"""Compare momentum variants on two repository Nasdaq-100-style universes."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from portfolio_opt.backtest import summarize_return_series
from portfolio_opt.fixed_allocation import run_fixed_allocation_backtest
from scripts import benchmark_momentum_variants as benchmark

MODELS = {
    "current_nasdaq100": ROOT / "examples/nasdaq100_universe.json",
    "pre2016_filtered_nasdaq_style": ROOT
    / "examples/nasdaq100_sector_universe_b2016filtered.json",
}
CANDIDATES = (
    "dual_2_equal",
    "dual_5_score_cap",
    "dual_5_equal_cap",
    "dual_5_score_63d_rebalance",
    "dual_5_inverse_vol_cap",
    "dual_5_score_20pct_vol",
    "dual_5_score_126d",
    "spy_buy_hold",
    "qqq_buy_hold",
    "equal_weight_universe",
)
COSTS = (0.001, 0.003)
FULL = ROOT / ".cache/nasdaq100_sensitivity_full.json"
SUMMARY = ROOT / "docs/data/nasdaq100_sensitivity_2026-09-29.json"


def load_panel(
    model_path: Path,
) -> tuple[
    list[str], list[str], dict[str, list[float]], dict[str, str], dict[str, str]
]:
    model = json.loads(model_path.read_text())
    symbols: list[str] = model["symbols"]
    raw: dict[str, dict[str, float]] = {}
    files: dict[str, str] = {}
    for symbol in dict.fromkeys([*symbols, "SPY", "QQQ"]):
        matches: list[tuple[Path, dict[str, Any], str]] = []
        for path in (ROOT / ".cache").glob(f"yfinance_closes_v2_{symbol}_*.json"):
            blob = path.read_bytes()
            cache = json.loads(blob)
            if cache.get("symbol") == symbol:
                matches.append(
                    (path, cache["closes"], hashlib.sha256(blob).hexdigest())
                )
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one cache for {symbol}; found {len(matches)}"
            )
        path, prices, digest = matches[0]
        raw[symbol] = prices
        files[symbol] = f"{path.relative_to(ROOT)} sha256:{digest}"
    dates = sorted(set.intersection(*(set(prices) for prices in raw.values())))
    dates = [day for day in dates if "2016-01-01" <= day <= "2026-09-29"]
    if len(dates) <= benchmark.WARMUP + 1:
        raise ValueError(f"Insufficient common history for {model_path}")
    closes = {symbol: [float(raw[symbol][day]) for day in dates] for symbol in raw}
    return symbols, dates, closes, model["asset_classes"], files


def run_qqq(closes: dict[str, list[float]], cost: float) -> benchmark.VariantResult:
    result = run_fixed_allocation_backtest(
        symbols=["QQQ"],
        closes_by_symbol={"QQQ": closes["QQQ"]},
        weights_by_symbol={"QQQ": 1.0},
        start_day=benchmark.WARMUP,
        rebalance_every=100_000,
        linear_trade_cost=cost,
    )
    values = np.asarray(result.daily_values, dtype=float)
    metrics = summarize_return_series(values[1:] / values[:-1] - 1.0)
    return {
        "cagr": float(metrics.annualized_return),
        "volatility": float(metrics.annualized_volatility),
        "max_drawdown": float(metrics.max_drawdown),
        "sortino": float(metrics.sortino_ratio),
        "average_turnover": float(result.average_turnover),
        "values": values.tolist(),
    }


def main() -> None:
    full: dict[str, Any] = {}
    compact: dict[str, Any] = {}
    evaluation_dates: dict[str, list[str]] = {}
    for label, model_path in MODELS.items():
        symbols, dates, closes, groups, files = load_panel(model_path)
        evaluation_dates[label] = dates[benchmark.WARMUP :]
        results: dict[str, dict[str, benchmark.VariantResult]] = {}
        for cost in COSTS:
            by_candidate: dict[str, benchmark.VariantResult] = {}
            for candidate in CANDIDATES:
                print(f"{label} {cost} {candidate}", flush=True)
                if candidate == "qqq_buy_hold":
                    by_candidate[candidate] = run_qqq(closes, cost)
                else:
                    name = (
                        "equal_weight_49"
                        if candidate == "equal_weight_universe"
                        else candidate
                    )
                    by_candidate[candidate] = benchmark.run_one(
                        name, cost, symbols, closes, groups
                    )
            results[str(cost)] = by_candidate
        first_cost = results[str(COSTS[0])]
        baseline = first_cost["dual_2_equal"]["values"]
        diagnostics = {
            name: benchmark.paired_interval(first_cost[name]["values"], baseline)
            for name in (
                "dual_5_score_cap",
                "dual_5_equal_cap",
                "dual_5_score_63d_rebalance",
            )
        }
        metadata = {
            "model": str(model_path.relative_to(ROOT)),
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "symbols": symbols,
            "symbol_count": len(symbols),
            "price_start": dates[0],
            "evaluation_start": dates[benchmark.WARMUP],
            "evaluation_end": dates[-1],
            "return_days": len(dates) - benchmark.WARMUP - 1,
            "warmup_days": benchmark.WARMUP,
            "costs": list(COSTS),
            "price_cache_files": files,
            "candidate_specs": {
                name: (
                    ("fixed", {"symbol": "QQQ", "rebalance_every": 100_000})
                    if name == "qqq_buy_hold"
                    else ("fixed", {"rebalance_every": 21})
                    if name == "equal_weight_universe"
                    else benchmark.SPECS[name]
                )
                for name in CANDIDATES
            },
            "paired_annual_log_excess_vs_dual_2_equal": diagnostics,
        }
        full[label] = {**metadata, "results": results}
        compact[label] = {
            **metadata,
            "results": {
                cost: {
                    name: {key: value for key, value in row.items() if key != "values"}
                    for name, row in by_candidate.items()
                }
                for cost, by_candidate in results.items()
            },
        }
    overlap_dates = sorted(
        set.intersection(*(set(days) for days in evaluation_dates.values()))
    )
    overlap: dict[str, Any] = {
        "start": overlap_dates[0],
        "end": overlap_dates[-1],
        "return_days": len(overlap_dates) - 1,
        "results": {},
    }
    for label in MODELS:
        positions = {day: index for index, day in enumerate(evaluation_dates[label])}
        compared: dict[str, dict[str, float]] = {}
        for candidate in (
            "dual_2_equal",
            "dual_5_score_cap",
            "dual_5_equal_cap",
            "qqq_buy_hold",
        ):
            curve = np.asarray(
                full[label]["results"][str(COSTS[0])][candidate]["values"]
            )
            selected = curve[[positions[day] for day in overlap_dates]]
            metrics = summarize_return_series(selected[1:] / selected[:-1] - 1.0)
            compared[candidate] = {
                "cagr": float(metrics.annualized_return),
                "max_drawdown": float(metrics.max_drawdown),
            }
        overlap["results"][label] = compared
    full["overlap"] = overlap
    compact["overlap"] = overlap
    FULL.write_text(json.dumps(full, indent=2) + "\n")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(compact, indent=2) + "\n")
    print(SUMMARY)


if __name__ == "__main__":
    main()
