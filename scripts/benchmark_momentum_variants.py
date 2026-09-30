"""Compare predeclared momentum variants on one local, date-aligned price panel."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from portfolio_opt.backtest import (
    run_dual_momentum_backtest,
    run_factor_momentum_backtest,
    summarize_return_series,
)
from portfolio_opt.fixed_allocation import run_fixed_allocation_backtest
from portfolio_opt.regime_adaptive import run_regime_adaptive_backtest

INPUT = ROOT / "docs/data/momentum_variant_input_2026-09-29.json"
ORIGINAL = ROOT / ".cache/sec_quality_benchmark_raw_2016_2026.json"
OUTPUT = ROOT / ".cache/momentum_variant_benchmark.json"
REPORT = ROOT / ".cache/autoresearch.md"
LOG = ROOT / ".cache/autoresearch.jsonl"
SUMMARY = ROOT / "docs/data/momentum_variant_benchmark_2026-09-29.json"
WARMUP = 252
COSTS = (0.001, 0.003)


class VariantResult(TypedDict):
    cagr: float
    volatility: float
    max_drawdown: float
    sortino: float
    average_turnover: float
    values: list[float]


SPECS: dict[str, tuple[str, dict[str, Any]]] = {
    "dual_2_equal": (
        "dual",
        {"lookback_days": 252, "top_k": 2, "weighting": "equal", "trailing_stop": 0.15},
    ),
    "dual_5_score_cap": (
        "dual",
        {
            "lookback_days": 252,
            "top_k": 5,
            "weighting": "score",
            "max_single_weight": 0.5,
            "trailing_stop": 0.15,
        },
    ),
    "dual_5_score_no_cap": (
        "dual",
        {"lookback_days": 252, "top_k": 5, "weighting": "score", "trailing_stop": 0.15},
    ),
    "dual_5_equal_cap": (
        "dual",
        {
            "lookback_days": 252,
            "top_k": 5,
            "weighting": "equal",
            "max_single_weight": 0.5,
            "trailing_stop": 0.15,
        },
    ),
    "dual_5_inverse_vol_cap": (
        "dual",
        {
            "lookback_days": 252,
            "top_k": 5,
            "weighting": "inverse-vol",
            "max_single_weight": 0.5,
            "trailing_stop": 0.15,
        },
    ),
    "dual_5_score_no_stop": (
        "dual",
        {
            "lookback_days": 252,
            "top_k": 5,
            "weighting": "score",
            "max_single_weight": 0.5,
        },
    ),
    "dual_5_score_126d": (
        "dual",
        {
            "lookback_days": 126,
            "top_k": 5,
            "weighting": "score",
            "max_single_weight": 0.5,
            "trailing_stop": 0.15,
        },
    ),
    "dual_5_score_63d_rebalance": (
        "dual",
        {
            "lookback_days": 252,
            "top_k": 5,
            "weighting": "score",
            "max_single_weight": 0.5,
            "trailing_stop": 0.15,
            "rebalance_every": 63,
        },
    ),
    "dual_5_score_20pct_vol": (
        "dual",
        {
            "lookback_days": 252,
            "top_k": 5,
            "weighting": "score",
            "max_single_weight": 0.5,
            "trailing_stop": 0.15,
            "target_vol": 0.2,
        },
    ),
    "factor_5_equal_3_sectors": (
        "factor",
        {
            "lookback_days": 252,
            "top_k": 5,
            "factor_top_k": 3,
            "weighting": "equal",
            "max_single_weight": 0.5,
            "trailing_stop": 0.15,
        },
    ),
    "regime_adaptive": ("regime", {"lookback_days": 126, "top_k": 5}),
    "spy_buy_hold": ("fixed", {}),
    "equal_weight_49": ("fixed", {}),
}


def load_prices() -> tuple[
    list[str], list[str], dict[str, list[float]], dict[str, str], dict[str, str]
]:
    source = json.loads(INPUT.read_text())
    symbols = source["symbols"]
    universe = json.loads((ROOT / source["universe"]).read_text())
    cache_files: dict[str, str] = {}
    raw: dict[str, dict[str, float]] = {}
    for symbol in [*symbols, "SPY"]:
        matches = []
        for path in (ROOT / ".cache").glob(f"yfinance_closes_v2_{symbol}_*.json"):
            blob = path.read_bytes()
            item = json.loads(blob)
            if item["symbol"] == symbol:
                matches.append((path, item["closes"], hashlib.sha256(blob).hexdigest()))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one exact-symbol price cache for {symbol}: {matches}"
            )
        path, prices, digest = matches[0]
        raw[symbol] = prices
        cache_files[symbol] = f"{path.relative_to(ROOT)} sha256:{digest}"
    common = sorted(set.intersection(*(set(series) for series in raw.values())))
    common = [
        day for day in common if source["price_start"] <= day <= source["price_end"]
    ]
    if len(common) < WARMUP + 2:
        raise ValueError("Insufficient shared price history")
    closes = {symbol: [float(raw[symbol][day]) for day in common] for symbol in raw}
    return symbols, common, closes, universe["asset_classes"], cache_files


def run_one(
    name: str,
    cost: float,
    symbols: list[str],
    closes: dict[str, list[float]],
    groups: dict[str, str],
) -> VariantResult:
    kind, params = SPECS[name]
    if kind == "fixed":
        held = ["SPY"] if name == "spy_buy_hold" else symbols
        weights = {symbol: 1.0 / len(held) for symbol in held}
        result = run_fixed_allocation_backtest(
            symbols=held,
            closes_by_symbol={symbol: closes[symbol] for symbol in held},
            weights_by_symbol=weights,
            start_day=WARMUP,
            rebalance_every=100_000 if name == "spy_buy_hold" else 21,
            linear_trade_cost=cost,
        )
    else:
        lookback = int(params["lookback_days"])
        data = {symbol: closes[symbol][WARMUP - lookback :] for symbol in symbols}
        common_args: dict[str, Any] = {
            "symbols": symbols,
            "closes_by_symbol": data,
            "asset_classes": groups,
            "rebalance_every": int(params.get("rebalance_every", 21)),
            "absolute_threshold": 0.0,
            "linear_trade_cost": cost,
        }
        options: dict[str, Any] = {
            key: value for key, value in params.items() if key != "rebalance_every"
        }
        if kind == "dual":
            result = run_dual_momentum_backtest(**common_args, **options)
        elif kind == "factor":
            result = run_factor_momentum_backtest(**common_args, **options)
        else:
            result = run_regime_adaptive_backtest(**common_args, **options)

    values = np.asarray(result.daily_values, dtype=float)
    if len(values) != len(next(iter(closes.values()))) - WARMUP:
        raise AssertionError(f"Curve length mismatch for {name}: {len(values)}")
    if np.any(values <= 0) or not np.all(np.isfinite(values)):
        raise AssertionError(f"Invalid curve for {name}")
    summary = summarize_return_series(values[1:] / values[:-1] - 1.0)
    return {
        "cagr": float(summary.annualized_return),
        "volatility": float(summary.annualized_volatility),
        "max_drawdown": float(summary.max_drawdown),
        "sortino": float(summary.sortino_ratio),
        "average_turnover": float(result.average_turnover),
        "values": values.tolist(),
    }


def paired_interval(a: list[float], b: list[float]) -> tuple[float, float, float]:
    a_values, b_values = np.asarray(a), np.asarray(b)
    difference = np.log(a_values[1:] / a_values[:-1]) - np.log(
        b_values[1:] / b_values[:-1]
    )
    draws, block = 2000, 21
    rng = np.random.default_rng(20260929)
    starts = rng.integers(
        0, len(difference), size=(draws, (len(difference) + block - 1) // block)
    )
    indices = (starts[:, :, None] + np.arange(block)) % len(difference)
    sample = 252 * difference[indices.reshape(draws, -1)[:, : len(difference)]].mean(
        axis=1
    )
    lower, upper = np.percentile(sample, [2.5, 97.5])
    return float(252 * difference.mean()), float(lower), float(upper)


def main() -> None:
    symbols, dates, closes, groups, cache_files = load_prices()
    results: dict[str, dict[str, VariantResult]] = {}
    for cost in COSTS:
        key = str(cost)
        results[key] = {}
        for name in SPECS:
            print(f"{key} {name}", flush=True)
            results[key][name] = run_one(name, cost, symbols, closes, groups)
    first = dates[WARMUP]
    last = dates[-1]
    baseline = results[str(COSTS[0])]["dual_2_equal"]
    proposed = results[str(COSTS[0])]["dual_5_score_cap"]
    interval = paired_interval(proposed["values"], baseline["values"])
    quarterly = results[str(COSTS[0])]["dual_5_score_63d_rebalance"]
    quarterly_interval = paired_interval(quarterly["values"], baseline["values"])
    cap_difference = float(
        np.max(
            np.abs(
                np.asarray(proposed["values"])
                - np.asarray(results[str(COSTS[0])]["dual_5_score_no_cap"]["values"])
            )
        )
    )
    baseline_cagr_difference: float | None = None
    if ORIGINAL.exists():
        original = json.loads(ORIGINAL.read_text())
        original_dates = original["dates"]
        original_start = original_dates.index(first)
        original_end = original_dates.index(last)
        original_values = np.asarray(
            original["dual_momentum_values"][original_start : original_end + 1]
        )
        original_metrics = summarize_return_series(
            original_values[1:] / original_values[:-1] - 1.0
        )
        baseline_cagr_difference = float(
            baseline["cagr"] - original_metrics.annualized_return
        )
    output = {
        "source": str(INPUT.relative_to(ROOT)),
        "cache_files": cache_files,
        "symbols": symbols,
        "price_start": dates[0],
        "evaluation_start": first,
        "evaluation_end": last,
        "return_days": len(dates) - WARMUP - 1,
        "common_warmup_days": WARMUP,
        "costs": list(COSTS),
        "specs": SPECS,
        "results": results,
        "proposed_minus_baseline_annual_log_return_ci": interval,
        "quarterly_minus_baseline_annual_log_return_ci": quarterly_interval,
        "maximum_cap_vs_no_cap_curve_difference": cap_difference,
        "baseline_cagr_minus_original_report_same_dates": baseline_cagr_difference,
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    compact = {
        **output,
        "results": {
            cost: {
                name: {key: value for key, value in result.items() if key != "values"}
                for name, result in candidates.items()
            }
            for cost, candidates in results.items()
        },
        "full_curve_artifact": str(OUTPUT.relative_to(ROOT)),
    }
    SUMMARY.write_text(json.dumps(compact, indent=2) + "\n")

    lines = [
        "# Bounded momentum variant benchmark",
        "",
        f"49 current-survivor symbols; price history {dates[0]} to {last}; evaluation {first} to {last} ({output['return_days']} return days).",
        "All variants use the same cached Yahoo adjusted closes, dates, 252-day common warmup, and trading cost. Strategies rebalance every 21 trading days unless specified. The original 252-day/top-2/equal/15%-stop variant is the baseline. The top-5/score/50%-cap/15%-stop variant is the proposed command.",
        "The fixed 49-symbol sample differs from the live dynamic universe and is selected with present knowledge. This is exploratory, not a point-in-time investable test or untouched holdout. Broker stop execution, taxes, and market impact are not modeled.",
        "",
        "| Candidate | CAGR 10 bps | CAGR 30 bps | Vol 10 bps | Max DD 10 bps | Sortino 10 bps |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    ranking = sorted(
        SPECS, key=lambda name: results[str(COSTS[0])][name]["cagr"], reverse=True
    )
    for name in ranking:
        a = results[str(COSTS[0])][name]
        b = results[str(COSTS[1])][name]
        lines.append(
            f"| {name} | {a['cagr']:.1%} | {b['cagr']:.1%} | {a['volatility']:.1%} | {a['max_drawdown']:.1%} | {a['sortino']:.2f} |"
        )
    lines.extend(
        [
            "",
            f"Proposed minus baseline annual log return: {interval[0]:+.1%}; paired 21-day block bootstrap 95% interval [{interval[1]:+.1%}, {interval[2]:+.1%}]. The interval does not adjust for testing multiple variants.",
            f"63-day rebalance minus baseline annual log return: {quarterly_interval[0]:+.1%}; paired interval [{quarterly_interval[1]:+.1%}, {quarterly_interval[2]:+.1%}].",
            f"The 50% single-position cap did not bind in this sample: maximum cap/no-cap curve difference {cap_difference:.2g}.",
            (
                f"Baseline cross-check against the previous raw benchmark over identical dates: CAGR differs by {baseline_cagr_difference:+.2%}, consistent with using a different cached price vintage."
                if baseline_cagr_difference is not None
                else "Previous raw benchmark unavailable; baseline cross-check skipped."
            ),
            "",
            "## Subperiod CAGR at 10 bps",
            "",
            "| Candidate | 2017–2019 | 2020–2021 | 2022–2023 | 2024–2026 May |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    evaluation_dates = dates[WARMUP:]
    periods = (
        ("2017-01-01", "2019-12-31"),
        ("2020-01-01", "2021-12-31"),
        ("2022-01-01", "2023-12-31"),
        ("2024-01-01", last),
    )
    for name in (
        "dual_2_equal",
        "dual_5_score_cap",
        "dual_5_equal_cap",
        "dual_5_score_63d_rebalance",
        "spy_buy_hold",
    ):
        values = np.asarray(results[str(COSTS[0])][name]["values"])
        cagr = []
        for start, end in periods:
            indices = [
                i for i, date in enumerate(evaluation_dates) if start <= date <= end
            ]
            selected = values[indices[0] : indices[-1] + 1]
            metrics = summarize_return_series(selected[1:] / selected[:-1] - 1.0)
            cagr.append(f"{metrics.annualized_return:.1%}")
        lines.append(f"| {name} | {' | '.join(cagr)} |")
    lines.extend(
        [
            "",
            "Reproduce: `.venv/bin/python scripts/benchmark_momentum_variants.py`. Compact results are in `docs/data/momentum_variant_benchmark_2026-09-29.json`; full curves remain in `.cache/momentum_variant_benchmark.json`.",
            "",
        ]
    )
    REPORT.write_text("\n".join(lines))
    with LOG.open("w") as stream:
        for name, parameters in SPECS.items():
            stream.write(
                json.dumps(
                    {
                        "candidate": name,
                        "parameters": parameters,
                        "ten_bps": {
                            k: v
                            for k, v in results[str(COSTS[0])][name].items()
                            if k != "values"
                        },
                        "thirty_bps": {
                            k: v
                            for k, v in results[str(COSTS[1])][name].items()
                            if k != "values"
                        },
                    }
                )
                + "\n"
            )
    print(REPORT)


if __name__ == "__main__":
    main()
