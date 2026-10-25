#!/usr/bin/env python3
"""Run the research pair-as-asset backtest on a sector universe.

This is a thin CLI over ``portfolio_opt.pair_momentum``. It reads cached
yfinance adjusted closes, never places orders, and does not touch the live
strategy path.

Pairs are selected point-in-time inside each ``sector_*`` group from the
universe file. The default universe is stock-only and excludes the sector ETF
proxies. Pairs are not random: candidates are filtered by return correlation,
fitted with an OLS beta, and ranked by residual volatility.

Example:
    python scripts/backtest_pair_momentum.py
    python scripts/backtest_pair_momentum.py --mode pair-rank --signal reversal
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np

from portfolio_opt.backtest import align_close_history, summarize_return_series
from portfolio_opt.pair_momentum import (
    PairBacktestConfig,
    PairBacktestResult,
    PairMode,
    PairSignal,
    run_pair_momentum_backtest,
)

os.chdir(Path(__file__).resolve().parents[1])

DEFAULT_UNIVERSE = "examples/pair_universe_curated.json"
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_START = "2015-01-02"
DEFAULT_END = "2026-05-15"


def load_cached_closes(
    symbols: set[str], cache_dir: str
) -> dict[str, dict[str, float]]:
    closes: dict[str, dict[str, float]] = {}
    pattern = str(Path(cache_dir) / "yfinance_closes_v2_*.json")
    for path in glob.glob(pattern):
        try:
            payload = json.loads(Path(path).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        symbol = payload.get("symbol")
        values = payload.get("closes")
        if symbol in symbols and isinstance(values, dict):
            closes[str(symbol)] = values
    return closes


def build_data(
    *,
    universe_path: str,
    cache_dir: str,
    start: str,
    end: str,
) -> tuple[
    list[str],
    list[str],
    np.ndarray,
    dict[str, list[int]],
    np.ndarray,
]:
    universe = json.loads(Path(universe_path).read_text())
    pair_symbols: list[str] = []
    for pair in universe.get("pairs", []):
        if "legs" in pair:
            pair_symbols.extend(str(leg["symbol"]) for leg in pair["legs"])
        else:
            pair_symbols.extend([str(pair["long"]), str(pair["short"])])
    all_symbols = list(dict.fromkeys([*universe["symbols"], *pair_symbols]))
    asset_classes = {
        str(symbol): str(asset_class)
        for symbol, asset_class in universe.get("asset_classes", {}).items()
    }
    sectors = sorted(
        {value for value in asset_classes.values() if value.startswith("sector_")}
    )
    cached = load_cached_closes(set(all_symbols) | {"SPY"}, cache_dir)
    if "SPY" not in cached:
        raise RuntimeError("SPY is missing from the yfinance cache.")

    dates = [date for date in sorted(cached["SPY"]) if start <= date <= end]
    symbols: list[str] = []
    for symbol in all_symbols:
        if symbol not in cached or asset_classes.get(symbol) not in sectors:
            continue
        if all(date in cached[symbol] for date in dates):
            symbols.append(symbol)
    if not symbols:
        raise RuntimeError("No sector symbols have complete cached history.")

    closes_by_symbol = {
        symbol: [cached[symbol][date] for date in dates] for symbol in symbols
    }
    aligned = align_close_history(symbols, closes_by_symbol)
    prices = np.array([aligned[symbol] for symbol in symbols], dtype=float)
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    sector_indices = {
        sector: [
            index
            for index, symbol in enumerate(symbols)
            if asset_classes.get(symbol) == sector
        ]
        for sector in sectors
    }
    spy_prices = np.array([cached["SPY"][date] for date in dates], dtype=float)
    spy_returns = spy_prices[1:] / spy_prices[:-1] - 1.0
    return symbols, dates, returns, sector_indices, spy_returns


def load_fixed_pairs(
    universe_path: str,
    symbols: list[str],
    *,
    screen_pairs: bool,
) -> list[tuple[int, int]] | None:
    if screen_pairs:
        return None
    universe = json.loads(Path(universe_path).read_text())
    definitions = universe.get("pairs")
    if not definitions:
        return None
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    fixed: list[tuple[int, int]] = []
    for definition in definitions:
        long_symbol = str(definition["long"])
        short_symbol = str(definition["short"])
        if long_symbol in symbol_index and short_symbol in symbol_index:
            fixed.append((symbol_index[long_symbol], symbol_index[short_symbol]))
    return fixed or None


def format_table(results: list[tuple[str, PairBacktestResult]]) -> str:
    header = (
        f"{'strategy':<24} {'CAGR':>8} {'vol':>8} {'maxDD':>8} "
        f"{'Sortino':>8} {'turnover':>9}"
    )
    lines = [header, "-" * len(header)]
    for name, result in results:
        lines.append(
            f"{name:<24} {result.annualized_return:>7.2%} "
            f"{result.annualized_volatility:>7.2%} "
            f"{result.max_drawdown:>7.2%} {result.sortino_ratio:>8.2f} "
            f"{result.average_turnover:>9.2f}"
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--formation-window", type=int, default=252)
    parser.add_argument("--formation-every", type=int, default=126)
    parser.add_argument("--momentum-min", type=int, default=63)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--pairs-per-sector", type=int, default=8)
    parser.add_argument("--max-pairs-per-leg", type=int, default=2)
    parser.add_argument("--min-corr", type=float, default=0.2)
    parser.add_argument("--pair-cost", type=float, default=0.0010)
    parser.add_argument("--borrow", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--cat-k", type=int, default=3)
    parser.add_argument("--pairs-per-cat", type=int, default=3)
    parser.add_argument(
        "--screen-pairs",
        action="store_true",
        help="Ignore any fixed pair list and select pairs systematically.",
    )
    parser.add_argument(
        "--mode",
        choices=("equal-all", "pair-rank", "category-index", "category-pairs"),
        default=None,
        help="Run one mode; otherwise print the standard comparison table.",
    )
    parser.add_argument(
        "--signal",
        choices=("momentum", "reversal"),
        default="momentum",
        help="Pair ranking direction inside a category or across all pairs.",
    )
    parser.add_argument("--json", default=None, help="Optional JSON output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PairBacktestConfig(
        formation_window=args.formation_window,
        formation_every=args.formation_every,
        momentum_min=args.momentum_min,
        rebalance_every=args.rebalance_every,
        pairs_per_sector=args.pairs_per_sector,
        max_pairs_per_leg=args.max_pairs_per_leg,
        min_corr=args.min_corr,
        pair_cost=args.pair_cost,
        borrow_rate=args.borrow,
        top_k=args.top_k,
        category_top_k=args.cat_k,
        pairs_per_category=args.pairs_per_cat,
    )
    symbols, dates, returns, sector_indices, spy_returns = build_data(
        universe_path=args.universe,
        cache_dir=args.cache_dir,
        start=args.start,
        end=args.end,
    )
    fixed_pair_indices = load_fixed_pairs(
        args.universe,
        symbols,
        screen_pairs=args.screen_pairs,
    )
    pair_source = (
        f"systematic screen ({len(symbols)} symbols)"
        if fixed_pair_indices is None
        else f"manual ({len(fixed_pair_indices)} fixed pairs)"
    )
    variants = (
        [
            ("equal-all", "momentum"),
            ("pair-rank", "momentum"),
            ("pair-rank", "reversal"),
            ("category-index", "momentum"),
            ("category-pairs", "momentum"),
            ("category-index", "reversal"),
            ("category-pairs", "reversal"),
        ]
        if args.mode is None
        else [(args.mode, args.signal)]
    )

    start_index = config.formation_window + config.momentum_min
    spy_summary = summarize_return_series(
        spy_returns[start_index:], trading_days_per_year=config.trading_days_per_year
    )
    results: list[tuple[str, PairBacktestResult]] = [
        (
            "SPY",
            PairBacktestResult(
                final_value=spy_summary.final_value,
                total_return=spy_summary.total_return,
                annualized_return=spy_summary.annualized_return,
                annualized_volatility=spy_summary.annualized_volatility,
                max_drawdown=spy_summary.max_drawdown,
                rebalance_count=0,
                average_turnover=0.0,
                daily_values=(),
                sortino_ratio=spy_summary.sortino_ratio,
                latest_pair_weights={},
            ),
        )
    ]
    for mode, signal in variants:
        result = run_pair_momentum_backtest(
            symbols=symbols,
            returns=returns,
            sector_indices=sector_indices,
            config=config,
            mode=cast(PairMode, mode),
            signal=cast(PairSignal, signal),
            fixed_pair_indices=fixed_pair_indices,
        )
        results.append((f"{mode}:{signal}", result))

    print(
        f"Universe: {args.universe} | {pair_source} | "
        f"{dates[0]} to {dates[-1]} | pair cost {config.pair_cost:.2%} | "
        f"borrow {config.borrow_rate:.2%}"
    )
    print(format_table(results))

    if args.json:
        payload = {
            "config": asdict(config),
            "universe": args.universe,
            "pair_source": pair_source,
            "fixed_pair_count": (
                len(fixed_pair_indices) if fixed_pair_indices is not None else None
            ),
            "symbols": len(symbols),
            "dates": {"start": dates[0], "end": dates[-1]},
            "results": [{"name": name, **asdict(result)} for name, result in results],
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
