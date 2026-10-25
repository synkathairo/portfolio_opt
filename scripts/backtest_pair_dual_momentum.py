#!/usr/bin/env python3
"""Research backtest: existing dual momentum applied to pair NAVs.

The pair layer is a fixed curated list from a universe file. Each pair is a
beta-neutral synthetic asset. The second layer is the repository's native dual
momentum selector: absolute filter versus cash, relative top-k ranking, and a
cash fallback.

This is research-only. It reads cached yfinance closes and never places orders.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import numpy as np

from portfolio_opt.backtest import (
    align_close_history,
    run_dual_momentum_backtest,
    summarize_return_series,
)
from portfolio_opt.pair_momentum import PairBacktestConfig, build_pair_navs

os.chdir(Path(__file__).resolve().parents[1])

DEFAULT_UNIVERSE = "examples/pair_universe_curated.json"
DEFAULT_CACHE_DIR = ".cache"
DEFAULT_START = "2015-01-02"
DEFAULT_END = "2026-05-15"
CASH_SYMBOL = "BIL"


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
    dict[str, str],
    np.ndarray,
    list[tuple[int, int]],
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
    pair_definitions = universe.get("pairs", [])
    sectors = sorted(
        {value for value in asset_classes.values() if value.startswith("sector_")}
    )
    cached = load_cached_closes(set(all_symbols) | {CASH_SYMBOL, "SPY"}, cache_dir)
    for required in (CASH_SYMBOL, "SPY"):
        if required not in cached:
            raise RuntimeError(f"{required} is missing from the yfinance cache.")

    cash_dates = set(cached[CASH_SYMBOL])
    dates = [
        date
        for date in sorted(cached["SPY"])
        if start <= date <= end and date in cash_dates
    ]
    symbols: list[str] = []
    for symbol in all_symbols:
        if symbol not in cached or asset_classes.get(symbol) not in sectors:
            continue
        if all(date in cached[symbol] for date in dates):
            symbols.append(symbol)
    if not symbols:
        raise RuntimeError("No symbols have complete cached history.")

    closes_by_symbol = {
        symbol: [cached[symbol][date] for date in dates] for symbol in symbols
    }
    aligned = align_close_history(symbols, closes_by_symbol)
    prices = np.array([aligned[symbol] for symbol in symbols], dtype=float)
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    spy_prices = np.array([cached["SPY"][date] for date in dates], dtype=float)
    spy_returns = spy_prices[1:] / spy_prices[:-1] - 1.0
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    fixed_pairs = [
        (symbol_index[str(definition["long"])], symbol_index[str(definition["short"])])
        for definition in pair_definitions
        if str(definition["long"]) in symbol_index
        and str(definition["short"]) in symbol_index
    ]
    return symbols, dates, returns, asset_classes, spy_returns, fixed_pairs


def format_result(name: str, result) -> str:
    return (
        f"{name:<34} {result.annualized_return:>7.2%} "
        f"{result.annualized_volatility:>7.2%} "
        f"{result.max_drawdown:>7.2%} {result.sortino_ratio:>8.2f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--formation-window", type=int, default=252)
    parser.add_argument("--formation-every", type=int, default=126)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--pair-cost", type=float, default=0.0010)
    parser.add_argument("--borrow", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--lookback",
        type=int,
        action="append",
        default=None,
        help="Dual-momentum lookback(s). Defaults to 126 and 252.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PairBacktestConfig(
        formation_window=args.formation_window,
        formation_every=args.formation_every,
        rebalance_every=args.rebalance_every,
        pair_cost=args.pair_cost,
        borrow_rate=args.borrow,
        top_k=args.top_k,
    )
    symbols, dates, returns, asset_classes, spy_returns, pairs = build_data(
        universe_path=args.universe,
        cache_dir=args.cache_dir,
        start=args.start,
        end=args.end,
    )
    if not pairs:
        raise RuntimeError("The universe file does not define any usable pairs.")

    pair_navs = build_pair_navs(returns, pairs, config, net=True)
    pair_labels = [f"{symbols[long]}/{symbols[short]}" for long, short in pairs]
    cash_closes = load_cached_closes({CASH_SYMBOL}, args.cache_dir)[CASH_SYMBOL]
    cash_prices = [cash_closes[date] for date in dates]
    closes_by_symbol = {
        label: pair_navs[pair].tolist()
        for label, pair in zip(pair_labels, pairs, strict=True)
    }
    closes_by_symbol[CASH_SYMBOL] = cash_prices
    pair_asset_classes = {
        label: asset_classes.get(symbols[pair[0]], "sector_unknown")
        for label, pair in zip(pair_labels, pairs, strict=True)
    }
    pair_asset_classes[CASH_SYMBOL] = "cash_like"

    lookbacks = args.lookback or [126, 252]
    print(
        f"Universe: {args.universe} | {len(pairs)} fixed pairs | "
        f"{dates[0]} to {dates[-1]} | pair cost {config.pair_cost:.2%} | "
        f"borrow {config.borrow_rate:.2%}"
    )
    print(f"{'strategy':<34} {'CAGR':>7} {'vol':>7} {'maxDD':>7} {'Sortino':>8}")
    print("-" * 68)

    spy_summary = summarize_return_series(
        spy_returns[config.formation_window :],
        trading_days_per_year=config.trading_days_per_year,
    )
    print(format_result("SPY", spy_summary))

    pair_returns = np.array(
        [np.diff(nav) / nav[:-1] for nav in pair_navs.values()],
        dtype=float,
    )
    equal_returns = pair_returns.mean(axis=0)
    equal_summary = summarize_return_series(
        equal_returns[config.formation_window :],
        trading_days_per_year=config.trading_days_per_year,
    )
    print(format_result("equal-weight all pairs", equal_summary))

    stock_navs = {
        symbol: np.concatenate(([1.0], np.cumprod(1.0 + returns[index])))
        for index, symbol in enumerate(symbols)
    }

    for lookback in lookbacks:
        panel_start = config.formation_window - lookback
        if panel_start < 0:
            continue
        pair_panel = {
            label: values[panel_start:] for label, values in closes_by_symbol.items()
        }
        pair_result = run_dual_momentum_backtest(
            symbols=list(pair_panel),
            closes_by_symbol=pair_panel,
            asset_classes=pair_asset_classes,
            lookback_days=lookback,
            rebalance_every=config.rebalance_every,
            top_k=config.top_k,
            absolute_threshold=0.0,
            weighting="equal",
            linear_trade_cost=config.pair_cost,
        )
        print(
            format_result(
                f"pair dual momentum {lookback}d top{config.top_k}", pair_result
            )
        )

        stock_panel = {
            symbol: nav[panel_start:].tolist() for symbol, nav in stock_navs.items()
        }
        stock_panel[CASH_SYMBOL] = cash_prices[panel_start:]
        stock_asset_classes = {
            symbol: asset_classes.get(symbol, "sector_unknown") for symbol in symbols
        }
        stock_asset_classes[CASH_SYMBOL] = "cash_like"
        stock_result = run_dual_momentum_backtest(
            symbols=list(stock_panel),
            closes_by_symbol=stock_panel,
            asset_classes=stock_asset_classes,
            lookback_days=lookback,
            rebalance_every=config.rebalance_every,
            top_k=config.top_k,
            absolute_threshold=0.0,
            weighting="equal",
            linear_trade_cost=config.pair_cost,
        )
        print(
            format_result(
                f"stock dual momentum {lookback}d top{config.top_k}", stock_result
            )
        )


if __name__ == "__main__":
    main()
