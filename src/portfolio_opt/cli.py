from __future__ import annotations

import argparse
import bisect
import itertools
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from typing import cast

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from .alpaca_interface import AlpacaClient, format_order_plans
from .backtest import (
    compute_dual_momentum_weights,
    compute_factor_momentum_weights,
    rolling_window_comparison,
    run_backtest,
    run_dual_momentum_backtest,
    run_factor_momentum_backtest,
    run_fixed_weight_benchmark,
)
from .config import AlpacaConfig, OptimizationConfig
from .csv_data import (
    fetch_closes as csv_fetch_closes,
    write_json_caches as csv_write_json_caches,
    write_yfinance_compatible_caches as csv_write_yfinance_compatible_caches,
)
from .estimation import estimate_inputs_from_momentum, estimate_inputs_from_prices
from .black_litterman import estimate_inputs_from_black_litterman
from .risk_parity import estimate_inputs_risk_parity
from .model import ModelInputs, load_model_inputs
from .optimizer import effective_turnover_penalty, optimize_weights, project_weights
from .rebalance import build_order_plan, build_trailing_stop_plan, current_weights
from .runtime import configure_local_cache_dirs
from .stockanalysis_data import fetch_closes as stockanalysis_fetch_closes
from .yfinance_data import fetch_closes as yf_fetch_closes
from utils.fetch_tickers import fetch_ticker_dict, filter_tickers_before

configure_local_cache_dirs()


def _calculate_trading_date_offset(trading_days: int) -> datetime:
    """Return the date exactly `trading_days` trading sessions ago."""
    cal = xcals.get_calendar("XNYS")
    sessions = cal.sessions

    # Convert index to list of datetime objects for easy bisect
    session_dates: list[datetime] = sessions.tolist()
    # exchange_calendars returns naive timestamps (no timezone)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    # Find the index of the last trading day on or before today
    idx = bisect.bisect_right(session_dates, today)
    if idx == len(session_dates) or session_dates[idx] > today:
        idx -= 1

    cutoff_idx = idx - trading_days
    if cutoff_idx < 0:
        raise ValueError(
            f"Cannot look back {trading_days} trading days; not enough history in calendar."
        )
    return cast(datetime, session_dates[cutoff_idx])


def _run_sweep_point(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    lookback_days: int,
    rebalance_every: int,
    return_model: str,
    mean_shrinkage: float,
    min_weight: float,
    max_weight: float,
    rebalance_threshold: float,
    max_turnover: float | None,
    class_min_weights: dict[str, float],
    class_max_weights: dict[str, float],
    risk_aversion: float,
    min_cash_weight: float,
    min_invested_weight: float,
    turnover_penalty: float,
    momentum_window: int,
    asset_class_matrix: np.ndarray | None,
) -> dict[str, float | int | dict[str, float]] | tuple[str, str]:
    """Run a single sweep parameter combination. Returns result or error tuple."""
    sweep_config = OptimizationConfig(
        risk_aversion=risk_aversion,
        min_weight=min_weight,
        max_weight=max_weight,
        rebalance_threshold=rebalance_threshold,
        turnover_penalty=turnover_penalty,
        force_full_investment=False,
        min_cash_weight=min_cash_weight,
        max_turnover=max_turnover,
        min_invested_weight=min_invested_weight,
        class_min_weights=class_min_weights,
        class_max_weights=class_max_weights,
    )
    try:
        sweep_backtest = run_backtest(
            symbols=symbols,
            closes_by_symbol=closes_by_symbol,
            lookback_days=lookback_days,
            rebalance_every=rebalance_every,
            return_model=return_model,
            mean_shrinkage=mean_shrinkage,
            momentum_window=momentum_window,
            opt_config=sweep_config,
            asset_class_matrix=asset_class_matrix,
        )
    except RuntimeError as exc:
        return (
            "error",
            json.dumps(
                {
                    "risk_aversion": risk_aversion,
                    "min_cash_weight": min_cash_weight,
                    "min_invested_weight": min_invested_weight,
                    "turnover_penalty": turnover_penalty,
                    "momentum_window": momentum_window,
                    "reason": str(exc),
                }
            ),
        )
    return {
        "risk_aversion": risk_aversion,
        "min_cash_weight": min_cash_weight,
        "min_invested_weight": min_invested_weight,
        "turnover_penalty": turnover_penalty,
        "momentum_window": momentum_window,
        "annualized_return": round(float(sweep_backtest.annualized_return), 6),
        "annualized_volatility": round(float(sweep_backtest.annualized_volatility), 6),
        "max_drawdown": round(float(sweep_backtest.max_drawdown), 6),
        "average_turnover": round(float(sweep_backtest.average_turnover), 6),
    }


def build_asset_class_matrix(
    symbols: list[str],
    asset_classes: dict[str, str],
    class_names: list[str],
) -> np.ndarray:
    if not class_names:
        return np.zeros((0, len(symbols)), dtype=float)
    matrix = np.zeros((len(class_names), len(symbols)), dtype=float)
    class_index = {class_name: index for index, class_name in enumerate(class_names)}
    for symbol_index, symbol in enumerate(symbols):
        class_name = asset_classes.get(symbol)
        if class_name is None:
            continue
        if class_name in class_index:
            matrix[class_index[class_name], symbol_index] = 1.0
    return matrix


def clean_weights(weights: np.ndarray, tolerance: float = 1e-5) -> np.ndarray:
    cleaned = np.array(weights, dtype=float)
    cleaned[np.abs(cleaned) < tolerance] = 0.0
    return cleaned


def asset_class_exposures(
    symbols: list[str],
    weights: np.ndarray,
    asset_classes: dict[str, str],
) -> dict[str, float]:
    return {
        class_name: round(
            float(
                sum(
                    weights[index]
                    for index, symbol in enumerate(symbols)
                    if asset_classes.get(symbol) == class_name
                )
            ),
            6,
        )
        for class_name in sorted(set(asset_classes.values()))
    }


def _validate_backtest_history(
    closes_by_symbol: dict[str, list[float]],
    *,
    lookback_days: int,
    backtest_days: int,
) -> None:
    if not closes_by_symbol:
        raise ValueError("No price history was loaded for the requested backtest.")

    lengths = {symbol: len(closes) for symbol, closes in closes_by_symbol.items()}
    common_history_days = min(lengths.values())
    required_history_days = lookback_days + backtest_days + 1
    if common_history_days >= required_history_days:
        return

    available_backtest_days = max(0, common_history_days - lookback_days - 1)
    limiting_symbols = sorted(lengths.items(), key=lambda item: item[1])[:5]
    limiting_summary = ", ".join(
        f"{symbol}={length}" for symbol, length in limiting_symbols
    )
    raise ValueError(
        "Not enough common history for the requested backtest. "
        f"Requested {backtest_days} backtest days with lookback {lookback_days}, "
        f"which needs {required_history_days} prices, but only {common_history_days} "
        f"common prices are available across the universe. That supports at most "
        f"{available_backtest_days} backtest days. Shortest histories: {limiting_summary}"
    )


def _json_key(value: str) -> str:
    return "".join(char.lower() if char.isalnum() else "_" for char in value).strip("_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a mean-variance rebalance against Alpaca."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", help="Path to model input JSON file.")
    group.add_argument(
        "--dynamic-universe",
        action="store_true",
        help="Fetch current index constituents dynamically instead of using a model file.",
    )
    parser.add_argument(
        "--filter-before",
        type=str,
        default=None,
        help="Only include tickers that started trading before this ISO date (e.g. 2020-01-01).",
    )
    parser.add_argument(
        "--ticker-basket",
        nargs="*",
        default=[],
        help="Universe components for --dynamic-universe (uses fetch_ticker_dict defaults if empty).",
    )
    parser.add_argument("--risk-aversion", type=float, default=4.0)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--rebalance-threshold", type=float, default=0.02)
    parser.add_argument("--turnover-penalty", type=float, default=0.02)
    parser.add_argument(
        "--allow-cash",
        action="store_true",
        help="Allow the optimizer to leave part of the portfolio in cash.",
    )
    parser.add_argument(
        "--min-cash-weight",
        type=float,
        default=0.0,
        help="Minimum cash weight to hold when --allow-cash is enabled.",
    )
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="Hard cap on one-step turnover, measured as sum(abs(target-current)).",
    )
    parser.add_argument(
        "--min-invested-weight",
        type=float,
        default=0.0,
        help="Minimum total risky-asset weight when cash is allowed.",
    )
    parser.add_argument(
        "--estimate-from-history",
        action="store_true",
        help="Estimate expected returns and covariance from Alpaca daily bars.",
    )
    parser.add_argument("--lookback-days", type=int, default=60)
    parser.add_argument(
        "--mean-shrinkage",
        type=float,
        default=0.75,
        help="Shrink sample mean returns toward zero to reduce estimation noise.",
    )
    parser.add_argument(
        "--return-model",
        choices=("sample-mean", "momentum", "black-litterman", "risk-parity"),
        default="sample-mean",
        help="How to estimate expected returns when using --estimate-from-history.",
    )
    parser.add_argument(
        "--strategy",
        choices=("mean-variance", "dual-momentum", "factor-momentum"),
        default="mean-variance",
        help="Strategy for live or backtest rebalancing. Momentum strategies use live prices when --estimate-from-history is set.",
    )
    parser.add_argument(
        "--momentum-window",
        type=int,
        default=63,
        help="Trailing trading-day window used by the momentum return model.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of assets to hold in dual momentum mode.",
    )
    parser.add_argument(
        "--factor-top-k",
        type=int,
        default=1,
        help="Number of top factor/sleeve groups to search in factor momentum mode.",
    )
    parser.add_argument(
        "--dual-momentum-weighting",
        choices=("equal", "score", "inverse-vol", "softmax"),
        default="equal",
        help="How to weight the selected basket in dual momentum mode.",
    )
    parser.add_argument(
        "--softmax-temperature",
        type=float,
        default=0.05,
        help="Temperature for softmax weighting in dual momentum mode.",
    )
    parser.add_argument(
        "--absolute-momentum-threshold",
        type=float,
        default=0.0,
        help="Minimum trailing return required for dual momentum if no cash proxy is present.",
    )
    parser.add_argument(
        "--target-vol",
        type=float,
        default=None,
        help="Target annualized portfolio volatility for the risky basket (vol targeting).",
    )
    parser.add_argument(
        "--vol-window",
        type=int,
        default=63,
        help="Trailing trading-day window used to estimate volatility for --target-vol.",
    )
    parser.add_argument(
        "--max-single-weight",
        type=float,
        default=None,
        help="Maximum weight for any single asset in the dual momentum basket.",
    )
    parser.add_argument(
        "--trailing-stop",
        type=float,
        default=None,
        help="Trailing stop-loss threshold per asset (e.g. 0.08 to exit an 8%% drawdown from peak).",
    )
    parser.add_argument(
        "--basket-opt",
        choices=("mean-variance",),
        default=None,
        help="How to size the momentum-selected basket (overrides --dual-momentum-weighting).",
    )
    parser.add_argument(
        "--basket-risk-aversion",
        type=float,
        default=1.0,
        help="Risk aversion for basket mean-variance optimization.",
    )
    parser.add_argument(
        "--data-source",
        choices=("alpaca", "yfinance", "csv", "csv+yfinance", "stockanalysis"),
        default="alpaca",
        help="Source for historical price data in backtest mode.",
    )
    parser.add_argument(
        "--csv-dir",
        default=".cache/csv",
        help=(
            "Directory of local OHLCV CSV files when --data-source csv is used. "
            "Rows must be symbol,date,open,high,low,close,volume."
        ),
    )
    parser.add_argument(
        "--csv-write-json-cache",
        action="store_true",
        help="Write provider-neutral JSON close caches from --csv-dir before running.",
    )
    parser.add_argument(
        "--stockanalysis-start",
        default="1980-01-01",
        help="Start date for --data-source stockanalysis chart JSON.",
    )
    parser.add_argument(
        "--stockanalysis-end",
        default=None,
        help="End date for --data-source stockanalysis chart JSON. Defaults to today.",
    )
    parser.add_argument(
        "--yfinance-max-workers",
        type=int,
        default=10,
        help="Maximum concurrent yfinance symbol downloads when --data-source yfinance is used.",
    )
    parser.add_argument(
        "--yfinance-retry-delay",
        type=float,
        default=1.0,
        help="Seconds to wait between yfinance retry attempts.",
    )
    parser.add_argument(
        "--yfinance-symbol-delay",
        type=float,
        default=0.02,
        help=(
            "Seconds to wait between yfinance symbol downloads when "
            "--yfinance-max-workers is 1."
        ),
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help=(
            "Additional benchmark ticker to compare in backtest mode. "
            "Can be repeated, e.g. --benchmark ^HSI."
        ),
    )
    parser.add_argument(
        "--backtest-days",
        type=int,
        default=0,
        help="Run a simple offline backtest over this many trading days instead of a live rebalance.",
    )
    parser.add_argument(
        "--rebalance-every",
        type=int,
        default=21,
        help="Trading-day interval between rebalances in backtest mode.",
    )
    parser.add_argument(
        "--rolling-window-days",
        type=int,
        default=0,
        help="If set, compare the strategy to SPY over rolling windows of this many trading days.",
    )
    parser.add_argument(
        "--rolling-step-days",
        type=int,
        default=21,
        help="Trading-day step between rolling comparison windows.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a simple parameter sweep in backtest mode.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top parameter combinations to show in sweep mode.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit market orders to Alpaca. Default behavior is dry-run output only.",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached Alpaca data when available.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh cached Alpaca data from the API.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Use cached data only and never call Alpaca.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Explicit dry-run mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    alpaca: AlpacaClient | None = None
    yfinance_max_workers = getattr(args, "yfinance_max_workers", 10)
    yfinance_retry_delay = getattr(args, "yfinance_retry_delay", 1.0)
    yfinance_symbol_delay = getattr(args, "yfinance_symbol_delay", 0.02)
    csv_dir = getattr(args, "csv_dir", ".cache/csv")
    stockanalysis_start = getattr(args, "stockanalysis_start", "1980-01-01")
    stockanalysis_end = getattr(args, "stockanalysis_end", None)
    factor_top_k = getattr(args, "factor_top_k", 1)
    if args.trailing_stop is not None and args.trailing_stop <= 0:
        raise ValueError("--trailing-stop must be greater than 0.")
    if factor_top_k <= 0:
        raise ValueError("--factor-top-k must be greater than 0.")
    if yfinance_max_workers < 1:
        raise ValueError("--yfinance-max-workers must be at least 1.")
    if yfinance_retry_delay < 0:
        raise ValueError("--yfinance-retry-delay cannot be negative.")
    if yfinance_symbol_delay < 0:
        raise ValueError("--yfinance-symbol-delay cannot be negative.")

    # Dynamic universe is only useful for live/dry-run trading, not backtesting
    if args.dynamic_universe and args.backtest_days > 0:
        raise SystemExit("--dynamic-universe cannot be used with --backtest-days")
    if args.dynamic_universe and args.rolling_window_days > 0:
        raise SystemExit("--dynamic-universe cannot be used with --rolling-window-days")

    # Build model inputs from either a static file or a dynamic universe
    if args.dynamic_universe:
        kwargs = {}
        if args.ticker_basket:
            kwargs["ticker_basket"] = args.ticker_basket
        fetched = fetch_ticker_dict(**kwargs)
        dynamic_symbols = set(fetched["symbols"])
        asset_classes = dict(fetched["asset_classes"])

        # Merge with currently held positions so the model can decide to exit
        # even if a stock gets delisted between universe updates.
        alpaca = AlpacaClient(AlpacaConfig.from_env())
        positions = alpaca.get_positions(
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        held_symbols = {p.symbol for p in positions}
        all_symbols = sorted(dynamic_symbols | held_symbols)

        # Ensure held symbols have an asset class entry
        for sym in held_symbols:
            if sym not in asset_classes:
                asset_classes[sym] = f"{sym} (Unknown)"

        # Filter out stocks that haven't been trading long enough for the lookback window.
        # If the user didn't specify a date, calculate the exact date from trading days.
        cutoff = None
        if args.filter_before:
            cutoff = datetime.strptime(args.filter_before, "%Y-%m-%d")
        else:
            try:
                cutoff = _calculate_trading_date_offset(args.lookback_days)
            except ValueError:
                pass  # Ignore if we can't look back far enough (unlikely)

        if cutoff:
            all_symbols = filter_tickers_before(all_symbols, cutoff)

        model = ModelInputs(
            symbols=all_symbols,
            expected_returns=None,
            covariance=None,
            asset_classes=asset_classes,
            class_min_weights={},
            class_max_weights={},
        )
    else:
        model = load_model_inputs(args.model)

    opt_config = OptimizationConfig(
        risk_aversion=args.risk_aversion,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        rebalance_threshold=args.rebalance_threshold,
        turnover_penalty=args.turnover_penalty,
        force_full_investment=not args.allow_cash,
        min_cash_weight=args.min_cash_weight,
        max_turnover=args.max_turnover,
        min_invested_weight=args.min_invested_weight,
        class_min_weights=model.class_min_weights,
        class_max_weights=model.class_max_weights,
    )

    constrained_class_names = list(model.class_min_weights) + [
        name for name in model.class_max_weights if name not in model.class_min_weights
    ]
    asset_class_matrix = build_asset_class_matrix(
        symbols=model.symbols,
        asset_classes=model.asset_classes,
        class_names=constrained_class_names,
    )

    if args.backtest_days > 0:
        total_days = args.lookback_days + args.backtest_days + 1
        benchmark_symbols = [
            symbol for symbol in getattr(args, "benchmark", []) if symbol
        ]
        closes_for_benchmarks: dict[str, list[float]] = {}
        symbols_for_benchmarks = model.symbols
        if args.data_source in {"yfinance", "csv", "csv+yfinance", "stockanalysis"}:
            requested_symbols = list(dict.fromkeys([*model.symbols, *benchmark_symbols]))
            if args.data_source == "csv+yfinance":
                csv_write_yfinance_compatible_caches(
                    csv_dir=csv_dir,
                    symbols=requested_symbols,
                )
            if args.data_source in {"yfinance", "csv+yfinance"}:
                closes_by_symbol = yf_fetch_closes(
                    requested_symbols,
                    period="max",
                    use_cache=args.use_cache or args.data_source == "csv+yfinance",
                    refresh_cache=(
                        args.refresh_cache and args.data_source != "csv+yfinance"
                    ),
                    offline=args.offline,
                    max_workers=yfinance_max_workers,
                    retry_delay=yfinance_retry_delay,
                    symbol_delay=yfinance_symbol_delay,
                )
            else:
                if args.data_source == "stockanalysis":
                    closes_by_symbol = stockanalysis_fetch_closes(
                        requested_symbols,
                        start=stockanalysis_start,
                        end=stockanalysis_end,
                        use_cache=args.use_cache,
                        refresh_cache=args.refresh_cache,
                        offline=args.offline,
                    )
                elif getattr(args, "csv_write_json_cache", False):
                    csv_write_json_caches(csv_dir=csv_dir, symbols=requested_symbols)
                    closes_by_symbol = csv_fetch_closes(
                        requested_symbols, csv_dir=csv_dir
                    )
                else:
                    closes_by_symbol = csv_fetch_closes(
                        requested_symbols, csv_dir=csv_dir
                    )
            # Trim to the requested total_days from the most recent
            for s in closes_by_symbol:
                closes_by_symbol[s] = closes_by_symbol[s][-total_days:]
            closes_for_benchmarks = closes_by_symbol
            symbols_for_benchmarks = requested_symbols
            closes_by_symbol = {
                symbol: closes_by_symbol[symbol] for symbol in model.symbols
            }
        else:
            if benchmark_symbols:
                raise ValueError(
                    "--benchmark is only supported with --data-source yfinance, csv, or stockanalysis."
                )
            if alpaca is None:
                alpaca = AlpacaClient(AlpacaConfig.from_env())
            closes_by_symbol = alpaca.get_daily_closes_for_period(
                model.symbols,
                total_days,
                use_cache=args.use_cache,
                refresh_cache=args.refresh_cache,
                offline=args.offline,
            )
        _validate_backtest_history(
            closes_by_symbol,
            lookback_days=args.lookback_days,
            backtest_days=args.backtest_days,
        )
        if args.sweep:
            if args.strategy in {"dual-momentum", "factor-momentum"}:
                raise ValueError(
                    "Sweep mode is only implemented for the mean-variance path."
                )
            risk_grid = [1.0, 2.0, 4.0]
            cash_grid = [0.05, 0.10, 0.20]
            invested_grid = [0.20, 0.30, 0.40]
            turnover_grid = [0.02, 0.05]
            momentum_grid = [42, 63, 84]

            grid_params = list(
                itertools.product(
                    risk_grid,
                    cash_grid,
                    invested_grid,
                    turnover_grid,
                    momentum_grid,
                )
            )

            try:
                with ProcessPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            _run_sweep_point,
                            symbols=model.symbols,
                            closes_by_symbol=closes_by_symbol,
                            lookback_days=args.lookback_days,
                            rebalance_every=args.rebalance_every,
                            return_model=args.return_model,
                            mean_shrinkage=args.mean_shrinkage,
                            min_weight=args.min_weight,
                            max_weight=args.max_weight,
                            rebalance_threshold=args.rebalance_threshold,
                            max_turnover=args.max_turnover,
                            class_min_weights=model.class_min_weights,
                            class_max_weights=model.class_max_weights,
                            risk_aversion=risk_aversion,
                            min_cash_weight=min_cash_weight,
                            min_invested_weight=min_invested_weight,
                            turnover_penalty=turnover_penalty,
                            momentum_window=momentum_window,
                            asset_class_matrix=(
                                asset_class_matrix if constrained_class_names else None
                            ),
                        )
                        for risk_aversion, min_cash_weight, min_invested_weight, turnover_penalty, momentum_window in grid_params
                    ]
                    outcomes = [future.result() for future in futures]
            except (NotImplementedError, PermissionError, OSError):
                outcomes = [
                    _run_sweep_point(
                        symbols=model.symbols,
                        closes_by_symbol=closes_by_symbol,
                        lookback_days=args.lookback_days,
                        rebalance_every=args.rebalance_every,
                        return_model=args.return_model,
                        mean_shrinkage=args.mean_shrinkage,
                        min_weight=args.min_weight,
                        max_weight=args.max_weight,
                        rebalance_threshold=args.rebalance_threshold,
                        max_turnover=args.max_turnover,
                        class_min_weights=model.class_min_weights,
                        class_max_weights=model.class_max_weights,
                        risk_aversion=risk_aversion,
                        min_cash_weight=min_cash_weight,
                        min_invested_weight=min_invested_weight,
                        turnover_penalty=turnover_penalty,
                        momentum_window=momentum_window,
                        asset_class_matrix=(
                            asset_class_matrix if constrained_class_names else None
                        ),
                    )
                    for risk_aversion, min_cash_weight, min_invested_weight, turnover_penalty, momentum_window in grid_params
                ]

            results: list[dict[str, float | int | dict[str, float]]] = []
            skipped: list[dict[str, float | int | str]] = []

            for outcome in outcomes:
                if isinstance(outcome, tuple):
                    skipped.append(json.loads(outcome[1]))
                else:
                    results.append(outcome)

            results.sort(
                key=lambda item: (
                    float(item["annualized_return"]),
                    -float(item["max_drawdown"]),
                ),
                reverse=True,
            )
            print(
                json.dumps(
                    {
                        "symbols": model.symbols,
                        "sweep": {
                            "days": args.backtest_days,
                            "rebalance_every": args.rebalance_every,
                            "top_n": args.top_n,
                            "tested": len(results),
                            "skipped": len(skipped),
                            "results": results[: args.top_n],
                            "skipped_examples": skipped[: min(5, len(skipped))],
                        },
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return
        if args.strategy == "dual-momentum":
            backtest = run_dual_momentum_backtest(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                rebalance_every=args.rebalance_every,
                top_k=args.top_k,
                absolute_threshold=args.absolute_momentum_threshold,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
                trailing_stop=args.trailing_stop,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
            )
        elif args.strategy == "factor-momentum":
            backtest = run_factor_momentum_backtest(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                rebalance_every=args.rebalance_every,
                top_k=args.top_k,
                factor_top_k=factor_top_k,
                absolute_threshold=args.absolute_momentum_threshold,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
                trailing_stop=args.trailing_stop,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
            )
        else:
            backtest = run_backtest(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                lookback_days=args.lookback_days,
                rebalance_every=args.rebalance_every,
                return_model=args.return_model,
                mean_shrinkage=args.mean_shrinkage,
                momentum_window=args.momentum_window,
                opt_config=opt_config,
                asset_class_matrix=(
                    asset_class_matrix if constrained_class_names else None
                ),
            )
        latest_weights = clean_weights(backtest.latest_weights)
        rolling_comparison = None
        if args.rolling_window_days > 0:
            rolling_comparison = rolling_window_comparison(
                strategy=args.strategy,
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                window_days=args.rolling_window_days,
                step_days=args.rolling_step_days,
                rebalance_every=args.rebalance_every,
                return_model=args.return_model,
                mean_shrinkage=args.mean_shrinkage,
                momentum_window=args.momentum_window,
                opt_config=opt_config,
                asset_class_matrix=(
                    asset_class_matrix if constrained_class_names else None
                ),
                top_k=args.top_k,
                factor_top_k=factor_top_k,
                absolute_threshold=args.absolute_momentum_threshold,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
            )
        benchmark_results = {
            "spy": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={"SPY": 1.0},
                start_day=args.lookback_days,
            ),
            "qqq": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={"QQQ": 1.0},
                start_day=args.lookback_days,
            ),
            "sixty_forty_spy_tlt": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={"SPY": 0.6, "TLT": 0.4},
                start_day=args.lookback_days,
            ),
            "equal_weight": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={
                    symbol: 1.0 / len(model.symbols) for symbol in model.symbols
                },
                start_day=args.lookback_days,
            ),
            "half_spy_half_cash": run_fixed_weight_benchmark(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                weights_by_symbol={"SPY": 0.5},
                start_day=args.lookback_days,
            ),
        }
        for symbol in benchmark_symbols:
            key = f"benchmark_{_json_key(symbol)}"
            benchmark_results[key] = run_fixed_weight_benchmark(
                symbols=symbols_for_benchmarks,
                closes_by_symbol=closes_for_benchmarks,
                weights_by_symbol={symbol: 1.0},
                start_day=args.lookback_days,
            )
        result = {
            "symbols": model.symbols,
            "backtest": {
                "strategy": args.strategy,
                "dual_momentum_weighting": (
                    args.dual_momentum_weighting
                    if args.strategy in {"dual-momentum", "factor-momentum"}
                    else None
                ),
                "factor_top_k": (
                    factor_top_k if args.strategy == "factor-momentum" else None
                ),
                "target_vol": args.target_vol,
                "vol_window": (
                    args.vol_window
                    if args.strategy in {"dual-momentum", "factor-momentum"}
                    else None
                ),
                "max_single_weight": args.max_single_weight,
                "basket_opt": args.basket_opt,
                "basket_risk_aversion": (
                    args.basket_risk_aversion
                    if args.strategy in {"dual-momentum", "factor-momentum"}
                    and args.basket_opt
                    else None
                ),
                "days": args.backtest_days,
                "rebalance_every": args.rebalance_every,
                "final_value": round(float(backtest.final_value), 6),
                "total_return": round(float(backtest.total_return), 6),
                "annualized_return": round(float(backtest.annualized_return), 6),
                "annualized_volatility": round(
                    float(backtest.annualized_volatility), 6
                ),
                "max_drawdown": round(float(backtest.max_drawdown), 6),
                "rebalance_count": backtest.rebalance_count,
                "average_turnover": round(float(backtest.average_turnover), 6),
                "daily_values": [round(v, 6) for v in backtest.daily_values],
            },
            "latest_target_weights": {
                symbol: round(float(weight), 6)
                for symbol, weight in zip(model.symbols, latest_weights, strict=True)
            },
            "latest_cash_weight": round(max(0.0, 1.0 - float(latest_weights.sum())), 6),
            "latest_asset_class_exposures": asset_class_exposures(
                symbols=model.symbols,
                weights=latest_weights,
                asset_classes=model.asset_classes,
            ),
            "benchmarks": benchmark_results,
        }
        if rolling_comparison is not None:
            result["rolling_vs_spy"] = rolling_comparison
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    if alpaca is None:
        alpaca = AlpacaClient(AlpacaConfig.from_env())

    # Even in dry-run mode we fetch live account state for the rebalance path,
    # because the order plan depends on current equity, holdings, and prices.
    account = alpaca.get_account(
        use_cache=args.use_cache, refresh_cache=args.refresh_cache, offline=args.offline
    )
    positions = alpaca.get_positions(
        use_cache=args.use_cache, refresh_cache=args.refresh_cache, offline=args.offline
    )
    existing_weights_map = current_weights(model.symbols, account, positions)
    existing_weights = np.array(
        [existing_weights_map[symbol] for symbol in model.symbols], dtype=float
    )
    scaled_turnover_penalty = effective_turnover_penalty(opt_config, existing_weights)
    target_weights: np.ndarray | None = None

    if args.estimate_from_history:
        history_days = args.lookback_days
        if args.strategy in {"dual-momentum", "factor-momentum"}:
            history_days = max(history_days, args.lookback_days + 1)
            if args.target_vol is not None:
                history_days = max(history_days, args.vol_window + 1)
        closes_by_symbol = alpaca.get_daily_closes(
            model.symbols,
            history_days,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )

        # Filter out symbols with insufficient history.
        # This handles cases where Alpaca (IEX data) or Yahoo has gaps,
        # or stocks are new IPOs/SPACs with limited history.
        original_count = len(closes_by_symbol)
        closes_by_symbol = {
            s: bars
            for s, bars in closes_by_symbol.items()
            if len(bars) >= history_days
        }
        dropped_count = original_count - len(closes_by_symbol)
        if dropped_count > 0:
            print(
                f"Warning: Dropped {dropped_count} symbols with insufficient history "
                f"(< {history_days} prices).",
                file=sys.stderr,
            )
            # Update model to match available data
            model = ModelInputs(
                symbols=list(closes_by_symbol.keys()),
                expected_returns=model.expected_returns,
                covariance=model.covariance,
                asset_classes=model.asset_classes,
                class_min_weights=model.class_min_weights,
                class_max_weights=model.class_max_weights,
            )
            # Recompute existing weights for the filtered symbol list
            existing_weights_map = current_weights(model.symbols, account, positions)
            existing_weights = np.array(
                [existing_weights_map[symbol] for symbol in model.symbols], dtype=float
            )
            asset_class_matrix = build_asset_class_matrix(
                symbols=model.symbols,
                asset_classes=model.asset_classes,
                class_names=constrained_class_names,
            )
            scaled_turnover_penalty = effective_turnover_penalty(
                opt_config, existing_weights
            )

        if args.strategy == "dual-momentum":
            dm_weights = compute_dual_momentum_weights(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                top_k=args.top_k,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                absolute_threshold=args.absolute_momentum_threshold,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
                trailing_stop=None,
            )
            target_weights = np.array(
                [dm_weights[s] for s in model.symbols], dtype=float
            )
            estimation_metadata = {
                "method": "dual_momentum",
                "lookback_days": args.lookback_days,
                "top_k": args.top_k,
                "weighting": args.dual_momentum_weighting,
                "vol_window": args.vol_window,
            }
        elif args.strategy == "factor-momentum":
            fm_weights = compute_factor_momentum_weights(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                asset_classes=model.asset_classes,
                lookback_days=args.lookback_days,
                top_k=args.top_k,
                factor_top_k=factor_top_k,
                weighting=args.dual_momentum_weighting,
                softmax_temperature=args.softmax_temperature,
                absolute_threshold=args.absolute_momentum_threshold,
                basket_opt=args.basket_opt,
                basket_risk_aversion=args.basket_risk_aversion,
                target_vol=args.target_vol,
                max_single_weight=args.max_single_weight,
                vol_window=args.vol_window,
            )
            target_weights = np.array(
                [fm_weights[s] for s in model.symbols], dtype=float
            )
            estimation_metadata = {
                "method": "factor_momentum",
                "lookback_days": args.lookback_days,
                "top_k": args.top_k,
                "factor_top_k": factor_top_k,
                "weighting": args.dual_momentum_weighting,
                "vol_window": args.vol_window,
            }
        elif args.return_model == "momentum":
            estimated = estimate_inputs_from_momentum(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                mean_shrinkage=args.mean_shrinkage,
                momentum_window=args.momentum_window,
            )
            expected_returns = estimated.expected_returns
            covariance = estimated.covariance
            estimation_metadata = {
                "method": "alpaca_daily_bars",
                "return_model": args.return_model,
                "lookback_days": args.lookback_days,
                "mean_shrinkage": args.mean_shrinkage,
                "observations": estimated.observations,
                "momentum_window": min(args.momentum_window, args.lookback_days),
            }
        elif args.return_model == "black-litterman":
            estimated = estimate_inputs_from_black_litterman(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                momentum_window=args.momentum_window,
                mean_shrinkage=args.mean_shrinkage,
            )
            expected_returns = estimated.expected_returns
            covariance = estimated.covariance
            estimation_metadata = {
                "method": "black_litterman",
                "lookback_days": args.lookback_days,
                "mean_shrinkage": args.mean_shrinkage,
                "observations": estimated.observations,
                "momentum_window": min(args.momentum_window, args.lookback_days),
            }
        elif args.return_model == "risk-parity":
            estimated = estimate_inputs_risk_parity(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                lookback_days=args.lookback_days,
            )
            target_weights = project_weights(
                target_weights=estimated.weights,
                config=opt_config,
                current_weights=existing_weights,
                asset_class_matrix=(
                    asset_class_matrix if constrained_class_names else None
                ),
            )
            estimation_metadata = {
                "method": "risk_parity",
                "lookback_days": args.lookback_days,
                "observations": estimated.observations,
            }
        else:
            estimated = estimate_inputs_from_prices(
                symbols=model.symbols,
                closes_by_symbol=closes_by_symbol,
                mean_shrinkage=args.mean_shrinkage,
            )
            expected_returns = estimated.expected_returns
            covariance = estimated.covariance
            estimation_metadata = {
                "method": "alpaca_daily_bars",
                "return_model": args.return_model,
                "lookback_days": args.lookback_days,
                "mean_shrinkage": args.mean_shrinkage,
                "observations": estimated.observations,
            }
    else:
        if model.expected_returns is None or model.covariance is None:
            raise ValueError(
                "Model file must include expected_returns and covariance unless "
                "--estimate-from-history is used."
            )
        expected_returns = model.expected_returns
        covariance = model.covariance
        estimation_metadata = {"method": "model_file"}

    if args.strategy not in {"dual-momentum", "factor-momentum"} and target_weights is None:
        target_weights = optimize_weights(
            expected_returns=expected_returns,
            covariance=covariance,
            config=opt_config,
            current_weights=existing_weights,
            asset_class_matrix=asset_class_matrix if constrained_class_names else None,
        )
    if target_weights is None:
        raise RuntimeError("Failed to produce target weights for the requested strategy.")
    target_weights = clean_weights(target_weights)
    target_cash_weight = max(0.0, 1.0 - float(target_weights.sum()))
    current_cash_weight = max(0.0, 1.0 - float(existing_weights.sum()))
    realized_turnover = float(np.abs(target_weights - existing_weights).sum())
    current_weights_clean = clean_weights(existing_weights)

    # Fetch open orders to prevent double-submission
    open_orders = alpaca.get_open_orders()
    symbols_with_open_orders = {
        str(order.get("symbol"))
        for order in open_orders
        if order.get("symbol") in model.symbols
    }
    symbols_needing_prices = [
        symbol
        for symbol, target_weight, current_weight in zip(
            model.symbols, target_weights, existing_weights, strict=True
        )
        if abs(float(target_weight - current_weight)) >= args.rebalance_threshold
        or symbol in symbols_with_open_orders
    ]

    # Convert target weights into dollar notional orders using latest prices.
    # Only request symbols that could actually produce an order; refreshing a
    # 500-symbol dynamic universe can otherwise timeout on one slow quote.
    latest_prices = (
        alpaca.get_latest_prices(
            symbols_needing_prices,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        if symbols_needing_prices
        else {}
    )

    plan = build_order_plan(
        symbols=model.symbols,
        target_weights=target_weights.tolist(),
        account=account,
        positions=positions,
        latest_prices=latest_prices,
        config=opt_config,
        open_orders=open_orders,
    )
    changed_symbols = [item.symbol for item in plan]
    trailing_stop_cancellations = (
        [
            {
                "id": order.get("id"),
                "symbol": order.get("symbol"),
                "qty": order.get("qty"),
                "trail_percent": order.get("trail_percent"),
            }
            for order in open_orders
            if order.get("symbol") in changed_symbols
            and order.get("type") == "trailing_stop"
            and order.get("side") == "sell"
        ]
        if args.trailing_stop is not None
        else []
    )
    trailing_stop_plan = (
        build_trailing_stop_plan(
            symbols=model.symbols,
            target_weights=target_weights.tolist(),
            positions=positions,
            open_orders=open_orders,
            trailing_stop=args.trailing_stop,
            rebalance_threshold=args.rebalance_threshold,
        )
        if args.trailing_stop is not None
        else []
    )

    result = {
        "symbols": model.symbols,
        "estimation": estimation_metadata,
        "optimization": {
            "risk_aversion": args.risk_aversion,
            "turnover_penalty": args.turnover_penalty,
            "effective_turnover_penalty": round(float(scaled_turnover_penalty), 6),
            "max_turnover": args.max_turnover,
            "min_weight": args.min_weight,
            "max_weight": args.max_weight,
            "rebalance_threshold": args.rebalance_threshold,
            "allow_cash": args.allow_cash,
            "min_cash_weight": args.min_cash_weight,
            "min_invested_weight": args.min_invested_weight,
            "class_min_weights": model.class_min_weights,
            "class_max_weights": model.class_max_weights,
        },
        "target_weights": {
            symbol: round(float(weight), 6)
            for symbol, weight in zip(model.symbols, target_weights, strict=True)
        },
        "current_weights": {
            symbol: round(float(weight), 6)
            for symbol, weight in zip(model.symbols, current_weights_clean, strict=True)
        },
        "expected_returns": (
            {
                symbol: round(float(value), 6)
                for symbol, value in zip(model.symbols, expected_returns, strict=True)
            }
            if args.strategy != "dual-momentum"
            and args.strategy != "factor-momentum"
            else None
        ),
        "cash": {
            "current_weight": round(current_cash_weight, 6),
            "target_weight": round(target_cash_weight, 6),
        },
        "asset_class_exposures": {
            "current": asset_class_exposures(
                symbols=model.symbols,
                weights=current_weights_clean,
                asset_classes=model.asset_classes,
            ),
            "target": asset_class_exposures(
                symbols=model.symbols,
                weights=target_weights,
                asset_classes=model.asset_classes,
            ),
        },
        "turnover": {
            "proposed": round(realized_turnover, 6),
            "max_allowed": args.max_turnover,
        },
        "orders": [asdict(item) for item in plan],
        "trailing_stop_cancellations": trailing_stop_cancellations,
        "trailing_stop_orders": [asdict(item) for item in trailing_stop_plan],
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.submit:
        canceled_trailing_stops = (
            alpaca.cancel_open_trailing_stops(changed_symbols, open_orders=open_orders)
            if args.trailing_stop is not None
            else []
        )
        submitted_orders = alpaca.submit_order_plan(plan)
        fill_statuses: list[dict] = []
        submitted_trailing_stops: list[dict] = []
        if args.trailing_stop is not None:
            fill_statuses = alpaca.wait_for_submitted_orders(submitted_orders)
            refreshed_positions = alpaca.get_positions()
            refreshed_open_orders = alpaca.get_open_orders()
            trailing_stop_plan = build_trailing_stop_plan(
                symbols=model.symbols,
                target_weights=target_weights.tolist(),
                positions=refreshed_positions,
                open_orders=refreshed_open_orders,
                trailing_stop=args.trailing_stop,
                rebalance_threshold=args.rebalance_threshold,
            )
            submitted_trailing_stops = alpaca.submit_trailing_stop_plan(
                trailing_stop_plan
            )
        print(format_order_plans(plan))
        if args.trailing_stop is not None:
            print(
                json.dumps(
                    {
                        "order_fill_statuses": fill_statuses,
                        "canceled_trailing_stops": canceled_trailing_stops,
                        "trailing_stop_orders": [
                            asdict(item) for item in trailing_stop_plan
                        ],
                        "submitted_trailing_stops": submitted_trailing_stops,
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
