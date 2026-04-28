from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_opt.backtest import TRADING_DAYS_PER_YEAR
from portfolio_opt.runtime import configure_local_cache_dirs

from .backtest import (
    format_backtest,
    run_cvxportfolio_backtest,
    run_cvxportfolio_sweep,
    run_framework_comparison,
)

configure_local_cache_dirs()


def run_from_args(args: argparse.Namespace) -> dict:
    if args.compare_custom:
        custom_config = json.loads(Path("examples/custom_best_preset.json").read_text())
        cvxportfolio_config = {
            "risk_aversion": args.risk_aversion,
            "mean_shrinkage": args.mean_shrinkage,
            "momentum_window": args.momentum_window,
            "min_cash_weight": args.min_cash_weight,
            "min_invested_weight": args.min_invested_weight,
            "max_weight": args.max_weight,
            "core_symbol": args.core_symbol,
            "core_weight": args.core_weight,
            "target_volatility": args.target_volatility,
            "max_leverage": args.max_leverage,
            "benchmark_symbol": args.benchmark_symbol,
            "benchmark_weight": args.benchmark_weight,
            "linear_trade_cost": args.linear_trade_cost,
            "planning_horizon": args.planning_horizon,
            "trading_days_per_year": args.trading_days_per_year,
        }
        return run_framework_comparison(
            model_path=args.model,
            lookback_days=args.lookback_days,
            backtest_days=args.backtest_days,
            cvxportfolio_config=cvxportfolio_config,
            custom_config=custom_config,
            trading_days_per_year=args.trading_days_per_year,
            data_source=args.data_source,
            csv_dir=args.csv_dir,
            csv_write_json_cache=args.csv_write_json_cache,
            stockanalysis_start=args.stockanalysis_start,
            stockanalysis_end=args.stockanalysis_end,
            yfinance_max_workers=args.yfinance_max_workers,
            yfinance_retry_delay=args.yfinance_retry_delay,
            yfinance_symbol_delay=args.yfinance_symbol_delay,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
    if args.sweep:
        return run_cvxportfolio_sweep(
            model_path=args.model,
            lookback_days=args.lookback_days,
            backtest_days=args.backtest_days,
            top_n=args.top_n,
            linear_trade_cost=args.linear_trade_cost,
            planning_horizon=args.planning_horizon,
            core_symbol=args.core_symbol,
            core_weight=args.core_weight,
            target_volatility=args.target_volatility,
            max_leverage=args.max_leverage,
            benchmark_symbol=args.benchmark_symbol,
            benchmark_weight=args.benchmark_weight,
            trading_days_per_year=args.trading_days_per_year,
            data_source=args.data_source,
            csv_dir=args.csv_dir,
            csv_write_json_cache=args.csv_write_json_cache,
            stockanalysis_start=args.stockanalysis_start,
            stockanalysis_end=args.stockanalysis_end,
            yfinance_max_workers=args.yfinance_max_workers,
            yfinance_retry_delay=args.yfinance_retry_delay,
            yfinance_symbol_delay=args.yfinance_symbol_delay,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
    return run_cvxportfolio_backtest(
        model_path=args.model,
        lookback_days=args.lookback_days,
        backtest_days=args.backtest_days,
        risk_aversion=args.risk_aversion,
        min_cash_weight=args.min_cash_weight,
        min_invested_weight=args.min_invested_weight,
        max_weight=args.max_weight,
        core_symbol=args.core_symbol,
        core_weight=args.core_weight,
        target_volatility=args.target_volatility,
        max_leverage=args.max_leverage,
        benchmark_symbol=args.benchmark_symbol,
        benchmark_weight=args.benchmark_weight,
        mean_shrinkage=args.mean_shrinkage,
        momentum_window=args.momentum_window,
        linear_trade_cost=args.linear_trade_cost,
        planning_horizon=args.planning_horizon,
        rolling_window_days=args.rolling_window_days,
        rolling_step_days=args.rolling_step_days,
        trading_days_per_year=args.trading_days_per_year,
        data_source=args.data_source,
        csv_dir=args.csv_dir,
        csv_write_json_cache=args.csv_write_json_cache,
        stockanalysis_start=args.stockanalysis_start,
        stockanalysis_end=args.stockanalysis_end,
        yfinance_max_workers=args.yfinance_max_workers,
        yfinance_retry_delay=args.yfinance_retry_delay,
        yfinance_symbol_delay=args.yfinance_symbol_delay,
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
        offline=args.offline,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a minimal cvxportfolio backtest on the repo universe."
    )
    parser.add_argument(
        "--model", required=True, help="Path to the model or universe JSON file."
    )
    parser.add_argument("--lookback-days", type=int, default=126)
    parser.add_argument("--backtest-days", type=int, default=252)
    parser.add_argument("--risk-aversion", type=float, default=1.0)
    parser.add_argument("--mean-shrinkage", type=float, default=0.75)
    parser.add_argument("--momentum-window", type=int, default=63)
    parser.add_argument("--min-cash-weight", type=float, default=0.10)
    parser.add_argument("--min-invested-weight", type=float, default=0.30)
    parser.add_argument("--max-weight", type=float, default=0.35)
    parser.add_argument("--core-symbol", default=None)
    parser.add_argument("--core-weight", type=float, default=0.0)
    parser.add_argument("--target-volatility", type=float, default=None)
    parser.add_argument("--max-leverage", type=float, default=None)
    parser.add_argument("--benchmark-symbol", default=None)
    parser.add_argument("--benchmark-weight", type=float, default=1.0)
    parser.add_argument("--linear-trade-cost", type=float, default=0.0)
    parser.add_argument("--planning-horizon", type=int, default=1)
    parser.add_argument(
        "--data-source",
        choices=("alpaca", "yfinance", "csv", "csv+yfinance", "stockanalysis"),
        default="alpaca",
        help="Source for historical close data.",
    )
    parser.add_argument(
        "--csv-dir",
        default=".cache/csv",
        help="Directory of local OHLCV CSV files when --data-source csv is used.",
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
    parser.add_argument("--yfinance-max-workers", type=int, default=10)
    parser.add_argument("--yfinance-retry-delay", type=float, default=1.0)
    parser.add_argument("--yfinance-symbol-delay", type=float, default=0.02)
    parser.add_argument(
        "--trading-days-per-year",
        type=int,
        default=TRADING_DAYS_PER_YEAR,
        help="Trading sessions per year used for annualized metrics.",
    )
    parser.add_argument("--rolling-window-days", type=int, default=0)
    parser.add_argument("--rolling-step-days", type=int, default=21)
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run a simple parameter sweep for the cvxportfolio path.",
    )
    parser.add_argument(
        "--compare-custom",
        action="store_true",
        help="Compare the current cvxportfolio config against the repo's custom baseline preset.",
    )
    parser.add_argument("--top-n", type=int, default=5)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(format_backtest(run_from_args(args)))


if __name__ == "__main__":
    main()
