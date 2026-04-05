from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_opt.runtime import configure_local_cache_dirs

configure_local_cache_dirs()

from .backtest import format_backtest, run_cvxportfolio_backtest, run_cvxportfolio_sweep, run_framework_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal cvxportfolio backtest on the repo universe.")
    parser.add_argument("--model", required=True, help="Path to the model or universe JSON file.")
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
    parser.add_argument("--linear-trade-cost", type=float, default=0.0)
    parser.add_argument("--planning-horizon", type=int, default=1)
    parser.add_argument("--rolling-window-days", type=int, default=0)
    parser.add_argument("--rolling-step-days", type=int, default=21)
    parser.add_argument("--sweep", action="store_true", help="Run a simple parameter sweep for the cvxportfolio path.")
    parser.add_argument(
        "--compare-custom",
        action="store_true",
        help="Compare the current cvxportfolio config against the repo's custom baseline preset.",
    )
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--use-cache", action="store_true", help="Use cached Alpaca data when available.")
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh cached Alpaca data from the API.")
    parser.add_argument("--offline", action="store_true", help="Use cached data only and never call Alpaca.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
            "linear_trade_cost": args.linear_trade_cost,
            "planning_horizon": args.planning_horizon,
        }
        result = run_framework_comparison(
            model_path=args.model,
            lookback_days=args.lookback_days,
            backtest_days=args.backtest_days,
            cvxportfolio_config=cvxportfolio_config,
            custom_config=custom_config,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        print(format_backtest(result))
        return
    if args.sweep:
        result = run_cvxportfolio_sweep(
            model_path=args.model,
            lookback_days=args.lookback_days,
            backtest_days=args.backtest_days,
            top_n=args.top_n,
            linear_trade_cost=args.linear_trade_cost,
            planning_horizon=args.planning_horizon,
            core_symbol=args.core_symbol,
            core_weight=args.core_weight,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            offline=args.offline,
        )
        print(format_backtest(result))
        return
    result = run_cvxportfolio_backtest(
        model_path=args.model,
        lookback_days=args.lookback_days,
        backtest_days=args.backtest_days,
        risk_aversion=args.risk_aversion,
        min_cash_weight=args.min_cash_weight,
        min_invested_weight=args.min_invested_weight,
        max_weight=args.max_weight,
        core_symbol=args.core_symbol,
        core_weight=args.core_weight,
        mean_shrinkage=args.mean_shrinkage,
        momentum_window=args.momentum_window,
        linear_trade_cost=args.linear_trade_cost,
        planning_horizon=args.planning_horizon,
        rolling_window_days=args.rolling_window_days,
        rolling_step_days=args.rolling_step_days,
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
        offline=args.offline,
    )
    print(format_backtest(result))


if __name__ == "__main__":
    main()
