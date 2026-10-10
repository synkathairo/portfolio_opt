"""Compare automatically rebalanced multi-asset ETF strategies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from portfolio_opt.backtest import run_dual_momentum_backtest
from portfolio_opt.fixed_allocation import run_fixed_allocation_backtest
from portfolio_opt.runtime import configure_local_cache_dirs
from portfolio_opt.yfinance_data import fetch_closes

configure_local_cache_dirs()


def _metrics(result: object) -> dict[str, float]:
    return {
        name: float(getattr(result, name))
        for name in (
            "final_value",
            "total_return",
            "annualized_return",
            "annualized_volatility",
            "max_drawdown",
            "sortino_ratio",
            "average_turnover",
        )
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--universe", default="examples/sector_universe_pre2020.json")
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--linear-trade-cost", type=float, default=0.001)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", default=".cache/etf_automatic_research.json")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    universe = json.loads(Path(args.universe).read_text())
    symbols = [str(symbol).upper() for symbol in universe["symbols"]]
    asset_classes = {
        str(symbol).upper(): str(value)
        for symbol, value in universe.get("asset_classes", {}).items()
    }
    closes = fetch_closes(
        symbols,
        period="max",
        use_cache=True,
        offline=args.offline,
    )
    common_length = min(len(closes[symbol]) for symbol in symbols)
    closes = {symbol: closes[symbol][-common_length:] for symbol in symbols}
    results = {
        "dual_momentum": run_dual_momentum_backtest(
            symbols=symbols,
            closes_by_symbol=closes,
            asset_classes=asset_classes,
            lookback_days=args.lookback_days,
            rebalance_every=args.rebalance_every,
            top_k=2,
            absolute_threshold=0.0,
            weighting="equal",
            trailing_stop=0.15,
            linear_trade_cost=args.linear_trade_cost,
        ),
        "trend_inverse_vol_target": run_dual_momentum_backtest(
            symbols=symbols,
            closes_by_symbol=closes,
            asset_classes=asset_classes,
            lookback_days=args.lookback_days,
            rebalance_every=args.rebalance_every,
            top_k=3,
            absolute_threshold=0.0,
            weighting="inverse-vol",
            target_vol=0.15,
            max_single_weight=0.60,
            trailing_stop=0.15,
            linear_trade_cost=args.linear_trade_cost,
        ),
        "spy": run_fixed_allocation_backtest(
            symbols=["SPY"],
            closes_by_symbol={"SPY": closes["SPY"]},
            weights_by_symbol={"SPY": 1.0},
            start_day=args.lookback_days,
            rebalance_every=args.rebalance_every,
            linear_trade_cost=args.linear_trade_cost,
        ),
        "sixty_forty": run_fixed_allocation_backtest(
            symbols=["SPY", "IEF"],
            closes_by_symbol={"SPY": closes["SPY"], "IEF": closes["IEF"]},
            weights_by_symbol={"SPY": 0.60, "IEF": 0.40},
            start_day=args.lookback_days,
            rebalance_every=args.rebalance_every,
            linear_trade_cost=args.linear_trade_cost,
        ),
    }
    output = {
        "universe": args.universe,
        "symbols": symbols,
        "lookback_days": args.lookback_days,
        "rebalance_every": args.rebalance_every,
        "linear_trade_cost": args.linear_trade_cost,
        "strategies": {name: _metrics(result) for name, result in results.items()},
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({"report": str(output_path), "symbols": len(symbols)}))


if __name__ == "__main__":
    main()
