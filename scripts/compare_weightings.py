#!/usr/bin/env python3
"""Compare dual-momentum weighting modes and basket-mean-variance vs benchmarks.

Usage:
    uv run python scripts/compare_weightings.py                              # defaults
    uv run python scripts/compare_weightings.py --model examples/sector_universe.json
    uv run python scripts/compare_weightings.py --top-k 5 --top-k 2
    uv run python scripts/compare_weightings.py --offline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from portfolio_opt.backtest import align_close_history, run_dual_momentum_backtest
from portfolio_opt.model import load_model_inputs as load_model
from portfolio_opt.yfinance_data import fetch_closes


def _print_row(mode, ret, ann, vol, dd, sharpe, sortino, rebal=0, to=0.0):
    print(
        f"{mode:<22s} {ret:7.2f}% {ann:6.2f}% {vol:5.2f}% {dd:6.2f}% "
        f"{sharpe:6.2f} {sortino:7.2f} {rebal:5d} {to:6.3f}"
    )


def _run_dm(
    symbols, closes, ac, args, weighting, top_k, basket_opt=None, basket_ra=1.0
):
    return run_dual_momentum_backtest(
        symbols,
        closes,
        ac,
        lookback_days=args.lookback_days,
        rebalance_every=args.rebalance_every,
        top_k=top_k,
        absolute_threshold=args.abs_threshold,
        weighting=weighting,
        softmax_temperature=args.softmax_temperature,
        basket_opt=basket_opt,
        basket_risk_aversion=basket_ra,
    )


def _spy_bench(closes, n_bt):
    spy = np.array(closes["SPY"])
    bt = spy[-n_bt:]
    ret = float(bt[-1] / bt[0] - 1.0)
    ann = float((1.0 + ret) ** (252.0 / (len(bt) - 1)) - 1.0)
    vol = float(np.std(np.diff(np.log(bt)))) * np.sqrt(252.0)
    dd = float(np.max(1.0 - bt / np.maximum.accumulate(bt)))
    return ret, ann, vol, dd, ann / vol if vol > 0 else 0.0


def main():
    p = argparse.ArgumentParser(description="Compare weighting modes")
    p.add_argument("--model", default="examples/sector_universe_pre2016.json")
    p.add_argument("--top-k", type=int, nargs="+", default=[2, 5])
    p.add_argument("--lookback-days", type=int, default=252)
    p.add_argument(
        "--backtest-days",
        type=int,
        default=0,
        help="Max backtest length (0 = all available)",
    )
    p.add_argument("--rebalance-every", type=int, default=21)
    p.add_argument("--abs-threshold", type=float, default=0.0)
    p.add_argument("--softmax-temperature", type=float, default=0.05)
    p.add_argument(
        "--basket-risk-aversion", type=float, nargs="+", default=[0.5, 1.0, 2.0]
    )
    p.add_argument("--offline", action="store_true")
    p.add_argument("--use-cache", action="store_true")
    args = p.parse_args()

    model = load_model(args.model)
    symbols = model.symbols
    ac = model.asset_classes

    # Fetch closes directly from yfinance cache, no min_history_days filter
    # (load_close_history's min_history_days drops symbols with shorter caches)
    closes = fetch_closes(
        symbols,
        period="max",
        use_cache=args.use_cache or args.offline,
        offline=args.offline,
    )

    aligned = align_close_history(symbols, closes)
    n_common = len(next(iter(aligned.values())))
    n_bt = n_common - args.lookback_days
    if args.backtest_days > 0:
        n_bt = min(n_bt, args.backtest_days)

    years = n_bt / 252.0
    print(f"Universe: {args.model}  ({len(symbols)} symbols)")
    print(f"Common history: {n_common} days ({n_common / 252:.1f}y)")
    print(
        f"Backtest: {n_bt} days ({years:.1f}y)  "
        f"lookback={args.lookback_days}  rebalance={args.rebalance_every}"
    )

    hdr = (
        f"{'Mode':<22s} {'Ret%':>8s} {'Ann%':>7s} {'Vol%':>6s} "
        f"{'MaxDD%':>7s} {'Sharpe':>6s} {'Sortino':>7s} {'Rebal':>5s} {'T/O':>6s}"
    )
    print("\n" + hdr)
    print("-" * len(hdr))

    weightings = ["equal", "score", "inverse-vol", "softmax"]

    for k in args.top_k:
        print(f"\n--- top_k={k} ---")
        for w in weightings:
            r = _run_dm(symbols, closes, ac, args, w, k)
            sh = (
                r.annualized_return / r.annualized_volatility
                if r.annualized_volatility > 0
                else 0
            )
            _print_row(
                w,
                r.total_return * 100,
                r.annualized_return * 100,
                r.annualized_volatility * 100,
                r.max_drawdown * 100,
                sh,
                r.sortino_ratio,
                r.rebalance_count,
                r.average_turnover,
            )

        for ra in args.basket_risk_aversion:
            r = _run_dm(
                symbols,
                closes,
                ac,
                args,
                "equal",
                k,
                basket_opt="mean-variance",
                basket_ra=ra,
            )
            sh = (
                r.annualized_return / r.annualized_volatility
                if r.annualized_volatility > 0
                else 0
            )
            _print_row(
                f"basket-mv ra={ra}",
                r.total_return * 100,
                r.annualized_return * 100,
                r.annualized_volatility * 100,
                r.max_drawdown * 100,
                sh,
                r.sortino_ratio,
                r.rebalance_count,
                r.average_turnover,
            )

    print("\n--- SPY buy-hold ---")
    spy_ret, spy_ann, spy_vol, spy_dd, spy_sh = _spy_bench(closes, n_bt)
    _print_row(
        "SPY (buy-hold)",
        spy_ret * 100,
        spy_ann * 100,
        spy_vol * 100,
        spy_dd * 100,
        spy_sh,
        0.0,
    )


if __name__ == "__main__":
    main()
