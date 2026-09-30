#!/usr/bin/env python3
"""Run a preliminary SEC-quality research backtest and write a chart.

This script deliberately uses a small, configurable basket by default.  A
current or historical ticker file is not a survivorship-bias-free constituent
database; the report says so explicitly and should not be treated as a final
performance study.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import pandas as pd
import yfinance as yf

from portfolio_opt.backtest import run_dual_momentum_backtest
from portfolio_opt.fixed_allocation import run_fixed_allocation_backtest
from portfolio_opt.quality_backtest import run_quality_backtest
from portfolio_opt.quality_momentum import run_quality_momentum_backtest
from portfolio_opt.regime_adaptive import run_regime_adaptive_backtest
from portfolio_opt.residual_reversion import run_residual_reversion_backtest
from portfolio_opt.runtime import configure_local_cache_dirs
from portfolio_opt.sec_edgar import SecEdgarClient
from portfolio_opt.value_quality_trend import run_value_quality_trend_backtest

configure_local_cache_dirs()
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        default="examples/nasdaq100_sp500_sector_universe_b2016filtered.json",
        help="JSON universe containing symbols and optional asset_classes.",
    )
    parser.add_argument("--max-symbols", type=int, default=50)
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--rebalance-every", type=int, default=21)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--dual-lookback-days",
        type=int,
        default=252,
        help="Lookback used by the dual-momentum comparison.",
    )
    parser.add_argument(
        "--dual-top-k",
        type=int,
        default=2,
        help="Number of holdings used by the dual-momentum comparison.",
    )
    parser.add_argument(
        "--adaptive-lookback-days",
        type=int,
        default=126,
        help="Long trend window used by the regime-adaptive comparison.",
    )
    parser.add_argument(
        "--quality-momentum-lookback-days",
        type=int,
        default=126,
        help="Trend window used by the quality-plus-momentum comparison.",
    )
    parser.add_argument(
        "--quality-momentum-weight",
        type=float,
        default=0.5,
        help="Weight assigned to quality in the quality-plus-momentum blend.",
    )
    parser.add_argument(
        "--vqt-trend-window",
        type=int,
        default=126,
        help="Trend window for the value-quality-trend comparison.",
    )
    parser.add_argument(
        "--vqt-rebalance-every",
        type=int,
        default=63,
        help="Rebalance interval for the value-quality-trend comparison.",
    )
    parser.add_argument("--linear-trade-cost", type=float, default=0.001)
    parser.add_argument(
        "--max-turnover",
        type=float,
        default=None,
        help="Optional per-rebalance turnover cap for residual reversal.",
    )
    parser.add_argument("--output", default=".cache/sec_quality_research.json")
    parser.add_argument("--plot", default=".cache/sec_quality_research.png")
    return parser.parse_args()


def _close_frame(
    symbols: list[str], start: str, end: str | None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    requested = [*symbols, "SPY"]
    frame = yf.download(
        requested,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )
    if frame.empty:
        raise RuntimeError("Yahoo Finance returned no prices for the requested basket.")

    def extract(field: str) -> pd.DataFrame:
        result = (
            frame[field] if isinstance(frame.columns, pd.MultiIndex) else frame[[field]]
        )
        if isinstance(result, pd.Series):
            result = result.to_frame(name=requested[0])
        result = result.rename_axis("date").sort_index()
        result.columns = [str(column).upper() for column in result.columns]
        return result

    close = extract("Adj Close")
    market_close = extract("Close")
    missing = [symbol for symbol in requested if symbol not in close.columns]
    if missing:
        raise RuntimeError("Yahoo Finance omitted prices for: " + ", ".join(missing))
    start_date = pd.Timestamp(start)
    end_date = pd.Timestamp(end) if end else close.index.max()
    coverage_cutoff = end_date - pd.Timedelta(days=90)
    eligible = [
        symbol
        for symbol in symbols
        if (
            not close[symbol].dropna().empty
            and close[symbol].dropna().index[0] <= start_date + pd.Timedelta(days=90)
            and close[symbol].dropna().index[-1] >= coverage_cutoff
        )
    ]
    if len(eligible) < 2:
        raise RuntimeError("Fewer than two symbols cover the requested price interval.")
    columns = [*eligible, "SPY"]
    close = close[columns]
    market_close = market_close[columns]
    valid = close.notna().all(axis=1) & market_close.notna().all(axis=1)
    return close.loc[valid], market_close.loc[valid]


def _stratified_symbols(
    symbols: list[str], groups: dict[str, str], maximum: int
) -> list[str]:
    """Take a deterministic round-robin sample across available sectors."""
    if maximum >= len(symbols):
        return symbols
    unique_groups = {groups.get(symbol, "__unknown__") for symbol in symbols}
    if len(unique_groups) > maximum // 2:
        # Some repository universes use a unique descriptive label per symbol
        # rather than sectors. Spread the sample across the full ordered list.
        return [symbols[(index * len(symbols)) // maximum] for index in range(maximum)]
    grouped: dict[str, list[str]] = {}
    for symbol in symbols:
        grouped.setdefault(groups.get(symbol, "__unknown__"), []).append(symbol)
    buckets = [grouped[key] for key in sorted(grouped)]
    selected: list[str] = []
    cursor = 0
    while len(selected) < maximum:
        added = False
        for bucket in buckets:
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                added = True
                if len(selected) == maximum:
                    break
        if not added:
            break
        cursor += 1
    return selected


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


def _run_period(
    prices: pd.DataFrame,
    market_prices: pd.DataFrame,
    symbols: list[str],
    facts: dict[str, dict],
    groups: dict[str, str],
    start: str,
    end: str,
    *,
    rebalance_every: int,
    top_k: int,
    linear_trade_cost: float,
    max_turnover: float | None,
    dual_lookback_days: int,
    dual_top_k: int,
    adaptive_lookback_days: int,
    quality_momentum_lookback_days: int,
    quality_momentum_weight: float,
    vqt_trend_window: int,
    vqt_rebalance_every: int,
) -> dict[str, dict[str, float]] | None:
    period_prices = prices.loc[start:end]
    period_market_prices = market_prices.loc[period_prices.index]
    if len(period_prices) < 90:
        return None
    dates = [timestamp.date() for timestamp in period_prices.index]
    closes = {
        symbol: period_prices[symbol].astype(float).tolist() for symbol in symbols
    }
    quality = run_quality_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        trading_dates=dates,
        companyfacts_by_symbol=facts,
        groups=groups,
        rebalance_every=rebalance_every,
        top_k=top_k,
        linear_trade_cost=linear_trade_cost,
    )
    residual = run_residual_reversion_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        trading_dates=dates,
        groups=groups,
        start_day=max(63, rebalance_every),
        rebalance_every=rebalance_every,
        top_k=top_k,
        linear_trade_cost=linear_trade_cost,
        max_turnover=max_turnover,
    )
    dual = (
        run_dual_momentum_backtest(
            symbols=symbols,
            closes_by_symbol=closes,
            asset_classes=groups,
            lookback_days=dual_lookback_days,
            rebalance_every=rebalance_every,
            top_k=dual_top_k,
            absolute_threshold=0.0,
            weighting="equal",
            trailing_stop=0.15,
            linear_trade_cost=linear_trade_cost,
        )
        if len(dates) > dual_lookback_days + 1
        else None
    )
    adaptive = (
        run_regime_adaptive_backtest(
            symbols=symbols,
            closes_by_symbol=closes,
            asset_classes=groups,
            lookback_days=adaptive_lookback_days,
            rebalance_every=rebalance_every,
            top_k=top_k,
            absolute_threshold=0.0,
            selection_window=63,
            mean_reversion_window=5,
            linear_trade_cost=linear_trade_cost,
        )
        if len(dates) > adaptive_lookback_days + 1
        else None
    )
    quality_momentum = (
        run_quality_momentum_backtest(
            symbols=symbols,
            closes_by_symbol=closes,
            trading_dates=dates,
            companyfacts_by_symbol=facts,
            groups=groups,
            lookback_days=quality_momentum_lookback_days,
            rebalance_every=rebalance_every,
            top_k=top_k,
            quality_weight=quality_momentum_weight,
            linear_trade_cost=linear_trade_cost,
        )
        if len(dates) > quality_momentum_lookback_days + 1
        else None
    )
    value_quality_trend = (
        run_value_quality_trend_backtest(
            symbols=symbols,
            closes_by_symbol=closes,
            market_closes_by_symbol={
                symbol: period_market_prices[symbol].astype(float).tolist()
                for symbol in symbols
            },
            trading_dates=dates,
            companyfacts_by_symbol=facts,
            groups=groups,
            trend_window=vqt_trend_window,
            rebalance_every=vqt_rebalance_every,
            top_k=top_k,
            max_per_group=2,
            linear_trade_cost=linear_trade_cost,
        )
        if len(dates) > vqt_trend_window + 1
        else None
    )
    spy = run_fixed_allocation_backtest(
        symbols=["SPY"],
        closes_by_symbol={"SPY": period_prices["SPY"].astype(float).tolist()},
        weights_by_symbol={"SPY": 1.0},
        start_day=0,
        rebalance_every=rebalance_every,
        linear_trade_cost=linear_trade_cost,
    )
    equal_weight = run_fixed_allocation_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        weights_by_symbol={symbol: 1.0 / len(symbols) for symbol in symbols},
        start_day=0,
        rebalance_every=rebalance_every,
        linear_trade_cost=linear_trade_cost,
    )
    result = {
        "sec_quality": _metrics(quality),
        "residual_reversion": _metrics(residual),
        "spy": _metrics(spy),
        "equal_weight": _metrics(equal_weight),
    }
    if dual is not None:
        result["dual_momentum"] = _metrics(dual)
    if adaptive is not None:
        result["regime_adaptive"] = _metrics(adaptive)
    if quality_momentum is not None:
        result["quality_momentum"] = _metrics(quality_momentum)
    if value_quality_trend is not None:
        result["value_quality_trend"] = _metrics(value_quality_trend)
    return result


def main() -> None:
    args = _parse_args()
    universe = json.loads(Path(args.universe).read_text())
    all_symbols = [str(symbol).upper() for symbol in universe["symbols"]]
    groups = {
        str(symbol).upper(): str(group)
        for symbol, group in universe.get("asset_classes", {}).items()
    }
    if args.max_symbols < 2:
        raise SystemExit("--max-symbols must be at least 2.")
    client = SecEdgarClient()
    ciks = client.ticker_ciks()
    available_symbols = [
        symbol for symbol in all_symbols if symbol in ciks and symbol != "SPY"
    ]
    candidate_symbols = _stratified_symbols(available_symbols, groups, args.max_symbols)
    if not candidate_symbols:
        raise SystemExit("No candidate symbols were found in the SEC ticker map.")
    print(
        f"Downloading prices for {len(candidate_symbols)} symbols...", file=sys.stderr
    )
    prices, market_prices = _close_frame(candidate_symbols, args.start, args.end)
    priced_symbols = [symbol for symbol in candidate_symbols if symbol in prices]

    facts: dict[str, dict] = {}
    for index, symbol in enumerate(priced_symbols, start=1):
        try:
            facts[symbol] = client.company_facts(ciks[symbol])
        except Exception as exc:  # noqa: BLE001 - one bad issuer should not abort research
            print(f"Skipping {symbol}: {exc}", file=sys.stderr)
        print(f"SEC facts {index}/{len(priced_symbols)}: {symbol}", file=sys.stderr)
    symbols = [symbol for symbol in priced_symbols if symbol in facts]
    if len(symbols) < 2:
        raise SystemExit("Fewer than two symbols have both prices and SEC facts.")
    prices = prices[[*symbols, "SPY"]]
    market_prices = market_prices[[*symbols, "SPY"]]
    dates = [timestamp.date() for timestamp in prices.index]
    closes = {symbol: prices[symbol].astype(float).tolist() for symbol in symbols}

    strategy = run_quality_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        trading_dates=dates,
        companyfacts_by_symbol=facts,
        groups=groups,
        rebalance_every=args.rebalance_every,
        top_k=args.top_k,
        linear_trade_cost=args.linear_trade_cost,
    )
    residual = run_residual_reversion_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        trading_dates=dates,
        groups=groups,
        start_day=max(63, args.rebalance_every),
        rebalance_every=args.rebalance_every,
        lookback_days=5,
        volatility_window=63,
        top_k=args.top_k,
        linear_trade_cost=args.linear_trade_cost,
        max_turnover=args.max_turnover,
    )
    dual = run_dual_momentum_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        asset_classes=groups,
        lookback_days=args.dual_lookback_days,
        rebalance_every=args.rebalance_every,
        top_k=args.dual_top_k,
        absolute_threshold=0.0,
        weighting="equal",
        trailing_stop=0.15,
        linear_trade_cost=args.linear_trade_cost,
    )
    adaptive = run_regime_adaptive_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        asset_classes=groups,
        lookback_days=args.adaptive_lookback_days,
        rebalance_every=args.rebalance_every,
        top_k=args.top_k,
        absolute_threshold=0.0,
        selection_window=63,
        mean_reversion_window=5,
        linear_trade_cost=args.linear_trade_cost,
    )
    quality_momentum = run_quality_momentum_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        trading_dates=dates,
        companyfacts_by_symbol=facts,
        groups=groups,
        lookback_days=args.quality_momentum_lookback_days,
        rebalance_every=args.rebalance_every,
        top_k=args.top_k,
        quality_weight=args.quality_momentum_weight,
        linear_trade_cost=args.linear_trade_cost,
    )
    value_quality_trend = run_value_quality_trend_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        market_closes_by_symbol={
            symbol: market_prices[symbol].astype(float).tolist() for symbol in symbols
        },
        trading_dates=dates,
        companyfacts_by_symbol=facts,
        groups=groups,
        trend_window=args.vqt_trend_window,
        rebalance_every=args.vqt_rebalance_every,
        top_k=args.top_k,
        max_per_group=2,
        linear_trade_cost=args.linear_trade_cost,
    )
    spy = run_fixed_allocation_backtest(
        symbols=["SPY"],
        closes_by_symbol={"SPY": prices["SPY"].astype(float).tolist()},
        weights_by_symbol={"SPY": 1.0},
        start_day=0,
        rebalance_every=args.rebalance_every,
        linear_trade_cost=args.linear_trade_cost,
    )
    equal_weight = run_fixed_allocation_backtest(
        symbols=symbols,
        closes_by_symbol=closes,
        weights_by_symbol={symbol: 1.0 / len(symbols) for symbol in symbols},
        start_day=0,
        rebalance_every=args.rebalance_every,
        linear_trade_cost=args.linear_trade_cost,
    )
    regime_results = {
        label: result
        for label, (period_start, period_end) in {
            "2016-2019": ("2016-01-01", "2019-12-31"),
            "2020-2021": ("2020-01-01", "2021-12-31"),
            "2022": ("2022-01-01", "2022-12-31"),
            "2023-present": ("2023-01-01", args.end or "2100-01-01"),
        }.items()
        if (
            result := _run_period(
                prices,
                market_prices,
                symbols,
                facts,
                groups,
                period_start,
                period_end,
                rebalance_every=args.rebalance_every,
                top_k=args.top_k,
                linear_trade_cost=args.linear_trade_cost,
                max_turnover=args.max_turnover,
                dual_lookback_days=args.dual_lookback_days,
                dual_top_k=args.dual_top_k,
                adaptive_lookback_days=args.adaptive_lookback_days,
                quality_momentum_lookback_days=args.quality_momentum_lookback_days,
                quality_momentum_weight=args.quality_momentum_weight,
                vqt_trend_window=args.vqt_trend_window,
                vqt_rebalance_every=args.vqt_rebalance_every,
            )
        )
        is not None
    }

    output = {
        "kind": "preliminary_sec_quality_research",
        "universe": args.universe,
        "survivorship_warning": "Ticker file and SEC ticker map are not historical constituent membership.",
        "symbols": symbols,
        "selection_method": "deterministic group-stratified or evenly spaced sample",
        "start": dates[0].isoformat(),
        "end": dates[-1].isoformat(),
        "linear_trade_cost": args.linear_trade_cost,
        "max_turnover": args.max_turnover,
        "dual_momentum_config": {
            "lookback_days": args.dual_lookback_days,
            "top_k": args.dual_top_k,
            "trailing_stop": 0.15,
        },
        "regime_adaptive_config": {
            "lookback_days": args.adaptive_lookback_days,
            "selection_window": 63,
            "mean_reversion_window": 5,
        },
        "quality_momentum_config": {
            "lookback_days": args.quality_momentum_lookback_days,
            "quality_weight": args.quality_momentum_weight,
        },
        "value_quality_trend_config": {
            "trend_window": args.vqt_trend_window,
            "rebalance_every": args.vqt_rebalance_every,
            "max_per_group": 2,
            "weights": {"quality": 0.4, "value": 0.4, "trend": 0.2},
        },
        "strategy": _metrics(strategy),
        "residual_reversion": _metrics(residual),
        "dual_momentum": _metrics(dual),
        "regime_adaptive": _metrics(adaptive),
        "quality_momentum": _metrics(quality_momentum),
        "value_quality_trend": _metrics(value_quality_trend),
        "spy": _metrics(spy),
        "equal_weight": _metrics(equal_weight),
        "regimes": regime_results,
        "dates": [item.isoformat() for item in dates],
        "strategy_values": list(strategy.daily_values),
        "residual_reversion_values": [1.0] * (len(dates) - len(residual.daily_values))
        + list(residual.daily_values),
        "dual_momentum_values": [1.0] * (len(dates) - len(dual.daily_values))
        + list(dual.daily_values),
        "regime_adaptive_values": [1.0] * (len(dates) - len(adaptive.daily_values))
        + list(adaptive.daily_values),
        "quality_momentum_values": [1.0]
        * (len(dates) - len(quality_momentum.daily_values))
        + list(quality_momentum.daily_values),
        "value_quality_trend_values": [1.0]
        * (len(dates) - len(value_quality_trend.daily_values))
        + list(value_quality_trend.daily_values),
        "spy_values": list(spy.daily_values),
        "equal_weight_values": list(equal_weight.daily_values),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")

    figure, axis = plt.subplots(figsize=(11, 6))
    axis.plot(dates, strategy.daily_values, label="SEC quality", linewidth=2)
    residual_dates = dates[-len(residual.daily_values) :]
    residual_label = "Residual reversal (rolling factors)"
    if args.max_turnover is not None:
        residual_label += f", cap {args.max_turnover:.0%}"
    axis.plot(
        residual_dates,
        residual.daily_values,
        label=residual_label,
        linewidth=1.5,
    )
    dual_dates = dates[-len(dual.daily_values) :]
    axis.plot(
        dual_dates,
        dual.daily_values,
        label=f"Dual momentum ({args.dual_lookback_days}d)",
        linewidth=1.25,
    )
    adaptive_dates = dates[-len(adaptive.daily_values) :]
    axis.plot(
        adaptive_dates,
        adaptive.daily_values,
        label=f"Regime adaptive ({args.adaptive_lookback_days}d)",
        linewidth=1.25,
    )
    quality_momentum_dates = dates[-len(quality_momentum.daily_values) :]
    axis.plot(
        quality_momentum_dates,
        quality_momentum.daily_values,
        label="Quality + momentum",
        linewidth=1.5,
    )
    value_quality_trend_dates = dates[-len(value_quality_trend.daily_values) :]
    axis.plot(
        value_quality_trend_dates,
        value_quality_trend.daily_values,
        label="Value + quality + trend",
        linewidth=1.5,
    )
    axis.plot(dates, spy.daily_values, label="SPY", linestyle="--")
    axis.plot(dates, equal_weight.daily_values, label="Equal weight", alpha=0.8)
    axis.set_title("Research strategy comparison (not survivorship-bias-free)")
    axis.set_ylabel("Growth of $1")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.autofmt_xdate()
    figure.tight_layout()
    plot_path = Path(args.plot)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(plot_path, dpi=150)
    print(
        json.dumps(
            {
                "report": str(output_path),
                "plot": str(plot_path),
                "symbols": len(symbols),
            }
        )
    )


if __name__ == "__main__":
    main()
