"""Research-only filing-date-safe value, quality, and trend strategy."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from .backtest import TRADING_DAYS_PER_YEAR, BacktestResult, summarize_return_series
from .fundamental_quality import build_snapshot, rank_quality_scores
from .sec_edgar import iter_facts, latest_facts_as_of


def _group_percentiles(
    values: dict[str, float], groups: dict[str, str] | None
) -> dict[str, float]:
    buckets: dict[str, list[str]] = {}
    for symbol in values:
        buckets.setdefault((groups or {}).get(symbol, "__all__"), []).append(symbol)
    scores: dict[str, float] = {}
    for members in buckets.values():
        ordered = sorted(members, key=lambda symbol: (values[symbol], symbol))
        scores.update(
            {symbol: (index + 1) / len(ordered) for index, symbol in enumerate(ordered)}
        )
    return scores


def _shares_outstanding(payload: dict[str, Any], as_of: date) -> float | None:
    observations = iter_facts(
        payload,
        tag="EntityCommonStockSharesOutstanding",
        unit="shares",
        forms=("10-K", "10-Q", "10-Q/A"),
        taxonomy="dei",
    )
    available = latest_facts_as_of(observations, as_of)
    eligible = [item for item in available.values() if item.end <= as_of]
    if not eligible:
        return None
    shares = max(eligible, key=lambda item: item.end).value
    return shares if shares > 0.0 else None


def run_value_quality_trend_backtest(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    market_closes_by_symbol: dict[str, list[float]],
    trading_dates: list[date],
    companyfacts_by_symbol: dict[str, dict[str, Any]],
    groups: dict[str, str] | None = None,
    trend_window: int = 126,
    rebalance_every: int = 63,
    top_k: int = 10,
    max_per_group: int = 2,
    quality_weight: float = 0.4,
    value_weight: float = 0.4,
    trend_weight: float = 0.2,
    minimum_metrics: int = 3,
    linear_trade_cost: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Rank profitable, inexpensive firms with a modest trend confirmation."""
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if trend_window < 1 or rebalance_every < 1 or top_k < 1 or max_per_group < 1:
        raise ValueError("windows, top_k, and max_per_group must be positive.")
    if min(quality_weight, value_weight, trend_weight) < 0.0:
        raise ValueError("factor weights cannot be negative.")
    if not np.isclose(quality_weight + value_weight + trend_weight, 1.0):
        raise ValueError("factor weights must sum to 1.0.")
    if minimum_metrics < 1 or linear_trade_cost < 0.0:
        raise ValueError(
            "minimum_metrics must be positive and cost cannot be negative."
        )
    lengths = {len(closes_by_symbol.get(symbol, [])) for symbol in symbols}
    if len(lengths) != 1 or len(trading_dates) != next(iter(lengths), 0):
        raise ValueError("All prices and trading_dates must have the same length.")
    if any(
        len(market_closes_by_symbol.get(symbol, [])) != len(trading_dates)
        for symbol in symbols
    ):
        raise ValueError("Unadjusted market closes must match trading_dates.")
    missing = [symbol for symbol in symbols if symbol not in companyfacts_by_symbol]
    if missing:
        raise ValueError("Missing SEC companyfacts for: " + ", ".join(missing))
    if len(trading_dates) < trend_window + 2:
        raise ValueError("Not enough dated prices for the requested trend window.")

    prices = np.array([closes_by_symbol[symbol] for symbol in symbols], dtype=float)
    market_prices = np.array(
        [market_closes_by_symbol[symbol] for symbol in symbols], dtype=float
    )
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    weights = np.zeros(len(symbols), dtype=float)
    values = [1.0]
    period_returns: list[float] = []
    turnovers: list[float] = []
    value = peak_value = 1.0
    max_drawdown = 0.0
    rebalance_count = 0

    for step in range(trend_window, returns.shape[1]):
        trade_cost = 0.0
        if (step - trend_window) % rebalance_every == 0:
            snapshots = {
                symbol: snapshot
                for symbol in symbols
                if (
                    snapshot := build_snapshot(
                        symbol,
                        companyfacts_by_symbol[symbol],
                        as_of=trading_dates[step],
                    )
                )
            }
            quality = rank_quality_scores(
                list(snapshots.values()),
                groups=groups,
                minimum_metrics=minimum_metrics,
            )
            earnings_yield: dict[str, float] = {}
            trend: dict[str, float] = {}
            for index, symbol in enumerate(symbols):
                snapshot = snapshots.get(symbol)
                shares = _shares_outstanding(
                    companyfacts_by_symbol[symbol], trading_dates[step]
                )
                if (
                    symbol not in quality
                    or snapshot is None
                    or snapshot.net_income is None
                    or snapshot.net_income <= 0.0
                    or shares is None
                ):
                    continue
                market_value = shares * market_prices[index, step]
                if market_value <= 0.0:
                    continue
                earnings_yield[symbol] = snapshot.net_income / market_value
                trend[symbol] = (
                    prices[index, step] / prices[index, step - trend_window] - 1.0
                )
            value_scores = _group_percentiles(earnings_yield, groups)
            trend_scores = _group_percentiles(trend, groups)
            combined = {
                symbol: quality_weight * quality[symbol]
                + value_weight * value_scores[symbol]
                + trend_weight * trend_scores[symbol]
                for symbol in quality
                if symbol in value_scores and symbol in trend_scores
            }
            selected: list[str] = []
            group_counts: dict[str, int] = {}
            for symbol in sorted(combined, key=lambda item: (-combined[item], item)):
                group = (groups or {}).get(symbol, "__all__")
                if group_counts.get(group, 0) < max_per_group:
                    selected.append(symbol)
                    group_counts[group] = group_counts.get(group, 0) + 1
                if len(selected) == top_k:
                    break
            target = np.zeros(len(symbols), dtype=float)
            if selected:
                target[[symbols.index(symbol) for symbol in selected]] = 1.0 / len(
                    selected
                )
            turnover = float(np.abs(target - weights).sum())
            turnovers.append(turnover)
            weights = target
            trade_cost = linear_trade_cost * turnover
            rebalance_count += 1

        gross_return = float(np.dot(weights, returns[:, step]))
        period_return = gross_return - trade_cost
        value *= 1.0 + period_return
        if value <= 0.0:
            raise ValueError("Portfolio gross value became nonpositive.")
        values.append(value)
        period_returns.append(period_return)
        peak_value = max(peak_value, value)
        max_drawdown = max(max_drawdown, 1.0 - value / peak_value)
        weights = weights * (1.0 + returns[:, step]) / (1.0 + gross_return)

    summary = summarize_return_series(
        np.array(period_returns),
        trading_days_per_year=trading_days_per_year,
        risk_free_rate=risk_free_rate,
    )
    return BacktestResult(
        final_value=value,
        total_return=value - 1.0,
        annualized_return=summary.annualized_return,
        annualized_volatility=summary.annualized_volatility,
        max_drawdown=max_drawdown,
        rebalance_count=rebalance_count,
        average_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        latest_weights=weights,
        daily_values=tuple(values),
        sortino_ratio=summary.sortino_ratio,
    )
