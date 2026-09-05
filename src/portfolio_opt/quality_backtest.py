"""Research-only backtest for point-in-time SEC quality rankings."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

from .backtest import (
    TRADING_DAYS_PER_YEAR,
    BacktestResult,
    summarize_return_series,
)
from .fundamental_quality import build_snapshot, rank_quality_scores


def run_quality_backtest(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    trading_dates: list[date],
    companyfacts_by_symbol: dict[str, dict[str, Any]],
    groups: dict[str, str] | None = None,
    start_day: int = 0,
    rebalance_every: int = 21,
    top_k: int = 10,
    minimum_metrics: int = 3,
    linear_trade_cost: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Backtest a long-only quality ranking using only filed information.

    ``trading_dates`` must correspond to the close rows.  A target chosen at
    the close on date ``t`` is applied to the return from ``t`` to ``t+1``;
    this prevents a filing on date ``t`` from leaking into that day's return.
    """
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if start_day < 0 or rebalance_every < 1 or top_k < 1:
        raise ValueError("start_day, rebalance_every, and top_k must be positive.")
    if minimum_metrics < 1 or linear_trade_cost < 0.0:
        raise ValueError(
            "minimum_metrics must be positive and cost cannot be negative."
        )
    lengths = {len(closes_by_symbol.get(symbol, [])) for symbol in symbols}
    if len(lengths) != 1 or len(trading_dates) != next(iter(lengths), 0):
        raise ValueError("All prices and trading_dates must have the same length.")
    if len(trading_dates) < 2 or start_day >= len(trading_dates) - 1:
        raise ValueError("Not enough dated prices for the requested backtest.")
    missing_facts = [
        symbol for symbol in symbols if symbol not in companyfacts_by_symbol
    ]
    if missing_facts:
        raise ValueError("Missing SEC companyfacts for: " + ", ".join(missing_facts))

    prices = np.array([closes_by_symbol[symbol] for symbol in symbols], dtype=float)
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    weights = np.zeros(len(symbols), dtype=float)
    values = [1.0]
    period_returns: list[float] = []
    turnovers: list[float] = []
    rebalance_count = 0
    value = 1.0
    peak_value = value
    max_drawdown = 0.0

    for step in range(start_day, returns.shape[1]):
        if step == start_day or (step - start_day) % rebalance_every == 0:
            snapshots = [
                snapshot
                for symbol in symbols
                if (
                    snapshot := build_snapshot(
                        symbol,
                        companyfacts_by_symbol[symbol],
                        as_of=trading_dates[step],
                    )
                )
            ]
            scores = rank_quality_scores(
                snapshots, groups=groups, minimum_metrics=minimum_metrics
            )
            selected = sorted(scores, key=lambda symbol: (-scores[symbol], symbol))[
                :top_k
            ]
            target = np.zeros(len(symbols), dtype=float)
            if selected:
                selected_indices = [symbols.index(symbol) for symbol in selected]
                target[selected_indices] = 1.0 / len(selected_indices)
            turnover = float(np.abs(target - weights).sum())
            turnovers.append(turnover)
            weights = target
            rebalance_count += 1
            trade_cost = linear_trade_cost * turnover
        else:
            trade_cost = 0.0

        gross_return = float(np.dot(weights, returns[:, step]))
        period_return = gross_return - trade_cost
        value *= 1.0 + period_return
        if value <= 0.0:
            raise ValueError("Portfolio gross value became nonpositive.")
        values.append(value)
        period_returns.append(period_return)
        peak_value = max(peak_value, value)
        max_drawdown = max(max_drawdown, 1.0 - value / peak_value)

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
