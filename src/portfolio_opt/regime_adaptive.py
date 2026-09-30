"""Research-only adaptive strategies built from daily closing prices.

The functions here deliberately use only information available before each
holding period.  They are backtest tools, not a claim of future outperformance.
"""

from __future__ import annotations

import numpy as np

from .backtest import (
    TRADING_DAYS_PER_YEAR,
    BacktestResult,
    align_close_history,
    compute_protective_momentum_weights,
    summarize_return_series,
)


def _defensive_indices(symbols: list[str], asset_classes: dict[str, str]) -> list[int]:
    return [
        index
        for index, symbol in enumerate(symbols)
        if asset_classes.get(symbol, "").lower().startswith(("cash", "bond"))
    ]


def compute_trend_filtered_mean_reversion_weights(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    asset_classes: dict[str, str],
    trend_window: int,
    mean_reversion_window: int,
    top_k: int,
) -> dict[str, float]:
    """Buy short-term laggards which remain above their longer-term trend.

    This is cross-sectional mean reversion, constrained by a trend filter to
    avoid mechanically buying the weakest assets during a broad sell-off.
    """
    if trend_window < 1 or mean_reversion_window < 1:
        raise ValueError("trend and mean-reversion windows must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    aligned = align_close_history(symbols, closes_by_symbol)
    prices = np.array([aligned[symbol] for symbol in symbols], dtype=float)
    required = max(trend_window, mean_reversion_window) + 1
    if prices.shape[1] < required:
        raise ValueError("Not enough price history for the requested windows.")

    defensive = _defensive_indices(symbols, asset_classes)
    risky = [index for index in range(len(symbols)) if index not in defensive]
    long_return = prices[:, -1] / prices[:, -1 - trend_window] - 1.0
    short_return = prices[:, -1] / prices[:, -1 - mean_reversion_window] - 1.0
    cash_return = max((long_return[index] for index in defensive), default=0.0)
    eligible = [index for index in risky if long_return[index] > cash_return]
    selected = sorted(
        eligible, key=lambda index: (short_return[index], symbols[index])
    )[:top_k]
    if not selected:
        selected = defensive
    if not selected:
        raise ValueError("At least one risky or defensive asset is required.")

    weight = 1.0 / len(selected)
    return {
        symbol: weight if index in selected else 0.0
        for index, symbol in enumerate(symbols)
    }


def run_regime_adaptive_backtest(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    asset_classes: dict[str, str],
    lookback_days: int,
    rebalance_every: int,
    top_k: int,
    absolute_threshold: float,
    selection_window: int = 63,
    mean_reversion_window: int = 5,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    linear_trade_cost: float = 0.0,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Choose trend or mean-reversion from their *prior* virtual performance.

    Both sleeves are simulated independently.  At a rebalance the selected
    sleeve is the one with the greater trailing return over ``selection_window``;
    no return from the current holding period participates in the decision.
    """
    if lookback_days < 1 or rebalance_every < 1 or selection_window < 1:
        raise ValueError(
            "lookback_days, rebalance_every, and selection_window must be positive."
        )
    if linear_trade_cost < 0:
        raise ValueError("linear_trade_cost cannot be negative.")
    aligned = align_close_history(symbols, closes_by_symbol)
    prices = np.array([aligned[symbol] for symbol in symbols], dtype=float)
    if prices.shape[1] < max(lookback_days, mean_reversion_window) + 2:
        raise ValueError("Not enough price history to run the backtest.")
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    start = max(lookback_days, mean_reversion_window)
    active_weights = np.zeros(len(symbols), dtype=float)
    trend_weights = active_weights.copy()
    reversion_weights = active_weights.copy()
    value = trend_value = reversion_value = 1.0
    values = [value]
    trend_values = [trend_value]
    reversion_values = [reversion_value]
    period_returns: list[float] = []
    turnovers: list[float] = []
    rebalances = 0
    peak_value = value
    max_drawdown = 0.0

    for step in range(start, returns.shape[1]):
        trend_cost = reversion_cost = active_cost = 0.0
        if (step - start) % rebalance_every == 0:
            history = {symbol: aligned[symbol][: step + 1] for symbol in symbols}
            trend_dict = compute_protective_momentum_weights(
                symbols=symbols,
                closes_by_symbol=history,
                asset_classes=asset_classes,
                lookback_days=lookback_days,
                top_k=top_k,
                absolute_threshold=absolute_threshold,
            )
            reversion_dict = compute_trend_filtered_mean_reversion_weights(
                symbols=symbols,
                closes_by_symbol=history,
                asset_classes=asset_classes,
                trend_window=lookback_days,
                mean_reversion_window=mean_reversion_window,
                top_k=top_k,
            )
            next_trend = np.array([trend_dict[symbol] for symbol in symbols])
            next_reversion = np.array([reversion_dict[symbol] for symbol in symbols])
            trend_cost = linear_trade_cost * float(
                np.abs(next_trend - trend_weights).sum()
            )
            reversion_cost = linear_trade_cost * float(
                np.abs(next_reversion - reversion_weights).sum()
            )
            trend_weights, reversion_weights = next_trend, next_reversion
            if len(trend_values) > selection_window:
                trend_score = (
                    trend_values[-1] / trend_values[-1 - selection_window] - 1.0
                )
                reversion_score = (
                    reversion_values[-1] / reversion_values[-1 - selection_window] - 1.0
                )
                target_weights = (
                    reversion_weights
                    if reversion_score > trend_score
                    else trend_weights
                )
            else:
                target_weights = trend_weights
            turnover = float(np.abs(target_weights - active_weights).sum())
            active_cost = linear_trade_cost * turnover
            active_weights = target_weights.copy()
            turnovers.append(turnover)
            rebalances += 1

        daily_returns = returns[:, step]
        trend_return = float(np.dot(trend_weights, daily_returns))
        reversion_return = float(np.dot(reversion_weights, daily_returns))
        active_return = float(np.dot(active_weights, daily_returns))
        trend_value *= 1.0 + trend_return - trend_cost
        reversion_value *= 1.0 + reversion_return - reversion_cost
        realized_return = active_return - active_cost
        value *= 1.0 + realized_return
        period_returns.append(realized_return)
        values.append(value)
        trend_values.append(trend_value)
        reversion_values.append(reversion_value)
        peak_value = max(peak_value, value)
        max_drawdown = max(max_drawdown, 1.0 - value / peak_value)
        trend_weights = trend_weights * (1.0 + daily_returns) / (1.0 + trend_return)
        reversion_weights = (
            reversion_weights * (1.0 + daily_returns) / (1.0 + reversion_return)
        )
        active_weights = active_weights * (1.0 + daily_returns) / (1.0 + active_return)

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
        rebalance_count=rebalances,
        average_turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        latest_weights=active_weights,
        daily_values=tuple(values),
        sortino_ratio=summary.sortino_ratio,
    )
