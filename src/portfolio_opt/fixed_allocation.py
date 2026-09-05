"""Cost-aware fixed-allocation backtests with realistic weight drift."""

from __future__ import annotations

import numpy as np

from .backtest import (
    TRADING_DAYS_PER_YEAR,
    BacktestResult,
    align_close_history,
    summarize_return_series,
)


def parse_fixed_weights(
    values: list[str],
    *,
    allowed_symbols: list[str],
) -> dict[str, float]:
    """Parse ``SYMBOL=WEIGHT`` CLI values and validate a long-only allocation."""
    weights: dict[str, float] = {}
    allowed = set(allowed_symbols)
    for value in values:
        symbol, separator, raw_weight = value.partition("=")
        symbol = symbol.strip().upper()
        if not separator or not symbol:
            raise ValueError(f"Invalid fixed weight {value!r}; expected SYMBOL=WEIGHT.")
        if symbol not in allowed:
            raise ValueError(
                f"Fixed-weight symbol {symbol} is not in the model universe."
            )
        if symbol in weights:
            raise ValueError(f"Duplicate fixed-weight symbol: {symbol}.")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(
                f"Invalid fixed weight for {symbol}: {raw_weight!r}."
            ) from exc
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Fixed weight for {symbol} must be finite and nonnegative."
            )
        weights[symbol] = weight

    if not weights:
        raise ValueError("At least one --fixed-weight SYMBOL=WEIGHT is required.")
    total = sum(weights.values())
    if not np.isclose(total, 1.0, atol=1e-8):
        raise ValueError(f"Fixed weights must sum to 1.0; received {total:.8f}.")
    return weights


def run_fixed_allocation_backtest(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    weights_by_symbol: dict[str, float],
    start_day: int,
    rebalance_every: int,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    linear_trade_cost: float = 0.0,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Backtest fixed targets, drifting weights between scheduled rebalances."""
    if start_day < 0:
        raise ValueError("start_day cannot be negative.")
    if rebalance_every < 1:
        raise ValueError("rebalance_every must be at least 1.")
    if linear_trade_cost < 0.0:
        raise ValueError("linear_trade_cost cannot be negative.")

    unknown = set(weights_by_symbol) - set(symbols)
    if unknown:
        raise ValueError(f"Unknown fixed-weight symbols: {sorted(unknown)}.")
    target = np.array([weights_by_symbol.get(symbol, 0.0) for symbol in symbols])
    if np.any(target < 0.0) or not np.all(np.isfinite(target)):
        raise ValueError("Fixed weights must be finite and nonnegative.")
    if not np.isclose(float(target.sum()), 1.0, atol=1e-8):
        raise ValueError("Fixed weights must sum to 1.0.")

    aligned = align_close_history(symbols, closes_by_symbol)
    prices = np.array([aligned[symbol] for symbol in symbols], dtype=float)
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    if start_day >= returns.shape[1]:
        raise ValueError(
            "Not enough price history to run the fixed-allocation backtest."
        )

    weights = np.zeros(len(symbols), dtype=float)
    value = 1.0
    peak_value = value
    max_drawdown = 0.0
    daily_values = [value]
    period_returns: list[float] = []
    turnovers: list[float] = []
    rebalance_count = 0

    for step in range(start_day, returns.shape[1]):
        trade_cost = 0.0
        if (step - start_day) % rebalance_every == 0:
            turnover = float(np.abs(target - weights).sum())
            turnovers.append(turnover)
            trade_cost = linear_trade_cost * turnover
            weights = target.copy()
            rebalance_count += 1

        asset_returns = returns[:, step]
        gross_return = float(np.dot(weights, asset_returns))
        period_return = gross_return - trade_cost
        value *= 1.0 + period_return
        period_returns.append(period_return)
        daily_values.append(value)
        peak_value = max(peak_value, value)
        max_drawdown = max(max_drawdown, 1.0 - value / peak_value)

        gross_growth = 1.0 + gross_return
        if gross_growth <= 0.0:
            raise ValueError("Portfolio gross value became nonpositive.")
        weights = weights * (1.0 + asset_returns) / gross_growth

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
        daily_values=tuple(daily_values),
        sortino_ratio=summary.sortino_ratio,
    )
