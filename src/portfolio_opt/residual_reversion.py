"""Research-only sector/market residual short-term reversal strategy."""

from __future__ import annotations

from datetime import date

import numpy as np

from .backtest import TRADING_DAYS_PER_YEAR, BacktestResult, summarize_return_series


def _capped_inverse_volatility(
    selected: list[int], volatility: np.ndarray, max_single_weight: float | None
) -> np.ndarray:
    weights = np.zeros(len(volatility), dtype=float)
    if not selected:
        return weights
    raw = np.array(
        [1.0 / max(float(volatility[index]), 1e-8) for index in selected], dtype=float
    )
    raw /= float(raw.sum())
    if max_single_weight is None:
        weights[selected] = raw
        return weights
    if not 0.0 < max_single_weight <= 1.0:
        raise ValueError("max_single_weight must be in (0, 1].")
    if max_single_weight * len(selected) < 1.0 - 1e-12:
        raise ValueError("max_single_weight is too low for the selected basket.")
    remaining = set(range(len(selected)))
    assigned = np.zeros(len(selected), dtype=float)
    remaining_total = 1.0
    while remaining:
        proposed_sum = float(raw[list(remaining)].sum())
        capped = [
            index
            for index in remaining
            if remaining_total * raw[index] / proposed_sum > max_single_weight
        ]
        if not capped:
            for index in remaining:
                assigned[index] = remaining_total * raw[index] / proposed_sum
            break
        for index in capped:
            assigned[index] = max_single_weight
            remaining_total -= max_single_weight
            remaining.remove(index)
        if remaining_total <= 1e-12:
            break
    weights[selected] = assigned
    return weights


def _residual_scores(
    returns: np.ndarray,
    *,
    groups: list[str] | None,
    lookback_days: int,
) -> np.ndarray:
    log_returns = np.log1p(np.clip(returns, -0.999999, None))
    market = log_returns.mean(axis=0)
    has_groups = groups is not None and len(set(groups)) < len(groups)
    residual = np.empty_like(log_returns)
    for index in range(log_returns.shape[0]):
        factors = [np.ones(log_returns.shape[1]), market]
        if has_groups and groups is not None:
            indices = [
                item for item, value in enumerate(groups) if value == groups[index]
            ]
            if len(indices) > 1:
                factors.append(log_returns[indices].mean(axis=0))
        design = np.column_stack(factors)
        coefficients, *_ = np.linalg.lstsq(design, log_returns[index], rcond=None)
        residual[index] = log_returns[index] - design @ coefficients
    return residual[:, -lookback_days:].sum(axis=1)


def run_residual_reversion_backtest(
    *,
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    trading_dates: list[date],
    groups: dict[str, str] | None = None,
    start_day: int = 63,
    rebalance_every: int = 21,
    lookback_days: int = 5,
    volatility_window: int = 63,
    top_k: int = 10,
    max_single_weight: float | None = 0.20,
    max_turnover: float | None = None,
    linear_trade_cost: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Buy the weakest recent market/sector residual performers.

    Signals use prices through date ``t`` and are applied only to the return
    from ``t`` to ``t+1``.  This is a long-only reversal test; it does not
    assume shorting, leverage, or borrow availability.
    """
    if not symbols:
        raise ValueError("At least one symbol is required.")
    if start_day <= lookback_days or rebalance_every < 1 or top_k < 1:
        raise ValueError(
            "start_day must exceed lookback_days; other intervals must be positive."
        )
    if volatility_window < 2 or linear_trade_cost < 0.0:
        raise ValueError(
            "volatility_window must be at least 2 and cost cannot be negative."
        )
    if max_turnover is not None and not 0.0 <= max_turnover <= 2.0:
        raise ValueError("max_turnover must be between 0 and 2.")
    lengths = {len(closes_by_symbol.get(symbol, [])) for symbol in symbols}
    if len(lengths) != 1 or len(trading_dates) != next(iter(lengths), 0):
        raise ValueError("All prices and trading_dates must have the same length.")
    if len(trading_dates) < 2 or start_day >= len(trading_dates) - 1:
        raise ValueError("Not enough dated prices for the requested backtest.")

    prices = np.array([closes_by_symbol[symbol] for symbol in symbols], dtype=float)
    returns = prices[:, 1:] / prices[:, :-1] - 1.0
    group_values = (
        [groups.get(symbol, "__all__") for symbol in symbols] if groups else None
    )
    weights = np.zeros(len(symbols), dtype=float)
    value = 1.0
    peak_value = value
    max_drawdown = 0.0
    values = [value]
    period_returns: list[float] = []
    turnovers: list[float] = []
    rebalance_count = 0

    for step in range(start_day, returns.shape[1]):
        if step == start_day or (step - start_day) % rebalance_every == 0:
            history = returns[
                :, max(0, step - max(volatility_window, lookback_days + 1)) : step
            ]
            scores = _residual_scores(
                history, groups=group_values, lookback_days=lookback_days
            )
            selected = np.argsort(scores)[: min(top_k, len(symbols))].tolist()
            recent = returns[:, max(0, step - volatility_window) : step]
            volatility = np.std(recent, axis=1, ddof=0)
            target = _capped_inverse_volatility(selected, volatility, max_single_weight)
            turnover = float(np.abs(target - weights).sum())
            if max_turnover is not None and turnover > max_turnover:
                target = weights + (target - weights) * (max_turnover / turnover)
                turnover = max_turnover
            turnovers.append(turnover)
            weights = target
            trade_cost = linear_trade_cost * turnover
            rebalance_count += 1
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
