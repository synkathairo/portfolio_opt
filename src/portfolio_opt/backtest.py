from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import OptimizationConfig
from .estimation import estimate_inputs_from_momentum, estimate_inputs_from_prices
from .optimizer import optimize_weights


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestResult:
    final_value: float
    total_return: float
    annualized_return: float
    annualized_volatility: float
    max_drawdown: float
    rebalance_count: int
    average_turnover: float
    latest_weights: np.ndarray


def run_backtest(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    lookback_days: int,
    rebalance_every: int,
    return_model: str,
    mean_shrinkage: float,
    momentum_window: int,
    opt_config: OptimizationConfig,
    asset_class_matrix: np.ndarray | None,
) -> BacktestResult:
    price_matrix = np.array([closes_by_symbol[symbol] for symbol in symbols], dtype=float)
    if price_matrix.ndim != 2 or price_matrix.shape[1] <= lookback_days + 1:
        raise ValueError("Not enough price history to run the backtest.")

    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    portfolio_value = 1.0
    weights = np.zeros(len(symbols), dtype=float)
    portfolio_returns: list[float] = []
    turnovers: list[float] = []
    rebalance_count = 0
    peak_value = portfolio_value
    max_drawdown = 0.0

    # Re-estimate inputs at each rebalance date using only prior history, then
    # hold those weights until the next rebalance.
    for step in range(lookback_days, returns.shape[1]):
        if (step - lookback_days) % rebalance_every == 0:
            window_closes = {
                symbol: price_matrix[index, step - lookback_days : step + 1].tolist()
                for index, symbol in enumerate(symbols)
            }
            if return_model == "momentum":
                estimated = estimate_inputs_from_momentum(
                    symbols=symbols,
                    closes_by_symbol=window_closes,
                    mean_shrinkage=mean_shrinkage,
                    momentum_window=momentum_window,
                )
            else:
                estimated = estimate_inputs_from_prices(
                    symbols=symbols,
                    closes_by_symbol=window_closes,
                    mean_shrinkage=mean_shrinkage,
                )
            target_weights = optimize_weights(
                expected_returns=estimated.expected_returns,
                covariance=estimated.covariance,
                config=opt_config,
                current_weights=weights,
                asset_class_matrix=asset_class_matrix,
            )
            turnovers.append(float(np.abs(target_weights - weights).sum()))
            weights = target_weights
            rebalance_count += 1

        cash_weight = max(0.0, 1.0 - float(weights.sum()))
        period_return = float(np.dot(weights, returns[:, step]) + cash_weight * 0.0)
        portfolio_returns.append(period_return)
        portfolio_value *= 1.0 + period_return
        peak_value = max(peak_value, portfolio_value)
        max_drawdown = max(max_drawdown, 1.0 - portfolio_value / peak_value)

    returns_array = np.array(portfolio_returns, dtype=float)
    periods = max(len(returns_array), 1)
    annualized_return = portfolio_value ** (TRADING_DAYS_PER_YEAR / periods) - 1.0
    annualized_volatility = returns_array.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR)
    average_turnover = float(np.mean(turnovers)) if turnovers else 0.0

    return BacktestResult(
        final_value=portfolio_value,
        total_return=portfolio_value - 1.0,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        max_drawdown=max_drawdown,
        rebalance_count=rebalance_count,
        average_turnover=average_turnover,
        latest_weights=weights,
    )
