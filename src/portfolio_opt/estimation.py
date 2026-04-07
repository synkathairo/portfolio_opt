from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class EstimatedInputs:
    expected_returns: np.ndarray
    covariance: np.ndarray
    observations: int


def estimate_inputs_from_prices(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    mean_shrinkage: float,
) -> EstimatedInputs:
    if not symbols:
        raise ValueError("At least one symbol is required.")

    price_matrix = np.array(
        [closes_by_symbol[symbol] for symbol in symbols], dtype=float
    )
    if price_matrix.ndim != 2 or price_matrix.shape[1] < 2:
        raise ValueError("Need at least two prices per symbol to estimate returns.")

    # Align all assets on the same close-to-close return grid.
    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    if returns.shape[1] < 2:
        raise ValueError(
            "Need at least two return observations to estimate covariance."
        )

    sample_mean = returns.mean(axis=1)
    # Shrink noisy return estimates back toward zero so the optimizer does not
    # overreact to short windows of lucky performance.
    shrunk_mean = sample_mean * (1.0 - mean_shrinkage)
    covariance = np.cov(returns, bias=False)

    # Convert daily statistics to annualized inputs for the optimizer.
    expected_returns = shrunk_mean * TRADING_DAYS_PER_YEAR
    annualized_covariance = covariance * TRADING_DAYS_PER_YEAR

    # Add a small diagonal ridge term to keep the covariance numerically stable.
    annualized_covariance += np.eye(len(symbols)) * 1e-6
    return EstimatedInputs(
        expected_returns=expected_returns,
        covariance=annualized_covariance,
        observations=returns.shape[1],
    )


def estimate_inputs_from_momentum(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    mean_shrinkage: float,
    momentum_window: int,
) -> EstimatedInputs:
    if not symbols:
        raise ValueError("At least one symbol is required.")

    price_matrix = np.array(
        [closes_by_symbol[symbol] for symbol in symbols], dtype=float
    )
    if price_matrix.ndim != 2 or price_matrix.shape[1] < 2:
        raise ValueError("Need at least two prices per symbol to estimate returns.")

    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    if returns.shape[1] < 2:
        raise ValueError(
            "Need at least two return observations to estimate covariance."
        )

    # Use recent cumulative return as a simple, interpretable trend signal
    # rather than trusting the noisy sample mean of daily returns.
    effective_window = min(momentum_window, price_matrix.shape[1] - 1)
    if effective_window < 1:
        raise ValueError("Momentum window must allow at least one return observation.")
    momentum = price_matrix[:, -1] / price_matrix[:, -(effective_window + 1)] - 1.0

    # Shrink the momentum signal toward zero for the same reason we shrink the
    # sample mean: recent performance is informative, but far from certain.
    expected_returns = momentum * (1.0 - mean_shrinkage)
    covariance = np.cov(returns, bias=False) * TRADING_DAYS_PER_YEAR
    covariance += np.eye(len(symbols)) * 1e-6

    return EstimatedInputs(
        expected_returns=expected_returns,
        covariance=covariance,
        observations=returns.shape[1],
    )
