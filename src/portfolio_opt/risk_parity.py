"""Risk parity portfolio construction.

Weights each asset so it contributes equal marginal risk to the portfolio.
In the simple long-only case this is inverse-vol weighting across all assets.

References:
    Maillard, Roncalli, Teïletche (2010): "The Properties of Equally-Weighted
    Risk Contribution Portfolios"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RiskParityInputs:
    weights: np.ndarray
    covariance: np.ndarray
    observations: int


def risk_parity_weights(
    covariance: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-10,
) -> np.ndarray:
    """Compute risk-parity weights using the cyclical coordinate descent algorithm.

    For long-only portfolios this converges to weights where each asset
    contributes equally to portfolio variance: w_i * (Sigma w)_i = constant.
    """
    n = covariance.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    # Start from inverse-vol initialization
    vols = np.sqrt(np.diag(covariance))
    w = 1.0 / np.maximum(vols, 1e-8)
    w /= w.sum()

    for _ in range(max_iter):
        w_old = w.copy()
        for i in range(n):
            sigma_i = covariance[i, :] @ w
            other = covariance[i, :] @ w - covariance[i, i] * w[i]
            # Solve: w_i^2 * sigma_ii + w_i * other - target = 0
            # where target = w_j * (Sigma w)_j for all j (equal risk contribution)
            # The closed-form update for RC = 1/n is:
            if abs(other) < 1e-16:
                continue
            w[i] = max(
                0.0,
                (-other + np.sqrt(other**2 + 4 * covariance[i, i] * (1.0 / n)))
                / (2 * covariance[i, i]),
            )
        # Normalize
        total = w.sum()
        if total > 0:
            w /= total
        if np.max(np.abs(w - w_old)) < tol:
            break

    return w


def estimate_inputs_risk_parity(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    lookback_days: int = 252,
) -> RiskParityInputs:
    """Estimate risk-parity weights from recent covariance."""
    price_matrix = np.array([closes_by_symbol[s] for s in symbols], dtype=float)
    if price_matrix.ndim != 2 or price_matrix.shape[1] < 2:
        raise ValueError("Not enough price history for risk parity estimation.")

    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    cov_252 = np.cov(returns) * 252
    cov_252 += 1e-8 * np.eye(len(symbols))

    weights = risk_parity_weights(cov_252)

    return RiskParityInputs(
        weights=weights,
        covariance=cov_252,
        observations=returns.shape[1],
    )
