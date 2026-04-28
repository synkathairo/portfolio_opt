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

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "cvxpy is required for risk parity. Install dependencies with `uv sync`."
    ) from exc


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
    """Compute equal-risk-contribution long-only weights.

    The log-barrier formulation solves for an unconstrained positive vector and
    then normalizes it; the normalized weights have equal risk contributions.
    """
    del max_iter, tol
    covariance = np.array(covariance, dtype=float)
    if covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if not np.all(np.isfinite(covariance)):
        raise ValueError("Covariance matrix must contain only finite values.")
    covariance = (covariance + covariance.T) / 2.0

    n = covariance.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    if n == 1:
        return np.ones(1, dtype=float)

    min_variance = float(np.min(np.diag(covariance)))
    if min_variance <= 0.0:
        covariance = covariance + np.eye(n) * (abs(min_variance) + 1e-8)

    unscaled = cp.Variable(n, pos=True)
    objective = cp.Minimize(
        0.5 * cp.quad_form(unscaled, covariance) - (1.0 / n) * cp.sum(cp.log(unscaled))
    )
    problem = cp.Problem(objective)
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(
            f"Risk parity optimization failed with status {problem.status}."
        )
    if unscaled.value is None:
        raise RuntimeError("Risk parity optimization returned no weights.")

    weights = np.maximum(np.array(unscaled.value, dtype=float), 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise RuntimeError("Risk parity optimization returned non-positive weights.")
    return weights / total


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
