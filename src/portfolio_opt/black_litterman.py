"""Black-Litterman expected return estimation.

Blends market equilibrium returns with investor views (e.g. momentum signals)
to produce more stable expected return estimates than raw sample means.

References:
    Black & Litterman (1992): "Global Portfolio Optimization"
    Idzorek (2004): "A Step-by-Step Guide to the Black-Litterman Model"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BlackLittermanInputs:
    expected_returns: np.ndarray
    covariance: np.ndarray
    observations: int


def black_litterman_expected_returns(
    covariance: np.ndarray,
    momentum_returns: np.ndarray,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    view_confidence: float = 0.5,
    market_weights: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Black-Litterman posterior expected returns.

    Parameters
    ----------
    covariance : np.ndarray
        Annualized covariance matrix (n x n).
    momentum_returns : np.ndarray
        Trailing momentum returns for each asset (n,). Used as investor views.
    tau : float
        Scaling factor for prior covariance uncertainty (typically 0.01-0.05).
    risk_aversion : float
        Market risk aversion coefficient (typically 2-3 for equities).
    view_confidence : float
        Confidence in each view (0 = no confidence, 1 = full confidence).
    market_weights : np.ndarray | None
        Market-cap weights. If None, equal-weight is used as a proxy.
    """
    n = covariance.shape[0]

    if market_weights is None:
        market_weights = np.ones(n, dtype=float) / n

    # Equilibrium excess returns: Pi = delta * Sigma * w_mkt
    pi = risk_aversion * covariance @ market_weights

    # Views: each asset has a view equal to its momentum return
    # P = identity (each view is on a single asset)
    P = np.eye(n, dtype=float)

    # View returns vector
    Q = momentum_returns

    # View uncertainty: Omega = diagonal, derived from confidence
    # omega_ii = (1/conf - 1) * tau * sigma_ii
    omega_diag = (1.0 / max(view_confidence, 1e-6) - 1.0) * tau * np.diag(covariance)
    Omega = np.diag(omega_diag)

    # Black-Litterman posterior:
    # E[R] = [(tau*Sigma)^-1 + P^T * Omega^-1 * P]^-1 * [(tau*Sigma)^-1 * Pi + P^T * Omega^-1 * Q]
    tau_sigma = tau * covariance
    tau_sigma_inv = np.linalg.inv(tau_sigma + 1e-10 * np.eye(n))
    omega_inv = np.linalg.inv(Omega + 1e-10 * np.eye(n))

    posterior_cov = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    posterior_mean = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

    return posterior_mean


def estimate_inputs_from_black_litterman(
    symbols: list[str],
    closes_by_symbol: dict[str, list[float]],
    momentum_window: int = 84,
    mean_shrinkage: float = 0.5,
    tau: float = 0.05,
    risk_aversion: float = 2.5,
    view_confidence: float = 0.5,
) -> BlackLittermanInputs:
    """Estimate expected returns and covariance using Black-Litterman.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols.
    closes_by_symbol : dict[str, list[float]]
        Daily close prices per symbol, aligned to common length.
    momentum_window : int
        Trailing days for momentum signal.
    mean_shrinkage : float
        Shrinkage applied to momentum returns before using as views.
    tau : float
        Prior uncertainty scaling.
    risk_aversion : float
        Market risk aversion for equilibrium returns.
    view_confidence : float
        Confidence in momentum views.
    """
    price_matrix = np.array([closes_by_symbol[s] for s in symbols], dtype=float)
    if price_matrix.ndim != 2 or price_matrix.shape[1] < 2:
        raise ValueError("Not enough price history for Black-Litterman estimation.")

    returns = price_matrix[:, 1:] / price_matrix[:, :-1] - 1.0
    cov_252 = np.cov(returns) * 252
    # Add diagonal ridge
    cov_252 += 1e-8 * np.eye(len(symbols))

    # Momentum returns as views
    effective_window = min(momentum_window, price_matrix.shape[1] - 1)
    if effective_window < 1:
        raise ValueError("Momentum window must allow at least one return observation.")
    mom_returns = (
        np.array(
            [
                price_matrix[i, -1] / price_matrix[i, -(effective_window + 1)] - 1.0
                for i in range(len(symbols))
            ],
            dtype=float,
        )
        * 252
        / effective_window
    )

    # Apply shrinkage to views
    mom_returns = mom_returns * (1.0 - mean_shrinkage)

    bl_returns = black_litterman_expected_returns(
        covariance=cov_252,
        momentum_returns=mom_returns,
        tau=tau,
        risk_aversion=risk_aversion,
        view_confidence=view_confidence,
    )

    return BlackLittermanInputs(
        expected_returns=bl_returns,
        covariance=cov_252,
        observations=returns.shape[1],
    )
