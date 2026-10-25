"""Point-in-time construction of synthetic spread NAVs.

The pair layer turns a ``SpreadSpec`` plus aligned leg returns into a NAV series.
It is deliberately below the strategy layer: dual momentum, mean variance, and
other strategies consume the resulting NAV like any other instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .instruments import Leg, SpreadSpec

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class PairLayerConfig:
    formation_window: int = 252
    formation_every: int = 126
    borrow_rate: float = 0.03
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR
    min_beta: float = 0.1
    max_beta: float = 5.0

    def __post_init__(self) -> None:
        if self.formation_window < 2:
            raise ValueError("formation_window must be at least 2.")
        if self.formation_every < 1:
            raise ValueError("formation_every must be positive.")
        if self.borrow_rate < 0.0:
            raise ValueError("borrow_rate cannot be negative.")
        if self.trading_days_per_year <= 0:
            raise ValueError("trading_days_per_year must be positive.")
        if self.min_beta <= 0.0 or self.max_beta < self.min_beta:
            raise ValueError("Beta bounds must be positive and ordered.")


def estimate_pair_beta(
    long_returns: np.ndarray,
    short_returns: np.ndarray,
    *,
    min_beta: float = 0.1,
    max_beta: float = 5.0,
) -> float | None:
    """Estimate the OLS sensitivity of the long leg to the short leg."""
    long_values = np.asarray(long_returns, dtype=float)
    short_values = np.asarray(short_returns, dtype=float)
    if long_values.shape != short_values.shape or long_values.size < 2:
        return None
    variance = float(np.var(short_values, ddof=0))
    if variance <= 1e-18:
        return None
    beta = float(np.cov(long_values, short_values, ddof=0)[0, 1] / variance)
    if not np.isfinite(beta) or beta <= min_beta:
        return None
    return float(np.clip(beta, min_beta, max_beta))


def normalized_leg_weights(legs: tuple[Leg, ...]) -> np.ndarray:
    """Return signed leg weights normalized to gross exposure 1.0."""
    raw = np.array([leg.weight for leg in legs], dtype=float)
    gross = float(np.abs(raw).sum())
    if gross <= 0.0:
        raise ValueError("Leg gross weight must be positive.")
    return raw / gross


def spread_period_returns(
    leg_returns: np.ndarray,
    leg_weights: np.ndarray,
    *,
    borrow_rate: float = 0.0,
    trading_days_per_year: int = TRADING_DAYS_PER_YEAR,
) -> np.ndarray:
    """Compute net spread returns from aligned leg returns and signed weights."""
    returns = np.asarray(leg_returns, dtype=float)
    weights = np.asarray(leg_weights, dtype=float)
    if returns.ndim != 2:
        raise ValueError("leg_returns must be a two-dimensional array.")
    if weights.shape != (returns.shape[0],):
        raise ValueError("leg_weights must match the number of legs.")
    gross = float(np.abs(weights).sum())
    if gross <= 0.0:
        raise ValueError("Leg gross weight must be positive.")
    normalized = weights / gross
    result = normalized @ returns
    short_exposure = float(np.abs(normalized[normalized < 0.0]).sum())
    if borrow_rate and short_exposure:
        result = result - borrow_rate / trading_days_per_year * short_exposure
    return result


def _formation_weights(
    spec: SpreadSpec,
    returns_by_symbol: dict[str, np.ndarray],
    formation_index: int,
    config: PairLayerConfig,
) -> np.ndarray:
    if spec.hedge == "beta" and len(spec.legs) == 2:
        start = max(0, formation_index - config.formation_window)
        long_returns = returns_by_symbol[spec.legs[0].symbol][start:formation_index]
        short_returns = returns_by_symbol[spec.legs[1].symbol][start:formation_index]
        beta = estimate_pair_beta(
            long_returns,
            short_returns,
            min_beta=config.min_beta,
            max_beta=config.max_beta,
        )
        if beta is not None:
            return np.array([1.0, -beta], dtype=float)
    return np.array([leg.weight for leg in spec.legs], dtype=float)


def build_spread_nav(
    spec: SpreadSpec,
    returns_by_symbol: dict[str, np.ndarray],
    config: PairLayerConfig,
) -> np.ndarray:
    """Build a point-in-time spread NAV for one instrument.

    The NAV is flat before the first formation. At each formation date, hedge
    weights are estimated using only data through that date. The resulting
    spread returns are net of borrow.
    """
    leg_returns = []
    period_count: int | None = None
    for leg in spec.legs:
        if leg.symbol not in returns_by_symbol:
            raise ValueError(f"Missing returns for leg symbol {leg.symbol!r}.")
        values = np.asarray(returns_by_symbol[leg.symbol], dtype=float)
        if values.ndim != 1:
            raise ValueError("Leg returns must be one-dimensional.")
        if period_count is None:
            period_count = len(values)
        elif len(values) != period_count:
            raise ValueError("All leg return series must have the same length.")
        leg_returns.append(values)
    if period_count is None:
        raise ValueError("Spread must have at least one leg.")

    spread_values = np.zeros(period_count, dtype=float)
    for formation in range(
        config.formation_window,
        period_count,
        config.formation_every,
    ):
        weights = _formation_weights(spec, returns_by_symbol, formation, config)
        end = min(formation + config.formation_every, period_count)
        leg_matrix = np.array(
            [returns[formation:end] for returns in leg_returns],
            dtype=float,
        )
        spread_values[formation:end] = spread_period_returns(
            leg_matrix,
            weights,
            borrow_rate=config.borrow_rate,
            trading_days_per_year=config.trading_days_per_year,
        )

    return np.concatenate(([1.0], np.cumprod(1.0 + spread_values)))
