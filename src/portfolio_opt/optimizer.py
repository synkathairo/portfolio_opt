from __future__ import annotations

import numpy as np

from .config import OptimizationConfig

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "cvxpy is required for optimization. Install dependencies with `pip install -e .`."
    ) from exc


def effective_turnover_penalty(
    config: OptimizationConfig,
    current_weights: np.ndarray | None,
) -> float:
    if current_weights is None:
        return 0.0
    invested_weight = float(np.clip(np.sum(np.array(current_weights, dtype=float)), 0.0, 1.0))
    # Scale the turnover penalty by the currently invested risky weight so an
    # all-cash portfolio is not punished for entering its first positions.
    return config.turnover_penalty * invested_weight


def optimize_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    config: OptimizationConfig,
    current_weights: np.ndarray | None = None,
    asset_class_matrix: np.ndarray | None = None,
) -> np.ndarray:
    asset_count = expected_returns.shape[0]
    weights = cp.Variable(asset_count)
    baseline_weights = (
        np.array(current_weights, dtype=float)
        if current_weights is not None
        else np.zeros(asset_count, dtype=float)
    )
    scaled_turnover_penalty = effective_turnover_penalty(config, baseline_weights)

    # This objective still uses single-period mean-variance optimization, but it
    # is more realistic than the original version because it penalizes turnover
    # away from the current holdings. The turnover penalty is scaled by the
    # currently invested weight so a cash-only account can establish an initial
    # portfolio without being artificially discouraged from investing at all.
    objective = cp.Maximize(
        expected_returns @ weights
        - config.risk_aversion * cp.quad_form(weights, covariance)
        - scaled_turnover_penalty * cp.norm1(weights - baseline_weights)
    )
    constraints = [weights >= config.min_weight, weights <= config.max_weight]
    if config.force_full_investment:
        constraints.append(cp.sum(weights) == 1)
    else:
        # Allowing partial investment leaves the remainder in cash.
        constraints.append(cp.sum(weights) <= 1.0 - config.min_cash_weight)
        constraints.append(cp.sum(weights) >= config.min_invested_weight)
    if config.max_turnover is not None:
        # Hard turnover cap for operational control on top of the soft penalty.
        constraints.append(cp.norm1(weights - baseline_weights) <= config.max_turnover)
    if asset_class_matrix is not None:
        class_exposures = asset_class_matrix @ weights
        for class_index, min_weight in enumerate(config.class_min_weights.values()):
            constraints.append(class_exposures[class_index] >= min_weight)
        for class_index, max_weight in enumerate(config.class_max_weights.values()):
            constraints.append(class_exposures[class_index] <= max_weight)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None:
        raise RuntimeError(f"Optimization failed with status {problem.status}.")

    solution = np.array(weights.value, dtype=float)
    # Normalize after clipping to keep the output usable even if the solver
    # returns tiny numerical violations around the bounds.
    clipped = np.clip(solution, config.min_weight, config.max_weight)
    total = clipped.sum()
    if config.force_full_investment:
        if total <= 0:
            raise RuntimeError("Optimization returned non-positive total weight.")
        # With full investment enabled this keeps weights summing to one.
        return clipped / total

    # In partial-investment mode the leftover weight is explicit cash, so keep
    # the raw clipped solution instead of renormalizing it back to 100%.
    return clipped


def optimize_basket_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    min_weight: float = 0.0,
    max_weight: float = 1.0,
    risk_aversion: float = 1.0,
    force_full_investment: bool = True,
) -> np.ndarray:
    """Lightweight mean-variance optimization for a small pre-selected basket.

    Unlike the full-asset optimizer this has no turnover penalty, no asset-class
    constraints, and no partial-investment logic — just risk-aware sizing of
    the assets the momentum signal already picked.
    """
    asset_count = expected_returns.shape[0]
    weights = cp.Variable(asset_count)

    objective = cp.Maximize(
        expected_returns @ weights
        - risk_aversion * cp.quad_form(weights, covariance)
    )
    constraints = [weights >= min_weight, weights <= max_weight]
    if force_full_investment:
        constraints.append(cp.sum(weights) == 1)
    else:
        constraints.append(cp.sum(weights) <= 1.0)

    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None:
        raise RuntimeError(f"Basket optimization failed with status {problem.status}.")

    solution = np.array(weights.value, dtype=float)
    return np.clip(solution, min_weight, max_weight)
