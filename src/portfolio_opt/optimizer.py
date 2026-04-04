from __future__ import annotations

import numpy as np

from .config import OptimizationConfig

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "cvxpy is required for optimization. Install dependencies with `pip install -e .`."
    ) from exc


def optimize_weights(
    expected_returns: np.ndarray,
    covariance: np.ndarray,
    config: OptimizationConfig,
    current_weights: np.ndarray | None = None,
) -> np.ndarray:
    asset_count = expected_returns.shape[0]
    weights = cp.Variable(asset_count)
    baseline_weights = (
        np.array(current_weights, dtype=float)
        if current_weights is not None
        else np.zeros(asset_count, dtype=float)
    )

    # This objective still uses single-period mean-variance optimization, but it
    # is more realistic than the original version because it penalizes turnover
    # away from the current holdings.
    objective = cp.Maximize(
        expected_returns @ weights
        - config.risk_aversion * cp.quad_form(weights, covariance)
        - config.turnover_penalty * cp.norm1(weights - baseline_weights)
    )
    constraints = [weights >= config.min_weight, weights <= config.max_weight]
    if config.force_full_investment:
        constraints.append(cp.sum(weights) == 1)
    else:
        constraints.append(cp.sum(weights) <= 1)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None:
        raise RuntimeError(f"Optimization failed with status {problem.status}.")

    solution = np.array(weights.value, dtype=float)
    # Normalize after clipping to keep the output usable even if the solver
    # returns tiny numerical violations around the bounds.
    # With force_full_investment enabled this keeps the weights summing to one.
    clipped = np.clip(solution, config.min_weight, config.max_weight)
    total = clipped.sum()
    if total <= 0:
        raise RuntimeError("Optimization returned non-positive total weight.")
    return clipped / total
