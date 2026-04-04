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
) -> np.ndarray:
    asset_count = expected_returns.shape[0]
    weights = cp.Variable(asset_count)

    # Classic single-period mean-variance objective with box constraints.
    objective = cp.Maximize(
        expected_returns @ weights - config.risk_aversion * cp.quad_form(weights, covariance)
    )
    constraints = [
        cp.sum(weights) == 1,
        weights >= config.min_weight,
        weights <= config.max_weight,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if weights.value is None:
        raise RuntimeError(f"Optimization failed with status {problem.status}.")

    solution = np.array(weights.value, dtype=float)
    # Normalize after clipping to keep the output usable even if the solver
    # returns tiny numerical violations around the bounds.
    clipped = np.clip(solution, config.min_weight, config.max_weight)
    total = clipped.sum()
    if total <= 0:
        raise RuntimeError("Optimization returned non-positive total weight.")
    return clipped / total
