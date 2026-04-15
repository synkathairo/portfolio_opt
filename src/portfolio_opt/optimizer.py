from __future__ import annotations

import numpy as np

from .config import OptimizationConfig

try:
    import cvxpy as cp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "cvxpy is required for optimization. Install dependencies with `pip install -e .`."
    ) from exc

SOLVER = cp.CLARABEL
# SOLVER = cp.ECOS


def _class_constraint_names(config: OptimizationConfig) -> list[str]:
    return list(config.class_min_weights) + [
        name
        for name in config.class_max_weights
        if name not in config.class_min_weights
    ]


def _build_constraints(
    weights: cp.Variable,
    config: OptimizationConfig,
    asset_class_matrix: np.ndarray | None,
    baseline_weights: np.ndarray | None = None,
) -> list[cp.Constraint]:
    constraints = [weights >= config.min_weight, weights <= config.max_weight]
    if config.force_full_investment:
        constraints.append(cp.sum(weights) == 1)
    else:
        constraints.append(cp.sum(weights) <= 1.0 - config.min_cash_weight)
        constraints.append(cp.sum(weights) >= config.min_invested_weight)
    if baseline_weights is not None and config.max_turnover is not None:
        constraints.append(cp.norm1(weights - baseline_weights) <= config.max_turnover)
    if asset_class_matrix is not None:
        class_exposures = asset_class_matrix @ weights
        for class_index, class_name in enumerate(_class_constraint_names(config)):
            min_weight = config.class_min_weights.get(class_name)
            if min_weight is not None:
                constraints.append(class_exposures[class_index] >= min_weight)
            max_weight = config.class_max_weights.get(class_name)
            if max_weight is not None:
                constraints.append(class_exposures[class_index] <= max_weight)
    return constraints


def _finalize_solution(
    solution: np.ndarray,
    config: OptimizationConfig,
) -> np.ndarray:
    clipped = np.clip(solution, config.min_weight, config.max_weight)
    total = clipped.sum()
    if config.force_full_investment:
        if total <= 0:
            raise RuntimeError("Optimization returned non-positive total weight.")
        return clipped / total
    return clipped


def effective_turnover_penalty(
    config: OptimizationConfig,
    current_weights: np.ndarray | None,
) -> float:
    if current_weights is None:
        return 0.0
    invested_weight = float(
        np.clip(np.sum(np.array(current_weights, dtype=float)), 0.0, 1.0)
    )
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
    constraints = _build_constraints(
        weights,
        config,
        asset_class_matrix,
        baseline_weights=baseline_weights,
    )
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=SOLVER)

    if weights.value is None:
        raise RuntimeError(f"Optimization failed with status {problem.status}.")

    solution = np.array(weights.value, dtype=float)
    return _finalize_solution(solution, config)


def project_weights(
    target_weights: np.ndarray,
    config: OptimizationConfig,
    current_weights: np.ndarray | None = None,
    asset_class_matrix: np.ndarray | None = None,
) -> np.ndarray:
    asset_count = target_weights.shape[0]
    weights = cp.Variable(asset_count)
    baseline_weights = (
        np.array(current_weights, dtype=float)
        if current_weights is not None
        else np.zeros(asset_count, dtype=float)
    )
    scaled_turnover_penalty = effective_turnover_penalty(config, baseline_weights)

    objective = cp.Minimize(
        cp.sum_squares(weights - np.array(target_weights, dtype=float))
        + scaled_turnover_penalty * cp.norm1(weights - baseline_weights)
    )
    constraints = _build_constraints(
        weights,
        config,
        asset_class_matrix,
        baseline_weights=baseline_weights,
    )
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=SOLVER)

    if weights.value is None:
        raise RuntimeError(f"Projection failed with status {problem.status}.")

    return _finalize_solution(np.array(weights.value, dtype=float), config)


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
        expected_returns @ weights - risk_aversion * cp.quad_form(weights, covariance)
    )
    constraints = [weights >= min_weight, weights <= max_weight]
    if force_full_investment:
        constraints.append(cp.sum(weights) == 1)
    else:
        constraints.append(cp.sum(weights) <= 1.0)

    problem = cp.Problem(objective, constraints)
    problem.solve(solver=SOLVER)

    if weights.value is None:
        raise RuntimeError(f"Basket optimization failed with status {problem.status}.")

    solution = np.array(weights.value, dtype=float)
    return np.clip(solution, min_weight, max_weight)
