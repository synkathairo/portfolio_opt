from __future__ import annotations

import pandas as pd


def build_policy(
    cvx,
    symbols: list[str],
    forecasts: pd.DataFrame,
    risk_aversion: float,
    max_weight: float,
    min_cash_weight: float,
    min_invested_weight: float,
    class_min_weights: dict[str, float],
    class_max_weights: dict[str, float],
    asset_classes: dict[str, str],
    core_symbol: str | None = None,
    core_weight: float = 0.0,
    planning_horizon: int = 1,
):
    objective = cvx.ReturnsForecast(r_hat=forecasts) - risk_aversion * cvx.FullCovariance()
    constraints = [
        cvx.LongOnly(applies_to_cash=True),
        cvx.MaxWeights(max_weight),
    ]

    total_exposure = pd.Series(1.0, index=symbols)
    constraints.append(cvx.FactorMaxLimit(total_exposure, 1.0 - min_cash_weight))
    if min_invested_weight > 0.0:
        constraints.append(cvx.FactorMinLimit(total_exposure, min_invested_weight))
    if core_symbol is not None and core_weight > 0.0:
        if core_symbol not in symbols:
            raise ValueError(f"Core symbol {core_symbol} is not in the model universe.")
        if core_weight > max_weight:
            raise ValueError(
                f"Core weight {core_weight} for {core_symbol} exceeds max_weight {max_weight}."
            )
        core_exposure = pd.Series(
            [1.0 if symbol == core_symbol else 0.0 for symbol in symbols],
            index=symbols,
        )
        constraints.append(cvx.FactorMinLimit(core_exposure, core_weight))

    for class_name, lower_bound in class_min_weights.items():
        exposure = pd.Series(
            [1.0 if asset_classes.get(symbol) == class_name else 0.0 for symbol in symbols],
            index=symbols,
        )
        constraints.append(cvx.FactorMinLimit(exposure, lower_bound))

    for class_name, upper_bound in class_max_weights.items():
        exposure = pd.Series(
            [1.0 if asset_classes.get(symbol) == class_name else 0.0 for symbol in symbols],
            index=symbols,
        )
        constraints.append(cvx.FactorMaxLimit(exposure, upper_bound))

    if planning_horizon > 1:
        return cvx.MultiPeriodOptimization(
            objective,
            constraints,
            planning_horizon=planning_horizon,
        )

    return cvx.SinglePeriodOptimization(
        objective=objective,
        constraints=constraints,
        include_cash_return=True,
        fallback_solver="SCS",
        benchmark=cvx.AllCash,
    )
