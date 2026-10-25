"""Thin wrappers and registry for strategies over an instrument universe.

Every wrapper receives the same ``StrategyData``. The pair layer is already
embedded in the instrument NAVs; these functions do not contain pair logic.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .backtest import (
    BacktestResult,
    run_backtest,
    run_dual_momentum_backtest,
    run_factor_momentum_backtest,
    run_protective_momentum_backtest,
)
from .fixed_allocation import run_fixed_allocation_backtest
from .regime_adaptive import run_regime_adaptive_backtest
from .strategy_data import StrategyData, UniverseBacktestResult

StrategyRunner = Callable[..., BacktestResult]


def run_dual_momentum(data: StrategyData, **kwargs: Any) -> BacktestResult:
    kwargs.setdefault("absolute_threshold", 0.0)
    return run_dual_momentum_backtest(
        symbols=list(data.closes),
        closes_by_symbol=data.closes,
        asset_classes=data.asset_classes,
        **kwargs,
    )


def run_mean_variance(data: StrategyData, **kwargs: Any) -> BacktestResult:
    return run_backtest(
        symbols=list(data.closes),
        closes_by_symbol=data.closes,
        **kwargs,
    )


def run_factor_momentum(data: StrategyData, **kwargs: Any) -> BacktestResult:
    kwargs.setdefault("absolute_threshold", 0.0)
    return run_factor_momentum_backtest(
        symbols=list(data.closes),
        closes_by_symbol=data.closes,
        asset_classes=data.asset_classes,
        **kwargs,
    )


def run_protective_momentum(data: StrategyData, **kwargs: Any) -> BacktestResult:
    kwargs.setdefault("absolute_threshold", 0.0)
    return run_protective_momentum_backtest(
        symbols=list(data.closes),
        closes_by_symbol=data.closes,
        asset_classes=data.asset_classes,
        **kwargs,
    )


def run_regime_adaptive(data: StrategyData, **kwargs: Any) -> BacktestResult:
    kwargs.setdefault("absolute_threshold", 0.0)
    return run_regime_adaptive_backtest(
        symbols=list(data.closes),
        closes_by_symbol=data.closes,
        asset_classes=data.asset_classes,
        **kwargs,
    )


def run_fixed_allocation(data: StrategyData, **kwargs: Any) -> BacktestResult:
    return run_fixed_allocation_backtest(
        symbols=list(data.closes),
        closes_by_symbol=data.closes,
        **kwargs,
    )


STRATEGIES: dict[str, StrategyRunner] = {
    "dual-momentum": run_dual_momentum,
    "mean-variance": run_mean_variance,
    "factor-momentum": run_factor_momentum,
    "protective-momentum": run_protective_momentum,
    "regime-adaptive": run_regime_adaptive,
    "fixed-allocation": run_fixed_allocation,
}


def run_strategy(
    name: str,
    data: StrategyData,
    **kwargs: Any,
) -> UniverseBacktestResult:
    """Dispatch a strategy over already-normalized instrument data."""
    try:
        runner = STRATEGIES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown strategy: {name}") from exc
    result = runner(data, **kwargs)
    expected = len(data.ids) + 1
    if len(result.latest_weights) != expected:
        raise RuntimeError(
            f"Strategy {name!r} returned {len(result.latest_weights)} weights; "
            f"expected {expected}."
        )
    return UniverseBacktestResult(
        result=result,
        instrument_ids=data.ids,
        cash_id=data.cash_id,
    )
