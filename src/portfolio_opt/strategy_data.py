"""Common strategy inputs derived from an instrument universe.

This module knows about instruments, NAVs, metadata, and cash. It knows nothing
about any particular strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backtest import BacktestResult
from .instruments import InstrumentUniverse


@dataclass(frozen=True)
class StrategyData:
    ids: tuple[str, ...]
    navs: np.ndarray
    closes: dict[str, list[float]]
    returns: np.ndarray
    asset_classes: dict[str, str]
    categories: dict[str, str]
    kinds: dict[str, str]
    cash_id: str
    cash_nav: np.ndarray

    @property
    def period_count(self) -> int:
        return self.navs.shape[1]


@dataclass(frozen=True)
class UniverseBacktestResult:
    result: BacktestResult
    instrument_ids: tuple[str, ...]
    cash_id: str

    @property
    def latest_weights(self) -> dict[str, float]:
        weights = np.asarray(self.result.latest_weights, dtype=float)
        return {
            instrument_id: float(weights[index])
            for index, instrument_id in enumerate(self.instrument_ids)
        }

    @property
    def latest_cash_weight(self) -> float:
        return float(self.result.latest_weights[len(self.instrument_ids)])


def build_strategy_data(
    *,
    universe: InstrumentUniverse,
    cash_nav: list[float] | np.ndarray | None = None,
    cash_id: str = "CASH",
) -> StrategyData:
    """Convert an instrument universe into a generic strategy panel."""
    ids, navs = universe.nav_matrix()
    cash_values = (
        np.ones(universe.period_count, dtype=float)
        if cash_nav is None
        else np.asarray(cash_nav, dtype=float)
    )
    if cash_values.shape != (universe.period_count,):
        raise ValueError("cash_nav must match the instrument NAV length.")
    if np.any(cash_values <= 0.0):
        raise ValueError("cash_nav must be positive.")
    if cash_id in universe.instruments:
        raise ValueError(f"Cash id collides with an instrument: {cash_id}")

    closes = {
        instrument_id: navs[index].tolist() for index, instrument_id in enumerate(ids)
    }
    closes[cash_id] = cash_values.tolist()
    return_columns = []
    for values in closes.values():
        prices = np.asarray(values, dtype=float)
        return_columns.append(prices[1:] / prices[:-1] - 1.0)
    asset_classes = universe.asset_classes()
    asset_classes[cash_id] = "cash_like"
    categories = {
        instrument_id: instrument.category
        for instrument_id, instrument in universe.instruments.items()
    }
    categories[cash_id] = "cash"
    kinds: dict[str, str] = dict(universe.kinds())
    kinds[cash_id] = "cash"
    return StrategyData(
        ids=tuple(ids),
        navs=navs,
        closes=closes,
        returns=np.array(return_columns, dtype=float),
        asset_classes=asset_classes,
        categories=categories,
        kinds=kinds,
        cash_id=cash_id,
        cash_nav=cash_values,
    )
