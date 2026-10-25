"""Generic instrument and spread definitions for strategy backtests.

A pair or basket spread is a synthetic instrument. Strategies operate on
instrument NAVs and metadata; the legs are implementation details used by the
pair layer and, later, by execution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

InstrumentKind = Literal["stock", "etf", "spread"]
HedgeMode = Literal["fixed", "beta"]


@dataclass(frozen=True)
class Leg:
    symbol: str
    weight: float

    def __post_init__(self) -> None:
        if not self.symbol:
            raise ValueError("Leg symbol must be non-empty.")
        if not np.isfinite(self.weight) or self.weight == 0.0:
            raise ValueError("Leg weight must be finite and nonzero.")


@dataclass(frozen=True)
class SpreadSpec:
    id: str
    legs: tuple[Leg, ...]
    category: str
    asset_class: str
    hedge: HedgeMode = "beta"
    shortable: bool = True

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Spread id must be non-empty.")
        if len(self.legs) < 2:
            raise ValueError("A spread must have at least two legs.")
        symbols = [leg.symbol for leg in self.legs]
        if len(set(symbols)) != len(symbols):
            raise ValueError("Spread legs must have unique symbols.")
        if not any(leg.weight > 0.0 for leg in self.legs):
            raise ValueError("A spread must have at least one positive leg.")
        if not any(leg.weight < 0.0 for leg in self.legs):
            raise ValueError("A spread must have at least one negative leg.")
        if sum(abs(leg.weight) for leg in self.legs) <= 0.0:
            raise ValueError("Spread gross weight must be positive.")
        if self.hedge not in {"fixed", "beta"}:
            raise ValueError("Spread hedge must be 'fixed' or 'beta'.")
        if not self.category or not self.asset_class:
            raise ValueError("Spread category and asset_class must be non-empty.")


@dataclass(frozen=True)
class Instrument:
    id: str
    kind: InstrumentKind
    nav: tuple[float, ...]
    asset_class: str
    category: str
    available: tuple[bool, ...]
    legs: tuple[Leg, ...] | None = None
    shortable: bool = False

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("Instrument id must be non-empty.")
        if not self.nav:
            raise ValueError("Instrument NAV must be non-empty.")
        if len(self.nav) != len(self.available):
            raise ValueError("Instrument NAV and availability must have equal length.")
        if not np.all(np.isfinite(np.asarray(self.nav, dtype=float))):
            raise ValueError("Instrument NAV must be finite.")
        if self.kind == "spread" and not self.legs:
            raise ValueError("Spread instruments must carry legs.")


@dataclass(frozen=True)
class InstrumentUniverse:
    instruments: dict[str, Instrument]

    def __post_init__(self) -> None:
        if not self.instruments:
            raise ValueError("Instrument universe must not be empty.")
        lengths = {len(instrument.nav) for instrument in self.instruments.values()}
        if len(lengths) != 1:
            raise ValueError("All instruments must have the same NAV length.")

    @property
    def period_count(self) -> int:
        return len(next(iter(self.instruments.values())).nav)

    def ids(self) -> list[str]:
        return list(self.instruments)

    def nav_matrix(self) -> tuple[list[str], np.ndarray]:
        ids = self.ids()
        matrix = np.array(
            [self.instruments[instrument_id].nav for instrument_id in ids],
            dtype=float,
        )
        return ids, matrix

    def asset_classes(self) -> dict[str, str]:
        return {
            instrument_id: instrument.asset_class
            for instrument_id, instrument in self.instruments.items()
        }

    def kinds(self) -> dict[str, InstrumentKind]:
        return {
            instrument_id: instrument.kind
            for instrument_id, instrument in self.instruments.items()
        }
