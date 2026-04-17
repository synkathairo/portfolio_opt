from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    market_value: float


@dataclass(frozen=True)
class AccountSnapshot:
    equity: float


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    current_weight: float
    target_weight: float
    delta_weight: float
    side: str
    notional_usd: float


@dataclass(frozen=True)
class TrailingStopPlan:
    symbol: str
    qty: float
    side: str
    trail_percent: float
    time_in_force: str


@dataclass(frozen=True)
class UnprotectedTrailingStopQty:
    symbol: str
    position_qty: float
    unprotected_qty: float


@dataclass(frozen=True)
class TrailingStopPlanResult:
    orders: list[TrailingStopPlan]
    unprotected_qty: list[UnprotectedTrailingStopQty]
