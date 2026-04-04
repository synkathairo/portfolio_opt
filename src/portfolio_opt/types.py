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
