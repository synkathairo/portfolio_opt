from __future__ import annotations

from .config import OptimizationConfig
from .types import AccountSnapshot, OrderPlan, Position


def current_weights(
    symbols: list[str],
    account: AccountSnapshot,
    positions: list[Position],
) -> dict[str, float]:
    by_symbol = {position.symbol: position for position in positions}
    if account.equity <= 0:
        raise ValueError("Account equity must be positive.")
    return {
        symbol: by_symbol.get(symbol, Position(symbol=symbol, qty=0.0, market_value=0.0)).market_value
        / account.equity
        for symbol in symbols
    }


def build_order_plan(
    symbols: list[str],
    target_weights: list[float],
    account: AccountSnapshot,
    positions: list[Position],
    latest_prices: dict[str, float],
    config: OptimizationConfig,
) -> list[OrderPlan]:
    weights_now = current_weights(symbols, account, positions)
    plans: list[OrderPlan] = []
    for symbol, target_weight in zip(symbols, target_weights, strict=True):
        current_weight = weights_now.get(symbol, 0.0)
        delta_weight = float(target_weight - current_weight)
        notional_usd = abs(delta_weight) * account.equity
        # Ignore small drifts so the strategy does not churn on every run.
        if abs(delta_weight) < config.rebalance_threshold:
            continue
        if latest_prices.get(symbol, 0.0) <= 0.0:
            continue
        plans.append(
            OrderPlan(
                symbol=symbol,
                current_weight=round(current_weight, 6),
                target_weight=round(float(target_weight), 6),
                delta_weight=round(delta_weight, 6),
                side="buy" if delta_weight > 0 else "sell",
                notional_usd=round(notional_usd, 2),
            )
        )
    return plans
