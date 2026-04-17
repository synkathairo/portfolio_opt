from __future__ import annotations

from math import floor

from .config import OptimizationConfig
from .types import (
    AccountSnapshot,
    OrderPlan,
    Position,
    TrailingStopPlan,
    TrailingStopPlanResult,
    UnprotectedTrailingStopQty,
)


def current_weights(
    symbols: list[str],
    account: AccountSnapshot,
    positions: list[Position],
) -> dict[str, float]:
    by_symbol = {position.symbol: position for position in positions}
    if account.equity <= 0:
        raise ValueError("Account equity must be positive.")
    # Any symbol not currently held is treated as a zero-weight position so the
    # optimizer and rebalance layer can work off the same ordered universe.
    return {
        symbol: by_symbol.get(
            symbol, Position(symbol=symbol, qty=0.0, market_value=0.0)
        ).market_value
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
    open_orders: list[dict] | None = None,
) -> list[OrderPlan]:
    weights_now = current_weights(symbols, account, positions)

    # Adjust current weights for any pending orders so we don't
    # double-submit or submit conflicting trades.
    if open_orders:
        for order in open_orders:
            symbol = order.get("symbol")
            if symbol in symbols:
                if _order_value(order.get("type")) == "trailing_stop":
                    continue
                # Calculate notional value of the open order
                qty = float(order.get("qty", 0) or 0)
                if order.get("side") == "sell":
                    qty = -qty  # Sell reduces position
                price = latest_prices.get(symbol, 0.0)
                notional = qty * price
                # Adjust weight
                if account.equity > 0:
                    weights_now[symbol] += notional / account.equity

    plans: list[OrderPlan] = []
    for symbol, target_weight in zip(symbols, target_weights, strict=True):
        current_weight = weights_now.get(symbol, 0.0)
        delta_weight = float(target_weight - current_weight)
        # Convert weight deltas into notional dollars so order sizing is tied
        # to portfolio equity instead of per-asset share math in this layer.
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


def build_trailing_stop_plan(
    *,
    symbols: list[str],
    target_weights: list[float],
    positions: list[Position],
    open_orders: list[dict] | None,
    trailing_stop: float,
    rebalance_threshold: float,
) -> TrailingStopPlanResult:
    target_by_symbol = {
        symbol: float(weight)
        for symbol, weight in zip(symbols, target_weights, strict=True)
    }
    protected_symbols = {
        str(order.get("symbol"))
        for order in open_orders or []
        if _order_value(order.get("type")) == "trailing_stop"
        and _order_value(order.get("side")) == "sell"
    }

    trail_percent = round(float(trailing_stop) * 100.0, 6)
    plans: list[TrailingStopPlan] = []
    unprotected_qty: list[UnprotectedTrailingStopQty] = []
    for position in positions:
        if position.symbol not in target_by_symbol:
            continue
        position_qty = float(position.qty)
        if position_qty <= 0.0:
            continue
        if target_by_symbol[position.symbol] < rebalance_threshold:
            continue
        if position.symbol in protected_symbols:
            continue
        whole_share_qty = float(floor(position_qty))
        remainder_qty = round(position_qty - whole_share_qty, 6)
        if remainder_qty > 0.0:
            unprotected_qty.append(
                UnprotectedTrailingStopQty(
                    symbol=position.symbol,
                    position_qty=round(position_qty, 6),
                    unprotected_qty=remainder_qty,
                )
            )
        if whole_share_qty >= 1.0:
            plans.append(
                TrailingStopPlan(
                    symbol=position.symbol,
                    qty=whole_share_qty,
                    side="sell",
                    trail_percent=trail_percent,
                    time_in_force="gtc",
                )
            )
    return TrailingStopPlanResult(orders=plans, unprotected_qty=unprotected_qty)


def _order_value(value: object) -> str:
    raw = getattr(value, "value", value)
    return str(raw)
