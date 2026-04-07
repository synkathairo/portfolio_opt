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
