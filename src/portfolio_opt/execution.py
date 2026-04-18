from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Protocol

from .config import OptimizationConfig
from .rebalance import build_order_plan
from .types import AccountSnapshot, OrderPlan, Position


class RebalanceBroker(Protocol):
    def submit_order_plan(self, plans: list[OrderPlan]) -> list[dict[str, Any]]: ...

    def wait_for_submitted_orders(
        self,
        submitted_orders: list[dict[str, Any]],
        *,
        timeout_seconds: float = 60.0,
        poll_seconds: float = 2.0,
    ) -> list[dict[str, Any]]: ...

    def get_account(
        self,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> AccountSnapshot: ...

    def get_positions(
        self,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> list[Position]: ...

    def get_open_orders(self) -> list[dict[str, Any]]: ...

    def get_latest_prices(
        self,
        symbols: list[str],
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, float]: ...


@dataclass(frozen=True)
class RebalanceExecutionResult:
    submitted_orders: list[dict[str, Any]]
    sell_fill_statuses: list[dict[str, Any]]
    buy_plan: list[OrderPlan]
    skipped_buys_reason: str | None = None


def _all_orders_filled(statuses: list[dict[str, Any]]) -> bool:
    return all(str(item.get("status")) == "filled" for item in statuses)


def submit_rebalance_sell_first(
    *,
    broker: RebalanceBroker,
    plan: list[OrderPlan],
    symbols: list[str],
    target_weights: list[float],
    config: OptimizationConfig,
    use_cache: bool = False,
    refresh_cache: bool = False,
    offline: bool = False,
) -> RebalanceExecutionResult:
    sell_plan = [item for item in plan if item.side == "sell"]
    buy_plan = [item for item in plan if item.side == "buy"]
    submitted_orders: list[dict[str, Any]] = []
    sell_fill_statuses: list[dict[str, Any]] = []

    if sell_plan:
        submitted_sells = broker.submit_order_plan(sell_plan)
        submitted_orders.extend(submitted_sells)
        if len(submitted_sells) != len(sell_plan):
            reason = "one or more sell orders were not accepted"
            print(f"Skipping buy orders because {reason}.", file=sys.stderr)
            return RebalanceExecutionResult(
                submitted_orders=submitted_orders,
                sell_fill_statuses=sell_fill_statuses,
                buy_plan=[],
                skipped_buys_reason=reason,
            )
        sell_fill_statuses = broker.wait_for_submitted_orders(submitted_sells)
        if not _all_orders_filled(sell_fill_statuses):
            reason = "one or more sell orders did not fill"
            print(f"Skipping buy orders because {reason}.", file=sys.stderr)
            return RebalanceExecutionResult(
                submitted_orders=submitted_orders,
                sell_fill_statuses=sell_fill_statuses,
                buy_plan=[],
                skipped_buys_reason=reason,
            )

    if not buy_plan:
        return RebalanceExecutionResult(
            submitted_orders=submitted_orders,
            sell_fill_statuses=sell_fill_statuses,
            buy_plan=[],
        )

    refreshed_account = broker.get_account(
        use_cache=False,
        refresh_cache=refresh_cache,
        offline=offline,
    )
    refreshed_positions = broker.get_positions(
        use_cache=False,
        refresh_cache=refresh_cache,
        offline=offline,
    )
    refreshed_open_orders = broker.get_open_orders()
    latest_buy_prices = broker.get_latest_prices(
        [item.symbol for item in buy_plan],
        use_cache=use_cache,
        refresh_cache=refresh_cache,
        offline=offline,
    )
    refreshed_plan = build_order_plan(
        symbols=symbols,
        target_weights=target_weights,
        account=refreshed_account,
        positions=refreshed_positions,
        latest_prices=latest_buy_prices,
        config=config,
        open_orders=refreshed_open_orders,
    )
    refreshed_buy_plan = [item for item in refreshed_plan if item.side == "buy"]
    if not refreshed_buy_plan:
        return RebalanceExecutionResult(
            submitted_orders=submitted_orders,
            sell_fill_statuses=sell_fill_statuses,
            buy_plan=[],
        )

    submitted_buys = broker.submit_order_plan(refreshed_buy_plan)
    submitted_orders.extend(submitted_buys)

    return RebalanceExecutionResult(
        submitted_orders=submitted_orders,
        sell_fill_statuses=sell_fill_statuses,
        buy_plan=refreshed_buy_plan,
    )
