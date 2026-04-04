from __future__ import annotations

import json
from dataclasses import asdict
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import AlpacaConfig
from .types import AccountSnapshot, OrderPlan, Position


class AlpacaClient:
    def __init__(self, config: AlpacaConfig) -> None:
        self._config = config

    def get_account(self) -> AccountSnapshot:
        payload = self._request_json("GET", "/v2/account")
        return AccountSnapshot(equity=float(payload["equity"]))

    def get_positions(self) -> list[Position]:
        payload = self._request_json("GET", "/v2/positions")
        return [
            Position(
                symbol=row["symbol"],
                qty=float(row["qty"]),
                market_value=float(row["market_value"]),
            )
            for row in payload
        ]

    def get_latest_prices(self, symbols: list[str]) -> dict[str, float]:
        query = urlencode({"symbols": ",".join(symbols), "feed": "iex"})
        payload = self._request_json("GET", f"/v2/stocks/trades/latest?{query}", data_api=True)
        trades = payload.get("trades", {})
        return {
            symbol: float(trade["p"])
            for symbol, trade in trades.items()
        }

    def submit_order_plan(self, plans: list[OrderPlan]) -> None:
        for plan in plans:
            # Notional market orders keep the rebalance layer independent from
            # per-symbol share rounding logic.
            order = {
                "symbol": plan.symbol,
                "notional": round(plan.notional_usd, 2),
                "side": plan.side,
                "type": "market",
                "time_in_force": "day",
            }
            self._request_json("POST", "/v2/orders", payload=order)

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        data_api: bool = False,
    ) -> dict | list:
        base_url = self._config.data_url if data_api else self._config.base_url
        headers = {
            "APCA-API-KEY-ID": self._config.api_key,
            "APCA-API-SECRET-KEY": self._config.api_secret,
            "Content-Type": "application/json",
        }
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(f"{base_url}{path}", data=body, headers=headers, method=method)
        try:
            with urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca API error {exc.code}: {message}") from exc
        except URLError as exc:  # pragma: no cover
            raise RuntimeError(f"Alpaca connection error: {exc.reason}") from exc


def format_order_plans(plans: list[OrderPlan]) -> str:
    return json.dumps([asdict(plan) for plan in plans], indent=2)
