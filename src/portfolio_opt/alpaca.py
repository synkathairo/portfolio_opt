from __future__ import annotations

import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .cache import cache_path, read_cache, write_cache
from .config import AlpacaConfig
from .types import AccountSnapshot, OrderPlan, Position


class AlpacaClient:
    def __init__(self, config: AlpacaConfig) -> None:
        self._config = config

    def get_account(
        self,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> AccountSnapshot:
        payload = self._cached_json(
            "account",
            {"kind": "account"},
            lambda: self._request_json("GET", "/v2/account"),
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
        return AccountSnapshot(equity=float(payload["equity"]))

    def get_positions(
        self,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> list[Position]:
        payload = self._cached_json(
            "positions",
            {"kind": "positions"},
            lambda: self._request_json("GET", "/v2/positions"),
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
        # Positions are always a list; cast away the dict[str, Any] annotation.
        rows: list[dict[str, Any]] = cast(list, payload)
        return [
            Position(
                symbol=row["symbol"],
                qty=float(row["qty"]),
                market_value=float(row["market_value"]),
            )
            for row in rows
        ]

    def get_latest_prices(
        self,
        symbols: list[str],
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, float]:
        payload = self._cached_json(
            "latest_prices",
            {"kind": "latest_prices", "symbols": symbols},
            lambda: self._latest_prices_payload(symbols),
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
        trades = payload.get("trades", {})
        return {symbol: float(trade["p"]) for symbol, trade in trades.items()}

    def get_daily_closes(
        self,
        symbols: list[str],
        lookback_days: int,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, list[float]]:
        cached = self._cached_json(
            "daily_closes",
            {
                "kind": "daily_closes",
                "symbols": symbols,
                "lookback_days": lookback_days,
            },
            lambda: self._daily_closes_payload(symbols, lookback_days),
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
        return {
            symbol: [float(value) for value in values]
            for symbol, values in cached.items()
        }

    def get_daily_bars(
        self,
        symbols: list[str],
        lookback_days: int,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, list[dict[str, str | float]]]:
        cached = self._cached_json(
            "daily_bars",
            {"kind": "daily_bars", "symbols": symbols, "lookback_days": lookback_days},
            lambda: self._daily_bars_payload(symbols, lookback_days),
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
        return {
            symbol: [
                {"timestamp": str(row["timestamp"]), "close": float(row["close"])}
                for row in values
            ]
            for symbol, values in cached.items()
        }

    def _daily_bars_payload(
        self, symbols: list[str], lookback_days: int
    ) -> dict[str, list[dict[str, str | float]]]:
        end = datetime.now(UTC)
        start = end - timedelta(days=max(lookback_days * 3, 30))
        bars_by_symbol: dict[str, list[dict[str, str | float]]] = {}
        for symbol in symbols:
            query = urlencode(
                {
                    "timeframe": "1Day",
                    "start": start.isoformat().replace("+00:00", "Z"),
                    "end": end.isoformat().replace("+00:00", "Z"),
                    "limit": str(lookback_days + 5),
                    "adjustment": "all",
                    "feed": "iex",
                }
            )
            payload = self._request_json(
                "GET", f"/v2/stocks/{symbol}/bars?{query}", data_api=True
            )
            series = payload.get("bars", [])
            bars = [
                {
                    "timestamp": str(row["t"]),
                    "close": float(row["c"]),
                }
                for row in series
            ][-lookback_days:]
            if len(bars) < 2:
                raise RuntimeError(
                    f"Not enough daily bars returned for {symbol}. "
                    f"Requested {lookback_days} trading days, got {len(bars)}."
                )
            bars_by_symbol[symbol] = bars
        return bars_by_symbol

    def _daily_closes_payload(
        self, symbols: list[str], lookback_days: int
    ) -> dict[str, list[float]]:
        end = datetime.now(UTC)
        # Ask for more calendar days than the trading lookback to survive weekends
        # and holidays while still ending up with enough bars.
        start = end - timedelta(days=max(lookback_days * 3, 30))
        closes_by_symbol: dict[str, list[float]] = {}
        for symbol in symbols:
            # Use the single-symbol endpoint here. Alpaca's multi-symbol bars
            # endpoint sorts results by symbol first, so a modest limit can
            # return only the first symbol and leave the rest empty.
            query = urlencode(
                {
                    "timeframe": "1Day",
                    "start": start.isoformat().replace("+00:00", "Z"),
                    "end": end.isoformat().replace("+00:00", "Z"),
                    "limit": str(lookback_days + 5),
                    "adjustment": "all",
                    "feed": "iex",
                }
            )
            payload = self._request_json(
                "GET", f"/v2/stocks/{symbol}/bars?{query}", data_api=True
            )
            series = payload.get("bars", [])
            closes = [float(row["c"]) for row in series][-lookback_days:]
            if len(closes) < 2:
                raise RuntimeError(
                    f"Not enough daily bars returned for {symbol}. "
                    f"Requested {lookback_days} trading days, got {len(closes)}."
                )
            closes_by_symbol[symbol] = closes
        return closes_by_symbol

    def _latest_prices_payload(self, symbols: list[str]) -> dict[str, Any]:
        query = urlencode({"symbols": ",".join(symbols), "feed": "iex"})
        return self._request_json(
            "GET", f"/v2/stocks/trades/latest?{query}", data_api=True
        )

    def get_daily_closes_for_period(
        self,
        symbols: list[str],
        total_days: int,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, list[float]]:
        return self.get_daily_closes(
            symbols,
            total_days,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )

    def get_daily_bars_for_period(
        self,
        symbols: list[str],
        total_days: int,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, list[dict[str, str | float]]]:
        return self.get_daily_bars(
            symbols,
            total_days,
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Fetch currently active orders to avoid double-submitting."""
        try:
            payload = self._request_json("GET", "/v2/orders?status=open")
            return payload if isinstance(payload, list) else []
        except Exception:
            return []

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
            try:
                self._request_json("POST", "/v2/orders", payload=order)
            except Exception as exc:
                # Log failure but continue with other orders to avoid
                # leaving the portfolio in a partially filled state.
                print(
                    f"Failed to submit order for {plan.symbol}: {exc}", file=sys.stderr
                )

    def _cached_json(
        self,
        name: str,
        key_payload: dict,
        fetcher,
        *,
        use_cache: bool,
        refresh_cache: bool,
        offline: bool,
    ) -> dict[str, Any]:
        path = cache_path(name, key_payload)
        if offline:
            if not path.exists():
                if name == "daily_closes":
                    fallback = self._find_offline_closes_fallback(
                        key_payload["symbols"],
                        int(key_payload["lookback_days"]),
                    )
                    if fallback is not None:
                        return fallback
                if name == "daily_bars":
                    fallback = self._find_offline_bars_fallback(
                        key_payload["symbols"],
                        int(key_payload["lookback_days"]),
                    )
                    if fallback is not None:
                        return fallback
                raise RuntimeError(
                    f"Offline mode requested but cache is missing: {path}"
                )
            return read_cache(path)
        if use_cache and path.exists() and not refresh_cache:
            return read_cache(path)
        payload = fetcher()
        if use_cache or refresh_cache:
            write_cache(path, payload)
        return payload

    def _find_offline_closes_fallback(
        self,
        symbols: list[str],
        lookback_days: int,
    ) -> dict[str, list[float]] | None:
        for path in sorted(Path(".cache").glob("daily_closes_*.json")):
            payload = read_cache(path)
            if not isinstance(payload, dict) or list(payload.keys()) != symbols:
                continue
            lengths = [len(values) for values in payload.values()]
            if min(lengths, default=0) >= lookback_days:
                return {
                    symbol: [float(value) for value in values[-lookback_days:]]
                    for symbol, values in payload.items()
                }
        for path in sorted(Path(".cache").glob("daily_bars_*.json")):
            payload = read_cache(path)
            if not isinstance(payload, dict) or list(payload.keys()) != symbols:
                continue
            lengths = [len(values) for values in payload.values()]
            if min(lengths, default=0) >= lookback_days:
                return {
                    symbol: [float(row["close"]) for row in values[-lookback_days:]]
                    for symbol, values in payload.items()
                }
        return None

    def _find_offline_bars_fallback(
        self,
        symbols: list[str],
        lookback_days: int,
    ) -> dict[str, list[dict[str, str | float]]] | None:
        for path in sorted(Path(".cache").glob("daily_bars_*.json")):
            payload = read_cache(path)
            if not isinstance(payload, dict) or list(payload.keys()) != symbols:
                continue
            lengths = [len(values) for values in payload.values()]
            if min(lengths, default=0) >= lookback_days:
                return {
                    symbol: [
                        {
                            "timestamp": str(row["timestamp"]),
                            "close": float(row["close"]),
                        }
                        for row in values[-lookback_days:]
                    ]
                    for symbol, values in payload.items()
                }
        return None

    def _request_json(
        self,
        method: str,
        path: str,
        payload: dict | None = None,
        data_api: bool = False,
    ) -> dict[str, Any]:
        base_url = self._config.data_url if data_api else self._config.base_url
        headers = {
            "APCA-API-KEY-ID": self._config.api_key,
            "APCA-API-SECRET-KEY": self._config.api_secret,
            "Content-Type": "application/json",
        }
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = Request(
            f"{base_url}{path}", data=body, headers=headers, method=method
        )
        try:
            with urlopen(request) as response:
                return json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:  # pragma: no cover
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Alpaca API error {exc.code}: {message}") from exc
        except URLError as exc:  # pragma: no cover
            raise RuntimeError(f"Alpaca connection error: {exc.reason}") from exc

    def fetch_yahoo_closes(
        self,
        symbols: list[str],
        period_days: int,
    ) -> dict[str, list[float]]:
        """Fetch historical daily closes from Yahoo Finance."""
        import yfinance as yf

        closes: dict[str, list[float]] = {}
        # Map days to Yahoo Range
        if period_days > 600:
            range_str = "max"
        elif period_days > 252:
            range_str = "2y"
        else:
            range_str = "1y"

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=range_str)
                if not hist.empty:
                    closes[symbol] = hist["Close"].tolist()
            except Exception:
                # Skip symbols that fail (e.g., delisted, invalid)
                continue
        return closes


def format_order_plans(plans: list[OrderPlan]) -> str:
    return json.dumps([asdict(plan) for plan in plans], indent=2)
