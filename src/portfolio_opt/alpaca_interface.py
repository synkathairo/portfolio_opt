from __future__ import annotations

import json
import sys
import warnings
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

warnings.filterwarnings(
    "ignore",
    message="websockets\\.legacy is deprecated.*",
    category=DeprecationWarning,
    module="websockets\\.legacy",
)

from alpaca.data.enums import Adjustment, DataFeed
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.models.bars import Bar, BarSet
from alpaca.data.models.trades import TradeSet
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestTradeRequest,
    StockLatestQuoteRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.models import Order, Position as AlpacaPosition, TradeAccount
from alpaca.trading.models import PortfolioHistory as AlpacaPortfolioHistory
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest

from .cache import cache_path, read_cache, write_cache
from .config import AlpacaConfig
from .types import AccountSnapshot, OrderPlan, Position


class AlpacaClient:
    def __init__(self, config: AlpacaConfig) -> None:
        self._config = config
        self._trading = TradingClient(
            api_key=config.api_key, secret_key=config.api_secret
        )
        self._data = StockHistoricalDataClient(
            api_key=config.api_key, secret_key=config.api_secret
        )

    def get_account(
        self,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> AccountSnapshot:
        payload = self._cached_json(
            "account",
            {"kind": "account"},
            lambda: self._account_payload(),
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
            lambda: self._positions_payload(),
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
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
        trades_payload = payload.get("trades", payload)
        return {
            symbol: float(trade["p"])
            for symbol, trade in trades_payload.items()
            if isinstance(trade, dict) and "p" in trade
        }

    def get_daily_closes(
        self,
        symbols: list[str],
        lookback_days: int,
        use_cache: bool = False,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, list[float]]:
        if use_cache or refresh_cache or offline:
            incremental = self._incremental_daily_closes(
                symbols=symbols,
                lookback_days=lookback_days,
                refresh_cache=refresh_cache,
                offline=offline,
            )
            if incremental is not None:
                return incremental

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

    def _account_payload(self) -> dict[str, Any]:
        account = cast(TradeAccount, self._trading.get_account())
        return {"equity": account.equity}

    def _positions_payload(self) -> list[dict[str, Any]]:
        positions = cast(list[AlpacaPosition], self._trading.get_all_positions())
        return [
            {
                "symbol": p.symbol,
                "qty": p.qty,
                "market_value": p.market_value,
            }
            for p in positions
        ]

    def _latest_prices_payload(self, symbols: list[str]) -> dict[str, Any]:
        req = StockLatestTradeRequest(symbol_or_symbols=symbols)
        result = self._data.get_stock_latest_trade(req)
        trades = {}
        # Handle both TradeSet object and dict return types
        data = result.data if hasattr(result, "data") else result
        if isinstance(data, dict):
            for symbol, trade_data in data.items():
                trade = (
                    trade_data[0]
                    if isinstance(trade_data, list) and trade_data
                    else trade_data
                )
                if isinstance(trade, dict):
                    trade_payload = cast(dict[str, Any], trade)
                    price = trade_payload.get("p")
                    if price is None:
                        price = trade_payload.get("price")
                else:
                    price = getattr(trade, "price", None)
                if price is not None:
                    trades[symbol] = {"p": float(price)}
        return trades

    def _daily_bars_payload(
        self, symbols: list[str], lookback_days: int
    ) -> dict[str, list[dict[str, str | float]]]:
        end = datetime.now(UTC)
        start = end - timedelta(days=max(lookback_days * 3, 30))

        bars_by_symbol: dict[str, list[dict[str, str | float]]] = {}

        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=lookback_days + 5,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX,
            )
            bars = cast(BarSet, self._data.get_stock_bars(request))
            for symbol, bar_list in bars.data.items():
                bars_data = [
                    {
                        "timestamp": str(b.timestamp),
                        "close": float(b.close),
                    }
                    for b in bar_list
                ][-lookback_days:]
                if len(bars_data) < 2:
                    raise RuntimeError(
                        f"Not enough daily bars returned for {symbol}. "
                        f"Requested {lookback_days} trading days, got {len(bars_data)}."
                    )
                bars_by_symbol[symbol] = bars_data

        # Retry any symbols still missing
        missing = [s for s in symbols if s not in bars_by_symbol]
        for symbol in missing:
            single_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=lookback_days + 5,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX,
            )
            single_bars = cast(BarSet, self._data.get_stock_bars(single_req))
            if symbol in single_bars.data:
                bar_list = single_bars.data[symbol]
                bars_data = [
                    {
                        "timestamp": str(b.timestamp),
                        "close": float(b.close),
                    }
                    for b in bar_list
                ][-lookback_days:]
                if len(bars_data) >= 2:
                    bars_by_symbol[symbol] = bars_data

        return bars_by_symbol

    def _daily_closes_payload(
        self, symbols: list[str], lookback_days: int
    ) -> dict[str, list[float]]:
        end = datetime.now(UTC)
        start = end - timedelta(days=max(lookback_days * 3, 30))

        closes_by_symbol: dict[str, list[float]] = {}

        # Fetch in batches to avoid Alpaca API limits on large universes.
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=lookback_days + 5,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX,
            )
            bars = cast(BarSet, self._data.get_stock_bars(request))
            for symbol, bar_list in bars.data.items():
                closes = [float(b.close) for b in bar_list][-lookback_days:]
                if len(closes) < 2:
                    raise RuntimeError(
                        f"Not enough daily bars returned for {symbol}. "
                        f"Requested {lookback_days} trading days, got {len(closes)}."
                    )
                closes_by_symbol[symbol] = closes

        # Retry any symbols still missing, one at a time.
        missing = [s for s in symbols if s not in closes_by_symbol]
        for symbol in missing:
            single_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=lookback_days + 5,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX,
            )
            single_bars = cast(BarSet, self._data.get_stock_bars(single_req))
            if symbol in single_bars.data:
                bar_list = single_bars.data[symbol]
                closes = [float(b.close) for b in bar_list][-lookback_days:]
                if len(closes) >= 2:
                    closes_by_symbol[symbol] = closes

        return closes_by_symbol

    def _incremental_daily_closes(
        self,
        *,
        symbols: list[str],
        lookback_days: int,
        refresh_cache: bool,
        offline: bool,
    ) -> dict[str, list[float]] | None:
        cached_rows: dict[str, list[dict[str, str | float]]] = {}
        missing_symbols: list[str] = []

        for symbol in symbols:
            path = self._daily_closes_v2_cache_path(symbol)
            if path.exists():
                payload = read_cache(path)
                rows = self._normalize_daily_bar_rows(payload)
                if rows:
                    cached_rows[symbol] = rows
                    continue
            missing_symbols.append(symbol)

        if offline:
            if missing_symbols:
                return None
            return self._daily_close_rows_to_close_map(
                cached_rows, symbols, lookback_days
            )

        if not refresh_cache:
            if missing_symbols:
                return None
            closes = self._daily_close_rows_to_close_map(
                cached_rows, symbols, lookback_days
            )
            return closes

        fetch_groups: dict[str, tuple[datetime, list[str], int]] = {}
        now = datetime.now(UTC)
        full_fetch_start = now - timedelta(days=max(lookback_days * 3, 30))
        full_fetch_key = full_fetch_start.date().isoformat()
        fetch_groups[full_fetch_key] = (full_fetch_start, [], lookback_days + 5)

        for symbol in symbols:
            rows = cached_rows.get(symbol, [])
            if not rows:
                fetch_groups[full_fetch_key][1].append(symbol)
                continue
            if len(rows) < lookback_days:
                fetch_groups[full_fetch_key][1].append(symbol)
                continue

            last_date = str(rows[-1]["timestamp"])
            try:
                next_start = (
                    datetime.strptime(last_date, "%Y-%m-%d").replace(tzinfo=UTC)
                    + timedelta(days=1)
                )
            except ValueError:
                fetch_groups[full_fetch_key][1].append(symbol)
                continue
            if next_start.date() > now.date():
                continue

            fetch_key = next_start.date().isoformat()
            if fetch_key not in fetch_groups:
                fetch_groups[fetch_key] = (next_start, [], lookback_days + 5)
            fetch_groups[fetch_key][1].append(symbol)

        for start, group_symbols, limit in fetch_groups.values():
            if not group_symbols:
                continue
            fetched = self._daily_bar_rows_payload(
                group_symbols,
                start=start,
                end=now,
                limit=limit,
            )
            for symbol in group_symbols:
                merged = self._merge_daily_bar_rows(
                    cached_rows.get(symbol, []),
                    fetched.get(symbol, []),
                    lookback_days=lookback_days,
                )
                if merged:
                    cached_rows[symbol] = merged
                    write_cache(self._daily_closes_v2_cache_path(symbol), merged)

        return self._daily_close_rows_to_close_map(cached_rows, symbols, lookback_days)

    def _daily_closes_v2_cache_path(self, symbol: str) -> Path:
        return cache_path(
            "daily_closes_v2",
            {
                "kind": "daily_closes_v2",
                "symbol": symbol,
                "adjustment": "all",
                "feed": "iex",
            },
        )

    def _daily_bar_rows_payload(
        self,
        symbols: list[str],
        *,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> dict[str, list[dict[str, str | float]]]:
        rows_by_symbol: dict[str, list[dict[str, str | float]]] = {}
        batch_size = 100
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            request = StockBarsRequest(
                symbol_or_symbols=batch,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=limit,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX,
            )
            bars = cast(BarSet, self._data.get_stock_bars(request))
            for symbol, bar_list in bars.data.items():
                rows_by_symbol[symbol] = [
                    {
                        "timestamp": self._bar_timestamp_date(b.timestamp),
                        "close": float(b.close),
                    }
                    for b in bar_list
                ]

        missing = [symbol for symbol in symbols if symbol not in rows_by_symbol]
        for symbol in missing:
            single_req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
                limit=limit,
                adjustment=Adjustment.ALL,
                feed=DataFeed.IEX,
            )
            single_bars = cast(BarSet, self._data.get_stock_bars(single_req))
            if symbol in single_bars.data:
                rows_by_symbol[symbol] = [
                    {
                        "timestamp": self._bar_timestamp_date(b.timestamp),
                        "close": float(b.close),
                    }
                    for b in single_bars.data[symbol]
                ]
        return rows_by_symbol

    def _bar_timestamp_date(self, timestamp: Any) -> str:
        if isinstance(timestamp, datetime):
            return timestamp.date().isoformat()
        return str(timestamp)[:10]

    def _normalize_daily_bar_rows(
        self, payload: Any
    ) -> list[dict[str, str | float]]:
        if not isinstance(payload, list):
            return []
        rows: list[dict[str, str | float]] = []
        for row in payload:
            if not isinstance(row, dict):
                continue
            if "timestamp" not in row or "close" not in row:
                continue
            rows.append(
                {
                    "timestamp": self._bar_timestamp_date(row["timestamp"]),
                    "close": float(row["close"]),
                }
            )
        return self._merge_daily_bar_rows([], rows, lookback_days=len(rows))

    def _merge_daily_bar_rows(
        self,
        existing: list[dict[str, str | float]],
        new: list[dict[str, str | float]],
        *,
        lookback_days: int,
    ) -> list[dict[str, str | float]]:
        by_date: dict[str, dict[str, str | float]] = {}
        for row in existing + new:
            if "timestamp" not in row or "close" not in row:
                continue
            by_date[str(row["timestamp"])] = {
                "timestamp": str(row["timestamp"]),
                "close": float(row["close"]),
            }
        return [by_date[key] for key in sorted(by_date)][-lookback_days:]

    def _daily_close_rows_to_close_map(
        self,
        rows_by_symbol: dict[str, list[dict[str, str | float]]],
        symbols: list[str],
        lookback_days: int,
    ) -> dict[str, list[float]] | None:
        if any(symbol not in rows_by_symbol for symbol in symbols):
            return None
        if any(len(rows_by_symbol[symbol]) < lookback_days for symbol in symbols):
            return None
        return {
            symbol: [
                float(row["close"])
                for row in rows_by_symbol[symbol][-lookback_days:]
            ]
            for symbol in symbols
        }

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

    def get_portfolio_history(
        self, period: str = "1M", timeframe: str = "1D"
    ) -> dict[str, Any]:
        """Get portfolio history for plotting."""
        from alpaca.trading.requests import GetPortfolioHistoryRequest

        history = cast(
            AlpacaPortfolioHistory,
            self._trading.get_portfolio_history(
                GetPortfolioHistoryRequest(period=period, timeframe=timeframe)
            ),
        )
        return {
            "timestamp": history.timestamp,
            "equity": history.equity,
            "profit_loss": history.profit_loss,
            "profit_loss_pct": history.profit_loss_pct,
        }

    def get_stock_bars_raw(
        self, symbol: str, start: datetime, end: datetime, limit: int = 1000
    ) -> list[dict[str, Any]]:
        """Get raw stock bars for plotting benchmarks."""
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
            limit=limit,
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
        )
        bars = cast(BarSet, self._data.get_stock_bars(request))
        result: list[dict[str, Any]] = []
        for symbol, bar_list in bars.data.items():
            for b in bar_list:
                result.append(
                    {
                        "timestamp": b.timestamp,
                        "open": b.open,
                        "high": b.high,
                        "low": b.low,
                        "close": b.close,
                        "volume": b.volume,
                    }
                )
        return result

    def get_open_orders(self) -> list[dict[str, Any]]:
        """Fetch currently active orders to avoid double-submitting."""
        try:
            req = GetOrdersRequest(status=QueryOrderStatus.OPEN)
            orders = cast(list[Order], self._trading.get_orders(req))
            return [
                {
                    "symbol": o.symbol,
                    "qty": o.qty,
                    "side": o.side,
                    "type": o.type,
                }
                for o in orders
            ]
        except Exception:
            return []

    def submit_order_plan(self, plans: list[OrderPlan]) -> None:
        for plan in plans:
            try:
                order_data = MarketOrderRequest(
                    symbol=plan.symbol,
                    notional=round(plan.notional_usd, 2),
                    side=plan.side,
                    time_in_force="day",
                )
                self._trading.submit_order(order_data)
            except Exception as exc:
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
            extracted = self._extract_close_subset(payload, symbols, lookback_days)
            if extracted is not None:
                return extracted
        for path in sorted(Path(".cache").glob("daily_bars_*.json")):
            payload = read_cache(path)
            extracted = self._extract_closes_from_bars_subset(
                payload, symbols, lookback_days
            )
            if extracted is not None:
                return extracted
        return None

    def _find_offline_bars_fallback(
        self,
        symbols: list[str],
        lookback_days: int,
    ) -> dict[str, list[dict[str, str | float]]] | None:
        for path in sorted(Path(".cache").glob("daily_bars_*.json")):
            payload = read_cache(path)
            extracted = self._extract_bar_subset(payload, symbols, lookback_days)
            if extracted is not None:
                return extracted
        return None

    def _extract_close_subset(
        self,
        payload: Any,
        symbols: list[str],
        lookback_days: int,
    ) -> dict[str, list[float]] | None:
        if not isinstance(payload, dict):
            return None
        if any(symbol not in payload for symbol in symbols):
            return None
        selected = {symbol: payload[symbol] for symbol in symbols}
        lengths = [
            len(values)
            for values in selected.values()
            if isinstance(values, list)
        ]
        if len(lengths) != len(symbols) or min(lengths, default=0) < lookback_days:
            return None
        return {
            symbol: [float(value) for value in selected[symbol][-lookback_days:]]
            for symbol in symbols
        }

    def _extract_closes_from_bars_subset(
        self,
        payload: Any,
        symbols: list[str],
        lookback_days: int,
    ) -> dict[str, list[float]] | None:
        bars_subset = self._extract_bar_subset(payload, symbols, lookback_days)
        if bars_subset is None:
            return None
        return {
            symbol: [float(row["close"]) for row in rows]
            for symbol, rows in bars_subset.items()
        }

    def _extract_bar_subset(
        self,
        payload: Any,
        symbols: list[str],
        lookback_days: int,
    ) -> dict[str, list[dict[str, str | float]]] | None:
        if not isinstance(payload, dict):
            return None
        if any(symbol not in payload for symbol in symbols):
            return None
        selected = {symbol: payload[symbol] for symbol in symbols}
        lengths = [
            len(values)
            for values in selected.values()
            if isinstance(values, list)
        ]
        if len(lengths) != len(symbols) or min(lengths, default=0) < lookback_days:
            return None
        return {
            symbol: [
                {
                    "timestamp": str(row["timestamp"]),
                    "close": float(row["close"]),
                }
                for row in selected[symbol][-lookback_days:]
            ]
            for symbol in symbols
        }

    def fetch_yahoo_closes(
        self,
        symbols: list[str],
        period_days: int,
    ) -> dict[str, list[float]]:
        """Fetch historical daily closes from Yahoo Finance."""
        import yfinance as yf

        closes: dict[str, list[float]] = {}
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
                continue

        return closes


def format_order_plans(plans: list[OrderPlan]) -> str:
    return json.dumps([asdict(plan) for plan in plans], indent=2)
