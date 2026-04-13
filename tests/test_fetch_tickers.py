from __future__ import annotations

from datetime import datetime
from typing import Any

from utils import fetch_tickers


def test_ticker_info_cache_reused_for_first_trade_date(monkeypatch) -> None:
    class DummyPath:
        def __init__(self) -> None:
            self.exists_value = False

        def exists(self) -> bool:
            return self.exists_value

    path = DummyPath()
    cache: dict[str, Any] = {}
    calls: list[str] = []
    timestamp_ms = int(datetime(2020, 1, 2, 12).timestamp() * 1000)

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def info(self) -> dict[str, Any]:
            calls.append(self.symbol)
            return {
                "shortName": "PNW Utility",
                "sector": "Utilities",
                "firstTradeDateMilliseconds": timestamp_ms,
            }

    def fake_write_cache(_path: DummyPath, payload: dict[str, Any]) -> None:
        cache["payload"] = payload
        path.exists_value = True

    monkeypatch.setattr(fetch_tickers, "_TICKER_INFO_MEMORY_CACHE", {})
    monkeypatch.setattr(fetch_tickers, "cache_path", lambda _name, _payload: path)
    monkeypatch.setattr(fetch_tickers, "read_cache", lambda _path: cache["payload"])
    monkeypatch.setattr(fetch_tickers, "write_cache", fake_write_cache)
    monkeypatch.setattr(fetch_tickers.yf, "Ticker", FakeTicker)

    assert fetch_tickers._get_ticker_info("PNW") == (
        "PNW",
        "PNW Utility (Utilities)",
    )
    assert fetch_tickers.get_ticker_firstTradeDate("PNW") == datetime(2020, 1, 2)
    assert calls == ["PNW"]
