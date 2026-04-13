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


def test_fetch_yfiua_index_constituents_uses_monthly_json(monkeypatch) -> None:
    requested: list[tuple[str, dict[str, str]]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {
                "constituents": [
                    {"symbol": "AAPL"},
                    {"ticker": "MSFT"},
                    {"symbol": "BA/.L"},
                    {"ticker": "BT/A.L"},
                    {"symbol": "ABC/.TO"},
                    {"ticker": "XYZ/A.TO"},
                    {"symbol": "000333/SZ"},
                    {"symbol": "600519/SS"},
                    {"symbol": "0700/HK"},
                    {"symbol": "SAP/DE"},
                ]
            }

    def fake_get(url: str, *, headers: dict[str, str]):
        requested.append((url, headers))
        return FakeResponse()

    monkeypatch.setattr(fetch_tickers.requests, "get", fake_get)

    assert fetch_tickers.fetch_yfiua_index_constituents(
        "nasdaq100",
        year=2026,
        month=4,
    ) == [
        "AAPL",
        "MSFT",
        "BA.L",
        "BT-A.L",
        "ABC.TO",
        "XYZ-A.TO",
        "000333.SZ",
        "600519.SS",
        "0700.HK",
        "SAP.DE",
    ]
    assert requested == [
        (
            "https://yfiua.github.io/index-constituents/2026/04/constituents-nasdaq100.json",
            {"User-Agent": "Mozilla/5.0"},
        )
    ]


def test_fetch_yfiua_index_constituents_uses_current_json_by_default(
    monkeypatch,
) -> None:
    requested: list[str] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, list[str]]:
            return {"symbols": ["AAPL"]}

    def fake_get(url: str, *, headers: dict[str, str]):
        requested.append(url)
        return FakeResponse()

    monkeypatch.setattr(fetch_tickers.requests, "get", fake_get)

    assert fetch_tickers.fetch_yfiua_index_constituents("nasdaq100") == ["AAPL"]
    assert requested == [
        "https://yfiua.github.io/index-constituents/constituents-nasdaq100.json"
    ]


def test_fetch_yfiua_index_constituents_requires_complete_date() -> None:
    try:
        fetch_tickers.fetch_yfiua_index_constituents("nasdaq100", year=2026)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("partial yfiua date should fail")

    assert "year and month" in message


def test_fetch_ftse_tickers_uses_yfiua_constituents(monkeypatch) -> None:
    called: list[str] = []

    def fake_fetch(code: str) -> list[str]:
        called.append(code)
        return ["AZN.L", "HSBA.L"]

    monkeypatch.setattr(fetch_tickers, "fetch_yfiua_index_constituents", fake_fetch)

    assert fetch_tickers.fetch_ftse_tickers() == ["AZN.L", "HSBA.L"]
    assert called == ["ftse100"]


def test_fetch_ticker_dict_supports_opt_in_ftse100(monkeypatch) -> None:
    called: list[str] = []

    def fake_fetch(code: str) -> list[str]:
        called.append(code)
        return ["AZN.L"]

    monkeypatch.setattr(fetch_tickers, "fetch_yfiua_index_constituents", fake_fetch)
    monkeypatch.setattr(
        fetch_tickers,
        "_format_ticker_dict",
        lambda tickers: {"symbols": tickers, "asset_classes": {}},
    )

    assert fetch_tickers.fetch_ticker_dict(ticker_basket=["ftse100"]) == {
        "symbols": ["AZN.L"],
        "asset_classes": {},
    }
    assert called == ["ftse100"]


def test_fetch_ticker_dict_supports_yfiua_prefixed_builtin(monkeypatch) -> None:
    called: list[str] = []

    def fake_fetch(code: str) -> list[str]:
        called.append(code)
        return ["AAPL"]

    monkeypatch.setattr(fetch_tickers, "fetch_nasdaq100_tickers", lambda: ["NVDA"])
    monkeypatch.setattr(fetch_tickers, "fetch_yfiua_index_constituents", fake_fetch)
    monkeypatch.setattr(
        fetch_tickers,
        "_format_ticker_dict",
        lambda tickers: {"symbols": tickers, "asset_classes": {}},
    )

    assert fetch_tickers.fetch_ticker_dict(ticker_basket=["yfiua:nasdaq100"]) == {
        "symbols": ["AAPL"],
        "asset_classes": {},
    }
    assert called == ["nasdaq100"]


def test_fetch_ticker_dict_preserves_builtin_nasdaq_provider(monkeypatch) -> None:
    monkeypatch.setattr(fetch_tickers, "fetch_nasdaq100_tickers", lambda: ["NVDA"])
    monkeypatch.setattr(
        fetch_tickers,
        "fetch_yfiua_index_constituents",
        lambda code: (_ for _ in ()).throw(
            AssertionError("bare nasdaq100 should use the existing Nasdaq provider")
        ),
    )
    monkeypatch.setattr(
        fetch_tickers,
        "_format_ticker_dict",
        lambda tickers: {"symbols": tickers, "asset_classes": {}},
    )

    assert fetch_tickers.fetch_ticker_dict(ticker_basket=["nasdaq100"]) == {
        "symbols": ["NVDA"],
        "asset_classes": {},
    }
