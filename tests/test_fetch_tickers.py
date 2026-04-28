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
        "sector_utilities",
    )
    assert fetch_tickers.get_ticker_firstTradeDate("PNW") == datetime(2020, 1, 2)
    assert calls == ["PNW"]
    assert cache["payload"] == {
        "shortName": "PNW Utility",
        "sector": "Utilities",
        "asset_class": "sector_utilities",
        "firstTradeDateMilliseconds": timestamp_ms,
    }


def test_ticker_info_normalizes_yahoo_real_estate_sector(monkeypatch) -> None:
    class DummyPath:
        def exists(self) -> bool:
            return False

    class FakeTicker:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        @property
        def info(self) -> dict[str, str]:
            return {"shortName": "Alexandria", "sector": "Real Estate"}

    monkeypatch.setattr(fetch_tickers, "_TICKER_INFO_MEMORY_CACHE", {})
    monkeypatch.setattr(
        fetch_tickers, "cache_path", lambda _name, _payload: DummyPath()
    )
    monkeypatch.setattr(fetch_tickers, "write_cache", lambda _path, _payload: None)
    monkeypatch.setattr(fetch_tickers.yf, "Ticker", FakeTicker)

    assert fetch_tickers._get_ticker_info("ARE") == ("ARE", "sector_real_estate")


def test_ticker_info_uses_cached_asset_class_without_network(monkeypatch) -> None:
    class DummyPath:
        def exists(self) -> bool:
            return True

    timestamp_ms = int(datetime(2020, 1, 2, 12).timestamp() * 1000)
    monkeypatch.setattr(fetch_tickers, "_TICKER_INFO_MEMORY_CACHE", {})
    monkeypatch.setattr(
        fetch_tickers, "cache_path", lambda _name, _payload: DummyPath()
    )
    monkeypatch.setattr(
        fetch_tickers,
        "read_cache",
        lambda _path: {
            "shortName": "Cached Name",
            "sector": "Technology",
            "asset_class": "sector_technology",
            "firstTradeDateMilliseconds": timestamp_ms,
        },
    )
    monkeypatch.setattr(
        fetch_tickers.yf,
        "Ticker",
        lambda _symbol: (_ for _ in ()).throw(
            AssertionError("network should not be used on cache hit")
        ),
    )

    assert fetch_tickers._get_ticker_info("AAPL") == ("AAPL", "sector_technology")
    assert fetch_tickers.get_ticker_firstTradeDate("AAPL") == datetime(2020, 1, 2)


def test_ticker_info_uses_dash_alias_when_dot_symbol_cache_is_unknown(
    monkeypatch,
) -> None:
    class DummyPath:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def exists(self) -> bool:
            return self.symbol in cache

    cache = {
        "BRK.B": {
            "symbol": "BRK.B",
            "asset_class": "sector_unknown",
        },
        "BRK-B": {
            "symbol": "BRK-B",
            "sector": "Financial Services",
            "asset_class": "sector_financials",
        },
    }

    monkeypatch.setattr(fetch_tickers, "_TICKER_INFO_MEMORY_CACHE", {})
    monkeypatch.setattr(
        fetch_tickers,
        "cache_path",
        lambda _name, payload: DummyPath(str(payload["symbol"])),
    )
    monkeypatch.setattr(fetch_tickers, "read_cache", lambda path: cache[path.symbol])
    monkeypatch.setattr(
        fetch_tickers.yf,
        "Ticker",
        lambda _symbol: (_ for _ in ()).throw(
            AssertionError("network should not be used on alias cache hit")
        ),
    )

    assert fetch_tickers._get_ticker_info("BRK.B") == ("BRK.B", "sector_financials")


def test_fetch_yfiua_index_constituents_uses_monthly_json(monkeypatch) -> None:
    requested: list[tuple[str, dict[str, str]]] = []

    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict[str, list[dict[str, str]]]:
            return {
                "constituents": [
                    {"symbol": "AAPL"},
                    {"ticker": "MSFT", "Name": "Microsoft"},
                    {"symbol": "BA/.L"},
                    {"ticker": "BT/A.L"},
                    {"symbol": "ABC/.TO"},
                    {"ticker": "XYZ/A.TO"},
                    {"symbol": "000333/SZ", "name": "Midea"},
                    {"symbol": "600519/SS", "Name": "Kweichow Moutai"},
                    {"symbol": "0700/HK", "Name": "Tencent"},
                    {"symbol": "SAP/DE", "Name": "SAP"},
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
    assert fetch_tickers.fetch_yfiua_index_constituent_names(
        "nasdaq100",
        year=2026,
        month=4,
    ) == {
        "MSFT": "Microsoft",
        "000333.SZ": "Midea",
        "600519.SS": "Kweichow Moutai",
        "0700.HK": "Tencent",
        "SAP.DE": "SAP",
    }
    assert fetch_tickers.fetch_yfiua_index_constituents_with_names(
        "nasdaq100",
        year=2026,
        month=4,
    ) == (
        [
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
        ],
        {
            "MSFT": "Microsoft",
            "000333.SZ": "Midea",
            "600519.SS": "Kweichow Moutai",
            "0700.HK": "Tencent",
            "SAP.DE": "SAP",
        },
    )


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


def test_fetch_nasdaq100_tickers_retries_malformed_json(monkeypatch) -> None:
    responses = [
        {
            "data": {
                "data": None,
            }
        },
        {
            "data": {
                "data": {
                    "rows": [
                        {"symbol": "AAPL"},
                        {"symbol": "MSFT"},
                        {"symbol": ""},
                        {"companyName": "Missing Symbol"},
                    ]
                }
            }
        },
    ]
    requested: list[tuple[str, dict[str, str], int]] = []
    sleeps: list[float] = []

    class FakeResponse:
        status_code = 200

        def __init__(self, payload: dict[str, Any]) -> None:
            self._payload = payload

        def json(self) -> dict[str, Any]:
            return self._payload

    def fake_get(url: str, *, headers: dict[str, str], timeout: int):
        requested.append((url, headers, timeout))
        return FakeResponse(responses.pop(0))

    monkeypatch.setattr(fetch_tickers.requests, "get", fake_get)
    monkeypatch.setattr(fetch_tickers.time, "sleep", lambda delay: sleeps.append(delay))

    assert fetch_tickers.fetch_nasdaq100_tickers(retries=2, backoff=0.1) == [
        "AAPL",
        "MSFT",
    ]
    assert len(requested) == 2
    assert requested[0][2] == 20
    assert sleeps == [0.1]


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


def test_fetch_historical_sp500_tickers_selects_latest_prior_row(monkeypatch) -> None:
    csv = "\n".join(
        [
            "date,tickers",
            '2015-12-31,"AAA,BBB"',
            '2016-01-05,"AAA,CCC"',
        ]
    )
    requested: list[str] = []

    class FakeResponse:
        text = csv

        def raise_for_status(self) -> None:
            pass

    def fake_get(url: str, *, headers: dict[str, str]):
        requested.append(url)
        return FakeResponse()

    monkeypatch.setattr(fetch_tickers.requests, "get", fake_get)
    monkeypatch.setattr(
        fetch_tickers,
        "cache_path",
        lambda _name, _payload: type(
            "MissingPath", (), {"exists": lambda self: False}
        )(),
    )
    monkeypatch.setattr(fetch_tickers, "write_cache", lambda _path, _payload: None)

    assert fetch_tickers.fetch_historical_sp500_tickers("2016-01-04") == [
        "AAA",
        "BBB",
    ]
    assert requested == [fetch_tickers.SP500_HISTORICAL_COMPONENTS_URL]


def test_fetch_historical_sp500_tickers_reuses_cache(monkeypatch) -> None:
    payload = {
        "rows": [
            {"date": "2015-12-31", "tickers": "AAA,BBB"},
            {"date": "2016-01-05", "tickers": "AAA,CCC"},
        ]
    }

    class ExistingPath:
        def exists(self) -> bool:
            return True

    monkeypatch.setattr(
        fetch_tickers,
        "cache_path",
        lambda _name, _payload: ExistingPath(),
    )
    monkeypatch.setattr(fetch_tickers, "read_cache", lambda _path: payload)
    monkeypatch.setattr(
        fetch_tickers.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network should not be used on cache hit")
        ),
    )

    assert fetch_tickers.fetch_historical_sp500_tickers("2016-01-06") == [
        "AAA",
        "CCC",
    ]


def test_fetch_historical_sp500_tickers_errors_before_first_row() -> None:
    payload = {"rows": [{"date": "2015-12-31", "tickers": "AAA,BBB"}]}

    try:
        fetch_tickers._historical_sp500_symbols_from_payload(
            payload,
            fetch_tickers.pd.Timestamp("2015-01-01"),
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("date before first historical S&P row should fail")

    assert "No historical S&P 500 data" in message


def test_parse_nikkei225_component_html() -> None:
    html = """
    <div class="idx-index-components table-responsive-md">
      <h3 class="idx-section-subheading">Automobiles</h3>
      <table><tbody>
        <tr><td>543A</td><td>ARCHION CORP.</td></tr>
        <tr><td>7203</td><td>TOYOTA MOTOR CORP.</td></tr>
        <tr><td>7267</td><td>HONDA MOTOR CO., LTD.</td></tr>
      </tbody></table>
    </div>
    <div class="idx-index-components table-responsive-md">
      <h3 class="idx-section-subheading">Banks</h3>
      <table><tbody>
        <tr><td>8306</td><td>MITSUBISHI UFJ FINANCIAL GROUP, INC.</td></tr>
      </tbody></table>
    </div>
    """

    assert fetch_tickers._parse_nikkei225_component_html(html) == [
        fetch_tickers.NikkeiConstituent(
            symbol="543A.T",
            name="ARCHION CORP.",
            sector="Automobiles",
        ),
        fetch_tickers.NikkeiConstituent(
            symbol="7203.T",
            name="TOYOTA MOTOR CORP.",
            sector="Automobiles",
        ),
        fetch_tickers.NikkeiConstituent(
            symbol="7267.T",
            name="HONDA MOTOR CO., LTD.",
            sector="Automobiles",
        ),
        fetch_tickers.NikkeiConstituent(
            symbol="8306.T",
            name="MITSUBISHI UFJ FINANCIAL GROUP, INC.",
            sector="Banks",
        ),
    ]


def test_fetch_nikkei225_constituents_writes_and_reuses_cache(monkeypatch) -> None:
    class DummyPath:
        def __init__(self) -> None:
            self.exists_value = False

        def exists(self) -> bool:
            return self.exists_value

    class FakeResponse:
        text = """
        <div class="idx-index-components table-responsive-md">
          <h3 class="idx-section-subheading">Automobiles</h3>
          <table><tbody><tr><td>7203</td><td>TOYOTA MOTOR CORP.</td></tr></tbody></table>
        </div>
        """

        def raise_for_status(self) -> None:
            pass

    path = DummyPath()
    cache: dict[str, Any] = {}
    requested: list[tuple[str, dict[str, str], int]] = []

    def fake_get(url: str, *, headers: dict[str, str], timeout: int):
        requested.append((url, headers, timeout))
        return FakeResponse()

    def fake_write_cache(_path: DummyPath, payload: dict[str, Any]) -> None:
        cache["payload"] = payload
        path.exists_value = True

    monkeypatch.setattr(fetch_tickers, "cache_path", lambda _name, _payload: path)
    monkeypatch.setattr(fetch_tickers, "read_cache", lambda _path: cache["payload"])
    monkeypatch.setattr(fetch_tickers, "write_cache", fake_write_cache)
    monkeypatch.setattr(fetch_tickers.requests, "get", fake_get)

    expected = [
        fetch_tickers.NikkeiConstituent(
            symbol="7203.T",
            name="TOYOTA MOTOR CORP.",
            sector="Automobiles",
        )
    ]
    assert fetch_tickers.fetch_nikkei225_constituents() == expected
    assert fetch_tickers.fetch_nikkei225_constituents() == expected
    assert len(requested) == 1
    assert requested[0][0] == fetch_tickers.NIKKEI225_COMPONENTS_URL
    assert "Chrome" in requested[0][1]["User-Agent"]
    assert requested[0][2] == 20


def test_fetch_ticker_dict_supports_opt_in_ftse100(monkeypatch) -> None:
    called: list[str] = []

    def fake_fetch(code: str) -> list[str]:
        called.append(code)
        return ["AZN.L"]

    monkeypatch.setattr(fetch_tickers, "fetch_yfiua_index_constituents", fake_fetch)
    monkeypatch.setattr(
        fetch_tickers,
        "_format_ticker_dict",
        lambda tickers, **_kwargs: {"symbols": tickers, "asset_classes": {}},
    )

    assert fetch_tickers.fetch_ticker_dict(ticker_basket=["ftse100"]) == {
        "symbols": ["AZN.L"],
        "asset_classes": {},
    }
    assert called == ["ftse100"]


def test_fetch_ticker_dict_supports_nikkei225(monkeypatch) -> None:
    monkeypatch.setattr(fetch_tickers, "fetch_nikkei225_tickers", lambda: ["7203.T"])
    monkeypatch.setattr(
        fetch_tickers,
        "_format_ticker_dict",
        lambda tickers, **_kwargs: {"symbols": tickers, "asset_classes": {}},
    )

    assert fetch_tickers.fetch_ticker_dict(ticker_basket=["nikkei225"]) == {
        "symbols": ["7203.T"],
        "asset_classes": {},
    }


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
        lambda tickers, **_kwargs: {"symbols": tickers, "asset_classes": {}},
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
        lambda tickers, **_kwargs: {"symbols": tickers, "asset_classes": {}},
    )

    assert fetch_tickers.fetch_ticker_dict(ticker_basket=["nasdaq100"]) == {
        "symbols": ["NVDA"],
        "asset_classes": {},
    }


def test_fetch_ticker_dict_merges_static_basket_asset_classes_without_fetch(
    monkeypatch,
) -> None:
    captured: dict[str, list[str]] = {}

    def fake_format(tickers):
        captured["tickers"] = list(tickers)
        return {
            "symbols": tickers,
            "asset_classes": {symbol: "sector_unknown" for symbol in tickers},
        }

    monkeypatch.setattr(fetch_tickers, "_format_ticker_dict", fake_format)

    result = fetch_tickers.fetch_ticker_dict(
        ticker_basket=["indexes", "cashlike", "commodities", "realestate"]
    )

    assert result["asset_classes"]["VNQ"] == "sector_real_estate"
    assert result["asset_classes"]["SGOV"] == "cash_like"
    assert result["asset_classes"]["TLT"] == "bond_long"
    assert result["asset_classes"]["GLD"] == "commodity_gold"
    assert result["asset_classes"]["SPY"] == "equity_us_large"
    assert captured["tickers"] == []


def test_format_ticker_dict_normalizes_yahoo_real_estate_sector(monkeypatch) -> None:
    monkeypatch.setattr(
        fetch_tickers,
        "_get_ticker_info",
        lambda ticker: (ticker, "sector_real_estate"),
    )

    result = fetch_tickers._format_ticker_dict(["ARE"])

    assert result == {
        "symbols": ["ARE"],
        "asset_classes": {"ARE": "sector_real_estate"},
    }


def test_fetch_ticker_dict_rejects_empty_builtin_basket(monkeypatch) -> None:
    monkeypatch.setattr(fetch_tickers, "fetch_nasdaq100_tickers", lambda: [])

    try:
        fetch_tickers.fetch_ticker_dict(ticker_basket=["nasdaq100"])
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("empty builtin basket should fail")

    assert "nasdaq100" in message
