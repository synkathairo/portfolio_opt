from __future__ import annotations

import pandas as pd

from portfolio_opt import yfinance_data


def test_yahoo_symbol_candidates_try_original_before_dash_fallback() -> None:
    assert yfinance_data._yahoo_symbol_candidates("BRK.B") == ["BRK.B", "BRK-B"]
    assert yfinance_data._yahoo_symbol_candidates("AZN.L") == ["AZN.L"]
    assert yfinance_data._yahoo_symbol_candidates("BT-A.L") == ["BT-A.L"]
    assert yfinance_data._yahoo_symbol_candidates("600519.SS") == ["600519.SS"]
    assert yfinance_data._yahoo_symbol_candidates("300750.SZ") == ["300750.SZ"]


def test_fetch_closes_aligns_symbols_by_actual_dates(monkeypatch) -> None:
    series_by_symbol = {
        "ACTIVE": pd.Series(
            [10.0, 11.0, 12.0, 13.0],
            index=pd.to_datetime(
                ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
            ),
        ),
        "SHORT": pd.Series(
            [20.0, 21.0, 22.0],
            index=pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]),
        ),
    }

    monkeypatch.setattr(
        yfinance_data,
        "_fetch_single_symbol",
        lambda symbol, period, retries, retry_delay: (symbol, series_by_symbol[symbol]),
    )

    closes = yfinance_data.fetch_closes(["ACTIVE", "SHORT"], max_workers=1)

    assert closes == {
        "ACTIVE": [11.0, 12.0, 13.0],
        "SHORT": [20.0, 21.0, 22.0],
    }


def test_fetch_closes_does_not_pair_stale_history_with_recent_prices(
    monkeypatch,
) -> None:
    series_by_symbol = {
        "ACTIVE": pd.Series(
            [10.0, 11.0, 12.0],
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
        ),
        "DELISTED": pd.Series(
            [20.0, 21.0, 22.0],
            index=pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"]),
        ),
    }

    monkeypatch.setattr(
        yfinance_data,
        "_fetch_single_symbol",
        lambda symbol, period, retries, retry_delay: (symbol, series_by_symbol[symbol]),
    )

    try:
        yfinance_data.fetch_closes(["ACTIVE", "DELISTED"], max_workers=1)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("fetch_closes should reject non-overlapping histories")

    assert "Not enough date-aligned common history" in message


def test_fetch_closes_symbol_delay_paces_single_worker_downloads(monkeypatch) -> None:
    calls: list[str] = []
    sleeps: list[float] = []
    series_by_symbol = {
        "AAA": pd.Series(
            [10.0, 11.0],
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        ),
        "BBB": pd.Series(
            [20.0, 21.0],
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        ),
        "CCC": pd.Series(
            [30.0, 31.0],
            index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        ),
    }

    def fake_fetch_single_symbol(symbol, period, retries, retry_delay):
        calls.append(symbol)
        return symbol, series_by_symbol[symbol]

    monkeypatch.setattr(yfinance_data, "_fetch_single_symbol", fake_fetch_single_symbol)
    monkeypatch.setattr(yfinance_data.time, "sleep", sleeps.append)

    closes = yfinance_data.fetch_closes(
        ["AAA", "BBB", "CCC"],
        max_workers=1,
        symbol_delay=2.5,
    )

    assert calls == ["AAA", "BBB", "CCC"]
    assert sleeps == [2.5, 2.5]
    assert closes == {
        "AAA": [10.0, 11.0],
        "BBB": [20.0, 21.0],
        "CCC": [30.0, 31.0],
    }


def test_fetch_symbols_calls_success_callback_before_later_failure(monkeypatch) -> None:
    series = pd.Series(
        [10.0, 11.0],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    writes: dict[str, pd.Series] = {}

    def fake_fetch_single_symbol(symbol, period, retries, retry_delay):
        if symbol == "BBB":
            raise RuntimeError("rate limited")
        return symbol, series

    monkeypatch.setattr(yfinance_data, "_fetch_single_symbol", fake_fetch_single_symbol)

    try:
        yfinance_data._fetch_symbols(
            ["AAA", "BBB"],
            period="max",
            retries=1,
            retry_delay=0,
            max_workers=1,
            symbol_delay=0,
            on_success=lambda symbol, closes: writes.__setitem__(symbol, closes),
        )
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("fetch should report the later failed symbol")

    assert "Failed to fetch 1/2 symbols: BBB" in message
    assert list(writes) == ["AAA"]
    assert writes["AAA"].equals(series)


def test_fetch_closes_use_cache_avoids_yfinance_download(monkeypatch) -> None:
    cached = {
        "SPY": {
            "symbol": "SPY",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-01": 100.0,
                "2024-01-02": 101.0,
            },
        },
        "QQQ": {
            "symbol": "QQQ",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-01": 200.0,
                "2024-01-02": 201.0,
            },
        },
    }

    class DummyPath:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def exists(self) -> bool:
            return True

        @property
        def name(self) -> str:
            return f"{self.symbol}_hash.json"

        def with_name(self, _name: str):
            return self

    monkeypatch.setattr(
        yfinance_data,
        "cache_path",
        lambda _name, payload: DummyPath(payload["symbol"]),
    )
    monkeypatch.setattr(yfinance_data, "read_cache", lambda path: cached[path.symbol])
    monkeypatch.setattr(
        yfinance_data,
        "_fetch_single_symbol",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("yfinance should not be called on cache hit")
        ),
    )

    closes = yfinance_data.fetch_closes(["SPY", "QQQ"], use_cache=True)

    assert closes == {
        "SPY": [100.0, 101.0],
        "QQQ": [200.0, 201.0],
    }


def test_fetch_closes_refresh_backfills_short_v2_cache(
    monkeypatch,
) -> None:
    cached = {
        "SPY": {
            "symbol": "SPY",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-01": 100.0,
                "2024-01-02": 101.0,
            },
        },
        "QQQ": {
            "symbol": "QQQ",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-01": 200.0,
                "2024-01-02": 201.0,
            },
        },
    }
    writes: dict[str, dict] = {}

    class DummyPath:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def exists(self) -> bool:
            return self.symbol in cached

        @property
        def name(self) -> str:
            return f"{self.symbol}_hash.json"

        def with_name(self, _name: str):
            return self

    def fake_cache_path(_name, payload):
        return DummyPath(payload["symbol"])

    def fake_fetch_symbols(symbols, **_kwargs):
        assert symbols == ["SPY", "QQQ"]
        fetched = {
            "SPY": pd.Series(
                [99.0, 100.0, 101.0, 102.0],
                index=pd.to_datetime(
                    ["2023-12-29", "2024-01-01", "2024-01-02", "2024-01-03"]
                ),
            ),
            "QQQ": pd.Series(
                [199.0, 200.0, 201.0, 202.0],
                index=pd.to_datetime(
                    ["2023-12-29", "2024-01-01", "2024-01-02", "2024-01-03"]
                ),
            ),
        }
        on_success = _kwargs.get("on_success")
        if on_success is not None:
            for symbol, series in fetched.items():
                on_success(symbol, series)
        return fetched

    monkeypatch.setattr(yfinance_data, "cache_path", fake_cache_path)
    monkeypatch.setattr(
        yfinance_data,
        "read_cache",
        lambda path: cached[path.symbol],
    )
    monkeypatch.setattr(
        yfinance_data,
        "write_cache",
        lambda path, payload: writes.__setitem__(path.symbol, payload),
    )
    monkeypatch.setattr(yfinance_data, "_fetch_symbols", fake_fetch_symbols)
    monkeypatch.setattr(
        yfinance_data,
        "_fetch_symbols_since",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tail refresh should not run for short cache")
        ),
    )

    closes = yfinance_data.fetch_closes(
        ["SPY", "QQQ"],
        min_history_days=4,
        use_cache=True,
        refresh_cache=True,
    )

    assert closes == {
        "SPY": [99.0, 100.0, 101.0, 102.0],
        "QQQ": [199.0, 200.0, 201.0, 202.0],
    }
    assert writes["SPY"] == {
        "symbol": "SPY",
        "source": "yfinance",
        "adjustment": "auto",
        "closes": {
            "2023-12-29": 99.0,
            "2024-01-01": 100.0,
            "2024-01-02": 101.0,
            "2024-01-03": 102.0,
        },
    }
    assert writes["QQQ"] == {
        "symbol": "QQQ",
        "source": "yfinance",
        "adjustment": "auto",
        "closes": {
            "2023-12-29": 199.0,
            "2024-01-01": 200.0,
            "2024-01-02": 201.0,
            "2024-01-03": 202.0,
        },
    }


def test_fetch_closes_refresh_fetches_only_missing_tail_for_sufficient_v2_cache(
    monkeypatch,
) -> None:
    cached = {
        "SPY": {
            "symbol": "SPY",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-01": 100.0,
                "2024-01-02": 101.0,
            },
        },
        "QQQ": {
            "symbol": "QQQ",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-01": 200.0,
                "2024-01-02": 201.0,
            },
        },
    }
    writes: dict[str, dict] = {}

    class DummyPath:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def exists(self) -> bool:
            return self.symbol in cached

        @property
        def name(self) -> str:
            return f"{self.symbol}_hash.json"

        def with_name(self, _name: str):
            return self

    def fake_cache_path(_name, payload):
        return DummyPath(payload["symbol"])

    def fake_fetch_symbols(symbols, **_kwargs):
        assert symbols == []
        return {}

    def fake_fetch_symbols_since(starts_by_symbol, **_kwargs):
        assert starts_by_symbol == {
            "SPY": pd.Timestamp("2024-01-03"),
            "QQQ": pd.Timestamp("2024-01-03"),
        }
        fetched = {
            "SPY": pd.Series(
                [102.0],
                index=pd.to_datetime(["2024-01-03"]),
            ),
            "QQQ": pd.Series(
                [202.0],
                index=pd.to_datetime(["2024-01-03"]),
            ),
        }
        on_success = _kwargs.get("on_success")
        if on_success is not None:
            for symbol, series in fetched.items():
                on_success(symbol, series)
        return fetched

    monkeypatch.setattr(yfinance_data, "cache_path", fake_cache_path)
    monkeypatch.setattr(
        yfinance_data,
        "read_cache",
        lambda path: cached[path.symbol],
    )
    monkeypatch.setattr(
        yfinance_data,
        "write_cache",
        lambda path, payload: writes.__setitem__(path.symbol, payload),
    )
    monkeypatch.setattr(yfinance_data, "_fetch_symbols", fake_fetch_symbols)
    monkeypatch.setattr(yfinance_data, "_fetch_symbols_since", fake_fetch_symbols_since)

    closes = yfinance_data.fetch_closes(
        ["SPY", "QQQ"],
        min_history_days=2,
        use_cache=True,
        refresh_cache=True,
    )

    assert closes == {
        "SPY": [100.0, 101.0, 102.0],
        "QQQ": [200.0, 201.0, 202.0],
    }
    assert writes["SPY"]["closes"]["2024-01-03"] == 102.0
    assert writes["QQQ"]["closes"]["2024-01-03"] == 202.0


def test_fetch_closes_min_history_filters_before_common_alignment(monkeypatch) -> None:
    cached = {
        "OLD": {
            "symbol": "OLD",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-01": 100.0,
                "2024-01-02": 101.0,
                "2024-01-03": 102.0,
                "2024-01-04": 103.0,
            },
        },
        "NEW": {
            "symbol": "NEW",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-03": 50.0,
                "2024-01-04": 51.0,
            },
        },
    }

    class DummyPath:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def exists(self) -> bool:
            return self.symbol in cached

        @property
        def name(self) -> str:
            return f"{self.symbol}_hash.json"

        def with_name(self, _name: str):
            return self

    monkeypatch.setattr(
        yfinance_data,
        "cache_path",
        lambda _name, payload: DummyPath(payload["symbol"]),
    )
    monkeypatch.setattr(yfinance_data, "read_cache", lambda path: cached[path.symbol])

    closes = yfinance_data.fetch_closes(
        ["OLD", "NEW"],
        min_history_days=4,
        use_cache=True,
        offline=True,
    )

    assert closes == {"OLD": [100.0, 101.0, 102.0, 103.0]}


def test_fetch_closes_use_cache_fetches_only_missing_v2_symbols(monkeypatch) -> None:
    cached = {
        "SPY": {
            "symbol": "SPY",
            "source": "yfinance",
            "adjustment": "auto",
            "closes": {
                "2024-01-01": 100.0,
                "2024-01-02": 101.0,
            },
        },
    }
    writes: dict[str, dict] = {}

    class DummyPath:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def exists(self) -> bool:
            return self.symbol in cached

        @property
        def name(self) -> str:
            return f"{self.symbol}_hash.json"

        def with_name(self, _name: str):
            return self

    def fake_cache_path(_name, payload):
        return DummyPath(payload["symbol"])

    def fake_fetch_symbols(symbols, **_kwargs):
        assert symbols == ["QQQ"]
        fetched = {
            "QQQ": pd.Series(
                [200.0, 201.0],
                index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
            )
        }
        on_success = _kwargs.get("on_success")
        if on_success is not None:
            for symbol, series in fetched.items():
                on_success(symbol, series)
        return fetched

    monkeypatch.setattr(yfinance_data, "cache_path", fake_cache_path)
    monkeypatch.setattr(
        yfinance_data,
        "read_cache",
        lambda path: cached[path.symbol],
    )
    monkeypatch.setattr(
        yfinance_data,
        "write_cache",
        lambda path, payload: writes.__setitem__(path.symbol, payload),
    )
    monkeypatch.setattr(yfinance_data, "_fetch_symbols", fake_fetch_symbols)
    monkeypatch.setattr(
        yfinance_data,
        "_fetch_symbols_since",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("tail refresh should not run without refresh_cache")
        ),
    )

    closes = yfinance_data.fetch_closes(["SPY", "QQQ"], use_cache=True)

    assert closes == {
        "SPY": [100.0, 101.0],
        "QQQ": [200.0, 201.0],
    }
    assert writes["QQQ"] == {
        "symbol": "QQQ",
        "source": "yfinance",
        "adjustment": "auto",
        "closes": {
            "2024-01-01": 200.0,
            "2024-01-02": 201.0,
        },
    }
