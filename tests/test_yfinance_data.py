from __future__ import annotations

import pandas as pd

from portfolio_opt import yfinance_data


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


def test_fetch_closes_use_cache_avoids_yfinance_download(monkeypatch) -> None:
    cached = {"SPY": [100.0, 101.0], "QQQ": [200.0, 201.0]}

    class DummyPath:
        def exists(self) -> bool:
            return True

        @property
        def name(self) -> str:
            return "dummy_hash.json"

        def with_name(self, _name: str):
            return self

    monkeypatch.setattr(yfinance_data, "cache_path", lambda name, payload: DummyPath())
    monkeypatch.setattr(yfinance_data, "read_cache", lambda path: cached)
    monkeypatch.setattr(
        yfinance_data,
        "_fetch_single_symbol",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("yfinance should not be called on cache hit")
        ),
    )

    closes = yfinance_data.fetch_closes(["SPY", "QQQ"], use_cache=True)

    assert closes == cached


def test_fetch_closes_refresh_fetches_only_missing_tail_for_v2_cache(
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
        return {
            "SPY": pd.Series(
                [102.0],
                index=pd.to_datetime(["2024-01-03"]),
            ),
            "QQQ": pd.Series(
                [202.0],
                index=pd.to_datetime(["2024-01-03"]),
            ),
        }

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
        use_cache=True,
        refresh_cache=True,
    )

    assert closes == {
        "SPY": [100.0, 101.0, 102.0],
        "QQQ": [200.0, 201.0, 202.0],
    }
    assert writes["SPY"] == {
        "symbol": "SPY",
        "source": "yfinance",
        "adjustment": "auto",
        "closes": {
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
            "2024-01-01": 200.0,
            "2024-01-02": 201.0,
            "2024-01-03": 202.0,
        },
    }
