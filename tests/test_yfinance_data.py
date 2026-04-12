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
