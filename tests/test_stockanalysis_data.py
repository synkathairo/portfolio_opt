from __future__ import annotations

import pytest

from portfolio_opt import stockanalysis_data


def _payload(closes: list[tuple[str, float]]) -> dict:
    return {
        "status": 200,
        "data": [
            {"t": day, "o": close - 1, "h": close + 1, "l": close - 2, "c": close, "v": 1000}
            for day, close in closes
        ],
    }


def test_fetch_closes_fetches_json_writes_cache_and_aligns_dates(
    tmp_path, monkeypatch
) -> None:
    payloads = {
        "AAA": _payload(
            [
                ("2024-01-01", 10.0),
                ("2024-01-02", 11.0),
                ("2024-01-03", 12.0),
            ]
        ),
        "BBB": _payload(
            [
                ("2024-01-02", 20.0),
                ("2024-01-03", 21.0),
                ("2024-01-04", 22.0),
            ]
        ),
    }
    writes: dict[str, dict] = {}
    requested_urls: list[str] = []

    class FakeResponse:
        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return payloads[self.symbol]

    def fake_get(url: str, **_kwargs):
        requested_urls.append(url)
        symbol = "AAA" if "/aaa/" in url else "BBB"
        return FakeResponse(symbol)

    monkeypatch.setattr(stockanalysis_data.requests, "get", fake_get)
    monkeypatch.setattr(
        stockanalysis_data,
        "cache_path",
        lambda _name, payload: tmp_path / f"{payload['symbol']}_hash.json",
    )
    monkeypatch.setattr(
        stockanalysis_data,
        "write_cache",
        lambda path, payload: writes.__setitem__(path.name, payload),
    )

    closes = stockanalysis_data.fetch_closes(
        ["AAA", "BBB"],
        start="2024-01-01",
        end="2024-01-04",
        use_cache=True,
    )

    assert closes == {"AAA": [11.0, 12.0], "BBB": [20.0, 21.0]}
    assert len(requested_urls) == 2
    assert set(writes) == {
        "stockanalysis_chart_AAA_hash.json",
        "stockanalysis_chart_BBB_hash.json",
    }


def test_fetch_closes_offline_reads_json_cache(tmp_path, monkeypatch) -> None:
    cached = {"AAA": _payload([("2024-01-01", 10.0), ("2024-01-02", 11.0)])}

    monkeypatch.setattr(
        stockanalysis_data,
        "cache_path",
        lambda _name, payload: tmp_path / f"{payload['symbol']}_hash.json",
    )
    monkeypatch.setattr(
        stockanalysis_data,
        "read_cache",
        lambda path: cached[path.name.split("_")[2]],
    )
    (tmp_path / "stockanalysis_chart_AAA_hash.json").write_text("{}")
    monkeypatch.setattr(
        stockanalysis_data.requests,
        "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("network should not be used in offline cache mode")
        ),
    )

    assert stockanalysis_data.fetch_closes(["AAA"], offline=True) == {
        "AAA": [10.0, 11.0]
    }


def test_fetch_closes_reports_empty_symbol(tmp_path, monkeypatch) -> None:
    class FakeResponse:
        def raise_for_status(self) -> None:
            pass

        def json(self) -> dict:
            return {"status": 200, "data": []}

    monkeypatch.setattr(stockanalysis_data.requests, "get", lambda *_args, **_kwargs: FakeResponse())
    monkeypatch.setattr(
        stockanalysis_data,
        "cache_path",
        lambda _name, payload: tmp_path / f"{payload['symbol']}_hash.json",
    )

    with pytest.raises(ValueError, match="Missing StockAnalysis data.*AAA"):
        stockanalysis_data.fetch_closes(["AAA"])
