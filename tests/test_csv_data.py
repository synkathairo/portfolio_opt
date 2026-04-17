from __future__ import annotations

import pytest

from portfolio_opt import csv_data
from portfolio_opt.csv_data import fetch_closes


def test_fetch_closes_reads_headerless_csvs_and_aligns_dates(tmp_path) -> None:
    (tmp_path / "prices_a.csv").write_text(
        "\n".join(
            [
                "AAA,2024-01-01,9,11,8,10,1000",
                "AAA,2024-01-02,10,12,9,11,1000",
                "AAA,2024-01-03,11,13,10,12,1000",
            ]
        )
    )
    (tmp_path / "prices_b.csv").write_text(
        "\n".join(
            [
                "BBB,2024-01-02,19,21,18,20,2000",
                "BBB,2024-01-03,20,22,19,21,2000",
                "BBB,2024-01-04,21,23,20,22,2000",
            ]
        )
    )

    assert fetch_closes(["AAA", "BBB"], csv_dir=tmp_path) == {
        "AAA": [11.0, 12.0],
        "BBB": [20.0, 21.0],
    }


def test_fetch_closes_reads_headered_csv_and_keeps_last_duplicate_date(
    tmp_path,
) -> None:
    (tmp_path / "prices.csv").write_text(
        "\n".join(
            [
                "symbol,date,open,high,low,close,volume",
                "AAA,2024-01-01,9,11,8,10,1000",
                "AAA,2024-01-02,10,12,9,11,1000",
                "AAA,2024-01-02,10,12,9,12,1000",
                "BBB,2024-01-01,19,21,18,20,2000",
                "BBB,2024-01-02,20,22,19,21,2000",
            ]
        )
    )

    assert fetch_closes(["AAA", "BBB"], csv_dir=tmp_path) == {
        "AAA": [10.0, 12.0],
        "BBB": [20.0, 21.0],
    }


def test_fetch_closes_reports_missing_symbols(tmp_path) -> None:
    (tmp_path / "prices.csv").write_text("AAA,2024-01-01,9,11,8,10,1000\n")

    with pytest.raises(ValueError, match="Missing CSV data.*BBB"):
        fetch_closes(["AAA", "BBB"], csv_dir=tmp_path)


def test_write_json_caches_writes_provider_neutral_payload(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "prices.csv").write_text(
        "\n".join(
            [
                "AAA,2024-01-01,9,11,8,10,1000",
                "AAA,2024-01-02,10,12,9,11,1000",
            ]
        )
    )
    writes: dict[str, dict] = {}

    def fake_cache_path(name, _payload):
        return tmp_path / f"{name}_hash.json"

    monkeypatch.setattr(csv_data, "cache_path", fake_cache_path)
    monkeypatch.setattr(
        csv_data,
        "write_cache",
        lambda path, payload: writes.__setitem__(path.name, payload),
    )

    paths = csv_data.write_json_caches(tmp_path)

    assert [path.name for path in paths] == ["csv_closes_v2_AAA_hash.json"]
    assert writes["csv_closes_v2_AAA_hash.json"] == {
        "symbol": "AAA",
        "source": "csv",
        "columns": csv_data.CSV_COLUMNS,
        "closes": {
            "2024-01-01": 10.0,
            "2024-01-02": 11.0,
        },
    }


def test_write_yfinance_compatible_caches_uses_existing_cache_layout(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "prices.csv").write_text("AAA,2024-01-01,9,11,8,10,1000\n")
    writes: dict[str, dict] = {}

    def fake_cache_path(name, _payload):
        return tmp_path / f"{name}_hash.json"

    monkeypatch.setattr(csv_data, "cache_path", fake_cache_path)
    monkeypatch.setattr(
        csv_data,
        "write_cache",
        lambda path, payload: writes.__setitem__(path.name, payload),
    )

    paths = csv_data.write_yfinance_compatible_caches(tmp_path, symbols=["AAA", "BBB"])

    assert [path.name for path in paths] == ["yfinance_closes_v2_AAA_hash.json"]
    assert writes["yfinance_closes_v2_AAA_hash.json"] == {
        "symbol": "AAA",
        "source": "csv",
        "columns": csv_data.CSV_COLUMNS,
        "adjustment": "auto",
        "closes": {
            "2024-01-01": 10.0,
        },
    }
