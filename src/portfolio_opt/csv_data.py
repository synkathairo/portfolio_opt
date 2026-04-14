"""Read daily closes from local OHLCV CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .cache import cache_path, write_cache

CSV_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume"]


def _read_ohlcv_csv(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=CSV_COLUMNS)

    normalized_columns = [str(column).strip().lower() for column in frame.columns]
    if normalized_columns[: len(CSV_COLUMNS)] != CSV_COLUMNS:
        frame = pd.read_csv(path, names=CSV_COLUMNS)
    else:
        frame.columns = normalized_columns

    missing = [column for column in CSV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")

    frame = frame[CSV_COLUMNS].copy()
    frame["symbol"] = frame["symbol"].astype(str).str.strip()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["symbol", "date", "close"])
    frame = frame[frame["symbol"] != ""]
    return frame


def load_close_series_by_symbol(csv_dir: str | Path = ".cache/csv") -> dict[str, pd.Series]:
    csv_dir = Path(csv_dir)
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory does not exist: {csv_dir}")
    if not csv_dir.is_dir():
        raise NotADirectoryError(f"CSV path is not a directory: {csv_dir}")

    series_by_symbol: dict[str, pd.Series] = {}
    for path in sorted(csv_dir.glob("*.csv")):
        frame = _read_ohlcv_csv(path)
        for raw_symbol, group in frame.groupby("symbol", sort=False):
            symbol = str(raw_symbol)
            series = (
                group[["date", "close"]]
                .drop_duplicates(subset=["date"], keep="last")
                .set_index("date")["close"]
                .astype(float)
                .sort_index()
            )
            if series.empty:
                continue
            if symbol in series_by_symbol:
                series = pd.concat([series_by_symbol[symbol], series])
                series = series[~series.index.duplicated(keep="last")].sort_index()
            series_by_symbol[symbol] = series

    return series_by_symbol


def _series_to_cache_payload(
    symbol: str,
    series: pd.Series,
    *,
    source: str,
) -> dict[str, Any]:
    closes: dict[str, float] = {}
    for index, value in series.sort_index().items():
        closes[str(index)[:10]] = float(value)
    return {
        "symbol": symbol,
        "source": source,
        "columns": CSV_COLUMNS,
        "closes": closes,
    }


def _safe_symbol(symbol: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in symbol.upper())


def _csv_closes_cache_path(symbol: str) -> Path:
    path = cache_path(
        "csv_closes_v2",
        {
            "kind": "csv_closes_v2",
            "symbol": symbol,
            "columns": CSV_COLUMNS,
        },
    )
    return path.with_name(f"csv_closes_v2_{_safe_symbol(symbol)}_{path.name.rsplit('_', 1)[-1]}")


def _yfinance_compatible_cache_path(symbol: str) -> Path:
    path = cache_path(
        "yfinance_closes_v2",
        {
            "kind": "yfinance_closes_v2",
            "symbol": symbol,
            "adjustment": "auto",
        },
    )
    return path.with_name(
        f"yfinance_closes_v2_{_safe_symbol(symbol)}_{path.name.rsplit('_', 1)[-1]}"
    )


def write_json_caches(
    csv_dir: str | Path = ".cache/csv",
    *,
    symbols: list[str] | None = None,
) -> list[Path]:
    """Write provider-neutral JSON close caches from local CSV files."""
    series_by_symbol = load_close_series_by_symbol(csv_dir)
    selected_symbols = symbols if symbols is not None else sorted(series_by_symbol)
    paths: list[Path] = []
    for symbol in selected_symbols:
        series = series_by_symbol.get(symbol)
        if series is None or series.empty:
            continue
        path = _csv_closes_cache_path(symbol)
        write_cache(path, _series_to_cache_payload(symbol, series, source="csv"))
        paths.append(path)
    return paths


def write_yfinance_compatible_caches(
    csv_dir: str | Path = ".cache/csv",
    *,
    symbols: list[str] | None = None,
) -> list[Path]:
    """Write CSV closes into the existing per-symbol JSON cache layout.

    This lets ``--data-source csv+yfinance`` reuse local CSV data for symbols
    present on disk while yfinance fills symbols that are missing from CSV.
    """
    series_by_symbol = load_close_series_by_symbol(csv_dir)
    selected_symbols = symbols if symbols is not None else sorted(series_by_symbol)
    paths: list[Path] = []
    for symbol in selected_symbols:
        series = series_by_symbol.get(symbol)
        if series is None or series.empty:
            continue
        path = _yfinance_compatible_cache_path(symbol)
        payload = _series_to_cache_payload(symbol, series, source="csv")
        payload["adjustment"] = "auto"
        write_cache(path, payload)
        paths.append(path)
    return paths


def fetch_closes(
    symbols: list[str],
    csv_dir: str | Path = ".cache/csv",
) -> dict[str, list[float]]:
    """Read date-aligned close prices from local OHLCV CSV files.

    The reader accepts CSVs with or without a header row. Rows must use:

    ``symbol,date,open,high,low,close,volume``
    """
    if not symbols:
        return {}

    series_by_symbol = load_close_series_by_symbol(csv_dir)
    missing = [symbol for symbol in symbols if symbol not in series_by_symbol]
    if missing:
        raise ValueError(
            f"Missing CSV data for {len(missing)} symbol(s): {', '.join(missing)}"
        )

    close_frame = pd.concat(
        [series_by_symbol[symbol].rename(symbol) for symbol in symbols],
        axis=1,
        join="inner",
    ).dropna(how="any")
    if len(close_frame) < 2:
        lengths = {symbol: len(series_by_symbol[symbol]) for symbol in symbols}
        shortest_symbol = min(lengths, key=lambda symbol: lengths[symbol])
        raise ValueError(
            "Not enough date-aligned common CSV history. "
            f"Shortest raw history: {shortest_symbol} with "
            f"{lengths[shortest_symbol]} bars."
        )

    return {
        symbol: [float(value) for value in close_frame[symbol].to_numpy()]
        for symbol in symbols
    }
