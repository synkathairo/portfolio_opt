"""Fetch and cache daily OHLCV data from StockAnalysis chart JSON."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from .cache import cache_path, read_cache, write_cache

DEFAULT_START = "1980-01-01"


def _safe_symbol(symbol: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in symbol.upper())


def _chart_url(symbol: str, *, start: str, end: str) -> str:
    encoded = quote(symbol.lower(), safe="")
    return (
        f"https://stockanalysis.com/api/charts/s/{encoded}/MAX/c"
        f"?chartiq=true&start={start}&end={end}"
    )


def _chart_cache_path(symbol: str, *, start: str, end: str) -> Path:
    path = cache_path(
        "stockanalysis_chart",
        {
            "kind": "stockanalysis_chart",
            "symbol": symbol,
            "start": start,
            "end": end,
            "interval": "MAX/c",
            "chartiq": True,
        },
    )
    return path.with_name(
        f"stockanalysis_chart_{_safe_symbol(symbol)}_{path.name.rsplit('_', 1)[-1]}"
    )


def _payload_to_close_series(symbol: str, payload: Any) -> pd.Series:
    if not isinstance(payload, dict):
        return pd.Series(dtype=float)
    rows = payload.get("data")
    if not isinstance(rows, list):
        return pd.Series(dtype=float)

    dates: list[str] = []
    closes: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_date = row.get("t")
        raw_close = row.get("c")
        if raw_date is None or raw_close is None:
            continue
        try:
            closes.append(float(raw_close))
        except (TypeError, ValueError):
            continue
        dates.append(str(raw_date)[:10])

    if not dates:
        return pd.Series(dtype=float, name=symbol)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(dates, errors="coerce"),
            "close": closes,
        }
    ).dropna()
    if frame.empty:
        return pd.Series(dtype=float, name=symbol)
    frame["date"] = frame["date"].dt.normalize()
    series = frame.set_index("date")["close"].astype(float)
    series.name = symbol
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _fetch_symbol_payload(symbol: str, *, start: str, end: str) -> dict[str, Any]:
    response = requests.get(
        _chart_url(symbol, start=start, end=end),
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected StockAnalysis payload for {symbol}.")
    return payload


def fetch_closes(
    symbols: list[str],
    *,
    start: str = DEFAULT_START,
    end: str | None = None,
    use_cache: bool = False,
    refresh_cache: bool = False,
    offline: bool = False,
) -> dict[str, list[float]]:
    """Fetch or read date-aligned close prices from StockAnalysis chart JSON."""
    if not symbols:
        return {}

    end = end or date.today().isoformat()
    series_by_symbol: dict[str, pd.Series] = {}
    missing: list[str] = []

    for symbol in symbols:
        path = _chart_cache_path(symbol, start=start, end=end)
        payload: Any | None = None
        if not refresh_cache and path.exists():
            payload = read_cache(path)
        elif offline:
            missing.append(symbol)
            continue
        else:
            payload = _fetch_symbol_payload(symbol, start=start, end=end)
            if use_cache or refresh_cache:
                write_cache(path, payload)

        series = _payload_to_close_series(symbol, payload)
        if series.empty:
            missing.append(symbol)
            continue
        series_by_symbol[symbol] = series

    if missing:
        raise ValueError(
            "Missing StockAnalysis data for "
            f"{len(missing)} symbol(s): {', '.join(missing)}"
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
            "Not enough date-aligned common StockAnalysis history. "
            f"Shortest raw history: {shortest_symbol} with "
            f"{lengths[shortest_symbol]} bars."
        )

    return {
        symbol: [float(value) for value in close_frame[symbol].to_numpy()]
        for symbol in symbols
    }
