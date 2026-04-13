"""Fetch historical daily closes from Yahoo Finance via yfinance.

Provides a drop-in replacement for the Alpaca daily-closes cache so the
custom backtest can reach back to ETF inception (e.g. the 2008 crisis).
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import pandas as pd

from .cache import cache_path, read_cache, write_cache

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "yfinance is required for historical data. Install with `pip install yfinance`."
    ) from exc

logger = logging.getLogger(__name__)


def _yahoo_symbol_candidates(symbol: str) -> list[str]:
    candidates = [symbol]
    if "." in symbol:
        fallback = symbol.replace(".", "-")
        candidates.append(fallback)
    return list(dict.fromkeys(candidates))


def _fetch_single_symbol(
    symbol: str,
    period: str,
    retries: int,
    retry_delay: float,
) -> tuple[str, pd.Series]:
    """Fetch daily adjusted closes for a single symbol from Yahoo Finance."""
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        for yahoo_sym in _yahoo_symbol_candidates(symbol):
            try:
                ticker = yf.Ticker(yahoo_sym)
                hist = ticker.history(period=period, auto_adjust=True)
                if hist.empty or "Close" not in hist.columns:
                    raise ValueError(f"No close data for {symbol}")
                closes = hist["Close"].astype(float).dropna()
                if closes.empty:
                    raise ValueError(f"Empty close series for {symbol}")
                closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
                closes = closes[~closes.index.duplicated(keep="last")].sort_index()
                return symbol, closes
            except Exception as exc:
                last_err = exc
                logger.debug("Failed to fetch %s as %s: %s", symbol, yahoo_sym, exc)
                continue
        if attempt < retries:
            logger.debug("Retry %d/%d for %s: %s", attempt, retries, symbol, last_err)
            time.sleep(retry_delay)
    raise RuntimeError(f"Failed to fetch {symbol} after {retries} retries: {last_err}")


def _fetch_single_symbol_from(
    symbol: str,
    start: pd.Timestamp,
    retries: int,
    retry_delay: float,
) -> tuple[str, pd.Series]:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        for yahoo_sym in _yahoo_symbol_candidates(symbol):
            try:
                ticker = yf.Ticker(yahoo_sym)
                hist = ticker.history(start=start.date().isoformat(), auto_adjust=True)
                if hist.empty or "Close" not in hist.columns:
                    return symbol, pd.Series(dtype=float)
                closes = hist["Close"].astype(float).dropna()
                if closes.empty:
                    return symbol, pd.Series(dtype=float)
                closes.index = pd.to_datetime(closes.index).tz_localize(None).normalize()
                closes = closes[~closes.index.duplicated(keep="last")].sort_index()
                return symbol, closes
            except Exception as exc:
                last_err = exc
                logger.debug("Failed to fetch %s as %s: %s", symbol, yahoo_sym, exc)
                continue
        if attempt < retries:
            logger.debug("Retry %d/%d for %s: %s", attempt, retries, symbol, last_err)
            time.sleep(retry_delay)
    raise RuntimeError(f"Failed to fetch {symbol} after {retries} retries: {last_err}")


def fetch_closes(
    symbols: list[str],
    period: str = "max",
    retries: int = 3,
    retry_delay: float = 1.0,
    max_workers: int = 10,
    use_cache: bool = False,
    refresh_cache: bool = False,
    offline: bool = False,
) -> dict[str, list[float]]:
    """Download daily adjusted closes for *symbols* from Yahoo Finance.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols (e.g. ``["SPY", "QQQ", "GLD"]``).
    period : str
        yfinance period string — ``"max"``, ``"10y"``, ``"5y"``, etc.
    retries : int
        Number of retry attempts per symbol on transient failures.
    retry_delay : float
        Seconds to wait between retries.
    max_workers : int
        Maximum number of concurrent threads for fetching symbols.
    use_cache : bool
        Reuse a cached yfinance result when available, otherwise fetch and write it.
    refresh_cache : bool
        Fetch from yfinance and overwrite the cached result.
    offline : bool
        Read from cache only and never call yfinance.

    Returns
    -------
    dict[str, list[float]]
        Mapping of symbol to adjusted close prices, oldest first. All lists are
        aligned by actual trading date so delisted or stale histories cannot be
        paired with unrelated recent bars from active tickers.
    """
    if not symbols:
        return {}

    if use_cache or refresh_cache or offline:
        incremental = _incremental_fetch_closes(
            symbols=symbols,
            period=period,
            retries=retries,
            retry_delay=retry_delay,
            max_workers=max_workers,
            refresh_cache=refresh_cache,
            offline=offline,
        )
        if incremental is not None:
            return incremental

    path = cache_path(
        "yfinance_closes",
        {"kind": "yfinance_closes", "symbols": symbols, "period": period},
    )
    if offline:
        if not path.exists():
            raise RuntimeError(f"Offline mode requested but cache is missing: {path}")
        return {
            symbol: [float(value) for value in values]
            for symbol, values in read_cache(path).items()
        }
    if use_cache and path.exists() and not refresh_cache:
        return {
            symbol: [float(value) for value in values]
            for symbol, values in read_cache(path).items()
        }

    closes_by_symbol: dict[str, pd.Series] = {}
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_symbol, s, period, retries, retry_delay): s
            for s in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                fetched_symbol, closes = future.result()
                closes_by_symbol[fetched_symbol] = closes
            except RuntimeError as exc:
                failed.append(symbol)
                logger.warning("Failed to fetch %s: %s", symbol, exc)

    if failed:
        raise RuntimeError(
            f"Failed to fetch {len(failed)}/{len(symbols)} symbols: "
            f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}"
        )

    close_frame = pd.concat(
        [closes_by_symbol[symbol].rename(symbol) for symbol in symbols],
        axis=1,
        join="inner",
    ).dropna(how="any")
    if len(close_frame) < 2:
        lengths = {symbol: len(series) for symbol, series in closes_by_symbol.items()}
        raise ValueError(
            "Not enough date-aligned common history. "
            f"Shortest raw history: {min(lengths, key=lambda s: lengths[s])} "
            f"with {min(lengths.values())} bars."
        )

    aligned = {
        symbol: [float(value) for value in close_frame[symbol].to_numpy()]
        for symbol in symbols
    }
    if use_cache or refresh_cache:
        write_cache(path, aligned)
    return aligned


def _incremental_fetch_closes(
    *,
    symbols: list[str],
    period: str,
    retries: int,
    retry_delay: float,
    max_workers: int,
    refresh_cache: bool,
    offline: bool,
) -> dict[str, list[float]] | None:
    cached_by_symbol: dict[str, pd.Series] = {}
    missing_symbols: list[str] = []
    for symbol in symbols:
        path = _symbol_closes_cache_path(symbol)
        if not path.exists():
            missing_symbols.append(symbol)
            continue
        series = _series_from_cached_rows(read_cache(path))
        if series.empty:
            missing_symbols.append(symbol)
            continue
        cached_by_symbol[symbol] = series

    if offline:
        if missing_symbols:
            return None
        return _align_close_series(symbols, cached_by_symbol)

    if not refresh_cache:
        if missing_symbols:
            return None
        return _align_close_series(symbols, cached_by_symbol)

    refreshed = _fetch_symbols(
        missing_symbols,
        period=period,
        retries=retries,
        retry_delay=retry_delay,
        max_workers=max_workers,
    )
    today = pd.Timestamp.today().normalize()
    starts_by_symbol: dict[str, pd.Timestamp] = {}
    for symbol, series in cached_by_symbol.items():
        if symbol in missing_symbols or series.empty:
            continue
        next_start = series.index[-1] + pd.Timedelta(days=1)
        if next_start <= today:
            starts_by_symbol[symbol] = next_start
    refreshed.update(
        _fetch_symbols_since(
            starts_by_symbol,
            retries=retries,
            retry_delay=retry_delay,
            max_workers=max_workers,
        )
    )
    for symbol, series in refreshed.items():
        if series.empty and symbol in cached_by_symbol:
            continue
        merged = _merge_close_series(cached_by_symbol.get(symbol), series)
        cached_by_symbol[symbol] = merged
        write_cache(_symbol_closes_cache_path(symbol), _series_to_cached_rows(symbol, merged))

    if any(symbol not in cached_by_symbol for symbol in symbols):
        return None
    return _align_close_series(symbols, cached_by_symbol)


def _fetch_symbols(
    symbols: list[str],
    *,
    period: str,
    retries: int,
    retry_delay: float,
    max_workers: int,
) -> dict[str, pd.Series]:
    if not symbols:
        return {}
    closes_by_symbol: dict[str, pd.Series] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_symbol, s, period, retries, retry_delay): s
            for s in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                fetched_symbol, closes = future.result()
                closes_by_symbol[fetched_symbol] = closes
            except RuntimeError as exc:
                failed.append(symbol)
                logger.warning("Failed to fetch %s: %s", symbol, exc)

    if failed:
        raise RuntimeError(
            f"Failed to fetch {len(failed)}/{len(symbols)} symbols: "
            f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}"
        )
    return closes_by_symbol


def _fetch_symbols_since(
    starts_by_symbol: dict[str, pd.Timestamp],
    *,
    retries: int,
    retry_delay: float,
    max_workers: int,
) -> dict[str, pd.Series]:
    if not starts_by_symbol:
        return {}
    closes_by_symbol: dict[str, pd.Series] = {}
    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _fetch_single_symbol_from,
                symbol,
                start,
                retries,
                retry_delay,
            ): symbol
            for symbol, start in starts_by_symbol.items()
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                fetched_symbol, closes = future.result()
                closes_by_symbol[fetched_symbol] = closes
            except RuntimeError as exc:
                failed.append(symbol)
                logger.warning("Failed to fetch %s: %s", symbol, exc)

    if failed:
        raise RuntimeError(
            f"Failed to fetch {len(failed)}/{len(starts_by_symbol)} symbols: "
            f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}"
        )
    return closes_by_symbol


def _symbol_closes_cache_path(symbol: str):
    safe_symbol = "".join(char if char.isalnum() else "_" for char in symbol.upper())
    path = cache_path(
        "yfinance_closes_v2",
        {
            "kind": "yfinance_closes_v2",
            "symbol": symbol,
            "adjustment": "auto",
        },
    )
    return path.with_name(f"yfinance_closes_v2_{safe_symbol}_{path.name.rsplit('_', 1)[-1]}")


def _series_from_cached_rows(payload: Any) -> pd.Series:
    if isinstance(payload, dict):
        closes = payload.get("closes")
        if isinstance(closes, dict):
            payload = [
                {"timestamp": date, "close": close}
                for date, close in closes.items()
            ]
    if not isinstance(payload, list):
        return pd.Series(dtype=float)
    dates: list[str] = []
    values: list[float] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        if "timestamp" not in row or "close" not in row:
            continue
        dates.append(str(row["timestamp"])[:10])
        values.append(float(row["close"]))
    if not dates:
        return pd.Series(dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(dates))
    series = pd.Series(values, index=index, dtype=float)
    series = series[~series.index.duplicated(keep="last")].sort_index()
    return series


def _series_to_cached_rows(
    symbol: str,
    series: pd.Series,
) -> dict[str, Any]:
    closes: dict[str, float] = {}
    for index, value in series.sort_index().items():
        closes[str(index)[:10]] = float(value)
    return {
        "symbol": symbol,
        "source": "yfinance",
        "adjustment": "auto",
        "closes": closes,
    }


def _merge_close_series(
    existing: pd.Series | None,
    new: pd.Series,
) -> pd.Series:
    if existing is None or existing.empty:
        merged = new
    else:
        merged = pd.concat([existing, new])
    merged = merged.astype(float)
    merged.index = pd.DatetimeIndex(pd.to_datetime(merged.index)).tz_localize(None)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def _align_close_series(
    symbols: list[str],
    closes_by_symbol: dict[str, pd.Series],
) -> dict[str, list[float]] | None:
    if any(symbol not in closes_by_symbol for symbol in symbols):
        return None
    close_frame = pd.concat(
        [closes_by_symbol[symbol].rename(symbol) for symbol in symbols],
        axis=1,
        join="inner",
    ).dropna(how="any")
    if len(close_frame) < 2:
        return None
    return {
        symbol: [float(value) for value in close_frame[symbol].to_numpy()]
        for symbol in symbols
    }
