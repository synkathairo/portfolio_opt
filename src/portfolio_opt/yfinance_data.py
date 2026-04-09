"""Fetch historical daily closes from Yahoo Finance via yfinance.

Provides a drop-in replacement for the Alpaca daily-closes cache so the
custom backtest can reach back to ETF inception (e.g. the 2008 crisis).
"""

from __future__ import annotations

import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "yfinance is required for historical data. Install with `pip install yfinance`."
    ) from exc

logger = logging.getLogger(__name__)


def _to_yahoo_symbol(symbol: str) -> str:
    """Convert a standard ticker symbol to Yahoo Finance format.

    Yahoo Finance uses ``-`` instead of ``.`` for share class suffixes
    (e.g. ``BRK.B`` → ``BRK-B``, ``BF.B`` → ``BF-B``).
    """
    parts = symbol.split(".")
    if len(parts) == 2 and len(parts[1]) <= 2:
        return f"{parts[0]}-{parts[1]}"
    return symbol


def _fetch_single_symbol(
    symbol: str,
    period: str,
    retries: int,
    retry_delay: float,
) -> tuple[str, list[float]]:
    """Fetch daily adjusted closes for a single symbol from Yahoo Finance."""
    yahoo_sym = _to_yahoo_symbol(symbol)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(yahoo_sym)
            hist = ticker.history(period=period, auto_adjust=True)
            if hist.empty or "Close" not in hist.columns:
                raise ValueError(f"No close data for {symbol}")
            closes = [float(c) for c in hist["Close"].values]
            if not closes:
                raise ValueError(f"Empty close series for {symbol}")
            return symbol, closes
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                logger.debug("Retry %d/%d for %s: %s", attempt, retries, symbol, exc)
                time.sleep(retry_delay)
    raise RuntimeError(f"Failed to fetch {symbol} after {retries} retries: {last_err}")


def fetch_closes(
    symbols: list[str],
    period: str = "max",
    retries: int = 3,
    retry_delay: float = 1.0,
    max_workers: int = 10,
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

    Returns
    -------
    dict[str, list[float]]
        Mapping of symbol to a list of adjusted close prices, oldest first.
        All lists are aligned to the same trailing length (the minimum across
        symbols) so the backtest can use them directly.
    """
    if not symbols:
        return {}

    closes_by_symbol: dict[str, list[float]] = {}
    failed: list[str] = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_fetch_single_symbol, s, period, retries, retry_delay): s
            for s in symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                symbol, closes = future.result()
                closes_by_symbol[symbol] = closes
            except RuntimeError as exc:
                failed.append(symbol)
                logger.warning("Failed to fetch %s: %s", symbol, exc)

    if failed:
        raise RuntimeError(
            f"Failed to fetch {len(failed)}/{len(symbols)} symbols: "
            f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}"
        )

    # Clean NaN values: trailing NaN (incomplete today bar) is dropped from
    # all symbols; middle NaN values (historical data gaps) are interpolated
    # as the mean of the surrounding bars.
    if closes_by_symbol:
        # Check if any symbol has a trailing NaN
        any_trailing_nan = any(math.isnan(v[-1]) for v in closes_by_symbol.values())
        if any_trailing_nan:
            for s in closes_by_symbol:
                closes_by_symbol[s] = closes_by_symbol[s][:-1]

        # Interpolate any remaining middle NaN values
        for s in closes_by_symbol:
            prices = closes_by_symbol[s]
            for i in range(len(prices)):
                if math.isnan(prices[i]):
                    if i > 0 and i < len(prices) - 1:
                        prices[i] = (prices[i - 1] + prices[i + 1]) / 2.0
                    else:
                        # Edge NaN with no neighbor — should not happen for
                        # liquid symbols, but handle defensively
                        raise ValueError(
                            f"Unrecoverable NaN at edge of close series for {s}"
                        )

    # Align to common trailing history — different ETFs have different inception
    # dates, so we trim to the shortest series.
    lengths = {s: len(v) for s, v in closes_by_symbol.items()}
    min_len = min(lengths.values())
    if min_len < 2:
        raise ValueError(
            f"Not enough common history. Shortest: {min(lengths, key=lambda s: lengths[s])} "
            f"with {min_len} bars."
        )

    trimmed = {s: v[-min_len:] for s, v in closes_by_symbol.items()}
    return trimmed
