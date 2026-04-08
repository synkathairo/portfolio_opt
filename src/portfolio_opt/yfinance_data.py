"""Fetch historical daily closes from Yahoo Finance via yfinance.

Provides a drop-in replacement for the Alpaca daily-closes cache so the
custom backtest can reach back to ETF inception (e.g. the 2008 crisis).
"""

from __future__ import annotations

import logging
import time

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


def fetch_closes(
    symbols: list[str],
    period: str = "max",
    retries: int = 3,
    retry_delay: float = 1.0,
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

    for symbol in symbols:
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
                closes_by_symbol[symbol] = closes
                break
            except Exception as exc:
                last_err = exc
                if attempt < retries:
                    logger.debug(
                        "Retry %d/%d for %s: %s", attempt, retries, symbol, exc
                    )
                    time.sleep(retry_delay)
        else:
            failed.append(symbol)
            logger.warning("Failed to fetch %s after %d retries: %s", symbol, retries, last_err)

    if failed:
        raise RuntimeError(
            f"Failed to fetch {len(failed)}/{len(symbols)} symbols: "
            f"{', '.join(failed[:10])}{'...' if len(failed) > 10 else ''}"
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
