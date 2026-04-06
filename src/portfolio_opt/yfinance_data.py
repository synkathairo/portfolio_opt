"""Fetch historical daily closes from Yahoo Finance via yfinance.

Provides a drop-in replacement for the Alpaca daily-closes cache so the
custom backtest can reach back to ETF inception (e.g. the 2008 crisis).
"""

from __future__ import annotations

import logging

import pandas as pd

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "yfinance is required for historical data. Install with `pip install yfinance`."
    ) from exc

logger = logging.getLogger(__name__)


def fetch_closes(
    symbols: list[str],
    period: str = "max",
) -> dict[str, list[float]]:
    """Download daily adjusted closes for *symbols* from Yahoo Finance.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols (e.g. ``["SPY", "QQQ", "GLD"]``).
    period : str
        yfinance period string — ``"max"``, ``"10y"``, ``"5y"``, etc.

    Returns
    -------
    dict[str, list[float]]
        Mapping of symbol to a list of adjusted close prices, oldest first.
        All lists are aligned to the same trailing length (the minimum across
        symbols) so the backtest can use them directly.
    """
    if not symbols:
        return {}

    # yfinance can fetch multiple tickers in one call, returning a DataFrame
    # with a MultiIndex column (symbol, price_field).  We only need "Adj Close".
    tickers = yf.Tickers(" ".join(symbols))
    closes_by_symbol: dict[str, list[float]] = {}

    for symbol in symbols:
        try:
            ticker = tickers.tickers.get(symbol) or tickers.tickers.get(symbol.upper())
            if ticker is None:
                raise ValueError(f"yfinance returned no data for {symbol}")
            hist = ticker.history(period=period, auto_adjust=True)
            if hist.empty or "Close" not in hist.columns:
                raise ValueError(f"No close data for {symbol}")
            closes = [float(c) for c in hist["Close"].values]
            if not closes:
                raise ValueError(f"Empty close series for {symbol}")
            closes_by_symbol[symbol] = closes
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch {symbol} from yfinance: {exc}") from exc

    # Align to common trailing history — different ETFs have different inception
    # dates, so we trim to the shortest series.
    lengths = {s: len(v) for s, v in closes_by_symbol.items()}
    min_len = min(lengths.values())
    if min_len < 2:
        raise ValueError(
            f"Not enough common history. Shortest: {min(lengths, key=lengths.get)} "
            f"with {min_len} bars."
        )

    trimmed = {s: v[-min_len:] for s, v in closes_by_symbol.items()}
    return trimmed
