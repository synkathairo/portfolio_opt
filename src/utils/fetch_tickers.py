import re
import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from portfolio_opt.cache import cache_path, read_cache, write_cache

_TICKER_INFO_MEMORY_CACHE: dict[str, dict[str, Any]] = {}

YFIUA_INDEX_STARTS: dict[str, tuple[int, int]] = {
    "csi300": (2023, 7),
    "csi500": (2024, 1),
    "csi1000": (2024, 1),
    "sse": (2023, 7),
    "szse": (2023, 7),
    "nasdaq100": (2023, 7),
    "sp500": (2023, 7),
    "dowjones": (2023, 7),
    "dax": (2023, 7),
    "hsi": (2023, 7),
    "ftse100": (2023, 7),
    "ibex35": (2024, 3),
    "ftsemib": (2024, 3),
    "nifty50": (2024, 3),
    "asx200": (2024, 3),
}

YFIUA_BASKET_PREFIX = "yfiua:"
_NON_YFIUA_BUILTIN_BASKETS = {
    "nasdaq100",
    "sp500",
    "sectors",
    "indexes",
    "cashlike",
    "commodities",
    "realestate",
}


# present a json-like dict object(?)

# alternatively, if json provided, pipe into a dict.
# these should match in layout


def _get_ticker_info_payload(ticker: str) -> dict[str, Any]:
    if ticker in _TICKER_INFO_MEMORY_CACHE:
        return _TICKER_INFO_MEMORY_CACHE[ticker]
    path = cache_path("ticker_info", {"symbol": ticker})
    if path.exists():
        cached = read_cache(path)
        if isinstance(cached, dict):
            _TICKER_INFO_MEMORY_CACHE[ticker] = cached
            return cached
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        _TICKER_INFO_MEMORY_CACHE[ticker] = {}
        return {}
    if isinstance(info, dict) and info:
        write_cache(path, info)
        _TICKER_INFO_MEMORY_CACHE[ticker] = info
        return info
    _TICKER_INFO_MEMORY_CACHE[ticker] = {}
    return {}


def _get_ticker_info(ticker: str) -> tuple[str, str]:
    """Fetch info for a single ticker. Returns (symbol, formatted_name)."""
    try:
        info = _get_ticker_info_payload(ticker)
        name = info.get("shortName") or info.get("longName") or ticker
        sector = info.get("sector") or "Unknown"
        return ticker, f"{name} ({sector})"
    except Exception:
        return ticker, f"{ticker} (Unknown)"


def _format_ticker_dict(tickers: list[str], max_workers: int = 3) -> dict:
    asset_classes: dict[str, str] = {}

    # Process in small batches to avoid Yahoo rate limiting
    batch_size = max_workers * 2
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_get_ticker_info, t): t for t in batch}
            results: dict[str, tuple[str, str]] = {}
            for future in as_completed(futures):
                ticker = futures[future]
                results[ticker] = future.result()

        # Reconstruct in original order
        for ticker in batch:
            asset_classes[ticker] = results[ticker][1]

        # Small delay between batches to avoid Yahoo rate limits
        if i + batch_size < len(tickers):
            time.sleep(1.0)

    result = {"symbols": tickers, "asset_classes": asset_classes}
    return result


def fetch_nasdaq100_tickers(retries: int = 3, backoff: float = 1.5) -> list[str]:
    url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    for i in range(retries):
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            # success
            data = r.json()
            return [row["symbol"] for row in data["data"]["data"]["rows"]]
        elif r.status_code in (429, 500, 502, 503, 504):
            # retryable errors
            pass
        else:
            # permanent error, don't retry
            r.raise_for_status()
    return []


def fetch_sp500_tickers(retries: int = 3, backoff: float = 1.5) -> list[str]:
    # url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks"
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/refs/heads/main/data/constituents.csv"
    df = pd.read_csv(url)
    symbols = df["Symbol"].tolist()
    return symbols


def fetch_yfiua_index_constituents(
    code: str,
    *,
    year: int | None = None,
    month: int | None = None,
) -> list[str]:
    if (year is None) != (month is None):
        raise ValueError("year and month must be provided together.")
    if year is None and month is None:
        url = f"https://yfiua.github.io/index-constituents/constituents-{code}.json"
    else:
        url = (
            "https://yfiua.github.io/index-constituents/"
            f"{year:04d}/{month:02d}/constituents-{code}.json"
        )
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()
    payload = response.json()
    return _extract_yfiua_symbols(payload)


def fetch_ftse_tickers() -> list[str]:
    return fetch_yfiua_index_constituents("ftse100")


def _yfiua_codes_from_basket(ticker_basket: list[str]) -> list[str]:
    codes: list[str] = []
    for basket in ticker_basket:
        if basket.startswith(YFIUA_BASKET_PREFIX):
            code = basket.removeprefix(YFIUA_BASKET_PREFIX)
        elif basket in YFIUA_INDEX_STARTS and basket not in _NON_YFIUA_BUILTIN_BASKETS:
            code = basket
        else:
            continue
        if code not in YFIUA_INDEX_STARTS:
            raise ValueError(
                f"Unsupported yfiua index code {code!r}. "
                f"Supported codes: {', '.join(sorted(YFIUA_INDEX_STARTS))}"
            )
        codes.append(code)
    return codes


def _extract_yfiua_symbols(payload: Any) -> list[str]:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict):
        values = (
            payload.get("symbols")
            or payload.get("constituents")
            or payload.get("data")
            or payload.get("tickers")
        )
    else:
        values = None

    if not isinstance(values, list):
        raise ValueError("Unexpected index constituent payload format.")

    symbols: list[str] = []
    for item in values:
        if isinstance(item, str):
            symbol = item
        elif isinstance(item, dict):
            raw_symbol = (
                item.get("symbol")
                or item.get("ticker")
                or item.get("Symbol")
                or item.get("Ticker")
            )
            if raw_symbol is None:
                continue
            symbol = str(raw_symbol)
        else:
            continue
        symbol = symbol.strip()
        if symbol:
            symbols.append(_normalize_yfiua_symbol(symbol))
    return symbols


def _normalize_yfiua_symbol(symbol: str) -> str:
    symbol = symbol.replace("/.", ".")
    symbol = re.sub(r"/([A-Z]{2,3})$", r".\1", symbol)
    return symbol.replace("/", "-")


# a basket of tickers as needed, customizable from ticker_selection
# notes: this can be slow with all options! but results perhaps can be cached,
# nasdaq100 and sp500 are updated on fixed basis
def fetch_ticker_dict(
    preexisting: list[str] = [],
    ticker_basket: list[str] = [
        "nasdaq100",
        "sp500",
        "sectors",
        "indexes",
        "cashlike",
        "commodities",
        "realestate",
    ],
) -> dict:
    tickers = set()
    if "nasdaq100" in ticker_basket:
        tickers |= set(fetch_nasdaq100_tickers())
    if "sp500" in ticker_basket:
        tickers |= set(fetch_sp500_tickers())
    for code in _yfiua_codes_from_basket(ticker_basket):
        tickers |= set(fetch_yfiua_index_constituents(code))
    # note: below were statically coded, so need to be mindful of when the ETFs were created when querying
    if "sectors" in ticker_basket:
        tickers |= {"XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU"}
    if "indexes" in ticker_basket:
        tickers |= {"SPY", "QQQ", "IWM"}
    if "cashlike" in ticker_basket:
        tickers |= {"SGOV", "IEF", "TLT", "TIP"}
    if "commodities" in ticker_basket:
        tickers |= {"GLD", "SLV", "DBC"}
    if "realestate" in ticker_basket:
        tickers |= {"VNQ"}
    tickers |= set(preexisting)
    ticker_list = list(tickers)
    ticker_list.sort()
    return _format_ticker_dict(ticker_list)


def get_ticker_firstTradeDate(symbol: str) -> Optional[datetime]:
    try:
        info = _get_ticker_info_payload(symbol)

        # This field provides the first trade date in Unix timestamp (Epoch)
        epoch_time = info.get("firstTradeDateMilliseconds")

        if epoch_time:
            # Convert Unix timestamp to a readable date
            listing_date = datetime.strptime(
                datetime.fromtimestamp(epoch_time / 1000).strftime("%Y-%m-%d"),
                "%Y-%m-%d",
            )
            # tickers_comb_dict[symbol] = listing_date
            # print(symbol, listing_date)
            return listing_date
        else:
            return None

    except Exception as e:
        print(f"{symbol:<8} | Error: {e}")
        return None


def filter_tickers_before(
    tickers: list[str], date: datetime, max_workers: int = 3
) -> list[str]:
    results: dict[str, Optional[datetime]] = {}

    # Process in small batches to avoid Yahoo rate limiting
    batch_size = max_workers * 2
    for i in range(0, len(tickers), batch_size):
        batch = tickers[i : i + batch_size]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(get_ticker_firstTradeDate, t): t for t in batch}
            for future in as_completed(futures):
                ticker = futures[future]
                results[ticker] = future.result()

        # Small delay between batches to avoid Yahoo rate limits
        if i + batch_size < len(tickers):
            time.sleep(1.0)

    return [
        ticker
        for ticker in tickers
        if (trade_date := results.get(ticker)) is not None and trade_date < date
    ]


# def get_tickers_from_ticker_json(dict)
