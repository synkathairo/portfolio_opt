import time
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed


# present a json-like dict object(?)

# alternatively, if json provided, pipe into a dict.
# these should match in layout


def _get_ticker_info(ticker: str) -> tuple[str, str]:
    """Fetch info for a single ticker. Returns (symbol, formatted_name)."""
    try:
        info = yf.Ticker(ticker).info or {}
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
        ticker_obj = yf.Ticker(symbol)

        # Fetch the metadata (one quick request)
        info: dict = ticker_obj.info

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
