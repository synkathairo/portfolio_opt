import requests
import yfinance as yf
import pandas as pd

# present a json-like dict object(?)

# alternatively, if json provided, pipe into a dict.
# these should match in layout


def _format_ticker_dict(tickers: list[str]) -> dict:
    symbols = tickers
    asset_classes = {}

    for t in tickers:
        try:
            info = yf.Ticker(t).info or {}
            name = info.get("shortName") or info.get("longName") or t
            sector = info.get("sector") or "Unknown"
            asset_classes[t] = f"{name} ({sector})"
        except Exception:
            asset_classes[t] = f"{t} (Unknown)"

    result = {"symbols": symbols, "asset_classes": asset_classes}
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
    ticker_selection: list[str] = [
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
    if "nasdaq100" in ticker_selection:
        tickers |= set(fetch_nasdaq100_tickers())
    if "sp500" in ticker_selection:
        tickers |= set(fetch_sp500_tickers())
    # note: below were statically coded, so need to be mindful of when the ETFs were created when querying
    if "sectors" in ticker_selection:
        tickers |= {"XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLU"}
    if "indexes" in ticker_selection:
        tickers |= {"SPY", "QQQ", "IWM"}
    if "cashlike" in ticker_selection:
        tickers |= {"SGOV", "IEF", "TLT", "TIP"}
    if "commodities" in ticker_selection:
        tickers |= {"GLD", "SLV", "DBC"}
    if "realestate" in ticker_selection:
        tickers |= {"VNQ"}
    tickers |= set(preexisting)
    ticker_list = list(tickers)
    ticker_list.sort()
    return _format_ticker_dict(ticker_list)
