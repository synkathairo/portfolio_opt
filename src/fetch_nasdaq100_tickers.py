import requests
import yfinance as yf

# present a json-like dict object(?)

# alternatively, if json provided, pipe into a dict.
# these should match in layout


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
            # permanent error → don't retry
            r.raise_for_status()
    return []


def fetch_nasdaq100_ticker_dict(preexisting: list[str] = []) -> dict:

    tickers: list[str] = fetch_nasdaq100_tickers()
    tickers += [entry for entry in preexisting if entry not in tickers]

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
