import re
import time
import requests
import yfinance as yf
import pandas as pd
from io import StringIO
from datetime import datetime
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Optional, cast
from concurrent.futures import ThreadPoolExecutor, as_completed

from portfolio_opt.cache import cache_path, read_cache, write_cache

_TICKER_INFO_MEMORY_CACHE: dict[str, dict[str, Any]] = {}

YFIUA_INDEX_STARTS: dict[str, tuple[int, int]] = {
    "csi300": (2023, 7),
    "csi500": (2024, 2),
    "csi1000": (2024, 2),
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
NIKKEI225_BASKET = "nikkei225"
NIKKEI225_COMPONENTS_URL = (
    "https://indexes.nikkei.co.jp/en/nkave/index/component?idx=nk225"
)
SP500_HISTORICAL_COMPONENTS_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%2801-17-2026%29.csv"
)
_NON_YFIUA_BUILTIN_BASKETS = {
    "nasdaq100",
    "sp500",
    NIKKEI225_BASKET,
    "sectors",
    "indexes",
    "cashlike",
    "commodities",
    "realestate",
}
DEFAULT_TICKER_BASKET: tuple[str, ...] = (
    "nasdaq100",
    "sp500",
    "sectors",
    "indexes",
    "cashlike",
    "commodities",
    "realestate",
)


@dataclass(frozen=True)
class NikkeiConstituent:
    symbol: str
    name: str
    sector: str


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
            time.sleep(0.02)

    result = {"symbols": tickers, "asset_classes": asset_classes}
    return result


def fetch_nasdaq100_tickers(retries: int = 3, backoff: float = 1.5) -> list[str]:
    url = "https://api.nasdaq.com/api/quote/list-type/nasdaq100"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }

    for i in range(retries):
        r = requests.get(url, headers=headers, timeout=20)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, dict):
                outer_data = data.get("data")
                if isinstance(outer_data, dict):
                    inner_data = outer_data.get("data")
                    if isinstance(inner_data, dict):
                        rows = inner_data.get("rows")
                        if isinstance(rows, list):
                            return [
                                symbol
                                for row in rows
                                if isinstance(row, dict)
                                and isinstance(symbol := row.get("symbol"), str)
                                and symbol
                            ]
        elif r.status_code in (429, 500, 502, 503, 504):
            # retryable errors
            pass
        else:
            # permanent error, don't retry
            r.raise_for_status()
        if i + 1 < retries:
            time.sleep(backoff)
    return []


def fetch_sp500_tickers(retries: int = 3, backoff: float = 1.5) -> list[str]:
    # url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks"
    url = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/refs/heads/main/data/constituents.csv"
    df = pd.read_csv(url)
    symbols = df["Symbol"].tolist()
    return symbols


def fetch_historical_sp500_tickers(
    snapshot_date: str | datetime,
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[str]:
    """Fetch S&P 500 constituents for the latest available row on/before a date."""
    snapshot = cast(pd.Timestamp, pd.Timestamp(snapshot_date))
    if pd.isna(snapshot):
        raise ValueError(f"Invalid snapshot date: {snapshot_date!r}")
    snapshot = snapshot.normalize()
    path = cache_path(
        "sp500_historical_components",
        {"source": SP500_HISTORICAL_COMPONENTS_URL, "format": 1},
    )
    if use_cache and not refresh_cache and path.exists():
        payload = read_cache(path)
        if isinstance(payload, dict):
            return _historical_sp500_symbols_from_payload(payload, snapshot)

    response = requests.get(
        SP500_HISTORICAL_COMPONENTS_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    frame = pd.read_csv(StringIO(response.text))
    payload = {
        "source": SP500_HISTORICAL_COMPONENTS_URL,
        "rows": [
            {"date": str(row["date"])[:10], "tickers": str(row["tickers"])}
            for _, row in frame.iterrows()
        ],
    }
    if use_cache or refresh_cache:
        write_cache(path, payload)
    return _historical_sp500_symbols_from_payload(payload, snapshot)


def _historical_sp500_symbols_from_payload(
    payload: dict[str, Any],
    snapshot: pd.Timestamp,
) -> list[str]:
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Unexpected historical S&P 500 payload format.")

    selected: dict[str, Any] | None = None
    selected_date: pd.Timestamp | None = None
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_date = row.get("date")
        if raw_date is None:
            continue
        row_date = cast(pd.Timestamp, pd.Timestamp(str(raw_date)))
        if pd.isna(row_date):
            continue
        row_date = row_date.normalize()
        if row_date <= snapshot and (selected_date is None or row_date > selected_date):
            selected = row
            selected_date = row_date

    if selected is None:
        raise ValueError(f"No historical S&P 500 data on or before {snapshot.date()}.")
    tickers = selected.get("tickers")
    if not isinstance(tickers, str):
        raise ValueError("Unexpected historical S&P 500 ticker row format.")
    return [ticker.strip() for ticker in tickers.split(",") if ticker.strip()]


def fetch_yfiua_index_constituents(
    code: str,
    *,
    year: int | None = None,
    month: int | None = None,
) -> list[str]:
    payload = _fetch_yfiua_index_payload(code, year=year, month=month)
    return _extract_yfiua_symbols(payload)


def fetch_yfiua_index_constituent_names(
    code: str,
    *,
    year: int | None = None,
    month: int | None = None,
) -> dict[str, str]:
    payload = _fetch_yfiua_index_payload(code, year=year, month=month)
    return _extract_yfiua_symbol_names(payload)


def fetch_yfiua_index_constituents_with_names(
    code: str,
    *,
    year: int | None = None,
    month: int | None = None,
) -> tuple[list[str], dict[str, str]]:
    payload = _fetch_yfiua_index_payload(code, year=year, month=month)
    records = _extract_yfiua_symbol_records(payload)
    symbols = [symbol for symbol, _name in records]
    names = {symbol: name for symbol, name in records if name}
    return symbols, names


def _fetch_yfiua_index_payload(
    code: str,
    *,
    year: int | None = None,
    month: int | None = None,
) -> Any:
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
    return response.json()


def fetch_ftse_tickers() -> list[str]:
    return fetch_yfiua_index_constituents("ftse100")


def fetch_nikkei225_tickers(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[str]:
    constituents = fetch_nikkei225_constituents(
        use_cache=use_cache,
        refresh_cache=refresh_cache,
    )
    return [constituent.symbol for constituent in constituents]


def fetch_nikkei225_constituents(
    *,
    use_cache: bool = True,
    refresh_cache: bool = False,
) -> list[NikkeiConstituent]:
    path = cache_path(
        "nikkei225_constituents",
        {"source": NIKKEI225_COMPONENTS_URL, "format": 1},
    )
    if use_cache and not refresh_cache and path.exists():
        return _nikkei_constituents_from_cache(read_cache(path))

    response = requests.get(
        NIKKEI225_COMPONENTS_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
        timeout=20,
    )
    response.raise_for_status()
    if "idx-index-components" not in response.text:
        raise ValueError("Unexpected Nikkei 225 component page format.")

    constituents = _parse_nikkei225_component_html(response.text)
    if not constituents:
        raise ValueError("No Nikkei 225 constituents were found.")

    if use_cache:
        write_cache(
            path,
            {
                "source": NIKKEI225_COMPONENTS_URL,
                "symbols": [constituent.symbol for constituent in constituents],
                "constituents": [
                    {
                        "symbol": constituent.symbol,
                        "name": constituent.name,
                        "sector": constituent.sector,
                    }
                    for constituent in constituents
                ],
            },
        )
    return constituents


def _nikkei_constituents_from_cache(payload: Any) -> list[NikkeiConstituent]:
    if not isinstance(payload, dict):
        raise ValueError("Unexpected cached Nikkei 225 payload format.")
    records = payload.get("constituents")
    if not isinstance(records, list):
        raise ValueError("Unexpected cached Nikkei 225 constituents format.")
    constituents: list[NikkeiConstituent] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        symbol = record.get("symbol")
        name = record.get("name")
        sector = record.get("sector")
        if symbol is None or name is None or sector is None:
            continue
        constituents.append(
            NikkeiConstituent(
                symbol=str(symbol),
                name=str(name),
                sector=str(sector),
            )
        )
    return constituents


class _Nikkei225ComponentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.records: list[tuple[str, str, str]] = []
        self._in_component_div = False
        self._component_div_depth = 0
        self._in_heading = False
        self._in_row = False
        self._in_cell = False
        self._current_sector = ""
        self._current_row: list[str] = []
        self._cell_text: list[str] = []
        self._heading_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = dict(attrs)
        classes = set((attr_map.get("class") or "").split())
        if tag == "div" and "idx-index-components" in classes:
            self._in_component_div = True
            self._component_div_depth = 1
            return
        if self._in_component_div and tag == "div":
            self._component_div_depth += 1
        if not self._in_component_div:
            return
        if tag == "h3":
            self._in_heading = True
            self._heading_text = []
        elif tag == "tr":
            self._in_row = True
            self._current_row = []
        elif tag == "td" and self._in_row:
            self._in_cell = True
            self._cell_text = []

    def handle_data(self, data: str) -> None:
        if self._in_heading:
            self._heading_text.append(data)
        if self._in_cell:
            self._cell_text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if self._in_heading and tag == "h3":
            self._current_sector = " ".join("".join(self._heading_text).split())
            self._in_heading = False
        elif self._in_cell and tag == "td":
            self._current_row.append(" ".join("".join(self._cell_text).split()))
            self._in_cell = False
        elif self._in_row and tag == "tr":
            if len(self._current_row) >= 2 and re.fullmatch(
                r"[0-9A-Z]{4}",
                self._current_row[0],
            ):
                self.records.append(
                    (
                        self._current_row[0],
                        self._current_row[1],
                        self._current_sector,
                    )
                )
            self._in_row = False
        if self._in_component_div and tag == "div":
            self._component_div_depth -= 1
            if self._component_div_depth == 0:
                self._in_component_div = False


def _parse_nikkei225_component_html(html: str) -> list[NikkeiConstituent]:
    parser = _Nikkei225ComponentParser()
    parser.feed(html)
    seen: set[str] = set()
    constituents: list[NikkeiConstituent] = []
    for code, name, sector in parser.records:
        symbol = f"{code}.T"
        if symbol in seen:
            continue
        seen.add(symbol)
        constituents.append(
            NikkeiConstituent(
                symbol=symbol,
                name=name,
                sector=sector or "Unknown",
            )
        )
    return constituents


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
    return [symbol for symbol, _name in _extract_yfiua_symbol_records(payload)]


def _extract_yfiua_symbol_names(payload: Any) -> dict[str, str]:
    return {
        symbol: name for symbol, name in _extract_yfiua_symbol_records(payload) if name
    }


def _extract_yfiua_symbol_records(payload: Any) -> list[tuple[str, str | None]]:
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

    records: list[tuple[str, str | None]] = []
    for item in values:
        name: str | None = None
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
            raw_name = item.get("name") or item.get("Name") or item.get("security")
            if raw_name is not None:
                name = str(raw_name).strip() or None
        else:
            continue
        symbol = symbol.strip()
        if symbol:
            records.append((_normalize_yfiua_symbol(symbol), name))
    return records


def _normalize_yfiua_symbol(symbol: str) -> str:
    symbol = symbol.replace("/.", ".")
    symbol = re.sub(r"/([A-Z]{2,3})$", r".\1", symbol)
    return symbol.replace("/", "-")


def _require_tickers(source: str, tickers: list[str]) -> list[str]:
    if not tickers:
        raise RuntimeError(f"No tickers were fetched for {source}.")
    return tickers


# a basket of tickers as needed, customizable from ticker_selection
# notes: this can be slow with all options! but results perhaps can be cached,
# nasdaq100 and sp500 are updated on fixed basis
def fetch_ticker_dict(
    preexisting: list[str] | None = None,
    ticker_basket: list[str] | tuple[str, ...] | None = None,
) -> dict:
    if preexisting is None:
        preexisting = []
    if ticker_basket is None:
        ticker_basket = list(DEFAULT_TICKER_BASKET)
    else:
        ticker_basket = list(ticker_basket)
    tickers = set()
    if "nasdaq100" in ticker_basket:
        tickers |= set(_require_tickers("nasdaq100", fetch_nasdaq100_tickers()))
    if "sp500" in ticker_basket:
        tickers |= set(_require_tickers("sp500", fetch_sp500_tickers()))
    if NIKKEI225_BASKET in ticker_basket:
        tickers |= set(
            _require_tickers(NIKKEI225_BASKET, fetch_nikkei225_tickers())
        )
    for code in _yfiua_codes_from_basket(ticker_basket):
        tickers |= set(
            _require_tickers(
                f"{YFIUA_BASKET_PREFIX}{code}",
                fetch_yfiua_index_constituents(code),
            )
        )
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
