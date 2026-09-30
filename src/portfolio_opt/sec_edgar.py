"""Cache-first access to SEC EDGAR company facts.

The SEC publishes these endpoints without an API key.  Values are retained
with their filing dates so research can use only information public at a
given point in time.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from .cache import read_cache, write_cache

_DATA_BASE_URL = "https://data.sec.gov"
_TICKER_URL = "https://www.sec.gov/files/company_tickers.json"
_DEFAULT_CACHE_DIR = Path(".cache/sec_edgar")
_DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = 0.12


@dataclass(frozen=True)
class FactObservation:
    """One reported XBRL fact, including the date it became public."""

    tag: str
    unit: str
    value: float
    start: date | None
    end: date
    filed: date
    form: str
    fiscal_year: int | None
    fiscal_period: str | None
    accession_number: str | None


def _parse_date(value: object, *, field: str) -> date:
    if not isinstance(value, str):
        raise TypeError(f"SEC fact {field} must be an ISO date.")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"SEC fact {field} is not an ISO date: {value!r}") from exc


def normalize_cik(cik: str | int) -> str:
    """Return the SEC's ten-digit CIK representation."""
    raw = str(cik).strip().upper()
    raw = raw.removeprefix("CIK")
    if not raw.isdigit():
        raise ValueError(f"Invalid CIK: {cik!r}")
    return raw.zfill(10)


def _cache_file(cache_dir: Path, name: str) -> Path:
    return cache_dir / f"{name}.json"


class SecEdgarClient:
    """Respectful EDGAR client with deterministic on-disk response caching."""

    def __init__(
        self,
        *,
        user_agent: str | None = None,
        cache_dir: Path = _DEFAULT_CACHE_DIR,
        min_request_interval_seconds: float = _DEFAULT_MIN_REQUEST_INTERVAL_SECONDS,
        timeout_seconds: float = 30.0,
        session: requests.Session | Any | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        load_dotenv()
        resolved_user_agent = user_agent or os.environ.get("SEC_USER_AGENT")
        if not resolved_user_agent:
            raise ValueError(
                "SEC_USER_AGENT is required. Set it to an identifying application "
                "name and contact address before accessing EDGAR."
            )
        self.user_agent = resolved_user_agent
        if min_request_interval_seconds < 0.1:
            raise ValueError(
                "min_request_interval_seconds must be at least 0.1 to stay at or "
                "below the SEC's 10 requests/second guidance."
            )
        self.cache_dir = Path(cache_dir)
        self.min_request_interval_seconds = min_request_interval_seconds
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()
        self._clock = clock
        self._sleep = sleep
        self._last_request_at: float | None = None

    def _fetch_json(self, url: str) -> dict[str, Any]:
        if self._last_request_at is not None:
            remaining = self.min_request_interval_seconds - (
                self._clock() - self._last_request_at
            )
            if remaining > 0:
                self._sleep(remaining)
        response = self.session.get(
            url,
            headers={"User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"},
            timeout=self.timeout_seconds,
        )
        self._last_request_at = self._clock()
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise TypeError(f"Unexpected non-object JSON response from {url}.")
        return payload

    def _cached_json(
        self,
        *,
        url: str,
        cache_name: str,
        use_cache: bool,
        refresh_cache: bool,
        offline: bool,
    ) -> dict[str, Any]:
        path = _cache_file(self.cache_dir, cache_name)
        if path.exists() and (use_cache or offline) and not refresh_cache:
            payload = read_cache(path)
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid JSON object cache: {path}")
            return payload
        if offline:
            raise FileNotFoundError(f"No cached SEC response available: {path}")
        payload = self._fetch_json(url)
        if use_cache or refresh_cache:
            write_cache(path, payload)
        return payload

    def ticker_ciks(
        self,
        *,
        use_cache: bool = True,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, str]:
        """Return the current SEC ticker-to-CIK map, keyed by upper-case ticker."""
        payload = self._cached_json(
            url=_TICKER_URL,
            cache_name="company_tickers",
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )
        result: dict[str, str] = {}
        for entry in payload.values():
            if not isinstance(entry, dict):
                continue
            ticker, cik = entry.get("ticker"), entry.get("cik_str")
            if isinstance(ticker, str) and isinstance(cik, (str, int)):
                result[ticker.upper()] = normalize_cik(cik)
        if not result:
            raise ValueError("SEC ticker response contained no ticker/CIK entries.")
        return result

    def company_facts(
        self,
        cik: str | int,
        *,
        use_cache: bool = True,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, Any]:
        """Return the companyfacts response for a CIK."""
        normalized = normalize_cik(cik)
        return self._cached_json(
            url=f"{_DATA_BASE_URL}/api/xbrl/companyfacts/CIK{normalized}.json",
            cache_name=f"companyfacts_CIK{normalized}",
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )

    def company_facts_for_symbol(
        self,
        symbol: str,
        *,
        use_cache: bool = True,
        refresh_cache: bool = False,
        offline: bool = False,
    ) -> dict[str, Any]:
        ciks = self.ticker_ciks(
            use_cache=use_cache, refresh_cache=refresh_cache, offline=offline
        )
        normalized_symbol = symbol.upper()
        if normalized_symbol not in ciks:
            raise KeyError(
                f"No current SEC CIK found for symbol {normalized_symbol!r}."
            )
        return self.company_facts(
            ciks[normalized_symbol],
            use_cache=use_cache,
            refresh_cache=refresh_cache,
            offline=offline,
        )


def iter_facts(
    payload: dict[str, Any],
    *,
    tag: str,
    unit: str = "USD",
    forms: Iterable[str] = ("10-K", "10-Q"),
    taxonomy: str = "us-gaap",
) -> list[FactObservation]:
    """Extract dated observations for a companyfacts taxonomy tag."""
    allowed_forms = set(forms)
    facts = payload.get("facts")
    namespace = facts.get(taxonomy) if isinstance(facts, dict) else None
    fact = namespace.get(tag) if isinstance(namespace, dict) else None
    units = fact.get("units") if isinstance(fact, dict) else None
    raw_observations = units.get(unit) if isinstance(units, dict) else None
    if not isinstance(raw_observations, list):
        return []

    result: list[FactObservation] = []
    for raw in raw_observations:
        if not isinstance(raw, dict) or raw.get("form") not in allowed_forms:
            continue
        value = raw.get("val")
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        try:
            end = _parse_date(raw.get("end"), field="end")
            filed = _parse_date(raw.get("filed"), field="filed")
            start_raw = raw.get("start")
            start = _parse_date(start_raw, field="start") if start_raw else None
        except ValueError:
            continue
        fiscal_year = raw.get("fy")
        result.append(
            FactObservation(
                tag=tag,
                unit=unit,
                value=float(value),
                start=start,
                end=end,
                filed=filed,
                form=str(raw["form"]),
                fiscal_year=fiscal_year if isinstance(fiscal_year, int) else None,
                fiscal_period=raw.get("fp") if isinstance(raw.get("fp"), str) else None,
                accession_number=(
                    raw.get("accn") if isinstance(raw.get("accn"), str) else None
                ),
            )
        )
    return sorted(
        result, key=lambda item: (item.filed, item.end, item.accession_number or "")
    )


def latest_facts_as_of(
    observations: Iterable[FactObservation], as_of: date
) -> dict[date, FactObservation]:
    """Return the latest public filing for each reporting-period end at ``as_of``."""
    result: dict[date, FactObservation] = {}
    for observation in observations:
        if observation.filed <= as_of:
            result[observation.end] = observation
    return result
