from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import requests
import yfinance as yf

from portfolio_opt.yfinance_data import _fetch_single_symbol_from, _yahoo_symbol_candidates
from utils.fetch_tickers import YFIUA_INDEX_STARTS, fetch_yfiua_index_constituents

YFIUA_EXCHANGE_SUFFIXES = {"AX", "DE", "HK", "L", "MC", "MI", "NS", "SS", "SZ"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate yfiua index universe examples."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples"),
        help="Directory where generated universe JSON files should be written.",
    )
    parser.add_argument(
        "--codes",
        nargs="*",
        default=sorted(YFIUA_INDEX_STARTS),
        help="Subset of yfiua index codes to generate.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=12,
        help="Concurrent fallback yfinance checks for current-valid historical universes.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Number of yfinance candidate symbols to validate per batch.",
    )
    parser.add_argument(
        "--current-valid-lookback-days",
        type=int,
        default=14,
        help="Calendar-day window used to decide whether yfinance still has a current quote.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    today = cast(pd.Timestamp, pd.Timestamp(datetime.now()))
    start_check = cast(
        pd.Timestamp,
        today.normalize() - pd.Timedelta(days=args.current_valid_lookback_days),
    )

    for code in args.codes:
        if code not in YFIUA_INDEX_STARTS:
            raise SystemExit(f"Unsupported yfiua code: {code}")

        year, month = YFIUA_INDEX_STARTS[code]
        try:
            historical_symbols = fetch_yfiua_index_constituents(
                code,
                year=year,
                month=month,
            )
            current_symbols = fetch_yfiua_index_constituents(code)
        except requests.HTTPError as exc:
            print(f"{code}: skipped unavailable yfiua data ({exc})")
            continue
        current_valid_symbols = _filter_current_valid_symbols(
            historical_symbols,
            start=start_check,
            max_workers=args.max_workers,
            batch_size=args.batch_size,
        )

        _write_universe(
            args.output_dir / f"yfiua_{code}_{year:04d}{month:02d}_universe.json",
            code=code,
            symbols=historical_symbols,
            snapshot=f"{year:04d}-{month:02d}",
            filtered_for_current_yfinance_data=False,
        )
        _write_universe(
            args.output_dir
            / f"yfiua_{code}_{year:04d}{month:02d}_current_valid_universe.json",
            code=code,
            symbols=current_valid_symbols,
            snapshot=f"{year:04d}-{month:02d}",
            filtered_for_current_yfinance_data=True,
        )
        _write_universe(
            args.output_dir / f"yfiua_{code}_current_universe.json",
            code=code,
            symbols=current_symbols,
            snapshot="current",
            filtered_for_current_yfinance_data=False,
        )
        print(
            f"{code}: historical={len(historical_symbols)} "
            f"current-valid={len(current_valid_symbols)} current={len(current_symbols)}"
        )


def _filter_current_valid_symbols(
    symbols: list[str],
    *,
    start: pd.Timestamp,
    max_workers: int,
    batch_size: int,
) -> list[str]:
    candidate_to_symbol: dict[str, str] = {}
    for symbol in symbols:
        for candidate in _yfiua_yahoo_validation_candidates(symbol):
            candidate_to_symbol.setdefault(candidate, symbol)

    valid: set[str] = set()
    candidates = list(candidate_to_symbol)
    failed_candidates: list[str] = []
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        try:
            valid_candidates = _valid_yfinance_candidates(batch, start)
        except Exception:
            failed_candidates.extend(batch)
            continue
        for candidate in valid_candidates:
            valid.add(candidate_to_symbol[candidate])

    failed_symbols = sorted(
        {candidate_to_symbol[candidate] for candidate in failed_candidates}
    )
    if not failed_symbols:
        return [symbol for symbol in symbols if symbol in valid]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_has_current_yfinance_close, symbol, start): symbol
            for symbol in failed_symbols
        }
        for future in as_completed(futures):
            symbol = futures[future]
            if future.result():
                valid.add(symbol)
    return [symbol for symbol in symbols if symbol in valid]


def _yfiua_yahoo_validation_candidates(symbol: str) -> list[str]:
    match = re.search(r"\.([A-Z]{1,3})$", symbol)
    if match and match.group(1) in YFIUA_EXCHANGE_SUFFIXES:
        return [symbol]
    return _yahoo_symbol_candidates(symbol)


def _valid_yfinance_candidates(
    candidates: list[str],
    start: pd.Timestamp,
) -> set[str]:
    if not candidates:
        return set()
    data = yf.download(
        candidates,
        start=start.date().isoformat(),
        auto_adjust=True,
        group_by="ticker",
        progress=False,
        threads=True,
        timeout=15,
    )
    valid: set[str] = set()
    if data.empty:
        return valid
    if isinstance(data.columns, pd.MultiIndex):
        for candidate in candidates:
            if candidate not in data.columns.get_level_values(0):
                continue
            symbol_frame = data[candidate]
            if "Close" in symbol_frame and not symbol_frame["Close"].dropna().empty:
                valid.add(candidate)
        return valid
    if len(candidates) == 1 and "Close" in data and not data["Close"].dropna().empty:
        valid.add(candidates[0])
    return valid


def _has_current_yfinance_close(symbol: str, start: pd.Timestamp) -> bool:
    try:
        _, closes = _fetch_single_symbol_from(
            symbol,
            start=start,
            retries=1,
            retry_delay=0.0,
        )
    except RuntimeError:
        return False
    return not closes.empty


def _write_universe(
    path: Path,
    *,
    code: str,
    symbols: list[str],
    snapshot: str,
    filtered_for_current_yfinance_data: bool,
) -> None:
    payload = {
        "source": "yfiua/index-constituents",
        "index_code": code,
        "snapshot": snapshot,
        "filtered_for_current_yfinance_data": filtered_for_current_yfinance_data,
        "symbols": symbols,
        "asset_classes": {
            symbol: f"{symbol} (yfiua:{code})" for symbol in symbols
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
