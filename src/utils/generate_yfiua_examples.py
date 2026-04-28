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

from portfolio_opt.yfinance_data import (
    _fetch_single_symbol_from,
    _yahoo_symbol_candidates,
)
from utils.fetch_tickers import (
    YFIUA_INDEX_STARTS,
    _format_ticker_dict,
    fetch_yfiua_index_constituents_with_names,
)

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
        "--year",
        type=int,
        default=None,
        help="Optional historical snapshot year override for all selected codes.",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Optional historical snapshot month override for all selected codes.",
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
    parser.add_argument(
        "--refresh-current-valid",
        action="store_true",
        help=(
            "Revalidate current-valid historical universes against yfinance. "
            "By default, existing current-valid files are reused to avoid rate limits."
        ),
    )
    parser.add_argument(
        "--asset-class-workers",
        type=int,
        default=3,
        help="Concurrent Yahoo metadata fetches for canonical sector labels.",
    )
    args = parser.parse_args()
    if (args.year is None) != (args.month is None):
        raise SystemExit("--year and --month must be provided together.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    today = cast(pd.Timestamp, pd.Timestamp(datetime.now()))
    start_check = cast(
        pd.Timestamp,
        today.normalize() - pd.Timedelta(days=args.current_valid_lookback_days),
    )

    for code in args.codes:
        if code not in YFIUA_INDEX_STARTS:
            raise SystemExit(f"Unsupported yfiua code: {code}")

        default_year, default_month = YFIUA_INDEX_STARTS[code]
        year = args.year if args.year is not None else default_year
        month = args.month if args.month is not None else default_month
        try:
            historical_symbols, historical_names = (
                fetch_yfiua_index_constituents_with_names(
                    code,
                    year=year,
                    month=month,
                )
            )
            current_symbols, current_names = fetch_yfiua_index_constituents_with_names(
                code
            )
        except requests.HTTPError as exc:
            print(f"{code}: skipped unavailable yfiua data ({exc})")
            continue
        current_valid_path = (
            args.output_dir
            / f"yfiua_{code}_{year:04d}{month:02d}_current_valid_universe.json"
        )
        current_valid_symbols = None
        if not args.refresh_current_valid and current_valid_path.exists():
            current_valid_symbols = _read_existing_symbols(current_valid_path)
        if current_valid_symbols is None:
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
            asset_classes=_canonical_asset_classes(
                historical_symbols,
                max_workers=args.asset_class_workers,
            ),
            snapshot=f"{year:04d}-{month:02d}",
            filtered_for_current_yfinance_data=False,
        )
        _write_universe(
            current_valid_path,
            code=code,
            symbols=current_valid_symbols,
            asset_classes=_canonical_asset_classes(
                current_valid_symbols,
                max_workers=args.asset_class_workers,
            ),
            snapshot=f"{year:04d}-{month:02d}",
            filtered_for_current_yfinance_data=True,
        )
        _write_universe(
            args.output_dir / f"yfiua_{code}_current_universe.json",
            code=code,
            symbols=current_symbols,
            asset_classes=_canonical_asset_classes(
                current_symbols,
                max_workers=args.asset_class_workers,
            ),
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


def _read_existing_symbols(path: Path) -> list[str]:
    payload = json.loads(path.read_text())
    symbols = payload.get("symbols")
    if not isinstance(symbols, list):
        raise ValueError(f"Existing universe symbols must be a list: {path}")
    return [str(symbol) for symbol in symbols]


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
    asset_classes: dict[str, str],
    snapshot: str,
    filtered_for_current_yfinance_data: bool,
) -> None:
    payload = {
        "source": "yfiua/index-constituents",
        "index_code": code,
        "snapshot": snapshot,
        "filtered_for_current_yfinance_data": filtered_for_current_yfinance_data,
        "symbols": symbols,
        "asset_classes": {symbol: asset_classes[symbol] for symbol in symbols},
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _canonical_asset_classes(symbols: list[str], *, max_workers: int) -> dict[str, str]:
    if max_workers < 1:
        raise SystemExit("--asset-class-workers must be positive.")
    formatted = _format_ticker_dict(symbols, max_workers=max_workers)
    asset_classes = formatted.get("asset_classes", {})
    if not isinstance(asset_classes, dict):
        raise ValueError("Canonical asset class formatting returned invalid payload.")
    return {str(symbol): str(asset_classes[symbol]) for symbol in symbols}


if __name__ == "__main__":
    main()
