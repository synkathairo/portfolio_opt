from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from portfolio_opt.yfinance_data import _fetch_single_symbol
from utils.fetch_tickers import NikkeiConstituent, fetch_nikkei225_constituents


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Nikkei 225 universe examples from the official Nikkei page."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("examples"),
        help="Directory where generated universe JSON files should be written.",
    )
    parser.add_argument(
        "--refresh-components",
        action="store_true",
        help="Refetch the official Nikkei component page instead of reusing cache.",
    )
    parser.add_argument(
        "--skip-backtest-valid",
        action="store_true",
        help="Only write the current component universe.",
    )
    parser.add_argument(
        "--min-history-prices",
        type=int,
        default=661,
        help=(
            "Minimum adjusted-close count required for the backtest-valid universe. "
            "The default supports 252 lookback days plus roughly 408 backtest days."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent yfinance history checks for the backtest-valid universe.",
    )
    parser.add_argument(
        "--symbol-delay",
        type=float,
        default=0.02,
        help="Seconds to wait between yfinance submissions.",
    )
    args = parser.parse_args()
    if args.min_history_prices < 1:
        raise SystemExit("--min-history-prices must be positive.")
    if args.max_workers < 1:
        raise SystemExit("--max-workers must be positive.")
    if args.symbol_delay < 0:
        raise SystemExit("--symbol-delay must be non-negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    constituents = fetch_nikkei225_constituents(
        use_cache=True,
        refresh_cache=args.refresh_components,
    )
    _write_universe(
        args.output_dir / "nikkei225_current_universe.json",
        constituents=constituents,
        backtest_valid=False,
        min_history_prices=None,
    )

    if args.skip_backtest_valid:
        print(f"nikkei225: current={len(constituents)}")
        return

    valid_symbols = _filter_min_history_symbols(
        [constituent.symbol for constituent in constituents],
        min_history_prices=args.min_history_prices,
        max_workers=args.max_workers,
        symbol_delay=args.symbol_delay,
    )
    valid_constituents = [
        constituent
        for constituent in constituents
        if constituent.symbol in valid_symbols
    ]
    _write_universe(
        args.output_dir / "nikkei225_current_backtest_valid_universe.json",
        constituents=valid_constituents,
        backtest_valid=True,
        min_history_prices=args.min_history_prices,
    )
    print(
        f"nikkei225: current={len(constituents)} "
        f"backtest-valid={len(valid_constituents)} "
        f"min-history-prices={args.min_history_prices}"
    )


def _filter_min_history_symbols(
    symbols: list[str],
    *,
    min_history_prices: int,
    max_workers: int,
    symbol_delay: float,
) -> set[str]:
    valid: set[str] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for symbol in symbols:
            futures[
                executor.submit(
                    _has_min_history,
                    symbol,
                    min_history_prices,
                )
            ] = symbol
            if symbol_delay:
                time.sleep(symbol_delay)
        for future in as_completed(futures):
            symbol = futures[future]
            if future.result():
                valid.add(symbol)
    return valid


def _has_min_history(symbol: str, min_history_prices: int) -> bool:
    try:
        _symbol, closes = _fetch_single_symbol(
            symbol,
            period="max",
            retries=2,
            retry_delay=0.5,
        )
    except RuntimeError:
        return False
    return len(closes) >= min_history_prices


def _write_universe(
    path: Path,
    *,
    constituents: list[NikkeiConstituent],
    backtest_valid: bool,
    min_history_prices: int | None,
) -> None:
    payload = {
        "source": "nikkei/indexes.nikkei.co.jp",
        "index_code": "nikkei225",
        "snapshot": "current",
        "filtered_for_backtest_history": backtest_valid,
        "min_history_prices": min_history_prices,
        "symbols": [constituent.symbol for constituent in constituents],
        "asset_classes": {
            constituent.symbol: f"{constituent.name} (nikkei225:{constituent.sector})"
            for constituent in constituents
        },
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
