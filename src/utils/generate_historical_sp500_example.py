from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.fetch_tickers import (
    SP500_HISTORICAL_COMPONENTS_URL,
    fetch_historical_sp500_tickers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a historical S&P 500 universe example."
    )
    parser.add_argument(
        "--date",
        required=True,
        help="Snapshot date, using the latest source row on or before this date.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to examples/sp500_YYYYMMDD_universe.json.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Refresh the cached historical components CSV.",
    )
    args = parser.parse_args()

    symbols = fetch_historical_sp500_tickers(
        args.date,
        refresh_cache=args.refresh_cache,
    )
    date_slug = args.date.replace("-", "")
    output = args.output or Path(f"examples/sp500_{date_slug}_universe.json")
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "source": SP500_HISTORICAL_COMPONENTS_URL,
        "index_code": "sp500",
        "snapshot": args.date,
        "symbols": symbols,
        "asset_classes": {
            symbol: f"{symbol} (sp500:{args.date})" for symbol in symbols
        },
    }
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"{output}: {len(symbols)} symbols")


if __name__ == "__main__":
    main()
