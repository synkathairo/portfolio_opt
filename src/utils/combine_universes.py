from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine existing universe JSON files into a new example universe."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Input universe JSON file. Can be repeated.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output universe JSON file.",
    )
    parser.add_argument(
        "--index-code",
        required=True,
        help="Logical index/universe code to write into the output metadata.",
    )
    parser.add_argument(
        "--snapshot",
        required=True,
        help="Snapshot label to write into the output metadata.",
    )
    parser.add_argument(
        "--component-index",
        action="append",
        default=[],
        help="Component index metadata entry. Can be repeated.",
    )
    parser.add_argument(
        "--benchmark",
        action="append",
        default=[],
        help=(
            "Benchmark symbol and label as SYMBOL=Label. Can be repeated, "
            "e.g. --benchmark '510300.SS=Huatai-PB CSI 300 ETF (benchmark_equity)'."
        ),
    )
    parser.add_argument(
        "--source",
        default="combined",
        help="Source metadata label.",
    )
    parser.add_argument(
        "--filtered-for-current-yfinance-data",
        action="store_true",
        help="Mark output as filtered for current Yahoo Finance availability.",
    )
    args = parser.parse_args()

    benchmark_asset_classes = _parse_symbol_labels(args.benchmark)
    symbols: list[str] = []
    asset_classes: dict[str, str] = {}
    for path in args.input:
        payload = json.loads(path.read_text())
        input_symbols = payload["symbols"]
        if not isinstance(input_symbols, list):
            raise ValueError(f"Input symbols must be a list: {path}")
        input_asset_classes = payload.get("asset_classes", {})
        if not isinstance(input_asset_classes, dict):
            input_asset_classes = {}
        for symbol in input_symbols:
            symbol = str(symbol)
            symbols.append(symbol)
            label = input_asset_classes.get(symbol)
            if isinstance(label, str):
                asset_classes.setdefault(symbol, label)

    symbols = _unique([*symbols, *benchmark_asset_classes])
    for symbol in symbols:
        asset_classes.setdefault(symbol, f"{symbol} ({args.index_code})")
    asset_classes.update(benchmark_asset_classes)

    output = {
        "source": args.source,
        "index_code": args.index_code,
        "component_indices": args.component_index,
        "snapshot": args.snapshot,
        "filtered_for_current_yfinance_data": args.filtered_for_current_yfinance_data,
        "includes_benchmark_etfs": bool(benchmark_asset_classes),
        "symbols": symbols,
        "asset_classes": {symbol: asset_classes[symbol] for symbol in symbols},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n")
    print(f"{args.output}: {len(symbols)} symbols")


def _parse_symbol_labels(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected benchmark as SYMBOL=Label, got: {value}")
        symbol, label = value.split("=", 1)
        symbol = symbol.strip()
        label = label.strip()
        if not symbol or not label:
            raise ValueError(f"Expected benchmark as SYMBOL=Label, got: {value}")
        parsed[symbol] = label
    return parsed


def _unique(values: list[Any]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


if __name__ == "__main__":
    main()
