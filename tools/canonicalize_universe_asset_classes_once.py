from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from portfolio_opt.cache import cache_path, read_cache
from portfolio_opt.model import load_model_inputs
from portfolio_opt.yfinance_data import _yahoo_symbol_candidates
from utils.fetch_tickers import (
    CASHLIKE_ASSET_CLASSES,
    COMMODITY_ASSET_CLASSES,
    INDEX_ASSET_CLASSES,
    REALESTATE_ASSET_CLASSES,
    SECTOR_ASSET_CLASSES,
    YAHOO_SECTOR_ASSET_CLASSES,
)

STATIC_OVERRIDES = {
    **SECTOR_ASSET_CLASSES,
    **INDEX_ASSET_CLASSES,
    **CASHLIKE_ASSET_CLASSES,
    **COMMODITY_ASSET_CLASSES,
    **REALESTATE_ASSET_CLASSES,
}
LEGACY_RENAMES = {"real_estate": "sector_real_estate"}
PAREN_LABEL_RE = re.compile(r"\(([^()]*)\)\s*$")


def is_unknown_label(label: str) -> bool:
    return label == "sector_unknown" or "(Unknown)" in label


def cached_asset_class(symbol: str) -> str | None:
    for candidate in _yahoo_symbol_candidates(symbol):
        path = cache_path("ticker_info", {"symbol": candidate})
        if not path.exists():
            continue
        cached = read_cache(path)
        if not isinstance(cached, dict):
            continue
        asset_class = cached.get("asset_class")
        if isinstance(asset_class, str) and asset_class != "sector_unknown":
            return asset_class
    return None


def canonical_asset_class(symbol: str, label: str) -> str:
    if symbol in STATIC_OVERRIDES:
        return STATIC_OVERRIDES[symbol]
    if label in LEGACY_RENAMES:
        return LEGACY_RENAMES[label]
    if label in YAHOO_SECTOR_ASSET_CLASSES:
        return YAHOO_SECTOR_ASSET_CLASSES[label]
    if is_unknown_label(label):
        cached = cached_asset_class(symbol)
        if cached is not None:
            return cached

    match = PAREN_LABEL_RE.search(label)
    if match is None:
        return label
    sector = match.group(1).strip()
    return YAHOO_SECTOR_ASSET_CLASSES.get(sector, label)


def canonicalize(payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    symbols = [str(symbol) for symbol in payload.get("symbols", [])]
    raw_asset_classes = payload.get("asset_classes", {})
    if not isinstance(raw_asset_classes, dict):
        return payload, []

    before_classes = {str(k): str(v) for k, v in raw_asset_classes.items()}
    after_classes = dict(before_classes)
    changes: list[str] = []

    for symbol in symbols:
        before = after_classes.get(symbol)
        if before is None:
            continue
        after = canonical_asset_class(symbol, before)
        if after != before:
            after_classes[symbol] = after
            changes.append(f"{symbol}: {before!r} -> {after!r}")

    output = dict(payload)
    output["asset_classes"] = {
        symbol: after_classes[symbol]
        for symbol in before_classes
        if symbol in after_classes
    }

    for section in ("class_min_weights", "class_max_weights"):
        constraints = output.get(section)
        if not isinstance(constraints, dict):
            continue
        output[section] = {
            LEGACY_RENAMES.get(str(name), str(name)): value
            for name, value in constraints.items()
        }

    constrained: set[str] = set()
    for section in ("class_min_weights", "class_max_weights"):
        constraints = output.get(section)
        if isinstance(constraints, dict):
            constrained.update(str(name) for name in constraints)
    unresolved = sorted(constrained - set(after_classes.values()))
    if unresolved:
        changes.append(f"unresolved constraints after migration: {unresolved}")

    return output, changes


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    try:
        load_model_inputs(tmp_path)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-time migration for old universe asset_class labels."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Universe JSON files. Defaults to examples/*.json.",
    )
    parser.add_argument("--write", action="store_true", help="Update files in place.")
    args = parser.parse_args()

    paths = args.paths or sorted(Path("examples").glob("*.json"))
    changed = 0
    for path in paths:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or "symbols" not in payload:
            continue
        migrated, changes = canonicalize(payload)
        if not changes:
            print(f"{path}: ok")
            continue
        changed += 1
        print(f"{path}:")
        for change in changes[:20]:
            print(f"  {change}")
        if len(changes) > 20:
            print(f"  ... {len(changes) - 20} more")
        if args.write and not any(
            change.startswith("unresolved ") for change in changes
        ):
            write_json_atomic(path, migrated)
            print("  wrote")
    print(f"{'changed' if args.write else 'would change'}: {changed}")


if __name__ == "__main__":
    main()
