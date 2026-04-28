from __future__ import annotations

import json
from pathlib import Path
from typing import Any

YAHOO_SECTOR_ASSET_CLASSES = {
    "Basic Materials": "sector_basic_materials",
    "Communication Services": "sector_communication_services",
    "Consumer Cyclical": "sector_consumer_cyclical",
    "Consumer Defensive": "sector_consumer_defensive",
    "Energy": "sector_energy",
    "Financial Services": "sector_financials",
    "Healthcare": "sector_health_care",
    "Industrials": "sector_industrials",
    "Real Estate": "sector_real_estate",
    "Technology": "sector_technology",
    "Utilities": "sector_utilities",
}


def main() -> None:
    cache_dir = Path(".cache")
    paths = sorted(cache_dir.glob("ticker_info_*.json"))
    changed = 0
    unchanged = 0
    skipped = 0

    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except Exception as exc:
            print(f"{path}: skipped unreadable JSON ({exc})")
            skipped += 1
            continue
        if not isinstance(payload, dict):
            print(f"{path}: skipped non-object JSON")
            skipped += 1
            continue

        asset_class = _asset_class_for(payload)
        if payload.get("asset_class") == asset_class:
            unchanged += 1
            continue

        payload["asset_class"] = asset_class
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        changed += 1

    print(
        f"ticker_info cache files: changed={changed} "
        f"unchanged={unchanged} skipped={skipped} total={len(paths)}"
    )


def _asset_class_for(payload: dict[str, Any]) -> str:
    sector = payload.get("sector")
    if not isinstance(sector, str):
        return "sector_unknown"
    return YAHOO_SECTOR_ASSET_CLASSES.get(sector, "sector_unknown")


if __name__ == "__main__":
    main()
