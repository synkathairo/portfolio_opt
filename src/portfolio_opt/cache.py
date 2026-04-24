from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any

CACHE_DIR = Path(".cache")


def cache_path(name: str, payload: dict[str, Any]) -> Path:
    CACHE_DIR.mkdir(exist_ok=True)
    digest = sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[
        :16
    ]
    return CACHE_DIR / f"{name}_{digest}.json"


def read_cache(path: Path) -> Any:
    return json.loads(path.read_text())


def write_cache(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    tmp_path.replace(path)
