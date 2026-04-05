from __future__ import annotations

import os
from pathlib import Path


def configure_local_cache_dirs() -> None:
    """Keep third-party cache writes inside the repo during local runs."""
    cache_root = Path(".cache")
    matplotlib_dir = cache_root / "matplotlib"
    fontconfig_dir = cache_root / "fontconfig"
    matplotlib_dir.mkdir(parents=True, exist_ok=True)
    fontconfig_dir.mkdir(parents=True, exist_ok=True)

    # Matplotlib and Fontconfig otherwise try user-level cache directories
    # outside the repo, which is noisy in sandboxed and CI environments.
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_dir))
    os.environ.setdefault("FONTCONFIG_PATH", str(fontconfig_dir))
    os.environ.setdefault("FONTCONFIG_FILE", str(fontconfig_dir / "fonts.conf"))
