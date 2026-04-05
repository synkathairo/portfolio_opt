from __future__ import annotations

import os

from portfolio_opt.runtime import configure_local_cache_dirs


def test_configure_local_cache_dirs_sets_repo_local_defaults(monkeypatch) -> None:
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.delenv("MPLCONFIGDIR", raising=False)
    monkeypatch.delenv("FONTCONFIG_PATH", raising=False)
    monkeypatch.delenv("FONTCONFIG_FILE", raising=False)

    configure_local_cache_dirs()

    assert os.environ["XDG_CACHE_HOME"] == ".cache"
    assert os.environ["MPLCONFIGDIR"] == ".cache/matplotlib"
    assert os.environ["FONTCONFIG_PATH"] == ".cache/fontconfig"
    assert os.environ["FONTCONFIG_FILE"] == ".cache/fontconfig/fonts.conf"
