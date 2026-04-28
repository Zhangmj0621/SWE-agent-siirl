from __future__ import annotations

import os
from pathlib import Path


def _find_repo_root(start: Path) -> Path | None:
    for parent in [start] + list(start.parents):
        if (parent / ".git").is_dir():
            return parent
    return None


def get_repo_temp_dir() -> str | None:
    override = os.getenv("SWE_TMP_DIR")
    if override:
        path = Path(override)
    else:
        repo_root = _find_repo_root(Path(__file__).resolve())
        if repo_root is None:
            return None
        path = repo_root / "tmp"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return str(path)
