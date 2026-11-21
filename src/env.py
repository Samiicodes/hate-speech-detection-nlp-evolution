from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"


def load_env(path: Path | None = None, override: bool = False) -> list[str]:
    path = path or ENV_FILE
    if not path.exists():
        return []

    loaded: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if override or key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded


def require(name: str = "ANTHROPIC_API_KEY") -> str:
    load_env()
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set.\n"
            f"  Put it in {ENV_FILE} as:\n"
            f"      {name}=sk-ant-...\n"
            f"  (.env is gitignored; the key must never be committed.)\n"
            f"  Or export it in the shell for this session only."
        )
    return value


def has(name: str = "ANTHROPIC_API_KEY") -> bool:
    load_env()
    return bool(os.environ.get(name))


# Load on import
load_env()

__all__ = ["ENV_FILE", "has", "load_env", "require"]
