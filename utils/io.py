"""I/O helpers for Phase 1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def load_json_array(path: str | Path) -> list[dict[str, Any]]:
    resolved = ensure_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"JSON file not found: {resolved}")

    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {resolved}, got {type(data).__name__}")

    return data


def write_json(path: str | Path, payload: Any) -> Path:
    resolved = ensure_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return resolved


def write_text(path: str | Path, payload: str) -> Path:
    resolved = ensure_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        handle.write(payload)
    return resolved
