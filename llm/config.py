"""LLM provider configuration for Phase 5."""

from __future__ import annotations

import os
from pathlib import Path


TIMELYGPT_API_BASE_URL = "https://hello.timelygpt.co.kr/api/v2/chat/bridge/openai"
DEEPINFRA_API_BASE_URL = "https://api.deepinfra.com/v1/openai"


def load_env_file(path: str | Path) -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_project_env(project_root: str | Path) -> None:
    load_env_file(Path(project_root) / ".env")


def get_provider_settings(provider: str) -> dict:
    if provider == "openai":
        return {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": os.getenv("OPENAI_API_BASE_URL") or None,
            "model": os.getenv("OPENAI_MODEL", "gpt-4o"),
        }
    if provider == "deepinfra":
        return {
            "api_key": os.getenv("DEEPINFRA_API_KEY"),
            "base_url": os.getenv("DEEPINFRA_API_BASE_URL", DEEPINFRA_API_BASE_URL),
            "model": os.getenv("DEEPINFRA_MODEL", "Qwen/Qwen3-Next-80B-A3B-Instruct"),
        }
    if provider == "timelygpt":
        return {
            "api_key": os.getenv("TIMELYGPT_API_KEY"),
            "base_url": os.getenv("TIMELYGPT_API_BASE_URL", TIMELYGPT_API_BASE_URL),
            "model": os.getenv("TIMELYGPT_MODEL", "openai/gpt-4.1-mini"),
        }
    if provider == "mock":
        return {"api_key": "", "base_url": "", "model": "mock-json"}
    raise ValueError(f"Unsupported LLM provider: {provider}")
