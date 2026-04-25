"""Text helpers for normalization and compact inspection output."""

from __future__ import annotations

import re


def normalize_title(title: str) -> str:
    return str(title).strip().lower()


def compact_text(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."

