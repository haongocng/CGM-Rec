"""Text helpers for normalization and compact inspection output."""

from __future__ import annotations

import html
import re


def normalize_title(title: str) -> str:
    return str(title).strip().lower()


def normalize_item_name(item_name: str) -> str:
    """Normalize item names for robust LLM output and ground-truth matching."""
    if not isinstance(item_name, str):
        item_name = str(item_name)

    cleaned = html.unescape(item_name).strip().lower()
    cleaned = cleaned.replace('"', "").replace("'", "")
    cleaned = re.sub(r"^\s*\d+[\.\)\-]?\s*", "", cleaned)
    cleaned = re.sub(r"^movie:\s*", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.replace("[", "").replace("]", "")
    cleaned = cleaned.replace("(", "").replace(")", "")
    return cleaned


def compact_text(text: str, limit: int = 120) -> str:
    text = re.sub(r"\s+", " ", str(text).strip())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."
