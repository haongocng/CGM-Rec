"""Parsing logic for raw session strings."""

from __future__ import annotations

import re

from constants import CANDIDATE_MARKER, DEFAULT_CANDIDATE_COUNT, ITEM_PATTERN
from data.schema import ParsedInput


class InputParser:
    def __init__(self, expected_candidate_count: int = DEFAULT_CANDIDATE_COUNT):
        self.expected_candidate_count = expected_candidate_count
        self._item_pattern = re.compile(ITEM_PATTERN)

    def parse(self, raw_input: str) -> ParsedInput:
        raw_input = str(raw_input)
        parts = raw_input.split(CANDIDATE_MARKER, maxsplit=1)

        session_part = parts[0]
        candidate_part = parts[1] if len(parts) == 2 else ""

        session_items = self._item_pattern.findall(session_part)
        candidate_items = self._item_pattern.findall(candidate_part)

        if self.expected_candidate_count and len(candidate_items) != self.expected_candidate_count:
            raise ValueError(
                f"Expected {self.expected_candidate_count} candidates, found {len(candidate_items)}"
            )

        return ParsedInput(
            raw_input=raw_input,
            session_items=session_items,
            candidate_items=candidate_items,
        )

