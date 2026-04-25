"""Strict JSON parsing and candidate compliance for Phase 5 LLM outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedRerank:
    ranked_titles: list[str]
    reasoning: str
    valid: bool
    errors: list[str] = field(default_factory=list)
    raw_response: str = ""


class RerankOutputParser:
    def parse(self, raw_response: str, candidate_items: list[str]) -> ParsedRerank:
        errors: list[str] = []
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            return ParsedRerank(
                ranked_titles=[],
                reasoning="",
                valid=False,
                errors=[f"invalid_json: {exc}"],
                raw_response=raw_response,
            )

        recommendations = payload.get("recommendations", [])
        reasoning = str(payload.get("reasoning", ""))
        if not isinstance(recommendations, list):
            errors.append("recommendations_not_list")
            recommendations = []

        candidates_by_normalized = {self._norm(item): item for item in candidate_items}
        seen: set[str] = set()
        ranked: list[str] = []
        for item in recommendations:
            key = self._norm(str(item))
            if key not in candidates_by_normalized:
                errors.append(f"unknown_candidate:{item}")
                continue
            if key in seen:
                errors.append(f"duplicate_candidate:{item}")
                continue
            seen.add(key)
            ranked.append(candidates_by_normalized[key])

        if len(ranked) != len(candidate_items):
            errors.append(f"expected_{len(candidate_items)}_items_got_{len(ranked)}")

        return ParsedRerank(
            ranked_titles=ranked,
            reasoning=reasoning,
            valid=not errors,
            errors=errors,
            raw_response=raw_response,
        )

    @staticmethod
    def _norm(value: str) -> str:
        return " ".join(value.strip().lower().split())
