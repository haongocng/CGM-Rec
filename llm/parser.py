"""Strict JSON parsing and candidate compliance for Phase 5 LLM outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from utils.text import normalize_item_name


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
            payload = json.loads(self._extract_json(raw_response))
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

        candidates_by_normalized = {normalize_item_name(item): item for item in candidate_items}
        seen: set[str] = set()
        ranked: list[str] = []
        for item in recommendations:
            item_text = self._extract_item_text(item)
            key = normalize_item_name(item_text)
            if key not in candidates_by_normalized:
                errors.append(f"unknown_candidate:{item_text}")
                continue
            if key in seen:
                errors.append(f"duplicate_candidate:{item_text}")
                continue
            seen.add(key)
            ranked.append(candidates_by_normalized[key])

        if len(ranked) != len(candidate_items):
            errors.append(f"expected_{len(candidate_items)}_items_got_{len(ranked)}")
            for candidate in candidate_items:
                key = normalize_item_name(candidate)
                if key not in seen:
                    seen.add(key)
                    ranked.append(candidate)

        return ParsedRerank(
            ranked_titles=ranked,
            reasoning=reasoning,
            valid=not errors,
            errors=errors,
            raw_response=raw_response,
        )

    @staticmethod
    def _extract_json(raw_response: str) -> str:
        response = str(raw_response).strip()
        if response.startswith("```"):
            lines = response.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            response = "\n".join(lines).strip()

        start = response.find("{")
        end = response.rfind("}")
        if start != -1 and end != -1 and start < end:
            return response[start : end + 1]
        return response

    @staticmethod
    def _extract_item_text(item: object) -> str:
        if isinstance(item, dict):
            for key in ("title", "item", "name", "product"):
                value = item.get(key)
                if value:
                    return str(value)
        return str(item)
