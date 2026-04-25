"""Advisory LLM lesson agent for Phase 5."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from llm.manager import LanguageModelManager
from llm.prompt_builder import Phase5PromptBuilder


@dataclass(slots=True)
class LessonAdvice:
    pattern: str
    signal: str
    failure_cause: str
    rule: str
    priority: str
    edge_hint_types: list[str] = field(default_factory=list)
    advice_confidence: float = 0.0
    valid: bool = True
    raw_response: str = ""

    def to_dict(self) -> dict:
        return {
            "pattern": self.pattern,
            "signal": self.signal,
            "failure_cause": self.failure_cause,
            "rule": self.rule,
            "priority": self.priority,
            "edge_hint_types": self.edge_hint_types,
            "advice_confidence": self.advice_confidence,
            "valid": self.valid,
            "raw_response": self.raw_response,
        }


class LLMLessonAgent:
    def __init__(
        self,
        provider: str,
        prompt_builder: Phase5PromptBuilder,
        model_manager: LanguageModelManager | None = None,
    ):
        self.provider = provider
        self.prompt_builder = prompt_builder
        self.model_manager = model_manager or LanguageModelManager(provider=provider)
        self.model = self.model_manager.get_model(model_tier="json")
        self.system_prompt = "You produce advisory JSON lessons for a graph-memory recommender."

    def infer(
        self,
        session_items: list[str],
        candidate_items: list[str],
        graph_evidence: str,
        final_ranking: list[str],
        target_item: str,
        target_rank: int,
    ) -> LessonAdvice:
        human_prompt = self.prompt_builder.build_lesson_prompt(
            session_items=session_items,
            candidate_items=candidate_items,
            graph_evidence=graph_evidence,
            final_ranking=final_ranking,
            target_item=target_item,
            target_rank=target_rank,
        )
        raw_response = self.model.generate(
            self.system_prompt,
            human_prompt,
            context={"task": "lesson", "target_rank": target_rank},
        )
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return LessonAdvice(
                pattern="invalid_lesson_json",
                signal="",
                failure_cause="",
                rule="",
                priority="low",
                valid=False,
                raw_response=raw_response,
            )

        priority = str(payload.get("priority", "low")).lower()
        if priority not in {"low", "medium", "high"}:
            priority = "low"
        edge_hint_types = payload.get("edge_hint_types", [])
        if not isinstance(edge_hint_types, list):
            edge_hint_types = []

        return LessonAdvice(
            pattern=str(payload.get("pattern", "")),
            signal=str(payload.get("signal", "")),
            failure_cause=str(payload.get("failure_cause", "")),
            rule=str(payload.get("rule", "")),
            priority=priority,
            edge_hint_types=[str(item) for item in edge_hint_types],
            advice_confidence=self._clamp_float(payload.get("advice_confidence", 0.0)),
            valid=True,
            raw_response=raw_response,
        )

    @staticmethod
    def _clamp_float(value: object) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0
