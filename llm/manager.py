"""KG4PO-style LLM manager adapted for CGM-Rec Phase 5."""

from __future__ import annotations

import json

from llm.config import get_provider_settings


class MockJsonModel:
    def __init__(self, model_tier: str = "json"):
        self.model_tier = model_tier

    def generate(self, system_prompt: str, human_prompt: str, context: dict | None = None) -> str:
        context = context or {}
        if context.get("task") == "lesson":
            target_rank = int(context.get("target_rank", 999))
            outcome = "success" if target_rank <= 5 else "failure"
            return json.dumps(
                {
                    "pattern": f"mock_{{outcome}}_pattern",
                    "signal": "Use graph/scorer disagreement and target rank as the observable signal.",
                    "failure_cause": "The mock lesson agent does not infer semantic causes.",
                    "rule": "Keep graph edits gated by diagnostics; use this advice only as weak confidence.",
                    "priority": "medium" if outcome == "failure" else "low",
                    "edge_hint_types": ["co_occurs_with"],
                    "advice_confidence": 0.5,
                }
            )

        ranked = context.get("scorer_ranking") or context.get("candidate_items") or []
        return json.dumps(
            {
                "reasoning": "Mock reranker preserves the scorer order for deterministic local testing.",
                "recommendations": ranked[:20],
            }
        )


class LangChainJsonModel:
    def __init__(self, provider: str, model_tier: str):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "Phase 5 real LLM providers require langchain-openai. "
                "Install cgm/requirements.txt or use --llm-provider mock."
            ) from exc

        settings = get_provider_settings(provider)
        if not settings.get("api_key"):
            raise ValueError(f"Missing API key for provider '{provider}'. Check cgm/.env.")

        temperature = 0.5 if model_tier in {"json", "power"} else 0.0
        kwargs = {
            "api_key": settings["api_key"],
            "model": settings["model"],
            "temperature": temperature,
        }
        if settings.get("base_url"):
            kwargs["base_url"] = settings["base_url"]
        if model_tier == "json":
            kwargs["model_kwargs"] = {"response_format": {"type": "json_object"}}
        self.model = ChatOpenAI(**kwargs)

    def generate(self, system_prompt: str, human_prompt: str, context: dict | None = None) -> str:
        response = self.model.invoke(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        return str(getattr(response, "content", response))


class LanguageModelManager:
    def __init__(self, provider: str = "mock"):
        self.provider = provider

    def get_model(self, model_tier: str = "json"):
        if self.provider == "mock":
            return MockJsonModel(model_tier=model_tier)
        return LangChainJsonModel(provider=self.provider, model_tier=model_tier)
