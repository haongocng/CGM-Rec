"""Phase 5 fixed LLM reranker."""

from __future__ import annotations

from dataclasses import dataclass, field

from llm.manager import LanguageModelManager
from llm.parser import RerankOutputParser
from llm.prompt_builder import Phase5PromptBuilder


@dataclass(slots=True)
class LLMRerankResult:
    ranked_titles: list[str]
    reasoning: str
    parser_valid: bool
    fallback_used: bool
    raw_response: str
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ranked_titles": self.ranked_titles,
            "reasoning": self.reasoning,
            "parser_valid": self.parser_valid,
            "fallback_used": self.fallback_used,
            "raw_response": self.raw_response,
            "errors": self.errors,
        }


class LLMReranker:
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
        self.parser = RerankOutputParser()
        self.system_prompt = self.prompt_builder.load_system_prompt()

    def rerank(
        self,
        session_items: list[str],
        candidate_items: list[str],
        graph_evidence: str,
        episodic_hints: str,
        scorer_ranking: list[str],
    ) -> LLMRerankResult:
        human_prompt = self.prompt_builder.build_rerank_prompt(
            session_items=session_items,
            candidate_items=candidate_items,
            graph_evidence=graph_evidence,
            episodic_hints=episodic_hints,
            scorer_ranking=scorer_ranking,
        )
        raw_response = self.model.generate(
            self.system_prompt,
            human_prompt,
            context={
                "task": "rerank",
                "candidate_items": candidate_items,
                "scorer_ranking": scorer_ranking,
            },
        )
        parsed = self.parser.parse(raw_response, candidate_items=candidate_items)
        if parsed.valid:
            return LLMRerankResult(
                ranked_titles=parsed.ranked_titles,
                reasoning=parsed.reasoning,
                parser_valid=True,
                fallback_used=False,
                raw_response=raw_response,
            )
        return LLMRerankResult(
            ranked_titles=list(scorer_ranking),
            reasoning=parsed.reasoning,
            parser_valid=False,
            fallback_used=True,
            raw_response=raw_response,
            errors=parsed.errors,
        )
