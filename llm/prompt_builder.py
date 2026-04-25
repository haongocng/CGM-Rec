"""Prompt construction for the Phase 5 LLM reranker and lesson agent."""

from __future__ import annotations

from pathlib import Path


DEFAULT_SYSTEM_PROMPT = """
Your task is to analyze the user's current session interactions and the candidate set of items to accurately infer the user's preferences and intent.
1. Consider patterns or combinations of items within the session that may indicate the user's genre preference or other relevant criteria.
2. Evaluate the context and relevance of the items in the candidate set to the user's session interactions.
3. Deduce the user's interactive intent within each combination.
4. Rearrange the candidate set according to the likelihood of potential user interactions.
Provide the rearranged list of items from the candidate set."""


DEFAULT_RERANK_TEMPLATE = """
This is a recommendation task. Your task is to analyze the user's current session history, graph-derived evidence, advisory memory hints, and scorer ranking to rerank the provided Candidate Set.

1. User interactions:
{session_items}

2. Candidate Set (YOU MUST SELECT AND RERANK EXACTLY FROM THESE 20 ITEMS):
{candidate_set}

3. Graph-derived evidence from current Continual Graph Memory:
{graph_evidence}

4. Episodic memory hints:
{episodic_hints}

5. Current frozen scorer ranking:
{scorer_ranking}

CRITICAL RULES:
- Do not hallucinate. Do not suggest any item that is not explicitly listed in the Candidate Set.
- Return all 20 items from the candidate set, ordered by relevance.
- Return strictly in JSON format with no markdown:
{{
  "reasoning": "short analysis of user intent and ranking evidence",
  "recommendations": ["item name 1", "item name 2", "..."]
}}"""


LESSON_TEMPLATE = """
You are an advisory lesson agent for a continual graph-memory recommender.
You do not edit the graph directly. Infer a compact JSON lesson that can help the existing diagnostics and write policy decide whether graph evidence should be reinforced, suppressed, or treated as tentative.

Session:
{session_items}

Candidate Set:
{candidate_set}

Graph Evidence:
{graph_evidence}

Final LLM Ranking:
{final_ranking}

Ground Truth:
{target_item}

Target Rank:
{target_rank}

Return strictly JSON:
{{
  "pattern": "short reusable pattern name",
  "signal": "observable trigger signal",
  "failure_cause": "why the ranking likely succeeded or failed",
  "rule": "advisory graph-memory rule",
  "priority": "low | medium | high",
  "edge_hint_types": ["co_occurs_with", "belongs_to", "has_keyword"],
  "advice_confidence": 0.0
}}"""


class Phase5PromptBuilder:
    def __init__(self, project_root: str | Path, kg4po_root: str | Path | None = None):
        self.project_root = Path(project_root)
        self.kg4po_root = Path(kg4po_root) if kg4po_root else self.project_root.parent / "KG4PO"

    def load_system_prompt(self) -> str:
        path = self.kg4po_root / "prompts" / "best_system_prompt.txt"
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
        return DEFAULT_SYSTEM_PROMPT

    def build_rerank_prompt(
        self,
        session_items: list[str],
        candidate_items: list[str],
        graph_evidence: str,
        episodic_hints: str,
        scorer_ranking: list[str],
    ) -> str:
        return DEFAULT_RERANK_TEMPLATE.format(
            session_items=self._numbered(session_items),
            candidate_set=self._numbered(candidate_items),
            graph_evidence=graph_evidence or "No graph evidence available.",
            episodic_hints=episodic_hints or "No relevant episodic hints.",
            scorer_ranking=self._numbered(scorer_ranking),
        )

    def build_lesson_prompt(
        self,
        session_items: list[str],
        candidate_items: list[str],
        graph_evidence: str,
        final_ranking: list[str],
        target_item: str,
        target_rank: int,
    ) -> str:
        return LESSON_TEMPLATE.format(
            session_items=self._numbered(session_items),
            candidate_set=self._numbered(candidate_items),
            graph_evidence=graph_evidence or "No graph evidence available.",
            final_ranking=self._numbered(final_ranking),
            target_item=target_item,
            target_rank=target_rank,
        )

    @staticmethod
    def _numbered(items: list[str]) -> str:
        return "\n".join(f'{idx + 1}."{item}"' for idx, item in enumerate(items))
