"""Build compact graph-memory evidence text for Phase 5 LLM prompts."""

from __future__ import annotations

from constants import (
    FEATURE_CANDIDATE_IN_DEGREE,
    FEATURE_CO_OCCUR_SCORE,
    FEATURE_SHARED_CATEGORIES,
    FEATURE_SHARED_KEYWORDS,
)
from memory.episodic_memory import EpisodeMatch
from model.scorer import ScoreOutput
from retrieval.semantic_retriever import RetrievalBundle


class LLMEvidenceBuilder:
    def build_graph_evidence(
        self,
        bundle: RetrievalBundle,
        score_output: ScoreOutput,
        limit_candidates: int = 20,
    ) -> str:
        score_by_title = {score.title: score for score in score_output.candidate_scores}
        lines: list[str] = []
        for evidence in bundle.candidate_evidence[:limit_candidates]:
            score = score_by_title.get(evidence.title)
            probability = score.probability if score else 0.0
            features = evidence.features
            parts = [
                f"- Candidate '{evidence.title}'",
                f"score_p={probability:.4f}",
                f"shared_categories={features.get(FEATURE_SHARED_CATEGORIES, 0.0):.0f}",
                f"shared_keywords={features.get(FEATURE_SHARED_KEYWORDS, 0.0):.0f}",
                f"co_occur={features.get(FEATURE_CO_OCCUR_SCORE, 0.0):.2f}",
                f"in_degree={features.get(FEATURE_CANDIDATE_IN_DEGREE, 0.0):.0f}",
            ]
            if evidence.metadata_edges:
                parts.append("edges=" + ", ".join(evidence.metadata_edges[:4]))
            lines.append("; ".join(parts) + ".")
        return "\n".join(lines)

    def build_episodic_hints(self, matches: list[EpisodeMatch]) -> str:
        if not matches:
            return "No similar episodes found."
        lines: list[str] = []
        for match in matches[:5]:
            lines.append(
                "- Similar episode "
                f"{match.episode.episode_id}: outcome={match.episode.outcome_type}, "
                f"target={match.episode.target_item}, rank={match.episode.target_rank}, "
                f"overlap_items={match.overlap_items[:3]}, "
                f"overlap_candidates={match.overlap_candidates[:3]}"
            )
        return "\n".join(lines)
