"""Semantic retrieval and feature extraction for Phase 3."""

from __future__ import annotations

from dataclasses import dataclass, field

from constants import (
    EDGE_REL_BELONGS_TO,
    EDGE_REL_CO_OCCURS,
    EDGE_REL_HAS_DESCRIPTION,
    EDGE_REL_HAS_KEYWORD,
    FEATURE_CANDIDATE_CATEGORY_COUNT,
    FEATURE_CANDIDATE_IN_DEGREE,
    FEATURE_CANDIDATE_KEYWORD_COUNT,
    FEATURE_CO_OCCUR_SCORE,
    FEATURE_HAS_DESCRIPTION,
    FEATURE_SHARED_CATEGORIES,
    FEATURE_SHARED_KEYWORDS,
)
from graph.schema import SeedGraph
from memory.semantic_memory import SemanticMemory
from utils.text import normalize_title


@dataclass(slots=True)
class CandidateEvidence:
    title: str
    candidate_id: str
    features: dict[str, float]
    metadata_edges: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalBundle:
    session_items: list[str]
    candidate_items: list[str]
    candidate_evidence: list[CandidateEvidence]


class SemanticRetriever:
    def __init__(self, graph: SeedGraph | SemanticMemory):
        if isinstance(graph, SemanticMemory):
            self.memory = graph
        else:
            self.memory = SemanticMemory.from_seed_graph(graph)

    def retrieve(self, session_items: list[str], candidate_items: list[str]) -> RetrievalBundle:
        session_ids = [self._item_id(title) for title in session_items]
        session_categories = self._collect_neighbor_ids(session_ids, EDGE_REL_BELONGS_TO)
        session_keywords = self._collect_neighbor_ids(session_ids, EDGE_REL_HAS_KEYWORD)

        candidate_evidence: list[CandidateEvidence] = []
        for title in candidate_items:
            candidate_id = self._item_id(title)
            category_ids = self._neighbor_ids(candidate_id, EDGE_REL_BELONGS_TO)
            keyword_ids = self._neighbor_ids(candidate_id, EDGE_REL_HAS_KEYWORD)
            has_description = 1.0 if self._neighbor_ids(candidate_id, EDGE_REL_HAS_DESCRIPTION) else 0.0
            co_occur_score = self._co_occur_score(session_ids, candidate_id)
            features = {
                FEATURE_SHARED_CATEGORIES: float(len(session_categories & category_ids)),
                FEATURE_SHARED_KEYWORDS: float(len(session_keywords & keyword_ids)),
                FEATURE_CO_OCCUR_SCORE: float(co_occur_score),
                FEATURE_CANDIDATE_CATEGORY_COUNT: float(len(category_ids)),
                FEATURE_CANDIDATE_KEYWORD_COUNT: float(len(keyword_ids)),
                FEATURE_HAS_DESCRIPTION: has_description,
                FEATURE_CANDIDATE_IN_DEGREE: float(self.memory.in_degree(candidate_id)),
            }
            candidate_evidence.append(
                CandidateEvidence(
                    title=title,
                    candidate_id=candidate_id,
                    features=features,
                    metadata_edges=self._describe_edges(candidate_id),
                )
            )

        return RetrievalBundle(
            session_items=session_items,
            candidate_items=candidate_items,
            candidate_evidence=candidate_evidence,
        )

    def _collect_neighbor_ids(self, item_ids: list[str], relation: str) -> set[str]:
        collected: set[str] = set()
        for item_id in item_ids:
            collected.update(self._neighbor_ids(item_id, relation))
        return collected

    def _neighbor_ids(self, item_id: str, relation: str) -> set[str]:
        return self.memory.neighbor_ids(item_id, relation)

    def _co_occur_score(self, session_ids: list[str], candidate_id: str) -> float:
        score = 0.0
        for session_id in session_ids:
            for edge in self.memory.get_edges(session_id, relation=EDGE_REL_CO_OCCURS):
                if edge.dst == candidate_id:
                    score += edge.weight
        return score

    def _describe_edges(self, item_id: str, limit: int = 6) -> list[str]:
        return self.memory.describe_edges(item_id, limit=limit)

    @staticmethod
    def _item_id(title: str) -> str:
        return f"item::{normalize_title(title)}"
