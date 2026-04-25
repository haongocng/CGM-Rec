"""Hybrid diagnostics for continual graph-memory edits."""

from __future__ import annotations

from dataclasses import dataclass

from constants import (
    EDGE_REL_BELONGS_TO,
    EDGE_REL_CO_OCCURS,
    EDGE_REL_HAS_DESCRIPTION,
    EDGE_REL_HAS_KEYWORD,
    FEATURE_CO_OCCUR_SCORE,
    FEATURE_HAS_DESCRIPTION,
    FEATURE_SHARED_CATEGORIES,
    FEATURE_SHARED_KEYWORDS,
)
from data.schema import SessionSample
from memory.schema import EditProposal, LessonPayload
from memory.semantic_memory import SemanticEdgeState, SemanticMemory
from model.scorer import CandidateScore, ScoreOutput
from retrieval.semantic_retriever import RetrievalBundle
from utils.text import normalize_title


@dataclass(slots=True)
class DiagnosticConfig:
    success_top_k: int = 5
    max_success_edits: int = 4
    max_failure_edits: int = 6
    reserved_tentative_edits: int = 2
    reinforce_weight_delta: float = 0.2
    reinforce_confidence_delta: float = 0.05
    suppress_weight_delta: float = -0.15
    suppress_confidence_delta: float = -0.04
    tentative_weight_delta: float = 0.1
    tentative_confidence_delta: float = 0.02


class HybridDiagnostics:
    def __init__(self, memory: SemanticMemory, config: DiagnosticConfig | None = None):
        self.memory = memory
        self.config = config or DiagnosticConfig()

    def analyze(
        self,
        sample: SessionSample,
        retrieval: RetrievalBundle,
        score_output: ScoreOutput,
        supporting_episode_ids: list[str] | None = None,
    ) -> LessonPayload:
        ranked_titles = list(score_output.ranked_titles)
        target_rank = self._get_rank(ranked_titles, sample.target)
        target_score = self._candidate_by_title(score_output, sample.target)
        if target_score is None:
            raise ValueError(f"Target '{sample.target}' is missing from score output")

        if target_rank <= self.config.success_top_k:
            diagnosis, proposals = self._diagnose_success(
                sample=sample,
                retrieval=retrieval,
                ranked_titles=ranked_titles,
                target_score=target_score,
            )
            outcome_type = "success"
        else:
            diagnosis, proposals = self._diagnose_failure(
                sample=sample,
                retrieval=retrieval,
                score_output=score_output,
                ranked_titles=ranked_titles,
                target_score=target_score,
            )
            outcome_type = "failure"

        return LessonPayload(
            sample_id=sample.sample_id,
            session_items=list(sample.parsed_input.session_items),
            candidate_items=list(sample.parsed_input.candidate_items),
            target_item=sample.target,
            predicted_ranking=ranked_titles,
            target_rank=target_rank,
            outcome_type=outcome_type,
            diagnosis=diagnosis,
            proposed_edits=proposals,
            signals={
                "target_rank": float(target_rank),
                "top_probability": score_output.candidate_scores[0].probability if score_output.candidate_scores else 0.0,
                "target_probability": target_score.probability,
            },
            supporting_episode_ids=supporting_episode_ids or [],
        )

    def _diagnose_success(
        self,
        sample: SessionSample,
        retrieval: RetrievalBundle,
        ranked_titles: list[str],
        target_score: CandidateScore,
    ) -> tuple[str, list[EditProposal]]:
        proposals: list[EditProposal] = []
        target_id = self._item_id(sample.target)
        session_ids = [self._item_id(title) for title in sample.parsed_input.session_items]
        positive_features = self._positive_feature_names(target_score)

        if FEATURE_CO_OCCUR_SCORE in positive_features:
            for session_id in session_ids:
                edge = self._find_edge(session_id, EDGE_REL_CO_OCCURS, target_id)
                if edge is None:
                    continue
                proposals.append(
                    self._proposal_from_edge(
                        action="reinforce_edge",
                        edge=edge,
                        weight_delta=self.config.reinforce_weight_delta,
                        confidence_delta=self.config.reinforce_confidence_delta,
                        reason="Successful ranking was supported by recent co-occurrence evidence.",
                        evidence={
                            "feature": FEATURE_CO_OCCUR_SCORE,
                            "contribution": target_score.feature_contributions.get(FEATURE_CO_OCCUR_SCORE, 0.0),
                            "session_item": self.memory.nodes.get(session_id).label if session_id in self.memory.nodes else session_id,
                        },
                    )
                )

        if FEATURE_SHARED_CATEGORIES in positive_features:
            proposals.extend(
                self._propose_reinforce_shared_neighbors(
                    target_id=target_id,
                    session_ids=session_ids,
                    relation=EDGE_REL_BELONGS_TO,
                    feature_name=FEATURE_SHARED_CATEGORIES,
                    contribution=target_score.feature_contributions.get(FEATURE_SHARED_CATEGORIES, 0.0),
                )
            )

        if FEATURE_SHARED_KEYWORDS in positive_features:
            proposals.extend(
                self._propose_reinforce_shared_neighbors(
                    target_id=target_id,
                    session_ids=session_ids,
                    relation=EDGE_REL_HAS_KEYWORD,
                    feature_name=FEATURE_SHARED_KEYWORDS,
                    contribution=target_score.feature_contributions.get(FEATURE_SHARED_KEYWORDS, 0.0),
                )
            )

        if FEATURE_HAS_DESCRIPTION in positive_features:
            description_edge = self._first_neighbor_edge(target_id, EDGE_REL_HAS_DESCRIPTION)
            if description_edge is not None:
                proposals.append(
                    self._proposal_from_edge(
                        action="reinforce_edge",
                        edge=description_edge,
                        weight_delta=0.05,
                        confidence_delta=0.01,
                        reason="Description evidence aligned with a successful recommendation.",
                        evidence={
                            "feature": FEATURE_HAS_DESCRIPTION,
                            "contribution": target_score.feature_contributions.get(FEATURE_HAS_DESCRIPTION, 0.0),
                        },
                    )
                )

        proposals = self._dedupe_proposals(proposals)[: self.config.max_success_edits]
        diagnosis = (
            f"Success lesson: target ranked at position {self._get_rank(ranked_titles, sample.target)}. "
            f"Reinforce graph evidence that aligned with the correct Top-{self.config.success_top_k} result."
        )
        return diagnosis, proposals

    def _diagnose_failure(
        self,
        sample: SessionSample,
        retrieval: RetrievalBundle,
        score_output: ScoreOutput,
        ranked_titles: list[str],
        target_score: CandidateScore,
    ) -> tuple[str, list[EditProposal]]:
        proposals: list[EditProposal] = []
        target_id = self._item_id(sample.target)
        session_ids = [self._item_id(title) for title in sample.parsed_input.session_items]
        target_rank = self._get_rank(ranked_titles, sample.target)
        higher_ranked = [score for score in score_output.candidate_scores if score.title != sample.target][: max(1, self.config.success_top_k)]

        for competitor in higher_ranked:
            proposals.extend(
                self._propose_suppress_misleading_paths(
                    competitor=competitor,
                    session_ids=session_ids,
                )
            )

        target_has_bridge = any(
            self._find_edge(session_id, EDGE_REL_CO_OCCURS, target_id) is not None for session_id in session_ids
        )
        if not target_has_bridge:
            for session_id in session_ids[-2:]:
                proposals.append(
                    EditProposal(
                        action="add_tentative_edge",
                        src=session_id,
                        relation=EDGE_REL_CO_OCCURS,
                        dst=target_id,
                        weight_delta=self.config.tentative_weight_delta,
                        confidence_delta=self.config.tentative_confidence_delta,
                        support_delta=1,
                        reason="Failure lesson suggests a missing short-term bridge from the current session to the true target.",
                        evidence={
                            "feature": FEATURE_CO_OCCUR_SCORE,
                            "target_rank": target_rank,
                            "target_probability": target_score.probability,
                            "session_item": self.memory.nodes.get(session_id).label if session_id in self.memory.nodes else session_id,
                        },
                    )
                )

        proposals = self._limit_failure_proposals(self._dedupe_proposals(proposals))
        diagnosis = (
            f"Failure lesson: target ranked at position {target_rank}, outside Top-{self.config.success_top_k}. "
            "Suppress misleading high-support paths and add tentative episodic bridges toward the true target."
        )
        return diagnosis, proposals

    def _propose_reinforce_shared_neighbors(
        self,
        target_id: str,
        session_ids: list[str],
        relation: str,
        feature_name: str,
        contribution: float,
    ) -> list[EditProposal]:
        proposals: list[EditProposal] = []
        target_neighbors = self._neighbor_ids(target_id, relation)
        session_neighbors: set[str] = set()
        for session_id in session_ids:
            session_neighbors.update(self._neighbor_ids(session_id, relation))

        for neighbor_id in sorted(target_neighbors & session_neighbors):
            edge = self._find_edge(target_id, relation, neighbor_id)
            if edge is None:
                continue
            proposals.append(
                self._proposal_from_edge(
                    action="reinforce_edge",
                    edge=edge,
                    weight_delta=self.config.reinforce_weight_delta,
                    confidence_delta=self.config.reinforce_confidence_delta,
                        reason="Successful ranking aligned with shared semantic evidence.",
                        evidence={
                            "feature": feature_name,
                            "contribution": contribution,
                            "shared_neighbor": self.memory.nodes.get(neighbor_id).label if neighbor_id in self.memory.nodes else neighbor_id,
                        },
                    )
                )
        return proposals

    def _propose_suppress_misleading_paths(
        self,
        competitor: CandidateScore,
        session_ids: list[str],
    ) -> list[EditProposal]:
        proposals: list[EditProposal] = []
        competitor_id = self._item_id(competitor.title)
        positive_features = self._positive_feature_names(competitor)

        if FEATURE_CO_OCCUR_SCORE in positive_features:
            for session_id in session_ids:
                edge = self._find_edge(session_id, EDGE_REL_CO_OCCURS, competitor_id)
                if edge is None:
                    continue
                proposals.append(
                    self._proposal_from_edge(
                        action="suppress_edge",
                        edge=edge,
                        weight_delta=self.config.suppress_weight_delta,
                        confidence_delta=self.config.suppress_confidence_delta,
                        reason="This co-occurrence path strongly supported a wrong candidate ranked above the target.",
                        evidence={
                            "feature": FEATURE_CO_OCCUR_SCORE,
                            "competitor": competitor.title,
                            "contribution": competitor.feature_contributions.get(FEATURE_CO_OCCUR_SCORE, 0.0),
                        },
                    )
                )

        if FEATURE_SHARED_CATEGORIES in positive_features:
            proposals.extend(
                self._propose_suppress_shared_neighbors(
                    candidate_id=competitor_id,
                    session_ids=session_ids,
                    relation=EDGE_REL_BELONGS_TO,
                    feature_name=FEATURE_SHARED_CATEGORIES,
                    contribution=competitor.feature_contributions.get(FEATURE_SHARED_CATEGORIES, 0.0),
                    competitor_title=competitor.title,
                )
            )

        if FEATURE_SHARED_KEYWORDS in positive_features:
            proposals.extend(
                self._propose_suppress_shared_neighbors(
                    candidate_id=competitor_id,
                    session_ids=session_ids,
                    relation=EDGE_REL_HAS_KEYWORD,
                    feature_name=FEATURE_SHARED_KEYWORDS,
                    contribution=competitor.feature_contributions.get(FEATURE_SHARED_KEYWORDS, 0.0),
                    competitor_title=competitor.title,
                )
            )

        return proposals

    def _propose_suppress_shared_neighbors(
        self,
        candidate_id: str,
        session_ids: list[str],
        relation: str,
        feature_name: str,
        contribution: float,
        competitor_title: str,
    ) -> list[EditProposal]:
        proposals: list[EditProposal] = []
        candidate_neighbors = self._neighbor_ids(candidate_id, relation)
        session_neighbors: set[str] = set()
        for session_id in session_ids:
            session_neighbors.update(self._neighbor_ids(session_id, relation))

        for neighbor_id in sorted(candidate_neighbors & session_neighbors):
            edge = self._find_edge(candidate_id, relation, neighbor_id)
            if edge is None:
                continue
            proposals.append(
                self._proposal_from_edge(
                    action="suppress_edge",
                    edge=edge,
                    weight_delta=self.config.suppress_weight_delta,
                    confidence_delta=self.config.suppress_confidence_delta,
                    reason="Shared semantic evidence over-supported an incorrect candidate.",
                        evidence={
                            "feature": feature_name,
                            "competitor": competitor_title,
                            "contribution": contribution,
                            "shared_neighbor": self.memory.nodes.get(neighbor_id).label if neighbor_id in self.memory.nodes else neighbor_id,
                        },
                    )
                )
        return proposals

    def _proposal_from_edge(
        self,
        action: str,
        edge: SemanticEdgeState,
        weight_delta: float,
        confidence_delta: float,
        reason: str,
        evidence: dict,
    ) -> EditProposal:
        support_delta = 1 if action == "reinforce_edge" else 0
        return EditProposal(
            action=action,
            src=edge.src,
            relation=edge.relation,
            dst=edge.dst,
            weight_delta=weight_delta,
            confidence_delta=confidence_delta,
            support_delta=support_delta,
            reason=reason,
            evidence=evidence,
        )

    def _dedupe_proposals(self, proposals: list[EditProposal]) -> list[EditProposal]:
        deduped: dict[tuple[str, str, str, str], EditProposal] = {}
        for proposal in proposals:
            key = (proposal.action, proposal.src, proposal.relation, proposal.dst)
            if key not in deduped:
                deduped[key] = proposal
                continue
            current = deduped[key]
            current.weight_delta += proposal.weight_delta
            current.confidence_delta += proposal.confidence_delta
            current.support_delta += proposal.support_delta
        return list(deduped.values())

    def _limit_failure_proposals(self, proposals: list[EditProposal]) -> list[EditProposal]:
        tentative = [proposal for proposal in proposals if proposal.action == "add_tentative_edge"]
        other = [proposal for proposal in proposals if proposal.action != "add_tentative_edge"]
        keep_tentative = tentative[: self.config.reserved_tentative_edits]
        remaining_slots = max(0, self.config.max_failure_edits - len(keep_tentative))
        return keep_tentative + other[:remaining_slots]

    def _positive_feature_names(self, candidate: CandidateScore) -> list[str]:
        scored = [
            (name, value)
            for name, value in candidate.feature_contributions.items()
            if value > 0.0
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [name for name, _value in scored]

    def _candidate_by_title(self, score_output: ScoreOutput, title: str) -> CandidateScore | None:
        for candidate in score_output.candidate_scores:
            if candidate.title == title:
                return candidate
        return None

    def _get_rank(self, ranked_titles: list[str], title: str) -> int:
        for idx, candidate in enumerate(ranked_titles):
            if candidate == title:
                return idx + 1
        return 999

    def _neighbor_ids(self, item_id: str, relation: str) -> set[str]:
        return self.memory.neighbor_ids(item_id, relation)

    def _find_edge(self, src: str, relation: str, dst: str) -> SemanticEdgeState | None:
        return self.memory.get_edge(src, relation, dst)

    def _first_neighbor_edge(self, src: str, relation: str) -> SemanticEdgeState | None:
        edges = self.memory.get_edges(src, relation=relation)
        return edges[0] if edges else None

    @staticmethod
    def _item_id(title: str) -> str:
        return f"item::{normalize_title(title)}"
