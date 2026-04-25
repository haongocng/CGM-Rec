"""Structured records shared by continual-memory modules."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class EditProposal:
    action: str
    src: str
    relation: str
    dst: str
    weight_delta: float
    confidence_delta: float
    reason: str
    support_delta: int = 0
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "src": self.src,
            "relation": self.relation,
            "dst": self.dst,
            "weight_delta": self.weight_delta,
            "confidence_delta": self.confidence_delta,
            "support_delta": self.support_delta,
            "reason": self.reason,
            "evidence": self.evidence,
        }


@dataclass(slots=True)
class LessonPayload:
    sample_id: str
    session_items: list[str]
    candidate_items: list[str]
    target_item: str
    predicted_ranking: list[str]
    target_rank: int
    outcome_type: str
    diagnosis: str
    proposed_edits: list[EditProposal] = field(default_factory=list)
    signals: dict[str, float] = field(default_factory=dict)
    supporting_episode_ids: list[str] = field(default_factory=list)
    lesson_advice: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "sample_id": self.sample_id,
            "session_items": self.session_items,
            "candidate_items": self.candidate_items,
            "target_item": self.target_item,
            "predicted_ranking": self.predicted_ranking,
            "target_rank": self.target_rank,
            "outcome_type": self.outcome_type,
            "diagnosis": self.diagnosis,
            "signals": self.signals,
            "supporting_episode_ids": self.supporting_episode_ids,
            "lesson_advice": self.lesson_advice,
            "proposed_edits": [proposal.to_dict() for proposal in self.proposed_edits],
        }


@dataclass(slots=True)
class EpisodeRecord:
    episode_id: str
    sample_id: str
    session_items: list[str]
    candidate_items: list[str]
    target_item: str
    predicted_ranking: list[str]
    outcome_type: str
    target_rank: int
    proposed_edits: list[EditProposal] = field(default_factory=list)
    diagnosis: str = ""
    signals: dict[str, float] = field(default_factory=dict)
    lesson_advice: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "sample_id": self.sample_id,
            "session_items": self.session_items,
            "candidate_items": self.candidate_items,
            "target_item": self.target_item,
            "predicted_ranking": self.predicted_ranking,
            "outcome_type": self.outcome_type,
            "target_rank": self.target_rank,
            "diagnosis": self.diagnosis,
            "signals": self.signals,
            "lesson_advice": self.lesson_advice,
            "proposed_edits": [proposal.to_dict() for proposal in self.proposed_edits],
        }
