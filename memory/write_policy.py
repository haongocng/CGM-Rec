"""Gatekeeping heuristics for writing continual graph-memory edits."""

from __future__ import annotations

from dataclasses import dataclass, field

from memory.schema import EditProposal, LessonPayload
from memory.semantic_memory import SemanticEdgeState, SemanticMemory


@dataclass(slots=True)
class WriteScore:
    action: str
    value: float
    accepted: bool
    target_memory: str
    components: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "value": self.value,
            "accepted": self.accepted,
            "target_memory": self.target_memory,
            "components": self.components,
            "rationale": self.rationale,
        }


@dataclass(slots=True)
class WritePolicyConfig:
    success_top_k: int = 5
    reinforce_threshold: float = 0.35
    suppress_threshold: float = 0.45
    tentative_threshold: float = 0.30
    contribution_scale: float = 0.25
    support_scale: float = 5.0
    min_existing_confidence_for_suppress: float = 0.15


class WritePolicy:
    def __init__(self, config: WritePolicyConfig | None = None):
        self.config = config or WritePolicyConfig()

    def evaluate(
        self,
        lesson: LessonPayload,
        proposal: EditProposal,
        semantic_memory: SemanticMemory,
    ) -> WriteScore:
        existing = semantic_memory.get_edge(proposal.src, proposal.relation, proposal.dst)
        contribution = abs(float(proposal.evidence.get("contribution", 0.0)))
        contribution_signal = min(1.0, contribution / self.config.contribution_scale)
        support_signal = self._support_signal(existing)
        confidence_signal = existing.confidence if existing else 0.0
        rank_signal = self._rank_signal(lesson)
        novelty_signal = 1.0 if existing is None else 0.0
        target_probability = float(lesson.signals.get("target_probability", 0.0))
        uncertainty_signal = 1.0 - max(0.0, min(1.0, target_probability))
        outcome_signal = self._outcome_signal(lesson, proposal)
        advice_signal = self._advice_signal(lesson, proposal)

        if proposal.action == "reinforce_edge":
            # ACL §4.6: reinforce edges that contributed to correct ranking.
            # Outcome signal provides a strong bonus when outcome == "success"
            # but does NOT block reinforcement during failures — an edge that
            # correctly supported the target is still worth reinforcing even
            # if the target didn't reach top-K.
            value = (
                0.25 * rank_signal
                + 0.25 * contribution_signal
                + 0.20 * outcome_signal
                + 0.15 * support_signal
                + 0.10 * confidence_signal
                + 0.05 * advice_signal
            )
            accepted = value >= self.config.reinforce_threshold
            rationale = (
                "Reinforce when evidence contribution and support are strong. "
                "Success outcome gives a bonus but is not strictly required."
            )
        elif proposal.action == "suppress_edge":
            # ACL §4.6: suppress edges that repeatedly mislead ranking.
            # Outcome signal provides a strong bonus when outcome == "failure"
            # but suppression can also fire during success if a competitor
            # was over-supported for the wrong reason.
            value = (
                0.30 * rank_signal
                + 0.25 * contribution_signal
                + 0.20 * outcome_signal
                + 0.10 * support_signal
                + 0.10 * confidence_signal
                + 0.05 * advice_signal
            )
            accepted = (
                existing is not None
                and confidence_signal >= self.config.min_existing_confidence_for_suppress
                and value >= self.config.suppress_threshold
            )
            rationale = (
                "Suppress when a confident existing path contributed to a misleading ranking. "
                "Failure outcome gives a bonus but is not strictly required."
            )
        elif proposal.action == "add_tentative_edge":
            # ACL §4.6: tentative edges bridge gaps discovered after a
            # failure, but can also be added during success if coverage
            # analysis suggests a missing path.
            value = (
                0.25 * rank_signal
                + 0.20 * outcome_signal
                + 0.20 * novelty_signal
                + 0.15 * uncertainty_signal
                + 0.10 * support_signal
                + 0.10 * advice_signal
            )
            accepted = value >= self.config.tentative_threshold
            rationale = (
                "Tentative edges require enough novelty and evidence pressure. "
                "Failure gives a bonus but is not strictly required."
            )
        else:
            value = 0.0
            accepted = False
            rationale = f"Unsupported action: {proposal.action}"

        return WriteScore(
            action=proposal.action,
            value=value,
            accepted=accepted,
            target_memory="semantic" if accepted else "episodic_only",
            components={
                "rank_signal": rank_signal,
                "contribution_signal": contribution_signal,
                "outcome_signal": outcome_signal,
                "support_signal": support_signal,
                "confidence_signal": confidence_signal,
                "novelty_signal": novelty_signal,
                "uncertainty_signal": uncertainty_signal,
                "advice_signal": advice_signal,
            },
            rationale=rationale,
        )

    def _support_signal(self, edge: SemanticEdgeState | None) -> float:
        if edge is None:
            return 0.0
        return min(1.0, edge.support_count / self.config.support_scale)

    def _rank_signal(self, lesson: LessonPayload) -> float:
        if lesson.outcome_type == "success":
            return max(0.0, (self.config.success_top_k - lesson.target_rank + 1) / self.config.success_top_k)
        severity = max(0.0, lesson.target_rank - self.config.success_top_k)
        return min(1.0, severity / 10.0)

    def _outcome_signal(self, lesson: LessonPayload, proposal: EditProposal) -> float:
        """Outcome alignment: how well does the proposal's action match the session outcome?

        Returns a value in [0, 1] where 1 means perfect alignment:
        - reinforce + success → 1.0, reinforce + failure → 0.3
        - suppress  + failure → 1.0, suppress  + success → 0.3
        - tentative + failure → 1.0, tentative + success → 0.2
        """
        is_success = lesson.outcome_type == "success"
        if proposal.action == "reinforce_edge":
            return 1.0 if is_success else 0.3
        if proposal.action == "suppress_edge":
            return 1.0 if not is_success else 0.3
        if proposal.action == "add_tentative_edge":
            return 1.0 if not is_success else 0.2
        return 0.0

    def _advice_signal(self, lesson: LessonPayload, proposal: EditProposal) -> float:
        advice = lesson.lesson_advice or {}
        if not advice.get("valid", False):
            return 0.0
        try:
            confidence = max(0.0, min(1.0, float(advice.get("advice_confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            confidence = 0.0
        priority = str(advice.get("priority", "low")).lower()
        priority_weight = {"low": 0.3, "medium": 0.7, "high": 1.0}.get(priority, 0.3)
        edge_hint_types = set(str(item) for item in advice.get("edge_hint_types", []))
        relation_match = 1.0 if proposal.relation in edge_hint_types else 0.5
        return confidence * priority_weight * relation_match
