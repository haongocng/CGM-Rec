"""Apply diagnostic lessons to semantic and episodic memory."""

from __future__ import annotations

from dataclasses import dataclass

from constants import EDGE_SOURCE_ONLINE, EDGE_SOURCE_TENTATIVE
from memory.episodic_memory import EpisodicMemory
from memory.schema import EditProposal, LessonPayload
from memory.semantic_memory import SemanticEdgeState, SemanticMemory
from memory.write_policy import WritePolicy, WriteScore


@dataclass(slots=True)
class AppliedEdit:
    proposal: EditProposal
    write_score: WriteScore
    applied: bool
    resulting_edge: dict | None = None

    def to_dict(self) -> dict:
        return {
            "proposal": self.proposal.to_dict(),
            "write_score": self.write_score.to_dict(),
            "applied": self.applied,
            "resulting_edge": self.resulting_edge,
        }


@dataclass(slots=True)
class MemoryWriteResult:
    episode_id: str
    applied_edits: list[AppliedEdit]
    rejected_edits: list[AppliedEdit]
    maintenance: dict[str, int]

    def counts(self) -> dict[str, int]:
        output = {
            "stored_episodes": 1,
            "reinforced_edges": 0,
            "suppressed_edges": 0,
            "tentative_edges": 0,
            "rejected_edits": len(self.rejected_edits),
            "promoted_edges": 0,
        }
        for entry in self.applied_edits:
            action = entry.proposal.action
            if action == "reinforce_edge":
                output["reinforced_edges"] += 1
            elif action == "suppress_edge":
                output["suppressed_edges"] += 1
            elif action == "add_tentative_edge":
                output["tentative_edges"] += 1
        return output

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "counts": self.counts(),
            "applied_edits": [entry.to_dict() for entry in self.applied_edits],
            "rejected_edits": [entry.to_dict() for entry in self.rejected_edits],
            "maintenance": self.maintenance,
        }


@dataclass(slots=True)
class MemoryWriterConfig:
    # --- tentative edge management ---
    tentative_weight_decay: float = 0.98
    tentative_confidence_decay: float = 0.99
    prune_weight_threshold: float = 0.02
    prune_confidence_threshold: float = 0.01
    max_tentative_edges: int = 250
    # --- stable edge decay (ACL §5.5) ---
    stable_weight_decay: float = 0.999
    stable_confidence_decay: float = 0.999
    stable_confidence_floor: float = 0.1
    stable_prune_weight_threshold: float = 0.01
    stable_prune_confidence_threshold: float = 0.05
    stable_prune_min_age: int = 50
    stable_prune_min_support: int = 1
    # --- promotion (ACL §5.4) ---
    promotion_min_support: int = 3
    promotion_min_confidence: float = 0.5
    promotion_min_unique_contexts: int = 2
    promotion_check_interval: int = 20


class MemoryWriter:
    def __init__(
        self,
        semantic_memory: SemanticMemory,
        episodic_memory: EpisodicMemory,
        write_policy: WritePolicy | None = None,
        config: MemoryWriterConfig | None = None,
    ):
        self.semantic_memory = semantic_memory
        self.episodic_memory = episodic_memory
        self.write_policy = write_policy or WritePolicy()
        self.config = config or MemoryWriterConfig()
        self.audit_log: list[dict] = []

    def apply_lesson(self, lesson: LessonPayload, step_index: int) -> MemoryWriteResult:
        episode = self.episodic_memory.store_lesson(lesson)
        applied_edits: list[AppliedEdit] = []
        rejected_edits: list[AppliedEdit] = []

        for proposal in lesson.proposed_edits:
            write_score = self.write_policy.evaluate(lesson, proposal, self.semantic_memory)
            if write_score.accepted:
                edge = self._apply_edit(proposal, write_score, step_index)
                applied_edits.append(
                    AppliedEdit(
                        proposal=proposal,
                        write_score=write_score,
                        applied=edge is not None,
                        resulting_edge=edge.to_dict() if edge is not None else None,
                    )
                )
            else:
                rejected_edits.append(
                    AppliedEdit(
                        proposal=proposal,
                        write_score=write_score,
                        applied=False,
                    )
                )

        maintenance = self._run_maintenance(step_index)
        result = MemoryWriteResult(
            episode_id=episode.episode_id,
            applied_edits=applied_edits,
            rejected_edits=rejected_edits,
            maintenance=maintenance,
        )
        self.audit_log.append(
            {
                "step_index": step_index,
                "sample_id": lesson.sample_id,
                "outcome_type": lesson.outcome_type,
                "target_rank": lesson.target_rank,
                "episode_id": episode.episode_id,
                "result": result.to_dict(),
            }
        )
        return result

    def aggregate_counts(self) -> dict[str, int]:
        totals = {
            "stored_episodes": len(self.audit_log),
            "reinforced_edges": 0,
            "suppressed_edges": 0,
            "tentative_edges": 0,
            "rejected_edits": 0,
            "promoted_edges": 0,
        }
        for entry in self.audit_log:
            counts = entry["result"]["counts"]
            for key in totals:
                if key == "stored_episodes":
                    continue
                totals[key] += counts.get(key, 0)
        totals["promoted_edges"] = self._total_promoted
        return totals

    def _apply_edit(self, proposal: EditProposal, write_score: WriteScore, step_index: int) -> SemanticEdgeState | None:
        metadata_updates = {
            "last_action": proposal.action,
            "last_updated_step": step_index,
            "write_score": write_score.value,
        }

        if proposal.action == "add_tentative_edge":
            metadata_updates["tentative"] = True
            metadata_updates["created_step"] = step_index
            return self.semantic_memory.upsert_edge(
                src=proposal.src,
                relation=proposal.relation,
                dst=proposal.dst,
                weight_delta=proposal.weight_delta,
                confidence_delta=proposal.confidence_delta,
                support_delta=max(1, proposal.support_delta),
                source_kind=EDGE_SOURCE_TENTATIVE,
                metadata_updates=metadata_updates,
                create_if_missing=True,
                step_index=step_index,
            )

        edge = self.semantic_memory.upsert_edge(
            src=proposal.src,
            relation=proposal.relation,
            dst=proposal.dst,
            weight_delta=proposal.weight_delta,
            confidence_delta=proposal.confidence_delta,
            support_delta=proposal.support_delta,
            source_kind=EDGE_SOURCE_ONLINE,
            metadata_updates=metadata_updates,
            create_if_missing=False,
            step_index=step_index,
        )
        if edge is not None and edge.metadata.get("tentative") and proposal.action == "reinforce_edge":
            edge.metadata["tentative"] = False
            edge.source_kind = EDGE_SOURCE_ONLINE
        return edge

    # ------------------------------------------------------------------
    # Maintenance: decay, pruning, promotion
    # ------------------------------------------------------------------

    @property
    def _total_promoted(self) -> int:
        """Count edges that were promoted from tentative/episodic to stable."""
        count = 0
        for edge in self.semantic_memory.iter_edges():
            if edge.metadata.get("promoted"):
                count += 1
        return count

    def _run_maintenance(self, step_index: int) -> dict[str, int]:
        decayed = 0
        pruned = 0
        promoted = 0

        # --- Tentative edge decay and pruning ---
        tentative_edges = [
            edge
            for edge in self.semantic_memory.iter_edges()
            if edge.metadata.get("tentative")
        ]

        for edge in tentative_edges:
            if edge.last_update_step < step_index:
                edge.weight *= self.config.tentative_weight_decay
                edge.confidence *= self.config.tentative_confidence_decay
                decayed += 1

        for edge in list(tentative_edges):
            if edge.weight < self.config.prune_weight_threshold or edge.confidence < self.config.prune_confidence_threshold:
                if self.semantic_memory.remove_edge(edge.src, edge.relation, edge.dst):
                    pruned += 1

        remaining_tentative = [
            edge
            for edge in self.semantic_memory.iter_edges()
            if edge.metadata.get("tentative")
        ]
        if len(remaining_tentative) > self.config.max_tentative_edges:
            ranked = sorted(
                remaining_tentative,
                key=lambda edge: (
                    edge.last_update_step,
                    edge.confidence,
                    edge.weight,
                ),
            )
            overflow = len(remaining_tentative) - self.config.max_tentative_edges
            for edge in ranked[:overflow]:
                if self.semantic_memory.remove_edge(edge.src, edge.relation, edge.dst):
                    pruned += 1

        # --- Stable edge decay (ACL §5.5, eq. 43-44) ---
        stable_decayed = 0
        stable_pruned = 0
        stable_edges = [
            edge
            for edge in self.semantic_memory.iter_edges()
            if not edge.metadata.get("tentative")
        ]

        for edge in stable_edges:
            if edge.last_update_step < step_index:
                edge.weight *= self.config.stable_weight_decay
                edge.confidence = max(
                    self.config.stable_confidence_floor,
                    edge.confidence * self.config.stable_confidence_decay,
                )
                stable_decayed += 1

        for edge in list(stable_edges):
            age = step_index - edge.last_update_step
            if (
                edge.weight < self.config.stable_prune_weight_threshold
                and edge.confidence < self.config.stable_prune_confidence_threshold
                and age > self.config.stable_prune_min_age
                and edge.support_count < self.config.stable_prune_min_support
            ):
                if self.semantic_memory.remove_edge(edge.src, edge.relation, edge.dst):
                    stable_pruned += 1

        # --- Episodic promotion (ACL §5.4, eq. 37-41) ---
        if step_index > 0 and step_index % self.config.promotion_check_interval == 0:
            promoted = self._run_promotion(step_index)

        return {
            "decayed_tentative_edges": decayed,
            "pruned_tentative_edges": pruned,
            "decayed_stable_edges": stable_decayed,
            "pruned_stable_edges": stable_pruned,
            "promoted_edges": promoted,
        }

    def _run_promotion(self, step_index: int) -> int:
        """Promote recurring episodic co-occurrence patterns into stable semantic edges (ACL §5.4).

        Scans episodic records for (session_item, target_item) pairs that appear
        across enough distinct episodes with consistent success signals.  When
        thresholds are met, the tentative / episodic evidence is promoted into
        a stable semantic edge.
        """
        from collections import Counter

        promoted = 0

        # Aggregate how many episodes support each (session_item -> target) pair
        pair_support: Counter[tuple[str, str]] = Counter()
        pair_contexts: dict[tuple[str, str], set[str]] = {}
        pair_successes: dict[tuple[str, str], int] = {}

        for record in self.episodic_memory.records:
            target_id = f"item::{record.target_item.strip().lower()}"
            for session_item in record.session_items:
                src_id = f"item::{session_item.strip().lower()}"
                if src_id == target_id:
                    continue
                pair = (src_id, target_id)
                pair_support[pair] += 1
                pair_contexts.setdefault(pair, set()).add(record.sample_id)
                if record.outcome_type == "success":
                    pair_successes[pair] = pair_successes.get(pair, 0) + 1

        for pair, support in pair_support.items():
            if support < self.config.promotion_min_support:
                continue
            unique_contexts = len(pair_contexts.get(pair, set()))
            if unique_contexts < self.config.promotion_min_unique_contexts:
                continue

            src_id, dst_id = pair
            existing = self.semantic_memory.get_edge(src_id, "co_occurs_with", dst_id)

            # Check if already promoted or already a strong stable edge
            if existing and not existing.metadata.get("tentative") and existing.confidence >= self.config.promotion_min_confidence:
                continue

            success_count = pair_successes.get(pair, 0)
            confidence = min(1.0, 0.4 + 0.1 * support + 0.1 * success_count)
            if confidence < self.config.promotion_min_confidence:
                continue

            edge = self.semantic_memory.upsert_edge(
                src=src_id,
                relation="co_occurs_with",
                dst=dst_id,
                weight_delta=float(support) * 0.2 if existing is None else 0.1,
                confidence_delta=confidence if existing is None else 0.05,
                support_delta=support,
                source_kind=EDGE_SOURCE_ONLINE,
                metadata_updates={
                    "tentative": False,
                    "promoted": True,
                    "promoted_at_step": step_index,
                    "promotion_support": support,
                    "promotion_unique_contexts": unique_contexts,
                },
                create_if_missing=True,
                step_index=step_index,
            )
            if edge is not None:
                promoted += 1

        return promoted
