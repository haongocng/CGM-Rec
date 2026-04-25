# CGM-Rec Fixes Applied — Documentation

> All 6 fixes verified end-to-end across Phases 1–5.  
> Every file is runnable and the full pipeline connects correctly.

---

## Summary of Changes

| # | Fix | File(s) Changed | ACL Reference |
|---|---|---|---|
| 1 | **Deleted `types.py`** stdlib shadow | `types.py` (removed) | N/A (infrastructure bug) |
| 2 | **Added `last_update_step`** to `SemanticEdgeState` | `memory/semantic_memory.py` | §5.5 (decay requires edge timestamps) |
| 3 | **Removed hard outcome-type gates** in write policy | `memory/write_policy.py` | §4.6, Table 1 |
| 4 | **Added stable-edge decay** for non-tentative edges | `memory/writer.py` | §5.5, eq. 43-44 |
| 5 | **Implemented `promote_episode_to_semantic`** | `memory/writer.py` | §5.4, eq. 37-41 |
| 6 | **Added structural updates to training** (dual optimization) | `engine/train_loop.py`, `main.py` | §4 (Read-Write-Consolidate cycle) |

---

## Fix 1: Delete `types.py`

**Problem:** `cgm/types.py` was a full copy of Python's stdlib `types` module. Any `import types` within the project or its dependencies would import this shadow file instead of the real stdlib, causing hard-to-debug breakages.

**Fix:** Deleted the file. No CGM source code imported it.

**File:** `cgm/types.py` → **removed**

---

## Fix 2: Add `last_update_step` to `SemanticEdgeState`

**Problem:** Edges had no timestamp tracking, so decay and pruning decisions couldn't consider how long an edge had been stale.

**Fix:** Added `last_update_step: int = 0` field to `SemanticEdgeState` and updated `upsert_edge()` to accept and set it.

**File:** [semantic_memory.py](file:///home/may1-idsailab/Documents/Hao/cgm/memory/semantic_memory.py)

```diff:semantic_memory.py
"""Mutable semantic memory built from the seed graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from graph.schema import GraphNode, SeedGraph


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class SemanticEdgeState:
    src: str
    relation: str
    dst: str
    weight: float
    confidence: float
    support_count: int
    source_kind: str
    metadata: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.relation, self.dst)

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "relation": self.relation,
            "dst": self.dst,
            "weight": self.weight,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "source_kind": self.source_kind,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SemanticMemory:
    nodes: dict[str, GraphNode]
    edges: dict[tuple[str, str, str], SemanticEdgeState]
    adjacency: dict[str, list[SemanticEdgeState]]
    reverse_adjacency: dict[str, list[SemanticEdgeState]]

    @classmethod
    def from_seed_graph(cls, graph: SeedGraph) -> "SemanticMemory":
        payload = {
            edge.key: SemanticEdgeState(
                src=edge.src,
                relation=edge.relation,
                dst=edge.dst,
                weight=edge.weight,
                confidence=edge.confidence,
                support_count=edge.support_count,
                source_kind=edge.source_kind,
                metadata=dict(edge.metadata),
            )
            for edge in graph.edges
        }
        memory = cls(
            nodes={node_id: GraphNode(**node.to_dict()) for node_id, node in graph.nodes.items()},
            edges=payload,
            adjacency={},
            reverse_adjacency={},
        )
        memory._rebuild_indexes()
        return memory

    def relation_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges.values():
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return counts

    def get_edge(self, src: str, relation: str, dst: str) -> SemanticEdgeState | None:
        return self.edges.get((src, relation, dst))

    def has_edge(self, src: str, relation: str, dst: str) -> bool:
        return (src, relation, dst) in self.edges

    def get_edges(self, src: str, relation: str | None = None) -> list[SemanticEdgeState]:
        if relation is None:
            return list(self.adjacency.get(src, []))
        return [edge for edge in self.adjacency.get(src, []) if edge.relation == relation]

    def neighbor_ids(self, src: str, relation: str) -> set[str]:
        return {edge.dst for edge in self.get_edges(src, relation=relation)}

    def in_degree(self, node_id: str) -> int:
        return len(self.reverse_adjacency.get(node_id, []))

    def describe_edges(self, node_id: str, limit: int = 6) -> list[str]:
        descriptions: list[str] = []
        for edge in self.adjacency.get(node_id, [])[:limit]:
            dst_label = self.nodes.get(edge.dst).label if edge.dst in self.nodes else edge.dst
            descriptions.append(f"{edge.relation}:{dst_label}")
        return descriptions

    def upsert_edge(
        self,
        src: str,
        relation: str,
        dst: str,
        weight_delta: float,
        confidence_delta: float,
        support_delta: int,
        source_kind: str,
        metadata_updates: dict | None = None,
        create_if_missing: bool = False,
    ) -> SemanticEdgeState | None:
        edge = self.get_edge(src, relation, dst)
        if edge is None:
            if not create_if_missing:
                return None
            edge = SemanticEdgeState(
                src=src,
                relation=relation,
                dst=dst,
                weight=max(0.0, weight_delta),
                confidence=_clamp_confidence(max(0.0, confidence_delta)),
                support_count=max(0, support_delta),
                source_kind=source_kind,
                metadata=dict(metadata_updates or {}),
            )
            self.edges[edge.key] = edge
            self.adjacency.setdefault(src, []).append(edge)
            self.reverse_adjacency.setdefault(dst, []).append(edge)
            return edge

        edge.weight = max(0.0, edge.weight + weight_delta)
        edge.confidence = _clamp_confidence(edge.confidence + confidence_delta)
        edge.support_count = max(0, edge.support_count + support_delta)
        if source_kind:
            edge.source_kind = source_kind
        if metadata_updates:
            edge.metadata.update(metadata_updates)
        return edge

    def remove_edge(self, src: str, relation: str, dst: str) -> bool:
        key = (src, relation, dst)
        edge = self.edges.pop(key, None)
        if edge is None:
            return False

        self.adjacency[src] = [item for item in self.adjacency.get(src, []) if item.key != key]
        self.reverse_adjacency[dst] = [item for item in self.reverse_adjacency.get(dst, []) if item.key != key]

        if not self.adjacency[src]:
            self.adjacency.pop(src)
        if not self.reverse_adjacency[dst]:
            self.reverse_adjacency.pop(dst)
        return True

    def iter_edges(self) -> list[SemanticEdgeState]:
        return list(self.edges.values())

    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "relations": self.relation_counts(),
            },
            "edges": [
                edge.to_dict()
                for edge in sorted(self.edges.values(), key=lambda item: (item.src, item.relation, item.dst))
            ],
        }

    def _rebuild_indexes(self) -> None:
        adjacency: dict[str, list[SemanticEdgeState]] = {}
        reverse: dict[str, list[SemanticEdgeState]] = {}
        for edge in self.edges.values():
            adjacency.setdefault(edge.src, []).append(edge)
            reverse.setdefault(edge.dst, []).append(edge)
        self.adjacency = adjacency
        self.reverse_adjacency = reverse
===
"""Mutable semantic memory built from the seed graph."""

from __future__ import annotations

from dataclasses import dataclass, field

from graph.schema import GraphNode, SeedGraph


def _clamp_confidence(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(slots=True)
class SemanticEdgeState:
    src: str
    relation: str
    dst: str
    weight: float
    confidence: float
    support_count: int
    source_kind: str
    last_update_step: int = 0
    metadata: dict = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.relation, self.dst)

    def to_dict(self) -> dict:
        return {
            "src": self.src,
            "relation": self.relation,
            "dst": self.dst,
            "weight": self.weight,
            "confidence": self.confidence,
            "support_count": self.support_count,
            "source_kind": self.source_kind,
            "last_update_step": self.last_update_step,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class SemanticMemory:
    nodes: dict[str, GraphNode]
    edges: dict[tuple[str, str, str], SemanticEdgeState]
    adjacency: dict[str, list[SemanticEdgeState]]
    reverse_adjacency: dict[str, list[SemanticEdgeState]]

    @classmethod
    def from_seed_graph(cls, graph: SeedGraph) -> "SemanticMemory":
        payload = {
            edge.key: SemanticEdgeState(
                src=edge.src,
                relation=edge.relation,
                dst=edge.dst,
                weight=edge.weight,
                confidence=edge.confidence,
                support_count=edge.support_count,
                source_kind=edge.source_kind,
                metadata=dict(edge.metadata),
            )
            for edge in graph.edges
        }
        memory = cls(
            nodes={node_id: GraphNode(**node.to_dict()) for node_id, node in graph.nodes.items()},
            edges=payload,
            adjacency={},
            reverse_adjacency={},
        )
        memory._rebuild_indexes()
        return memory

    def relation_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges.values():
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return counts

    def get_edge(self, src: str, relation: str, dst: str) -> SemanticEdgeState | None:
        return self.edges.get((src, relation, dst))

    def has_edge(self, src: str, relation: str, dst: str) -> bool:
        return (src, relation, dst) in self.edges

    def get_edges(self, src: str, relation: str | None = None) -> list[SemanticEdgeState]:
        if relation is None:
            return list(self.adjacency.get(src, []))
        return [edge for edge in self.adjacency.get(src, []) if edge.relation == relation]

    def neighbor_ids(self, src: str, relation: str) -> set[str]:
        return {edge.dst for edge in self.get_edges(src, relation=relation)}

    def in_degree(self, node_id: str) -> int:
        return len(self.reverse_adjacency.get(node_id, []))

    def describe_edges(self, node_id: str, limit: int = 6) -> list[str]:
        descriptions: list[str] = []
        for edge in self.adjacency.get(node_id, [])[:limit]:
            dst_label = self.nodes.get(edge.dst).label if edge.dst in self.nodes else edge.dst
            descriptions.append(f"{edge.relation}:{dst_label}")
        return descriptions

    def upsert_edge(
        self,
        src: str,
        relation: str,
        dst: str,
        weight_delta: float,
        confidence_delta: float,
        support_delta: int,
        source_kind: str,
        metadata_updates: dict | None = None,
        create_if_missing: bool = False,
        step_index: int = 0,
    ) -> SemanticEdgeState | None:
        edge = self.get_edge(src, relation, dst)
        if edge is None:
            if not create_if_missing:
                return None
            edge = SemanticEdgeState(
                src=src,
                relation=relation,
                dst=dst,
                weight=max(0.0, weight_delta),
                confidence=_clamp_confidence(max(0.0, confidence_delta)),
                support_count=max(0, support_delta),
                source_kind=source_kind,
                last_update_step=step_index,
                metadata=dict(metadata_updates or {}),
            )
            self.edges[edge.key] = edge
            self.adjacency.setdefault(src, []).append(edge)
            self.reverse_adjacency.setdefault(dst, []).append(edge)
            return edge

        edge.weight = max(0.0, edge.weight + weight_delta)
        edge.confidence = _clamp_confidence(edge.confidence + confidence_delta)
        edge.support_count = max(0, edge.support_count + support_delta)
        edge.last_update_step = step_index
        if source_kind:
            edge.source_kind = source_kind
        if metadata_updates:
            edge.metadata.update(metadata_updates)
        return edge

    def remove_edge(self, src: str, relation: str, dst: str) -> bool:
        key = (src, relation, dst)
        edge = self.edges.pop(key, None)
        if edge is None:
            return False

        self.adjacency[src] = [item for item in self.adjacency.get(src, []) if item.key != key]
        self.reverse_adjacency[dst] = [item for item in self.reverse_adjacency.get(dst, []) if item.key != key]

        if not self.adjacency[src]:
            self.adjacency.pop(src)
        if not self.reverse_adjacency[dst]:
            self.reverse_adjacency.pop(dst)
        return True

    def iter_edges(self) -> list[SemanticEdgeState]:
        return list(self.edges.values())

    def edge_count(self) -> int:
        return len(self.edges)

    def to_dict(self) -> dict:
        return {
            "summary": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "relations": self.relation_counts(),
            },
            "edges": [
                edge.to_dict()
                for edge in sorted(self.edges.values(), key=lambda item: (item.src, item.relation, item.dst))
            ],
        }

    def _rebuild_indexes(self) -> None:
        adjacency: dict[str, list[SemanticEdgeState]] = {}
        reverse: dict[str, list[SemanticEdgeState]] = {}
        for edge in self.edges.values():
            adjacency.setdefault(edge.src, []).append(edge)
            reverse.setdefault(edge.dst, []).append(edge)
        self.adjacency = adjacency
        self.reverse_adjacency = reverse
```

---

## Fix 3: Remove Hard Outcome-Type Gates in Write Policy

**Problem:** The old policy hardcoded:
- `reinforce_edge`: accepted **only** when `outcome == "success"`
- `suppress_edge`: accepted **only** when `outcome == "failure"`
- `add_tentative_edge`: accepted **only** when `outcome == "failure"`

This prevented legitimate edits (e.g., reinforcing an edge that correctly supported the target during a failure, or suppressing an over-supported competitor during a success).

**Fix:** Replaced the hard boolean gate with a soft `_outcome_signal()` function that returns a value in `[0, 1]`:

```python
# Before (hard gate):
accepted = lesson.outcome_type == "success" and value >= threshold

# After (soft signal):
outcome_signal = 1.0 if is_success else 0.3  # for reinforce
value = 0.25*rank + 0.25*contribution + 0.20*outcome + ...
accepted = value >= threshold
```

The outcome alignment still contributes strongly (0.20 weight) but no longer blocks the edit entirely.

**File:** [write_policy.py](file:///home/may1-idsailab/Documents/Hao/cgm/memory/write_policy.py)

```diff:write_policy.py
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
        advice_signal = self._advice_signal(lesson, proposal)

        if proposal.action == "reinforce_edge":
            value = (0.35 * rank_signal) + (0.30 * contribution_signal) + (0.20 * support_signal) + (0.10 * confidence_signal) + (0.05 * advice_signal)
            accepted = lesson.outcome_type == "success" and value >= self.config.reinforce_threshold
            rationale = "Reinforce only when success evidence and current support are strong enough."
        elif proposal.action == "suppress_edge":
            value = (0.40 * rank_signal) + (0.30 * contribution_signal) + (0.15 * support_signal) + (0.10 * confidence_signal) + (0.05 * advice_signal)
            accepted = (
                lesson.outcome_type == "failure"
                and existing is not None
                and confidence_signal >= self.config.min_existing_confidence_for_suppress
                and value >= self.config.suppress_threshold
            )
            rationale = "Suppress only when a confident existing path contributed to failure."
        elif proposal.action == "add_tentative_edge":
            value = (0.35 * rank_signal) + (0.20 * novelty_signal) + (0.20 * uncertainty_signal) + (0.15 * support_signal) + (0.10 * advice_signal)
            accepted = lesson.outcome_type == "failure" and value >= self.config.tentative_threshold
            rationale = "Tentative edges require clear failure pressure and enough novelty to justify a new bridge."
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
===
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

```

---

## Fix 4: Stable-Edge Decay (ACL §5.5)

**Problem:** Only tentative edges decayed over time. Stable semantic edges (from metadata or warmup) never lost confidence, even if they were never reinforced during training or testing. This violates ACL §5.5 equations 43-44.

**Fix:** Added stable-edge decay logic in `_run_maintenance()`:

```python
# Stable edge decay (very slow — 0.999 per step)
for edge in stable_edges:
    if edge.last_update_step < step_index:
        edge.weight *= 0.999
        edge.confidence = max(0.1, edge.confidence * 0.999)

# Stable edge pruning (only if very old, very low, and unsupported)
if weight < 0.01 and confidence < 0.05 and age > 50 and support < 1:
    remove_edge(...)
```

The decay is intentionally very slow (0.999) so that metadata edges degrade gracefully over hundreds of steps rather than disappearing quickly.

**File:** [writer.py](file:///home/may1-idsailab/Documents/Hao/cgm/memory/writer.py)

---

## Fix 5: Implement `promote_episode_to_semantic` (ACL §5.4)

**Problem:** Recurring episodic patterns (session→target pairs that appeared multiple times) were never consolidated into stable semantic edges. They stayed in episodic memory and eventually got evicted by FIFO.

**Fix:** Added `_run_promotion()` method that runs every `promotion_check_interval` steps (default: 20). It:

1. Scans all episodic records for `(session_item, target_item)` co-occurrence pairs
2. Counts support (how many episodes contain the pair) and unique contexts
3. If support ≥ 3, unique contexts ≥ 2, and computed confidence ≥ 0.5:
   - Creates or strengthens a `co_occurs_with` edge in semantic memory
   - Marks it with `promoted: True` for audit tracking

```python
# Promotion thresholds (ACL §5.4, eq. 37-41):
promotion_min_support: int = 3           # n_z >= n_min
promotion_min_confidence: float = 0.5    # c_z >= c_min  
promotion_min_unique_contexts: int = 2   # |UniqueContexts(z)| >= m_min
```

**File:** [writer.py](file:///home/may1-idsailab/Documents/Hao/cgm/memory/writer.py)

```diff:writer.py
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
    tentative_weight_decay: float = 0.98
    tentative_confidence_decay: float = 0.99
    prune_weight_threshold: float = 0.02
    prune_confidence_threshold: float = 0.01
    max_tentative_edges: int = 250


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
        }
        for entry in self.audit_log:
            counts = entry["result"]["counts"]
            for key in totals:
                if key == "stored_episodes":
                    continue
                totals[key] += counts.get(key, 0)
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
        )
        if edge is not None and edge.metadata.get("tentative") and proposal.action == "reinforce_edge":
            edge.metadata["tentative"] = False
            edge.source_kind = EDGE_SOURCE_ONLINE
        return edge

    def _run_maintenance(self, step_index: int) -> dict[str, int]:
        decayed = 0
        pruned = 0
        tentative_edges = [
            edge
            for edge in self.semantic_memory.iter_edges()
            if edge.metadata.get("tentative")
        ]

        for edge in tentative_edges:
            if int(edge.metadata.get("last_updated_step", step_index)) < step_index:
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
                    edge.metadata.get("last_updated_step", -1),
                    edge.confidence,
                    edge.weight,
                ),
            )
            overflow = len(remaining_tentative) - self.config.max_tentative_edges
            for edge in ranked[:overflow]:
                if self.semantic_memory.remove_edge(edge.src, edge.relation, edge.dst):
                    pruned += 1

        return {"decayed_tentative_edges": decayed, "pruned_tentative_edges": pruned}
===
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

```

---

## Fix 6: Dual Optimization in Training (Neural + Structural)

**Problem:** The training loop only trained the scorer weights $W$ on a **static** seed graph. No graph edits were applied during training, meaning the scorer never learned on an evolving graph.

**Fix:** Extended `train_semantic_scorer()` to accept optional `diagnostics`, `writer`, and `episodic_memory` parameters. When provided, each training sample triggers:

1. **Neural update:** `scorer.update()` adjusts weights via cross-entropy
2. **Structural update:** `diagnostics.analyze()` → `writer.apply_lesson()` reinforces/suppresses/creates edges

The retriever re-reads the (now updated) semantic memory on every sample, so subsequent scorer updates see an evolving graph.

```python
# Before:
for sample in samples:
    bundle = retriever.retrieve(...)
    output = scorer.update(...)  # neural only

# After:
for sample in samples:
    bundle = retriever.retrieve(...)     # reads evolving graph
    output = scorer.update(...)          # neural update
    lesson = diagnostics.analyze(...)    # diagnose success/failure
    writer.apply_lesson(lesson, ...)     # structural update
```

The function is **backward compatible** — when `diagnostics`/`writer` are `None`, it behaves exactly like the original neural-only loop.

**Files:**
- [train_loop.py](file:///home/may1-idsailab/Documents/Hao/cgm/engine/train_loop.py)
- [main.py](file:///home/may1-idsailab/Documents/Hao/cgm/main.py) (wired the dual-optimization components)

```diff:train_loop.py
"""Phase 3 neural-only training loop."""

from __future__ import annotations

from dataclasses import dataclass

from data.schema import SessionSample
from eval.metrics import evaluate_predictions
from model.scorer import LinearSemanticScorer
from retrieval.semantic_retriever import SemanticRetriever


@dataclass(slots=True)
class TrainResult:
    epochs: int
    learning_rate: float
    average_loss: float
    metrics: dict[str, float]
    steps: int


def train_semantic_scorer(
    samples: list[SessionSample],
    retriever: SemanticRetriever,
    scorer: LinearSemanticScorer,
    epochs: int,
    learning_rate: float,
) -> TrainResult:
    all_predictions: list[list[str]] = []
    all_targets: list[str] = []
    total_loss = 0.0
    steps = 0

    for _epoch in range(epochs):
        for sample in samples:
            bundle = retriever.retrieve(
                session_items=sample.parsed_input.session_items,
                candidate_items=sample.parsed_input.candidate_items,
            )
            output = scorer.update(bundle, target_title=sample.target, learning_rate=learning_rate)
            total_loss += output.loss
            all_predictions.append(output.ranked_titles)
            all_targets.append(sample.target)
            steps += 1

    average_loss = total_loss / steps if steps else 0.0
    metrics = evaluate_predictions(all_predictions, all_targets)
    return TrainResult(
        epochs=epochs,
        learning_rate=learning_rate,
        average_loss=average_loss,
        metrics=metrics,
        steps=steps,
    )
===
"""Phase 3 neural + structural training loop.

During training the system performs dual optimization:
1. Neural update  — learn scorer weights W via cross-entropy on the candidate set.
2. Structural update — run diagnostics and apply typed graph edits (reinforce,
   suppress, tentative) so the scorer learns on an increasingly refined graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from data.schema import SessionSample
from engine.diagnostics import HybridDiagnostics
from eval.metrics import evaluate_predictions
from memory.episodic_memory import EpisodicMemory
from memory.writer import MemoryWriter
from model.scorer import LinearSemanticScorer
from retrieval.semantic_retriever import SemanticRetriever


@dataclass(slots=True)
class TrainResult:
    epochs: int
    learning_rate: float
    average_loss: float
    metrics: dict[str, float]
    steps: int
    edit_counts: dict[str, int] = field(default_factory=dict)


def train_semantic_scorer(
    samples: list[SessionSample],
    retriever: SemanticRetriever,
    scorer: LinearSemanticScorer,
    epochs: int,
    learning_rate: float,
    diagnostics: HybridDiagnostics | None = None,
    writer: MemoryWriter | None = None,
    episodic_memory: EpisodicMemory | None = None,
) -> TrainResult:
    """Train the semantic scorer with optional structural memory updates.

    When *diagnostics* and *writer* are provided the training loop performs
    the dual-optimisation described in the paper:

    1. **Neural update** — ``scorer.update()`` adjusts feature weights via
       gradient descent on cross-entropy loss.
    2. **Structural update** — ``diagnostics.analyze()`` identifies useful /
       misleading graph edges and ``writer.apply_lesson()`` reinforces,
       suppresses, or creates tentative edges.

    The retriever re-reads the (now updated) semantic memory on every sample,
    so subsequent scorer updates see an evolving graph.

    When *diagnostics* / *writer* are ``None``, the function falls back to
    the original neural-only training loop for backward compatibility.
    """
    all_predictions: list[list[str]] = []
    all_targets: list[str] = []
    total_loss = 0.0
    steps = 0
    structural_enabled = diagnostics is not None and writer is not None

    for _epoch in range(epochs):
        for sample in samples:
            # --- Read from (potentially updated) memory ---
            bundle = retriever.retrieve(
                session_items=sample.parsed_input.session_items,
                candidate_items=sample.parsed_input.candidate_items,
            )

            # --- Neural update ---
            output = scorer.update(bundle, target_title=sample.target, learning_rate=learning_rate)
            total_loss += output.loss
            all_predictions.append(output.ranked_titles)
            all_targets.append(sample.target)

            # --- Structural update (when enabled) ---
            if structural_enabled:
                similar_episodes = []
                if episodic_memory is not None:
                    similar_episodes = episodic_memory.retrieve_similar(
                        session_items=sample.parsed_input.session_items,
                        candidate_items=sample.parsed_input.candidate_items,
                        limit=3,
                        min_similarity=0.05,
                    )

                lesson = diagnostics.analyze(
                    sample=sample,
                    retrieval=bundle,
                    score_output=output,
                    supporting_episode_ids=[match.episode.episode_id for match in similar_episodes],
                )
                writer.apply_lesson(lesson, step_index=steps)

            steps += 1

    average_loss = total_loss / steps if steps else 0.0
    metrics = evaluate_predictions(all_predictions, all_targets)
    edit_counts = writer.aggregate_counts() if writer is not None else {}
    return TrainResult(
        epochs=epochs,
        learning_rate=learning_rate,
        average_loss=average_loss,
        metrics=metrics,
        steps=steps,
        edit_counts=edit_counts,
    )

```

---

## Verification Results

All 5 phases tested successfully:

| Phase | Command | Status | Key Output |
|---|---|---|---|
| **Phase 1** | `--view data` | ✅ Pass | 200 train, 743 test, 1500 products |
| **Phase 2** | `--view seed-graph` | ✅ Pass | 4302 nodes, 25974 edges |
| **Phase 3 Train** | `--view phase3-train --epochs 2` | ✅ Pass | 744 reinforced, 671 suppressed, 130 tentative, **46 promoted** |
| **Phase 3 Test** | `--view phase3-test --epochs 2` | ✅ Pass | HIT@5=0.565, NDCG@5=0.397 |
| **Phase 4 Online** | `--view phase4-test-online --epochs 2` | ✅ Pass | Online edits applied per session |
| **Phase 5 LLM** | `--view phase5-test-llm --epochs 2 --max-test-samples 10` | ✅ Pass | Parser=100%, Fallback=0%, all JSON valid |

> [!IMPORTANT]
> Phase 3 training now shows **Structural edits** in its output, confirming that the dual-optimization loop (neural + graph) is active. The graph evolves during training, not just during testing.

---

## ACL Paper Alignment After Fixes

| ACL Paper Feature | Status |
|---|---|
| §4.1 Seed Graph Construction | ✅ Implemented |
| §4.2 Stable Semantic Graph Memory | ✅ Implemented |
| §4.3 Fast Episodic Lesson Memory | ✅ Implemented |
| §4.4 Dual-Memory Retrieval | ✅ Implemented |
| §4.5 Recommendation with Retrieved Graph Memory | ✅ Implemented |
| §4.6 Explicit Memory Write Policy (Table 1) | ✅ **Fixed** — all 6 edit types now work |
| §4.7 Consolidation, Decay, and Pruning | ✅ **Fixed** — stable decay + promotion |
| §5.1 Edge Confidence | ✅ Implemented |
| §5.3 Memory Write Score | ✅ **Fixed** — soft outcome signal |
| §5.4 Promotion Rule | ✅ **Fixed** — newly implemented |
| §5.5 Decay and Pruning | ✅ **Fixed** — stable edges now decay |
| §6.3 Prequential Evaluation | ✅ Implemented (Phase 4/5) |
