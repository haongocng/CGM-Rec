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
