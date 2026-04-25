"""Graph-facing schema for Phase 2 seed graph construction."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class GraphNode:
    node_id: str
    node_type: str
    label: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class GraphEdge:
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
class SeedGraph:
    nodes: dict[str, GraphNode]
    edges: list[GraphEdge]
    adjacency: dict[str, list[GraphEdge]]
    reverse_adjacency: dict[str, list[GraphEdge]]

    def node_count_by_type(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes.values():
            counts[node.node_type] = counts.get(node.node_type, 0) + 1
        return counts

    def edge_count_by_relation(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for edge in self.edges:
            counts[edge.relation] = counts.get(edge.relation, 0) + 1
        return counts

    def to_dict(self) -> dict:
        return {
            "summary": {
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
                "nodes_by_type": self.node_count_by_type(),
                "edges_by_relation": self.edge_count_by_relation(),
            },
            "nodes": [node.to_dict() for node in sorted(self.nodes.values(), key=lambda n: n.node_id)],
            "edges": [
                edge.to_dict()
                for edge in sorted(self.edges, key=lambda e: (e.src, e.relation, e.dst))
            ],
        }

    def to_text(self) -> str:
        lines: list[str] = []
        lines.append("CGM-Rec Seed Graph")
        lines.append(f"Nodes: {len(self.nodes)}")
        lines.append(f"Edges: {len(self.edges)}")
        lines.append(f"Nodes by type: {self.node_count_by_type()}")
        lines.append(f"Edges by relation: {self.edge_count_by_relation()}")
        lines.append("")
        lines.append("[Nodes]")
        for node in sorted(self.nodes.values(), key=lambda n: n.node_id):
            lines.append(f"{node.node_id} | type={node.node_type} | label={node.label} | metadata={node.metadata}")
        lines.append("")
        lines.append("[Edges]")
        for edge in sorted(self.edges, key=lambda e: (e.src, e.relation, e.dst)):
            lines.append(
                f"{edge.src} --{edge.relation}--> {edge.dst} "
                f"| weight={edge.weight:.2f} | conf={edge.confidence:.2f} "
                f"| support={edge.support_count} | source={edge.source_kind} | metadata={edge.metadata}"
            )
        return "\n".join(lines)
