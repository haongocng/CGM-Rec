"""Seed graph construction from metadata and warm-up behavior."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

from constants import (
    EDGE_REL_BELONGS_TO,
    EDGE_REL_CO_OCCURS,
    EDGE_REL_HAS_DESCRIPTION,
    EDGE_REL_HAS_KEYWORD,
    EDGE_SOURCE_METADATA,
    EDGE_SOURCE_WARMUP,
    NODE_TYPE_CATEGORY,
    NODE_TYPE_DESCRIPTION,
    NODE_TYPE_ITEM,
    NODE_TYPE_KEYWORD,
)
from data.schema import ProductInfo, SessionSample
from graph.schema import GraphEdge, GraphNode, SeedGraph
from utils.text import normalize_title


class SeedGraphBuilder:
    def __init__(self, keyword_top_k: int = 5, include_description: bool = True, co_occur_window_size: int = 5):
        self.keyword_top_k = keyword_top_k
        self.include_description = include_description
        self.co_occur_window_size = max(2, co_occur_window_size)

    def build(self, products: dict[str, ProductInfo], warmup_samples: list[SessionSample]) -> SeedGraph:
        nodes: dict[str, GraphNode] = {}
        edge_map: dict[tuple[str, str, str], GraphEdge] = {}

        def ensure_node(node_id: str, node_type: str, label: str, metadata: dict | None = None) -> None:
            if node_id not in nodes:
                nodes[node_id] = GraphNode(
                    node_id=node_id,
                    node_type=node_type,
                    label=label,
                    metadata=metadata or {},
                )

        def add_edge(
            src: str,
            relation: str,
            dst: str,
            weight: float,
            confidence: float,
            support_count: int,
            source_kind: str,
            metadata: dict | None = None,
        ) -> None:
            key = (src, relation, dst)
            if key in edge_map:
                edge = edge_map[key]
                edge.support_count += support_count
                edge.weight += weight
                edge.confidence = max(edge.confidence, confidence)
            else:
                edge_map[key] = GraphEdge(
                    src=src,
                    relation=relation,
                    dst=dst,
                    weight=weight,
                    confidence=confidence,
                    support_count=support_count,
                    source_kind=source_kind,
                    metadata=metadata or {},
                )

        for product in products.values():
            item_id = self._item_id(product.title)
            ensure_node(item_id, NODE_TYPE_ITEM, product.title, {"normalized_title": product.normalized_title})

            for level_name, label in sorted(product.taxonomy_levels.items()):
                category_id = self._category_id(level_name, label)
                ensure_node(category_id, NODE_TYPE_CATEGORY, label, {"taxonomy_level": level_name})
                add_edge(
                    src=item_id,
                    relation=EDGE_REL_BELONGS_TO,
                    dst=category_id,
                    weight=1.0,
                    confidence=1.0,
                    support_count=1,
                    source_kind=EDGE_SOURCE_METADATA,
                    metadata={"taxonomy_level": level_name},
                )

            for rank, keyword in enumerate(product.keywords[: self.keyword_top_k]):
                keyword_id = self._keyword_id(keyword)
                ensure_node(keyword_id, NODE_TYPE_KEYWORD, keyword, {"rank": rank})
                add_edge(
                    src=item_id,
                    relation=EDGE_REL_HAS_KEYWORD,
                    dst=keyword_id,
                    weight=max(0.2, 1.0 - (rank * 0.1)),
                    confidence=0.9,
                    support_count=1,
                    source_kind=EDGE_SOURCE_METADATA,
                    metadata={"keyword_rank": rank},
                )

            if self.include_description and product.description:
                description_id = self._description_id(product.title)
                ensure_node(description_id, NODE_TYPE_DESCRIPTION, product.title, {})
                add_edge(
                    src=item_id,
                    relation=EDGE_REL_HAS_DESCRIPTION,
                    dst=description_id,
                    weight=1.0,
                    confidence=0.95,
                    support_count=1,
                    source_kind=EDGE_SOURCE_METADATA,
                )

        pair_counts: dict[tuple[str, str], int] = defaultdict(int)
        for sample in warmup_samples:
            item_ids = [self._item_id(title) for title in sample.parsed_input.session_items]
            for idx, src in enumerate(item_ids):
                end = min(len(item_ids), idx + self.co_occur_window_size)
                window_items = item_ids[idx:end]
                for left, right in combinations(window_items, 2):
                    pair = tuple(sorted((left, right)))
                    pair_counts[pair] += 1

        for (left, right), count in pair_counts.items():
            if left not in nodes or right not in nodes:
                continue
            weight = float(count)
            confidence = min(1.0, 0.4 + 0.1 * count)
            add_edge(
                src=left,
                relation=EDGE_REL_CO_OCCURS,
                dst=right,
                weight=weight,
                confidence=confidence,
                support_count=count,
                source_kind=EDGE_SOURCE_WARMUP,
            )
            add_edge(
                src=right,
                relation=EDGE_REL_CO_OCCURS,
                dst=left,
                weight=weight,
                confidence=confidence,
                support_count=count,
                source_kind=EDGE_SOURCE_WARMUP,
            )

        edges = list(edge_map.values())
        adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        reverse: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            adjacency[edge.src].append(edge)
            reverse[edge.dst].append(edge)

        return SeedGraph(
            nodes=nodes,
            edges=edges,
            adjacency=dict(adjacency),
            reverse_adjacency=dict(reverse),
        )

    @staticmethod
    def _item_id(title: str) -> str:
        return f"item::{normalize_title(title)}"

    @staticmethod
    def _category_id(level_name: str, label: str) -> str:
        return f"cat::{level_name.lower()}::{normalize_title(label)}"

    @staticmethod
    def _keyword_id(keyword: str) -> str:
        return f"kw::{normalize_title(keyword)}"

    @staticmethod
    def _description_id(title: str) -> str:
        return f"desc::{normalize_title(title)}"
