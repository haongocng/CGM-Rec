"""PyTorch graph-attention semantic scorer for CGM-Rec.

The scorer intentionally mirrors LinearSemanticScorer's public surface so the
Phase 3/4/5 loops can freeze or update it without changing their contracts.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from constants import (
    EDGE_REL_BELONGS_TO,
    EDGE_REL_CO_OCCURS,
    EDGE_REL_HAS_DESCRIPTION,
    EDGE_REL_HAS_KEYWORD,
    NODE_TYPE_CATEGORY,
    NODE_TYPE_DESCRIPTION,
    NODE_TYPE_ITEM,
    NODE_TYPE_KEYWORD,
    PHASE3_FEATURE_NAMES,
)
from memory.semantic_memory import SemanticEdgeState, SemanticMemory
from model.scorer import CandidateScore, ScoreOutput
from retrieval.semantic_retriever import CandidateEvidence, RetrievalBundle
from utils.text import normalize_title


NODE_TYPES = (
    NODE_TYPE_ITEM,
    NODE_TYPE_CATEGORY,
    NODE_TYPE_KEYWORD,
    NODE_TYPE_DESCRIPTION,
    "unknown",
)
RELATIONS = (
    EDGE_REL_BELONGS_TO,
    EDGE_REL_HAS_KEYWORD,
    EDGE_REL_HAS_DESCRIPTION,
    EDGE_REL_CO_OCCURS,
    f"{EDGE_REL_BELONGS_TO}__rev",
    f"{EDGE_REL_HAS_KEYWORD}__rev",
    f"{EDGE_REL_HAS_DESCRIPTION}__rev",
    f"{EDGE_REL_CO_OCCURS}__rev",
)


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "GraphAttentionSemanticScorer requires PyTorch. Install torch first, "
            "or run with --scorer-type linear."
        ) from exc
    return torch


def resolve_torch_device(requested: str = "auto") -> str:
    torch = _require_torch()
    normalized = (requested or "auto").lower()
    if normalized == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is False.")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device '{requested}'. Use auto, cuda, or cpu.")
    return normalized


@dataclass(slots=True)
class _SubgraphBatch:
    node_type_ids: Any
    node_degree_features: Any
    edge_src: Any
    edge_dst: Any
    edge_relation_ids: Any
    edge_features: Any
    session_indices: Any
    candidate_indices: Any
    candidate_features: Any
    candidate_evidence: list[CandidateEvidence]


class _TorchGraphAttentionModel:
    """Small dependency-light GAT implemented with torch.nn primitives."""

    def __new__(cls, *args, **kwargs):
        torch = _require_torch()
        nn = torch.nn

        class TorchGraphAttentionModel(nn.Module):
            def __init__(self, feature_dim: int, hidden_dim: int, node_type_count: int, relation_count: int):
                super().__init__()
                self.hidden_dim = hidden_dim
                self.node_type_embedding = nn.Embedding(node_type_count, hidden_dim)
                self.degree_projection = nn.Linear(2, hidden_dim)
                self.relation_embedding = nn.Embedding(relation_count, hidden_dim)
                self.edge_projection = nn.Linear(3, hidden_dim)
                self.attention_projection = nn.Linear(hidden_dim * 4, 1)
                self.message_projection = nn.Linear(hidden_dim * 3, hidden_dim)
                self.update_projection = nn.Linear(hidden_dim * 2, hidden_dim)
                self.gat_score_head = nn.Sequential(
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                self.legacy_projection = nn.Linear(feature_dim, 1)
                self.activation = nn.LeakyReLU(0.2)

            def forward(self, batch: _SubgraphBatch):
                h = self.node_type_embedding(batch.node_type_ids) + self.degree_projection(batch.node_degree_features)
                if batch.edge_src.numel() > 0:
                    rel = self.relation_embedding(batch.edge_relation_ids)
                    edge_state = self.edge_projection(batch.edge_features)
                    src_h = h[batch.edge_src]
                    dst_h = h[batch.edge_dst]
                    attention_input = torch.cat([src_h, dst_h, rel, edge_state], dim=-1)
                    attention_scores = self.activation(self.attention_projection(attention_input)).squeeze(-1)
                    messages = self.message_projection(torch.cat([src_h, rel, edge_state], dim=-1))

                    aggregated = []
                    for node_index in range(h.shape[0]):
                        mask = batch.edge_dst == node_index
                        if bool(mask.any()):
                            weights = torch.softmax(attention_scores[mask], dim=0).unsqueeze(-1)
                            aggregated.append((weights * messages[mask]).sum(dim=0))
                        else:
                            aggregated.append(torch.zeros(self.hidden_dim, device=h.device, dtype=h.dtype))
                    agg = torch.stack(aggregated, dim=0)
                    h = torch.relu(self.update_projection(torch.cat([h, agg], dim=-1)))

                candidate_h = h[batch.candidate_indices]
                if batch.session_indices.numel() > 0:
                    session_context = h[batch.session_indices].mean(dim=0, keepdim=True)
                    session_context = session_context.expand_as(candidate_h)
                else:
                    session_context = torch.zeros_like(candidate_h)

                pair_features = torch.cat(
                    [
                        candidate_h,
                        session_context,
                        candidate_h * session_context,
                        torch.abs(candidate_h - session_context),
                    ],
                    dim=-1,
                )
                gat_logits = self.gat_score_head(pair_features).squeeze(-1)
                legacy_logits = self.legacy_projection(batch.candidate_features).squeeze(-1)
                return gat_logits + legacy_logits

        return TorchGraphAttentionModel(*args, **kwargs)


class GraphAttentionSemanticScorer:
    def __init__(
        self,
        semantic_memory: SemanticMemory,
        device: str = "auto",
        feature_names: tuple[str, ...] = PHASE3_FEATURE_NAMES,
        hidden_dim: int = 32,
        max_edges_per_seed: int = 64,
    ):
        self.torch = _require_torch()
        self.semantic_memory = semantic_memory
        self.feature_names = feature_names
        self.hidden_dim = hidden_dim
        self.max_edges_per_seed = max_edges_per_seed
        self.node_type_to_id = {node_type: idx for idx, node_type in enumerate(NODE_TYPES)}
        self.relation_to_id = {relation: idx for idx, relation in enumerate(RELATIONS)}
        self.device = resolve_torch_device(device)
        self.model = _TorchGraphAttentionModel(
            feature_dim=len(self.feature_names),
            hidden_dim=hidden_dim,
            node_type_count=len(self.node_type_to_id),
            relation_count=len(self.relation_to_id),
        ).to(self.device)
        self.optimizer = None

    @property
    def weights(self) -> dict[str, float]:
        legacy = self.model.legacy_projection.weight.detach().cpu()[0]
        output = {
            feature_name: float(legacy[idx])
            for idx, feature_name in enumerate(self.feature_names)
        }
        total_norm = 0.0
        with self.torch.no_grad():
            for parameter in self.model.parameters():
                total_norm += float(parameter.detach().norm().cpu()) ** 2
        output["gat_parameter_norm"] = math.sqrt(total_norm)
        return output

    @property
    def bias(self) -> float:
        return float(self.model.legacy_projection.bias.detach().cpu()[0])

    def score_bundle(self, bundle: RetrievalBundle, target_title: str | None = None) -> ScoreOutput:
        self.model.eval()
        with self.torch.no_grad():
            batch = self._build_batch(bundle)
            logits = self.model(batch)
        return self._to_score_output(bundle, logits, target_title)

    def update(self, bundle: RetrievalBundle, target_title: str, learning_rate: float) -> ScoreOutput:
        target_index = self._target_index(bundle, target_title)
        if target_index is None:
            return self.score_bundle(bundle, target_title=target_title)

        if self.optimizer is None or self._optimizer_learning_rate() != learning_rate:
            self.optimizer = self.torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        self.model.train()
        batch = self._build_batch(bundle)
        logits = self.model(batch)
        target = self.torch.tensor([target_index], dtype=self.torch.long, device=self.device)
        loss = self.torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return self.score_bundle(bundle, target_title=target_title)

    def save(self, path: str) -> None:
        payload = {
            "scorer_type": "gat",
            "feature_names": list(self.feature_names),
            "hidden_dim": self.hidden_dim,
            "max_edges_per_seed": self.max_edges_per_seed,
            "node_type_to_id": self.node_type_to_id,
            "relation_to_id": self.relation_to_id,
            "state_dict": self.model.state_dict(),
        }
        self.torch.save(payload, path)

    @classmethod
    def load(cls, path: str, semantic_memory: SemanticMemory, device: str = "auto") -> "GraphAttentionSemanticScorer":
        torch = _require_torch()
        resolved_device = resolve_torch_device(device)
        payload = torch.load(path, map_location=resolved_device)
        scorer = cls(
            semantic_memory=semantic_memory,
            device=resolved_device,
            feature_names=tuple(payload.get("feature_names", PHASE3_FEATURE_NAMES)),
            hidden_dim=int(payload.get("hidden_dim", 32)),
            max_edges_per_seed=int(payload.get("max_edges_per_seed", 64)),
        )
        scorer.model.load_state_dict(payload["state_dict"])
        scorer.model.eval()
        return scorer

    def _build_batch(self, bundle: RetrievalBundle) -> _SubgraphBatch:
        node_ids, directed_edges = self._collect_subgraph(bundle)
        node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
        node_type_ids = [
            self.node_type_to_id.get(self._node_type(node_id), self.node_type_to_id["unknown"])
            for node_id in node_ids
        ]
        node_degree_features = [self._degree_features(node_id) for node_id in node_ids]
        edge_src = [node_index[edge.src] for edge in directed_edges]
        edge_dst = [node_index[edge.dst] for edge in directed_edges]
        edge_relation_ids = [self.relation_to_id[edge.relation] for edge in directed_edges]
        edge_features = [
            [
                float(edge.weight),
                float(edge.confidence),
                min(1.0, math.log1p(max(0, edge.support_count)) / 5.0),
            ]
            for edge in directed_edges
        ]
        session_indices = [
            node_index[item_id]
            for item_id in [self._item_id(title) for title in bundle.session_items]
            if item_id in node_index
        ]
        candidate_indices = [node_index[evidence.candidate_id] for evidence in bundle.candidate_evidence]
        candidate_features = [
            [float(evidence.features.get(feature_name, 0.0)) for feature_name in self.feature_names]
            for evidence in bundle.candidate_evidence
        ]

        return _SubgraphBatch(
            node_type_ids=self.torch.tensor(node_type_ids, dtype=self.torch.long, device=self.device),
            node_degree_features=self.torch.tensor(node_degree_features, dtype=self.torch.float32, device=self.device),
            edge_src=self.torch.tensor(edge_src, dtype=self.torch.long, device=self.device),
            edge_dst=self.torch.tensor(edge_dst, dtype=self.torch.long, device=self.device),
            edge_relation_ids=self.torch.tensor(edge_relation_ids, dtype=self.torch.long, device=self.device),
            edge_features=self.torch.tensor(edge_features, dtype=self.torch.float32, device=self.device)
            if edge_features
            else self.torch.zeros((0, 3), dtype=self.torch.float32, device=self.device),
            session_indices=self.torch.tensor(session_indices, dtype=self.torch.long, device=self.device),
            candidate_indices=self.torch.tensor(candidate_indices, dtype=self.torch.long, device=self.device),
            candidate_features=self.torch.tensor(candidate_features, dtype=self.torch.float32, device=self.device),
            candidate_evidence=list(bundle.candidate_evidence),
        )

    def _collect_subgraph(self, bundle: RetrievalBundle) -> tuple[list[str], list[SemanticEdgeState]]:
        seed_ids = [self._item_id(title) for title in bundle.session_items]
        seed_ids.extend(evidence.candidate_id for evidence in bundle.candidate_evidence)
        nodes: dict[str, None] = {}
        edge_by_key: dict[tuple[str, str, str], SemanticEdgeState] = {}

        for node_id in seed_ids:
            nodes[node_id] = None
            local_edges = self._limited_edges(self.semantic_memory.get_edges(node_id))
            local_edges.extend(self._limited_edges(self.semantic_memory.reverse_adjacency.get(node_id, [])))
            for edge in local_edges:
                nodes[edge.src] = None
                nodes[edge.dst] = None
                edge_by_key[(edge.src, edge.relation, edge.dst)] = edge
                reverse = SemanticEdgeState(
                    src=edge.dst,
                    relation=f"{edge.relation}__rev",
                    dst=edge.src,
                    weight=edge.weight,
                    confidence=edge.confidence,
                    support_count=edge.support_count,
                    source_kind=edge.source_kind,
                    last_update_step=edge.last_update_step,
                    metadata=edge.metadata,
                )
                edge_by_key[(reverse.src, reverse.relation, reverse.dst)] = reverse

        return list(nodes.keys()), list(edge_by_key.values())

    def _limited_edges(self, edges: list[SemanticEdgeState]) -> list[SemanticEdgeState]:
        ranked = sorted(
            edges,
            key=lambda edge: (edge.weight, edge.confidence, edge.support_count),
            reverse=True,
        )
        return ranked[: self.max_edges_per_seed]

    def _to_score_output(self, bundle: RetrievalBundle, logits: Any, target_title: str | None) -> ScoreOutput:
        probabilities = self.torch.softmax(logits.detach(), dim=0).cpu().tolist()
        raw_scores = logits.detach().cpu().tolist()
        legacy_weights = self.model.legacy_projection.weight.detach().cpu()[0].tolist()
        candidate_scores: list[CandidateScore] = []
        for evidence, raw_score, probability in zip(bundle.candidate_evidence, raw_scores, probabilities):
            contributions = {
                feature_name: float(legacy_weights[idx]) * float(evidence.features.get(feature_name, 0.0))
                for idx, feature_name in enumerate(self.feature_names)
            }
            contributions["gat_attention_score"] = float(raw_score) - sum(contributions.values()) - self.bias
            candidate_scores.append(
                CandidateScore(
                    title=evidence.title,
                    score=float(raw_score),
                    probability=float(probability),
                    feature_contributions=contributions,
                    features=dict(evidence.features),
                )
            )
        candidate_scores.sort(key=lambda item: item.probability, reverse=True)
        ranked_titles = [item.title for item in candidate_scores]
        loss = self._cross_entropy(candidate_scores, target_title)
        return ScoreOutput(ranked_titles=ranked_titles, candidate_scores=candidate_scores, loss=loss)

    def _node_type(self, node_id: str) -> str:
        node = self.semantic_memory.nodes.get(node_id)
        if node is not None:
            return node.node_type
        if node_id.startswith("item::"):
            return NODE_TYPE_ITEM
        return "unknown"

    def _degree_features(self, node_id: str) -> list[float]:
        out_degree = len(self.semantic_memory.get_edges(node_id))
        in_degree = self.semantic_memory.in_degree(node_id)
        return [
            min(1.0, math.log1p(out_degree) / 8.0),
            min(1.0, math.log1p(in_degree) / 8.0),
        ]

    def _optimizer_learning_rate(self) -> float | None:
        if self.optimizer is None:
            return None
        return float(self.optimizer.param_groups[0]["lr"])

    @staticmethod
    def _cross_entropy(candidate_scores: list[CandidateScore], target_title: str | None) -> float:
        if target_title is None:
            return 0.0
        for candidate in candidate_scores:
            if candidate.title == target_title:
                return -math.log(max(candidate.probability, 1e-12))
        return float("inf")

    @staticmethod
    def _target_index(bundle: RetrievalBundle, target_title: str) -> int | None:
        for idx, evidence in enumerate(bundle.candidate_evidence):
            if evidence.title == target_title:
                return idx
        return None

    @staticmethod
    def _item_id(title: str) -> str:
        return f"item::{normalize_title(title)}"
