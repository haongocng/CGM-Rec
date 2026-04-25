"""Evaluation metrics for Phase 3."""

from __future__ import annotations

import math


def get_rank(predictions: list[str], target: str) -> int:
    for idx, candidate in enumerate(predictions):
        if candidate == target:
            return idx + 1
    return 999


def hit_at_k(rank: int, k: int) -> float:
    return 1.0 if rank <= k else 0.0


def ndcg_at_k(rank: int, k: int) -> float:
    if rank > k:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate_predictions(predictions: list[list[str]], targets: list[str], ks: tuple[int, ...] = (1, 5, 10, 20)) -> dict[str, float]:
    ranks = [get_rank(prediction, target) for prediction, target in zip(predictions, targets)]
    metrics: dict[str, float] = {}
    if not ranks:
        return metrics
    for k in ks:
        metrics[f"HIT@{k}"] = sum(hit_at_k(rank, k) for rank in ranks) / len(ranks)
        metrics[f"NDCG@{k}"] = sum(ndcg_at_k(rank, k) for rank in ranks) / len(ranks)
    metrics["avg_rank"] = sum(ranks) / len(ranks)
    return metrics
