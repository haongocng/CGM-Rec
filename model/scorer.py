"""Pure-Python trainable semantic scorer for Phase 3."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

from constants import PHASE3_FEATURE_NAMES
from retrieval.semantic_retriever import CandidateEvidence, RetrievalBundle


@dataclass(slots=True)
class CandidateScore:
    title: str
    score: float
    probability: float
    feature_contributions: dict[str, float]
    features: dict[str, float]


@dataclass(slots=True)
class ScoreOutput:
    ranked_titles: list[str]
    candidate_scores: list[CandidateScore]
    loss: float


class LinearSemanticScorer:
    def __init__(self, feature_names: tuple[str, ...] = PHASE3_FEATURE_NAMES):
        self.feature_names = feature_names
        self.weights = {name: 0.0 for name in feature_names}
        self.bias = 0.0

    def score_bundle(self, bundle: RetrievalBundle, target_title: str | None = None) -> ScoreOutput:
        raw_scores = []
        for evidence in bundle.candidate_evidence:
            score, contributions = self._score_candidate(evidence)
            raw_scores.append((evidence, score, contributions))

        probabilities = self._softmax([score for _, score, _ in raw_scores])
        candidate_scores: list[CandidateScore] = []
        for (evidence, score, contributions), probability in zip(raw_scores, probabilities):
            candidate_scores.append(
                CandidateScore(
                    title=evidence.title,
                    score=score,
                    probability=probability,
                    feature_contributions=contributions,
                    features=dict(evidence.features),
                )
            )

        candidate_scores.sort(key=lambda item: item.probability, reverse=True)
        ranked_titles = [item.title for item in candidate_scores]
        loss = self._cross_entropy(candidate_scores, target_title) if target_title is not None else 0.0
        return ScoreOutput(ranked_titles=ranked_titles, candidate_scores=candidate_scores, loss=loss)

    def update(self, bundle: RetrievalBundle, target_title: str, learning_rate: float) -> ScoreOutput:
        raw_scores = []
        for evidence in bundle.candidate_evidence:
            score, contributions = self._score_candidate(evidence)
            raw_scores.append((evidence, score, contributions))
        probabilities = self._softmax([score for _, score, _ in raw_scores])

        for (evidence, _score, _contrib), probability in zip(raw_scores, probabilities):
            target_indicator = 1.0 if evidence.title == target_title else 0.0
            error = probability - target_indicator
            for feature_name in self.feature_names:
                self.weights[feature_name] -= learning_rate * error * evidence.features.get(feature_name, 0.0)
            self.bias -= learning_rate * error

        return self.score_bundle(bundle, target_title=target_title)

    def save(self, path: str) -> None:
        payload = {
            "feature_names": list(self.feature_names),
            "weights": self.weights,
            "bias": self.bias,
        }
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "LinearSemanticScorer":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        scorer = cls(feature_names=tuple(payload["feature_names"]))
        scorer.weights = {key: float(value) for key, value in payload["weights"].items()}
        scorer.bias = float(payload["bias"])
        return scorer

    def _score_candidate(self, evidence: CandidateEvidence) -> tuple[float, dict[str, float]]:
        contributions = {
            feature_name: self.weights[feature_name] * evidence.features.get(feature_name, 0.0)
            for feature_name in self.feature_names
        }
        score = self.bias + sum(contributions.values())
        return score, contributions

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        max_score = max(scores) if scores else 0.0
        exps = [math.exp(score - max_score) for score in scores]
        denom = sum(exps) or 1.0
        return [value / denom for value in exps]

    @staticmethod
    def _cross_entropy(candidate_scores: list[CandidateScore], target_title: str | None) -> float:
        if target_title is None:
            return 0.0
        for candidate in candidate_scores:
            if candidate.title == target_title:
                probability = max(candidate.probability, 1e-12)
                return -math.log(probability)
        return float("inf")
