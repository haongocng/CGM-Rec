"""Phase 3 loss helpers."""

from __future__ import annotations

from model.scorer import ScoreOutput


def cross_entropy_from_output(output: ScoreOutput) -> float:
    return output.loss
