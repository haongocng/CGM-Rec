"""Single-session Phase 3 execution."""

from __future__ import annotations

from dataclasses import dataclass

from data.schema import SessionSample
from model.scorer import LinearSemanticScorer, ScoreOutput
from retrieval.semantic_retriever import RetrievalBundle, SemanticRetriever


@dataclass(slots=True)
class SessionResult:
    sample_id: str
    target: str
    retrieval: RetrievalBundle
    score_output: ScoreOutput


def run_session(sample: SessionSample, retriever: SemanticRetriever, scorer: LinearSemanticScorer) -> SessionResult:
    retrieval = retriever.retrieve(
        session_items=sample.parsed_input.session_items,
        candidate_items=sample.parsed_input.candidate_items,
    )
    score_output = scorer.score_bundle(retrieval, target_title=sample.target)
    return SessionResult(
        sample_id=sample.sample_id,
        target=sample.target,
        retrieval=retrieval,
        score_output=score_output,
    )
