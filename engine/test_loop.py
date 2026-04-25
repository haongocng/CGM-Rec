"""Phase 3/4 evaluation loops."""

from __future__ import annotations

from dataclasses import dataclass, field

from data.schema import SessionSample
from engine.diagnostics import HybridDiagnostics
from eval.metrics import evaluate_predictions
from llm.lesson_agent import LLMLessonAgent
from llm.reranker import LLMReranker
from memory.episodic_memory import EpisodicMemory
from memory.writer import MemoryWriter
from model.scorer import CandidateScore, LinearSemanticScorer, ScoreOutput
from retrieval.llm_evidence_builder import LLMEvidenceBuilder
from retrieval.semantic_retriever import SemanticRetriever


@dataclass(slots=True)
class TestExample:
    sample_id: str
    target: str
    ranked_titles: list[str]
    top_probability: float
    target_rank: int | None = None
    outcome_type: str = ""
    applied_edit_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class TestResult:
    metrics: dict[str, float]
    examples: list[TestExample]


@dataclass(slots=True)
class OnlineTestResult:
    metrics: dict[str, float]
    examples: list[TestExample]
    edit_counts: dict[str, int]
    episodic_summary: dict
    semantic_summary: dict
    audit_log: list[dict]


@dataclass(slots=True)
class Phase5TestExample:
    sample_id: str
    target: str
    scorer_top5: list[str]
    llm_top5: list[str]
    final_top5: list[str]
    target_rank: int
    fallback_used: bool
    parser_valid: bool
    lesson_valid: bool
    applied_edit_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class Phase5TestResult:
    metrics: dict[str, float]
    examples: list[Phase5TestExample]
    edit_counts: dict[str, int]
    episodic_summary: dict
    semantic_summary: dict
    parser_valid_rate: float
    fallback_rate: float
    lesson_valid_rate: float
    audit_log: list[dict]


def evaluate_semantic_scorer(
    samples: list[SessionSample],
    retriever: SemanticRetriever,
    scorer: LinearSemanticScorer,
    max_examples: int = 10,
) -> TestResult:
    predictions: list[list[str]] = []
    targets: list[str] = []
    examples: list[TestExample] = []

    for sample in samples:
        bundle = retriever.retrieve(
            session_items=sample.parsed_input.session_items,
            candidate_items=sample.parsed_input.candidate_items,
        )
        output = scorer.score_bundle(bundle, target_title=sample.target)
        predictions.append(output.ranked_titles)
        targets.append(sample.target)
        if len(examples) < max_examples:
            top_probability = output.candidate_scores[0].probability if output.candidate_scores else 0.0
            examples.append(
                TestExample(
                    sample_id=sample.sample_id,
                    target=sample.target,
                    ranked_titles=output.ranked_titles[:5],
                    top_probability=top_probability,
                )
            )

    return TestResult(metrics=evaluate_predictions(predictions, targets), examples=examples)


def evaluate_semantic_scorer_online(
    samples: list[SessionSample],
    retriever: SemanticRetriever,
    scorer: LinearSemanticScorer,
    diagnostics: HybridDiagnostics,
    writer: MemoryWriter,
    episodic_memory: EpisodicMemory,
    max_examples: int = 10,
) -> OnlineTestResult:
    predictions: list[list[str]] = []
    targets: list[str] = []
    examples: list[TestExample] = []

    for step_index, sample in enumerate(samples):
        bundle = retriever.retrieve(
            session_items=sample.parsed_input.session_items,
            candidate_items=sample.parsed_input.candidate_items,
        )
        output = scorer.score_bundle(bundle, target_title=sample.target)
        predictions.append(output.ranked_titles)
        targets.append(sample.target)

        similar_episodes = episodic_memory.retrieve_similar(
            session_items=sample.parsed_input.session_items,
            candidate_items=sample.parsed_input.candidate_items,
            limit=5,
            min_similarity=0.05,
        )
        lesson = diagnostics.analyze(
            sample=sample,
            retrieval=bundle,
            score_output=output,
            supporting_episode_ids=[match.episode.episode_id for match in similar_episodes],
        )
        write_result = writer.apply_lesson(lesson, step_index=step_index)

        if len(examples) < max_examples:
            top_probability = output.candidate_scores[0].probability if output.candidate_scores else 0.0
            examples.append(
                TestExample(
                    sample_id=sample.sample_id,
                    target=sample.target,
                    ranked_titles=output.ranked_titles[:5],
                    top_probability=top_probability,
                    target_rank=lesson.target_rank,
                    outcome_type=lesson.outcome_type,
                    applied_edit_counts=write_result.counts(),
                )
            )

    return OnlineTestResult(
        metrics=evaluate_predictions(predictions, targets),
        examples=examples,
        edit_counts=writer.aggregate_counts(),
        episodic_summary=episodic_memory.summary(),
        semantic_summary={
            "edge_count": writer.semantic_memory.edge_count(),
            "relations": writer.semantic_memory.relation_counts(),
        },
        audit_log=list(writer.audit_log),
    )


def evaluate_llm_reranker_online(
    samples: list[SessionSample],
    retriever: SemanticRetriever,
    scorer: LinearSemanticScorer,
    diagnostics: HybridDiagnostics,
    writer: MemoryWriter,
    episodic_memory: EpisodicMemory,
    reranker: LLMReranker,
    lesson_agent: LLMLessonAgent,
    evidence_builder: LLMEvidenceBuilder,
    max_examples: int = 10,
) -> Phase5TestResult:
    predictions: list[list[str]] = []
    targets: list[str] = []
    examples: list[Phase5TestExample] = []
    parser_valid_count = 0
    fallback_count = 0
    lesson_valid_count = 0

    for step_index, sample in enumerate(samples):
        bundle = retriever.retrieve(
            session_items=sample.parsed_input.session_items,
            candidate_items=sample.parsed_input.candidate_items,
        )
        scorer_output = scorer.score_bundle(bundle, target_title=sample.target)
        similar_episodes = episodic_memory.retrieve_similar(
            session_items=sample.parsed_input.session_items,
            candidate_items=sample.parsed_input.candidate_items,
            limit=5,
            min_similarity=0.05,
        )
        graph_evidence = evidence_builder.build_graph_evidence(bundle, scorer_output)
        episodic_hints = evidence_builder.build_episodic_hints(similar_episodes)
        rerank_result = reranker.rerank(
            session_items=sample.parsed_input.session_items,
            candidate_items=sample.parsed_input.candidate_items,
            graph_evidence=graph_evidence,
            episodic_hints=episodic_hints,
            scorer_ranking=scorer_output.ranked_titles,
        )
        final_ranking = rerank_result.ranked_titles
        final_rank = _get_rank(final_ranking, sample.target)
        predictions.append(final_ranking)
        targets.append(sample.target)
        parser_valid_count += 1 if rerank_result.parser_valid else 0
        fallback_count += 1 if rerank_result.fallback_used else 0

        lesson_advice = lesson_agent.infer(
            session_items=sample.parsed_input.session_items,
            candidate_items=sample.parsed_input.candidate_items,
            graph_evidence=graph_evidence,
            final_ranking=final_ranking,
            target_item=sample.target,
            target_rank=final_rank,
        )
        lesson_valid_count += 1 if lesson_advice.valid else 0

        final_output = _reorder_score_output(scorer_output, final_ranking)
        lesson = diagnostics.analyze(
            sample=sample,
            retrieval=bundle,
            score_output=final_output,
            supporting_episode_ids=[match.episode.episode_id for match in similar_episodes],
        )
        lesson.lesson_advice = lesson_advice.to_dict()
        lesson.signals["llm_parser_valid"] = 1.0 if rerank_result.parser_valid else 0.0
        lesson.signals["llm_fallback_used"] = 1.0 if rerank_result.fallback_used else 0.0
        lesson.signals["llm_lesson_valid"] = 1.0 if lesson_advice.valid else 0.0
        write_result = writer.apply_lesson(lesson, step_index=step_index)

        if len(examples) < max_examples:
            examples.append(
                Phase5TestExample(
                    sample_id=sample.sample_id,
                    target=sample.target,
                    scorer_top5=scorer_output.ranked_titles[:5],
                    llm_top5=rerank_result.ranked_titles[:5],
                    final_top5=final_ranking[:5],
                    target_rank=final_rank,
                    fallback_used=rerank_result.fallback_used,
                    parser_valid=rerank_result.parser_valid,
                    lesson_valid=lesson_advice.valid,
                    applied_edit_counts=write_result.counts(),
                )
            )

    total = len(samples) or 1
    return Phase5TestResult(
        metrics=evaluate_predictions(predictions, targets),
        examples=examples,
        edit_counts=writer.aggregate_counts(),
        episodic_summary=episodic_memory.summary(),
        semantic_summary={
            "edge_count": writer.semantic_memory.edge_count(),
            "relations": writer.semantic_memory.relation_counts(),
        },
        parser_valid_rate=parser_valid_count / total,
        fallback_rate=fallback_count / total,
        lesson_valid_rate=lesson_valid_count / total,
        audit_log=list(writer.audit_log),
    )


def _reorder_score_output(output: ScoreOutput, ranked_titles: list[str]) -> ScoreOutput:
    by_title: dict[str, CandidateScore] = {candidate.title: candidate for candidate in output.candidate_scores}
    ordered = [by_title[title] for title in ranked_titles if title in by_title]
    return ScoreOutput(ranked_titles=list(ranked_titles), candidate_scores=ordered, loss=output.loss)


def _get_rank(predictions: list[str], target: str) -> int:
    for idx, candidate in enumerate(predictions):
        if candidate == target:
            return idx + 1
    return 999
