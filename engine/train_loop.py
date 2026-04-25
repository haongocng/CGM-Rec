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
