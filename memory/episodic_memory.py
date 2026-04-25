"""Fast episodic lesson memory for continual recommendation."""

from __future__ import annotations

from dataclasses import dataclass

from memory.schema import EpisodeRecord, LessonPayload
from utils.text import normalize_title


@dataclass(slots=True)
class EpisodeMatch:
    episode: EpisodeRecord
    similarity: float
    overlap_items: list[str]
    overlap_candidates: list[str]

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode.episode_id,
            "sample_id": self.episode.sample_id,
            "similarity": self.similarity,
            "overlap_items": self.overlap_items,
            "overlap_candidates": self.overlap_candidates,
            "target_item": self.episode.target_item,
            "outcome_type": self.episode.outcome_type,
            "target_rank": self.episode.target_rank,
        }


class EpisodicMemory:
    def __init__(self, max_records: int = 500):
        self.max_records = max(1, max_records)
        self.records: list[EpisodeRecord] = []
        self._next_episode_index = 0

    def store(self, record: EpisodeRecord) -> EpisodeRecord:
        self.records.append(record)
        self._next_episode_index += 1
        if len(self.records) > self.max_records:
            self.records = self.records[-self.max_records :]
        return record

    def store_lesson(self, lesson: LessonPayload) -> EpisodeRecord:
        return self.store(
            EpisodeRecord(
                episode_id=f"episode_{self._next_episode_index:06d}",
                sample_id=lesson.sample_id,
                session_items=list(lesson.session_items),
                candidate_items=list(lesson.candidate_items),
                target_item=lesson.target_item,
                predicted_ranking=list(lesson.predicted_ranking),
                outcome_type=lesson.outcome_type,
                target_rank=lesson.target_rank,
                proposed_edits=list(lesson.proposed_edits),
                diagnosis=lesson.diagnosis,
                signals=dict(lesson.signals),
                lesson_advice=dict(lesson.lesson_advice),
            )
        )

    def retrieve_similar(
        self,
        session_items: list[str],
        candidate_items: list[str] | None = None,
        limit: int = 5,
        min_similarity: float = 0.0,
    ) -> list[EpisodeMatch]:
        ranked: list[EpisodeMatch] = []
        session_set = self._normalize_items(session_items)
        candidate_set = self._normalize_items(candidate_items or [])

        for record in reversed(self.records):
            record_set = self._normalize_items(record.session_items)
            record_candidate_set = self._normalize_items(record.candidate_items)
            union = session_set | record_set
            candidate_union = candidate_set | record_candidate_set
            overlap = session_set & record_set
            overlap_candidates = candidate_set & record_candidate_set
            session_similarity = len(overlap) / len(union) if union else 0.0
            candidate_similarity = len(overlap_candidates) / len(candidate_union) if candidate_union else 0.0
            similarity = (0.7 * session_similarity) + (0.3 * candidate_similarity)
            if similarity < min_similarity:
                continue
            ranked.append(
                EpisodeMatch(
                    episode=record,
                    similarity=similarity,
                    overlap_items=sorted(overlap),
                    overlap_candidates=sorted(overlap_candidates),
                )
            )

        ranked.sort(
            key=lambda match: (
                match.similarity,
                -match.episode.target_rank,
                match.episode.episode_id,
            ),
            reverse=True,
        )
        return ranked[: max(1, limit)]

    def latest(self, limit: int = 5) -> list[EpisodeRecord]:
        return list(reversed(self.records[-max(1, limit) :]))

    def summary(self) -> dict:
        success_count = sum(1 for record in self.records if record.outcome_type == "success")
        failure_count = sum(1 for record in self.records if record.outcome_type == "failure")
        return {
            "records": len(self.records),
            "max_records": self.max_records,
            "success_records": success_count,
            "failure_records": failure_count,
        }

    @staticmethod
    def _normalize_items(items: list[str]) -> set[str]:
        return {normalize_title(item) for item in items if item.strip()}
