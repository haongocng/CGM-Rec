"""Data-facing schema for Phase 1."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class ParsedInput:
    raw_input: str
    session_items: list[str]
    candidate_items: list[str]


@dataclass(slots=True)
class SessionSample:
    sample_id: str
    target: str
    target_index: int | None
    raw_input: str
    parsed_input: ParsedInput
    target_position: int | None = None
    target_index_base: str | None = None


@dataclass(slots=True)
class ProductInfo:
    title: str
    normalized_title: str
    taxonomy_levels: dict[str, str] = field(default_factory=dict)
    full_path: str = ""
    description: str = ""
    keywords: list[str] = field(default_factory=list)
    raw: dict = field(default_factory=dict)


@dataclass(slots=True)
class DatasetBundle:
    train_samples: list[SessionSample]
    test_samples: list[SessionSample]
    products: dict[str, ProductInfo]
    dataset_name: str


@dataclass(slots=True)
class WarmupSplit:
    warmup_samples: list[SessionSample]
    stream_samples: list[SessionSample]

