"""Dataset loading for ML-100K Phase 1 foundations."""

from __future__ import annotations

from constants import DATASET_ML_100K, INDEX_BASE_ONE, INDEX_BASE_UNKNOWN, INDEX_BASE_ZERO
from data.parser import InputParser
from data.schema import DatasetBundle, ProductInfo, SessionSample
from utils.io import load_json_array
from utils.text import normalize_title


class DatasetLoader:
    def __init__(self, expected_candidate_count: int = 20):
        self.parser = InputParser(expected_candidate_count=expected_candidate_count)

    def load_ml100k(self, train_path: str, test_path: str, info_path: str) -> DatasetBundle:
        train_samples = self._load_session_samples(train_path, split_name="train")
        test_samples = self._load_session_samples(test_path, split_name="test")
        products = self._load_products(info_path)
        return DatasetBundle(
            train_samples=train_samples,
            test_samples=test_samples,
            products=products,
            dataset_name=DATASET_ML_100K,
        )

    def _load_session_samples(self, path: str, split_name: str) -> list[SessionSample]:
        raw_samples = load_json_array(path)
        samples: list[SessionSample] = []
        for idx, entry in enumerate(raw_samples):
            parsed = self.parser.parse(entry.get("input", ""))
            target = str(entry.get("target", "")).strip()
            target_index = self._parse_optional_int(entry.get("target_index"))
            target_position = self._find_target_position(parsed.candidate_items, target)
            target_index_base = self._infer_index_base(target_index, target_position)

            if target_position is None:
                raise ValueError(f"Target '{target}' is not present in candidate set for {split_name}[{idx}]")

            samples.append(
                SessionSample(
                    sample_id=f"{split_name}_{idx:05d}",
                    target=target,
                    target_index=target_index,
                    raw_input=parsed.raw_input,
                    parsed_input=parsed,
                    target_position=target_position,
                    target_index_base=target_index_base,
                )
            )
        return samples

    def _load_products(self, path: str) -> dict[str, ProductInfo]:
        raw_products = load_json_array(path)
        products: dict[str, ProductInfo] = {}
        for entry in raw_products:
            title = str(entry.get("title", "")).strip()
            normalized = normalize_title(title)
            details = entry.get("details", {}) or {}
            keywords = details.get("keywords", []) or []
            if isinstance(keywords, str):
                keywords = [part.strip() for part in keywords.split(",") if part.strip()]

            products[normalized] = ProductInfo(
                title=title,
                normalized_title=normalized,
                taxonomy_levels=dict(entry.get("taxonomy", {}) or {}),
                full_path=str(entry.get("full_path", "") or ""),
                description=str(details.get("description", "") or entry.get("description", "") or ""),
                keywords=[str(keyword).strip() for keyword in keywords if str(keyword).strip()],
                raw=entry,
            )
        return products

    @staticmethod
    def _parse_optional_int(value: object) -> int | None:
        if value is None or value == "":
            return None
        return int(value)

    @staticmethod
    def _find_target_position(candidate_items: list[str], target: str) -> int | None:
        for idx, candidate in enumerate(candidate_items):
            if candidate == target:
                return idx
        return None

    @staticmethod
    def _infer_index_base(target_index: int | None, target_position: int | None) -> str:
        if target_index is None or target_position is None:
            return INDEX_BASE_UNKNOWN
        if target_index == target_position:
            return INDEX_BASE_ZERO
        if target_index == target_position + 1:
            return INDEX_BASE_ONE
        return INDEX_BASE_UNKNOWN

