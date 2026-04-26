"""Dataset loading for CGM-Rec — supports ml_100k (rich format) and
bundle / ml_1m / games (lightweight format: title + category list only).

Format detection happens in _load_products(): the presence of the ``taxonomy``
key indicates the rich ml_100k format; the presence of ``category`` (a list)
indicates the lightweight format, which is normalised into the same
ProductInfo schema so downstream components (SeedGraphBuilder, feature
extractors) require no changes.
"""

from __future__ import annotations

from constants import INDEX_BASE_ONE, INDEX_BASE_UNKNOWN, INDEX_BASE_ZERO
from data.parser import InputParser
from data.schema import DatasetBundle, ProductInfo, SessionSample
from utils.io import load_json_array
from utils.text import normalize_title


class DatasetLoader:
    def __init__(self, expected_candidate_count: int = 20):
        self.parser = InputParser(expected_candidate_count=expected_candidate_count)

    # ------------------------------------------------------------------
    # Public loaders
    # ------------------------------------------------------------------

    def load_dataset(self, dataset_name: str, train_path: str, test_path: str, info_path: str) -> DatasetBundle:
        """Generic loader — works for all supported datasets."""
        train_samples = self._load_session_samples(train_path, split_name="train")
        test_samples = self._load_session_samples(test_path, split_name="test")
        products = self._load_products(info_path)
        return DatasetBundle(
            train_samples=train_samples,
            test_samples=test_samples,
            products=products,
            dataset_name=dataset_name,
        )

    # Kept for backward compatibility (scratch scripts, tests, etc.)
    def load_ml100k(self, train_path: str, test_path: str, info_path: str) -> DatasetBundle:
        return self.load_dataset("ml_100k", train_path, test_path, info_path)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
        """Load product metadata.

        Supports two formats:
        - **Rich format** (ml_100k): entry has ``taxonomy`` (dict), ``full_path``,
          and ``details`` (with ``keywords`` list and ``description``).
        - **Simple format** (bundle, ml_1m, games): entry has only ``title`` and
          ``category`` (a list of strings). The category list is converted to a
          ``taxonomy_levels`` dict (Level_1, Level_2, …) so that downstream
          components behave identically regardless of the source dataset.
        """
        raw_products = load_json_array(path)
        products: dict[str, ProductInfo] = {}
        for entry in raw_products:
            title = str(entry.get("title", "")).strip()
            if not title:
                continue
            normalized = normalize_title(title)

            if "taxonomy" in entry:
                # ---- Rich format (ml_100k) ----
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
                    keywords=[str(kw).strip() for kw in keywords if str(kw).strip()],
                    raw=entry,
                )
            else:
                # ---- Simple format (bundle, ml_1m, games) ----
                # Convert flat category list → taxonomy_levels dict
                raw_categories = entry.get("category", []) or []
                if isinstance(raw_categories, str):
                    raw_categories = [raw_categories]
                taxonomy_levels = {
                    f"Level_{i + 1}": str(cat).strip()
                    for i, cat in enumerate(raw_categories)
                    if str(cat).strip()
                }
                full_path = " > ".join(taxonomy_levels.values())
                products[normalized] = ProductInfo(
                    title=title,
                    normalized_title=normalized,
                    taxonomy_levels=taxonomy_levels,
                    full_path=full_path,
                    description="",   # not available in simple format
                    keywords=[],      # not available in simple format
                    raw=entry,
                )
        return products

    # ------------------------------------------------------------------
    # Static utilities
    # ------------------------------------------------------------------

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
