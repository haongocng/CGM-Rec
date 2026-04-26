"""Phase 1 configuration for CGM-Rec foundations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from constants import (
    DATASET_ML_100K,
    DEFAULT_CO_OCCUR_WINDOW,
    DEFAULT_INCLUDE_DESCRIPTION,
    DEFAULT_KEYWORD_TOP_K,
    DEFAULT_PHASE3_EPOCHS,
    DEFAULT_PHASE3_LEARNING_RATE,
    DEFAULT_RANDOM_SEED,
    DEFAULT_WARMUP_COUNT,
    DEFAULT_WARMUP_MODE,
    DEFAULT_WARMUP_RATIO,
)


@dataclass(slots=True)
class Phase1Config:
    dataset_root: Path
    dataset_name: str = DATASET_ML_100K
    train_path: Path | None = None
    test_path: Path | None = None
    info_path: Path | None = None
    warmup_mode: str = DEFAULT_WARMUP_MODE
    warmup_ratio: float = DEFAULT_WARMUP_RATIO
    warmup_count: int = DEFAULT_WARMUP_COUNT
    random_seed: int = DEFAULT_RANDOM_SEED
    keyword_top_k: int = DEFAULT_KEYWORD_TOP_K
    include_description: bool = DEFAULT_INCLUDE_DESCRIPTION
    co_occur_window_size: int = DEFAULT_CO_OCCUR_WINDOW
    phase3_epochs: int = DEFAULT_PHASE3_EPOCHS
    phase3_learning_rate: float = DEFAULT_PHASE3_LEARNING_RATE

    def __post_init__(self) -> None:
        """Resolve dataset file paths.

        File naming convention: underscores are removed from the dataset
        identifier to form the file stem, e.g.:
          ml_100k  → train_ml100k.json / test_ml100k.json / info_ml100k.json
          ml_1m    → train_ml1m.json   / test_ml1m.json   / info_ml1m.json
          bundle   → train_bundle.json / test_bundle.json / info_bundle.json
          games    → train_games.json  / test_games.json  / info_games.json
        """
        dataset_dir = self.dataset_root / self.dataset_name
        file_stem = self.dataset_name.replace("_", "")
        if self.train_path is None:
            self.train_path = dataset_dir / f"train_{file_stem}.json"
        if self.test_path is None:
            self.test_path = dataset_dir / f"test_{file_stem}.json"
        if self.info_path is None:
            self.info_path = dataset_dir / f"info_{file_stem}.json"


def default_config(project_root: str | Path) -> Phase1Config:
    project_root = Path(project_root).resolve()
    dataset_root = project_root / "dataset"
    return Phase1Config(dataset_root=dataset_root)
