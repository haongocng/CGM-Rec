"""Reproducibility helpers."""

from __future__ import annotations

import random


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np  # type: ignore

        np.random.seed(seed)
    except Exception:
        pass

