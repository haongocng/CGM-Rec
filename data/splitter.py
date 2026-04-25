"""Chronological warm-up splitting utilities."""

from __future__ import annotations

from constants import SUPPORTED_WARMUP_MODES, WARMUP_MODE_COUNT, WARMUP_MODE_RATIO
from data.schema import SessionSample, WarmupSplit


def split_warmup(
    samples: list[SessionSample],
    mode: str,
    warmup_ratio: float,
    warmup_count: int,
) -> WarmupSplit:
    if mode not in SUPPORTED_WARMUP_MODES:
        raise ValueError(f"Unsupported warmup mode: {mode}")

    total = len(samples)
    if total == 0:
        return WarmupSplit(warmup_samples=[], stream_samples=[])

    if mode == WARMUP_MODE_RATIO:
        warmup_size = max(1, int(total * warmup_ratio))
    elif mode == WARMUP_MODE_COUNT:
        warmup_size = warmup_count
    else:
        raise ValueError(f"Unhandled warmup mode: {mode}")

    warmup_size = max(1, min(warmup_size, total - 1 if total > 1 else 1))
    return WarmupSplit(
        warmup_samples=samples[:warmup_size],
        stream_samples=samples[warmup_size:],
    )

