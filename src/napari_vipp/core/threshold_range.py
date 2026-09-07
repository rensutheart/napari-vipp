"""Shared validation for an authored, inclusive intensity interval."""

from __future__ import annotations

import math


def validate_threshold_range(low_threshold, high_threshold) -> tuple[float, float]:
    """Reject invalid bounds; never reorder or repair scientific parameters."""
    try:
        low, high = float(low_threshold), float(high_threshold)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Binary Threshold range limits must be finite numbers."
        ) from exc
    if not math.isfinite(low) or not math.isfinite(high):
        raise ValueError("Binary Threshold range limits must be finite numbers.")
    if low > high:
        raise ValueError(
            "Binary Threshold low threshold must not exceed high threshold."
        )
    return low, high
