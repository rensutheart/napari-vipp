"""Finite owner-approved continuation metadata, never a model grant tool."""

from __future__ import annotations

import math
from copy import deepcopy

HARD_ALLOWANCE = {
    "max_attempts": 5,
    "max_actions": 64,
    "max_model_calls": 80,
    "max_seconds": 1800,
    "max_image_bytes": 20 * 1024 * 1024,
}
MAX_CONTINUATION_GRANTS = 64
TIME_POLICY = "safe_boundary_active_seconds"


def allowance_dict(limits):
    """Copy approved absolute totals, not an increment or remaining allowance."""
    return {name: getattr(limits, name) for name in HARD_ALLOWANCE}


def validate_continuation_context(value):
    """Validate optional v2 wire metadata without granting any authority."""
    if (
        type(value) is not dict
        or set(value)
        != {"version", "allowance", "hard_limits", "grants_used", "time_policy"}
        or type(value["version"]) is not int
        or value["version"] != 1
        or value["time_policy"] != TIME_POLICY
        or type(value["grants_used"]) is not int
        or not 0 <= value["grants_used"] <= MAX_CONTINUATION_GRANTS
    ):
        raise ValueError("Invalid bounded continuation metadata.")
    for key in ("allowance", "hard_limits"):
        limits = value[key]
        if type(limits) is not dict or set(limits) != set(HARD_ALLOWANCE):
            raise ValueError("Continuation needs exact finite allowance fields.")
        for name, ceiling in HARD_ALLOWANCE.items():
            number = limits[name]
            if name == "max_seconds":
                valid = (
                    type(number) in {int, float}
                    and math.isfinite(number)
                    and 0 < number <= ceiling
                )
            else:
                valid = type(number) is int and 1 <= number <= ceiling
            if not valid or key == "hard_limits" and number != ceiling:
                raise ValueError("Continuation cannot change hard safety ceilings.")
    return deepcopy(value)
