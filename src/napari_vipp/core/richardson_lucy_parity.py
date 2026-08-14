"""Operation-owned production parity gates for Richardson--Lucy variants."""

from __future__ import annotations

import math

import numpy as np

from napari_vipp.core.compute_benchmark import ParityResult

RICHARDSON_LUCY_PARITY_OPERATION_IDS = frozenset({"richardson_lucy_deconvolution"})
RICHARDSON_LUCY_TV_PARITY_OPERATION_IDS = frozenset(
    {"richardson_lucy_tv_deconvolution"}
)
RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT = 5e-3
RICHARDSON_LUCY_FLOAT32_ABSOLUTE_FLOOR = 1e-12
RICHARDSON_LUCY_FLOAT32_MAX_ABS_BASE = 1e-6
RICHARDSON_LUCY_FLOAT32_MAX_ABS_PEAK_FACTOR = 5e-3
RICHARDSON_LUCY_FLOAT32_NEAR_IDENTITY_NRMSE_LIMIT = 2e-6
RICHARDSON_LUCY_TV_FLOAT32_NRMSE_LIMIT = RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT
RICHARDSON_LUCY_TV_FLOAT32_MAX_ABS_BASE = RICHARDSON_LUCY_FLOAT32_MAX_ABS_BASE
RICHARDSON_LUCY_TV_FLOAT32_MAX_ABS_PEAK_FACTOR = (
    RICHARDSON_LUCY_FLOAT32_MAX_ABS_PEAK_FACTOR
)


def richardson_lucy_float32_parity(
    reference: object,
    candidate: object,
) -> ParityResult:
    """Gate ordinary RL with the official float32 scientific-equivalence policy."""

    return _richardson_lucy_scientific_equivalence(
        reference,
        candidate,
        operation_name="Richardson-Lucy",
    )


def richardson_lucy_tv_float32_parity(
    reference: object,
    candidate: object,
) -> ParityResult:
    """Gate RL-TV with the shared float32 scientific-equivalence policy."""

    return _richardson_lucy_scientific_equivalence(
        reference,
        candidate,
        operation_name="Richardson-Lucy TV",
    )


def _richardson_lucy_scientific_equivalence(
    reference: object,
    candidate: object,
    *,
    operation_name: str,
) -> ParityResult:
    """Compare two RL-family outputs without mistaking identity for validity.

    The 0.5% bounds are the official CPU/GPU scientific-equivalence decision.
    The former 2e-6 NRMSE screen is retained in the detail string only as a
    useful near-identity diagnostic; it never passes or rejects an output.
    """

    try:
        expected = np.asarray(reference)
        actual = np.asarray(candidate)
    except Exception as exc:
        return ParityResult(False, f"outputs are not host arrays: {exc}")
    mismatch = _array_contract_mismatch(expected, actual)
    if mismatch:
        return ParityResult(False, mismatch)
    if expected.dtype != np.dtype(np.float32):
        return ParityResult(
            False,
            f"{operation_name} production benchmark requires float32, "
            f"got {expected.dtype}",
        )
    expected_finite = np.isfinite(expected)
    actual_finite = np.isfinite(actual)
    if not np.array_equal(expected_finite, actual_finite):
        return ParityResult(False, "finite/non-finite masks differ")
    if not bool(np.all(expected_finite)):
        return ParityResult(
            False,
            f"{operation_name} admitted region must be completely finite",
        )
    if bool(np.any(expected < 0)) or bool(np.any(actual < 0)):
        return ParityResult(False, f"{operation_name} outputs must be non-negative")

    expected64 = expected.astype(np.float64)
    actual64 = actual.astype(np.float64)
    difference = actual64 - expected64
    max_abs = float(np.max(np.abs(difference))) if difference.size else 0.0
    peak = float(np.max(np.abs(expected64))) if expected64.size else 0.0
    max_abs_limit = RICHARDSON_LUCY_FLOAT32_MAX_ABS_BASE + (
        RICHARDSON_LUCY_FLOAT32_MAX_ABS_PEAK_FACTOR * peak
    )
    denominator = max(
        float(np.linalg.norm(expected64.ravel())),
        float(math.sqrt(expected64.size) * RICHARDSON_LUCY_FLOAT32_ABSOLUTE_FLOOR),
    )
    numerator = float(np.linalg.norm(difference.ravel()))
    nrmse = numerator / denominator if denominator else 0.0
    max_ulp = _maximum_float32_ulp_distance(expected, actual)
    near_identity = nrmse <= RICHARDSON_LUCY_FLOAT32_NEAR_IDENTITY_NRMSE_LIMIT
    passed = bool(
        nrmse <= RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT and max_abs <= max_abs_limit
    )
    return ParityResult(
        passed,
        f"nrmse={nrmse:.9g} "
        f"(limit={RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT:.9g}); "
        f"max_abs={max_abs:.9g} (limit={max_abs_limit:.9g}); "
        "near_identity="
        f"{str(near_identity).lower()} "
        "(diagnostic_nrmse_limit="
        f"{RICHARDSON_LUCY_FLOAT32_NEAR_IDENTITY_NRMSE_LIMIT:.9g}); "
        f"max_ulp={max_ulp} (diagnostic)",
    )


def _array_contract_mismatch(expected: np.ndarray, actual: np.ndarray) -> str:
    if expected.shape != actual.shape:
        return f"shape differs: CPU {expected.shape}, candidate {actual.shape}"
    if expected.dtype != actual.dtype:
        return f"dtype differs: CPU {expected.dtype}, candidate {actual.dtype}"
    return ""


def _maximum_float32_ulp_distance(
    expected: np.ndarray,
    actual: np.ndarray,
) -> int:
    if not expected.size:
        return 0

    def ordered(values: np.ndarray) -> np.ndarray:
        bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32)
        negative = (bits & np.uint32(0x80000000)) != 0
        return np.where(
            negative,
            np.uint64(0xFFFFFFFF) - bits.astype(np.uint64),
            np.uint64(0x80000000) + bits.astype(np.uint64),
        )

    expected_ordered = ordered(expected)
    actual_ordered = ordered(actual)
    distance = np.maximum(expected_ordered, actual_ordered) - np.minimum(
        expected_ordered,
        actual_ordered,
    )
    return int(np.max(distance))


__all__ = [
    "RICHARDSON_LUCY_FLOAT32_ABSOLUTE_FLOOR",
    "RICHARDSON_LUCY_FLOAT32_MAX_ABS_BASE",
    "RICHARDSON_LUCY_FLOAT32_MAX_ABS_PEAK_FACTOR",
    "RICHARDSON_LUCY_FLOAT32_NEAR_IDENTITY_NRMSE_LIMIT",
    "RICHARDSON_LUCY_FLOAT32_NRMSE_LIMIT",
    "RICHARDSON_LUCY_PARITY_OPERATION_IDS",
    "RICHARDSON_LUCY_TV_FLOAT32_MAX_ABS_BASE",
    "RICHARDSON_LUCY_TV_FLOAT32_MAX_ABS_PEAK_FACTOR",
    "RICHARDSON_LUCY_TV_FLOAT32_NRMSE_LIMIT",
    "RICHARDSON_LUCY_TV_PARITY_OPERATION_IDS",
    "richardson_lucy_float32_parity",
    "richardson_lucy_tv_float32_parity",
]
