#!/usr/bin/env python
"""Build reproducible CPU/CuPy Sigma Filter admission and timing evidence.

The default ``full`` profile exercises exact, branch-sensitive parity and then
times radii 0.5, 2, 5, and 10 on 256² through 2048² planes plus representative
plane-wise stacks.  Timings distinguish case-cold calls, warm resident compute,
explicit transfers, and transfer-inclusive end-to-end calls.  The resulting
JSON and Markdown are machine-local screening evidence, not a portable speed
claim or a durable optimizer record.

Importing this module, asking for ``--help``, and ``--validate-existing`` do not
import CuPy or initialize CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

SCHEMA = "napari-vipp-cupy-sigma-filter-evidence"
SCHEMA_VERSION = 1
OPERATION_ID = "sigma_filter"
IMPLEMENTATION_ID = "cupy-sigma-filter-v1"
GENERATOR_ID = "numpy-pcg64-structured-confocal-plane-v1"
BENCHMARK_ROUNDS = 7
ROUND_ORDER = (
    "gpu",
    "cpu",
    "cpu",
    "gpu",
    "gpu",
    "cpu",
    "cpu",
    "gpu",
    "gpu",
    "cpu",
    "cpu",
    "gpu",
    "gpu",
    "cpu",
)
BOOTSTRAP_SAMPLES = 2_000
BOOTSTRAP_SEED = 20_260_802
CONFIDENCE_LEVEL = 0.95
MINIMUM_CONFIDENT_SPEEDUP = 1.20
MINIMUM_RELATIVE_SAVING = 0.05
MINIMUM_ABSOLUTE_SAVING_SECONDS = 0.020
PLANE_EXTENTS = (256, 512, 1024, 2048)
RADII = (0.5, 2.0, 5.0, 10.0)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/benchmarks/sigma-filter-cupy-windows-rtx5090.json"
SOURCE_PROVENANCE_PATHS = (
    Path("src/napari_vipp/core/sigma_filter.py"),
    Path("src/napari_vipp/core/gpu/cupy_sigma.py"),
    Path("scripts/benchmark_gpu_sigma.py"),
)

REQUIRED_ADMISSION_COVERAGE = frozenset(
    {
        "dtype:uint8",
        "dtype:uint16",
        "dtype:float32",
        "radius:0.5",
        "radius:2",
        "radius:5",
        "radius:10",
        "sigma-width:0-inclusive",
        "sigma-width:default",
        "minimum-fraction:0",
        "minimum-fraction:0.2",
        "minimum-fraction:0.8",
        "minimum-fraction:1",
        "fallback:exclude-center",
        "fallback:full-mean",
        "boundary:nearest-clamp",
        "restore:float32-half-up",
        "float32:negative-zero",
        "float32:subnormal-sample",
        "float32:subnormal-square",
        "axes:leading-planes",
        "axes:explicit-channel",
        "plane:tiny",
    }
)
REQUIRED_REJECTION_COVERAGE = frozenset(
    {
        "reject:dtype",
        "reject:radius",
        "reject:sigma-width",
        "reject:minimum-fraction",
        "reject:outlier-aware",
        "reject:channel-axis",
        "reject:byte-order",
        "reject:nonfinite",
        "reject:float32-square-overflow",
    }
)

_ROOT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "created_utc",
        "kind",
        "portable_performance_claim",
        "durable_optimizer_record",
        "profile",
        "method",
        "platform",
        "packages",
        "source_provenance",
        "admission",
        "rejections",
        "lifecycle",
        "performance",
    }
)
_PERFORMANCE_RESULT_KEYS = frozenset(
    {
        "case_id",
        "label",
        "source_kind",
        "source_metadata",
        "shape",
        "element_count",
        "input_bytes",
        "dtype",
        "radius",
        "parameters",
        "input_sha256",
        "parity",
        "samples",
        "summary",
        "bootstrap_seed",
        "cleanup",
    }
)


class EvidenceError(RuntimeError):
    """A complete and internally consistent evidence artifact was not produced."""


@dataclass(frozen=True, slots=True)
class _AdmissionCase:
    case_id: str
    data: Any
    parameters: Mapping[str, object]
    coverage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _RejectionCase:
    case_id: str
    data: Any
    parameters: Mapping[str, object]
    coverage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PerformanceCase:
    case_id: str
    label: str
    shape: tuple[int, ...]
    radius: float
    seed: int

    @property
    def source_kind(self) -> str:
        return (
            "deterministic-synthetic-stack"
            if len(self.shape) > 2
            else ("deterministic-synthetic-plane")
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Canonical JSON output (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Readable summary (default: output with an .md suffix).",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="full",
        help="Use the required full matrix or a short development smoke profile.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device index (default: 0).",
    )
    parser.add_argument(
        "--validate-existing",
        type=Path,
        help="Validate existing JSON/Markdown without importing CuPy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_existing is not None:
        try:
            artifact = validate_existing(args.validate_existing)
        except (OSError, TypeError, ValueError, EvidenceError) as exc:
            print(f"Sigma Filter evidence validation failed: {exc}", file=sys.stderr)
            return 2
        print(f"Sigma Filter evidence is current: {artifact}")
        return 0

    if isinstance(args.device_index, bool) or args.device_index < 0:
        print("CUDA device index must be non-negative.", file=sys.stderr)
        return 2
    markdown = args.markdown or args.output.with_suffix(".md")
    try:
        document = run_evidence(
            profile=args.profile,
            device_index=args.device_index,
        )
        _atomic_write_artifacts(args.output, markdown, document)
    except (OSError, TypeError, ValueError, EvidenceError) as exc:
        print(f"Sigma Filter evidence run failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # concise boundary for optional CUDA failures
        print(
            f"Sigma Filter evidence run failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"Wrote Sigma Filter evidence to {args.output.resolve()}")
    print(f"Wrote readable summary to {markdown.resolve()}")
    return 0


def run_evidence(
    *,
    profile: str = "full",
    device_index: int = 0,
) -> dict[str, object]:
    """Run exact admission and bounded machine-local performance evidence."""

    selected_profile = str(profile).strip().lower()
    if selected_profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'.")
    if isinstance(device_index, bool) or int(device_index) < 0:
        raise ValueError("CUDA device index must be a non-negative integer.")

    source_provenance = _source_provenance()
    import cupy

    from napari_vipp.core.gpu.cupy_sigma import sigma_filter as gpu_sigma
    from napari_vipp.core.sigma_filter import sigma_filter as cpu_sigma

    with cupy.cuda.Device(int(device_index)):
        platform_record = _platform_record(cupy, int(device_index))
        admission = _run_admission(
            cupy=cupy,
            cpu_function=cpu_sigma,
            gpu_function=gpu_sigma,
        )
        rejections = _run_rejections(
            cupy=cupy,
            cpu_function=cpu_sigma,
            gpu_function=gpu_sigma,
        )
        lifecycle = _run_lifecycle(cupy=cupy, gpu_function=gpu_sigma)
        performance = _run_performance(
            _performance_cases(selected_profile),
            cupy=cupy,
            cpu_function=cpu_sigma,
            gpu_function=gpu_sigma,
        )

    _require_source_snapshot_unchanged(source_provenance)
    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "scientific-admission-and-machine-local-performance-evidence",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": selected_profile,
        "method": _method_record(selected_profile),
        "platform": platform_record,
        "packages": _package_versions(),
        "source_provenance": source_provenance,
        "admission": admission,
        "rejections": rejections,
        "lifecycle": lifecycle,
        "performance": performance,
    }
    _validate_document_contract(document)
    return document


def _method_record(profile: str) -> dict[str, object]:
    return {
        "operation_id": OPERATION_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "generator_id": GENERATOR_ID,
        "profile": profile,
        "timed_rounds": BENCHMARK_ROUNDS,
        "round_order": list(ROUND_ORDER),
        "bootstrap_samples": BOOTSTRAP_SAMPLES,
        "bootstrap_confidence_level": CONFIDENCE_LEVEL,
        "minimum_confident_speedup": MINIMUM_CONFIDENT_SPEEDUP,
        "minimum_relative_saving": MINIMUM_RELATIVE_SAVING,
        "minimum_absolute_saving_seconds": MINIMUM_ABSOLUTE_SAVING_SECONDS,
        "cpu_timing_scope": "synchronous-production-operation-call-v1",
        "gpu_case_cold_scope": (
            "unwarmed-case-host-to-device-compute-device-to-host-v1"
        ),
        "gpu_end_to_end_scope": (
            "host-to-device-synchronized-compute-device-to-host-v1"
        ),
        "gpu_resident_scope": "synchronized-resident-compute-v1",
        "gpu_transfer_scope": "host-to-device-plus-device-to-host-v1",
        "provider_jit_precompiled_by_admission": True,
        "disk_io_included": False,
        "input_generation_included": False,
        "exact_parity_required_before_timing": True,
        "durable_optimizer_record": False,
    }


def _admission_cases() -> tuple[_AdmissionCase, ...]:
    import numpy as np

    structured = np.asarray(
        [
            [0, 0, 1, 3, 8, 13, 21],
            [0, 2, 4, 6, 8, 10, 12],
            [1, 4, 9, 250, 9, 4, 1],
            [2, 6, 10, 14, 10, 6, 2],
            [50, 12, 8, 4, 0, 4, 8],
        ]
    )
    half_up = np.zeros((3, 3), dtype=np.uint8)
    half_up[1, 1] = 10
    half_up[1, 0] = half_up[1, 2] = 1
    smallest = np.nextafter(np.float32(0.0), np.float32(1.0))
    subnormal_samples = np.asarray(
        [
            [0.0, smallest, -smallest],
            [smallest * 2, -smallest * 2, 0.0],
            [smallest * 4, smallest, -smallest],
        ],
        dtype=np.float32,
    )
    subnormal_squares = np.asarray(
        [
            [0.0, 1e-20, -1e-20],
            [2e-20, -2e-20, 0.0],
            [4e-20, 1e-20, -1e-20],
        ],
        dtype=np.float32,
    )
    negative_zero = np.full((3, 3), np.float32(-0.0), dtype=np.float32)
    leading = np.arange(2 * 3 * 5 * 7, dtype=np.uint16).reshape(2, 3, 5, 7)
    channel = np.arange(2 * 3 * 5 * 7, dtype=np.uint16).reshape(2, 3, 5, 7)
    cases = (
        _AdmissionCase(
            "uint8-half-up-border-fallback",
            half_up,
            {
                "radius": 0.5,
                "sigma_width": 0.0,
                "minimum_pixel_fraction": 1.0,
                "outlier_aware": True,
            },
            (
                "dtype:uint8",
                "radius:0.5",
                "sigma-width:0-inclusive",
                "minimum-fraction:1",
                "fallback:exclude-center",
                "boundary:nearest-clamp",
                "restore:float32-half-up",
            ),
        ),
        _AdmissionCase(
            "uint16-default-structured",
            structured.astype(np.uint16),
            {"radius": 2.0},
            (
                "dtype:uint16",
                "radius:2",
                "sigma-width:default",
                "minimum-fraction:0.2",
            ),
        ),
        _AdmissionCase(
            "uint16-full-mean-wide",
            structured.astype(np.uint16),
            {
                "radius": 5.0,
                "sigma_width": 3.0,
                "minimum_pixel_fraction": 0.8,
                "outlier_aware": False,
            },
            (
                "radius:5",
                "minimum-fraction:0.8",
                "fallback:full-mean",
            ),
        ),
        _AdmissionCase(
            "uint16-radius-ten-zero-minimum",
            structured.astype(np.uint16),
            {
                "radius": 10.0,
                "minimum_pixel_fraction": 0.0,
            },
            ("radius:10", "minimum-fraction:0"),
        ),
        _AdmissionCase(
            "float32-negative-zero",
            negative_zero,
            {"radius": 0.5},
            ("dtype:float32", "float32:negative-zero"),
        ),
        _AdmissionCase(
            "float32-subnormal-samples",
            subnormal_samples,
            {
                "radius": 2.0,
                "sigma_width": 0.0,
                "minimum_pixel_fraction": 1.0,
            },
            ("float32:subnormal-sample",),
        ),
        _AdmissionCase(
            "float32-subnormal-squares",
            subnormal_squares,
            {
                "radius": 2.0,
                "sigma_width": 1.0,
                "minimum_pixel_fraction": 0.8,
                "outlier_aware": False,
            },
            ("float32:subnormal-square",),
        ),
        _AdmissionCase(
            "leading-plane-stack",
            leading,
            {"radius": 2.0},
            ("axes:leading-planes",),
        ),
        _AdmissionCase(
            "explicit-channel-axis",
            channel,
            {"radius": 2.0, "channel_axis": 1},
            ("axes:explicit-channel",),
        ),
        _AdmissionCase(
            "tiny-plane-radius-ten",
            np.asarray([[65_535]], dtype=np.uint16),
            {"radius": 10.0},
            ("plane:tiny",),
        ),
    )
    for case in cases:
        case.data.setflags(write=False)
    return cases


def _run_admission(*, cupy, cpu_function, gpu_function) -> dict[str, object]:
    import numpy as np

    records: list[dict[str, object]] = []
    covered: set[str] = set()
    for case in _admission_cases():
        host_before = np.asarray(case.data).copy()
        cpu_output = np.asarray(cpu_function(case.data, **case.parameters))
        device_input = cupy.asarray(case.data)
        device_before = device_input.copy()
        device_output = gpu_function(device_input, **case.parameters)
        if not isinstance(device_output, cupy.ndarray):
            raise EvidenceError(f"{case.case_id} did not return a resident array.")
        cupy.cuda.get_current_stream().synchronize()
        gpu_output = np.asarray(cupy.asnumpy(device_output))
        mismatch_count = _bitwise_mismatch_count(cpu_output, gpu_output)
        if mismatch_count:
            raise EvidenceError(
                f"{case.case_id} produced {mismatch_count} bitwise mismatches."
            )
        if not bool(cupy.array_equal(device_input, device_before).item()):
            raise EvidenceError(f"{case.case_id} mutated its device input.")
        np.testing.assert_array_equal(np.asarray(case.data), host_before)
        if not bool(device_output.flags.c_contiguous):
            raise EvidenceError(f"{case.case_id} returned non-contiguous output.")
        records.append(
            {
                "case_id": case.case_id,
                "shape": list(case.data.shape),
                "dtype": case.data.dtype.name,
                "parameters": dict(case.parameters),
                "coverage": list(case.coverage),
                "exact": True,
                "mismatch_count": 0,
                "gpu_output_resident": True,
                "gpu_output_contiguous": True,
                "input_immutable": True,
                "cpu_output_sha256": _array_sha256(cpu_output),
                "gpu_output_sha256": _array_sha256(gpu_output),
            }
        )
        covered.update(case.coverage)
        del device_input, device_before, device_output
    missing = REQUIRED_ADMISSION_COVERAGE - covered
    if missing:
        raise EvidenceError(f"Admission coverage is incomplete: {sorted(missing)}")
    return {
        "status": "pass",
        "parity_profile": "bitwise-identical-dtype-shape-and-signed-zero-v1",
        "case_count": len(records),
        "coverage": sorted(covered),
        "cases": records,
    }


def _rejection_cases() -> tuple[_RejectionCase, ...]:
    import numpy as np

    nonfinite = np.ones((3, 3), dtype=np.float32)
    nonfinite[1, 1] = np.nan
    overflow = np.ones((3, 3), dtype=np.float32)
    overflow[1, 1] = np.finfo(np.float32).max
    return (
        _RejectionCase(
            "unsupported-float64",
            np.ones((3, 3), dtype=np.float64),
            {},
            ("reject:dtype",),
        ),
        _RejectionCase(
            "non-native-uint16",
            np.ones((3, 3), dtype=">u2"),
            {},
            ("reject:byte-order",),
        ),
        _RejectionCase(
            "non-native-float32",
            np.ones((3, 3), dtype=">f4"),
            {},
            ("reject:byte-order",),
        ),
        _RejectionCase(
            "radius-below-minimum",
            np.ones((3, 3), dtype=np.uint8),
            {"radius": 0.49},
            ("reject:radius",),
        ),
        _RejectionCase(
            "negative-sigma-width",
            np.ones((3, 3), dtype=np.uint8),
            {"sigma_width": -1.0},
            ("reject:sigma-width",),
        ),
        _RejectionCase(
            "fraction-above-one",
            np.ones((3, 3), dtype=np.uint8),
            {"minimum_pixel_fraction": 1.01},
            ("reject:minimum-fraction",),
        ),
        _RejectionCase(
            "nonboolean-outlier-mode",
            np.ones((3, 3), dtype=np.uint8),
            {"outlier_aware": 1},
            ("reject:outlier-aware",),
        ),
        _RejectionCase(
            "boolean-channel-axis",
            np.ones((3, 3, 3), dtype=np.uint8),
            {"channel_axis": True},
            ("reject:channel-axis",),
        ),
        _RejectionCase(
            "nonfinite-float32",
            nonfinite,
            {},
            ("reject:nonfinite",),
        ),
        _RejectionCase(
            "square-unsafe-float32",
            overflow,
            {},
            ("reject:float32-square-overflow",),
        ),
    )


def _run_rejections(*, cupy, cpu_function, gpu_function) -> dict[str, object]:
    records: list[dict[str, object]] = []
    covered: set[str] = set()
    for case in _rejection_cases():
        cpu_error = _captured_value_error(
            lambda case=case: cpu_function(case.data, **case.parameters)
        )
        gpu_input = (
            case.data
            if "reject:byte-order" in case.coverage
            else cupy.asarray(case.data)
        )
        gpu_error = _captured_value_error(
            lambda case=case, gpu_input=gpu_input: gpu_function(
                gpu_input,
                **case.parameters,
            )
        )
        if cpu_error != gpu_error:
            raise EvidenceError(
                f"{case.case_id} CPU/GPU rejection differed: "
                f"{cpu_error!r} != {gpu_error!r}."
            )
        records.append(
            {
                "case_id": case.case_id,
                "coverage": list(case.coverage),
                "error_type": "ValueError",
                "message": cpu_error,
                "cpu_gpu_message_exact": True,
            }
        )
        covered.update(case.coverage)
    missing = REQUIRED_REJECTION_COVERAGE - covered
    if missing:
        raise EvidenceError(f"Rejection coverage is incomplete: {sorted(missing)}")
    return {
        "status": "pass",
        "case_count": len(records),
        "coverage": sorted(covered),
        "cases": records,
    }


def _captured_value_error(call) -> str:
    try:
        call()
    except ValueError as exc:
        return str(exc)
    raise EvidenceError("Expected ValueError was not raised.")


def _run_lifecycle(*, cupy, gpu_function) -> dict[str, object]:
    from napari_vipp.core.progress import OperationCancelled, ProgressContext

    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 2,
        reporter=updates.append,
    )
    cancelled = False
    try:
        gpu_function(
            cupy.ones((130, 64), dtype=cupy.uint16),
            radius=10.0,
            progress=progress,
        )
    except OperationCancelled:
        cancelled = True
    cupy.cuda.get_current_stream().synchronize()
    pool = cupy.get_default_memory_pool()
    pinned_pool = cupy.get_default_pinned_memory_pool()
    pool.free_all_blocks()
    pinned_pool.free_all_blocks()
    cupy.cuda.get_current_stream().synchronize()
    if not cancelled or len(updates) != 2:
        raise EvidenceError("Sigma Filter cancellation was not observed at a tile.")
    if int(pool.used_bytes()) or int(pool.total_bytes()):
        raise EvidenceError("CUDA memory pool did not drain after cancellation.")
    return {
        "status": "pass",
        "cancelled": True,
        "boundary": "synchronized-row-tile-v1",
        "reported_progress": [
            {
                "current": int(update.current),
                "total": int(update.total),
                "message": update.message,
            }
            for update in updates
        ],
        "device_pool_used_bytes_after_cleanup": int(pool.used_bytes()),
        "device_pool_reserved_bytes_after_cleanup": int(pool.total_bytes()),
    }


def _performance_cases(profile: str) -> tuple[_PerformanceCase, ...]:
    if profile == "quick":
        return (
            _PerformanceCase("plane-64-r0p5", "64² plane, r=0.5", (64, 64), 0.5, 641),
            _PerformanceCase("plane-128-r2", "128² plane, r=2", (128, 128), 2.0, 1282),
            _PerformanceCase(
                "stack-2x128-r10",
                "2×128² stack, r=10",
                (2, 128, 128),
                10.0,
                212810,
            ),
        )
    if profile != "full":
        raise ValueError("profile must be 'quick' or 'full'.")
    cases = []
    for radius in RADII:
        radius_id = str(radius).replace(".", "p")
        for extent in PLANE_EXTENTS:
            cases.append(
                _PerformanceCase(
                    f"plane-{extent}-r{radius_id}",
                    f"{extent}² plane, r={radius:g}",
                    (extent, extent),
                    radius,
                    90_000 + extent,
                )
            )
    cases.extend(
        (
            _PerformanceCase(
                "stack-8x512-r2p0",
                "8×512² plane-wise stack, r=2",
                (8, 512, 512),
                2.0,
                805_122,
            ),
            _PerformanceCase(
                "stack-4x1024-r10p0",
                "4×1024² plane-wise stack, r=10",
                (4, 1024, 1024),
                10.0,
                410_241,
            ),
        )
    )
    return tuple(cases)


@cache
def _synthetic_image(shape: tuple[int, ...], seed: int):
    """Return deterministic uint16 texture with edges, spots, and outliers."""

    import numpy as np

    rows, columns = shape[-2:]
    leading_shape = shape[:-2]
    plane_count = math.prod(leading_shape) if leading_shape else 1
    result = np.empty((plane_count, rows, columns), dtype=np.uint16)
    yy = np.arange(rows, dtype=np.int32)[:, None]
    xx = np.arange(columns, dtype=np.int32)[None, :]
    rng = np.random.default_rng(seed)
    for plane_index in range(plane_count):
        checker = ((yy // 24) ^ (xx // 24)) * 223
        pattern = (yy * 37 + xx * 19 + checker + plane_index * 131) % 4096
        noise = rng.integers(0, 512, size=(rows, columns), dtype=np.uint16)
        plane = pattern.astype(np.uint16)
        plane += noise
        for spot_index in range(4):
            center_y = (rows * (spot_index + 1)) // 5
            center_x = (columns * (4 - spot_index)) // 5
            spot_radius = max(min(rows, columns) // (18 + spot_index * 2), 2)
            mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= spot_radius**2
            plane[mask] = np.minimum(
                plane[mask].astype(np.uint32) + 20_000 + 3_000 * spot_index,
                65_535,
            ).astype(np.uint16)
        plane[0, 0] = 0
        plane[-1, -1] = 65_535
        result[plane_index] = plane
    restored = np.ascontiguousarray(result.reshape(shape))
    restored.setflags(write=False)
    return restored


def _run_performance(
    cases: Sequence[_PerformanceCase],
    *,
    cupy,
    cpu_function,
    gpu_function,
) -> dict[str, object]:
    results = []
    for index, definition in enumerate(cases, start=1):
        print(
            f"[{index}/{len(cases)}] Timing {definition.label}...",
            flush=True,
        )
        data = _synthetic_image(definition.shape, definition.seed)
        _resource_preflight(data, cupy=cupy)
        record = _benchmark_case(
            definition,
            data,
            cupy=cupy,
            cpu_function=cpu_function,
            gpu_function=gpu_function,
            bootstrap_seed=BOOTSTRAP_SEED + index,
        )
        results.append(record)
        summary = record["summary"]
        print(
            f"[{index}/{len(cases)}] CPU "
            f"{summary['cpu_median_seconds']:.4f}s; GPU end-to-end "
            f"{summary['gpu_end_to_end_median_seconds']:.4f}s; "
            f"{summary['gpu_end_to_end_speedup']:.2f}x; "
            f"{summary['screening_choice']}.",
            flush=True,
        )
    return {
        "status": "pass",
        "case_count": len(results),
        "results": results,
        "crossover": _crossover_summary(results),
    }


def _resource_preflight(data, *, cupy) -> None:
    free_bytes, total_bytes = (int(value) for value in cupy.cuda.runtime.memGetInfo())
    conservative_need = int(data.nbytes) * 12 + 256 * 1024**2
    if free_bytes < conservative_need:
        raise EvidenceError(
            f"CUDA preflight needs {conservative_need:,} free bytes for "
            f"{data.shape}, but only {free_bytes:,} of {total_bytes:,} are free."
        )


def _benchmark_case(
    definition: _PerformanceCase,
    data,
    *,
    cupy,
    cpu_function,
    gpu_function,
    bootstrap_seed: int,
) -> dict[str, object]:
    import numpy as np

    parameters = {
        "radius": definition.radius,
        "sigma_width": 2.0,
        "minimum_pixel_fraction": 0.2,
        "outlier_aware": True,
        "channel_axis": None,
    }
    cpu_started = time.perf_counter()
    cpu_reference = np.asarray(cpu_function(data, **parameters))
    cpu_cold_seconds = time.perf_counter() - cpu_started

    pool = cupy.get_default_memory_pool()
    pinned_pool = cupy.get_default_pinned_memory_pool()
    pool.free_all_blocks()
    pinned_pool.free_all_blocks()
    cupy.cuda.get_current_stream().synchronize()
    gpu_started = time.perf_counter()
    cold_input = cupy.asarray(data)
    cold_output_device = gpu_function(cold_input, **parameters)
    if not isinstance(cold_output_device, cupy.ndarray):
        raise EvidenceError(f"{definition.case_id} GPU output was not resident.")
    cold_output = np.asarray(cupy.asnumpy(cold_output_device))
    cupy.cuda.get_current_stream().synchronize()
    gpu_cold_seconds = time.perf_counter() - gpu_started
    mismatch_count = _bitwise_mismatch_count(cpu_reference, cold_output)
    if mismatch_count:
        raise EvidenceError(
            f"Timing refused: {definition.case_id} had {mismatch_count} mismatches."
        )
    del cold_input, cold_output_device

    resident_input = cupy.asarray(data)
    warm_resident = gpu_function(resident_input, **parameters)
    cupy.cuda.get_current_stream().synchronize()
    del warm_resident

    cpu_seconds: list[float] = []
    gpu_end_to_end_seconds: list[float] = []
    for candidate in ROUND_ORDER:
        if candidate == "cpu":
            started = time.perf_counter()
            cpu_function(data, **parameters)
            cpu_seconds.append(time.perf_counter() - started)
        else:
            cupy.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            device_input = cupy.asarray(data)
            device_output = gpu_function(device_input, **parameters)
            cupy.asnumpy(device_output)
            cupy.cuda.get_current_stream().synchronize()
            gpu_end_to_end_seconds.append(time.perf_counter() - started)
            del device_input, device_output

    gpu_resident_seconds: list[float] = []
    for _ in range(BENCHMARK_ROUNDS):
        cupy.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        resident_output = gpu_function(resident_input, **parameters)
        cupy.cuda.get_current_stream().synchronize()
        gpu_resident_seconds.append(time.perf_counter() - started)
        del resident_output

    transfer_seconds: list[float] = []
    for _ in range(BENCHMARK_ROUNDS):
        cupy.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        transfer_input = cupy.asarray(data)
        cupy.asnumpy(transfer_input)
        cupy.cuda.get_current_stream().synchronize()
        transfer_seconds.append(time.perf_counter() - started)
        del transfer_input

    del resident_input
    cupy.cuda.get_current_stream().synchronize()
    used_before_cleanup = int(pool.used_bytes())
    pool.free_all_blocks()
    pinned_pool.free_all_blocks()
    cupy.cuda.get_current_stream().synchronize()
    cleanup = {
        "used_bytes_before_cleanup": used_before_cleanup,
        "used_bytes_after_cleanup": int(pool.used_bytes()),
        "reserved_bytes_after_cleanup": int(pool.total_bytes()),
    }
    if cleanup["used_bytes_after_cleanup"] or cleanup["reserved_bytes_after_cleanup"]:
        raise EvidenceError(f"{definition.case_id} CUDA pool cleanup failed.")

    summary = _timing_summary(
        cpu_seconds,
        gpu_end_to_end_seconds,
        gpu_resident_seconds,
        transfer_seconds,
        cpu_cold_seconds=cpu_cold_seconds,
        gpu_cold_seconds=gpu_cold_seconds,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "case_id": definition.case_id,
        "label": definition.label,
        "source_kind": definition.source_kind,
        "source_metadata": {"generator": GENERATOR_ID, "seed": definition.seed},
        "shape": list(definition.shape),
        "element_count": int(data.size),
        "input_bytes": int(data.nbytes),
        "dtype": data.dtype.name,
        "radius": definition.radius,
        "parameters": parameters,
        "input_sha256": _array_sha256(data),
        "parity": {
            "profile": "bitwise-identical-uint16-v1",
            "passed": True,
            "mismatch_count": 0,
            "gpu_output_resident": True,
            "cpu_output_sha256": _array_sha256(cpu_reference),
            "gpu_output_sha256": _array_sha256(cold_output),
        },
        "samples": {
            "cpu_seconds": cpu_seconds,
            "gpu_end_to_end_seconds": gpu_end_to_end_seconds,
            "gpu_resident_seconds": gpu_resident_seconds,
            "gpu_transfer_seconds": transfer_seconds,
            "cpu_case_cold_seconds": cpu_cold_seconds,
            "gpu_case_cold_end_to_end_seconds": gpu_cold_seconds,
        },
        "summary": summary,
        "bootstrap_seed": bootstrap_seed,
        "cleanup": cleanup,
    }


def _timing_summary(
    cpu_seconds: Sequence[float],
    gpu_end_to_end_seconds: Sequence[float],
    gpu_resident_seconds: Sequence[float],
    transfer_seconds: Sequence[float],
    *,
    cpu_cold_seconds: float,
    gpu_cold_seconds: float,
    bootstrap_seed: int,
) -> dict[str, object]:
    cpu_median = statistics.median(cpu_seconds)
    gpu_e2e_median = statistics.median(gpu_end_to_end_seconds)
    gpu_resident_median = statistics.median(gpu_resident_seconds)
    transfer_median = statistics.median(transfer_seconds)
    paired_speedups = [
        cpu / gpu
        for cpu, gpu in zip(
            cpu_seconds,
            gpu_end_to_end_seconds,
            strict=True,
        )
    ]
    confidence_low, confidence_high = _bootstrap_median_interval(
        paired_speedups,
        seed=bootstrap_seed,
    )
    absolute_saving = cpu_median - gpu_e2e_median
    required_saving = max(
        MINIMUM_ABSOLUTE_SAVING_SECONDS,
        MINIMUM_RELATIVE_SAVING * cpu_median,
    )
    speed_gate = confidence_low >= MINIMUM_CONFIDENT_SPEEDUP
    saving_gate = absolute_saving >= required_saving
    auto_gate = speed_gate and saving_gate
    return {
        "cpu_median_seconds": cpu_median,
        "gpu_end_to_end_median_seconds": gpu_e2e_median,
        "gpu_resident_median_seconds": gpu_resident_median,
        "gpu_transfer_median_seconds": transfer_median,
        "cpu_case_cold_seconds": float(cpu_cold_seconds),
        "gpu_case_cold_end_to_end_seconds": float(gpu_cold_seconds),
        "gpu_end_to_end_speedup": cpu_median / gpu_e2e_median,
        "gpu_resident_speedup": cpu_median / gpu_resident_median,
        "paired_speedups": paired_speedups,
        "paired_speedup_median": statistics.median(paired_speedups),
        "paired_speedup_confidence_low": confidence_low,
        "paired_speedup_confidence_high": confidence_high,
        "absolute_median_saving_seconds": absolute_saving,
        "required_saving_seconds": required_saving,
        "confidence_speed_gate_passed": speed_gate,
        "material_saving_gate_passed": saving_gate,
        "auto_performance_gate_passed": auto_gate,
        "screening_choice": "GPU-CuPy" if auto_gate else "CPU",
    }


def _bootstrap_median_interval(
    values: Sequence[float],
    *,
    seed: int,
) -> tuple[float, float]:
    if len(values) != BENCHMARK_ROUNDS:
        raise EvidenceError("Paired speedup samples are incomplete.")
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(tuple(values), k=len(values)))
        for _ in range(BOOTSTRAP_SAMPLES)
    )
    tail = (1.0 - CONFIDENCE_LEVEL) / 2.0
    low_index = max(int(math.floor(tail * BOOTSTRAP_SAMPLES)), 0)
    high_index = min(
        int(math.ceil((1.0 - tail) * BOOTSTRAP_SAMPLES)) - 1,
        BOOTSTRAP_SAMPLES - 1,
    )
    return float(medians[low_index]), float(medians[high_index])


def _crossover_summary(results: Sequence[Mapping[str, object]]) -> dict[str, object]:
    per_radius: dict[str, object] = {}
    plane_results = [item for item in results if len(item["shape"]) == 2]
    for radius in sorted({float(item["radius"]) for item in plane_results}):
        matching = sorted(
            (item for item in plane_results if float(item["radius"]) == radius),
            key=lambda item: int(item["element_count"]),
        )
        first_confident = None
        for index, item in enumerate(matching):
            if all(
                bool(candidate["summary"]["auto_performance_gate_passed"])
                for candidate in matching[index:]
            ):
                first_confident = item
                break
        per_radius[f"{radius:g}"] = {
            "tested_extents": [int(item["shape"][-1]) for item in matching],
            "all_tested_cases_choose_gpu": all(
                bool(item["summary"]["auto_performance_gate_passed"])
                for item in matching
            ),
            "smallest_confident_gpu_extent": (
                None if first_confident is None else int(first_confident["shape"][-1])
            ),
            "smallest_confident_gpu_pixels": (
                None
                if first_confident is None
                else int(first_confident["element_count"])
            ),
            "observed_cpu_winner_extents": [
                int(item["shape"][-1])
                for item in matching
                if not bool(item["summary"]["auto_performance_gate_passed"])
            ],
        }
    stack_results = [item for item in results if len(item["shape"]) > 2]
    return {
        "per_radius": per_radius,
        "stack_cases_all_choose_gpu": all(
            bool(item["summary"]["auto_performance_gate_passed"])
            for item in stack_results
        ),
        "observed_crossover_interpretation": (
            "The smallest confident GPU extent is bounded by the tested grid. "
            "Do not extrapolate below it or to another machine."
        ),
    }


def _platform_record(cupy, device_index: int) -> dict[str, object]:
    uname = platform.uname()
    properties = cupy.cuda.runtime.getDeviceProperties(device_index)
    name = properties.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    return {
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "python": platform.python_version(),
        "device_index": device_index,
        "device_name": str(name),
        "compute_capability": (
            f"{int(properties.get('major', 0))}.{int(properties.get('minor', 0))}"
        ),
        "total_device_memory_bytes": int(properties.get("totalGlobalMem", 0)),
        "cuda_driver_version": int(cupy.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
    }


def _package_versions() -> dict[str, str]:
    names = (
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy-cuda13x",
    )
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _source_provenance() -> list[dict[str, str]]:
    return [
        {
            "path": path.as_posix(),
            "sha256": _file_sha256(PROJECT_ROOT / path),
        }
        for path in SOURCE_PROVENANCE_PATHS
    ]


def _require_source_snapshot_unchanged(
    snapshot: Sequence[Mapping[str, str]],
) -> None:
    if list(snapshot) != _source_provenance():
        raise EvidenceError("Source files changed while evidence was running.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value) -> str:
    import numpy as np

    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _bitwise_mismatch_count(expected, actual) -> int:
    import numpy as np

    first = np.asarray(expected)
    second = np.asarray(actual)
    if first.shape != second.shape or first.dtype != second.dtype:
        return max(int(first.size), int(second.size), 1)
    itemsize = int(first.dtype.itemsize)
    first_bytes = np.ascontiguousarray(first).view(np.uint8).reshape(-1, itemsize)
    second_bytes = np.ascontiguousarray(second).view(np.uint8).reshape(-1, itemsize)
    return int(np.count_nonzero(np.any(first_bytes != second_bytes, axis=1)))


def render_markdown(document: Mapping[str, object]) -> str:
    _validate_document_contract(document)
    platform_record = _mapping(document["platform"], "platform")
    admission = _mapping(document["admission"], "admission")
    rejections = _mapping(document["rejections"], "rejections")
    lifecycle = _mapping(document["lifecycle"], "lifecycle")
    performance = _mapping(document["performance"], "performance")
    lines = [
        "# CuPy Sigma Filter admission and performance evidence",
        "",
        f"- Generated: `{document['created_utc']}`",
        f"- Device: `{platform_record['device_name']}`",
        f"- Profile: `{document['profile']}`",
        f"- Exact admission cases: `{admission['case_count']}`",
        f"- Matched rejection cases: `{rejections['case_count']}`",
        f"- Timed cases: `{performance['case_count']}`",
        f"- Cancellation and cleanup: `{lifecycle['status']}`",
        "",
        "Every admission and timed workload compared the production CPU operation",
        "with the resident CuPy provider bit for bit, including float32 signed zero",
        "and subnormal arithmetic. Timings are a short machine-local screen, not a",
        "portable claim or durable optimizer record. Case-cold GPU timings include",
        "allocations and transfers but not first-process JIT, because admission",
        "compiled the provider first.",
        "",
        "| Case | Shape | Radius | CPU warm | GPU case-cold E2E | GPU warm E2E | "
        "GPU resident | Transfers | E2E speedup | 95% paired lower | Choice |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for raw_result in performance["results"]:
        result = _mapping(raw_result, "performance result")
        summary = _mapping(result["summary"], "timing summary")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(result["label"]),
                    "×".join(str(value) for value in result["shape"]),
                    f"{float(result['radius']):g}",
                    _seconds(summary["cpu_median_seconds"]),
                    _seconds(summary["gpu_case_cold_end_to_end_seconds"]),
                    _seconds(summary["gpu_end_to_end_median_seconds"]),
                    _seconds(summary["gpu_resident_median_seconds"]),
                    _seconds(summary["gpu_transfer_median_seconds"]),
                    _ratio(summary["gpu_end_to_end_speedup"]),
                    _ratio(summary["paired_speedup_confidence_low"]),
                    str(summary["screening_choice"]),
                )
            )
            + " |"
        )

    crossover = _mapping(performance["crossover"], "crossover")
    lines.extend(("", "## Reviewed crossover screen", ""))
    per_radius = _mapping(crossover["per_radius"], "per-radius crossover")
    for radius in sorted(per_radius, key=float):
        raw_record = per_radius[radius]
        record = _mapping(raw_record, "radius crossover")
        extent = record["smallest_confident_gpu_extent"]
        if extent is None:
            statement = "no tested extent cleared both gates"
        else:
            statement = f"GPU cleared both gates from {int(extent)}²"
        lines.append(f"- **Radius {radius}:** {statement} on this machine.")
    lines.extend(
        (
            "",
            str(crossover["observed_crossover_interpretation"]),
            "",
            "## Gates and interpretation",
            "",
            f"- The lower bound of the paired {CONFIDENCE_LEVEL:.0%} bootstrap "
            f"interval must be at least `{MINIMUM_CONFIDENT_SPEEDUP:.2f}x`.",
            "- Median end-to-end saving must exceed "
            f"`{MINIMUM_RELATIVE_SAVING:.0%}` of CPU time or "
            f"`{MINIMUM_ABSOLUTE_SAVING_SECONDS:.3f} s`, whichever is larger.",
            "- GPU end-to-end includes H2D, synchronized compute, and D2H.",
            "- Resident timing models a pipeline that already holds the image on GPU.",
            "- Disk I/O and synthetic image generation are excluded.",
            "- Re-run VIPP's optimizer on the actual workload before persisting a",
            "  backend choice; this report must not be copied across machines.",
            "",
            "## Exact coverage",
            "",
        )
    )
    for tag in admission["coverage"]:
        lines.append(f"- `{tag}`")
    lines.extend(("", "## Matched rejection coverage", ""))
    for tag in rejections["coverage"]:
        lines.append(f"- `{tag}`")
    lines.append("")
    return "\n".join(lines)


def _seconds(value: object) -> str:
    return f"{float(value):.6f} s"


def _ratio(value: object) -> str:
    return f"{float(value):.2f}x"


def _validate_document_contract(document: Mapping[str, object]) -> None:
    _require_exact_keys(document, _ROOT_KEYS, "evidence root")
    if document.get("schema") != SCHEMA:
        raise EvidenceError("Unexpected evidence schema.")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("Unexpected evidence schema version.")
    if document.get("kind") != (
        "scientific-admission-and-machine-local-performance-evidence"
    ):
        raise EvidenceError("Unexpected evidence kind.")
    if document.get("portable_performance_claim") is not False:
        raise EvidenceError("Evidence must reject portable performance claims.")
    if document.get("durable_optimizer_record") is not False:
        raise EvidenceError("Evidence must not claim to be an optimizer record.")
    try:
        datetime.fromisoformat(str(document["created_utc"]))
    except (KeyError, ValueError) as exc:
        raise EvidenceError("Evidence created_utc is invalid.") from exc
    profile = str(document.get("profile", ""))
    if profile not in {"quick", "full"}:
        raise EvidenceError("Evidence profile must be quick or full.")
    if dict(_mapping(document["method"], "method")) != _method_record(profile):
        raise EvidenceError("Evidence method does not match the fixed contract.")
    _mapping(document["platform"], "platform")
    _mapping(document["packages"], "packages")
    _validate_source_provenance(document["source_provenance"])
    _validate_admission(_mapping(document["admission"], "admission"))
    _validate_rejections(_mapping(document["rejections"], "rejections"))
    _validate_lifecycle(_mapping(document["lifecycle"], "lifecycle"))
    _validate_performance(
        _mapping(document["performance"], "performance"),
        profile=profile,
    )


def _validate_source_provenance(value: object) -> None:
    if not isinstance(value, list) or len(value) != len(SOURCE_PROVENANCE_PATHS):
        raise EvidenceError("Source provenance is incomplete.")
    expected = _source_provenance()
    if value != expected:
        raise EvidenceError("Evidence source fingerprints are stale.")


def _validate_admission(admission: Mapping[str, object]) -> None:
    if admission.get("status") != "pass":
        raise EvidenceError("Admission did not pass.")
    coverage = admission.get("coverage")
    if not isinstance(coverage, list) or not REQUIRED_ADMISSION_COVERAGE <= set(
        coverage
    ):
        raise EvidenceError("Admission coverage is incomplete.")
    cases = admission.get("cases")
    if not isinstance(cases, list) or admission.get("case_count") != len(cases):
        raise EvidenceError("Admission cases are incomplete.")
    for raw_case in cases:
        case = _mapping(raw_case, "admission case")
        if (
            case.get("exact") is not True
            or case.get("mismatch_count") != 0
            or case.get("gpu_output_resident") is not True
            or case.get("gpu_output_contiguous") is not True
            or case.get("input_immutable") is not True
            or case.get("cpu_output_sha256") != case.get("gpu_output_sha256")
        ):
            raise EvidenceError("An admission case is not exact and resident.")
        _hex_digest(case.get("cpu_output_sha256"), "admission output SHA-256")


def _validate_rejections(rejections: Mapping[str, object]) -> None:
    if rejections.get("status") != "pass":
        raise EvidenceError("Rejection audit did not pass.")
    coverage = rejections.get("coverage")
    if not isinstance(coverage, list) or not REQUIRED_REJECTION_COVERAGE <= set(
        coverage
    ):
        raise EvidenceError("Rejection coverage is incomplete.")
    cases = rejections.get("cases")
    if not isinstance(cases, list) or rejections.get("case_count") != len(cases):
        raise EvidenceError("Rejection cases are incomplete.")
    if any(
        _mapping(case, "rejection case").get("cpu_gpu_message_exact") is not True
        for case in cases
    ):
        raise EvidenceError("A CPU/GPU rejection message differed.")


def _validate_lifecycle(lifecycle: Mapping[str, object]) -> None:
    if lifecycle.get("status") != "pass" or lifecycle.get("cancelled") is not True:
        raise EvidenceError("Lifecycle cancellation did not pass.")
    if (
        lifecycle.get("device_pool_used_bytes_after_cleanup") != 0
        or lifecycle.get("device_pool_reserved_bytes_after_cleanup") != 0
    ):
        raise EvidenceError("Lifecycle cleanup did not drain the device pool.")
    updates = lifecycle.get("reported_progress")
    if not isinstance(updates, list) or len(updates) != 2:
        raise EvidenceError("Lifecycle progress is incomplete.")
    if [int(_mapping(item, "progress")["current"]) for item in updates] != [0, 1]:
        raise EvidenceError("Lifecycle cancellation was not at a row-tile boundary.")


def _validate_performance(
    performance: Mapping[str, object],
    *,
    profile: str,
) -> None:
    if performance.get("status") != "pass":
        raise EvidenceError("Performance timing did not complete.")
    results = performance.get("results")
    definitions = _performance_cases(profile)
    if not isinstance(results, list) or len(results) != len(definitions):
        raise EvidenceError("Performance cases are incomplete.")
    if performance.get("case_count") != len(results):
        raise EvidenceError("Performance case_count is inconsistent.")
    by_id = {definition.case_id: definition for definition in definitions}
    if [str(_mapping(item, "result").get("case_id")) for item in results] != [
        definition.case_id for definition in definitions
    ]:
        raise EvidenceError("Performance case identities are incomplete.")
    for result_index, raw_result in enumerate(results, start=1):
        result = _mapping(raw_result, "performance result")
        _require_exact_keys(result, _PERFORMANCE_RESULT_KEYS, "performance result")
        definition = by_id[str(result["case_id"])]
        if (
            result.get("shape") != list(definition.shape)
            or float(result.get("radius", -1)) != definition.radius
            or result.get("dtype") != "uint16"
            or result.get("element_count") != math.prod(definition.shape)
            or result.get("input_bytes") != math.prod(definition.shape) * 2
        ):
            raise EvidenceError("Performance workload metadata is inconsistent.")
        expected_parameters = {
            "radius": definition.radius,
            "sigma_width": 2.0,
            "minimum_pixel_fraction": 0.2,
            "outlier_aware": True,
            "channel_axis": None,
        }
        if result.get("parameters") != expected_parameters:
            raise EvidenceError("Performance parameters left the fixed contract.")
        expected_source = {
            "generator": GENERATOR_ID,
            "seed": definition.seed,
        }
        if (
            result.get("source_kind") != definition.source_kind
            or result.get("source_metadata") != expected_source
        ):
            raise EvidenceError("Performance source metadata is inconsistent.")
        _hex_digest(result.get("input_sha256"), "performance input SHA-256")
        parity = _mapping(result.get("parity"), "performance parity")
        if (
            parity.get("passed") is not True
            or parity.get("mismatch_count") != 0
            or parity.get("gpu_output_resident") is not True
            or parity.get("cpu_output_sha256") != parity.get("gpu_output_sha256")
        ):
            raise EvidenceError("Performance parity is not exact and resident.")
        samples = _mapping(result.get("samples"), "timing samples")
        for name in (
            "cpu_seconds",
            "gpu_end_to_end_seconds",
            "gpu_resident_seconds",
            "gpu_transfer_seconds",
        ):
            _validate_timing_samples(samples.get(name), name)
        cpu_cold = _positive_finite(
            samples.get("cpu_case_cold_seconds"),
            "CPU case-cold time",
        )
        gpu_cold = _positive_finite(
            samples.get("gpu_case_cold_end_to_end_seconds"),
            "GPU case-cold time",
        )
        bootstrap_seed = int(result.get("bootstrap_seed", -1))
        if bootstrap_seed != BOOTSTRAP_SEED + result_index:
            raise EvidenceError("Performance bootstrap seed is inconsistent.")
        expected = _timing_summary(
            [float(value) for value in samples["cpu_seconds"]],
            [float(value) for value in samples["gpu_end_to_end_seconds"]],
            [float(value) for value in samples["gpu_resident_seconds"]],
            [float(value) for value in samples["gpu_transfer_seconds"]],
            cpu_cold_seconds=cpu_cold,
            gpu_cold_seconds=gpu_cold,
            bootstrap_seed=bootstrap_seed,
        )
        summary = _mapping(result.get("summary"), "timing summary")
        if set(summary) != set(expected):
            raise EvidenceError("Timing summary fields are incomplete.")
        for name, expected_value in expected.items():
            actual = summary.get(name)
            if isinstance(expected_value, float):
                if not math.isclose(
                    float(actual), expected_value, rel_tol=1e-12, abs_tol=1e-12
                ):
                    raise EvidenceError(f"Timing summary {name!r} is inconsistent.")
            elif actual != expected_value:
                raise EvidenceError(f"Timing summary {name!r} is inconsistent.")
        cleanup = _mapping(result.get("cleanup"), "performance cleanup")
        if (
            cleanup.get("used_bytes_after_cleanup") != 0
            or cleanup.get("reserved_bytes_after_cleanup") != 0
        ):
            raise EvidenceError("Performance cleanup did not drain the pool.")
    expected_crossover = _crossover_summary(results)
    if performance.get("crossover") != expected_crossover:
        raise EvidenceError("Crossover summary is inconsistent.")


def _validate_timing_samples(value: object, name: str) -> None:
    if not isinstance(value, list) or len(value) != BENCHMARK_ROUNDS:
        raise EvidenceError(f"{name} samples are incomplete.")
    for sample in value:
        _positive_finite(sample, name)


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{name} must be an object.")
    return value


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise EvidenceError(f"{name} must be a positive finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise EvidenceError(f"{name} must be a positive finite number.") from exc
    if not math.isfinite(result) or result <= 0.0:
        raise EvidenceError(f"{name} must be a positive finite number.")
    return result


def _hex_digest(value: object, name: str) -> str:
    text = str(value)
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise EvidenceError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise EvidenceError(
            f"{name} fields differ: missing={sorted(expected - set(value))}, "
            f"extra={sorted(set(value) - expected)}."
        )


def validate_existing(path: Path) -> Path:
    artifact = path.expanduser().resolve()
    with artifact.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise EvidenceError("Evidence root must be an object.")
    _validate_document_contract(document)
    markdown = artifact.with_suffix(".md")
    if markdown.is_file():
        expected = render_markdown(document)
        if markdown.read_text(encoding="utf-8") != expected:
            raise EvidenceError("Markdown summary is stale or was edited manually.")
    return artifact


def _atomic_write_artifacts(
    output: Path,
    markdown: Path,
    document: Mapping[str, object],
) -> None:
    _validate_document_contract(document)
    _atomic_write_text(
        output,
        json.dumps(document, indent=2, sort_keys=True) + "\n",
    )
    _atomic_write_text(markdown, render_markdown(document))


def _atomic_write_text(path: Path, text: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
