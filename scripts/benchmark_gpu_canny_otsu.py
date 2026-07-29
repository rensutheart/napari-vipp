"""Validate and screen VIPP's exact CuPy Canny and Otsu providers.

The command has two deliberately separate jobs:

* run a deterministic scientific-admission matrix against VIPP's production
  CPU operations, requiring bit-for-bit identical boolean masks; and
* collect short, machine-local timings on a large structured stack.  CPU and
  transfer-inclusive GPU calls are measured independently, while a second GPU
  timing keeps the input resident to represent an already-GPU pipeline.

An optional private ND2 source may add one lazily selected T/C, ZYX volume.
Its path, filename, content digest, and pixels are never serialized.  Import,
``--help``, and ``--validate-existing`` do not import CuPy or initialize CUDA.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import re
import statistics
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

SCHEMA = "napari-vipp-cupy-canny-otsu-evidence"
SCHEMA_VERSION = 3
GENERATOR_ID = "numpy-pcg64-structured-confocal-like-stack-v1"
PROFILE = "practical"
SYNTHETIC_SHAPE = (8, 1024, 1024)
WARMUP_ROUNDS = 1
BENCHMARK_ROUNDS = 3
ROUND_ORDER = ("gpu", "cpu", "cpu", "gpu", "gpu", "cpu")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/benchmarks/canny-otsu-cupy-windows-rtx5090.json"
SOURCE_PROVENANCE_PATHS = (
    "scripts/benchmark_gpu_canny_otsu.py",
    "src/napari_vipp/core/compute.py",
    "src/napari_vipp/core/compute_planning.py",
    "src/napari_vipp/core/compute_registry.py",
    "src/napari_vipp/core/operations.py",
    "src/napari_vipp/core/gpu/cupy_canny.py",
    "src/napari_vipp/core/gpu/cupy_otsu.py",
    "src/napari_vipp/core/compute_specs.py",
    "src/napari_vipp/core/compute_policy.py",
    "src/napari_vipp/core/execution.py",
    "src/napari_vipp/core/progress.py",
)
CANNY_IMPLEMENTATION_ID = "cupyx-canny-edges-exact-v1"
OTSU_IMPLEMENTATION_ID = "cupy-otsu-threshold-exact-v1"
PRIVATE_SOURCE_ID = "private-real-nd2-single-channel-zyx"
PRIVATE_SOURCE_LABEL = "Private real-acquisition single-channel ZYX volume"
PRIVATE_SOURCE_KIND = "private-nd2-volume"
PUBLIC_V3_SCIENTIFIC_STACK = {
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit-image": "0.26.0",
}
PUBLIC_V3_CUPY_VERSION = "14.1.1"
PUBLIC_V3_CUDA_DEVICE_NAME = "NVIDIA GeForce RTX 5090"
PUBLIC_V3_CUDA_COMPUTE_CAPABILITY = "12.0"
PUBLIC_V3_CUDA_DRIVER_VERSION = 13_030
PUBLIC_V3_CUDA_RUNTIME_VERSION = 13_020
_PACKAGE_KEYS = frozenset(
    {
        "napari-vipp",
        *PUBLIC_V3_SCIENTIFIC_STACK,
        "cupy-cuda12x",
        "cupy-cuda13x",
    }
)
_HOST_PLATFORM_KEYS = frozenset(
    {
        "system",
        "release",
        "machine",
        "processor",
        "python",
        "python_implementation",
        "python_abi",
        "execution_mode",
    }
)
_PLATFORM_KEYS = frozenset(
    {
        *_HOST_PLATFORM_KEYS,
        "cuda_device_index",
        "cuda_device_name",
        "cuda_compute_capability",
        "cuda_driver_version",
        "cuda_runtime_version",
        "total_accelerator_memory_bytes",
    }
)
_PERFORMANCE_SOURCE_KEYS = frozenset(
    {
        "source_id",
        "label",
        "source_kind",
        "source_metadata",
        "direct_private_identifiers_published",
        "shape",
        "dtype",
        "element_count",
        "input_bytes",
        "input_sha256",
        "operations",
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
        "performance",
        "lifecycle",
    }
)
_METHOD_KEYS = frozenset(
    {
        "cpu_path",
        "gpu_path",
        "parity",
        "warmup_rounds",
        "timed_rounds",
        "cpu_timing_scope",
        "gpu_end_to_end_timing_scope",
        "gpu_resident_timing_scope",
        "disk_io_included",
        "input_generation_included",
        "exact_parity_required_before_timing",
    }
)
_SOURCE_PROVENANCE_KEYS = frozenset({"path", "sha256"})
_ADMISSION_KEYS = frozenset(
    {"status", "case_count", "failure_count", "coverage", "cases"}
)
_ADMISSION_CASE_KEYS = frozenset(
    {
        "case_id",
        "operation_id",
        "input_shape",
        "input_dtype",
        "input_sha256",
        "parameters",
        "coverage",
        "output_shape",
        "output_dtype",
        "gpu_output_dtype",
        "gpu_output_resident",
        "cpu_foreground_pixels",
        "gpu_foreground_pixels",
        "mismatch_count",
        "exact_mask_match",
    }
)
_PERFORMANCE_KEYS = frozenset({"status", "source_count", "sources"})
_SYNTHETIC_SOURCE_METADATA_KEYS = frozenset({"generator"})
_PERFORMANCE_OPERATION_KEYS = frozenset(
    {
        "operation_id",
        "implementation_id",
        "parameters",
        "parity",
        "samples",
        "memory",
        "summary",
    }
)
_PARITY_KEYS = frozenset(
    {
        "profile",
        "passed",
        "mismatch_count",
        "foreground_pixels",
        "cpu_output_dtype",
        "gpu_output_dtype",
        "gpu_output_resident",
    }
)
_TIMING_SAMPLE_KEYS = frozenset(
    {"cpu_seconds", "gpu_end_to_end_seconds", "gpu_resident_seconds"}
)
_MEMORY_KEYS = frozenset(
    {
        "model_id",
        "runtime_managed_peak_bytes",
        "uncertainty_bytes",
        "admitted_device_peak_bytes",
        "observed_private_pool_peak_bytes",
        "observed_within_admitted_peak",
        "cleanup",
    }
)
_CLEANUP_KEYS = frozenset(
    {
        "passed",
        "used_bytes_after_cleanup",
        "reserved_bytes_after_cleanup",
        "error",
    }
)
_TIMING_SUMMARY_KEYS = frozenset(
    {
        "cpu_median_seconds",
        "gpu_end_to_end_median_seconds",
        "gpu_resident_median_seconds",
        "gpu_end_to_end_speedup",
        "gpu_resident_speedup",
        "screening_choice",
    }
)
_LIFECYCLE_KEYS = frozenset({"status", "operations"})
_LIFECYCLE_OPERATION_KEYS = frozenset(
    {
        "operation_id",
        "implementation_id",
        "parameters",
        "reported_progress",
        "cancellation_requested",
        "cancellation_observed",
        "cleanup",
    }
)
_PROGRESS_UPDATE_KEYS = frozenset({"current", "total", "message"})
_PRIVATE_SOURCE_METADATA_KEYS = frozenset(
    {
        "original_axes",
        "original_shape",
        "original_dtype",
        "selected_indices",
        "direct_identifiers_omitted",
    }
)
_PRIVATE_ND2_AXIS_ORDER = "TPZCYX"
_PRIVATE_ND2_AXES = frozenset(_PRIVATE_ND2_AXIS_ORDER)
_MAX_PRIVATE_AXIS_EXTENT = 2**31 - 1
_MAX_PRIVATE_ELEMENT_COUNT = 2**63 - 1
_PERFORMANCE_OPERATION_CONTRACTS = {
    "canny_edges": {
        "implementation_id": CANNY_IMPLEMENTATION_ID,
        "memory_model_id": "cupyx-canny-exact-memory-v1",
        "parameters": {"sigma": 1.5, "low_quantile": 0.1, "high_quantile": 0.2},
    },
    "otsu_threshold": {
        "implementation_id": OTSU_IMPLEMENTATION_ID,
        "memory_model_id": "cupy-otsu-histogram-memory-v1",
        "parameters": {
            "threshold_scope": "Stack histogram",
            "histogram_bins": 256,
        },
    },
}
_LIFECYCLE_OPERATION_CONTRACTS = {
    "canny_edges": {
        "implementation_id": CANNY_IMPLEMENTATION_ID,
        "parameters": {"sigma": 1.5, "low_quantile": 0.1, "high_quantile": 0.2},
        "progress": (
            {"current": 0, "total": 2, "message": "Canny planes"},
            {"current": 1, "total": 2, "message": "Canny planes"},
        ),
    },
    "otsu_threshold": {
        "implementation_id": OTSU_IMPLEMENTATION_ID,
        "parameters": {
            "threshold_scope": "Slice histogram",
            "histogram_bins": 256,
        },
        "progress": (
            {"current": 0, "total": 2, "message": "Otsu slice histograms"},
            {"current": 1, "total": 2, "message": "Otsu histogram ready"},
        ),
    },
}
REQUIRED_COVERAGE = {
    "canny_edges": frozenset(
        {
            "dtype:bool",
            "dtype:uint8",
            "dtype:uint16",
            "sigma:zero",
            "sigma:negative-clamped",
            "sigma:positive",
            "sigma:upper-bound",
            "quantile:endpoints",
            "quantile:equal",
            "quantile:ordered",
            "layout:leading-blocks",
            "layout:rgb",
            "layout:rgba",
            "topology:narrow",
            "topology:flat",
            "topology:border",
        }
    ),
    "otsu_threshold": frozenset(
        {
            "dtype:bool",
            "dtype:int8",
            "dtype:uint8",
            "dtype:int16",
            "dtype:uint16",
            "dtype:int32",
            "dtype:uint32",
            "dtype:int64",
            "dtype:uint64",
            "dtype:float16",
            "dtype:float32",
            "dtype:float64",
            "scope:stack",
            "scope:slice",
            "bins:2",
            "bins:256",
            "bins:65536",
            "layout:leading-blocks",
            "layout:rgb",
            "layout:rgba",
            "values:constant",
            "values:nonfinite",
            "values:native-integer-levels",
            "integer-span:65536",
            "range:float32-extreme",
            "range:int64-rgb-luma",
        }
    ),
}


class EvidenceError(RuntimeError):
    """The evidence run or its persisted contract is incomplete."""


@dataclass(frozen=True, slots=True)
class _AdmissionCase:
    case_id: str
    operation_id: str
    data: Any
    parameters: Mapping[str, object]
    coverage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PerformanceSource:
    source_id: str
    label: str
    data: Any
    source_kind: str
    metadata: Mapping[str, object]
    private: bool = False


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
        "--nd2",
        type=Path,
        help=(
            "Optional private ND2 source. One T/C ZYX volume is timed without "
            "publishing its path, filename, digest, or pixels."
        ),
    )
    parser.add_argument(
        "--nd2-time-index",
        type=int,
        default=0,
        help="T index for --nd2 (default: 0).",
    )
    parser.add_argument(
        "--nd2-channel-index",
        type=int,
        default=0,
        help="C index for --nd2 (default: 0).",
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
            print(f"Canny/Otsu evidence validation failed: {exc}", file=sys.stderr)
            return 2
        print(f"Canny/Otsu evidence is current: {artifact}")
        return 0

    markdown = args.markdown or args.output.with_suffix(".md")
    try:
        document = run_evidence(
            device_index=args.device_index,
            nd2_path=args.nd2,
            nd2_time_index=args.nd2_time_index,
            nd2_channel_index=args.nd2_channel_index,
        )
        _atomic_write_artifacts(args.output, markdown, document)
    except (OSError, TypeError, ValueError, EvidenceError) as exc:
        print(f"Canny/Otsu evidence run failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # optional runtime failures need a concise CLI edge
        print(
            f"Canny/Otsu evidence run failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"Wrote Canny/Otsu evidence to {args.output.resolve()}")
    print(f"Wrote readable summary to {markdown.resolve()}")
    return 0


def run_evidence(
    *,
    device_index: int = 0,
    nd2_path: Path | None = None,
    nd2_time_index: int = 0,
    nd2_channel_index: int = 0,
) -> dict[str, object]:
    """Run exact admission and large-stack timing on one CUDA device."""

    if isinstance(device_index, bool) or device_index < 0:
        raise ValueError("CUDA device index must be a non-negative integer.")

    packages = _package_versions()
    _validate_package_contract(packages)
    _validate_public_v3_host_contract(_host_platform_record())

    import cupy

    from napari_vipp.core.gpu.cupy_canny import canny_edges as gpu_canny
    from napari_vipp.core.gpu.cupy_otsu import otsu_threshold as gpu_otsu
    from napari_vipp.core.operations import canny_edges as cpu_canny
    from napari_vipp.core.operations import otsu_threshold as cpu_otsu

    cpu_functions = {
        "canny_edges": cpu_canny,
        "otsu_threshold": cpu_otsu,
    }
    gpu_functions = {
        "canny_edges": gpu_canny,
        "otsu_threshold": gpu_otsu,
    }
    provenance = _source_provenance()
    with cupy.cuda.Device(device_index):
        platform_record = _platform_record(cupy, device_index)
        _validate_public_v3_environment_contract(platform_record, packages)
        admission = _run_admission_matrix(
            cupy=cupy,
            cpu_functions=cpu_functions,
            gpu_functions=gpu_functions,
        )
        sources = [_synthetic_performance_source()]
        if nd2_path is not None:
            sources.append(
                _load_private_nd2_volume(
                    nd2_path,
                    time_index=nd2_time_index,
                    channel_index=nd2_channel_index,
                )
            )
        performance = _run_performance(
            sources,
            cupy=cupy,
            cpu_functions=cpu_functions,
            gpu_functions=gpu_functions,
        )
        lifecycle = _run_lifecycle_audit(
            cupy=cupy,
            gpu_functions=gpu_functions,
        )
    _require_source_snapshot_unchanged(provenance)
    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "scientific-admission-and-machine-local-performance-evidence",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": PROFILE,
        "method": {
            "cpu_path": "production-napari-vipp-operations",
            "gpu_path": "production-exact-cupy-providers",
            "parity": "bitwise-identical-boolean-mask",
            "warmup_rounds": WARMUP_ROUNDS,
            "timed_rounds": BENCHMARK_ROUNDS,
            "cpu_timing_scope": "synchronous-operation-call-v1",
            "gpu_end_to_end_timing_scope": (
                "host-to-device-plus-synchronized-compute-plus-device-to-host-v1"
            ),
            "gpu_resident_timing_scope": "synchronized-resident-compute-v1",
            "disk_io_included": False,
            "input_generation_included": False,
            "exact_parity_required_before_timing": True,
        },
        "platform": platform_record,
        "packages": packages,
        "source_provenance": provenance,
        "admission": admission,
        "performance": performance,
        "lifecycle": lifecycle,
    }
    _validate_document_contract(document)
    return document


def _run_admission_matrix(
    *,
    cupy,
    cpu_functions: Mapping[str, Any],
    gpu_functions: Mapping[str, Any],
) -> dict[str, object]:
    import numpy as np

    records: list[dict[str, object]] = []
    covered = {operation: set() for operation in REQUIRED_COVERAGE}
    for case in _admission_cases():
        cpu_output = np.asarray(
            cpu_functions[case.operation_id](case.data, **case.parameters)
        )
        if cpu_output.dtype != np.dtype(bool):
            raise EvidenceError(
                f"Authoritative CPU {case.operation_id} returned "
                f"{cpu_output.dtype.name}, not bool."
            )
        gpu_output_device = gpu_functions[case.operation_id](
            cupy.asarray(case.data),
            **case.parameters,
        )
        if not isinstance(gpu_output_device, cupy.ndarray):
            raise EvidenceError(
                f"GPU {case.operation_id} did not preserve device residency."
            )
        if gpu_output_device.dtype != cupy.dtype(bool):
            raise EvidenceError(
                f"GPU {case.operation_id} returned "
                f"{gpu_output_device.dtype.name}, not bool."
            )
        cupy.cuda.get_current_stream().synchronize()
        gpu_output = np.asarray(cupy.asnumpy(gpu_output_device))
        mismatch_count = int(np.count_nonzero(cpu_output != gpu_output))
        exact = (
            cpu_output.shape == gpu_output.shape
            and cpu_output.dtype == gpu_output.dtype
            and mismatch_count == 0
        )
        covered[case.operation_id].update(case.coverage)
        records.append(
            {
                "case_id": case.case_id,
                "operation_id": case.operation_id,
                "input_shape": list(case.data.shape),
                "input_dtype": np.dtype(case.data.dtype).name,
                "input_sha256": _array_sha256(case.data),
                "parameters": dict(case.parameters),
                "coverage": list(case.coverage),
                "output_shape": list(cpu_output.shape),
                "output_dtype": cpu_output.dtype.name,
                "gpu_output_dtype": gpu_output.dtype.name,
                "gpu_output_resident": True,
                "cpu_foreground_pixels": int(np.count_nonzero(cpu_output)),
                "gpu_foreground_pixels": int(np.count_nonzero(gpu_output)),
                "mismatch_count": mismatch_count,
                "exact_mask_match": exact,
            }
        )
        del gpu_output_device

    missing = {
        operation: sorted(required - covered[operation])
        for operation, required in REQUIRED_COVERAGE.items()
        if required - covered[operation]
    }
    failures = [
        record["case_id"] for record in records if not record["exact_mask_match"]
    ]
    if missing:
        raise EvidenceError(f"Admission matrix is missing required coverage: {missing}")
    if failures:
        raise EvidenceError(f"Exact Canny/Otsu parity failed for: {failures}")
    return {
        "status": "pass",
        "case_count": len(records),
        "failure_count": 0,
        "coverage": {
            operation: sorted(values) for operation, values in covered.items()
        },
        "cases": records,
    }


def _admission_cases() -> tuple[_AdmissionCase, ...]:
    import numpy as np

    rng = np.random.default_rng(20_260_729)
    unit = rng.random((37, 43), dtype=np.float32)
    ramp = np.arange(37, dtype=np.float32)[:, None] * np.float32(0.61) + np.arange(
        43, dtype=np.float32
    )[None, :] * np.float32(0.39)
    ramp /= ramp.max()
    structured = np.clip(unit * np.float32(0.35) + ramp * np.float32(0.65), 0, 1)
    bool_image = structured > np.float32(0.53)
    u8 = np.rint(structured * np.float32(255)).astype(np.uint8)
    u16 = np.rint(structured * np.float32(65535)).astype(np.uint16)

    canny_cases = [
        _AdmissionCase(
            "canny-bool-default",
            "canny_edges",
            bool_image,
            {},
            ("dtype:bool", "sigma:positive", "quantile:ordered"),
        ),
        _AdmissionCase(
            "canny-u8-zero-endpoints",
            "canny_edges",
            u8,
            {"sigma": 0.0, "low_quantile": 0.0, "high_quantile": 1.0},
            (
                "dtype:uint8",
                "sigma:zero",
                "quantile:endpoints",
            ),
        ),
        _AdmissionCase(
            "canny-u8-negative-sigma-clamped",
            "canny_edges",
            np.roll(u8, 4, 1),
            {"sigma": -3.0, "low_quantile": 0.2, "high_quantile": 0.8},
            ("dtype:uint8", "sigma:negative-clamped", "quantile:ordered"),
        ),
        _AdmissionCase(
            "canny-u16-upper-sigma-equal-quantile",
            "canny_edges",
            u16,
            {"sigma": 12.0, "low_quantile": 0.35, "high_quantile": 0.35},
            (
                "dtype:uint16",
                "sigma:upper-bound",
                "quantile:equal",
            ),
        ),
        _AdmissionCase(
            "canny-u16-leading-blocks",
            "canny_edges",
            np.stack((u16, u16[::-1], np.roll(u16, 7, 1))),
            {"sigma": 1.25, "low_quantile": 0.15, "high_quantile": 0.75},
            (
                "dtype:uint16",
                "sigma:positive",
                "quantile:ordered",
                "layout:leading-blocks",
            ),
        ),
        _AdmissionCase(
            "canny-rgb-last-axis",
            "canny_edges",
            np.stack((u8, np.roll(u8, 3, 0), np.roll(u8, 5, 1)), axis=-1),
            {"channel_axis": -1, "sigma": 0.75},
            ("layout:rgb",),
        ),
        _AdmissionCase(
            "canny-rgba-first-axis",
            "canny_edges",
            np.stack((u16, np.roll(u16, 2, 0), np.roll(u16, 4, 1), u16), axis=0),
            {"channel_axis": 0, "sigma": 2.0},
            ("layout:rgba",),
        ),
        _AdmissionCase(
            "canny-narrow-border",
            "canny_edges",
            np.pad(np.full((1, 15), 255, dtype=np.uint8), ((1, 1), (1, 1))),
            {"sigma": 0.0, "low_quantile": 0.0, "high_quantile": 0.0},
            ("topology:narrow", "topology:border"),
        ),
        _AdmissionCase(
            "canny-flat-tie",
            "canny_edges",
            np.full((19, 23), 32768, dtype=np.uint16),
            {"sigma": 4.0, "low_quantile": 0.5, "high_quantile": 0.5},
            ("topology:flat", "quantile:equal"),
        ),
    ]

    integer_types = (
        np.int8,
        np.uint8,
        np.int16,
        np.uint16,
        np.int32,
        np.uint32,
        np.int64,
        np.uint64,
    )
    otsu_cases: list[_AdmissionCase] = [
        _AdmissionCase(
            "otsu-bool-identity",
            "otsu_threshold",
            bool_image,
            {},
            ("dtype:bool", "scope:stack"),
        )
    ]
    for index, dtype in enumerate(integer_types):
        info = np.iinfo(dtype)
        if np.issubdtype(dtype, np.signedinteger):
            minimum = max(info.min, -20_000 - index)
            maximum = min(info.max, minimum + 40_000)
        else:
            minimum = min(info.max, index * 17)
            maximum = min(info.max, minimum + 40_000)
        levels = np.linspace(minimum, maximum, num=structured.size, dtype=np.float64)
        integer_image = np.rint(levels).reshape(structured.shape).astype(dtype)
        otsu_cases.append(
            _AdmissionCase(
                f"otsu-{np.dtype(dtype).name}-native-levels",
                "otsu_threshold",
                integer_image,
                {"histogram_bins": 3},  # intentionally ignored for scalar integers
                (
                    f"dtype:{np.dtype(dtype).name}",
                    "values:native-integer-levels",
                    "scope:stack",
                ),
            )
        )

    f16 = structured.astype(np.float16)
    f32_nonfinite = structured.copy()
    f32_nonfinite[0, 0] = np.nan
    f32_nonfinite[1, 1] = np.inf
    f32_nonfinite[2, 2] = -np.inf
    f64 = structured.astype(np.float64)
    otsu_cases.extend(
        (
            _AdmissionCase(
                "otsu-f16-two-bins",
                "otsu_threshold",
                f16,
                {"histogram_bins": 2},
                ("dtype:float16", "bins:2", "scope:stack"),
            ),
            _AdmissionCase(
                "otsu-f32-nonfinite-256-bins",
                "otsu_threshold",
                f32_nonfinite,
                {"histogram_bins": 256},
                (
                    "dtype:float32",
                    "bins:256",
                    "values:nonfinite",
                ),
            ),
            _AdmissionCase(
                "otsu-f64-65536-bins",
                "otsu_threshold",
                f64,
                {"histogram_bins": 65_536},
                ("dtype:float64", "bins:65536"),
            ),
            _AdmissionCase(
                "otsu-constant",
                "otsu_threshold",
                np.full((17, 21), 1234, dtype=np.uint16),
                {},
                ("values:constant",),
            ),
            _AdmissionCase(
                "otsu-u16-complete-native-span",
                "otsu_threshold",
                np.arange(65_536, dtype=np.uint16).reshape(256, 256),
                {},
                ("integer-span:65536", "values:native-integer-levels"),
            ),
            _AdmissionCase(
                "otsu-slice-leading-blocks",
                "otsu_threshold",
                np.stack((u16, np.roll(u16, 9, 1), u16 // 2)),
                {"threshold_scope": "Slice histogram"},
                ("scope:slice", "layout:leading-blocks"),
            ),
            _AdmissionCase(
                "otsu-rgb-last-axis",
                "otsu_threshold",
                np.stack((u8, np.roll(u8, 2, 0), np.roll(u8, 5, 1)), axis=-1),
                {"channel_axis": -1, "histogram_bins": 256},
                ("layout:rgb",),
            ),
            _AdmissionCase(
                "otsu-rgba-first-axis-slices",
                "otsu_threshold",
                np.stack((u16, np.roll(u16, 3, 0), u16 // 2, u16), axis=1),
                {
                    "channel_axis": 1,
                    "threshold_scope": "Slice histogram",
                    "histogram_bins": 256,
                },
                ("layout:rgba", "scope:slice"),
            ),
            _AdmissionCase(
                "otsu-f32-extreme-range",
                "otsu_threshold",
                np.asarray(
                    [
                        [-1.0, -1.0, np.finfo(np.float32).max / 2.0],
                        [np.finfo(np.float32).max / 2.0, 0.0, 1.0],
                        [
                            np.finfo(np.float32).max / 2.0,
                            1.0,
                            -np.finfo(np.float32).max / 2.0,
                        ],
                    ],
                    dtype=np.float32,
                ),
                {"histogram_bins": 3},
                ("dtype:float32", "range:float32-extreme"),
            ),
            _AdmissionCase(
                "otsu-int64-rgb-wide-range",
                "otsu_threshold",
                np.random.default_rng(0).integers(
                    -(2**63),
                    2**63 - 1,
                    size=(7, 9, 3),
                    dtype=np.int64,
                ),
                {"channel_axis": -1, "histogram_bins": 256},
                ("dtype:int64", "layout:rgb", "range:int64-rgb-luma"),
            ),
        )
    )
    return tuple(canny_cases + otsu_cases)


def _synthetic_performance_source() -> _PerformanceSource:
    import numpy as np
    from scipy import ndimage

    rng = np.random.default_rng(20_260_730)
    shape = SYNTHETIC_SHAPE
    noise = rng.standard_normal(shape, dtype=np.float32)
    smooth = ndimage.gaussian_filter(noise, sigma=(0.0, 5.0, 5.0), mode="reflect")
    y = np.linspace(0.0, 8.0 * np.pi, shape[-2], dtype=np.float32)
    x = np.linspace(0.0, 10.0 * np.pi, shape[-1], dtype=np.float32)
    texture = np.sin(y)[:, None] + np.cos(x)[None, :]
    z_gain = np.linspace(0.75, 1.25, shape[0], dtype=np.float32)[:, None, None]
    values = smooth * np.float32(0.45) + texture[None, :, :] * z_gain
    # Sparse diffraction-like puncta make the workload less like random noise
    # while keeping the generator compact and completely deterministic.
    yy, xx = np.ogrid[: shape[-2], : shape[-1]]
    for _ in range(48):
        plane = int(rng.integers(0, shape[0]))
        center_y = int(rng.integers(16, shape[-2] - 16))
        center_x = int(rng.integers(16, shape[-1] - 16))
        radius = int(rng.integers(3, 13))
        disk = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
        values[plane, disk] += np.float32(rng.uniform(2.0, 5.0))
    values -= values.min()
    maximum = float(values.max())
    if not math.isfinite(maximum) or maximum <= 0:
        raise EvidenceError("Synthetic performance generator became degenerate.")
    values *= np.float32(60_000.0 / maximum)
    data = np.ascontiguousarray(np.rint(values), dtype=np.uint16)
    shape_label = "x".join(str(size) for size in shape)
    return _PerformanceSource(
        source_id="synthetic-structured-large-stack",
        label=f"Structured synthetic {shape_label} uint16 stack",
        data=data,
        source_kind="deterministic-synthetic",
        metadata={"generator": GENERATOR_ID},
    )


def _load_private_nd2_volume(
    path: Path,
    *,
    time_index: int,
    channel_index: int,
) -> _PerformanceSource:
    import numpy as np

    try:
        import nd2
    except ImportError as exc:
        raise EvidenceError(
            "--nd2 requires the optional nd2 package in the benchmark environment."
        ) from exc

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ND2 source does not exist: {resolved}")
    with nd2.ND2File(str(resolved)) as nd_file:
        shape = tuple(int(size) for size in nd_file.shape)
        sizes = getattr(nd_file, "sizes", {})
        axes = "".join(str(name).upper() for name in sizes)
        original_dtype = np.dtype(nd_file.dtype).name
    if (
        len(axes) != len(shape)
        or len(set(axes)) != len(axes)
        or not set(axes) <= _PRIVATE_ND2_AXES
        or not all(axis in axes for axis in "ZYX")
    ):
        raise EvidenceError(
            "ND2 must expose unique privacy-safe T/P/Z/C/Y/X axes including "
            f"ordered ZYX; got axes={axes!r}, shape={shape}."
        )

    selection: list[int | slice] = []
    selected_spatial_axes: list[str] = []
    selected_indices: dict[str, int] = {}
    for axis, size in zip(axes, shape, strict=True):
        if axis == "T":
            index = _validated_index(time_index, size, "T")
            selection.append(index)
            selected_indices[axis] = index
        elif axis == "C":
            index = _validated_index(channel_index, size, "C")
            selection.append(index)
            selected_indices[axis] = index
        elif axis in "ZYX":
            selection.append(slice(None))
            selected_spatial_axes.append(axis)
        elif axis == "P" and size == 1:
            selection.append(0)
        else:
            raise EvidenceError(
                f"Unsupported non-spatial ND2 axis {axis!r} with size {size}; "
                "only a singleton P axis may be removed implicitly."
            )
    if "".join(selected_spatial_axes) != "ZYX":
        raise EvidenceError(
            "Selected ND2 spatial order must be ZYX, got "
            f"{''.join(selected_spatial_axes)!r}."
        )

    lazy = nd2.imread(str(resolved), dask=True)
    selected = lazy[tuple(selection)]
    if hasattr(selected, "compute"):
        selected = selected.compute()
    data = np.ascontiguousarray(np.asarray(selected))
    if data.ndim != 3:
        raise EvidenceError(f"Selected ND2 volume must be ZYX, got {data.shape}.")
    supported_dtypes = (
        np.dtype(np.uint8),
        np.dtype(np.uint16),
    )
    if data.dtype not in supported_dtypes:
        raise EvidenceError(
            "The private common Canny/Otsu timing input must already be uint8 "
            f"or uint16; got {data.dtype}. No benchmark-only cast was made."
        )
    return _PerformanceSource(
        source_id=PRIVATE_SOURCE_ID,
        label=PRIVATE_SOURCE_LABEL,
        data=data,
        source_kind=PRIVATE_SOURCE_KIND,
        metadata={
            "original_axes": axes,
            "original_shape": list(shape),
            "original_dtype": original_dtype,
            "selected_indices": selected_indices,
            "direct_identifiers_omitted": True,
        },
        private=True,
    )


def _validated_index(value: int, size: int, axis: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{axis} index must be an integer.")
    if value < 0 or value >= size:
        raise IndexError(f"{axis} index {value} is outside 0..{size - 1}.")
    return value


def _run_performance(
    sources: Sequence[_PerformanceSource],
    *,
    cupy,
    cpu_functions: Mapping[str, Any],
    gpu_functions: Mapping[str, Any],
) -> dict[str, object]:
    records = []
    for source in sources:
        _resource_preflight(source.data, cupy)
        operations = []
        for operation_id, parameters in (
            (
                "canny_edges",
                {"sigma": 1.5, "low_quantile": 0.1, "high_quantile": 0.2},
            ),
            (
                "otsu_threshold",
                {"threshold_scope": "Stack histogram", "histogram_bins": 256},
            ),
        ):
            print(f"Timing {operation_id} on {source.label}...", flush=True)
            operations.append(
                _benchmark_operation(
                    source.data,
                    operation_id=operation_id,
                    parameters=parameters,
                    cpu_function=cpu_functions[operation_id],
                    gpu_function=gpu_functions[operation_id],
                    cupy=cupy,
                )
            )
        record: dict[str, object] = {
            "source_id": source.source_id,
            "label": source.label,
            "source_kind": source.source_kind,
            "source_metadata": dict(source.metadata),
            "direct_private_identifiers_published": False,
            "shape": list(source.data.shape),
            "dtype": source.data.dtype.name,
            "element_count": int(source.data.size),
            "input_bytes": int(source.data.nbytes),
            "input_sha256": None if source.private else _array_sha256(source.data),
            "operations": operations,
        }
        records.append(record)
    return {
        "status": "pass",
        "source_count": len(records),
        "sources": records,
    }


def _benchmark_operation(
    data,
    *,
    operation_id: str,
    parameters: Mapping[str, object],
    cpu_function,
    gpu_function,
    cupy,
) -> dict[str, object]:
    import numpy as np

    memory = _audit_memory_and_cleanup(
        data,
        operation_id=operation_id,
        parameters=parameters,
        gpu_function=gpu_function,
        cupy=cupy,
    )

    cpu_reference = np.asarray(cpu_function(data, **parameters))
    if cpu_reference.dtype != np.dtype(bool):
        raise EvidenceError(
            f"Performance timing refused: CPU {operation_id} returned "
            f"{cpu_reference.dtype.name}, not bool."
        )
    resident_input = cupy.asarray(data)
    gpu_reference_device = gpu_function(resident_input, **parameters)
    if not isinstance(gpu_reference_device, cupy.ndarray):
        raise EvidenceError(
            f"Performance timing refused: GPU {operation_id} did not return a "
            "resident CuPy array."
        )
    if gpu_reference_device.dtype != cupy.dtype(bool):
        raise EvidenceError(
            f"Performance timing refused: GPU {operation_id} returned "
            f"{gpu_reference_device.dtype.name}, not bool."
        )
    cupy.cuda.get_current_stream().synchronize()
    gpu_reference = np.asarray(cupy.asnumpy(gpu_reference_device))
    mismatch_count = int(np.count_nonzero(cpu_reference != gpu_reference))
    if cpu_reference.shape != gpu_reference.shape or mismatch_count:
        raise EvidenceError(
            f"Performance timing refused: {operation_id} exact parity had "
            f"{mismatch_count} mismatches."
        )

    # Untimed warmups remove first-call imports, CUDA compilation, and allocator
    # setup from the short descriptive samples.
    for _ in range(WARMUP_ROUNDS):
        cpu_function(data, **parameters)
        warm_device = gpu_function(resident_input, **parameters)
        cupy.cuda.get_current_stream().synchronize()
        del warm_device
        end_to_end_input = cupy.asarray(data)
        end_to_end_output = gpu_function(end_to_end_input, **parameters)
        cupy.asnumpy(end_to_end_output)
        cupy.cuda.get_current_stream().synchronize()
        del end_to_end_input, end_to_end_output

    cpu_seconds: list[float] = []
    gpu_end_to_end_seconds: list[float] = []
    # The fixed interleaving prevents every CPU or GPU sample from always
    # occurring first while remaining fully reproducible.
    for candidate in ROUND_ORDER:
        if candidate == "cpu":
            started = time.perf_counter()
            cpu_function(data, **parameters)
            elapsed = time.perf_counter() - started
            cpu_seconds.append(elapsed)
        else:
            cupy.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            device_input = cupy.asarray(data)
            device_output = gpu_function(device_input, **parameters)
            cupy.asnumpy(device_output)
            cupy.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter() - started
            gpu_end_to_end_seconds.append(elapsed)
            del device_input, device_output

    gpu_resident_seconds: list[float] = []
    for _ in range(BENCHMARK_ROUNDS):
        cupy.cuda.get_current_stream().synchronize()
        started = time.perf_counter()
        resident_output = gpu_function(resident_input, **parameters)
        cupy.cuda.get_current_stream().synchronize()
        gpu_resident_seconds.append(time.perf_counter() - started)
        del resident_output

    del resident_input, gpu_reference_device
    cupy.get_default_memory_pool().free_all_blocks()
    cupy.get_default_pinned_memory_pool().free_all_blocks()
    cupy.cuda.get_current_stream().synchronize()

    cpu_median = statistics.median(cpu_seconds)
    gpu_e2e_median = statistics.median(gpu_end_to_end_seconds)
    gpu_resident_median = statistics.median(gpu_resident_seconds)
    implementation_id = (
        CANNY_IMPLEMENTATION_ID
        if operation_id == "canny_edges"
        else OTSU_IMPLEMENTATION_ID
    )
    return {
        "operation_id": operation_id,
        "implementation_id": implementation_id,
        "parameters": dict(parameters),
        "parity": {
            "profile": "bitwise-identical-boolean-mask",
            "passed": True,
            "mismatch_count": 0,
            "foreground_pixels": int(np.count_nonzero(cpu_reference)),
            "cpu_output_dtype": cpu_reference.dtype.name,
            "gpu_output_dtype": gpu_reference.dtype.name,
            "gpu_output_resident": True,
        },
        "samples": {
            "cpu_seconds": cpu_seconds,
            "gpu_end_to_end_seconds": gpu_end_to_end_seconds,
            "gpu_resident_seconds": gpu_resident_seconds,
        },
        "memory": memory,
        "summary": {
            "cpu_median_seconds": cpu_median,
            "gpu_end_to_end_median_seconds": gpu_e2e_median,
            "gpu_resident_median_seconds": gpu_resident_median,
            "gpu_end_to_end_speedup": cpu_median / gpu_e2e_median,
            "gpu_resident_speedup": cpu_median / gpu_resident_median,
            "screening_choice": "GPU-CuPy" if gpu_e2e_median < cpu_median else "CPU",
        },
    }


def _audit_memory_and_cleanup(
    data,
    *,
    operation_id: str,
    parameters: Mapping[str, object],
    gpu_function,
    cupy,
) -> dict[str, object]:
    """Measure one isolated allocator peak against the production policy model."""

    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import estimate_candidate_memory
    from napari_vipp.core.compute_specs import compute_specs_for

    implementation_id = (
        CANNY_IMPLEMENTATION_ID
        if operation_id == "canny_edges"
        else OTSU_IMPLEMENTATION_ID
    )
    spec = next(
        item
        for item in compute_specs_for(operation_id, include_cpu=False)
        if item.implementation_id == implementation_id
    )
    workload = WorkloadDescriptor(
        f"evidence-{operation_id}",
        operation_id,
        (tuple(int(size) for size in data.shape),),
        (data.dtype.name,),
        parameters=tuple(parameters.items()),
        resolved_spatial_ndim=2,
    )
    estimate = estimate_candidate_memory(spec, workload)
    admitted_peak = estimate.total_device_peak_bytes + estimate.uncertainty_bytes

    pool = cupy.cuda.MemoryPool()
    device_input = device_output = None
    observed_peak = 0
    cleanup_error = ""
    try:
        with cupy.cuda.using_allocator(pool.malloc):
            device_input = cupy.asarray(data)
            device_output = gpu_function(device_input, **parameters)
            if not isinstance(device_output, cupy.ndarray):
                raise EvidenceError(
                    f"Memory audit: GPU {operation_id} did not return a CuPy array."
                )
            if device_output.dtype != cupy.dtype(bool):
                raise EvidenceError(
                    f"Memory audit: GPU {operation_id} returned "
                    f"{device_output.dtype.name}, not bool."
                )
            cupy.cuda.get_current_stream().synchronize()
            # A private pool retains freed intermediate blocks, so total_bytes
            # captures the operation's allocator high-water capacity at this
            # synchronized boundary rather than only the live output.
            observed_peak = int(pool.total_bytes())
    finally:
        device_output = device_input = None
        try:
            cupy.cuda.get_current_stream().synchronize()
            pool.free_all_blocks()
            cupy.cuda.get_current_stream().synchronize()
        except BaseException as exc:  # pragma: no cover - hardware failure path
            cleanup_error = f"{type(exc).__name__}: {exc}"

    cleanup_used = int(pool.used_bytes())
    cleanup_reserved = int(pool.total_bytes())
    cleanup_passed = not cleanup_error and cleanup_used == 0 and cleanup_reserved == 0
    if observed_peak > admitted_peak:
        raise EvidenceError(
            f"Memory audit: {operation_id} observed {observed_peak:,} private-pool "
            f"bytes, above the admitted model total {admitted_peak:,}."
        )
    if not cleanup_passed:
        raise EvidenceError(
            f"Memory audit: {operation_id} private allocator did not drain "
            f"(used={cleanup_used}, reserved={cleanup_reserved}, "
            f"error={cleanup_error!r})."
        )
    return {
        "model_id": estimate.model_id,
        "runtime_managed_peak_bytes": estimate.runtime_managed_peak_bytes,
        "uncertainty_bytes": estimate.uncertainty_bytes,
        "admitted_device_peak_bytes": admitted_peak,
        "observed_private_pool_peak_bytes": observed_peak,
        "observed_within_admitted_peak": True,
        "cleanup": {
            "passed": True,
            "used_bytes_after_cleanup": cleanup_used,
            "reserved_bytes_after_cleanup": cleanup_reserved,
            "error": cleanup_error,
        },
    }


def _run_lifecycle_audit(
    *,
    cupy,
    gpu_functions: Mapping[str, Any],
) -> dict[str, object]:
    """Prove cooperative cancellation and private-pool cleanup per provider."""

    import numpy as np

    from napari_vipp.core.progress import OperationCancelled, ProgressContext

    data = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    records = []
    for operation_id, parameters in (
        (
            "canny_edges",
            {"sigma": 1.5, "low_quantile": 0.1, "high_quantile": 0.2},
        ),
        (
            "otsu_threshold",
            {"threshold_scope": "Slice histogram", "histogram_bins": 256},
        ),
    ):
        cancel_event = threading.Event()
        reported_progress: list[dict[str, object]] = []

        def reporter(
            update,
            event=cancel_event,
            sink=reported_progress,
        ) -> None:
            sink.append(
                {
                    "current": int(update.current),
                    "total": int(update.total),
                    "message": str(update.message),
                }
            )
            if int(update.current) == 1:
                event.set()

        progress = ProgressContext(
            cancelled=cancel_event.is_set,
            reporter=reporter,
        )
        pool = cupy.cuda.MemoryPool()
        device_input = None
        cancellation_observed = False
        cleanup_error = ""
        try:
            with cupy.cuda.using_allocator(pool.malloc):
                device_input = cupy.asarray(data)
                try:
                    gpu_functions[operation_id](
                        device_input,
                        progress=progress,
                        **parameters,
                    )
                except OperationCancelled:
                    cancellation_observed = True
        finally:
            device_input = None
            try:
                cupy.cuda.get_current_stream().synchronize()
                pool.free_all_blocks()
                cupy.cuda.get_current_stream().synchronize()
            except BaseException as exc:  # pragma: no cover - hardware failure path
                cleanup_error = f"{type(exc).__name__}: {exc}"
        cleanup_used = int(pool.used_bytes())
        cleanup_reserved = int(pool.total_bytes())
        cleanup_passed = (
            not cleanup_error and cleanup_used == 0 and cleanup_reserved == 0
        )
        if not cancel_event.is_set() or not cancellation_observed or not cleanup_passed:
            raise EvidenceError(
                f"Lifecycle audit failed for {operation_id}: requested="
                f"{cancel_event.is_set()}, observed={cancellation_observed}, "
                f"used={cleanup_used}, reserved={cleanup_reserved}, "
                f"error={cleanup_error!r}."
            )
        records.append(
            {
                "operation_id": operation_id,
                "implementation_id": (
                    CANNY_IMPLEMENTATION_ID
                    if operation_id == "canny_edges"
                    else OTSU_IMPLEMENTATION_ID
                ),
                "parameters": dict(parameters),
                # Persist the actual synchronized provider milestones observed
                # by the cancellation callback.  Do not turn a hard-coded
                # expected plane count into purported runtime evidence.
                "reported_progress": reported_progress,
                "cancellation_requested": True,
                "cancellation_observed": True,
                "cleanup": {
                    "passed": True,
                    "used_bytes_after_cleanup": cleanup_used,
                    "reserved_bytes_after_cleanup": cleanup_reserved,
                    "error": cleanup_error,
                },
            }
        )
    return {"status": "pass", "operations": records}


def _resource_preflight(data, cupy) -> None:
    # Canny is plane-wise, but its float32/float64 work arrays dominate one
    # plane.  The conservative multiplier also leaves room for CUDA context and
    # allocator fragmentation without pretending to be the policy memory model.
    plane_elements = math.prod(data.shape[-2:])
    estimated_device_bytes = int(data.nbytes + data.size + plane_elements * 128)
    free_device_bytes, _total = cupy.cuda.runtime.memGetInfo()
    if estimated_device_bytes > int(free_device_bytes * 0.75):
        raise EvidenceError(
            "CUDA-memory preflight rejected the performance stack: estimated "
            f"{estimated_device_bytes:,} bytes exceeds 75% of currently free memory."
        )


def _host_platform_record() -> dict[str, object]:
    """Return host fields that can be checked before importing CUDA."""

    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "machine": uname.machine,
        "processor": uname.processor,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "python_abi": str(getattr(sys.implementation, "cache_tag", "")),
        "execution_mode": _execution_mode(uname),
    }


def _platform_record(cupy, device_index: int) -> dict[str, object]:
    properties = cupy.cuda.runtime.getDeviceProperties(device_index)
    raw_name = properties.get("name", "unknown CUDA device")
    if isinstance(raw_name, bytes):
        raw_name = raw_name.decode(errors="replace")
    result = _host_platform_record()
    result.update(
        {
            "cuda_device_index": device_index,
            "cuda_device_name": str(raw_name),
            "cuda_compute_capability": (
                f"{int(properties['major'])}.{int(properties['minor'])}"
            ),
            "cuda_driver_version": int(cupy.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": int(cupy.cuda.runtime.runtimeGetVersion()),
            "total_accelerator_memory_bytes": int(properties["totalGlobalMem"]),
        }
    )
    return result


def _execution_mode(uname: platform.uname_result) -> str:
    details = f"{uname.release} {uname.version}".casefold()
    if uname.system.casefold() == "linux" and "microsoft" in details:
        return "wsl"
    return "native"


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in (
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy-cuda12x",
        "cupy-cuda13x",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def _source_provenance() -> list[dict[str, str]]:
    result = []
    for relative in SOURCE_PROVENANCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise EvidenceError(
                f"Required source provenance file is missing: {relative}"
            )
        result.append({"path": relative, "sha256": _file_sha256(path)})
    return result


def _require_source_snapshot_unchanged(
    provenance: Sequence[Mapping[str, str]],
) -> None:
    for item in provenance:
        relative = str(item["path"])
        if _file_sha256(PROJECT_ROOT / relative) != str(item["sha256"]):
            raise EvidenceError(f"Source changed while benchmarking: {relative}")


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


def render_markdown(document: Mapping[str, object]) -> str:
    _validate_document_contract(document)
    platform_info = _mapping(document["platform"], "platform")
    admission = _mapping(document["admission"], "admission")
    performance = _mapping(document["performance"], "performance")
    lifecycle = _mapping(document["lifecycle"], "lifecycle")
    lines = [
        "# Exact CuPy Canny and Otsu evidence",
        "",
        f"- Generated: `{document['created_utc']}`",
        f"- Device: `{platform_info['cuda_device_name']}`",
        f"- Admission cases: `{admission['case_count']}` (all exact)",
        f"- Timed sources: `{performance['source_count']}`",
        "- Cancellation/cleanup providers: "
        f"`{len(lifecycle['operations'])}` (all pass)",
        "",
        "The admission matrix compares the production CPU operations with the",
        "production CuPy providers and requires identical boolean masks. Timing",
        "is a short machine-local screen, not a portable speed claim or a durable",
        "optimizer record. GPU end-to-end includes both transfers and synchronized",
        "compute; resident GPU time models an already-GPU pipeline.",
        "",
        "| Source | Operation | Elements | CPU | GPU end-to-end | GPU resident | "
        "End-to-end speedup | Resident speedup | Screen winner |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for raw_source in performance["sources"]:
        source = _mapping(raw_source, "performance source")
        for raw_operation in source["operations"]:
            operation = _mapping(raw_operation, "performance operation")
            summary = _mapping(operation["summary"], "timing summary")
            lines.append(
                "| "
                + " | ".join(
                    (
                        str(source["label"]),
                        str(operation["operation_id"]),
                        f"{int(source['element_count']):,}",
                        _seconds(summary["cpu_median_seconds"]),
                        _seconds(summary["gpu_end_to_end_median_seconds"]),
                        _seconds(summary["gpu_resident_median_seconds"]),
                        _ratio(summary["gpu_end_to_end_speedup"]),
                        _ratio(summary["gpu_resident_speedup"]),
                        str(summary["screening_choice"]),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "## Memory and lifecycle evidence",
            "",
        )
    )
    for raw_source in performance["sources"]:
        source = _mapping(raw_source, "performance source")
        for raw_operation in source["operations"]:
            operation = _mapping(raw_operation, "performance operation")
            memory = _mapping(operation["memory"], "memory evidence")
            lines.append(
                f"- **{source['label']} / {operation['operation_id']}:** observed "
                f"`{int(memory['observed_private_pool_peak_bytes']):,}` private-pool "
                f"bytes within `{int(memory['admitted_device_peak_bytes']):,}` "
                "admitted; "
                "cleanup passed."
            )
    for raw_record in lifecycle["operations"]:
        record = _mapping(raw_record, "lifecycle operation")
        lines.append(
            f"- **{record['operation_id']}:** cooperative cancellation observed; "
            "private allocator cleanup passed."
        )
    lines.extend(
        (
            "",
            "## Admission coverage",
            "",
        )
    )
    coverage = _mapping(admission["coverage"], "admission coverage")
    for operation_id in sorted(coverage):
        lines.append(f"- **{operation_id}:** " + ", ".join(coverage[operation_id]))
    lines.extend(
        (
            "",
            "## Interpretation limits",
            "",
            "- Exact admission applies only to the versioned regions represented",
            "  by the checked coverage tags and production policy contracts.",
            "- The structured synthetic stack stresses large plane-wise work; it",
            "  is not a claim that synthetic textures reproduce confocal biology.",
            "- The optional private acquisition is a real-data anchor. Its path,",
            "  filename, content digest, and pixels are deliberately absent.",
            "- Timings exclude disk I/O and input generation. Re-run the pipeline",
            "  optimizer on the user's actual workload before persisting a choice.",
            "",
        )
    )
    return "\n".join(lines)


def _seconds(value: object) -> str:
    return f"{float(value):.4f} s"


def _ratio(value: object) -> str:
    return f"{float(value):.2f}x"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{name} must be an object.")
    return value


def _validate_public_v3_environment_contract(
    platform_record: Mapping[str, object],
    packages: Mapping[str, object],
) -> None:
    """Require the exact host/runtime envelope admitted by public policy v3."""

    _validate_platform_contract(platform_record)
    installed_cupy_distribution = _validate_package_contract(packages)
    runtime_major = int(platform_record["cuda_runtime_version"]) // 1000
    expected_distribution = f"cupy-cuda{runtime_major}x"
    if installed_cupy_distribution != expected_distribution:
        raise EvidenceError(
            "CUDA runtime/package track mismatch: public-v3 evidence requires "
            f"{expected_distribution} for CUDA {runtime_major}, found "
            f"{installed_cupy_distribution}."
        )


def _validate_public_v3_host_contract(
    platform_record: Mapping[str, object],
) -> None:
    """Validate the CUDA-independent public-v3 host fields."""

    _require_exact_keys(
        platform_record,
        _HOST_PLATFORM_KEYS,
        "Public-v3 host platform",
    )
    system = _bounded_metadata_text(
        platform_record.get("system"),
        "platform system",
    )
    if system != "Windows":
        raise EvidenceError("Public-v3 Canny/Otsu evidence requires native Windows.")
    if platform_record.get("execution_mode") != "native":
        raise EvidenceError(
            "Public-v3 Canny/Otsu evidence requires native execution, not WSL."
        )
    if platform_record.get("python_implementation") != "CPython":
        raise EvidenceError("Public-v3 evidence requires CPython.")
    if platform_record.get("python_abi") != "cpython-312":
        raise EvidenceError("Public-v3 evidence requires the cpython-312 ABI.")
    python_version = _bounded_metadata_text(
        platform_record.get("python"),
        "Python version",
    )
    if re.fullmatch(r"3\.12\.\d+", python_version) is None:
        raise EvidenceError("Public-v3 evidence requires Python 3.12.x.")
    _bounded_metadata_text(platform_record.get("release"), "platform release")
    _bounded_metadata_text(platform_record.get("machine"), "platform machine")
    _bounded_metadata_text(
        platform_record.get("processor"),
        "platform processor",
        allow_empty=True,
        maximum_length=512,
    )


def _validate_platform_contract(platform_record: Mapping[str, object]) -> None:
    _require_exact_keys(
        platform_record,
        _PLATFORM_KEYS,
        "Public-v3 platform/device metadata",
    )
    host_record = {key: platform_record[key] for key in _HOST_PLATFORM_KEYS}
    _validate_public_v3_host_contract(host_record)
    _nonnegative_int(platform_record.get("cuda_device_index"), "CUDA device index")
    device_name = _bounded_metadata_text(
        platform_record.get("cuda_device_name"),
        "CUDA device name",
    )
    if device_name != PUBLIC_V3_CUDA_DEVICE_NAME:
        raise EvidenceError(
            "Public-v3 evidence is admitted only for the exact reviewed device "
            f"{PUBLIC_V3_CUDA_DEVICE_NAME!r}; found {device_name!r}."
        )
    capability = _bounded_metadata_text(
        platform_record.get("cuda_compute_capability"),
        "CUDA compute capability",
    )
    if capability != PUBLIC_V3_CUDA_COMPUTE_CAPABILITY:
        raise EvidenceError(
            "Public-v3 evidence requires CUDA compute capability "
            f"{PUBLIC_V3_CUDA_COMPUTE_CAPABILITY}; found {capability!r}."
        )
    driver_version = _nonnegative_int(
        platform_record.get("cuda_driver_version"),
        "CUDA driver version",
    )
    runtime_version = _nonnegative_int(
        platform_record.get("cuda_runtime_version"),
        "CUDA runtime version",
    )
    if driver_version != PUBLIC_V3_CUDA_DRIVER_VERSION:
        raise EvidenceError(
            "Public-v3 evidence requires exact CUDA driver version "
            f"{PUBLIC_V3_CUDA_DRIVER_VERSION}; found {driver_version}."
        )
    if runtime_version != PUBLIC_V3_CUDA_RUNTIME_VERSION:
        raise EvidenceError(
            "Public-v3 evidence requires exact CUDA runtime version "
            f"{PUBLIC_V3_CUDA_RUNTIME_VERSION}; found {runtime_version}."
        )
    total_memory = _nonnegative_int(
        platform_record.get("total_accelerator_memory_bytes"),
        "total accelerator memory",
    )
    if total_memory == 0:
        raise EvidenceError("Total accelerator memory must be positive.")


def _validate_package_contract(packages: Mapping[str, object]) -> str:
    _require_exact_keys(packages, _PACKAGE_KEYS, "Public-v3 package metadata")
    project_version = _bounded_metadata_text(
        packages.get("napari-vipp"),
        "napari-vipp package version",
    )
    if (
        project_version == "not-installed"
        or re.fullmatch(
            r"[0-9A-Za-z][0-9A-Za-z.!+_-]*",
            project_version,
        )
        is None
    ):
        raise EvidenceError("napari-vipp package version metadata is invalid.")
    for distribution, expected in PUBLIC_V3_SCIENTIFIC_STACK.items():
        actual = packages.get(distribution)
        if actual != expected:
            raise EvidenceError(
                "Public-v3 Canny/Otsu evidence requires the exact scientific "
                f"stack; {distribution} must be {expected}, found {actual!r}."
            )
    installed_cupy = tuple(
        distribution
        for distribution in ("cupy-cuda12x", "cupy-cuda13x")
        if packages.get(distribution) != "not-installed"
    )
    if len(installed_cupy) != 1:
        raise EvidenceError(
            "Public-v3 evidence requires exactly one installed CuPy CUDA distribution."
        )
    distribution = installed_cupy[0]
    if packages.get(distribution) != PUBLIC_V3_CUPY_VERSION:
        raise EvidenceError(
            f"Public-v3 evidence requires {distribution} {PUBLIC_V3_CUPY_VERSION}."
        )
    return distribution


def _bounded_metadata_text(
    value: object,
    name: str,
    *,
    allow_empty: bool = False,
    maximum_length: int = 256,
) -> str:
    if not isinstance(value, str):
        raise EvidenceError(f"{name} must be text.")
    if (not value and not allow_empty) or len(value) > maximum_length:
        raise EvidenceError(f"{name} has an invalid length.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise EvidenceError(f"{name} contains control characters.")
    if "/" in value or "\\" in value:
        raise EvidenceError(f"{name} must not contain a filesystem path.")
    return value


def _validate_source_provenance_contract(value: object) -> None:
    if not isinstance(value, list):
        raise EvidenceError("Evidence source provenance must be a list.")
    if len(value) != len(SOURCE_PROVENANCE_PATHS):
        raise EvidenceError(
            "Evidence source provenance must contain every canonical source once."
        )
    for index, expected_path in enumerate(SOURCE_PROVENANCE_PATHS):
        item = _mapping(value[index], "source provenance entry")
        _require_exact_keys(
            item,
            _SOURCE_PROVENANCE_KEYS,
            "Source provenance entry",
        )
        if item.get("path") != expected_path:
            raise EvidenceError(
                "Evidence source provenance order/path changed at index "
                f"{index}: expected {expected_path!r}."
            )
        _hex_digest(item.get("sha256"), "source provenance SHA-256")


def _validate_document_contract(document: Mapping[str, object]) -> None:
    _require_exact_keys(document, _ROOT_KEYS, "Evidence root")
    if document.get("schema") != SCHEMA:
        raise EvidenceError("Unexpected evidence schema.")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise EvidenceError("Unexpected evidence schema version.")
    if document.get("kind") != (
        "scientific-admission-and-machine-local-performance-evidence"
    ):
        raise EvidenceError("Unexpected evidence kind.")
    if document.get("portable_performance_claim") is not False:
        raise EvidenceError("Evidence must reject portable speed claims.")
    if document.get("durable_optimizer_record") is not False:
        raise EvidenceError("This short screen must not be a durable optimizer record.")
    if document.get("profile") != PROFILE:
        raise EvidenceError("Unexpected performance profile.")
    created_utc = document.get("created_utc")
    if not isinstance(created_utc, str):
        raise EvidenceError("Evidence created_utc must be text.")
    try:
        created = datetime.fromisoformat(created_utc)
    except ValueError as exc:
        raise EvidenceError("Evidence created_utc is invalid.") from exc
    if created.tzinfo is None or created.utcoffset() != timedelta(0):
        raise EvidenceError("Evidence created_utc must be timezone-aware UTC.")

    method = _mapping(document.get("method"), "method")
    _require_exact_keys(method, _METHOD_KEYS, "Evidence method")
    required_method = {
        "cpu_path": "production-napari-vipp-operations",
        "gpu_path": "production-exact-cupy-providers",
        "parity": "bitwise-identical-boolean-mask",
        "warmup_rounds": WARMUP_ROUNDS,
        "timed_rounds": BENCHMARK_ROUNDS,
        "cpu_timing_scope": "synchronous-operation-call-v1",
        "gpu_end_to_end_timing_scope": (
            "host-to-device-plus-synchronized-compute-plus-device-to-host-v1"
        ),
        "gpu_resident_timing_scope": "synchronized-resident-compute-v1",
        "disk_io_included": False,
        "input_generation_included": False,
        "exact_parity_required_before_timing": True,
    }
    for key, expected in required_method.items():
        if method.get(key) != expected:
            raise EvidenceError(f"Evidence method field {key!r} changed.")

    platform_record = _mapping(document.get("platform"), "platform")
    packages = _mapping(document.get("packages"), "packages")
    _validate_public_v3_environment_contract(platform_record, packages)
    _validate_source_provenance_contract(document.get("source_provenance"))
    admission = _mapping(document.get("admission"), "admission")
    _validate_admission_contract(admission)
    performance = _mapping(document.get("performance"), "performance")
    _validate_performance_contract(performance)
    lifecycle = _mapping(document.get("lifecycle"), "lifecycle")
    _validate_lifecycle_contract(lifecycle)
    _validate_privacy(document)


def _validate_admission_contract(admission: Mapping[str, object]) -> None:
    _require_exact_keys(admission, _ADMISSION_KEYS, "Admission evidence")
    cases = admission.get("cases")
    if admission.get("status") != "pass" or admission.get("failure_count") != 0:
        raise EvidenceError("Admission evidence did not pass.")
    if not isinstance(cases, list):
        raise EvidenceError("Admission evidence has no cases.")
    expected_cases = _admission_cases()
    if (
        admission.get("case_count") != len(cases)
        or len(cases) != len(expected_cases)
    ):
        raise EvidenceError("Admission case count is inconsistent.")
    covered = {operation: set() for operation in REQUIRED_COVERAGE}
    for raw_case, expected_case in zip(cases, expected_cases, strict=True):
        case = _mapping(raw_case, "admission case")
        _require_exact_keys(case, _ADMISSION_CASE_KEYS, "Admission case")
        case_id = case.get("case_id")
        if case_id != expected_case.case_id:
            raise EvidenceError(
                "Admission case order/identity changed; expected "
                f"{expected_case.case_id!r}."
            )
        operation_id = case.get("operation_id")
        if operation_id != expected_case.operation_id:
            raise EvidenceError(f"Admission case {case_id!r} changed operation.")
        expected_input_shape = tuple(int(size) for size in expected_case.data.shape)
        input_shape = _positive_shape(case.get("input_shape"), "admission input shape")
        if input_shape != expected_input_shape:
            raise EvidenceError(f"Admission case {case_id!r} changed input shape.")
        expected_dtype = expected_case.data.dtype.name
        if case.get("input_dtype") != expected_dtype:
            raise EvidenceError(f"Admission case {case_id!r} changed input dtype.")
        if case.get("input_sha256") != _array_sha256(expected_case.data):
            raise EvidenceError(f"Admission case {case_id!r} changed input data.")
        parameters = _mapping(case.get("parameters"), "admission parameters")
        if dict(parameters) != dict(expected_case.parameters):
            raise EvidenceError(f"Admission case {case_id!r} changed parameters.")
        coverage = case.get("coverage")
        if coverage != list(expected_case.coverage):
            raise EvidenceError(f"Admission case {case_id!r} changed coverage tags.")
        expected_output_shape = _project_admission_output_shape(
            expected_input_shape,
            expected_case.parameters,
        )
        output_shape = _positive_shape(
            case.get("output_shape"),
            "admission output shape",
        )
        if output_shape != expected_output_shape:
            raise EvidenceError(f"Admission case {case_id!r} changed output shape.")
        if (
            case.get("exact_mask_match") is not True
            or case.get("mismatch_count") != 0
            or case.get("output_dtype") != "bool"
            or case.get("gpu_output_dtype") != "bool"
            or case.get("gpu_output_resident") is not True
        ):
            raise EvidenceError(f"Admission case {case_id!r} is not exact.")
        _hex_digest(case.get("input_sha256"), "admission input SHA-256")
        cpu_foreground = _nonnegative_int(
            case.get("cpu_foreground_pixels"),
            "CPU foreground pixels",
        )
        gpu_foreground = _nonnegative_int(
            case.get("gpu_foreground_pixels"),
            "GPU foreground pixels",
        )
        if (
            cpu_foreground != gpu_foreground
            or cpu_foreground > math.prod(output_shape)
        ):
            raise EvidenceError(
                f"Admission case {case_id!r} foreground counts are inconsistent."
            )
        covered[operation_id].update(coverage)
    for operation_id, required in REQUIRED_COVERAGE.items():
        if covered[operation_id] != set(required):
            raise EvidenceError(
                f"Admission operation {operation_id} coverage changed."
            )
    serialized_coverage = _mapping(admission.get("coverage"), "coverage")
    _require_exact_keys(
        serialized_coverage,
        frozenset(REQUIRED_COVERAGE),
        "Admission coverage",
    )
    for operation_id, required in REQUIRED_COVERAGE.items():
        if serialized_coverage.get(operation_id) != sorted(required):
            raise EvidenceError("Serialized admission coverage is inconsistent.")


def _project_admission_output_shape(
    input_shape: tuple[int, ...],
    parameters: Mapping[str, object],
) -> tuple[int, ...]:
    raw_axis = parameters.get("channel_axis")
    if raw_axis is None:
        return input_shape
    axis = int(raw_axis) % len(input_shape)
    return input_shape[:axis] + input_shape[axis + 1 :]


def _validate_performance_contract(performance: Mapping[str, object]) -> None:
    _require_exact_keys(performance, _PERFORMANCE_KEYS, "Performance evidence")
    sources = performance.get("sources")
    if performance.get("status") != "pass":
        raise EvidenceError("Performance evidence did not pass.")
    if not isinstance(sources, list) or len(sources) not in {1, 2}:
        raise EvidenceError("Performance evidence has no sources.")
    if performance.get("source_count") != len(sources):
        raise EvidenceError("Performance source count is inconsistent.")
    expected_source_ids = ["synthetic-structured-large-stack"]
    if len(sources) == 2:
        expected_source_ids.append(PRIVATE_SOURCE_ID)
    for raw_source, expected_source_id in zip(
        sources,
        expected_source_ids,
        strict=True,
    ):
        source = _mapping(raw_source, "performance source")
        _require_exact_keys(
            source,
            _PERFORMANCE_SOURCE_KEYS,
            "Performance source",
        )
        source_id = source.get("source_id")
        if source_id != expected_source_id:
            raise EvidenceError("Performance source order/identity changed.")
        shape = _positive_shape(source.get("shape"), "performance shape")
        if len(shape) != 3:
            raise EvidenceError("Performance inputs must be ZYX stacks.")
        element_count = math.prod(shape)
        if source.get("element_count") != element_count:
            raise EvidenceError("Performance element_count is inconsistent.")
        if source.get("direct_private_identifiers_published") is not False:
            raise EvidenceError("Private identifiers must not be published.")
        private = source.get("source_kind") == PRIVATE_SOURCE_KIND
        if private:
            if source_id != PRIVATE_SOURCE_ID:
                raise EvidenceError(
                    "Private performance source ID must use the fixed "
                    "privacy-safe value."
                )
            if source.get("label") != PRIVATE_SOURCE_LABEL:
                raise EvidenceError(
                    "Private performance source label must use the fixed "
                    "privacy-safe value."
                )
            if source.get("input_sha256") is not None:
                raise EvidenceError("Private performance input_sha256 must be null.")
            metadata = _mapping(source.get("source_metadata"), "private metadata")
            _require_exact_keys(
                metadata,
                _PRIVATE_SOURCE_METADATA_KEYS,
                "Private source metadata",
            )
            if metadata.get("direct_identifiers_omitted") is not True:
                raise EvidenceError("Private metadata is not explicitly redacted.")
            _validate_private_source_metadata(source, metadata)
        else:
            if source.get("source_kind") != "deterministic-synthetic":
                raise EvidenceError("Unexpected performance source kind.")
            if source_id != "synthetic-structured-large-stack":
                raise EvidenceError("Unexpected synthetic performance source ID.")
            expected_label = (
                "Structured synthetic "
                + "x".join(str(size) for size in SYNTHETIC_SHAPE)
                + " uint16 stack"
            )
            if source.get("label") != expected_label:
                raise EvidenceError("Synthetic performance source label changed.")
            metadata = _mapping(source.get("source_metadata"), "synthetic metadata")
            _require_exact_keys(
                metadata,
                _SYNTHETIC_SOURCE_METADATA_KEYS,
                "Synthetic source metadata",
            )
            if metadata.get("generator") != GENERATOR_ID:
                raise EvidenceError("Synthetic performance generator changed.")
            _hex_digest(source.get("input_sha256"), "performance input SHA-256")
            if tuple(shape) != SYNTHETIC_SHAPE or source.get("dtype") != "uint16":
                raise EvidenceError("Synthetic performance workload changed.")
        dtype_name = source.get("dtype")
        itemsize = {"uint8": 1, "uint16": 2}.get(dtype_name)
        if itemsize is None:
            raise EvidenceError("Performance source dtype is not admitted.")
        if source.get("input_bytes") != element_count * itemsize:
            raise EvidenceError("Performance input_bytes is inconsistent.")
        operations = source.get("operations")
        if not isinstance(operations, list) or len(operations) != 2:
            raise EvidenceError("Each performance source requires Canny and Otsu.")
        for raw_operation, operation_id in zip(
            operations,
            _PERFORMANCE_OPERATION_CONTRACTS,
            strict=True,
        ):
            operation = _mapping(raw_operation, "performance operation")
            _require_exact_keys(
                operation,
                _PERFORMANCE_OPERATION_KEYS,
                "Performance operation",
            )
            contract = _PERFORMANCE_OPERATION_CONTRACTS[operation_id]
            if operation.get("operation_id") != operation_id:
                raise EvidenceError("Performance operation order changed.")
            if operation.get("implementation_id") != contract["implementation_id"]:
                raise EvidenceError("Unexpected performance implementation ID.")
            parameters = _mapping(
                operation.get("parameters"),
                "performance parameters",
            )
            if dict(parameters) != contract["parameters"]:
                raise EvidenceError("Performance operation parameters changed.")
            parity = _mapping(operation.get("parity"), "performance parity")
            _require_exact_keys(parity, _PARITY_KEYS, "Performance parity")
            if (
                parity.get("profile") != "bitwise-identical-boolean-mask"
                or parity.get("passed") is not True
                or parity.get("mismatch_count") != 0
                or parity.get("cpu_output_dtype") != "bool"
                or parity.get("gpu_output_dtype") != "bool"
                or parity.get("gpu_output_resident") is not True
            ):
                raise EvidenceError("A timed implementation failed exact parity.")
            foreground = _nonnegative_int(
                parity.get("foreground_pixels"),
                "performance foreground pixels",
            )
            if foreground > element_count:
                raise EvidenceError("Performance foreground count is inconsistent.")
            memory = _mapping(operation.get("memory"), "memory evidence")
            _require_exact_keys(memory, _MEMORY_KEYS, "Memory evidence")
            if memory.get("model_id") != contract["memory_model_id"]:
                raise EvidenceError("A timed implementation lacks memory-model proof.")
            expected_memory = _expected_memory_estimate(
                operation_id,
                shape,
                str(dtype_name),
                parameters,
            )
            runtime_managed = _nonnegative_int(
                memory.get("runtime_managed_peak_bytes"),
                "runtime_managed_peak_bytes",
            )
            uncertainty = _nonnegative_int(
                memory.get("uncertainty_bytes"),
                "uncertainty_bytes",
            )
            admitted = _nonnegative_int(
                memory.get("admitted_device_peak_bytes"),
                "admitted_device_peak_bytes",
            )
            observed = _nonnegative_int(
                memory.get("observed_private_pool_peak_bytes"),
                "observed_private_pool_peak_bytes",
            )
            if (
                runtime_managed != expected_memory.runtime_managed_peak_bytes
                or uncertainty != expected_memory.uncertainty_bytes
                or admitted != runtime_managed + uncertainty
                or observed == 0
                or observed > admitted
                or memory.get("observed_within_admitted_peak") is not True
            ):
                raise EvidenceError("Observed device peak exceeds policy admission.")
            _validate_cleanup_contract(memory.get("cleanup"), "memory cleanup")
            samples = _mapping(operation.get("samples"), "timing samples")
            _require_exact_keys(samples, _TIMING_SAMPLE_KEYS, "Timing samples")
            for key in _TIMING_SAMPLE_KEYS:
                values = samples.get(key)
                if not isinstance(values, list) or len(values) != BENCHMARK_ROUNDS:
                    raise EvidenceError(f"Timing field {key!r} is incomplete.")
                for value in values:
                    _positive_finite(value, key)
            _validate_timing_summary(operation)


def _expected_memory_estimate(
    operation_id: str,
    shape: tuple[int, ...],
    dtype: str,
    parameters: Mapping[str, object],
):
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import estimate_candidate_memory
    from napari_vipp.core.compute_specs import compute_specs_for

    contract = _PERFORMANCE_OPERATION_CONTRACTS[operation_id]
    implementation_id = str(contract["implementation_id"])
    spec = next(
        item
        for item in compute_specs_for(operation_id, include_cpu=False)
        if item.implementation_id == implementation_id
    )
    workload = WorkloadDescriptor(
        f"evidence-{operation_id}",
        operation_id,
        (shape,),
        (dtype,),
        parameters=tuple(parameters.items()),
        resolved_spatial_ndim=2,
    )
    estimate = estimate_candidate_memory(spec, workload)
    if estimate.model_id != contract["memory_model_id"]:
        raise EvidenceError("Production memory-model identity changed.")
    return estimate


def _validate_lifecycle_contract(lifecycle: Mapping[str, object]) -> None:
    _require_exact_keys(lifecycle, _LIFECYCLE_KEYS, "Lifecycle evidence")
    records = lifecycle.get("operations")
    if (
        lifecycle.get("status") != "pass"
        or not isinstance(records, list)
        or len(records) != len(_LIFECYCLE_OPERATION_CONTRACTS)
    ):
        raise EvidenceError("Lifecycle evidence did not pass.")
    for raw_record, operation_id in zip(
        records,
        _LIFECYCLE_OPERATION_CONTRACTS,
        strict=True,
    ):
        record = _mapping(raw_record, "lifecycle operation")
        _require_exact_keys(
            record,
            _LIFECYCLE_OPERATION_KEYS,
            "Lifecycle operation",
        )
        contract = _LIFECYCLE_OPERATION_CONTRACTS[operation_id]
        if (
            record.get("operation_id") != operation_id
            or record.get("implementation_id") != contract["implementation_id"]
            or dict(_mapping(record.get("parameters"), "lifecycle parameters"))
            != contract["parameters"]
            or record.get("cancellation_requested") is not True
            or record.get("cancellation_observed") is not True
        ):
            raise EvidenceError(
                f"Lifecycle cancellation evidence failed for {operation_id!r}."
            )
        progress = record.get("reported_progress")
        if not isinstance(progress, list) or len(progress) != 2:
            raise EvidenceError(
                f"Lifecycle progress evidence failed for {operation_id!r}."
            )
        normalized_progress = []
        for raw_update in progress:
            update = _mapping(raw_update, "lifecycle progress update")
            _require_exact_keys(
                update,
                _PROGRESS_UPDATE_KEYS,
                "Lifecycle progress update",
            )
            normalized_progress.append(dict(update))
        if tuple(normalized_progress) != contract["progress"]:
            raise EvidenceError(
                f"Lifecycle progress evidence failed for {operation_id!r}."
            )
        _validate_cleanup_contract(record.get("cleanup"), "lifecycle cleanup")


def _validate_cleanup_contract(value: object, name: str) -> None:
    cleanup = _mapping(value, name)
    _require_exact_keys(cleanup, _CLEANUP_KEYS, name)
    if (
        cleanup.get("passed") is not True
        or _nonnegative_int(
            cleanup.get("used_bytes_after_cleanup"),
            "used_bytes_after_cleanup",
        )
        != 0
        or _nonnegative_int(
            cleanup.get("reserved_bytes_after_cleanup"),
            "reserved_bytes_after_cleanup",
        )
        != 0
        or cleanup.get("error") != ""
    ):
        raise EvidenceError(f"{name} did not drain the private allocator.")


def _validate_timing_summary(operation: Mapping[str, object]) -> None:
    samples = _mapping(operation["samples"], "timing samples")
    summary = _mapping(operation.get("summary"), "timing summary")
    _require_exact_keys(summary, _TIMING_SUMMARY_KEYS, "Timing summary")
    cpu = tuple(float(value) for value in samples["cpu_seconds"])
    end_to_end = tuple(float(value) for value in samples["gpu_end_to_end_seconds"])
    resident = tuple(float(value) for value in samples["gpu_resident_seconds"])
    cpu_median = statistics.median(cpu)
    e2e_median = statistics.median(end_to_end)
    resident_median = statistics.median(resident)
    expected = {
        "cpu_median_seconds": cpu_median,
        "gpu_end_to_end_median_seconds": e2e_median,
        "gpu_resident_median_seconds": resident_median,
        "gpu_end_to_end_speedup": cpu_median / e2e_median,
        "gpu_resident_speedup": cpu_median / resident_median,
        "screening_choice": "GPU-CuPy" if e2e_median < cpu_median else "CPU",
    }
    for key, value in expected.items():
        actual = summary.get(key)
        if isinstance(value, str):
            if actual != value:
                raise EvidenceError(f"Timing summary field {key!r} is inconsistent.")
        else:
            actual_value = _positive_finite(actual, key)
            if not math.isclose(
                actual_value,
                value,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise EvidenceError(
                    f"Timing summary field {key!r} is inconsistent."
                )


def _validate_privacy(document: Mapping[str, object]) -> None:
    serialized = json.dumps(document, sort_keys=True).lower()
    forbidden = (
        ".nd2",
        "file://",
        "/home/",
        "/users/",
        "/volumes/",
        "/mnt/",
        "/media/",
    )
    for marker in forbidden:
        if marker in serialized:
            raise EvidenceError(
                f"Evidence contains forbidden private marker {marker!r}."
            )
    if re.search(r"(?<![a-z0-9])[a-z]:[\\/]", serialized) is not None:
        raise EvidenceError("Evidence contains a forbidden absolute drive path.")
    if re.search(r"\\\\[a-z0-9]", serialized) is not None:
        raise EvidenceError("Evidence contains a forbidden UNC path.")


def _validate_private_source_metadata(
    source: Mapping[str, object],
    metadata: Mapping[str, object],
) -> None:
    """Validate only bounded, non-identifying acquisition descriptors."""

    axes = metadata.get("original_axes")
    if (
        not isinstance(axes, str)
        or not axes
        or len(axes) > len(_PRIVATE_ND2_AXES)
        or len(set(axes)) != len(axes)
        or not set(axes) <= _PRIVATE_ND2_AXES
        or "".join(axis for axis in axes if axis in "ZYX") != "ZYX"
    ):
        raise EvidenceError("Private source axes metadata is invalid.")
    original_shape = _positive_shape(
        metadata.get("original_shape"),
        "private original shape",
    )
    if len(original_shape) != len(axes):
        raise EvidenceError("Private source axes and original shape are inconsistent.")
    if any(size > _MAX_PRIVATE_AXIS_EXTENT for size in original_shape):
        raise EvidenceError("Private source axis extent exceeds the metadata bound.")
    if math.prod(original_shape) > _MAX_PRIVATE_ELEMENT_COUNT:
        raise EvidenceError("Private source shape exceeds the metadata bound.")
    axis_sizes = dict(zip(axes, original_shape, strict=True))
    if "P" in axis_sizes and axis_sizes["P"] != 1:
        raise EvidenceError("Private source P axis must be singleton.")
    source_shape = _positive_shape(source.get("shape"), "private source shape")
    expected_source_shape = tuple(axis_sizes[axis] for axis in "ZYX")
    if source_shape != expected_source_shape:
        raise EvidenceError(
            "Private source shape does not match the selected ZYX extents."
        )

    original_dtype = metadata.get("original_dtype")
    if original_dtype not in {"uint8", "uint16"}:
        raise EvidenceError("Private source dtype metadata is not admitted.")
    if source.get("dtype") != original_dtype:
        raise EvidenceError("Private source dtype metadata is inconsistent.")

    selected_indices = _mapping(
        metadata.get("selected_indices"),
        "private selected indices",
    )
    expected_index_axes = {axis for axis in ("T", "C") if axis in axes}
    if set(selected_indices) != expected_index_axes:
        raise EvidenceError(
            "Private selected-index metadata does not match the fixed "
            "privacy-safe schema."
        )
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= axis_sizes[axis]
        for axis, value in selected_indices.items()
    ):
        raise EvidenceError(
            "Private selected indices must be in range for their original axes."
        )


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise EvidenceError(f"{name} does not match the fixed privacy-safe schema.")


def _positive_shape(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise EvidenceError(f"{name} must be a nonempty list.")
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in value
    ):
        raise EvidenceError(f"{name} must contain positive integers.")
    return tuple(value)


def _positive_finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise EvidenceError(f"{name} must be finite and positive.")
    return float(value)


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EvidenceError(f"{name} must be a non-negative integer.")
    return value


def _hex_digest(value: object, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise EvidenceError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def validate_existing(path: Path) -> Path:
    artifact = path.expanduser().resolve()
    with artifact.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise EvidenceError("Evidence root must be an object.")
    _validate_document_contract(document)
    provenance = document.get("source_provenance")
    if not isinstance(provenance, list):
        raise EvidenceError("Evidence source provenance is missing.")
    expected = _source_provenance()
    actual = [dict(_mapping(item, "source provenance")) for item in provenance]
    if actual != expected:
        raise EvidenceError(
            "Evidence source fingerprints do not match the current checkout."
        )
    markdown = artifact.with_suffix(".md")
    if markdown.is_file() and markdown.read_text(encoding="utf-8") != render_markdown(
        document
    ):
        raise EvidenceError("Readable Markdown does not match the JSON renderer.")
    return artifact


def _atomic_write_artifacts(
    output: Path,
    markdown: Path,
    document: Mapping[str, object],
) -> None:
    _validate_document_contract(document)
    json_text = (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    markdown_text = render_markdown(document)
    _atomic_write_text(output, json_text)
    _atomic_write_text(markdown, markdown_text)


def _atomic_write_text(path: Path, text: str) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
