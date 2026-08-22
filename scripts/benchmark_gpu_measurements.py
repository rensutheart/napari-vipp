#!/usr/bin/env python
"""Build CPU/CuPy basic object-measurement evidence on a real CUDA device.

The harness treats a GPU measurement as a vertical slice: resident label and
intensity inputs, a resident packed ``float64`` result, mandatory device-to-host
transfer, and the operation-owned typed ``TableData`` finalizer.  Admission
proves complete public-table parity for morphology and intensity measurements;
performance records resident compute separately from the unavoidable table
boundary and from full transfer-inclusive execution.

The ``full`` profile covers 2D and 3D measurements, leading blocks, arbitrary
sparse and repeated label IDs, empty labels, bool/uint8/uint16/finite-float32
intensity, calibrated and reordered axes, deterministic repeats, synchronized
progress/cancellation, private-pool cleanup, the production memory bound, and
large confocal-like stacks.  Results are machine-local evidence, not portable
speed claims or durable optimizer records.

Importing this module, asking for ``--help``, and ``--validate-existing`` do not
import CuPy, NumPy, or initialize CUDA.
"""

from __future__ import annotations

import argparse
import gc
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
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "napari-vipp-cupy-basic-measurements-evidence"
SCHEMA_VERSION = 1
MORPHOLOGY_OPERATION_ID = "measure_objects"
INTENSITY_OPERATION_ID = "measure_objects_intensity"
IMPLEMENTATION_IDS = {
    MORPHOLOGY_OPERATION_ID: "cupy-measure-objects-basic-v1",
    INTENSITY_OPERATION_ID: "cupy-measure-objects-intensity-basic-v1",
}
MEMORY_MODEL_ID = "cupy-basic-measurements-memory-v1"
PARITY_POLICY_ID = "basic-measurement-table-v1"
GENERATOR_ID = "numpy-pcg64-sparse-object-measurements-v1"
BENCHMARK_ROUNDS = 5
ADMISSION_REPEATS = 3
PLANE_EXTENTS = (256, 512, 1024, 2048)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs/benchmarks/measurements-cupy-windows-rtx5090.json"
)
HISTORICAL_CUCIM_OUTPUT = (
    PROJECT_ROOT / "docs/benchmarks/measurements-cucim-windows-rtx5090.json"
)
SOURCE_PROVENANCE_PATHS = (
    Path("src/napari_vipp/core/measurements.py"),
    Path("src/napari_vipp/core/operations.py"),
    Path("src/napari_vipp/core/compute_policy.py"),
    Path("src/napari_vipp/core/compute_specs.py"),
    Path("src/napari_vipp/core/gpu/cupy_measurements.py"),
    Path("scripts/benchmark_gpu_measurements.py"),
)

REQUIRED_ADMISSION_COVERAGE = frozenset(
    {
        "operation:morphology",
        "operation:intensity",
        "dimension:2d",
        "dimension:3d",
        "axes:leading-blocks",
        "axes:reordered-spatial",
        "calibration:physical",
        "labels:sparse-ids",
        "labels:repeated-disconnected-id",
        "labels:empty",
        "dtype:intensity-bool",
        "dtype:intensity-uint8",
        "dtype:intensity-uint16",
        "dtype:intensity-float32-finite",
        "table:typed-schema-order-units",
        "repeat:deterministic",
    }
)
REQUIRED_REJECTION_COVERAGE = frozenset(
    {
        "reject:label-dtype",
        "reject:label-negative",
        "reject:label-byte-order",
        "reject:intensity-dtype",
        "reject:intensity-nonfinite",
        "reject:intensity-shape",
        "reject:extended-shape",
        "reject:extended-axes",
        "reject:extended-boundary",
        "reject:extended-ratios",
        "reject:extended-moments",
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
        "environment",
        "packages",
        "source_provenance",
        "operation_contracts",
        "admission",
        "rejections",
        "lifecycle",
        "performance",
    }
)


class EvidenceError(RuntimeError):
    """Raised when evidence is incomplete, stale, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class AdmissionDefinition:
    case_id: str
    canonical_shape: tuple[int, ...]
    spatial_ndim: int
    pattern: str
    intensity_dtype: str | None
    seed: int
    axis_order: tuple[int, ...]
    calibrated: bool
    coverage: tuple[str, ...]

    @property
    def operation_id(self) -> str:
        return (
            INTENSITY_OPERATION_ID
            if self.intensity_dtype is not None
            else MORPHOLOGY_OPERATION_ID
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.canonical_shape[index] for index in self.axis_order)

    @property
    def spatial_mode(self) -> str:
        return "2D YX" if self.spatial_ndim == 2 else "3D ZYX"


@dataclass(frozen=True, slots=True)
class RejectionDefinition:
    case_id: str
    kind: str
    expected_pattern: str
    coverage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerformanceDefinition:
    case_id: str
    label: str
    canonical_shape: tuple[int, ...]
    spatial_ndim: int
    intensity_dtype: str | None
    seed: int
    family: str
    pattern: str = "sparse"

    @property
    def operation_id(self) -> str:
        return (
            INTENSITY_OPERATION_ID
            if self.intensity_dtype is not None
            else MORPHOLOGY_OPERATION_ID
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.canonical_shape

    @property
    def spatial_mode(self) -> str:
        return "2D YX" if self.spatial_ndim == 2 else "3D ZYX"


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
        help="Run the complete evidence matrix or a short development profile.",
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
        help="Validate JSON and Markdown without importing CUDA libraries.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_existing is not None:
        try:
            path = validate_existing(args.validate_existing)
        except (OSError, TypeError, ValueError, EvidenceError) as exc:
            print(f"Measurements evidence is invalid: {exc}", file=sys.stderr)
            return 2
        print(f"Measurements evidence is current: {path}")
        return 0
    if isinstance(args.device_index, bool) or args.device_index < 0:
        print("CUDA device index must be non-negative.", file=sys.stderr)
        return 2

    output = args.output.resolve()
    markdown = (args.markdown or output.with_suffix(".md")).resolve()
    try:
        document = build_evidence(args.profile, args.device_index)
        _atomic_write_artifacts(output, markdown, document)
        validate_existing(output)
    except (OSError, TypeError, ValueError, EvidenceError) as exc:
        print(f"Measurements benchmark failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # concise boundary for optional CUDA failures
        print(
            f"Measurements benchmark failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"Wrote {output}")
    print(f"Wrote {markdown}")
    return 0


def build_evidence(profile: str, device_index: int) -> dict[str, object]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'.")
    source_snapshot = _source_provenance()
    contract_snapshot = _operation_contracts()
    np = _numpy()
    cp = _cupy()
    functions = _operation_functions()
    rounds = BENCHMARK_ROUNDS if profile == "full" else 3

    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - host-specific failure
        raise EvidenceError(f"CUDA runtime probe failed: {exc}") from exc
    if device_index < 0 or device_index >= device_count:
        raise EvidenceError(
            f"CUDA device index {device_index} is unavailable; found {device_count}."
        )

    with cp.cuda.Device(device_index):
        _warm_runtime(cp, functions)
        admission = _run_admission(cp, functions)
        rejections = _run_rejections(cp, functions)
        lifecycle = _run_lifecycle(cp, functions)
        results = []
        for definition in _performance_cases(profile):
            print(f"Timing {definition.case_id} ...", flush=True)
            results.append(_run_performance_case(cp, functions, definition, rounds))
    performance = {
        "status": "pass",
        "rounds": rounds,
        "case_count": len(results),
        "all_memory_estimates_cover_observed": all(
            result["memory"]["estimate_with_uncertainty_covers_observed"]
            for result in results
        ),
        "results": results,
    }
    if not performance["all_memory_estimates_cover_observed"]:
        raise EvidenceError("The production memory estimate did not cover observation.")
    if source_snapshot != _source_provenance():
        raise EvidenceError("Tracked source changed while evidence was collected.")
    if contract_snapshot != _operation_contracts():
        raise EvidenceError("Operation contract changed while evidence was collected.")

    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "scientific-admission-and-machine-local-performance-evidence",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": profile,
        "method": _method_record(profile, rounds),
        "environment": _environment_record(cp, device_index),
        "packages": _package_record(cp, np),
        "source_provenance": source_snapshot,
        "operation_contracts": contract_snapshot,
        "admission": admission,
        "rejections": rejections,
        "lifecycle": lifecycle,
        "performance": performance,
    }
    _validate_document_contract(document)
    return document


def _warm_runtime(cp, functions: Mapping[str, object]) -> None:
    # The provider lazily compiles its CuPy RawKernels and materializes small
    # process-lifetime Euler coefficient arrays for each spatial rank. Initialize
    # both ranks with the ordinary/default allocator before any isolated evidence
    # pool; otherwise first-call caches can look like leaked private execution
    # allocations. Intensity calls also compile both grouped-reduction paths.
    for shape, spatial_mode in (((16, 16), "2D YX"), ((8, 12, 12), "3D ZYX")):
        labels = cp.zeros(shape, dtype=cp.int32)
        labels[tuple(slice(2, min(6, size)) for size in shape)] = 7
        packed = functions["gpu_morphology"](
            labels,
            spatial_mode=spatial_mode,
        )
        intensity = cp.arange(labels.size, dtype=cp.uint16).reshape(shape)
        intensity_packed = functions["gpu_intensity"](
            [labels, intensity],
            spatial_mode=spatial_mode,
        )
        cp.cuda.get_current_stream().synchronize()
        del packed, intensity_packed, intensity, labels
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()


def _admission_cases() -> tuple[AdmissionDefinition, ...]:
    identity2 = (0, 1)
    identity3 = (0, 1, 2)
    identity4 = (0, 1, 2, 3)
    return (
        AdmissionDefinition(
            "morph-2d-sparse",
            (61, 67),
            2,
            "sparse",
            None,
            26_060_101,
            identity2,
            False,
            (
                "operation:morphology",
                "dimension:2d",
                "labels:sparse-ids",
                "table:typed-schema-order-units",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "morph-3d-repeated",
            (15, 29, 31),
            3,
            "repeated",
            None,
            26_060_102,
            identity3,
            False,
            (
                "operation:morphology",
                "dimension:3d",
                "labels:repeated-disconnected-id",
                "table:typed-schema-order-units",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "morph-leading-2d",
            (3, 43, 47),
            2,
            "sparse",
            None,
            26_060_103,
            identity3,
            False,
            (
                "operation:morphology",
                "dimension:2d",
                "axes:leading-blocks",
                "labels:sparse-ids",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "morph-leading-3d",
            (2, 11, 23, 25),
            3,
            "sparse",
            None,
            26_060_104,
            identity4,
            False,
            (
                "operation:morphology",
                "dimension:3d",
                "axes:leading-blocks",
                "labels:sparse-ids",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "morph-empty-2d",
            (31, 37),
            2,
            "empty",
            None,
            26_060_105,
            identity2,
            False,
            (
                "operation:morphology",
                "dimension:2d",
                "labels:empty",
                "table:typed-schema-order-units",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "morph-reordered-calibrated",
            (2, 41, 43),
            2,
            "repeated",
            None,
            26_060_106,
            (1, 0, 2),
            True,
            (
                "operation:morphology",
                "dimension:2d",
                "axes:leading-blocks",
                "axes:reordered-spatial",
                "calibration:physical",
                "labels:repeated-disconnected-id",
                "table:typed-schema-order-units",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "intensity-bool-2d",
            (59, 63),
            2,
            "repeated",
            "bool",
            26_060_201,
            identity2,
            False,
            (
                "operation:intensity",
                "dimension:2d",
                "dtype:intensity-bool",
                "labels:repeated-disconnected-id",
                "table:typed-schema-order-units",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "intensity-uint8-3d",
            (13, 27, 29),
            3,
            "sparse",
            "uint8",
            26_060_202,
            identity3,
            False,
            (
                "operation:intensity",
                "dimension:3d",
                "dtype:intensity-uint8",
                "labels:sparse-ids",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "intensity-uint16-leading-2d",
            (3, 45, 49),
            2,
            "sparse",
            "uint16",
            26_060_203,
            identity3,
            False,
            (
                "operation:intensity",
                "dimension:2d",
                "axes:leading-blocks",
                "dtype:intensity-uint16",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "intensity-float32-reordered-calibrated",
            (2, 12, 31, 33),
            3,
            "repeated",
            "float32",
            26_060_204,
            (1, 0, 2, 3),
            True,
            (
                "operation:intensity",
                "dimension:3d",
                "axes:leading-blocks",
                "axes:reordered-spatial",
                "calibration:physical",
                "dtype:intensity-float32-finite",
                "labels:repeated-disconnected-id",
                "table:typed-schema-order-units",
                "repeat:deterministic",
            ),
        ),
        AdmissionDefinition(
            "intensity-empty-2d",
            (33, 35),
            2,
            "empty",
            "uint8",
            26_060_205,
            identity2,
            False,
            (
                "operation:intensity",
                "dimension:2d",
                "labels:empty",
                "dtype:intensity-uint8",
                "table:typed-schema-order-units",
                "repeat:deterministic",
            ),
        ),
    )


def _rejection_cases() -> tuple[RejectionDefinition, ...]:
    return (
        RejectionDefinition(
            "reject-label-uint16",
            "label-uint16",
            r"native int32 labels",
            ("reject:label-dtype",),
        ),
        RejectionDefinition(
            "reject-label-negative",
            "label-negative",
            r"non-negative label IDs",
            ("reject:label-negative",),
        ),
        RejectionDefinition(
            "reject-label-byte-order",
            "label-byte-order",
            r"native int32 labels",
            ("reject:label-byte-order",),
        ),
        RejectionDefinition(
            "reject-intensity-float64",
            "intensity-float64",
            r"native bool, uint8, uint16, and finite float32",
            ("reject:intensity-dtype",),
        ),
        RejectionDefinition(
            "reject-intensity-nonfinite",
            "intensity-nonfinite",
            r"finite data",
            ("reject:intensity-nonfinite",),
        ),
        RejectionDefinition(
            "reject-intensity-shape",
            "intensity-shape",
            r"same shape",
            ("reject:intensity-shape",),
        ),
        RejectionDefinition(
            "reject-extended-shape",
            "extended-shape",
            r"extended measurement options",
            ("reject:extended-shape",),
        ),
        RejectionDefinition(
            "reject-extended-axes",
            "extended-axes",
            r"extended measurement options",
            ("reject:extended-axes",),
        ),
        RejectionDefinition(
            "reject-extended-boundary",
            "extended-boundary",
            r"extended measurement options",
            ("reject:extended-boundary",),
        ),
        RejectionDefinition(
            "reject-extended-ratios",
            "extended-ratios",
            r"extended measurement options",
            ("reject:extended-ratios",),
        ),
        RejectionDefinition(
            "reject-extended-moments",
            "extended-moments",
            r"extended measurement options",
            ("reject:extended-moments",),
        ),
    )


def _performance_cases(profile: str) -> tuple[PerformanceDefinition, ...]:
    extents = PLANE_EXTENTS if profile == "full" else PLANE_EXTENTS[:2]
    definitions: list[PerformanceDefinition] = []
    for extent in extents:
        definitions.extend(
            (
                PerformanceDefinition(
                    f"plane-{extent}-morphology",
                    f"{extent}² morphology",
                    (extent, extent),
                    2,
                    None,
                    50_900 + extent,
                    "plane-morphology",
                ),
                PerformanceDefinition(
                    f"plane-{extent}-intensity-uint16",
                    f"{extent}² morphology + uint16 intensity",
                    (extent, extent),
                    2,
                    "uint16",
                    51_900 + extent,
                    "plane-intensity-uint16",
                ),
            )
        )
    representative = (
        PerformanceDefinition(
            "volume-32x256x256-morphology",
            "32×256×256 morphology",
            (32, 256, 256),
            3,
            None,
            52_001,
            "volume-morphology",
        ),
        PerformanceDefinition(
            "volume-32x256x256-intensity-uint16",
            "32×256×256 morphology + uint16 intensity",
            (32, 256, 256),
            3,
            "uint16",
            52_002,
            "volume-intensity-uint16",
        ),
        PerformanceDefinition(
            "stack-8x512x512-morphology",
            "8×512² plane-wise morphology",
            (8, 512, 512),
            2,
            None,
            52_003,
            "stack-morphology",
        ),
        PerformanceDefinition(
            "stack-6x512x512-intensity-float32",
            "6×512² plane-wise morphology + float32 intensity",
            (6, 512, 512),
            2,
            "float32",
            52_004,
            "stack-intensity-float32",
        ),
        PerformanceDefinition(
            "confocal-volume-64x512x512-intensity-uint16",
            "64×512×512 confocal-like volume + uint16 intensity",
            (64, 512, 512),
            3,
            "uint16",
            52_005,
            "confocal-volume-intensity-uint16",
        ),
        PerformanceDefinition(
            "confocal-stack-16x1024x1024-morphology",
            "16×1024² confocal-like plane-wise morphology",
            (16, 1024, 1024),
            2,
            None,
            52_006,
            "confocal-stack-morphology",
        ),
        PerformanceDefinition(
            "many-objects-256x256-intensity-uint16",
            "256² morphology + uint16 intensity, 1,024 objects",
            (256, 256),
            2,
            "uint16",
            52_007,
            "many-object-intensity-uint16",
            "many-objects",
        ),
    )
    if profile == "quick":
        representative = (representative[0], representative[2])
    definitions.extend(representative)
    return tuple(definitions)


def _make_inputs(definition: AdmissionDefinition | PerformanceDefinition):
    canonical_labels = _make_labels(
        definition.canonical_shape,
        definition.spatial_ndim,
        definition.pattern,
        definition.seed,
    )
    order = (
        definition.axis_order
        if isinstance(definition, AdmissionDefinition)
        else tuple(range(len(definition.canonical_shape)))
    )
    labels = _readonly_contiguous(_numpy().transpose(canonical_labels, order), "int32")
    if definition.intensity_dtype is None:
        return labels, None
    canonical_intensity = _make_intensity(
        definition.canonical_shape,
        definition.intensity_dtype,
        definition.seed + 1_000_003,
    )
    intensity = _readonly_contiguous(
        _numpy().transpose(canonical_intensity, order),
        definition.intensity_dtype,
    )
    return labels, intensity


def _make_labels(
    canonical_shape: tuple[int, ...],
    spatial_ndim: int,
    pattern: str,
    seed: int,
):
    np = _numpy()
    labels = np.zeros(canonical_shape, dtype=np.int32)
    leading_shape = canonical_shape[:-spatial_ndim]
    indexes = np.ndindex(leading_shape) if leading_shape else ((),)
    for block_number, leading_index in enumerate(indexes):
        block = labels[leading_index] if leading_shape else labels
        if pattern == "empty":
            continue
        rng = np.random.default_rng(seed + block_number * 10_007)
        spatial_elements = math.prod(block.shape)
        if pattern == "many-objects":
            object_count = min(1_024, spatial_elements // 3)
            flattened = block.reshape(-1)
            flattened[: object_count * 3] = np.repeat(
                np.arange(1, object_count + 1, dtype=np.int32),
                3,
            )
            continue
        object_count = min(96, max(4, spatial_elements // 16_384))
        if spatial_elements < 16_384:
            object_count = min(12, max(4, spatial_elements // 384))
        for index in range(object_count):
            extents = tuple(max(1, min(6, size // 12)) for size in block.shape)
            starts = tuple(
                int(rng.integers(0, max(size - extent + 1, 1)))
                for size, extent in zip(block.shape, extents, strict=True)
            )
            slices = tuple(
                slice(start, min(start + extent, size))
                for start, extent, size in zip(
                    starts,
                    extents,
                    block.shape,
                    strict=True,
                )
            )
            block[slices] = np.int32(1 + index * 17 + block_number * 100_003)
        if pattern == "repeated":
            repeated_id = np.int32(700_001 + block_number * 2)
            first = tuple(slice(1, min(3, size)) for size in block.shape)
            second = tuple(
                slice(max(size - 3, 0), max(size - 1, 1)) for size in block.shape
            )
            block[first] = repeated_id
            block[second] = repeated_id
        elif pattern != "sparse":
            raise ValueError(f"Unknown label pattern {pattern!r}.")
    labels = np.ascontiguousarray(labels, dtype=np.int32)
    labels.setflags(write=False)
    return labels


def _make_intensity(shape: tuple[int, ...], dtype: str, seed: int):
    np = _numpy()
    rng = np.random.default_rng(seed)
    normalized = np.dtype(dtype)
    if normalized == np.dtype(bool):
        values = rng.random(shape) > 0.42
    elif normalized == np.dtype(np.uint8):
        values = rng.integers(0, 256, shape, dtype=np.uint8)
    elif normalized == np.dtype(np.uint16):
        values = rng.integers(0, 65_536, shape, dtype=np.uint16)
    elif normalized == np.dtype(np.float32):
        values = rng.normal(140.0, 35.0, shape).astype(np.float32)
        values += np.linspace(-0.25, 0.25, values.size, dtype=np.float32).reshape(shape)
    else:
        raise ValueError(f"Unsupported evidence intensity dtype {dtype!r}.")
    values = np.ascontiguousarray(values, dtype=normalized)
    values.setflags(write=False)
    return values


def _readonly_contiguous(data, dtype):
    result = _numpy().ascontiguousarray(data, dtype=dtype)
    result.setflags(write=False)
    return result


def _parameters(
    definition: AdmissionDefinition | PerformanceDefinition,
) -> dict[str, object]:
    ndim = len(definition.shape)
    spatial_ndim = definition.spatial_ndim
    leading_ndim = ndim - spatial_ndim
    canonical_names = tuple(f"series_{index}" for index in range(leading_ndim)) + (
        ("y", "x") if spatial_ndim == 2 else ("z", "y", "x")
    )
    canonical_types = tuple("unknown" for _ in range(leading_ndim)) + tuple(
        "space" for _ in range(spatial_ndim)
    )
    calibrated = isinstance(definition, AdmissionDefinition) and definition.calibrated
    if calibrated:
        canonical_scales = tuple(None for _ in range(leading_ndim)) + (
            (0.31, 0.19) if spatial_ndim == 2 else (1.25, 0.31, 0.19)
        )
        canonical_units = tuple(None for _ in range(leading_ndim)) + tuple(
            "um" for _ in range(spatial_ndim)
        )
    else:
        canonical_scales = tuple(None for _ in range(ndim))
        canonical_units = tuple(None for _ in range(ndim))
    order = (
        definition.axis_order
        if isinstance(definition, AdmissionDefinition)
        else tuple(range(ndim))
    )
    operation_id = definition.operation_id
    return {
        "spatial_mode": definition.spatial_mode,
        "measurement_set": (
            "Basic morphology + intensity"
            if operation_id == INTENSITY_OPERATION_ID
            else "Basic morphology"
        ),
        "include_shape_descriptors": False,
        "include_axis_descriptors": False,
        "include_2d_boundary_descriptors": False,
        "include_derived_shape_ratios": False,
        "include_2d_shape_moments": False,
        "axis_names": tuple(canonical_names[index] for index in order),
        "axis_types": tuple(canonical_types[index] for index in order),
        "axis_scales": tuple(canonical_scales[index] for index in order),
        "axis_units": tuple(canonical_units[index] for index in order),
        "source_name": "deterministic GPU measurements evidence",
    }


def _run_admission(cp, functions: Mapping[str, object]) -> dict[str, object]:
    cases = []
    coverage: set[str] = set()
    for definition in _admission_cases():
        labels, intensity = _make_inputs(definition)
        parameters = _parameters(definition)
        cpu_table = _cpu_call(
            functions, definition.operation_id, labels, intensity, parameters
        )
        pool = cp.cuda.MemoryPool()
        hashes: list[str] = []
        parity_details: list[str] = []
        output_contiguous = True
        input_immutable = True
        packed_shapes: list[list[int]] = []
        with cp.cuda.using_allocator(pool.malloc):
            device_labels = cp.asarray(labels)
            device_intensity = cp.asarray(intensity) if intensity is not None else None
            labels_before = device_labels.copy()
            intensity_before = (
                device_intensity.copy() if device_intensity is not None else None
            )
            outputs = []
            for _ in range(ADMISSION_REPEATS):
                packed = _gpu_call(
                    functions,
                    definition.operation_id,
                    device_labels,
                    device_intensity,
                    parameters,
                )
                cp.cuda.get_current_stream().synchronize()
                if not isinstance(packed, cp.ndarray) or packed.dtype != cp.float64:
                    raise EvidenceError(
                        f"{definition.case_id} did not return resident float64 rows."
                    )
                output_contiguous &= bool(packed.flags.c_contiguous)
                packed_shapes.append([int(size) for size in packed.shape])
                table = _finalize_packed(
                    functions,
                    definition.operation_id,
                    cp.asnumpy(packed),
                    labels.shape,
                    parameters,
                )
                parity = functions["parity"](
                    cpu_table,
                    table,
                    intensity_dtype=definition.intensity_dtype,
                )
                parity_details.append(str(parity.detail))
                if not parity.passed:
                    raise EvidenceError(
                        f"Admission parity failed for {definition.case_id}: "
                        f"{parity.detail}"
                    )
                hashes.append(_table_sha256(table))
                outputs.append(packed)
            input_immutable = bool(cp.array_equal(device_labels, labels_before).item())
            if device_intensity is not None and intensity_before is not None:
                input_immutable &= bool(
                    cp.array_equal(device_intensity, intensity_before).item()
                )
            del packed, table, outputs, labels_before, device_labels
            if device_intensity is not None:
                del device_intensity
            if intensity_before is not None:
                del intensity_before
            cp.cuda.get_current_stream().synchronize()
        cleanup = _drain_pool(cp, pool)
        deterministic = len(set(hashes)) == 1
        if not deterministic or not output_contiguous or not input_immutable:
            raise EvidenceError(f"Admission lifecycle failed for {definition.case_id}.")
        if definition.pattern == "empty" and cpu_table.row_count != 0:
            raise EvidenceError(f"Empty-label case {definition.case_id} emitted rows.")
        coverage.update(definition.coverage)
        cases.append(
            {
                "case_id": definition.case_id,
                "operation_id": definition.operation_id,
                "implementation_id": IMPLEMENTATION_IDS[definition.operation_id],
                "shape": list(definition.shape),
                "canonical_shape": list(definition.canonical_shape),
                "spatial_ndim": definition.spatial_ndim,
                "spatial_mode": definition.spatial_mode,
                "pattern": definition.pattern,
                "intensity_dtype": definition.intensity_dtype,
                "seed": definition.seed,
                "coverage": sorted(definition.coverage),
                "parameters": _json_value(parameters),
                "label_ids": _positive_label_ids(labels),
                "row_count": cpu_table.row_count,
                "column_count": cpu_table.column_count,
                "columns": list(cpu_table.columns),
                "column_units": [list(item) for item in cpu_table.column_units],
                "packed_shapes": packed_shapes,
                "parity": {
                    "policy_id": PARITY_POLICY_ID,
                    "passed": True,
                    "details": parity_details,
                    "cpu_table_sha256": _table_sha256(cpu_table),
                    "gpu_table_sha256": hashes[0],
                },
                "resident_output_dtype": "float64",
                "resident_output_contiguous": output_contiguous,
                "input_immutable": input_immutable,
                "repeat_count": ADMISSION_REPEATS,
                "repeat_deterministic": deterministic,
                "gpu_repeat_table_sha256": hashes,
                "label_input_sha256": _array_sha256(labels),
                "intensity_input_sha256": (
                    _array_sha256(intensity) if intensity is not None else None
                ),
                "cleanup": cleanup,
            }
        )
    if not REQUIRED_ADMISSION_COVERAGE <= coverage:
        missing = sorted(REQUIRED_ADMISSION_COVERAGE - coverage)
        raise EvidenceError(f"Admission coverage is incomplete: {missing}.")
    return {
        "status": "pass",
        "case_count": len(cases),
        "repeat_count": ADMISSION_REPEATS,
        "parity_policy_id": PARITY_POLICY_ID,
        "coverage": sorted(coverage),
        "cases": cases,
    }


def _run_rejections(cp, functions: Mapping[str, object]) -> dict[str, object]:
    np = _numpy()
    cases = []
    coverage: set[str] = set()
    for definition in _rejection_cases():
        labels = np.zeros((21, 23), dtype=np.int32)
        labels[2:5, 3:7] = 7
        intensity = np.ones(labels.shape, dtype=np.uint16)
        parameters = {
            "spatial_mode": "2D YX",
            "include_shape_descriptors": False,
            "include_axis_descriptors": False,
            "include_2d_boundary_descriptors": False,
            "include_derived_shape_ratios": False,
            "include_2d_shape_moments": False,
        }
        operation_id = MORPHOLOGY_OPERATION_ID
        if definition.kind == "label-uint16":
            labels = labels.astype(np.uint16)
        elif definition.kind == "label-negative":
            labels[0, 0] = -1
        elif definition.kind == "label-byte-order":
            labels = labels.byteswap().view(labels.dtype.newbyteorder())
        elif definition.kind == "intensity-float64":
            operation_id = INTENSITY_OPERATION_ID
            intensity = intensity.astype(np.float64)
        elif definition.kind == "intensity-nonfinite":
            operation_id = INTENSITY_OPERATION_ID
            intensity = intensity.astype(np.float32)
            intensity[0, 0] = np.nan
        elif definition.kind == "intensity-shape":
            operation_id = INTENSITY_OPERATION_ID
            intensity = intensity[:-1]
        elif definition.kind == "extended-shape":
            parameters["include_shape_descriptors"] = True
        elif definition.kind == "extended-axes":
            parameters["include_axis_descriptors"] = True
        elif definition.kind == "extended-boundary":
            parameters["include_2d_boundary_descriptors"] = True
        elif definition.kind == "extended-ratios":
            parameters["include_derived_shape_ratios"] = True
        elif definition.kind == "extended-moments":
            parameters["include_2d_shape_moments"] = True
        else:  # pragma: no cover - private manifest guard
            raise RuntimeError(definition.kind)
        pool = cp.cuda.MemoryPool()
        message = ""
        rejected = False
        with cp.cuda.using_allocator(pool.malloc):
            try:
                _gpu_call(
                    functions,
                    operation_id,
                    labels,
                    intensity if operation_id == INTENSITY_OPERATION_ID else None,
                    parameters,
                )
                cp.cuda.get_current_stream().synchronize()
            except (TypeError, ValueError) as exc:
                rejected = True
                message = str(exc)
                exc.__traceback__ = None
        cleanup = _drain_pool(cp, pool)
        if not rejected or re.search(definition.expected_pattern, message) is None:
            raise EvidenceError(
                f"Unexpected rejection for {definition.case_id}: {message!r}."
            )
        coverage.update(definition.coverage)
        cases.append(
            {
                "case_id": definition.case_id,
                "operation_id": operation_id,
                "coverage": sorted(definition.coverage),
                "expected_pattern": definition.expected_pattern,
                "rejected": True,
                "message": message,
                "cleanup": cleanup,
            }
        )
    if not REQUIRED_REJECTION_COVERAGE <= coverage:
        missing = sorted(REQUIRED_REJECTION_COVERAGE - coverage)
        raise EvidenceError(f"Rejection coverage is incomplete: {missing}.")
    return {
        "status": "pass",
        "case_count": len(cases),
        "coverage": sorted(coverage),
        "cases": cases,
    }


def _run_lifecycle(cp, functions: Mapping[str, object]) -> dict[str, object]:
    from napari_vipp.core.progress import OperationCancelled, ProgressContext

    results = []
    for operation_id, intensity_dtype in (
        (MORPHOLOGY_OPERATION_ID, None),
        (INTENSITY_OPERATION_ID, "uint16"),
    ):
        definition = AdmissionDefinition(
            f"lifecycle-{operation_id}",
            (3, 192, 192),
            2,
            "sparse",
            intensity_dtype,
            26_061_000 + len(results),
            (0, 1, 2),
            False,
            (),
        )
        labels, intensity = _make_inputs(definition)
        parameters = _parameters(definition)
        pool = cp.cuda.MemoryPool()
        cancellation_state = {"cancelled": False}
        updates: list[dict[str, object]] = []

        def reporter(
            update,
            *,
            _updates=updates,
            _state=cancellation_state,
        ) -> None:
            _updates.append(
                {
                    "current": int(update.current),
                    "total": int(update.total),
                    "message": str(update.message),
                }
            )
            if int(update.current) == 1:
                _state["cancelled"] = True

        cancellation_observed = False
        with cp.cuda.using_allocator(pool.malloc):
            device_labels = cp.asarray(labels)
            device_intensity = cp.asarray(intensity) if intensity is not None else None
            progress = ProgressContext(
                cancelled=lambda _state=cancellation_state: _state["cancelled"],
                reporter=reporter,
            )
            try:
                _gpu_call(
                    functions,
                    operation_id,
                    device_labels,
                    device_intensity,
                    parameters,
                    progress=progress,
                )
            except OperationCancelled as exc:
                cancellation_observed = True
                exc.__traceback__ = None
            if not cancellation_observed:
                raise EvidenceError(
                    f"Cancellation was not observed for {operation_id}."
                )
            cancellation_state["cancelled"] = False
            complete_updates: list[dict[str, object]] = []
            complete_progress = ProgressContext(
                reporter=lambda update, _updates=complete_updates: _updates.append(
                    {
                        "current": int(update.current),
                        "total": int(update.total),
                        "message": str(update.message),
                    }
                )
            )
            packed = _gpu_call(
                functions,
                operation_id,
                device_labels,
                device_intensity,
                parameters,
                progress=complete_progress,
            )
            cp.cuda.get_current_stream().synchronize()
            table = _finalize_packed(
                functions,
                operation_id,
                cp.asnumpy(packed),
                labels.shape,
                parameters,
            )
            expected = _cpu_call(
                functions,
                operation_id,
                labels,
                intensity,
                parameters,
            )
            parity = functions["parity"](
                expected,
                table,
                intensity_dtype=intensity_dtype,
            )
            if not parity.passed:
                raise EvidenceError(
                    "Post-cancellation reuse failed for "
                    f"{operation_id}: {parity.detail}"
                )
            del packed, device_labels
            if device_intensity is not None:
                del device_intensity
            cp.cuda.get_current_stream().synchronize()
        cleanup = _drain_pool(cp, pool)
        stages_per_block = 6 if intensity_dtype is not None else 4
        expected_total = 3 * stages_per_block + 1
        expected_cancel_currents = [0, 1]
        if [update["current"] for update in updates] != expected_cancel_currents:
            raise EvidenceError(
                f"Cancellation progress was not staged for {operation_id}."
            )
        if [update["current"] for update in complete_updates] != list(
            range(expected_total + 1)
        ) or {update["total"] for update in complete_updates} != {expected_total}:
            raise EvidenceError(
                f"Completion progress was incomplete for {operation_id}."
            )
        messages = "\n".join(str(update["message"]) for update in complete_updates)
        required_messages = (
            "preparing",
            "compacting labels",
            "measuring morphology",
            "measuring topology",
            "packing rows",
            "assembling packed table",
        )
        if intensity_dtype is not None:
            required_messages += (
                "measuring intensity ranges and means",
                "measuring intensity variation",
            )
        if any(message not in messages for message in required_messages):
            raise EvidenceError(
                f"Progress stage messages were incomplete for {operation_id}."
            )
        results.append(
            {
                "operation_id": operation_id,
                "implementation_id": IMPLEMENTATION_IDS[operation_id],
                "cancellation_requested": True,
                "cancellation_observed": True,
                "cancel_after_completed_stage": 1,
                "cancel_updates": updates,
                "complete_updates": complete_updates,
                "expected_total": expected_total,
                "post_cancellation_reuse_parity": True,
                "parity_detail": parity.detail,
                "cleanup": cleanup,
            }
        )
    return {
        "status": "pass",
        "boundary": "synchronized-measurement-block-stage-v1",
        "case_count": len(results),
        "cases": results,
    }


def _run_performance_case(
    cp,
    functions: Mapping[str, object],
    definition: PerformanceDefinition,
    rounds: int,
) -> dict[str, object]:
    labels, intensity = _make_inputs(definition)
    parameters = _parameters(definition)
    started = time.perf_counter()
    cpu_reference = _cpu_call(
        functions,
        definition.operation_id,
        labels,
        intensity,
        parameters,
    )
    cpu_case_cold = time.perf_counter() - started
    cpu_seconds = []
    for _ in range(rounds):
        started = time.perf_counter()
        output = _cpu_call(
            functions,
            definition.operation_id,
            labels,
            intensity,
            parameters,
        )
        cpu_seconds.append(time.perf_counter() - started)
        del output

    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        started = time.perf_counter()
        cold_labels = cp.asarray(labels)
        cold_intensity = cp.asarray(intensity) if intensity is not None else None
        cold_packed = _gpu_call(
            functions,
            definition.operation_id,
            cold_labels,
            cold_intensity,
            parameters,
        )
        cold_host = cp.asnumpy(cold_packed)
        cold_table = _finalize_packed(
            functions,
            definition.operation_id,
            cold_host,
            labels.shape,
            parameters,
        )
        cp.cuda.get_current_stream().synchronize()
        gpu_case_cold = time.perf_counter() - started
        del cold_packed, cold_host, cold_labels
        if cold_intensity is not None:
            del cold_intensity

        resident_labels = cp.asarray(labels)
        resident_intensity = cp.asarray(intensity) if intensity is not None else None
        warm = _gpu_call(
            functions,
            definition.operation_id,
            resident_labels,
            resident_intensity,
            parameters,
        )
        cp.cuda.get_current_stream().synchronize()
        del warm

        gpu_resident_compute_seconds = []
        for _ in range(rounds):
            started = time.perf_counter()
            packed = _gpu_call(
                functions,
                definition.operation_id,
                resident_labels,
                resident_intensity,
                parameters,
            )
            cp.cuda.get_current_stream().synchronize()
            gpu_resident_compute_seconds.append(time.perf_counter() - started)
            del packed

        gpu_resident_public_seconds = []
        for _ in range(rounds):
            started = time.perf_counter()
            packed = _gpu_call(
                functions,
                definition.operation_id,
                resident_labels,
                resident_intensity,
                parameters,
            )
            host = cp.asnumpy(packed)
            table = _finalize_packed(
                functions,
                definition.operation_id,
                host,
                labels.shape,
                parameters,
            )
            cp.cuda.get_current_stream().synchronize()
            gpu_resident_public_seconds.append(time.perf_counter() - started)
            del table, host, packed

        gpu_transfer_inclusive_seconds = []
        for _ in range(rounds):
            started = time.perf_counter()
            transfer_labels = cp.asarray(labels)
            transfer_intensity = (
                cp.asarray(intensity) if intensity is not None else None
            )
            packed = _gpu_call(
                functions,
                definition.operation_id,
                transfer_labels,
                transfer_intensity,
                parameters,
            )
            host = cp.asnumpy(packed)
            table = _finalize_packed(
                functions,
                definition.operation_id,
                host,
                labels.shape,
                parameters,
            )
            cp.cuda.get_current_stream().synchronize()
            gpu_transfer_inclusive_seconds.append(time.perf_counter() - started)
            del table, host, packed, transfer_labels
            if transfer_intensity is not None:
                del transfer_intensity
        del resident_labels
        if resident_intensity is not None:
            del resident_intensity
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)

    parity = functions["parity"](
        cpu_reference,
        cold_table,
        intensity_dtype=definition.intensity_dtype,
    )
    if not parity.passed:
        raise EvidenceError(
            f"Performance parity failed for {definition.case_id}: {parity.detail}"
        )
    memory = _measure_memory(cp, functions, definition, labels, intensity, parameters)
    return {
        "case_id": definition.case_id,
        "label": definition.label,
        "family": definition.family,
        "operation_id": definition.operation_id,
        "implementation_id": IMPLEMENTATION_IDS[definition.operation_id],
        "shape": list(definition.shape),
        "spatial_ndim": definition.spatial_ndim,
        "spatial_mode": definition.spatial_mode,
        "intensity_dtype": definition.intensity_dtype,
        "seed": definition.seed,
        "element_count": int(labels.size),
        "input_bytes": int(
            labels.nbytes + (intensity.nbytes if intensity is not None else 0)
        ),
        "object_row_count": cpu_reference.row_count,
        "label_input_sha256": _array_sha256(labels),
        "intensity_input_sha256": _array_sha256(intensity)
        if intensity is not None
        else None,
        "parity": {
            "policy_id": PARITY_POLICY_ID,
            "passed": True,
            "detail": parity.detail,
            "cpu_table_sha256": _table_sha256(cpu_reference),
            "gpu_table_sha256": _table_sha256(cold_table),
        },
        "samples": {
            "cpu_seconds": cpu_seconds,
            "gpu_resident_compute_seconds": gpu_resident_compute_seconds,
            "gpu_resident_public_seconds": gpu_resident_public_seconds,
            "gpu_transfer_inclusive_seconds": gpu_transfer_inclusive_seconds,
            "cpu_case_cold_seconds": cpu_case_cold,
            "gpu_case_cold_transfer_inclusive_seconds": gpu_case_cold,
        },
        "summary": _timing_summary(
            cpu_seconds,
            gpu_resident_compute_seconds,
            gpu_resident_public_seconds,
            gpu_transfer_inclusive_seconds,
            cpu_case_cold,
            gpu_case_cold,
        ),
        "memory": memory,
        "cleanup": cleanup,
    }


def _measure_memory(
    cp,
    functions: Mapping[str, object],
    definition: PerformanceDefinition,
    labels,
    intensity,
    parameters: Mapping[str, object],
) -> dict[str, object]:
    pool = cp.cuda.MemoryPool()
    packed_bytes = 0
    with cp.cuda.using_allocator(pool.malloc):
        device_labels = cp.asarray(labels)
        device_intensity = cp.asarray(intensity) if intensity is not None else None
        used_with_inputs = int(pool.used_bytes())
        packed = _gpu_call(
            functions,
            definition.operation_id,
            device_labels,
            device_intensity,
            parameters,
        )
        cp.cuda.get_current_stream().synchronize()
        packed_bytes = int(packed.nbytes)
        used_with_output = int(pool.used_bytes())
        observed_reserved = int(pool.total_bytes())
        host = cp.asnumpy(packed)
        table = _finalize_packed(
            functions,
            definition.operation_id,
            host,
            labels.shape,
            parameters,
        )
        del table, host, packed, device_labels
        if device_intensity is not None:
            del device_intensity
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    estimate = _estimated_memory(
        labels.shape,
        definition.spatial_ndim,
        int(labels.nbytes + (intensity.nbytes if intensity is not None else 0)),
        include_intensity=intensity is not None,
    )
    covered = estimate["device_peak_with_uncertainty_bytes"] >= observed_reserved
    return {
        "scope": "isolated-cupy-private-pool-reserved-high-water-v1",
        "model_id": MEMORY_MODEL_ID,
        **estimate,
        "actual_packed_output_bytes": packed_bytes,
        "observed_reserved_bytes": observed_reserved,
        "observed_used_bytes_with_inputs": used_with_inputs,
        "observed_used_bytes_with_output": used_with_output,
        "observed_to_estimated_ratio": (
            observed_reserved / estimate["device_peak_with_uncertainty_bytes"]
        ),
        "estimate_with_uncertainty_covers_observed": covered,
        **cleanup,
    }


def _estimated_memory(
    shape: Sequence[int],
    spatial_ndim: int,
    input_bytes: int,
    *,
    include_intensity: bool,
) -> dict[str, int]:
    element_count = math.prod(int(size) for size in shape)
    spatial_elements = math.prod(int(size) for size in shape[-spatial_ndim:])
    leading_ndim = len(tuple(shape)) - spatial_ndim
    packed_width = leading_ndim + 3 + 3 * spatial_ndim + (5 if include_intensity else 0)
    output_upper = element_count * packed_width * 8
    working_copies = int(input_bytes)
    per_block_workspace = spatial_elements * (224 if include_intensity else 128)
    managed_peak = (
        int(input_bytes)
        + output_upper
        + working_copies
        + per_block_workspace
        + output_upper
    )
    uncertainty = max(64 * 1024**2, managed_peak // 4)
    return {
        "input_bytes": int(input_bytes),
        "packed_output_upper_bound_bytes": output_upper,
        "working_copy_bytes": working_copies,
        "active_block_workspace_bytes": per_block_workspace,
        "runtime_managed_peak_bytes": managed_peak,
        "uncertainty_bytes": uncertainty,
        "device_peak_with_uncertainty_bytes": managed_peak + uncertainty,
        "host_materialization_peak_bytes": output_upper * 5,
    }


def _timing_summary(
    cpu_seconds: Sequence[float],
    resident_compute: Sequence[float],
    resident_public: Sequence[float],
    transfer_inclusive: Sequence[float],
    cpu_cold: float,
    gpu_cold: float,
) -> dict[str, object]:
    cpu_median = statistics.median(cpu_seconds)
    compute_median = statistics.median(resident_compute)
    public_median = statistics.median(resident_public)
    transfer_median = statistics.median(transfer_inclusive)
    return {
        "cpu_median_seconds": cpu_median,
        "gpu_resident_compute_median_seconds": compute_median,
        "gpu_resident_public_median_seconds": public_median,
        "gpu_transfer_inclusive_median_seconds": transfer_median,
        "cpu_case_cold_seconds": cpu_cold,
        "gpu_case_cold_transfer_inclusive_seconds": gpu_cold,
        "gpu_resident_compute_speedup": cpu_median / compute_median,
        "gpu_resident_public_speedup": cpu_median / public_median,
        "gpu_transfer_inclusive_speedup": cpu_median / transfer_median,
        "mandatory_table_boundary_seconds": max(public_median - compute_median, 0.0),
        "input_transfer_and_allocation_seconds": max(
            transfer_median - public_median, 0.0
        ),
        "screening_choice": "GPU-CuPy" if transfer_median < cpu_median else "CPU",
    }


def _cpu_call(functions, operation_id, labels, intensity, parameters):
    if operation_id == MORPHOLOGY_OPERATION_ID:
        return functions["cpu_morphology"](labels, **parameters)
    return functions["cpu_intensity"]([labels, intensity], **parameters)


def _gpu_call(
    functions,
    operation_id,
    labels,
    intensity,
    parameters,
    *,
    progress=None,
):
    kwargs = dict(parameters)
    if progress is not None:
        kwargs["progress"] = progress
    if operation_id == MORPHOLOGY_OPERATION_ID:
        return functions["gpu_morphology"](labels, **kwargs)
    return functions["gpu_intensity"]([labels, intensity], **kwargs)


def _finalize_packed(functions, operation_id, packed, input_shape, parameters):
    call = SimpleNamespace(
        operation_id=operation_id,
        kwargs=dict(parameters),
        input_states=(SimpleNamespace(shape=tuple(input_shape)),),
        inputs=(None,),
    )
    return functions["finalizer"]((packed,), call=call)


def _positive_label_ids(labels) -> list[int]:
    values = _numpy().unique(labels)
    return [int(value) for value in values if int(value) > 0]


def _source_provenance() -> list[dict[str, str]]:
    records = []
    for relative_path in SOURCE_PROVENANCE_PATHS:
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            raise EvidenceError(f"Required source file is missing: {relative_path}.")
        records.append(
            {
                "path": relative_path.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return records


def _operation_contracts() -> dict[str, object]:
    snapshots = []
    for operation_id, include_intensity in (
        (MORPHOLOGY_OPERATION_ID, False),
        (INTENSITY_OPERATION_ID, True),
    ):
        snapshot = {
            "operation_id": operation_id,
            "implementation_id": IMPLEMENTATION_IDS[operation_id],
            "runtime_id": "cuda-cupy",
            "implementation_library_id": "cupy",
            "provider_version": "implementation-v1",
            "parameter_policy_id": "basic-measurements-parameters-v1",
            "parity_policy_id": PARITY_POLICY_ID,
            "memory_model_id": MEMORY_MODEL_ID,
            "host_finalizer_ref": (
                "napari_vipp.core.measurements:finalize_basic_measurement_outputs"
            ),
            "resident_payload": "c-contiguous-float64-packed-rows-v1",
            "spatial_ndims": [2, 3],
            "label_region": "native-nonnegative-int32",
            "intensity_region": (
                ["bool", "uint8", "uint16", "finite-float32"]
                if include_intensity
                else []
            ),
            "extended_measurement_options": "all-disabled",
            "leading_block_order": "c-order",
            "object_order": "ascending-positive-label-id-per-block",
            "table_boundary": "mandatory-typed-host-finalizer",
        }
        encoded = json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        snapshots.append(
            {"snapshot": snapshot, "sha256": hashlib.sha256(encoded).hexdigest()}
        )
    encoded_all = json.dumps(
        snapshots,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "contracts": snapshots,
        "sha256": hashlib.sha256(encoded_all).hexdigest(),
    }


def _method_record(profile: str, rounds: int) -> dict[str, object]:
    return {
        "profile": profile,
        "generator": GENERATOR_ID,
        "admission_repeats": ADMISSION_REPEATS,
        "benchmark_rounds": rounds,
        "cpu_timing_scope": "case-cold-and-warm-complete-host-table-v1",
        "gpu_resident_compute_scope": "synchronized-resident-packed-compute-v1",
        "gpu_resident_public_scope": (
            "resident-compute-plus-d2h-plus-typed-host-finalizer-v1"
        ),
        "gpu_transfer_inclusive_scope": (
            "h2d-plus-compute-plus-d2h-plus-typed-host-finalizer-v1"
        ),
        "memory_observation_scope": "isolated-cupy-private-pool-reserved-high-water-v1",
        "memory_model_id": MEMORY_MODEL_ID,
        "parity_policy_id": PARITY_POLICY_ID,
        "cancellation": "synchronized-measurement-block-stage-v1",
        "provider_cache_warmup": (
            "2d-and-3d-morphology-and-intensity-before-private-pools-v1"
        ),
        "provider_cache_scope": (
            "process-lifetime-cupy-kernels-and-lookups-excluded-from-private-pool-v1"
        ),
        "timing_note": (
            "The typed host table is mandatory. Resident packed compute is diagnostic; "
            "selection compares the complete public output boundary."
        ),
    }


def _environment_record(cp, device_index: int) -> dict[str, object]:
    properties = cp.cuda.runtime.getDeviceProperties(device_index)
    name = properties.get("name", b"")
    if isinstance(name, bytes):
        name = name.decode(errors="replace").rstrip("\x00")
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "python_executable": _executable_name(sys.executable),
        "device_index": device_index,
        "device_name": str(name),
        "compute_capability": (
            f"{int(properties.get('major', 0))}.{int(properties.get('minor', 0))}"
        ),
        "device_total_memory_bytes": int(properties.get("totalGlobalMem", 0)),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
        "device_count": int(cp.cuda.runtime.getDeviceCount()),
    }


def _executable_name(executable: str) -> str:
    """Return a privacy-safe interpreter label for machine-local evidence."""

    normalized = str(executable).strip().replace("\\", "/").rstrip("/")
    return normalized.rsplit("/", 1)[-1] or "python"


def _package_record(cp, np) -> dict[str, str]:
    packages = {"numpy": str(np.__version__), "cupy": str(cp.__version__)}
    for distribution in (
        "scipy",
        "scikit-image",
        "cupy-cuda13x",
        "napari-vipp",
    ):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed-as-distribution"
    return packages


def validate_existing(path: Path | str) -> Path:
    artifact = Path(path).resolve()
    document = json.loads(artifact.read_text(encoding="utf-8"))
    _validate_document_contract(document)
    if artifact.read_text(encoding="utf-8") != _canonical_json(document):
        raise EvidenceError("Evidence JSON is not canonical.")
    markdown = artifact.with_suffix(".md")
    if not markdown.is_file():
        raise EvidenceError(f"Readable Markdown is missing: {markdown}.")
    if markdown.read_text(encoding="utf-8") != _render_markdown(document):
        raise EvidenceError("Readable Markdown is stale or edited.")
    return artifact


def _validate_document_contract(document: object) -> None:
    if not isinstance(document, Mapping):
        raise EvidenceError("Evidence root must be an object.")
    if frozenset(document) != _ROOT_KEYS:
        raise EvidenceError("Evidence root fields differ from the schema.")
    if (
        document.get("schema") != SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise EvidenceError("Evidence schema or version is unsupported.")
    if (
        document.get("kind")
        != "scientific-admission-and-machine-local-performance-evidence"
    ):
        raise EvidenceError("Evidence kind is invalid.")
    if document.get("portable_performance_claim") is not False:
        raise EvidenceError("Evidence must not claim portable performance.")
    if document.get("durable_optimizer_record") is not False:
        raise EvidenceError("Evidence must not masquerade as an optimizer record.")
    if document.get("profile") not in {"quick", "full"}:
        raise EvidenceError("Evidence profile is invalid.")
    try:
        datetime.fromisoformat(str(document["created_utc"]))
    except ValueError as exc:
        raise EvidenceError("Evidence timestamp is invalid.") from exc
    if document.get("source_provenance") != _source_provenance():
        raise EvidenceError("Source provenance fingerprints are stale.")
    if document.get("operation_contracts") != _operation_contracts():
        raise EvidenceError("Operation contracts are stale.")
    _validate_admission(document.get("admission"))
    _validate_rejections(document.get("rejections"))
    _validate_lifecycle(document.get("lifecycle"))
    _validate_performance(document.get("performance"), str(document["profile"]))


def _validate_admission(value: object) -> None:
    if not isinstance(value, Mapping) or value.get("status") != "pass":
        raise EvidenceError("Admission did not pass.")
    coverage = frozenset(str(item) for item in value.get("coverage", ()))
    if not REQUIRED_ADMISSION_COVERAGE <= coverage:
        raise EvidenceError("Admission coverage is incomplete.")
    cases = value.get("cases")
    if not isinstance(cases, list) or value.get("case_count") != len(cases):
        raise EvidenceError("Admission case count is inconsistent.")
    if len(cases) != len(_admission_cases()):
        raise EvidenceError("Admission manifest coverage is incomplete.")
    expected_ids = {definition.case_id for definition in _admission_cases()}
    if {
        str(case.get("case_id")) for case in cases if isinstance(case, Mapping)
    } != expected_ids:
        raise EvidenceError("Admission case IDs differ from the manifest.")
    for case in cases:
        if not isinstance(case, Mapping):
            raise EvidenceError("Admission case must be an object.")
        parity = case.get("parity")
        if not isinstance(parity, Mapping) or parity.get("passed") is not True:
            raise EvidenceError("Admission parity did not pass.")
        hashes = case.get("gpu_repeat_table_sha256")
        if (
            not isinstance(hashes, list)
            or len(hashes) != ADMISSION_REPEATS
            or len(set(hashes)) != 1
        ):
            raise EvidenceError("Admission output is not exactly deterministic.")
        if (
            case.get("input_immutable") is not True
            or case.get("resident_output_contiguous") is not True
        ):
            raise EvidenceError("Admission residency lifecycle is incomplete.")
        _validate_cleanup(case.get("cleanup"))


def _validate_rejections(value: object) -> None:
    if not isinstance(value, Mapping) or value.get("status") != "pass":
        raise EvidenceError("Rejection matrix did not pass.")
    coverage = frozenset(str(item) for item in value.get("coverage", ()))
    if not REQUIRED_REJECTION_COVERAGE <= coverage:
        raise EvidenceError("Rejection coverage is incomplete.")
    cases = value.get("cases")
    if not isinstance(cases, list) or value.get("case_count") != len(cases):
        raise EvidenceError("Rejection case count is inconsistent.")
    if len(cases) != len(_rejection_cases()) or not all(
        isinstance(case, Mapping) and case.get("rejected") is True for case in cases
    ):
        raise EvidenceError("Required provider rejection was not observed.")
    for case in cases:
        _validate_cleanup(case.get("cleanup"))


def _validate_lifecycle(value: object) -> None:
    if not isinstance(value, Mapping) or value.get("status") != "pass":
        raise EvidenceError("Lifecycle evidence did not pass.")
    cases = value.get("cases")
    if not isinstance(cases, list) or len(cases) != 2 or value.get("case_count") != 2:
        raise EvidenceError("Lifecycle operation coverage is incomplete.")
    expected = {MORPHOLOGY_OPERATION_ID, INTENSITY_OPERATION_ID}
    if {case.get("operation_id") for case in cases} != expected:
        raise EvidenceError("Lifecycle operation IDs are incomplete.")
    for case in cases:
        if (
            case.get("cancellation_observed") is not True
            or case.get("post_cancellation_reuse_parity") is not True
        ):
            raise EvidenceError("Cancellation or post-cancellation reuse failed.")
        updates = case.get("complete_updates")
        if (
            not isinstance(updates, list)
            or [update.get("current") for update in updates]
            != list(range(int(case.get("expected_total", -1)) + 1))
            or {update.get("total") for update in updates}
            != {case.get("expected_total")}
        ):
            raise EvidenceError("Truthful completion progress is incomplete.")
        _validate_cleanup(case.get("cleanup"))


def _validate_performance(value: object, profile: str) -> None:
    if not isinstance(value, Mapping) or value.get("status") != "pass":
        raise EvidenceError("Performance evidence did not pass.")
    cases = value.get("results")
    if not isinstance(cases, list) or value.get("case_count") != len(cases):
        raise EvidenceError("Performance case count is inconsistent.")
    expected_ids = {definition.case_id for definition in _performance_cases(profile)}
    if {
        str(case.get("case_id")) for case in cases if isinstance(case, Mapping)
    } != expected_ids:
        raise EvidenceError("Performance case IDs differ from the manifest.")
    rounds = int(value.get("rounds", 0))
    if rounds != (BENCHMARK_ROUNDS if profile == "full" else 3):
        raise EvidenceError("Performance round count is inconsistent.")
    for case in cases:
        if (
            not isinstance(case, Mapping)
            or case.get("parity", {}).get("passed") is not True
        ):
            raise EvidenceError("Performance parity did not pass.")
        samples = case.get("samples")
        summary = case.get("summary")
        if not isinstance(samples, Mapping) or not isinstance(summary, Mapping):
            raise EvidenceError("Performance timing fields are malformed.")
        names = (
            ("cpu_seconds", "cpu_median_seconds"),
            ("gpu_resident_compute_seconds", "gpu_resident_compute_median_seconds"),
            ("gpu_resident_public_seconds", "gpu_resident_public_median_seconds"),
            ("gpu_transfer_inclusive_seconds", "gpu_transfer_inclusive_median_seconds"),
        )
        for sample_name, median_name in names:
            values = samples.get(sample_name)
            if (
                not isinstance(values, list)
                or len(values) != rounds
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or item <= 0
                    for item in values
                )
            ):
                raise EvidenceError("Performance timing samples are invalid.")
            if not math.isclose(
                float(summary.get(median_name, -1.0)),
                statistics.median(values),
                rel_tol=1e-12,
                abs_tol=1e-15,
            ):
                raise EvidenceError("Performance timing summary is inconsistent.")
        memory = case.get("memory")
        if (
            not isinstance(memory, Mapping)
            or memory.get("estimate_with_uncertainty_covers_observed") is not True
        ):
            raise EvidenceError("Performance memory evidence is inconsistent.")
        if int(memory.get("device_peak_with_uncertainty_bytes", 0)) < int(
            memory.get("observed_reserved_bytes", 0)
        ):
            raise EvidenceError(
                "Performance memory estimate does not cover observation."
            )
        _validate_cleanup(memory)
        _validate_cleanup(case.get("cleanup"))
    if value.get("all_memory_estimates_cover_observed") is not True:
        raise EvidenceError("Performance memory coverage summary is false.")


def _validate_cleanup(value: object) -> None:
    if not isinstance(value, Mapping):
        raise EvidenceError("CUDA cleanup record is missing.")
    if (
        int(value.get("device_pool_used_bytes_after_cleanup", -1)) != 0
        or int(value.get("device_pool_reserved_bytes_after_cleanup", -1)) != 0
    ):
        raise EvidenceError("CUDA private-pool cleanup is incomplete.")


def _drain_pool(cp, pool) -> dict[str, int]:
    cp.cuda.get_current_stream().synchronize()
    gc.collect()
    pool.free_all_blocks()
    cp.cuda.get_current_stream().synchronize()
    used = int(pool.used_bytes())
    reserved = int(pool.total_bytes())
    if used or reserved:
        raise EvidenceError(
            f"CUDA pool cleanup failed: used={used}, reserved={reserved}."
        )
    return {
        "device_pool_used_bytes_after_cleanup": used,
        "device_pool_reserved_bytes_after_cleanup": reserved,
    }


def _array_sha256(data) -> str:
    arr = _numpy().ascontiguousarray(data)
    digest = hashlib.sha256()
    digest.update(arr.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(arr.shape), separators=(",", ":")).encode("ascii"))
    digest.update(arr.tobytes(order="C"))
    return digest.hexdigest()


def _table_sha256(table) -> str:
    def cell(value):
        if type(value) is float:
            return {"type": "float", "value": value.hex()}
        if type(value) is int:
            return {"type": "int", "value": value}
        if type(value) is bool:
            return {"type": "bool", "value": value}
        if type(value) is str:
            return {"type": "str", "value": value}
        raise EvidenceError(f"Unsupported public table scalar {type(value).__name__}.")

    payload = {
        "columns": list(table.columns),
        "rows": [[cell(value) for value in row] for row in table.rows],
        "name": table.name,
        "table_kind": table.table_kind,
        "source_name": table.source_name,
        "column_units": [list(item) for item in table.column_units],
    }
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _canonical_json(document: Mapping[str, object]) -> str:
    return (
        json.dumps(
            document,
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def _historical_provider_comparison(
    document: Mapping[str, object],
) -> tuple[str, ...]:
    """Summarize like-for-like transfer-inclusive medians when history exists."""

    try:
        historical = json.loads(HISTORICAL_CUCIM_OUTPUT.read_text(encoding="utf-8"))
        current_results = document["performance"]["results"]
        historical_results = historical["performance"]["results"]
    except (FileNotFoundError, KeyError, TypeError, json.JSONDecodeError):
        return ()
    historical_by_id = {item["case_id"]: item for item in historical_results}
    ratios: list[float] = []
    for current in current_results:
        previous = historical_by_id.get(current["case_id"])
        if previous is None:
            continue
        if (
            previous.get("label_input_sha256")
            != current.get("label_input_sha256")
            or previous.get("intensity_input_sha256")
            != current.get("intensity_input_sha256")
        ):
            continue
        current_seconds = float(
            current["summary"]["gpu_transfer_inclusive_median_seconds"]
        )
        previous_seconds = float(
            previous["summary"]["gpu_transfer_inclusive_median_seconds"]
        )
        if current_seconds > 0.0 and previous_seconds > 0.0:
            ratios.append(previous_seconds / current_seconds)
    if not ratios:
        return ()
    geometric_mean = math.exp(statistics.fmean(math.log(value) for value in ratios))
    wins = sum(value > 1.0 for value in ratios)
    return (
        "## Historical-provider comparison",
        "",
        (
            "The preserved `measurements-cucim-windows-rtx5090.json` artifact "
            f"contains {len(ratios)} matching case IDs and input SHA-256 values. "
            "Comparing transfer-inclusive medians, the production CuPy provider "
            f"is faster in {wins} of {len(ratios)} matched cases: "
            f"**{geometric_mean:.2f}×** geometric mean, with a "
            f"**{min(ratios):.2f}–{max(ratios):.2f}×** range. This comparison "
            "is the basis for removing cuCIM from the active measurement and "
            "installation paths; the old artifact remains immutable historical "
            "evidence."
        ),
        "",
    )


def _render_markdown(document: Mapping[str, object]) -> str:
    environment = document["environment"]
    admission = document["admission"]
    rejections = document["rejections"]
    lifecycle = document["lifecycle"]
    performance = document["performance"]
    lines = [
        "# GPU basic Measurements evidence",
        "",
        f"Generated: `{document['created_utc']}`",
        "",
        (
            f"This is machine-local evidence from **{environment['device_name']}** "
            f"on {environment['system']} with Python {environment['python']}. It is "
            "not a portable performance claim or a durable optimizer record."
        ),
        "",
        "The public GPU timing includes the mandatory packed-result transfer and "
        "typed `TableData` finalizer. Resident packed-compute timing is shown only "
        "to explain where time is spent.",
        "",
        "## Admission and lifecycle",
        "",
        f"- Admission cases: **{admission['case_count']}** (all passed)",
        f"- Scientifically ineligible cases rejected: **{rejections['case_count']}**",
        f"- Progress/cancellation lifecycle cases: **{lifecycle['case_count']}**",
        f"- Parity policy: `{admission['parity_policy_id']}`",
        "- Private CUDA pools after every case: **0 used / 0 reserved bytes**",
        "",
        "Admission covers 2D/3D, leading blocks, reordered/calibrated axes, sparse "
        "and repeated IDs, zero-row tables, all supported intensity dtypes, exact "
        "schema/order/units/scalar types, deterministic repeats, cancellation, and "
        "post-cancellation reuse.",
        "",
        "## Performance",
        "",
        (
            "| Workload | Rows | CPU (s) | GPU resident packed (s) | "
            "GPU resident + table (s) | GPU full public (s) | Full speedup | "
            "Screen | Peak VRAM / bound |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for result in performance["results"]:
        summary = result["summary"]
        memory = result["memory"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(result["label"]),
                    str(result["object_row_count"]),
                    f"{summary['cpu_median_seconds']:.6f}",
                    f"{summary['gpu_resident_compute_median_seconds']:.6f}",
                    f"{summary['gpu_resident_public_median_seconds']:.6f}",
                    f"{summary['gpu_transfer_inclusive_median_seconds']:.6f}",
                    f"{summary['gpu_transfer_inclusive_speedup']:.2f}×",
                    str(summary["screening_choice"]),
                    (
                        f"{memory['observed_reserved_bytes'] / 1024**2:.1f} / "
                        f"{memory['device_peak_with_uncertainty_bytes'] / 1024**2:.1f} "
                        "MiB"
                    ),
                )
            )
            + " |"
        )
    lines.extend(("", *_historical_provider_comparison(document)))
    lines.extend(
        (
            "## Method notes",
            "",
            "- CPU samples are complete typed-table calls.",
            "- GPU resident packed samples end at a synchronized device matrix.",
            "- GPU resident + table samples include D2H and typed host finalization.",
            "- GPU full public samples additionally include authored input transfers.",
            "- Screening compares CPU with the full public GPU boundary.",
            (
                "- The memory bound is the production "
                "`cupy-basic-measurements-memory-v1` model including its "
                "uncertainty reserve."
            ),
            "",
            "Reproduce with:",
            "",
            "```powershell",
            "python scripts/benchmark_gpu_measurements.py --profile full",
            (
                "python scripts/benchmark_gpu_measurements.py --validate-existing "
                "docs/benchmarks/measurements-cupy-windows-rtx5090.json"
            ),
            "```",
            "",
        )
    )
    return "\n".join(lines)


def _atomic_write_artifacts(
    output: Path,
    markdown: Path,
    document: Mapping[str, object],
) -> None:
    _validate_document_contract(document)
    _atomic_write_text(output, _canonical_json(document))
    _atomic_write_text(markdown, _render_markdown(document))


def _atomic_write_text(path: Path, content: str) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


@cache
def _numpy():
    import numpy as np

    return np


@cache
def _cupy():
    import cupy as cp

    return cp


@cache
def _operation_functions() -> dict[str, object]:
    from napari_vipp.core.gpu.cupy_measurements import (
        measure_objects as gpu_morphology,
    )
    from napari_vipp.core.gpu.cupy_measurements import (
        measure_objects_with_intensity as gpu_intensity,
    )
    from napari_vipp.core.measurements import (
        finalize_basic_measurement_outputs,
        measurement_table_parity,
    )
    from napari_vipp.core.operations import measure_objects as cpu_morphology
    from napari_vipp.core.operations import (
        measure_objects_with_intensity as cpu_intensity,
    )

    return {
        "cpu_morphology": cpu_morphology,
        "cpu_intensity": cpu_intensity,
        "gpu_morphology": gpu_morphology,
        "gpu_intensity": gpu_intensity,
        "finalizer": finalize_basic_measurement_outputs,
        "parity": measurement_table_parity,
    }


if __name__ == "__main__":
    raise SystemExit(main())
