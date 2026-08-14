#!/usr/bin/env python
"""Qualify VIPP's exact CuPyX boolean-mask cleanup implementations.

This owner covers complete-hole filling and boolean small-object removal as one
strict scientific admission suite.  It records exact CPU parity, adversarial
workloads, structural metadata, input integrity, conservative memory, truthful
block-boundary cancellation, cleanup and reuse, safe CPU fallback, provenance,
and both resident and transfer-inclusive timing.

Importing this module, asking for ``--help``, or validating an existing JSON
artifact does not import CuPy or initialize CUDA.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "napari-vipp-cupyx-mask-cleanup-evidence"
SCHEMA_VERSION = 1
EVIDENCE_KIND = "scientific-admission-and-machine-local-performance-evidence"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/benchmarks/mask-cleanup-cupyx-local.json"
ADMISSION_REPEATS = 3
QUICK_ROUNDS = 3
FULL_ROUNDS = 7
RUNTIME_ID = "cuda-cupy"
LIBRARY_ID = "cupyx"
ENVIRONMENT_POLICY_ID = "cuda-cupy-14.1.1-cpython312-windows-native-v3"
REQUIRED_FACETS = (
    "cpu_oracle_parity",
    "adversarial_workloads",
    "metadata",
    "input_integrity",
    "memory",
    "cancellation",
    "cleanup",
    "fallback",
    "provenance",
    "transfer_inclusive_timing",
)
IMPLEMENTATION_IDS = {
    "fill_holes": "cupyx-fill-holes-all-v1",
    "remove_small_objects": "cupyx-remove-small-objects-bool-v1",
}
SOURCE_PROVENANCE_PATHS = (
    Path("src/napari_vipp/core/operations.py"),
    Path("src/napari_vipp/core/connected_components.py"),
    Path("src/napari_vipp/core/metadata.py"),
    Path("src/napari_vipp/core/pipeline.py"),
    Path("src/napari_vipp/core/gpu/cupy_fill_holes.py"),
    Path("src/napari_vipp/core/gpu/cupy_remove_small_objects.py"),
    Path("src/napari_vipp/core/compute_specs.py"),
    Path("src/napari_vipp/core/compute_policy.py"),
    Path("src/napari_vipp/core/compute_planning.py"),
    Path("src/napari_vipp/core/execution.py"),
    Path("src/napari_vipp/core/compute_benchmark_adapter.py"),
    Path("scripts/benchmark_gpu_mask_cleanup.py"),
)
REQUIRED_ADMISSION_COVERAGE = {
    "fill_holes": frozenset(
        {
            "dtype:bool",
            "rank:2",
            "rank:3",
            "leading-blocks:independent",
            "layout:contiguous",
            "layout:noncontiguous",
            "connectivity:face",
            "connectivity:full",
            "topology:enclosed-and-boundary-open",
            "topology:checkerboard",
            "shape:odd",
            "shape:empty",
            "repeat:deterministic",
        }
    ),
    "remove_small_objects": frozenset(
        {
            "dtype:bool",
            "rank:2",
            "rank:3",
            "leading-blocks:independent",
            "layout:contiguous",
            "layout:noncontiguous",
            "connectivity:face",
            "connectivity:full",
            "size:min-minus-one-min-plus-one",
            "size:identity-zero-one",
            "topology:checkerboard",
            "shape:odd",
            "shape:empty",
            "repeat:deterministic",
        }
    ),
}
_ROOT_KEYS = frozenset(
    {
        "schema",
        "schema_version",
        "created_utc",
        "kind",
        "portable_performance_claim",
        "durable_optimizer_record",
        "profile",
        "implementations",
        "environment",
        "packages",
        "environment_packages_sha256",
        "source_provenance",
        "operation_contracts",
        "method",
        "facets",
        "admission",
        "metadata",
        "lifecycle",
        "fallback",
        "performance",
    }
)
_ENVIRONMENT_KEYS = frozenset(
    {
        "system",
        "release",
        "machine",
        "python_implementation",
        "python_abi",
        "python",
        "python_executable",
        "device_index",
        "device_name",
        "compute_capability",
        "device_total_memory_bytes",
        "cuda_driver_version",
        "cuda_runtime_version",
    }
)
_PACKAGE_KEYS = frozenset(
    {
        "numpy",
        "scipy",
        "scikit-image",
        "cupy",
        "cupy-cuda13x",
        "napari-vipp",
    }
)


class EvidenceError(RuntimeError):
    """Raised when mask-cleanup evidence is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class AdmissionCase:
    operation_id: str
    case_id: str
    kind: str
    coverage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerformanceCase:
    operation_id: str
    case_id: str
    shape: tuple[int, ...]
    spatial_ndim: int
    spatial_mode: str
    connectivity: str
    size_parameter: int
    pattern: str


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Evidence JSON path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="full",
        help="Short development run or complete qualification run.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=0,
        help="CUDA device ordinal (default: 0).",
    )
    parser.add_argument(
        "--validate-existing",
        type=Path,
        help="Validate an existing artifact without importing CuPy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.validate_existing is not None:
            validated = validate_existing(args.validate_existing)
            print(f"Mask-cleanup evidence is current: {validated}")
            return 0
        if args.device_index < 0:
            raise ValueError("device index must be nonnegative")
        document = build_evidence(args.profile, args.device_index)
        _atomic_write_json(args.output.resolve(), document)
        validate_existing(args.output.resolve())
    except (EvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"Mask-cleanup evidence failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output.resolve()}")
    return 0


def build_evidence(profile: str, device_index: int) -> dict[str, object]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'.")
    source_snapshot = _source_provenance()
    contract_snapshot = _operation_contracts()
    np = _numpy()
    cp = _cupy()
    cpu_functions, gpu_functions = _operation_functions()
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - machine-specific failure
        raise EvidenceError(f"CUDA runtime probe failed: {exc}") from exc
    if device_index >= device_count:
        raise EvidenceError(
            f"CUDA device index {device_index} is unavailable; found {device_count}."
        )
    environment = _environment_record(cp, device_index)
    packages = _package_record(cp, np)
    environment_packages_sha256 = _environment_packages_sha256(
        environment,
        packages,
    )
    _validate_environment_and_packages(
        {
            "environment": environment,
            "packages": packages,
            "environment_packages_sha256": environment_packages_sha256,
        }
    )

    rounds = FULL_ROUNDS if profile == "full" else QUICK_ROUNDS
    with cp.cuda.Device(device_index):
        _warm_runtime(cp, gpu_functions)
        admission = _run_admission(cp, cpu_functions, gpu_functions)
        metadata = _run_metadata(cpu_functions)
        lifecycle = _run_lifecycle(cp, cpu_functions, gpu_functions)
        fallback = _run_fallback(cpu_functions)
        performance_results = [
            _run_performance_case(
                cp,
                cpu_functions[item.operation_id],
                gpu_functions[item.operation_id],
                item,
                rounds,
            )
            for item in _performance_cases(profile)
        ]

    if source_snapshot != _source_provenance():
        raise EvidenceError("Tracked source changed while evidence was collected.")
    if contract_snapshot != _operation_contracts():
        raise EvidenceError("A mask-cleanup contract changed during collection.")

    performance = {
        "status": "pass",
        "rounds": rounds,
        "case_count": len(performance_results),
        "results": performance_results,
        "all_memory_estimates_cover_observed": all(
            item["memory"]["estimate_covers_observed"] for item in performance_results
        ),
    }
    if not performance["all_memory_estimates_cover_observed"]:
        raise EvidenceError("A declared cleanup memory estimate was exceeded.")

    implementations = _implementation_records()
    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": EVIDENCE_KIND,
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": profile,
        "implementations": implementations,
        "environment": environment,
        "packages": packages,
        "environment_packages_sha256": environment_packages_sha256,
        "source_provenance": source_snapshot,
        "operation_contracts": contract_snapshot,
        "method": _method_record(profile, rounds),
        "facets": {
            _implementation_key(item["operation_id"]): {
                facet: "pass" for facet in REQUIRED_FACETS
            }
            for item in implementations
        },
        "admission": admission,
        "metadata": metadata,
        "lifecycle": lifecycle,
        "fallback": fallback,
        "performance": performance,
    }
    _validate_document(document, require_current_sources=True)
    return document


def _admission_cases() -> tuple[AdmissionCase, ...]:
    return (
        AdmissionCase(
            "fill_holes",
            "fill-2d-face-odd-leading",
            "fill-2d-face-odd-leading",
            (
                "dtype:bool",
                "rank:2",
                "leading-blocks:independent",
                "layout:contiguous",
                "connectivity:face",
                "topology:enclosed-and-boundary-open",
                "shape:odd",
            ),
        ),
        AdmissionCase(
            "fill_holes",
            "fill-2d-full-strided-checker",
            "fill-2d-full-strided-checker",
            (
                "dtype:bool",
                "rank:2",
                "leading-blocks:independent",
                "layout:noncontiguous",
                "connectivity:full",
                "topology:checkerboard",
                "shape:odd",
            ),
        ),
        AdmissionCase(
            "fill_holes",
            "fill-3d-face-leading",
            "fill-3d-face-leading",
            (
                "dtype:bool",
                "rank:3",
                "leading-blocks:independent",
                "layout:contiguous",
                "connectivity:face",
                "topology:enclosed-and-boundary-open",
                "shape:odd",
            ),
        ),
        AdmissionCase(
            "fill_holes",
            "fill-empty-spatial-blocks",
            "fill-empty-spatial-blocks",
            (
                "dtype:bool",
                "rank:2",
                "leading-blocks:independent",
                "layout:contiguous",
                "connectivity:face",
                "shape:empty",
            ),
        ),
        AdmissionCase(
            "remove_small_objects",
            "remove-2d-face-threshold-odd",
            "remove-2d-face-threshold-odd",
            (
                "dtype:bool",
                "rank:2",
                "leading-blocks:independent",
                "layout:contiguous",
                "connectivity:face",
                "size:min-minus-one-min-plus-one",
                "shape:odd",
            ),
        ),
        AdmissionCase(
            "remove_small_objects",
            "remove-2d-full-strided-checker",
            "remove-2d-full-strided-checker",
            (
                "dtype:bool",
                "rank:2",
                "leading-blocks:independent",
                "layout:noncontiguous",
                "connectivity:full",
                "topology:checkerboard",
                "shape:odd",
            ),
        ),
        AdmissionCase(
            "remove_small_objects",
            "remove-3d-face-leading",
            "remove-3d-face-leading",
            (
                "dtype:bool",
                "rank:3",
                "leading-blocks:independent",
                "layout:contiguous",
                "connectivity:face",
                "size:min-minus-one-min-plus-one",
                "shape:odd",
            ),
        ),
        AdmissionCase(
            "remove_small_objects",
            "remove-empty-identity",
            "remove-empty-identity",
            (
                "dtype:bool",
                "rank:2",
                "leading-blocks:independent",
                "layout:contiguous",
                "connectivity:full",
                "size:identity-zero-one",
                "shape:empty",
            ),
        ),
    )


def _host_case(kind: str):
    np = _numpy()
    if kind == "fill-2d-face-odd-leading":
        mask = np.ones((2, 17, 19), dtype=bool)
        mask[0, 4:7, 5:8] = False
        mask[0, 0:6, 13] = False
        mask[1, 8, 9] = False
        mask[1, 12:15, 3:7] = False
        return mask
    if kind == "fill-2d-full-strided-checker":
        return _noncontiguous_case_base(kind)[:, :, 1::2]
    if kind == "fill-3d-face-leading":
        mask = np.ones((2, 9, 13, 15), dtype=bool)
        mask[0, 3:6, 4:9, 5:10] = False
        mask[0, 0:4, 10, 10] = False
        mask[1, 4, 6, 7] = False
        return mask
    if kind == "fill-empty-spatial-blocks":
        return np.empty((3, 0, 17), dtype=bool)
    if kind == "remove-2d-face-threshold-odd":
        mask = np.zeros((2, 17, 19), dtype=bool)
        mask[0, 1, 1:6] = True
        mask[0, 4:6, 2:5] = True
        mask[0, 9:12, 8:11] = True
        mask[1, 1:3, 1:4] = True
        mask[1, 7:9, 10:14] = True
        return mask
    if kind == "remove-2d-full-strided-checker":
        return _noncontiguous_case_base(kind)[:, :, 1::2]
    if kind == "remove-3d-face-leading":
        mask = np.zeros((2, 9, 13, 15), dtype=bool)
        mask[0, 1, 1, 1:7] = True
        mask[0, 3:5, 3:5, 3:5] = True
        mask[0, 5:8, 7:10, 8:11] = True
        mask[1, 2:4, 2:4, 2:4] = True
        mask[1, 5:8, 8:11, 9:12] = True
        return mask
    if kind == "remove-empty-identity":
        return np.empty((0, 13, 15), dtype=bool)
    raise ValueError(f"Unknown admission case kind {kind!r}.")


def _noncontiguous_case_base(kind: str):
    np = _numpy()
    if kind == "fill-2d-full-strided-checker":
        base = np.ones((2, 23, 50), dtype=bool)
        grid = np.indices((19, 46)).sum(axis=0) % 2 == 0
        base[:, 2:21, 2:48] = grid
        base[:, 8:15, 18:32] = True
        base[:, 10:13, 22:28] = False
        return base
    if kind == "remove-2d-full-strided-checker":
        base = np.zeros((2, 23, 50), dtype=bool)
        grid = np.indices((19, 46)).sum(axis=0) % 2 == 0
        base[:, 2:21, 2:48] = grid
        base[0, 8:14, 18:30] = True
        base[1, 4:7, 34:42] = True
        return base
    raise ValueError(f"Unknown strided case kind {kind!r}.")


def _case_parameters(kind: str) -> dict[str, object]:
    if kind == "fill-2d-face-odd-leading":
        return {
            "max_hole_size": 0,
            "spatial_mode": "2D YX",
            "connectivity": "Face connected",
        }
    if kind == "fill-2d-full-strided-checker":
        return {
            "max_hole_size": 0,
            "spatial_mode": "2D YX",
            "connectivity": "Full connectivity",
        }
    if kind == "fill-3d-face-leading":
        return {
            "max_hole_size": 0,
            "spatial_mode": "3D ZYX",
            "connectivity": "Face connected",
        }
    if kind == "fill-empty-spatial-blocks":
        return {
            "max_hole_size": 0,
            "spatial_mode": "2D YX",
            "connectivity": "Face connected",
        }
    if kind == "remove-2d-face-threshold-odd":
        return {
            "min_size": 6,
            "spatial_mode": "2D YX",
            "connectivity": "Face connected",
        }
    if kind == "remove-2d-full-strided-checker":
        return {
            "min_size": 4,
            "spatial_mode": "2D YX",
            "connectivity": "Full connectivity",
        }
    if kind == "remove-3d-face-leading":
        return {
            "min_size": 8,
            "spatial_mode": "3D ZYX",
            "connectivity": "Face connected",
        }
    if kind == "remove-empty-identity":
        return {
            "min_size": 1,
            "spatial_mode": "2D YX",
            "connectivity": "Full connectivity",
        }
    raise ValueError(f"Unknown admission case kind {kind!r}.")


def _device_case(cp, host, kind: str):
    if "strided" in kind:
        keepalive = cp.asarray(_noncontiguous_case_base(kind))
        return keepalive[:, :, 1::2], keepalive
    return cp.asarray(host), None


def _run_admission(cp, cpu_functions, gpu_functions) -> dict[str, object]:
    grouped = {operation_id: [] for operation_id in IMPLEMENTATION_IDS}
    coverage = {
        operation_id: {"repeat:deterministic"} for operation_id in IMPLEMENTATION_IDS
    }
    for definition in _admission_cases():
        host = _host_case(definition.kind)
        host.setflags(write=False)
        parameters = _case_parameters(definition.kind)
        expected = cpu_functions[definition.operation_id](host, **parameters)
        input_hash = _array_sha256(host)
        pool = cp.cuda.MemoryPool()
        output_hashes = []
        output_resident = True
        output_independent = True
        with cp.cuda.using_allocator(pool.malloc):
            device, keepalive = _device_case(cp, host, definition.kind)
            if "layout:noncontiguous" in definition.coverage and bool(
                device.flags.c_contiguous
            ):
                raise EvidenceError(
                    f"{definition.case_id} did not construct a strided CUDA input."
                )
            for _ in range(ADMISSION_REPEATS):
                output = gpu_functions[definition.operation_id](device, **parameters)
                cp.cuda.get_current_stream().synchronize()
                output_resident &= isinstance(output, cp.ndarray)
                output_independent &= output.dtype == cp.bool_ and (
                    not output.size or int(output.data.ptr) != int(device.data.ptr)
                )
                actual = cp.asnumpy(output)
                if not _numpy().array_equal(actual, expected):
                    raise EvidenceError(f"Parity failed for {definition.case_id}.")
                output_hashes.append(_array_sha256(actual))
                del actual, output
            resident_after = cp.asnumpy(device)
            input_immutable = _array_sha256(resident_after) == input_hash
            del resident_after, device, keepalive
            cp.cuda.get_current_stream().synchronize()
        cleanup = _drain_pool(cp, pool)
        if (
            not output_resident
            or not output_independent
            or not input_immutable
            or len(set(output_hashes)) != 1
        ):
            raise EvidenceError(f"Integrity failed for {definition.case_id}.")
        coverage[definition.operation_id].update(definition.coverage)
        grouped[definition.operation_id].append(
            {
                "case_id": definition.case_id,
                "shape": list(host.shape),
                "input_dtype": str(host.dtype),
                "output_dtype": str(expected.dtype),
                "parameters": _json_value(parameters),
                "coverage": sorted(definition.coverage),
                "input_sha256": input_hash,
                "output_integrity_contract": "independent-bool-allocation-v1",
                "cpu_gpu_bitwise_equal": True,
                "gpu_output_resident": output_resident,
                "input_immutable": input_immutable,
                "repeat_deterministic": len(set(output_hashes)) == 1,
                "repeat_count": ADMISSION_REPEATS,
                "cpu_output_sha256": _array_sha256(expected),
                "gpu_output_sha256": output_hashes[-1],
                "cleanup": cleanup,
            }
        )
    for operation_id, required in REQUIRED_ADMISSION_COVERAGE.items():
        if not required <= coverage[operation_id]:
            missing = sorted(required - coverage[operation_id])
            raise EvidenceError(f"{operation_id} admission misses {missing}.")
    return {
        operation_id: {
            "status": "pass",
            "case_count": len(grouped[operation_id]),
            "repeat_count": ADMISSION_REPEATS,
            "coverage": sorted(coverage[operation_id]),
            "cases": grouped[operation_id],
        }
        for operation_id in sorted(IMPLEMENTATION_IDS)
    }


def _run_metadata(cpu_functions) -> dict[str, object]:
    records = _expected_metadata_records(cpu_functions)
    if any(item["status"] != "pass" for item in records.values()):
        raise EvidenceError("Mask-cleanup structural metadata projection failed.")
    return records


def _expected_metadata_records(cpu_functions=None) -> dict[str, object]:
    np = _numpy()
    if cpu_functions is None:
        cpu_functions, _gpu_functions = _operation_functions()
    from napari_vipp.core import execution as execution_module
    from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
    from napari_vipp.core.pipeline import PrototypePipeline

    definitions = {
        "fill_holes": {
            "shape": (2, 17, 19),
            "axes": (
                AxisMetadata("t", "time", "s", 2.5),
                AxisMetadata("y", "space", "micrometer", 0.4),
                AxisMetadata("x", "space", "micrometer", 0.6),
            ),
            "parameters": {
                "max_hole_size": 0,
                "spatial_mode": "Auto from axes",
                "connectivity": "Face connected",
            },
            "resolved": 2,
        },
        "remove_small_objects": {
            "shape": (2, 5, 13, 15),
            "axes": (
                AxisMetadata("t", "time", "s", 3.0),
                AxisMetadata("z", "space", "micrometer", 1.2),
                AxisMetadata("y", "space", "micrometer", 0.4),
                AxisMetadata("x", "space", "micrometer", 0.6),
            ),
            "parameters": {
                "min_size": 7,
                "spatial_mode": "Auto from axes",
                "connectivity": "Full connectivity",
            },
            "resolved": 3,
        },
    }
    records = {}
    for operation_id, definition in definitions.items():
        data = np.zeros(definition["shape"], dtype=bool)
        state = image_state_from_array(
            data,
            axes=definition["axes"],
            history=("Imported calibrated boolean segmentation",),
            source_name="Mask cleanup metadata fixture",
        )
        if state is None:
            raise EvidenceError("Metadata fixture did not produce image state.")
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        source = pipeline.add_node("binary_threshold")
        node = pipeline.add_node(operation_id)
        for name, value in definition["parameters"].items():
            pipeline.set_param(node.id, name, value)
        if not pipeline.connect(source.id, node.id).success:
            raise EvidenceError("Metadata fixture graph could not be connected.")
        call = pipeline.prepare_node_call(node.id, (data,), (state,))
        cpu_output = cpu_functions[operation_id](data, **call.kwargs)
        (_final_output, cpu_state) = pipeline.finalize_node_call(call, cpu_output)[0]
        (resident_state,) = execution_module._predict_device_node_states(
            pipeline,
            call,
            _gpu_spec(operation_id),
            (SimpleNamespace(shape=data.shape, dtype=np.dtype(bool)),),
        )
        if cpu_state is None or resident_state is None:
            raise EvidenceError("Mask-cleanup metadata state is missing.")
        cpu_structure = _metadata_structure(cpu_state)
        resident_structure = _metadata_structure(resident_state)
        records[operation_id] = {
            "status": "pass",
            "resolved_spatial_ndim": call.kwargs.get("resolved_spatial_ndim"),
            "expected_spatial_ndim": definition["resolved"],
            "cpu_structure": cpu_structure,
            "resident_structure": resident_structure,
            "structural_metadata_equal": cpu_structure == resident_structure,
            "shape_preserved": tuple(cpu_state.shape) == tuple(data.shape),
            "bool_mask_kind_preserved": (
                cpu_state.dtype == "bool"
                and resident_state.dtype == "bool"
                and cpu_state.kind == "binary mask"
                and resident_state.kind == "binary mask"
            ),
        }
    return records


def _metadata_structure(state) -> dict[str, object]:
    return {
        "shape": list(state.shape),
        "dtype": state.dtype,
        "kind": state.kind,
        "bit_depth": state.bit_depth,
        "axes": [
            {
                "name": axis.name,
                "type": axis.type,
                "unit": axis.unit,
                "scale": axis.scale,
                "translation": axis.translation,
            }
            for axis in state.axes
        ],
        "source_name": state.source_name,
        "history": list(state.history),
        "channels": [_canonical_value(channel) for channel in state.channels],
        "acquisition": _canonical_value(state.acquisition),
        "source": _canonical_value(state.source),
    }


class _CancelAfterFirstBlock:
    def __init__(self) -> None:
        self.cancelled = False
        self.reports: list[dict[str, object]] = []

    def report(self, current: int, total: int, message: str) -> None:
        self.reports.append(
            {"current": current, "total": total, "message": str(message)}
        )
        if current == 1:
            self.cancelled = True

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("evidence cancellation")


def _run_lifecycle(cp, cpu_functions, gpu_functions) -> dict[str, object]:
    np = _numpy()
    records = {}
    for operation_id in sorted(IMPLEMENTATION_IDS):
        if operation_id == "fill_holes":
            host = np.ones((3, 17, 19), dtype=bool)
            host[:, 8, 9] = False
            parameters = {
                "max_hole_size": 0,
                "spatial_mode": "2D YX",
                "connectivity": "Face connected",
            }
            message = "Fill-hole blocks"
        else:
            host = np.zeros((3, 17, 19), dtype=bool)
            host[:, 3:6, 4:8] = True
            host[:, 12, 14] = True
            parameters = {
                "min_size": 5,
                "spatial_mode": "2D YX",
                "connectivity": "Face connected",
            }
            message = "Small-object blocks"
        expected = cpu_functions[operation_id](host, **parameters)
        input_hash = _array_sha256(host)
        progress = _CancelAfterFirstBlock()
        pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            device = cp.asarray(host)
            cancellation_observed = False
            try:
                gpu_functions[operation_id](
                    device,
                    progress=progress,
                    **parameters,
                )
            except RuntimeError as exc:
                cancellation_observed = str(exc) == "evidence cancellation"
            reused = gpu_functions[operation_id](device, **parameters)
            cp.cuda.get_current_stream().synchronize()
            reused_host = cp.asnumpy(reused)
            reuse_hash = _array_sha256(reused_host)
            resident_after = cp.asnumpy(device)
            input_immutable = _array_sha256(resident_after) == input_hash
            del reused_host, resident_after, reused, device
            cp.cuda.get_current_stream().synchronize()
        cleanup = _drain_pool(cp, pool)
        reuse_exact = reuse_hash == _array_sha256(expected)
        if not cancellation_observed or not reuse_exact or not input_immutable:
            raise EvidenceError(f"{operation_id} lifecycle contract failed.")
        records[operation_id] = {
            "status": "pass",
            "boundary": "synchronized-spatial-block-boundary-v1",
            "cancellation_requested": True,
            "cancellation_observed": cancellation_observed,
            "reported_progress": progress.reports,
            "expected_progress_message": message,
            "post_cancellation_reuse_exact": reuse_exact,
            "input_immutable": input_immutable,
            **cleanup,
        }
    return records


def _fallback_case_definitions() -> tuple[dict[str, object], ...]:
    return (
        {
            "case_id": "fill-positive-size-limit",
            "operation_id": "fill_holes",
            "shape": (9, 11),
            "dtype": "bool",
            "spatial_ndim": 2,
            "parameters": (
                ("max_hole_size", 4),
                ("spatial_mode", "2D YX"),
                ("connectivity", "Face connected"),
            ),
            "category": "safe_cpu_fallback",
        },
        {
            "case_id": "fill-numeric-nonzero-mask",
            "operation_id": "fill_holes",
            "shape": (9, 11),
            "dtype": "uint8",
            "spatial_ndim": 2,
            "parameters": (
                ("max_hole_size", 0),
                ("spatial_mode", "2D YX"),
                ("connectivity", "Face connected"),
            ),
            "category": "safe_cpu_fallback",
        },
        {
            "case_id": "remove-integer-label-preservation",
            "operation_id": "remove_small_objects",
            "shape": (9, 11),
            "dtype": "int32",
            "spatial_ndim": 2,
            "parameters": (
                ("min_size", 5),
                ("spatial_mode", "2D YX"),
                ("connectivity", "Face connected"),
            ),
            "category": "safe_cpu_fallback",
        },
        {
            "case_id": "fill-invalid-connectivity",
            "operation_id": "fill_holes",
            "shape": (9, 11),
            "dtype": "bool",
            "spatial_ndim": 2,
            "parameters": (
                ("max_hole_size", 0),
                ("spatial_mode", "2D YX"),
                ("connectivity", "Edge connected"),
            ),
            "category": "invalid_authored",
        },
        {
            "case_id": "remove-spatial-rank-disagreement",
            "operation_id": "remove_small_objects",
            "shape": (3, 9, 11),
            "dtype": "bool",
            "spatial_ndim": 2,
            "parameters": (
                ("min_size", 5),
                ("spatial_mode", "3D ZYX"),
                ("connectivity", "Face connected"),
            ),
            "category": "invalid_authored",
        },
        {
            "case_id": "remove-noninteger-size",
            "operation_id": "remove_small_objects",
            "shape": (9, 11),
            "dtype": "bool",
            "spatial_ndim": 2,
            "parameters": (
                ("min_size", 2.5),
                ("spatial_mode", "2D YX"),
                ("connectivity", "Face connected"),
            ),
            "category": "invalid_authored",
        },
    )


def _run_fallback(cpu_functions) -> dict[str, object]:
    record = _expected_fallback_record(cpu_functions)
    if record["status"] != "pass":
        raise EvidenceError("Mask-cleanup fallback evidence failed.")
    return record


def _expected_fallback_record(cpu_functions=None) -> dict[str, object]:
    if cpu_functions is None:
        cpu_functions, _gpu_functions = _operation_functions()
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import evaluate_candidate_workload_support

    safe = []
    invalid = []
    for definition in _fallback_case_definitions():
        workload = WorkloadDescriptor(
            definition["case_id"],
            definition["operation_id"],
            (definition["shape"],),
            (definition["dtype"],),
            parameters=definition["parameters"],
            resolved_spatial_ndim=definition["spatial_ndim"],
        )
        decision = evaluate_candidate_workload_support(
            _gpu_spec(definition["operation_id"]),
            workload,
        )
        item = {
            "case_id": definition["case_id"],
            "operation_id": definition["operation_id"],
            "shape": list(definition["shape"]),
            "input_dtype": definition["dtype"],
            "parameters": _json_value(dict(definition["parameters"])),
            "supported": decision.supported,
            "fallback_allowed": decision.fallback_allowed,
            "reason_text": decision.reason_text,
        }
        if definition["category"] == "safe_cpu_fallback":
            data = _fallback_input(definition)
            output = cpu_functions[definition["operation_id"]](
                data,
                **dict(definition["parameters"]),
            )
            item.update(
                {
                    "cpu_authority_executed": True,
                    "cpu_output_dtype": str(output.dtype),
                    "cpu_output_sha256": _array_sha256(output),
                    "cpu_input_immutable": _array_sha256(data)
                    == _array_sha256(_fallback_input(definition)),
                }
            )
            if decision.supported or not decision.fallback_allowed:
                raise EvidenceError(
                    f"Safe fallback failed for {definition['case_id']}."
                )
            safe.append(item)
        else:
            item["cpu_authority_executed"] = False
            if decision.supported or decision.fallback_allowed:
                raise EvidenceError(
                    f"Invalid authored case was not rejected: {definition['case_id']}."
                )
            invalid.append(item)
    return {
        "status": "pass",
        "safe_cpu_fallback_case_count": len(safe),
        "invalid_authored_case_count": len(invalid),
        "safe_cpu_fallback_cases": safe,
        "invalid_authored_cases": invalid,
    }


def _fallback_input(definition: Mapping[str, object]):
    np = _numpy()
    case_id = str(definition["case_id"])
    shape = tuple(definition["shape"])
    if case_id == "fill-positive-size-limit":
        data = np.ones(shape, dtype=bool)
        data[2, 2] = False
        data[5:7, 6:8] = False
        return data
    if case_id == "fill-numeric-nonzero-mask":
        data = np.ones(shape, dtype=np.uint8)
        data[3:6, 4:7] = 0
        return data
    if case_id == "remove-integer-label-preservation":
        data = np.zeros(shape, dtype=np.int32)
        data[1:3, 1:3] = 7
        data[5:8, 5:9] = 42
        return data
    return np.zeros(shape, dtype=np.dtype(str(definition["dtype"])))


def _performance_cases(profile: str) -> tuple[PerformanceCase, ...]:
    quick = (
        PerformanceCase(
            "fill_holes",
            "fill-checker-3x257x259-odd-padded",
            (3, 257, 259),
            2,
            "2D YX",
            "Face connected",
            0,
            "checkerboard",
        ),
        PerformanceCase(
            "remove_small_objects",
            "remove-checker-3x257x259-odd",
            (3, 257, 259),
            2,
            "2D YX",
            "Face connected",
            17,
            "checkerboard",
        ),
    )
    if profile == "quick":
        return quick
    if profile != "full":
        raise ValueError("profile must be 'quick' or 'full'.")
    return quick + (
        PerformanceCase(
            "fill_holes",
            "fill-volume-2x33x65x67-odd-padded",
            (2, 33, 65, 67),
            3,
            "3D ZYX",
            "Full connectivity",
            0,
            "cavities",
        ),
        PerformanceCase(
            "remove_small_objects",
            "remove-volume-2x33x65x67-odd",
            (2, 33, 65, 67),
            3,
            "3D ZYX",
            "Face connected",
            31,
            "components",
        ),
    )


def _performance_input(definition: PerformanceCase):
    np = _numpy()
    spatial_shape = definition.shape[-definition.spatial_ndim :]
    leading_shape = definition.shape[: -definition.spatial_ndim]
    if definition.pattern == "checkerboard":
        spatial = np.indices(spatial_shape).sum(axis=0) % 2 == 0
        if leading_shape:
            return np.broadcast_to(spatial, definition.shape).copy()
        return spatial
    if definition.pattern == "cavities":
        result = np.ones(definition.shape, dtype=bool)
        result[..., 5:-5:4, 7:-7:6, 9:-9:8] = False
        result[..., 0:8, 3, 3] = False
        return result
    if definition.pattern == "components":
        result = np.zeros(definition.shape, dtype=bool)
        result[..., 2:-2:5, 3:-3:7, 4:-4:9] = True
        result[..., 8:18, 12:24, 16:30] = True
        return result
    raise ValueError(f"Unknown performance pattern {definition.pattern!r}.")


def _performance_parameters(definition: PerformanceCase) -> dict[str, object]:
    parameters = {
        "spatial_mode": definition.spatial_mode,
        "connectivity": definition.connectivity,
    }
    if definition.operation_id == "fill_holes":
        parameters["max_hole_size"] = definition.size_parameter
    else:
        parameters["min_size"] = definition.size_parameter
    return parameters


def _run_performance_case(cp, cpu_function, gpu_function, definition, rounds):
    host = _performance_input(definition)
    parameters = _performance_parameters(definition)
    cpu_seconds = []
    for _ in range(rounds):
        started = time.perf_counter()
        expected = cpu_function(host, **parameters)
        cpu_seconds.append(time.perf_counter() - started)

    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device = cp.asarray(host)
        warm = gpu_function(device, **parameters)
        cp.cuda.get_current_stream().synchronize()
        del warm
        resident_seconds = []
        for _ in range(rounds):
            started = time.perf_counter()
            output = gpu_function(device, **parameters)
            cp.cuda.get_current_stream().synchronize()
            resident_seconds.append(time.perf_counter() - started)
            del output
        transfer_seconds = []
        final_host = None
        for _ in range(rounds):
            started = time.perf_counter()
            transfer_input = cp.asarray(host)
            transfer_output = gpu_function(transfer_input, **parameters)
            final_host = cp.asnumpy(transfer_output)
            cp.cuda.get_current_stream().synchronize()
            transfer_seconds.append(time.perf_counter() - started)
            del transfer_output, transfer_input
        device_after = cp.asnumpy(device)
        input_immutable = _array_sha256(device_after) == _array_sha256(host)
        del device_after, device
        cp.cuda.get_current_stream().synchronize()
    timing_cleanup = _drain_pool(cp, pool)
    if (
        final_host is None
        or not _numpy().array_equal(final_host, expected)
        or not input_immutable
    ):
        raise EvidenceError(f"Performance parity failed for {definition.case_id}.")

    memory = _measure_memory(cp, gpu_function, host, definition)
    return {
        "operation_id": definition.operation_id,
        "case_id": definition.case_id,
        "shape": list(definition.shape),
        "spatial_ndim": definition.spatial_ndim,
        "spatial_block_elements": math.prod(
            definition.shape[-definition.spatial_ndim :]
        ),
        "padded_spatial_block_elements": math.prod(
            size + 2 for size in definition.shape[-definition.spatial_ndim :]
        ),
        "input_dtype": "bool",
        "element_count": math.prod(definition.shape),
        "input_bytes": int(host.nbytes),
        "output_bytes": int(expected.nbytes),
        "pattern": definition.pattern,
        "parameters": _json_value(parameters),
        "parity": {
            "passed": True,
            "policy": "mask-bitwise-v1",
            "cpu_output_sha256": _array_sha256(expected),
            "gpu_output_sha256": _array_sha256(final_host),
            "input_immutable": input_immutable,
        },
        "samples": {
            "cpu_seconds": cpu_seconds,
            "gpu_resident_seconds": resident_seconds,
            "gpu_transfer_inclusive_seconds": transfer_seconds,
        },
        "summary": _timing_summary(cpu_seconds, resident_seconds, transfer_seconds),
        "memory": memory,
        "cleanup": timing_cleanup,
    }


def _measure_memory(cp, gpu_function, host, definition) -> dict[str, object]:
    estimate = _performance_memory_estimate(definition)
    parameters = _performance_parameters(definition)
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device = cp.asarray(host)
        used_with_input = int(pool.used_bytes())
        output = gpu_function(device, **parameters)
        cp.cuda.get_current_stream().synchronize()
        used_with_output = int(pool.used_bytes())
        observed_reserved = int(pool.total_bytes())
        output_independent = not output.size or int(output.data.ptr) != int(
            device.data.ptr
        )
        del output, device
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    estimated_with_uncertainty = (
        estimate.total_device_peak_bytes + estimate.uncertainty_bytes
    )
    covered = estimated_with_uncertainty >= observed_reserved
    if not output_independent:
        raise EvidenceError("Mask cleanup did not allocate an independent output.")
    return {
        "scope": "isolated-cupy-memory-pool-reserved-high-water-v1",
        "model_id": estimate.model_id,
        "runtime_managed_peak_bytes": estimate.runtime_managed_peak_bytes,
        "total_device_peak_bytes": estimate.total_device_peak_bytes,
        "uncertainty_bytes": estimate.uncertainty_bytes,
        "estimated_peak_with_uncertainty_bytes": estimated_with_uncertainty,
        "observed_reserved_bytes": observed_reserved,
        "observed_used_bytes_with_input": used_with_input,
        "observed_used_bytes_with_output": used_with_output,
        "output_shares_input_allocation": not output_independent,
        "estimate_covers_observed": covered,
        **cleanup,
    }


def _performance_memory_estimate(definition: PerformanceCase):
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import estimate_candidate_memory

    workload = WorkloadDescriptor(
        definition.case_id,
        definition.operation_id,
        (definition.shape,),
        ("bool",),
        parameters=tuple(_performance_parameters(definition).items()),
        resolved_spatial_ndim=definition.spatial_ndim,
    )
    return estimate_candidate_memory(_gpu_spec(definition.operation_id), workload)


def _timing_summary(cpu, resident, transfer) -> dict[str, float]:
    return {
        "cpu_median_seconds": statistics.median(cpu),
        "gpu_resident_median_seconds": statistics.median(resident),
        "gpu_transfer_inclusive_median_seconds": statistics.median(transfer),
    }


def _warm_runtime(cp, gpu_functions) -> None:
    fill_source = cp.ones((2, 9, 11), dtype=bool)
    fill_source[:, 4, 5] = False
    filled = gpu_functions["fill_holes"](
        fill_source,
        max_hole_size=0,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )
    remove_source = cp.zeros((2, 9, 11), dtype=bool)
    remove_source[:, 2:5, 3:7] = True
    removed = gpu_functions["remove_small_objects"](
        remove_source,
        min_size=5,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )
    cp.cuda.get_current_stream().synchronize()
    del removed, remove_source, filled, fill_source
    cp.get_default_memory_pool().free_all_blocks()


def _drain_pool(cp, pool) -> dict[str, int]:
    cp.cuda.get_current_stream().synchronize()
    gc.collect()
    pool.free_all_blocks()
    cp.cuda.get_current_stream().synchronize()
    used = int(pool.used_bytes())
    reserved = int(pool.total_bytes())
    if used or reserved:
        raise EvidenceError(
            f"CUDA private-pool cleanup failed: used={used}, reserved={reserved}."
        )
    return {
        "device_pool_used_bytes_after_cleanup": used,
        "device_pool_reserved_bytes_after_cleanup": reserved,
    }


def _method_record(profile: str, rounds: int) -> dict[str, object]:
    return {
        "profile": profile,
        "admission_repeats": ADMISSION_REPEATS,
        "benchmark_rounds": rounds,
        "public_region": "bool-mask-full-hole-fill-and-small-object-removal-v1",
        "parity": "mask-bitwise-v1",
        "gpu_resident_timing_scope": "synchronized-resident-operation-v1",
        "gpu_transfer_inclusive_timing_scope": (
            "host-to-device-plus-operation-plus-device-to-host-synchronized-v1"
        ),
        "memory_observation_scope": (
            "isolated-cupy-memory-pool-reserved-high-water-v1"
        ),
        "cancellation": "synchronized-spatial-block-boundaries-v1",
        "memory_adversaries": "odd-padded-and-checkerboard-spatial-blocks-v1",
    }


def _environment_record(cp, device_index: int) -> dict[str, object]:
    properties = cp.cuda.runtime.getDeviceProperties(device_index)
    name = properties.get("name", b"")
    if isinstance(name, bytes):
        name = name.decode(errors="replace").rstrip("\x00")
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python_abi": sys.implementation.cache_tag,
        "python": platform.python_version(),
        "python_executable": Path(sys.executable).name,
        "device_index": device_index,
        "device_name": str(name),
        "compute_capability": (
            f"{int(properties.get('major', 0))}.{int(properties.get('minor', 0))}"
        ),
        "device_total_memory_bytes": int(properties.get("totalGlobalMem", 0)),
        "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
    }


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


def _project_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream).get("project", {})
    version = str(project.get("version", "")).strip()
    if not version:
        raise EvidenceError("pyproject.toml has no project version.")
    return version


def _environment_packages_sha256(
    environment: Mapping[str, object],
    packages: Mapping[str, object],
) -> str:
    payload = {"environment": dict(environment), "packages": dict(packages)}
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_environment_and_packages(document: Mapping[str, object]) -> None:
    environment = _mapping(document.get("environment"), "environment")
    packages = _mapping(document.get("packages"), "packages")
    if set(environment) != _ENVIRONMENT_KEYS:
        raise EvidenceError("Environment record differs from the canonical schema.")
    if set(packages) != _PACKAGE_KEYS:
        raise EvidenceError("Package record differs from the canonical schema.")
    expected_binding = _environment_packages_sha256(environment, packages)
    if document.get("environment_packages_sha256") != expected_binding:
        raise EvidenceError("Environment/package evidence binding is invalid.")

    def nonempty_string(name: str) -> str:
        value = environment.get(name)
        if not isinstance(value, str) or not value.strip():
            raise EvidenceError(f"Environment field {name!r} is invalid.")
        return value.strip()

    if nonempty_string("system") != "Windows":
        raise EvidenceError("Mask-cleanup evidence requires native Windows.")
    nonempty_string("release")
    if nonempty_string("machine").lower() not in {"amd64", "x86_64"}:
        raise EvidenceError("Mask-cleanup evidence requires Windows x86-64.")
    if nonempty_string("python_implementation") != "CPython":
        raise EvidenceError("Mask-cleanup evidence requires CPython.")
    if nonempty_string("python_abi") != "cpython-312":
        raise EvidenceError("Mask-cleanup evidence requires cpython-312 ABI.")
    python_parts = nonempty_string("python").split(".")
    if (
        len(python_parts) != 3
        or python_parts[:2] != ["3", "12"]
        or not python_parts[2].isdigit()
    ):
        raise EvidenceError("Mask-cleanup evidence requires CPython 3.12.x.")
    if nonempty_string("python_executable").lower() != "python.exe":
        raise EvidenceError("The evidence Python executable is not native Windows.")
    device_index = environment.get("device_index")
    if (
        isinstance(device_index, bool)
        or not isinstance(device_index, int)
        or device_index < 0
    ):
        raise EvidenceError("The CUDA device index is invalid.")
    nonempty_string("device_name")
    try:
        capability = tuple(
            int(value) for value in nonempty_string("compute_capability").split(".")
        )
    except ValueError as exc:
        raise EvidenceError("The CUDA compute capability is invalid.") from exc
    if len(capability) != 2 or capability < (7, 5):
        raise EvidenceError("The CUDA compute capability is outside admission.")
    for name in (
        "device_total_memory_bytes",
        "cuda_driver_version",
        "cuda_runtime_version",
    ):
        value = environment.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise EvidenceError(f"Environment field {name!r} is invalid.")
    if environment["cuda_driver_version"] < 13030:
        raise EvidenceError("The CUDA driver is outside the admitted environment.")
    if environment["cuda_runtime_version"] != 13020:
        raise EvidenceError("The CUDA runtime differs from the exact admitted pin.")

    expected_packages = {
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "scikit-image": "0.26.0",
        "cupy": "14.1.1",
        "cupy-cuda13x": "14.1.1",
        "napari-vipp": _project_version(),
    }
    if dict(packages) != expected_packages:
        raise EvidenceError("Package versions differ from the exact admitted pins.")


def _source_provenance() -> dict[str, object]:
    files = []
    for relative in SOURCE_PROVENANCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise EvidenceError(f"Required source file is missing: {relative}.")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    return {"files": files, "git": _git_provenance()}


def _git_provenance() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--untracked-files=normal",
                    "--",
                    *(path.as_posix() for path in SOURCE_PROVENANCE_PATHS),
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "commit": "", "worktree_dirty": None}
    return {"available": True, "commit": commit, "worktree_dirty": dirty}


def _operation_contracts() -> dict[str, object]:
    records = {}
    for operation_id in sorted(IMPLEMENTATION_IDS):
        snapshot = _canonical_value(_gpu_spec(operation_id))
        encoded = json.dumps(
            snapshot,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        records[operation_id] = {
            "snapshot": snapshot,
            "sha256": hashlib.sha256(encoded).hexdigest(),
        }
    return records


def _gpu_spec(operation_id: str):
    from napari_vipp.core.compute_specs import accelerator_compute_specs

    implementation_id = IMPLEMENTATION_IDS[operation_id]
    matches = [
        spec
        for spec in accelerator_compute_specs()
        if spec.operation_id == operation_id
        and spec.implementation_id == implementation_id
    ]
    if len(matches) != 1:
        raise EvidenceError(
            f"Expected one {operation_id}/{implementation_id} compute contract."
        )
    spec = matches[0]
    if (
        spec.runtime_id != RUNTIME_ID
        or spec.implementation_library_id != LIBRARY_ID
        or spec.validated_environment_policy_id != ENVIRONMENT_POLICY_ID
    ):
        raise EvidenceError(f"{operation_id} runtime/environment identity is stale.")
    return spec


def _implementation_records() -> list[dict[str, str]]:
    return [
        {
            "operation_id": operation_id,
            "implementation_id": IMPLEMENTATION_IDS[operation_id],
            "implementation_version": "1",
            "runtime_id": RUNTIME_ID,
            "library_id": LIBRARY_ID,
            "validated_environment_policy_id": ENVIRONMENT_POLICY_ID,
        }
        for operation_id in sorted(IMPLEMENTATION_IDS)
    ]


def _implementation_key(operation_id: str) -> str:
    return f"{operation_id}::{IMPLEMENTATION_IDS[operation_id]}"


def _canonical_value(value):
    if dataclasses.is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return _canonical_value(value.value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (tuple, list)):
        return [_canonical_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise EvidenceError(f"Operation contract contains unsupported value {value!r}.")


def _operation_functions():
    from napari_vipp.core.operations import fill_holes, remove_small_objects

    fill_module = importlib.import_module("napari_vipp.core.gpu.cupy_fill_holes")
    remove_module = importlib.import_module(
        "napari_vipp.core.gpu.cupy_remove_small_objects"
    )
    return (
        {
            "fill_holes": fill_holes,
            "remove_small_objects": remove_small_objects,
        },
        {
            "fill_holes": fill_module.fill_holes,
            "remove_small_objects": remove_module.remove_small_objects,
        },
    )


def _numpy():
    return importlib.import_module("numpy")


def _cupy():
    try:
        return importlib.import_module("cupy")
    except (ImportError, OSError) as exc:
        raise EvidenceError(f"CuPy is unavailable: {exc}") from exc


def _array_sha256(array) -> str:
    contiguous = _numpy().ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _json_value(value):
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def validate_existing(path: Path | str) -> Path:
    artifact = Path(path).resolve(strict=True)
    raw = artifact.read_text(encoding="utf-8")
    document = json.loads(raw)
    if raw != _canonical_json(document):
        raise EvidenceError("Evidence JSON is not canonical sorted, indented JSON.")
    _validate_document(document, require_current_sources=True)
    return artifact


def _validate_identity_claims(document: Mapping[str, object]) -> None:
    created_utc = document.get("created_utc")
    try:
        created_at = (
            datetime.fromisoformat(created_utc)
            if isinstance(created_utc, str)
            else None
        )
    except ValueError as exc:
        raise EvidenceError("Evidence creation timestamp is invalid.") from exc
    if (
        document.get("schema") != SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("kind") != EVIDENCE_KIND
        or created_at is None
        or created_at.tzinfo is None
        or created_at.utcoffset() != UTC.utcoffset(created_at)
        or document.get("profile") not in {"quick", "full"}
        or document.get("portable_performance_claim") is not False
        or document.get("durable_optimizer_record") is not False
        or document.get("implementations") != _implementation_records()
    ):
        raise EvidenceError("Evidence identity/profile claims are invalid.")


def _validate_document(document: object, *, require_current_sources: bool) -> None:
    if not isinstance(document, Mapping) or set(document) != _ROOT_KEYS:
        raise EvidenceError("Evidence root differs from the canonical schema.")
    _validate_identity_claims(document)
    expected_facets = {
        _implementation_key(operation_id): {facet: "pass" for facet in REQUIRED_FACETS}
        for operation_id in sorted(IMPLEMENTATION_IDS)
    }
    if document.get("facets") != expected_facets:
        raise EvidenceError("Required facet status is incomplete.")
    _validate_environment_and_packages(document)
    if require_current_sources:
        if document.get("source_provenance") != _source_provenance():
            raise EvidenceError("Source provenance fingerprints are stale.")
        if document.get("operation_contracts") != _operation_contracts():
            raise EvidenceError("Operation contract fingerprints are stale.")

    _validate_admission_records(document.get("admission"))
    _validate_metadata_records(document.get("metadata"))
    lifecycle = _mapping(document.get("lifecycle"), "lifecycle")
    for operation_id, message in {
        "fill_holes": "Fill-hole blocks",
        "remove_small_objects": "Small-object blocks",
    }.items():
        life = _mapping(lifecycle.get(operation_id), "lifecycle item")
        reports = life.get("reported_progress")
        if (
            life.get("status") != "pass"
            or life.get("boundary") != "synchronized-spatial-block-boundary-v1"
            or life.get("cancellation_requested") is not True
            or life.get("cancellation_observed") is not True
            or not isinstance(reports, list)
            or reports[:2]
            != [
                {"current": 0, "total": 3, "message": message},
                {"current": 1, "total": 3, "message": message},
            ]
            or len(reports) != 2
            or life.get("expected_progress_message") != message
            or life.get("post_cancellation_reuse_exact") is not True
            or life.get("input_immutable") is not True
            or not _cleanup_passed(life)
        ):
            raise EvidenceError(f"{operation_id} lifecycle evidence is invalid.")

    if document.get("fallback") != _expected_fallback_record():
        raise EvidenceError("Fallback evidence is invalid.")
    _validate_performance(document)
    profile = str(document["profile"])
    rounds = FULL_ROUNDS if profile == "full" else QUICK_ROUNDS
    if document.get("method") != _method_record(profile, rounds):
        raise EvidenceError("Method record is inconsistent with the profile.")


def _validate_admission_records(value: object) -> None:
    admission = _mapping(value, "admission")
    for operation_id in sorted(IMPLEMENTATION_IDS):
        section = _mapping(admission.get(operation_id), operation_id)
        definitions = [
            case for case in _admission_cases() if case.operation_id == operation_id
        ]
        expected_coverage = {
            item for definition in definitions for item in definition.coverage
        } | {"repeat:deterministic"}
        cases = section.get("cases")
        if (
            section.get("status") != "pass"
            or not isinstance(cases, list)
            or [item.get("case_id") for item in cases]
            != [definition.case_id for definition in definitions]
            or section.get("case_count") != len(cases)
            or section.get("repeat_count") != ADMISSION_REPEATS
            or section.get("coverage") != sorted(expected_coverage)
            or not REQUIRED_ADMISSION_COVERAGE[operation_id] <= expected_coverage
        ):
            raise EvidenceError(f"{operation_id} admission aggregate is invalid.")
        for item, definition in zip(cases, definitions, strict=True):
            case = _mapping(item, "admission case")
            host = _host_case(definition.kind)
            parameters = _case_parameters(definition.kind)
            if (
                case.get("shape") != list(host.shape)
                or case.get("input_dtype") != "bool"
                or case.get("output_dtype") != "bool"
                or case.get("parameters") != _json_value(parameters)
                or case.get("coverage") != sorted(definition.coverage)
                or case.get("input_sha256") != _array_sha256(host)
                or case.get("output_integrity_contract")
                != "independent-bool-allocation-v1"
                or case.get("cpu_gpu_bitwise_equal") is not True
                or case.get("gpu_output_resident") is not True
                or case.get("input_immutable") is not True
                or case.get("repeat_deterministic") is not True
                or case.get("repeat_count") != ADMISSION_REPEATS
                or case.get("cpu_output_sha256") != case.get("gpu_output_sha256")
                or not _cleanup_passed(_mapping(case.get("cleanup"), "cleanup"))
            ):
                raise EvidenceError("Admission parity/integrity evidence is invalid.")


def _validate_metadata_records(value: object) -> None:
    metadata = _mapping(value, "metadata")
    if metadata != _expected_metadata_records():
        raise EvidenceError("Mask-cleanup metadata evidence is invalid.")


def _validate_performance(document: Mapping[str, object]) -> None:
    performance = _mapping(document.get("performance"), "performance")
    profile = str(document["profile"])
    definitions = _performance_cases(profile)
    rounds = FULL_ROUNDS if profile == "full" else QUICK_ROUNDS
    results = performance.get("results")
    if (
        performance.get("status") != "pass"
        or performance.get("rounds") != rounds
        or not isinstance(results, list)
        or performance.get("case_count") != len(results)
        or [item.get("case_id") for item in results]
        != [item.case_id for item in definitions]
        or performance.get("all_memory_estimates_cover_observed") is not True
    ):
        raise EvidenceError("Performance aggregate is invalid.")
    for result, definition in zip(results, definitions, strict=True):
        item = _mapping(result, "performance case")
        parity = _mapping(item.get("parity"), "parity")
        samples = _mapping(item.get("samples"), "samples")
        memory = _mapping(item.get("memory"), "memory")
        estimate = _performance_memory_estimate(definition)
        expected_peak = estimate.total_device_peak_bytes + estimate.uncertainty_bytes
        observed = memory.get("observed_reserved_bytes")
        if (
            item.get("operation_id") != definition.operation_id
            or item.get("shape") != list(definition.shape)
            or item.get("spatial_ndim") != definition.spatial_ndim
            or item.get("input_dtype") != "bool"
            or item.get("pattern") != definition.pattern
            or item.get("parameters")
            != _json_value(_performance_parameters(definition))
            or item.get("spatial_block_elements")
            != math.prod(definition.shape[-definition.spatial_ndim :])
            or item.get("padded_spatial_block_elements")
            != math.prod(
                size + 2 for size in definition.shape[-definition.spatial_ndim :]
            )
            or parity.get("passed") is not True
            or parity.get("policy") != "mask-bitwise-v1"
            or parity.get("cpu_output_sha256") != parity.get("gpu_output_sha256")
            or parity.get("input_immutable") is not True
            or memory.get("model_id") != estimate.model_id
            or memory.get("runtime_managed_peak_bytes")
            != estimate.runtime_managed_peak_bytes
            or memory.get("total_device_peak_bytes") != estimate.total_device_peak_bytes
            or memory.get("uncertainty_bytes") != estimate.uncertainty_bytes
            or memory.get("estimated_peak_with_uncertainty_bytes") != expected_peak
            or isinstance(observed, bool)
            or not isinstance(observed, int)
            or observed < 0
            or observed > expected_peak
            or memory.get("estimate_covers_observed") is not True
            or memory.get("output_shares_input_allocation") is not False
            or not _cleanup_passed(memory)
            or not _cleanup_passed(_mapping(item.get("cleanup"), "cleanup"))
        ):
            raise EvidenceError("Performance parity/memory evidence is invalid.")
        for name in (
            "cpu_seconds",
            "gpu_resident_seconds",
            "gpu_transfer_inclusive_seconds",
        ):
            values = samples.get(name)
            if (
                not isinstance(values, list)
                or len(values) != rounds
                or any(not _positive_finite(value) for value in values)
            ):
                raise EvidenceError(f"{name} timing samples are invalid.")
        if item.get("summary") != _timing_summary(
            samples["cpu_seconds"],
            samples["gpu_resident_seconds"],
            samples["gpu_transfer_inclusive_seconds"],
        ):
            raise EvidenceError("Timing summary is inconsistent with samples.")


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{name} must be an object.")
    return value


def _cleanup_passed(value: Mapping[str, object]) -> bool:
    return (
        value.get("device_pool_used_bytes_after_cleanup") == 0
        and value.get("device_pool_reserved_bytes_after_cleanup") == 0
    )


def _positive_finite(value: object) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number > 0


def _canonical_json(document: object) -> str:
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(_canonical_json(document))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
