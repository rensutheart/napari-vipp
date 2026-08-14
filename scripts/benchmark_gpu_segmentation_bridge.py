#!/usr/bin/env python
"""Qualify VIPP's Extract Channel and Binary Threshold CUDA bridge.

The combined evidence owner follows the first useful resident segmentation
corridor across a zero-allocation semantic channel view and an exact float32
fixed threshold.  It records all ten public-admission facets, including safe
CPU fallback and both resident and transfer-inclusive timings.

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

SCHEMA = "napari-vipp-cupy-segmentation-bridge-evidence"
SCHEMA_VERSION = 1
EVIDENCE_KIND = "scientific-admission-and-machine-local-performance-evidence"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/benchmarks/segmentation-bridge-cupy-local.json"
ADMISSION_REPEATS = 3
QUICK_ROUNDS = 3
FULL_ROUNDS = 7
RUNTIME_ID = "cuda-cupy"
LIBRARY_ID = "cupy"
ENVIRONMENT_POLICY_ID = "cuda-cupy-core-14.1.1-cpython312-windows-native-v1"
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
    "binary_threshold": "cupy-binary-threshold-f32-exact-v1",
    "extract_channel": "cupy-extract-channel-view-v1",
}
SOURCE_PROVENANCE_PATHS = (
    Path("src/napari_vipp/core/operations.py"),
    Path("src/napari_vipp/core/metadata.py"),
    Path("src/napari_vipp/core/gpu/cupy_binary_threshold.py"),
    Path("src/napari_vipp/core/gpu/cupy_extract_channel.py"),
    Path("src/napari_vipp/core/compute_specs.py"),
    Path("src/napari_vipp/core/compute_policy.py"),
    Path("src/napari_vipp/core/compute_planning.py"),
    Path("src/napari_vipp/core/execution.py"),
    Path("src/napari_vipp/core/compute_benchmark_adapter.py"),
    Path("scripts/benchmark_gpu_segmentation_bridge.py"),
)
REQUIRED_ADMISSION_COVERAGE = {
    "binary_threshold": frozenset(
        {
            "dtype:float32",
            "rank:2",
            "rank:3",
            "layout:contiguous",
            "layout:noncontiguous",
            "values:nonfinite",
            "values:signed-zero",
            "threshold:float32-rounding-boundary",
            "repeat:deterministic",
        }
    ),
    "extract_channel": frozenset(
        {
            "dtype:bool",
            "dtype:uint8",
            "dtype:uint16",
            "dtype:float32",
            "rank:3",
            "rank:4",
            "rank:5",
            "layout:contiguous",
            "layout:noncontiguous",
            "axis:semantic-name",
            "axis:semantic-type",
            "axis:type-precedence",
            "channel:negative-index",
            "output:allocation-sharing-view",
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
    """Raised when bridge evidence is incomplete or inconsistent."""


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
    dtype: str
    parameters: tuple[tuple[str, object], ...]


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
            print(f"Segmentation bridge evidence is current: {validated}")
            return 0
        if args.device_index < 0:
            raise ValueError("device index must be nonnegative")
        document = build_evidence(args.profile, args.device_index)
        _atomic_write_json(args.output.resolve(), document)
        validate_existing(args.output.resolve())
    except (EvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"Segmentation bridge evidence failed: {exc}", file=sys.stderr)
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
    environment_record = _environment_record(cp, device_index)
    package_record = _package_record(cp, np)
    environment_packages_sha256 = _environment_packages_sha256(
        environment_record,
        package_record,
    )
    _validate_environment_and_packages(
        {
            "environment": environment_record,
            "packages": package_record,
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
        raise EvidenceError("A bridge operation contract changed during collection.")

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
        raise EvidenceError("A declared memory estimate did not cover observation.")

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
        "environment": environment_record,
        "packages": package_record,
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
            "binary_threshold",
            "binary-boundaries-plane",
            "binary-boundaries",
            (
                "dtype:float32",
                "rank:2",
                "layout:contiguous",
                "values:signed-zero",
                "threshold:float32-rounding-boundary",
            ),
        ),
        AdmissionCase(
            "binary_threshold",
            "binary-nonfinite-strided",
            "binary-nonfinite-strided",
            (
                "dtype:float32",
                "rank:2",
                "layout:noncontiguous",
                "values:nonfinite",
                "values:signed-zero",
            ),
        ),
        AdmissionCase(
            "binary_threshold",
            "binary-volume",
            "binary-volume",
            ("dtype:float32", "rank:3", "layout:contiguous"),
        ),
        AdmissionCase(
            "extract_channel",
            "extract-bool-names",
            "extract-bool-names",
            (
                "dtype:bool",
                "rank:3",
                "layout:contiguous",
                "axis:semantic-name",
                "output:allocation-sharing-view",
            ),
        ),
        AdmissionCase(
            "extract_channel",
            "extract-u8-types",
            "extract-u8-types",
            (
                "dtype:uint8",
                "rank:4",
                "layout:contiguous",
                "axis:semantic-type",
                "output:allocation-sharing-view",
            ),
        ),
        AdmissionCase(
            "extract_channel",
            "extract-u16-negative-strided",
            "extract-u16-negative-strided",
            (
                "dtype:uint16",
                "rank:4",
                "layout:noncontiguous",
                "axis:semantic-name",
                "channel:negative-index",
                "output:allocation-sharing-view",
            ),
        ),
        AdmissionCase(
            "extract_channel",
            "extract-f32-type-precedence",
            "extract-f32-type-precedence",
            (
                "dtype:float32",
                "rank:5",
                "layout:contiguous",
                "axis:semantic-name",
                "axis:semantic-type",
                "axis:type-precedence",
                "output:allocation-sharing-view",
            ),
        ),
    )


def _host_case(kind: str):
    np = _numpy()
    if kind == "binary-boundaries":
        return np.asarray(
            [
                [-0.0, 0.0, 1.0, np.nextafter(np.float32(1), np.float32(2))],
                [-1.0, 0.5, 2.0, 7.0],
            ],
            dtype=np.float32,
        )
    if kind == "binary-nonfinite-strided":
        base = np.asarray(
            [
                [np.nan, -np.inf, -0.0, 0.0, np.inf, 1.0],
                [2.0, 0.25, 0.5, 0.75, -1.0, np.nan],
                [3.0, -3.0, np.inf, -np.inf, 0.0, -0.0],
                [4.0, 2.0, 1.0, 0.5, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        return base[::-1, 1::2]
    if kind == "binary-volume":
        return np.linspace(-2.0, 3.0, 3 * 17 * 19, dtype=np.float32).reshape(3, 17, 19)
    if kind == "extract-bool-names":
        return (np.arange(3 * 11 * 13).reshape(3, 11, 13) % 3) == 0
    if kind == "extract-u8-types":
        return (
            np.arange(2 * 4 * 9 * 11, dtype=np.uint32).reshape(2, 4, 9, 11) % 251
        ).astype(np.uint8)
    if kind == "extract-u16-negative-strided":
        base = np.arange(5 * 4 * 14 * 18, dtype=np.uint16).reshape(5, 4, 14, 18)
        return base[:, :, ::2, 1::2]
    if kind == "extract-f32-type-precedence":
        result = np.linspace(
            -5.0,
            9.0,
            2 * 3 * 4 * 7 * 9,
            dtype=np.float32,
        ).reshape(2, 3, 4, 7, 9)
        result[0, 1, 2, 3, 4] = np.nan
        result[1, 2, 3, 4, 5] = np.inf
        return result
    raise ValueError(f"Unknown admission case kind {kind!r}.")


def _case_parameters(kind: str) -> dict[str, object]:
    if kind == "binary-boundaries":
        return {"threshold": 1.00000008, "channel_axis": None}
    if kind == "binary-nonfinite-strided":
        return {"threshold": 0.0, "channel_axis": None}
    if kind == "binary-volume":
        return {"threshold": 0.75, "channel_axis": None}
    if kind == "extract-bool-names":
        return {"channel": 1, "axis_names": ("c", "y", "x"), "axis_types": ()}
    if kind == "extract-u8-types":
        return {
            "channel": 2,
            "axis_names": (),
            "axis_types": ("time", "channel", "space", "space"),
        }
    if kind == "extract-u16-negative-strided":
        return {
            "channel": -1,
            "axis_names": ("t", "c", "y", "x"),
            "axis_types": (),
        }
    if kind == "extract-f32-type-precedence":
        return {
            "channel": 2,
            "axis_names": ("c", "z", "y", "x", "t"),
            "axis_types": ("time", "channel", "space", "space", "space"),
        }
    raise ValueError(f"Unknown admission case kind {kind!r}.")


def _device_case(cp, host, kind: str):
    """Recreate authored non-contiguous layouts in the resident domain."""

    np = _numpy()
    if kind == "binary-nonfinite-strided":
        base = np.asarray(
            [
                [np.nan, -np.inf, -0.0, 0.0, np.inf, 1.0],
                [2.0, 0.25, 0.5, 0.75, -1.0, np.nan],
                [3.0, -3.0, np.inf, -np.inf, 0.0, -0.0],
                [4.0, 2.0, 1.0, 0.5, 0.0, -1.0],
            ],
            dtype=np.float32,
        )
        keepalive = cp.asarray(base)
        return keepalive[::-1, 1::2], keepalive
    if kind == "extract-u16-negative-strided":
        base = np.arange(5 * 4 * 14 * 18, dtype=np.uint16).reshape(5, 4, 14, 18)
        keepalive = cp.asarray(base)
        return keepalive[:, :, ::2, 1::2], keepalive
    device = cp.asarray(host)
    return device, None


def _run_admission(cp, cpu_functions, gpu_functions) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = {
        operation_id: [] for operation_id in IMPLEMENTATION_IDS
    }
    coverage: dict[str, set[str]] = {
        operation_id: {"repeat:deterministic"} for operation_id in IMPLEMENTATION_IDS
    }
    for definition in _admission_cases():
        host = _host_case(definition.kind)
        host.setflags(write=False)
        parameters = _case_parameters(definition.kind)
        expected = cpu_functions[definition.operation_id](host, **parameters)
        input_hash = _array_sha256(host)
        pool = cp.cuda.MemoryPool()
        output_hashes: list[str] = []
        output_resident = True
        integrity_contract = True
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
                if definition.operation_id == "binary_threshold":
                    integrity_contract &= output.dtype == cp.bool_ and (
                        not output.size or int(output.data.ptr) != int(device.data.ptr)
                    )
                else:
                    integrity_contract &= output.dtype == device.dtype and int(
                        output.data.mem.ptr
                    ) == int(device.data.mem.ptr)
                actual = cp.asnumpy(output)
                if not _numpy().array_equal(actual, expected, equal_nan=True):
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
            or not integrity_contract
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
                "cpu_gpu_bitwise_equal": True,
                "gpu_output_resident": True,
                "input_immutable": True,
                "output_integrity_contract": (
                    "independent-bool-allocation-v1"
                    if definition.operation_id == "binary_threshold"
                    else "shared-input-allocation-view-v1"
                ),
                "repeat_count": ADMISSION_REPEATS,
                "repeat_deterministic": True,
                "input_sha256": input_hash,
                "cpu_output_sha256": _array_sha256(expected),
                "gpu_output_sha256": output_hashes[0],
                "cleanup": cleanup,
            }
        )
    result: dict[str, object] = {}
    for operation_id, cases in grouped.items():
        missing = REQUIRED_ADMISSION_COVERAGE[operation_id] - coverage[operation_id]
        if missing:
            raise EvidenceError(
                f"{operation_id} admission coverage is incomplete: {sorted(missing)}."
            )
        result[operation_id] = {
            "status": "pass",
            "case_count": len(cases),
            "repeat_count": ADMISSION_REPEATS,
            "coverage": sorted(coverage[operation_id]),
            "cases": cases,
        }
    return result


def _run_metadata(cpu_functions) -> dict[str, object]:
    np = _numpy()
    from napari_vipp.core.metadata import (
        ChannelMetadata,
        image_state_from_array,
        transform_image_state,
    )

    binary_source = np.linspace(-1, 1, 3 * 11 * 13, dtype=np.float32).reshape(3, 11, 13)
    binary_state = image_state_from_array(
        binary_source,
        layer_metadata={"axes": "ZYX"},
        source_name="segmentation-bridge-admission",
        metadata_source="admission-authored axes",
        history=("Synthetic qualification source",),
    )
    binary_output = cpu_functions["binary_threshold"](
        binary_source,
        threshold=0.25,
    )
    binary_transformed = transform_image_state(
        binary_output,
        binary_state,
        operation_id="binary_threshold",
        operation_title="Binary Threshold",
        params={"threshold": 0.25, "channel_axis": None},
    )

    extract_source = np.arange(3 * 11 * 13, dtype=np.uint16).reshape(3, 11, 13)
    extract_state = image_state_from_array(
        extract_source,
        layer_metadata={"axes": "CYX"},
        channels=tuple(ChannelMetadata(name=name) for name in ("R", "G", "B")),
        source_name="segmentation-bridge-admission",
        metadata_source="admission-authored axes",
        history=("Synthetic qualification source",),
    )
    extract_parameters = {
        "channel": 1,
        "axis_names": ("c", "y", "x"),
        "axis_types": (),
    }
    extract_output = cpu_functions["extract_channel"](
        extract_source,
        **extract_parameters,
    )
    extract_transformed = transform_image_state(
        extract_output,
        extract_state,
        operation_id="extract_channel",
        operation_title="Extract Channel",
        params=extract_parameters,
    )
    if any(
        item is None
        for item in (
            binary_state,
            binary_transformed,
            extract_state,
            extract_transformed,
        )
    ):
        raise EvidenceError("Bridge metadata state was not produced.")
    binary_passed = all(
        (
            binary_transformed.shape == binary_state.shape,
            binary_transformed.dtype == "bool",
            binary_transformed.axes == binary_state.axes,
            binary_transformed.source == binary_state.source,
            binary_transformed.acquisition == binary_state.acquisition,
            len(binary_transformed.history) == len(binary_state.history) + 1,
        )
    )
    extract_passed = all(
        (
            extract_transformed.shape == (11, 13),
            extract_transformed.dtype == "uint16",
            extract_transformed.axis_order == "YX",
            extract_transformed.source == extract_state.source,
            extract_transformed.acquisition == extract_state.acquisition,
            tuple(channel.name for channel in extract_transformed.channels) == ("G",),
            len(extract_transformed.history) == len(extract_state.history) + 1,
        )
    )
    if not binary_passed or not extract_passed:
        raise EvidenceError("Bridge metadata transformation failed.")
    return _expected_metadata_records()


def _expected_metadata_records() -> dict[str, object]:
    return {
        "binary_threshold": {
            "status": "pass",
            "shape_preserved": True,
            "dtype_updated_to_bool": True,
            "axes_preserved": True,
            "source_preserved": True,
            "acquisition_preserved": True,
            "history_appended": True,
        },
        "extract_channel": {
            "status": "pass",
            "selected_channel_axis_removed": True,
            "selected_channel_metadata_preserved": True,
            "dtype_preserved": True,
            "source_preserved": True,
            "acquisition_preserved": True,
            "history_appended": True,
        },
    }


def _run_lifecycle(cp, cpu_functions, gpu_functions) -> dict[str, object]:
    from napari_vipp.core.progress import OperationCancelled, ProgressContext

    np = _numpy()
    definitions = (
        (
            "binary_threshold",
            np.linspace(-1, 1, 257 * 263, dtype=np.float32).reshape(257, 263),
            {"threshold": 0.125, "channel_axis": None},
            "Applying binary threshold",
        ),
        (
            "extract_channel",
            np.arange(3 * 257 * 263, dtype=np.uint16).reshape(3, 257, 263),
            {"channel": 1, "axis_names": ("c", "y", "x"), "axis_types": ()},
            "Extracting channel",
        ),
    )
    results: dict[str, object] = {}
    for operation_id, host, parameters, message in definitions:
        expected = cpu_functions[operation_id](host, **parameters)
        state: dict[str, object] = {"cancelled": False, "updates": []}

        def reporter(update, *, state=state) -> None:
            updates = state["updates"]
            assert isinstance(updates, list)
            updates.append(
                {
                    "current": int(update.current),
                    "total": int(update.total),
                    "message": str(update.message),
                }
            )
            if int(update.current) == 0:
                state["cancelled"] = True

        pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            device = cp.asarray(host)
            progress = ProgressContext(
                cancelled=lambda state=state: bool(state["cancelled"]),
                reporter=reporter,
            )
            cancellation_observed = False
            try:
                gpu_functions[operation_id](device, progress=progress, **parameters)
            except OperationCancelled as exc:
                cancellation_observed = True
                exc.__traceback__ = None
            if not cancellation_observed:
                raise EvidenceError(f"{operation_id} cancellation was not observed.")
            state["cancelled"] = False
            output = gpu_functions[operation_id](device, **parameters)
            cp.cuda.get_current_stream().synchronize()
            reuse_exact = _numpy().array_equal(
                cp.asnumpy(output), expected, equal_nan=True
            )
            input_immutable = _array_sha256(cp.asnumpy(device)) == _array_sha256(host)
            del output, device
            cp.cuda.get_current_stream().synchronize()
        cleanup = _drain_pool(cp, pool)
        expected_updates = [{"current": 0, "total": 1, "message": message}]
        updates = state["updates"]
        if updates != expected_updates or not reuse_exact or not input_immutable:
            raise EvidenceError(f"{operation_id} cancellation/reuse failed.")
        results[operation_id] = {
            "status": "pass",
            "cancellation_requested": True,
            "cancellation_observed": True,
            "reported_progress": updates,
            "post_cancellation_reuse_exact": True,
            "input_immutable": True,
            "boundary": (
                "monolithic-synchronized-boundary-v1"
                if operation_id == "binary_threshold"
                else "constant-time-view-boundary-v1"
            ),
            **cleanup,
        }
    return results


def _run_fallback(cpu_functions) -> dict[str, object]:
    np = _numpy()
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import evaluate_candidate_workload_support

    safe_definitions = (
        (
            "binary_threshold",
            "binary-uint16",
            ((5, 7),),
            ("uint16",),
            {"threshold": 5.0, "channel_axis": None},
            np.arange(35, dtype=np.uint16).reshape(5, 7),
        ),
        (
            "binary_threshold",
            "binary-luma",
            ((3, 5, 7),),
            ("float32",),
            {"threshold": 0.5, "channel_axis": 0},
            np.linspace(0, 1, 3 * 5 * 7, dtype=np.float32).reshape(3, 5, 7),
        ),
        (
            "binary_threshold",
            "binary-nonfinite-threshold",
            ((5, 7),),
            ("float32",),
            {"threshold": "nan", "channel_axis": None},
            np.linspace(0, 1, 35, dtype=np.float32).reshape(5, 7),
        ),
        (
            "extract_channel",
            "extract-float64",
            ((3, 5, 7),),
            ("float64",),
            {"channel": 1, "axis_names": ("c", "y", "x"), "axis_types": ()},
            np.arange(3 * 5 * 7, dtype=np.float64).reshape(3, 5, 7),
        ),
    )
    safe_cases: list[dict[str, object]] = []
    for operation_id, case_id, shapes, dtypes, parameters, source in safe_definitions:
        workload = WorkloadDescriptor(
            case_id,
            operation_id,
            shapes,
            dtypes,
            parameters=tuple(parameters.items()),
            resolved_spatial_ndim=2,
        )
        decision = evaluate_candidate_workload_support(
            _gpu_spec(operation_id), workload
        )
        if decision.supported or not decision.fallback_allowed:
            raise EvidenceError(f"Safe CPU fallback was not selected for {case_id}.")
        output = cpu_functions[operation_id](source, **parameters)
        safe_cases.append(
            {
                "operation_id": operation_id,
                "case_id": case_id,
                "gpu_supported": False,
                "fallback_allowed": True,
                "reason": decision.reason_text,
                "cpu_output_dtype": str(output.dtype),
                "cpu_output_sha256": _array_sha256(output),
            }
        )

    invalid_definitions = (
        (
            "binary_threshold",
            "binary-invalid-axis",
            ((3, 5, 7),),
            ("float32",),
            {"threshold": 0.5, "channel_axis": 9},
        ),
        (
            "extract_channel",
            "extract-missing-axis",
            ((3, 5, 7),),
            ("uint16",),
            {"channel": 1, "axis_names": ("z", "y", "x"), "axis_types": ()},
        ),
        (
            "extract_channel",
            "extract-invalid-index",
            ((3, 5, 7),),
            ("uint16",),
            {"channel": 9, "axis_names": ("c", "y", "x"), "axis_types": ()},
        ),
    )
    invalid_cases = []
    for operation_id, case_id, shapes, dtypes, parameters in invalid_definitions:
        workload = WorkloadDescriptor(
            case_id,
            operation_id,
            shapes,
            dtypes,
            parameters=tuple(parameters.items()),
            resolved_spatial_ndim=2,
        )
        decision = evaluate_candidate_workload_support(
            _gpu_spec(operation_id), workload
        )
        if decision.supported or decision.fallback_allowed:
            raise EvidenceError(f"Invalid authored case {case_id} did not fail closed.")
        invalid_cases.append(
            {
                "operation_id": operation_id,
                "case_id": case_id,
                "gpu_supported": False,
                "fallback_allowed": False,
                "reason": decision.reason_text,
            }
        )
    return {
        "status": "pass",
        "safe_cpu_fallback_case_count": len(safe_cases),
        "safe_cpu_fallback_cases": safe_cases,
        "invalid_authored_case_count": len(invalid_cases),
        "invalid_authored_cases": invalid_cases,
    }


def _expected_fallback_record() -> dict[str, object]:
    cpu_functions, _gpu_functions = _operation_functions()
    return _run_fallback(cpu_functions)


def _performance_cases(profile: str) -> tuple[PerformanceCase, ...]:
    quick = (
        PerformanceCase(
            "binary_threshold",
            "binary-plane-512",
            (512, 512),
            "float32",
            (("threshold", 0.4), ("channel_axis", None)),
        ),
        PerformanceCase(
            "extract_channel",
            "extract-c3-plane-512",
            (3, 512, 512),
            "uint16",
            (
                ("channel", 1),
                ("axis_names", ("c", "y", "x")),
                ("axis_types", ()),
            ),
        ),
        PerformanceCase(
            "extract_channel",
            "extract-c3-plane-31x37-allocator-rounding",
            (3, 31, 37),
            "uint16",
            (
                ("channel", 1),
                ("axis_names", ("c", "y", "x")),
                ("axis_types", ()),
            ),
        ),
        PerformanceCase(
            "extract_channel",
            "extract-plane-31x37-c3-contiguous-staging",
            (31, 37, 3),
            "uint16",
            (
                ("channel", 1),
                ("axis_names", ("y", "x", "c")),
                ("axis_types", ()),
            ),
        ),
    )
    if profile == "quick":
        return quick
    return quick + (
        PerformanceCase(
            "binary_threshold",
            "binary-stack-16x512",
            (16, 512, 512),
            "float32",
            (("threshold", 0.6), ("channel_axis", None)),
        ),
        PerformanceCase(
            "extract_channel",
            "extract-t4-c3-plane-512",
            (4, 3, 512, 512),
            "float32",
            (
                ("channel", 2),
                ("axis_names", ("t", "c", "y", "x")),
                ("axis_types", ()),
            ),
        ),
    )


def _performance_input(definition: PerformanceCase):
    np = _numpy()
    elements = math.prod(definition.shape)
    if definition.dtype == "float32":
        return np.linspace(-1.0, 2.0, elements, dtype=np.float32).reshape(
            definition.shape
        )
    maximum = int(np.iinfo(np.dtype(definition.dtype)).max)
    return (
        (np.arange(elements, dtype=np.uint64) * 257 % (maximum + 1))
        .astype(definition.dtype)
        .reshape(definition.shape)
    )


def _run_performance_case(cp, cpu_function, gpu_function, definition, rounds):
    host = _performance_input(definition)
    parameters = dict(definition.parameters)
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
        del device
        cp.cuda.get_current_stream().synchronize()
    timing_cleanup = _drain_pool(cp, pool)
    if final_host is None or not _numpy().array_equal(
        final_host, expected, equal_nan=True
    ):
        raise EvidenceError(f"Performance parity failed for {definition.case_id}.")

    memory = _measure_memory(cp, gpu_function, host, definition)
    return {
        "operation_id": definition.operation_id,
        "case_id": definition.case_id,
        "shape": list(definition.shape),
        "input_dtype": definition.dtype,
        "element_count": math.prod(definition.shape),
        "input_bytes": int(host.nbytes),
        "output_bytes": int(expected.nbytes),
        "parameters": _json_value(parameters),
        "parity": {
            "passed": True,
            "policy": "bitwise-array-v1",
            "cpu_output_sha256": _array_sha256(expected),
            "gpu_output_sha256": _array_sha256(final_host),
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
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device = cp.asarray(host)
        used_with_input = int(pool.used_bytes())
        output = gpu_function(device, **dict(definition.parameters))
        cp.cuda.get_current_stream().synchronize()
        used_with_output = int(pool.used_bytes())
        materialized_output = cp.ascontiguousarray(output)
        cp.cuda.get_current_stream().synchronize()
        used_with_materialization = int(pool.used_bytes())
        observed_reserved = int(pool.total_bytes())
        output_shares_input_allocation = int(output.data.mem.ptr) == int(
            device.data.mem.ptr
        )
        del materialized_output, output, device
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    estimated_with_uncertainty = (
        estimate.total_device_peak_bytes + estimate.uncertainty_bytes
    )
    covered = estimated_with_uncertainty >= observed_reserved
    if definition.operation_id == "extract_channel":
        if not output_shares_input_allocation or used_with_output != used_with_input:
            raise EvidenceError("Extract Channel allocated device output storage.")
    elif output_shares_input_allocation:
        raise EvidenceError("Binary Threshold did not allocate an independent mask.")
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
        "observed_used_bytes_with_materialization": used_with_materialization,
        "output_shares_input_allocation": output_shares_input_allocation,
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
        (definition.dtype,),
        parameters=definition.parameters,
        resolved_spatial_ndim=min(len(definition.shape), 3),
    )
    return estimate_candidate_memory(_gpu_spec(definition.operation_id), workload)


def _timing_summary(cpu, resident, transfer) -> dict[str, float]:
    return {
        "cpu_median_seconds": statistics.median(cpu),
        "gpu_resident_median_seconds": statistics.median(resident),
        "gpu_transfer_inclusive_median_seconds": statistics.median(transfer),
    }


def _warm_runtime(cp, gpu_functions) -> None:
    binary = cp.arange(16, dtype=cp.float32).reshape(4, 4)
    mask = gpu_functions["binary_threshold"](binary, threshold=7.5)
    channels = cp.arange(3 * 4 * 4, dtype=cp.uint16).reshape(3, 4, 4)
    selected = gpu_functions["extract_channel"](
        channels,
        channel=1,
        axis_names=("c", "y", "x"),
    )
    cp.cuda.get_current_stream().synchronize()
    del selected, channels, mask, binary
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
        "public_region": ("semantic-channel-view-plus-float32-scalar-threshold-v1"),
        "parity": "bitwise-array-v1",
        "gpu_resident_timing_scope": "synchronized-resident-operation-v1",
        "gpu_transfer_inclusive_timing_scope": (
            "host-to-device-plus-operation-plus-device-to-host-synchronized-v1"
        ),
        "memory_observation_scope": (
            "isolated-cupy-memory-pool-reserved-high-water-v1"
        ),
        "cancellation": ("synchronized-threshold-and-constant-time-view-boundaries-v1"),
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
        raise EvidenceError("Segmentation bridge evidence requires native Windows.")
    nonempty_string("release")
    if nonempty_string("machine").lower() not in {"amd64", "x86_64"}:
        raise EvidenceError("Segmentation bridge evidence requires Windows x86-64.")
    if nonempty_string("python_implementation") != "CPython":
        raise EvidenceError("Segmentation bridge evidence requires CPython.")
    if nonempty_string("python_abi") != "cpython-312":
        raise EvidenceError("Segmentation bridge evidence requires cpython-312 ABI.")
    python_version = nonempty_string("python")
    python_parts = python_version.split(".")
    if (
        len(python_parts) != 3
        or python_parts[:2] != ["3", "12"]
        or not python_parts[2].isdigit()
    ):
        raise EvidenceError("Segmentation bridge evidence requires CPython 3.12.x.")
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
        capability_parts = nonempty_string("compute_capability").split(".")
        capability = tuple(int(value) for value in capability_parts)
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
    from napari_vipp.core.operations import binary_threshold, extract_channel

    binary_module = importlib.import_module(
        "napari_vipp.core.gpu.cupy_binary_threshold"
    )
    extract_module = importlib.import_module(
        "napari_vipp.core.gpu.cupy_extract_channel"
    )
    return (
        {
            "binary_threshold": binary_threshold,
            "extract_channel": extract_channel,
        },
        {
            "binary_threshold": binary_module.binary_threshold,
            "extract_channel": extract_module.extract_channel,
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
    for operation_id in sorted(IMPLEMENTATION_IDS):
        life = _mapping(lifecycle.get(operation_id), "lifecycle")
        expected_message = (
            "Applying binary threshold"
            if operation_id == "binary_threshold"
            else "Extracting channel"
        )
        expected_boundary = (
            "monolithic-synchronized-boundary-v1"
            if operation_id == "binary_threshold"
            else "constant-time-view-boundary-v1"
        )
        if (
            life.get("status") != "pass"
            or life.get("cancellation_requested") is not True
            or life.get("cancellation_observed") is not True
            or life.get("reported_progress")
            != [{"current": 0, "total": 1, "message": expected_message}]
            or life.get("post_cancellation_reuse_exact") is not True
            or life.get("input_immutable") is not True
            or life.get("boundary") != expected_boundary
            or not _cleanup_passed(life)
        ):
            raise EvidenceError(f"{operation_id} lifecycle evidence is invalid.")

    fallback = _mapping(document.get("fallback"), "fallback")
    if fallback != _expected_fallback_record():
        raise EvidenceError("Fallback evidence is invalid.")

    performance = _mapping(document.get("performance"), "performance")
    results = performance.get("results")
    profile = str(document["profile"])
    rounds = FULL_ROUNDS if profile == "full" else QUICK_ROUNDS
    if (
        performance.get("status") != "pass"
        or performance.get("rounds") != rounds
        or not isinstance(results, list)
        or performance.get("case_count") != len(results)
        or [item.get("case_id") for item in results]
        != [item.case_id for item in _performance_cases(profile)]
        or performance.get("all_memory_estimates_cover_observed") is not True
    ):
        raise EvidenceError("Performance aggregate is invalid.")
    for result in results:
        item = _mapping(result, "performance case")
        parity = _mapping(item.get("parity"), "parity")
        samples = _mapping(item.get("samples"), "samples")
        memory = _mapping(item.get("memory"), "memory")
        cleanup = _mapping(item.get("cleanup"), "timing cleanup")
        operation_id = str(item.get("operation_id"))
        case_id = str(item.get("case_id"))
        definition = next(
            (
                case
                for case in _performance_cases(profile)
                if case.case_id == case_id
            ),
            None,
        )
        if definition is None or definition.operation_id != operation_id:
            raise EvidenceError("Performance case identity is invalid.")
        estimate = _performance_memory_estimate(definition)
        expected_peak = estimate.total_device_peak_bytes + estimate.uncertainty_bytes
        observed_reserved = memory.get("observed_reserved_bytes")
        if (
            item.get("shape") != list(definition.shape)
            or item.get("input_dtype") != definition.dtype
            or item.get("parameters") != _json_value(dict(definition.parameters))
            or parity.get("passed") is not True
            or parity.get("cpu_output_sha256") != parity.get("gpu_output_sha256")
            or memory.get("model_id") != estimate.model_id
            or memory.get("runtime_managed_peak_bytes")
            != estimate.runtime_managed_peak_bytes
            or memory.get("total_device_peak_bytes")
            != estimate.total_device_peak_bytes
            or memory.get("uncertainty_bytes") != estimate.uncertainty_bytes
            or memory.get("estimated_peak_with_uncertainty_bytes") != expected_peak
            or isinstance(observed_reserved, bool)
            or not isinstance(observed_reserved, int)
            or observed_reserved < 0
            or expected_peak < observed_reserved
            or memory.get("estimate_covers_observed") is not True
            or not _cleanup_passed(memory)
            or not _cleanup_passed(cleanup)
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
    if document.get("method") != _method_record(profile, rounds):
        raise EvidenceError("Method record is inconsistent with the profile.")


def _validate_admission_records(value: object) -> None:
    admission = _mapping(value, "admission")
    for operation_id in sorted(IMPLEMENTATION_IDS):
        section = _mapping(admission.get(operation_id), operation_id)
        expected_definitions = [
            case for case in _admission_cases() if case.operation_id == operation_id
        ]
        expected_cases = [case.case_id for case in expected_definitions]
        expected_coverage = {
            item
            for definition in expected_definitions
            for item in definition.coverage
        } | {"repeat:deterministic"}
        cases = section.get("cases")
        if (
            section.get("status") != "pass"
            or not isinstance(cases, list)
            or [item.get("case_id") for item in cases] != expected_cases
            or section.get("case_count") != len(cases)
            or section.get("repeat_count") != ADMISSION_REPEATS
            or section.get("coverage") != sorted(expected_coverage)
            or not REQUIRED_ADMISSION_COVERAGE[operation_id] <= expected_coverage
        ):
            raise EvidenceError(f"{operation_id} admission aggregate is invalid.")
        for case, definition in zip(cases, expected_definitions, strict=True):
            item = _mapping(case, "admission case")
            host = _host_case(definition.kind)
            parameters = _case_parameters(definition.kind)
            expected_integrity = (
                "independent-bool-allocation-v1"
                if operation_id == "binary_threshold"
                else "shared-input-allocation-view-v1"
            )
            if (
                item.get("shape") != list(host.shape)
                or item.get("input_dtype") != str(host.dtype)
                or item.get("output_dtype")
                != ("bool" if operation_id == "binary_threshold" else str(host.dtype))
                or item.get("parameters") != _json_value(parameters)
                or item.get("input_sha256") != _array_sha256(host)
                or item.get("coverage") != sorted(definition.coverage)
                or item.get("output_integrity_contract") != expected_integrity
                or item.get("cpu_gpu_bitwise_equal") is not True
                or item.get("gpu_output_resident") is not True
                or item.get("input_immutable") is not True
                or item.get("repeat_deterministic") is not True
                or item.get("repeat_count") != ADMISSION_REPEATS
                or item.get("cpu_output_sha256") != item.get("gpu_output_sha256")
                or not _cleanup_passed(_mapping(item.get("cleanup"), "cleanup"))
            ):
                raise EvidenceError("Admission parity/integrity evidence is invalid.")

def _validate_metadata_records(value: object) -> None:
    metadata = _mapping(value, "metadata")
    if metadata != _expected_metadata_records():
        raise EvidenceError("Bridge metadata evidence is invalid.")


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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
