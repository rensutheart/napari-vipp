#!/usr/bin/env python
"""Qualify VIPP's exact CuPy Remove Outliers (Binary) implementation.

The evidence is intentionally limited to the public bool-mask region: trailing
YX planes, ImageJ's historical footprint, radii from 0.5 through 25, and one
foreground/background polarity per invocation.  It covers exact CPU parity,
adversarial inputs, metadata, input integrity, the executable memory model,
tile-boundary cancellation, cleanup and reuse, safe CPU fallback, production
provenance, transfer-inclusive timing, and unseen/revisited runtime parameters.

Importing this module, asking for ``--help``, or validating an existing JSON
artifact does not import CuPy or initialize CUDA.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

SCHEMA = "napari-vipp-cupy-remove-binary-outliers-evidence"
SCHEMA_VERSION = 1
EVIDENCE_KIND = "scientific-admission-and-machine-local-performance-evidence"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/benchmarks/remove-binary-outliers-cupy-local.json"
OPERATION_ID = "remove_binary_outliers"
IMPLEMENTATION_ID = "cupy-remove-binary-outliers-v1"
IMPLEMENTATION_VERSION = "1"
RUNTIME_ID = "cuda-cupy"
LIBRARY_ID = "cupy"
ENVIRONMENT_POLICY_ID = "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1"
FOREGROUND = "Foreground (remove)"
BACKGROUND = "Background (fill)"
ADMISSION_REPEATS = 3
QUICK_ROUNDS = 3
FULL_ROUNDS = 5
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
RADIUS_SEQUENCE = (0.5, 1.0, 1.5, 2.0, 3.0, 8.0, 25.0, 1.5, 8.0)
POLARITY_SEQUENCE = (FOREGROUND, BACKGROUND, FOREGROUND, BACKGROUND)
SOURCE_PROVENANCE_PATHS = (
    Path("src/napari_vipp/core/remove_outliers.py"),
    Path("src/napari_vipp/core/gpu/cupy_remove_binary_outliers.py"),
    Path("src/napari_vipp/core/compute_specs.py"),
    Path("src/napari_vipp/core/compute_policy.py"),
    Path("src/napari_vipp/core/execution.py"),
    Path("src/napari_vipp/core/metadata.py"),
    Path("src/napari_vipp/core/pipeline.py"),
    Path("scripts/benchmark_gpu_remove_binary_outliers.py"),
)
REQUIRED_ADMISSION_COVERAGE = frozenset(
    {
        "dtype:bool",
        "rank:2",
        "rank:3",
        "rank:4",
        "leading-planes:independent",
        "layout:contiguous",
        "layout:noncontiguous",
        "boundary:nearest",
        "shape:tiny-smaller-than-radius",
        "shape:odd",
        "radius:0.5",
        "radius:1.5-imagej-adjustment",
        "radius:2.5-imagej-adjustment",
        "radius:25-public-maximum",
        "polarity:foreground-remove",
        "polarity:background-fill",
        "constant:foreground",
        "constant:background",
        "repeat:deterministic",
    }
)


class EvidenceError(RuntimeError):
    """The collected evidence is incomplete, stale, or inconsistent."""


@dataclass(frozen=True, slots=True)
class AdmissionCase:
    case_id: str
    data: object
    radius: float
    which_outliers: str
    coverage: tuple[str, ...]
    device_noncontiguous: bool = False


@dataclass(frozen=True, slots=True)
class PerformanceCase:
    case_id: str
    shape: tuple[int, ...]
    radius: float
    which_outliers: str
    seed: int


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
            path = validate_existing(args.validate_existing)
            print(f"Remove Outliers (Binary) evidence is current: {path}")
            return 0
        if args.device_index < 0:
            raise ValueError("device index must be nonnegative")
        document = build_evidence(args.profile, args.device_index)
        destination = _atomic_write_json(args.output, document)
        validate_existing(destination)
    except (EvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"Remove Outliers (Binary) evidence failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {destination}")
    return 0


def build_evidence(profile: str, device_index: int) -> dict[str, object]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'.")
    source_snapshot = _source_provenance()
    contract_snapshot = _operation_contract()
    np = _numpy()
    cp = _cupy()
    cpu_function, gpu_function, provider_module = _operation_functions()
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - hardware-specific failure
        raise EvidenceError(f"CUDA runtime probe failed: {exc}") from exc
    if device_index >= device_count:
        raise EvidenceError(
            f"CUDA device index {device_index} is unavailable; found {device_count}."
        )

    rounds = FULL_ROUNDS if profile == "full" else QUICK_ROUNDS
    with cp.cuda.Device(device_index):
        _warm_runtime(cp, gpu_function)
        admission = _run_admission(cp, cpu_function, gpu_function)
        parameter_sweep = _run_parameter_sweep(
            cp,
            cpu_function,
            gpu_function,
            provider_module,
        )
        metadata = _run_metadata(cpu_function)
        lifecycle = _run_lifecycle(cp, cpu_function, gpu_function)
        fallback = _run_fallback(cpu_function)
        provenance = _run_production_provenance(
            np,
            cpu_function,
            device_index=device_index,
        )
        performance_results = [
            _run_performance_case(
                cp,
                cpu_function,
                gpu_function,
                definition,
                rounds,
            )
            for definition in _performance_cases(profile)
        ]
        terminal_cleanup = _drain_default_pools(cp)

    if source_snapshot != _source_provenance():
        raise EvidenceError("Tracked source changed while evidence was collected.")
    if contract_snapshot != _operation_contract():
        raise EvidenceError("The public compute contract changed during collection.")
    if any(
        not item["memory"]["estimate_covers_observed"] for item in performance_results
    ):
        raise EvidenceError("The executable memory estimate was exceeded.")

    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": EVIDENCE_KIND,
        "status": "pass",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": profile,
        "implementation": {
            "operation_id": OPERATION_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": IMPLEMENTATION_VERSION,
            "runtime_id": RUNTIME_ID,
            "library_id": LIBRARY_ID,
        },
        "environment": _environment_record(cp, device_index),
        "packages": _package_versions(),
        "source_provenance": source_snapshot,
        "operation_contract": contract_snapshot,
        "method": {
            "cpu_oracle": ("napari_vipp.core.remove_outliers:remove_binary_outliers"),
            "gpu_provider": (
                "napari_vipp.core.gpu.cupy_remove_binary_outliers:"
                "remove_binary_outliers"
            ),
            "parity": "bitwise-identical-bool-shape-v1",
            "timing": (
                "synchronized wall clock; transfer-inclusive samples include "
                "host-to-device input and device-to-host output"
            ),
            "parameter_compilation": (
                "one warmed fixed-source RawKernel receives radius-derived row "
                "spans and polarity as runtime arguments"
            ),
        },
        "facets": {facet: "pass" for facet in REQUIRED_FACETS},
        "admission": admission,
        "parameter_sweep": parameter_sweep,
        "metadata": metadata,
        "lifecycle": lifecycle,
        "fallback": fallback,
        "provenance": provenance,
        "performance": {
            "status": "pass",
            "rounds": rounds,
            "case_count": len(performance_results),
            "results": performance_results,
            "all_memory_estimates_cover_observed": True,
        },
        "terminal_cleanup": terminal_cleanup,
    }
    _validate_document(document, require_current_sources=True)
    return document


def _admission_cases() -> tuple[AdmissionCase, ...]:
    np = _numpy()

    isolated = np.zeros((9, 11), dtype=bool)
    isolated[4, 5] = True
    isolated[1:4, 1:4] = True

    boundary = np.zeros((9, 13), dtype=bool)
    boundary[0, 0] = True
    boundary[0, -1] = True
    boundary[-1, 0] = True
    boundary[-1, -1] = True
    boundary[2:7, 3:10] = True
    boundary[4, 6] = False

    tiny = np.array([[True, False, True], [False, True, False]], dtype=bool)

    noncontiguous_base = np.indices((3, 15, 38)).sum(axis=0) % 5 < 2
    noncontiguous = noncontiguous_base[..., ::2]

    leading = np.indices((2, 3, 11, 13)).sum(axis=0) % 7 < 3
    leading[..., 3:8, 4:9] = True

    maximum = np.indices((31, 33)).sum(axis=0) % 4 == 0
    maximum[5:26, 6:27] = True
    maximum[14:17, 15:18] = False

    return (
        AdmissionCase(
            "foreground-radius-half",
            isolated,
            0.5,
            FOREGROUND,
            (
                "dtype:bool",
                "rank:2",
                "layout:contiguous",
                "shape:odd",
                "radius:0.5",
                "polarity:foreground-remove",
            ),
        ),
        AdmissionCase(
            "background-imagej-radius-one-half",
            boundary,
            1.5,
            BACKGROUND,
            (
                "dtype:bool",
                "rank:2",
                "layout:contiguous",
                "boundary:nearest",
                "shape:odd",
                "radius:1.5-imagej-adjustment",
                "polarity:background-fill",
            ),
        ),
        AdmissionCase(
            "foreground-imagej-radius-two-half",
            boundary,
            2.5,
            FOREGROUND,
            (
                "dtype:bool",
                "rank:2",
                "layout:contiguous",
                "boundary:nearest",
                "radius:2.5-imagej-adjustment",
                "polarity:foreground-remove",
            ),
        ),
        AdmissionCase(
            "tiny-background-radius-eight",
            tiny,
            8.0,
            BACKGROUND,
            (
                "dtype:bool",
                "rank:2",
                "layout:contiguous",
                "boundary:nearest",
                "shape:tiny-smaller-than-radius",
                "polarity:background-fill",
            ),
        ),
        AdmissionCase(
            "strided-three-dimensional-leading-planes",
            noncontiguous,
            3.0,
            FOREGROUND,
            (
                "dtype:bool",
                "rank:3",
                "leading-planes:independent",
                "layout:noncontiguous",
                "polarity:foreground-remove",
            ),
            device_noncontiguous=True,
        ),
        AdmissionCase(
            "four-dimensional-leading-planes",
            leading,
            2.0,
            BACKGROUND,
            (
                "dtype:bool",
                "rank:4",
                "leading-planes:independent",
                "layout:contiguous",
                "polarity:background-fill",
            ),
        ),
        AdmissionCase(
            "public-maximum-radius",
            maximum,
            25.0,
            FOREGROUND,
            (
                "dtype:bool",
                "rank:2",
                "layout:contiguous",
                "radius:25-public-maximum",
                "polarity:foreground-remove",
            ),
        ),
        AdmissionCase(
            "constant-foreground",
            np.ones((7, 9), dtype=bool),
            3.0,
            FOREGROUND,
            (
                "dtype:bool",
                "rank:2",
                "layout:contiguous",
                "constant:foreground",
                "polarity:foreground-remove",
            ),
        ),
        AdmissionCase(
            "constant-background",
            np.zeros((7, 9), dtype=bool),
            3.0,
            BACKGROUND,
            (
                "dtype:bool",
                "rank:2",
                "layout:contiguous",
                "constant:background",
                "polarity:background-fill",
            ),
        ),
    )


def _run_admission(cp, cpu_function, gpu_function) -> dict[str, object]:
    records: list[dict[str, object]] = []
    coverage: set[str] = {"repeat:deterministic"}
    for definition in _admission_cases():
        host = definition.data
        host_hash = _array_sha256(host)
        expected = cpu_function(
            host,
            radius=definition.radius,
            which_outliers=definition.which_outliers,
        )
        pool = cp.cuda.MemoryPool()
        output_hashes = []
        with cp.cuda.using_allocator(pool.malloc):
            if definition.device_noncontiguous:
                device_base = cp.asarray(
                    _numpy().ascontiguousarray(host.repeat(2, axis=-1))
                )
                device = device_base[..., ::2]
            else:
                device_base = None
                device = cp.asarray(host)
            device_before = device.copy()
            source_pointer = int(device.data.ptr)
            output_pointer = source_pointer
            output_contiguous = False
            for _repeat in range(ADMISSION_REPEATS):
                output = gpu_function(
                    device,
                    radius=definition.radius,
                    which_outliers=definition.which_outliers,
                )
                cp.cuda.get_current_stream().synchronize()
                actual = cp.asnumpy(output)
                if not _numpy().array_equal(expected, actual):
                    raise EvidenceError(
                        f"{definition.case_id} failed exact CPU/GPU parity."
                    )
                output_hashes.append(_array_sha256(actual))
                output_pointer = int(output.data.ptr)
                output_contiguous = bool(output.flags.c_contiguous)
                del output, actual
            input_immutable = bool(cp.array_equal(device, device_before).item())
            device_layout_observed = (
                "noncontiguous" if not bool(device.flags.c_contiguous) else "contiguous"
            )
            del device_before, device
            if device_base is not None:
                del device_base
            cp.cuda.get_current_stream().synchronize()
        cleanup = _drain_pool(cp, pool)
        repeat_deterministic = len(set(output_hashes)) == 1
        output_independent = output_pointer != source_pointer
        if not all(
            (
                input_immutable,
                repeat_deterministic,
                output_contiguous,
                output_independent,
            )
        ):
            raise EvidenceError(
                f"{definition.case_id} failed integrity or allocation guarantees."
            )
        if _array_sha256(host) != host_hash:
            raise EvidenceError(f"{definition.case_id} mutated its host input.")
        if (
            definition.device_noncontiguous
            and device_layout_observed != "noncontiguous"
        ):
            raise EvidenceError("The noncontiguous device lane was not exercised.")
        coverage.update(definition.coverage)
        records.append(
            {
                "case_id": definition.case_id,
                "shape": list(host.shape),
                "radius": definition.radius,
                "which_outliers": definition.which_outliers,
                "coverage": list(definition.coverage),
                "cpu_gpu_bitwise_equal": True,
                "mismatch_count": 0,
                "input_immutable": True,
                "gpu_output_resident": True,
                "gpu_output_contiguous": True,
                "gpu_output_independent": True,
                "device_input_layout": device_layout_observed,
                "repeat_count": ADMISSION_REPEATS,
                "repeat_deterministic": True,
                "cpu_output_sha256": _array_sha256(expected),
                "gpu_output_sha256": output_hashes[-1],
                "cleanup": cleanup,
            }
        )
    missing = REQUIRED_ADMISSION_COVERAGE - coverage
    if missing:
        raise EvidenceError(f"Admission coverage is incomplete: {sorted(missing)}")
    return {
        "status": "pass",
        "parity_profile": "bitwise-identical-bool-shape-v1",
        "case_count": len(records),
        "repeat_count": ADMISSION_REPEATS,
        "coverage": sorted(coverage),
        "cases": records,
    }


def _run_parameter_sweep(cp, cpu_function, gpu_function, provider_module):
    np = _numpy()
    host = np.indices((31, 37)).sum(axis=0) % 6 < 3
    host[4:27, 5:32] = True
    host[13:18, 15:22] = False
    kernel_factory = provider_module._remove_binary_outliers_kernel
    kernel = kernel_factory(cp)
    cache_before = kernel_factory.cache_info()
    records = []

    def run_lane(lane_id, parameter_name, values, base):
        seen: set[str] = set()
        steps = []
        for index, value in enumerate(values):
            parameters = dict(base)
            parameters[parameter_name] = value
            expected = cpu_function(host, **parameters)
            started = time.perf_counter()
            device = cp.asarray(host)
            output = gpu_function(device, **parameters)
            actual = cp.asnumpy(output)
            cp.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter() - started
            if not np.array_equal(expected, actual):
                raise EvidenceError(
                    f"Parameter sweep {lane_id}={value!r} failed exact parity."
                )
            if kernel_factory(cp) is not kernel:
                raise EvidenceError("A parameter change replaced the fixed RawKernel.")
            key = json.dumps(value, sort_keys=True)
            occurrence = "revisit" if key in seen else "unseen"
            if index == 0:
                occurrence = "startup"
            seen.add(key)
            steps.append(
                {
                    "index": index,
                    "authored_value": value,
                    "occurrence": occurrence,
                    "transfer_inclusive_seconds": elapsed,
                    "cpu_gpu_bitwise_equal": True,
                    "gpu_output_sha256": _array_sha256(actual),
                }
            )
            del actual, output, device
        if not any(item["occurrence"] == "revisit" for item in steps):
            raise EvidenceError(f"Parameter lane {lane_id!r} has no revisit.")
        records.append(
            {
                "lane_id": lane_id,
                "parameter_name": parameter_name,
                "values": list(values),
                "steps": steps,
                "all_exact": True,
            }
        )

    run_lane(
        "radius",
        "radius",
        RADIUS_SEQUENCE,
        {"radius": 2.0, "which_outliers": FOREGROUND},
    )
    run_lane(
        "outlier-polarity",
        "which_outliers",
        POLARITY_SEQUENCE,
        {"radius": 2.0, "which_outliers": FOREGROUND},
    )
    cp.cuda.get_current_stream().synchronize()
    cache_after = kernel_factory.cache_info()
    cleanup = _drain_default_pools(cp)
    if cache_after.misses != cache_before.misses:
        raise EvidenceError("Runtime parameter changes created a new kernel object.")
    return {
        "status": "pass",
        "lanes": records,
        "fixed_source_kernel": True,
        "kernel_object_reused": True,
        "runtime_parameterized_radius": True,
        "runtime_parameterized_polarity": True,
        "kernel_cache_misses_before": cache_before.misses,
        "kernel_cache_misses_after": cache_after.misses,
        "new_kernel_objects_during_sweep": 0,
        "timing_interpretation": (
            "machine-local synchronized transfer-inclusive observations; no "
            "portable absolute cliff threshold"
        ),
        "cleanup": cleanup,
    }


def _run_metadata(cpu_function) -> dict[str, object]:
    np = _numpy()
    from napari_vipp.core import execution as execution_module
    from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
    from napari_vipp.core.pipeline import PrototypePipeline

    data = np.indices((2, 3, 11, 13)).sum(axis=0) % 5 < 2
    axes = (
        AxisMetadata("t", "time", "s", 2.0),
        AxisMetadata("z", "space", "micrometer", 1.5),
        AxisMetadata("y", "space", "micrometer", 0.4),
        AxisMetadata("x", "space", "micrometer", 0.6),
    )
    state = image_state_from_array(
        data,
        axes=axes,
        history=("Imported calibrated segmentation",),
        source_name="Remove Outliers admission fixture",
    )
    if state is None:
        raise EvidenceError("Metadata fixture did not produce an image state.")
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source = pipeline.add_node("binary_threshold")
    node = pipeline.add_node(OPERATION_ID)
    pipeline.set_param(node.id, "radius", 2.5)
    pipeline.set_param(node.id, "which_outliers", BACKGROUND)
    if not pipeline.connect(source.id, node.id).success:
        raise EvidenceError("Metadata fixture graph could not be connected.")
    call = pipeline.prepare_node_call(node.id, (data,), (state,))
    output = cpu_function(data, **call.kwargs)
    (_final_output, cpu_state) = pipeline.finalize_node_call(call, output)[0]
    (gpu_state,) = execution_module._predict_device_node_states(
        pipeline,
        call,
        _gpu_spec(),
        (SimpleNamespace(shape=data.shape, dtype=np.dtype(bool)),),
    )
    if cpu_state is None or gpu_state is None:
        raise EvidenceError("Projected cleanup metadata is missing.")
    cpu_structure = _metadata_structure(cpu_state)
    gpu_structure = _metadata_structure(gpu_state)
    history_appended = len(cpu_state.history) == len(state.history) + 1
    passed = all(
        (
            cpu_structure == gpu_structure,
            tuple(cpu_state.shape) == tuple(data.shape),
            cpu_state.dtype == "bool",
            cpu_state.kind == "binary mask",
            cpu_state.axes == state.axes,
            history_appended,
        )
    )
    if not passed:
        raise EvidenceError("Structural metadata projection failed.")
    return {
        "status": "pass",
        "cpu_gpu_structural_metadata_equal": True,
        "shape_preserved": True,
        "axes_preserved": True,
        "bool_mask_kind_preserved": True,
        "history_appended": True,
        "cpu_structure": cpu_structure,
        "gpu_structure": gpu_structure,
    }


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
    }


class _CancelAfterFirstTile:
    def __init__(self) -> None:
        self.cancelled = False
        self.reports: list[dict[str, object]] = []

    def report(self, current: int, total: int, message: str) -> None:
        self.reports.append(
            {"current": int(current), "total": int(total), "message": str(message)}
        )
        if current == 1:
            self.cancelled = True

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("remove-outliers-evidence-cancelled")


def _run_lifecycle(cp, cpu_function, gpu_function) -> dict[str, object]:
    np = _numpy()
    host = np.indices((2, 130, 129)).sum(axis=0) % 5 < 2
    parameters = {"radius": 25.0, "which_outliers": BACKGROUND}
    expected = cpu_function(host, **parameters)
    host_hash = _array_sha256(host)
    progress = _CancelAfterFirstTile()
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device = cp.asarray(host)
        cancellation_observed = False
        try:
            gpu_function(device, progress=progress, **parameters)
        except RuntimeError as exc:
            cancellation_observed = str(exc) == "remove-outliers-evidence-cancelled"
        progress.cancelled = False
        reused = gpu_function(device, **parameters)
        cp.cuda.get_current_stream().synchronize()
        reused_host = cp.asnumpy(reused)
        reuse_exact = np.array_equal(expected, reused_host)
        input_immutable = _array_sha256(cp.asnumpy(device)) == host_hash
        del reused_host, reused, device
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    completed_tiles = [item for item in progress.reports if item["current"] == 1]
    if not all(
        (
            cancellation_observed,
            len(completed_tiles) == 1,
            reuse_exact,
            input_immutable,
            _array_sha256(host) == host_hash,
        )
    ):
        raise EvidenceError("Cancellation, reuse, or input integrity failed.")
    terminal = _drain_default_pools(cp)
    return {
        "status": "pass",
        "cancellation_requested": True,
        "cancellation_observed": True,
        "boundary": "synchronized-pixel-tile-boundary-v1",
        "reported_progress": progress.reports,
        "post_cancellation_reuse_exact": True,
        "input_immutable": True,
        "isolated_pool_cleanup": cleanup,
        "default_pool_cleanup_after_reuse": terminal,
    }


def _fallback_definitions() -> tuple[dict[str, object], ...]:
    return (
        {
            "case_id": "radius-above-public-through-direct-limit",
            "shape": (9, 11),
            "dtype": "bool",
            "parameters": (("radius", 26.0), ("which_outliers", FOREGROUND)),
            "category": "safe_cpu_fallback",
        },
        {
            "case_id": "canonical-uint8-mask",
            "shape": (9, 11),
            "dtype": "uint8",
            "parameters": (("radius", 2.0), ("which_outliers", BACKGROUND)),
            "category": "safe_cpu_fallback",
        },
        {
            "case_id": "radius-below-minimum",
            "shape": (9, 11),
            "dtype": "bool",
            "parameters": (("radius", 0.49), ("which_outliers", FOREGROUND)),
            "category": "invalid_authored",
        },
        {
            "case_id": "invalid-polarity",
            "shape": (9, 11),
            "dtype": "bool",
            "parameters": (("radius", 2.0), ("which_outliers", "Both")),
            "category": "invalid_authored",
        },
        {
            "case_id": "rank-one-mask",
            "shape": (11,),
            "dtype": "bool",
            "parameters": (("radius", 2.0), ("which_outliers", FOREGROUND)),
            "category": "invalid_authored",
        },
    )


def _run_fallback(cpu_function) -> dict[str, object]:
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import evaluate_candidate_workload_support

    safe = []
    invalid = []
    for definition in _fallback_definitions():
        workload = WorkloadDescriptor(
            definition["case_id"],
            OPERATION_ID,
            (definition["shape"],),
            (definition["dtype"],),
            parameters=definition["parameters"],
            resolved_spatial_ndim=2,
        )
        decision = evaluate_candidate_workload_support(_gpu_spec(), workload)
        record = {
            "case_id": definition["case_id"],
            "shape": list(definition["shape"]),
            "dtype": definition["dtype"],
            "parameters": dict(definition["parameters"]),
            "supported": decision.supported,
            "fallback_allowed": decision.fallback_allowed,
            "reason_text": decision.reason_text,
        }
        if definition["category"] == "safe_cpu_fallback":
            source = _fallback_input(definition)
            source_hash = _array_sha256(source)
            output = cpu_function(source, **dict(definition["parameters"]))
            record.update(
                {
                    "cpu_authority_executed": True,
                    "cpu_output_dtype": str(output.dtype),
                    "cpu_output_sha256": _array_sha256(output),
                    "input_immutable": _array_sha256(source) == source_hash,
                }
            )
            if decision.supported or not decision.fallback_allowed:
                raise EvidenceError(
                    f"Safe fallback failed for {definition['case_id']}."
                )
            safe.append(record)
        else:
            record["cpu_authority_executed"] = False
            if decision.supported or decision.fallback_allowed:
                raise EvidenceError(
                    f"Invalid authoring was not rejected: {definition['case_id']}."
                )
            invalid.append(record)
    return {
        "status": "pass",
        "safe_cpu_fallback_case_count": len(safe),
        "invalid_authored_case_count": len(invalid),
        "safe_cpu_fallback_cases": safe,
        "invalid_authored_cases": invalid,
    }


def _fallback_input(definition: Mapping[str, object]):
    np = _numpy()
    shape = tuple(definition["shape"])
    if definition["dtype"] == "uint8":
        source = np.zeros(shape, dtype=np.uint8)
        source[2:7, 3:8] = 255
        source[4, 5] = 0
        return source
    source = np.indices(shape).sum(axis=0) % 4 == 0
    return source.astype(bool, copy=False)


def _run_production_provenance(np, cpu_function, *, device_index: int):
    from napari_vipp.core.compute import ComputeMode, ComputeRequest, FallbackPolicy
    from napari_vipp.core.compute_registry import ComputeRegistry
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import serialize_workflow

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    target = pipeline.add_node(OPERATION_ID)
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(target.id, "radius", 2.5)
    pipeline.set_param(target.id, "which_outliers", BACKGROUND)
    if not pipeline.connect("input", threshold.id).success:
        raise EvidenceError("Production provenance threshold could not be connected.")
    if not pipeline.connect(threshold.id, target.id).success:
        raise EvidenceError("Production provenance target could not be connected.")
    data = np.linspace(0.0, 1.0, 67 * 71, dtype=np.float32).reshape(67, 71)
    data[20:47, 22:49] = 1.0
    data[32:35, 34:37] = 0.0
    expected_mask = data > np.float32(0.5)
    expected = cpu_function(expected_mask, radius=2.5, which_outliers=BACKGROUND)
    request = PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata={"axes": "YX"},
        input_name="remove-outliers-production-provenance",
        source_payloads={},
        compute_request=ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences={
                threshold.id: "cpu",
                target.id: f"implementation:{IMPLEMENTATION_ID}",
            },
            runtime_id=RUNTIME_ID,
            device_id=f"cuda:{device_index}",
            fallback_policy=FallbackPolicy.STRICT,
        ),
        retain_node_ids=frozenset({target.id}),
    )
    registry = ComputeRegistry()
    try:
        result = execute_pipeline_request(request, compute_registry=registry)
    finally:
        registry.close()
    if result.error or result.pipeline is None or result.execution_report is None:
        raise EvidenceError(
            f"Production provenance execution failed: {result.error or 'no report'}"
        )
    decision = next(
        (
            item
            for item in result.execution_report.actual_decisions
            if item.node_id == target.id
        ),
        None,
    )
    if decision is None:
        raise EvidenceError("Production provenance omitted the target decision.")
    actual = np.asarray(result.pipeline.outputs[target.id])
    exact = np.array_equal(actual, expected)
    identity_exact = all(
        (
            decision.runtime_id == RUNTIME_ID,
            decision.implementation_library_id == LIBRARY_ID,
            decision.implementation_id == IMPLEMENTATION_ID,
            decision.implementation_version == IMPLEMENTATION_VERSION,
            not decision.fallback_used,
        )
    )
    if not exact or not identity_exact or not result.execution_report.cleanup_succeeded:
        raise EvidenceError("Production provenance or cleanup was not exact.")
    return {
        "status": "pass",
        "actual_runtime_id": decision.runtime_id,
        "actual_library_id": decision.implementation_library_id,
        "actual_implementation_id": decision.implementation_id,
        "actual_implementation_version": decision.implementation_version,
        "decision_kind": decision.decision_kind.value,
        "fallback_used": decision.fallback_used,
        "cleanup_succeeded": result.execution_report.cleanup_succeeded,
        "cpu_gpu_bitwise_equal": True,
        "output_sha256": _array_sha256(actual),
        "environment": {
            "device_id": result.execution_report.environment.device_id,
            "device_name": result.execution_report.environment.device_name,
            "environment_fingerprint": (
                result.execution_report.environment.fingerprint
            ),
        },
    }


def _performance_cases(profile: str) -> tuple[PerformanceCase, ...]:
    quick = (
        PerformanceCase("plane-256-r2-foreground", (256, 256), 2.0, FOREGROUND, 1),
        PerformanceCase("plane-384-r8-background", (384, 384), 8.0, BACKGROUND, 2),
        PerformanceCase("plane-256-r25-foreground", (256, 256), 25.0, FOREGROUND, 3),
    )
    if profile == "quick":
        return quick
    return quick + (
        PerformanceCase("plane-1024-r2-foreground", (1024, 1024), 2.0, FOREGROUND, 4),
        PerformanceCase("plane-768-r8-background", (768, 768), 8.0, BACKGROUND, 5),
        PerformanceCase("stack-4x512-r3-foreground", (4, 512, 512), 3.0, FOREGROUND, 6),
        PerformanceCase("plane-512-r25-background", (512, 512), 25.0, BACKGROUND, 7),
    )


def _performance_input(definition: PerformanceCase):
    np = _numpy()
    rng = np.random.default_rng(definition.seed)
    source = rng.random(definition.shape) > 0.54
    source[
        ...,
        definition.shape[-2] // 4 : -definition.shape[-2] // 4,
        definition.shape[-1] // 4 : -definition.shape[-1] // 4,
    ] = True
    return source


def _run_performance_case(cp, cpu_function, gpu_function, definition, rounds):
    np = _numpy()
    host = _performance_input(definition)
    parameters = {
        "radius": definition.radius,
        "which_outliers": definition.which_outliers,
    }
    expected = cpu_function(host, **parameters)
    device = cp.asarray(host)
    warm = gpu_function(device, **parameters)
    cp.cuda.get_current_stream().synchronize()
    if not np.array_equal(cp.asnumpy(warm), expected):
        raise EvidenceError(f"{definition.case_id} warm parity failed.")
    del warm

    cpu_seconds = []
    resident_seconds = []
    transfer_seconds = []
    for _round in range(rounds):
        started = time.perf_counter()
        cpu_output = cpu_function(host, **parameters)
        cpu_seconds.append(time.perf_counter() - started)

        started = time.perf_counter()
        resident_output = gpu_function(device, **parameters)
        cp.cuda.get_current_stream().synchronize()
        resident_seconds.append(time.perf_counter() - started)

        started = time.perf_counter()
        transferred_input = cp.asarray(host)
        transferred_output = gpu_function(transferred_input, **parameters)
        transferred_host = cp.asnumpy(transferred_output)
        cp.cuda.get_current_stream().synchronize()
        transfer_seconds.append(time.perf_counter() - started)
        if not (
            np.array_equal(cpu_output, expected)
            and np.array_equal(cp.asnumpy(resident_output), expected)
            and np.array_equal(transferred_host, expected)
        ):
            raise EvidenceError(f"{definition.case_id} timed parity failed.")
        del (
            cpu_output,
            resident_output,
            transferred_input,
            transferred_output,
            transferred_host,
        )
    del device
    cp.cuda.get_current_stream().synchronize()
    memory = _measure_memory(cp, gpu_function, host, definition)
    cleanup = _drain_default_pools(cp)
    return {
        "case_id": definition.case_id,
        "shape": list(definition.shape),
        "radius": definition.radius,
        "which_outliers": definition.which_outliers,
        "rounds": rounds,
        "cpu_gpu_bitwise_equal": True,
        "samples": {
            "cpu_seconds": cpu_seconds,
            "gpu_resident_seconds": resident_seconds,
            "gpu_transfer_inclusive_seconds": transfer_seconds,
        },
        "summary": {
            "cpu_median_seconds": statistics.median(cpu_seconds),
            "gpu_resident_median_seconds": statistics.median(resident_seconds),
            "gpu_transfer_inclusive_median_seconds": statistics.median(
                transfer_seconds
            ),
        },
        "memory": memory,
        "cleanup": cleanup,
    }


def _measure_memory(cp, gpu_function, host, definition) -> dict[str, object]:
    estimate = _memory_estimate(definition)
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device = cp.asarray(host)
        used_with_input = int(pool.used_bytes())
        output = gpu_function(
            device,
            radius=definition.radius,
            which_outliers=definition.which_outliers,
        )
        cp.cuda.get_current_stream().synchronize()
        used_with_output = int(pool.used_bytes())
        observed_reserved = int(pool.total_bytes())
        output_independent = int(output.data.ptr) != int(device.data.ptr)
        del output, device
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    estimated_peak = estimate.total_device_peak_bytes + estimate.uncertainty_bytes
    covered = estimated_peak >= observed_reserved
    if not output_independent or not covered:
        raise EvidenceError(
            f"{definition.case_id} failed memory or output allocation evidence."
        )
    return {
        "scope": "isolated-cupy-memory-pool-reserved-high-water-v1",
        "model_id": estimate.model_id,
        "runtime_managed_peak_bytes": estimate.runtime_managed_peak_bytes,
        "total_device_peak_bytes": estimate.total_device_peak_bytes,
        "uncertainty_bytes": estimate.uncertainty_bytes,
        "estimated_peak_with_uncertainty_bytes": estimated_peak,
        "observed_reserved_bytes": observed_reserved,
        "observed_used_bytes_with_input": used_with_input,
        "observed_used_bytes_with_output": used_with_output,
        "output_shares_input_allocation": False,
        "estimate_covers_observed": True,
        **cleanup,
    }


def _memory_estimate(definition: PerformanceCase):
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import estimate_candidate_memory

    workload = WorkloadDescriptor(
        definition.case_id,
        OPERATION_ID,
        (definition.shape,),
        ("bool",),
        parameters=(
            ("radius", definition.radius),
            ("which_outliers", definition.which_outliers),
        ),
        resolved_spatial_ndim=2,
    )
    return estimate_candidate_memory(_gpu_spec(), workload)


def _warm_runtime(cp, gpu_function) -> None:
    source = cp.zeros((9, 11), dtype=bool)
    source[4, 5] = True
    output = gpu_function(source, radius=2.0, which_outliers=FOREGROUND)
    cp.cuda.get_current_stream().synchronize()
    del output, source
    _drain_default_pools(cp)


def _drain_pool(cp, pool) -> dict[str, int]:
    gc.collect()
    cp.cuda.get_current_stream().synchronize()
    pool.free_all_blocks()
    cp.cuda.get_current_stream().synchronize()
    used = int(pool.used_bytes())
    reserved = int(pool.total_bytes())
    if used or reserved:
        raise EvidenceError(
            f"Isolated CUDA pool did not drain (used={used}, reserved={reserved})."
        )
    return {
        "device_pool_used_bytes_after_cleanup": used,
        "device_pool_reserved_bytes_after_cleanup": reserved,
    }


def _drain_default_pools(cp) -> dict[str, int]:
    gc.collect()
    cp.cuda.get_current_stream().synchronize()
    device_pool = cp.get_default_memory_pool()
    pinned_pool = cp.get_default_pinned_memory_pool()
    device_pool.free_all_blocks()
    pinned_pool.free_all_blocks()
    cp.cuda.get_current_stream().synchronize()
    used = int(device_pool.used_bytes())
    reserved = int(device_pool.total_bytes())
    pinned_reserved = int(pinned_pool.n_free_blocks())
    if used or reserved or pinned_reserved:
        raise EvidenceError(
            "Default CUDA pools did not drain after synchronized cleanup."
        )
    return {
        "device_pool_used_bytes_after_cleanup": used,
        "device_pool_reserved_bytes_after_cleanup": reserved,
        "pinned_pool_free_blocks_after_cleanup": pinned_reserved,
    }


def _operation_functions():
    cpu_module = importlib.import_module("napari_vipp.core.remove_outliers")
    gpu_module = importlib.import_module(
        "napari_vipp.core.gpu.cupy_remove_binary_outliers"
    )
    return (
        cpu_module.remove_binary_outliers,
        gpu_module.remove_binary_outliers,
        gpu_module,
    )


def _gpu_spec():
    from napari_vipp.core.compute_specs import compute_specs_for

    specs = compute_specs_for(
        OPERATION_ID,
        include_cpu=False,
        allow_experimental=False,
    )
    if len(specs) != 1 or specs[0].implementation_id != IMPLEMENTATION_ID:
        raise EvidenceError("The public GPU implementation declaration is missing.")
    return specs[0]


def _operation_contract() -> dict[str, object]:
    spec = _gpu_spec()
    return {
        "operation_id": spec.operation_id,
        "implementation_id": spec.implementation_id,
        "implementation_version": spec.implementation_version,
        "runtime_id": spec.runtime_id,
        "library_id": spec.implementation_library_id,
        "callable_ref": spec.callable_ref,
        "admission_tier": spec.admission_tier.value,
        "validated_environment_policy_id": spec.validated_environment_policy_id,
        "parameter_policy_id": spec.parameter_policy_id,
        "workload_policy_id": spec.workload_policy_id,
        "parity_policy_id": spec.parity_policy_id,
        "memory_model_id": spec.memory_model_id,
        "boundary_policy_id": spec.boundary_policy_id,
        "progress_policy_id": spec.progress_policy_id,
        "cancellation_policy_id": spec.cancellation_policy_id,
        "supports_device_residency": spec.supports_device_residency,
        "public_input_dtypes": list(spec.input_ports[0].public_dtypes),
        "public_output_dtypes": list(spec.output_ports[0].public_dtypes),
        "limitations": list(spec.limitations),
    }


def _environment_record(cp, device_index: int) -> dict[str, object]:
    properties = cp.cuda.runtime.getDeviceProperties(device_index)
    name = properties.get("name", "")
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    major = int(properties.get("major", 0))
    minor = int(properties.get("minor", 0))
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python_implementation": platform.python_implementation(),
        "python": platform.python_version(),
        "python_abi": getattr(sys.implementation, "cache_tag", ""),
        # Keep durable evidence portable and avoid publishing a workstation-
        # specific user path. Package versions and the ABI tag retain the
        # interpreter identity needed for reproducibility.
        "python_executable": Path(sys.executable).name,
        "device_index": device_index,
        "device_name": str(name),
        "compute_capability": f"{major}.{minor}",
        "device_total_memory_bytes": int(properties.get("totalGlobalMem", 0)),
        "cuda_driver_version": str(cp.cuda.runtime.driverGetVersion()),
        "cuda_runtime_version": str(cp.cuda.runtime.runtimeGetVersion()),
    }


def _package_versions() -> dict[str, str]:
    result = {}
    for name in (
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy-cuda13x",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result


def _source_provenance() -> list[dict[str, str]]:
    records = []
    for relative in SOURCE_PROVENANCE_PATHS:
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.is_symlink():
            raise EvidenceError(f"Tracked source is unavailable: {relative.as_posix()}")
        records.append({"path": relative.as_posix(), "sha256": _file_sha256(path)})
    return records


def validate_existing(path: Path | str) -> Path:
    source = Path(path).expanduser().resolve()
    if not source.is_file() or source.is_symlink():
        raise EvidenceError("Evidence path must be a regular JSON file.")
    try:
        document = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise EvidenceError(f"Evidence is not strict JSON: {exc}") from exc
    _validate_document(document, require_current_sources=True)
    return source


def _validate_document(document: object, *, require_current_sources: bool) -> None:
    if not isinstance(document, dict):
        raise EvidenceError("Evidence root must be an object.")
    if (
        document.get("schema") != SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("kind") != EVIDENCE_KIND
        or document.get("status") != "pass"
        or document.get("portable_performance_claim") is not False
        or document.get("durable_optimizer_record") is not False
        or document.get("profile") not in {"quick", "full"}
    ):
        raise EvidenceError("Evidence root contract is invalid.")
    identity = _mapping(document.get("implementation"), "implementation")
    expected_identity = {
        "operation_id": OPERATION_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "implementation_version": IMPLEMENTATION_VERSION,
        "runtime_id": RUNTIME_ID,
        "library_id": LIBRARY_ID,
    }
    if identity != expected_identity:
        raise EvidenceError("Implementation identity does not match admission.")
    facets = _mapping(document.get("facets"), "facets")
    if facets != {facet: "pass" for facet in REQUIRED_FACETS}:
        raise EvidenceError("Required facet status is incomplete.")
    contract = _mapping(document.get("operation_contract"), "operation contract")
    if contract.get("validated_environment_policy_id") != ENVIRONMENT_POLICY_ID:
        raise EvidenceError("Unexpected validated environment policy.")
    if contract.get("memory_model_id") != "cupy-remove-binary-outliers-memory-v1":
        raise EvidenceError("Unexpected executable memory model.")
    if require_current_sources:
        if document.get("source_provenance") != _source_provenance():
            raise EvidenceError("Evidence source provenance is stale.")
        if contract != _operation_contract():
            raise EvidenceError("Evidence operation contract is stale.")

    admission = _mapping(document.get("admission"), "admission")
    if admission.get("status") != "pass":
        raise EvidenceError("Admission did not pass.")
    if not REQUIRED_ADMISSION_COVERAGE <= set(admission.get("coverage", ())):
        raise EvidenceError("Admission coverage is incomplete.")
    cases = admission.get("cases")
    if not isinstance(cases, list) or len(cases) != len(_admission_cases()):
        raise EvidenceError("Admission case list is incomplete.")
    for case in cases:
        item = _mapping(case, "admission case")
        if not all(
            item.get(key) is True
            for key in (
                "cpu_gpu_bitwise_equal",
                "input_immutable",
                "gpu_output_resident",
                "gpu_output_contiguous",
                "gpu_output_independent",
                "repeat_deterministic",
            )
        ):
            raise EvidenceError("An admission integrity check did not pass.")
        _require_cleanup(_mapping(item.get("cleanup"), "admission cleanup"))

    sweep = _mapping(document.get("parameter_sweep"), "parameter sweep")
    if not all(
        sweep.get(key) is True
        for key in (
            "fixed_source_kernel",
            "kernel_object_reused",
            "runtime_parameterized_radius",
            "runtime_parameterized_polarity",
        )
    ):
        raise EvidenceError("Fixed-kernel parameter evidence is incomplete.")
    if sweep.get("new_kernel_objects_during_sweep") != 0:
        raise EvidenceError("Parameter sweep created a kernel object.")
    lanes = _mapping_list(sweep.get("lanes"), "parameter lanes")
    lane_by_name = {item.get("parameter_name"): item for item in lanes}
    if tuple(lane_by_name.get("radius", {}).get("values", ())) != RADIUS_SEQUENCE:
        raise EvidenceError("Radius unseen/revisit sequence differs from admission.")
    if tuple(lane_by_name.get("which_outliers", {}).get("values", ())) != (
        POLARITY_SEQUENCE
    ):
        raise EvidenceError("Polarity sequence differs from admission.")
    for lane in lanes:
        if lane.get("all_exact") is not True:
            raise EvidenceError("A parameter lane failed exact parity.")
        steps = _mapping_list(lane.get("steps"), "parameter steps")
        if not any(item.get("occurrence") == "revisit" for item in steps):
            raise EvidenceError("A parameter lane omitted its revisit.")
        for step in steps:
            _finite_nonnegative(step.get("transfer_inclusive_seconds"), "sweep time")

    for key in ("metadata", "lifecycle", "fallback", "provenance"):
        if _mapping(document.get(key), key).get("status") != "pass":
            raise EvidenceError(f"{key} evidence did not pass.")
    lifecycle = _mapping(document["lifecycle"], "lifecycle")
    if not all(
        lifecycle.get(key) is True
        for key in (
            "cancellation_requested",
            "cancellation_observed",
            "post_cancellation_reuse_exact",
            "input_immutable",
        )
    ):
        raise EvidenceError("Lifecycle evidence is incomplete.")
    _require_cleanup(
        _mapping(lifecycle.get("isolated_pool_cleanup"), "lifecycle cleanup")
    )
    _require_default_cleanup(
        _mapping(
            lifecycle.get("default_pool_cleanup_after_reuse"),
            "lifecycle default cleanup",
        )
    )
    fallback = _mapping(document["fallback"], "fallback")
    if fallback.get("safe_cpu_fallback_case_count") != 2:
        raise EvidenceError("Safe CPU fallback coverage is incomplete.")
    if fallback.get("invalid_authored_case_count") != 3:
        raise EvidenceError("Invalid authoring coverage is incomplete.")
    provenance = _mapping(document["provenance"], "provenance")
    if not all(
        (
            provenance.get("actual_runtime_id") == RUNTIME_ID,
            provenance.get("actual_library_id") == LIBRARY_ID,
            provenance.get("actual_implementation_id") == IMPLEMENTATION_ID,
            provenance.get("actual_implementation_version") == IMPLEMENTATION_VERSION,
            provenance.get("fallback_used") is False,
            provenance.get("cleanup_succeeded") is True,
            provenance.get("cpu_gpu_bitwise_equal") is True,
        )
    ):
        raise EvidenceError("Production provenance evidence is incomplete.")

    performance = _mapping(document.get("performance"), "performance")
    if (
        performance.get("status") != "pass"
        or performance.get("all_memory_estimates_cover_observed") is not True
    ):
        raise EvidenceError("Performance or memory evidence did not pass.")
    results = _mapping_list(performance.get("results"), "performance results")
    if len(results) != len(_performance_cases(str(document["profile"]))):
        raise EvidenceError("Performance case list is incomplete.")
    for result in results:
        if result.get("cpu_gpu_bitwise_equal") is not True:
            raise EvidenceError("A timed case failed exact parity.")
        samples = _mapping(result.get("samples"), "timing samples")
        for name in (
            "cpu_seconds",
            "gpu_resident_seconds",
            "gpu_transfer_inclusive_seconds",
        ):
            values = samples.get(name)
            if not isinstance(values, list) or not values:
                raise EvidenceError(f"Timing series {name!r} is empty.")
            for value in values:
                _finite_nonnegative(value, name)
        memory = _mapping(result.get("memory"), "memory")
        if memory.get("estimate_covers_observed") is not True:
            raise EvidenceError("A memory estimate did not cover observation.")
        _require_cleanup(memory)
        _require_default_cleanup(_mapping(result.get("cleanup"), "cleanup"))
    _require_default_cleanup(
        _mapping(document.get("terminal_cleanup"), "terminal cleanup")
    )


def _require_cleanup(record: Mapping[str, object]) -> None:
    if (
        record.get("device_pool_used_bytes_after_cleanup") != 0
        or record.get("device_pool_reserved_bytes_after_cleanup") != 0
    ):
        raise EvidenceError("An isolated CUDA pool retained memory after cleanup.")


def _require_default_cleanup(record: Mapping[str, object]) -> None:
    _require_cleanup(record)
    if record.get("pinned_pool_free_blocks_after_cleanup") != 0:
        raise EvidenceError("Pinned CUDA pool retained blocks after cleanup.")


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{label} must be numeric.")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved < 0.0:
        raise EvidenceError(f"{label} must be finite and nonnegative.")
    return resolved


def _mapping(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be an object.")
    return value


def _mapping_list(value: object, label: str) -> list[dict[str, object]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise EvidenceError(f"{label} must be an array of objects.")
    return value


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _numpy():
    return importlib.import_module("numpy")


def _cupy():
    return importlib.import_module("cupy")


def _array_sha256(value) -> str:
    np = _numpy()
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path | str, document: Mapping[str, object]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


if __name__ == "__main__":
    raise SystemExit(main())
