#!/usr/bin/env python
"""Build exact CPU/CuPyX connected-components evidence on a real CUDA device.

The full profile proves exact label IDs, dtype, leading-block semantics, repeat
determinism, lifecycle cleanup, and the production memory estimate. It also
measures case-cold, synchronized resident, and transfer-inclusive execution
over practical 2D crossover sizes plus representative 3D and stack workloads.

Importing this module, asking for ``--help``, and ``--validate-existing`` do not
import CuPy or initialize CUDA.
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
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path

SCHEMA = "napari-vipp-cupyx-connected-components-evidence"
SCHEMA_VERSION = 1
OPERATION_ID = "label_connected_components"
IMPLEMENTATION_ID = "cupyx-connected-components-v1"
GENERATOR_ID = "numpy-pcg64-connected-components-mask-v1"
BENCHMARK_ROUNDS = 5
ADMISSION_REPEATS = 3
PLANE_EXTENTS = (256, 512, 1024, 2048)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs/benchmarks/connected-components-cupyx-windows-rtx5090.json"
)
SOURCE_PROVENANCE_PATHS = (
    Path("src/napari_vipp/core/connected_components.py"),
    Path("src/napari_vipp/core/gpu/cupy_connected_components.py"),
    Path("src/napari_vipp/core/compute_specs.py"),
    Path("src/napari_vipp/core/compute_policy.py"),
    Path("src/napari_vipp/core/compute_benchmark_adapter.py"),
    Path("scripts/benchmark_gpu_connected_components.py"),
)
REQUIRED_ADMISSION_COVERAGE = frozenset(
    {
        "dimension:2d",
        "dimension:3d",
        "connectivity:face",
        "connectivity:full",
        "pattern:sparse",
        "pattern:dense",
        "pattern:checkerboard",
        "axes:leading-blocks-2d",
        "axes:leading-blocks-3d",
        "labels:exact-int32-ids",
        "labels:restart-at-one",
        "repeat:deterministic",
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
        "operation_contract",
        "admission",
        "lifecycle",
        "performance",
    }
)


class EvidenceError(RuntimeError):
    """Raised when evidence is incomplete, stale, or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class AdmissionDefinition:
    case_id: str
    shape: tuple[int, ...]
    spatial_ndim: int
    connectivity: str
    pattern: str
    seed: int
    coverage: tuple[str, ...]

    @property
    def spatial_mode(self) -> str:
        return "2D YX" if self.spatial_ndim == 2 else "3D ZYX"


@dataclass(frozen=True, slots=True)
class PerformanceDefinition:
    case_id: str
    label: str
    shape: tuple[int, ...]
    spatial_ndim: int
    connectivity: str
    pattern: str
    seed: int
    family: str

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
        help="Validate JSON and Markdown without importing CuPy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_existing is not None:
        try:
            path = validate_existing(args.validate_existing)
        except (OSError, TypeError, ValueError, EvidenceError) as exc:
            print(f"Connected-components evidence is invalid: {exc}", file=sys.stderr)
            return 2
        print(f"Connected-components evidence is current: {path}")
        return 0

    output = args.output.resolve()
    markdown = (args.markdown or output.with_suffix(".md")).resolve()
    try:
        document = build_evidence(args.profile, args.device_index)
        _atomic_write_artifacts(output, markdown, document)
        validate_existing(output)
    except (OSError, TypeError, ValueError, EvidenceError) as exc:
        print(f"Connected-components benchmark failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {output}")
    print(f"Wrote {markdown}")
    return 0


def build_evidence(profile: str, device_index: int) -> dict[str, object]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'.")
    source_snapshot = _source_provenance()
    contract_snapshot = _operation_contract()
    np = _numpy()
    cp = _cupy()
    cpu_function, gpu_function = _operation_functions()
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
        _warm_runtime(cp, gpu_function)
        admission = _run_admission(cp, cpu_function, gpu_function)
        lifecycle = _run_lifecycle(cp, cpu_function, gpu_function)
        performance_results = []
        for definition in _performance_cases(profile):
            print(f"Timing {definition.case_id} ...", flush=True)
            performance_results.append(
                _run_performance_case(
                    cp,
                    cpu_function,
                    gpu_function,
                    definition,
                    rounds,
                )
            )

    performance = {
        "status": "pass",
        "rounds": rounds,
        "case_count": len(performance_results),
        "all_memory_estimates_cover_observed": all(
            result["memory"]["estimate_covers_observed"]
            for result in performance_results
        ),
        "results": performance_results,
        "crossovers": _crossover_summary(performance_results),
    }
    if not performance["all_memory_estimates_cover_observed"]:
        raise EvidenceError("The production memory estimate did not cover observation.")

    if source_snapshot != _source_provenance():
        raise EvidenceError(
            "Tracked source changed while evidence was being collected."
        )
    if contract_snapshot != _operation_contract():
        raise EvidenceError(
            "Operation contract changed while evidence was being collected."
        )

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
        "operation_contract": contract_snapshot,
        "admission": admission,
        "lifecycle": lifecycle,
        "performance": performance,
    }
    _validate_document_contract(document)
    return document


def _warm_runtime(cp, gpu_function) -> None:
    mask = cp.zeros((8, 8), dtype=cp.bool_)
    mask[1:3, 1:3] = True
    output = gpu_function(
        mask,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )
    cp.cuda.get_current_stream().synchronize()
    del output, mask
    cp.get_default_memory_pool().free_all_blocks()


def _admission_cases() -> tuple[AdmissionDefinition, ...]:
    definitions: list[AdmissionDefinition] = []
    for spatial_ndim, shape in ((2, (65, 67)), (3, (17, 31, 33))):
        dimension = f"dimension:{spatial_ndim}d"
        for pattern in ("sparse", "dense", "checkerboard"):
            for connectivity in ("Face connected", "Full connectivity"):
                short = "face" if connectivity.startswith("Face") else "full"
                definitions.append(
                    AdmissionDefinition(
                        case_id=(f"admit-{spatial_ndim}d-{pattern}-{short}"),
                        shape=shape,
                        spatial_ndim=spatial_ndim,
                        connectivity=connectivity,
                        pattern=pattern,
                        seed=20_260_802 + spatial_ndim * 100 + len(definitions),
                        coverage=(
                            dimension,
                            f"connectivity:{short}",
                            f"pattern:{pattern}",
                            "labels:exact-int32-ids",
                            "repeat:deterministic",
                        ),
                    )
                )
    for spatial_ndim, shape in ((2, (3, 41, 43)), (3, (2, 11, 23, 25))):
        for connectivity in ("Face connected", "Full connectivity"):
            short = "face" if connectivity.startswith("Face") else "full"
            definitions.append(
                AdmissionDefinition(
                    case_id=f"admit-leading-{spatial_ndim}d-{short}",
                    shape=shape,
                    spatial_ndim=spatial_ndim,
                    connectivity=connectivity,
                    pattern="leading-reset",
                    seed=20_260_900 + spatial_ndim,
                    coverage=(
                        f"dimension:{spatial_ndim}d",
                        f"connectivity:{short}",
                        f"axes:leading-blocks-{spatial_ndim}d",
                        "labels:exact-int32-ids",
                        "labels:restart-at-one",
                        "repeat:deterministic",
                    ),
                )
            )
    return tuple(definitions)


def _performance_cases(profile: str) -> tuple[PerformanceDefinition, ...]:
    extents = PLANE_EXTENTS if profile == "full" else PLANE_EXTENTS[:2]
    definitions: list[PerformanceDefinition] = []
    plane_profiles = (
        ("sparse", "Face connected"),
        ("dense", "Full connectivity"),
        ("checkerboard", "Face connected"),
    )
    for extent in extents:
        for pattern, connectivity in plane_profiles:
            short = "face" if connectivity.startswith("Face") else "full"
            definitions.append(
                PerformanceDefinition(
                    case_id=f"plane-{extent}-{pattern}-{short}",
                    label=f"{extent}² {pattern} {short}",
                    shape=(extent, extent),
                    spatial_ndim=2,
                    connectivity=connectivity,
                    pattern=pattern,
                    seed=50_900 + extent + len(definitions),
                    family=f"plane-{pattern}-{short}",
                )
            )
    representative = (
        PerformanceDefinition(
            "volume-48x128x128-sparse-face",
            "48×128×128 sparse face",
            (48, 128, 128),
            3,
            "Face connected",
            "sparse",
            51_901,
            "volume-sparse-face",
        ),
        PerformanceDefinition(
            "volume-64x192x192-dense-full",
            "64×192×192 dense full",
            (64, 192, 192),
            3,
            "Full connectivity",
            "dense",
            51_902,
            "volume-dense-full",
        ),
        PerformanceDefinition(
            "volume-32x256x256-checkerboard-face",
            "32×256×256 checkerboard face",
            (32, 256, 256),
            3,
            "Face connected",
            "checkerboard",
            51_903,
            "volume-checkerboard-face",
        ),
        PerformanceDefinition(
            "volume-64x512x512-sparse-face",
            "64×512×512 sparse face",
            (64, 512, 512),
            3,
            "Face connected",
            "sparse",
            51_906,
            "volume-confocal-sparse-face",
        ),
        PerformanceDefinition(
            "stack-8x512x512-sparse-face",
            "8×512² 2D sparse face",
            (8, 512, 512),
            2,
            "Face connected",
            "sparse",
            51_904,
            "stack-sparse-face",
        ),
        PerformanceDefinition(
            "stack-4x1024x1024-dense-full",
            "4×1024² 2D dense full",
            (4, 1024, 1024),
            2,
            "Full connectivity",
            "dense",
            51_905,
            "stack-dense-full",
        ),
    )
    if profile == "quick":
        representative = (representative[0], representative[-2])
    definitions.extend(representative)
    return tuple(definitions)


def _make_mask(
    shape: tuple[int, ...],
    pattern: str,
    seed: int,
    spatial_ndim: int,
):
    np = _numpy()
    if pattern == "sparse":
        mask = np.random.default_rng(seed).random(shape) < 0.03
    elif pattern == "dense":
        mask = np.random.default_rng(seed).random(shape) < 0.68
    elif pattern == "checkerboard":
        parity = np.zeros(shape, dtype=np.uint8)
        for axis, size in enumerate(shape):
            axis_shape = [1] * len(shape)
            axis_shape[axis] = size
            parity ^= (np.arange(size, dtype=np.uint8) & 1).reshape(axis_shape)
        mask = parity == 0
    elif pattern == "leading-reset":
        mask = np.zeros(shape, dtype=bool)
        leading_shape = shape[:-spatial_ndim]
        for index in np.ndindex(leading_shape):
            block = mask[index]
            first = (1,) * spatial_ndim
            second = tuple(max(size - 2, 0) for size in block.shape)
            block[first] = True
            block[second] = True
    else:
        raise ValueError(f"Unknown mask pattern {pattern!r}.")
    mask = np.ascontiguousarray(mask, dtype=bool)
    mask.setflags(write=False)
    return mask


def _run_admission(cp, cpu_function, gpu_function) -> dict[str, object]:
    cases = []
    coverage: set[str] = set()
    for definition in _admission_cases():
        data = _make_mask(
            definition.shape,
            definition.pattern,
            definition.seed,
            definition.spatial_ndim,
        )
        expected = cpu_function(
            data,
            spatial_mode=definition.spatial_mode,
            connectivity=definition.connectivity,
        )
        pool = cp.cuda.MemoryPool()
        gpu_hashes: list[str] = []
        mismatch_count = 0
        input_immutable = True
        gpu_output_contiguous = True
        with cp.cuda.using_allocator(pool.malloc):
            device_input = cp.asarray(data)
            before = device_input.copy()
            outputs = []
            for _ in range(ADMISSION_REPEATS):
                output = gpu_function(
                    device_input,
                    spatial_mode=definition.spatial_mode,
                    connectivity=definition.connectivity,
                )
                cp.cuda.get_current_stream().synchronize()
                if not isinstance(output, cp.ndarray) or output.dtype != cp.int32:
                    raise EvidenceError(
                        f"{definition.case_id} did not return resident int32 labels."
                    )
                gpu_output_contiguous &= bool(output.flags.c_contiguous)
                host_output = cp.asnumpy(output)
                mismatch_count += int((host_output != expected).sum())
                gpu_hashes.append(_array_sha256(host_output))
                outputs.append(output)
            input_immutable = bool(cp.all(device_input == before).item())
            del output, host_output, outputs, before, device_input
            cp.cuda.get_current_stream().synchronize()
        cleanup = _drain_pool(cp, pool)
        exact = mismatch_count == 0 and len(set(gpu_hashes)) == 1
        if not exact or not input_immutable or not gpu_output_contiguous:
            raise EvidenceError(f"Admission failed for {definition.case_id}.")
        block_counts = _block_component_counts(expected, definition.spatial_ndim)
        restart_verified = _block_ids_restart(expected, definition.spatial_ndim)
        if definition.pattern == "leading-reset" and not restart_verified:
            raise EvidenceError(
                f"Leading IDs did not restart for {definition.case_id}."
            )
        coverage.update(definition.coverage)
        cases.append(
            {
                "case_id": definition.case_id,
                "shape": list(definition.shape),
                "spatial_ndim": definition.spatial_ndim,
                "spatial_mode": definition.spatial_mode,
                "connectivity": definition.connectivity,
                "pattern": definition.pattern,
                "seed": definition.seed,
                "coverage": sorted(definition.coverage),
                "foreground_count": int(data.sum()),
                "block_component_counts": block_counts,
                "ids_restart_verified": restart_verified,
                "exact": exact,
                "mismatch_count": mismatch_count,
                "cpu_dtype": str(expected.dtype),
                "gpu_dtype": "int32",
                "gpu_output_resident": True,
                "gpu_output_contiguous": gpu_output_contiguous,
                "input_immutable": input_immutable,
                "repeat_count": ADMISSION_REPEATS,
                "repeat_deterministic": len(set(gpu_hashes)) == 1,
                "input_sha256": _array_sha256(data),
                "cpu_output_sha256": _array_sha256(expected),
                "gpu_output_sha256": gpu_hashes[0],
                "gpu_repeat_sha256": gpu_hashes,
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
        "parity_profile": "bitwise-identical-int32-label-ids-v1",
        "coverage": sorted(coverage),
        "cases": cases,
    }


def _run_lifecycle(cp, cpu_function, gpu_function) -> dict[str, object]:
    from napari_vipp.core.progress import OperationCancelled, ProgressContext

    data = _make_mask((3, 512, 512), "sparse", 52_509, 2)
    pool = cp.cuda.MemoryPool()
    updates: list[dict[str, object]] = []
    cancelled = False

    def reporter(update) -> None:
        nonlocal cancelled
        updates.append(
            {
                "current": int(update.current),
                "total": int(update.total),
                "message": str(update.message),
            }
        )
        if int(update.current) == 1:
            cancelled = True

    cancellation_observed = False
    reuse_exact = False
    with cp.cuda.using_allocator(pool.malloc):
        source = cp.asarray(data)
        progress = ProgressContext(cancelled=lambda: cancelled, reporter=reporter)
        try:
            gpu_function(
                source,
                spatial_mode="2D YX",
                connectivity="Face connected",
                progress=progress,
            )
        except OperationCancelled as exc:
            cancellation_observed = True
            exc.__traceback__ = None
        if not cancellation_observed:
            raise EvidenceError("Connected-components cancellation was not observed.")
        cancelled = False
        reuse_input = data[0]
        expected = cpu_function(
            reuse_input,
            spatial_mode="2D YX",
            connectivity="Face connected",
        )
        reuse_output = gpu_function(
            cp.asarray(reuse_input),
            spatial_mode="2D YX",
            connectivity="Face connected",
        )
        cp.cuda.get_current_stream().synchronize()
        reuse_exact = bool((cp.asnumpy(reuse_output) == expected).all())
        used_before_cleanup = int(pool.used_bytes())
        reserved_before_cleanup = int(pool.total_bytes())
        del reuse_output, source
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    if not reuse_exact:
        raise EvidenceError("CUDA provider was not reusable after cancellation.")
    expected_updates = [
        {"current": 0, "total": 3, "message": "Connected-component blocks"},
        {"current": 1, "total": 3, "message": "Connected-component blocks"},
    ]
    if updates != expected_updates:
        raise EvidenceError(f"Unexpected cancellation progress: {updates!r}.")
    return {
        "status": "pass",
        "cancellation_requested": True,
        "cancellation_observed": cancellation_observed,
        "boundary": "synchronized-leading-spatial-block-v1",
        "reported_progress": updates,
        "post_cancellation_reuse_exact": reuse_exact,
        "used_bytes_before_cleanup": used_before_cleanup,
        "reserved_bytes_before_cleanup": reserved_before_cleanup,
        **cleanup,
    }


def _run_performance_case(
    cp,
    cpu_function,
    gpu_function,
    definition: PerformanceDefinition,
    rounds: int,
) -> dict[str, object]:
    data = _make_mask(
        definition.shape,
        definition.pattern,
        definition.seed,
        definition.spatial_ndim,
    )
    parameters = {
        "spatial_mode": definition.spatial_mode,
        "connectivity": definition.connectivity,
    }

    started = time.perf_counter()
    cpu_reference = cpu_function(data, **parameters)
    cpu_cold = time.perf_counter() - started
    cpu_seconds = []
    for _ in range(rounds):
        started = time.perf_counter()
        cpu_output = cpu_function(data, **parameters)
        cpu_seconds.append(time.perf_counter() - started)
        del cpu_output

    timing_pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(timing_pool.malloc):
        started = time.perf_counter()
        cold_input = cp.asarray(data)
        cold_output = gpu_function(cold_input, **parameters)
        cold_host = cp.asnumpy(cold_output)
        cp.cuda.get_current_stream().synchronize()
        gpu_cold = time.perf_counter() - started
        del cold_output, cold_input

        resident_input = cp.asarray(data)
        warm_output = gpu_function(resident_input, **parameters)
        cp.cuda.get_current_stream().synchronize()
        del warm_output
        gpu_resident_seconds = []
        for _ in range(rounds):
            started = time.perf_counter()
            resident_output = gpu_function(resident_input, **parameters)
            cp.cuda.get_current_stream().synchronize()
            gpu_resident_seconds.append(time.perf_counter() - started)
            del resident_output

        gpu_transfer_inclusive_seconds = []
        for _ in range(rounds):
            started = time.perf_counter()
            transfer_input = cp.asarray(data)
            transfer_output = gpu_function(transfer_input, **parameters)
            cp.asnumpy(transfer_output)
            cp.cuda.get_current_stream().synchronize()
            gpu_transfer_inclusive_seconds.append(time.perf_counter() - started)
            del transfer_output, transfer_input
        del resident_input
        cp.cuda.get_current_stream().synchronize()
    timing_cleanup = _drain_pool(cp, timing_pool)

    mismatch_count = int((cold_host != cpu_reference).sum())
    if mismatch_count:
        raise EvidenceError(f"Performance parity failed for {definition.case_id}.")
    memory = _measure_memory(cp, gpu_function, data, definition)
    summary = _timing_summary(
        cpu_seconds,
        gpu_resident_seconds,
        gpu_transfer_inclusive_seconds,
        cpu_case_cold_seconds=cpu_cold,
        gpu_case_cold_transfer_inclusive_seconds=gpu_cold,
    )
    return {
        "case_id": definition.case_id,
        "label": definition.label,
        "family": definition.family,
        "shape": list(definition.shape),
        "spatial_ndim": definition.spatial_ndim,
        "spatial_mode": definition.spatial_mode,
        "connectivity": definition.connectivity,
        "pattern": definition.pattern,
        "seed": definition.seed,
        "element_count": int(data.size),
        "input_bytes": int(data.nbytes),
        "foreground_count": int(data.sum()),
        "input_sha256": _array_sha256(data),
        "parity": {
            "profile": "bitwise-identical-int32-label-ids-v1",
            "passed": True,
            "mismatch_count": mismatch_count,
            "cpu_dtype": str(cpu_reference.dtype),
            "gpu_dtype": str(cold_host.dtype),
            "gpu_output_resident": True,
            "cpu_output_sha256": _array_sha256(cpu_reference),
            "gpu_output_sha256": _array_sha256(cold_host),
        },
        "samples": {
            "cpu_seconds": cpu_seconds,
            "gpu_resident_seconds": gpu_resident_seconds,
            "gpu_transfer_inclusive_seconds": gpu_transfer_inclusive_seconds,
            "cpu_case_cold_seconds": cpu_cold,
            "gpu_case_cold_transfer_inclusive_seconds": gpu_cold,
        },
        "summary": summary,
        "memory": memory,
        "cleanup": timing_cleanup,
    }


def _measure_memory(cp, gpu_function, data, definition) -> dict[str, object]:
    pool = cp.cuda.MemoryPool()
    parameters = {
        "spatial_mode": definition.spatial_mode,
        "connectivity": definition.connectivity,
    }
    with cp.cuda.using_allocator(pool.malloc):
        device_input = cp.asarray(data)
        used_with_input = int(pool.used_bytes())
        output = gpu_function(device_input, **parameters)
        cp.cuda.get_current_stream().synchronize()
        used_with_output = int(pool.used_bytes())
        observed_reserved = int(pool.total_bytes())
        del output, device_input
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    estimated = _estimated_peak_bytes(definition.shape, definition.spatial_ndim)
    return {
        "scope": "isolated-cupy-memory-pool-reserved-high-water-v1",
        "model_id": "cupyx-connected-components-memory-v1",
        "estimated_peak_bytes": estimated,
        "observed_reserved_bytes": observed_reserved,
        "observed_used_bytes_with_input": used_with_input,
        "observed_used_bytes_with_output": used_with_output,
        "observed_to_estimated_ratio": observed_reserved / estimated,
        "estimate_covers_observed": estimated >= observed_reserved,
        **cleanup,
    }


def _estimated_peak_bytes(shape: Sequence[int], spatial_ndim: int) -> int:
    element_count = math.prod(int(size) for size in shape)
    block_elements = math.prod(int(size) for size in shape[-spatial_ndim:])
    return element_count * 5 + block_elements * 7


def _timing_summary(
    cpu_seconds: Sequence[float],
    gpu_resident_seconds: Sequence[float],
    gpu_transfer_inclusive_seconds: Sequence[float],
    *,
    cpu_case_cold_seconds: float,
    gpu_case_cold_transfer_inclusive_seconds: float,
) -> dict[str, object]:
    cpu_median = statistics.median(cpu_seconds)
    resident_median = statistics.median(gpu_resident_seconds)
    transfer_median = statistics.median(gpu_transfer_inclusive_seconds)
    decision = "GPU-CuPyX" if transfer_median < cpu_median else "CPU"
    return {
        "cpu_median_seconds": cpu_median,
        "gpu_resident_median_seconds": resident_median,
        "gpu_transfer_inclusive_median_seconds": transfer_median,
        "cpu_case_cold_seconds": cpu_case_cold_seconds,
        "gpu_case_cold_transfer_inclusive_seconds": (
            gpu_case_cold_transfer_inclusive_seconds
        ),
        "gpu_resident_speedup": cpu_median / resident_median,
        "gpu_transfer_inclusive_speedup": cpu_median / transfer_median,
        "gpu_case_cold_speedup": (
            cpu_case_cold_seconds / gpu_case_cold_transfer_inclusive_seconds
        ),
        "transfer_and_allocation_overhead_seconds": max(
            transfer_median - resident_median,
            0.0,
        ),
        "screening_choice": decision,
    }


def _crossover_summary(
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    families: dict[str, list[Mapping[str, object]]] = {}
    for result in results:
        if len(result["shape"]) != 2:
            continue
        families.setdefault(str(result["family"]), []).append(result)
    summaries = []
    for family in sorted(families):
        ordered = sorted(families[family], key=lambda result: int(result["shape"][0]))
        end_to_end = next(
            (
                int(result["shape"][0])
                for result in ordered
                if result["summary"]["gpu_transfer_inclusive_speedup"] > 1.0
            ),
            None,
        )
        resident = next(
            (
                int(result["shape"][0])
                for result in ordered
                if result["summary"]["gpu_resident_speedup"] > 1.0
            ),
            None,
        )
        summaries.append(
            {
                "family": family,
                "tested_extents": [int(result["shape"][0]) for result in ordered],
                "first_gpu_faster_transfer_inclusive_extent": end_to_end,
                "first_gpu_faster_resident_extent": resident,
            }
        )
    return summaries


def _block_component_counts(labels, spatial_ndim: int) -> list[int]:
    np = _numpy()
    leading_shape = labels.shape[:-spatial_ndim]
    if not leading_shape:
        return [int(labels.max(initial=0))]
    return [int(labels[index].max(initial=0)) for index in np.ndindex(leading_shape)]


def _block_ids_restart(labels, spatial_ndim: int) -> bool:
    np = _numpy()
    leading_shape = labels.shape[:-spatial_ndim]
    indexes = (None,) if not leading_shape else np.ndindex(leading_shape)
    for index in indexes:
        block = labels if index is None else labels[index]
        unique = np.unique(block)
        positive = unique[unique > 0]
        if positive.size and not np.array_equal(
            positive,
            np.arange(1, positive.size + 1, dtype=positive.dtype),
        ):
            return False
    return True


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


def _method_record(profile: str, rounds: int) -> dict[str, object]:
    return {
        "profile": profile,
        "generator": GENERATOR_ID,
        "admission_repeats": ADMISSION_REPEATS,
        "benchmark_rounds": rounds,
        "cpu_timing_scope": "case-cold-and-warm-host-scipy-v1",
        "gpu_resident_timing_scope": "synchronized-resident-compute-v1",
        "gpu_transfer_inclusive_timing_scope": (
            "host-to-device-plus-compute-plus-device-to-host-synchronized-v1"
        ),
        "gpu_case_cold_scope": (
            "empty-private-pool-allocation-and-transfers-after-runtime-warmup-v1"
        ),
        "memory_observation_scope": (
            "isolated-cupy-memory-pool-reserved-high-water-v1"
        ),
        "memory_model": (
            "full-input-and-int32-output-plus-seven-bytes-per-active-spatial-block"
        ),
        "parity": "bitwise-identical-int32-label-ids-v1",
        "cancellation": "synchronized-leading-spatial-block-boundary-v1",
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
    packages: dict[str, str] = {
        "numpy": str(np.__version__),
        "cupy": str(cp.__version__),
    }
    for distribution in ("scipy", "cupy-cuda13x", "napari-vipp"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed-as-distribution"
    return packages


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


def _operation_contract() -> dict[str, object]:
    from napari_vipp.core.compute_specs import accelerator_compute_specs

    matches = [
        spec
        for spec in accelerator_compute_specs()
        if spec.operation_id == OPERATION_ID
        and spec.implementation_id == IMPLEMENTATION_ID
    ]
    if len(matches) != 1:
        raise EvidenceError(
            f"Expected one {OPERATION_ID}/{IMPLEMENTATION_ID} compute contract."
        )
    snapshot = _canonical_value(matches[0])
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "snapshot": snapshot,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


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


def _array_sha256(array) -> str:
    np = _numpy()
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def render_markdown(document: Mapping[str, object]) -> str:
    environment = document["environment"]
    admission = document["admission"]
    lifecycle = document["lifecycle"]
    performance = document["performance"]
    lines = [
        "# Connected Components CPU/CuPyX evidence",
        "",
        f"Generated: `{document['created_utc']}`",
        "",
        "This is machine-local screening evidence, not a portable performance claim.",
        "",
        "## Outcome",
        "",
        (
            f"- Exact admission cases: **{admission['case_count']} / "
            f"{admission['case_count']} passed**."
        ),
        (
            f"- GPU: **{environment['device_name']}**; CUDA runtime "
            f"`{environment['cuda_runtime_version']}`."
        ),
        (
            f"- Cancellation and cleanup: **{lifecycle['status']}** at "
            "synchronized leading-block boundaries."
        ),
        (
            "- Memory estimate covered every observed private-pool high-water "
            f"mark: **{performance['all_memory_estimates_cover_observed']}**."
        ),
        (
            "- Labels must match SciPy bit for bit as native `int32`; "
            "equivalence up to relabeling is insufficient."
        ),
        "- Leading non-spatial blocks are independent and restart label IDs at one.",
        "",
        "## Timing method",
        "",
        "- CPU: case-cold call plus warm host medians.",
        (
            "- GPU resident: synchronized compute with input already on device "
            "and output left resident."
        ),
        (
            "- GPU transfer-inclusive: host-to-device, compute, device-to-host, "
            "then synchronization."
        ),
        (
            "- GPU case-cold: empty private allocator pool after one "
            "process-level runtime warmup."
        ),
        (
            "- VRAM observation: isolated CuPy pool reserved high-water; it is "
            "not device-wide telemetry."
        ),
        "",
        "## Measured workloads",
        "",
        (
            "| Workload | Shape | Pattern / connectivity | CPU median | GPU "
            "resident | GPU transfer-inclusive | E2E speedup | VRAM observed / "
            "estimated | Choice |"
        ),
        "|---|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in performance["results"]:
        summary = result["summary"]
        memory = result["memory"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(result["case_id"]),
                    "×".join(str(size) for size in result["shape"]),
                    f"{result['pattern']} / {result['connectivity']}",
                    _seconds(summary["cpu_median_seconds"]),
                    _seconds(summary["gpu_resident_median_seconds"]),
                    _seconds(summary["gpu_transfer_inclusive_median_seconds"]),
                    f"{summary['gpu_transfer_inclusive_speedup']:.2f}×",
                    (
                        f"{_mib(memory['observed_reserved_bytes'])} / "
                        f"{_mib(memory['estimated_peak_bytes'])}"
                    ),
                    str(summary["screening_choice"]),
                )
            )
            + " |"
        )
    lines.extend(["", "## Plane crossover screening", ""])
    for crossover in performance["crossovers"]:
        end_to_end = crossover["first_gpu_faster_transfer_inclusive_extent"]
        resident = crossover["first_gpu_faster_resident_extent"]
        lines.append(
            f"- `{crossover['family']}`: resident crossover `{resident}`; "
            f"transfer-inclusive crossover `{end_to_end}` among tested extents "
            f"`{crossover['tested_extents']}`."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "Sparse masks with many isolated components can keep CPU "
                "competitive to larger extents; dense and checkerboard workloads "
                "can strongly favor resident CuPyX. Auto selection should "
                "therefore use measured workload records rather than a size-only "
                "rule."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def validate_existing(path: Path | str) -> Path:
    artifact = Path(path).resolve()
    raw = artifact.read_text(encoding="utf-8")
    document = json.loads(raw)
    if raw != _canonical_json(document):
        raise EvidenceError("JSON is not in canonical sorted, indented form.")
    _validate_document_contract(document)
    markdown_path = artifact.with_suffix(".md")
    if not markdown_path.is_file():
        raise EvidenceError(f"Markdown companion is missing: {markdown_path}.")
    if markdown_path.read_text(encoding="utf-8") != render_markdown(document):
        raise EvidenceError("Markdown companion does not match canonical rendering.")
    return artifact


def _validate_document_contract(document: object) -> None:
    if not isinstance(document, Mapping):
        raise EvidenceError("Evidence root must be an object.")
    if set(document) != _ROOT_KEYS:
        raise EvidenceError("Evidence root fields differ from the canonical schema.")
    if (
        document.get("schema") != SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise EvidenceError("Evidence schema or version is unsupported.")
    if document.get("profile") not in {"quick", "full"}:
        raise EvidenceError("Evidence profile is invalid.")
    if document.get("portable_performance_claim") is not False:
        raise EvidenceError("Evidence must not claim portable performance.")
    if document.get("durable_optimizer_record") is not False:
        raise EvidenceError("Evidence must not claim to be an optimizer record.")
    if document.get("source_provenance") != _source_provenance():
        raise EvidenceError("Source provenance fingerprints are stale.")
    if document.get("operation_contract") != _operation_contract():
        raise EvidenceError("Operation-specific compute contract is stale.")
    if not isinstance(document.get("environment"), Mapping):
        raise EvidenceError("Environment record is missing.")
    if not isinstance(document.get("packages"), Mapping):
        raise EvidenceError("Package record is missing.")

    profile = str(document["profile"])
    performance = _require_mapping(document.get("performance"), "performance")
    rounds = int(performance.get("rounds", 0))
    if document.get("method") != _method_record(profile, rounds):
        raise EvidenceError("Method record does not match the profile and rounds.")
    _validate_admission(document.get("admission"))
    _validate_lifecycle(document.get("lifecycle"))
    _validate_performance(performance, profile, rounds)


def _validate_admission(value: object) -> None:
    admission = _require_mapping(value, "admission")
    cases = admission.get("cases")
    if not isinstance(cases, list):
        raise EvidenceError("Admission cases must be a list.")
    expected_ids = [case.case_id for case in _admission_cases()]
    if [
        case.get("case_id") for case in cases if isinstance(case, Mapping)
    ] != expected_ids:
        raise EvidenceError("Admission case manifest differs from the required matrix.")
    if admission.get("status") != "pass" or admission.get("case_count") != len(cases):
        raise EvidenceError("Admission status or case count is invalid.")
    if admission.get("repeat_count") != ADMISSION_REPEATS:
        raise EvidenceError("Admission repeat count is invalid.")
    coverage = set(admission.get("coverage", ()))
    if not REQUIRED_ADMISSION_COVERAGE <= coverage:
        raise EvidenceError("Admission coverage is incomplete.")
    for case in cases:
        item = _require_mapping(case, "admission case")
        if (
            item.get("exact") is not True
            or item.get("mismatch_count") != 0
            or item.get("cpu_dtype") != "int32"
            or item.get("gpu_dtype") != "int32"
            or item.get("gpu_output_resident") is not True
            or item.get("gpu_output_contiguous") is not True
            or item.get("input_immutable") is not True
            or item.get("repeat_deterministic") is not True
            or item.get("repeat_count") != ADMISSION_REPEATS
        ):
            raise EvidenceError("Admission parity or residency claim is invalid.")
        hashes = item.get("gpu_repeat_sha256")
        if not isinstance(hashes, list) or len(hashes) != ADMISSION_REPEATS:
            raise EvidenceError("Admission repeat hashes are incomplete.")
        if len(set(hashes)) != 1 or hashes[0] != item.get("cpu_output_sha256"):
            raise EvidenceError("Admission labels are not exactly deterministic.")
        _validate_cleanup(item.get("cleanup"), "admission cleanup")


def _validate_lifecycle(value: object) -> None:
    lifecycle = _require_mapping(value, "lifecycle")
    if (
        lifecycle.get("status") != "pass"
        or lifecycle.get("cancellation_requested") is not True
        or lifecycle.get("cancellation_observed") is not True
        or lifecycle.get("post_cancellation_reuse_exact") is not True
        or lifecycle.get("boundary") != "synchronized-leading-spatial-block-v1"
    ):
        raise EvidenceError("Lifecycle cancellation contract is invalid.")
    expected_progress = [
        {"current": 0, "total": 3, "message": "Connected-component blocks"},
        {"current": 1, "total": 3, "message": "Connected-component blocks"},
    ]
    if lifecycle.get("reported_progress") != expected_progress:
        raise EvidenceError("Lifecycle progress does not prove a block boundary.")
    _validate_cleanup(lifecycle, "lifecycle cleanup")


def _validate_performance(
    performance: Mapping[str, object],
    profile: str,
    rounds: int,
) -> None:
    if performance.get("status") != "pass" or rounds < 3:
        raise EvidenceError("Performance status or round count is invalid.")
    cases = performance.get("results")
    if not isinstance(cases, list):
        raise EvidenceError("Performance results must be a list.")
    expected_definitions = _performance_cases(profile)
    if [case.get("case_id") for case in cases if isinstance(case, Mapping)] != [
        definition.case_id for definition in expected_definitions
    ]:
        raise EvidenceError("Performance case manifest differs from the profile.")
    if performance.get("case_count") != len(cases):
        raise EvidenceError("Performance case count is invalid.")
    for case, definition in zip(cases, expected_definitions, strict=True):
        item = _require_mapping(case, "performance case")
        parity = _require_mapping(item.get("parity"), "performance parity")
        if (
            parity.get("passed") is not True
            or parity.get("mismatch_count") != 0
            or parity.get("cpu_dtype") != "int32"
            or parity.get("gpu_dtype") != "int32"
            or parity.get("gpu_output_resident") is not True
            or parity.get("cpu_output_sha256") != parity.get("gpu_output_sha256")
        ):
            raise EvidenceError("Performance parity is invalid.")
        samples = _require_mapping(item.get("samples"), "performance samples")
        cpu = _positive_samples(samples.get("cpu_seconds"), rounds, "CPU")
        resident = _positive_samples(
            samples.get("gpu_resident_seconds"), rounds, "GPU resident"
        )
        transfer = _positive_samples(
            samples.get("gpu_transfer_inclusive_seconds"),
            rounds,
            "GPU transfer-inclusive",
        )
        cpu_cold = _positive_finite(samples.get("cpu_case_cold_seconds"), "CPU cold")
        gpu_cold = _positive_finite(
            samples.get("gpu_case_cold_transfer_inclusive_seconds"),
            "GPU cold",
        )
        expected_summary = _timing_summary(
            cpu,
            resident,
            transfer,
            cpu_case_cold_seconds=cpu_cold,
            gpu_case_cold_transfer_inclusive_seconds=gpu_cold,
        )
        if item.get("summary") != expected_summary:
            raise EvidenceError("Performance timing summary is inconsistent.")
        memory = _require_mapping(item.get("memory"), "performance memory")
        estimated = _estimated_peak_bytes(definition.shape, definition.spatial_ndim)
        observed = int(memory.get("observed_reserved_bytes", -1))
        if (
            memory.get("model_id") != "cupyx-connected-components-memory-v1"
            or memory.get("estimated_peak_bytes") != estimated
            or observed < 0
            or memory.get("estimate_covers_observed") is not (estimated >= observed)
            or not math.isclose(
                float(memory.get("observed_to_estimated_ratio", -1)),
                observed / estimated,
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            raise EvidenceError("Performance memory evidence is inconsistent.")
        _validate_cleanup(memory, "memory cleanup")
        _validate_cleanup(item.get("cleanup"), "timing cleanup")
    all_covered = all(
        bool(case["memory"]["estimate_covers_observed"]) for case in cases
    )
    if performance.get("all_memory_estimates_cover_observed") is not all_covered:
        raise EvidenceError("Aggregate memory coverage is inconsistent.")
    if not all_covered:
        raise EvidenceError("At least one memory estimate did not cover observation.")
    if performance.get("crossovers") != _crossover_summary(cases):
        raise EvidenceError("Crossover summary is inconsistent.")


def _validate_cleanup(value: object, name: str) -> None:
    cleanup = _require_mapping(value, name)
    if (
        cleanup.get("device_pool_used_bytes_after_cleanup") != 0
        or cleanup.get("device_pool_reserved_bytes_after_cleanup") != 0
    ):
        raise EvidenceError(f"{name} did not drain the private CUDA pool.")


def _positive_samples(value: object, count: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != count:
        raise EvidenceError(f"{name} samples are incomplete.")
    return [_positive_finite(item, name) for item in value]


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise EvidenceError(f"{name} must be a positive finite number.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise EvidenceError(f"{name} must be a positive finite number.")
    return number


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise EvidenceError(f"{name} must be an object.")
    return value


def _atomic_write_artifacts(
    output: Path,
    markdown: Path,
    document: Mapping[str, object],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(output, _canonical_json(document))
    _atomic_write_text(markdown, render_markdown(document))


def _canonical_json(document: object) -> str:
    return (
        json.dumps(
            document,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )


def _atomic_write_text(path: Path, content: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _seconds(value: object) -> str:
    seconds = float(value)
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f} µs"
    if seconds < 1.0:
        return f"{seconds * 1_000:.2f} ms"
    return f"{seconds:.3f} s"


def _mib(value: object) -> str:
    return f"{int(value) / (1024**2):.2f} MiB"


@cache
def _numpy():
    return importlib.import_module("numpy")


@cache
def _cupy():
    return importlib.import_module("cupy")


@cache
def _operation_functions():
    cpu_module = importlib.import_module("napari_vipp.core.connected_components")
    gpu_module = importlib.import_module(
        "napari_vipp.core.gpu.cupy_connected_components"
    )
    return (
        cpu_module.label_connected_components,
        gpu_module.label_connected_components,
    )


if __name__ == "__main__":
    raise SystemExit(main())
