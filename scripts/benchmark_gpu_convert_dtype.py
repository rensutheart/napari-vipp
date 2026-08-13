#!/usr/bin/env python
"""Qualify VIPP's public lossless Convert Dtype CUDA implementation.

The evidence is intentionally limited to the declared public region:
``uint8``/``uint16`` inputs converted to resident ``float32`` with Preserve
scaling.  It proves bitwise CPU parity, awkward input handling, metadata,
input integrity, memory accounting, cancellation/reuse, cleanup, explicit CPU
fallback, source provenance, and synchronized resident/transfer-inclusive
timing on a real CUDA device.

Importing this module, asking for ``--help``, or validating an existing JSON
artifact does not import CuPy and does not initialize CUDA.
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "napari-vipp-cupyx-convert-dtype-evidence"
SCHEMA_VERSION = 1
OPERATION_ID = "convert_dtype"
IMPLEMENTATION_ID = "cupyx-convert-dtype-preserve-f32-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "docs/benchmarks/convert-dtype-cupyx-local.json"
ADMISSION_REPEATS = 3
QUICK_ROUNDS = 3
FULL_ROUNDS = 7
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
SOURCE_PROVENANCE_PATHS = (
    Path("src/napari_vipp/core/operations.py"),
    Path("src/napari_vipp/core/gpu/cupy_convert_dtype.py"),
    Path("src/napari_vipp/core/compute_specs.py"),
    Path("src/napari_vipp/core/compute_policy.py"),
    Path("src/napari_vipp/core/compute_benchmark_adapter.py"),
    Path("scripts/benchmark_gpu_convert_dtype.py"),
)
REQUIRED_ADMISSION_COVERAGE = frozenset(
    {
        "dtype:uint8",
        "dtype:uint16",
        "rank:1",
        "rank:2",
        "rank:3",
        "layout:contiguous",
        "layout:noncontiguous",
        "values:empty",
        "values:boundaries",
        "values:uint16-maximum",
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
        "implementation",
        "environment",
        "packages",
        "source_provenance",
        "operation_contract",
        "method",
        "facets",
        "admission",
        "metadata",
        "lifecycle",
        "fallback",
        "performance",
    }
)


class EvidenceError(RuntimeError):
    """Raised when evidence is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class AdmissionCase:
    case_id: str
    kind: str
    coverage: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerformanceCase:
    case_id: str
    shape: tuple[int, ...]
    dtype: str


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
            print(f"Convert Dtype evidence is current: {validated}")
            return 0
        if args.device_index < 0:
            raise ValueError("device index must be nonnegative")
        document = build_evidence(args.profile, args.device_index)
        _atomic_write_json(args.output.resolve(), document)
        validate_existing(args.output.resolve())
    except (EvidenceError, OSError, TypeError, ValueError) as exc:
        print(f"Convert Dtype evidence failed: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {args.output.resolve()}")
    return 0


def build_evidence(profile: str, device_index: int) -> dict[str, object]:
    if profile not in {"quick", "full"}:
        raise ValueError("profile must be 'quick' or 'full'.")
    source_snapshot = _source_provenance()
    contract_snapshot = _operation_contract()
    np = _numpy()
    cp = _cupy()
    cpu_function, gpu_function = _operation_functions()
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:  # pragma: no cover - machine-specific failure
        raise EvidenceError(f"CUDA runtime probe failed: {exc}") from exc
    if device_index >= device_count:
        raise EvidenceError(
            f"CUDA device index {device_index} is unavailable; found {device_count}."
        )

    rounds = FULL_ROUNDS if profile == "full" else QUICK_ROUNDS
    with cp.cuda.Device(device_index):
        _warm_runtime(cp, gpu_function)
        admission = _run_admission(cp, cpu_function, gpu_function)
        metadata = _run_metadata(cpu_function)
        lifecycle = _run_lifecycle(cp, cpu_function, gpu_function)
        fallback = _run_fallback(cpu_function)
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

    if source_snapshot != _source_provenance():
        raise EvidenceError("Tracked source changed while evidence was collected.")
    if contract_snapshot != _operation_contract():
        raise EvidenceError(
            "The operation contract changed during evidence collection."
        )

    performance = {
        "status": "pass",
        "rounds": rounds,
        "case_count": len(performance_results),
        "results": performance_results,
        "all_memory_estimates_cover_observed": all(
            item["memory"]["estimate_covers_observed"]
            for item in performance_results
        ),
    }
    if not performance["all_memory_estimates_cover_observed"]:
        raise EvidenceError("The declared memory estimate did not cover observation.")

    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "scientific-admission-and-machine-local-performance-evidence",
        "portable_performance_claim": False,
        "durable_optimizer_record": False,
        "profile": profile,
        "implementation": {
            "operation_id": OPERATION_ID,
            "implementation_id": IMPLEMENTATION_ID,
            "implementation_version": "1",
            "runtime_id": "cuda-cupy",
            "library_id": "cupyx",
        },
        "environment": _environment_record(cp, device_index),
        "packages": _package_record(cp, np),
        "source_provenance": source_snapshot,
        "operation_contract": contract_snapshot,
        "method": _method_record(profile, rounds),
        "facets": {facet: "pass" for facet in REQUIRED_FACETS},
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
            "u8-empty-1d",
            "u8-empty",
            ("dtype:uint8", "rank:1", "layout:contiguous", "values:empty"),
        ),
        AdmissionCase(
            "u8-boundaries-1d",
            "u8-boundaries",
            (
                "dtype:uint8",
                "rank:1",
                "layout:contiguous",
                "values:boundaries",
            ),
        ),
        AdmissionCase(
            "u8-pattern-2d",
            "u8-pattern-2d",
            ("dtype:uint8", "rank:2", "layout:contiguous"),
        ),
        AdmissionCase(
            "u16-strided-2d",
            "u16-strided-2d",
            (
                "dtype:uint16",
                "rank:2",
                "layout:noncontiguous",
                "values:boundaries",
                "values:uint16-maximum",
            ),
        ),
        AdmissionCase(
            "u16-transposed-3d",
            "u16-transposed-3d",
            (
                "dtype:uint16",
                "rank:3",
                "layout:noncontiguous",
                "values:uint16-maximum",
            ),
        ),
    )


def _host_case(kind: str):
    np = _numpy()
    if kind == "u8-empty":
        return np.empty((0,), dtype=np.uint8)
    if kind == "u8-boundaries":
        return np.asarray([0, 1, 2, 127, 128, 254, 255], dtype=np.uint8)
    if kind == "u8-pattern-2d":
        return ((np.arange(31 * 37, dtype=np.uint16) * 29) % 256).astype(
            np.uint8
        ).reshape(31, 37)
    if kind == "u16-strided-2d":
        base = np.arange(14 * 22, dtype=np.uint16).reshape(14, 22)
        base[-2, -2] = np.iinfo(np.uint16).max
        return base[::-2, 1::2]
    if kind == "u16-transposed-3d":
        base = np.arange(5 * 7 * 11, dtype=np.uint16).reshape(5, 7, 11)
        base[-1, -1, -1] = np.iinfo(np.uint16).max
        return base.transpose(1, 0, 2)
    raise ValueError(f"Unknown admission case kind {kind!r}.")


def _device_case(cp, host, kind: str):
    """Create a genuinely non-contiguous resident view where required."""

    if kind == "u16-strided-2d":
        base_host = _numpy().ascontiguousarray(host[::-1, ::-1])
        # Reconstruct a larger resident base and slice it to the authored values.
        source = _numpy().arange(14 * 22, dtype=_numpy().uint16).reshape(14, 22)
        source[-2, -2] = _numpy().iinfo(_numpy().uint16).max
        device_base = cp.asarray(source)
        device = device_base[::-2, 1::2]
        del base_host
        return device, device_base
    if kind == "u16-transposed-3d":
        source = _numpy().arange(5 * 7 * 11, dtype=_numpy().uint16).reshape(5, 7, 11)
        source[-1, -1, -1] = _numpy().iinfo(_numpy().uint16).max
        device_base = cp.asarray(source)
        return device_base.transpose(1, 0, 2), device_base
    device = cp.asarray(host)
    return device, None


def _run_admission(cp, cpu_function, gpu_function) -> dict[str, object]:
    cases: list[dict[str, object]] = []
    coverage: set[str] = {"repeat:deterministic"}
    for definition in _admission_cases():
        host = _host_case(definition.kind)
        expected = cpu_function(host, output_dtype="float32", scaling="preserve")
        host_before = _array_bytes(host)
        pool = cp.cuda.MemoryPool()
        with cp.cuda.using_allocator(pool.malloc):
            device, keepalive = _device_case(cp, host, definition.kind)
            if "layout:noncontiguous" in definition.coverage and bool(
                device.flags.c_contiguous
            ):
                raise EvidenceError(
                    f"{definition.case_id} did not construct a strided CUDA input."
                )
            before = device.copy()
            output_hashes: list[str] = []
            pointer_distinct = True
            for _ in range(ADMISSION_REPEATS):
                output = gpu_function(
                    device,
                    output_dtype="float32",
                    scaling="preserve",
                )
                cp.cuda.get_current_stream().synchronize()
                if not isinstance(output, cp.ndarray) or output.dtype != cp.float32:
                    raise EvidenceError(
                        f"{definition.case_id} did not return resident float32 data."
                    )
                if output.size:
                    pointer_distinct &= int(output.data.ptr) != int(device.data.ptr)
                actual = cp.asnumpy(output)
                if not _numpy().array_equal(actual, expected):
                    raise EvidenceError(f"Parity failed for {definition.case_id}.")
                output_hashes.append(_array_sha256(actual))
                del actual, output
            input_immutable = bool(cp.array_equal(device, before).item())
            del before, device, keepalive
            cp.cuda.get_current_stream().synchronize()
        cleanup = _drain_pool(cp, pool)
        if not input_immutable or not pointer_distinct:
            raise EvidenceError(f"Input integrity failed for {definition.case_id}.")
        if len(set(output_hashes)) != 1:
            raise EvidenceError(f"Repeat determinism failed for {definition.case_id}.")
        if _array_bytes(host) != host_before:
            raise EvidenceError(f"Host input changed for {definition.case_id}.")
        coverage.update(definition.coverage)
        cases.append(
            {
                "case_id": definition.case_id,
                "shape": list(host.shape),
                "input_dtype": str(host.dtype),
                "output_dtype": str(expected.dtype),
                "coverage": sorted(definition.coverage),
                "cpu_gpu_bitwise_equal": True,
                "gpu_output_resident": True,
                "gpu_output_pointer_distinct": pointer_distinct,
                "input_immutable": input_immutable,
                "repeat_count": ADMISSION_REPEATS,
                "repeat_deterministic": True,
                "input_sha256": _array_sha256(host),
                "cpu_output_sha256": _array_sha256(expected),
                "gpu_output_sha256": output_hashes[0],
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
        "coverage": sorted(coverage),
        "parity_policy": "bitwise-uint8-uint16-to-float32-preserve-v1",
        "cases": cases,
    }


def _run_metadata(cpu_function) -> dict[str, object]:
    np = _numpy()
    from napari_vipp.core.metadata import image_state_from_array, transform_image_state

    source = np.arange(3 * 11 * 13, dtype=np.uint16).reshape(3, 11, 13)
    output = cpu_function(source, output_dtype="float32", scaling="preserve")
    input_state = image_state_from_array(
        source,
        source_name="convert-dtype-admission",
        metadata_source="admission-authored axes",
        history=("Synthetic qualification source",),
    )
    output_state = transform_image_state(
        output,
        input_state,
        operation_id=OPERATION_ID,
        operation_title="Convert Dtype",
        params={"output_dtype": "float32", "scaling": "preserve"},
    )
    if input_state is None or output_state is None:
        raise EvidenceError("Convert Dtype metadata state was not produced.")
    axes_preserved = output_state.axes == input_state.axes
    source_preserved = output_state.source == input_state.source
    acquisition_preserved = output_state.acquisition == input_state.acquisition
    history_appended = (
        len(output_state.history) == len(input_state.history) + 1
        and "float32 via preserve" in output_state.history[-1]
    )
    passed = all(
        (
            output_state.shape == input_state.shape,
            output_state.dtype == "float32",
            axes_preserved,
            source_preserved,
            acquisition_preserved,
            history_appended,
        )
    )
    if not passed:
        raise EvidenceError("Convert Dtype metadata preservation failed.")
    return {
        "status": "pass",
        "shape_preserved": True,
        "dtype_updated_to_float32": True,
        "axes_preserved": axes_preserved,
        "source_preserved": source_preserved,
        "acquisition_preserved": acquisition_preserved,
        "history_appended": history_appended,
        "history_entry": output_state.history[-1],
    }


def _run_lifecycle(cp, cpu_function, gpu_function) -> dict[str, object]:
    from napari_vipp.core.progress import OperationCancelled, ProgressContext

    np = _numpy()
    host = np.arange(257 * 263, dtype=np.uint16).reshape(257, 263)
    expected = cpu_function(host, output_dtype="float32", scaling="preserve")
    cancelled = False
    updates: list[dict[str, object]] = []

    def reporter(update) -> None:
        nonlocal cancelled
        updates.append(
            {
                "current": int(update.current),
                "total": int(update.total),
                "message": str(update.message),
            }
        )
        if int(update.current) == 0:
            cancelled = True

    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device = cp.asarray(host)
        before = device.copy()
        progress = ProgressContext(cancelled=lambda: cancelled, reporter=reporter)
        cancellation_observed = False
        try:
            gpu_function(
                device,
                output_dtype="float32",
                scaling="preserve",
                progress=progress,
            )
        except OperationCancelled as exc:
            cancellation_observed = True
            exc.__traceback__ = None
        if not cancellation_observed:
            raise EvidenceError("Pre-operation cancellation was not observed.")
        cancelled = False
        output = gpu_function(
            device,
            output_dtype="float32",
            scaling="preserve",
        )
        cp.cuda.get_current_stream().synchronize()
        reuse_exact = bool(_numpy().array_equal(cp.asnumpy(output), expected))
        input_immutable = bool(cp.array_equal(device, before).item())
        del output, device, before
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    expected_updates = [{"current": 0, "total": 1, "message": "Converting dtype"}]
    if updates != expected_updates or not reuse_exact or not input_immutable:
        raise EvidenceError("Cancellation cleanup/reuse contract failed.")
    return {
        "status": "pass",
        "cancellation_requested": True,
        "cancellation_observed": cancellation_observed,
        "boundary": "monolithic-pre-operation-boundary-v1",
        "reported_progress": updates,
        "post_cancellation_reuse_exact": reuse_exact,
        "input_immutable": input_immutable,
        **cleanup,
    }


def _run_fallback(cpu_function) -> dict[str, object]:
    np = _numpy()
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import evaluate_candidate_workload_support

    spec = _gpu_spec()
    definitions = (
        ("uint16-to-uint8-preserve", "uint16", "uint8", "preserve"),
        ("uint16-to-float32-clip", "uint16", "float32", "clip"),
        ("float32-to-float32-preserve", "float32", "float32", "preserve"),
    )
    cases: list[dict[str, object]] = []
    for case_id, input_dtype, output_dtype, scaling in definitions:
        workload = WorkloadDescriptor(
            case_id,
            OPERATION_ID,
            ((2, 3),),
            (input_dtype,),
            parameters=(("output_dtype", output_dtype), ("scaling", scaling)),
            resolved_spatial_ndim=2,
        )
        decision = evaluate_candidate_workload_support(spec, workload)
        if decision.supported or not decision.fallback_allowed:
            raise EvidenceError(f"Safe CPU fallback was not selected for {case_id}.")
        source = np.asarray([[0, 1, 2], [3, 4, 5]], dtype=input_dtype)
        output = cpu_function(
            source,
            output_dtype=output_dtype,
            scaling=scaling,
        )
        cases.append(
            {
                "case_id": case_id,
                "gpu_supported": False,
                "fallback_allowed": True,
                "reason": decision.reason_text,
                "cpu_output_dtype": str(output.dtype),
                "cpu_output_sha256": _array_sha256(output),
            }
        )

    invalid = WorkloadDescriptor(
        "invalid-output",
        OPERATION_ID,
        ((2, 3),),
        ("uint16",),
        parameters=(("output_dtype", "int8"), ("scaling", "preserve")),
        resolved_spatial_ndim=2,
    )
    invalid_decision = evaluate_candidate_workload_support(spec, invalid)
    if invalid_decision.supported or invalid_decision.fallback_allowed:
        raise EvidenceError("Invalid authored conversion incorrectly allowed fallback.")
    return {
        "status": "pass",
        "safe_cpu_fallback_case_count": len(cases),
        "cases": cases,
        "invalid_authored_conversion_fails_closed": True,
        "invalid_reason": invalid_decision.reason_text,
    }


def _performance_cases(profile: str) -> tuple[PerformanceCase, ...]:
    quick = (
        PerformanceCase("u8-plane-512", (512, 512), "uint8"),
        PerformanceCase("u16-stack-8x512", (8, 512, 512), "uint16"),
    )
    if profile == "quick":
        return quick
    return quick + (
        PerformanceCase("u8-plane-2048", (2048, 2048), "uint8"),
        PerformanceCase("u16-stack-32x512", (32, 512, 512), "uint16"),
    )


def _run_performance_case(cp, cpu_function, gpu_function, definition, rounds):
    np = _numpy()
    dtype = np.dtype(definition.dtype)
    maximum = int(np.iinfo(dtype).max)
    elements = math.prod(definition.shape)
    host = (np.arange(elements, dtype=np.uint64) * 257 % (maximum + 1)).astype(
        dtype
    ).reshape(definition.shape)

    cpu_seconds = []
    for _ in range(rounds):
        started = time.perf_counter()
        expected = cpu_function(host, output_dtype="float32", scaling="preserve")
        cpu_seconds.append(time.perf_counter() - started)

    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device = cp.asarray(host)
        warm = gpu_function(device, output_dtype="float32", scaling="preserve")
        cp.cuda.get_current_stream().synchronize()
        del warm
        resident_seconds = []
        for _ in range(rounds):
            started = time.perf_counter()
            output = gpu_function(
                device,
                output_dtype="float32",
                scaling="preserve",
            )
            cp.cuda.get_current_stream().synchronize()
            resident_seconds.append(time.perf_counter() - started)
            del output
        transfer_seconds = []
        final_host = None
        for _ in range(rounds):
            started = time.perf_counter()
            transfer_input = cp.asarray(host)
            transfer_output = gpu_function(
                transfer_input,
                output_dtype="float32",
                scaling="preserve",
            )
            final_host = cp.asnumpy(transfer_output)
            cp.cuda.get_current_stream().synchronize()
            transfer_seconds.append(time.perf_counter() - started)
            del transfer_output, transfer_input
        del device
        cp.cuda.get_current_stream().synchronize()
    timing_cleanup = _drain_pool(cp, pool)
    if final_host is None or not np.array_equal(final_host, expected):
        raise EvidenceError(f"Performance parity failed for {definition.case_id}.")

    memory = _measure_memory(cp, gpu_function, host, definition)
    return {
        "case_id": definition.case_id,
        "shape": list(definition.shape),
        "input_dtype": definition.dtype,
        "element_count": elements,
        "input_bytes": int(host.nbytes),
        "output_bytes": int(expected.nbytes),
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
    from napari_vipp.core.compute import WorkloadDescriptor
    from napari_vipp.core.compute_policy import estimate_candidate_memory

    workload = WorkloadDescriptor(
        definition.case_id,
        OPERATION_ID,
        (definition.shape,),
        (definition.dtype,),
        parameters=(("output_dtype", "float32"), ("scaling", "preserve")),
        resolved_spatial_ndim=min(len(definition.shape), 3),
    )
    estimate = estimate_candidate_memory(_gpu_spec(), workload)
    pool = cp.cuda.MemoryPool()
    with cp.cuda.using_allocator(pool.malloc):
        device = cp.asarray(host)
        used_with_input = int(pool.used_bytes())
        output = gpu_function(
            device,
            output_dtype="float32",
            scaling="preserve",
        )
        cp.cuda.get_current_stream().synchronize()
        used_with_output = int(pool.used_bytes())
        observed_reserved = int(pool.total_bytes())
        del output, device
        cp.cuda.get_current_stream().synchronize()
    cleanup = _drain_pool(cp, pool)
    estimated_with_uncertainty = (
        estimate.total_device_peak_bytes + estimate.uncertainty_bytes
    )
    covered = estimated_with_uncertainty >= observed_reserved
    return {
        "scope": "isolated-cupy-memory-pool-reserved-high-water-v1",
        "model_id": estimate.model_id,
        "runtime_managed_peak_bytes": estimate.runtime_managed_peak_bytes,
        "uncertainty_bytes": estimate.uncertainty_bytes,
        "estimated_peak_with_uncertainty_bytes": estimated_with_uncertainty,
        "observed_reserved_bytes": observed_reserved,
        "observed_used_bytes_with_input": used_with_input,
        "observed_used_bytes_with_output": used_with_output,
        "estimate_covers_observed": covered,
        **cleanup,
    }


def _timing_summary(cpu, resident, transfer) -> dict[str, float]:
    return {
        "cpu_median_seconds": statistics.median(cpu),
        "gpu_resident_median_seconds": statistics.median(resident),
        "gpu_transfer_inclusive_median_seconds": statistics.median(transfer),
    }


def _warm_runtime(cp, gpu_function) -> None:
    source = cp.arange(16, dtype=cp.uint16)
    output = gpu_function(source, output_dtype="float32", scaling="preserve")
    cp.cuda.get_current_stream().synchronize()
    del output, source
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
        "public_region": "uint8-uint16-to-float32-preserve-only-v1",
        "parity": "bitwise-array-v1",
        "gpu_resident_timing_scope": "synchronized-resident-compute-v1",
        "gpu_transfer_inclusive_timing_scope": (
            "host-to-device-plus-compute-plus-device-to-host-synchronized-v1"
        ),
        "memory_observation_scope": (
            "isolated-cupy-memory-pool-reserved-high-water-v1"
        ),
        "cancellation": "monolithic-pre-operation-boundary-and-reuse-v1",
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
    for distribution in ("cupy-cuda13x", "napari-vipp"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = "not-installed-as-distribution"
    return packages


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
                ["git", "status", "--porcelain", "--untracked-files=no"],
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


def _operation_contract() -> dict[str, object]:
    spec = _gpu_spec()
    snapshot = _canonical_value(spec)
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


def _gpu_spec():
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
    return matches[0]


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
    from napari_vipp.core.operations import convert_dtype as cpu_function

    module = importlib.import_module("napari_vipp.core.gpu.cupy_convert_dtype")
    return cpu_function, module.convert_dtype


def _numpy():
    return importlib.import_module("numpy")


def _cupy():
    try:
        return importlib.import_module("cupy")
    except (ImportError, OSError) as exc:
        raise EvidenceError(f"CuPy is unavailable: {exc}") from exc


def _array_bytes(array) -> bytes:
    return _numpy().ascontiguousarray(array).tobytes(order="C")


def _array_sha256(array) -> str:
    contiguous = _numpy().ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def validate_existing(path: Path | str) -> Path:
    artifact = Path(path).resolve(strict=True)
    raw = artifact.read_text(encoding="utf-8")
    document = json.loads(raw)
    if raw != _canonical_json(document):
        raise EvidenceError("Evidence JSON is not canonical sorted, indented JSON.")
    _validate_document(document, require_current_sources=True)
    return artifact


def _validate_document(document: object, *, require_current_sources: bool) -> None:
    if not isinstance(document, Mapping) or set(document) != _ROOT_KEYS:
        raise EvidenceError("Evidence root differs from the canonical schema.")
    if (
        document.get("schema") != SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
        or document.get("profile") not in {"quick", "full"}
        or document.get("portable_performance_claim") is not False
        or document.get("durable_optimizer_record") is not False
    ):
        raise EvidenceError("Evidence identity/profile claims are invalid.")
    implementation = _mapping(document.get("implementation"), "implementation")
    if (
        implementation.get("operation_id") != OPERATION_ID
        or implementation.get("implementation_id") != IMPLEMENTATION_ID
        or implementation.get("implementation_version") != "1"
    ):
        raise EvidenceError("Implementation identity is stale.")
    facets = _mapping(document.get("facets"), "facets")
    if facets != {facet: "pass" for facet in REQUIRED_FACETS}:
        raise EvidenceError("Required facet status is incomplete.")
    if require_current_sources:
        if document.get("source_provenance") != _source_provenance():
            raise EvidenceError("Source provenance fingerprints are stale.")
        if document.get("operation_contract") != _operation_contract():
            raise EvidenceError("Operation contract fingerprint is stale.")

    admission = _mapping(document.get("admission"), "admission")
    cases = admission.get("cases")
    if not isinstance(cases, list) or [
        case.get("case_id") for case in cases if isinstance(case, Mapping)
    ] != [case.case_id for case in _admission_cases()]:
        raise EvidenceError("Admission case manifest is incomplete.")
    if (
        admission.get("status") != "pass"
        or admission.get("case_count") != len(cases)
        or admission.get("repeat_count") != ADMISSION_REPEATS
        or not REQUIRED_ADMISSION_COVERAGE
        <= set(admission.get("coverage", ()))
    ):
        raise EvidenceError("Admission aggregate is invalid.")
    for case in cases:
        item = _mapping(case, "admission case")
        cleanup = _mapping(item.get("cleanup"), "admission cleanup")
        if (
            item.get("cpu_gpu_bitwise_equal") is not True
            or item.get("gpu_output_resident") is not True
            or item.get("gpu_output_pointer_distinct") is not True
            or item.get("input_immutable") is not True
            or item.get("repeat_deterministic") is not True
            or item.get("repeat_count") != ADMISSION_REPEATS
            or item.get("cpu_output_sha256") != item.get("gpu_output_sha256")
            or not _cleanup_passed(cleanup)
        ):
            raise EvidenceError("Admission parity/integrity evidence is invalid.")

    metadata = _mapping(document.get("metadata"), "metadata")
    if metadata.get("status") != "pass" or not all(
        metadata.get(key) is True
        for key in (
            "shape_preserved",
            "dtype_updated_to_float32",
            "axes_preserved",
            "source_preserved",
            "acquisition_preserved",
            "history_appended",
        )
    ):
        raise EvidenceError("Metadata evidence is invalid.")

    lifecycle = _mapping(document.get("lifecycle"), "lifecycle")
    if (
        lifecycle.get("status") != "pass"
        or lifecycle.get("cancellation_requested") is not True
        or lifecycle.get("cancellation_observed") is not True
        or lifecycle.get("post_cancellation_reuse_exact") is not True
        or lifecycle.get("input_immutable") is not True
        or not _cleanup_passed(lifecycle)
    ):
        raise EvidenceError("Cancellation/cleanup evidence is invalid.")

    fallback = _mapping(document.get("fallback"), "fallback")
    fallback_cases = fallback.get("cases")
    if (
        fallback.get("status") != "pass"
        or fallback.get("invalid_authored_conversion_fails_closed") is not True
        or not isinstance(fallback_cases, list)
        or fallback.get("safe_cpu_fallback_case_count") != len(fallback_cases)
        or not fallback_cases
        or not all(
            case.get("gpu_supported") is False
            and case.get("fallback_allowed") is True
            and bool(case.get("cpu_output_sha256"))
            for case in fallback_cases
            if isinstance(case, Mapping)
        )
    ):
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
        or [item.get("case_id") for item in results if isinstance(item, Mapping)]
        != [item.case_id for item in _performance_cases(profile)]
        or performance.get("all_memory_estimates_cover_observed") is not True
    ):
        raise EvidenceError("Performance aggregate is invalid.")
    for result in results:
        item = _mapping(result, "performance case")
        parity = _mapping(item.get("parity"), "performance parity")
        samples = _mapping(item.get("samples"), "performance samples")
        memory = _mapping(item.get("memory"), "performance memory")
        if (
            parity.get("passed") is not True
            or parity.get("cpu_output_sha256") != parity.get("gpu_output_sha256")
            or memory.get("model_id") != "cupy-convert-dtype-memory-v1"
            or memory.get("estimate_covers_observed") is not True
            or not _cleanup_passed(memory)
            or not _cleanup_passed(_mapping(item.get("cleanup"), "timing cleanup"))
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
        expected_summary = _timing_summary(
            samples["cpu_seconds"],
            samples["gpu_resident_seconds"],
            samples["gpu_transfer_inclusive_seconds"],
        )
        if item.get("summary") != expected_summary:
            raise EvidenceError("Timing summary is inconsistent with samples.")

    if document.get("method") != _method_record(profile, rounds):
        raise EvidenceError("Method record is inconsistent with the profile.")


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


if __name__ == "__main__":
    raise SystemExit(main())
