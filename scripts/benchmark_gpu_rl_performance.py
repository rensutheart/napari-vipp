"""Benchmark production CPU and CuPy Richardson--Lucy on large 3D stacks.

This developer-only command measures VIPP's registered production benchmark
path.  GPU end-to-end samples include both input transfers, synchronized
resident compute, output transfer, and private-scope cleanup.  The same samples
also retain transfer-only and resident-compute timing.  Every candidate must
first pass the production scientific parity gate for the exact workload.

Importing this module, asking for help, or validating an existing artifact does
not import CuPy, initialize CUDA, or open an ND2 file.
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
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = "napari-vipp-cupy-rl-performance-evidence"
SCHEMA_VERSION = 1
GENERATOR_ID = "numpy-pcg64-large-rl-v1"
OPERATION_ID = "richardson_lucy_deconvolution"
IMPLEMENTATION_ID = "rl-cupy-f32-v1"
RUNTIME_ID = "cuda-cupy"
BENCHMARK_ROUNDS = 3
BOOTSTRAP_SAMPLES = 0
BOOTSTRAP_SEED = 0
ROUND_ORDER_SEED = 20_260_729
FILTER_EPSILON = 1e-8
ITERATIONS = 25
MIB = 1024**2
GIB = 1024**3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "docs/benchmarks/rl-cupy-performance-windows-rtx5090.json"
)
SOURCE_PROVENANCE_PATHS = (
    "scripts/benchmark_gpu_rl_performance.py",
    "src/napari_vipp/core/compute.py",
    "src/napari_vipp/core/accelerator_lease.py",
    "src/napari_vipp/core/compute_benchmark.py",
    "src/napari_vipp/core/compute_benchmark_adapter.py",
    "src/napari_vipp/core/compute_planning.py",
    "src/napari_vipp/core/compute_policy.py",
    "src/napari_vipp/core/compute_registry.py",
    "src/napari_vipp/core/compute_specs.py",
    "src/napari_vipp/core/node_execution.py",
    "src/napari_vipp/core/operations.py",
    "src/napari_vipp/core/progress.py",
    "src/napari_vipp/core/gpu/cupy_rl.py",
    "src/napari_vipp/core/gpu/cupy_runtime.py",
)


class PerformanceBenchmarkError(RuntimeError):
    """A complete, parity-qualified performance artifact was not produced."""


@dataclass(frozen=True, slots=True)
class _CaseDefinition:
    case_id: str
    label: str
    shape: tuple[int, int, int]
    psf_shape: tuple[int, int, int]
    psf_sigma: tuple[float, float, float]
    seed: int | None
    source_kind: str
    source_metadata: Mapping[str, object]
    private_source: bool = False


SYNTHETIC_CASES = (
    _CaseDefinition(
        case_id="synthetic-shape-stress-medium-3d",
        label="Medium 3D shape stress (synthetic)",
        shape=(64, 512, 512),
        psf_shape=(15, 31, 31),
        psf_sigma=(2.5, 4.0, 4.0),
        seed=20_260_731,
        source_kind="deterministic-synthetic",
        source_metadata={"generator": GENERATOR_ID},
    ),
    _CaseDefinition(
        case_id="synthetic-shape-stress-large-3d",
        label="Large 3D shape stress (synthetic)",
        shape=(64, 1024, 1024),
        psf_shape=(15, 31, 31),
        psf_sigma=(2.5, 4.0, 4.0),
        seed=20_260_733,
        source_kind="deterministic-synthetic",
        source_metadata={"generator": GENERATOR_ID},
    ),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=("medium", "large"),
        default="large",
        help="medium runs 64x512x512; large runs medium plus 64x1024x1024.",
    )
    parser.add_argument(
        "--nd2",
        type=Path,
        help=(
            "Optional private ND2 source. One selected T,C ZYX volume is added; "
            "the path, filename, content hash, and pixels are never serialized."
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
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON evidence path (default: {DEFAULT_OUTPUT}).",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        help="Readable summary path (default: output path with .md suffix).",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help="Exact CUDA device ID (default: the probed selected device).",
    )
    parser.add_argument(
        "--validate-existing",
        type=Path,
        help="Validate an existing JSON artifact without importing CuPy.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.validate_existing is not None:
        try:
            artifact = validate_existing(args.validate_existing)
        except (OSError, TypeError, ValueError, PerformanceBenchmarkError) as exc:
            print(f"RL performance evidence validation failed: {exc}", file=sys.stderr)
            return 2
        print(f"RL performance evidence is current: {artifact}")
        return 0

    markdown = args.markdown or args.output.with_suffix(".md")
    try:
        document = run_benchmarks(
            profile=args.profile,
            nd2_path=args.nd2,
            nd2_time_index=args.nd2_time_index,
            nd2_channel_index=args.nd2_channel_index,
            device_id=args.device_id,
        )
        _atomic_write_artifacts(args.output, markdown, document)
    except (OSError, TypeError, ValueError, PerformanceBenchmarkError) as exc:
        print(f"RL performance benchmark failed: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # optional provider failures need a concise CLI edge
        print(
            f"RL performance benchmark failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 2
    print(f"Wrote RL performance evidence to {args.output.resolve()}")
    print(f"Wrote readable summary to {markdown.resolve()}")
    return 0


def run_benchmarks(
    *,
    profile: str,
    nd2_path: Path | None,
    nd2_time_index: int,
    nd2_channel_index: int,
    device_id: str = "",
) -> dict[str, object]:
    """Run fixed large-stack cases through the registered production adapter."""

    import numpy as np

    from napari_vipp.core.compute import (
        ComputeMode,
        ComputeRequest,
        WorkloadDescriptor,
    )
    from napari_vipp.core.compute_benchmark import NodeBenchmarkService
    from napari_vipp.core.compute_benchmark_adapter import (
        build_registered_node_benchmark,
    )
    from napari_vipp.core.compute_planning import probe_compute_environment
    from napari_vipp.core.compute_policy import (
        ArrayFacts,
        FactCompleteness,
        estimate_candidate_memory,
        evaluate_candidate_support,
        evaluate_memory_support,
    )
    from napari_vipp.core.compute_registry import ComputeRegistry
    from napari_vipp.core.node_execution import PreparedNodeCall
    from napari_vipp.core.operations import richardson_lucy_deconvolution

    selected_profile = str(profile).strip().lower()
    if selected_profile not in {"medium", "large"}:
        raise ValueError("profile must be 'medium' or 'large'.")

    source_provenance = _source_provenance()
    request = ComputeRequest(
        mode=ComputeMode.SELECTIVE,
        runtime_id=RUNTIME_ID,
        device_id=str(device_id).strip(),
        allow_experimental=True,
    )
    registry = ComputeRegistry()
    try:
        matches = tuple(
            spec
            for spec in registry.implementations_for_operation(
                OPERATION_ID,
                allow_experimental=True,
            )
            if spec.implementation_id == IMPLEMENTATION_ID
        )
        if len(matches) != 1:
            raise PerformanceBenchmarkError(
                f"Expected exactly one {IMPLEMENTATION_ID!r} provider, got "
                f"{len(matches)}."
            )
        spec = matches[0]
        environment, warnings = probe_compute_environment(
            registry,
            request,
            (spec,),
        )
        _require_environment(environment, spec)
        resolved_device_id = str(environment.device_id)
        total_device_bytes = int(environment.total_accelerator_memory_bytes)
        memory_limit_bytes = max(1, int(total_device_bytes * 0.8))
        safety_reserve_bytes = max(512 * MIB, total_device_bytes // 10)

        definitions = list(
            SYNTHETIC_CASES[:1] if selected_profile == "medium" else SYNTHETIC_CASES
        )
        loaded_nd2: tuple[_CaseDefinition, Any] | None = None
        if nd2_path is not None:
            loaded_nd2 = _load_private_nd2_volume(
                nd2_path,
                time_index=nd2_time_index,
                channel_index=nd2_channel_index,
            )
            definitions.insert(0, loaded_nd2[0])

        results: list[dict[str, object]] = []
        for case_index, definition in enumerate(definitions, start=1):
            print(
                f"[{case_index}/{len(definitions)}] Preparing "
                f"{definition.label} {definition.shape}.",
                flush=True,
            )
            if loaded_nd2 is not None and definition is loaded_nd2[0]:
                image = np.ascontiguousarray(loaded_nd2[1], dtype=np.float32)
                loaded_nd2 = None
            else:
                image = _synthetic_image(definition)
            psf = _gaussian_psf(
                definition.psf_shape,
                definition.psf_sigma,
            )
            parameters = _parameters()
            preflight_workload = WorkloadDescriptor(
                node_id=f"rl-performance-{definition.case_id}",
                operation_id=OPERATION_ID,
                input_shapes=(
                    tuple(int(size) for size in image.shape),
                    tuple(int(size) for size in psf.shape),
                ),
                input_dtypes=(image.dtype.name, psf.dtype.name),
                parameters=tuple(
                    (name, value)
                    for name, value in parameters.items()
                    if name != "progress"
                ),
                resolved_spatial_ndim=3,
            )
            memory_estimate = estimate_candidate_memory(spec, preflight_workload)
            memory_support = evaluate_memory_support(
                memory_estimate,
                memory_cap_bytes=memory_limit_bytes,
                total_device_bytes=total_device_bytes,
                safety_reserve_bytes=safety_reserve_bytes,
            )
            if not memory_support.supported:
                raise PerformanceBenchmarkError(
                    f"{definition.case_id} failed VRAM preflight: "
                    f"{memory_support.reason_text}"
                )
            preflight = _resource_preflight(
                image=image,
                psf=psf,
                admitted_device_bytes=(
                    memory_estimate.total_device_peak_bytes
                    + memory_estimate.uncertainty_bytes
                ),
                safety_reserve_bytes=safety_reserve_bytes,
                device_id=resolved_device_id,
            )
            call = PreparedNodeCall(
                node_id=f"rl-performance-{definition.case_id}",
                operation_id=OPERATION_ID,
                cpu_function=richardson_lucy_deconvolution,
                inputs=(image, psf),
                kwargs=parameters,
                multiple_inputs=True,
                output_port_count=1,
            )
            built = build_registered_node_benchmark(
                call,
                admitted_specs=(spec,),
                registry=registry,
                environment_fingerprint=environment.fingerprint,
                device_id=resolved_device_id,
                memory_limit_bytes=memory_limit_bytes,
                safety_reserve_bytes=safety_reserve_bytes,
                warm_rounds=BENCHMARK_ROUNDS,
                max_warm_rounds=BENCHMARK_ROUNDS,
                allow_experimental=True,
                paired_bootstrap_samples=BOOTSTRAP_SAMPLES,
                paired_bootstrap_seed=BOOTSTRAP_SEED,
                paired_confidence_level=0.95,
            )
            if memory_estimate != estimate_candidate_memory(
                spec,
                built.request.workload,
            ):
                raise PerformanceBenchmarkError(
                    "The coarse preflight workload and detached benchmark "
                    "workload produced different memory estimates."
                )
            facts = tuple(
                _complete_array_facts(
                    value,
                    revision_fingerprint=built.request.workload.facts_fingerprint,
                    array_facts_type=ArrayFacts,
                    completeness=FactCompleteness.COMPLETE,
                )
                for value in (image, psf)
            )
            support = evaluate_candidate_support(
                spec,
                built.request.workload,
                environment,
                allow_experimental=True,
                array_facts=facts,
            )
            if not support.supported:
                raise PerformanceBenchmarkError(
                    f"{definition.case_id} was not scientifically admitted: "
                    f"{support.reason.value}: {support.reason_text}"
                )
            service = NodeBenchmarkService(
                rng=random.Random(ROUND_ORDER_SEED + case_index),
            )
            progress = _ProgressPrinter(definition.case_id)
            record = service.benchmark(
                built.request,
                progress=progress,
            )
            serialized = _serialize_case(
                definition,
                image,
                psf,
                spec,
                built,
                record,
                memory_estimate=memory_estimate,
                preflight=preflight,
            )
            results.append(serialized)
            summary = serialized["summary"]
            print(
                f"[{case_index}/{len(definitions)}] Completed "
                f"{definition.case_id}: CPU {summary['cpu_median_seconds']:.3f}s, "
                f"GPU {summary['gpu_end_to_end_median_seconds']:.3f}s, "
                f"{summary['paired_speedup_median']:.2f}x.",
                flush=True,
            )
            del image, psf, call, built, facts, record

        document: dict[str, object] = {
            "schema": SCHEMA,
            "schema_version": SCHEMA_VERSION,
            "created_utc": datetime.now(UTC).isoformat(),
            "kind": "machine-local-production-path-screening-evidence",
            "portable_performance_claim": False,
            "profile": selected_profile,
            "method": {
                "operation_id": OPERATION_ID,
                "implementation_id": IMPLEMENTATION_ID,
                "iterations": ITERATIONS,
                "filter_epsilon": FILTER_EPSILON,
                "warmup_rounds": 1,
                "paired_warm_rounds": BENCHMARK_ROUNDS,
                "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
                "sampling_profile": "short-descriptive-3-paired-v1",
                "durable_optimizer_record": False,
                "gpu_timing_scope": "synchronized-end-to-end-v1",
                "disk_io_included": False,
                "input_generation_included": False,
                "exact_workload_parity_required_before_timing": True,
            },
            "platform": _platform_provenance(environment),
            "packages": _package_provenance(),
            "environment": environment.as_dict(),
            "probe_warnings": list(warnings),
            "source_provenance": source_provenance,
            "results": results,
        }
        _require_source_snapshot_unchanged(source_provenance)
        return document
    finally:
        registry.close()


class _ProgressPrinter:
    def __init__(self, case_id: str) -> None:
        self.case_id = case_id
        self._last: tuple[object, ...] | None = None

    def __call__(self, update: Any) -> None:
        operation_total = int(getattr(update, "operation_total", 0))
        operation_completed = int(getattr(update, "operation_completed", 0))
        if operation_total:
            stride = max(1, operation_total // 5)
            if operation_completed not in {0, operation_total} and (
                operation_completed % stride
            ):
                return
        key = (
            str(update.phase),
            update.implementation_id,
            int(update.completed),
            int(update.total),
            operation_completed,
            operation_total,
        )
        if key == self._last:
            return
        self._last = key
        suffix = (
            f"; operation {operation_completed}/{operation_total}"
            if operation_total
            else ""
        )
        print(
            f"  {update.implementation_id}: {update.phase.value} "
            f"{update.completed}/{update.total}{suffix}",
            flush=True,
        )


def _parameters() -> dict[str, object]:
    return {
        "spatial_mode": "3D ZYX",
        "resolved_spatial_ndim": 3,
        "iterations": ITERATIONS,
        "normalize_psf": True,
        "clip_negative_input": True,
        "clip_output_negative": True,
        "preserve_input_scale": True,
        "filter_epsilon": FILTER_EPSILON,
        "progress": None,
    }


def _synthetic_image(definition: _CaseDefinition):
    import numpy as np

    if definition.seed is None:
        raise ValueError("A synthetic case requires a deterministic seed.")
    rng = np.random.default_rng(definition.seed)
    image = rng.random(definition.shape, dtype=np.float32)
    # A low positive background plus sparse brighter samples better resembles
    # normalized microscopy intensity than a perfectly uniform field while the
    # FFT workload remains content-independent.
    image *= np.float32(0.08)
    flat = image.reshape(-1)
    bead_count = max(1, flat.size // 65_536)
    bead_indices = rng.integers(0, flat.size, size=bead_count)
    flat[bead_indices] += rng.random(bead_count, dtype=np.float32) * np.float32(0.92)
    return np.ascontiguousarray(image, dtype=np.float32)


def _gaussian_psf(
    shape: tuple[int, int, int],
    sigma: tuple[float, float, float],
):
    import numpy as np

    if any(size <= 0 or size % 2 == 0 for size in shape):
        raise ValueError("Performance PSF extents must be positive and odd.")
    coordinates = np.indices(shape, dtype=np.float32)
    exponent = np.zeros(shape, dtype=np.float32)
    for axis, (size, width) in enumerate(zip(shape, sigma, strict=True)):
        if not math.isfinite(width) or width <= 0:
            raise ValueError("Performance PSF sigmas must be finite and positive.")
        centered = (coordinates[axis] - np.float32((size - 1) / 2)) / np.float32(width)
        exponent += centered * centered
    psf = np.exp(np.float32(-0.5) * exponent).astype(np.float32, copy=False)
    psf /= np.float32(psf.sum(dtype=np.float64))
    return np.ascontiguousarray(psf, dtype=np.float32)


def _load_private_nd2_volume(
    path: Path,
    *,
    time_index: int,
    channel_index: int,
) -> tuple[_CaseDefinition, object]:
    import numpy as np

    try:
        import nd2
    except ImportError as exc:
        raise PerformanceBenchmarkError(
            "--nd2 requires the optional nd2 package in the benchmark environment."
        ) from exc

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"ND2 source does not exist: {resolved}")
    with nd2.ND2File(str(resolved)) as nd_file:
        shape = tuple(int(size) for size in nd_file.shape)
        sizes = getattr(nd_file, "sizes", {})
        axes = "".join(str(name).upper() for name in sizes)
        dtype = np.dtype(nd_file.dtype).name
    if len(axes) != len(shape) or not all(axis in axes for axis in "ZYX"):
        raise PerformanceBenchmarkError(
            f"ND2 must expose ordered ZYX axes; got axes={axes!r}, shape={shape}."
        )
    selection: list[int | slice] = []
    selected_axes: list[str] = []
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
            selected_axes.append(axis)
        elif size == 1:
            selection.append(0)
        else:
            raise PerformanceBenchmarkError(
                f"Unsupported non-spatial ND2 axis {axis!r} with size {size}."
            )
    if "".join(selected_axes) != "ZYX":
        raise PerformanceBenchmarkError(
            f"Selected ND2 spatial order must be ZYX, got {''.join(selected_axes)!r}."
        )
    lazy = nd2.imread(str(resolved), dask=True)
    selected = lazy[tuple(selection)]
    if hasattr(selected, "compute"):
        selected = selected.compute()
    volume = np.ascontiguousarray(np.asarray(selected), dtype=np.float32)
    if volume.ndim != 3:
        raise PerformanceBenchmarkError(
            f"Selected ND2 volume must be 3D ZYX, got {volume.shape}."
        )
    psf_shape = tuple(
        _bounded_odd_extent(requested, image_extent)
        for requested, image_extent in zip((9, 31, 31), volume.shape, strict=True)
    )
    if any(extent < 3 for extent in psf_shape):
        raise PerformanceBenchmarkError(
            "Selected ND2 volume is too small for a 3D timing PSF."
        )
    definition = _CaseDefinition(
        case_id="private-real-nd2-volume-3d",
        label="Private real-acquisition single-channel ZYX volume",
        shape=tuple(int(size) for size in volume.shape),
        psf_shape=psf_shape,
        psf_sigma=tuple(
            max(1.0, min(default_sigma, extent / 4))
            for default_sigma, extent in zip((2.5, 4.0, 4.0), psf_shape, strict=True)
        ),
        seed=None,
        source_kind="private-nd2-volume",
        source_metadata={
            "original_axes": axes,
            "original_shape": list(shape),
            "original_dtype": dtype,
            "selected_indices": selected_indices,
            "direct_identifiers_omitted": True,
        },
        private_source=True,
    )
    return definition, volume


def _bounded_odd_extent(requested: int, image_extent: int) -> int:
    extent = min(int(requested), int(image_extent))
    return extent if extent % 2 else extent - 1


def _validated_index(value: int, size: int, axis: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{axis} index must be an integer.")
    if value < 0 or value >= size:
        raise IndexError(f"{axis} index {value} is outside 0..{size - 1}.")
    return value


def _complete_array_facts(
    value: Any,
    *,
    revision_fingerprint: str,
    array_facts_type: Any,
    completeness: Any,
) -> Any:
    import numpy as np

    array = np.asarray(value)
    finite = np.isfinite(array)
    finite_values = array[finite]
    guarantees = []
    if not bool(np.any((array == 0) & np.signbit(array))):
        guarantees.append("no-negative-zero")
    if finite_values.size and bool(np.min(finite_values) >= 0):
        guarantees.append("nonnegative")
    return array_facts_type(
        shape=tuple(int(size) for size in array.shape),
        dtype=array.dtype.name,
        element_count=int(array.size),
        revision_fingerprint=revision_fingerprint,
        completeness=completeness,
        finite_count=int(np.count_nonzero(finite)),
        minimum=float(np.min(finite_values)) if finite_values.size else None,
        maximum=float(np.max(finite_values)) if finite_values.size else None,
        strides=tuple(int(stride) for stride in array.strides),
        contiguous=bool(array.flags.c_contiguous),
        guarantees=tuple(guarantees),
    )


def _resource_preflight(
    *,
    image: Any,
    psf: Any,
    admitted_device_bytes: int,
    safety_reserve_bytes: int,
    device_id: str,
) -> dict[str, int]:
    import cupy

    input_bytes = int(image.nbytes) + int(psf.nbytes)
    host_available = _available_host_memory_bytes()
    # The registered parity gate temporarily owns CPU/GPU outputs, float64
    # views, and a float64 difference.  The FFT memory model is a conservative
    # additional host-pressure proxy for SciPy's convolution workspace.
    required_host = max(
        4 * GIB,
        int(admitted_device_bytes) + 5 * input_bytes,
    )
    if host_available and required_host > host_available:
        raise PerformanceBenchmarkError(
            "Host-memory preflight refused the workload: requires about "
            f"{required_host / GIB:.2f} GiB, only "
            f"{host_available / GIB:.2f} GiB is currently available."
        )
    device_index = _cuda_device_index(device_id)
    with cupy.cuda.Device(device_index):
        free_device, total_device = cupy.cuda.runtime.memGetInfo()
    usable_device = max(0, int(free_device) - int(safety_reserve_bytes))
    if admitted_device_bytes > usable_device:
        raise PerformanceBenchmarkError(
            "VRAM preflight refused the workload: requires about "
            f"{admitted_device_bytes / GIB:.2f} GiB after the model, only "
            f"{usable_device / GIB:.2f} GiB is currently usable after reserve."
        )
    return {
        "input_bytes": input_bytes,
        "host_available_bytes": host_available,
        "required_host_bytes": required_host,
        "device_free_bytes": int(free_device),
        "device_total_bytes": int(total_device),
        "device_usable_after_reserve_bytes": usable_device,
        "admitted_device_bytes": int(admitted_device_bytes),
        "device_index": device_index,
    }


def _cuda_device_index(device_id: str) -> int:
    runtime, separator, raw_index = str(device_id).strip().partition(":")
    if (
        runtime != "cuda"
        or not separator
        or not raw_index.isascii()
        or not (raw_index.isdecimal())
    ):
        raise PerformanceBenchmarkError(
            f"Expected a canonical CUDA device ID, got {device_id!r}."
        )
    return int(raw_index)


def _available_host_memory_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        if sys.platform != "win32":
            return 0
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = (
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            )

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return 0
        return int(status.ullAvailPhys)


def _require_environment(environment: Any, spec: Any) -> None:
    if environment.probe_status != "available":
        raise PerformanceBenchmarkError(
            environment.probe_reason or "CUDA environment probe was unavailable."
        )
    if RUNTIME_ID not in environment.runtime_ids:
        raise PerformanceBenchmarkError(f"Required runtime {RUNTIME_ID!r} is absent.")
    if spec.implementation_library_id not in environment.implementation_libraries:
        raise PerformanceBenchmarkError(
            f"Required library {spec.implementation_library_id!r} is absent."
        )
    if not str(environment.device_id).startswith("cuda:"):
        raise PerformanceBenchmarkError("The selected environment is not CUDA.")


def _serialize_case(
    definition: _CaseDefinition,
    image: Any,
    psf: Any,
    spec: Any,
    built: Any,
    record: Any,
    *,
    memory_estimate: Any,
    preflight: Mapping[str, int],
) -> dict[str, object]:
    cpu = next(
        item for item in record.candidates if item.implementation_id.startswith("cpu-")
    )
    gpu = next(
        item
        for item in record.candidates
        if item.implementation_id == spec.implementation_id
    )
    if not gpu.parity_passed or gpu.error:
        raise PerformanceBenchmarkError(
            f"{definition.case_id} GPU parity failed: "
            f"{gpu.error or 'production parity gate rejected the output'}"
        )
    if len(cpu.warm_seconds) != BENCHMARK_ROUNDS:
        raise PerformanceBenchmarkError("CPU warm timing rounds are incomplete.")
    if len(gpu.warm_seconds) != BENCHMARK_ROUNDS:
        raise PerformanceBenchmarkError("GPU warm timing rounds are incomplete.")
    if len(gpu.warm_transfer_seconds) != BENCHMARK_ROUNDS:
        raise PerformanceBenchmarkError("GPU transfer timing rounds are incomplete.")
    if len(gpu.warm_resident_seconds) != BENCHMARK_ROUNDS:
        raise PerformanceBenchmarkError("GPU resident timing rounds are incomplete.")
    runs = built.observations.runs(spec.implementation_id)
    expected_runs = 1 + int(built.request.warmup_rounds) + BENCHMARK_ROUNDS
    terminal_zero = all(
        run.terminal_snapshot.runtime_live_bytes == 0
        and run.terminal_snapshot.runtime_reserved_bytes == 0
        for run in runs
    )
    cleanup_succeeded = all(run.cleanup_succeeded for run in runs)
    if (
        len(runs) != expected_runs
        or not cleanup_succeeded
        or not terminal_zero
        or not gpu.synchronized
        or not gpu.transfers_included
        or gpu.timing_scope != "synchronized-end-to-end-v1"
    ):
        raise PerformanceBenchmarkError(
            f"{definition.case_id} did not produce complete synchronized cleanup "
            "evidence."
        )
    cpu_median = statistics.median(cpu.warm_seconds)
    gpu_median = statistics.median(gpu.warm_seconds)
    paired_speedups = tuple(
        cpu_seconds / gpu_seconds
        for cpu_seconds, gpu_seconds in zip(
            cpu.warm_seconds,
            gpu.warm_seconds,
            strict=True,
        )
    )
    material_saving = cpu_median - gpu_median > max(0.010, 0.05 * cpu_median)
    summary = {
        "screening_choice": "GPU-CuPy" if material_saving else "CPU",
        "cpu_median_seconds": cpu_median,
        "gpu_end_to_end_median_seconds": gpu_median,
        "gpu_resident_median_seconds": statistics.median(gpu.warm_resident_seconds),
        "gpu_transfer_median_seconds": statistics.median(gpu.warm_transfer_seconds),
        "paired_speedup_median": statistics.median(paired_speedups),
        "paired_speedups": list(paired_speedups),
        "absolute_median_saving_seconds": cpu_median - gpu_median,
        "peak_observed_device_bytes": gpu.peak_memory_bytes,
    }
    input_sha256 = None
    workload_fingerprint = None
    benchmark_record_digest = None
    if not definition.private_source:
        input_sha256 = hashlib.sha256(memoryview(image).cast("B")).hexdigest()
        workload_fingerprint = built.request.workload.fingerprint
        benchmark_record_digest = record.key.digest
    return {
        "case_id": definition.case_id,
        "label": definition.label,
        "source_kind": definition.source_kind,
        "source_metadata": dict(definition.source_metadata),
        "direct_private_identifiers_published": False,
        "shape": list(definition.shape),
        "voxel_count": math.prod(definition.shape),
        "dtype": str(image.dtype),
        "image_bytes": int(image.nbytes),
        "psf_shape": list(definition.psf_shape),
        "psf_sigma_voxels": list(definition.psf_sigma),
        "psf_sha256": hashlib.sha256(memoryview(psf).cast("B")).hexdigest(),
        "input_sha256": input_sha256,
        "parameters": {
            name: value for name, value in _parameters().items() if name != "progress"
        },
        "workload_fingerprint": workload_fingerprint,
        "benchmark_record_digest": benchmark_record_digest,
        "benchmark_policy_id": record.benchmark_policy_id,
        "environment_fingerprint": built.request.environment_fingerprint,
        "service_screening_winner": record.accepted_implementation_id,
        "memory_estimate": asdict(memory_estimate),
        "preflight": dict(preflight),
        "summary": summary,
        "candidates": [asdict(item) for item in record.candidates],
        "gpu_cleanup": {
            "invocation_count": len(runs),
            "all_cleanup_succeeded": cleanup_succeeded,
            "all_runtime_pool_terminal_zero": terminal_zero,
            "terminal_snapshots": [run.terminal_snapshot.as_dict() for run in runs],
        },
    }


def _platform_provenance(environment: Any) -> dict[str, object]:
    uname = platform.uname()
    return {
        "system": uname.system,
        "release": uname.release,
        "version": uname.version,
        "machine": uname.machine,
        "processor": uname.processor,
        "python": platform.python_version(),
        "device_id": environment.device_id,
        "device_name": environment.device_name,
        "driver_version": environment.driver_version,
        "total_accelerator_memory_bytes": (environment.total_accelerator_memory_bytes),
    }


def _package_provenance() -> dict[str, str]:
    names = ("napari-vipp", "numpy", "scipy", "scikit-image", "cupy-cuda13x")
    versions: dict[str, str] = {}
    for name in names:
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
            raise PerformanceBenchmarkError(
                f"Required source provenance file is missing: {relative}"
            )
        result.append({"path": relative, "sha256": _file_sha256(path)})
    return result


def _require_source_snapshot_unchanged(
    provenance: Sequence[Mapping[str, str]],
) -> None:
    for item in provenance:
        relative = str(item["path"])
        expected = str(item["sha256"])
        if _file_sha256(PROJECT_ROOT / relative) != expected:
            raise PerformanceBenchmarkError(
                f"Source changed while benchmarking: {relative}"
            )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_markdown(document: Mapping[str, object]) -> str:
    _validate_document_contract(document)
    platform_info = _mapping(document["platform"], "platform")
    method = _mapping(document["method"], "method")
    lines = [
        "# CuPy Richardson-Lucy large-stack performance",
        "",
        f"- Generated: `{document['created_utc']}`",
        f"- Device: `{platform_info['device_name']}`",
        f"- Host processor: `{platform_info['processor']}`",
        f"- Iterations: `{method['iterations']}`",
        f"- Filter epsilon: `{float(method['filter_epsilon']):.0e}`",
        f"- Paired warm rounds: `{method['paired_warm_rounds']}`",
        "",
        "This is machine-local production-path evidence, not a portable speed",
        "claim or a reusable optimizer record. It is a short three-pair",
        "descriptive screen on deliberately expensive workloads. Disk I/O and",
        "input generation are excluded. GPU end-to-end time",
        "includes Image/PSF H2D transfer, synchronized resident compute, output",
        "D2H transfer, and private allocator cleanup. Scientific parity passed",
        "before the observed screening winner was reported.",
        "",
        "| Workload | Voxels | CPU median | GPU end-to-end | GPU resident | "
        "Transfer | Speedup | Screen winner |",
        "|---|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for raw in document["results"]:
        case = _mapping(raw, "result")
        summary = _mapping(case["summary"], "result summary")
        lines.append(
            "| "
            + " | ".join(
                (
                    str(case["label"]),
                    f"{int(case['voxel_count']):,}",
                    _seconds(summary["cpu_median_seconds"]),
                    _seconds(summary["gpu_end_to_end_median_seconds"]),
                    _seconds(summary["gpu_resident_median_seconds"]),
                    _seconds(summary["gpu_transfer_median_seconds"]),
                    _ratio(summary["paired_speedup_median"]),
                    str(summary["screening_choice"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "- The CPU and GPU columns compare the same authored operation and",
            "  parameters. Resident time is shown only to explain pipeline-context",
            "  gains; the screen winner uses transfer-inclusive GPU time. Three",
            "  paired rounds are enough for a descriptive large-stack result but",
            "  do not replace VIPP's longer durable node-benchmark evidence.",
            "- Synthetic cases are shape-and-memory stress tests, not claims that",
            "  independent random voxels reproduce confocal image statistics. The",
            "  optional private ND2 volume supplies the real-acquisition anchor.",
            "- `filter_epsilon=1e-8` is the currently admitted measured point, not",
            "  an assertion that it is inherently the only useful GPU epsilon.",
            "- Large-stack results do not broaden the scientific region. New",
            "  epsilon, iteration, PSF, dtype, or safety-option regions require a",
            "  versioned numerical study across adversarial fixtures.",
            "- The optional private ND2 case publishes workload metadata and timings",
            "  but no path, filename, content digest, or pixels. Its generated",
            "  Gaussian timing PSF is not a measured restoration-quality PSF.",
            "",
        )
    )
    return "\n".join(lines)


def _seconds(value: object) -> str:
    return f"{float(value):.3f} s"


def _ratio(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.2f}x"


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise PerformanceBenchmarkError(f"{name} must be an object.")
    return value


def _validate_document_shape(document: Mapping[str, object]) -> None:
    if document.get("schema") != SCHEMA:
        raise PerformanceBenchmarkError("Unexpected evidence schema.")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise PerformanceBenchmarkError("Unexpected evidence schema version.")
    results = document.get("results")
    if not isinstance(results, list) or not results:
        raise PerformanceBenchmarkError("Evidence must contain at least one result.")
    _mapping(document.get("platform"), "platform")
    _mapping(document.get("method"), "method")


def _validate_document_contract(document: Mapping[str, object]) -> None:
    _validate_document_shape(document)
    if document.get("kind") != "machine-local-production-path-screening-evidence":
        raise PerformanceBenchmarkError("Unexpected performance evidence kind.")
    if document.get("portable_performance_claim") is not False:
        raise PerformanceBenchmarkError("Evidence must reject portable speed claims.")
    profile = document.get("profile")
    if profile not in {"medium", "large"}:
        raise PerformanceBenchmarkError("Evidence profile must be medium or large.")
    try:
        datetime.fromisoformat(str(document["created_utc"]))
    except (KeyError, ValueError) as exc:
        raise PerformanceBenchmarkError("Evidence created_utc is invalid.") from exc
    method = _mapping(document["method"], "method")
    required_method = {
        "operation_id": OPERATION_ID,
        "implementation_id": IMPLEMENTATION_ID,
        "iterations": ITERATIONS,
        "filter_epsilon": FILTER_EPSILON,
        "warmup_rounds": 1,
        "paired_warm_rounds": BENCHMARK_ROUNDS,
        "paired_bootstrap_samples": BOOTSTRAP_SAMPLES,
        "sampling_profile": "short-descriptive-3-paired-v1",
        "durable_optimizer_record": False,
        "gpu_timing_scope": "synchronized-end-to-end-v1",
        "disk_io_included": False,
        "input_generation_included": False,
        "exact_workload_parity_required_before_timing": True,
    }
    for name, expected in required_method.items():
        if method.get(name) != expected:
            raise PerformanceBenchmarkError(
                f"Evidence method field {name!r} does not match the fixed contract."
            )

    results = document["results"]
    identifiers = {str(_mapping(item, "result").get("case_id", "")) for item in results}
    required_ids = {"synthetic-shape-stress-medium-3d"}
    if profile == "large":
        required_ids.add("synthetic-shape-stress-large-3d")
    optional_private = "private-real-nd2-volume-3d"
    if not required_ids.issubset(identifiers):
        raise PerformanceBenchmarkError("Evidence is missing a required profile case.")
    if identifiers - required_ids - {optional_private}:
        raise PerformanceBenchmarkError("Evidence contains an unexpected case ID.")
    if len(results) != len(identifiers):
        raise PerformanceBenchmarkError("Evidence case IDs must be unique.")
    for raw in results:
        _validate_case_contract(_mapping(raw, "result"))


def _validate_case_contract(case: Mapping[str, object]) -> None:
    shape = _positive_shape(case.get("shape"), "image shape")
    psf_shape = _positive_shape(case.get("psf_shape"), "PSF shape")
    if len(shape) != 3 or len(psf_shape) != 3:
        raise PerformanceBenchmarkError("Performance cases must be true-3D.")
    if any(extent % 2 == 0 for extent in psf_shape):
        raise PerformanceBenchmarkError("Performance PSF extents must be odd.")
    if any(psf > image for psf, image in zip(psf_shape, shape, strict=True)):
        raise PerformanceBenchmarkError("Performance PSF must fit the image.")
    if case.get("dtype") != "float32":
        raise PerformanceBenchmarkError("Performance input dtype must be float32.")
    if case.get("voxel_count") != math.prod(shape):
        raise PerformanceBenchmarkError("Performance voxel_count is inconsistent.")
    if case.get("image_bytes") != math.prod(shape) * 4:
        raise PerformanceBenchmarkError("Performance image_bytes is inconsistent.")
    parameters = _mapping(case.get("parameters"), "case parameters")
    expected_parameters = {
        name: value for name, value in _parameters().items() if name != "progress"
    }
    if dict(parameters) != expected_parameters:
        raise PerformanceBenchmarkError("Case parameters left the admitted region.")
    if case.get("direct_private_identifiers_published") is not False:
        raise PerformanceBenchmarkError("Direct private identifiers must be omitted.")
    _hex_digest(case.get("psf_sha256"), "PSF SHA-256")

    private = case.get("source_kind") == "private-nd2-volume"
    if private:
        for name in ("input_sha256", "workload_fingerprint", "benchmark_record_digest"):
            if case.get(name) is not None:
                raise PerformanceBenchmarkError(
                    f"Private evidence field {name!r} must be null."
                )
        metadata = _mapping(case.get("source_metadata"), "private source metadata")
        allowed = {
            "original_axes",
            "original_shape",
            "original_dtype",
            "selected_indices",
            "direct_identifiers_omitted",
        }
        if (
            set(metadata) != allowed
            or metadata.get("direct_identifiers_omitted") is not True
        ):
            raise PerformanceBenchmarkError(
                "Private source metadata violated its exact redaction contract."
            )
        serialized_metadata = json.dumps(metadata, sort_keys=True).lower()
        if ".nd2" in serialized_metadata or ":\\" in serialized_metadata:
            raise PerformanceBenchmarkError("Private source metadata contains a path.")
    else:
        if case.get("source_kind") != "deterministic-synthetic":
            raise PerformanceBenchmarkError("Unexpected non-private source kind.")
        _hex_digest(case.get("input_sha256"), "input SHA-256")
        for name in ("workload_fingerprint", "benchmark_record_digest"):
            if not str(case.get(name, "")).strip():
                raise PerformanceBenchmarkError(f"Evidence field {name!r} is empty.")

    summary = _mapping(case.get("summary"), "case summary")
    for name in (
        "cpu_median_seconds",
        "gpu_end_to_end_median_seconds",
        "gpu_resident_median_seconds",
        "gpu_transfer_median_seconds",
        "paired_speedup_median",
    ):
        _positive_finite(summary.get(name), name)
    if summary.get("screening_choice") not in {"CPU", "GPU-CuPy"}:
        raise PerformanceBenchmarkError("Screening choice is invalid.")
    paired_speedups = summary.get("paired_speedups")
    if (
        not isinstance(paired_speedups, list)
        or len(paired_speedups) != BENCHMARK_ROUNDS
    ):
        raise PerformanceBenchmarkError("Paired speedup samples are incomplete.")
    for value in paired_speedups:
        _positive_finite(value, "paired speedup")

    candidates = case.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise PerformanceBenchmarkError("Evidence requires CPU and GPU candidates.")
    cpu = next(
        (
            item
            for item in candidates
            if str(_mapping(item, "candidate").get("implementation_id", "")).startswith(
                "cpu-"
            )
        ),
        None,
    )
    gpu = next(
        (
            item
            for item in candidates
            if _mapping(item, "candidate").get("implementation_id") == IMPLEMENTATION_ID
        ),
        None,
    )
    if cpu is None or gpu is None:
        raise PerformanceBenchmarkError("CPU/GPU candidate identities are incomplete.")
    cpu_record = _mapping(cpu, "CPU candidate")
    gpu_record = _mapping(gpu, "GPU candidate")
    _validate_candidate_samples(cpu_record, gpu=False)
    _validate_candidate_samples(gpu_record, gpu=True)
    cpu_times = tuple(float(value) for value in cpu_record["warm_seconds"])
    gpu_times = tuple(float(value) for value in gpu_record["warm_seconds"])
    expected_speedups = tuple(
        cpu_seconds / gpu_seconds
        for cpu_seconds, gpu_seconds in zip(cpu_times, gpu_times, strict=True)
    )
    checks = {
        "cpu_median_seconds": statistics.median(cpu_times),
        "gpu_end_to_end_median_seconds": statistics.median(gpu_times),
        "gpu_resident_median_seconds": statistics.median(
            tuple(float(value) for value in gpu_record["warm_resident_seconds"])
        ),
        "gpu_transfer_median_seconds": statistics.median(
            tuple(float(value) for value in gpu_record["warm_transfer_seconds"])
        ),
        "paired_speedup_median": statistics.median(expected_speedups),
    }
    for name, expected in checks.items():
        if not math.isclose(
            float(summary[name]), expected, rel_tol=1e-12, abs_tol=1e-12
        ):
            raise PerformanceBenchmarkError(f"Summary field {name!r} is inconsistent.")

    cleanup = _mapping(case.get("gpu_cleanup"), "GPU cleanup")
    if cleanup.get("invocation_count") != BENCHMARK_ROUNDS + 2:
        raise PerformanceBenchmarkError("GPU cleanup invocation count is incomplete.")
    if cleanup.get("all_cleanup_succeeded") is not True:
        raise PerformanceBenchmarkError("GPU cleanup did not succeed.")
    if cleanup.get("all_runtime_pool_terminal_zero") is not True:
        raise PerformanceBenchmarkError("GPU runtime pool did not return to zero.")
    snapshots = cleanup.get("terminal_snapshots")
    if not isinstance(snapshots, list) or len(snapshots) != BENCHMARK_ROUNDS + 2:
        raise PerformanceBenchmarkError("GPU terminal snapshots are incomplete.")
    for raw_snapshot in snapshots:
        snapshot = _mapping(raw_snapshot, "terminal snapshot")
        if (
            snapshot.get("runtime_live_bytes") != 0
            or snapshot.get("runtime_reserved_bytes") != 0
        ):
            raise PerformanceBenchmarkError("GPU terminal memory is not zero.")

    memory = _mapping(case.get("memory_estimate"), "memory estimate")
    if memory.get("model_id") != "cupyx-richardson-lucy-fft-memory-v2":
        raise PerformanceBenchmarkError("Unexpected RL memory model.")
    preflight = _mapping(case.get("preflight"), "resource preflight")
    admitted = int(memory.get("total_device_peak_bytes", -1)) + int(
        memory.get("uncertainty_bytes", -1)
    )
    if preflight.get("admitted_device_bytes") != admitted:
        raise PerformanceBenchmarkError("Preflight and memory model disagree.")


def _validate_candidate_samples(candidate: Mapping[str, object], *, gpu: bool) -> None:
    if candidate.get("parity_passed") is not True or candidate.get("error"):
        raise PerformanceBenchmarkError("A benchmark candidate failed parity.")
    warm = candidate.get("warm_seconds")
    if not isinstance(warm, (list, tuple)) or len(warm) != BENCHMARK_ROUNDS:
        raise PerformanceBenchmarkError("Candidate warm samples are incomplete.")
    for value in warm:
        _positive_finite(value, "candidate warm sample")
    if not gpu:
        return
    if candidate.get("timing_scope") != "synchronized-end-to-end-v1":
        raise PerformanceBenchmarkError("GPU timing scope is invalid.")
    if candidate.get("synchronized") is not True:
        raise PerformanceBenchmarkError("GPU samples were not synchronized.")
    if candidate.get("transfers_included") is not True:
        raise PerformanceBenchmarkError("GPU samples omitted transfers.")
    for name in ("warm_transfer_seconds", "warm_resident_seconds"):
        values = candidate.get(name)
        if not isinstance(values, (list, tuple)) or len(values) != BENCHMARK_ROUNDS:
            raise PerformanceBenchmarkError(f"GPU field {name!r} is incomplete.")
        for value in values:
            _positive_finite(value, name)


def _positive_shape(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise PerformanceBenchmarkError(f"{name} must be a nonempty list.")
    shape = tuple(value)
    if any(
        isinstance(size, bool) or not isinstance(size, int) or size <= 0
        for size in shape
    ):
        raise PerformanceBenchmarkError(f"{name} must contain positive integers.")
    return shape


def _positive_finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise PerformanceBenchmarkError(f"{name} must be finite and positive.")
    return float(value)


def _hex_digest(value: object, name: str) -> str:
    text = str(value or "")
    if len(text) != 64 or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise PerformanceBenchmarkError(f"{name} must be a lowercase SHA-256 digest.")
    return text


def validate_existing(path: Path) -> Path:
    artifact = path.expanduser().resolve()
    with artifact.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise PerformanceBenchmarkError("Evidence root must be an object.")
    _validate_document_contract(document)
    provenance = document.get("source_provenance")
    if not isinstance(provenance, list):
        raise PerformanceBenchmarkError("Evidence source provenance is missing.")
    expected = {item["path"]: item["sha256"] for item in _source_provenance()}
    actual: dict[str, str] = {}
    for raw in provenance:
        item = _mapping(raw, "source provenance entry")
        actual[str(item.get("path", ""))] = str(item.get("sha256", ""))
    if actual != expected:
        raise PerformanceBenchmarkError(
            "Evidence source fingerprints do not match the current checkout."
        )
    markdown = artifact.with_suffix(".md")
    if markdown.is_file():
        expected_markdown = render_markdown(document)
        actual_markdown = markdown.read_text(encoding="utf-8")
        if actual_markdown != expected_markdown:
            raise PerformanceBenchmarkError(
                "Readable Markdown does not match the JSON evidence renderer."
            )
    return artifact


def _atomic_write_artifacts(
    output: Path,
    markdown: Path,
    document: Mapping[str, object],
) -> None:
    _validate_document_contract(document)
    json_text = json.dumps(document, indent=2, sort_keys=True) + "\n"
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
