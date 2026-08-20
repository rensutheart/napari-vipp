#!/usr/bin/env python
"""Calibrate VIPP thumbnail-statistics CPU/CuPy crossover points.

Every synthetic dtype/size case runs in a fresh Python process.  Its first
``Prefer GPU`` calculation therefore includes lazy CuPy import, CUDA probing,
context setup, upload, the first dtype-specific kernel, download, and cleanup.
The following calls use the same production ``ThumbnailStatisticsEngine`` and
are reported separately as warm calls.  CPU and GPU limits must agree exactly.

The default matrix brackets the conservative uint8/uint16 production
thresholds.  ``--dtypes float32`` adds the qualified floating-point path
without making its extra bounded radix passes part of every routine
calibration.  ``--nd2`` optionally adds one real, channel-selected stack.  The
private path, filename, pixels, hashes, and calculated limits are never written
to the JSON artifact or human summary.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "napari-vipp-thumbnail-statistics-calibration"
SCHEMA_VERSION = 1
WORKER_MARKER = "VIPP_THUMBNAIL_BENCHMARK_RESULT="
DEFAULT_SIZES_MIB = (2, 4, 8, 32, 128, 256, 384, 512)
DEFAULT_DTYPES = ("uint8", "uint16")
SUPPORTED_DTYPES = (*DEFAULT_DTYPES, "float32")
DEFAULT_CPU_ROUNDS = 3
DEFAULT_WARM_GPU_ROUNDS = 3
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = Path(__file__).resolve()
SOURCE_PATHS = (
    Path("scripts/benchmark_thumbnail_statistics.py"),
    Path("src/napari_vipp/core/thumbnail_statistics.py"),
    Path("src/napari_vipp/core/gpu/cupy_thumbnail_statistics.py"),
)


class BenchmarkError(RuntimeError):
    """Raised when calibration evidence would be incomplete or misleading."""


@dataclass(frozen=True, slots=True)
class _WorkerSpec:
    source_kind: str
    dtype: str = ""
    size_mib: int = 0
    nd2_path: Path | None = None
    nd2_channel_index: int = 0
    nd2_time_index: int | None = None

    @property
    def case_id(self) -> str:
        if self.source_kind == "synthetic":
            return f"synthetic-{self.dtype}-{self.size_mib:04d}mib"
        return "private-nd2-channel-stack"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sizes-mib",
        default=",".join(str(value) for value in DEFAULT_SIZES_MIB),
        help="Comma-separated synthetic input sizes in MiB.",
    )
    parser.add_argument(
        "--dtypes",
        default=",".join(DEFAULT_DTYPES),
        help="Comma-separated synthetic dtypes: uint8,uint16,float32.",
    )
    parser.add_argument(
        "--cpu-rounds",
        type=int,
        default=DEFAULT_CPU_ROUNDS,
        help="Production CPU calculations per case (default: 3).",
    )
    parser.add_argument(
        "--warm-gpu-rounds",
        type=int,
        default=DEFAULT_WARM_GPU_ROUNDS,
        help="GPU calculations after the first cold call (default: 3).",
    )
    parser.add_argument(
        "--device-id",
        default="",
        help="Optional production device id, for example cuda:0.",
    )
    parser.add_argument(
        "--nd2",
        type=Path,
        help="Optional private ND2 input. Its path and pixels are never saved.",
    )
    parser.add_argument(
        "--nd2-channel-index",
        type=int,
        default=0,
        help="Channel selected from --nd2 (default: 0).",
    )
    parser.add_argument(
        "--nd2-time-index",
        type=int,
        help="Optionally select one T index; by default the full T stack is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Write strict, privacy-safe JSON to this file.",
    )
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Return a nonzero status if any case cannot execute with CuPy.",
    )
    parser.add_argument(
        "--worker-timeout-seconds",
        type=float,
        default=300.0,
        help="Fresh-process timeout per case (default: 300 seconds).",
    )

    # Private worker protocol.  Keeping it in this file guarantees that every
    # case executes the exact same source snapshot as the orchestrator.
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-source", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-dtype", help=argparse.SUPPRESS)
    parser.add_argument("--_worker-size-mib", type=int, help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args._worker:
        try:
            result = _run_worker_from_arguments(args)
        except Exception as exc:
            print(
                f"Thumbnail benchmark worker failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 3
        print(WORKER_MARKER + json.dumps(result, allow_nan=False, sort_keys=True))
        return 0

    try:
        sizes_mib = _parse_sizes_mib(args.sizes_mib)
        dtypes = _parse_dtypes(args.dtypes)
        cpu_rounds = _positive_integer(args.cpu_rounds, "--cpu-rounds")
        warm_gpu_rounds = _positive_integer(
            args.warm_gpu_rounds,
            "--warm-gpu-rounds",
        )
        timeout_seconds = float(args.worker_timeout_seconds)
        if not np.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("--worker-timeout-seconds must be finite and positive.")
        document = run_benchmark(
            sizes_mib=sizes_mib,
            dtypes=dtypes,
            cpu_rounds=cpu_rounds,
            warm_gpu_rounds=warm_gpu_rounds,
            device_id=str(args.device_id).strip(),
            nd2_path=args.nd2,
            nd2_channel_index=args.nd2_channel_index,
            nd2_time_index=args.nd2_time_index,
            worker_timeout_seconds=timeout_seconds,
        )
        _validate_document(document)
        if args.output is not None:
            written = _atomic_write_json(args.output, document)
        else:
            written = None
    except (BenchmarkError, OSError, TypeError, ValueError) as exc:
        detail = _redact_private_text(str(exc), args.nd2)
        print(f"Thumbnail-statistics calibration failed: {detail}", file=sys.stderr)
        return 2

    print(render_human_summary(document))
    if written is not None:
        print(f"\nJSON evidence: {written}")
    gpu_missing = any(
        result["gpu"]["status"] != "available" for result in document["results"]
    )
    return 4 if args.require_gpu and gpu_missing else 0


def run_benchmark(
    *,
    sizes_mib: Sequence[int] = DEFAULT_SIZES_MIB,
    dtypes: Sequence[str] = DEFAULT_DTYPES,
    cpu_rounds: int = DEFAULT_CPU_ROUNDS,
    warm_gpu_rounds: int = DEFAULT_WARM_GPU_ROUNDS,
    device_id: str = "",
    nd2_path: Path | None = None,
    nd2_channel_index: int = 0,
    nd2_time_index: int | None = None,
    worker_timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Collect process-isolated production timings and return one document."""

    specs = [
        _WorkerSpec("synthetic", dtype=dtype, size_mib=int(size_mib))
        for dtype in dtypes
        for size_mib in sizes_mib
    ]
    private_path = Path(nd2_path) if nd2_path is not None else None
    if private_path is not None:
        specs.append(
            _WorkerSpec(
                "private-nd2",
                nd2_path=private_path,
                nd2_channel_index=int(nd2_channel_index),
                nd2_time_index=nd2_time_index,
            )
        )

    results: list[dict[str, object]] = []
    for index, spec in enumerate(specs, start=1):
        label = (
            f"{spec.dtype} {spec.size_mib} MiB"
            if spec.source_kind == "synthetic"
            else "private ND2 channel stack"
        )
        print(
            f"[{index}/{len(specs)}] {label}: fresh-process cold + warm timing...",
            flush=True,
        )
        result = _invoke_worker(
            spec,
            cpu_rounds=cpu_rounds,
            warm_gpu_rounds=warm_gpu_rounds,
            device_id=device_id,
            timeout_seconds=worker_timeout_seconds,
        )
        results.append(result)

    document: dict[str, object] = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(),
        "kind": "machine-local calibration; not a portable performance claim",
        "method": {
            "production_api": (
                "napari_vipp.core.thumbnail_statistics.ThumbnailStatisticsEngine"
            ),
            "contrast_mode": "Percentile",
            "percentile_range": [0.5, 99.9],
            "exact_parity_required": True,
            "case_process_isolation": "one fresh Python process per input",
            "cpu_timing": f"median of {cpu_rounds} complete CPU engine calls",
            "cold_gpu_timing": (
                "first complete Prefer-GPU engine call in a fresh process; includes "
                "lazy runtime import/probe, context, upload, first kernel, download, "
                "and production cleanup"
            ),
            "warm_gpu_timing": (
                f"median of {warm_gpu_rounds} subsequent complete engine calls in "
                "the same process"
            ),
            "break_even_rule": (
                "smallest observed size at which GPU is no slower than CPU for "
                "that size and every larger measured size"
            ),
            "private_limits_and_pixels_omitted": True,
        },
        "matrix": {
            "synthetic_sizes_mib": [int(value) for value in sizes_mib],
            "synthetic_dtypes": list(dtypes),
            "cpu_rounds": cpu_rounds,
            "warm_gpu_rounds": warm_gpu_rounds,
            "private_nd2_included": private_path is not None,
        },
        "platform": _platform_provenance(),
        "source_provenance": _source_provenance(),
        "results": results,
        "observed_crossovers": _observed_crossovers(results),
    }
    _assert_private_source_redacted(document, private_path)
    return document


def _invoke_worker(
    spec: _WorkerSpec,
    *,
    cpu_rounds: int,
    warm_gpu_rounds: int,
    device_id: str,
    timeout_seconds: float,
) -> dict[str, object]:
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        "--_worker",
        "--_worker-source",
        spec.source_kind,
        "--cpu-rounds",
        str(cpu_rounds),
        "--warm-gpu-rounds",
        str(warm_gpu_rounds),
    ]
    if device_id:
        command.extend(("--device-id", device_id))
    if spec.source_kind == "synthetic":
        command.extend(
            (
                "--_worker-dtype",
                spec.dtype,
                "--_worker-size-mib",
                str(spec.size_mib),
            )
        )
    else:
        if spec.nd2_path is None:
            raise BenchmarkError("The private ND2 worker has no input.")
        command.extend(
            (
                "--nd2",
                str(spec.nd2_path),
                "--nd2-channel-index",
                str(spec.nd2_channel_index),
            )
        )
        if spec.nd2_time_index is not None:
            command.extend(("--nd2-time-index", str(spec.nd2_time_index)))

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_seconds,
    )
    payload_line = next(
        (
            line[len(WORKER_MARKER) :]
            for line in reversed(completed.stdout.splitlines())
            if line.startswith(WORKER_MARKER)
        ),
        None,
    )
    if completed.returncode != 0 or payload_line is None:
        detail = completed.stderr.strip() or "worker returned no result"
        detail = _redact_private_text(detail, spec.nd2_path)
        raise BenchmarkError(f"Fresh benchmark worker failed: {detail}")
    try:
        result = json.loads(payload_line)
    except json.JSONDecodeError as exc:
        raise BenchmarkError("Fresh benchmark worker returned invalid JSON.") from exc
    if not isinstance(result, dict) or result.get("case_id") != spec.case_id:
        raise BenchmarkError("Fresh benchmark worker returned the wrong case.")
    _assert_private_source_redacted(result, spec.nd2_path)
    return result


def _run_worker_from_arguments(args: argparse.Namespace) -> dict[str, object]:
    cpu_rounds = _positive_integer(args.cpu_rounds, "--cpu-rounds")
    warm_gpu_rounds = _positive_integer(args.warm_gpu_rounds, "--warm-gpu-rounds")
    if args._worker_source == "synthetic":
        dtype = _parse_dtypes(args._worker_dtype or "")
        if len(dtype) != 1:
            raise ValueError("A synthetic worker requires exactly one dtype.")
        size_mib = _positive_integer(args._worker_size_mib, "--_worker-size-mib")
        data = _synthetic_stack(dtype[0], size_mib)
        spec = _WorkerSpec("synthetic", dtype=dtype[0], size_mib=size_mib)
        source = {
            "source_kind": "deterministic-synthetic-stack",
            "generator_id": "repeating-native-integer-ramp-v1",
        }
    elif args._worker_source == "private-nd2":
        if args.nd2 is None:
            raise ValueError("A private ND2 worker requires --nd2.")
        data, source = _load_private_nd2(
            args.nd2,
            channel_index=args.nd2_channel_index,
            time_index=args.nd2_time_index,
        )
        spec = _WorkerSpec("private-nd2", nd2_path=args.nd2)
    else:
        raise ValueError("Unknown private worker source kind.")
    return _benchmark_array(
        data,
        case_id=spec.case_id,
        source=source,
        cpu_rounds=cpu_rounds,
        warm_gpu_rounds=warm_gpu_rounds,
        device_id=str(args.device_id).strip(),
    )


def _benchmark_array(
    data: np.ndarray,
    *,
    case_id: str,
    source: Mapping[str, object],
    cpu_rounds: int,
    warm_gpu_rounds: int,
    device_id: str,
) -> dict[str, object]:
    Engine, Request, Backend, ComputeMode = _production_api()
    array = np.ascontiguousarray(data)
    cpu_engine = Engine()
    cpu_request = Request(
        data=array,
        contrast_mode="Percentile",
        compute_mode=ComputeMode.CPU,
        device_id=device_id,
    )
    auto_request = Request(
        data=array,
        contrast_mode="Percentile",
        compute_mode=ComputeMode.AUTO,
        device_id=device_id,
    )
    auto_before = cpu_engine.select(auto_request)
    cpu_results = [cpu_engine.calculate(cpu_request) for _ in range(cpu_rounds)]
    reference = cpu_results[0].limits
    if any(result.actual_backend is not Backend.CPU_NUMPY for result in cpu_results):
        raise BenchmarkError("Production CPU timing did not execute on CPU.")
    if any(not _limits_equal(result.limits, reference) for result in cpu_results):
        raise BenchmarkError("Repeated production CPU limits were not deterministic.")
    cpu_samples = [float(result.elapsed_seconds) for result in cpu_results]

    gpu_engine = Engine()
    gpu_request = Request(
        data=array,
        contrast_mode="Percentile",
        compute_mode=ComputeMode.PREFER_GPU,
        device_id=device_id,
    )
    cold = gpu_engine.calculate(gpu_request)
    gpu: dict[str, object]
    parity = False
    if cold.actual_backend is not Backend.GPU_CUPY:
        gpu = {
            "status": "unavailable",
            "reason_code": cold.fallback_reason_code or cold.decision.reason_code,
            "fallback_elapsed_seconds": float(cold.elapsed_seconds),
            "cold_seconds": None,
            "cold_transfer": _transfer_record(cold),
            "warm_samples_seconds": [],
            "warm_transfers": [],
            "warm_median_seconds": None,
            "runtime_id": cold.runtime_id,
            "device_id": cold.device_id,
            "algorithm_id": str(getattr(cold, "algorithm_id", "")),
        }
    else:
        parity = _limits_equal(cold.limits, reference)
        if not parity:
            raise BenchmarkError("Cold GPU limits differ from production CPU limits.")
        warm_results = [
            gpu_engine.calculate(gpu_request) for _ in range(warm_gpu_rounds)
        ]
        if any(
            result.actual_backend is not Backend.GPU_CUPY for result in warm_results
        ):
            raise BenchmarkError("A warm production GPU timing unexpectedly fell back.")
        if any(not _limits_equal(result.limits, reference) for result in warm_results):
            raise BenchmarkError("Warm GPU limits differ from production CPU limits.")
        warm_samples = [float(result.elapsed_seconds) for result in warm_results]
        gpu = {
            "status": "available",
            "reason_code": "",
            "fallback_elapsed_seconds": None,
            "cold_seconds": float(cold.elapsed_seconds),
            "cold_transfer": _transfer_record(cold),
            "warm_samples_seconds": warm_samples,
            "warm_transfers": [_transfer_record(result) for result in warm_results],
            "warm_median_seconds": float(statistics.median(warm_samples)),
            "runtime_id": cold.runtime_id,
            "device_id": cold.device_id,
            "algorithm_id": str(getattr(cold, "algorithm_id", "")),
        }

    auto_after = gpu_engine.select(auto_request)

    cpu_median = float(statistics.median(cpu_samples))
    cold_seconds = gpu["cold_seconds"]
    warm_median = gpu["warm_median_seconds"]
    result: dict[str, object] = {
        "case_id": case_id,
        "source": dict(source),
        "shape": [int(value) for value in array.shape],
        "dtype": array.dtype.name,
        "element_count": int(array.size),
        "input_bytes": int(array.nbytes),
        "cpu": {
            "samples_seconds": cpu_samples,
            "median_seconds": cpu_median,
            "algorithm_id": cpu_results[0].algorithm_id,
        },
        "gpu": gpu,
        "production_auto_policy": {
            "before_gpu_evidence": _decision_record(auto_before),
            "after_this_engine_attempt": _decision_record(auto_after),
        },
        "exact_parity": parity,
        "speedup": {
            "cpu_over_cold_gpu": (
                cpu_median / float(cold_seconds) if cold_seconds else None
            ),
            "cpu_over_warm_gpu": (
                cpu_median / float(warm_median) if warm_median else None
            ),
        },
        # Exact limits are deliberately omitted even for synthetic inputs so
        # private cases cannot accidentally diverge into a less safe schema.
        "contrast_limits_omitted": True,
    }
    return result


def _production_api():
    source_root = PROJECT_ROOT / "src"
    source_text = str(source_root)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    from napari_vipp.core.compute import ComputeMode
    from napari_vipp.core.thumbnail_statistics import (
        ThumbnailStatisticsBackend,
        ThumbnailStatisticsEngine,
        ThumbnailStatisticsRequest,
    )

    return (
        ThumbnailStatisticsEngine,
        ThumbnailStatisticsRequest,
        ThumbnailStatisticsBackend,
        ComputeMode,
    )


def _decision_record(decision: Any) -> dict[str, object]:
    backend = decision.backend
    return {
        "backend": str(getattr(backend, "value", backend)),
        "reason_code": str(decision.reason_code),
        "threshold_bytes": int(decision.threshold_bytes),
        "gpu_warm": bool(decision.gpu_warm),
    }


def _transfer_record(result: Any) -> dict[str, object]:
    """Return privacy-safe logical transfer observations from production."""

    return {
        "input_path": str(getattr(result, "input_path", "")),
        "logical_input_host_to_device_bytes": int(
            getattr(result, "logical_input_host_to_device_bytes", 0)
        ),
        "auxiliary_host_to_device_bytes": int(
            getattr(result, "auxiliary_host_to_device_bytes", 0)
        ),
        "device_to_host_bytes": int(getattr(result, "device_to_host_bytes", 0)),
        "device_to_host_values": int(getattr(result, "device_to_host_values", 0)),
    }


def _synthetic_stack(dtype_name: str, size_mib: int) -> np.ndarray:
    dtype = np.dtype(dtype_name)
    byte_count = int(size_mib) * 1024**2
    element_count = byte_count // dtype.itemsize
    if element_count * dtype.itemsize != byte_count:
        raise ValueError("Synthetic byte size is not divisible by the dtype size.")
    # Every supported native dtype produces one MiB per logical plane while
    # retaining a useful stack shape and the requested exact byte count.
    plane_shapes = {
        np.dtype(np.uint8): (1024, 1024),
        np.dtype(np.uint16): (512, 1024),
        np.dtype(np.float32): (256, 1024),
    }
    try:
        plane_shape = plane_shapes[dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported synthetic dtype: {dtype_name!r}.") from exc
    plane_elements = int(np.prod(plane_shape))
    planes, remainder = divmod(element_count, plane_elements)
    if remainder or planes < 1:
        raise ValueError("Synthetic size cannot be represented as the test stack.")
    values = np.empty(element_count, dtype=dtype)
    chunk_elements = min(element_count, 1024**2)
    if dtype == np.dtype(np.float32):
        # Sparse bit-constructed edge values exercise the exact float policy
        # without relying on host conversions that could erase signed zero or
        # subnormals.  The remainder is a representative finite distribution.
        indices = np.arange(chunk_elements, dtype=np.uint32)
        pattern = (
            (indices % np.uint32(4096)).astype(np.float32) - np.float32(64.0)
        ) / np.float32(7.0)
        special_bits = np.asarray(
            (
                0x00000000,  # +0
                0x80000000,  # -0
                0x00000001,  # smallest positive subnormal
                0x80000001,  # smallest negative subnormal
                0x00800000,  # smallest positive normal
                0x7F800000,  # +inf (excluded)
                0xFF800000,  # -inf (clipped when finite negatives exist)
                0x7FC00001,  # NaN (excluded)
            ),
            dtype=np.uint32,
        ).view(np.float32)
        pattern[: special_bits.size] = special_bits
    else:
        pattern = np.arange(chunk_elements, dtype=dtype)
    for start in range(0, element_count, chunk_elements):
        stop = min(start + chunk_elements, element_count)
        values[start:stop] = pattern[: stop - start]
    result = values.reshape((planes, *plane_shape))
    result.flags.writeable = False
    return result


def _load_private_nd2(
    path: Path,
    *,
    channel_index: int,
    time_index: int | None,
) -> tuple[np.ndarray, dict[str, object]]:
    try:
        import nd2
    except ImportError as exc:
        raise BenchmarkError("--nd2 requires the optional nd2 package.") from exc

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError("The private ND2 source does not exist.")
    with nd2.ND2File(str(resolved)) as nd_file:
        shape = tuple(int(value) for value in nd_file.shape)
        sizes = getattr(nd_file, "sizes", {})
        axes = "".join(str(axis).upper() for axis in sizes)
    if len(axes) != len(shape) or len(set(axes)) != len(axes):
        raise BenchmarkError("The ND2 source has ambiguous axis metadata.")
    if "C" not in axes or not all(axis in axes for axis in "YX"):
        raise BenchmarkError("The ND2 source must expose C, Y, and X axes.")

    selection: list[int | slice] = []
    selected_axes: list[str] = []
    for axis, extent in zip(axes, shape, strict=True):
        if axis == "C":
            selection.append(_validated_index(channel_index, extent, "C"))
        elif axis == "T" and time_index is not None:
            selection.append(_validated_index(time_index, extent, "T"))
        elif axis in {"T", "Z", "Y", "X"}:
            selection.append(slice(None))
            selected_axes.append(axis)
        elif extent == 1:
            selection.append(0)
        else:
            raise BenchmarkError(
                f"Unsupported non-singleton ND2 axis {axis!r}; "
                "select it before benchmarking."
            )

    lazy = nd2.imread(str(resolved), dask=True)
    selected = lazy[tuple(selection)]
    if hasattr(selected, "compute"):
        selected = selected.compute()
    data = np.ascontiguousarray(np.asarray(selected))
    if data.dtype not in {
        np.dtype(np.uint8),
        np.dtype(np.uint16),
        np.dtype(np.float32),
    }:
        raise BenchmarkError(
            "The selected ND2 channel must already be native uint8, uint16, "
            "or float32; the calibration does not synthesize a cast."
        )
    return data, {
        "source_kind": "private-nd2-channel-stack",
        "selected_axes": "".join(selected_axes),
        "selected_channel_index": int(channel_index),
        "selected_time_index": time_index,
        "direct_identifiers_omitted": True,
    }


def _validated_index(index: int, extent: int, axis: str) -> int:
    if isinstance(index, bool) or not isinstance(index, int):
        raise TypeError(f"{axis} index must be an integer.")
    if index < 0 or index >= extent:
        raise IndexError(f"{axis} index is outside the available range.")
    return index


def _observed_crossovers(
    results: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    crossovers: dict[str, object] = {}
    observed_dtypes = tuple(
        dict.fromkeys(
            str(result["dtype"])
            for result in results
            if result.get("source", {}).get("source_kind")
            == "deterministic-synthetic-stack"
        )
    )
    for dtype in observed_dtypes:
        cases = sorted(
            (
                result
                for result in results
                if result.get("dtype") == dtype
                and result.get("source", {}).get("source_kind")
                == "deterministic-synthetic-stack"
            ),
            key=lambda result: int(result["input_bytes"]),
        )
        available = [case for case in cases if case["gpu"]["status"] == "available"]
        cold_thresholds = {
            int(
                case["production_auto_policy"]["before_gpu_evidence"]["threshold_bytes"]
            )
            for case in cases
            if "production_auto_policy" in case
        }
        warm_thresholds = {
            int(
                case["production_auto_policy"]["after_this_engine_attempt"][
                    "threshold_bytes"
                ]
            )
            for case in available
            if "production_auto_policy" in case
        }
        crossovers[dtype] = {
            "cold_sustained_gpu_no_slower_from_bytes": _sustained_crossover(
                available,
                gpu_key="cold_seconds",
            ),
            "warm_sustained_gpu_no_slower_from_bytes": _sustained_crossover(
                available,
                gpu_key="warm_median_seconds",
            ),
            "all_cases_gpu_available": len(available) == len(cases) and bool(cases),
            "production_cold_auto_threshold_bytes": _single_value_or_none(
                cold_thresholds
            ),
            "production_warm_auto_threshold_bytes": _single_value_or_none(
                warm_thresholds
            ),
        }
    return crossovers


def _single_value_or_none(values: set[int]) -> int | None:
    if len(values) > 1:
        raise BenchmarkError("Production Auto thresholds changed within one dtype.")
    return next(iter(values), None)


def _sustained_crossover(
    cases: Sequence[Mapping[str, object]],
    *,
    gpu_key: str,
) -> int | None:
    for index, case in enumerate(cases):
        tail = cases[index:]
        if all(
            float(item["gpu"][gpu_key]) <= float(item["cpu"]["median_seconds"])
            for item in tail
            if item["gpu"].get(gpu_key) is not None
        ) and all(item["gpu"].get(gpu_key) is not None for item in tail):
            return int(case["input_bytes"])
    return None


def render_human_summary(document: Mapping[str, object]) -> str:
    lines = [
        "VIPP thumbnail-statistics CPU/CuPy calibration",
        "Cold = first production GPU call in a fresh process; warm = later calls.",
        "Times include production transfer, runtime ownership, and cleanup.",
        "",
        f"{'dtype':<7} {'MiB':>6} {'CPU med':>10} {'GPU cold':>10} "
        f"{'GPU warm':>10} {'cold x':>8} {'warm x':>8} {'parity':>8}",
    ]
    for result in document["results"]:
        size_mib = float(result["input_bytes"]) / 1024**2
        gpu = result["gpu"]
        cold = _format_seconds(gpu["cold_seconds"])
        warm = _format_seconds(gpu["warm_median_seconds"])
        cold_speedup = _format_speedup(result["speedup"]["cpu_over_cold_gpu"])
        warm_speedup = _format_speedup(result["speedup"]["cpu_over_warm_gpu"])
        parity = "exact" if result["exact_parity"] else gpu["status"]
        lines.append(
            f"{result['dtype']:<7} {size_mib:6.1f} "
            f"{_format_seconds(result['cpu']['median_seconds']):>10} "
            f"{cold:>10} {warm:>10} {cold_speedup:>8} {warm_speedup:>8} "
            f"{parity:>8}"
        )
    lines.extend(("", "Observed sustained crossover (measured matrix only):"))
    for dtype, values in document["observed_crossovers"].items():
        cold = _format_bytes(values["cold_sustained_gpu_no_slower_from_bytes"])
        warm = _format_bytes(values["warm_sustained_gpu_no_slower_from_bytes"])
        cold_policy = _format_bytes(values["production_cold_auto_threshold_bytes"])
        warm_policy = _format_bytes(values["production_warm_auto_threshold_bytes"])
        lines.append(
            f"  {dtype}: observed cold {cold}, warm {warm}; "
            f"production Auto cold {cold_policy}, warm {warm_policy}"
        )
    lines.append(
        "These machine-local observations inform thresholds; they are not a "
        "portable guarantee."
    )
    return "\n".join(lines)


def _format_seconds(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.4f}s"


def _format_speedup(value: object) -> str:
    return "n/a" if value is None else f"{float(value):.2f}x"


def _format_bytes(value: object) -> str:
    return "not observed" if value is None else f"{int(value) / 1024**2:.0f} MiB"


def _parse_sizes_mib(text: str) -> tuple[int, ...]:
    try:
        values = tuple(
            int(part.strip()) for part in str(text).split(",") if part.strip()
        )
    except ValueError as exc:
        raise ValueError("--sizes-mib must contain comma-separated integers.") from exc
    if not values or any(value <= 0 for value in values):
        raise ValueError("--sizes-mib must contain positive integers.")
    if len(values) != len(set(values)):
        raise ValueError("--sizes-mib values must be unique.")
    return tuple(sorted(values))


def _parse_dtypes(text: str) -> tuple[str, ...]:
    values = tuple(
        part.strip().lower() for part in str(text).split(",") if part.strip()
    )
    if not values or any(value not in SUPPORTED_DTYPES for value in values):
        raise ValueError("--dtypes must contain uint8, uint16, and/or float32.")
    if len(values) != len(set(values)):
        raise ValueError("--dtypes values must be unique.")
    return values


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return value


def _limits_equal(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            _limits_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def _platform_provenance() -> dict[str, object]:
    packages = {}
    for distribution in ("numpy", "cupy-cuda12x", "cupy-cuda13x", "napari-vipp"):
        try:
            packages[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            packages[distribution] = None
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "executable": Path(sys.executable).name,
        "packages": packages,
        "hostname_omitted": True,
    }


def _source_provenance() -> dict[str, object]:
    hashes = {}
    for relative in SOURCE_PATHS:
        path = PROJECT_ROOT / relative
        hashes[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"hash_algorithm": "sha256", "files": hashes, "git": _git_identity()}


def _git_identity() -> dict[str, object]:
    try:
        head = subprocess.run(
            ("git", "-C", str(PROJECT_ROOT), "rev-parse", "--verify", "HEAD"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        status = subprocess.run(
            ("git", "-C", str(PROJECT_ROOT), "status", "--porcelain=v1"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False}
    if head.returncode != 0 or status.returncode != 0:
        return {"available": False}
    return {
        "available": True,
        "head": head.stdout.strip(),
        "worktree_dirty": bool(status.stdout.strip()),
    }


def _validate_document(document: Mapping[str, object]) -> None:
    if (
        document.get("schema") != SCHEMA
        or document.get("schema_version") != SCHEMA_VERSION
    ):
        raise BenchmarkError("Unexpected calibration evidence schema.")
    results = document.get("results")
    if not isinstance(results, list) or not results:
        raise BenchmarkError("Calibration evidence has no results.")
    matrix = document.get("matrix")
    if not isinstance(matrix, Mapping):
        raise BenchmarkError("Calibration evidence has no timing matrix.")
    cpu_rounds = _positive_integer(matrix.get("cpu_rounds"), "cpu_rounds")
    warm_rounds = _positive_integer(
        matrix.get("warm_gpu_rounds"),
        "warm_gpu_rounds",
    )
    case_ids: set[str] = set()
    for result in results:
        if not isinstance(result, Mapping):
            raise BenchmarkError("Calibration result must be an object.")
        case_id = str(result.get("case_id", ""))
        if not case_id or case_id in case_ids:
            raise BenchmarkError("Calibration case ids must be nonempty and unique.")
        case_ids.add(case_id)
        if int(result.get("input_bytes", 0)) <= 0:
            raise BenchmarkError("Calibration input byte counts must be positive.")
        cpu = result.get("cpu")
        gpu = result.get("gpu")
        if not isinstance(cpu, Mapping) or not isinstance(gpu, Mapping):
            raise BenchmarkError("Calibration result lacks CPU/GPU timing objects.")
        cpu_samples = cpu.get("samples_seconds")
        if not isinstance(cpu_samples, list) or len(cpu_samples) != cpu_rounds:
            raise BenchmarkError("CPU timing round count does not match the matrix.")
        for seconds in cpu_samples:
            if not np.isfinite(seconds) or seconds < 0:
                raise BenchmarkError("CPU timing is not finite and non-negative.")
        if float(cpu["median_seconds"]) != float(statistics.median(cpu_samples)):
            raise BenchmarkError("CPU median does not match the stored samples.")
        status = gpu.get("status")
        if status not in {"available", "unavailable"}:
            raise BenchmarkError("GPU status must be available or unavailable.")
        warm_samples = gpu.get("warm_samples_seconds")
        if not isinstance(warm_samples, list):
            raise BenchmarkError("GPU warm samples must be a list.")
        if status == "available":
            if not result.get("exact_parity"):
                raise BenchmarkError(
                    "A timed GPU result does not have exact CPU parity."
                )
            if len(warm_samples) != warm_rounds:
                raise BenchmarkError(
                    "GPU warm timing round count does not match the matrix."
                )
            if gpu.get("cold_seconds") is None:
                raise BenchmarkError("Available GPU result has no cold timing.")
            if float(gpu["warm_median_seconds"]) != float(
                statistics.median(warm_samples)
            ):
                raise BenchmarkError("GPU median does not match the stored samples.")
        elif (
            result.get("exact_parity")
            or gpu.get("cold_seconds") is not None
            or gpu.get("warm_median_seconds") is not None
            or warm_samples
        ):
            raise BenchmarkError(
                "Unavailable GPU result must not contain GPU timing or parity claims."
            )
        for seconds in warm_samples:
            if not np.isfinite(seconds) or seconds < 0:
                raise BenchmarkError("GPU timing is not finite and non-negative.")
        for key in ("cold_seconds", "warm_median_seconds"):
            seconds = gpu[key]
            if seconds is not None and (not np.isfinite(seconds) or seconds < 0):
                raise BenchmarkError("GPU timing is not finite and non-negative.")
    try:
        json.dumps(document, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("Calibration evidence is not strict JSON.") from exc


def _assert_private_source_redacted(document: object, path: Path | None) -> None:
    if path is None:
        return
    serialized = json.dumps(document, ensure_ascii=False, default=str).casefold()
    resolved = Path(path).expanduser().resolve(strict=False)
    forbidden = {str(resolved).casefold(), resolved.name.casefold()}
    if any(value and value in serialized for value in forbidden):
        raise BenchmarkError(
            "Private ND2 path or filename reached serialized evidence."
        )


def _redact_private_text(text: str, path: Path | None) -> str:
    result = str(text)
    if path is None:
        return result
    resolved = Path(path).expanduser().resolve(strict=False)
    for private in (str(path), str(resolved), resolved.name):
        if private:
            result = result.replace(private, "<private ND2 omitted>")
    return result


def _atomic_write_json(output: Path, document: Mapping[str, object]) -> Path:
    try:
        encoded = (
            json.dumps(
                document,
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise BenchmarkError("Calibration evidence is not strict JSON.") from exc
    requested = Path(output).expanduser()
    if requested.is_symlink():
        raise BenchmarkError("--output must not be a symbolic link.")
    path = requested.resolve(strict=False)
    if path.exists() and path.is_dir():
        raise BenchmarkError("--output refers to a directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


if __name__ == "__main__":
    raise SystemExit(main())
