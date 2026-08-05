"""Bounded CuPy histograms for presentation-only thumbnail statistics."""

from __future__ import annotations

import importlib
from functools import cache
from types import ModuleType

import numpy as np

from napari_vipp.core.progress import ProgressContext

_THREADS_PER_BLOCK = 256
_MAXIMUM_BLOCKS = 65_535


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only after the core selector explicitly chooses CUDA."""

    return importlib.import_module("cupy")


@cache
def _histogram_kernel(cupy: ModuleType, dtype_name: str):
    if dtype_name == "uint8":
        native_type = "unsigned char"
        kernel_name = "vipp_thumbnail_histogram_uint8"
    elif dtype_name == "uint16":
        native_type = "unsigned short"
        kernel_name = "vipp_thumbnail_histogram_uint16"
    else:  # pragma: no cover - guarded by the public adapter
        raise TypeError(f"Unsupported thumbnail histogram dtype {dtype_name!r}.")
    return cupy.RawKernel(
        rf"""
        extern "C" __global__
        void {kernel_name}(
            const {native_type}* values,
            const unsigned long long size,
            const unsigned long long channel_count,
            const unsigned long long channel_stride,
            const unsigned long long level_count,
            unsigned long long* counts)
        {{
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long grid_stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < size; index += grid_stride) {{
                const unsigned long long channel =
                    channel_count == 1ULL
                    ? 0ULL
                    : (index / channel_stride) % channel_count;
                const unsigned long long level =
                    (unsigned long long)values[index];
                atomicAdd(
                    counts + channel * level_count + level,
                    1ULL);
            }}
        }}
        """,
        kernel_name,
        options=("--std=c++11",),
    )


def exact_uint_histogram_counts(
    runtime,
    data,
    *,
    device_id: str,
    channel_axis: int | None = None,
    progress: ProgressContext | None = None,
) -> np.ndarray:
    """Count every uint8/uint16 level within an active private runtime scope.

    The input is uploaded once.  A channel-aware kernel writes only bounded
    uint64 counts, and only those counts return to the host.  Device arrays are
    released and all Python aliases dropped before the caller exits its runtime
    scope, allowing the established CuPy lifecycle checks to verify zero live
    private allocations.
    """

    arr = np.asarray(data)
    if arr.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        raise TypeError("CuPy thumbnail histograms require native uint8 or uint16.")
    if arr.size == 0:
        level_count = 256 if arr.dtype == np.dtype(np.uint8) else 65_536
        if channel_axis is None:
            return np.zeros(level_count, dtype=np.uint64)
        axis = _normalized_channel_axis(channel_axis, arr.ndim)
        return np.zeros((arr.shape[axis], level_count), dtype=np.uint64)

    axis = (
        None
        if channel_axis is None
        else _normalized_channel_axis(channel_axis, arr.ndim)
    )
    channel_count = 1 if axis is None else int(arr.shape[axis])
    channel_stride = (
        1
        if axis is None or axis == arr.ndim - 1
        else int(np.prod(arr.shape[axis + 1 :], dtype=np.int64))
    )
    level_count = 256 if arr.dtype == np.dtype(np.uint8) else 65_536
    total = int(arr.size)
    _report(
        progress,
        0,
        total,
        "Uploading thumbnail statistics to GPU · cancel applies after this pass",
    )

    cupy = _cupy_module()
    device_values = None
    device_counts = None
    try:
        device_values = runtime.to_device(arr, device_id=device_id)
        _report(
            progress,
            max(1, total // 4),
            total,
            "Counting exact thumbnail intensity levels on GPU · "
            "cancel applies after this pass",
        )
        device_counts = cupy.zeros(
            (channel_count, level_count),
            dtype=cupy.uint64,
        )
        block_count = min(
            (total + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,
            _MAXIMUM_BLOCKS,
        )
        kernel = _histogram_kernel(cupy, np.dtype(arr.dtype).name)
        kernel(
            (block_count,),
            (_THREADS_PER_BLOCK,),
            (
                device_values,
                np.uint64(total),
                np.uint64(channel_count),
                np.uint64(channel_stride),
                np.uint64(level_count),
                device_counts,
            ),
        )
        runtime.synchronize(device_id=device_id)
        _report(
            progress,
            max(1, 3 * total // 4),
            total,
            "Returning bounded thumbnail histogram from GPU · "
            "cancel applies after this pass",
        )
        host_counts = np.asarray(runtime.to_host(device_counts), dtype=np.uint64)
        _report(progress, total, total, "Thumbnail GPU histogram ready")
    finally:
        release_errors = []
        for value in (device_counts, device_values):
            if value is None:
                continue
            try:
                runtime.release(value)
            except Exception as exc:
                release_errors.append(exc)
        device_counts = None
        device_values = None
        if release_errors:
            detail = "; ".join(f"{type(exc).__name__}: {exc}" for exc in release_errors)
            raise RuntimeError(
                "Thumbnail GPU allocations could not be relinquished cleanly: " + detail
            ) from release_errors[0]
    return host_counts[0] if axis is None else host_counts


def _normalized_channel_axis(channel_axis: int, ndim: int) -> int:
    if ndim <= 0:
        raise ValueError("A scalar array cannot have a channel axis.")
    axis = int(channel_axis)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"Channel axis {channel_axis} is outside an array with {ndim} axes."
        )
    return axis


def _report(
    progress: ProgressContext | None,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress is not None:
        progress.report(current, total, message)


__all__ = ["exact_uint_histogram_counts"]
