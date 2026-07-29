"""Device-resident CuPy adapter for VIPP's Otsu threshold contract.

The image-sized finite-value histogram and final mask are evaluated on the
GPU.  Only the bounded histogram is finalized on the host so VIPP retains the
authoritative NumPy float64 cumulative arithmetic and first-maximum tie
breaking.  Optional CUDA imports remain lazy.
"""

from __future__ import annotations

import importlib
from functools import cache
from itertools import product
from numbers import Integral
from types import ModuleType

import numpy as np

_DEFAULT_HISTOGRAM_BINS = 256
_MAX_NATIVE_INTEGER_HISTOGRAM_BINS = 65_536
_HISTOGRAM_THREADS_PER_BLOCK = 256
_HISTOGRAM_MAXIMUM_BLOCKS = 65_535


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only for an explicit accelerator execution."""

    return importlib.import_module("cupy")


@cache
def _atomic_bincount_kernel(cupy: ModuleType, dtype_name: str):
    """Compile a bounded-workspace exact histogram for signed/unsigned indices.

    CuPy's public ``bincount`` can dispatch wide integer histograms through
    CUB.  CUB owns one privatized histogram per resident sweep block when the
    bin count does not fit in shared memory, so a 65,536-level image can reserve
    hundreds of MiB independently of the logical input/output arrays.  VIPP's
    admission model cannot express that device-occupancy-dependent workspace.

    Our indices are already exact integer bin numbers.  Counting them directly
    into one uint64 histogram preserves those semantics while making temporary
    storage independent of both CUDA occupancy and the number of bins.  The
    two kernels avoid an image-sized uint64-to-int64 conversion on the native
    integer path.
    """

    if dtype_name == "uint64":
        index_type = "unsigned long long"
        valid = "value < bin_count"
        kernel_name = "vipp_otsu_bincount_uint64"
    elif dtype_name == "int64":
        index_type = "long long"
        valid = "value >= 0 && (unsigned long long)value < bin_count"
        kernel_name = "vipp_otsu_bincount_int64"
    else:  # pragma: no cover - private helper contract
        raise ValueError(f"Unsupported exact histogram index dtype {dtype_name!r}.")

    return cupy.RawKernel(
        rf"""
        extern "C" __global__
        void {kernel_name}(
            const {index_type}* values,
            const unsigned long long size,
            const unsigned long long bin_count,
            unsigned long long* counts)
        {{
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < size; index += stride) {{
                const {index_type} value = values[index];
                if ({valid}) {{
                    atomicAdd(counts + (unsigned long long)value, 1ULL);
                }}
            }}
        }}
        """,
        kernel_name,
        options=("--std=c++11",),
    )


def _exact_bincount(indices, *, bin_count: int, cupy):
    """Count validated int64/uint64 indices with bounded device storage."""

    flattened = indices.reshape(-1)
    dtype_name = np.dtype(flattened.dtype).name
    if dtype_name not in {"int64", "uint64"}:
        raise ValueError(
            "Exact Otsu histogram indices must use int64 or uint64 storage."
        )
    counts = cupy.zeros(int(bin_count), dtype=cupy.uint64)
    if not int(flattened.size):
        return counts
    block_count = min(
        (int(flattened.size) + _HISTOGRAM_THREADS_PER_BLOCK - 1)
        // _HISTOGRAM_THREADS_PER_BLOCK,
        _HISTOGRAM_MAXIMUM_BLOCKS,
    )
    kernel = _atomic_bincount_kernel(cupy, dtype_name)
    kernel(
        (block_count,),
        (_HISTOGRAM_THREADS_PER_BLOCK,),
        (
            flattened,
            np.uint64(flattened.size),
            np.uint64(bin_count),
            counts,
        ),
    )
    return counts


def otsu_threshold(
    data,
    threshold_scope: str = "Stack histogram",
    histogram_bins: int = _DEFAULT_HISTOGRAM_BINS,
    channel_axis: int | None = None,
    progress=None,
):
    """Return VIPP's exact boolean Otsu mask in the CuPy array domain.

    The adapter preserves the CPU operation's finite-value policy, exact
    native integer levels, float histogram edges, RGB/RGBA BT.601 reduction,
    stack/slice scope, constant and boolean behavior, and ``array > threshold``
    foreground rule.
    """

    cupy = _cupy_module()
    array = cupy.asarray(data)
    if array.size == 0:
        raise ValueError("Otsu threshold requires non-empty image data.")
    array = _to_explicit_grayscale(
        array,
        channel_axis=channel_axis,
        cupy=cupy,
    )
    scope = str(threshold_scope).strip().casefold()
    if scope not in {"stack histogram", "slice histogram"}:
        raise ValueError(
            "Threshold scope must be 'Stack histogram' or 'Slice histogram'."
        )

    # A scalar boolean image is already a segmentation.  This identity applies
    # only when no RGB/RGBA conversion has changed its dtype to float32.
    if np.dtype(array.dtype) == np.dtype(bool):
        if progress is not None:
            progress.check_cancelled()
        result = array.copy()
        if progress is not None:
            cupy.cuda.get_current_stream().synchronize()
            progress.report(1, 1, "Otsu histograms")
        return result

    if scope == "stack histogram" or array.ndim <= 2:
        return _threshold_with_progress(
            array,
            histogram_bins=histogram_bins,
            completed=0,
            total=1,
            progress=progress,
            cupy=cupy,
        )

    leading_shape = tuple(int(size) for size in array.shape[:-2])
    total = int(np.prod(leading_shape, dtype=np.int64))
    output = cupy.empty(array.shape, dtype=cupy.bool_)
    if progress is not None:
        progress.report(0, total, "Otsu slice histograms")
    for completed, index in enumerate(_indices(leading_shape)):
        output[index] = _threshold_with_progress(
            array[index],
            histogram_bins=histogram_bins,
            completed=completed,
            total=total,
            progress=progress,
            cupy=cupy,
            report_start=False,
        )
    return output


def _threshold_with_progress(
    array,
    *,
    histogram_bins,
    completed: int,
    total: int,
    progress,
    cupy,
    report_start: bool = True,
):
    if progress is not None:
        if report_start:
            progress.report(completed, total, "Otsu histogram")
        else:
            progress.check_cancelled()
    threshold = _otsu_value(array, histogram_bins=histogram_bins, cupy=cupy)
    mask = array > threshold
    if np.issubdtype(np.dtype(array.dtype), np.inexact):
        mask &= cupy.isfinite(array)
    if progress is not None:
        # Make the milestone truthful and surface device failures before the
        # operation advances or cooperative cancellation unwinds it.
        cupy.cuda.get_current_stream().synchronize()
        progress.report(completed + 1, total, "Otsu histogram ready")
    return mask


def _otsu_value(array, *, histogram_bins, cupy) -> int | float:
    dtype = np.dtype(array.dtype)
    if dtype == np.dtype(bool):
        # Kept for direct helper use; the public operation returns bool identity.
        false_count = int(cupy.count_nonzero(~array).item())
        true_count = int(array.size) - false_count
        if not true_count:
            return 0
        if not false_count:
            return 1
        counts = np.asarray((false_count, true_count), dtype=np.intp)
        return _threshold_from_histogram(
            counts,
            np.asarray((0, 1), dtype=np.uint8),
        )
    if np.issubdtype(dtype, np.integer):
        minimum = int(cupy.min(array).item())
        maximum = int(cupy.max(array).item())
        if minimum == maximum:
            return minimum
        span = maximum - minimum + 1
        if span > _MAX_NATIVE_INTEGER_HISTOGRAM_BINS:
            raise ValueError(
                f"Integer intensity span contains {span:,} levels; automatic "
                "thresholding supports at most "
                f"{_MAX_NATIVE_INTEGER_HISTOGRAM_BINS:,} exact integer levels. "
                "Convert or rescale the image to uint16 or floating point instead "
                "of silently collapsing integer levels."
            )
        relative = array.astype(cupy.uint64, copy=True)
        minimum_native = cupy.asarray(minimum, dtype=array.dtype).astype(cupy.uint64)
        relative -= minimum_native
        counts_device = _exact_bincount(
            relative,
            bin_count=span,
            cupy=cupy,
        )
        counts = np.asarray(cupy.asnumpy(counts_device), dtype=np.intp)
        centers = np.arange(span, dtype=np.int64)
        return minimum + int(_threshold_from_histogram(counts, centers))
    if not np.issubdtype(dtype, np.floating):
        raise ValueError(
            "Automatic histogram thresholds require boolean, integer, or "
            "floating-point image data."
        )

    bin_count = _validated_histogram_bins(histogram_bins)
    finite = cupy.isfinite(array)
    finite_values = array[finite]
    if not int(finite_values.size):
        raise ValueError(
            "Automatic thresholding requires at least one finite input value."
        )
    minimum = float(cupy.min(finite_values).item())
    maximum = float(cupy.max(finite_values).item())
    if minimum == maximum:
        return minimum

    # NumPy deliberately constructs float32 edges for float32 input (likewise
    # other floating dtypes).  Creating these bounded edges on the host exactly
    # preserves that public numerical contract.
    edge_limits = np.asarray((minimum, maximum), dtype=dtype)
    edges = np.histogram_bin_edges(edge_limits, bins=bin_count)
    values64 = finite_values.astype(cupy.float64, copy=False)
    counts_device = _exact_float_histogram(
        values64,
        edges=edges,
        cupy=cupy,
    )
    counts = np.asarray(cupy.asnumpy(counts_device), dtype=np.intp)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return float(_threshold_from_histogram(counts, centers))


def _exact_float_histogram(values64, *, edges: np.ndarray, cupy):
    """Count NumPy-authored bins without CuPy's uniform-range shortcut.

    ``cupy.histogram`` computes uniform-bin indices from a normalized range.
    That shortcut can overflow for finite float32 values near their limits and
    can round differently for wide float64 ranges.  NumPy's authoritative
    result is comparison-defined: internal bins are half-open and the final
    edge is inclusive.  Searching the exact NumPy-authored edge array on the
    device preserves those semantics for every finite range while transferring
    only the bounded counts to the host.
    """

    edges_device = cupy.asarray(edges)
    bin_count = int(edges.size - 1)
    indices = cupy.searchsorted(edges_device, values64, side="right") - 1
    # ``searchsorted(..., side='right')`` maps a value equal to the final edge
    # one position past the last bin.  NumPy includes that endpoint in the last
    # bin; all values are otherwise bounded by edges derived from their extrema.
    cupy.minimum(indices, bin_count - 1, out=indices)
    return _exact_bincount(
        indices,
        bin_count=bin_count,
        cupy=cupy,
    )


def _threshold_from_histogram(
    hist: np.ndarray,
    centers: np.ndarray,
) -> int | float | np.number:
    """Apply the authoritative CPU Otsu arithmetic to bounded host data."""

    calculation_centers = (
        centers.astype(np.float64, copy=False)
        if np.issubdtype(centers.dtype, np.integer)
        else centers
    )
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * calculation_centers) / np.maximum(weight1, 1)
    mean2 = (
        np.cumsum((hist * calculation_centers)[::-1])
        / np.maximum(weight2[::-1], 1)
    )[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if variance12.size == 0:
        return (centers[0] + centers[-1]) / 2.0
    return centers[:-1][np.argmax(variance12)]


def _to_explicit_grayscale(array, *, channel_axis, cupy):
    axis = _validated_channel_axis(
        channel_axis,
        array.ndim,
        operation="Otsu threshold",
    )
    dtype = np.dtype(array.dtype)
    if axis is None:
        if not (
            dtype == np.dtype(bool)
            or np.issubdtype(dtype, np.integer)
            or np.issubdtype(dtype, np.floating)
        ):
            raise ValueError(
                "Automatic histogram thresholds require boolean, integer, or "
                "floating-point image data."
            )
        return array

    channel_count = int(array.shape[axis])
    if channel_count not in {3, 4}:
        raise ValueError(
            "Otsu threshold channel_axis must contain exactly 3 RGB or 4 RGBA "
            f"channels, not {channel_count}."
        )
    if not (
        dtype == np.dtype(bool)
        or np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    ):
        raise ValueError(
            "Otsu threshold RGB/RGBA conversion requires real-valued boolean, "
            "integer, or floating image data."
        )

    moved = cupy.moveaxis(array, axis, -1)
    work_dtype = np.result_type(dtype, np.float32)
    rgb = moved[..., :3].astype(work_dtype, copy=False)
    coefficients = cupy.asarray((0.299, 0.587, 0.114), dtype=work_dtype)
    return cupy.sum(rgb * coefficients, axis=-1, dtype=work_dtype)


def _validated_channel_axis(value, ndim: int, *, operation: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{operation} channel_axis must be an integer or None.")
    if ndim < 3:
        raise ValueError(
            f"{operation} requires at least two spatial dimensions when "
            "channel_axis is set."
        )
    axis = int(value)
    if axis < -ndim or axis >= ndim:
        raise ValueError(
            f"{operation} channel_axis {axis} is out of range for {ndim}D input."
        )
    return axis % ndim


def _validated_histogram_bins(histogram_bins: int) -> int:
    error = "Float histogram bins must be an integer from 2 to 65,536."
    if isinstance(histogram_bins, (bool, np.bool_)):
        raise ValueError(error)
    try:
        bin_count = int(histogram_bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if isinstance(histogram_bins, (float, np.floating)) and not float(
        histogram_bins
    ).is_integer():
        raise ValueError(error)
    if not 2 <= bin_count <= _MAX_NATIVE_INTEGER_HISTOGRAM_BINS:
        raise ValueError(error)
    return bin_count


def _indices(shape: tuple[int, ...]):
    return product(*(range(size) for size in shape))


__all__ = ["otsu_threshold"]
