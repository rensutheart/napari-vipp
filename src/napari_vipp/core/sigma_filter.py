"""Authoritative CPU implementation of VIPP's Sigma Filter.

This module owns the complete clean-room scientific contract shared with the
CuPy implementation.  :mod:`napari_vipp.core.operations` re-exports the two
public callables so existing workflows and callers retain the same API.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import lru_cache
from numbers import Integral

import numpy as np

_SUPPORTED_DTYPES = frozenset(
    {np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)}
)
_ROW_BLOCK_SIZE = 64
_FLOAT32_SQUARE_LIMIT = float(np.float32(math.sqrt(float(np.finfo(np.float32).max))))


def sigma_filter_footprint(
    radius: float,
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Return Fiji Sigma Filter Plus' ordered circular YX footprint.

    The returned values are ``(squared_radius_limit, extent, offsets)``.  The
    offsets use deterministic row-major ``(dy, dx)`` order so the authoritative
    CPU implementation and the CUDA implementation can accumulate samples in
    the same order.
    """
    normalized = _validated_float(
        radius,
        name="radius",
        minimum=0.5,
        maximum=10.0,
    )
    if 1.5 <= normalized < 1.75:
        normalized = 1.75
    elif 2.5 <= normalized < 2.85:
        normalized = 2.85
    return _cached_footprint(normalized)


@lru_cache(maxsize=128)
def _cached_footprint(
    normalized_radius: float,
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    squared_radius_limit = math.floor(normalized_radius * normalized_radius) + 1
    extent = math.isqrt(squared_radius_limit)
    offsets = tuple(
        (dy, dx)
        for dy in range(-extent, extent + 1)
        for dx in range(-extent, extent + 1)
        if dx * dx + dy * dy <= squared_radius_limit
    )
    return squared_radius_limit, extent, offsets


def sigma_filter(
    data,
    radius: float = 2.0,
    sigma_width: float = 2.0,
    minimum_pixel_fraction: float = 0.2,
    outlier_aware: bool = True,
    channel_axis: int | None = None,
    progress=None,
) -> np.ndarray:
    """Apply a slice-wise edge-preserving Lee sigma filter on resolved YX.

    This clean-room implementation follows the documented behavior of Fiji's
    Sigma Filter Plus.  Each channel and every leading stack plane is processed
    independently.  The input is immutable, nearest/clamped boundary extension
    is used, and output keeps the input dtype.  Version 1 deliberately supports
    only finite ``uint8``, ``uint16``, and ``float32`` images.

    Population variance is accumulated in float64 from a float32 sample
    workspace.  Because a mathematical variance cannot be negative, any
    negative value produced by cancellation is clamped to positive zero before
    taking the square root.  Unsigned integer output uses clip plus half-up
    rounding, matching Fiji rather than NumPy's ties-to-even rounding.
    """
    arr = np.asarray(data)
    channel_axis = _validated_channel_axis(channel_axis, arr.ndim)
    if arr.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "Sigma Filter supports only uint8, uint16, and float32 input; "
            f"received {arr.dtype}."
        )
    if arr.size == 0:
        raise ValueError("Sigma Filter requires non-empty image data.")
    if arr.ndim - (channel_axis is not None) < 2:
        raise ValueError("Sigma Filter requires two resolved YX spatial axes.")
    _squared_radius_limit, footprint_extent, offsets = sigma_filter_footprint(radius)
    sigma_width = _validated_float(
        sigma_width,
        name="sigma_width",
        minimum=0.0,
    )
    minimum_pixel_fraction = _validated_float(
        minimum_pixel_fraction,
        name="minimum_pixel_fraction",
        minimum=0.0,
        maximum=1.0,
    )
    if not isinstance(outlier_aware, (bool, np.bool_)):
        raise ValueError("Sigma Filter outlier_aware must be a boolean.")

    y_axis, x_axis = _xy_axes(arr, channel_axis=channel_axis)
    leading_axes = tuple(
        axis for axis in range(arr.ndim) if axis not in {y_axis, x_axis}
    )
    permutation = leading_axes + (y_axis, x_axis)
    working = np.transpose(arr, permutation)
    leading_shape = working.shape[:-2]
    plane_count = int(np.prod(leading_shape, dtype=np.int64)) if leading_shape else 1
    rows = int(working.shape[-2])
    row_blocks = math.ceil(rows / _ROW_BLOCK_SIZE)
    validation_blocks = plane_count * row_blocks
    total_blocks = validation_blocks * 2
    if progress is not None:
        progress.report(0, total_blocks, "Sigma Filter validation")
        progress.check_cancelled()
    completed_blocks = _validate_working_input(
        working,
        progress=progress,
        total_blocks=total_blocks,
    )
    output = np.empty(working.shape, dtype=arr.dtype)

    minimum_count = math.ceil(len(offsets) * minimum_pixel_fraction)
    plane_indices = np.ndindex(leading_shape) if leading_shape else iter(((),))
    for plane_index in plane_indices:
        source_plane = np.ascontiguousarray(working[plane_index], dtype=np.float32)
        padded = np.pad(
            source_plane,
            footprint_extent,
            mode="edge",
        )
        destination_plane = output[plane_index]
        for row_start in range(0, rows, _ROW_BLOCK_SIZE):
            if progress is not None:
                progress.check_cancelled()
            row_stop = min(row_start + _ROW_BLOCK_SIZE, rows)
            filtered = _row_block(
                source_plane,
                padded,
                row_start=row_start,
                row_stop=row_stop,
                footprint_extent=footprint_extent,
                offsets=offsets,
                sigma_width=sigma_width,
                minimum_count=minimum_count,
                outlier_aware=bool(outlier_aware),
                check_cancelled=(
                    progress.check_cancelled if progress is not None else None
                ),
            )
            destination_plane[row_start:row_stop] = _restore_dtype(
                filtered,
                arr.dtype,
            )
            completed_blocks += 1
            if progress is not None:
                progress.report(
                    completed_blocks,
                    total_blocks,
                    "Sigma Filter rows",
                )

    inverse_permutation = tuple(int(axis) for axis in np.argsort(permutation))
    return np.ascontiguousarray(np.transpose(output, inverse_permutation))


def _row_block(
    source_plane: np.ndarray,
    padded: np.ndarray,
    *,
    row_start: int,
    row_stop: int,
    footprint_extent: int,
    offsets: tuple[tuple[int, int], ...],
    sigma_width: float,
    minimum_count: int,
    outlier_aware: bool,
    check_cancelled: Callable[[], None] | None,
) -> np.ndarray:
    width = int(source_plane.shape[1])
    block_shape = (row_stop - row_start, width)
    full_sum = np.zeros(block_shape, dtype=np.float64)
    full_sum_squared = np.zeros(block_shape, dtype=np.float64)
    scratch = np.empty(block_shape, dtype=np.float64)
    sample_squared = np.empty(block_shape, dtype=np.float32)

    for offset_index, (dy, dx) in enumerate(offsets):
        if check_cancelled is not None and offset_index % 16 == 0:
            check_cancelled()
        sample = padded[
            row_start + footprint_extent + dy : row_stop + footprint_extent + dy,
            footprint_extent + dx : footprint_extent + dx + width,
        ]
        np.add(full_sum, sample, out=full_sum)
        # Fiji's documented implementation squares the float sample before
        # accumulating into a double.  Keep that precision boundary explicit;
        # the resulting tiny negative variance is clamped below.
        np.multiply(sample, sample, out=sample_squared)
        np.add(full_sum_squared, sample_squared, out=full_sum_squared)

    footprint_count = len(offsets)
    mean = full_sum / footprint_count
    variance = full_sum_squared / footprint_count
    np.square(mean, out=scratch)
    variance -= scratch
    # Cancellation can make an exactly non-negative population variance a
    # tiny negative number.  Clamping every negative result is deterministic,
    # mathematically faithful, and shared verbatim by the CUDA contract.
    np.maximum(variance, 0.0, out=variance)
    np.sqrt(variance, out=variance)
    with np.errstate(over="ignore", invalid="ignore"):
        variance *= sigma_width
        center = source_plane[row_start:row_stop]
        lower = center - variance
        upper = center + variance

    selected_sum = np.zeros(block_shape, dtype=np.float64)
    selected_count = np.zeros(block_shape, dtype=np.uint16)
    selected = np.empty(block_shape, dtype=bool)
    upper_selected = np.empty(block_shape, dtype=bool)
    for offset_index, (dy, dx) in enumerate(offsets):
        if check_cancelled is not None and offset_index % 16 == 0:
            check_cancelled()
        sample = padded[
            row_start + footprint_extent + dy : row_stop + footprint_extent + dy,
            footprint_extent + dx : footprint_extent + dx + width,
        ]
        np.greater_equal(sample, lower, out=selected)
        np.less_equal(sample, upper, out=upper_selected)
        np.logical_and(selected, upper_selected, out=selected)
        np.add(selected_sum, sample, out=selected_sum, where=selected)
        np.add(selected_count, selected, out=selected_count, casting="unsafe")

    selected_mean = selected_sum / selected_count
    if outlier_aware:
        fallback = (full_sum - center) / (footprint_count - 1)
    else:
        fallback = mean
    return np.where(selected_count >= minimum_count, selected_mean, fallback)


def _validate_working_input(
    working: np.ndarray,
    *,
    progress,
    total_blocks: int,
) -> int:
    completed_blocks = 0
    leading_shape = working.shape[:-2]
    plane_indices = np.ndindex(leading_shape) if leading_shape else iter(((),))
    for plane_index in plane_indices:
        plane = working[plane_index]
        for row_start in range(0, plane.shape[0], _ROW_BLOCK_SIZE):
            if progress is not None:
                progress.check_cancelled()
            block = np.asarray(
                plane[row_start : row_start + _ROW_BLOCK_SIZE],
                dtype=np.float32,
            )
            if not np.isfinite(block).all():
                raise ValueError("Sigma Filter requires finite image intensities.")
            if working.dtype == np.float32 and float(np.max(np.abs(block))) > (
                _FLOAT32_SQUARE_LIMIT
            ):
                raise ValueError(
                    "Sigma Filter float32 input magnitude would overflow the "
                    "Fiji-compatible float32 square workspace."
                )
            completed_blocks += 1
            if progress is not None:
                progress.report(
                    completed_blocks,
                    total_blocks,
                    "Sigma Filter validation",
                )
    return completed_blocks


def _validated_float(
    value,
    *,
    name: str,
    minimum: float,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"Sigma Filter {name} must be a finite number.")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Sigma Filter {name} must be a finite number.") from exc
    if not math.isfinite(normalized):
        raise ValueError(f"Sigma Filter {name} must be finite.")
    if normalized < minimum or (maximum is not None and normalized > maximum):
        interval = (
            f"{minimum} or greater"
            if maximum is None
            else f"between {minimum} and {maximum} inclusive"
        )
        raise ValueError(f"Sigma Filter {name} must be {interval}.")
    # Canonicalize negative zero so equivalent authored parameter values share
    # one footprint/cache identity and one CPU/GPU threshold contract.
    return 0.0 if normalized == 0.0 else normalized


def _restore_dtype(
    values: np.ndarray,
    dtype: np.dtype,
) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype in {np.dtype(np.uint8), np.dtype(np.uint16)}:
        info = np.iinfo(dtype)
        fiji_values = np.asarray(values, dtype=np.float32)
        rounded = np.floor(fiji_values + np.float32(0.5))
        return np.clip(rounded, 0, info.max).astype(dtype)
    return np.asarray(values, dtype=np.float32)


def _validated_channel_axis(value, ndim: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Sigma Filter channel_axis must be an integer or None.")
    if ndim < 3:
        raise ValueError(
            "Sigma Filter requires at least two spatial dimensions when "
            "channel_axis is set."
        )
    axis = int(value)
    if axis < -ndim or axis >= ndim:
        raise ValueError(
            f"Sigma Filter channel_axis {axis} is out of range for {ndim}D input."
        )
    return axis % ndim


def _xy_axes(
    arr: np.ndarray,
    *,
    channel_axis: int | None = None,
) -> tuple[int, int]:
    """Return trailing spatial Y/X axes without shape inference."""
    spatial_axes = list(range(arr.ndim))
    if channel_axis is not None:
        spatial_axes.remove(channel_axis)
    if len(spatial_axes) >= 2:
        return spatial_axes[-2], spatial_axes[-1]
    if len(spatial_axes) == 1:
        return spatial_axes[0], spatial_axes[0]
    return 0, 0


__all__ = ["sigma_filter", "sigma_filter_footprint"]
