"""Exact device-resident ImageJ-style cleanup for boolean masks.

CuPy remains optional and is imported only after this provider is selected.
The fixed-source CUDA kernel receives the ImageJ row spans, radius-dependent
point count, and selected polarity as runtime data so changing an authored
parameter never creates a new compiled-kernel specialization.
"""

from __future__ import annotations

import importlib
import math
from functools import cache
from types import ModuleType

import numpy as np

from napari_vipp.core.remove_outliers import (
    FOREGROUND_OUTLIERS,
    OUTLIER_CHOICES,
    imagej_remove_outliers_footprint,
)

_ROW_TILE_SIZE = 64
_TARGET_SAMPLE_VISITS_PER_TILE = 32 * 1024 * 1024
_THREADS_PER_BLOCK = 256
_MAXIMUM_BLOCKS = 65_535
_PROGRESS_MESSAGE = "Remove Outliers (Binary) pixels"
_KERNEL_OPTIONS = ("--std=c++11",)


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only for an explicitly selected accelerator execution."""

    return importlib.import_module("cupy")


@cache
def _remove_binary_outliers_kernel(cupy: ModuleType):
    """Compile the one bool-only, parameter-independent CUDA kernel."""

    return cupy.RawKernel(
        r"""
        extern "C" __global__
        void vipp_remove_binary_outliers_bool(
            const unsigned char* source,
            const long long rows,
            const long long columns,
            const long long pixel_start,
            const long long pixel_stop,
            const int* row_half_widths,
            const int footprint_rows,
            const unsigned int majority_threshold,
            const int remove_foreground,
            unsigned char* output)
        {
            unsigned long long local_index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long)blockDim.x * gridDim.x;
            const unsigned long long tile_size =
                (unsigned long long)(pixel_stop - pixel_start);
            const int center_row = footprint_rows / 2;

            for (; local_index < tile_size; local_index += stride) {
                const long long output_index = pixel_start + (long long)local_index;
                const long long y = output_index / columns;
                const long long x =
                    output_index % columns;
                unsigned int foreground_count = 0U;

                for (int row_index = 0;
                     row_index < footprint_rows;
                     ++row_index) {
                    long long yy = y + (long long)(row_index - center_row);
                    yy = yy < 0 ? 0 : (yy >= rows ? rows - 1 : yy);
                    const int half_width = row_half_widths[row_index];
                    for (int dx = -half_width; dx <= half_width; ++dx) {
                        long long xx = x + (long long)dx;
                        xx = xx < 0 ? 0 : (xx >= columns ? columns - 1 : xx);
                        foreground_count +=
                            source[yy * columns + xx] != 0U ? 1U : 0U;
                    }
                }

                const bool center_foreground = source[output_index] != 0U;
                const bool local_majority =
                    foreground_count > majority_threshold;
                output[output_index] = remove_foreground
                    ? (unsigned char)(center_foreground && local_majority)
                    : (unsigned char)(center_foreground || local_majority);
            }
        }
        """,
        "vipp_remove_binary_outliers_bool",
        options=_KERNEL_OPTIONS,
    )


def remove_binary_outliers(
    data,
    radius: float = 2.0,
    which_outliers: str = FOREGROUND_OUTLIERS,
    progress=None,
):
    """Remove foreground specks or fill background notches in resident masks.

    Every leading index is processed as an independent trailing-YX plane.
    ImageJ's exact historical circular footprint and nearest-edge extension are
    shared with the authoritative CPU operation.  The admitted accelerator
    region is bool-only and the returned resident bool array is always a new,
    contiguous allocation.
    """

    if progress is not None:
        progress.check_cancelled()

    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None and np.dtype(source_dtype) != np.dtype(bool):
        raise ValueError(
            "Remove Outliers (Binary) GPU execution requires a boolean mask; "
            "canonical uint8 validation remains authoritative on CPU."
        )
    footprint = imagej_remove_outliers_footprint(radius)
    polarity = _validated_outlier_choice(which_outliers)

    cupy = _cupy_module()
    array = cupy.asarray(data)
    if np.dtype(array.dtype) != np.dtype(bool):
        raise ValueError(
            "Remove Outliers (Binary) GPU execution requires a boolean mask; "
            "canonical uint8 validation remains authoritative on CPU."
        )
    if int(array.size) == 0:
        raise ValueError("Remove Outliers (Binary) does not accept empty masks.")
    if int(array.ndim) < 2:
        raise ValueError(
            "Remove Outliers (Binary) requires at least two dimensions with "
            "trailing YX axes."
        )

    working = cupy.ascontiguousarray(array)
    leading_shape = tuple(int(size) for size in working.shape[:-2])
    plane_count = math.prod(leading_shape) if leading_shape else 1
    rows, columns = (int(size) for size in working.shape[-2:])
    row_half_widths = _footprint_row_half_widths(footprint)
    footprint_point_count = int(np.count_nonzero(footprint))
    plane_pixels = rows * columns
    tile_pixels = _pixel_tile_size(
        columns=columns,
        footprint_point_count=footprint_point_count,
    )
    tiles_per_plane = math.ceil(plane_pixels / tile_pixels)
    total_tiles = plane_count * tiles_per_plane
    device_row_half_widths = cupy.asarray(row_half_widths)
    output = cupy.empty(working.shape, dtype=bool)
    working_planes = working.reshape(plane_count, rows, columns)
    output_planes = output.reshape(plane_count, rows, columns)
    kernel = _remove_binary_outliers_kernel(cupy)
    remove_foreground = polarity == FOREGROUND_OUTLIERS

    completed = 0
    if progress is not None:
        progress.report(0, total_tiles, _PROGRESS_MESSAGE)
        progress.check_cancelled()
    for plane_index in range(plane_count):
        for pixel_start in range(0, plane_pixels, tile_pixels):
            if progress is not None:
                progress.check_cancelled()
            pixel_stop = min(pixel_start + tile_pixels, plane_pixels)
            active_tile_pixels = pixel_stop - pixel_start
            block_count = min(
                math.ceil(active_tile_pixels / _THREADS_PER_BLOCK),
                _MAXIMUM_BLOCKS,
            )
            kernel(
                (block_count,),
                (_THREADS_PER_BLOCK,),
                (
                    working_planes[plane_index],
                    np.int64(rows),
                    np.int64(columns),
                    np.int64(pixel_start),
                    np.int64(pixel_stop),
                    device_row_half_widths,
                    np.int32(row_half_widths.size),
                    np.uint32(footprint_point_count // 2),
                    np.int32(remove_foreground),
                    output_planes[plane_index],
                ),
            )
            completed += 1
            if progress is not None:
                cupy.cuda.get_current_stream().synchronize()
                progress.check_cancelled()
                progress.report(completed, total_tiles, _PROGRESS_MESSAGE)
                # A cancellation raised by the terminal report must still
                # prevent the synchronized result from being returned.
                progress.check_cancelled()
    return output


def _pixel_tile_size(*, columns: int, footprint_point_count: int) -> int:
    """Bound one launch by both 64 rows and a sample-visit budget."""

    row_cap = _ROW_TILE_SIZE * columns
    sample_cap = max(1, _TARGET_SAMPLE_VISITS_PER_TILE // footprint_point_count)
    return max(1, min(row_cap, sample_cap))


def _footprint_row_half_widths(footprint: np.ndarray) -> np.ndarray:
    """Return each non-empty ImageJ footprint row's contiguous half-width."""

    center = footprint.shape[1] // 2
    half_widths = np.empty(footprint.shape[0], dtype=np.int32)
    for row_index, row in enumerate(footprint):
        populated = np.flatnonzero(row)
        half_widths[row_index] = int(populated[-1]) - center
    return half_widths


def _validated_outlier_choice(which_outliers: str) -> str:
    choice = str(which_outliers).strip()
    if choice not in OUTLIER_CHOICES:
        expected = " or ".join(repr(value) for value in OUTLIER_CHOICES)
        raise ValueError(f"Remove Outliers (Binary) outlier type must be {expected}.")
    return choice


__all__ = ["remove_binary_outliers"]
