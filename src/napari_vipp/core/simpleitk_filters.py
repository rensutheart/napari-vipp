"""Qualified CPU filtering adapters; importing this module does not load ITK.

The median adapter implements the existing SciPy half-sample-symmetric XY
contract, not SimpleITK's default boundary convention. Non-XY dimensions remain
independent. It selects an existing sample without intensity conversion.
"""

from __future__ import annotations

import importlib
import math
import os
from functools import cache
from types import ModuleType
from typing import Literal

import numpy as np
from scipy import ndimage as ndi

MEDIAN_ADAPTER_VERSION = "1"
MEDIAN_MIN_PLANE_PIXELS = 65_536
MEDIAN_MAX_SIZE = 51
_MAX_CHUNK_BYTES = 64 * 1024 * 1024
_MAX_THREADS = 12
_MEDIAN_DTYPES = frozenset(
    np.dtype(name)
    for name in (
        "uint8",
        "uint16",
        "uint32",
        "int8",
        "int16",
        "int32",
        "float32",
        "float64",
    )
)


@cache
def _simpleitk_module() -> ModuleType:
    """Load the packaged CPU backend only when a qualified call uses it."""
    return importlib.import_module("SimpleITK")


def median_filter_backend(
    array: np.ndarray,
    *,
    size: int,
    xy_axes: tuple[int, int],
) -> Literal["simpleitk", "scipy"]:
    """Deterministically choose the implementation for canonical arguments.

    The caller owns public parameter and channel-axis validation. This same
    decision is available to execution provenance; it never imports SimpleITK.
    Small/degenerate planes, unqualified sizes, wide integers, Boolean data and
    non-native dtypes keep their previous SciPy implementation. Floating-point
    data must be finite and contain no negative zero: median tie selection is
    not bitwise interchangeable between implementations for those values.
    """
    if (
        array.ndim < 2
        or array.size == 0
        or xy_axes[0] == xy_axes[1]
        or array.dtype not in _MEDIAN_DTYPES
        or not array.dtype.isnative
        or not 3 <= size <= MEDIAN_MAX_SIZE
        or size % 2 == 0
    ):
        return "scipy"
    height, width = (array.shape[axis] for axis in xy_axes)
    if min(height, width) < size or height * width < MEDIAN_MIN_PLANE_PIXELS:
        return "scipy"
    if array.dtype.kind == "f":
        # Iterate in bounded chunks even for non-contiguous/read-only input.
        iterator = np.nditer(
            array,
            flags=["external_loop", "buffered", "zerosize_ok"],
            op_flags=["readonly"],
            buffersize=_MAX_CHUNK_BYTES // array.dtype.itemsize,
        )
        for chunk in iterator:
            if not np.isfinite(chunk).all() or np.any((chunk == 0) & np.signbit(chunk)):
                return "scipy"
    return "simpleitk"


def scipy_median_filter(
    array: np.ndarray,
    *,
    size: int,
    xy_axes: tuple[int, int],
) -> np.ndarray:
    """Unchanged authoritative CPU reference and unqualified-input path."""
    filter_size = [1] * array.ndim
    for axis in xy_axes:
        filter_size[axis] = size
    return ndi.median_filter(array, size=filter_size)


def median_filter(
    array: np.ndarray,
    *,
    size: int,
    xy_axes: tuple[int, int],
) -> np.ndarray:
    """Run the qualified backend, preserving input shape, dtype and buffers."""
    backend = median_filter_backend(array, size=size, xy_axes=xy_axes)
    if backend == "simpleitk":
        return _simpleitk_median_filter(array, size=size, xy_axes=xy_axes)
    return scipy_median_filter(array, size=size, xy_axes=xy_axes)


def _simpleitk_median_filter(
    array: np.ndarray,
    *,
    size: int,
    xy_axes: tuple[int, int],
) -> np.ndarray:
    """Filter admitted arguments; private entry point for exact qualification.

    Symmetric padding includes the edge sample and matches SciPy ``reflect``.
    The halo is discarded, so SimpleITK's own edge extension is never observed.
    ITK sees a private scalar 3D buffer with radius zero on the plane axis. No
    channel, time or Z samples can enter another plane's median footprint.
    """
    sitk = _simpleitk_module()
    radius = size // 2
    source = np.moveaxis(array, xy_axes, (-2, -1))
    result = np.empty(array.shape, dtype=array.dtype)
    destination = np.moveaxis(result, xy_axes, (-2, -1))
    height, width = source.shape[-2:]
    leading_shape = source.shape[:-2]
    plane_count = math.prod(leading_shape)
    padded_shape = (height + 2 * radius, width + 2 * radius)
    plane_bytes = math.prod(padded_shape) * array.dtype.itemsize
    chunk_planes = max(1, _MAX_CHUNK_BYTES // plane_bytes)

    method = sitk.MedianImageFilter()
    method.SetRadius([radius, radius, 0])
    threads = max(1, min(os.cpu_count() or 1, _MAX_THREADS))
    method.SetNumberOfThreads(threads)
    method.SetNumberOfWorkUnits(threads)

    for start in range(0, plane_count, chunk_planes):
        count = min(chunk_planes, plane_count - start)
        padded = np.empty((count, *padded_shape), dtype=array.dtype)
        indices = [
            np.unravel_index(index, leading_shape) if leading_shape else ()
            for index in range(start, start + count)
        ]
        for local_index, index in enumerate(indices):
            padded[local_index] = np.pad(
                source[index],
                ((radius, radius), (radius, radius)),
                mode="symmetric",
            )
        filtered = sitk.GetArrayFromImage(
            method.Execute(sitk.GetImageFromArray(padded, isVector=False))
        )
        for local_index, index in enumerate(indices):
            destination[index] = filtered[
                local_index, radius : radius + height, radius : radius + width
            ]
    return result


__all__ = [
    "MEDIAN_ADAPTER_VERSION",
    "median_filter",
    "median_filter_backend",
    "scipy_median_filter",
]
