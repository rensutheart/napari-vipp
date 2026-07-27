"""Lazy CuPyX adapter for VIPP's slice-wise median filter contract."""

from __future__ import annotations

import importlib
from functools import cache
from numbers import Integral
from types import ModuleType


@cache
def _cupy_modules() -> tuple[ModuleType, ModuleType]:
    """Load optional CUDA modules only when the adapter is executed."""

    cupy = importlib.import_module("cupy")
    ndimage = importlib.import_module("cupyx.scipy.ndimage")
    return cupy, ndimage


def median_filter(
    data,
    size: int = 5,
    channel_axis: int | None = None,
):
    """Apply VIPP's median filter while keeping input and output on device.

    The active footprint is the trailing two non-channel axes (or the sole
    non-channel axis for 1D data), matching
    :func:`napari_vipp.core.operations.median_filter`.  CuPyX's explicit
    ``reflect`` mode matches SciPy's default boundary convention.
    """

    cupy, ndimage = _cupy_modules()
    array = cupy.asarray(data)
    channel_axis = _validated_channel_axis(channel_axis, array.ndim)
    canonical_size = _odd_size(size, minimum=1)
    footprint = [1] * array.ndim
    for axis in _xy_axes(array.ndim, channel_axis=channel_axis):
        footprint[axis] = canonical_size
    return ndimage.median_filter(
        array,
        size=tuple(footprint),
        mode="reflect",
    )


def _validated_channel_axis(value, ndim: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Median filter channel_axis must be an integer or None.")
    if ndim < 3:
        raise ValueError(
            "Median filter requires at least two spatial dimensions when "
            "channel_axis is set."
        )
    axis = int(value)
    if axis < -ndim or axis >= ndim:
        raise ValueError(
            f"Median filter channel_axis {axis} is out of range for {ndim}D input."
        )
    return axis % ndim


def _odd_size(value: int | float, minimum: int = 1) -> int:
    size = max(int(round(float(value))), minimum)
    if size % 2 == 0:
        size += 1
    return max(size, minimum)


def _xy_axes(ndim: int, *, channel_axis: int | None) -> tuple[int, int]:
    spatial_axes = list(range(ndim))
    if channel_axis is not None:
        spatial_axes.remove(channel_axis)
    if len(spatial_axes) >= 2:
        return spatial_axes[-2], spatial_axes[-1]
    if len(spatial_axes) == 1:
        return spatial_axes[0], spatial_axes[0]
    return 0, 0


__all__ = ["median_filter"]
