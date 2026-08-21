"""Lazy, device-resident CUDA adapters for VIPP Gaussian operations.

The optional CUDA stack is imported only when an adapter is executed.  Inputs,
outputs, and intermediate image buffers stay in the CuPy array domain.
"""

from __future__ import annotations

import importlib
import math
from functools import cache
from numbers import Integral
from types import ModuleType

import numpy as np

_THREADS_PER_BLOCK = 256
_MAXIMUM_BLOCKS = 65_535


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only for an explicit accelerator execution."""

    return importlib.import_module("cupy")


@cache
def _cupyx_ndimage_module() -> ModuleType:
    """Load CuPyX only for a non-public or integral execution path."""

    return importlib.import_module("cupyx.scipy.ndimage")


@cache
def _float32_gaussian_axis_kernel():
    """Return one radius-independent float32 correlation kernel.

    CuPyX specializes its generated correlation kernel on the Gaussian weight
    shape.  Interactive sigma edits therefore compile a new CUDA kernel for
    every previously unseen radius.  This kernel keeps the radius and weights
    as runtime inputs, so one compilation serves every reviewed sigma value.
    The flattened contiguous input also keeps the generated CUDA signature
    independent of image rank.
    """

    cupy = _cupy_module()
    kernel_name = "vipp_dynamic_float32_gaussian_axis_v2"
    return cupy.RawKernel(
        rf"""
        extern "C" __global__
        void {kernel_name}(
            const float* input_values,
            const float* weights,
            float* output_values,
            const unsigned long long value_count,
            const long long axis_length,
            const long long inner_stride,
            const int radius)
        {{
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < value_count; index += stride) {{
                const long long coordinate =
                    ((long long)index / inner_stride) % axis_length;
                const long long period = 2 * axis_length;
                float total = 0.0f;
                for (int offset = -radius; offset <= radius; ++offset) {{
                    long long reflected = coordinate + offset;
                    reflected %= period;
                    if (reflected < 0) {{
                        reflected += period;
                    }}
                    if (reflected >= axis_length) {{
                        reflected = period - 1 - reflected;
                    }}
                    const long long source_index =
                        (long long)index
                        + (reflected - coordinate) * inner_stride;
                    total += input_values[source_index]
                        * weights[offset + radius];
                }}
                output_values[index] = total;
            }}
        }}
        """,
        kernel_name,
    )


def gaussian_blur(
    data,
    sigma: float = 1.0,
    channel_axis: int | None = None,
):
    """Apply VIPP's slice-wise Y/X Gaussian contract on a CuPy array."""

    cupy = _cupy_module()
    array = cupy.asarray(data)
    if array.dtype == cupy.bool_:
        array = array.astype(cupy.float32)
    channel_axis = _validated_channel_axis(
        channel_axis,
        array.ndim,
        operation="Gaussian blur",
    )
    sigma = _finite_nonnegative_sigma(sigma, "Gaussian blur sigma")
    if sigma == 0.0:
        return array.copy()

    sigma_by_axis = [0.0] * array.ndim
    for axis in _xy_axes(array.ndim, channel_axis=channel_axis):
        sigma_by_axis[axis] = sigma
    return _gaussian_filter(
        array,
        tuple(sigma_by_axis),
        cupy=cupy,
    )


def gaussian_blur_3d(
    data,
    sigma_z: float = 2.0,
    sigma_y: float = 2.0,
    sigma_x: float = 2.0,
    lock_xy: bool = True,
    channel_axis: int | None = None,
):
    """Apply VIPP's resolved trailing Z/Y/X Gaussian contract on device."""

    del lock_xy  # UI convenience; the three sigma values are authoritative.
    cupy = _cupy_module()
    array = cupy.asarray(data)
    if array.dtype == cupy.bool_:
        array = array.astype(cupy.float32)
    channel_axis = _validated_channel_axis(
        channel_axis,
        array.ndim,
        operation="Gaussian blur 3D",
    )
    spatial_axes = _spatial_axes(array.ndim, channel_axis=channel_axis)
    if not spatial_axes:
        return array.copy()

    values = (
        _finite_nonnegative_sigma(sigma_z, "Gaussian blur 3D sigma_z"),
        _finite_nonnegative_sigma(sigma_y, "Gaussian blur 3D sigma_y"),
        _finite_nonnegative_sigma(sigma_x, "Gaussian blur 3D sigma_x"),
    )
    sigma_by_axis = [0.0] * array.ndim
    active_axes = spatial_axes[-3:]
    active_values = values[-len(active_axes) :]
    for axis, value in zip(active_axes, active_values, strict=True):
        sigma_by_axis[axis] = value
    if not any(sigma_by_axis):
        return array.copy()
    return _gaussian_filter(
        array,
        tuple(sigma_by_axis),
        cupy=cupy,
    )


def _gaussian_filter(array, sigmas, *, cupy: ModuleType):
    """Use SciPy-compatible integer intermediates without leaving device.

    CuPyX may accumulate integral inputs at reduced precision, which produces
    sparse one-unit disagreements for uint16.  SciPy uses double-precision
    correlation followed by a cast back to the public integer dtype after each
    active axis.  Reproducing that separable sequence on device restores the
    authoritative integer behavior while retaining CuPy arrays throughout.
    Reviewed float32 inputs use VIPP's radius-independent separable kernel so
    interactive parameter changes do not repeatedly compile radius-specialized
    CuPyX kernels. Other floating inputs retain CuPyX's implementation and are
    outside the public GPU contract.
    """

    if array.dtype == cupy.float32:
        return _dynamic_float32_gaussian_filter(array, sigmas, cupy=cupy)

    ndimage = _cupyx_ndimage_module()
    if not cupy.issubdtype(array.dtype, cupy.integer):
        return ndimage.gaussian_filter(
            array,
            sigma=sigmas,
            mode="reflect",
        )

    public_dtype = array.dtype
    result = array
    filtered_any_axis = False
    for axis, sigma in enumerate(sigmas):
        # SciPy/CuPyX omit axes whose sigma is effectively zero.  The adapter's
        # public parameters are finite, so no other special cases are needed.
        if sigma <= 1e-15:
            continue
        filtered_any_axis = True
        work = result.astype(cupy.float64)
        filtered = ndimage.gaussian_filter1d(
            work,
            sigma=sigma,
            axis=axis,
            output=cupy.float64,
            mode="reflect",
        )
        result = filtered.astype(public_dtype)
    return result if filtered_any_axis else array.copy()


def _dynamic_float32_gaussian_filter(array, sigmas, *, cupy: ModuleType):
    """Apply separable reflect-mode Gaussian correlation with dynamic radii."""

    result = cupy.ascontiguousarray(array)
    kernel = _float32_gaussian_axis_kernel()
    filtered_any_axis = False
    for axis, sigma in enumerate(sigmas):
        if sigma <= 1e-15:
            continue
        radius = int(4.0 * float(sigma) + 0.5)
        # CuPyX treats a rounded zero-radius axis as an exact copy.  Avoid
        # arithmetic here so signed zeros and NaN payload bits are preserved.
        if radius <= 0:
            continue
        filtered_any_axis = True
        weights = cupy.asarray(
            _gaussian_weights(float(sigma), radius),
            dtype=cupy.float32,
        )
        shape = result.shape
        inner_stride = int(np.prod(shape[axis + 1 :], dtype=np.int64))
        flat = result.reshape(-1)
        filtered = cupy.empty_like(flat)
        if flat.size:
            blocks = min(
                max(
                    (int(flat.size) + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK,
                    1,
                ),
                _MAXIMUM_BLOCKS,
            )
            kernel(
                (blocks,),
                (_THREADS_PER_BLOCK,),
                (
                    flat,
                    weights,
                    filtered,
                    np.uint64(flat.size),
                    np.int64(shape[axis]),
                    np.int64(inner_stride),
                    np.int32(radius),
                ),
            )
        result = filtered.reshape(shape)
    return result if filtered_any_axis else array.copy()


def _gaussian_weights(sigma: float, radius: int) -> np.ndarray:
    """Return CuPyX-compatible normalized float32 Gaussian weights."""

    positions = np.arange(-radius, radius + 1, dtype=np.float64)
    weights = np.exp(-0.5 / (sigma * sigma) * positions * positions)
    weights /= weights.sum()
    return weights.astype(np.float32)


def _finite_nonnegative_sigma(value, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    # The authoritative CPU operations clamp negative sigma values to zero.
    return max(parsed, 0.0)


def _validated_channel_axis(
    value,
    ndim: int,
    *,
    operation: str,
) -> int | None:
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


def _xy_axes(ndim: int, *, channel_axis: int | None) -> tuple[int, int]:
    spatial_axes = _spatial_axes(ndim, channel_axis=channel_axis)
    if len(spatial_axes) >= 2:
        return spatial_axes[-2], spatial_axes[-1]
    if len(spatial_axes) == 1:
        return spatial_axes[0], spatial_axes[0]
    return 0, 0


def _spatial_axes(ndim: int, *, channel_axis: int | None) -> list[int]:
    axes = list(range(ndim))
    if channel_axis is not None:
        axes.remove(channel_axis)
    return axes


__all__ = ["gaussian_blur", "gaussian_blur_3d"]
