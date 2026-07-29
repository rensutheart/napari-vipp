"""Device-resident cuCIM adapters for VIPP rolling-ball operations.

The optional CUDA stack is imported only when an adapter is called.  These
functions reproduce the public CPU operations around cuCIM's primitive; they
do not expose the raw primitive as a VIPP implementation.
"""

from __future__ import annotations

import importlib
from functools import cache
from numbers import Integral
from types import ModuleType

import numpy as np


@cache
def _gpu_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Load optional providers only for explicit accelerator execution."""

    cupy = importlib.import_module("cupy")
    ndimage = importlib.import_module("cupyx.scipy.ndimage")
    restoration = importlib.import_module("cucim.skimage.restoration")
    return cupy, ndimage, restoration


def rolling_ball_background(
    data,
    radius: float = 50.0,
    light_background: bool = False,
    disable_smoothing: bool = False,
    spatial_mode: str = "2D YX",
    resolved_spatial_ndim: int | None = None,
    progress=None,
    channel_axis: int | None = None,
):
    """Estimate VIPP's rolling-ball background in the common CuPy domain."""

    cupy, ndimage, restoration = _gpu_modules()
    array = cupy.asarray(data)
    channel_axis = _validated_channel_axis(
        channel_axis,
        array.ndim,
        operation="Rolling-ball background",
    )
    if array.dtype == cupy.bool_:
        return cupy.zeros_like(array)
    background = _estimate_background(
        array,
        radius=radius,
        light_background=bool(light_background),
        disable_smoothing=bool(disable_smoothing),
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        channel_axis=channel_axis,
        progress=progress,
        cupy=cupy,
        ndimage=ndimage,
        restoration=restoration,
    )
    return _restore_numeric_dtype(background, array, cupy=cupy)


def subtract_background(
    data,
    radius: float = 50.0,
    light_background: bool = False,
    disable_smoothing: bool = False,
    clip_negative: bool = True,
    spatial_mode: str = "2D YX",
    resolved_spatial_ndim: int | None = None,
    progress=None,
    channel_axis: int | None = None,
):
    """Subtract VIPP's rolling-ball background without leaving the GPU."""

    cupy, ndimage, restoration = _gpu_modules()
    array = cupy.asarray(data)
    channel_axis = _validated_channel_axis(
        channel_axis,
        array.ndim,
        operation="Subtract background",
    )
    if array.dtype == cupy.bool_:
        return array.copy()
    background = _estimate_background(
        array,
        radius=radius,
        light_background=bool(light_background),
        disable_smoothing=bool(disable_smoothing),
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        channel_axis=channel_axis,
        progress=progress,
        cupy=cupy,
        ndimage=ndimage,
        restoration=restoration,
    )
    values = array.astype(background.dtype, copy=False)
    corrected = background - values if bool(light_background) else values - background
    if bool(clip_negative):
        corrected = cupy.maximum(corrected, 0)
    return _restore_numeric_dtype(corrected, array, cupy=cupy)


def _estimate_background(
    array,
    *,
    radius,
    light_background: bool,
    disable_smoothing: bool,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
    channel_axis: int | None,
    progress,
    cupy,
    ndimage,
    restoration,
):
    if array.ndim == 0:
        output_dtype = cupy.float64 if array.dtype == cupy.float64 else cupy.float32
        return array.astype(output_dtype, copy=True)

    radius_pixels = max(int(round(float(radius))), 1)
    spatial_ndim = _resolved_spatial_ndim(
        array.ndim,
        spatial_mode,
        resolved_spatial_ndim,
    )
    output_dtype = cupy.float64 if array.dtype == cupy.float64 else cupy.float32

    def estimate(block):
        if progress is not None:
            progress.check_cancelled()
        return _background_block(
            block,
            radius_pixels=radius_pixels,
            light_background=light_background,
            disable_smoothing=disable_smoothing,
            output_dtype=output_dtype,
            cupy=cupy,
            ndimage=ndimage,
            restoration=restoration,
        )

    if channel_axis is not None:
        channels_last = cupy.moveaxis(array, channel_axis, -1)
        block_ndim = min(spatial_ndim, max(channels_last.ndim - 1, 1))
        block_count = _spatial_block_count(
            channels_last[..., 0].shape,
            block_ndim,
        )
        total = max(block_count * int(channels_last.shape[-1]), 1)
        channels = [
            _apply_spatial_blocks(
                channels_last[..., channel],
                block_ndim,
                estimate,
                dtype=output_dtype,
                cupy=cupy,
                progress=progress,
                progress_start=channel * block_count,
                progress_total=total,
                progress_message=f"Rolling-ball channel {channel + 1}",
            )
            for channel in range(channels_last.shape[-1])
        ]
        stacked = cupy.stack(channels, axis=-1).astype(output_dtype, copy=False)
        return cupy.moveaxis(stacked, -1, channel_axis)

    return _apply_spatial_blocks(
        array,
        spatial_ndim,
        estimate,
        dtype=output_dtype,
        cupy=cupy,
        progress=progress,
        progress_message="Rolling-ball background",
    )


def _background_block(
    block,
    *,
    radius_pixels: int,
    light_background: bool,
    disable_smoothing: bool,
    output_dtype,
    cupy,
    ndimage,
    restoration,
):
    values = block.astype(output_dtype, copy=True)
    if values.size == 0:
        return values

    low, high = _finite_bounds(values, cupy=cupy)
    finite = cupy.isfinite(values)
    safe = cupy.where(
        finite,
        values,
        cupy.where(cupy.isposinf(values), high, low),
    ).astype(output_dtype, copy=False)

    if not disable_smoothing and safe.ndim > 0:
        safe = _uniform_filter_size_three(
            safe,
            output_dtype=output_dtype,
            cupy=cupy,
            ndimage=ndimage,
        )

    if light_background:
        low, high = _finite_bounds(
            safe,
            cupy=cupy,
            default_low=low,
            default_high=high,
        )
        offset = low + high
        inverted = offset - safe
        background = offset - restoration.rolling_ball(
            inverted,
            radius=radius_pixels,
        )
    else:
        background = restoration.rolling_ball(safe, radius=radius_pixels)
    return background.astype(output_dtype, copy=False)


def _uniform_filter_size_three(values, *, output_dtype, cupy, ndimage):
    """Match SciPy's double accumulator and per-axis public dtype cast.

    CuPyX's multidimensional ``uniform_filter`` accumulates float32 inputs in
    float32 and differs from SciPy by one or two ULPs.  SciPy uses a double
    line accumulator, casts to the requested dtype after each axis, and feeds
    that rounded intermediate into the next axis.  This separable device path
    preserves those semantics without transferring image data to the host.
    """

    result = values
    for axis in range(values.ndim):
        work = result.astype(cupy.float64)
        filtered = ndimage.uniform_filter1d(
            work,
            size=3,
            axis=axis,
            output=cupy.float64,
            mode="nearest",
        )
        result = filtered.astype(output_dtype)
    return result


def _finite_bounds(
    values,
    *,
    cupy,
    default_low=None,
    default_high=None,
):
    """Return finite extrema as device scalars, including all-invalid input."""

    finite = cupy.isfinite(values)
    any_finite = cupy.any(finite)
    low_candidate = cupy.min(cupy.where(finite, values, cupy.inf))
    high_candidate = cupy.max(cupy.where(finite, values, -cupy.inf))
    zero = values.dtype.type(0)
    low_fallback = zero if default_low is None else default_low
    high_fallback = zero if default_high is None else default_high
    low = cupy.where(any_finite, low_candidate, low_fallback)
    high = cupy.where(any_finite, high_candidate, high_fallback)
    return low, high


def _apply_spatial_blocks(
    array,
    spatial_ndim: int,
    function,
    *,
    dtype,
    cupy,
    progress=None,
    progress_start: int = 0,
    progress_total: int | None = None,
    progress_message: str = "",
):
    spatial_ndim = min(max(int(spatial_ndim), 1), max(array.ndim, 1))
    block_count = _spatial_block_count(array.shape, spatial_ndim)
    denominator = int(progress_total or block_count)
    completed = 0
    if progress is not None:
        progress.report(progress_start, denominator, progress_message)
    if array.ndim <= spatial_ndim:
        result = function(array).astype(dtype, copy=False)
        _synchronize_completed_block(cupy, progress)
        if progress is not None:
            progress.report(progress_start + 1, denominator, progress_message)
        return result

    result = cupy.empty(array.shape, dtype=dtype)
    leading_shape = array.shape[: array.ndim - spatial_ndim]
    for index in np.ndindex(leading_shape):
        if progress is not None:
            progress.check_cancelled()
        result[index] = function(array[index])
        _synchronize_completed_block(cupy, progress)
        completed += 1
        if progress is not None:
            progress.report(
                progress_start + completed,
                denominator,
                progress_message,
            )
    return result


def _synchronize_completed_block(cupy, progress) -> None:
    """Make a block's output assignment real before exposing its completion."""

    # Keep this boundary even without a reporter: cuCIM/CuPy kernels and the
    # final assignment are asynchronous, and every block must surface failures
    # before the operation advances to the next block.
    cupy.cuda.get_current_stream().synchronize()
    if progress is not None:
        progress.check_cancelled()


def _spatial_block_count(shape: tuple[int, ...], spatial_ndim: int) -> int:
    ndim = len(shape)
    spatial_ndim = min(max(int(spatial_ndim), 1), max(ndim, 1))
    if ndim <= spatial_ndim:
        return 1
    return int(np.prod(shape[: ndim - spatial_ndim], dtype=np.int64))


def _resolved_spatial_ndim(
    ndim: int,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
) -> int:
    requested = _spatial_mode_dimension(spatial_mode)
    if requested is None and resolved_spatial_ndim is not None:
        requested = _validated_resolved_spatial_ndim(resolved_spatial_ndim)
    if requested is None:
        if ndim > 2:
            raise ValueError(
                "Auto from axes requires explicit axis semantics. Supply "
                "resolved_spatial_ndim or select an explicit 2D/3D mode."
            )
        requested = max(ndim, 1)
    if requested > max(ndim, 1):
        raise ValueError(
            f"{requested}D spatial processing cannot be applied to a {ndim}D array."
        )
    return requested


def _spatial_mode_dimension(spatial_mode: str) -> int | None:
    mode = str(spatial_mode).strip().casefold()
    dimensions = {
        "auto from axes": None,
        "2d yx": 2,
        "2d per xy slice (advanced)": 2,
        "3d zyx": 3,
        "3d zyx volume": 3,
    }
    if mode not in dimensions:
        raise ValueError(
            "Spatial mode must be Auto from axes, 2D YX, "
            "2D per XY slice (advanced), 3D ZYX, or 3D ZYX volume."
        )
    return dimensions[mode]


def _validated_resolved_spatial_ndim(value) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("resolved_spatial_ndim must be an integer from 1 to 3.")
    resolved = int(value)
    if resolved not in {1, 2, 3}:
        raise ValueError("resolved_spatial_ndim must be an integer from 1 to 3.")
    return resolved


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


def _restore_numeric_dtype(values, original, *, cupy):
    dtype = np.dtype(original.dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        safe = cupy.nan_to_num(
            values,
            nan=0.0,
            posinf=float(info.max),
            neginf=float(info.min),
        )
        rounded = cupy.rint(safe)
        return cupy.clip(rounded, info.min, info.max).astype(original.dtype)
    return values.astype(original.dtype, copy=False)


__all__ = ["rolling_ball_background", "subtract_background"]
