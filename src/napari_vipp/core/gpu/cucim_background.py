"""Device-resident CUDA adapters for VIPP rolling-ball operations.

The optional CUDA stack is imported only when an adapter is called.  These
functions reproduce the public CPU operations without exposing provider
primitives as VIPP implementations.  The rolling-ball radius is a runtime
kernel argument so interactive edits do not compile a new CUDA program for
every previously unseen radius.
"""

from __future__ import annotations

import importlib
from functools import cache
from numbers import Integral
from types import ModuleType

import numpy as np

_THREADS_PER_BLOCK = 256
_MAXIMUM_BLOCKS = 65_535
_KERNEL_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)
_FLOAT32_CONVERSION_HELPERS = r"""
    __device__ __forceinline__ double vipp_background_float_to_double(
        const float value)
    {
        const unsigned int bits = __float_as_uint(value);
        const unsigned int magnitude = bits & 0x7fffffffU;
        if (magnitude != 0U && magnitude < 0x00800000U) {
            double converted = ldexp((double)magnitude, -149);
            return (bits & 0x80000000U) ? -converted : converted;
        }
        return (double)value;
    }

    __device__ __forceinline__ float vipp_background_double_to_float(
        const double value)
    {
        const double magnitude = fabs(value);
        if (magnitude > 0.0
            && magnitude < 1.17549435082228750796873653722224568e-38) {
            // NVRTC/CUDA float arithmetic can flush subnormal values. Build
            // those results from their IEEE-754 bits so the CPU contract is
            // preserved without transferring the image to the host.
            unsigned int mantissa =
                __double2uint_rn(ldexp(magnitude, 149));
            if (mantissa > 0x00800000U) {
                mantissa = 0x00800000U;
            }
            const unsigned int sign =
                signbit(value) ? 0x80000000U : 0U;
            return __uint_as_float(sign | mantissa);
        }
        return (float)value;
    }
"""


@cache
def _gpu_modules() -> tuple[ModuleType, ModuleType]:
    """Load optional providers only for explicit accelerator execution."""

    cupy = importlib.import_module("cupy")
    ndimage = importlib.import_module("cupyx.scipy.ndimage")
    return cupy, ndimage


@cache
def _float32_uniform_filter_axis_kernel(cupy):
    """Return SciPy's rolling size-three float32 smoothing kernel.

    One thread owns one complete axis line.  That deliberately mirrors
    ``NI_UniformFilter1D``: initialize the first reflected window, then add
    the entering sample before subtracting the leaving sample.  A thread per
    output value would perform three independent additions and can differ by
    an ULP from SciPy even when all values are ordinary finite floats.
    """

    source = rf"""
        {_FLOAT32_CONVERSION_HELPERS}

        extern "C" __global__
        void vipp_background_uniform_filter_axis_float32_v2(
            const float* input_values,
            float* output_values,
            const unsigned long long line_count,
            const long long axis_length,
            const long long inner_stride)
        {{
            unsigned long long line =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; line < line_count; line += stride) {{
                const long long outer = (long long)line / inner_stride;
                const long long inner = (long long)line % inner_stride;
                const long long base =
                    outer * axis_length * inner_stride + inner;
                const long long second = axis_length > 1 ? 1 : 0;
                double total =
                    2.0 * vipp_background_float_to_double(input_values[base])
                    + vipp_background_float_to_double(
                        input_values[base + second * inner_stride]);
                output_values[base] =
                    vipp_background_double_to_float(total / 3.0);
                for (long long coordinate = 1;
                     coordinate < axis_length;
                     ++coordinate) {{
                    const long long entering =
                        coordinate + 1 < axis_length
                        ? coordinate + 1
                        : axis_length - 1;
                    const long long leaving = coordinate > 1
                        ? coordinate - 2
                        : 0;
                    total += vipp_background_float_to_double(
                        input_values[base + entering * inner_stride]);
                    total -= vipp_background_float_to_double(
                        input_values[base + leaving * inner_stride]);
                    output_values[base + coordinate * inner_stride] =
                        vipp_background_double_to_float(total / 3.0);
                }}
            }}
        }}
    """
    return cupy.RawKernel(
        source,
        "vipp_background_uniform_filter_axis_float32_v2",
        options=_KERNEL_OPTIONS,
    )


@cache
def _float32_light_transform_kernel(cupy):
    """Return the subnormal-preserving light-background affine transform."""

    source = rf"""
        {_FLOAT32_CONVERSION_HELPERS}

        extern "C" __global__
        void vipp_background_light_transform_float32_v1(
            const float* input_values,
            float* output_values,
            const unsigned long long value_count,
            const float* low_value,
            const float* high_value)
        {{
            // NumPy's weak-scalar expression rounds the offset to float32
            // before subtracting the array.  Preserve that intermediate
            // rounding while using software conversion for subnormals.
            const float offset_value = vipp_background_double_to_float(
                vipp_background_float_to_double(low_value[0])
                + vipp_background_float_to_double(high_value[0]));
            const double offset =
                vipp_background_float_to_double(offset_value);
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < value_count; index += stride) {{
                output_values[index] = vipp_background_double_to_float(
                    offset
                    - vipp_background_float_to_double(input_values[index]));
            }}
        }}
    """
    return cupy.RawKernel(
        source,
        "vipp_background_light_transform_float32_v1",
        options=_KERNEL_OPTIONS,
    )


@cache
def _float32_subtract_kernel(cupy):
    """Return exact float32 subtraction without subnormal flush-to-zero."""

    source = rf"""
        {_FLOAT32_CONVERSION_HELPERS}

        extern "C" __global__
        void vipp_background_subtract_float32_v1(
            const float* values,
            const float* background,
            float* output_values,
            const unsigned long long value_count,
            const int light_background,
            const int clip_negative)
        {{
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < value_count; index += stride) {{
                const double value = light_background
                    ? vipp_background_float_to_double(background[index])
                        - vipp_background_float_to_double(values[index])
                    : vipp_background_float_to_double(values[index])
                        - vipp_background_float_to_double(background[index]);
                const double clipped = clip_negative && value <= 0.0
                    ? 0.0
                    : value;
                output_values[index] =
                    vipp_background_double_to_float(clipped);
            }}
        }}
    """
    return cupy.RawKernel(
        source,
        "vipp_background_subtract_float32_v1",
        options=_KERNEL_OPTIONS,
    )


@cache
def _float32_zero_bound_tie_kernel(cupy):
    """Return NumPy-compatible signed-zero tie correction for min/max."""

    source = r"""
        extern "C" __global__
        void vipp_background_float32_zero_bound_tie_v1(
            const float* values,
            const unsigned long long value_count,
            float* low_value,
            float* high_value)
        {
            if (blockIdx.x != 0 || threadIdx.x != 0) {
                return;
            }
            const unsigned int low_bits = __float_as_uint(low_value[0]);
            const unsigned int high_bits = __float_as_uint(high_value[0]);
            if ((low_bits & 0x7fffffffU) != 0U
                || (high_bits & 0x7fffffffU) != 0U) {
                return;
            }
            for (unsigned long long index = value_count;
                 index > 0;
                 --index) {
                const float value = values[index - 1];
                if (isfinite(value)) {
                    const unsigned int bits = __float_as_uint(value);
                    if ((bits & 0x7fffffffU) == 0U) {
                        low_value[0] = value;
                        high_value[0] = value;
                    }
                    return;
                }
            }
        }
    """
    return cupy.RawKernel(
        source,
        "vipp_background_float32_zero_bound_tie_v1",
        options=_KERNEL_OPTIONS,
    )


@cache
def _dynamic_rolling_ball_kernel(cupy, spatial_ndim: int, dtype_name: str):
    """Return one radius-independent erosion kernel for a spatial rank.

    cuCIM constructs a radius-sized footprint and delegates to CuPyX's
    generated grey-erosion kernel.  CuPyX embeds the footprint shape in the
    CUDA source, so each new interactive radius incurs another compilation.
    Here the radius and array extents are scalar runtime inputs.  The source
    varies only with the reviewed spatial rank (one, two, or three) and the
    float32/float64 workspace dtype.

    The inner calculation is the same non-flat spherical erosion used by
    cuCIM: ``min(image - (sqrt(radius**2 - distance**2) - radius))`` over
    in-bounds points inside the ball.  Omitting out-of-bounds candidates is
    equivalent to its constant ``+inf`` boundary mode.
    """

    ndim = int(spatial_ndim)
    if ndim not in {1, 2, 3}:
        raise ValueError("Dynamic rolling-ball erosion supports one to three axes.")
    ctype_by_dtype = {"float32": "float", "float64": "double"}
    try:
        ctype = ctype_by_dtype[dtype_name]
    except KeyError as exc:  # pragma: no cover - private invariant
        raise TypeError(
            f"Unsupported rolling-ball workspace dtype {dtype_name!r}."
        ) from exc
    square_root = "sqrtf" if dtype_name == "float32" else "sqrt"

    coordinate_lines = ["long long remainder = (long long)index;"]
    for axis in range(ndim - 1, 0, -1):
        coordinate_lines.extend(
            (
                f"long long coordinate_{axis} = remainder % extent_{axis};",
                f"remainder /= extent_{axis};",
            )
        )
    coordinate_lines.append("long long coordinate_0 = remainder;")

    distance = " + ".join(f"distance_{axis}" for axis in range(ndim))
    source_index_lines = ["long long source_index = position_0;"]
    for axis in range(1, ndim):
        source_index_lines.append(
            f"source_index = source_index * extent_{axis} + position_{axis};"
        )
    body = "\n".join(
        (
            f"{ctype} distance_square = {distance};",
            "if (distance_square > radius_square) {",
            "    continue;",
            "}",
            *source_index_lines,
            f"{ctype} structure = {square_root}(radius_square - distance_square) "
            "- radius_value;",
            f"{ctype} candidate = input_values[source_index] - structure;",
            "if (candidate < best) {",
            "    best = candidate;",
            "}",
        )
    )
    for axis in range(ndim - 1, -1, -1):
        body = "\n".join(
            (
                f"for (int delta_{axis} = -radius; delta_{axis} <= radius; "
                f"++delta_{axis}) {{",
                f"    long long position_{axis} = coordinate_{axis} + delta_{axis};",
                f"    if (position_{axis} < 0 || position_{axis} >= extent_{axis}) {{",
                "        continue;",
                "    }",
                f"    {ctype} offset_{axis} = ({ctype})delta_{axis};",
                f"    {ctype} distance_{axis} = offset_{axis} * offset_{axis};",
                body,
                "}",
            )
        )

    kernel_name = f"vipp_dynamic_rolling_ball_{ndim}d_{dtype_name}_v1"
    store_line = (
        "output_values[index] = "
        "((__float_as_uint(best) & 0x7fffffffU) == 0U) "
        "? __uint_as_float(0U) : best;"
        if dtype_name == "float32"
        else "output_values[index] = best == 0.0 ? 0.0 : best;"
    )
    operation = "\n".join(
        (
            'extern "C" __global__',
            f"void {kernel_name}(",
            f"    const {ctype}* input_values,",
            f"    {ctype}* output_values,",
            "    const unsigned long long value_count,",
            *(f"    const long long extent_{axis}," for axis in range(ndim)),
            "    const int radius)",
            "{",
            "    unsigned long long index =",
            "        (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;",
            "    const unsigned long long stride =",
            "        (unsigned long long)blockDim.x * gridDim.x;",
            "    for (; index < value_count; index += stride) {",
            *(f"        {line}" for line in coordinate_lines),
            f"        {ctype} radius_value = ({ctype})radius;",
            f"        {ctype} radius_square = radius_value * radius_value;",
            f"        {ctype} best = input_values[index];",
            *(f"        {line}" for line in body.splitlines()),
            f"        {store_line}",
            "    }",
            "}",
        )
    )
    return cupy.RawKernel(
        operation,
        kernel_name,
        options=_KERNEL_OPTIONS,
    )


def _dynamic_rolling_ball(values, radius_pixels: int, *, cupy):
    """Apply exact spherical erosion with radius supplied at launch time."""

    contiguous = cupy.ascontiguousarray(values)
    dtype_name = np.dtype(contiguous.dtype).name
    kernel = _dynamic_rolling_ball_kernel(cupy, contiguous.ndim, dtype_name)
    extents = tuple(np.int64(extent) for extent in contiguous.shape)
    output = cupy.empty_like(contiguous)
    blocks = min(
        max((int(contiguous.size) + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1),
        _MAXIMUM_BLOCKS,
    )
    kernel(
        (blocks,),
        (_THREADS_PER_BLOCK,),
        (
            contiguous,
            output,
            np.uint64(contiguous.size),
            *extents,
            np.int32(radius_pixels),
        ),
    )
    return output


def _launch_1d_kernel(kernel, value_count: int, arguments) -> None:
    """Launch a bounded grid for one output thread per logical value."""

    blocks = min(
        max((int(value_count) + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, 1),
        _MAXIMUM_BLOCKS,
    )
    kernel(
        (blocks,),
        (_THREADS_PER_BLOCK,),
        arguments,
    )


def _float32_light_transform(values, low, high, *, cupy):
    """Compute ``low + high - values`` without flushing subnormals."""

    contiguous = cupy.ascontiguousarray(values)
    output = cupy.empty_like(contiguous)
    _launch_1d_kernel(
        _float32_light_transform_kernel(cupy),
        int(contiguous.size),
        (
            contiguous,
            output,
            np.uint64(contiguous.size),
            low,
            high,
        ),
    )
    return output


def _subtract_float32(
    values,
    background,
    *,
    light_background: bool,
    clip_negative: bool,
    cupy,
):
    """Apply authoritative float32 subtraction with exact tiny-value handling."""

    contiguous_values = cupy.ascontiguousarray(values)
    contiguous_background = cupy.ascontiguousarray(background)
    output = cupy.empty_like(contiguous_values)
    _launch_1d_kernel(
        _float32_subtract_kernel(cupy),
        int(contiguous_values.size),
        (
            contiguous_values,
            contiguous_background,
            output,
            np.uint64(contiguous_values.size),
            np.int32(bool(light_background)),
            np.int32(bool(clip_negative)),
        ),
    )
    return output


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

    cupy, ndimage = _gpu_modules()
    _require_native_endian_input(data, operation="Rolling-ball background")
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

    cupy, ndimage = _gpu_modules()
    _require_native_endian_input(data, operation="Subtract background")
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
    )
    values = array.astype(background.dtype, copy=False)
    if array.dtype == cupy.float32:
        corrected = _subtract_float32(
            values,
            background,
            light_background=bool(light_background),
            clip_negative=bool(clip_negative),
            cupy=cupy,
        )
    else:
        corrected = (
            background - values if bool(light_background) else values - background
        )
        if bool(clip_negative):
            corrected = cupy.maximum(corrected, 0)
    return _restore_numeric_dtype(corrected, array, cupy=cupy)


def _require_native_endian_input(data, *, operation: str) -> None:
    """Fail closed before CuPy silently normalizes a foreign byte order."""

    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None and not np.dtype(source_dtype).isnative:
        raise ValueError(
            f"{operation} GPU execution requires native-endian input data."
        )


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
        if output_dtype == cupy.float32:
            inverted = _float32_light_transform(safe, low, high, cupy=cupy)
            background = _float32_light_transform(
                _dynamic_rolling_ball(
                    inverted,
                    radius_pixels,
                    cupy=cupy,
                ),
                low,
                high,
                cupy=cupy,
            )
        else:
            offset = low + high
            inverted = offset - safe
            background = offset - _dynamic_rolling_ball(
                inverted,
                radius_pixels,
                cupy=cupy,
            )
    else:
        background = _dynamic_rolling_ball(
            safe,
            radius_pixels,
            cupy=cupy,
        )
    return background.astype(output_dtype, copy=False)


def _uniform_filter_size_three(
    values,
    *,
    output_dtype,
    cupy,
    ndimage,
):
    """Match SciPy's double accumulator and per-axis public dtype cast.

    CuPyX's multidimensional ``uniform_filter`` accumulates float32 inputs in
    float32 and differs from SciPy by one or two ULPs.  SciPy uses a double
    line accumulator, casts to the requested dtype after each axis, and feeds
    that rounded intermediate into the next axis.  This separable device path
    preserves those semantics without transferring image data to the host.
    """

    if output_dtype == cupy.float32:
        result = cupy.ascontiguousarray(values)
        kernel = _float32_uniform_filter_axis_kernel(cupy)
        for axis in range(values.ndim):
            filtered = cupy.empty_like(result)
            inner_stride = int(np.prod(result.shape[axis + 1 :], dtype=np.int64))
            line_count = int(result.size) // int(result.shape[axis])
            _launch_1d_kernel(
                kernel,
                line_count,
                (
                    result,
                    filtered,
                    np.uint64(line_count),
                    np.int64(result.shape[axis]),
                    np.int64(inner_stride),
                ),
            )
            result = filtered
        return result

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
    if values.dtype == cupy.float32:
        # NumPy keeps the last equal zero lane, while CuPy may select another
        # reduction lane.  The sign is observable in light-background output.
        # This fixed kernel returns immediately for ordinary nonzero bounds and
        # only scans backward when every finite value compares equal to zero.
        contiguous = cupy.ascontiguousarray(values)
        _float32_zero_bound_tie_kernel(cupy)(
            (1,),
            (1,),
            (
                contiguous,
                np.uint64(contiguous.size),
                low_candidate,
                high_candidate,
            ),
        )
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

    # Keep this boundary even without a reporter: CuPy kernels and the
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
