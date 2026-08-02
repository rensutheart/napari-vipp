"""Device-resident CuPy implementation of VIPP's Sigma Filter.

CuPy is imported only when :func:`sigma_filter` is called.  The CUDA kernel
keeps every image-sized buffer on the device and mirrors the authoritative CPU
operation's ordered Fiji-compatible arithmetic: float32 samples and squares,
ordered float64 accumulators, population variance clamped to positive zero,
inclusive sigma limits, and float32 half-up restoration for unsigned output.
"""

from __future__ import annotations

import importlib
import math
from functools import cache
from numbers import Integral
from types import ModuleType

import numpy as np

_SUPPORTED_DTYPES = frozenset(
    {np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)}
)
_ROW_TILE_SIZE = 64
_THREADS_PER_BLOCK = 256
_MAXIMUM_BLOCKS = 65_535
_FLOAT32_SQUARE_LIMIT = np.float32(math.sqrt(float(np.finfo(np.float32).max)))
_KERNEL_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--ftz=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only for an explicit accelerator execution."""

    return importlib.import_module("cupy")


@cache
def _validation_kernel(cupy: ModuleType):
    """Compile bounded-storage validation for finite, square-safe float32."""

    return cupy.RawKernel(
        r"""
        extern "C" __global__
        void vipp_sigma_validate_float32(
            const float* values,
            const unsigned long long size,
            const float square_limit,
            unsigned int* status)
        {
            unsigned long long index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long)blockDim.x * gridDim.x;
            for (; index < size; index += stride) {
                const float value = values[index];
                if (!isfinite(value)) {
                    atomicOr(status, 1U);
                } else if (fabsf(value) > square_limit) {
                    atomicOr(status, 2U);
                }
            }
        }
        """,
        "vipp_sigma_validate_float32",
        options=_KERNEL_OPTIONS,
    )


@cache
def _sigma_filter_kernel(cupy: ModuleType, dtype_name: str):
    """Compile one deterministic output-thread kernel for ``dtype_name``."""

    output_declarations = {
        "uint8": "unsigned char* output",
        "uint16": "unsigned short* output",
        "float32": "float* output",
    }
    output_stores = {
        "uint8": r"""
            const float fiji_value = (float)filtered;
            float rounded = floorf(fiji_value + 0.5f);
            rounded = fminf(fmaxf(rounded, 0.0f), 255.0f);
            output[output_index] = (unsigned char)rounded;
        """,
        "uint16": r"""
            const float fiji_value = (float)filtered;
            float rounded = floorf(fiji_value + 0.5f);
            rounded = fminf(fmaxf(rounded, 0.0f), 65535.0f);
            output[output_index] = (unsigned short)rounded;
        """,
        "float32": "output[output_index] = vipp_double_to_float(filtered);",
    }
    if dtype_name not in output_declarations:  # pragma: no cover - private guard
        raise ValueError(f"Unsupported Sigma Filter output dtype {dtype_name!r}.")

    kernel_name = f"vipp_sigma_filter_{dtype_name}"
    source = rf"""
        __device__ __forceinline__ double vipp_float_to_double(
            const float value)
        {{
            const unsigned int bits = __float_as_uint(value);
            const unsigned int magnitude = bits & 0x7fffffffU;
            if (magnitude != 0U && magnitude < 0x00800000U) {{
                double converted = ldexp((double)magnitude, -149);
                return (bits & 0x80000000U) ? -converted : converted;
            }}
            return (double)value;
        }}

        __device__ __forceinline__ float vipp_double_to_float(
            const double value)
        {{
            const double magnitude = fabs(value);
            if (magnitude > 0.0
                && magnitude < 1.17549435082228750796873653722224568e-38) {{
                // CuPy/NVRTC may append --ftz=true even when the caller asks
                // for --ftz=false. Construct subnormal float32 results by bits
                // so a scientifically valid value cannot be silently flushed.
                unsigned int mantissa =
                    __double2uint_rn(ldexp(magnitude, 149));
                if (mantissa > 0x00800000U) {{
                    mantissa = 0x00800000U;
                }}
                const unsigned int sign =
                    signbit(value) ? 0x80000000U : 0U;
                return __uint_as_float(sign | mantissa);
            }}
            return (float)value;
        }}

        extern "C" __global__
        void {kernel_name}(
            const float* source,
            const long long rows,
            const long long columns,
            const long long row_start,
            const long long row_stop,
            const int* offsets,
            const int footprint_count,
            const double sigma_width,
            const unsigned int minimum_count,
            const int outlier_aware,
            {output_declarations[dtype_name]})
        {{
            unsigned long long local_index =
                (unsigned long long)blockDim.x * blockIdx.x + threadIdx.x;
            const unsigned long long stride =
                (unsigned long long)blockDim.x * gridDim.x;
            const unsigned long long tile_size =
                (unsigned long long)(row_stop - row_start)
                * (unsigned long long)columns;

            for (; local_index < tile_size; local_index += stride) {{
                const long long y = row_start
                    + (long long)(local_index / (unsigned long long)columns);
                const long long x =
                    (long long)(local_index % (unsigned long long)columns);
                const long long output_index = y * columns + x;

                double full_sum = 0.0;
                double full_sum_squared = 0.0;
                for (int offset_index = 0;
                     offset_index < footprint_count;
                     ++offset_index) {{
                    long long yy = y + (long long)offsets[2 * offset_index];
                    long long xx = x + (long long)offsets[2 * offset_index + 1];
                    yy = yy < 0 ? 0 : (yy >= rows ? rows - 1 : yy);
                    xx = xx < 0 ? 0 : (xx >= columns ? columns - 1 : xx);
                    const float sample = source[yy * columns + xx];
                    const double sample_value = vipp_float_to_double(sample);
                    const float sample_squared = vipp_double_to_float(
                        sample_value * sample_value);
                    full_sum += sample_value;
                    full_sum_squared += vipp_float_to_double(sample_squared);
                }}

                const double mean = full_sum / (double)footprint_count;
                double variance = full_sum_squared / (double)footprint_count;
                variance -= mean * mean;
                if (variance < 0.0) {{
                    variance = 0.0;
                }}
                const double spread = sigma_width * sqrt(variance);
                const double center =
                    vipp_float_to_double(source[output_index]);
                const double lower = center - spread;
                const double upper = center + spread;

                double selected_sum = 0.0;
                unsigned int selected_count = 0U;
                for (int offset_index = 0;
                     offset_index < footprint_count;
                     ++offset_index) {{
                    long long yy = y + (long long)offsets[2 * offset_index];
                    long long xx = x + (long long)offsets[2 * offset_index + 1];
                    yy = yy < 0 ? 0 : (yy >= rows ? rows - 1 : yy);
                    xx = xx < 0 ? 0 : (xx >= columns ? columns - 1 : xx);
                    const float sample = source[yy * columns + xx];
                    const double sample_value = vipp_float_to_double(sample);
                    if (sample_value >= lower && sample_value <= upper) {{
                        selected_sum += sample_value;
                        ++selected_count;
                    }}
                }}

                double filtered;
                if (selected_count >= minimum_count) {{
                    filtered = selected_sum / (double)selected_count;
                }} else if (outlier_aware) {{
                    filtered = (full_sum - center)
                        / (double)(footprint_count - 1);
                }} else {{
                    filtered = mean;
                }}
                {output_stores[dtype_name]}
            }}
        }}
    """
    return cupy.RawKernel(
        source,
        kernel_name,
        options=_KERNEL_OPTIONS,
    )


def sigma_filter(
    data,
    radius: float = 2.0,
    sigma_width: float = 2.0,
    minimum_pixel_fraction: float = 0.2,
    outlier_aware: bool = True,
    channel_axis: int | None = None,
    progress=None,
):
    """Apply VIPP's slice-wise Sigma Filter to a resident CuPy array.

    Resolved Y/X planes are processed independently for every leading index and
    channel.  Non-contiguous inputs and non-trailing channel axes are
    canonicalized explicitly; the returned array is contiguous in the input's
    original axis order and retains its dtype.
    """

    source_dtype = getattr(data, "dtype", None)
    if source_dtype is not None:
        source_dtype = np.dtype(source_dtype)
        if not source_dtype.isnative:
            raise ValueError(
                "Sigma Filter supports only uint8, uint16, and float32 input; "
                f"received {source_dtype}."
            )

    cupy = _cupy_module()
    array = cupy.asarray(data)
    channel_axis = _validated_channel_axis(channel_axis, array.ndim)
    dtype = np.dtype(array.dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "Sigma Filter supports only uint8, uint16, and float32 input; "
            f"received {dtype}."
        )
    if int(array.size) == 0:
        raise ValueError("Sigma Filter requires non-empty image data.")
    if array.ndim - (channel_axis is not None) < 2:
        raise ValueError("Sigma Filter requires two resolved YX spatial axes.")

    _squared_radius_limit, _extent, offsets = _shared_footprint(radius)
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

    spatial_axes = [axis for axis in range(array.ndim) if axis != channel_axis]
    y_axis, x_axis = spatial_axes[-2:]
    leading_axes = tuple(
        axis for axis in range(array.ndim) if axis not in {y_axis, x_axis}
    )
    permutation = leading_axes + (y_axis, x_axis)
    working = cupy.ascontiguousarray(
        cupy.transpose(array, permutation),
        dtype=cupy.float32,
    )
    if dtype == np.dtype(np.float32):
        _validate_float32_input(working, cupy=cupy)

    leading_shape = tuple(int(size) for size in working.shape[:-2])
    plane_count = math.prod(leading_shape) if leading_shape else 1
    rows, columns = (int(size) for size in working.shape[-2:])
    row_tiles = math.ceil(rows / _ROW_TILE_SIZE)
    total_tiles = plane_count * row_tiles
    minimum_count = math.ceil(len(offsets) * minimum_pixel_fraction)
    device_offsets = cupy.asarray(
        np.asarray(offsets, dtype=np.int32).reshape(-1),
    )
    output = cupy.empty(working.shape, dtype=array.dtype)
    working_planes = working.reshape(plane_count, rows, columns)
    output_planes = output.reshape(plane_count, rows, columns)
    kernel = _sigma_filter_kernel(cupy, dtype.name)

    completed = 0
    if progress is not None:
        progress.report(0, total_tiles, "Sigma Filter rows")
        progress.check_cancelled()
    for plane_index in range(plane_count):
        for row_start in range(0, rows, _ROW_TILE_SIZE):
            if progress is not None:
                progress.check_cancelled()
            row_stop = min(row_start + _ROW_TILE_SIZE, rows)
            tile_pixels = (row_stop - row_start) * columns
            block_count = min(
                math.ceil(tile_pixels / _THREADS_PER_BLOCK),
                _MAXIMUM_BLOCKS,
            )
            kernel(
                (block_count,),
                (_THREADS_PER_BLOCK,),
                (
                    working_planes[plane_index],
                    np.int64(rows),
                    np.int64(columns),
                    np.int64(row_start),
                    np.int64(row_stop),
                    device_offsets,
                    np.int32(len(offsets)),
                    np.float64(sigma_width),
                    np.uint32(minimum_count),
                    np.int32(bool(outlier_aware)),
                    output_planes[plane_index],
                ),
            )
            completed += 1
            if progress is not None:
                cupy.cuda.get_current_stream().synchronize()
                # Delay the last update until the result has been restored to
                # the caller's original, contiguous axis order below.
                if completed < total_tiles:
                    progress.report(completed, total_tiles, "Sigma Filter rows")

    inverse_permutation = tuple(int(axis) for axis in np.argsort(permutation))
    result = cupy.ascontiguousarray(cupy.transpose(output, inverse_permutation))
    if progress is not None:
        cupy.cuda.get_current_stream().synchronize()
        progress.report(total_tiles, total_tiles, "Sigma Filter rows")
    return result


def _validate_float32_input(values, *, cupy: ModuleType) -> None:
    status = cupy.zeros(1, dtype=cupy.uint32)
    block_count = min(
        math.ceil(int(values.size) / _THREADS_PER_BLOCK),
        _MAXIMUM_BLOCKS,
    )
    _validation_kernel(cupy)(
        (block_count,),
        (_THREADS_PER_BLOCK,),
        (
            values,
            np.uint64(values.size),
            np.float32(_FLOAT32_SQUARE_LIMIT),
            status,
        ),
    )
    validation_status = int(status.item())
    if validation_status & 1:
        raise ValueError("Sigma Filter requires finite image intensities.")
    if validation_status & 2:
        raise ValueError(
            "Sigma Filter float32 input magnitude would overflow the "
            "Fiji-compatible float32 square workspace."
        )


def _shared_footprint(radius):
    """Load the authoritative footprint helper without eager CPU dependencies."""

    from napari_vipp.core.operations import sigma_filter_footprint

    return sigma_filter_footprint(radius)


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
    return 0.0 if normalized == 0.0 else normalized


__all__ = ["sigma_filter"]
