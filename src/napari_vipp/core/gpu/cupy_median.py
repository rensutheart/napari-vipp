"""Lazy, device-resident median filter with a runtime-sized footprint.

CuPyX generates a different CUDA program for every median footprint shape.
That is scientifically correct, but it makes interactive edits pause while
NVRTC compiles each previously unseen size.  This adapter instead selects the
median with a radix histogram whose CUDA source depends on dtype, never on the
authored filter size.
"""

from __future__ import annotations

import importlib
from functools import cache
from numbers import Integral
from types import ModuleType

import numpy as np

_SUPPORTED_DTYPES = frozenset(
    {np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)}
)
_WARPS_PER_BLOCK = 4
_THREADS_PER_BLOCK = 32 * _WARPS_PER_BLOCK
_KERNEL_OPTIONS = (
    "--std=c++11",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
)


@cache
def _cupy_module() -> ModuleType:
    """Load CuPy only for an explicit accelerator execution."""

    return importlib.import_module("cupy")


@cache
def _median_filter_kernel(cupy: ModuleType, dtype_name: str):
    """Compile one size-independent median-selection kernel per dtype."""

    declarations = {
        "uint8": "const unsigned char* input, unsigned char* output",
        "uint16": "const unsigned short* input, unsigned short* output",
        "float32": "const float* input, float* output",
    }
    key_functions = {
        "uint8": r"""
            __device__ __forceinline__ unsigned int vipp_median_key(
                const unsigned char value)
            {
                return (unsigned int)value;
            }
        """,
        "uint16": r"""
            __device__ __forceinline__ unsigned int vipp_median_key(
                const unsigned short value)
            {
                return (unsigned int)value;
            }
        """,
        "float32": r"""
            __device__ __forceinline__ unsigned int vipp_median_key(
                const float value)
            {
                const unsigned int bits = __float_as_uint(value);
                return (bits & 0x80000000U)
                    ? ~bits
                    : (bits ^ 0x80000000U);
            }
        """,
    }
    stores = {
        "uint8": "output[output_index] = (unsigned char)prefix[warp];",
        "uint16": "output[output_index] = (unsigned short)prefix[warp];",
        "float32": r"""
            const unsigned int key = prefix[warp];
            const unsigned int bits =
                (key & 0x80000000U) ? (key ^ 0x80000000U) : ~key;
            output[output_index] = __uint_as_float(bits);
        """,
    }
    bit_counts = {"uint8": 8, "uint16": 16, "float32": 32}
    if dtype_name not in declarations:  # pragma: no cover - private guard
        raise ValueError(f"Unsupported Median Filter dtype {dtype_name!r}.")

    kernel_name = f"vipp_dynamic_median_{dtype_name}"
    source = rf"""
        {key_functions[dtype_name]}

        __device__ __forceinline__ long long vipp_reflect_index(
            long long coordinate,
            const long long extent)
        {{
            if (extent <= 1) {{
                return 0;
            }}
            const long long period = extent * 2;
            coordinate %= period;
            if (coordinate < 0) {{
                coordinate += period;
            }}
            return coordinate < extent
                ? coordinate
                : period - coordinate - 1;
        }}

        extern "C" __global__
        void {kernel_name}(
            {declarations[dtype_name]},
            const long long output_size,
            const int ndim,
            const long long* shape,
            const long long* strides,
            const int y_axis,
            const int x_axis,
            const int footprint_size)
        {{
            const int warp = threadIdx.x >> 5;
            const int lane = threadIdx.x & 31;
            const long long output_index =
                (long long)blockIdx.x * {_WARPS_PER_BLOCK} + warp;
            if (output_index >= output_size) {{
                return;
            }}

            __shared__ unsigned int histogram[{_WARPS_PER_BLOCK}][16];
            __shared__ unsigned int prefix[{_WARPS_PER_BLOCK}];
            __shared__ unsigned int remaining_rank[{_WARPS_PER_BLOCK}];
            __shared__ long long base_offset[{_WARPS_PER_BLOCK}];
            __shared__ long long y_coordinate[{_WARPS_PER_BLOCK}];
            __shared__ long long x_coordinate[{_WARPS_PER_BLOCK}];

            if (lane == 0) {{
                long long remainder = output_index;
                long long base = 0;
                long long y = 0;
                long long x = 0;
                for (int axis = ndim - 1; axis >= 0; --axis) {{
                    const long long coordinate = remainder % shape[axis];
                    remainder /= shape[axis];
                    if (axis == y_axis) {{
                        y = coordinate;
                    }}
                    if (axis == x_axis) {{
                        x = coordinate;
                    }}
                    if (axis != y_axis && axis != x_axis) {{
                        base += coordinate * strides[axis];
                    }}
                }}
                const unsigned int sample_count = y_axis == x_axis
                    ? (unsigned int)footprint_size
                    : (unsigned int)(footprint_size * footprint_size);
                base_offset[warp] = base;
                y_coordinate[warp] = y;
                x_coordinate[warp] = x;
                prefix[warp] = 0U;
                remaining_rank[warp] = sample_count >> 1;
            }}
            __syncwarp();

            const int radius = footprint_size >> 1;
            const int sample_count = y_axis == x_axis
                ? footprint_size
                : footprint_size * footprint_size;
            const int pass_count = {bit_counts[dtype_name]} / 4;
            for (int pass = 0; pass < pass_count; ++pass) {{
                if (lane < 16) {{
                    histogram[warp][lane] = 0U;
                }}
                __syncwarp();

                const int shift = {bit_counts[dtype_name]} - 4 * (pass + 1);
                for (int sample = lane; sample < sample_count; sample += 32) {{
                    long long input_offset;
                    if (y_axis == x_axis) {{
                        const long long coordinate = vipp_reflect_index(
                            y_coordinate[warp] + sample - radius,
                            shape[y_axis]);
                        input_offset =
                            base_offset[warp] + coordinate * strides[y_axis];
                    }} else {{
                        const int y_delta = sample / footprint_size - radius;
                        const int x_delta = sample % footprint_size - radius;
                        const long long y = vipp_reflect_index(
                            y_coordinate[warp] + y_delta,
                            shape[y_axis]);
                        const long long x = vipp_reflect_index(
                            x_coordinate[warp] + x_delta,
                            shape[x_axis]);
                        input_offset = base_offset[warp]
                            + y * strides[y_axis]
                            + x * strides[x_axis];
                    }}
                    const unsigned int key =
                        vipp_median_key(input[input_offset]);
                    if (pass == 0
                        || (key >> (shift + 4)) == prefix[warp]) {{
                        atomicAdd(
                            &histogram[warp][(key >> shift) & 15U],
                            1U);
                    }}
                }}
                __syncwarp();

                if (lane == 0) {{
                    unsigned int rank = remaining_rank[warp];
                    unsigned int selected = 0U;
                    for (unsigned int digit = 0U; digit < 16U; ++digit) {{
                        const unsigned int count = histogram[warp][digit];
                        if (rank < count) {{
                            selected = digit;
                            break;
                        }}
                        rank -= count;
                    }}
                    remaining_rank[warp] = rank;
                    prefix[warp] = (prefix[warp] << 4) | selected;
                }}
                __syncwarp();
            }}

            if (lane == 0) {{
                {stores[dtype_name]}
            }}
        }}
    """
    return cupy.RawKernel(
        source,
        kernel_name,
        options=_KERNEL_OPTIONS,
    )


def median_filter(
    data,
    size: int = 5,
    channel_axis: int | None = None,
):
    """Apply VIPP's exact median filter without size-specific compilation.

    The active footprint is the trailing two non-channel axes (or the sole
    non-channel axis for 1D data), matching
    :func:`napari_vipp.core.operations.median_filter`. Boundary coordinates
    use SciPy's half-sample-symmetric ``reflect`` convention. The admitted
    float32 region is finite and excludes negative zero, which permits an
    exact IEEE-754 order key and bitwise output restoration.
    """

    cupy = _cupy_module()
    array = cupy.asarray(data)
    channel_axis = _validated_channel_axis(channel_axis, array.ndim)
    canonical_size = _odd_size(size, minimum=1)
    y_axis, x_axis = _xy_axes(array.ndim, channel_axis=channel_axis)
    dtype = np.dtype(array.dtype)
    if dtype not in _SUPPORTED_DTYPES:
        raise TypeError(
            "Median Filter GPU execution supports uint8, uint16, and float32; "
            f"received {dtype.name}."
        )

    output = cupy.empty(array.shape, dtype=array.dtype)
    if array.size == 0:
        return output

    itemsize = int(array.dtype.itemsize)
    shape = cupy.asarray(np.asarray(array.shape, dtype=np.int64))
    strides = cupy.asarray(
        np.asarray(
            tuple(int(stride) // itemsize for stride in array.strides),
            dtype=np.int64,
        )
    )
    blocks = (int(array.size) + _WARPS_PER_BLOCK - 1) // _WARPS_PER_BLOCK
    kernel = _median_filter_kernel(cupy, dtype.name)
    kernel(
        (blocks,),
        (_THREADS_PER_BLOCK,),
        (
            array,
            output,
            np.int64(array.size),
            np.int32(array.ndim),
            shape,
            strides,
            np.int32(y_axis),
            np.int32(x_axis),
            np.int32(canonical_size),
        ),
    )
    return output


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
