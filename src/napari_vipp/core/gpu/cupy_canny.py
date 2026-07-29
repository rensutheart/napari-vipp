"""Device-resident CuPyX implementation of VIPP's Canny edge detector.

The optional CUDA stack is imported only when :func:`canny_edges` is called.
The implementation mirrors the authoritative CPU operation rather than
delegating to a second high-level Canny implementation: scalar images are
processed as trailing Y/X planes, an explicitly declared RGB/RGBA axis is
reduced with VIPP's BT.601 coefficients, and scikit-image's constant-boundary
Gaussian, Sobel, bilinear non-maximum suppression, and eight-connected
hysteresis conventions are retained.
"""

from __future__ import annotations

import importlib
import math
from functools import cache
from itertools import product
from numbers import Integral
from types import ModuleType

import numpy as np


@cache
def _cupy_modules() -> tuple[ModuleType, ModuleType]:
    """Load optional CUDA modules only for explicit accelerator execution."""

    cupy = importlib.import_module("cupy")
    ndimage = importlib.import_module("cupyx.scipy.ndimage")
    return cupy, ndimage


@cache
def _nonmaximum_kernel(cupy: ModuleType):
    """Compile scikit-image's float32 bilinear suppression rule for CUDA."""

    # This is the branch structure used by
    # skimage.feature._canny_cy._nonmaximum_suppression_bilinear.  Keeping it
    # in one kernel avoids materialising the many image-sized boolean and
    # neighbour arrays required by a vectorised transcription.
    return cupy.RawKernel(
        r"""
        extern "C" __global__
        void vipp_canny_nonmaximum(
            const float* isobel,
            const float* jsobel,
            const float* magnitude,
            const unsigned char* eroded_mask,
            const float* low_threshold,
            const long long rows,
            const long long cols,
            float* out)
        {
            const long long index =
                (long long)blockDim.x * blockIdx.x + threadIdx.x;
            const long long size = rows * cols;
            if (index >= size) {
                return;
            }
            out[index] = 0.0f;

            const float m = magnitude[index];
            if (!eroded_mask[index] || !(m >= *low_threshold)) {
                return;
            }

            const float i = isobel[index];
            const float j = jsobel[index];
            const bool is_down = i <= 0.0f;
            const bool is_up = i >= 0.0f;
            const bool is_left = j <= 0.0f;
            const bool is_right = j >= 0.0f;
            const bool cond1 =
                (is_up && is_right) || (is_down && is_left);
            const bool cond2 =
                (is_down && is_right) || (is_up && is_left);
            if (!cond1 && !cond2) {
                return;
            }

            const float abs_i = fabsf(i);
            const float abs_j = fabsf(j);
            float weight;
            float neighbour1_1;
            float neighbour1_2;
            float neighbour2_1;
            float neighbour2_2;

            if (cond1) {
                if (abs_i > abs_j) {
                    weight = abs_j / abs_i;
                    neighbour1_1 = magnitude[index + cols];
                    neighbour1_2 = magnitude[index + cols + 1];
                    neighbour2_1 = magnitude[index - cols];
                    neighbour2_2 = magnitude[index - cols - 1];
                } else {
                    weight = abs_i / abs_j;
                    neighbour1_1 = magnitude[index + 1];
                    neighbour1_2 = magnitude[index + cols + 1];
                    neighbour2_1 = magnitude[index - 1];
                    neighbour2_2 = magnitude[index - cols - 1];
                }
            } else {
                if (abs_i < abs_j) {
                    weight = abs_i / abs_j;
                    neighbour1_1 = magnitude[index + 1];
                    neighbour1_2 = magnitude[index - cols + 1];
                    neighbour2_1 = magnitude[index - 1];
                    neighbour2_2 = magnitude[index + cols - 1];
                } else {
                    weight = abs_j / abs_i;
                    neighbour1_1 = magnitude[index - cols];
                    neighbour1_2 = magnitude[index - cols + 1];
                    neighbour2_1 = magnitude[index + cols];
                    neighbour2_2 = magnitude[index + cols - 1];
                }
            }

            // Cython's ``1.0 - w`` contains a C-double literal.  The first
            // product is therefore rounded as float32 while the second term
            // and final addition are evaluated as float64.
            const float plus_first = neighbour1_2 * weight;
            const double plus_value = (double)plus_first
                + (double)neighbour1_1 * (1.0 - (double)weight);
            const bool plus = plus_value <= (double)m;
            if (plus) {
                const float minus_first = neighbour2_2 * weight;
                const double minus_value = (double)minus_first
                    + (double)neighbour2_1 * (1.0 - (double)weight);
                const bool minus = minus_value <= (double)m;
                if (minus) {
                    out[index] = m;
                }
            }
        }
        """,
        "vipp_canny_nonmaximum",
        options=("--std=c++11", "--fmad=false"),
    )


@cache
def _correlate1d_kernel(cupy: ModuleType):
    """Compile SciPy's deterministic odd-kernel correlation order."""

    # scipy.ndimage's NI_Correlate1D does not accumulate taps from left to
    # right.  It detects symmetric/antisymmetric kernels, starts with the
    # centre tap, and then adds paired samples from the outside in.  That
    # order is observable for finite float32 inputs with severe cancellation:
    # changing it can change a Canny edge bit even though both calculations
    # use double accumulators.  CuPyX is therefore not exact enough for this
    # particular primitive; this kernel reproduces SciPy's ordering explicitly.
    return cupy.RawKernel(
        r"""
        __device__ __forceinline__ long long reflect_index(
            long long coordinate,
            const long long length)
        {
            if (length <= 1) {
                return 0;
            }
            const long long period = 2 * length;
            coordinate %= period;
            if (coordinate < 0) {
                coordinate += period;
            }
            if (coordinate >= length) {
                coordinate = period - coordinate - 1;
            }
            return coordinate;
        }

        __device__ __forceinline__ float sample_axis(
            const float* input,
            const long long row,
            const long long col,
            const long long rows,
            const long long cols,
            const int axis,
            const int reflect_mode)
        {
            long long sampled_row = row;
            long long sampled_col = col;
            if (axis == 0) {
                if (!reflect_mode && (sampled_row < 0 || sampled_row >= rows)) {
                    return 0.0f;
                }
                sampled_row = reflect_index(sampled_row, rows);
            } else {
                if (!reflect_mode && (sampled_col < 0 || sampled_col >= cols)) {
                    return 0.0f;
                }
                sampled_col = reflect_index(sampled_col, cols);
            }
            return input[sampled_row * cols + sampled_col];
        }

        extern "C" __global__
        void vipp_scipy_correlate1d(
            const float* input,
            const double* weights,
            const long long rows,
            const long long cols,
            const int radius,
            const int axis,
            const int reflect_mode,
            const int symmetry,
            float* output)
        {
            const long long index =
                (long long)blockDim.x * blockIdx.x + threadIdx.x;
            const long long size = rows * cols;
            if (index >= size) {
                return;
            }
            const long long row = index / cols;
            const long long col = index - row * cols;
            const double centre = (double)sample_axis(
                input, row, col, rows, cols, axis, reflect_mode);
            double value;

            if (symmetry != 0) {
                value = centre * weights[radius];
                // Matches NI_Correlate1D's jj=-radius..-1 loop exactly.
                for (int offset = radius; offset >= 1; --offset) {
                    const long long row_offset = axis == 0 ? offset : 0;
                    const long long col_offset = axis == 1 ? offset : 0;
                    const double left = (double)sample_axis(
                        input,
                        row - row_offset,
                        col - col_offset,
                        rows,
                        cols,
                        axis,
                        reflect_mode);
                    const double right = (double)sample_axis(
                        input,
                        row + row_offset,
                        col + col_offset,
                        rows,
                        cols,
                        axis,
                        reflect_mode);
                    const double pair = symmetry > 0
                        ? left + right
                        : left - right;
                    value += pair * weights[radius - offset];
                }
            } else {
                const long long row_offset = axis == 0 ? radius : 0;
                const long long col_offset = axis == 1 ? radius : 0;
                value = (double)sample_axis(
                    input,
                    row + row_offset,
                    col + col_offset,
                    rows,
                    cols,
                    axis,
                    reflect_mode) * weights[2 * radius];
                for (int offset = -radius; offset < radius; ++offset) {
                    const long long tap_row = axis == 0 ? row + offset : row;
                    const long long tap_col = axis == 1 ? col + offset : col;
                    value += (double)sample_axis(
                        input,
                        tap_row,
                        tap_col,
                        rows,
                        cols,
                        axis,
                        reflect_mode) * weights[radius + offset];
                }
            }
            output[index] = (float)value;
        }
        """,
        "vipp_scipy_correlate1d",
        options=("--std=c++11", "--fmad=false"),
    )


def canny_edges(
    data,
    sigma: float = 1.0,
    low_quantile: float = 0.1,
    high_quantile: float = 0.2,
    channel_axis: int | None = None,
    progress=None,
):
    """Return VIPP's slice-wise Canny mask without leaving the GPU.

    Input planes are explicitly converted to float32, exactly as in the CPU
    operation.  The output is a resident boolean CuPy array and the input is
    never mutated. Exact public execution is gated by compute policy to bool,
    uint8, and uint16 inputs; direct float32 calls remain useful for numerical
    study but are not public because CUDA can flush subnormal intermediates.
    """

    cupy, ndimage = _cupy_modules()
    array = cupy.asarray(data)
    if array.size == 0:
        raise ValueError("Canny requires non-empty image data.")

    array = _to_explicit_grayscale(
        array,
        channel_axis=channel_axis,
        cupy=cupy,
    )
    low, high = _validated_threshold_pair(low_quantile, high_quantile)
    sigma = _nonnegative_sigma(sigma)

    if array.ndim <= 2:
        if progress is not None:
            progress.report(0, 1, "Canny planes")
            progress.check_cancelled()
        result = _canny_plane(
            array,
            sigma=sigma,
            low_quantile=low,
            high_quantile=high,
            cupy=cupy,
            ndimage=ndimage,
        )
        cupy.cuda.get_current_stream().synchronize()
        if progress is not None:
            progress.report(1, 1, "Canny planes")
        return result

    leading_shape = tuple(int(size) for size in array.shape[:-2])
    total = math.prod(leading_shape)
    if progress is not None:
        progress.report(0, total, "Canny planes")
        progress.check_cancelled()
    output = cupy.empty(array.shape, dtype=cupy.bool_)
    completed = 0
    for index in _indices(leading_shape):
        if completed and progress is not None:
            progress.check_cancelled()
        output[index] = _canny_plane(
            array[index],
            sigma=sigma,
            low_quantile=low,
            high_quantile=high,
            cupy=cupy,
            ndimage=ndimage,
        )
        cupy.cuda.get_current_stream().synchronize()
        completed += 1
        if progress is not None:
            progress.report(completed, total, "Canny planes")
    return output


def _canny_plane(
    plane,
    *,
    sigma: float,
    low_quantile: float,
    high_quantile: float,
    cupy: ModuleType,
    ndimage: ModuleType,
):
    values = cupy.asarray(plane, dtype=cupy.float32)
    if values.ndim != 2:
        raise ValueError("The parameter `image` must be a 2-dimensional array")

    # scikit-image's default mode is constant.  Its Canny preprocessing
    # corrects the smoothed image for the fractional Gaussian contribution of
    # an all-true mask, then excludes the one-pixel border from NMS.
    mask = cupy.ones(values.shape, dtype=cupy.float32)
    bleed_over = _scipy_order_gaussian_filter(
        mask,
        sigma=sigma,
        cupy=cupy,
        ndimage=ndimage,
    )
    bleed_over += cupy.finfo(cupy.float32).eps
    smoothed = _scipy_order_gaussian_filter(
        values,
        sigma=sigma,
        cupy=cupy,
        ndimage=ndimage,
    )
    smoothed /= bleed_over

    jsobel = _scipy_order_sobel(
        smoothed,
        axis=1,
        cupy=cupy,
        ndimage=ndimage,
    )
    isobel = _scipy_order_sobel(
        smoothed,
        axis=0,
        cupy=cupy,
        ndimage=ndimage,
    )
    magnitude = isobel * isobel
    magnitude += jsobel * jsobel
    cupy.sqrt(magnitude, out=magnitude)

    thresholds = cupy.percentile(
        magnitude,
        cupy.asarray(
            (100.0 * low_quantile, 100.0 * high_quantile),
            dtype=cupy.float64,
        ),
    )
    low_threshold = thresholds[0].astype(cupy.float32, copy=False)
    high_threshold = thresholds[1].astype(cupy.float32, copy=False)
    # The CPU Cython kernel deliberately excludes zero-magnitude pixels when
    # the requested low quantile resolves to zero.
    low_threshold = cupy.where(
        low_threshold == cupy.float32(0.0),
        cupy.float32(1e-14),
        low_threshold,
    )

    eroded_mask = cupy.ones(values.shape, dtype=cupy.uint8)
    eroded_mask[:1, :] = 0
    eroded_mask[-1:, :] = 0
    eroded_mask[:, :1] = 0
    eroded_mask[:, -1:] = 0
    low_masked = cupy.empty(values.shape, dtype=cupy.float32)
    size = int(values.size)
    threads = 256
    blocks = (size + threads - 1) // threads
    _nonmaximum_kernel(cupy)(
        (blocks,),
        (threads,),
        (
            cupy.ascontiguousarray(isobel),
            cupy.ascontiguousarray(jsobel),
            cupy.ascontiguousarray(magnitude),
            cupy.ascontiguousarray(eroded_mask),
            low_threshold,
            int(values.shape[0]),
            int(values.shape[1]),
            low_masked,
        ),
    )

    low_mask = low_masked > cupy.float32(0.0)
    structure = cupy.ones((3, 3), dtype=cupy.bool_)
    labels, count = ndimage.label(low_mask, structure=structure)
    if int(count) == 0:
        return low_mask

    high_mask = low_mask & (low_masked >= high_threshold)
    good_labels = cupy.zeros(int(count) + 1, dtype=cupy.bool_)
    good_labels[cupy.unique(labels[high_mask])] = True
    return good_labels[labels]


def _scipy_order_gaussian_filter(
    array,
    *,
    sigma: float,
    cupy: ModuleType,
    ndimage: ModuleType,
):
    """Match SciPy's separable Gaussian accumulation and float32 stores."""

    if sigma <= 1e-15:
        return array.copy()
    radius = int(4.0 * sigma + 0.5)
    weights = _gaussian_weights(sigma, radius)
    result = cupy.asarray(array, dtype=cupy.float32)
    for axis in (0, 1):
        result = _scipy_order_correlate1d(
            result,
            weights,
            axis=axis,
            mode="constant",
            cupy=cupy,
            ndimage=ndimage,
        )
    return result


def _gaussian_weights(sigma: float, radius: int) -> np.ndarray:
    """Build exactly the order-zero float64 kernel used by SciPy."""

    sigma_squared = sigma * sigma
    positions = np.arange(-radius, radius + 1)
    weights = np.exp(-0.5 / sigma_squared * positions**2)
    return weights / weights.sum()


def _scipy_order_sobel(
    array,
    *,
    axis: int,
    cupy: ModuleType,
    ndimage: ModuleType,
):
    """Match SciPy's derivative-then-smoothing float32 Sobel sequence."""

    intermediate = _scipy_order_correlate1d(
        array,
        (-1.0, 0.0, 1.0),
        axis=axis,
        mode="reflect",
        cupy=cupy,
        ndimage=ndimage,
    )
    return _scipy_order_correlate1d(
        intermediate,
        (1.0, 2.0, 1.0),
        axis=1 - axis,
        mode="reflect",
        cupy=cupy,
        ndimage=ndimage,
    )


def _scipy_order_correlate1d(
    array,
    weights,
    *,
    axis: int,
    mode: str,
    cupy: ModuleType,
    ndimage: ModuleType,
):
    """Accumulate in float64, then reproduce SciPy's float32 axis store."""

    del ndimage
    values = cupy.ascontiguousarray(array, dtype=cupy.float32)
    if values.ndim != 2:
        raise ValueError("Canny correlation requires a two-dimensional plane.")
    if axis not in {0, 1}:
        raise ValueError("Canny correlation axis must be 0 or 1.")
    if mode not in {"constant", "reflect"}:
        raise ValueError("Canny correlation supports constant or reflect mode.")

    host_weights = np.asarray(weights, dtype=np.float64)
    if host_weights.ndim != 1 or host_weights.size % 2 != 1:
        raise ValueError("Canny correlation requires an odd one-dimensional kernel.")
    radius = int(host_weights.size // 2)
    tolerance = np.finfo(np.float64).eps
    symmetry = 1
    for offset in range(1, radius + 1):
        if (
            abs(
                host_weights[radius + offset]
                - host_weights[radius - offset]
            )
            > tolerance
        ):
            symmetry = 0
            break
    if symmetry == 0:
        symmetry = -1
        for offset in range(1, radius + 1):
            if (
                abs(
                    host_weights[radius + offset]
                    + host_weights[radius - offset]
                )
                > tolerance
            ):
                symmetry = 0
                break

    output = cupy.empty(values.shape, dtype=cupy.float32)
    size = int(values.size)
    threads = 256
    blocks = (size + threads - 1) // threads
    _correlate1d_kernel(cupy)(
        (blocks,),
        (threads,),
        (
            values,
            cupy.asarray(host_weights, dtype=cupy.float64),
            int(values.shape[0]),
            int(values.shape[1]),
            radius,
            axis,
            int(mode == "reflect"),
            symmetry,
            output,
        ),
    )
    return output


def _to_explicit_grayscale(array, *, channel_axis, cupy: ModuleType):
    axis = _validated_channel_axis(channel_axis, array.ndim)
    if axis is None:
        return array
    channel_count = int(array.shape[axis])
    if channel_count not in {3, 4}:
        raise ValueError(
            "Canny channel_axis must contain exactly 3 RGB or 4 RGBA "
            f"channels, not {channel_count}."
        )
    if not (
        array.dtype == cupy.bool_
        or cupy.issubdtype(array.dtype, cupy.integer)
        or cupy.issubdtype(array.dtype, cupy.floating)
    ):
        raise ValueError(
            "Canny RGB/RGBA conversion requires real-valued boolean, integer, "
            "or floating image data."
        )

    moved = cupy.moveaxis(array, axis, -1)
    work_dtype = cupy.result_type(array.dtype, cupy.float32)
    rgb = moved[..., :3].astype(work_dtype, copy=False)
    coefficients = cupy.asarray((0.299, 0.587, 0.114), dtype=work_dtype)
    return cupy.sum(rgb * coefficients, axis=-1, dtype=work_dtype)


def _validated_channel_axis(value, ndim: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError("Canny channel_axis must be an integer or None.")
    if ndim < 3:
        raise ValueError(
            "Canny requires at least two spatial dimensions when channel_axis "
            "is set."
        )
    axis = int(value)
    if axis < -ndim or axis >= ndim:
        raise ValueError(
            f"Canny channel_axis {axis} is out of range for {ndim}D input."
        )
    return axis % ndim


def _validated_threshold_pair(low, high) -> tuple[float, float]:
    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Canny low and high thresholds must be finite numbers."
        ) from exc
    if not math.isfinite(low_value) or not math.isfinite(high_value):
        raise ValueError("Canny low and high thresholds must be finite numbers.")
    if low_value < 0.0 or high_value < 0.0:
        raise ValueError("Canny low and high thresholds must be at least 0.")
    if low_value > 1.0 or high_value > 1.0:
        raise ValueError("Canny low and high thresholds must be at most 1.")
    if high_value < low_value:
        raise ValueError(
            f"Canny low threshold ({low_value:g}) must not exceed the high "
            f"threshold ({high_value:g})."
        )
    return low_value, high_value


def _nonnegative_sigma(value) -> float:
    # The CPU operation intentionally clamps negative values to zero and
    # otherwise lets scikit-image/SciPy surface invalid non-finite inputs.
    return max(float(value), 0.0)


def _indices(shape: tuple[int, ...]):
    if not shape:
        yield ()
        return
    yield from product(*(range(size) for size in shape))


__all__ = ["canny_edges"]
