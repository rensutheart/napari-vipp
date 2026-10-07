"""ImageJ 1.x direct Gaussian convolution on independent scalar YX planes.

This is a source port of the non-downsampling branch of ImageJ's
``ij.plugin.filter.GaussianBlur`` (Michael Schmid / Stephan Saalfeld), including
its polynomial tail correction, edge tail sums and float32 accumulation order.
ImageJ is public-domain software; attribution is retained here:
https://imagej.net/ij/developer/source/ij/plugin/filter/GaussianBlur.java.html
https://imagej.net/ij/docs/intro.html

The target is ImageJ 1.54p's menu Gaussian Blur behavior. Independent Java
reference evidence must establish parity for any claimed use; this source port
alone is not proof of equivalence. This module does not change VIPP's existing
Gaussian operations.
"""

from __future__ import annotations

import math
from numbers import Real

import numpy as np

_SUPPORTED_DTYPES = (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"))
_MAX_DIRECT_SIGMA = 8.5
_FINITE_VALIDATION_CHUNK_SIZE = 64 * 1024


def _make_kernel(
    sigma: float, accuracy: float, line_length: int
) -> tuple[np.ndarray, np.ndarray]:
    """Port ImageJ's one-sided coefficients and out-of-line tail sums."""
    radius = math.ceil(sigma * math.sqrt(-2.0 * math.log(accuracy))) + 1
    max_radius = max(line_length, 50)
    radius = min(radius, max_radius)
    kernel = np.empty(radius, dtype=np.float32)
    for index in range(radius):
        kernel[index] = math.exp(-0.5 * index * index / sigma / sigma)

    if radius < max_radius and radius > 3:
        sqrt_slope = math.inf
        index = radius
        while index > radius // 2:
            index -= 1
            candidate = math.sqrt(float(kernel[index])) / (radius - index)
            if candidate < sqrt_slope:
                sqrt_slope = candidate
            else:
                break
        for tail_index in range(index + 2, radius):
            distance = radius - tail_index
            kernel[tail_index] = distance * distance * sqrt_slope * sqrt_slope

    if radius < max_radius:
        normalization = float(kernel[0])
        for index in range(1, radius):
            # Java's integer-times-float product is rounded to float32 first.
            normalization += float(np.float32(2) * kernel[index])
    else:
        normalization = sigma * math.sqrt(2.0 * math.pi)

    running_sum = 0.5 + 0.5 * float(kernel[0]) / normalization
    tail_sums = np.empty(radius, dtype=np.float32)
    for index in range(radius):
        value = float(kernel[index]) / normalization
        kernel[index] = value
        running_sum -= value
        tail_sums[index] = running_sum
    return kernel, tail_sums


def _convolve_direction(
    source: np.ndarray,
    kernel: np.ndarray,
    tail_sums: np.ndarray,
    axis: int,
    progress=None,
) -> np.ndarray:
    """Evaluate Java's paired-neighbor sums in the same float32 order."""
    if progress is not None:
        progress.check_cancelled()
    lines = np.moveaxis(source, axis, -1)
    length = lines.shape[-1]
    radius = len(kernel)
    first_part = min(radius, length)
    result = lines * kernel[0]

    # The left and right tails precede the neighbor products in ImageJ.
    result[..., :first_part] += (
        lines[..., :1] * tail_sums[np.arange(first_part)]
    )
    positions = np.arange(length)
    right_tail = np.flatnonzero(
        ((positions < first_part) & (positions + radius > length))
        | ((positions >= first_part) & (positions + radius >= length))
    )
    if right_tail.size:
        result[..., right_tail] += (
            lines[..., -1:] * tail_sums[length - right_tail - 1]
        )

    paired = np.empty_like(lines)
    product = np.empty_like(lines)
    inside_end = length - radius
    for offset in range(1, radius):
        if progress is not None:
            progress.check_cancelled()
        paired.fill(0)
        if offset < length:
            # Preserve the Java order: v=0; v+=left; v+=right; result+=k*v.
            paired[..., offset:] += lines[..., :-offset]
            paired[..., :-offset] += lines[..., offset:]
            if radius < inside_end:
                # ImageJ's interior branch sums the two neighbors directly.
                # Adding +0 first would erase a pair of negative-zero values.
                np.add(
                    lines[..., radius - offset : inside_end - offset],
                    lines[..., radius + offset : inside_end + offset],
                    out=paired[..., radius:inside_end],
                )
        np.multiply(kernel[offset], paired, out=product)
        np.add(result, product, out=result)
    return np.moveaxis(result, -1, axis)


def _validate_finite_float_input(array: np.ndarray, progress=None) -> None:
    """Validate every float value before filtering, using bounded buffers."""
    chunks = np.nditer(
        array,
        flags=["external_loop", "buffered"],
        op_flags=["readonly"],
        order="K",
        buffersize=_FINITE_VALIDATION_CHUNK_SIZE,
    )
    for chunk in chunks:
        if progress is not None:
            progress.check_cancelled()
        if not np.isfinite(chunk).all():
            raise ValueError("ImageJ Gaussian Blur requires finite input values.")


def imagej_gaussian_blur(data, sigma: float = 1.5, progress=None) -> np.ndarray:
    """Blur each trailing YX plane with the ImageJ direct Gaussian algorithm.

    Only scalar uint8, uint16 and finite float32 data are supported. The caller
    must validate canonical trailing YX semantics; leading positions are
    independent and are never mixed. Sigma is a pixel standard deviation in
    [0, 8.5]; larger values require ImageJ's unsupported downsampling branch.
    There is no ROI, calibration-unit conversion or channel interpretation.

    Float32 convolution runs X then Y with nearest-edge extension and the
    dtype-dependent ImageJ menu accuracy (0.002 for uint8; 0.0002 otherwise).
    Integer outputs use ImageJ's final float32 +0.5, clamp, truncate conversion;
    float32 outputs retain signed values. Float32 arithmetic overflow fails
    visibly. A same-shape, same-dtype owned output is returned, including sigma
    zero; input arrays, including read-only/strided arrays, are never changed.
    """
    if progress is not None:
        progress.check_cancelled()
    array = np.asarray(data)
    if array.ndim < 2 or any(length == 0 for length in array.shape):
        raise ValueError("ImageJ Gaussian Blur requires nonempty scalar YX planes.")
    if array.dtype not in _SUPPORTED_DTYPES:
        raise ValueError(
            "ImageJ Gaussian Blur requires uint8, uint16 or float32 input; "
            "choose an explicit conversion before this operation."
        )
    if isinstance(sigma, (bool, np.bool_)) or not isinstance(sigma, Real):
        raise ValueError("ImageJ Gaussian sigma must be a real scalar in [0, 8.5].")
    sigma_value = float(sigma)
    if not math.isfinite(sigma_value) or not 0 <= sigma_value <= _MAX_DIRECT_SIGMA:
        raise ValueError(
            "ImageJ Gaussian sigma must be finite and in [0, 8.5] pixels; "
            "larger sigma requires the unsupported ImageJ downsampling algorithm."
        )
    if array.dtype == np.dtype("float32"):
        _validate_finite_float_input(array, progress)
    if progress is not None:
        progress.check_cancelled()
    if sigma_value == 0:
        return array.copy()

    accuracy = 0.002 if array.dtype == np.dtype("uint8") else 0.0002
    kernel_x, tail_x = _make_kernel(sigma_value, accuracy, array.shape[-1])
    kernel_y, tail_y = _make_kernel(sigma_value, accuracy, array.shape[-2])
    output = np.empty_like(array)
    leading_shape = array.shape[:-2]
    total = 2 * math.prod(leading_shape)
    indexes = ((),) if not leading_shape else np.ndindex(leading_shape)
    for index, position in enumerate(indexes):
        if progress is not None:
            progress.check_cancelled()
            progress.report(2 * index, total, "ImageJ Gaussian XY planes")
        source = array[position].astype(np.float32, copy=True)
        try:
            with np.errstate(over="raise", invalid="raise"):
                filtered = _convolve_direction(
                    source, kernel_x, tail_x, axis=1, progress=progress
                )
                if progress is not None:
                    progress.check_cancelled()
                    progress.report(2 * index + 1, total, "ImageJ Gaussian XY planes")
                filtered = _convolve_direction(
                    filtered, kernel_y, tail_y, axis=0, progress=progress
                )
        except FloatingPointError as error:
            raise ValueError(
                "ImageJ Gaussian float32 arithmetic overflowed; "
                "use an explicit intensity-range conversion before this operation."
            ) from error
        if array.dtype != np.dtype("float32"):
            filtered += np.float32(0.5)
            np.clip(filtered, 0, np.iinfo(array.dtype).max, out=filtered)
        output[position] = filtered
        if progress is not None:
            progress.check_cancelled()
            progress.report(2 * index + 2, total, "ImageJ Gaussian XY planes")
    return output
