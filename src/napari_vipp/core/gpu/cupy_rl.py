"""Device-resident CuPyX Richardson--Lucy deconvolution.

The optional CUDA modules are imported only when the provider is called.  The
public callable mirrors VIPP's authoritative CPU operation: it receives the
ordered ``[Image, PSF]`` input list, applies a two- or three-dimensional
deconvolution to each leading block, and returns a float32 array with the image
shape.  Image, PSF, output, and all image-sized intermediates remain in the
CuPy array domain.
"""

from __future__ import annotations

import importlib
from functools import cache
from itertools import product
from numbers import Integral
from types import ModuleType

from napari_vipp.core.gpu.cupy_imports import import_cupyx_signal_module


@cache
def _cupy_modules() -> tuple[ModuleType, ModuleType]:
    """Load optional CUDA modules only for explicit accelerator execution."""

    cupy = importlib.import_module("cupy")
    signal = import_cupyx_signal_module()
    return cupy, signal


def richardson_lucy_deconvolution(
    inputs,
    spatial_mode: str = "Auto from axes",
    iterations: int = 25,
    normalize_psf: bool = True,
    clip_negative_input: bool = True,
    clip_output_negative: bool = True,
    preserve_input_scale: bool = True,
    filter_epsilon: float = 1e-12,
    resolved_spatial_ndim: int | None = None,
    progress=None,
):
    """Restore a resident float32 image with Richardson--Lucy iterations.

    ``inputs`` is the established VIPP multi-input positional argument: an
    ordered iterable whose first two values are Image and PSF.  The validated
    production region is finite float32 input on two- or three-dimensional
    spatial blocks.  The implementation nevertheless retains the CPU
    operation's cleaning and float32 conversion at this provider boundary.
    """

    cupy, signal = _cupy_modules()
    image, psf = _deconvolution_inputs(inputs, cupy=cupy)
    spatial_ndim = _resolved_deconvolution_spatial_ndim(
        image.ndim,
        spatial_mode,
        resolved_spatial_ndim,
    )
    kernel = _deconvolution_psf(
        psf,
        spatial_ndim,
        normalize_psf=bool(normalize_psf),
        cupy=cupy,
    )
    iteration_count = max(int(iterations), 1)
    epsilon = float(filter_epsilon)

    def restore_block(block, iteration_done=None):
        values, output_scale = _deconvolution_observed_block(
            block,
            clip_negative_input=bool(clip_negative_input),
            preserve_input_scale=bool(preserve_input_scale),
            cupy=cupy,
        )
        restored = _richardson_lucy_block(
            values,
            kernel,
            iterations=iteration_count,
            filter_epsilon=epsilon,
            iteration_done=iteration_done,
            check_cancelled=(
                progress.check_cancelled if progress is not None else None
            ),
            sanitize_each_iteration=progress is not None,
            cupy=cupy,
            signal=signal,
        )
        return _deconvolution_output_block(
            restored,
            output_scale=output_scale,
            clip_output_negative=bool(clip_output_negative),
            cupy=cupy,
        )

    return _apply_deconvolution_blocks(
        image,
        spatial_ndim,
        restore_block,
        iterations=iteration_count,
        progress=progress,
        progress_message="Richardson-Lucy deconvolution",
        cupy=cupy,
    )


def _deconvolution_inputs(inputs, *, cupy):
    try:
        image, psf = list(inputs)[:2]
    except Exception as exc:
        raise ValueError("Deconvolution requires two inputs: Image and PSF.") from exc
    if image is None or psf is None:
        raise ValueError("Deconvolution requires connected Image and PSF inputs.")
    return cupy.asarray(image), cupy.asarray(psf)


def _resolved_deconvolution_spatial_ndim(
    image_ndim: int,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
) -> int:
    requested = _spatial_mode_dimension(spatial_mode)
    if requested is None and resolved_spatial_ndim is not None:
        requested = _validated_resolved_spatial_ndim(resolved_spatial_ndim)
    if requested is None:
        if image_ndim > 2:
            raise ValueError(
                "Auto from axes requires explicit axis semantics. Supply "
                "resolved_spatial_ndim or select an explicit 2D/3D mode."
            )
        requested = max(image_ndim, 1)
    if requested > max(image_ndim, 1):
        raise ValueError(
            f"{requested}D spatial processing cannot be applied to a "
            f"{image_ndim}D array."
        )
    if requested not in {2, 3}:
        raise ValueError("Deconvolution requires 2D YX or 3D ZYX spatial processing.")
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


def _deconvolution_psf(
    psf,
    spatial_ndim: int,
    *,
    normalize_psf: bool,
    cupy,
):
    kernel = cupy.asarray(psf, dtype=cupy.float32)
    if kernel.ndim != spatial_ndim:
        raise ValueError(
            f"PSF dimensionality ({kernel.ndim}D) must match the resolved "
            f"spatial dimensionality ({spatial_ndim}D)."
        )
    if kernel.size == 0 or any(size <= 0 for size in kernel.shape):
        raise ValueError("PSF is empty.")
    kernel = cupy.nan_to_num(
        kernel,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    kernel = cupy.maximum(kernel, cupy.float32(0.0))
    total = cupy.sum(kernel, dtype=cupy.float64)
    if not _device_scalar_bool(cupy.isfinite(total)) or not (
        _device_scalar_float(total) > 1e-12
    ):
        raise ValueError(
            "PSF is empty or invalid after cleaning; sum is below the "
            "minimum valid threshold."
        )
    maximum = cupy.max(kernel)
    if not _device_scalar_bool(cupy.isfinite(maximum)):
        raise ValueError("PSF maximum is not finite after cleaning.")
    if normalize_psf:
        kernel = kernel / total.astype(cupy.float32)
    return cupy.ascontiguousarray(kernel.astype(cupy.float32, copy=False))


def _device_scalar_bool(value) -> bool:
    """Read only a validation scalar; image and PSF buffers stay resident."""

    return bool(value.item())


def _device_scalar_float(value) -> float:
    """Read only a validation scalar; image and PSF buffers stay resident."""

    return float(value.item())


def _deconvolution_observed_block(
    block,
    *,
    clip_negative_input: bool,
    preserve_input_scale: bool,
    cupy,
):
    values = cupy.asarray(block, dtype=cupy.float32)
    finite = cupy.isfinite(values)
    finite_maximum = cupy.max(cupy.where(finite, values, -cupy.inf))
    any_finite = cupy.any(finite)
    posinf_value = cupy.where(
        any_finite,
        finite_maximum,
        cupy.float32(0.0),
    )
    posinf_value = cupy.maximum(posinf_value, cupy.float32(0.0))
    values = cupy.nan_to_num(
        values,
        nan=0.0,
        posinf=posinf_value,
        neginf=0.0,
    ).astype(cupy.float32, copy=False)
    if clip_negative_input:
        values = cupy.maximum(values, cupy.float32(0.0))

    scale = cupy.max(values)
    positive_scale = cupy.isfinite(scale) & (scale > cupy.float32(0.0))
    safe_scale = cupy.where(positive_scale, scale, cupy.float32(1.0))
    values = cupy.where(positive_scale, values, cupy.zeros_like(values))
    if preserve_input_scale:
        values = values / safe_scale.astype(cupy.float32)
        output_scale = safe_scale
    else:
        output_scale = cupy.float32(1.0)
    return values.astype(cupy.float32, copy=False), output_scale


def _deconvolution_output_block(
    restored,
    *,
    output_scale,
    clip_output_negative: bool,
    cupy,
):
    output = restored.astype(cupy.float32, copy=False) * cupy.asarray(
        output_scale,
        dtype=cupy.float32,
    )
    cupy.nan_to_num(
        output,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    if clip_output_negative:
        cupy.maximum(output, cupy.float32(0.0), out=output)
    return output.astype(cupy.float32, copy=False)


def _apply_deconvolution_blocks(
    image,
    spatial_ndim: int,
    block_function,
    *,
    iterations: int,
    progress,
    progress_message: str,
    cupy,
):
    block_count = _spatial_block_count(image.shape, spatial_ndim)
    total = max(block_count * int(iterations), 1)
    completed = 0
    message = str(progress_message)
    if progress is not None:
        progress.report(0, total, message)

    def iteration_done() -> None:
        nonlocal completed
        # CuPy kernels are asynchronous.  Synchronizing here makes an emitted
        # iteration milestone truthful and surfaces failures before progress
        # advances or cancellation unwinds the operation.
        cupy.cuda.get_current_stream().synchronize()
        progress.check_cancelled()
        completed += 1
        progress.report(completed, total, message)

    if image.ndim <= spatial_ndim:
        if progress is not None:
            progress.check_cancelled()
        restored = block_function(
            image,
            iteration_done if progress is not None else None,
        )
        return cupy.ascontiguousarray(restored.astype(cupy.float32, copy=False))

    result = cupy.empty(image.shape, dtype=cupy.float32)
    leading_shape = image.shape[: image.ndim - spatial_ndim]
    for index in _ndindex(leading_shape):
        if progress is not None:
            progress.check_cancelled()
        result[index] = block_function(
            image[index],
            iteration_done if progress is not None else None,
        )
    return cupy.ascontiguousarray(result)


def _spatial_block_count(shape: tuple[int, ...], spatial_ndim: int) -> int:
    if len(shape) <= spatial_ndim:
        return 1
    count = 1
    for size in shape[: len(shape) - spatial_ndim]:
        count *= int(size)
    return count


def _ndindex(shape: tuple[int, ...]):
    return product(*(range(int(size)) for size in shape))


def _richardson_lucy_block(
    image,
    psf,
    *,
    iterations: int,
    filter_epsilon: float,
    iteration_done,
    check_cancelled,
    sanitize_each_iteration: bool,
    cupy,
    signal,
):
    estimate = cupy.full(image.shape, cupy.float32(0.5), dtype=cupy.float32)
    psf_mirror = cupy.ascontiguousarray(cupy.flip(psf))
    epsilon = cupy.float32(1e-12)
    threshold = float(filter_epsilon)
    # The CPU operation uses its native, progress-aware loop when a progress
    # context exists.  Its no-progress skimage path treats any nonzero epsilon
    # as enabled, while the native loop enables only positive thresholds.
    use_threshold = threshold > 0 if sanitize_each_iteration else bool(threshold)

    for _ in range(int(iterations)):
        if check_cancelled is not None:
            check_cancelled()
        blurred = signal.convolve(estimate, psf, mode="same", method="fft") + epsilon
        if use_threshold:
            relative_blur = cupy.where(
                blurred < threshold,
                cupy.float32(0.0),
                image / blurred,
            )
        else:
            relative_blur = image / blurred
        correction = signal.convolve(
            relative_blur,
            psf_mirror,
            mode="same",
            method="fft",
        )
        estimate *= correction
        if sanitize_each_iteration:
            cupy.nan_to_num(
                estimate,
                copy=False,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            cupy.maximum(estimate, cupy.float32(0.0), out=estimate)
        if iteration_done is not None:
            iteration_done()
    return estimate.astype(cupy.float32, copy=False)


__all__ = ["richardson_lucy_deconvolution"]
