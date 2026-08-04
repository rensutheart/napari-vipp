"""Authoritative CPU Richardson--Lucy and RL-TV implementations.

This operation-owned module contains the complete scientific contract shared
with the optional CuPy providers.  :mod:`napari_vipp.core.operations`
re-exports the two public callables so existing workflows and callers retain
their API while evidence can fingerprint this focused implementation boundary.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import signal
from skimage import restoration


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
) -> np.ndarray:
    """Restore an image with baseline Richardson-Lucy deconvolution."""

    image, psf = _deconvolution_inputs(inputs)
    image_arr = np.asarray(image)
    spatial_ndim = _resolved_deconvolution_spatial_ndim(
        image_arr,
        spatial_mode,
        resolved_spatial_ndim,
    )
    kernel = _deconvolution_psf(
        psf,
        spatial_ndim,
        normalize_psf=bool(normalize_psf),
    )
    iterations = max(int(iterations), 1)

    def restore_block(block: np.ndarray, iteration_done=None) -> np.ndarray:
        values, output_scale = _deconvolution_observed_block(
            block,
            clip_negative_input=bool(clip_negative_input),
            preserve_input_scale=bool(preserve_input_scale),
        )
        if progress is None:
            restored = restoration.richardson_lucy(
                values,
                kernel,
                num_iter=iterations,
                clip=False,
                filter_epsilon=float(filter_epsilon),
            )
        else:
            restored = _richardson_lucy_native_block(
                values,
                kernel,
                iterations=iterations,
                filter_epsilon=float(filter_epsilon),
                iteration_done=iteration_done,
                check_cancelled=(
                    progress.check_cancelled if progress is not None else None
                ),
            )
        return _deconvolution_output_block(
            restored,
            output_scale=output_scale,
            clip_output_negative=bool(clip_output_negative),
        )

    return _apply_deconvolution_blocks(
        image_arr,
        spatial_ndim,
        restore_block,
        iterations=iterations,
        progress=progress,
        progress_message="Richardson-Lucy deconvolution",
    )


def richardson_lucy_tv_deconvolution(
    inputs,
    spatial_mode: str = "Auto from axes",
    iterations: int = 25,
    tv_regularization: float = 0.002,
    tv_epsilon: float = 1e-6,
    normalize_psf: bool = True,
    clip_negative_input: bool = True,
    clip_output_negative: bool = True,
    preserve_input_scale: bool = True,
    filter_epsilon: float = 1e-12,
    denominator_floor: float = 0.05,
    resolved_spatial_ndim: int | None = None,
    progress=None,
) -> np.ndarray:
    """Restore an image with Richardson-Lucy total-variation deconvolution."""

    image, psf = _deconvolution_inputs(inputs)
    image_arr = np.asarray(image)
    spatial_ndim = _resolved_deconvolution_spatial_ndim(
        image_arr,
        spatial_mode,
        resolved_spatial_ndim,
    )
    kernel = _deconvolution_psf(
        psf,
        spatial_ndim,
        normalize_psf=bool(normalize_psf),
    )
    iterations = max(int(iterations), 1)

    def restore_block(block: np.ndarray, iteration_done=None) -> np.ndarray:
        values, output_scale = _deconvolution_observed_block(
            block,
            clip_negative_input=bool(clip_negative_input),
            preserve_input_scale=bool(preserve_input_scale),
        )
        restored = _richardson_lucy_tv_native_block(
            values,
            kernel,
            iterations=iterations,
            tv_regularization=max(float(tv_regularization), 0.0),
            tv_epsilon=max(float(tv_epsilon), 1e-12),
            filter_epsilon=float(filter_epsilon),
            denominator_floor=max(float(denominator_floor), 1e-6),
            iteration_done=iteration_done,
            check_cancelled=(
                progress.check_cancelled if progress is not None else None
            ),
        )
        return _deconvolution_output_block(
            restored,
            output_scale=output_scale,
            clip_output_negative=bool(clip_output_negative),
        )

    return _apply_deconvolution_blocks(
        image_arr,
        spatial_ndim,
        restore_block,
        iterations=iterations,
        progress=progress,
        progress_message="Richardson-Lucy TV deconvolution",
    )


def _deconvolution_inputs(inputs) -> tuple[np.ndarray, np.ndarray]:
    try:
        image, psf = list(inputs)[:2]
    except Exception as exc:
        raise ValueError("Deconvolution requires two inputs: Image and PSF.") from exc
    if image is None or psf is None:
        raise ValueError("Deconvolution requires connected Image and PSF inputs.")
    return np.asarray(image), np.asarray(psf)


def _resolved_deconvolution_spatial_ndim(
    arr: np.ndarray,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
) -> int:
    requested = _spatial_mode_dimension(spatial_mode)
    if requested is None and resolved_spatial_ndim is not None:
        requested = _validated_resolved_spatial_ndim(resolved_spatial_ndim)
    if requested is None:
        if arr.ndim > 2:
            raise ValueError(
                "Auto from axes requires explicit axis semantics. Supply "
                "resolved_spatial_ndim or select an explicit 2D/3D mode."
            )
        requested = max(arr.ndim, 1)
    if requested > max(arr.ndim, 1):
        raise ValueError(
            f"{requested}D spatial processing cannot be applied to a {arr.ndim}D array."
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
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
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
) -> np.ndarray:
    kernel = np.asarray(psf, dtype=np.float32)
    if kernel.ndim != spatial_ndim:
        raise ValueError(
            f"PSF dimensionality ({kernel.ndim}D) must match the resolved "
            f"spatial dimensionality ({spatial_ndim}D)."
        )
    if kernel.size == 0 or any(size <= 0 for size in kernel.shape):
        raise ValueError("PSF is empty.")
    kernel = np.nan_to_num(kernel, nan=0.0, posinf=0.0, neginf=0.0)
    kernel = np.maximum(kernel, 0.0)
    kernel = _validate_psf_sum(kernel, minimum_valid_sum=1e-12)
    if bool(normalize_psf):
        kernel = kernel / np.float32(kernel.sum(dtype=np.float64))
    return np.ascontiguousarray(kernel.astype(np.float32, copy=False))


def _validate_psf_sum(
    psf: np.ndarray,
    *,
    minimum_valid_sum: float,
) -> np.ndarray:
    total = float(np.sum(psf, dtype=np.float64))
    if not np.isfinite(total) or total <= float(minimum_valid_sum):
        raise ValueError(
            "PSF is empty or invalid after cleaning; sum is below the "
            "minimum valid threshold."
        )
    if not np.isfinite(float(np.max(psf))):
        raise ValueError("PSF maximum is not finite after cleaning.")
    return psf


def _deconvolution_observed_block(
    block: np.ndarray,
    *,
    clip_negative_input: bool,
    preserve_input_scale: bool,
) -> tuple[np.ndarray, float]:
    values = np.asarray(block, dtype=np.float32)
    finite = values[np.isfinite(values)]
    posinf_value = float(finite.max()) if finite.size else 0.0
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=max(posinf_value, 0.0),
        neginf=0.0,
    ).astype(np.float32, copy=False)
    if bool(clip_negative_input):
        values = np.maximum(values, 0.0)
    finite = values[np.isfinite(values)]
    scale = float(finite.max()) if finite.size else 0.0
    if not np.isfinite(scale) or scale <= 0.0:
        return np.zeros_like(values, dtype=np.float32), 1.0
    if bool(preserve_input_scale):
        return (values / np.float32(scale)).astype(np.float32, copy=False), scale
    return values.astype(np.float32, copy=False), 1.0


def _deconvolution_output_block(
    restored: np.ndarray,
    *,
    output_scale: float,
    clip_output_negative: bool,
) -> np.ndarray:
    output = np.asarray(restored, dtype=np.float32) * np.float32(output_scale)
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)
    if bool(clip_output_negative):
        output = np.maximum(output, 0.0)
    return output.astype(np.float32, copy=False)


def _apply_deconvolution_blocks(
    arr: np.ndarray,
    spatial_ndim: int,
    block_func: Callable[[np.ndarray, Callable[[], None] | None], np.ndarray],
    *,
    iterations: int,
    progress=None,
    progress_message: str,
) -> np.ndarray:
    arr = np.asarray(arr)
    block_count = _spatial_block_count(arr, spatial_ndim)
    total = max(block_count * int(iterations), 1)
    completed = 0
    if progress is not None:
        progress.report(0, total, progress_message)

    def iteration_done() -> None:
        nonlocal completed
        completed += 1
        if progress is not None:
            progress.report(completed, total, progress_message)

    if arr.ndim <= spatial_ndim:
        if progress is not None:
            progress.check_cancelled()
        return np.ascontiguousarray(
            block_func(arr, iteration_done if progress is not None else None)
        )

    result = np.empty(arr.shape, dtype=np.float32)
    leading_shape = arr.shape[: arr.ndim - spatial_ndim]
    for index in np.ndindex(leading_shape):
        if progress is not None:
            progress.check_cancelled()
        result[index] = block_func(
            arr[index],
            iteration_done if progress is not None else None,
        )
    return np.ascontiguousarray(result)


def _spatial_block_count(arr: np.ndarray, spatial_ndim: int) -> int:
    arr = np.asarray(arr)
    spatial_ndim = int(np.clip(spatial_ndim, 1, max(arr.ndim, 1)))
    if arr.ndim <= spatial_ndim:
        return 1
    return int(np.prod(arr.shape[: arr.ndim - spatial_ndim], dtype=np.int64))


def _richardson_lucy_native_block(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    iterations: int,
    filter_epsilon: float,
    iteration_done: Callable[[], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> np.ndarray:
    estimate = np.full(image.shape, 0.5, dtype=np.float32)
    psf_mirror = np.flip(psf)
    eps = np.float32(1e-12)
    filter_epsilon = float(filter_epsilon)
    for _ in range(int(iterations)):
        if check_cancelled is not None:
            check_cancelled()
        blurred = signal.convolve(estimate, psf, mode="same") + eps
        if filter_epsilon > 0:
            relative_blur = np.where(blurred < filter_epsilon, 0.0, image / blurred)
        else:
            relative_blur = image / blurred
        estimate *= signal.convolve(relative_blur, psf_mirror, mode="same")
        estimate = np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)
        estimate = np.maximum(estimate, 0.0).astype(np.float32, copy=False)
        if iteration_done is not None:
            iteration_done()
    return estimate.astype(np.float32, copy=False)


def _richardson_lucy_tv_native_block(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    iterations: int,
    tv_regularization: float,
    tv_epsilon: float,
    filter_epsilon: float,
    denominator_floor: float,
    iteration_done: Callable[[], None] | None = None,
    check_cancelled: Callable[[], None] | None = None,
) -> np.ndarray:
    estimate = np.full(image.shape, 0.5, dtype=np.float32)
    psf_mirror = np.flip(psf)
    eps = np.float32(1e-12)
    filter_epsilon = float(filter_epsilon)
    for _ in range(int(iterations)):
        if check_cancelled is not None:
            check_cancelled()
        blurred = signal.convolve(estimate, psf, mode="same") + eps
        if filter_epsilon > 0:
            relative_blur = np.where(blurred < filter_epsilon, 0.0, image / blurred)
        else:
            relative_blur = image / blurred
        correction = signal.convolve(relative_blur, psf_mirror, mode="same")
        if tv_regularization > 0:
            tv = _tv_divergence(estimate, epsilon=tv_epsilon)
            denom = np.maximum(
                1.0 - np.float32(tv_regularization) * tv,
                np.float32(denominator_floor),
            )
            estimate = estimate * correction / denom
        else:
            estimate *= correction
        estimate = np.nan_to_num(estimate, nan=0.0, posinf=0.0, neginf=0.0)
        estimate = np.maximum(estimate, 0.0).astype(np.float32, copy=False)
        if iteration_done is not None:
            iteration_done()
    return estimate.astype(np.float32, copy=False)


def _tv_divergence(values: np.ndarray, *, epsilon: float) -> np.ndarray:
    gradients = np.gradient(values.astype(np.float32, copy=False))
    norm = np.sqrt(
        np.sum(
            np.stack([gradient * gradient for gradient in gradients], axis=0),
            axis=0,
            dtype=np.float32,
        )
        + np.float32(epsilon) ** 2
    )
    normalized = [gradient / norm for gradient in gradients]
    divergence = np.zeros(values.shape, dtype=np.float32)
    for axis, component in enumerate(normalized):
        divergence += np.gradient(component.astype(np.float32, copy=False), axis=axis)
    return np.nan_to_num(divergence, nan=0.0, posinf=0.0, neginf=0.0).astype(
        np.float32,
        copy=False,
    )


__all__ = [
    "richardson_lucy_deconvolution",
    "richardson_lucy_tv_deconvolution",
]
