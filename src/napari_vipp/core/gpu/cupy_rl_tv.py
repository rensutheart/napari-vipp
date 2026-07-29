"""Device-resident CuPyX Richardson--Lucy total-variation deconvolution.

This provider deliberately mirrors VIPP's authoritative CPU RL-TV loop.  It
reuses the ordinary Richardson--Lucy input, PSF, block, progress, and output
substrate while preserving the shipped TV sign, central-difference stencil,
constant initialization, zero-extension convolution, numerical guards, and
float32 output contract.

Optional CUDA modules remain lazily imported by :mod:`cupy_rl`; importing this
module is therefore safe on CPU-only installations.
"""

from __future__ import annotations

from .cupy_rl import (
    _apply_deconvolution_blocks,
    _cupy_modules,
    _deconvolution_inputs,
    _deconvolution_observed_block,
    _deconvolution_output_block,
    _deconvolution_psf,
    _resolved_deconvolution_spatial_ndim,
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
):
    """Restore resident float32 blocks with VIPP's RL-TV update.

    ``inputs`` follows the established ordered ``[Image, PSF]`` contract.  No
    dtype conversion, PSF preparation, spacing correction, or alternative TV
    discretization is introduced by accelerator selection.
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
    regularization = max(float(tv_regularization), 0.0)
    gradient_epsilon = max(float(tv_epsilon), 1e-12)
    ratio_epsilon = float(filter_epsilon)
    floor = max(float(denominator_floor), 1e-6)

    def restore_block(block, iteration_done=None):
        values, output_scale = _deconvolution_observed_block(
            block,
            clip_negative_input=bool(clip_negative_input),
            preserve_input_scale=bool(preserve_input_scale),
            cupy=cupy,
        )
        restored = _richardson_lucy_tv_block(
            values,
            kernel,
            iterations=iteration_count,
            tv_regularization=regularization,
            tv_epsilon=gradient_epsilon,
            filter_epsilon=ratio_epsilon,
            denominator_floor=floor,
            iteration_done=iteration_done,
            check_cancelled=(
                progress.check_cancelled if progress is not None else None
            ),
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
        progress_message="Richardson-Lucy TV deconvolution",
        cupy=cupy,
    )


def _richardson_lucy_tv_block(
    image,
    psf,
    *,
    iterations: int,
    tv_regularization: float,
    tv_epsilon: float,
    filter_epsilon: float,
    denominator_floor: float,
    iteration_done,
    check_cancelled,
    diagnostics_observer=None,
    cupy,
    signal,
):
    """Apply the production RL-TV recurrence to one spatial block."""

    estimate = cupy.full(image.shape, cupy.float32(0.5), dtype=cupy.float32)
    psf_mirror = cupy.ascontiguousarray(cupy.flip(psf))
    epsilon = cupy.float32(1e-12)
    threshold = float(filter_epsilon)

    for iteration_index in range(int(iterations)):
        if check_cancelled is not None:
            check_cancelled()
        blurred = signal.convolve(estimate, psf, mode="same", method="fft") + epsilon
        if threshold > 0:
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
        if tv_regularization > 0:
            tv = _tv_divergence(estimate, epsilon=tv_epsilon, cupy=cupy)
            raw_denominator = cupy.float32(1.0) - cupy.float32(tv_regularization) * tv
            if diagnostics_observer is not None:
                floor_value = cupy.float32(denominator_floor)
                diagnostics_observer(
                    iteration_index,
                    float(cupy.min(raw_denominator).item()),
                    float(cupy.mean(raw_denominator < floor_value).item()),
                )
            denominator = cupy.maximum(
                raw_denominator,
                cupy.float32(denominator_floor),
            )
            estimate = estimate * correction / denominator
        else:
            estimate *= correction
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


def _tv_divergence(values, *, epsilon: float, cupy):
    """Reproduce the CPU central-gradient divergence in float32."""

    gradients = cupy.gradient(values.astype(cupy.float32, copy=False))
    norm = cupy.sqrt(
        cupy.sum(
            cupy.stack(
                [gradient * gradient for gradient in gradients],
                axis=0,
            ),
            axis=0,
            dtype=cupy.float32,
        )
        + cupy.float32(epsilon) ** 2
    )
    normalized = [gradient / norm for gradient in gradients]
    divergence = cupy.zeros(values.shape, dtype=cupy.float32)
    for axis, component in enumerate(normalized):
        divergence += cupy.gradient(
            component.astype(cupy.float32, copy=False),
            axis=axis,
        )
    cupy.nan_to_num(
        divergence,
        copy=False,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    return divergence.astype(cupy.float32, copy=False)


__all__ = ["richardson_lucy_tv_deconvolution"]
