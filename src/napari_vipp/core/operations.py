"""Prototype image-processing operations."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations, product
from numbers import Integral
from pathlib import Path

import numpy as np
from scipy import integrate, signal, special
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull, QhullError
from skimage import (
    feature,
    filters,
    measure,
    morphology,
    restoration,
    segmentation,
    transform,
)

from napari_vipp.core.channel_colors import channel_color_table, color_value_to_rgb
from napari_vipp.core.io import write_image
from napari_vipp.core.tables import TableData, table_from_columns

NO_TABLE_COLUMNS_VALUE = "__none__"

RGB_CHANNELS = (3, 4)

COMPOSITE_RGB_PRESERVE_VALUES = "Preserve numeric values"
COMPOSITE_RGB_PERCENTILE_1_99 = "Per-channel 1st-99th percentile (lossy)"

BORN_WOLF_PSF_AUTO_PARAMETERS = (
    "wavelength_nm",
    "numerical_aperture",
    "refractive_index",
    "pixel_size_xy_um",
    "z_step_um",
)
BORN_WOLF_PSF_MANUAL_DEFAULTS = {
    "wavelength_nm": 520.0,
    "numerical_aperture": 1.4,
    "refractive_index": 1.515,
    "pixel_size_xy_um": 0.1,
    "z_step_um": 0.3,
    "channel": 0,
}

_GLOBAL_THRESHOLD_HISTOGRAM_BINS = 256
_THRESHOLD_CHUNK_SIZE = 1_048_576
_MAX_NATIVE_INTEGER_HISTOGRAM_BINS = 65_536
_FLOAT64_EXACT_INTEGER_LIMIT = 2**53


@dataclass(frozen=True)
class BornWolfPsfParameterResolution:
    name: str
    value: float | int | None
    source: str
    message: str
    required: bool = True

    @property
    def resolved(self) -> bool:
        return self.value is not None


@dataclass(frozen=True)
class BornWolfPsfResolution:
    spatial_ndim: int
    values: dict[str, float | int]
    parameters: dict[str, BornWolfPsfParameterResolution]
    unresolved: tuple[str, ...]


def crop_stack(
    data,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    channel_axis: int | None = None,
    axis_names: Sequence[str] = (),
) -> np.ndarray:
    """Crop declared Y/X axes, or trailing scalar axes without metadata.

    Arrays are scalar by default. In particular, a trailing dimension of
    length 3 or 4 is treated as X unless ``channel_axis`` is supplied.
    """
    arr = np.asarray(data)
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Crop stack",
    )
    if arr.ndim < 2:
        if any(value != 0 for value in (top, bottom, left, right)):
            raise ValueError("Crop stack requires at least two spatial axes.")
        return arr.copy()

    y_axis, x_axis = _xy_axes(
        arr,
        channel_axis=channel_axis,
        axis_names=axis_names,
    )
    slices = [slice(None)] * arr.ndim
    top, bottom = _crop_pair(top, bottom, arr.shape[y_axis])
    left, right = _crop_pair(left, right, arr.shape[x_axis])
    slices[y_axis] = slice(top, arr.shape[y_axis] - bottom)
    slices[x_axis] = slice(left, arr.shape[x_axis] - right)
    return np.ascontiguousarray(arr[tuple(slices)])


def reorder_axes(
    data,
    order: str = "",
    axis_names: Sequence[str] = (),
) -> np.ndarray:
    """Transpose an array to a full output axis order."""
    arr = np.asarray(data)
    indices = _axis_order_indices(order, arr.ndim, axis_names)
    if indices is None:
        raise ValueError(
            "Reorder Axes order must be a complete numeric permutation or a "
            "complete set of declared axis names."
        )
    return np.ascontiguousarray(np.transpose(arr, indices))


def linear_scale_offset(data, alpha: float = 3.0, beta: float = 1.0) -> np.ndarray:
    """Apply a linear scale-and-offset contrast operation."""
    arr = np.asarray(data)
    alpha = _finite_float_parameter(alpha, "Linear Scale + Offset alpha")
    beta = _finite_float_parameter(beta, "Linear Scale + Offset beta")
    if arr.dtype == bool:
        return arr.copy()
    scaled = arr.astype(np.float32, copy=False) * alpha + beta
    return _restore_numeric_dtype(scaled, arr)


def gamma_correction(data, gamma: float = 0.5) -> np.ndarray:
    """Apply power-law gamma correction."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return arr.copy()

    gamma = _finite_float_parameter(gamma, "Gamma Correction gamma")
    if gamma <= 0.0:
        raise ValueError("Gamma Correction gamma must be greater than zero.")
    if not (
        np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError("Gamma Correction requires real-valued image data.")
    finite = arr[np.isfinite(arr)]
    if finite.size != arr.size:
        raise ValueError("Gamma Correction input must contain only finite values.")
    if finite.size and finite.min() < 0:
        raise ValueError("Gamma Correction input values must be non-negative.")
    scale = _intensity_scale(arr)
    corrected = np.power(np.clip(arr.astype(np.float32) / scale, 0, 1), gamma) * scale
    return _restore_numeric_dtype(corrected, arr)


def _finite_float_parameter(value, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    return parsed


def average_blur(
    data,
    size: int = 5,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Apply a slice-wise mean blur over explicitly resolved X/Y axes.

    Arrays are scalar by default, including trailing dimensions of length 3
    or 4. A declared channel axis is preserved and excluded from filtering.
    """
    arr = _float_if_bool(np.asarray(data))
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Average blur",
    )
    size = max(int(size), 1)
    filter_size = [1] * arr.ndim
    for axis in _xy_axes(arr, channel_axis=channel_axis):
        filter_size[axis] = size
    return ndi.uniform_filter(arr, size=filter_size)


def gaussian_blur(
    data,
    sigma: float = 1.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Apply slice-wise Gaussian blur over explicitly resolved X/Y axes.

    Arrays are scalar by default, including trailing dimensions of length 3
    or 4. A declared channel axis is preserved and excluded from filtering.
    """
    arr = _float_if_bool(np.asarray(data))
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Gaussian blur",
    )
    sigma = max(float(sigma), 0.0)
    if sigma == 0:
        return arr.copy()

    sigma_by_axis = [0.0] * arr.ndim
    for axis in _xy_axes(arr, channel_axis=channel_axis):
        sigma_by_axis[axis] = sigma
    return ndi.gaussian_filter(arr, sigma=sigma_by_axis)


def gaussian_blur_3d(
    data,
    sigma_z: float = 2.0,
    sigma_y: float = 2.0,
    sigma_x: float = 2.0,
    lock_xy: bool = True,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Apply Gaussian blur across resolved Z/Y/X volume axes.

    Arrays are scalar by default, including trailing dimensions of length 3
    or 4. A declared channel axis is preserved and excluded from filtering.
    """
    del lock_xy  # UI convenience flag; sigma_y/sigma_x carry the actual values.
    arr = _float_if_bool(np.asarray(data))
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Gaussian blur 3D",
    )
    spatial_axes = _spatial_axes(arr, channel_axis=channel_axis)
    if not spatial_axes:
        return arr.copy()

    sigma_by_axis = [0.0] * arr.ndim
    values = [
        max(float(sigma_z), 0.0),
        max(float(sigma_y), 0.0),
        max(float(sigma_x), 0.0),
    ]
    active_axes = spatial_axes[-3:]
    active_values = values[-len(active_axes) :]
    for axis, value in zip(active_axes, active_values, strict=False):
        sigma_by_axis[axis] = value
    if not any(sigma_by_axis):
        return arr.copy()
    return ndi.gaussian_filter(arr, sigma=sigma_by_axis)


def born_wolf_psf(
    data,
    spatial_mode: str = "Auto from axes",
    auto_parameters: bool = True,
    wavelength_nm: float = 0.0,
    numerical_aperture: float = 0.0,
    refractive_index: float = 0.0,
    pixel_size_xy_um: float = 0.0,
    z_step_um: float = 0.0,
    xy_size: int = 65,
    z_size: int = 33,
    channel: int = -1,
    pupil_samples: int = 256,
    normalize: bool = True,
    resolved_spatial_ndim: int | None = None,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    axis_scales: Sequence[float] = (),
    axis_units: Sequence[str | None] = (),
    channel_emission_wavelengths: Sequence[float | None] = (),
    channel_emission_wavelength_units: Sequence[str | None] = (),
    channel_excitation_wavelengths: Sequence[float | None] = (),
    channel_excitation_wavelength_units: Sequence[str | None] = (),
    objective_lens_na: float | None = None,
    objective_refractive_index: float | None = None,
) -> np.ndarray:
    """Generate a scalar Born-Wolf point-spread function from image metadata."""
    shape = tuple(int(size) for size in getattr(data, "shape", ()) or ())
    resolution = resolve_born_wolf_psf_parameters(
        shape,
        spatial_mode,
        auto_parameters=auto_parameters,
        wavelength_nm=wavelength_nm,
        numerical_aperture=numerical_aperture,
        refractive_index=refractive_index,
        pixel_size_xy_um=pixel_size_xy_um,
        z_step_um=z_step_um,
        channel=channel,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_types=axis_types,
        axis_names=axis_names,
        axis_scales=axis_scales,
        axis_units=axis_units,
        channel_emission_wavelengths=channel_emission_wavelengths,
        channel_emission_wavelength_units=channel_emission_wavelength_units,
        channel_excitation_wavelengths=channel_excitation_wavelengths,
        channel_excitation_wavelength_units=channel_excitation_wavelength_units,
        objective_lens_na=objective_lens_na,
        objective_refractive_index=objective_refractive_index,
    )
    if resolution.unresolved:
        unresolved = ", ".join(resolution.unresolved)
        mode = "auto" if bool(auto_parameters) else "manual"
        raise ValueError(
            f"Cannot generate Born-Wolf PSF in {mode} mode; unresolved "
            f"parameter(s): {unresolved}."
        )
    return _born_wolf_psf_from_resolution(
        resolution,
        xy_size=xy_size,
        z_size=z_size,
        pupil_samples=pupil_samples,
        normalize=normalize,
    )


def born_wolf_psf_outputs(
    data,
    spatial_mode: str = "Auto from axes",
    auto_parameters: bool = True,
    wavelength_nm: float = 0.0,
    numerical_aperture: float = 0.0,
    refractive_index: float = 0.0,
    pixel_size_xy_um: float = 0.0,
    z_step_um: float = 0.0,
    xy_size: int = 65,
    z_size: int = 33,
    channel: int = -1,
    pupil_samples: int = 256,
    normalize: bool = True,
    resolved_spatial_ndim: int | None = None,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    axis_scales: Sequence[float] = (),
    axis_units: Sequence[str | None] = (),
    channel_emission_wavelengths: Sequence[float | None] = (),
    channel_emission_wavelength_units: Sequence[str | None] = (),
    channel_excitation_wavelengths: Sequence[float | None] = (),
    channel_excitation_wavelength_units: Sequence[str | None] = (),
    objective_lens_na: float | None = None,
    objective_refractive_index: float | None = None,
) -> tuple[np.ndarray, ...]:
    """Generate one Born-Wolf PSF per requested channel for graph nodes."""
    channel_indices = _born_wolf_psf_output_channel_indices(
        channel,
        auto_parameters=auto_parameters,
        channel_emission_wavelengths=channel_emission_wavelengths,
        channel_excitation_wavelengths=channel_excitation_wavelengths,
    )
    return tuple(
        born_wolf_psf(
            data,
            spatial_mode=spatial_mode,
            auto_parameters=auto_parameters,
            wavelength_nm=wavelength_nm,
            numerical_aperture=numerical_aperture,
            refractive_index=refractive_index,
            pixel_size_xy_um=pixel_size_xy_um,
            z_step_um=z_step_um,
            xy_size=xy_size,
            z_size=z_size,
            channel=channel_index,
            pupil_samples=pupil_samples,
            normalize=normalize,
            resolved_spatial_ndim=resolved_spatial_ndim,
            axis_names=axis_names,
            axis_types=axis_types,
            axis_scales=axis_scales,
            axis_units=axis_units,
            channel_emission_wavelengths=channel_emission_wavelengths,
            channel_emission_wavelength_units=channel_emission_wavelength_units,
            channel_excitation_wavelengths=channel_excitation_wavelengths,
            channel_excitation_wavelength_units=channel_excitation_wavelength_units,
            objective_lens_na=objective_lens_na,
            objective_refractive_index=objective_refractive_index,
        )
        for channel_index in channel_indices
    )


def _born_wolf_psf_output_channel_indices(
    channel: int,
    *,
    auto_parameters: bool,
    channel_emission_wavelengths: Sequence[float | None],
    channel_excitation_wavelengths: Sequence[float | None],
) -> tuple[int, ...]:
    try:
        requested = int(channel)
    except Exception:
        requested = -1
    count = _psf_channel_count(
        channel_emission_wavelengths,
        channel_excitation_wavelengths,
    )
    if bool(auto_parameters) and requested < 0 and count > 1:
        return tuple(range(count))
    return (requested,)


def _born_wolf_psf_from_resolution(
    resolution: BornWolfPsfResolution,
    *,
    xy_size: int,
    z_size: int,
    pupil_samples: int,
    normalize: bool,
) -> np.ndarray:
    spatial_ndim = resolution.spatial_ndim
    xy_size = _odd_size(xy_size, minimum=9, maximum=1025)
    z_size = _odd_size(z_size, minimum=1, maximum=1025)
    if spatial_ndim < 3:
        z_size = 1

    wavelength_nm = float(resolution.values["wavelength_nm"])
    wavelength_um = max(wavelength_nm / 1000.0, 1e-6)
    numerical_aperture = float(resolution.values["numerical_aperture"])
    refractive_index = float(resolution.values["refractive_index"])
    if numerical_aperture >= refractive_index:
        raise ValueError(
            "Born-Wolf PSF numerical aperture must be smaller than the "
            "refractive index."
        )
    pixel_size_xy_um = float(resolution.values["pixel_size_xy_um"])
    z_step_um = float(resolution.values["z_step_um"])

    y_coords = (np.arange(xy_size, dtype=np.float64) - xy_size // 2) * pixel_size_xy_um
    x_coords = (np.arange(xy_size, dtype=np.float64) - xy_size // 2) * pixel_size_xy_um
    yy, xx = np.meshgrid(y_coords, x_coords, indexing="ij")
    radius_um = np.hypot(yy, xx)
    radial_phase = (2.0 * np.pi * numerical_aperture / wavelength_um) * radius_um

    z_coords = (
        np.arange(z_size, dtype=np.float64) - z_size // 2
    ) * z_step_um
    pupil_count = max(int(pupil_samples), 16)
    pupil = np.linspace(0.0, 1.0, pupil_count, dtype=np.float64)
    pupil_weight = pupil[None, None, :]
    bessel = special.j0(radial_phase[..., None] * pupil_weight)
    alpha = np.arcsin(np.clip(numerical_aperture / refractive_index, 0.0, 0.999999))
    defocus_scale = (
        8.0
        * np.pi
        * refractive_index
        * (np.sin(alpha / 2.0) ** 2)
        / wavelength_um
    )

    planes = []
    for z_um in z_coords:
        phase = np.exp(1j * defocus_scale * z_um * (pupil**2))[None, None, :]
        amplitude = 2.0 * integrate.trapezoid(
            bessel * phase * pupil_weight,
            pupil,
            axis=-1,
        )
        planes.append(np.abs(amplitude) ** 2)
    psf = np.stack(planes, axis=0).astype(np.float32, copy=False)
    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        total = float(psf.sum(dtype=np.float64))
        if total > 0:
            psf = psf / np.float32(total)
    if spatial_ndim < 3:
        return np.ascontiguousarray(psf[0])
    return np.ascontiguousarray(psf)


def prepare_validate_psf(
    data,
    center_mode: str = "Peak",
    clip_negatives: bool = True,
    normalize_sum: bool = True,
    minimum_valid_sum: float = 1e-12,
    force_odd_shape: bool = True,
    crop_empty_border: bool = False,
) -> np.ndarray:
    """Clean, center, and normalize a scalar 2D or 3D PSF kernel."""
    arr = np.asarray(data)
    if arr.ndim not in {2, 3}:
        raise ValueError(
            "Prepare / Validate PSF expects a scalar 2D YX or 3D ZYX PSF "
            "without time or channel axes."
        )
    if arr.size == 0 or any(size <= 0 for size in arr.shape):
        raise ValueError("PSF is empty.")

    minimum_valid_sum = max(float(minimum_valid_sum), 0.0)
    psf = arr.astype(np.float32, copy=False)
    psf = np.nan_to_num(psf, nan=0.0, posinf=0.0, neginf=0.0)
    if bool(clip_negatives):
        psf = np.maximum(psf, 0.0)
    if bool(crop_empty_border):
        psf = _crop_empty_psf_border(psf)
    if bool(force_odd_shape):
        psf = _force_odd_psf_shape(psf)
    psf = _center_psf(psf, center_mode, minimum_valid_sum=minimum_valid_sum)
    psf = _validate_psf_sum(psf, minimum_valid_sum=minimum_valid_sum)
    if bool(normalize_sum):
        psf = psf / np.float32(psf.sum(dtype=np.float64))
    return np.ascontiguousarray(psf.astype(np.float32, copy=False))


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


def median_filter(
    data,
    size: int = 5,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Apply a slice-wise median filter over explicitly resolved X/Y axes.

    Arrays are scalar by default, including trailing dimensions of length 3
    or 4. A declared channel axis is preserved and excluded from filtering.
    """
    arr = np.asarray(data)
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Median filter",
    )
    size = _odd_size(size, minimum=1)
    filter_size = [1] * arr.ndim
    for axis in _xy_axes(arr, channel_axis=channel_axis):
        filter_size[axis] = size
    return ndi.median_filter(arr, size=filter_size)


def bilateral_filter(
    data,
    diameter: int = 5,
    sigma_color: float = 5.0,
    sigma_space: float = 5.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Denoise planes with ``sigma_color`` in input intensity units.

    One whole-array affine intensity scale is shared by every plane and inverted
    on output. Data is scalar by default, even when its last dimension has length
    3 or 4. Set ``channel_axis`` explicitly for multichannel data. Spatial
    boundaries use reflection instead of artificial zeros.
    """
    arr = np.asarray(data)
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Bilateral filter",
    )
    diameter = _validated_odd_filter_size(
        diameter,
        operation="Bilateral filter",
        name="diameter",
        minimum=3,
    )
    sigma_color = _validated_filter_scale(
        sigma_color,
        operation="Bilateral filter",
        name="sigma_color",
        minimum=0.0,
        inclusive=False,
    )
    sigma_space = _validated_filter_scale(
        sigma_space,
        operation="Bilateral filter",
        name="sigma_space",
        minimum=0.0,
        inclusive=False,
    )
    unit_arr, intensity_scale = _to_float_unit(
        arr,
        operation="Bilateral filter",
    )
    unit_sigma_color = sigma_color / intensity_scale.span
    if not np.isfinite(unit_sigma_color) or unit_sigma_color <= 0.0:
        raise ValueError(
            "Bilateral filter sigma_color is too small for the input intensity span."
        )

    def filter_plane(plane: np.ndarray) -> np.ndarray:
        return restoration.denoise_bilateral(
            plane,
            win_size=diameter,
            sigma_color=unit_sigma_color,
            sigma_spatial=sigma_space,
            mode="reflect",
            channel_axis=-1 if channel_axis is not None else None,
        ).astype(unit_arr.dtype, copy=False)

    filtered = _apply_filter_plane_wise(
        unit_arr,
        filter_plane,
        channel_axis=channel_axis,
    )
    return _restore_unit_float_dtype(
        filtered,
        arr,
        intensity_scale,
        operation="Bilateral filter",
    )


def difference_of_gaussians_filter(
    data,
    low_sigma: float = 1.0,
    high_sigma: float = 3.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Enhance structures between ordered Gaussian scales on resolved X/Y.

    Arrays are scalar by default, including trailing dimensions of length 3
    or 4. A declared channel axis is preserved and excluded from filtering.
    """
    original = np.asarray(data)
    dtype = original.dtype if np.issubdtype(original.dtype, np.floating) else np.float32
    arr = _float_if_bool(original).astype(dtype, copy=False)
    low_sigma, high_sigma = _validated_sigma_pair(low_sigma, high_sigma)
    low = gaussian_blur(arr, sigma=low_sigma, channel_axis=channel_axis)
    high = gaussian_blur(arr, sigma=high_sigma, channel_axis=channel_axis)
    return low.astype(dtype, copy=False) - high.astype(dtype, copy=False)


def unsharp_mask_filter(
    data,
    radius: float = 1.0,
    amount: float = 1.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Sharpen planes while preserving the input float intensity units.

    One whole-array affine intensity scale is shared by every plane and inverted
    on output. Data is scalar by default, even when its last dimension has length
    3 or 4. Set ``channel_axis`` explicitly for multichannel data. Float
    sharpening overshoot is retained in the original units.
    """
    arr = np.asarray(data)
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Unsharp mask",
    )
    radius = _validated_filter_scale(
        radius,
        operation="Unsharp mask",
        name="radius",
        minimum=0.0,
    )
    amount = _validated_filter_scale(
        amount,
        operation="Unsharp mask",
        name="amount",
        minimum=0.0,
    )
    unit_arr, intensity_scale = _to_float_unit(
        arr,
        operation="Unsharp mask",
    )

    def sharpen_plane(plane: np.ndarray) -> np.ndarray:
        return filters.unsharp_mask(
            plane,
            radius=radius,
            amount=amount,
            channel_axis=-1 if channel_axis is not None else None,
            preserve_range=True,
        ).astype(unit_arr.dtype, copy=False)

    sharpened = _apply_filter_plane_wise(
        unit_arr,
        sharpen_plane,
        channel_axis=channel_axis,
    )
    return _restore_unit_float_dtype(
        sharpened,
        arr,
        intensity_scale,
        operation="Unsharp mask",
    )


def rolling_ball_background(
    data,
    radius: float = 50.0,
    light_background: bool = False,
    disable_smoothing: bool = False,
    spatial_mode: str = "2D YX",
    resolved_spatial_ndim: int | None = None,
    progress=None,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Estimate smooth background with a Fiji/ImageJ-style rolling ball.

    Arrays are scalar by default. Set ``channel_axis`` to process a declared
    multichannel array independently without inferring color from its shape.
    """
    arr = np.asarray(data)
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Rolling-ball background",
    )
    if arr.dtype == bool:
        return np.zeros_like(arr)
    background = _estimate_rolling_ball_background(
        arr,
        radius=radius,
        light_background=light_background,
        disable_smoothing=disable_smoothing,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        channel_axis=channel_axis,
        progress=progress,
    )
    return _restore_numeric_dtype(background, arr)


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
) -> np.ndarray:
    """Subtract a rolling-ball background estimate while preserving dtype.

    Arrays are scalar by default. Set ``channel_axis`` to process a declared
    multichannel array independently without inferring color from its shape.
    """
    arr = np.asarray(data)
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Subtract background",
    )
    if arr.dtype == bool:
        return arr.copy()
    background = _estimate_rolling_ball_background(
        arr,
        radius=radius,
        light_background=light_background,
        disable_smoothing=disable_smoothing,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        channel_axis=channel_axis,
        progress=progress,
    )
    values = arr.astype(background.dtype, copy=False)
    if bool(light_background):
        corrected = background - values
    else:
        corrected = values - background
    if bool(clip_negative):
        corrected = np.maximum(corrected, 0)
    return _restore_numeric_dtype(corrected, arr)


def non_local_means_filter(
    data,
    patch_size: int = 5,
    patch_distance: int = 6,
    h: float = 0.08,
    fast_mode: bool = True,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Denoise planes using one shared global intensity scale.

    ``h`` is expressed on the normalized global scale: 1.0 is the full integer
    dtype span, 1.0 for float data already in 0..1, or the whole-array finite
    min-to-max span for other float data. Data is scalar by default, including
    trailing dimensions of length 3 or 4. Set ``channel_axis`` explicitly for
    multichannel data. Output is restored to input units.
    """
    arr = np.asarray(data)
    channel_axis = _validated_filter_channel_axis(
        channel_axis,
        arr.ndim,
        operation="Non-local means",
    )
    patch_size = _validated_odd_filter_size(
        patch_size,
        operation="Non-local means",
        name="patch_size",
        minimum=3,
    )
    patch_distance = _validated_filter_integer(
        patch_distance,
        operation="Non-local means",
        name="patch_distance",
        minimum=1,
    )
    h = _validated_filter_scale(
        h,
        operation="Non-local means",
        name="h",
        minimum=0.0,
    )
    unit_arr, intensity_scale = _to_float_unit(
        arr,
        operation="Non-local means",
    )

    def denoise_plane(plane: np.ndarray) -> np.ndarray:
        return restoration.denoise_nl_means(
            plane,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
            fast_mode=bool(fast_mode),
            preserve_range=True,
            channel_axis=-1 if channel_axis is not None else None,
        ).astype(unit_arr.dtype, copy=False)

    denoised = _apply_filter_plane_wise(
        unit_arr,
        denoise_plane,
        channel_axis=channel_axis,
    )
    return _restore_unit_float_dtype(
        denoised,
        arr,
        intensity_scale,
        operation="Non-local means",
    )


def sobel_filter(data, channel_axis: int | None = None) -> np.ndarray:
    """Return the slice-wise Sobel edge magnitude of scalar image planes.

    Data is scalar by default. An explicitly declared RGB/RGBA axis is reduced
    to BT.601 luma before filtering; the channel axis is removed from the output.
    """
    original = np.asarray(data)
    arr = _to_explicit_grayscale(
        original,
        channel_axis=channel_axis,
        operation="Sobel filter",
    )
    result = _apply_scalar_plane_wise(
        arr,
        lambda plane: filters.sobel(plane.astype(np.float32, copy=False)),
    )
    return _restore_numeric_dtype(result, original)


def laplace_filter(
    data,
    kernel_size: int = 3,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a slice-wise scalar Laplace edge/detail response.

    Data is scalar by default. An explicitly declared RGB/RGBA axis is reduced
    to BT.601 luma before filtering; the channel axis is removed from the output.
    """
    original = np.asarray(data)
    arr = _to_explicit_grayscale(
        original,
        channel_axis=channel_axis,
        operation="Laplace filter",
    )
    kernel_size = _odd_size(kernel_size, minimum=3)
    result = _apply_scalar_plane_wise(
        arr,
        lambda plane: filters.laplace(
            plane.astype(np.float32, copy=False),
            ksize=kernel_size,
        ),
    )
    if np.issubdtype(original.dtype, np.floating):
        return result.astype(original.dtype, copy=False)
    return result.astype(np.float32, copy=False)


def canny_edges(
    data,
    sigma: float = 1.0,
    low_quantile: float = 0.1,
    high_quantile: float = 0.2,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a scalar-plane Canny mask from ordered 0..1 quantiles.

    Data is scalar by default. An explicitly declared RGB/RGBA axis is reduced
    to BT.601 luma before filtering; the channel axis is removed from the mask.
    """
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Canny",
    )
    low, high = _validated_threshold_pair(
        low_quantile,
        high_quantile,
        operation="Canny",
        minimum=0.0,
        maximum=1.0,
    )
    sigma = max(float(sigma), 0.0)

    def canny_plane(plane: np.ndarray) -> np.ndarray:
        values = plane.astype(np.float32, copy=False)
        return feature.canny(
            values,
            sigma=sigma,
            low_threshold=low,
            high_threshold=high,
            use_quantiles=True,
        )

    return _apply_scalar_plane_wise(arr, canny_plane)


def hysteresis_threshold(
    data,
    low_threshold: float = 0.25,
    high_threshold: float = 0.7,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a mask from finite intensity thresholds with low <= high.

    Data is scalar by default. An explicitly declared RGB/RGBA axis is reduced
    to BT.601 luma first; the channel axis is removed from the output mask.
    """
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Hysteresis",
    )
    low, high = _validated_threshold_pair(
        low_threshold,
        high_threshold,
        operation="Hysteresis",
    )
    spatial_ndim = _resolved_spatial_ndim(
        arr,
        spatial_mode,
        resolved_spatial_ndim,
    )

    def threshold_block(block: np.ndarray) -> np.ndarray:
        values = block.astype(np.float32, copy=False)
        return filters.apply_hysteresis_threshold(values, low, high)

    return _apply_spatial_blocks(
        arr,
        spatial_ndim,
        threshold_block,
        dtype=bool,
    )


def otsu_threshold(
    data,
    threshold_scope: str = "Stack histogram",
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return an Otsu mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Otsu threshold",
    )
    return _global_threshold(
        arr,
        lambda values: _otsu_value(values, histogram_bins),
        threshold_scope=threshold_scope,
    )


def triangle_threshold(
    data,
    threshold_scope: str = "Stack histogram",
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a triangle mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Triangle threshold",
    )
    return _global_threshold(
        arr,
        lambda values: _triangle_value(values, histogram_bins),
        threshold_scope=threshold_scope,
    )


def li_threshold(
    data,
    threshold_scope: str = "Stack histogram",
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a Li mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Li threshold",
    )
    return _global_threshold(
        arr,
        _li_value,
        threshold_scope=threshold_scope,
    )


def yen_threshold(
    data,
    threshold_scope: str = "Stack histogram",
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a Yen mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Yen threshold",
    )
    return _global_threshold(
        arr,
        lambda values: _yen_value(values, histogram_bins),
        threshold_scope=threshold_scope,
    )


def isodata_threshold(
    data,
    threshold_scope: str = "Stack histogram",
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return an Isodata mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Isodata threshold",
    )
    return _global_threshold(
        arr,
        lambda values: _isodata_value(values, histogram_bins),
        threshold_scope=threshold_scope,
    )


def minimum_threshold(
    data,
    threshold_scope: str = "Stack histogram",
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
    max_iterations: int = 10_000,
    progress=None,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a minimum mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Minimum threshold",
    )
    return _global_threshold(
        arr,
        lambda values: _minimum_value(
            values,
            histogram_bins,
            max_iterations=max_iterations,
            progress=progress,
        ),
        threshold_scope=threshold_scope,
    )


def automatic_threshold_value(
    data,
    operation_id: str,
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
    max_iterations: int = 10_000,
    progress=None,
    channel_axis: int | None = None,
) -> int | float | None:
    """Return a global threshold, optionally after declared RGB/RGBA reduction."""
    threshold_func = _automatic_threshold_function(operation_id)
    if threshold_func is None:
        return None
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Automatic threshold",
    )
    if arr.dtype == bool:
        return 0.5
    if str(operation_id) == "minimum_threshold":
        return _minimum_value(
            arr,
            histogram_bins,
            max_iterations=max_iterations,
            progress=progress,
        )
    return threshold_func(arr, histogram_bins)


def binary_threshold(
    data,
    threshold: float = 0.5,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a fixed mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Binary threshold",
    )
    return arr > float(threshold)


def adaptive_mean_threshold(
    data,
    block_size: int = 11,
    c: float = 2.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a local-mean mask, optionally reducing declared RGB/RGBA."""
    return _adaptive_threshold(
        data,
        block_size=block_size,
        c=c,
        method="mean",
        channel_axis=channel_axis,
    )


def adaptive_gaussian_threshold(
    data,
    block_size: int = 11,
    c: float = 2.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a local-Gaussian mask, optionally reducing declared RGB/RGBA."""
    return _adaptive_threshold(
        data,
        block_size=block_size,
        c=c,
        method="gaussian",
        channel_axis=channel_axis,
    )


def sauvola_threshold(
    data,
    window_size: int = 15,
    k: float = 0.2,
    dynamic_range: float = 0.0,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a Sauvola mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Sauvola threshold",
    )

    def threshold_plane(plane: np.ndarray) -> np.ndarray:
        window = _odd_size(window_size, minimum=3, maximum=min(plane.shape[-2:]))
        if window < 3:
            return plane > float(np.mean(plane))
        values = plane.astype(np.float32, copy=False)
        r = float(dynamic_range)
        if r <= 0.0:
            stats = _finite_array_stats(values)
            r = (
                float((stats.maximum - stats.minimum) / 2.0)
                if stats.count
                else 1.0
            )
            r = max(r, 1e-6)
        local = filters.threshold_sauvola(values, window_size=window, k=float(k), r=r)
        return values > local

    return _apply_scalar_plane_wise(arr, threshold_plane)


def niblack_threshold(
    data,
    window_size: int = 15,
    k: float = 0.2,
    channel_axis: int | None = None,
) -> np.ndarray:
    """Return a Niblack mask, optionally reducing declared RGB/RGBA to luma."""
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation="Niblack threshold",
    )

    def threshold_plane(plane: np.ndarray) -> np.ndarray:
        window = _odd_size(window_size, minimum=3, maximum=min(plane.shape[-2:]))
        if window < 3:
            return plane > float(np.mean(plane))
        values = plane.astype(np.float32, copy=False)
        local = filters.threshold_niblack(values, window_size=window, k=float(k))
        return values > local

    return _apply_scalar_plane_wise(arr, threshold_plane)


def dilate(data, size: int = 10, iterations: int = 1) -> np.ndarray:
    """Dilate a binary mask slice by slice."""
    mask = _to_bool_mask(data)
    return ndi.binary_dilation(
        mask,
        structure=_xy_structure(mask, size),
        iterations=max(int(iterations), 1),
    )


def erode(data, size: int = 10, iterations: int = 1) -> np.ndarray:
    """Erode a binary mask slice by slice."""
    mask = _to_bool_mask(data)
    return ndi.binary_erosion(
        mask,
        structure=_xy_structure(mask, size),
        iterations=max(int(iterations), 1),
    )


def opening(data, size: int = 2) -> np.ndarray:
    """Open a binary mask slice by slice."""
    mask = _to_bool_mask(data)
    return ndi.binary_opening(mask, structure=_xy_structure(mask, size))


def closing(data, size: int = 2) -> np.ndarray:
    """Close a binary mask slice by slice."""
    mask = _to_bool_mask(data)
    return ndi.binary_closing(mask, structure=_xy_structure(mask, size))


def top_hat(data, size: int = 2) -> np.ndarray:
    """White top-hat transform for binary masks."""
    mask = _to_bool_mask(data)
    opened = ndi.binary_opening(mask, structure=_xy_structure(mask, size))
    return mask & ~opened


def black_hat(data, size: int = 2) -> np.ndarray:
    """Black-hat transform for binary masks."""
    mask = _to_bool_mask(data)
    closed = ndi.binary_closing(mask, structure=_xy_structure(mask, size))
    return closed & ~mask


def morphological_gradient(data, size: int = 2) -> np.ndarray:
    """Binary morphological gradient."""
    mask = _to_bool_mask(data)
    structure = _xy_structure(mask, size)
    return ndi.binary_dilation(mask, structure=structure) ^ ndi.binary_erosion(
        mask,
        structure=structure,
    )


def fill_holes(
    data,
    max_hole_size: int = 0,
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Face connected",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Fill enclosed mask holes per XY slice or across a ZYX volume.

    ``max_hole_size=0`` fills every enclosed hole. Positive values fill only
    holes whose area or volume is at most the requested pixel/voxel count.
    """
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    rank = 1 if str(connectivity).lower().startswith("face") else spatial_ndim
    structure = ndi.generate_binary_structure(spatial_ndim, rank)
    maximum = max(int(max_hole_size), 0)

    def fill_block(block: np.ndarray) -> np.ndarray:
        filled = ndi.binary_fill_holes(block, structure=structure)
        if maximum == 0:
            return filled
        holes = filled & ~block
        hole_labels, count = ndi.label(holes, structure=structure)
        if count == 0:
            return block.copy()
        sizes = np.bincount(hole_labels.ravel())
        keep = sizes <= maximum
        keep[0] = False
        return block | keep[hole_labels]

    return _apply_spatial_blocks(mask, spatial_ndim, fill_block, dtype=bool)


def clear_border_objects(
    data,
    border_buffer: int = 0,
    boundary_mode: str = "All spatial borders",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Remove objects touching all spatial or lateral YX boundaries."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        objects = arr
    elif np.issubdtype(arr.dtype, np.integer):
        objects = _validated_labels(arr)
    else:
        raise ValueError(
            "Clear Border Objects requires a binary mask or integer label image."
        )

    spatial_ndim = _resolved_spatial_ndim(
        objects,
        "Auto from axes",
        resolved_spatial_ndim,
    )
    buffer_size = max(int(border_buffer), 0)
    mode = str(boundary_mode).strip().lower()

    def clear_block(block: np.ndarray) -> np.ndarray:
        if mode.startswith("lateral") and block.ndim >= 2:
            boundary_axes = tuple(range(block.ndim - 2, block.ndim))
        else:
            boundary_axes = tuple(range(block.ndim))
        if block.size and any(
            buffer_size >= block.shape[axis] for axis in boundary_axes
        ):
            raise ValueError(
                "Border buffer must be smaller than every processed "
                "boundary dimension."
            )
        valid_region = np.ones(block.shape, dtype=bool)
        extent = buffer_size + 1
        for axis in boundary_axes:
            start = [slice(None)] * block.ndim
            end = [slice(None)] * block.ndim
            start[axis] = slice(0, extent)
            end[axis] = slice(-extent, None)
            valid_region[tuple(start)] = False
            valid_region[tuple(end)] = False
        return segmentation.clear_border(block, mask=valid_region)

    return _apply_spatial_blocks(
        objects,
        spatial_ndim,
        clear_block,
        dtype=objects.dtype,
    )


def remove_small_objects(
    data,
    min_size: int = 10,
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Face connected",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Remove small mask components or labels within each spatial block."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        objects = arr
    elif np.issubdtype(arr.dtype, np.integer):
        objects = _validated_labels(arr)
    else:
        raise ValueError(
            "Remove Small Objects requires a binary mask or integer label image."
        )

    spatial_ndim = _resolved_spatial_ndim(
        objects,
        spatial_mode,
        resolved_spatial_ndim,
    )
    rank = 1 if str(connectivity).lower().startswith("face") else spatial_ndim
    structure = ndi.generate_binary_structure(spatial_ndim, rank)
    minimum = max(int(min_size), 0)

    def filter_block(block: np.ndarray) -> np.ndarray:
        if block.dtype == bool:
            labels, count = ndi.label(block, structure=structure)
            if count == 0 or minimum <= 1:
                return block.copy()
            sizes = np.bincount(labels.ravel())
            keep = sizes >= minimum
            keep[0] = False
            return keep[labels]

        values, counts = np.unique(block, return_counts=True)
        keep = (values != 0) & (counts >= minimum)
        kept_labels = values[keep]
        if kept_labels.size == 0:
            return np.zeros_like(block)
        return np.where(np.isin(block, kept_labels), block, 0)

    return _apply_spatial_blocks(
        objects,
        spatial_ndim,
        filter_block,
        dtype=objects.dtype,
    )


def label_connected_components(
    data,
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Full connectivity",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Assign an integer ID to each connected foreground structure."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    rank = 1 if str(connectivity).lower().startswith("face") else spatial_ndim
    structure = ndi.generate_binary_structure(spatial_ndim, rank)

    def label_block(block: np.ndarray) -> np.ndarray:
        labels, _count = ndi.label(block, structure=structure)
        return labels.astype(np.int32, copy=False)

    return _apply_spatial_blocks(mask, spatial_ndim, label_block, dtype=np.int32)


def euclidean_distance_transform(
    data,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Compute foreground distance to background per 2D plane or 3D volume."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )

    def distance_block(block: np.ndarray) -> np.ndarray:
        return ndi.distance_transform_edt(block).astype(np.float32, copy=False)

    return _apply_spatial_blocks(mask, spatial_ndim, distance_block, dtype=np.float32)


def h_maxima_markers(
    data,
    h: float = 1.0,
    spatial_mode: str = "Auto from axes",
    connectivity: str = "Full connectivity",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Generate labeled watershed seed markers from robust local maxima."""
    arr = np.asarray(data)
    spatial_ndim = _resolved_spatial_ndim(
        arr,
        spatial_mode,
        resolved_spatial_ndim,
    )
    rank = 1 if str(connectivity).lower().startswith("face") else spatial_ndim
    structure = ndi.generate_binary_structure(spatial_ndim, rank)
    height = max(float(h), 0.0)

    def marker_block(block: np.ndarray) -> np.ndarray:
        values = np.asarray(block, dtype=np.float32)
        stats = _finite_array_stats(values)
        if stats.count == 0 or stats.minimum == stats.maximum:
            return np.zeros(values.shape, dtype=np.int32)
        if stats.count == values.size:
            safe = values
        else:
            safe = np.nan_to_num(
                values,
                nan=stats.minimum,
                posinf=stats.maximum,
                neginf=stats.minimum,
            )
        if height > 0:
            peaks = morphology.h_maxima(safe, height)
        else:
            peaks = morphology.local_maxima(safe)
        peaks &= safe > stats.minimum
        labels, _count = ndi.label(peaks, structure=structure)
        return labels.astype(np.int32, copy=False)

    return _apply_spatial_blocks(arr, spatial_ndim, marker_block, dtype=np.int32)


def auto_watershed_from_mask(
    data,
    h: float = 1.0,
    connectivity: str = "Full connectivity",
    image_mode: str = "Distance map (invert)",
    compactness: float = 0.0,
    watershed_line: bool = False,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Segment touching foreground structures from one mask-like input.

    This convenience operation performs the common separation chain:
    mask -> distance transform -> h-maxima markers -> marker-controlled watershed.
    """
    mask = _to_bool_mask(data)
    distance = euclidean_distance_transform(
        mask,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
    )
    markers = h_maxima_markers(
        distance,
        h=h,
        spatial_mode=spatial_mode,
        connectivity=connectivity,
        resolved_spatial_ndim=resolved_spatial_ndim,
    )
    return marker_controlled_watershed(
        [distance, markers, mask],
        image_mode=image_mode,
        compactness=compactness,
        watershed_line=watershed_line,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
    )


def marker_controlled_watershed(
    inputs,
    image_mode: str = "Distance map (invert)",
    compactness: float = 0.0,
    watershed_line: bool = False,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Segment touching objects from image/elevation, marker, and mask inputs."""
    arrays = _matching_input_arrays(inputs, 3, "Marker-Controlled Watershed")
    image = arrays[0].astype(np.float32, copy=False)
    markers = _validated_labels(arrays[1]).astype(np.int32, copy=False)
    mask = _to_bool_mask(arrays[2])
    spatial_ndim = _resolved_spatial_ndim(
        image,
        spatial_mode,
        resolved_spatial_ndim,
    )
    invert = str(image_mode).strip().lower().startswith("distance")

    def watershed_block(
        image_block: np.ndarray,
        marker_block: np.ndarray,
        mask_block: np.ndarray,
    ) -> np.ndarray:
        if not np.any(mask_block) or int(marker_block.max(initial=0)) <= 0:
            return np.zeros(mask_block.shape, dtype=np.int32)
        elevation = -image_block if invert else image_block
        labels = segmentation.watershed(
            elevation,
            marker_block,
            mask=mask_block,
            compactness=max(float(compactness), 0.0),
            watershed_line=bool(watershed_line),
        )
        return labels.astype(np.int32, copy=False)

    return _apply_spatial_blocks_multi(
        (image, markers, mask),
        spatial_ndim,
        watershed_block,
        dtype=np.int32,
    )


def expand_labels_image(
    data,
    distance: float = 5.0,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Expand labels into nearby background without allowing overlaps."""
    labels = _validated_labels(data)
    spatial_ndim = _resolved_spatial_ndim(
        labels,
        spatial_mode,
        resolved_spatial_ndim,
    )
    expansion = max(float(distance), 0.0)

    def expand_block(block: np.ndarray) -> np.ndarray:
        if expansion <= 0 or int(block.max(initial=0)) <= 0:
            return block.copy()
        expanded = segmentation.expand_labels(block, distance=expansion)
        return np.asarray(expanded, dtype=labels.dtype)

    return _apply_spatial_blocks(
        labels,
        spatial_ndim,
        expand_block,
        dtype=labels.dtype,
    )


def filter_labels_by_volume(
    data,
    min_volume: int = 10,
    max_volume: int = 0,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Remove labeled objects outside an inclusive pixel/voxel volume range.

    ``max_volume=0`` disables the upper bound. Retained label IDs are preserved;
    use :func:`relabel_sequential` when compact IDs are desired.
    """
    labels = _validated_labels(data)
    spatial_ndim = _resolved_spatial_ndim(
        labels,
        spatial_mode,
        resolved_spatial_ndim,
    )
    minimum = max(int(min_volume), 0)
    maximum = max(int(max_volume), 0)

    def filter_block(block: np.ndarray) -> np.ndarray:
        values, counts = np.unique(block, return_counts=True)
        keep = values != 0
        keep &= counts >= minimum
        if maximum > 0:
            keep &= counts <= maximum
        kept_labels = values[keep]
        if kept_labels.size == 0:
            return np.zeros_like(block)
        return np.where(np.isin(block, kept_labels), block, 0)

    return _apply_spatial_blocks(
        labels,
        spatial_ndim,
        filter_block,
        dtype=labels.dtype,
    )


PROPERTY_FILTER_COLUMN_PRIORITY = (
    "volume_physical",
    "area_physical",
    "volume_voxels",
    "area_pixels",
    "length_physical",
    "length_pixels",
    "intensity_mean",
    "intensity_sum",
    "intensity_max",
    "intensity_min",
    "intensity_std",
    "skeleton_length_physical",
    "skeleton_length_pixels",
    "skeleton_length_voxels",
    "skeleton_voxel_count",
    "branch_count",
    "graph_edge_count",
    "voxel_graph_edge_count",
    "isolated_node_count",
    "endpoint_voxel_count",
    "junction_voxel_count",
    "cycle_count",
)


def filter_labels_by_property(
    inputs,
    property_column: str = "auto",
    min_value: float = 0.0,
    max_value: float = 0.0,
    keep_mode: str = "Keep inside range",
    unmatched_labels: str = "Remove unmatched labels",
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
) -> np.ndarray:
    """Filter label IDs using a per-object measurement table column."""
    labels_data, table = _labels_and_table_inputs(inputs)
    labels = _validated_labels(labels_data)
    table = _validated_table(table)
    label_column = _label_id_column(table)
    value_column = _property_filter_column(table, property_column)
    minimum = float(min_value)
    maximum = float(max_value)
    has_maximum = maximum > minimum if maximum > 0 else False
    remove_inside = str(keep_mode).strip().lower().startswith("remove")
    keep_unmatched = str(unmatched_labels).strip().lower().startswith("keep")

    spatial_ndim = _resolved_spatial_ndim(
        labels,
        spatial_mode,
        resolved_spatial_ndim,
    )
    axis_names = _measurement_axis_names(labels.ndim, axis_names)
    axis_types = _measurement_axis_types(labels.ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        labels.ndim,
        spatial_ndim,
        axis_names,
        axis_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(labels.ndim - spatial_ndim, labels.ndim))

    labels_for_filter = np.moveaxis(
        labels,
        spatial_axes,
        tuple(range(labels.ndim - spatial_ndim, labels.ndim)),
    )
    moved_axis_names = tuple(
        axis_names[index]
        for index in range(labels.ndim)
        if index not in spatial_axes
    ) + tuple(axis_names[index] for index in spatial_axes)
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[: labels.ndim - spatial_ndim],
        fallback=tuple(
            f"axis_{index}" for index in range(labels.ndim - spatial_ndim)
        ),
    )
    leading_shape = labels_for_filter.shape[: labels.ndim - spatial_ndim]
    kept_by_block, matched_by_block = _property_filter_records_by_block(
        table,
        leading_axis_names,
        value_column,
        label_column,
        minimum,
        maximum,
        has_maximum,
        remove_inside,
    )
    output = np.zeros_like(labels_for_filter)
    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = (
            labels_for_filter[block_index]
            if leading_shape
            else labels_for_filter
        )
        key = _property_filter_block_key(leading_axis_names, block_index)
        kept = _property_filter_labels_for_block(kept_by_block, key)
        if keep_unmatched:
            matched = _property_filter_labels_for_block(matched_by_block, key)
            block_output = block.copy()
            if kept:
                remove = np.isin(block, list(matched - kept))
                block_output[remove] = 0
            elif matched:
                block_output[np.isin(block, list(matched))] = 0
        elif kept:
            block_output = np.where(np.isin(block, list(kept)), block, 0)
        else:
            block_output = np.zeros_like(block)
        if leading_shape:
            output[block_index] = block_output
        else:
            output = block_output

    return np.moveaxis(
        output,
        tuple(range(labels.ndim - spatial_ndim, labels.ndim)),
        spatial_axes,
    ).astype(labels.dtype, copy=False)


def relabel_sequential(
    data,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Renumber positive labels to a compact ``1..N`` sequence per frame."""
    labels = _validated_labels(data)
    spatial_ndim = _resolved_spatial_ndim(
        labels,
        spatial_mode,
        resolved_spatial_ndim,
    )

    def relabel_block(block: np.ndarray) -> np.ndarray:
        relabeled, _forward_map, _inverse_map = segmentation.relabel_sequential(
            block
        )
        return np.asarray(relabeled, dtype=labels.dtype)

    return _apply_spatial_blocks(
        labels,
        spatial_ndim,
        relabel_block,
        dtype=labels.dtype,
    )


def skeletonize_mask(
    data,
    spatial_mode: str = "Auto from axes",
    method: str = "Auto",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Skeletonize a binary mask per XY plane or across a ZYX volume."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    method_value = _skeletonize_method(method)

    def skeletonize_block(block: np.ndarray) -> np.ndarray:
        if not np.any(block):
            return np.zeros_like(block, dtype=bool)
        if method_value == "zhang" and block.ndim != 2:
            raise ValueError("Zhang skeletonization is only valid for 2D blocks.")
        return morphology.skeletonize(block, method=method_value)

    return _apply_spatial_blocks(mask, spatial_ndim, skeletonize_block, dtype=bool)


def skeleton_keypoints(
    data,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract endpoint, junction, and isolated-node masks from a skeleton."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )

    return _apply_spatial_blocks_tuple(
        mask,
        spatial_ndim,
        _skeleton_keypoint_masks_block,
        dtypes=(bool, bool, bool),
    )


def label_skeleton_components(
    data,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Label connected skeleton components within each spatial block."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    return _apply_spatial_blocks(
        mask,
        spatial_ndim,
        _label_skeleton_components_block,
        dtype=np.int32,
    )


def label_skeleton_branches(
    data,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Label branch paths between skeleton endpoints and junctions."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    return _apply_spatial_blocks(
        mask,
        spatial_ndim,
        _label_skeleton_branches_block,
        dtype=np.int32,
    )


def skeleton_graph_overlay(
    data,
    display_mode: str = "Colored edges + colored nodes",
    node_size: int = 1,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Render skeleton branches and graph nodes as a channel-last RGB overlay."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )

    def overlay_block(block: np.ndarray) -> np.ndarray:
        return _skeleton_graph_overlay_block(
            block,
            display_mode=display_mode,
            node_size=node_size,
        )

    return _apply_spatial_blocks_rgb(mask, spatial_ndim, overlay_block)


def prune_skeleton_branches(
    data,
    min_branch_length: float = 3.0,
    length_units: str = "Pixels/voxels",
    iterations: int = 1,
    remove_isolated: bool = True,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
) -> np.ndarray:
    """Remove short terminal skeleton branches within each spatial block."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    del axis_units  # axis_scales carry the numeric conversion used for pruning.
    scales = _normalized_spatial_scales(
        spatial_ndim,
        tuple(axis_scales or ())[-spatial_ndim:],
    )

    def prune_block(block: np.ndarray) -> np.ndarray:
        return _prune_skeleton_branches_block(
            block,
            min_branch_length=float(min_branch_length),
            iterations=int(iterations),
            remove_isolated=bool(remove_isolated),
            length_units=length_units,
            scales=scales,
        )

    return _apply_spatial_blocks(mask, spatial_ndim, prune_block, dtype=bool)


def measure_objects(
    data,
    spatial_mode: str = "Auto from axes",
    measurement_set: str = "Basic morphology",
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
    include_derived_shape_ratios: bool = False,
    include_2d_shape_moments: bool = False,
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Measure labeled objects into a row-per-object table."""
    labels = _validated_labels(data)
    spatial_ndim = _resolved_spatial_ndim(
        labels,
        spatial_mode,
        resolved_spatial_ndim,
    )
    axis_names = _measurement_axis_names(labels.ndim, axis_names)
    axis_types = _measurement_axis_types(labels.ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        labels.ndim,
        spatial_ndim,
        axis_names,
        axis_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(labels.ndim - spatial_ndim, labels.ndim))
    labels_for_measure = np.moveaxis(
        labels,
        spatial_axes,
        tuple(range(labels.ndim - spatial_ndim, labels.ndim)),
    )
    moved_axis_names = tuple(
        axis_names[index]
        for index in range(labels.ndim)
        if index not in spatial_axes
    ) + tuple(axis_names[index] for index in spatial_axes)
    moved_axis_scales = _reordered_axis_values(axis_scales, labels.ndim, spatial_axes)
    moved_axis_units = _reordered_axis_values(axis_units, labels.ndim, spatial_axes)
    spatial_axis_names = _safe_axis_column_names(
        moved_axis_names[-spatial_ndim:],
        fallback=("z", "y", "x")[-spatial_ndim:],
    )
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[: labels.ndim - spatial_ndim],
        fallback=tuple(f"axis_{index}" for index in range(labels.ndim - spatial_ndim)),
    )
    leading_shape = labels_for_measure.shape[: labels.ndim - spatial_ndim]

    units = _measurement_units(
        spatial_ndim,
        moved_axis_scales[-spatial_ndim:],
        moved_axis_units[-spatial_ndim:],
    )
    columns = _measurement_empty_columns(
        leading_axis_names,
        spatial_axis_names,
        units,
        include_shape_descriptors=include_shape_descriptors,
        include_axis_descriptors=include_axis_descriptors,
        include_2d_boundary_descriptors=include_2d_boundary_descriptors,
        include_derived_shape_ratios=include_derived_shape_ratios,
        include_2d_shape_moments=include_2d_shape_moments,
        spatial_ndim=spatial_ndim,
    )
    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = labels_for_measure[block_index] if leading_shape else labels_for_measure
        block_columns = _measure_label_block(
            block,
            spatial_axis_names,
            spatial_ndim,
            axis_scales=moved_axis_scales[-spatial_ndim:],
            has_physical_calibration=bool(units.physical_column),
            include_shape_descriptors=include_shape_descriptors,
            include_axis_descriptors=include_axis_descriptors,
            include_2d_boundary_descriptors=include_2d_boundary_descriptors,
            include_derived_shape_ratios=include_derived_shape_ratios,
            include_2d_shape_moments=include_2d_shape_moments,
        )
        row_count = len(block_columns["label_id"])
        for axis_position, axis_name in enumerate(leading_axis_names):
            columns[f"{axis_name}_index"].extend(
                [int(block_index[axis_position])] * row_count
            )
        for name, values in block_columns.items():
            columns[name].extend(values)
        if units.physical_column:
            columns[units.physical_column].extend(
                [
                    float(value) * units.scale_product
                    for value in block_columns[units.size_column]
                ]
            )
            columns["physical_unit"].extend([units.unit_label] * row_count)

    return table_from_columns(
        columns,
        name="Object measurements",
        table_kind=_measurement_table_kind(
            str(measurement_set or "Basic morphology"),
            include_shape_descriptors=include_shape_descriptors,
            include_axis_descriptors=include_axis_descriptors,
            include_2d_boundary_descriptors=include_2d_boundary_descriptors,
            include_derived_shape_ratios=include_derived_shape_ratios,
            include_2d_shape_moments=include_2d_shape_moments,
            spatial_ndim=spatial_ndim,
        ),
        source_name=source_name,
        column_units=units.column_units,
    )


def measure_objects_with_intensity(
    inputs,
    spatial_mode: str = "Auto from axes",
    measurement_set: str = "Basic morphology + intensity",
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
    include_derived_shape_ratios: bool = False,
    include_2d_shape_moments: bool = False,
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Measure labeled objects and their matching intensity image."""
    labels_data, intensity_data = _labels_and_intensity_inputs(inputs)
    labels = _validated_labels(labels_data)
    intensity = np.asarray(intensity_data)
    if intensity.shape != labels.shape:
        raise ValueError(
            "Intensity-aware measurements require labels and intensity image "
            "with the same shape."
        )

    spatial_ndim = _resolved_spatial_ndim(
        labels,
        spatial_mode,
        resolved_spatial_ndim,
    )
    axis_names = _measurement_axis_names(labels.ndim, axis_names)
    axis_types = _measurement_axis_types(labels.ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        labels.ndim,
        spatial_ndim,
        axis_names,
        axis_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(labels.ndim - spatial_ndim, labels.ndim))
    labels_for_measure = np.moveaxis(
        labels,
        spatial_axes,
        tuple(range(labels.ndim - spatial_ndim, labels.ndim)),
    )
    intensity_for_measure = np.moveaxis(
        intensity,
        spatial_axes,
        tuple(range(labels.ndim - spatial_ndim, labels.ndim)),
    )
    moved_axis_names = tuple(
        axis_names[index]
        for index in range(labels.ndim)
        if index not in spatial_axes
    ) + tuple(axis_names[index] for index in spatial_axes)
    moved_axis_scales = _reordered_axis_values(axis_scales, labels.ndim, spatial_axes)
    moved_axis_units = _reordered_axis_values(axis_units, labels.ndim, spatial_axes)
    spatial_axis_names = _safe_axis_column_names(
        moved_axis_names[-spatial_ndim:],
        fallback=("z", "y", "x")[-spatial_ndim:],
    )
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[: labels.ndim - spatial_ndim],
        fallback=tuple(f"axis_{index}" for index in range(labels.ndim - spatial_ndim)),
    )
    leading_shape = labels_for_measure.shape[: labels.ndim - spatial_ndim]

    units = _measurement_units(
        spatial_ndim,
        moved_axis_scales[-spatial_ndim:],
        moved_axis_units[-spatial_ndim:],
    )
    columns = _measurement_empty_columns(
        leading_axis_names,
        spatial_axis_names,
        units,
        include_intensity=True,
        include_shape_descriptors=include_shape_descriptors,
        include_axis_descriptors=include_axis_descriptors,
        include_2d_boundary_descriptors=include_2d_boundary_descriptors,
        include_derived_shape_ratios=include_derived_shape_ratios,
        include_2d_shape_moments=include_2d_shape_moments,
        spatial_ndim=spatial_ndim,
    )
    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = labels_for_measure[block_index] if leading_shape else labels_for_measure
        intensity_block = (
            intensity_for_measure[block_index]
            if leading_shape
            else intensity_for_measure
        )
        block_columns = _measure_label_block(
            block,
            spatial_axis_names,
            spatial_ndim,
            intensity_block=intensity_block,
            axis_scales=moved_axis_scales[-spatial_ndim:],
            has_physical_calibration=bool(units.physical_column),
            include_shape_descriptors=include_shape_descriptors,
            include_axis_descriptors=include_axis_descriptors,
            include_2d_boundary_descriptors=include_2d_boundary_descriptors,
            include_derived_shape_ratios=include_derived_shape_ratios,
            include_2d_shape_moments=include_2d_shape_moments,
        )
        row_count = len(block_columns["label_id"])
        for axis_position, axis_name in enumerate(leading_axis_names):
            columns[f"{axis_name}_index"].extend(
                [int(block_index[axis_position])] * row_count
            )
        for name, values in block_columns.items():
            columns[name].extend(values)
        if units.physical_column:
            columns[units.physical_column].extend(
                [
                    float(value) * units.scale_product
                    for value in block_columns[units.size_column]
                ]
            )
            columns["physical_unit"].extend([units.unit_label] * row_count)

    table_kind = _measurement_table_kind(
        str(measurement_set or "Basic morphology + intensity"),
        include_shape_descriptors=include_shape_descriptors,
        include_axis_descriptors=include_axis_descriptors,
        include_2d_boundary_descriptors=include_2d_boundary_descriptors,
        include_derived_shape_ratios=include_derived_shape_ratios,
        include_2d_shape_moments=include_2d_shape_moments,
        spatial_ndim=spatial_ndim,
    )
    return table_from_columns(
        columns,
        name="Object intensity measurements",
        table_kind=table_kind,
        source_name=source_name,
        column_units=units.column_units,
    )


def measure_3d_mesh_morphology(
    data,
    spatial_mode: str = "Auto from axes",
    minimum_voxel_count: int = 16,
    include_convex_hull_metrics: bool = True,
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
    progress=None,
) -> TableData:
    """Measure 3D mesh/surface morphology for labeled objects."""
    labels = _validated_labels(data)
    spatial_ndim = _resolved_spatial_ndim(
        labels,
        spatial_mode,
        resolved_spatial_ndim,
    )
    if spatial_ndim != 3:
        raise ValueError("Measure 3D Mesh Morphology requires true 3D labels.")
    axis_names = _measurement_axis_names(labels.ndim, axis_names)
    axis_types = _measurement_axis_types(labels.ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        labels.ndim,
        spatial_ndim,
        axis_names,
        axis_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(labels.ndim - spatial_ndim, labels.ndim))
    labels_for_measure = np.moveaxis(
        labels,
        spatial_axes,
        tuple(range(labels.ndim - spatial_ndim, labels.ndim)),
    )
    moved_axis_names = tuple(
        axis_names[index]
        for index in range(labels.ndim)
        if index not in spatial_axes
    ) + tuple(axis_names[index] for index in spatial_axes)
    moved_axis_scales = _reordered_axis_values(axis_scales, labels.ndim, spatial_axes)
    moved_axis_units = _reordered_axis_values(axis_units, labels.ndim, spatial_axes)
    spatial_axis_names = _safe_axis_column_names(
        moved_axis_names[-spatial_ndim:],
        fallback=("z", "y", "x"),
    )
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[: labels.ndim - spatial_ndim],
        fallback=tuple(f"axis_{index}" for index in range(labels.ndim - spatial_ndim)),
    )
    leading_shape = labels_for_measure.shape[: labels.ndim - spatial_ndim]
    units = _mesh_units(
        moved_axis_scales[-spatial_ndim:],
        moved_axis_units[-spatial_ndim:],
        spatial_axis_names,
        include_convex_hull_metrics=bool(include_convex_hull_metrics),
    )
    columns = _mesh_morphology_empty_columns(
        leading_axis_names,
        spatial_axis_names,
        include_convex_hull_metrics=bool(include_convex_hull_metrics),
    )

    block_items = []
    total_labels = 0
    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = labels_for_measure[block_index] if leading_shape else labels_for_measure
        label_ids = _positive_label_ids(block)
        total_labels += len(label_ids)
        block_items.append((block_index, block, label_ids))
    if progress is not None:
        progress.report(0, max(total_labels, 1), "Preparing mesh measurements")
    completed_labels = 0
    for block_index, block, label_ids in block_items:
        block_columns = _measure_3d_mesh_morphology_block(
            block,
            spatial_axis_names,
            units,
            minimum_voxel_count=max(int(minimum_voxel_count), 1),
            include_convex_hull_metrics=bool(include_convex_hull_metrics),
            label_ids=label_ids,
            progress=progress,
            progress_start=completed_labels,
            progress_total=max(total_labels, 1),
        )
        completed_labels += len(label_ids)
        row_count = len(block_columns["label_id"])
        for axis_position, axis_name in enumerate(leading_axis_names):
            columns[f"{axis_name}_index"].extend(
                [int(block_index[axis_position])] * row_count
            )
        for name, values in block_columns.items():
            columns[name].extend(values)
        columns["physical_unit"].extend([units.length_unit] * row_count)
    if progress is not None:
        progress.report(max(total_labels, 1), max(total_labels, 1), "Mesh complete")

    return table_from_columns(
        columns,
        name="3D mesh morphology measurements",
        table_kind="3D mesh morphology",
        source_name=source_name,
        column_units=units.column_units,
    )


def analyze_skeleton(
    data,
    spatial_mode: str = "Auto from axes",
    input_mode: str = "Already skeletonized",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Analyze skeleton components into a per-component network table."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    axis_names = _measurement_axis_names(mask.ndim, axis_names)
    axis_types = _measurement_axis_types(mask.ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        mask.ndim,
        spatial_ndim,
        axis_names,
        axis_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(mask.ndim - spatial_ndim, mask.ndim))
    mask_for_analysis = np.moveaxis(
        mask,
        spatial_axes,
        tuple(range(mask.ndim - spatial_ndim, mask.ndim)),
    )
    moved_axis_names = tuple(
        axis_names[index]
        for index in range(mask.ndim)
        if index not in spatial_axes
    ) + tuple(axis_names[index] for index in spatial_axes)
    moved_axis_scales = _reordered_axis_values(axis_scales, mask.ndim, spatial_axes)
    moved_axis_units = _reordered_axis_values(axis_units, mask.ndim, spatial_axes)
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[: mask.ndim - spatial_ndim],
        fallback=tuple(f"axis_{index}" for index in range(mask.ndim - spatial_ndim)),
    )
    leading_shape = mask_for_analysis.shape[: mask.ndim - spatial_ndim]
    units = _skeleton_units(
        spatial_ndim,
        moved_axis_scales[-spatial_ndim:],
        moved_axis_units[-spatial_ndim:],
    )
    columns = _skeleton_empty_columns(leading_axis_names, units)
    should_skeletonize = str(input_mode).strip().lower().startswith("skeletonize")

    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = (
            mask_for_analysis[block_index]
            if leading_shape
            else mask_for_analysis
        )
        skeleton = (
            morphology.skeletonize(block).astype(bool)
            if should_skeletonize
            else block.astype(bool, copy=False)
        )
        block_columns = _analyze_skeleton_block(skeleton, units)
        row_count = len(block_columns["component_id"])
        for axis_position, axis_name in enumerate(leading_axis_names):
            columns[f"{axis_name}_index"].extend(
                [int(block_index[axis_position])] * row_count
            )
        for name, values in block_columns.items():
            columns[name].extend(values)
        if units.physical_column:
            columns["physical_unit"].extend([units.unit_label] * row_count)

    return table_from_columns(
        columns,
        name="Skeleton network measurements",
        table_kind="Skeleton network",
        source_name=source_name,
        column_units=units.column_units,
    )


def measure_skeleton_branches(
    data,
    spatial_mode: str = "Auto from axes",
    input_mode: str = "Already skeletonized",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Measure traced skeleton branches into a row-per-branch table."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    has_axis_names = axis_names is not None and len(axis_names) == mask.ndim
    axis_names = _measurement_axis_names(mask.ndim, axis_names)
    axis_types = _measurement_axis_types(mask.ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        mask.ndim,
        spatial_ndim,
        axis_names,
        axis_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(mask.ndim - spatial_ndim, mask.ndim))
    mask_for_analysis = np.moveaxis(
        mask,
        spatial_axes,
        tuple(range(mask.ndim - spatial_ndim, mask.ndim)),
    )
    moved_axis_names = tuple(
        axis_names[index]
        for index in range(mask.ndim)
        if index not in spatial_axes
    ) + tuple(axis_names[index] for index in spatial_axes)
    moved_axis_scales = _reordered_axis_values(axis_scales, mask.ndim, spatial_axes)
    moved_axis_units = _reordered_axis_values(axis_units, mask.ndim, spatial_axes)
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[: mask.ndim - spatial_ndim],
        fallback=tuple(f"axis_{index}" for index in range(mask.ndim - spatial_ndim)),
    )
    spatial_axis_names = _safe_axis_column_names(
        moved_axis_names[-spatial_ndim:]
        if has_axis_names
        else tuple("" for _axis in range(spatial_ndim)),
        fallback=("z", "y", "x")[-spatial_ndim:],
    )
    leading_shape = mask_for_analysis.shape[: mask.ndim - spatial_ndim]
    units = _skeleton_units(
        spatial_ndim,
        moved_axis_scales[-spatial_ndim:],
        moved_axis_units[-spatial_ndim:],
    )
    columns = _skeleton_branch_empty_columns(
        leading_axis_names,
        spatial_axis_names,
        units,
    )
    should_skeletonize = str(input_mode).strip().lower().startswith("skeletonize")

    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = (
            mask_for_analysis[block_index]
            if leading_shape
            else mask_for_analysis
        )
        skeleton = (
            morphology.skeletonize(block).astype(bool)
            if should_skeletonize
            else block.astype(bool, copy=False)
        )
        block_columns = _measure_skeleton_branches_block(
            skeleton,
            units,
            spatial_axis_names,
        )
        row_count = len(block_columns["component_id"])
        for axis_position, axis_name in enumerate(leading_axis_names):
            columns[f"{axis_name}_index"].extend(
                [int(block_index[axis_position])] * row_count
            )
        for name, values in block_columns.items():
            columns[name].extend(values)
        if units.physical_column:
            columns["physical_unit"].extend([units.unit_label] * row_count)

    return table_from_columns(
        columns,
        name="Skeleton branch measurements",
        table_kind="Skeleton branches",
        source_name=source_name,
        column_units=_skeleton_branch_column_units(units),
    )


def skeleton_graph_tables(
    data,
    spatial_mode: str = "Auto from axes",
    input_mode: str = "Already skeletonized",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> tuple[TableData, TableData]:
    """Export explicit skeleton graph node and edge tables."""
    (
        mask_for_analysis,
        _spatial_ndim,
        leading_axis_names,
        spatial_axis_names,
        leading_shape,
        units,
    ) = _skeleton_table_context(
        data,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    should_skeletonize = str(input_mode).strip().lower().startswith("skeletonize")
    node_columns = _skeleton_graph_node_empty_columns(
        leading_axis_names,
        spatial_axis_names,
    )
    edge_columns = _skeleton_graph_edge_empty_columns(
        leading_axis_names,
        spatial_axis_names,
        units,
    )
    if units.physical_column:
        node_columns["physical_unit"] = []

    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = (
            mask_for_analysis[block_index]
            if leading_shape
            else mask_for_analysis
        )
        skeleton = (
            morphology.skeletonize(block).astype(bool)
            if should_skeletonize
            else block.astype(bool, copy=False)
        )
        block_nodes, block_edges = _skeleton_graph_tables_block(
            skeleton,
            units,
            spatial_axis_names,
        )
        _extend_skeleton_block_columns(
            node_columns,
            block_nodes,
            leading_axis_names,
            block_index,
        )
        _extend_skeleton_block_columns(
            edge_columns,
            block_edges,
            leading_axis_names,
            block_index,
        )
        if units.physical_column:
            node_columns["physical_unit"].extend(
                [units.unit_label] * len(block_nodes["component_id"])
            )
            edge_columns["physical_unit"].extend(
                [units.unit_label] * len(block_edges["component_id"])
            )

    return (
        table_from_columns(
            node_columns,
            name="Skeleton graph nodes",
            table_kind="Skeleton graph nodes",
            source_name=source_name,
            column_units=_skeleton_graph_node_column_units(units),
        ),
        table_from_columns(
            edge_columns,
            name="Skeleton graph edges",
            table_kind="Skeleton graph edges",
            source_name=source_name,
            column_units=_skeleton_graph_edge_column_units(units),
        ),
    )


def measure_overall_skeleton_network(
    data,
    spatial_mode: str = "Auto from axes",
    input_mode: str = "Already skeletonized",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Measure whole-skeleton network topology per spatial block."""
    (
        mask_for_analysis,
        _spatial_ndim,
        leading_axis_names,
        _spatial_axis_names,
        leading_shape,
        units,
    ) = _skeleton_table_context(
        data,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    should_skeletonize = str(input_mode).strip().lower().startswith("skeletonize")
    columns = _overall_skeleton_network_empty_columns(leading_axis_names, units)

    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = (
            mask_for_analysis[block_index]
            if leading_shape
            else mask_for_analysis
        )
        skeleton = (
            morphology.skeletonize(block).astype(bool)
            if should_skeletonize
            else block.astype(bool, copy=False)
        )
        block_columns = _measure_overall_skeleton_network_block(skeleton, units)
        _extend_skeleton_block_columns(
            columns,
            block_columns,
            leading_axis_names,
            block_index,
        )
        if units.physical_column:
            columns["physical_unit"].extend([units.unit_label])

    return table_from_columns(
        columns,
        name="Overall skeleton network measurements",
        table_kind="Overall skeleton network",
        source_name=source_name,
        column_units=_overall_skeleton_network_column_units(units),
    )


IDENTITY_JOIN_COLUMNS = (
    "source_name",
    "sample_id",
    "sample",
    "series_index",
    "series",
    "t_index",
    "time_index",
    "c_index",
    "channel_index",
    "z_index",
    "label_id",
    "object_id",
    "component_id",
    "branch_id",
)


def merge_tables(
    inputs,
    input_count: int = 2,
    join_mode: str = "Left join",
    join_keys: str = "auto",
) -> TableData:
    """Join measurement tables into one analysis-ready table."""
    tables = _table_inputs(inputs, input_count)
    key_columns, row_position_join = _table_join_keys(tables, join_keys)
    mode = _normalized_join_mode(join_mode)
    records = [table.records() for table in tables]
    indexes = [
        _table_record_index(
            table_records,
            key_columns,
            row_position_join,
            table_number=index + 1,
        )
        for index, table_records in enumerate(records)
    ]
    orders = [
        _table_key_order(table_records, key_columns, row_position_join)
        for table_records in records
    ]
    output_keys = _joined_table_keys(mode, indexes, orders)
    output_columns: list[str] = []
    output_specs: list[tuple[int | None, str]] = []
    output_units: dict[str, str] = {}
    used_columns: set[str] = set()

    if not row_position_join:
        for column in key_columns:
            output_columns.append(column)
            output_specs.append((None, column))
            used_columns.add(column)
            unit = _first_table_unit(tables, column)
            if unit:
                output_units[column] = unit

    for table_index, table in enumerate(tables):
        suffix = f"table{table_index + 1}"
        for column in table.columns:
            if column in key_columns:
                continue
            output_column = _unique_table_column(column, used_columns, suffix)
            output_columns.append(output_column)
            output_specs.append((table_index, column))
            used_columns.add(output_column)
            unit = table.unit_for(column)
            if unit:
                output_units[output_column] = unit

    rows: list[tuple[object, ...]] = []
    for key in output_keys:
        values: list[object] = []
        for table_index, column in output_specs:
            if table_index is None:
                record = _first_available_record(indexes, key)
            else:
                record = indexes[table_index].get(key)
            values.append("" if record is None else record.get(column, ""))
        rows.append(tuple(values))

    return TableData(
        columns=tuple(output_columns),
        rows=tuple(rows),
        name="Merged measurement table",
        table_kind="Merged measurements",
        source_name=_merged_table_source_name(tables),
        column_units=tuple(output_units.items()),
    )


def add_metadata_columns(
    data,
    metadata_columns: str = "condition=control",
    overwrite: str = "no",
) -> TableData:
    """Add constant metadata columns to every row of a table."""
    table = _validated_table(data)
    additions = _parse_metadata_columns(metadata_columns)
    if not additions:
        return TableData(
            columns=table.columns,
            rows=table.rows,
            name=table.name,
            table_kind=table.table_kind,
            source_name=table.source_name,
            column_units=table.column_units,
        )

    should_overwrite = str(overwrite).strip().lower().startswith("y")
    columns = list(table.columns)
    rows = [list(row) for row in table.rows]
    overwritten: set[str] = set()
    for column, value in additions:
        if column in columns:
            if not should_overwrite:
                raise ValueError(
                    f"Metadata column {column!r} already exists. "
                    "Enable overwrite to replace it."
                )
            column_index = columns.index(column)
            overwritten.add(column)
            for row in rows:
                row[column_index] = value
            continue
        columns.append(column)
        for row in rows:
            row.append(value)

    units = tuple(
        (column, unit)
        for column, unit in table.column_units
        if column in columns and column not in overwritten
    )
    return TableData(
        columns=tuple(columns),
        rows=tuple(tuple(row) for row in rows),
        name=table.name or "Annotated table",
        table_kind=f"{table.table_kind} + metadata",
        source_name=table.source_name,
        column_units=units,
    )


def select_table_columns(
    data,
    columns: str = "auto",
    selection_mode: str = "Keep listed columns",
    append_unlisted: str = "no",
) -> TableData:
    """Keep, drop, or reorder table columns for analysis-ready export."""
    table = _validated_table(data)
    no_columns_requested = str(columns).strip().lower() in {
        NO_TABLE_COLUMNS_VALUE,
        "none",
        "<none>",
    }
    requested = () if no_columns_requested else _parse_table_column_list(columns)
    if no_columns_requested:
        selected = []
    elif not requested:
        selected = list(table.columns)
    else:
        missing = [column for column in requested if column not in table.columns]
        if missing:
            raise ValueError(
                "Select Table Columns could not find column(s): "
                + ", ".join(missing)
            )
        mode = str(selection_mode).strip().lower()
        if mode.startswith("drop"):
            drop = set(requested)
            selected = [column for column in table.columns if column not in drop]
        else:
            selected = list(dict.fromkeys(requested))
            if str(append_unlisted).strip().lower().startswith("y"):
                selected.extend(
                    column for column in table.columns if column not in selected
                )
    indices = [table.columns.index(column) for column in selected]
    rows = tuple(tuple(row[index] for index in indices) for row in table.rows)
    units = tuple(
        (column, table.unit_for(column))
        for column in selected
        if table.unit_for(column)
    )
    return TableData(
        columns=tuple(selected),
        rows=rows,
        name=table.name or "Selected table columns",
        table_kind=f"{table.table_kind} + column selection",
        source_name=table.source_name,
        column_units=units,
    )


SUMMARY_GROUP_COLUMN_PRIORITY = (
    "condition",
    "treatment",
    "group",
    "replicate",
    "batch",
    "sample_id",
    "sample",
    "source_name",
    "series_index",
    "series",
    "plate",
    "well",
    "field",
    "timepoint",
    "time_point",
    "t_index",
    "time_index",
    "c_index",
    "channel_index",
    "z_index",
)

SUMMARY_EXCLUDED_VALUE_COLUMNS = frozenset(
    set(IDENTITY_JOIN_COLUMNS) | set(SUMMARY_GROUP_COLUMN_PRIORITY)
)

DEFAULT_SUMMARY_STATISTICS = (
    "count",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "q25",
    "q75",
)


def summarize_measurements(
    data,
    group_by: str = "auto",
    value_columns: str = "auto",
    statistics: str = "count,mean,median,std,min,max,q25,q75",
) -> TableData:
    """Summarize measurement columns by metadata or axis-index groups."""
    table = _validated_table(data)
    group_columns = _summary_group_columns(table, group_by)
    numeric_columns = _summary_value_columns(table, value_columns, group_columns)
    summary_stats = _summary_statistics(statistics)

    output_columns = list(group_columns) + ["row_count"]
    output_units: dict[str, str] = {}
    for column in numeric_columns:
        for stat in summary_stats:
            output_column = f"{column}_{stat}"
            output_columns.append(output_column)
            unit = table.unit_for(column)
            if unit and stat != "count":
                output_units[output_column] = unit

    records = table.records()
    grouped_records: dict[tuple[object, ...], list[dict[str, object]]] = {}
    key_values: dict[tuple[object, ...], tuple[object, ...]] = {}
    key_order: list[tuple[object, ...]] = []
    for record in records:
        raw_key = tuple(record[column] for column in group_columns)
        key = tuple(_hashable_table_key(value) for value in raw_key)
        if key not in grouped_records:
            grouped_records[key] = []
            key_values[key] = raw_key
            key_order.append(key)
        grouped_records[key].append(record)

    rows: list[tuple[object, ...]] = []
    for key in key_order:
        group_records = grouped_records[key]
        row: list[object] = list(key_values[key])
        row.append(len(group_records))
        for column in numeric_columns:
            values = [
                value
                for record in group_records
                if (value := _table_numeric_value(record.get(column))) is not None
            ]
            for stat in summary_stats:
                row.append(_summary_statistic(values, stat))
        rows.append(tuple(row))

    return TableData(
        columns=tuple(output_columns),
        rows=tuple(rows),
        name="Measurement summary",
        table_kind="Grouped measurement summary",
        source_name=table.source_name,
        column_units=tuple(output_units.items()),
    )


BRANCH_TYPE_ORDER = (
    "endpoint_to_endpoint",
    "endpoint_to_junction",
    "junction_to_junction",
    "cycle",
    "isolated",
)


def summarize_skeleton_branches(
    data,
    group_by: str = "auto",
    statistics: str = "mean,median,std,min,max,q25,q75",
) -> TableData:
    """Summarize a skeleton branch table into per-group branch distributions."""
    table = _validated_table(data)
    _validate_skeleton_branch_table(table)
    group_columns = _summary_group_columns(table, group_by)
    summary_stats = _summary_statistics(statistics)
    length_columns = _skeleton_branch_length_columns(table)
    tortuosity_column = (
        "branch_tortuosity" if "branch_tortuosity" in table.columns else ""
    )
    branch_types = _skeleton_branch_types(table)

    output_columns = list(group_columns) + [
        "branch_count",
        "component_count",
    ]
    output_units: dict[str, str] = {
        "branch_count": "count",
        "component_count": "count",
    }
    for column in length_columns:
        output_columns.append(f"{column}_total")
        unit = table.unit_for(column)
        if unit:
            output_units[f"{column}_total"] = unit
        for stat in summary_stats:
            output_column = f"{column}_{stat}"
            output_columns.append(output_column)
            if unit and stat != "count":
                output_units[output_column] = unit
    if tortuosity_column:
        for stat in summary_stats:
            output_columns.append(f"{tortuosity_column}_{stat}")
            output_units[f"{tortuosity_column}_{stat}"] = "ratio"
    for branch_type in branch_types:
        safe_type = _safe_column_token(branch_type)
        count_column = f"branch_type_{safe_type}_count"
        fraction_column = f"branch_type_{safe_type}_fraction"
        output_columns.extend((count_column, fraction_column))
        output_units[count_column] = "count"
        output_units[fraction_column] = "fraction"

    records = table.records()
    grouped_records: dict[tuple[object, ...], list[dict[str, object]]] = {}
    key_values: dict[tuple[object, ...], tuple[object, ...]] = {}
    key_order: list[tuple[object, ...]] = []
    for record in records:
        raw_key = tuple(record[column] for column in group_columns)
        key = tuple(_hashable_table_key(value) for value in raw_key)
        if key not in grouped_records:
            grouped_records[key] = []
            key_values[key] = raw_key
            key_order.append(key)
        grouped_records[key].append(record)

    rows: list[tuple[object, ...]] = []
    for key in key_order:
        group_records = grouped_records[key]
        branch_count = len(group_records)
        row: list[object] = list(key_values[key])
        row.append(branch_count)
        row.append(_unique_non_empty_count(group_records, "component_id"))
        for column in length_columns:
            values = _numeric_values_for_records(group_records, column)
            row.append(float(np.sum(values)) if values else 0.0)
            for stat in summary_stats:
                row.append(_summary_statistic(values, stat))
        if tortuosity_column:
            tortuosity_values = _numeric_values_for_records(
                group_records,
                tortuosity_column,
            )
            for stat in summary_stats:
                row.append(_summary_statistic(tortuosity_values, stat))
        type_counts = Counter(
            str(record.get("branch_type", "") or "")
            for record in group_records
        )
        for branch_type in branch_types:
            count = int(type_counts.get(branch_type, 0))
            row.append(count)
            row.append(float(count / branch_count) if branch_count else 0.0)
        rows.append(tuple(row))

    return TableData(
        columns=tuple(output_columns),
        rows=tuple(rows),
        name="Skeleton branch summary",
        table_kind="Skeleton branch summary",
        source_name=table.source_name,
        column_units=tuple(output_units.items()),
    )


def _table_inputs(inputs, input_count: int) -> list[TableData]:
    if isinstance(inputs, TableData):
        candidates = [inputs]
    else:
        candidates = list(inputs)
    count = max(int(input_count), 1)
    tables = [_validated_table(table) for table in candidates[:count]]
    if len(tables) < count:
        raise ValueError(f"Expected {count} table input(s), received {len(tables)}.")
    if not tables:
        raise ValueError("Merge Tables needs at least one table input.")
    return tables


def _parse_table_column_list(text: str) -> tuple[str, ...]:
    cleaned = str(text or "").strip()
    if not cleaned or cleaned.lower() == "auto":
        return ()
    columns: list[str] = []
    seen: set[str] = set()
    for part in cleaned.split(","):
        column = part.strip()
        if not column or column in seen:
            continue
        columns.append(column)
        seen.add(column)
    return tuple(columns)


def _summary_group_columns(table: TableData, group_by: str) -> tuple[str, ...]:
    requested = _parse_table_column_list(group_by)
    if requested:
        missing = [column for column in requested if column not in table.columns]
        if missing:
            raise ValueError(
                "Summarize Measurements could not find group column(s): "
                + ", ".join(missing)
            )
        return requested
    return tuple(
        column for column in SUMMARY_GROUP_COLUMN_PRIORITY if column in table.columns
    )


def _summary_value_columns(
    table: TableData,
    value_columns: str,
    group_columns: tuple[str, ...],
) -> tuple[str, ...]:
    requested = _parse_table_column_list(value_columns)
    if requested:
        missing = [column for column in requested if column not in table.columns]
        if missing:
            raise ValueError(
                "Summarize Measurements could not find value column(s): "
                + ", ".join(missing)
            )
        non_numeric = [
            column
            for column in requested
            if not _table_column_has_numeric_values(table, column)
        ]
        if non_numeric:
            raise ValueError(
                "Summarize Measurements value column(s) have no numeric values: "
                + ", ".join(non_numeric)
            )
        return requested

    excluded = set(group_columns) | SUMMARY_EXCLUDED_VALUE_COLUMNS
    return tuple(
        column
        for column in table.columns
        if column not in excluded
        and not column.endswith("_index")
        and _table_column_has_numeric_values(table, column)
    )


def _summary_statistics(statistics: str) -> tuple[str, ...]:
    parsed = _parse_summary_statistics(statistics)
    if not parsed:
        return DEFAULT_SUMMARY_STATISTICS
    return parsed


def _parse_summary_statistics(statistics: str) -> tuple[str, ...]:
    aliases = {
        "n": "count",
        "count": "count",
        "mean": "mean",
        "average": "mean",
        "avg": "mean",
        "median": "median",
        "std": "std",
        "sd": "std",
        "stdev": "std",
        "standard_deviation": "std",
        "standard deviation": "std",
        "min": "min",
        "minimum": "min",
        "max": "max",
        "maximum": "max",
        "sum": "sum",
        "total": "sum",
        "q25": "q25",
        "p25": "q25",
        "25%": "q25",
        "q75": "q75",
        "p75": "q75",
        "75%": "q75",
    }
    cleaned = str(statistics or "").replace("\n", ",").replace(";", ",")
    stats: list[str] = []
    seen: set[str] = set()
    for part in cleaned.split(","):
        token = part.strip().lower().replace("-", "_")
        if not token or token == "auto":
            continue
        stat = aliases.get(token)
        if stat is None:
            raise ValueError(f"Unsupported summary statistic: {part.strip()!r}.")
        if stat not in seen:
            stats.append(stat)
            seen.add(stat)
    return tuple(stats)


def _summary_statistic(values: Sequence[float], statistic: str) -> object:
    count = len(values)
    if statistic == "count":
        return int(count)
    if count == 0:
        return ""
    arr = np.asarray(values, dtype=np.float64)
    if statistic == "mean":
        return float(np.mean(arr))
    if statistic == "median":
        return float(np.median(arr))
    if statistic == "std":
        return float(np.std(arr, ddof=1)) if count > 1 else 0.0
    if statistic == "min":
        return float(np.min(arr))
    if statistic == "max":
        return float(np.max(arr))
    if statistic == "sum":
        return float(np.sum(arr))
    if statistic == "q25":
        return float(np.percentile(arr, 25))
    if statistic == "q75":
        return float(np.percentile(arr, 75))
    raise ValueError(f"Unsupported summary statistic: {statistic!r}.")


def _validate_skeleton_branch_table(table: TableData) -> None:
    required = {"branch_type"}
    missing = sorted(column for column in required if column not in table.columns)
    if missing:
        raise ValueError(
            "Summarize Skeleton Branches expects a skeleton branch table. "
            "Missing column(s): " + ", ".join(missing)
        )
    if not _skeleton_branch_length_columns(table):
        raise ValueError(
            "Summarize Skeleton Branches could not find a branch length column."
        )


def _skeleton_branch_length_columns(table: TableData) -> tuple[str, ...]:
    preferred = (
        "branch_length_physical",
        "branch_length_voxels",
        "branch_length_pixels",
    )
    return tuple(column for column in preferred if column in table.columns)


def _skeleton_branch_types(table: TableData) -> tuple[str, ...]:
    observed = {
        str(record.get("branch_type", "") or "")
        for record in table.records()
        if str(record.get("branch_type", "") or "")
    }
    ordered = [
        branch_type for branch_type in BRANCH_TYPE_ORDER if branch_type in observed
    ]
    ordered.extend(sorted(observed - set(ordered)))
    return tuple(ordered or BRANCH_TYPE_ORDER)


def _numeric_values_for_records(
    records: Sequence[dict[str, object]],
    column: str,
) -> list[float]:
    return [
        value
        for record in records
        if (value := _table_numeric_value(record.get(column))) is not None
    ]


def _unique_non_empty_count(
    records: Sequence[dict[str, object]],
    column: str,
) -> int:
    values = {
        value
        for record in records
        if (value := record.get(column, "")) not in {"", None}
    }
    return len(values)


def _safe_column_token(value: object) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    token = token.strip("_")
    return token or "unknown"


def _validated_table(value) -> TableData:
    if not isinstance(value, TableData):
        raise TypeError("This operation expects VIPP TableData input.")
    return value


def _labels_and_table_inputs(inputs) -> tuple[object, TableData]:
    if not isinstance(inputs, Sequence) or len(inputs) < 2:
        raise ValueError("Filter Labels By Property needs labels and table inputs.")
    return inputs[0], _validated_table(inputs[1])


def _label_id_column(table: TableData) -> str:
    for column in ("label_id", "object_id"):
        if column in table.columns:
            return column
    raise ValueError("Filter Labels By Property table needs a label_id column.")


def _property_filter_column(table: TableData, requested: str) -> str:
    text = str(requested).strip()
    if text and text.lower() != "auto":
        if text not in table.columns:
            raise ValueError(f"Property column {text!r} is not in the table.")
        if not _table_column_has_numeric_values(table, text):
            raise ValueError(f"Property column {text!r} has no numeric values.")
        return text

    for column in PROPERTY_FILTER_COLUMN_PRIORITY:
        if column in table.columns and _table_column_has_numeric_values(table, column):
            return column
    for column in table.columns:
        if column in IDENTITY_JOIN_COLUMNS or column.endswith("_index"):
            continue
        if _table_column_has_numeric_values(table, column):
            return column
    raise ValueError("Could not find a numeric property column to filter labels.")


def _table_column_has_numeric_values(table: TableData, column: str) -> bool:
    column_index = table.columns.index(column)
    return any(
        _table_numeric_value(row[column_index]) is not None for row in table.rows
    )


def _property_filter_records_by_block(
    table: TableData,
    leading_axis_names: tuple[str, ...],
    value_column: str,
    label_column: str,
    minimum: float,
    maximum: float,
    has_maximum: bool,
    remove_inside: bool,
) -> tuple[dict[tuple[object, ...], set[int]], dict[tuple[object, ...], set[int]]]:
    value_index = table.columns.index(value_column)
    label_index = table.columns.index(label_column)
    axis_columns = tuple(
        (axis_name, f"{axis_name}_index", table.columns.index(f"{axis_name}_index"))
        for axis_name in leading_axis_names
        if f"{axis_name}_index" in table.columns
    )
    kept: dict[tuple[object, ...], set[int]] = {}
    matched: dict[tuple[object, ...], set[int]] = {}
    for row in table.rows:
        label_id = _table_int_value(row[label_index])
        if label_id is None or label_id <= 0:
            continue
        if axis_columns:
            key_items: list[tuple[str, int]] = []
            for axis_name, _name, index in axis_columns:
                axis_value = _table_int_value(row[index])
                if axis_value is None:
                    key_items = []
                    break
                key_items.append((axis_name, axis_value))
            if not key_items:
                continue
            key = tuple(key_items)
        else:
            key = (None,)
        matched.setdefault(key, set()).add(label_id)
        value = _table_numeric_value(row[value_index])
        if value is None:
            continue
        inside = value >= minimum and (not has_maximum or value <= maximum)
        if remove_inside:
            inside = not inside
        if inside:
            kept.setdefault(key, set()).add(label_id)
    return kept, matched


def _property_filter_block_key(
    leading_axis_names: tuple[str, ...],
    block_index: tuple[int, ...],
) -> tuple[object, ...]:
    if not leading_axis_names:
        return (None,)
    return tuple(
        (axis_name, int(block_index[position]))
        for position, axis_name in enumerate(leading_axis_names)
    )


def _property_filter_labels_for_block(
    labels_by_block: dict[tuple[object, ...], set[int]],
    key: tuple[object, ...],
) -> set[int]:
    labels: set[int] = set()
    key_items = set(key)
    for record_key, record_labels in labels_by_block.items():
        if record_key == (None,) or set(record_key) <= key_items:
            labels.update(record_labels)
    return labels


def _table_numeric_value(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def _table_int_value(value) -> int | None:
    number = _table_numeric_value(value)
    if number is None:
        return None
    rounded = int(round(number))
    return rounded if np.isclose(number, rounded) else None


def _table_join_keys(
    tables: Sequence[TableData],
    join_keys: str,
) -> tuple[tuple[str, ...], bool]:
    raw = str(join_keys).strip()
    if raw and raw.lower() != "auto":
        keys = tuple(part.strip() for part in raw.split(",") if part.strip())
        if not keys:
            raise ValueError("Join keys cannot be blank.")
        missing = {
            key
            for key in keys
            for table in tables
            if key not in table.columns
        }
        if missing:
            raise ValueError(
                "Manual join keys must exist in every table: "
                + ", ".join(sorted(missing))
            )
        return keys, False

    common = set(tables[0].columns)
    for table in tables[1:]:
        common &= set(table.columns)
    keys = tuple(column for column in IDENTITY_JOIN_COLUMNS if column in common)
    if keys:
        return keys, False
    row_counts = {table.row_count for table in tables}
    if len(row_counts) == 1:
        return (), True
    raise ValueError(
        "Could not infer table join keys. Enter comma-separated join keys, "
        "or use tables with the same row count for row-position joining."
    )


def _normalized_join_mode(value: str) -> str:
    text = str(value).strip().lower()
    if text.startswith("inner"):
        return "inner"
    if text.startswith("outer"):
        return "outer"
    return "left"


def _table_record_index(
    records: list[dict[str, object]],
    key_columns: tuple[str, ...],
    row_position_join: bool,
    *,
    table_number: int,
) -> dict[tuple[object, ...], dict[str, object]]:
    indexed: dict[tuple[object, ...], dict[str, object]] = {}
    for row_index, record in enumerate(records):
        key = (
            (int(row_index),)
            if row_position_join
            else tuple(
                _hashable_table_key(record.get(column))
                for column in key_columns
            )
        )
        if key in indexed:
            labels = ", ".join(
                f"{column}={record.get(column)!r}" for column in key_columns
            )
            raise ValueError(
                f"Table {table_number} has duplicate rows for join key {labels}."
            )
        indexed[key] = record
    return indexed


def _table_key_order(
    records: list[dict[str, object]],
    key_columns: tuple[str, ...],
    row_position_join: bool,
) -> list[tuple[object, ...]]:
    if row_position_join:
        return [(int(index),) for index in range(len(records))]
    return [
        tuple(_hashable_table_key(record.get(column)) for column in key_columns)
        for record in records
    ]


def _joined_table_keys(
    mode: str,
    indexes: list[dict[tuple[object, ...], dict[str, object]]],
    orders: list[list[tuple[object, ...]]],
) -> list[tuple[object, ...]]:
    if mode == "inner":
        return [
            key
            for key in orders[0]
            if all(key in index for index in indexes[1:])
        ]
    if mode == "outer":
        joined: list[tuple[object, ...]] = []
        seen: set[tuple[object, ...]] = set()
        for order in orders:
            for key in order:
                if key in seen:
                    continue
                joined.append(key)
                seen.add(key)
        return joined
    return list(orders[0])


def _hashable_table_key(value) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return tuple(_hashable_table_key(item) for item in value)
    if isinstance(value, dict):
        return tuple(
            (str(key), _hashable_table_key(item))
            for key, item in sorted(value.items())
        )
    return value


def _first_available_record(
    indexes: list[dict[tuple[object, ...], dict[str, object]]],
    key: tuple[object, ...],
) -> dict[str, object] | None:
    for index in indexes:
        record = index.get(key)
        if record is not None:
            return record
    return None


def _unique_table_column(column: str, used: set[str], suffix: str) -> str:
    if column not in used:
        return column
    base = f"{column}_{suffix}"
    candidate = base
    index = 2
    while candidate in used:
        candidate = f"{base}_{index}"
        index += 1
    return candidate


def _first_table_unit(tables: Sequence[TableData], column: str) -> str:
    for table in tables:
        unit = table.unit_for(column)
        if unit:
            return unit
    return ""


def _merged_table_source_name(tables: Sequence[TableData]) -> str:
    names: list[str] = []
    seen: set[str] = set()
    for table in tables:
        name = str(table.source_name).strip()
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    return ", ".join(names)


def _parse_metadata_columns(text: str) -> tuple[tuple[str, str], ...]:
    cleaned = str(text).replace("\n", ",").replace(";", ",")
    pairs: list[tuple[str, str]] = []
    seen: set[str] = set()
    for part in cleaned.split(","):
        item = part.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                "Metadata columns must use name=value entries separated by commas."
            )
        name, value = item.split("=", 1)
        column = "_".join(name.strip().split())
        if not column:
            raise ValueError("Metadata column names cannot be blank.")
        if column in seen:
            raise ValueError(f"Metadata column {column!r} is listed more than once.")
        pairs.append((column, value.strip()))
        seen.add(column)
    return tuple(pairs)


def extract_channel(
    data,
    channel: int = 0,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
) -> np.ndarray:
    """Extract a single channel from a semantically marked channel axis."""
    arr = np.asarray(data)
    axis = _strict_channel_axis(
        arr,
        axis_names=axis_names,
        axis_types=axis_types,
    )
    if axis is None:
        raise ValueError(
            "Extract Channel requires an explicitly declared channel axis."
        )
    return _extract_channel(
        arr,
        channel=channel,
        channel_axis=axis,
    )


def combine_channels(
    inputs,
    input_count: int = 2,
    channel_axis: int | None = None,
    channel_colors: str = "",
) -> np.ndarray:
    """Combine multiple same-shaped inputs into a multichannel image."""
    arrays = [np.asarray(item) for item in inputs if item is not None]
    arrays = _active_input_arrays(arrays, input_count, "Combine Channels")

    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("Combine Channels inputs must have matching shapes.")

    axis = _validated_insert_axis(
        channel_axis,
        arrays[0].ndim,
        operation="Combine Channels",
    )
    return np.stack(arrays, axis=axis)


def assign_channel_colors(data, channel_colors: str = "") -> np.ndarray:
    """Pass image data through while updating carried channel colour metadata."""
    return np.asarray(data).copy()


def calculate_weighted_image(
    inputs,
    input_count: int = 2,
    weights: str = "1,1",
    offset: float = 0.0,
) -> np.ndarray:
    """Create a float image using one finite weight per active input."""
    arrays = [np.asarray(item) for item in inputs if item is not None]
    arrays = _active_input_arrays(arrays, input_count, "Image Calculator")

    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("Image Calculator inputs must have matching shapes.")

    parsed_weights = _parse_finite_weight_list(weights)
    if len(parsed_weights) != len(arrays):
        raise ValueError(
            "Image Calculator requires exactly "
            f"{len(arrays)} weight(s), one for each active input; "
            f"received {len(parsed_weights)}."
        )

    try:
        resolved_offset = float(offset)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Image Calculator offset must be finite.") from exc
    if not np.isfinite(resolved_offset):
        raise ValueError("Image Calculator offset must be finite.")

    result = np.zeros(shape, dtype=np.float32)
    for array, weight in zip(arrays, parsed_weights, strict=True):
        result += array.astype(np.float32, copy=False) * float(weight)
    result += resolved_offset
    return result


def _active_input_arrays(
    arrays: list[np.ndarray],
    input_count,
    operation: str,
) -> list[np.ndarray]:
    if isinstance(input_count, (bool, np.bool_)) or not isinstance(
        input_count,
        Integral,
    ):
        raise ValueError(f"{operation} input_count must be an integer.")
    requested = int(input_count)
    if requested < 1:
        raise ValueError(f"{operation} input_count must be at least 1.")
    if len(arrays) < requested:
        raise ValueError(
            f"{operation} needs {requested} connected inputs; received "
            f"{len(arrays)}."
        )
    return arrays[:requested]


def _validated_insert_axis(value, ndim: int, *, operation: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{operation} channel_axis must be an integer.")
    axis = int(value)
    if axis < 0:
        axis += ndim + 1
    if axis < 0 or axis > ndim:
        raise ValueError(
            f"{operation} channel_axis {int(value)} is out of range for "
            f"insertion into a {ndim}D input."
        )
    return axis


def composite_to_rgb(
    data,
    channel_axis: int | None = None,
    red_channel: int = -1,
    green_channel: int = -1,
    blue_channel: int = -1,
    intensity_mapping: str = COMPOSITE_RGB_PRESERVE_VALUES,
    channel_axis_semantics: str = "",
    channel_colors: str | list[int | str] | tuple[int | str, ...] = "",
) -> np.ndarray:
    """Map an explicitly declared channel axis into channel-last RGB.

    ``Preserve numeric values`` retains the native intensity scale: channel
    values are copied or additively mixed without normalization or clipping.
    Inputs that cannot be represented safely in the RGB arithmetic are rejected
    instead of being rounded or overflowed silently.
    The legacy 1st/99th-percentile display mapping remains available only via
    ``Per-channel 1st-99th percentile (lossy)``. That mode normalizes every
    selected channel independently and clips additive colour mixtures to [0, 1].

    Only an axis explicitly declared as ``rgb`` or ``rgba`` preserves encoded
    RGB order automatically. Other channel axes use the declared/default
    fluorescence colour table unless explicit R/G/B channel indices are given.
    ``-1`` channel selections mean "use the positional default"; every explicit
    channel index must exist.
    """
    arr = np.asarray(data)
    axis = _validated_composite_channel_axis(channel_axis, arr.ndim)
    mapping = _validated_composite_intensity_mapping(intensity_mapping)
    _validate_composite_input_values(arr, mapping)

    moved = np.moveaxis(arr, axis, -1)
    count = int(moved.shape[-1])
    if count < 1:
        raise ValueError("Composite → RGB channel axis must not be empty.")
    semantic = _validated_composite_axis_semantics(
        channel_axis_semantics,
        count=count,
    )
    selections = _validated_composite_channel_selections(
        (red_channel, green_channel, blue_channel),
        count=count,
    )
    output_dtype = _composite_output_dtype(arr, mapping)

    manual_mapping = any(selection >= 0 for selection in selections)
    true_rgb = _is_true_rgb_channel_axis(
        count,
        channel_axis_semantics=semantic,
    )
    if count == 1 and not manual_mapping:
        gray = _composite_mapped_channel(
            moved[..., 0],
            intensity_mapping=mapping,
            output_dtype=output_dtype,
        )
        return np.stack([gray, gray, gray], axis=-1)
    if not manual_mapping and not true_rgb:
        color_table = _validated_composite_color_table(channel_colors, count)
        return _composite_to_rgb_by_color_table(
            moved,
            color_table,
            intensity_mapping=mapping,
            output_dtype=output_dtype,
        )

    defaults = _default_rgb_channel_indices(
        count,
        channel_axis_semantics=semantic,
    )
    blank = np.zeros(moved.shape[:-1], dtype=output_dtype)
    channels = []
    for position, selection in enumerate(selections):
        index = defaults[position] if selection < 0 else selection
        if index is None:
            channels.append(blank)
            continue
        channels.append(
            _composite_mapped_channel(
                moved[..., index],
                intensity_mapping=mapping,
                output_dtype=output_dtype,
            )
        )
    return np.stack(channels, axis=-1)


def split_channels(
    data,
    preview_channel: int = 0,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
) -> list[np.ndarray]:
    """Split a multi-channel image into one output array per channel.

    The channel axis is detected from carried axis metadata or conventional
    RGB/RGBA channel-last arrays. Each output is a single channel with the
    spatial shape of the input. Images without a channel axis should use
    ``split_axis`` instead of silently treating Z or time as channels.

    ``preview_channel`` is a UI-only hint used by the graph thumbnail and does
    not affect the returned channel outputs.
    """
    del preview_channel
    arr = np.asarray(data)
    axis = _strict_channel_axis(
        arr,
        axis_names=axis_names,
        axis_types=axis_types,
    )
    if axis is None:
        raise ValueError(
            "Split Channels needs a channel axis. Use Split Axis to split "
            "timepoints, Z slices, or another image axis."
        )

    moved = np.moveaxis(arr, axis, -1)
    count = moved.shape[-1]
    return [moved[..., index].copy() for index in range(count)]


def split_axis(
    data,
    axis: str = "axis:0",
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
) -> list[np.ndarray]:
    """Split an image into one output per index of a selected axis."""
    del axis_names, axis_types
    arr = np.asarray(data)
    if arr.ndim == 0:
        return [arr.copy()]
    axis_index = _axis_index_from_token(axis, arr.ndim)
    return [
        np.take(arr, index, axis=axis_index).copy()
        for index in range(arr.shape[axis_index])
    ]


def colocalization_threshold_values(
    inputs,
    threshold_mode: str = "Manual",
    channel_1_threshold: float = 25.0,
    channel_2_threshold: float = 25.0,
    intensity_max: float = 255.0,
) -> tuple[float, float]:
    """Return the threshold pair used by a colocalization node."""
    ch1, ch2, roi_mask, _warnings = _coloc_normalized_inputs_and_mask(
        inputs,
        intensity_max=intensity_max,
    )
    threshold_1, threshold_2, _costes = _coloc_thresholds(
        ch1,
        ch2,
        threshold_mode=threshold_mode,
        channel_1_threshold=channel_1_threshold,
        channel_2_threshold=channel_2_threshold,
        intensity_max=intensity_max,
        roi_mask=roi_mask,
    )
    return threshold_1, threshold_2


def colocalization_normalized_inputs(
    inputs,
    intensity_max: float = 255.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...]]:
    """Return normalized channel arrays and optional ROI mask for UI diagnostics."""
    return _coloc_normalized_inputs_and_mask(inputs, intensity_max=intensity_max)


def colocalization_metrics(
    inputs,
    threshold_mode: str = "Manual",
    channel_1_threshold: float = 25.0,
    channel_2_threshold: float = 25.0,
    intensity_max: float = 255.0,
) -> TableData:
    """Measure whole-image two-channel colocalization using normalized thresholds."""
    ch1, ch2, roi_mask, warnings = _coloc_normalized_inputs_and_mask(
        inputs,
        intensity_max=intensity_max,
    )
    threshold_1, threshold_2, costes = _coloc_thresholds(
        ch1,
        ch2,
        threshold_mode=threshold_mode,
        channel_1_threshold=channel_1_threshold,
        channel_2_threshold=channel_2_threshold,
        intensity_max=intensity_max,
        roi_mask=roi_mask,
    )
    if roi_mask is None:
        roi_mask = np.ones(ch1.shape, dtype=bool)
    roi_voxels = int(np.count_nonzero(roi_mask))
    positive_1 = (ch1 >= threshold_1) & roi_mask
    positive_2 = (ch2 >= threshold_2) & roi_mask
    overlap = positive_1 & positive_2

    sum_1_positive = float(np.sum(ch1[positive_1], dtype=np.float64))
    sum_2_positive = float(np.sum(ch2[positive_2], dtype=np.float64))
    sum_1_overlap = float(np.sum(ch1[overlap], dtype=np.float64))
    sum_2_overlap = float(np.sum(ch2[overlap], dtype=np.float64))

    columns = {
        "threshold_mode": [str(threshold_mode)],
        "channel_1_threshold": [float(threshold_1)],
        "channel_2_threshold": [float(threshold_2)],
        "threshold_units": [f"normalized_0_{_format_coloc_max(intensity_max)}"],
        "mask_restricted": [roi_voxels != int(ch1.size)],
        "total_voxels": [roi_voxels],
        "channel_1_positive_voxels": [int(np.count_nonzero(positive_1))],
        "channel_2_positive_voxels": [int(np.count_nonzero(positive_2))],
        "colocalized_voxels": [int(np.count_nonzero(overlap))],
        "colocalized_fraction": [
            _safe_fraction(np.count_nonzero(overlap), roi_voxels)
        ],
        "pearson_all": [_pearson(ch1[roi_mask], ch2[roi_mask])],
        "pearson_colocalized": [_pearson(ch1[overlap], ch2[overlap])],
        "manders_m1": [_safe_fraction(sum_1_overlap, sum_1_positive)],
        "manders_m2": [_safe_fraction(sum_2_overlap, sum_2_positive)],
        "overlap_coefficient_all": [
            _overlap_coefficient(ch1[roi_mask], ch2[roi_mask])
        ],
        "overlap_coefficient_colocalized": [
            _overlap_coefficient(ch1[overlap], ch2[overlap])
        ],
        "channel_1_positive_sum": [sum_1_positive],
        "channel_2_positive_sum": [sum_2_positive],
        "colocalized_channel_1_sum": [sum_1_overlap],
        "colocalized_channel_2_sum": [sum_2_overlap],
        "costes_slope": [float("nan") if costes is None else costes["slope"]],
        "costes_intercept": [
            float("nan") if costes is None else costes["intercept"]
        ],
        "costes_pearson_below": [
            float("nan") if costes is None else costes["pearson_below"]
        ],
        "costes_iterations": [0 if costes is None else int(costes["iterations"])],
        "normalization_warnings": ["; ".join(warnings)],
    }
    return table_from_columns(
        columns,
        name="Colocalization metrics",
        table_kind="colocalization metrics",
        column_units={
            "channel_1_threshold": f"normalized_0_{_format_coloc_max(intensity_max)}",
            "channel_2_threshold": f"normalized_0_{_format_coloc_max(intensity_max)}",
            "total_voxels": "voxels",
            "channel_1_positive_voxels": "voxels",
            "channel_2_positive_voxels": "voxels",
            "colocalized_voxels": "voxels",
        },
    )


def object_colocalization_metrics(
    inputs,
    threshold_mode: str = "Manual",
    channel_1_threshold: float = 25.0,
    channel_2_threshold: float = 25.0,
    intensity_max: float = 255.0,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Measure two-channel colocalization for each labeled object."""
    values = list(inputs)
    if len(values) < 3:
        raise ValueError(
            "Object colocalization requires labels plus two channel image inputs."
        )
    label_source = np.asarray(values[0])
    context = _spatial_table_context(
        label_source,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    labels = _objects_from_mask_or_labels(label_source, context=context)
    ch1_raw = _coloc_reduce_to_intensity(values[1])
    ch2_raw = _coloc_reduce_to_intensity(values[2])
    if labels.shape != ch1_raw.shape or labels.shape != ch2_raw.shape:
        raise ValueError(
            "Object colocalization requires labels and channels with matching "
            f"shapes; got {labels.shape}, {ch1_raw.shape}, and {ch2_raw.shape}."
        )

    ch1, ch2, warnings = _normalize_coloc_channels(
        ch1_raw,
        ch2_raw,
        output_max=float(intensity_max),
    )
    foreground = labels > 0
    threshold_1, threshold_2, costes = _coloc_thresholds(
        ch1,
        ch2,
        threshold_mode=threshold_mode,
        channel_1_threshold=channel_1_threshold,
        channel_2_threshold=channel_2_threshold,
        intensity_max=intensity_max,
        roi_mask=foreground if np.count_nonzero(foreground) else None,
    )
    labels_for_measure = _move_to_spatial_last(
        labels,
        context.spatial_axes,
        context.spatial_ndim,
    )
    ch1_for_measure = _move_to_spatial_last(
        ch1,
        context.spatial_axes,
        context.spatial_ndim,
    )
    ch2_for_measure = _move_to_spatial_last(
        ch2,
        context.spatial_axes,
        context.spatial_ndim,
    )

    columns = _object_colocalization_columns(context.leading_axis_names)
    for leading_index in np.ndindex(context.leading_shape or (1,)):
        block_index = () if not context.leading_shape else leading_index
        label_block = (
            labels_for_measure[block_index]
            if context.leading_shape
            else labels_for_measure
        )
        ch1_block = (
            ch1_for_measure[block_index] if context.leading_shape else ch1_for_measure
        )
        ch2_block = (
            ch2_for_measure[block_index] if context.leading_shape else ch2_for_measure
        )
        _append_object_colocalization_rows(
            columns,
            context.leading_axis_names,
            block_index,
            label_block,
            ch1_block,
            ch2_block,
            threshold_mode=threshold_mode,
            threshold_1=threshold_1,
            threshold_2=threshold_2,
            intensity_max=intensity_max,
            costes=costes,
            warnings=warnings,
        )

    threshold_unit = f"normalized_0_{_format_coloc_max(intensity_max)}"
    return table_from_columns(
        columns,
        name="Object colocalization metrics",
        table_kind="per-object colocalization metrics",
        source_name=source_name,
        column_units={
            "channel_1_threshold": threshold_unit,
            "channel_2_threshold": threshold_unit,
            "object_voxels": "voxels",
            "channel_1_positive_voxels": "voxels",
            "channel_2_positive_voxels": "voxels",
            "colocalized_voxels": "voxels",
            "colocalized_fraction": "fraction",
            "manders_m1": "fraction",
            "manders_m2": "fraction",
            "channel_1_positive_sum": "intensity",
            "channel_2_positive_sum": "intensity",
            "colocalized_channel_1_sum": "intensity",
            "colocalized_channel_2_sum": "intensity",
        },
    )


def label_overlap_association(
    inputs,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Return one row for each overlapping reference/target label pair."""
    reference_source, target_source, context = _label_pair_inputs_and_context(
        inputs,
        "Label overlap association requires reference and target label inputs.",
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    reference = _objects_from_mask_or_labels(reference_source, context=context)
    target = _objects_from_mask_or_labels(target_source, context=context)
    reference_for_measure = _move_to_spatial_last(
        reference,
        context.spatial_axes,
        context.spatial_ndim,
    )
    target_for_measure = _move_to_spatial_last(
        target,
        context.spatial_axes,
        context.spatial_ndim,
    )
    columns = _label_overlap_columns(context.leading_axis_names)
    for leading_index in np.ndindex(context.leading_shape or (1,)):
        block_index = () if not context.leading_shape else leading_index
        reference_block = (
            reference_for_measure[block_index]
            if context.leading_shape
            else reference_for_measure
        )
        target_block = (
            target_for_measure[block_index]
            if context.leading_shape
            else target_for_measure
        )
        _append_label_overlap_rows(
            columns,
            context.leading_axis_names,
            block_index,
            reference_block,
            target_block,
        )
    return table_from_columns(
        columns,
        name="Label overlap association",
        table_kind="label overlap association",
        source_name=source_name,
        column_units={
            "reference_voxels": "voxels",
            "target_voxels": "voxels",
            "overlap_voxels": "voxels",
            "reference_overlap_fraction": "fraction",
            "target_overlap_fraction": "fraction",
            "intersection_over_union": "fraction",
        },
    )


def nearest_object_distance(
    inputs,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Measure nearest target-object centroid distance for each reference label."""
    reference_source, target_source, context = _label_pair_inputs_and_context(
        inputs,
        "Nearest object distance requires reference and target label inputs.",
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    reference = _objects_from_mask_or_labels(reference_source, context=context)
    target = _objects_from_mask_or_labels(target_source, context=context)
    reference_for_measure = _move_to_spatial_last(
        reference,
        context.spatial_axes,
        context.spatial_ndim,
    )
    target_for_measure = _move_to_spatial_last(
        target,
        context.spatial_axes,
        context.spatial_ndim,
    )
    columns = _nearest_distance_columns(context)
    for leading_index in np.ndindex(context.leading_shape or (1,)):
        block_index = () if not context.leading_shape else leading_index
        reference_block = (
            reference_for_measure[block_index]
            if context.leading_shape
            else reference_for_measure
        )
        target_block = (
            target_for_measure[block_index]
            if context.leading_shape
            else target_for_measure
        )
        _append_nearest_distance_rows(
            columns,
            context,
            block_index,
            reference_block,
            target_block,
        )
    units = {"centroid_distance_pixels": "pixels"}
    if context.has_physical_calibration:
        units["centroid_distance_physical"] = context.physical_unit
        units["physical_unit"] = "text"
    return table_from_columns(
        columns,
        name="Nearest object distance",
        table_kind="nearest object distance",
        source_name=source_name,
        column_units=units,
    )


def event_localization(
    inputs,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_names: tuple[str, ...] | None = None,
    axis_types: tuple[str, ...] | None = None,
    axis_scales: tuple[float, ...] | None = None,
    axis_units: tuple[str | None, ...] | None = None,
    source_name: str = "",
) -> TableData:
    """Assign event/puncta objects to overlapping labels, masks, or ROIs."""
    values = list(inputs)
    if len(values) < 2:
        raise ValueError("Event localization requires event and region inputs.")
    event_source = np.asarray(values[0])
    region_source = np.asarray(values[1])
    if event_source.shape != region_source.shape:
        raise ValueError(
            "Event localization inputs must have matching shapes; got "
            f"{event_source.shape} and {region_source.shape}."
        )
    context = _spatial_table_context(
        event_source,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    events = _event_objects_from_mask_or_labels(event_source, context=context)
    regions = _regions_from_mask_or_labels(region_source)
    if events.shape != regions.shape:
        raise ValueError(
            "Event localization inputs must have matching shapes; got "
            f"{events.shape} and {regions.shape}."
        )
    events_for_measure = _move_to_spatial_last(
        events,
        context.spatial_axes,
        context.spatial_ndim,
    )
    regions_for_measure = _move_to_spatial_last(
        regions,
        context.spatial_axes,
        context.spatial_ndim,
    )
    columns = _event_localization_columns(context.leading_axis_names)
    for leading_index in np.ndindex(context.leading_shape or (1,)):
        block_index = () if not context.leading_shape else leading_index
        event_block = (
            events_for_measure[block_index]
            if context.leading_shape
            else events_for_measure
        )
        region_block = (
            regions_for_measure[block_index]
            if context.leading_shape
            else regions_for_measure
        )
        _append_event_localization_rows(
            columns,
            context.leading_axis_names,
            block_index,
            event_block,
            region_block,
        )
    return table_from_columns(
        columns,
        name="Event localization",
        table_kind="event localization",
        source_name=source_name,
        column_units={
            "event_voxels": "voxels",
            "overlap_voxels": "voxels",
            "event_overlap_fraction": "fraction",
        },
    )


def colocalized_voxels(
    inputs,
    threshold_mode: str = "Manual",
    channel_1_threshold: float = 25.0,
    channel_2_threshold: float = 25.0,
    display_mode: str = "White overlay on channels",
    channel_1_color: str = "Red",
    channel_2_color: str = "Green",
    intensity_max: float = 255.0,
) -> np.ndarray:
    """Render threshold-colocalized voxels as a channel-last RGB image."""
    ch1, ch2, roi_mask, _warnings = _coloc_normalized_inputs_and_mask(
        inputs,
        intensity_max=intensity_max,
    )
    threshold_1, threshold_2, _costes = _coloc_thresholds(
        ch1,
        ch2,
        threshold_mode=threshold_mode,
        channel_1_threshold=channel_1_threshold,
        channel_2_threshold=channel_2_threshold,
        intensity_max=intensity_max,
        roi_mask=roi_mask,
    )
    return _coloc_voxel_overlay(
        ch1,
        ch2,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        display_mode=display_mode,
        channel_1_color=channel_1_color,
        channel_2_color=channel_2_color,
        intensity_max=intensity_max,
        roi_mask=roi_mask,
    )


def colocalization_scatter_plot(
    inputs,
    threshold_mode: str = "Manual",
    channel_1_threshold: float = 25.0,
    channel_2_threshold: float = 25.0,
    bins: int = 128,
    log_counts: bool = True,
    intensity_max: float = 255.0,
) -> np.ndarray:
    """Render a two-channel scatter-density image with threshold guides."""
    ch1, ch2, roi_mask, _warnings = _coloc_normalized_inputs_and_mask(
        inputs,
        intensity_max=intensity_max,
    )
    threshold_1, threshold_2, _costes = _coloc_thresholds(
        ch1,
        ch2,
        threshold_mode=threshold_mode,
        channel_1_threshold=channel_1_threshold,
        channel_2_threshold=channel_2_threshold,
        intensity_max=intensity_max,
        roi_mask=roi_mask,
    )
    return _coloc_scatter_plot_image(
        ch1,
        ch2,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        bins=bins,
        log_counts=log_counts,
        intensity_max=intensity_max,
        roi_mask=roi_mask,
    )


def racc_index(
    inputs,
    threshold_mode: str = "Manual",
    channel_1_threshold: float = 25.0,
    channel_2_threshold: float = 25.0,
    theta_degrees: float = 45.0,
    include_percentile: float = 99.0,
    intensity_max: float = 255.0,
    output_dtype: str = "float32",
) -> np.ndarray:
    """Compute a RACC-like index image from two thresholded channels.

    The core calculation follows the same numerical model as the separate RACC
    napari plugin, but is exposed here as a VIPP graph node so it can be
    chained with VIPP preprocessing and export.
    """
    ch1, ch2, roi_mask, _warnings = _coloc_normalized_inputs_and_mask(
        inputs,
        intensity_max=intensity_max,
    )
    threshold_1, threshold_2, _costes = _coloc_thresholds(
        ch1,
        ch2,
        threshold_mode=threshold_mode,
        channel_1_threshold=channel_1_threshold,
        channel_2_threshold=channel_2_threshold,
        intensity_max=intensity_max,
        roi_mask=roi_mask,
    )
    output = _racc_index_image(
        ch1,
        ch2,
        threshold_1=threshold_1,
        threshold_2=threshold_2,
        theta_degrees=theta_degrees,
        include_percentile=include_percentile,
        intensity_max=intensity_max,
        roi_mask=roi_mask,
    )
    if str(output_dtype).strip().lower() == "uint8":
        return np.round(np.clip(output, 0.0, 1.0) * 255.0).astype(np.uint8)
    return output.astype(np.float32, copy=False)



def rescale_intensity(
    data,
    in_low_percentile: float = 0.0,
    in_high_percentile: float = 100.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
    in_low_value: float | None = None,
    in_high_value: float | None = None,
    *,
    cutoff_mode: str = "Percentiles",
) -> np.ndarray:
    """Rescale intensity from input cutoffs to a requested output range."""
    arr = np.asarray(data)
    mode = str(cutoff_mode).strip().casefold()
    if mode not in {"percentiles", "values"}:
        raise ValueError(
            "Rescale Intensity input cutoffs must be 'Percentiles' or 'Values'."
        )
    integer_data = np.issubdtype(arr.dtype, np.integer)
    output_minimum = (
        _required_finite_fraction(out_min, "Output minimum")
        if integer_data
        else _required_finite_float(out_min, "Output minimum")
    )
    output_maximum = (
        _required_finite_fraction(out_max, "Output maximum")
        if integer_data
        else _required_finite_float(out_max, "Output maximum")
    )
    if mode == "percentiles":
        low_p = _required_finite_float(in_low_percentile, "Low percentile")
        high_p = _required_finite_float(in_high_percentile, "High percentile")
        if not 0.0 <= low_p <= 100.0 or not 0.0 <= high_p <= 100.0:
            raise ValueError("Rescale percentiles must be between 0 and 100.")
        if low_p > high_p:
            raise ValueError(
                "Rescale low percentile must not exceed the high percentile."
            )
        if arr.dtype == bool:
            return arr.copy()
        if integer_data:
            low, high = exact_integer_percentiles(arr, (low_p, high_p))
            return _rescale_integer_intensity(
                arr,
                low=low,
                high=high,
                output_minimum=output_minimum,
                output_maximum=output_maximum,
                output_minimum_source=out_min,
                output_maximum_source=out_max,
            )
        values = arr[np.isfinite(arr)]
        if values.size == 0:
            raise ValueError(
                "Rescale percentile cutoffs require at least one finite input value."
            )
        low, high = np.percentile(
            values,
            [low_p, high_p],
            overwrite_input=True,
        )
    else:
        low = (
            _required_finite_fraction(in_low_value, "Low input value")
            if integer_data
            else _required_finite_float(in_low_value, "Low input value")
        )
        high = (
            _required_finite_fraction(in_high_value, "High input value")
            if integer_data
            else _required_finite_float(in_high_value, "High input value")
        )
        if low > high:
            raise ValueError(
                "Rescale low input value must not exceed the high input value."
            )
        if arr.dtype == bool:
            return arr.copy()
        if integer_data:
            _reject_rounded_wide_integer_control(
                arr,
                in_low_value,
                "Low input value",
            )
            _reject_rounded_wide_integer_control(
                arr,
                in_high_value,
                "High input value",
            )
            return _rescale_integer_intensity(
                arr,
                low=low,
                high=high,
                output_minimum=output_minimum,
                output_maximum=output_maximum,
                output_minimum_source=out_min,
                output_maximum_source=out_max,
            )
    if high == low:
        return _cast_rescaled_intensity(
            np.full(arr.shape, output_minimum, dtype=np.float64),
            arr.dtype,
        )
    scaled = (arr.astype(np.float64, copy=False) - float(low)) / float(high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    output = scaled * (output_maximum - output_minimum) + output_minimum
    return _cast_rescaled_intensity(output, arr.dtype)


def _required_finite_float(value, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    return parsed


def _required_finite_fraction(value, label: str) -> Fraction:
    """Parse a saved numeric value without discarding exact integer levels."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{label} must be a finite number.")
    if isinstance(value, Integral):
        return Fraction(int(value))
    try:
        parsed = float(value)
        exact = Fraction(str(value))
    except (TypeError, ValueError, ZeroDivisionError, OverflowError) as exc:
        raise ValueError(f"{label} must be a finite number.") from exc
    if not np.isfinite(parsed):
        raise ValueError(f"{label} must be a finite number.")
    return exact


def exact_integer_percentiles(
    data,
    percentiles: Sequence[float],
) -> tuple[int | Fraction, ...]:
    """Return exact linear percentiles without converting native levels to float."""
    arr = np.asarray(data)
    if not np.issubdtype(arr.dtype, np.integer) or arr.dtype == bool:
        raise TypeError("Exact integer percentiles require non-boolean integer data.")
    requested = tuple(
        _required_finite_float(value, "Percentile") for value in percentiles
    )
    if any(value < 0.0 or value > 100.0 for value in requested):
        raise ValueError("Percentiles must be between 0 and 100.")
    stats = _integer_array_stats(arr)
    if stats.count == 0:
        raise ValueError("Percentile cutoffs require at least one input value.")
    if stats.minimum == stats.maximum:
        return tuple(int(stats.minimum) for _value in requested)
    if all(value in {0.0, 100.0} for value in requested):
        return tuple(
            int(stats.minimum) if value == 0.0 else int(stats.maximum)
            for value in requested
        )

    ranks: list[Fraction] = []
    indices: set[int] = set()
    for value in requested:
        rank = (Fraction(str(value)) / 100) * (stats.count - 1)
        ranks.append(rank)
        indices.add(math.floor(rank))
        indices.add(math.ceil(rank))

    values = np.asarray(arr).reshape(-1).copy()
    values.partition(tuple(sorted(indices)))
    result: list[int | Fraction] = []
    for rank in ranks:
        lower_index = math.floor(rank)
        upper_index = math.ceil(rank)
        lower = int(values[lower_index])
        if lower_index == upper_index:
            result.append(lower)
            continue
        upper = int(values[upper_index])
        cutoff = Fraction(lower) + (rank - lower_index) * (upper - lower)
        result.append(int(cutoff) if cutoff.denominator == 1 else cutoff)
    return tuple(result)


def _rescale_integer_intensity(
    arr: np.ndarray,
    *,
    low: int | Fraction,
    high: int | Fraction,
    output_minimum: Fraction,
    output_maximum: Fraction,
    output_minimum_source,
    output_maximum_source,
) -> np.ndarray:
    """Rescale integer levels in translated coordinates and restore them exactly."""
    low = Fraction(low)
    high = Fraction(high)
    if arr.size == 0:
        return arr.copy()
    if output_minimum.denominator != 1 or output_maximum.denominator != 1:
        raise ValueError(
            "Integer Rescale Intensity output bounds must be whole numbers; "
            "convert the image to floating point for fractional output bounds."
        )
    output_low = int(output_minimum)
    output_high = int(output_maximum)
    info = np.iinfo(arr.dtype)
    if not info.min <= output_low <= info.max or not (
        info.min <= output_high <= info.max
    ):
        raise ValueError(
            f"Rescale Intensity output bounds must fit the {arr.dtype} range "
            f"{info.min}..{info.max}."
        )
    _reject_rounded_wide_integer_control(
        arr,
        output_minimum_source,
        "Output minimum",
    )
    _reject_rounded_wide_integer_control(
        arr,
        output_maximum_source,
        "Output maximum",
    )
    output_span = output_high - output_low
    if abs(output_span) > _FLOAT64_EXACT_INTEGER_LIMIT:
        raise ValueError(
            "Integer Rescale Intensity output span exceeds 2^53 levels, so "
            "float64 scaling cannot preserve every output level. Choose a "
            "narrower range or convert dtype explicitly."
        )
    if high == low:
        return np.full(arr.shape, output_low, dtype=arr.dtype)
    input_span = high - low
    if input_span > _FLOAT64_EXACT_INTEGER_LIMIT:
        raise ValueError(
            "Integer Rescale Intensity cutoff span exceeds 2^53 levels, so "
            "float64 scaling cannot distinguish every native input level. "
            "Choose narrower explicit cutoffs or convert dtype explicitly."
        )

    low_floor = math.floor(low)
    high_ceil = math.ceil(high)
    output = np.empty_like(arr)
    iterator = np.nditer(
        [arr, output],
        flags=["buffered", "external_loop", "refs_ok", "zerosize_ok"],
        op_flags=[["readonly"], ["writeonly"]],
        order="K",
        buffersize=_THRESHOLD_CHUNK_SIZE,
    )
    output_low_native = np.asarray(output_low, dtype=arr.dtype)
    output_high_native = np.asarray(output_high, dtype=arr.dtype)
    for source_chunk, output_chunk in iterator:
        source_values = np.asarray(source_chunk)
        output_values = np.asarray(output_chunk)
        low_mask = _integer_at_most(source_values, low_floor)
        high_mask = _integer_at_least(source_values, high_ceil)
        interior = ~(low_mask | high_mask)
        output_values[low_mask] = output_low_native
        output_values[high_mask] = output_high_native
        if not np.any(interior):
            continue

        relative = source_values[interior].astype(np.uint64, copy=True)
        relative -= np.uint64(low_floor % 2**64)
        normalized = relative.astype(np.float64, copy=False)
        normalized -= float(low - low_floor)
        normalized /= float(input_span)
        np.clip(normalized, 0.0, 1.0, out=normalized)
        normalized *= float(output_span)
        offsets = np.rint(normalized).astype(np.int64)
        restored = offsets.astype(np.uint64)
        restored += np.uint64(output_low % 2**64)
        if arr.dtype == np.dtype(np.int64):
            output_values[interior] = restored.view(np.int64)
        else:
            output_values[interior] = restored.astype(arr.dtype)
    return output


def _integer_at_most(arr: np.ndarray, bound: int) -> np.ndarray:
    info = np.iinfo(arr.dtype)
    if bound < info.min:
        return np.zeros(arr.shape, dtype=bool)
    if bound >= info.max:
        return np.ones(arr.shape, dtype=bool)
    return arr <= np.asarray(bound, dtype=arr.dtype)


def _integer_at_least(arr: np.ndarray, bound: int) -> np.ndarray:
    info = np.iinfo(arr.dtype)
    if bound <= info.min:
        return np.ones(arr.shape, dtype=bool)
    if bound > info.max:
        return np.zeros(arr.shape, dtype=bool)
    return arr >= np.asarray(bound, dtype=arr.dtype)


def _reject_rounded_wide_integer_control(
    arr: np.ndarray,
    source_value,
    label: str,
) -> None:
    """Reject GUI floating values that cannot identify adjacent wide levels."""
    if isinstance(source_value, Integral):
        return
    try:
        magnitude = abs(float(source_value))
    except (TypeError, ValueError, OverflowError):
        return
    if magnitude <= _FLOAT64_EXACT_INTEGER_LIMIT:
        return
    raise ValueError(
        f"{label} was saved as a floating-point value above 2^53, where adjacent "
        "integer levels cannot be represented. Use an exact integer workflow "
        "value, percentile cutoffs, or convert the image dtype first."
    )


def _cast_rescaled_intensity(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(bool):
        return values.astype(bool)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        rounded = np.rint(values)
        clipped = np.clip(rounded, info.min, info.max)
        return clipped.astype(dtype)
    if np.issubdtype(dtype, np.floating):
        return values.astype(dtype, copy=False)
    return values.astype(dtype, copy=False)


def normalize_image(
    data,
    method: str = "min-max",
) -> np.ndarray:
    """Normalize an image to min-max or z-score float output."""
    arr = np.asarray(data)
    output_dtype = arr.dtype if np.issubdtype(arr.dtype, np.floating) else np.float32
    if arr.dtype == bool:
        return arr.astype(output_dtype)
    values = arr[np.isfinite(arr)].astype(output_dtype, copy=False)
    if values.size == 0:
        return np.zeros_like(arr, dtype=output_dtype)
    if str(method).lower() == "z-score":
        mean = float(values.mean())
        std = float(values.std())
        if std == 0:
            return np.zeros_like(arr, dtype=output_dtype)
        return ((arr.astype(output_dtype, copy=False) - mean) / std).astype(
            output_dtype,
            copy=False,
        )
    return _rescale_values(arr.astype(output_dtype, copy=False), 0.0, 1.0).astype(
        output_dtype,
        copy=False,
    )


def clip_intensity(
    data,
    minimum: float = 0.0,
    maximum: float = 255.0,
    *,
    cutoff_mode: str = "Data range",
) -> np.ndarray:
    """Clip intensities while preserving the incoming dtype where possible."""
    arr = np.asarray(data)
    mode = str(cutoff_mode).strip().casefold()
    if mode not in {"data range", "values"}:
        raise ValueError("Clip input cutoffs must be 'Data range' or 'Values'.")
    if mode == "data range":
        return arr.copy()
    integer_data = np.issubdtype(arr.dtype, np.integer)
    low = (
        _required_finite_fraction(minimum, "Clip minimum")
        if integer_data
        else _required_finite_float(minimum, "Clip minimum")
    )
    high = (
        _required_finite_fraction(maximum, "Clip maximum")
        if integer_data
        else _required_finite_float(maximum, "Clip maximum")
    )
    if low > high:
        raise ValueError("Clip minimum must not exceed the maximum.")
    if arr.dtype == bool:
        return arr.copy()
    if integer_data:
        _reject_rounded_wide_integer_control(arr, minimum, "Clip minimum")
        _reject_rounded_wide_integer_control(arr, maximum, "Clip maximum")
        if low.denominator != 1 or high.denominator != 1:
            raise ValueError(
                "Integer Clip bounds must be whole numbers; convert the image "
                "to floating point before using fractional bounds."
            )
        info = np.iinfo(arr.dtype)
        low_value = int(low)
        high_value = int(high)
        if low_value > info.max or high_value < info.min:
            raise ValueError(
                f"Clip bounds lie outside the representable {arr.dtype} range "
                f"{info.min}..{info.max}."
            )
        low_value = max(low_value, int(info.min))
        high_value = min(high_value, int(info.max))
        return np.clip(
            arr,
            np.asarray(low_value, dtype=arr.dtype),
            np.asarray(high_value, dtype=arr.dtype),
        )
    clipped = np.clip(arr, low, high)
    return clipped.astype(arr.dtype, copy=False)


def add_images(inputs, input_count: int = 2) -> np.ndarray:
    """Add several same-shaped inputs."""
    arrays = _matching_input_arrays(inputs, input_count, "Add")
    result = np.zeros(arrays[0].shape, dtype=np.float32)
    for array in arrays:
        result += array.astype(np.float32, copy=False)
    return result


def subtract_images(inputs, input_count: int = 2) -> np.ndarray:
    """Subtract later same-shaped inputs from the first input."""
    arrays = _matching_input_arrays(inputs, input_count, "Subtract")
    result = arrays[0].astype(np.float32, copy=True)
    for array in arrays[1:]:
        result -= array.astype(np.float32, copy=False)
    return result


def ratio_image(inputs, input_count: int = 2, epsilon: float = 1e-6) -> np.ndarray:
    """Divide the first input by the second input with denominator protection."""
    arrays = _matching_input_arrays(inputs, input_count, "Ratio")
    numerator = arrays[0].astype(np.float32, copy=False)
    denominator = arrays[1].astype(np.float32, copy=False)
    return numerator / (denominator + float(epsilon))


def mask_image(
    inputs,
    outside_value: float = 0.0,
    invert_mask: str = "no",
    mask_axis_mapping: Sequence[int] | None = None,
) -> np.ndarray:
    """Apply a binary mask without guessing omitted axes from their sizes."""
    arrays = [np.asarray(item) for item in inputs if item is not None]
    if len(arrays) < 2:
        raise ValueError("Mask Image needs an image input and a mask input.")

    image = arrays[0]
    # This port accepts masks or labels, never color data. Treat every nonzero
    # label as selected without interpreting a trailing length-3/4 axis as RGB.
    mask = np.asarray(arrays[1]) != 0
    if str(invert_mask).lower() == "yes":
        mask = ~mask
    mask = _broadcast_mask_to_image(
        mask,
        image.shape,
        mask_axis_mapping=mask_axis_mapping,
    )
    output = np.asarray(image).copy()
    output[~mask] = np.asarray(outside_value, dtype=output.dtype)
    return output


def logical_and(inputs, input_count: int = 2) -> np.ndarray:
    """Logical AND over same-shaped inputs."""
    masks = [
        _to_bool_mask(array)
        for array in _matching_input_arrays(inputs, input_count, "Logical AND")
    ]
    return np.logical_and.reduce(masks)


def logical_or(inputs, input_count: int = 2) -> np.ndarray:
    """Logical OR over same-shaped inputs."""
    masks = [
        _to_bool_mask(array)
        for array in _matching_input_arrays(inputs, input_count, "Logical OR")
    ]
    return np.logical_or.reduce(masks)


def logical_xor(inputs, input_count: int = 2) -> np.ndarray:
    """Logical XOR over same-shaped inputs."""
    masks = [
        _to_bool_mask(array)
        for array in _matching_input_arrays(inputs, input_count, "Logical XOR")
    ]
    return np.logical_xor.reduce(masks)


def extract_channel_at_axis(
    data,
    channel: int = 0,
    channel_axis: int = 0,
) -> np.ndarray:
    """Extract a channel from an explicitly selected axis."""
    return _extract_channel(
        np.asarray(data),
        channel=channel,
        channel_axis=channel_axis,
    )


def _matching_input_arrays(
    inputs,
    input_count: int,
    operation_name: str,
) -> list[np.ndarray]:
    arrays = [np.asarray(item) for item in inputs if item is not None]
    requested = max(int(input_count), 1)
    if len(arrays) < requested:
        raise ValueError(
            f"{operation_name} needs {requested} connected input(s); "
            f"received {len(arrays)}."
        )
    arrays = arrays[:requested]
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError(f"{operation_name} inputs must have matching shapes.")
    return arrays


def _broadcast_mask_to_image(
    mask: np.ndarray,
    image_shape: tuple[int, ...],
    *,
    mask_axis_mapping: Sequence[int] | None = None,
) -> np.ndarray:
    """Broadcast a mask using an explicit source-to-target axis mapping."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == image_shape:
        return mask
    if mask.ndim > len(image_shape):
        raise ValueError(
            f"Mask Image mask rank {mask.ndim}D exceeds image rank "
            f"{len(image_shape)}D."
        )
    if mask_axis_mapping is None:
        raise ValueError(
            "Mask Image inputs with different shapes or ranks require an "
            "explicit semantic mask axis mapping; VIPP does not align axes by "
            "coincident sizes."
        )

    mapping = tuple(mask_axis_mapping)
    if len(mapping) != mask.ndim:
        raise ValueError(
            "Mask Image mask axis mapping must contain one image-axis index "
            f"for each of the mask's {mask.ndim} axes."
        )
    if any(
        isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral)
        for axis in mapping
    ):
        raise ValueError("Mask Image mask axis mapping must contain integers.")
    mapping = tuple(int(axis) for axis in mapping)
    if len(set(mapping)) != len(mapping):
        raise ValueError("Mask Image mask axis mapping must be one-to-one.")
    if any(axis < 0 or axis >= len(image_shape) for axis in mapping):
        raise ValueError(
            f"Mask Image mask axis mapping {mapping} is outside the "
            f"{len(image_shape)}D image."
        )

    source_order = tuple(sorted(range(mask.ndim), key=mapping.__getitem__))
    ordered_mask = np.transpose(mask, source_order)
    expanded_shape = [1] * len(image_shape)
    for source_axis in source_order:
        image_axis = mapping[source_axis]
        mask_size = int(mask.shape[source_axis])
        image_size = int(image_shape[image_axis])
        if mask_size != image_size:
            raise ValueError(
                f"Mask Image mapped axis sizes differ at image axis {image_axis} "
                f"({image_size} versus {mask_size})."
            )
        expanded_shape[image_axis] = mask_size
    return np.broadcast_to(ordered_mask.reshape(expanded_shape), image_shape)


def _extract_channel(
    arr: np.ndarray,
    channel: int = 0,
    channel_axis: int | None = None,
) -> np.ndarray:
    if channel_axis is None:
        raise ValueError("Extract Channel requires an explicit channel axis.")
    channel_axis = _validated_axis_index(
        channel_axis,
        arr.ndim,
        operation="Extract Channel",
    )
    if isinstance(channel, (bool, np.bool_)) or not isinstance(channel, Integral):
        raise ValueError("Extract Channel channel index must be an integer.")
    channel = int(channel)
    channel_count = int(arr.shape[channel_axis])
    if channel < 0:
        channel += channel_count
    if channel < 0 or channel >= channel_count:
        raise ValueError(
            f"Extract Channel channel index {channel!r} is out of range for "
            f"{channel_count} channels."
        )
    return np.take(arr, channel, axis=channel_axis)


def convert_dtype(
    data,
    output_dtype: str = "uint8",
    scaling: str = "rescale",
) -> np.ndarray:
    """Convert image dtype with an explicit scaling strategy."""
    arr = np.asarray(data)
    output_dtype = str(output_dtype).lower()
    scaling = str(scaling).lower()
    if output_dtype not in {"bool", "uint8", "uint16", "float32"}:
        raise ValueError(
            "Convert Dtype output_dtype must be bool, uint8, uint16, or float32."
        )
    if scaling not in {"rescale", "clip", "preserve"}:
        raise ValueError(
            "Convert Dtype scaling must be rescale, clip, or preserve."
        )

    if output_dtype == "bool":
        if arr.dtype == bool:
            return arr.copy()
        return arr != 0

    dtype = _target_dtype(output_dtype)
    if np.issubdtype(dtype, np.floating):
        return _convert_to_float(arr, dtype, scaling)
    if np.issubdtype(dtype, np.integer):
        return _convert_to_integer(arr, dtype, scaling)
    return arr.astype(dtype, copy=True)


def invert(data) -> np.ndarray:
    """Invert boolean masks or intensity images."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return np.logical_not(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr.copy()
    return finite.max() + finite.min() - arr


def save_array_output(
    data,
    path: str | Path,
    *,
    format: str = "auto",
    overwrite: bool = True,
    image_state=None,
) -> Path:
    """Write an array output through the shared headless I/O registry."""
    return write_image(
        data,
        path,
        format=format,
        overwrite=overwrite,
        image_state=image_state,
    )


def save_output(
    data,
    enabled: str = "off",
    path: str = "",
    format: str = "auto",
    overwrite: str = "no",
    image_state=None,
) -> np.ndarray:
    """Pipeline node that writes the current output and passes data downstream."""
    arr = np.asarray(data).copy()
    if str(enabled).lower() == "on" and str(path).strip():
        save_array_output(
            arr,
            path,
            format=format,
            overwrite=str(overwrite).lower() == "yes",
            image_state=image_state,
        )
    return arr


def batch_output(
    data,
    tag: str = "output",
    format: str = "batch default",
    subfolder: str = "",
    filename_template: str = "{source_stem}__{tag}",
    overwrite: str = "batch default",
):
    """Pass data through while marking it as an explicit batch output."""
    return data


def _imagej_tiff_payload(arr: np.ndarray, image_state) -> tuple[np.ndarray, str]:
    writable = _tiff_writable_array(arr)
    axes = _imagej_axes_for(writable, image_state)
    desired = "TZCYXS"
    if len(axes) > 1:
        order = [axes.index(axis) for axis in desired if axis in axes]
        if order != list(range(len(axes))):
            writable = np.transpose(writable, order)
            axes = "".join(axes[index] for index in order)
    return np.ascontiguousarray(writable), axes


def _is_label_image_state(image_state) -> bool:
    kind = getattr(image_state, "kind", None)
    if kind is None and isinstance(image_state, dict):
        kind = image_state.get("kind")
    return str(kind).lower() == "label image"


def _tiff_writable_array(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * np.uint8(255)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, arr.shape[0])
    return arr


def _imagej_axes_for(arr: np.ndarray, image_state) -> str:
    labels = _imagej_axis_labels_from_state(image_state, arr.ndim)
    if labels is None:
        labels = _fallback_imagej_axes(arr)
    if len(labels) != arr.ndim or any(label not in "TZCYXS" for label in labels):
        labels = _fallback_imagej_axes(arr)
    return labels


def _imagej_axis_labels_from_state(image_state, ndim: int) -> str | None:
    axes = getattr(image_state, "axes", None)
    if axes is None and isinstance(image_state, dict):
        axes = image_state.get("axes")
    if axes is None or len(axes) != ndim:
        return None

    labels = []
    for axis in axes:
        if isinstance(axis, dict):
            name = str(axis.get("name", "")).lower()
            axis_type = str(axis.get("type", "")).lower()
        else:
            name = str(getattr(axis, "name", "")).lower()
            axis_type = str(getattr(axis, "type", "")).lower()
        if name == "t" or axis_type == "time":
            labels.append("T")
        elif name == "z":
            labels.append("Z")
        elif name in {"c", "channel"}:
            labels.append("C")
        elif name == "y":
            labels.append("Y")
        elif name == "x":
            labels.append("X")
        elif name in {"rgb", "rgba"}:
            labels.append("S")
        elif axis_type == "channel":
            labels.append("C")
        else:
            return None
    return "".join(labels)


def _fallback_imagej_axes(arr: np.ndarray) -> str:
    if arr.ndim == 0:
        return "YX"
    if arr.ndim == 1:
        return "YX"
    if arr.ndim == 2:
        return "YX"
    if arr.ndim == 3 and arr.shape[-1] in RGB_CHANNELS:
        return "YXS"
    fallback = {
        3: "ZYX",
        4: "ZCYX",
        5: "TZCYX",
        6: "TZCYXS",
    }
    return fallback.get(arr.ndim, "YX")


def max_intensity_projection(data, axis: int = 0) -> np.ndarray:
    """Project an image along one axis using maximum intensity."""
    arr = np.asarray(data)
    if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, Integral):
        raise ValueError("Maximum Projection axis must be an integer.")
    axis = int(axis)
    if axis < 0 or axis >= arr.ndim:
        raise ValueError(
            f"Maximum Projection axis {axis} is out of range for {arr.ndim}D input."
        )
    if arr.ndim <= 2:
        return arr.copy()
    return np.max(arr, axis=axis)


def project_image(
    data,
    axes: str = "auto",
    method: str = "Maximum",
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
) -> np.ndarray:
    """Project an image over one or more selected axes."""
    arr = np.asarray(data)
    axis_indices = _projection_axis_indices(
        arr.ndim,
        axes,
        axis_names=axis_names,
        axis_types=axis_types,
        shape=arr.shape,
    )
    if not axis_indices:
        return arr.copy()
    return _project_array(arr, axis_indices, method)


def orthogonal_projection(
    data,
    method: str = "Maximum",
    use_physical_scale: bool = True,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    axis_scales: Sequence[float] = (),
    axis_units: Sequence[str | None] = (),
) -> np.ndarray:
    """Build an XY/XZ/YZ orthogonal projection montage from a 3D volume."""
    arr = np.asarray(data)
    spatial_axes = _orthogonal_spatial_axis_indices(
        arr.ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        shape=arr.shape,
    )
    if len(spatial_axes) < 3:
        return arr.copy()

    z_axis, y_axis, x_axis = spatial_axes[-3:]
    non_spatial_axes = [axis for axis in range(arr.ndim) if axis not in spatial_axes]
    processing_order = non_spatial_axes + [z_axis, y_axis, x_axis]
    moved = np.transpose(arr, processing_order)
    leading_shape = moved.shape[:-3]
    z_size, y_size, x_size = moved.shape[-3:]
    display_scales = _orthogonal_display_scales(
        spatial_axes[-3:],
        use_physical_scale=use_physical_scale,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    display_z_y, display_z_x, display_y, display_x = _orthogonal_display_sizes(
        (z_size, y_size, x_size),
        display_scales,
    )
    flat = moved.reshape((-1, z_size, y_size, x_size))

    montages = [
        _orthogonal_projection_block(
            block,
            method,
            (display_z_y, display_z_x, display_y, display_x),
        )
        for block in flat
    ]
    montage = np.stack(montages, axis=0).reshape(
        leading_shape + (display_y + display_z_y, display_x + display_z_x)
    )

    temp_labels = non_spatial_axes + [-2, -1]
    desired_labels: list[int] = []
    inserted_montage_axes = False
    spatial_set = set(spatial_axes)
    first_spatial_axis = min(spatial_axes)
    for axis in range(arr.ndim):
        if axis in spatial_set:
            if not inserted_montage_axes and axis == first_spatial_axis:
                desired_labels.extend([-2, -1])
                inserted_montage_axes = True
            continue
        desired_labels.append(axis)
    if not inserted_montage_axes:
        desired_labels.extend([-2, -1])

    transpose_order = [temp_labels.index(label) for label in desired_labels]
    return np.ascontiguousarray(np.transpose(montage, transpose_order))


def set_pixel_size(
    data,
    x_size: float = 1.0,
    y_size: float = 1.0,
    z_size: float = 1.0,
    unit: str = "micrometer",
) -> np.ndarray:
    """Pass image data through while updating carried pixel-size metadata."""
    del x_size, y_size, z_size, unit
    return np.asarray(data)


def rescale_axes(
    data,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    z_scale: float = 1.0,
    lock_xy: bool = True,
    interpolation: str = "Auto",
    anti_aliasing: bool = True,
    resize_mode: str = "Scale factor",
    x_size: int = 0,
    y_size: int = 0,
    z_size: int = 0,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    input_kind: str = "",
    progress=None,
) -> np.ndarray:
    """Resample spatial X/Y/Z axes by scale factors or explicit output sizes."""
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.copy()
    x_scale = _positive_float(x_scale, 1.0)
    y_scale = x_scale if bool(lock_xy) else _positive_float(y_scale, 1.0)
    z_scale = _positive_float(z_scale, 1.0)

    axis_map = _xyz_axis_indices(
        arr.ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        shape=arr.shape,
    )
    output_shape = list(arr.shape)
    if str(resize_mode).strip().lower().startswith("output"):
        requested_sizes = {"x": x_size, "y": y_size, "z": z_size}
        for role, axis in axis_map.items():
            output_shape[axis] = _positive_int(
                requested_sizes.get(role),
                arr.shape[axis],
            )
    else:
        scale_by_axis = {axis_map["x"]: x_scale}
        if "y" in axis_map:
            scale_by_axis[axis_map["y"]] = y_scale
        if "z" in axis_map:
            scale_by_axis[axis_map["z"]] = z_scale
        output_shape = [
            max(int(round(size * scale_by_axis.get(axis, 1.0))), 1)
            for axis, size in enumerate(arr.shape)
        ]
    output_shape = tuple(output_shape)
    if output_shape == arr.shape:
        return arr.copy()
    if progress is not None:
        progress.report(0, 1, "Resampling axes")

    semantic = _rescale_semantic_kind(arr, input_kind)
    order = _resize_order(interpolation, semantic)
    anti_alias = (
        bool(anti_aliasing)
        and order > 0
        and semantic == "image"
        and any(output_shape[axis] < arr.shape[axis] for axis in range(arr.ndim))
    )
    resized = transform.resize(
        arr,
        output_shape,
        order=order,
        mode="edge",
        preserve_range=True,
        anti_aliasing=anti_alias,
    )
    if progress is not None:
        progress.report(1, 1, "Resampling complete")
    return _restore_rescaled_axes_dtype(resized, arr, semantic)


def select_axis_slice(
    data,
    axis: int = 0,
    index: int = 0,
    axes: str = "",
    indices: str = "",
    ranges: str = "",
    range_mode: bool = False,
    remove_axes: str = "",
    remove_indices: str = "",
) -> np.ndarray:
    """Select one or more axis slices or retained axis ranges."""
    arr = np.asarray(data)
    if arr.ndim == 0:
        return arr.copy()
    if range_mode:
        result = _axis_range_selection(arr, ranges)
        selections = _slice_selections(
            result,
            0,
            0,
            remove_axes,
            remove_indices,
            use_default=False,
        )
    else:
        result = arr
        selections = _slice_selections(arr, axis, index, axes, indices)
    for axis_index, slice_index in sorted(selections.items(), reverse=True):
        result = np.take(result, slice_index, axis=axis_index)
    return result


def _adaptive_threshold(
    data,
    block_size: int,
    c: float,
    method: str,
    *,
    channel_axis: int | None = None,
) -> np.ndarray:
    operation = (
        "Adaptive mean threshold"
        if method == "mean"
        else "Adaptive Gaussian threshold"
    )
    arr = _to_explicit_grayscale(
        np.asarray(data),
        channel_axis=channel_axis,
        operation=operation,
    )

    def threshold_plane(plane: np.ndarray) -> np.ndarray:
        block = _odd_size(block_size, minimum=3, maximum=min(plane.shape[-2:]))
        if block < 3:
            return plane > float(np.mean(plane))
        local = filters.threshold_local(
            plane.astype(np.float32, copy=False),
            block,
            method=method,
            offset=float(c),
        )
        return plane > local

    return _apply_scalar_plane_wise(arr, threshold_plane)


def _global_threshold(
    arr: np.ndarray,
    threshold_func: Callable[[np.ndarray], int | float],
    *,
    threshold_scope: str = "Stack histogram",
) -> np.ndarray:
    scope = str(threshold_scope).strip().casefold()
    if scope not in {"stack histogram", "slice histogram"}:
        raise ValueError(
            "Threshold scope must be 'Stack histogram' or 'Slice histogram'."
        )
    # A boolean image is already a binary segmentation. Running histogram
    # algorithms on only two levels is redundant and some methods (notably
    # Triangle) choose the True level itself, which would erase foreground.
    if arr.dtype == bool:
        return arr.copy()
    if scope == "stack histogram":
        return _threshold_mask(arr, threshold_func(arr))

    def threshold_plane(plane: np.ndarray) -> np.ndarray:
        return _threshold_mask(plane, threshold_func(plane))

    return _apply_scalar_plane_wise(arr, threshold_plane)


def _threshold_mask(arr: np.ndarray, threshold: int | float) -> np.ndarray:
    mask = arr > threshold
    if np.issubdtype(arr.dtype, np.inexact):
        mask &= np.isfinite(arr)
    return mask


def _to_explicit_grayscale(
    arr: np.ndarray,
    *,
    channel_axis: int | None,
    operation: str,
) -> np.ndarray:
    """Reduce only an explicitly declared RGB/RGBA axis to scalar luma.

    The supported channel axis must contain exactly three RGB or four RGBA
    channels in that order. Boolean, integer, and floating arrays are accepted;
    complex and nonnumeric arrays are rejected. Luma uses the established
    BT.601 weights ``0.299 R + 0.587 G + 0.114 B`` and ignores alpha. Removing
    the channel axis preserves the relative order of every remaining axis.
    """
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{operation} requires non-empty image data.")
    axis = _validated_luma_channel_axis(
        arr,
        channel_axis,
        operation=operation,
    )
    if axis is None:
        return arr

    moved = np.moveaxis(arr, axis, -1)
    work_dtype = np.result_type(arr.dtype, np.float32)
    rgb = moved[..., :3].astype(work_dtype, copy=False)
    coefficients = np.asarray((0.299, 0.587, 0.114), dtype=work_dtype)
    return np.sum(rgb * coefficients, axis=-1, dtype=work_dtype)


def _to_bool_mask(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.dtype == bool:
        return arr
    return arr != 0


def _target_dtype(name: str) -> np.dtype:
    choices = {
        "uint8": np.dtype(np.uint8),
        "uint16": np.dtype(np.uint16),
        "float32": np.dtype(np.float32),
    }
    return choices[name]


def _convert_to_float(
    arr: np.ndarray,
    dtype: np.dtype,
    scaling: str,
) -> np.ndarray:
    if scaling == "preserve":
        return arr.astype(dtype, copy=True)
    values = arr.astype(np.float32, copy=False)
    if scaling == "clip":
        return np.nan_to_num(
            np.clip(values, 0.0, 1.0),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).astype(dtype)
    return _rescale_values(values, 0.0, 1.0).astype(dtype)


def _convert_to_integer(
    arr: np.ndarray,
    dtype: np.dtype,
    scaling: str,
) -> np.ndarray:
    info = np.iinfo(dtype)
    values = arr.astype(np.float64, copy=False)
    if scaling == "preserve":
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "Convert Dtype preserve cannot represent non-finite values "
                "in an integer output."
            )
        if values.size and (values.min() < info.min or values.max() > info.max):
            raise ValueError(
                "Convert Dtype preserve input values exceed the output dtype range; "
                "choose clip or rescale explicitly."
            )
        return values.astype(dtype)
    elif scaling == "clip":
        scaled = np.clip(values, info.min, info.max)
    else:
        scaled = _rescale_values(values, float(info.min), float(info.max))
    scaled = np.nan_to_num(
        scaled,
        nan=0.0,
        posinf=float(info.max),
        neginf=float(info.min),
    )
    return np.clip(scaled, info.min, info.max).astype(dtype)


def _rescale_values(
    values: np.ndarray,
    target_min: float,
    target_max: float,
) -> np.ndarray:
    values = np.asarray(values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=np.float64)

    source_min = float(finite.min())
    source_max = float(finite.max())
    if source_max == source_min:
        fill = target_max if source_max > 0 else target_min
        return np.full_like(values, fill, dtype=np.float64)

    scaled = (values.astype(np.float64) - source_min) / (source_max - source_min)
    return scaled * (target_max - target_min) + target_min


@dataclass(frozen=True)
class _FiniteArrayStats:
    count: int
    minimum: int | float
    maximum: int | float


@dataclass(frozen=True)
class _FiniteHistogram:
    stats: _FiniteArrayStats
    counts: np.ndarray | None = None
    centers: np.ndarray | None = None


def _finite_array_stats(arr: np.ndarray) -> _FiniteArrayStats:
    count = 0
    minimum = np.inf
    maximum = -np.inf
    for values in _finite_float64_chunks(arr):
        if values.size == 0:
            continue
        count += values.size
        minimum = min(minimum, float(values.min()))
        maximum = max(maximum, float(values.max()))
    if count == 0:
        return _FiniteArrayStats(0, 0.0, 0.0)
    return _FiniteArrayStats(count, minimum, maximum)


def _integer_array_stats(arr: np.ndarray) -> _FiniteArrayStats:
    """Return exact integer extrema without converting native levels to float."""
    count = int(arr.size)
    if count == 0:
        return _FiniteArrayStats(0, 0, 0)
    minimum: int | None = None
    maximum: int | None = None
    for values in _array_chunks(arr):
        chunk_minimum = int(values.min())
        chunk_maximum = int(values.max())
        minimum = chunk_minimum if minimum is None else min(minimum, chunk_minimum)
        maximum = chunk_maximum if maximum is None else max(maximum, chunk_maximum)
    assert minimum is not None and maximum is not None
    return _FiniteArrayStats(count, minimum, maximum)


def _finite_histogram(
    arr: np.ndarray,
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
) -> _FiniteHistogram:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return _boolean_histogram(arr)
    if np.issubdtype(arr.dtype, np.integer):
        return _native_integer_histogram(arr)
    if not np.issubdtype(arr.dtype, np.floating):
        raise ValueError(
            "Automatic histogram thresholds require boolean, integer, or "
            "floating-point image data."
        )

    bin_count = _validated_histogram_bins(histogram_bins)
    stats = _finite_array_stats(arr)
    if stats.count == 0 or stats.minimum == stats.maximum:
        return _FiniteHistogram(stats)

    # Match ``skimage.exposure.histogram`` for floating-point images. NumPy
    # deliberately builds float32 edges for float32 input (and likewise for
    # other floating dtypes); promoting the edges to float64 changes their
    # centres and can move the reported threshold slightly.
    edge_limits = np.asarray(
        [stats.minimum, stats.maximum],
        dtype=arr.dtype,
    )
    edges = np.histogram_bin_edges(edge_limits, bins=bin_count)
    counts = np.zeros(bin_count, dtype=np.intp)
    for values in _finite_float64_chunks(arr):
        if values.size:
            counts += np.histogram(values, bins=edges)[0]
    centers = (edges[:-1] + edges[1:]) / 2.0
    return _FiniteHistogram(stats, counts, centers)


def _boolean_histogram(arr: np.ndarray) -> _FiniteHistogram:
    count = int(arr.size)
    true_count = sum(int(np.count_nonzero(chunk)) for chunk in _array_chunks(arr))
    false_count = count - true_count
    if count == 0:
        stats = _FiniteArrayStats(0, 0, 0)
    else:
        stats = _FiniteArrayStats(
            count,
            0 if false_count else 1,
            1 if true_count else 0,
        )
    return _FiniteHistogram(
        stats,
        np.array([false_count, true_count], dtype=np.intp),
        np.array([0, 1], dtype=np.uint8),
    )


def _native_integer_histogram(arr: np.ndarray) -> _FiniteHistogram:
    stats = _integer_array_stats(arr)
    if stats.count == 0:
        return _FiniteHistogram(stats)
    minimum = int(stats.minimum)
    maximum = int(stats.maximum)

    span = maximum - minimum + 1
    if span > _MAX_NATIVE_INTEGER_HISTOGRAM_BINS:
        raise ValueError(
            f"Integer intensity span contains {span:,} levels; automatic "
            f"thresholding supports at most "
            f"{_MAX_NATIVE_INTEGER_HISTOGRAM_BINS:,} exact integer levels. "
            "Convert or rescale the image to uint16 or floating point instead "
            "of silently collapsing integer levels."
        )

    counts = np.zeros(span, dtype=np.intp)
    for values in _array_chunks(arr):
        if minimum >= np.iinfo(np.int64).min and maximum <= np.iinfo(np.int64).max:
            indices = values.astype(np.int64, copy=True)
            indices -= minimum
            counts += np.bincount(indices, minlength=span)
        else:
            levels, level_counts = np.unique(values, return_counts=True)
            indices = np.fromiter(
                (int(level) - minimum for level in levels),
                dtype=np.intp,
                count=levels.size,
            )
            counts[indices] += level_counts.astype(np.intp, copy=False)
    center_dtype = (
        np.uint64 if maximum > np.iinfo(np.int64).max else np.int64
    )
    centers = np.fromiter(
        (minimum + index for index in range(span)),
        dtype=center_dtype,
        count=span,
    )
    return _FiniteHistogram(stats, counts, centers)


def _validated_histogram_bins(histogram_bins: int) -> int:
    if isinstance(histogram_bins, (bool, np.bool_)):
        raise ValueError("Float histogram bins must be an integer from 2 to 65,536.")
    try:
        bin_count = int(histogram_bins)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Float histogram bins must be an integer from 2 to 65,536."
        ) from exc
    if isinstance(histogram_bins, (float, np.floating)) and not float(
        histogram_bins
    ).is_integer():
        raise ValueError("Float histogram bins must be an integer from 2 to 65,536.")
    if not 2 <= bin_count <= _MAX_NATIVE_INTEGER_HISTOGRAM_BINS:
        raise ValueError("Float histogram bins must be an integer from 2 to 65,536.")
    return bin_count


def _threshold_histogram_centers(
    arr: np.ndarray,
    summary: _FiniteHistogram,
) -> tuple[np.ndarray, int | None]:
    """Return numerically safe centers and an exact integer offset if needed."""
    assert summary.counts is not None and summary.centers is not None
    if np.issubdtype(np.asarray(arr).dtype, np.integer):
        return np.arange(summary.counts.size, dtype=np.int64), int(
            summary.stats.minimum
        )
    return summary.centers, None


def _restore_histogram_threshold(
    threshold: int | float | np.number,
    integer_offset: int | None,
) -> int | float:
    """Map a translated integer level back without a float round trip."""
    if integer_offset is not None:
        return integer_offset + int(threshold)
    return float(threshold)


def _otsu_value(
    arr: np.ndarray,
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
) -> int | float:
    summary = _finite_histogram(arr, histogram_bins)
    _require_finite_threshold_stats(summary.stats)
    if summary.stats.minimum == summary.stats.maximum:
        return summary.stats.minimum
    assert summary.counts is not None and summary.centers is not None

    hist = summary.counts
    centers, integer_offset = _threshold_histogram_centers(arr, summary)
    calculation_centers = (
        centers.astype(np.float64, copy=False)
        if integer_offset is not None
        else centers
    )
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * calculation_centers) / np.maximum(weight1, 1)
    mean2 = (
        np.cumsum((hist * calculation_centers)[::-1])
        / np.maximum(weight2[::-1], 1)
    )[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if variance12.size == 0:
        return (summary.stats.minimum + summary.stats.maximum) / 2.0
    return _restore_histogram_threshold(
        centers[:-1][np.argmax(variance12)],
        integer_offset,
    )


def _array_chunks(arr: np.ndarray) -> Iterator[np.ndarray]:
    """Yield bounded chunks without changing their numeric dtype."""
    iterator = np.nditer(
        np.asarray(arr),
        flags=["buffered", "external_loop", "refs_ok", "zerosize_ok"],
        op_flags=[["readonly"]],
        order="K",
        buffersize=_THRESHOLD_CHUNK_SIZE,
    )
    for chunk in iterator:
        yield np.asarray(chunk)


def _finite_float64_chunks(arr: np.ndarray) -> Iterator[np.ndarray]:
    """Yield bounded finite chunks for automatic-threshold calculations."""
    for chunk in _array_chunks(arr):
        values = chunk
        finite = np.isfinite(values)
        if not finite.all():
            values = values[finite]
        yield values.astype(np.float64, copy=False)


def _triangle_value(
    arr: np.ndarray,
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
) -> int | float:
    summary = _finite_histogram(arr, histogram_bins)
    _require_finite_threshold_stats(summary.stats)
    if summary.stats.minimum == summary.stats.maximum:
        return summary.stats.minimum
    assert summary.counts is not None and summary.centers is not None

    hist = summary.counts
    centers, integer_offset = _threshold_histogram_centers(arr, summary)
    bin_count = hist.size
    peak_index = int(np.argmax(hist))
    peak_height = float(hist[peak_index])
    low_index, high_index = np.flatnonzero(hist)[[0, -1]]
    flip = peak_index - low_index < high_index - peak_index
    if flip:
        hist = hist[::-1]
        low_index = bin_count - high_index - 1
        peak_index = bin_count - peak_index - 1

    width = peak_index - low_index
    if width <= 0:
        return _restore_histogram_threshold(
            centers[peak_index],
            integer_offset,
        )
    x_values = np.arange(width)
    y_values = hist[x_values + low_index]
    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    normalized_width = width / norm
    level = int(np.argmax(peak_height * x_values - normalized_width * y_values))
    level += low_index
    if flip:
        level = bin_count - level - 1
    return _restore_histogram_threshold(centers[level], integer_offset)


def _li_value(
    arr: np.ndarray,
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
) -> int | float:
    """Run Li's raw iteration; unlike the other methods it is not histogram based."""
    del histogram_bins
    return _safe_threshold(arr, filters.threshold_li)


def _histogram_threshold_value(
    arr: np.ndarray,
    threshold_func: Callable[..., int | float],
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
) -> int | float:
    summary = _finite_histogram(arr, histogram_bins)
    _require_finite_threshold_stats(summary.stats)
    if summary.stats.minimum == summary.stats.maximum:
        return summary.stats.minimum
    assert summary.counts is not None and summary.centers is not None
    centers, integer_offset = _threshold_histogram_centers(arr, summary)
    threshold = threshold_func(hist=(summary.counts, centers))
    return _restore_histogram_threshold(threshold, integer_offset)


def _yen_value(
    arr: np.ndarray,
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
) -> int | float:
    return _histogram_threshold_value(arr, filters.threshold_yen, histogram_bins)


def _isodata_value(
    arr: np.ndarray,
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
) -> int | float:
    return _histogram_threshold_value(arr, filters.threshold_isodata, histogram_bins)


def _minimum_value(
    arr: np.ndarray,
    histogram_bins: int = _GLOBAL_THRESHOLD_HISTOGRAM_BINS,
    *,
    max_iterations: int = 10_000,
    progress=None,
) -> int | float:
    error = "Minimum threshold iterations must be an integer from 1 to 10,000."
    if isinstance(max_iterations, (bool, np.bool_)):
        raise ValueError(error)
    try:
        iteration_limit = int(max_iterations)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not 1 <= iteration_limit <= 10_000 or iteration_limit != max_iterations:
        raise ValueError(error)

    summary = _finite_histogram(arr, histogram_bins)
    _require_finite_threshold_stats(summary.stats)
    if summary.stats.minimum == summary.stats.maximum:
        return summary.stats.minimum
    assert summary.counts is not None and summary.centers is not None

    smooth_histogram = summary.counts.astype(np.float32, copy=False)
    maxima: list[int] = []
    if progress is not None:
        progress.report(0, iteration_limit, "Smoothing threshold histogram")
    for iteration in range(iteration_limit):
        if progress is not None:
            progress.check_cancelled()
        smooth_histogram = ndi.uniform_filter1d(smooth_histogram, 3)
        maxima = _minimum_histogram_maxima(smooth_histogram)
        if len(maxima) < 3:
            break
        if progress is not None and (iteration + 1) % 25 == 0:
            progress.report(
                iteration + 1,
                iteration_limit,
                "Smoothing threshold histogram",
            )

    if len(maxima) != 2:
        raise RuntimeError("Unable to find two maxima in histogram")
    if iteration == iteration_limit - 1:
        raise RuntimeError("Maximum iteration reached for histogram smoothing")
    if progress is not None:
        progress.report(iteration + 1, iteration_limit, "Threshold histogram ready")
    valley_offset = int(
        np.argmin(smooth_histogram[maxima[0] : maxima[1] + 1])
    )
    centers, integer_offset = _threshold_histogram_centers(arr, summary)
    return _restore_histogram_threshold(
        centers[maxima[0] + valley_offset],
        integer_offset,
    )


def _minimum_histogram_maxima(histogram: np.ndarray) -> list[int]:
    """Return plateau-safe local maxima using Minimum's published rule."""
    maxima: list[int] = []
    direction = 1
    for index in range(histogram.size - 1):
        if direction > 0:
            if histogram[index + 1] < histogram[index]:
                direction = -1
                maxima.append(index)
        elif histogram[index + 1] > histogram[index]:
            direction = 1
    return maxima


def _validated_threshold_pair(
    low,
    high,
    *,
    operation: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> tuple[float, float]:
    try:
        low_value = float(low)
        high_value = float(high)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            f"{operation} low and high thresholds must be finite numbers."
        ) from exc
    if not np.isfinite(low_value) or not np.isfinite(high_value):
        raise ValueError(
            f"{operation} low and high thresholds must be finite numbers."
        )
    if minimum is not None and (low_value < minimum or high_value < minimum):
        raise ValueError(
            f"{operation} low and high thresholds must be at least {minimum:g}."
        )
    if maximum is not None and (low_value > maximum or high_value > maximum):
        raise ValueError(
            f"{operation} low and high thresholds must be at most {maximum:g}."
        )
    if high_value < low_value:
        raise ValueError(
            f"{operation} low threshold ({low_value:g}) must not exceed "
            f"the high threshold ({high_value:g})."
        )
    return low_value, high_value


def _validated_sigma_pair(low_sigma, high_sigma) -> tuple[float, float]:
    try:
        low_value = float(low_sigma)
        high_value = float(high_sigma)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "Difference of Gaussians sigmas must be finite numbers."
        ) from exc
    if not np.isfinite(low_value) or not np.isfinite(high_value):
        raise ValueError("Difference of Gaussians sigmas must be finite numbers.")
    if low_value < 0.0 or high_value < 0.0:
        raise ValueError("Difference of Gaussians sigmas must be non-negative.")
    if high_value <= low_value:
        raise ValueError(
            "Difference of Gaussians high sigma must be greater than low sigma."
        )
    return low_value, high_value


def _safe_threshold(
    arr: np.ndarray,
    threshold_func: Callable[[np.ndarray], float],
) -> int | float:
    arr = np.asarray(arr)
    if np.issubdtype(arr.dtype, np.integer):
        stats = _integer_array_stats(arr)
        _require_finite_threshold_stats(stats)
        if stats.minimum == stats.maximum:
            return stats.minimum
        minimum = int(stats.minimum)
        maximum = int(stats.maximum)
        relative_span = maximum - minimum
        if relative_span > 2**53:
            raise ValueError(
                f"Integer intensity span {relative_span:,} exceeds the exact "
                "float64 range required by Li thresholding. Convert or rescale "
                "the image before applying Li Threshold."
            )

        # Subtract in unsigned integer space before converting to float. The
        # modulo subtraction is the exact native difference for every signed
        # and unsigned integer dtype as long as the span is below 2**64.
        relative = arr.astype(np.uint64, copy=True)
        minimum_native = np.asarray(minimum, dtype=arr.dtype).astype(np.uint64)
        relative -= minimum_native
        relative_threshold = float(
            threshold_func(relative.astype(np.float64, copy=False))
        )
        exact_cutoff = minimum + math.floor(relative_threshold)
        restored_threshold = float(relative_threshold + minimum)
        if (
            math.floor(restored_threshold) == exact_cutoff
            and float(exact_cutoff) == exact_cutoff
            and float(exact_cutoff + 1) > restored_threshold
        ):
            return restored_threshold
        # Adding a large offset can round a fractional threshold across an
        # integer boundary, or collapse the next native level onto the same
        # float. For integer comparison, x > t is exactly x > floor(t).
        return exact_cutoff

    stats = _finite_array_stats(arr)
    _require_finite_threshold_stats(stats)
    if stats.minimum == stats.maximum:
        return stats.minimum
    values = np.asarray(arr)
    values = values[np.isfinite(values)].astype(np.float64, copy=False)
    return float(threshold_func(values))


def _require_finite_threshold_stats(stats: _FiniteArrayStats) -> None:
    if stats.count == 0:
        raise ValueError(
            "Automatic thresholding requires at least one finite input value."
        )


def _automatic_threshold_function(
    operation_id: str,
) -> Callable[[np.ndarray, int], int | float] | None:
    return {
        "otsu_threshold": _otsu_value,
        "triangle_threshold": _triangle_value,
        "li_threshold": _li_value,
        "yen_threshold": _yen_value,
        "isodata_threshold": _isodata_value,
        "minimum_threshold": _minimum_value,
    }.get(str(operation_id))


def _apply_plane_wise(arr: np.ndarray, func: Callable[[np.ndarray], np.ndarray]):
    arr = np.asarray(arr)
    if arr.ndim <= 2 or _is_color_image(arr):
        return func(arr)

    if _has_channel_axis(arr):
        plane_ndim = 3
    else:
        plane_ndim = 2
    leading_shape = arr.shape[: arr.ndim - plane_ndim]
    if not leading_shape:
        return func(arr)

    sample = np.asarray(func(arr[(0,) * len(leading_shape)]))
    out = np.empty(leading_shape + sample.shape, dtype=sample.dtype)
    out[(0,) * len(leading_shape)] = sample
    for index in np.ndindex(leading_shape):
        if all(i == 0 for i in index):
            continue
        out[index] = func(arr[index])
    return out


def _apply_scalar_plane_wise(
    arr: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply ``func`` over trailing YX planes without channel inference."""
    arr = np.asarray(arr)
    if arr.ndim <= 2:
        return np.asarray(func(arr))

    leading_shape = arr.shape[:-2]
    sample = np.asarray(func(arr[(0,) * len(leading_shape)]))
    out = np.empty(leading_shape + sample.shape, dtype=sample.dtype)
    out[(0,) * len(leading_shape)] = sample
    for index in np.ndindex(leading_shape):
        if all(i == 0 for i in index):
            continue
        out[index] = func(arr[index])
    return out


def _apply_filter_plane_wise(
    arr: np.ndarray,
    func: Callable[[np.ndarray], np.ndarray],
    *,
    channel_axis: int | None,
) -> np.ndarray:
    """Apply a filter to scalar YX or explicitly declared YXC planes."""
    arr = np.asarray(arr)
    if channel_axis is None:
        working = arr
        plane_ndim = 2
    else:
        working = np.moveaxis(arr, channel_axis, -1)
        plane_ndim = 3

    leading_shape = working.shape[: max(working.ndim - plane_ndim, 0)]
    if not leading_shape:
        filtered = np.asarray(func(working))
    else:
        sample = np.asarray(func(working[(0,) * len(leading_shape)]))
        filtered = np.empty(leading_shape + sample.shape, dtype=sample.dtype)
        filtered[(0,) * len(leading_shape)] = sample
        for index in np.ndindex(leading_shape):
            if all(i == 0 for i in index):
                continue
            filtered[index] = func(working[index])

    if channel_axis is not None:
        if filtered.ndim != arr.ndim:
            raise ValueError(
                "A multichannel plane filter must preserve the declared channel axis."
            )
        filtered = np.moveaxis(filtered, -1, channel_axis)
    return np.ascontiguousarray(filtered)


def _apply_spatial_blocks(
    arr: np.ndarray,
    spatial_ndim: int,
    func: Callable[[np.ndarray], np.ndarray],
    *,
    dtype,
    progress=None,
    progress_start: int = 0,
    progress_total: int | None = None,
    progress_message: str = "",
) -> np.ndarray:
    """Apply ``func`` independently over leading non-spatial dimensions."""
    spatial_ndim = int(np.clip(spatial_ndim, 1, max(arr.ndim, 1)))
    block_count = _spatial_block_count(arr, spatial_ndim)
    denominator = int(progress_total or block_count)
    completed = 0
    if progress is not None:
        progress.report(progress_start, denominator, progress_message)
    if arr.ndim <= spatial_ndim:
        result = np.asarray(func(arr), dtype=dtype)
        if progress is not None:
            progress.report(progress_start + 1, denominator, progress_message)
        return result

    result = np.empty(arr.shape, dtype=dtype)
    leading_shape = arr.shape[: arr.ndim - spatial_ndim]
    for index in np.ndindex(leading_shape):
        if progress is not None:
            progress.check_cancelled()
        result[index] = func(arr[index])
        completed += 1
        if progress is not None:
            progress.report(
                progress_start + completed,
                denominator,
                progress_message,
            )
    return result


def _spatial_block_count(arr: np.ndarray, spatial_ndim: int) -> int:
    arr = np.asarray(arr)
    spatial_ndim = int(np.clip(spatial_ndim, 1, max(arr.ndim, 1)))
    if arr.ndim <= spatial_ndim:
        return 1
    return int(np.prod(arr.shape[: arr.ndim - spatial_ndim], dtype=np.int64))


def _apply_spatial_blocks_tuple(
    arr: np.ndarray,
    spatial_ndim: int,
    func: Callable[[np.ndarray], tuple[np.ndarray, ...]],
    *,
    dtypes: tuple[object, ...],
) -> tuple[np.ndarray, ...]:
    """Apply ``func`` over leading blocks when it returns multiple arrays."""
    arr = np.asarray(arr)
    spatial_ndim = int(np.clip(spatial_ndim, 1, max(arr.ndim, 1)))
    if arr.ndim <= spatial_ndim:
        return tuple(
            np.asarray(value, dtype=dtype)
            for value, dtype in zip(func(arr), dtypes, strict=True)
        )

    result = tuple(np.empty(arr.shape, dtype=dtype) for dtype in dtypes)
    leading_shape = arr.shape[: arr.ndim - spatial_ndim]
    for index in np.ndindex(leading_shape):
        values = func(arr[index])
        for output, value in zip(result, values, strict=True):
            output[index] = value
    return result


def _apply_spatial_blocks_multi(
    arrays: Sequence[np.ndarray],
    spatial_ndim: int,
    func: Callable[..., np.ndarray],
    *,
    dtype,
) -> np.ndarray:
    """Apply ``func`` over matching leading blocks from several arrays."""
    arrays = tuple(np.asarray(array) for array in arrays)
    if not arrays:
        raise ValueError("At least one array is required.")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("Spatial block inputs must have matching shapes.")
    spatial_ndim = int(np.clip(spatial_ndim, 1, max(arrays[0].ndim, 1)))
    if arrays[0].ndim <= spatial_ndim:
        return np.asarray(func(*arrays), dtype=dtype)

    result = np.empty(shape, dtype=dtype)
    leading_shape = shape[: arrays[0].ndim - spatial_ndim]
    for index in np.ndindex(leading_shape):
        result[index] = func(*(array[index] for array in arrays))
    return result


def _resolved_spatial_ndim(
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
            f"{requested}D spatial processing cannot be applied to a "
            f"{arr.ndim}D array."
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
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError("resolved_spatial_ndim must be an integer from 1 to 3.")
    resolved = int(value)
    if resolved not in {1, 2, 3}:
        raise ValueError("resolved_spatial_ndim must be an integer from 1 to 3.")
    return resolved


def _deconvolution_inputs(inputs) -> tuple[np.ndarray, np.ndarray]:
    try:
        image, psf = list(inputs)[:2]
    except Exception as exc:
        raise ValueError(
            "Deconvolution requires two inputs: Image and PSF."
        ) from exc
    if image is None or psf is None:
        raise ValueError("Deconvolution requires connected Image and PSF inputs.")
    return np.asarray(image), np.asarray(psf)


def _resolved_deconvolution_spatial_ndim(
    arr: np.ndarray,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
) -> int:
    spatial_ndim = _resolved_spatial_ndim(
        arr,
        spatial_mode,
        resolved_spatial_ndim,
    )
    if spatial_ndim not in {2, 3}:
        raise ValueError(
            "Deconvolution requires 2D YX or 3D ZYX spatial processing."
        )
    return spatial_ndim


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


def _crop_empty_psf_border(psf: np.ndarray) -> np.ndarray:
    nonzero = psf != 0
    if not np.any(nonzero):
        raise ValueError("PSF is empty after cleaning.")
    coords = np.argwhere(nonzero)
    slices = tuple(
        slice(int(coords[:, axis].min()), int(coords[:, axis].max()) + 1)
        for axis in range(psf.ndim)
    )
    return np.ascontiguousarray(psf[slices])


def _force_odd_psf_shape(psf: np.ndarray) -> np.ndarray:
    pad_width = [
        (0, 1) if int(size) % 2 == 0 else (0, 0)
        for size in psf.shape
    ]
    if not any(after for _, after in pad_width):
        return psf
    return np.pad(psf, pad_width, mode="constant", constant_values=0)


def _center_psf(
    psf: np.ndarray,
    center_mode: str,
    *,
    minimum_valid_sum: float,
) -> np.ndarray:
    mode = str(center_mode).strip().lower()
    if mode in {"", "none", "off", "disabled"}:
        return psf
    target = np.asarray([size // 2 for size in psf.shape], dtype=int)
    if mode.startswith("centroid"):
        source = _psf_centroid_index(psf, minimum_valid_sum=minimum_valid_sum)
    elif mode.startswith("peak"):
        source = np.asarray(np.unravel_index(int(np.argmax(psf)), psf.shape))
    else:
        raise ValueError(f"Unknown PSF center mode: {center_mode!r}.")
    shift = tuple(int(dst - src) for dst, src in zip(target, source, strict=True))
    if not any(shift):
        return psf
    return _integer_shift_zero_fill(psf, shift)


def _psf_centroid_index(
    psf: np.ndarray,
    *,
    minimum_valid_sum: float,
) -> np.ndarray:
    weights = np.maximum(psf, 0.0)
    total = float(weights.sum(dtype=np.float64))
    if total <= float(minimum_valid_sum) or not np.isfinite(total):
        return np.asarray(np.unravel_index(int(np.argmax(psf)), psf.shape))
    coords = np.indices(psf.shape, dtype=np.float64)
    centroid = [
        float(np.sum(coords[axis] * weights, dtype=np.float64) / total)
        for axis in range(psf.ndim)
    ]
    return np.asarray(
        [
            int(np.clip(np.rint(value), 0, psf.shape[axis] - 1))
            for axis, value in enumerate(centroid)
        ],
        dtype=int,
    )


def _integer_shift_zero_fill(psf: np.ndarray, shift: tuple[int, ...]) -> np.ndarray:
    shifted = np.zeros_like(psf)
    source_slices: list[slice] = []
    target_slices: list[slice] = []
    for size, offset in zip(psf.shape, shift, strict=True):
        if abs(offset) >= size:
            return shifted
        if offset > 0:
            source_slices.append(slice(0, size - offset))
            target_slices.append(slice(offset, size))
        elif offset < 0:
            source_slices.append(slice(-offset, size))
            target_slices.append(slice(0, size + offset))
        else:
            source_slices.append(slice(None))
            target_slices.append(slice(None))
    shifted[tuple(target_slices)] = psf[tuple(source_slices)]
    return shifted


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


def _estimate_rolling_ball_background(
    arr: np.ndarray,
    *,
    radius: float,
    light_background: bool,
    disable_smoothing: bool,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
    channel_axis: int | None,
    progress=None,
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 0:
        return arr.astype(np.float32, copy=True)
    radius_pixels = max(int(round(float(radius))), 1)
    spatial_ndim = _resolved_spatial_ndim(
        arr,
        spatial_mode,
        resolved_spatial_ndim,
    )
    output_dtype = np.float64 if arr.dtype == np.float64 else np.float32

    def estimate(block: np.ndarray) -> np.ndarray:
        if progress is not None:
            progress.check_cancelled()
        return _rolling_ball_background_block(
            block,
            radius_pixels=radius_pixels,
            light_background=bool(light_background),
            disable_smoothing=bool(disable_smoothing),
            output_dtype=output_dtype,
        )

    if channel_axis is not None:
        channels_last = np.moveaxis(arr, channel_axis, -1)
        block_ndim = min(spatial_ndim, max(channels_last.ndim - 1, 1))
        block_count = _spatial_block_count(channels_last[..., 0], block_ndim)
        total = max(block_count * int(channels_last.shape[-1]), 1)
        channels = [
            _apply_spatial_blocks(
                channels_last[..., channel],
                block_ndim,
                estimate,
                dtype=output_dtype,
                progress=progress,
                progress_start=channel * block_count,
                progress_total=total,
                progress_message=f"Rolling-ball channel {channel + 1}",
            )
            for channel in range(channels_last.shape[-1])
        ]
        stacked = np.stack(channels, axis=-1).astype(output_dtype, copy=False)
        return np.moveaxis(stacked, -1, channel_axis)
    return _apply_spatial_blocks(
        arr,
        spatial_ndim,
        estimate,
        dtype=output_dtype,
        progress=progress,
        progress_message="Rolling-ball background",
    )


def _rolling_ball_background_block(
    block: np.ndarray,
    *,
    radius_pixels: int,
    light_background: bool,
    disable_smoothing: bool,
    output_dtype,
) -> np.ndarray:
    values = np.asarray(block, dtype=output_dtype)
    if values.size == 0:
        return values.copy()

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values, dtype=output_dtype)
    low = float(finite.min())
    high = float(finite.max())
    safe = np.nan_to_num(values, nan=low, posinf=high, neginf=low)

    if not bool(disable_smoothing) and safe.ndim > 0:
        safe = ndi.uniform_filter(safe, size=3, mode="nearest")

    if bool(light_background):
        finite = safe[np.isfinite(safe)]
        low = float(finite.min()) if finite.size else low
        high = float(finite.max()) if finite.size else high
        offset = low + high
        inverted = offset - safe
        background = offset - restoration.rolling_ball(
            inverted,
            radius=radius_pixels,
        )
    else:
        background = restoration.rolling_ball(safe, radius=radius_pixels)
    return np.asarray(background, dtype=output_dtype)


def _validated_labels(data) -> np.ndarray:
    labels = np.asarray(data)
    if labels.dtype == bool or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("Label operations require a non-negative integer label image.")
    if labels.size and int(labels.min()) < 0:
        raise ValueError("Label operations require non-negative label IDs.")
    return labels


@dataclass(frozen=True)
class _MeasurementUnits:
    size_column: str
    equivalent_diameter_column: str
    physical_column: str
    scale_product: float
    scales: tuple[float, ...]
    length_unit: str
    area_unit: str
    unit_label: str
    column_units: dict[str, str]


@dataclass(frozen=True)
class _SpatialTableContext:
    spatial_ndim: int
    spatial_axes: tuple[int, ...]
    leading_axis_names: tuple[str, ...]
    spatial_axis_names: tuple[str, ...]
    leading_shape: tuple[int, ...]
    spatial_scales: tuple[float, ...]
    physical_unit: str
    has_physical_calibration: bool


@dataclass(frozen=True)
class _SkeletonUnits:
    length_column: str
    physical_column: str
    scales: tuple[float, ...]
    unit_label: str
    column_units: dict[str, str]


@dataclass(frozen=True)
class _MeshUnits:
    scales: tuple[float, float, float]
    length_unit: str
    area_unit: str
    volume_unit: str
    voxel_volume: float
    column_units: dict[str, str]


@dataclass(frozen=True)
class _SkeletonBranchTrace:
    branch_id: int
    branch_type: str
    path: tuple[int, ...]
    start_node: int | None
    end_node: int | None
    edge_count: int
    pixel_length: float
    physical_length: float
    euclidean_pixel_distance: float
    euclidean_physical_distance: float


def _skeletonize_method(method: str) -> str | None:
    value = str(method).strip().lower()
    if value == "auto":
        return None
    if value.startswith("lee"):
        return "lee"
    if value.startswith("zhang"):
        return "zhang"
    return None


def _skeleton_adjacency(
    component: np.ndarray,
    scales: tuple[float, ...] = (),
) -> tuple[np.ndarray, list[list[int]], np.ndarray, int, float, float]:
    component = np.asarray(component, dtype=bool)
    coords = np.argwhere(component)
    voxel_count = int(coords.shape[0])
    adjacency: list[list[int]] = [[] for _ in range(voxel_count)]
    if voxel_count == 0:
        return coords, adjacency, np.asarray([], dtype=np.int32), 0, 0.0, 0.0

    scales = _normalized_spatial_scales(component.ndim, scales)
    index_by_coord = {
        tuple(int(value) for value in coord): index
        for index, coord in enumerate(coords)
    }
    voxel_graph_edge_count = 0
    pixel_length = 0.0
    physical_length = 0.0
    for index, coord_array in enumerate(coords):
        coord = tuple(int(value) for value in coord_array)
        for offset in _half_neighbor_offsets(component.ndim):
            neighbor = tuple(
                coord[axis] + offset[axis]
                for axis in range(component.ndim)
            )
            neighbor_index = index_by_coord.get(neighbor)
            if neighbor_index is None:
                continue
            if not _valid_skeleton_edge(component, coord, offset):
                continue
            adjacency[index].append(neighbor_index)
            adjacency[neighbor_index].append(index)
            voxel_graph_edge_count += 1
            pixel_length += _offset_length(offset, (1.0,) * component.ndim)
            physical_length += _offset_length(offset, scales)

    degrees = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.int32)
    return (
        coords,
        adjacency,
        degrees,
        int(voxel_graph_edge_count),
        float(pixel_length),
        float(physical_length),
    )


def _skeleton_keypoint_masks_block(
    block: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    skeleton = np.asarray(block, dtype=bool)
    endpoints = np.zeros_like(skeleton, dtype=bool)
    junctions = np.zeros_like(skeleton, dtype=bool)
    isolated = np.zeros_like(skeleton, dtype=bool)
    coords, _adjacency, degrees, _edge_count, _pixel_length, _physical_length = (
        _skeleton_adjacency(skeleton)
    )
    for index, coord_array in enumerate(coords):
        coord = tuple(int(value) for value in coord_array)
        degree = int(degrees[index])
        if degree == 0:
            isolated[coord] = True
        elif degree == 1:
            endpoints[coord] = True
        elif degree >= 3:
            junctions[coord] = True
    return endpoints, junctions, isolated


def _label_skeleton_components_block(block: np.ndarray) -> np.ndarray:
    skeleton = np.asarray(block, dtype=bool)
    structure = ndi.generate_binary_structure(max(skeleton.ndim, 1), skeleton.ndim)
    labels, _count = ndi.label(skeleton, structure=structure)
    return labels.astype(np.int32, copy=False)


def _skeleton_branch_labels_component(component: np.ndarray) -> np.ndarray:
    component = np.asarray(component, dtype=bool)
    labels = np.zeros(component.shape, dtype=np.int32)
    coords, adjacency, degrees, _edge_count, _pixel_length, _physical_length = (
        _skeleton_adjacency(component)
    )
    if coords.shape[0] <= 1:
        return labels

    key_nodes = {index for index, degree in enumerate(degrees) if degree != 2}
    if not key_nodes:
        labels[component] = 1
        return labels

    visited: set[tuple[int, int]] = set()
    next_label = 1
    for start in sorted(key_nodes):
        if int(degrees[start]) == 0:
            continue
        for neighbor in adjacency[start]:
            edge = _edge_key(start, neighbor)
            if edge in visited:
                continue
            visited.add(edge)
            path = [start, neighbor]
            previous = start
            current = neighbor
            while current not in key_nodes:
                next_nodes = [node for node in adjacency[current] if node != previous]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = _edge_key(current, next_node)
                if edge in visited:
                    break
                visited.add(edge)
                path.append(next_node)
                previous, current = current, next_node

            assigned = False
            for node in path:
                degree = int(degrees[node])
                if degree == 0 or degree >= 3:
                    continue
                coord = tuple(int(value) for value in coords[node])
                labels[coord] = next_label
                assigned = True
            if assigned:
                next_label += 1
    return labels


def _label_skeleton_branches_block(block: np.ndarray) -> np.ndarray:
    skeleton = np.asarray(block, dtype=bool)
    component_labels = _label_skeleton_components_block(skeleton)
    branch_labels = np.zeros(skeleton.shape, dtype=np.int32)
    next_label = 1
    for component_id in range(1, int(component_labels.max()) + 1):
        component = component_labels == component_id
        labels = _skeleton_branch_labels_component(component)
        for local_label in range(1, int(labels.max()) + 1):
            branch_labels[labels == local_label] = next_label
            next_label += 1
    return branch_labels


def _apply_spatial_blocks_rgb(
    arr: np.ndarray,
    spatial_ndim: int,
    func: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    arr = np.asarray(arr)
    spatial_ndim = int(np.clip(spatial_ndim, 1, max(arr.ndim, 1)))
    if arr.ndim <= spatial_ndim:
        return np.asarray(func(arr), dtype=np.float32)

    leading_shape = arr.shape[: arr.ndim - spatial_ndim]
    spatial_shape = arr.shape[arr.ndim - spatial_ndim :]
    result = np.zeros(leading_shape + spatial_shape + (3,), dtype=np.float32)
    for index in np.ndindex(leading_shape):
        result[index] = func(arr[index])
    return result


def _skeleton_graph_overlay_block(
    block: np.ndarray,
    *,
    display_mode: str,
    node_size: int,
) -> np.ndarray:
    skeleton = np.asarray(block, dtype=bool)
    overlay = np.zeros(skeleton.shape + (3,), dtype=np.float32)
    if not np.any(skeleton):
        return overlay

    mode = str(display_mode).strip().lower()
    color_edges = "colored edge" in mode
    color_nodes = "colored node" in mode or "nodes" in mode
    white_edges = "white edge" in mode or not color_edges
    branch_labels = _label_skeleton_branches_block(skeleton)
    if color_edges:
        for label_id in range(1, int(branch_labels.max()) + 1):
            overlay[branch_labels == label_id] = _skeleton_palette_color(label_id)
    elif white_edges:
        overlay[skeleton] = (1.0, 1.0, 1.0)

    endpoints, junctions, isolated = _skeleton_keypoint_masks_block(skeleton)
    key_nodes = endpoints | junctions | isolated
    if color_nodes:
        radius = max(int(node_size), 1)
        endpoints = _dilate_keypoint_mask(endpoints, radius)
        junctions = _dilate_keypoint_mask(junctions, radius)
        isolated = _dilate_keypoint_mask(isolated, radius)
        overlay[endpoints] = (0.0, 1.0, 0.0)
        overlay[junctions] = (1.0, 0.0, 1.0)
        overlay[isolated] = (0.0, 0.65, 1.0)
    elif color_edges:
        overlay[key_nodes] = (1.0, 1.0, 1.0)
    return overlay


def _dilate_keypoint_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 1 or not np.any(mask):
        return mask
    structure = ndi.generate_binary_structure(max(mask.ndim, 1), 1)
    return ndi.binary_dilation(mask, structure=structure, iterations=radius - 1)


def _skeleton_palette_color(index: int) -> tuple[float, float, float]:
    palette = (
        (0.95, 0.20, 0.20),
        (0.20, 0.75, 1.00),
        (1.00, 0.80, 0.20),
        (0.45, 0.90, 0.35),
        (0.80, 0.45, 1.00),
        (1.00, 0.45, 0.75),
        (0.30, 1.00, 0.80),
        (1.00, 0.55, 0.25),
    )
    return palette[(int(index) - 1) % len(palette)]


def _trace_terminal_branch(
    endpoint: int,
    adjacency: list[list[int]],
    degrees: np.ndarray,
) -> tuple[list[int], int, int]:
    neighbor = adjacency[endpoint][0]
    path = [endpoint, neighbor]
    previous = endpoint
    current = neighbor
    edge_count = 1
    while int(degrees[current]) == 2:
        next_nodes = [node for node in adjacency[current] if node != previous]
        if not next_nodes:
            break
        next_node = next_nodes[0]
        path.append(next_node)
        previous, current = current, next_node
        edge_count += 1
    return path, current, edge_count


def _prune_skeleton_branches_block(
    block: np.ndarray,
    *,
    min_branch_length: float,
    iterations: int,
    remove_isolated: bool,
    length_units: str = "Pixels/voxels",
    scales: tuple[float, ...] = (),
) -> np.ndarray:
    skeleton = np.asarray(block, dtype=bool).copy()
    minimum = max(float(min_branch_length), 0.0)
    iterations = max(int(iterations), 1)
    use_physical = str(length_units).strip().lower().startswith("physical")
    scales = _normalized_spatial_scales(skeleton.ndim, scales)
    for _iteration in range(iterations):
        coords, adjacency, degrees, _edge_count, _pixel_length, _physical_length = (
            _skeleton_adjacency(skeleton, scales)
        )
        if coords.shape[0] == 0:
            break

        remove = np.zeros(coords.shape[0], dtype=bool)
        if remove_isolated:
            remove |= degrees == 0
        for endpoint in np.flatnonzero(degrees == 1):
            path, terminal_node, edge_count = _trace_terminal_branch(
                int(endpoint),
                adjacency,
                degrees,
            )
            if int(degrees[terminal_node]) < 3:
                continue
            trace = _skeleton_branch_trace_from_path(
                0,
                "",
                path,
                coords,
                scales,
            )
            branch_length = (
                trace.physical_length if use_physical else trace.pixel_length
            )
            if float(branch_length) >= minimum:
                continue
            for node in path:
                if int(degrees[node]) < 3:
                    remove[node] = True

        if not np.any(remove):
            break
        for node in np.flatnonzero(remove):
            coord = tuple(int(value) for value in coords[int(node)])
            skeleton[coord] = False
    return skeleton


def _skeleton_table_context(
    data,
    *,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
    axis_names: tuple[str, ...] | None,
    axis_types: tuple[str, ...] | None,
    axis_scales: tuple[float, ...] | None,
    axis_units: tuple[str | None, ...] | None,
) -> tuple[
    np.ndarray,
    int,
    tuple[str, ...],
    tuple[str, ...],
    tuple[int, ...],
    _SkeletonUnits,
]:
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )
    has_axis_names = axis_names is not None and len(axis_names) == mask.ndim
    axis_names = _measurement_axis_names(mask.ndim, axis_names)
    axis_types = _measurement_axis_types(mask.ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        mask.ndim,
        spatial_ndim,
        axis_names,
        axis_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(mask.ndim - spatial_ndim, mask.ndim))
    mask_for_analysis = np.moveaxis(
        mask,
        spatial_axes,
        tuple(range(mask.ndim - spatial_ndim, mask.ndim)),
    )
    moved_axis_names = tuple(
        axis_names[index]
        for index in range(mask.ndim)
        if index not in spatial_axes
    ) + tuple(axis_names[index] for index in spatial_axes)
    moved_axis_scales = _reordered_axis_values(axis_scales, mask.ndim, spatial_axes)
    moved_axis_units = _reordered_axis_values(axis_units, mask.ndim, spatial_axes)
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[: mask.ndim - spatial_ndim],
        fallback=tuple(f"axis_{index}" for index in range(mask.ndim - spatial_ndim)),
    )
    spatial_axis_names = _safe_axis_column_names(
        moved_axis_names[-spatial_ndim:]
        if has_axis_names
        else tuple("" for _axis in range(spatial_ndim)),
        fallback=("z", "y", "x")[-spatial_ndim:],
    )
    leading_shape = mask_for_analysis.shape[: mask.ndim - spatial_ndim]
    units = _skeleton_units(
        spatial_ndim,
        moved_axis_scales[-spatial_ndim:],
        moved_axis_units[-spatial_ndim:],
    )
    return (
        mask_for_analysis,
        spatial_ndim,
        leading_axis_names,
        spatial_axis_names,
        leading_shape,
        units,
    )


def _extend_skeleton_block_columns(
    columns: dict[str, list[object]],
    block_columns: dict[str, list[object]],
    leading_axis_names: tuple[str, ...],
    block_index: tuple[int, ...],
) -> None:
    row_count = len(next(iter(block_columns.values()), []))
    for axis_position, axis_name in enumerate(leading_axis_names):
        columns[f"{axis_name}_index"].extend(
            [int(block_index[axis_position])] * row_count
        )
    for name, values in block_columns.items():
        columns[name].extend(values)


def _skeleton_empty_columns(
    leading_axis_names: tuple[str, ...],
    units: _SkeletonUnits,
) -> dict[str, list[object]]:
    columns: dict[str, list[object]] = {}
    for axis_name in leading_axis_names:
        columns[f"{axis_name}_index"] = []
    columns["component_id"] = []
    columns["component_count_in_block"] = []
    columns["component_voxel_fraction"] = []
    columns["skeleton_voxel_count"] = []
    columns["endpoint_voxel_count"] = []
    columns["junction_voxel_count"] = []
    columns["isolated_node_count"] = []
    columns["branch_count"] = []
    columns["graph_node_count"] = []
    columns["graph_edge_count"] = []
    columns["voxel_graph_edge_count"] = []
    columns["cycle_count"] = []
    columns[units.length_column] = []
    if units.physical_column:
        columns[units.physical_column] = []
        columns["physical_unit"] = []
    return columns


def _analyze_skeleton_block(
    skeleton: np.ndarray,
    units: _SkeletonUnits,
) -> dict[str, list[object]]:
    skeleton = np.asarray(skeleton, dtype=bool)
    ndim = skeleton.ndim
    structure = ndi.generate_binary_structure(ndim, ndim)
    component_labels, component_count = ndi.label(skeleton, structure=structure)
    total_voxel_count = int(np.count_nonzero(skeleton))
    columns: dict[str, list[object]] = {
        "component_id": [],
        "component_count_in_block": [],
        "component_voxel_fraction": [],
        "skeleton_voxel_count": [],
        "endpoint_voxel_count": [],
        "junction_voxel_count": [],
        "isolated_node_count": [],
        "branch_count": [],
        "graph_node_count": [],
        "graph_edge_count": [],
        "voxel_graph_edge_count": [],
        "cycle_count": [],
        units.length_column: [],
    }
    if units.physical_column:
        columns[units.physical_column] = []

    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        graph = _skeleton_component_graph(component, units.scales)
        columns["component_id"].append(component_id)
        columns["component_count_in_block"].append(int(component_count))
        columns["component_voxel_fraction"].append(
            float(graph.voxel_count / total_voxel_count)
            if total_voxel_count
            else 0.0
        )
        columns["skeleton_voxel_count"].append(graph.voxel_count)
        columns["endpoint_voxel_count"].append(graph.endpoint_count)
        columns["junction_voxel_count"].append(graph.junction_count)
        columns["isolated_node_count"].append(graph.isolated_node_count)
        columns["branch_count"].append(graph.branch_count)
        columns["graph_node_count"].append(graph.graph_node_count)
        columns["graph_edge_count"].append(graph.graph_edge_count)
        columns["voxel_graph_edge_count"].append(graph.voxel_graph_edge_count)
        columns["cycle_count"].append(graph.cycle_count)
        columns[units.length_column].append(graph.pixel_length)
        if units.physical_column:
            columns[units.physical_column].append(graph.physical_length)
    return columns


def _skeleton_branch_empty_columns(
    leading_axis_names: tuple[str, ...],
    spatial_axis_names: tuple[str, ...],
    units: _SkeletonUnits,
) -> dict[str, list[object]]:
    columns: dict[str, list[object]] = {}
    for axis_name in leading_axis_names:
        columns[f"{axis_name}_index"] = []
    columns["component_id"] = []
    columns["branch_id"] = []
    columns["branch_type"] = []
    columns["branch_voxel_count"] = []
    columns["branch_edge_count"] = []
    columns[_branch_length_column(units)] = []
    columns[_branch_euclidean_column(units)] = []
    columns["branch_tortuosity"] = []
    if units.physical_column:
        columns["branch_length_physical"] = []
        columns["branch_euclidean_distance_physical"] = []
        columns["physical_unit"] = []
    for axis_name in spatial_axis_names:
        columns[f"start_{axis_name}"] = []
    for axis_name in spatial_axis_names:
        columns[f"end_{axis_name}"] = []
    return columns


def _measure_skeleton_branches_block(
    skeleton: np.ndarray,
    units: _SkeletonUnits,
    spatial_axis_names: tuple[str, ...],
) -> dict[str, list[object]]:
    skeleton = np.asarray(skeleton, dtype=bool)
    ndim = skeleton.ndim
    structure = ndi.generate_binary_structure(ndim, ndim)
    component_labels, component_count = ndi.label(skeleton, structure=structure)
    columns = _skeleton_branch_empty_columns((), spatial_axis_names, units)

    global_branch_id = 1
    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        coords, traces = _skeleton_branch_traces(component, units.scales)
        for trace in traces:
            columns["component_id"].append(int(component_id))
            columns["branch_id"].append(global_branch_id)
            columns["branch_type"].append(trace.branch_type)
            columns["branch_voxel_count"].append(len(trace.path))
            columns["branch_edge_count"].append(trace.edge_count)
            columns[_branch_length_column(units)].append(trace.pixel_length)
            columns[_branch_euclidean_column(units)].append(
                trace.euclidean_pixel_distance
            )
            denominator = trace.euclidean_pixel_distance
            columns["branch_tortuosity"].append(
                float(trace.pixel_length / denominator)
                if denominator > 0
                else 0.0
            )
            if units.physical_column:
                columns["branch_length_physical"].append(trace.physical_length)
                columns["branch_euclidean_distance_physical"].append(
                    trace.euclidean_physical_distance
                )
            start_coord = _trace_coord(coords, trace.start_node)
            end_coord = _trace_coord(coords, trace.end_node)
            for axis_index, axis_name in enumerate(spatial_axis_names):
                columns[f"start_{axis_name}"].append(start_coord[axis_index])
            for axis_index, axis_name in enumerate(spatial_axis_names):
                columns[f"end_{axis_name}"].append(end_coord[axis_index])
            global_branch_id += 1
    return columns


def _skeleton_graph_node_empty_columns(
    leading_axis_names: tuple[str, ...],
    spatial_axis_names: tuple[str, ...],
) -> dict[str, list[object]]:
    columns: dict[str, list[object]] = {}
    for axis_name in leading_axis_names:
        columns[f"{axis_name}_index"] = []
    columns["component_id"] = []
    columns["node_id"] = []
    columns["node_type"] = []
    columns["node_degree"] = []
    columns["skeleton_voxel_index"] = []
    for axis_name in spatial_axis_names:
        columns[f"{axis_name}_coord"] = []
    return columns


def _skeleton_graph_edge_empty_columns(
    leading_axis_names: tuple[str, ...],
    spatial_axis_names: tuple[str, ...],
    units: _SkeletonUnits,
) -> dict[str, list[object]]:
    columns: dict[str, list[object]] = {}
    for axis_name in leading_axis_names:
        columns[f"{axis_name}_index"] = []
    columns["component_id"] = []
    columns["edge_id"] = []
    columns["start_node_id"] = []
    columns["end_node_id"] = []
    columns["branch_type"] = []
    columns["branch_voxel_count"] = []
    columns["branch_edge_count"] = []
    columns[_branch_length_column(units)] = []
    columns[_branch_euclidean_column(units)] = []
    columns["branch_tortuosity"] = []
    if units.physical_column:
        columns["branch_length_physical"] = []
        columns["branch_euclidean_distance_physical"] = []
        columns["physical_unit"] = []
    for axis_name in spatial_axis_names:
        columns[f"start_{axis_name}"] = []
    for axis_name in spatial_axis_names:
        columns[f"end_{axis_name}"] = []
    return columns


def _skeleton_graph_tables_block(
    skeleton: np.ndarray,
    units: _SkeletonUnits,
    spatial_axis_names: tuple[str, ...],
) -> tuple[dict[str, list[object]], dict[str, list[object]]]:
    skeleton = np.asarray(skeleton, dtype=bool)
    ndim = skeleton.ndim
    structure = ndi.generate_binary_structure(ndim, ndim)
    component_labels, component_count = ndi.label(skeleton, structure=structure)
    node_columns = _skeleton_graph_node_empty_columns((), spatial_axis_names)
    edge_columns = _skeleton_graph_edge_empty_columns((), spatial_axis_names, units)

    global_node_id = 1
    global_edge_id = 1
    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        coords, adjacency, degrees, _edge_count, _pixel_length, _physical_length = (
            _skeleton_adjacency(component, units.scales)
        )
        graph_node_by_voxel_index: dict[int, int] = {}
        for voxel_index, degree in enumerate(degrees):
            degree = int(degree)
            if degree == 2:
                continue
            graph_node_by_voxel_index[int(voxel_index)] = global_node_id
            coord = _trace_coord(coords, int(voxel_index))
            node_columns["component_id"].append(int(component_id))
            node_columns["node_id"].append(int(global_node_id))
            node_columns["node_type"].append(_skeleton_node_type(degree))
            node_columns["node_degree"].append(degree)
            node_columns["skeleton_voxel_index"].append(int(voxel_index))
            for axis_index, axis_name in enumerate(spatial_axis_names):
                node_columns[f"{axis_name}_coord"].append(coord[axis_index])
            global_node_id += 1

        _coords, traces = _skeleton_branch_traces(component, units.scales)
        for trace in traces:
            if trace.edge_count <= 0:
                continue
            start_node_id = (
                graph_node_by_voxel_index.get(int(trace.start_node), 0)
                if trace.start_node is not None
                else 0
            )
            end_node_id = (
                graph_node_by_voxel_index.get(int(trace.end_node), 0)
                if trace.end_node is not None
                else 0
            )
            edge_columns["component_id"].append(int(component_id))
            edge_columns["edge_id"].append(int(global_edge_id))
            edge_columns["start_node_id"].append(int(start_node_id))
            edge_columns["end_node_id"].append(int(end_node_id))
            edge_columns["branch_type"].append(trace.branch_type)
            edge_columns["branch_voxel_count"].append(len(trace.path))
            edge_columns["branch_edge_count"].append(trace.edge_count)
            edge_columns[_branch_length_column(units)].append(trace.pixel_length)
            edge_columns[_branch_euclidean_column(units)].append(
                trace.euclidean_pixel_distance
            )
            denominator = trace.euclidean_pixel_distance
            edge_columns["branch_tortuosity"].append(
                float(trace.pixel_length / denominator)
                if denominator > 0
                else 0.0
            )
            if units.physical_column:
                edge_columns["branch_length_physical"].append(trace.physical_length)
                edge_columns["branch_euclidean_distance_physical"].append(
                    trace.euclidean_physical_distance
                )
            start_coord = _trace_coord(coords, trace.start_node)
            end_coord = _trace_coord(coords, trace.end_node)
            for axis_index, axis_name in enumerate(spatial_axis_names):
                edge_columns[f"start_{axis_name}"].append(start_coord[axis_index])
            for axis_index, axis_name in enumerate(spatial_axis_names):
                edge_columns[f"end_{axis_name}"].append(end_coord[axis_index])
            global_edge_id += 1

    return node_columns, edge_columns


def _skeleton_graph_node_column_units(units: _SkeletonUnits) -> dict[str, str]:
    column_units = {
        "component_id": "id",
        "node_id": "id",
        "node_degree": "count",
        "skeleton_voxel_index": "index",
    }
    if units.physical_column:
        column_units["physical_unit"] = "text"
    return column_units


def _skeleton_graph_edge_column_units(units: _SkeletonUnits) -> dict[str, str]:
    column_units = {
        "component_id": "id",
        "edge_id": "id",
        "start_node_id": "id",
        "end_node_id": "id",
        "branch_voxel_count": "voxels",
        "branch_edge_count": "count",
        _branch_length_column(units): (
            "voxels" if units.length_column.endswith("voxels") else "pixels"
        ),
        _branch_euclidean_column(units): (
            "voxels" if units.length_column.endswith("voxels") else "pixels"
        ),
        "branch_tortuosity": "ratio",
    }
    if units.physical_column:
        column_units["branch_length_physical"] = units.unit_label
        column_units["branch_euclidean_distance_physical"] = units.unit_label
        column_units["physical_unit"] = "text"
    return column_units


def _overall_skeleton_network_empty_columns(
    leading_axis_names: tuple[str, ...],
    units: _SkeletonUnits,
) -> dict[str, list[object]]:
    columns: dict[str, list[object]] = {}
    for axis_name in leading_axis_names:
        columns[f"{axis_name}_index"] = []
    columns["component_count"] = []
    columns["skeleton_voxel_count"] = []
    columns["largest_component_voxel_count"] = []
    columns["largest_component_voxel_fraction"] = []
    columns["isolated_component_count"] = []
    columns["endpoint_voxel_count"] = []
    columns["junction_voxel_count"] = []
    columns["isolated_node_count"] = []
    columns["branch_count"] = []
    columns["graph_node_count"] = []
    columns["graph_edge_count"] = []
    columns["voxel_graph_edge_count"] = []
    columns["cycle_count"] = []
    columns[units.length_column] = []
    columns[f"mean_branch_{units.length_column.removeprefix('skeleton_')}"] = []
    columns[f"median_branch_{units.length_column.removeprefix('skeleton_')}"] = []
    columns[f"max_branch_{units.length_column.removeprefix('skeleton_')}"] = []
    columns["mean_branch_tortuosity"] = []
    columns["network_connectedness_fraction"] = []
    columns["fragmentation_index"] = []
    columns["isolated_component_fraction"] = []
    columns["branches_per_component"] = []
    columns["endpoints_per_component"] = []
    columns["junctions_per_component"] = []
    columns["cycles_per_component"] = []
    columns["components_per_skeleton_length"] = []
    columns["branches_per_skeleton_length"] = []
    columns["endpoints_per_skeleton_length"] = []
    columns["junctions_per_skeleton_length"] = []
    columns["cycles_per_skeleton_length"] = []
    if units.physical_column:
        columns[units.physical_column] = []
        columns["mean_branch_length_physical"] = []
        columns["median_branch_length_physical"] = []
        columns["max_branch_length_physical"] = []
        columns["components_per_physical_length"] = []
        columns["branches_per_physical_length"] = []
        columns["endpoints_per_physical_length"] = []
        columns["junctions_per_physical_length"] = []
        columns["cycles_per_physical_length"] = []
        columns["physical_unit"] = []
    return columns


def _measure_overall_skeleton_network_block(
    skeleton: np.ndarray,
    units: _SkeletonUnits,
) -> dict[str, list[object]]:
    skeleton = np.asarray(skeleton, dtype=bool)
    ndim = skeleton.ndim
    structure = ndi.generate_binary_structure(ndim, ndim)
    component_labels, component_count = ndi.label(skeleton, structure=structure)
    total_voxel_count = int(np.count_nonzero(skeleton))
    columns = _overall_skeleton_network_empty_columns((), units)
    component_voxel_counts: list[int] = []
    branch_lengths: list[float] = []
    branch_physical_lengths: list[float] = []
    branch_tortuosities: list[float] = []
    totals = {
        "endpoint_voxel_count": 0,
        "junction_voxel_count": 0,
        "isolated_node_count": 0,
        "branch_count": 0,
        "graph_node_count": 0,
        "graph_edge_count": 0,
        "voxel_graph_edge_count": 0,
        "cycle_count": 0,
        units.length_column: 0.0,
    }
    if units.physical_column:
        totals[units.physical_column] = 0.0

    isolated_component_count = 0
    for component_id in range(1, component_count + 1):
        component = component_labels == component_id
        graph = _skeleton_component_graph(component, units.scales)
        component_voxel_counts.append(graph.voxel_count)
        isolated_component_count += int(
            graph.voxel_count == 1 and graph.isolated_node_count == 1
        )
        totals["endpoint_voxel_count"] += graph.endpoint_count
        totals["junction_voxel_count"] += graph.junction_count
        totals["isolated_node_count"] += graph.isolated_node_count
        totals["branch_count"] += graph.branch_count
        totals["graph_node_count"] += graph.graph_node_count
        totals["graph_edge_count"] += graph.graph_edge_count
        totals["voxel_graph_edge_count"] += graph.voxel_graph_edge_count
        totals["cycle_count"] += graph.cycle_count
        totals[units.length_column] += graph.pixel_length
        if units.physical_column:
            totals[units.physical_column] += graph.physical_length

        _coords, traces = _skeleton_branch_traces(component, units.scales)
        for trace in traces:
            if trace.edge_count <= 0:
                continue
            branch_lengths.append(float(trace.pixel_length))
            branch_physical_lengths.append(float(trace.physical_length))
            if trace.euclidean_pixel_distance > 0:
                branch_tortuosities.append(
                    float(trace.pixel_length / trace.euclidean_pixel_distance)
                )

    largest_component_voxel_count = max(component_voxel_counts, default=0)
    largest_fraction = (
        float(largest_component_voxel_count / total_voxel_count)
        if total_voxel_count
        else 0.0
    )
    fragmentation_index = (
        float(component_count / total_voxel_count) if total_voxel_count else 0.0
    )
    total_length = float(totals[units.length_column])
    total_physical_length = (
        float(totals[units.physical_column]) if units.physical_column else 0.0
    )
    length_suffix = units.length_column.removeprefix("skeleton_")
    columns["component_count"].append(int(component_count))
    columns["skeleton_voxel_count"].append(total_voxel_count)
    columns["largest_component_voxel_count"].append(largest_component_voxel_count)
    columns["largest_component_voxel_fraction"].append(largest_fraction)
    columns["isolated_component_count"].append(int(isolated_component_count))
    for name in (
        "endpoint_voxel_count",
        "junction_voxel_count",
        "isolated_node_count",
        "branch_count",
        "graph_node_count",
        "graph_edge_count",
        "voxel_graph_edge_count",
        "cycle_count",
        units.length_column,
    ):
        columns[name].append(totals[name])
    columns[f"mean_branch_{length_suffix}"].append(_safe_mean(branch_lengths))
    columns[f"median_branch_{length_suffix}"].append(_safe_median(branch_lengths))
    columns[f"max_branch_{length_suffix}"].append(_safe_max(branch_lengths))
    columns["mean_branch_tortuosity"].append(_safe_mean(branch_tortuosities))
    columns["network_connectedness_fraction"].append(largest_fraction)
    columns["fragmentation_index"].append(fragmentation_index)
    columns["isolated_component_fraction"].append(
        _safe_ratio(isolated_component_count, component_count)
    )
    columns["branches_per_component"].append(
        _safe_ratio(totals["branch_count"], component_count)
    )
    columns["endpoints_per_component"].append(
        _safe_ratio(totals["endpoint_voxel_count"], component_count)
    )
    columns["junctions_per_component"].append(
        _safe_ratio(totals["junction_voxel_count"], component_count)
    )
    columns["cycles_per_component"].append(
        _safe_ratio(totals["cycle_count"], component_count)
    )
    columns["components_per_skeleton_length"].append(
        _safe_ratio(component_count, total_length)
    )
    columns["branches_per_skeleton_length"].append(
        _safe_ratio(totals["branch_count"], total_length)
    )
    columns["endpoints_per_skeleton_length"].append(
        _safe_ratio(totals["endpoint_voxel_count"], total_length)
    )
    columns["junctions_per_skeleton_length"].append(
        _safe_ratio(totals["junction_voxel_count"], total_length)
    )
    columns["cycles_per_skeleton_length"].append(
        _safe_ratio(totals["cycle_count"], total_length)
    )
    if units.physical_column:
        columns[units.physical_column].append(totals[units.physical_column])
        columns["mean_branch_length_physical"].append(
            _safe_mean(branch_physical_lengths)
        )
        columns["median_branch_length_physical"].append(
            _safe_median(branch_physical_lengths)
        )
        columns["max_branch_length_physical"].append(_safe_max(branch_physical_lengths))
        columns["components_per_physical_length"].append(
            _safe_ratio(component_count, total_physical_length)
        )
        columns["branches_per_physical_length"].append(
            _safe_ratio(totals["branch_count"], total_physical_length)
        )
        columns["endpoints_per_physical_length"].append(
            _safe_ratio(totals["endpoint_voxel_count"], total_physical_length)
        )
        columns["junctions_per_physical_length"].append(
            _safe_ratio(totals["junction_voxel_count"], total_physical_length)
        )
        columns["cycles_per_physical_length"].append(
            _safe_ratio(totals["cycle_count"], total_physical_length)
        )
    return columns


def _overall_skeleton_network_column_units(units: _SkeletonUnits) -> dict[str, str]:
    distance_unit = "voxels" if units.length_column.endswith("voxels") else "pixels"
    length_suffix = units.length_column.removeprefix("skeleton_")
    column_units = {
        "component_count": "count",
        "skeleton_voxel_count": "voxels",
        "largest_component_voxel_count": "voxels",
        "largest_component_voxel_fraction": "fraction",
        "isolated_component_count": "count",
        "endpoint_voxel_count": "voxels",
        "junction_voxel_count": "voxels",
        "isolated_node_count": "count",
        "branch_count": "count",
        "graph_node_count": "count",
        "graph_edge_count": "count",
        "voxel_graph_edge_count": "count",
        "cycle_count": "count",
        units.length_column: distance_unit,
        f"mean_branch_{length_suffix}": distance_unit,
        f"median_branch_{length_suffix}": distance_unit,
        f"max_branch_{length_suffix}": distance_unit,
        "mean_branch_tortuosity": "ratio",
        "network_connectedness_fraction": "fraction",
        "fragmentation_index": "components/voxel",
        "isolated_component_fraction": "fraction",
        "branches_per_component": "count/component",
        "endpoints_per_component": "count/component",
        "junctions_per_component": "count/component",
        "cycles_per_component": "count/component",
        "components_per_skeleton_length": f"count/{distance_unit}",
        "branches_per_skeleton_length": f"count/{distance_unit}",
        "endpoints_per_skeleton_length": f"count/{distance_unit}",
        "junctions_per_skeleton_length": f"count/{distance_unit}",
        "cycles_per_skeleton_length": f"count/{distance_unit}",
    }
    if units.physical_column:
        column_units[units.physical_column] = units.unit_label
        column_units["mean_branch_length_physical"] = units.unit_label
        column_units["median_branch_length_physical"] = units.unit_label
        column_units["max_branch_length_physical"] = units.unit_label
        column_units["components_per_physical_length"] = f"count/{units.unit_label}"
        column_units["branches_per_physical_length"] = f"count/{units.unit_label}"
        column_units["endpoints_per_physical_length"] = f"count/{units.unit_label}"
        column_units["junctions_per_physical_length"] = f"count/{units.unit_label}"
        column_units["cycles_per_physical_length"] = f"count/{units.unit_label}"
        column_units["physical_unit"] = "text"
    return column_units


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _safe_max(values: Sequence[float]) -> float:
    return float(np.max(values)) if values else 0.0


def _safe_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    return float(numerator) / denominator if denominator > 0 else 0.0


def _trace_coord(coords: np.ndarray, node: int | None) -> tuple[int, ...]:
    if node is None or coords.shape[0] == 0:
        return tuple(0 for _axis in range(coords.shape[1] if coords.ndim == 2 else 0))
    return tuple(int(value) for value in coords[int(node)])


def _branch_length_column(units: _SkeletonUnits) -> str:
    return (
        "branch_length_voxels"
        if units.length_column.endswith("voxels")
        else "branch_length_pixels"
    )


def _branch_euclidean_column(units: _SkeletonUnits) -> str:
    return (
        "branch_euclidean_distance_voxels"
        if units.length_column.endswith("voxels")
        else "branch_euclidean_distance_pixels"
    )


def _skeleton_branch_column_units(units: _SkeletonUnits) -> dict[str, str]:
    distance_unit = "voxels" if units.length_column.endswith("voxels") else "pixels"
    column_units = {
        "branch_voxel_count": "voxels",
        "branch_edge_count": "count",
        _branch_length_column(units): distance_unit,
        _branch_euclidean_column(units): distance_unit,
        "branch_tortuosity": "ratio",
    }
    if units.physical_column:
        column_units["branch_length_physical"] = units.unit_label
        column_units["branch_euclidean_distance_physical"] = units.unit_label
    return column_units


@dataclass(frozen=True)
class _SkeletonGraphMetrics:
    voxel_count: int
    endpoint_count: int
    junction_count: int
    isolated_node_count: int
    branch_count: int
    graph_node_count: int
    graph_edge_count: int
    voxel_graph_edge_count: int
    cycle_count: int
    pixel_length: float
    physical_length: float


def _skeleton_component_graph(
    component: np.ndarray,
    scales: tuple[float, ...],
) -> _SkeletonGraphMetrics:
    coords = np.argwhere(component)
    voxel_count = int(coords.shape[0])
    if voxel_count == 0:
        return _SkeletonGraphMetrics(
            voxel_count=0,
            endpoint_count=0,
            junction_count=0,
            isolated_node_count=0,
            branch_count=0,
            graph_node_count=0,
            graph_edge_count=0,
            voxel_graph_edge_count=0,
            cycle_count=0,
            pixel_length=0.0,
            physical_length=0.0,
        )

    (
        _coords,
        adjacency,
        degrees,
        voxel_graph_edge_count,
        pixel_length,
        physical_length,
    ) = _skeleton_adjacency(component, scales)
    isolated_node_count = int(np.count_nonzero(degrees == 0))
    endpoint_count = int(np.count_nonzero(degrees == 1))
    junction_count = int(np.count_nonzero(degrees >= 3))
    branch_count = _skeleton_branch_count(adjacency, degrees)
    graph_node_count = endpoint_count + junction_count + isolated_node_count
    graph_edge_count = branch_count
    cycle_count = max(0, int(voxel_graph_edge_count) - voxel_count + 1)
    return _SkeletonGraphMetrics(
        voxel_count=voxel_count,
        endpoint_count=endpoint_count,
        junction_count=junction_count,
        isolated_node_count=isolated_node_count,
        branch_count=branch_count,
        graph_node_count=graph_node_count,
        graph_edge_count=graph_edge_count,
        voxel_graph_edge_count=int(voxel_graph_edge_count),
        cycle_count=cycle_count,
        pixel_length=float(pixel_length),
        physical_length=float(physical_length),
    )


def _skeleton_branch_count(adjacency: list[list[int]], degrees: np.ndarray) -> int:
    if not adjacency:
        return 0
    if len(adjacency) == 1:
        return 0
    key_nodes = {index for index, degree in enumerate(degrees) if degree != 2}
    if not key_nodes:
        return 1

    visited: set[tuple[int, int]] = set()
    branches = 0
    for start in sorted(key_nodes):
        for neighbor in adjacency[start]:
            edge = _edge_key(start, neighbor)
            if edge in visited:
                continue
            visited.add(edge)
            previous = start
            current = neighbor
            while current not in key_nodes:
                next_nodes = [node for node in adjacency[current] if node != previous]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = _edge_key(current, next_node)
                if edge in visited:
                    break
                visited.add(edge)
                previous, current = current, next_node
            branches += 1
    return branches


def _skeleton_branch_traces(
    component: np.ndarray,
    scales: tuple[float, ...],
) -> tuple[np.ndarray, list[_SkeletonBranchTrace]]:
    coords, adjacency, degrees, edge_count, pixel_length, physical_length = (
        _skeleton_adjacency(component, scales)
    )
    if coords.shape[0] == 0:
        return coords, []
    if coords.shape[0] == 1:
        return coords, [
            _SkeletonBranchTrace(
                branch_id=1,
                branch_type="isolated",
                path=(0,),
                start_node=0,
                end_node=0,
                edge_count=0,
                pixel_length=0.0,
                physical_length=0.0,
                euclidean_pixel_distance=0.0,
                euclidean_physical_distance=0.0,
            )
        ]

    key_nodes = {index for index, degree in enumerate(degrees) if degree != 2}
    if not key_nodes:
        return coords, [
            _SkeletonBranchTrace(
                branch_id=1,
                branch_type="cycle",
                path=tuple(range(coords.shape[0])),
                start_node=None,
                end_node=None,
                edge_count=int(edge_count),
                pixel_length=float(pixel_length),
                physical_length=float(physical_length),
                euclidean_pixel_distance=0.0,
                euclidean_physical_distance=0.0,
            )
        ]

    visited: set[tuple[int, int]] = set()
    traces: list[_SkeletonBranchTrace] = []
    next_branch_id = 1
    for start in sorted(key_nodes):
        if int(degrees[start]) == 0:
            traces.append(
                _SkeletonBranchTrace(
                    branch_id=next_branch_id,
                    branch_type="isolated",
                    path=(start,),
                    start_node=start,
                    end_node=start,
                    edge_count=0,
                    pixel_length=0.0,
                    physical_length=0.0,
                    euclidean_pixel_distance=0.0,
                    euclidean_physical_distance=0.0,
                )
            )
            next_branch_id += 1
            continue
        for neighbor in adjacency[start]:
            edge = _edge_key(start, neighbor)
            if edge in visited:
                continue
            visited.add(edge)
            path = [start, neighbor]
            previous = start
            current = neighbor
            while current not in key_nodes:
                next_nodes = [node for node in adjacency[current] if node != previous]
                if not next_nodes:
                    break
                next_node = next_nodes[0]
                edge = _edge_key(current, next_node)
                if edge in visited:
                    break
                visited.add(edge)
                path.append(next_node)
                previous, current = current, next_node

            start_node = int(path[0])
            end_node = int(path[-1])
            branch_type = _skeleton_branch_type(
                int(degrees[start_node]),
                int(degrees[end_node]),
            )
            trace = _skeleton_branch_trace_from_path(
                next_branch_id,
                branch_type,
                path,
                coords,
                scales,
            )
            traces.append(trace)
            next_branch_id += 1
    return coords, traces


def _skeleton_branch_trace_from_path(
    branch_id: int,
    branch_type: str,
    path: Sequence[int],
    coords: np.ndarray,
    scales: tuple[float, ...],
) -> _SkeletonBranchTrace:
    scales = _normalized_spatial_scales(coords.shape[1], scales)
    pixel_length = 0.0
    physical_length = 0.0
    for first, second in zip(path, path[1:], strict=False):
        offset = tuple(
            int(coords[int(second), axis] - coords[int(first), axis])
            for axis in range(coords.shape[1])
        )
        pixel_length += _offset_length(offset, (1.0,) * coords.shape[1])
        physical_length += _offset_length(offset, scales)
    start_node = int(path[0]) if path else None
    end_node = int(path[-1]) if path else None
    euclidean_pixel = 0.0
    euclidean_physical = 0.0
    if start_node is not None and end_node is not None:
        delta = tuple(
            int(coords[end_node, axis] - coords[start_node, axis])
            for axis in range(coords.shape[1])
        )
        euclidean_pixel = _offset_length(delta, (1.0,) * coords.shape[1])
        euclidean_physical = _offset_length(delta, scales)
    return _SkeletonBranchTrace(
        branch_id=int(branch_id),
        branch_type=branch_type,
        path=tuple(int(node) for node in path),
        start_node=start_node,
        end_node=end_node,
        edge_count=max(len(path) - 1, 0),
        pixel_length=float(pixel_length),
        physical_length=float(physical_length),
        euclidean_pixel_distance=float(euclidean_pixel),
        euclidean_physical_distance=float(euclidean_physical),
    )


def _skeleton_branch_type(start_degree: int, end_degree: int) -> str:
    start_type = _skeleton_node_type(start_degree)
    end_type = _skeleton_node_type(end_degree)
    priority = {"isolated": 0, "endpoint": 1, "junction": 2, "path": 3}
    if priority.get(end_type, 99) < priority.get(start_type, 99):
        start_type, end_type = end_type, start_type
    return f"{start_type}_to_{end_type}"


def _skeleton_node_type(degree: int) -> str:
    if degree <= 0:
        return "isolated"
    if degree == 1:
        return "endpoint"
    if degree == 2:
        return "path"
    return "junction"


def _edge_key(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _half_neighbor_offsets(ndim: int) -> tuple[tuple[int, ...], ...]:
    zero = (0,) * ndim
    return tuple(
        offset
        for offset in product((-1, 0, 1), repeat=ndim)
        if offset != zero and offset > zero
    )


def _valid_skeleton_edge(
    component: np.ndarray,
    coord: tuple[int, ...],
    offset: tuple[int, ...],
) -> bool:
    nonzero_axes = [axis for axis, value in enumerate(offset) if value]
    if len(nonzero_axes) <= 1:
        return True
    # Avoid adding a diagonal shortcut when a lower-order skeleton voxel already
    # connects the same neighborhood through a face/edge path.
    for length in range(1, len(nonzero_axes)):
        for axes in combinations(nonzero_axes, length):
            intermediate = list(coord)
            for axis in axes:
                intermediate[axis] += offset[axis]
            intermediate_tuple = tuple(intermediate)
            if _coord_in_bounds(intermediate_tuple, component.shape) and component[
                intermediate_tuple
            ]:
                return False
    return True


def _coord_in_bounds(coord: tuple[int, ...], shape: tuple[int, ...]) -> bool:
    return all(0 <= value < shape[index] for index, value in enumerate(coord))


def _offset_length(offset: tuple[int, ...], scales: tuple[float, ...]) -> float:
    values = [
        (float(offset[index]) * float(scales[index])) ** 2
        for index in range(len(offset))
    ]
    return float(np.sqrt(np.sum(values)))


def _skeleton_units(
    spatial_ndim: int,
    axis_scales: Sequence[float | None],
    axis_units: Sequence[str | None],
) -> _SkeletonUnits:
    length_column = (
        "skeleton_length_voxels"
        if spatial_ndim >= 3
        else "skeleton_length_pixels"
    )
    physical_column = "skeleton_length_physical"
    scales = _normalized_spatial_scales(spatial_ndim, axis_scales)
    units = [
        str(value).strip()
        for value in tuple(axis_units)[-spatial_ndim:]
        if value not in {None, ""}
    ]
    calibrated = any(abs(scale - 1.0) > 1e-12 for scale in scales) or bool(units)
    unit_label = _physical_unit_label(
        units,
        1,
        "voxel" if spatial_ndim >= 3 else "pixel",
    )
    column_units = {
        "component_count_in_block": "count",
        "component_voxel_fraction": "fraction",
        "skeleton_voxel_count": "voxels",
        "endpoint_voxel_count": "voxels",
        "junction_voxel_count": "voxels",
        "isolated_node_count": "count",
        "branch_count": "count",
        "graph_node_count": "count",
        "graph_edge_count": "count",
        "voxel_graph_edge_count": "count",
        "cycle_count": "count",
        length_column: "voxels" if spatial_ndim >= 3 else "pixels",
    }
    if calibrated:
        column_units[physical_column] = unit_label
        column_units["physical_unit"] = "text"
    else:
        physical_column = ""
        unit_label = ""
    return _SkeletonUnits(
        length_column=length_column,
        physical_column=physical_column,
        scales=scales,
        unit_label=unit_label,
        column_units=column_units,
    )


def _normalized_spatial_scales(
    spatial_ndim: int,
    axis_scales: Sequence[float | None],
) -> tuple[float, ...]:
    scales = [
        float(value) if value not in {None, ""} else 1.0
        for value in tuple(axis_scales)[-spatial_ndim:]
    ]
    if len(scales) < spatial_ndim:
        scales = [1.0] * (spatial_ndim - len(scales)) + scales
    return tuple(scales)


def _measure_label_block(
    block: np.ndarray,
    spatial_axis_names: tuple[str, ...],
    spatial_ndim: int,
    intensity_block: np.ndarray | None = None,
    axis_scales: Sequence[float | None] = (),
    has_physical_calibration: bool = False,
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
    include_derived_shape_ratios: bool = False,
    include_2d_shape_moments: bool = False,
) -> dict[str, list[object]]:
    properties = _regionprops_measurement_properties(
        spatial_ndim,
        include_shape_descriptors=include_shape_descriptors,
        include_axis_descriptors=include_axis_descriptors,
        include_2d_boundary_descriptors=include_2d_boundary_descriptors,
        include_derived_shape_ratios=include_derived_shape_ratios,
        include_2d_shape_moments=include_2d_shape_moments,
    )
    raw = measure.regionprops_table(block, properties=properties)
    scales = _normalized_spatial_scales(spatial_ndim, axis_scales)
    physical_raw = _calibrated_regionprops_table(
        block,
        spatial_ndim,
        scales,
        include_shape_descriptors=include_shape_descriptors,
        include_axis_descriptors=include_axis_descriptors,
        include_2d_boundary_descriptors=include_2d_boundary_descriptors,
        include_derived_shape_ratios=include_derived_shape_ratios,
        include_2d_shape_moments=include_2d_shape_moments,
    ) if has_physical_calibration else {}
    units = _measurement_units(spatial_ndim, (), ())
    result: dict[str, list[object]] = {
        "label_id": [int(value) for value in raw.get("label", [])],
        units.size_column: [int(value) for value in raw.get("area", [])],
    }
    for axis_index, axis_name in enumerate(spatial_axis_names):
        result[f"centroid_{axis_name}"] = [
            float(value) for value in raw.get(f"centroid-{axis_index}", [])
        ]
    for axis_index, axis_name in enumerate(spatial_axis_names):
        result[f"bbox_{axis_name}_min"] = [
            int(value) for value in raw.get(f"bbox-{axis_index}", [])
        ]
    for axis_index, axis_name in enumerate(spatial_axis_names):
        bbox_index = spatial_ndim + axis_index
        result[f"bbox_{axis_name}_max"] = [
            int(value) for value in raw.get(f"bbox-{bbox_index}", [])
        ]
    result[units.equivalent_diameter_column] = [
        float(value) for value in raw.get("equivalent_diameter_area", [])
    ]
    if has_physical_calibration:
        _add_calibrated_measurements(
            result,
            raw,
            physical_raw,
            spatial_axis_names,
            spatial_ndim,
            scales,
            include_shape_descriptors=include_shape_descriptors,
            include_axis_descriptors=include_axis_descriptors,
            include_2d_boundary_descriptors=include_2d_boundary_descriptors,
            include_derived_shape_ratios=include_derived_shape_ratios,
            include_2d_shape_moments=include_2d_shape_moments,
        )
    result["extent"] = [float(value) for value in raw.get("extent", [])]
    result["euler_number"] = [
        int(value) for value in raw.get("euler_number", [])
    ]
    if include_shape_descriptors:
        _add_shape_descriptor_measurements(result, raw, spatial_ndim)
    if include_axis_descriptors:
        _add_axis_descriptor_measurements(result, raw, spatial_ndim)
    if include_2d_boundary_descriptors and spatial_ndim == 2:
        result["perimeter_pixels"] = _float_column(raw, "perimeter")
        result["perimeter_crofton_pixels"] = _float_column(
            raw,
            "perimeter_crofton",
        )
    if include_derived_shape_ratios and spatial_ndim >= 2:
        _add_derived_shape_ratio_measurements(result, raw, spatial_ndim)
    if include_2d_shape_moments and spatial_ndim == 2:
        _add_2d_shape_moment_measurements(result, raw)
    if intensity_block is not None:
        _add_intensity_measurements(result, block, intensity_block)
    return result


def _regionprops_measurement_properties(
    spatial_ndim: int,
    *,
    include_shape_descriptors: bool,
    include_axis_descriptors: bool,
    include_2d_boundary_descriptors: bool,
    include_derived_shape_ratios: bool,
    include_2d_shape_moments: bool,
) -> tuple[str, ...]:
    properties: list[str] = [
        "label",
        "area",
        "bbox",
        "centroid",
        "equivalent_diameter_area",
        "extent",
        "euler_number",
    ]
    if include_shape_descriptors:
        properties.extend(("area_bbox", "area_filled"))
        if spatial_ndim == 2:
            properties.extend(("area_convex", "solidity", "feret_diameter_max"))
    if include_axis_descriptors and spatial_ndim >= 2:
        properties.extend(
            (
                "axis_major_length",
                "axis_minor_length",
                "inertia_tensor_eigvals",
            )
        )
        if spatial_ndim == 2:
            properties.extend(("eccentricity", "orientation"))
    if include_2d_boundary_descriptors and spatial_ndim == 2:
        properties.extend(("perimeter", "perimeter_crofton"))
    if include_derived_shape_ratios and spatial_ndim >= 2:
        properties.extend(
            (
                "area_bbox",
                "axis_major_length",
                "axis_minor_length",
                "inertia_tensor_eigvals",
            )
        )
    if include_2d_shape_moments and spatial_ndim == 2:
        properties.extend(("moments_hu", "perimeter", "perimeter_crofton"))
    return tuple(dict.fromkeys(properties))


def _calibrated_regionprops_table(
    block: np.ndarray,
    spatial_ndim: int,
    scales: tuple[float, ...],
    *,
    include_shape_descriptors: bool,
    include_axis_descriptors: bool,
    include_2d_boundary_descriptors: bool,
    include_derived_shape_ratios: bool,
    include_2d_shape_moments: bool,
) -> dict[str, np.ndarray]:
    properties: list[str] = [
        "label",
        "area",
        "centroid",
        "equivalent_diameter_area",
    ]
    if include_shape_descriptors:
        properties.extend(("area_bbox", "area_filled"))
        if spatial_ndim == 2:
            properties.extend(("area_convex", "feret_diameter_max"))
    if include_axis_descriptors and spatial_ndim >= 2:
        properties.extend(
            (
                "axis_major_length",
                "axis_minor_length",
                "inertia_tensor_eigvals",
            )
        )
    if (
        spatial_ndim == 2
        and (include_2d_boundary_descriptors or include_2d_shape_moments)
        and _scales_are_isotropic(scales)
    ):
        properties.extend(("perimeter", "perimeter_crofton"))

    unique_properties = tuple(dict.fromkeys(properties))
    if not unique_properties:
        return {}
    try:
        return measure.regionprops_table(
            block,
            properties=unique_properties,
            spacing=scales,
        )
    except NotImplementedError:
        safe_properties = tuple(
            property_name
            for property_name in unique_properties
            if property_name not in {"perimeter", "perimeter_crofton"}
        )
        return measure.regionprops_table(
            block,
            properties=safe_properties,
            spacing=scales,
        )


def _add_calibrated_measurements(
    result: dict[str, list[object]],
    raw: dict[str, np.ndarray],
    physical_raw: dict[str, np.ndarray],
    spatial_axis_names: tuple[str, ...],
    spatial_ndim: int,
    scales: tuple[float, ...],
    *,
    include_shape_descriptors: bool,
    include_axis_descriptors: bool,
    include_2d_boundary_descriptors: bool,
    include_derived_shape_ratios: bool,
    include_2d_shape_moments: bool,
) -> None:
    row_count = len(result.get("label_id", []))
    result["equivalent_diameter_physical"] = _float_or_nan_column(
        physical_raw,
        "equivalent_diameter_area",
        row_count,
    )
    for axis_index, axis_name in enumerate(spatial_axis_names):
        scale = float(scales[axis_index])
        result[f"centroid_{axis_name}_physical"] = _float_or_scaled_column(
            physical_raw,
            f"centroid-{axis_index}",
            raw,
            f"centroid-{axis_index}",
            scale,
            row_count,
        )
        result[f"bbox_{axis_name}_min_physical"] = [
            float(value) * scale for value in _float_column(raw, f"bbox-{axis_index}")
        ]
        bbox_index = spatial_ndim + axis_index
        result[f"bbox_{axis_name}_max_physical"] = [
            float(value) * scale for value in _float_column(raw, f"bbox-{bbox_index}")
        ]
    if include_shape_descriptors:
        result[_physical_size_descriptor_column("bbox", spatial_ndim)] = (
            _float_or_nan_column(physical_raw, "area_bbox", row_count)
        )
        result[_physical_size_descriptor_column("filled", spatial_ndim)] = (
            _float_or_nan_column(physical_raw, "area_filled", row_count)
        )
        if spatial_ndim == 2:
            result["convex_area_physical"] = _float_or_nan_column(
                physical_raw,
                "area_convex",
                row_count,
            )
            result["feret_diameter_max_physical"] = _float_or_nan_column(
                physical_raw,
                "feret_diameter_max",
                row_count,
            )
    if include_axis_descriptors and spatial_ndim >= 2:
        result["major_axis_length_physical"] = _float_or_nan_column(
            physical_raw,
            "axis_major_length",
            row_count,
        )
        result["minor_axis_length_physical"] = _float_or_nan_column(
            physical_raw,
            "axis_minor_length",
            row_count,
        )
        for axis_index in range(spatial_ndim):
            result[f"inertia_tensor_eigval_{axis_index}_physical"] = (
                _float_or_nan_column(
                    physical_raw,
                    f"inertia_tensor_eigvals-{axis_index}",
                    row_count,
                )
            )
    if include_2d_boundary_descriptors and spatial_ndim == 2:
        result["perimeter_physical"] = _float_or_nan_column(
            physical_raw,
            "perimeter",
            row_count,
        )
        result["perimeter_crofton_physical"] = _float_or_nan_column(
            physical_raw,
            "perimeter_crofton",
            row_count,
        )
    if include_derived_shape_ratios and spatial_ndim >= 2:
        for axis_index in range(spatial_ndim):
            scale = float(scales[axis_index])
            result[f"bbox_axis_{axis_index}_length_physical"] = [
                float(maximum - minimum) * scale
                for minimum, maximum in zip(
                    _float_column(raw, f"bbox-{axis_index}"),
                    _float_column(raw, f"bbox-{spatial_ndim + axis_index}"),
                    strict=False,
                )
            ]
    if include_2d_shape_moments and spatial_ndim == 2:
        perimeter = _float_or_nan_column(physical_raw, "perimeter", row_count)
        areas = _float_or_nan_column(physical_raw, "area", row_count)
        result["perimeter_area_ratio_physical"] = _ratio_columns(perimeter, areas)


def _float_or_nan_column(
    raw: dict[str, np.ndarray],
    name: str,
    row_count: int,
) -> list[float]:
    values = _float_column(raw, name)
    if values:
        return values
    return [float("nan")] * int(row_count)


def _float_or_scaled_column(
    primary: dict[str, np.ndarray],
    primary_name: str,
    fallback: dict[str, np.ndarray],
    fallback_name: str,
    scale: float,
    row_count: int,
) -> list[float]:
    values = _float_column(primary, primary_name)
    if values:
        return values
    fallback_values = _float_column(fallback, fallback_name)
    if fallback_values:
        return [float(value) * float(scale) for value in fallback_values]
    return [float("nan")] * int(row_count)


def _scales_are_isotropic(scales: Sequence[float]) -> bool:
    values = [float(value) for value in scales]
    if not values:
        return True
    first = values[0]
    return all(abs(value - first) <= 1e-12 for value in values[1:])


def _add_shape_descriptor_measurements(
    result: dict[str, list[object]],
    raw: dict[str, np.ndarray],
    spatial_ndim: int,
) -> None:
    result[_size_descriptor_column("bbox", spatial_ndim)] = _int_column(
        raw,
        "area_bbox",
    )
    result[_size_descriptor_column("filled", spatial_ndim)] = _int_column(
        raw,
        "area_filled",
    )
    if spatial_ndim == 2:
        result["convex_area_pixels"] = _int_column(raw, "area_convex")
        result["solidity"] = _float_column(raw, "solidity")
        result["feret_diameter_max_pixels"] = _float_column(
            raw,
            "feret_diameter_max",
        )


def _add_axis_descriptor_measurements(
    result: dict[str, list[object]],
    raw: dict[str, np.ndarray],
    spatial_ndim: int,
) -> None:
    if spatial_ndim < 2:
        return
    suffix = "voxels" if spatial_ndim >= 3 else "pixels"
    result[f"major_axis_length_{suffix}"] = _float_column(
        raw,
        "axis_major_length",
    )
    result[f"minor_axis_length_{suffix}"] = _float_column(
        raw,
        "axis_minor_length",
    )
    for axis_index in range(spatial_ndim):
        result[f"inertia_tensor_eigval_{axis_index}"] = _float_column(
            raw,
            f"inertia_tensor_eigvals-{axis_index}",
        )
    if spatial_ndim == 2:
        result["eccentricity"] = _float_column(raw, "eccentricity")
        result["orientation_radians"] = _float_column(raw, "orientation")


def _add_derived_shape_ratio_measurements(
    result: dict[str, list[object]],
    raw: dict[str, np.ndarray],
    spatial_ndim: int,
) -> None:
    suffix = "voxels" if spatial_ndim >= 3 else "pixels"
    axis_major = _float_column(raw, "axis_major_length")
    axis_minor = _float_column(raw, "axis_minor_length")
    result["axis_ratio_major_minor"] = _ratio_columns(axis_major, axis_minor)
    bbox_sizes: list[list[float]] = []
    for axis_index in range(spatial_ndim):
        minimums = _float_column(raw, f"bbox-{axis_index}")
        maximums = _float_column(raw, f"bbox-{spatial_ndim + axis_index}")
        sizes = [
            float(maximum - minimum)
            for minimum, maximum in zip(minimums, maximums, strict=False)
        ]
        bbox_sizes.append(sizes)
        result[f"bbox_axis_{axis_index}_length_{suffix}"] = sizes
    for left in range(spatial_ndim):
        for right in range(left + 1, spatial_ndim):
            result[f"bbox_axis_ratio_{left}_{right}"] = _ratio_columns(
                bbox_sizes[left],
                bbox_sizes[right],
            )
    size_column = _measurement_units(spatial_ndim, (), ()).size_column
    result["bbox_fill_fraction"] = _ratio_columns(
        [float(value) for value in result.get(size_column, [])],
        _float_column(raw, "area_bbox"),
    )
    for left in range(spatial_ndim):
        for right in range(left + 1, spatial_ndim):
            result[f"inertia_eigval_ratio_{left}_{right}"] = _ratio_columns(
                _float_column(raw, f"inertia_tensor_eigvals-{left}"),
                _float_column(raw, f"inertia_tensor_eigvals-{right}"),
            )


def _add_2d_shape_moment_measurements(
    result: dict[str, list[object]],
    raw: dict[str, np.ndarray],
) -> None:
    areas = [float(value) for value in result.get("area_pixels", [])]
    perimeters = _float_column(raw, "perimeter")
    circularity_perimeters = _float_column(raw, "perimeter_crofton")
    if not circularity_perimeters:
        circularity_perimeters = perimeters
    result["circularity"] = [
        _safe_ratio(4.0 * np.pi * area, perimeter * perimeter)
        for area, perimeter in zip(areas, circularity_perimeters, strict=False)
    ]
    result["perimeter_area_ratio"] = _ratio_columns(perimeters, areas)
    for index in range(7):
        result[f"hu_moment_{index}"] = _float_column(raw, f"moments_hu-{index}")


def _ratio_columns(
    numerators: Sequence[float],
    denominators: Sequence[float],
) -> list[float]:
    return [
        _safe_ratio(float(numerator), float(denominator))
        for numerator, denominator in zip(numerators, denominators, strict=False)
    ]


def _size_descriptor_column(prefix: str, spatial_ndim: int) -> str:
    if spatial_ndim >= 3:
        return f"{prefix}_volume_voxels"
    if spatial_ndim == 2:
        return f"{prefix}_area_pixels"
    return f"{prefix}_length_pixels"


def _physical_size_descriptor_column(prefix: str, spatial_ndim: int) -> str:
    if spatial_ndim >= 3:
        return f"{prefix}_volume_physical"
    if spatial_ndim == 2:
        return f"{prefix}_area_physical"
    return f"{prefix}_length_physical"


def _float_column(raw: dict[str, np.ndarray], name: str) -> list[float]:
    return [float(value) for value in raw.get(name, [])]


def _int_column(raw: dict[str, np.ndarray], name: str) -> list[int]:
    return [int(round(float(value))) for value in raw.get(name, [])]


def _add_intensity_measurements(
    result: dict[str, list[object]],
    labels: np.ndarray,
    intensity: np.ndarray,
) -> None:
    means: list[float] = []
    minimums: list[float] = []
    maximums: list[float] = []
    sums: list[float] = []
    stds: list[float] = []
    intensity = np.asarray(intensity)
    for label_id in result["label_id"]:
        values = intensity[labels == int(label_id)].astype(np.float64, copy=False)
        if values.size == 0:
            means.append(float("nan"))
            minimums.append(float("nan"))
            maximums.append(float("nan"))
            sums.append(0.0)
            stds.append(float("nan"))
            continue
        means.append(float(np.mean(values)))
        minimums.append(float(np.min(values)))
        maximums.append(float(np.max(values)))
        sums.append(float(np.sum(values)))
        stds.append(float(np.std(values)))
    result["intensity_mean"] = means
    result["intensity_min"] = minimums
    result["intensity_max"] = maximums
    result["intensity_sum"] = sums
    result["intensity_std"] = stds


def _mesh_units(
    axis_scales: Sequence[float | None],
    axis_units: Sequence[str | None],
    spatial_axis_names: tuple[str, ...],
    *,
    include_convex_hull_metrics: bool,
) -> _MeshUnits:
    scales = _normalized_spatial_scales(3, axis_scales)
    units = [
        str(value).strip()
        for value in tuple(axis_units)[-3:]
        if value not in {None, ""}
    ]
    length_unit = _physical_unit_label(units, 1, "voxel")
    area_unit = _physical_unit_label(units, 2, "voxel^2")
    volume_unit = _physical_unit_label(units, 3, "voxel^3")
    column_units = {
        "label_id": "label",
        "mesh_status": "text",
        "mesh_error": "text",
        "voxel_count": "voxels",
        "voxel_volume_physical": volume_unit,
        "mesh_volume_physical": volume_unit,
        "mesh_surface_area_physical": area_unit,
        "surface_area_to_volume": f"1/{length_unit}",
        "equivalent_sphere_radius_physical": length_unit,
        "equivalent_sphere_diameter_physical": length_unit,
        "sphericity": "ratio",
        "physical_unit": "text",
    }
    for axis_name in spatial_axis_names:
        column_units[f"mesh_extent_{axis_name}_physical"] = length_unit
    for left, right in combinations(spatial_axis_names, 2):
        column_units[f"mesh_extent_ratio_{left}_{right}"] = "ratio"
    if include_convex_hull_metrics:
        column_units.update(
            {
                "convex_hull_volume_physical": volume_unit,
                "convex_hull_surface_area_physical": area_unit,
                "solidity_3d": "ratio",
                "surface_area_to_convex_hull_area": "ratio",
            }
        )
    return _MeshUnits(
        scales=tuple(float(scale) for scale in scales),
        length_unit=length_unit,
        area_unit=area_unit,
        volume_unit=volume_unit,
        voxel_volume=float(np.prod(scales)),
        column_units=column_units,
    )


def _mesh_morphology_empty_columns(
    leading_axis_names: tuple[str, ...],
    spatial_axis_names: tuple[str, ...],
    *,
    include_convex_hull_metrics: bool,
) -> dict[str, list[object]]:
    columns: dict[str, list[object]] = {}
    for axis_name in leading_axis_names:
        columns[f"{axis_name}_index"] = []
    columns["label_id"] = []
    columns["mesh_status"] = []
    columns["mesh_error"] = []
    columns["voxel_count"] = []
    columns["voxel_volume_physical"] = []
    columns["mesh_volume_physical"] = []
    columns["mesh_surface_area_physical"] = []
    columns["surface_area_to_volume"] = []
    columns["equivalent_sphere_radius_physical"] = []
    columns["equivalent_sphere_diameter_physical"] = []
    columns["sphericity"] = []
    for axis_name in spatial_axis_names:
        columns[f"mesh_extent_{axis_name}_physical"] = []
    for left, right in combinations(spatial_axis_names, 2):
        columns[f"mesh_extent_ratio_{left}_{right}"] = []
    if include_convex_hull_metrics:
        columns["convex_hull_volume_physical"] = []
        columns["convex_hull_surface_area_physical"] = []
        columns["solidity_3d"] = []
        columns["surface_area_to_convex_hull_area"] = []
    columns["physical_unit"] = []
    return columns


def _measure_3d_mesh_morphology_block(
    block: np.ndarray,
    spatial_axis_names: tuple[str, ...],
    units: _MeshUnits,
    *,
    minimum_voxel_count: int,
    include_convex_hull_metrics: bool,
    label_ids: Sequence[int] | None = None,
    progress=None,
    progress_start: int = 0,
    progress_total: int | None = None,
) -> dict[str, list[object]]:
    block = np.asarray(block)
    columns = _mesh_morphology_empty_columns(
        (),
        spatial_axis_names,
        include_convex_hull_metrics=include_convex_hull_metrics,
    )
    labels = list(label_ids) if label_ids is not None else _positive_label_ids(block)
    total = int(progress_total or max(len(labels), 1))

    def report_label_done(offset: int, label_id: int) -> None:
        if progress is not None:
            progress.report(
                progress_start + offset,
                total,
                f"Measured label {int(label_id)}",
            )

    for offset, label_id in enumerate(labels, start=1):
        if progress is not None:
            progress.check_cancelled()
        mask = block == int(label_id)
        voxel_count = int(np.count_nonzero(mask))
        base_values = {
            "label_id": int(label_id),
            "voxel_count": voxel_count,
            "voxel_volume_physical": float(voxel_count) * units.voxel_volume,
        }
        if voxel_count < int(minimum_voxel_count):
            _append_mesh_row(
                columns,
                spatial_axis_names,
                include_convex_hull_metrics=include_convex_hull_metrics,
                **base_values,
                mesh_status="skipped_too_few_voxels",
                mesh_error=(
                    f"voxel_count {voxel_count} is below minimum "
                    f"{int(minimum_voxel_count)}"
                ),
            )
            report_label_done(offset, label_id)
            continue
        try:
            metrics = _mesh_metrics_for_label_mask(mask, spatial_axis_names, units)
        except Exception as exc:
            _append_mesh_row(
                columns,
                spatial_axis_names,
                include_convex_hull_metrics=include_convex_hull_metrics,
                **base_values,
                mesh_status="mesh_failed",
                mesh_error=str(exc),
            )
            report_label_done(offset, label_id)
            continue
        if include_convex_hull_metrics:
            try:
                metrics.update(_convex_hull_metrics(metrics["mesh_vertices"]))
                metrics["solidity_3d"] = _nan_ratio(
                    metrics["mesh_volume_physical"],
                    metrics["convex_hull_volume_physical"],
                )
                metrics["surface_area_to_convex_hull_area"] = _nan_ratio(
                    metrics["mesh_surface_area_physical"],
                    metrics["convex_hull_surface_area_physical"],
                )
                mesh_status = "ok"
                mesh_error = ""
            except (QhullError, ValueError) as exc:
                mesh_status = "partial_convex_hull_failed"
                mesh_error = str(exc)
        else:
            mesh_status = "ok"
            mesh_error = ""
        metrics.pop("mesh_vertices", None)
        _append_mesh_row(
            columns,
            spatial_axis_names,
            include_convex_hull_metrics=include_convex_hull_metrics,
            **base_values,
            **metrics,
            mesh_status=mesh_status,
            mesh_error=mesh_error,
        )
        report_label_done(offset, label_id)
    return columns


def _positive_label_ids(block: np.ndarray) -> list[int]:
    values = np.unique(block)
    return [int(value) for value in values if int(value) > 0]


def _mesh_metrics_for_label_mask(
    mask: np.ndarray,
    spatial_axis_names: tuple[str, ...],
    units: _MeshUnits,
) -> dict[str, object]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("label has no voxels")
    minimum = coords.min(axis=0)
    maximum = coords.max(axis=0) + 1
    slices = tuple(
        slice(int(lo), int(hi))
        for lo, hi in zip(minimum, maximum, strict=True)
    )
    local = mask[slices]
    padded = np.pad(local.astype(np.float32, copy=False), 1, mode="constant")
    vertices, faces, _normals, _values = measure.marching_cubes(
        padded,
        level=0.5,
        spacing=units.scales,
    )
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("marching cubes produced an empty mesh")
    surface_area = float(measure.mesh_surface_area(vertices, faces))
    mesh_volume = abs(_signed_triangle_mesh_volume(vertices, faces))
    extents = np.ptp(vertices, axis=0)
    metrics: dict[str, object] = {
        "mesh_vertices": vertices,
        "mesh_volume_physical": mesh_volume,
        "mesh_surface_area_physical": surface_area,
        "surface_area_to_volume": _nan_ratio(surface_area, mesh_volume),
        "equivalent_sphere_radius_physical": _equivalent_sphere_radius(mesh_volume),
        "equivalent_sphere_diameter_physical": 2.0
        * _equivalent_sphere_radius(mesh_volume),
        "sphericity": _sphericity(mesh_volume, surface_area),
    }
    for axis_name, extent in zip(spatial_axis_names, extents, strict=True):
        metrics[f"mesh_extent_{axis_name}_physical"] = float(extent)
    for left_index, right_index in combinations(range(len(spatial_axis_names)), 2):
        left = spatial_axis_names[left_index]
        right = spatial_axis_names[right_index]
        metrics[f"mesh_extent_ratio_{left}_{right}"] = _nan_ratio(
            float(extents[left_index]),
            float(extents[right_index]),
        )
    return metrics


def _signed_triangle_mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=int)]
    if triangles.size == 0:
        return 0.0
    cross = np.cross(triangles[:, 1], triangles[:, 2])
    return float(np.einsum("ij,ij->", triangles[:, 0], cross) / 6.0)


def _convex_hull_metrics(vertices: np.ndarray) -> dict[str, float]:
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape[0] < 4:
        raise ValueError("at least four mesh vertices are required for a 3D hull")
    hull = ConvexHull(points)
    return {
        "convex_hull_volume_physical": float(hull.volume),
        "convex_hull_surface_area_physical": float(hull.area),
    }


def _append_mesh_row(
    columns: dict[str, list[object]],
    spatial_axis_names: tuple[str, ...],
    *,
    include_convex_hull_metrics: bool,
    label_id: int,
    mesh_status: str,
    mesh_error: str,
    voxel_count: int,
    voxel_volume_physical: float,
    mesh_volume_physical: float = float("nan"),
    mesh_surface_area_physical: float = float("nan"),
    surface_area_to_volume: float = float("nan"),
    equivalent_sphere_radius_physical: float = float("nan"),
    equivalent_sphere_diameter_physical: float = float("nan"),
    sphericity: float = float("nan"),
    convex_hull_volume_physical: float = float("nan"),
    convex_hull_surface_area_physical: float = float("nan"),
    solidity_3d: float = float("nan"),
    surface_area_to_convex_hull_area: float = float("nan"),
    **metrics,
) -> None:
    columns["label_id"].append(int(label_id))
    columns["mesh_status"].append(str(mesh_status))
    columns["mesh_error"].append(str(mesh_error))
    columns["voxel_count"].append(int(voxel_count))
    columns["voxel_volume_physical"].append(float(voxel_volume_physical))
    columns["mesh_volume_physical"].append(float(mesh_volume_physical))
    columns["mesh_surface_area_physical"].append(float(mesh_surface_area_physical))
    columns["surface_area_to_volume"].append(float(surface_area_to_volume))
    columns["equivalent_sphere_radius_physical"].append(
        float(equivalent_sphere_radius_physical)
    )
    columns["equivalent_sphere_diameter_physical"].append(
        float(equivalent_sphere_diameter_physical)
    )
    columns["sphericity"].append(float(sphericity))
    for axis_name in spatial_axis_names:
        columns[f"mesh_extent_{axis_name}_physical"].append(
            float(metrics.get(f"mesh_extent_{axis_name}_physical", float("nan")))
        )
    for left, right in combinations(spatial_axis_names, 2):
        columns[f"mesh_extent_ratio_{left}_{right}"].append(
            float(metrics.get(f"mesh_extent_ratio_{left}_{right}", float("nan")))
        )
    if include_convex_hull_metrics:
        columns["convex_hull_volume_physical"].append(
            float(convex_hull_volume_physical)
        )
        columns["convex_hull_surface_area_physical"].append(
            float(convex_hull_surface_area_physical)
        )
        columns["solidity_3d"].append(float(solidity_3d))
        columns["surface_area_to_convex_hull_area"].append(
            float(surface_area_to_convex_hull_area)
        )


def _nan_ratio(numerator: float, denominator: float) -> float:
    denominator = float(denominator)
    if denominator <= 0 or not np.isfinite(denominator):
        return float("nan")
    return float(numerator) / denominator


def _equivalent_sphere_radius(volume: float) -> float:
    volume = float(volume)
    if volume <= 0 or not np.isfinite(volume):
        return float("nan")
    return float(((3.0 * volume) / (4.0 * np.pi)) ** (1.0 / 3.0))


def _sphericity(volume: float, surface_area: float) -> float:
    volume = float(volume)
    surface_area = float(surface_area)
    if volume <= 0 or surface_area <= 0:
        return float("nan")
    return float(
        (np.pi ** (1.0 / 3.0))
        * ((6.0 * volume) ** (2.0 / 3.0))
        / surface_area
    )


def _measurement_empty_columns(
    leading_axis_names: tuple[str, ...],
    spatial_axis_names: tuple[str, ...],
    units: _MeasurementUnits,
    *,
    include_intensity: bool = False,
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
    include_derived_shape_ratios: bool = False,
    include_2d_shape_moments: bool = False,
    spatial_ndim: int = 2,
) -> dict[str, list[object]]:
    columns: dict[str, list[object]] = {}
    for axis_name in leading_axis_names:
        columns[f"{axis_name}_index"] = []
    columns["label_id"] = []
    columns[units.size_column] = []
    if units.physical_column:
        columns[units.physical_column] = []
        columns["physical_unit"] = []
    for axis_name in spatial_axis_names:
        columns[f"centroid_{axis_name}"] = []
    for axis_name in spatial_axis_names:
        columns[f"bbox_{axis_name}_min"] = []
    for axis_name in spatial_axis_names:
        columns[f"bbox_{axis_name}_max"] = []
    columns[units.equivalent_diameter_column] = []
    if units.physical_column:
        columns["equivalent_diameter_physical"] = []
        units.column_units["equivalent_diameter_physical"] = units.length_unit
        for axis_name in spatial_axis_names:
            columns[f"centroid_{axis_name}_physical"] = []
            columns[f"bbox_{axis_name}_min_physical"] = []
            columns[f"bbox_{axis_name}_max_physical"] = []
            units.column_units[f"centroid_{axis_name}_physical"] = units.length_unit
            units.column_units[f"bbox_{axis_name}_min_physical"] = units.length_unit
            units.column_units[f"bbox_{axis_name}_max_physical"] = units.length_unit
    columns["extent"] = []
    columns["euler_number"] = []
    if include_shape_descriptors:
        columns[_size_descriptor_column("bbox", spatial_ndim)] = []
        columns[_size_descriptor_column("filled", spatial_ndim)] = []
        if units.physical_column:
            columns[_physical_size_descriptor_column("bbox", spatial_ndim)] = []
            columns[_physical_size_descriptor_column("filled", spatial_ndim)] = []
            units.column_units[
                _physical_size_descriptor_column("bbox", spatial_ndim)
            ] = units.unit_label
            units.column_units[
                _physical_size_descriptor_column("filled", spatial_ndim)
            ] = units.unit_label
        if spatial_ndim == 2:
            columns["convex_area_pixels"] = []
            columns["solidity"] = []
            columns["feret_diameter_max_pixels"] = []
            if units.physical_column:
                columns["convex_area_physical"] = []
                columns["feret_diameter_max_physical"] = []
                units.column_units["convex_area_physical"] = units.unit_label
                units.column_units["feret_diameter_max_physical"] = units.length_unit
    if include_axis_descriptors and spatial_ndim >= 2:
        suffix = "voxels" if spatial_ndim >= 3 else "pixels"
        columns[f"major_axis_length_{suffix}"] = []
        columns[f"minor_axis_length_{suffix}"] = []
        if units.physical_column:
            columns["major_axis_length_physical"] = []
            columns["minor_axis_length_physical"] = []
            units.column_units["major_axis_length_physical"] = units.length_unit
            units.column_units["minor_axis_length_physical"] = units.length_unit
        for axis_index in range(spatial_ndim):
            columns[f"inertia_tensor_eigval_{axis_index}"] = []
            if units.physical_column:
                columns[f"inertia_tensor_eigval_{axis_index}_physical"] = []
                units.column_units[
                    f"inertia_tensor_eigval_{axis_index}_physical"
                ] = units.area_unit
        if spatial_ndim == 2:
            columns["eccentricity"] = []
            columns["orientation_radians"] = []
    if include_2d_boundary_descriptors and spatial_ndim == 2:
        columns["perimeter_pixels"] = []
        columns["perimeter_crofton_pixels"] = []
        if units.physical_column:
            columns["perimeter_physical"] = []
            columns["perimeter_crofton_physical"] = []
            units.column_units["perimeter_physical"] = units.length_unit
            units.column_units["perimeter_crofton_physical"] = units.length_unit
    if include_derived_shape_ratios and spatial_ndim >= 2:
        suffix = "voxels" if spatial_ndim >= 3 else "pixels"
        columns["axis_ratio_major_minor"] = []
        for axis_index in range(spatial_ndim):
            columns[f"bbox_axis_{axis_index}_length_{suffix}"] = []
            if units.physical_column:
                columns[f"bbox_axis_{axis_index}_length_physical"] = []
                units.column_units[
                    f"bbox_axis_{axis_index}_length_physical"
                ] = units.length_unit
        for left in range(spatial_ndim):
            for right in range(left + 1, spatial_ndim):
                columns[f"bbox_axis_ratio_{left}_{right}"] = []
        columns["bbox_fill_fraction"] = []
        for left in range(spatial_ndim):
            for right in range(left + 1, spatial_ndim):
                columns[f"inertia_eigval_ratio_{left}_{right}"] = []
    if include_2d_shape_moments and spatial_ndim == 2:
        columns["circularity"] = []
        columns["perimeter_area_ratio"] = []
        if units.physical_column:
            columns["perimeter_area_ratio_physical"] = []
            units.column_units[
                "perimeter_area_ratio_physical"
            ] = f"1/{units.length_unit}"
        for index in range(7):
            columns[f"hu_moment_{index}"] = []
    if include_intensity:
        columns["intensity_mean"] = []
        columns["intensity_min"] = []
        columns["intensity_max"] = []
        columns["intensity_sum"] = []
        columns["intensity_std"] = []
    return columns


def _measurement_units(
    spatial_ndim: int,
    axis_scales: Sequence[float | None],
    axis_units: Sequence[str | None],
) -> _MeasurementUnits:
    if spatial_ndim >= 3:
        size_column = "volume_voxels"
        equivalent_column = "equivalent_diameter_voxels"
        physical_column = "volume_physical"
        default_unit = "voxel^3"
    elif spatial_ndim == 2:
        size_column = "area_pixels"
        equivalent_column = "equivalent_diameter_pixels"
        physical_column = "area_physical"
        default_unit = "pixel^2"
    else:
        size_column = "length_pixels"
        equivalent_column = "equivalent_diameter_pixels"
        physical_column = "length_physical"
        default_unit = "pixel"

    scales = [
        float(value) if value not in {None, ""} else 1.0
        for value in tuple(axis_scales)[-spatial_ndim:]
    ]
    if len(scales) < spatial_ndim:
        scales = [1.0] * (spatial_ndim - len(scales)) + scales
    units = [
        str(value).strip()
        for value in tuple(axis_units)[-spatial_ndim:]
        if value not in {None, ""}
    ]
    calibrated = any(abs(scale - 1.0) > 1e-12 for scale in scales) or bool(units)
    scale_product = float(np.prod(scales)) if scales else 1.0
    length_default = "voxel" if spatial_ndim >= 3 else "pixel"
    area_default = "voxel^2" if spatial_ndim >= 3 else "pixel^2"
    length_unit = _physical_unit_label(units, 1, length_default)
    area_unit = _physical_unit_label(units, 2, area_default)
    unit_label = _physical_unit_label(units, spatial_ndim, default_unit)
    column_units = {
        size_column: "voxels" if spatial_ndim >= 3 else "pixels",
        equivalent_column: "voxels" if spatial_ndim >= 3 else "pixels",
        _size_descriptor_column("bbox", spatial_ndim): (
            "voxels" if spatial_ndim >= 3 else "pixels"
        ),
        _size_descriptor_column("filled", spatial_ndim): (
            "voxels" if spatial_ndim >= 3 else "pixels"
        ),
        "intensity_mean": "intensity",
        "intensity_min": "intensity",
        "intensity_max": "intensity",
        "intensity_sum": "intensity",
        "intensity_std": "intensity",
    }
    if spatial_ndim == 2:
        column_units.update(
            {
                "convex_area_pixels": "pixels",
                "feret_diameter_max_pixels": "pixels",
                "perimeter_pixels": "pixels",
                "perimeter_crofton_pixels": "pixels",
            }
        )
    if spatial_ndim >= 2:
        suffix = "voxels" if spatial_ndim >= 3 else "pixels"
        discrete_length_unit = "voxels" if spatial_ndim >= 3 else "pixels"
        squared_unit = "voxels^2" if spatial_ndim >= 3 else "pixels^2"
        column_units[f"major_axis_length_{suffix}"] = discrete_length_unit
        column_units[f"minor_axis_length_{suffix}"] = discrete_length_unit
        column_units["axis_ratio_major_minor"] = "ratio"
        for axis_index in range(spatial_ndim):
            column_units[f"bbox_axis_{axis_index}_length_{suffix}"] = (
                discrete_length_unit
            )
        for axis_index in range(spatial_ndim):
            column_units[f"inertia_tensor_eigval_{axis_index}"] = squared_unit
        for left in range(spatial_ndim):
            for right in range(left + 1, spatial_ndim):
                column_units[f"bbox_axis_ratio_{left}_{right}"] = "ratio"
                column_units[f"inertia_eigval_ratio_{left}_{right}"] = "ratio"
        column_units["bbox_fill_fraction"] = "fraction"
    column_units["solidity"] = "ratio"
    column_units["eccentricity"] = "ratio"
    column_units["orientation_radians"] = "radians"
    column_units["circularity"] = "ratio"
    column_units["perimeter_area_ratio"] = "1/pixel"
    for index in range(7):
        column_units[f"hu_moment_{index}"] = "dimensionless"
    if calibrated:
        column_units[physical_column] = unit_label
        column_units["physical_unit"] = "text"
    else:
        physical_column = ""
        unit_label = ""
    return _MeasurementUnits(
        size_column=size_column,
        equivalent_diameter_column=equivalent_column,
        physical_column=physical_column,
        scale_product=scale_product,
        scales=tuple(float(scale) for scale in scales),
        length_unit=length_unit,
        area_unit=area_unit,
        unit_label=unit_label,
        column_units=column_units,
    )


def _labels_and_intensity_inputs(inputs) -> tuple[object, object]:
    values = list(inputs)
    if len(values) < 2:
        raise ValueError(
            "Intensity-aware measurements require labels and intensity image inputs."
        )
    return values[0], values[1]


def _measurement_table_kind(
    base: str,
    *,
    include_shape_descriptors: bool,
    include_axis_descriptors: bool,
    include_2d_boundary_descriptors: bool,
    include_derived_shape_ratios: bool,
    include_2d_shape_moments: bool,
    spatial_ndim: int,
) -> str:
    extras: list[str] = []
    if include_shape_descriptors:
        extras.append("shape descriptors")
    if include_axis_descriptors:
        extras.append("axis/inertia descriptors")
    if include_2d_boundary_descriptors and spatial_ndim == 2:
        extras.append("2D boundary descriptors")
    if include_derived_shape_ratios and spatial_ndim >= 2:
        extras.append("derived shape ratios")
    if include_2d_shape_moments and spatial_ndim == 2:
        extras.append("2D shape moments")
    return f"{base} + {', '.join(extras)}" if extras else base


def _physical_unit_label(
    units: Sequence[str],
    spatial_ndim: int,
    default_unit: str,
) -> str:
    if not units:
        return default_unit
    if len(set(units)) == 1:
        unit = units[0]
        return unit if spatial_ndim == 1 else f"{unit}^{spatial_ndim}"
    return "*".join(units)


def _measurement_axis_names(
    ndim: int,
    axis_names: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if axis_names is not None and len(axis_names) == ndim:
        return tuple(
            str(name).strip().lower() or f"axis_{index}"
            for index, name in enumerate(axis_names)
        )
    return tuple(f"axis_{index}" for index in range(ndim))


def _measurement_axis_types(
    ndim: int,
    axis_types: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if axis_types is not None and len(axis_types) == ndim:
        return tuple(str(axis_type).strip().lower() for axis_type in axis_types)
    return tuple("space" if index >= ndim - 2 else "unknown" for index in range(ndim))


def _measurement_spatial_axes(
    ndim: int,
    spatial_ndim: int,
    axis_names: tuple[str, ...],
    axis_types: tuple[str, ...],
) -> tuple[int, ...]:
    desired_names = ("z", "y", "x")[-spatial_ndim:]
    if all(axis_names.count(name) == 1 for name in desired_names):
        return tuple(axis_names.index(name) for name in desired_names)
    spatial = tuple(
        index
        for index, axis_type in enumerate(axis_types)
        if axis_type == "space"
    )
    if len(spatial) >= spatial_ndim:
        return spatial[-spatial_ndim:]
    return tuple(range(ndim - spatial_ndim, ndim))


def _reordered_axis_values(
    values: Sequence | None,
    ndim: int,
    spatial_axes: tuple[int, ...],
) -> tuple:
    if values is None or len(tuple(values)) != ndim:
        values = tuple(None for _ in range(ndim))
    values = tuple(values)
    leading = tuple(values[index] for index in range(ndim) if index not in spatial_axes)
    spatial = tuple(values[index] for index in spatial_axes)
    return leading + spatial


def _safe_axis_column_names(
    names: tuple[str, ...],
    fallback: tuple[str, ...],
) -> tuple[str, ...]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(names):
        fallback_name = fallback[index] if index < len(fallback) else f"axis_{index}"
        candidate = _safe_column_fragment(name or fallback_name)
        if not candidate:
            candidate = _safe_column_fragment(fallback_name)
        if candidate in seen:
            candidate = f"{candidate}_{index}"
        seen.add(candidate)
        cleaned.append(candidate)
    return tuple(cleaned)


def _safe_column_fragment(value: str) -> str:
    text = str(value).strip().lower()
    chars = [character if character.isalnum() else "_" for character in text]
    compact = "_".join(part for part in "".join(chars).split("_") if part)
    return compact


def _resolved_psf_spatial_ndim(
    shape: Sequence[int],
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
    axis_types: Sequence[str],
) -> int:
    # ``resolve_born_wolf_psf_parameters`` is also used to render the editor
    # before its image input has been connected.  An empty shape therefore
    # means "no dimensional context yet", not a one-dimensional image.  Real
    # operation calls always provide an array shape and remain subject to the
    # dimensionality check below.
    ndim = len(tuple(shape))
    requested = _spatial_mode_dimension(spatial_mode)
    if requested is None and resolved_spatial_ndim is not None:
        requested = _validated_resolved_spatial_ndim(resolved_spatial_ndim)
    if requested is None:
        spatial_count = sum(
            str(axis_type).strip().lower() == "space"
            for axis_type in axis_types
        )
        if spatial_count:
            requested = 3 if spatial_count >= 3 else 2
        elif 0 < ndim <= 2:
            requested = 2
        elif ndim == 0:
            requested = 2
        else:
            raise ValueError(
                "Auto from axes requires explicit axis semantics. Supply "
                "axis_types, resolved_spatial_ndim, or an explicit 2D/3D mode."
            )
    if requested not in {2, 3}:
        raise ValueError("Born-Wolf PSF spatial dimensionality must be 2 or 3.")
    if ndim and requested > ndim:
        raise ValueError(
            f"{requested}D PSF generation cannot be derived from a {ndim}D shape."
        )
    return requested


def resolve_born_wolf_psf_parameters(
    shape: Sequence[int],
    spatial_mode: str = "Auto from axes",
    *,
    auto_parameters: bool = True,
    wavelength_nm: float = 0.0,
    numerical_aperture: float = 0.0,
    refractive_index: float = 0.0,
    pixel_size_xy_um: float = 0.0,
    z_step_um: float = 0.0,
    channel: int = -1,
    resolved_spatial_ndim: int | None = None,
    axis_types: Sequence[str] = (),
    axis_names: Sequence[str] = (),
    axis_scales: Sequence[float] = (),
    axis_units: Sequence[str | None] = (),
    channel_emission_wavelengths: Sequence[float | None] = (),
    channel_emission_wavelength_units: Sequence[str | None] = (),
    channel_excitation_wavelengths: Sequence[float | None] = (),
    channel_excitation_wavelength_units: Sequence[str | None] = (),
    objective_lens_na: float | None = None,
    objective_refractive_index: float | None = None,
) -> BornWolfPsfResolution:
    """Resolve Born-Wolf PSF auto/manual microscope parameters."""
    spatial_ndim = _resolved_psf_spatial_ndim(
        shape,
        spatial_mode,
        resolved_spatial_ndim,
        axis_types,
    )
    auto = bool(auto_parameters)
    parameters: dict[str, BornWolfPsfParameterResolution] = {}
    values: dict[str, float | int] = {}

    def add(
        name: str,
        value: float | int | None,
        source: str,
        message: str,
        *,
        required: bool = True,
    ) -> None:
        parameters[name] = BornWolfPsfParameterResolution(
            name,
            value,
            source,
            message,
            required,
        )
        if value is not None:
            values[name] = value

    channel_index = _psf_channel_index(
        channel,
        channel_emission_wavelengths,
        channel_excitation_wavelengths,
    )
    if int(channel) >= 0:
        add("channel", int(channel_index), "manual", "Manual channel index.")
    else:
        add(
            "channel",
            int(channel_index),
            "metadata" if _psf_channel_count(
                channel_emission_wavelengths,
                channel_excitation_wavelengths,
            )
            else "auto",
            "Auto channel index.",
            required=False,
        )

    wavelength_value, wavelength_source = _resolve_psf_wavelength_nm_optional(
        wavelength_nm,
        channel_index,
        channel_emission_wavelengths,
        channel_emission_wavelength_units,
        channel_excitation_wavelengths,
        channel_excitation_wavelength_units,
        auto=auto,
    )
    add(
        "wavelength_nm",
        wavelength_value,
        wavelength_source,
        (
            "Emission or excitation wavelength from channel metadata."
            if wavelength_value is not None and wavelength_source == "metadata"
            else "Manual emission wavelength."
            if wavelength_value is not None
            else "Missing emission/excitation wavelength metadata."
        ),
    )

    na_value, na_source = _resolve_psf_positive_optional(
        numerical_aperture,
        objective_lens_na,
        auto=auto,
    )
    add(
        "numerical_aperture",
        na_value,
        na_source,
        (
            "Objective numerical aperture from metadata."
            if na_value is not None and na_source == "metadata"
            else "Manual numerical aperture."
            if na_value is not None
            else "Missing objective numerical aperture metadata."
        ),
    )

    ri_value, ri_source = _resolve_psf_positive_optional(
        refractive_index,
        objective_refractive_index,
        auto=auto,
    )
    add(
        "refractive_index",
        ri_value,
        ri_source,
        (
            "Objective immersion refractive index from metadata."
            if ri_value is not None and ri_source == "metadata"
            else "Manual refractive index."
            if ri_value is not None
            else "Missing objective refractive index metadata."
        ),
    )

    xy_metadata = _metadata_xy_pixel_size_um_optional(
        axis_names,
        axis_types,
        axis_scales,
        axis_units,
    )
    xy_value, xy_source = _resolve_psf_positive_optional(
        pixel_size_xy_um,
        xy_metadata,
        auto=auto,
    )
    add(
        "pixel_size_xy_um",
        xy_value,
        xy_source,
        (
            "Mean X/Y pixel size from axis metadata."
            if xy_value is not None and xy_source == "metadata"
            else "Manual X/Y pixel size."
            if xy_value is not None
            else "Missing X/Y pixel-size metadata."
        ),
    )

    z_metadata = _metadata_axis_size_um(
        "z",
        axis_names,
        axis_types,
        axis_scales,
        axis_units,
    )
    z_value, z_source = _resolve_psf_positive_optional(
        z_step_um,
        z_metadata,
        auto=auto,
    )
    z_required = spatial_ndim >= 3
    if not z_required and z_value is None:
        z_value = 0.0
        z_source = "not used"
    add(
        "z_step_um",
        z_value,
        z_source,
        (
            "Z step from axis metadata."
            if z_value is not None and z_source == "metadata"
            else "Manual Z step."
            if z_value is not None
            else "Z step is not used for a 2D PSF."
            if not z_required
            else "Missing Z-step metadata."
        ),
        required=z_required,
    )

    unresolved = tuple(
        name
        for name, result in parameters.items()
        if result.required and not result.resolved
    )
    return BornWolfPsfResolution(spatial_ndim, values, parameters, unresolved)


def _psf_channel_index(
    channel: int,
    emission_wavelengths: Sequence[float | None],
    excitation_wavelengths: Sequence[float | None],
) -> int:
    count = _psf_channel_count(emission_wavelengths, excitation_wavelengths)
    if count <= 0:
        return 0
    try:
        index = int(channel)
    except Exception:
        index = -1
    if index < 0:
        return 0
    return int(np.clip(index, 0, count - 1))


def _psf_channel_count(
    emission_wavelengths: Sequence[float | None],
    excitation_wavelengths: Sequence[float | None],
) -> int:
    return max(len(tuple(emission_wavelengths)), len(tuple(excitation_wavelengths)))


def _resolve_psf_positive_optional(
    requested,
    metadata_value,
    *,
    auto: bool,
) -> tuple[float | None, str]:
    requested_value = _positive_float(requested, 0.0)
    if requested_value > 0:
        return requested_value, "manual"
    if auto:
        metadata_positive = _positive_float(metadata_value, 0.0)
        if metadata_positive > 0:
            return metadata_positive, "metadata"
    return None, "unresolved"


def _resolve_psf_wavelength_nm_optional(
    requested_nm: float,
    channel_index: int,
    emission_wavelengths: Sequence[float | None],
    emission_units: Sequence[str | None],
    excitation_wavelengths: Sequence[float | None],
    excitation_units: Sequence[str | None],
    *,
    auto: bool,
) -> tuple[float | None, str]:
    requested = _positive_float(requested_nm, 0.0)
    if requested > 0:
        return requested, "manual"
    if not auto:
        return None, "unresolved"
    for values, units in (
        (emission_wavelengths, emission_units),
        (excitation_wavelengths, excitation_units),
    ):
        if not values or channel_index >= len(values):
            continue
        converted = _length_to_nanometer(
            values[channel_index],
            units[channel_index] if channel_index < len(units) else None,
        )
        if converted is not None and converted > 0:
            return converted, "metadata"
    return None, "unresolved"


def _resolved_wavelength_nm(
    requested_nm: float,
    channel_index: int,
    emission_wavelengths: Sequence[float | None],
    emission_units: Sequence[str | None],
    excitation_wavelengths: Sequence[float | None],
    excitation_units: Sequence[str | None],
) -> float:
    requested = _positive_float(requested_nm, 0.0)
    if requested > 0:
        return requested
    for values, units in (
        (emission_wavelengths, emission_units),
        (excitation_wavelengths, excitation_units),
    ):
        if not values or channel_index >= len(values):
            continue
        converted = _length_to_nanometer(
            values[channel_index],
            units[channel_index] if channel_index < len(units) else None,
        )
        if converted is not None and converted > 0:
            return converted
    return 520.0


def _metadata_xy_pixel_size_um(
    axis_names: Sequence[str],
    axis_types: Sequence[str],
    axis_scales: Sequence[float],
    axis_units: Sequence[str | None],
) -> float:
    return (
        _metadata_xy_pixel_size_um_optional(
            axis_names,
            axis_types,
            axis_scales,
            axis_units,
        )
        or 0.1
    )


def _metadata_xy_pixel_size_um_optional(
    axis_names: Sequence[str],
    axis_types: Sequence[str],
    axis_scales: Sequence[float],
    axis_units: Sequence[str | None],
) -> float | None:
    values = [
        _metadata_axis_size_um(
            name,
            axis_names,
            axis_types,
            axis_scales,
            axis_units,
        )
        for name in ("x", "y")
    ]
    present = [value for value in values if value is not None and value > 0]
    if present:
        return float(np.mean(present))
    spatial = []
    for index, axis_type in enumerate(axis_types):
        if str(axis_type).strip().lower() != "space":
            continue
        if index >= len(axis_scales):
            continue
        converted = _length_to_micrometer(
            axis_scales[index],
            axis_units[index] if index < len(axis_units) else None,
        )
        if converted is not None and converted > 0:
            spatial.append(converted)
    if spatial:
        return float(np.mean(spatial[-2:]))
    return None


def _metadata_axis_size_um(
    name: str,
    axis_names: Sequence[str],
    axis_types: Sequence[str],
    axis_scales: Sequence[float],
    axis_units: Sequence[str | None],
) -> float | None:
    target = str(name).strip().lower()
    for index, axis_name in enumerate(axis_names):
        if str(axis_name).strip().lower() != target:
            continue
        if index >= len(axis_scales):
            return None
        return _length_to_micrometer(
            axis_scales[index],
            axis_units[index] if index < len(axis_units) else None,
        )
    if target == "z":
        spatial_indices = [
            index
            for index, axis_type in enumerate(axis_types)
            if str(axis_type).strip().lower() == "space"
        ]
        if len(spatial_indices) >= 3:
            index = spatial_indices[-3]
            return _length_to_micrometer(
                axis_scales[index],
                axis_units[index] if index < len(axis_units) else None,
            )
    return None


def _length_to_nanometer(value, unit: str | None) -> float | None:
    micrometer = _length_to_micrometer(value, unit, infer_wavelength_unit=True)
    return None if micrometer is None else micrometer * 1000.0


def _length_to_micrometer(
    value,
    unit: str | None,
    *,
    infer_wavelength_unit: bool = False,
) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if number <= 0 or not np.isfinite(number):
        return None
    normalized = _normalized_length_unit(unit)
    if normalized in {"meter", "m"}:
        return number * 1_000_000.0
    if normalized in {"millimeter", "millimetre", "mm"}:
        return number * 1000.0
    if normalized in {"micrometer", "micrometre", "um"}:
        return number
    if normalized in {"nanometer", "nanometre", "nm"}:
        return number / 1000.0
    if normalized in {"pixel", "pixels", "px"}:
        return None
    if infer_wavelength_unit and number <= 10.0:
        return number
    return number / 1000.0 if infer_wavelength_unit else number


def _normalized_length_unit(unit: str | None) -> str:
    text = str(unit or "").strip().lower()
    text = text.replace("\u00b5", "u").replace("\u03bc", "u")
    aliases = {
        "micron": "um",
        "microns": "um",
        "micrometer": "um",
        "micrometers": "um",
        "micrometre": "um",
        "micrometres": "um",
        "nanometer": "nm",
        "nanometers": "nm",
        "nanometre": "nm",
        "nanometres": "nm",
        "millimeter": "mm",
        "millimeters": "mm",
        "millimetre": "mm",
        "millimetres": "mm",
        "meter": "m",
        "meters": "m",
        "metre": "m",
        "metres": "m",
    }
    return aliases.get(text, text)


def _xy_structure(arr: np.ndarray, size: int) -> np.ndarray:
    structure = np.zeros([1] * arr.ndim, dtype=bool)
    slices = [0] * arr.ndim
    size = max(int(size), 1)
    for axis in _xy_axes(arr):
        shape = list(structure.shape)
        shape[axis] = size
        structure = np.broadcast_to(structure, shape).copy()
        slices[axis] = slice(None)
    structure[tuple(slices)] = True
    return structure


def _xy_axes(
    arr: np.ndarray,
    *,
    channel_axis: int | None = None,
    axis_names: Sequence[str] = (),
) -> tuple[int, int]:
    """Return declared Y/X axes or trailing axes without shape inference."""
    if axis_names:
        names = tuple(str(name).strip().casefold() for name in axis_names)
        if len(names) != arr.ndim:
            raise ValueError(
                "Declared axis names must match the input array rank."
            )
        if names.count("y") != 1 or names.count("x") != 1:
            raise ValueError(
                "A Y/X operation requires exactly one declared y axis and "
                "one declared x axis."
            )
        y_axis, x_axis = names.index("y"), names.index("x")
        if channel_axis in {y_axis, x_axis}:
            raise ValueError(
                "The declared channel axis cannot also be a Y/X spatial axis."
            )
        return y_axis, x_axis
    spatial_axes = _spatial_axes(arr, channel_axis=channel_axis)
    if len(spatial_axes) >= 2:
        return spatial_axes[-2], spatial_axes[-1]
    if len(spatial_axes) == 1:
        return spatial_axes[0], spatial_axes[0]
    return 0, 0


def _spatial_axes(
    arr: np.ndarray,
    *,
    channel_axis: int | None = None,
) -> list[int]:
    """Return array axes excluding only an explicitly supplied channel axis."""
    axes = list(range(arr.ndim))
    if channel_axis is not None:
        axes.remove(channel_axis)
    return axes


def _has_channel_axis(arr: np.ndarray) -> bool:
    return arr.ndim >= 3 and arr.shape[-1] in RGB_CHANNELS


def _is_color_image(arr: np.ndarray) -> bool:
    return arr.ndim == 3 and arr.shape[-1] in RGB_CHANNELS


def _default_channel_axis(arr: np.ndarray) -> int | None:
    if _has_channel_axis(arr):
        return arr.ndim - 1
    if arr.ndim >= 4:
        return 1 if arr.ndim >= 5 else 0
    if arr.ndim == 3 and arr.shape[0] <= 16:
        return 0
    return None


def _strict_channel_axis(
    arr: np.ndarray,
    *,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
) -> int | None:
    for index, axis_type in enumerate(axis_types[: arr.ndim]):
        if str(axis_type).strip().lower() == "channel":
            return index
    for index, axis_name in enumerate(axis_names[: arr.ndim]):
        if str(axis_name).strip().lower() in {"c", "channel", "rgb", "rgba"}:
            return index
    return None


def _axis_index_from_token(value, ndim: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Split Axis requires an axis:N integer selection.")
    if isinstance(value, Integral):
        axis = int(value)
        return _validated_axis_index(axis, ndim, operation="Split Axis")
    text = str(value).strip().casefold()
    if not text.startswith("axis:"):
        raise ValueError("Split Axis requires an axis:N integer selection.")
    try:
        axis = int(text.removeprefix("axis:"))
    except ValueError as exc:
        raise ValueError("Split Axis requires an axis:N integer selection.") from exc
    return _validated_axis_index(axis, ndim, operation="Split Axis")


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim <= 0:
        return 0
    axis = int(axis)
    if axis < 0:
        axis += ndim
    return int(np.clip(axis, 0, ndim - 1))


def _validated_axis_index(
    value: int,
    ndim: int,
    *,
    operation: str,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{operation} axis must be an integer.")
    axis = int(value)
    if axis < 0:
        axis += ndim
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"{operation} axis {int(value)} is out of range for {ndim}D input."
        )
    return axis


def _slice_selections(
    arr: np.ndarray,
    axis: int,
    index: int,
    axes,
    indices,
    *,
    use_default: bool = True,
) -> dict[int, int]:
    selected_axes = _strict_int_list(axes, field="selected axes")
    selected_indices = _strict_int_list(indices, field="selected indices")
    if not selected_axes:
        if not use_default:
            return {}
        selected_axes = _strict_int_list(axis, field="selected axis")
        selected_indices = _strict_int_list(index, field="selected index")
    if len(selected_axes) != len(selected_indices):
        raise ValueError(
            "Select Axis Slice requires exactly one index for every selected axis."
        )

    selections: dict[int, int] = {}
    for position, axis_value in enumerate(selected_axes):
        axis_index = _validated_axis_index(
            axis_value,
            arr.ndim,
            operation="Select Axis Slice",
        )
        if axis_index in selections:
            raise ValueError("Select Axis Slice axes must not contain duplicates.")
        slice_index = selected_indices[position]
        if slice_index < 0 or slice_index >= arr.shape[axis_index]:
            raise ValueError(
                f"Select Axis Slice index {slice_index} is out of range for "
                f"axis {axis_value} with size {arr.shape[axis_index]}."
            )
        selections[axis_index] = slice_index
    return selections


def _axis_range_selection(arr: np.ndarray, ranges) -> np.ndarray:
    parsed = _parse_axis_ranges(ranges, arr.shape)
    if not parsed:
        return arr.copy()
    slices = [slice(None)] * arr.ndim
    for axis, (start, end) in parsed.items():
        slices[axis] = slice(start, end + 1)
    return np.ascontiguousarray(arr[tuple(slices)])


def _parse_axis_ranges(value, shape: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    if not isinstance(value, str) or not value.strip():
        return {}
    ranges: dict[int, tuple[int, int]] = {}
    for part in value.split(";"):
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) != 3:
            raise ValueError(
                "Select Axis Slice ranges must use axis:start:end entries."
            )
        try:
            requested_axis = int(pieces[0])
            start = int(pieces[1])
            end = int(pieces[2])
        except ValueError as exc:
            raise ValueError(
                "Select Axis Slice ranges must contain integers."
            ) from exc
        axis = _validated_axis_index(
            requested_axis,
            len(shape),
            operation="Select Axis Slice",
        )
        if axis in ranges:
            raise ValueError("Select Axis Slice ranges must not repeat an axis.")
        if start < 0 or end < 0 or start >= shape[axis] or end >= shape[axis]:
            raise ValueError(
                f"Select Axis Slice range {start}:{end} is out of bounds for "
                f"axis {requested_axis} with size {shape[axis]}."
            )
        if start > end:
            raise ValueError(
                "Select Axis Slice range start must not exceed its end."
            )
        ranges[axis] = (start, end)
    return ranges


def _strict_int_list(value, *, field: str) -> list[int]:
    if value is None or (isinstance(value, str) and not value.strip()):
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        if any(not part for part in parts):
            raise ValueError(f"Select Axis Slice {field} contain an empty value.")
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    parsed: list[int] = []
    for part in parts:
        if isinstance(part, (bool, np.bool_)):
            raise ValueError(f"Select Axis Slice {field} must be integers.")
        if isinstance(part, Integral):
            parsed.append(int(part))
            continue
        if isinstance(part, str):
            try:
                parsed.append(int(part))
                continue
            except ValueError:
                pass
        raise ValueError(f"Select Axis Slice {field} must be integers.")
    return parsed


def _parse_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    parsed = []
    for part in parts:
        try:
            parsed.append(int(part))
        except (TypeError, ValueError):
            continue
    return parsed


def _project_array(
    arr: np.ndarray,
    axis_indices: Sequence[int],
    method: str,
) -> np.ndarray:
    axes = tuple(sorted({int(axis) for axis in axis_indices}))
    if not axes:
        return arr.copy()
    reducer = _projection_reducer(method)
    result = reducer(arr, axis=axes)
    return _projection_result_dtype(result, arr, method)


def _projection_reducer(method: str) -> Callable:
    normalized = _normalized_projection_method(method)
    reducers: dict[str, Callable] = {
        "maximum": np.max,
        "max": np.max,
        "mean": np.mean,
        "average": np.mean,
        "minimum": np.min,
        "min": np.min,
        "median": np.median,
        "sum": np.sum,
        "standard deviation": np.std,
        "std": np.std,
        "std dev": np.std,
    }
    return reducers.get(normalized, np.max)


def _projection_result_dtype(
    result: np.ndarray,
    original: np.ndarray,
    method: str,
) -> np.ndarray:
    normalized = _normalized_projection_method(method)
    if normalized in {"maximum", "max", "minimum", "min"}:
        return np.ascontiguousarray(result.astype(original.dtype, copy=False))
    if np.issubdtype(original.dtype, np.floating):
        return np.ascontiguousarray(result.astype(original.dtype, copy=False))
    return np.ascontiguousarray(result.astype(np.float32, copy=False))


def _normalized_projection_method(method: str) -> str:
    return str(method or "Maximum").strip().lower().replace("_", " ")


def _projection_axis_indices(
    ndim: int,
    axes,
    *,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    shape: Sequence[int] = (),
) -> tuple[int, ...]:
    tokens = _projection_axis_tokens(axes)
    if ndim <= 2:
        # Validate explicit requests even though image projection is a 3D+
        # operation and therefore leaves 2D data unchanged.
        for token in tokens:
            if token.startswith("axis:"):
                _projection_axis_index_from_token(token, ndim, (), ())
        return ()
    if not tokens or _tokens_request_auto_projection(tokens):
        return _auto_projection_axis_indices(
            ndim,
            axis_names=axis_names,
            axis_types=axis_types,
            shape=shape,
        )
    if _tokens_request_non_yx_spatial_projection(tokens):
        return _non_yx_spatial_axis_indices(
            ndim,
            axis_names=axis_names,
            axis_types=axis_types,
            shape=shape,
        )

    names = _normalized_axis_names(ndim, axis_names, shape)
    types = [str(axis_type).strip().lower() for axis_type in axis_types]
    parsed: list[int] = []
    for token in tokens:
        index = _projection_axis_index_from_token(token, ndim, names, types)
        if index is None:
            raise ValueError(
                f"Project Image axis selection {token!r} does not match an input axis."
            )
        if index not in parsed:
            parsed.append(index)
    return tuple(parsed)


def _projection_axis_tokens(axes) -> list[str]:
    if axes is None:
        return []
    if isinstance(axes, (list, tuple)):
        values = axes
    else:
        values = (axes,)
    return [
        token
        for token in (_projection_axis_token(value) for value in values)
        if token
    ]


def _projection_axis_token(value) -> str:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("Project Image axes must use auto, axis:N, or name:axis.")
    if isinstance(value, (int, np.integer)):
        return f"axis:{int(value)}"
    text = str(value).strip().lower()
    if text in {"", "auto", "non_yx_spatial"}:
        return text
    if text.startswith(("axis:", "name:")):
        return text
    raise ValueError("Project Image axes must use auto, axis:N, or name:axis.")


def _tokens_request_auto_projection(tokens: Sequence[str]) -> bool:
    text = " ".join(tokens).strip().lower()
    return text in {"", "auto"}


def _tokens_request_non_yx_spatial_projection(tokens: Sequence[str]) -> bool:
    text = " ".join(tokens).strip().lower()
    return text == "non_yx_spatial"


def _projection_axis_index_from_token(
    token: str,
    ndim: int,
    axis_names: Sequence[str],
    axis_types: Sequence[str],
) -> int | None:
    normalized = token.strip().lower()
    if not normalized:
        return None
    if normalized.startswith("axis:"):
        try:
            axis = int(normalized.removeprefix("axis:"))
        except ValueError as exc:
            raise ValueError("Project Image axis:N must contain an integer.") from exc
        return _validated_axis_index(axis, ndim, operation="Project Image")
    if not normalized.startswith("name:"):
        return None
    normalized = normalized.removeprefix("name:")

    aliases = {
        "time": "t",
        "channel": "c",
        "channels": "c",
        "sample": "s",
        "rgb": "rgb",
        "rgba": "rgba",
    }
    name = aliases.get(normalized, normalized)
    for index, axis_name in enumerate(axis_names[:ndim]):
        if axis_name == name:
            return index
    if name == "t":
        for index, axis_type in enumerate(axis_types[:ndim]):
            if axis_type == "time":
                return index
    if name in {"c", "rgb", "rgba"}:
        for index, axis_type in enumerate(axis_types[:ndim]):
            if axis_type == "channel":
                return index
    return None


def _auto_projection_axis_indices(
    ndim: int,
    *,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    shape: Sequence[int] = (),
) -> tuple[int, ...]:
    names = _normalized_axis_names(ndim, axis_names, shape)
    if "z" in names:
        return (names.index("z"),)
    spatial_axes = _metadata_spatial_axis_indices(
        ndim,
        axis_types=axis_types,
        axis_names=names,
    )
    if len(spatial_axes) >= 3:
        return (spatial_axes[-3],)
    fallback = _fallback_projection_spatial_indices(ndim, shape)
    if len(fallback) >= 3:
        return (fallback[-3],)
    return ()


def _non_yx_spatial_axis_indices(
    ndim: int,
    *,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    shape: Sequence[int] = (),
) -> tuple[int, ...]:
    names = _normalized_axis_names(ndim, axis_names, shape)
    spatial_axes = _metadata_spatial_axis_indices(
        ndim,
        axis_types=axis_types,
        axis_names=names,
    )
    if not spatial_axes:
        spatial_axes = _fallback_projection_spatial_indices(ndim, shape)
    if len(spatial_axes) <= 2:
        return ()
    if "y" in names and "x" in names:
        yx_axes = {names.index("y"), names.index("x")}
        return tuple(axis for axis in spatial_axes if axis not in yx_axes)
    return tuple(spatial_axes[:-2])


def _orthogonal_spatial_axis_indices(
    ndim: int,
    *,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    shape: Sequence[int] = (),
) -> tuple[int, ...]:
    if ndim < 3:
        return ()
    names = _normalized_axis_names(ndim, axis_names, shape)
    named = [names.index(name) for name in ("z", "y", "x") if name in names]
    if len(named) == 3:
        return tuple(named)
    spatial_axes = _metadata_spatial_axis_indices(
        ndim,
        axis_types=axis_types,
        axis_names=names,
    )
    if len(spatial_axes) >= 3:
        return tuple(spatial_axes[-3:])
    fallback = _fallback_projection_spatial_indices(ndim, shape)
    if len(fallback) >= 3:
        return tuple(fallback[-3:])
    return ()


def _metadata_spatial_axis_indices(
    ndim: int,
    *,
    axis_types: Sequence[str],
    axis_names: Sequence[str],
) -> list[int]:
    types = [str(axis_type).strip().lower() for axis_type in axis_types]
    spatial = [
        index
        for index, axis_type in enumerate(types[:ndim])
        if axis_type == "space"
    ]
    if spatial:
        return spatial
    return [
        index
        for index, axis_name in enumerate(axis_names[:ndim])
        if axis_name in {"z", "y", "x"}
    ]


def _fallback_projection_spatial_indices(
    ndim: int,
    shape: Sequence[int] = (),
) -> list[int]:
    names = _fallback_axis_names(ndim, shape)
    spatial = [
        index
        for index, axis_name in enumerate(names)
        if axis_name in {"z", "y", "x"}
    ]
    if spatial:
        return spatial
    if ndim == 3 and shape and int(shape[-1]) in RGB_CHANNELS:
        return [0, 1]
    return list(range(ndim))


def _normalized_axis_names(
    ndim: int,
    axis_names: Sequence[str] = (),
    shape: Sequence[int] = (),
) -> list[str]:
    names = [str(name).strip().lower() for name in axis_names[:ndim]]
    if len(names) == ndim and all(names):
        return names
    return list(_fallback_axis_names(ndim, shape))


def _fallback_axis_names(ndim: int, shape: Sequence[int] = ()) -> tuple[str, ...]:
    shape_tuple = tuple(int(size) for size in shape) if shape else ()
    if ndim == 5:
        return ("t", "c", "z", "y", "x")
    if ndim == 4:
        if shape_tuple and shape_tuple[-1] in RGB_CHANNELS:
            return ("z", "y", "x", "rgb")
        if shape_tuple and shape_tuple[0] <= 4:
            return ("c", "z", "y", "x")
        return ("t", "z", "y", "x")
    if ndim == 3:
        if shape_tuple and shape_tuple[-1] in RGB_CHANNELS:
            return ("y", "x", "rgb")
        return ("z", "y", "x")
    if ndim == 2:
        return ("y", "x")
    return tuple(f"axis{index}" for index in range(ndim))


def _xyz_axis_indices(
    ndim: int,
    *,
    axis_names: Sequence[str] = (),
    axis_types: Sequence[str] = (),
    shape: Sequence[int] = (),
) -> dict[str, int]:
    names = _normalized_axis_names(ndim, axis_names, shape)
    axis_map = {
        name: names.index(name)
        for name in ("x", "y", "z")
        if name in names
    }
    spatial_axes = _metadata_spatial_axis_indices(
        ndim,
        axis_types=axis_types,
        axis_names=names,
    )
    if not spatial_axes:
        spatial_axes = _fallback_projection_spatial_indices(ndim, shape)
    if spatial_axes:
        axis_map.setdefault("x", spatial_axes[-1])
    if len(spatial_axes) >= 2:
        axis_map.setdefault("y", spatial_axes[-2])
    if len(spatial_axes) >= 3:
        axis_map.setdefault("z", spatial_axes[-3])
    return axis_map


def _resize_order(interpolation: str, semantic: str) -> int:
    if semantic in {"mask", "labels"}:
        return 0
    normalized = str(interpolation or "Auto").strip().lower()
    if normalized.startswith("auto"):
        return 1
    if "nearest" in normalized:
        return 0
    if "linear" in normalized or "bilinear" in normalized or "trilinear" in normalized:
        return 1
    if "quadratic" in normalized:
        return 2
    if "cubic" in normalized or "bicubic" in normalized or "tricubic" in normalized:
        return 3
    if "quartic" in normalized:
        return 4
    if "quintic" in normalized:
        return 5
    return 1


def _rescale_semantic_kind(arr: np.ndarray, input_kind: str) -> str:
    text = str(input_kind or "").strip().lower()
    if arr.dtype == bool or "mask" in text:
        return "mask"
    if "label" in text:
        return "labels"
    return "image"


def _restore_rescaled_axes_dtype(
    resized: np.ndarray,
    original: np.ndarray,
    semantic: str,
) -> np.ndarray:
    if semantic == "mask" or original.dtype == bool:
        return np.ascontiguousarray(resized > 0.5)
    return _restore_numeric_dtype(resized, original)


def _orthogonal_projection_block(
    block: np.ndarray,
    method: str,
    display_shape: tuple[int, int, int, int],
) -> np.ndarray:
    display_z_y, display_z_x, display_y, display_x = display_shape
    xy = _project_array(block, (0,), method)
    xz = _project_array(block, (1,), method)
    yz = _project_array(block, (2,), method).T
    xy = _resize_projection_panel(xy, (display_y, display_x))
    xz = _resize_projection_panel(xz, (display_z_y, display_x))
    yz = _resize_projection_panel(yz, (display_y, display_z_x))
    fill_value = _projection_canvas_fill(block)
    canvas = np.full(
        (display_y + display_z_y, display_x + display_z_x),
        fill_value,
        dtype=xy.dtype,
    )
    canvas[:display_y, :display_x] = xy
    canvas[display_y:, :display_x] = xz
    canvas[:display_y, display_x:] = yz
    return canvas


def _orthogonal_display_scales(
    spatial_axes: Sequence[int],
    *,
    use_physical_scale: bool,
    axis_scales: Sequence[float] = (),
    axis_units: Sequence[str | None] = (),
) -> tuple[float, float, float]:
    if not use_physical_scale or len(spatial_axes) < 3:
        return (1.0, 1.0, 1.0)
    converted: list[float] = []
    normalized_units: list[str] = []
    for axis in spatial_axes[-3:]:
        scale = _positive_float(
            axis_scales[axis] if axis < len(axis_scales) else 1.0,
            1.0,
        )
        unit = axis_units[axis] if axis < len(axis_units) else None
        normalized_unit = _normalized_physical_unit(unit)
        factor = _unit_to_micrometer_factor(normalized_unit)
        converted.append(scale * factor if factor is not None else scale)
        normalized_units.append(normalized_unit)
    known_units = [unit for unit in normalized_units if unit not in {"", "pixel"}]
    if known_units and any(
        _unit_to_micrometer_factor(unit) is None for unit in known_units
    ):
        if len(set(known_units)) > 1:
            return (1.0, 1.0, 1.0)
    if any(value <= 0 or not np.isfinite(value) for value in converted):
        return (1.0, 1.0, 1.0)
    return tuple(converted)  # type: ignore[return-value]


def _orthogonal_display_sizes(
    shape: tuple[int, int, int],
    display_scales: tuple[float, float, float],
) -> tuple[int, int, int, int]:
    z_size, y_size, x_size = shape
    z_scale, y_scale, x_scale = display_scales
    if any(
        value <= 0 or not np.isfinite(value)
        for value in (z_scale, y_scale, x_scale)
    ):
        return z_size, z_size, y_size, x_size
    display_z_y = max(int(round(z_size * z_scale / y_scale)), 1)
    display_z_x = max(int(round(z_size * z_scale / x_scale)), 1)
    return display_z_y, display_z_x, y_size, x_size


def _resize_projection_panel(
    panel: np.ndarray,
    target_shape: tuple[int, int],
) -> np.ndarray:
    target_y, target_x = (max(int(size), 1) for size in target_shape)
    if panel.shape == (target_y, target_x):
        return np.ascontiguousarray(panel)
    zoom = (
        target_y / max(panel.shape[0], 1),
        target_x / max(panel.shape[1], 1),
    )
    order = 1 if np.issubdtype(panel.dtype, np.floating) else 0
    resized = ndi.zoom(panel, zoom=zoom, order=order)
    return _fit_projection_panel(resized, (target_y, target_x), panel.dtype)


def _fit_projection_panel(
    panel: np.ndarray,
    target_shape: tuple[int, int],
    dtype: np.dtype,
) -> np.ndarray:
    fitted = np.zeros(target_shape, dtype=dtype)
    y_size = min(panel.shape[0], target_shape[0])
    x_size = min(panel.shape[1], target_shape[1])
    fitted[:y_size, :x_size] = panel[:y_size, :x_size].astype(dtype, copy=False)
    return np.ascontiguousarray(fitted)


def _projection_canvas_fill(arr: np.ndarray):
    if arr.size == 0:
        return 0
    if np.issubdtype(arr.dtype, np.floating):
        finite = arr[np.isfinite(arr)]
        if finite.size:
            return finite.min()
        return 0.0
    return arr.min()


def _positive_float(value, default: float) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if result <= 0 or not np.isfinite(result):
        return default
    return result


def _positive_int(value, default: int) -> int:
    try:
        result = int(value)
    except Exception:
        return default
    return result if result > 0 else default


def _normalized_physical_unit(unit) -> str:
    text = str(unit or "").strip().lower()
    text = text.replace("\u00b5", "u").replace("\u03bc", "u")
    aliases = {
        "um": "micrometer",
        "micron": "micrometer",
        "microns": "micrometer",
        "micrometre": "micrometer",
        "micrometres": "micrometer",
        "micrometers": "micrometer",
        "nm": "nanometer",
        "nanometre": "nanometer",
        "nanometres": "nanometer",
        "nanometers": "nanometer",
        "mm": "millimeter",
        "millimetre": "millimeter",
        "millimetres": "millimeter",
        "millimeters": "millimeter",
        "m": "meter",
        "metre": "meter",
        "metres": "meter",
        "meters": "meter",
        "px": "pixel",
        "pixels": "pixel",
    }
    return aliases.get(text, text)


def _unit_to_micrometer_factor(unit: str) -> float | None:
    return {
        "nanometer": 0.001,
        "micrometer": 1.0,
        "millimeter": 1000.0,
        "meter": 1_000_000.0,
    }.get(unit)


def _axis_order_indices(
    order,
    ndim: int,
    axis_names: Sequence[str] = (),
) -> tuple[int, ...] | None:
    if ndim <= 1:
        return tuple(range(ndim))
    tokens = _axis_order_tokens(order)
    if not tokens:
        return tuple(range(ndim))
    if len(tokens) != ndim:
        return None

    indices = _numeric_axis_order(tokens, ndim)
    if indices is not None:
        return indices
    return _named_axis_order(tokens, ndim, axis_names)


def _axis_order_tokens(order) -> list[str]:
    if order is None:
        return []
    if isinstance(order, (list, tuple)):
        return [str(part).strip() for part in order if str(part).strip()]
    text = str(order).strip()
    if not text:
        return []
    if any(separator in text for separator in (",", ";", " ")):
        normalized = text.replace(";", ",").replace(" ", ",")
        return [part.strip() for part in normalized.split(",") if part.strip()]
    return list(text)


def _numeric_axis_order(tokens: Sequence[str], ndim: int) -> tuple[int, ...] | None:
    indices: list[int] = []
    try:
        for token in tokens:
            axis = int(token)
            if axis < 0:
                axis += ndim
            indices.append(axis)
    except ValueError:
        return None
    if sorted(indices) != list(range(ndim)):
        return None
    return tuple(indices)


def _named_axis_order(
    tokens: Sequence[str],
    ndim: int,
    axis_names: Sequence[str],
) -> tuple[int, ...] | None:
    names = [str(name).strip().lower() for name in axis_names]
    if len(names) != ndim:
        return None
    indices: list[int] = []
    used: set[int] = set()
    for token in tokens:
        target = str(token).strip().lower()
        matches = [
            index
            for index, name in enumerate(names)
            if name == target and index not in used
        ]
        if not matches:
            return None
        index = matches[0]
        indices.append(index)
        used.add(index)
    return tuple(indices)


def _parse_finite_weight_list(value) -> list[float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]

    parsed: list[float] = []
    for index, part in enumerate(parts, start=1):
        if isinstance(part, str) and not part:
            raise ValueError(f"Image Calculator weight {index} is empty.")
        try:
            number = float(part)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"Image Calculator weight {index} must be a finite number; "
                f"got {part!r}."
            ) from exc
        if not np.isfinite(number):
            raise ValueError(
                f"Image Calculator weight {index} must be a finite number; "
                f"got {part!r}."
            )
        parsed.append(number)
    return parsed


def _odd_size(value: int | float, minimum: int = 1, maximum: int | None = None) -> int:
    size = max(int(round(float(value))), minimum)
    if maximum is not None:
        maximum = max(int(maximum), minimum)
        if maximum % 2 == 0:
            maximum -= 1
        size = min(size, maximum)
    if size % 2 == 0:
        size += 1
    return max(size, minimum)


def _crop_pair(first: int, second: int, axis_size: int) -> tuple[int, int]:
    values = (first, second)
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in values
    ):
        raise ValueError("Crop margins must be integers.")
    first, second = (int(value) for value in values)
    if first < 0 or second < 0:
        raise ValueError("Crop margins must be non-negative.")
    if axis_size <= 0:
        raise ValueError("Crop stack cannot crop an empty spatial axis.")
    if first + second >= axis_size:
        raise ValueError(
            f"Crop margins {first} and {second} remove every sample from an "
            f"axis of length {axis_size}."
        )
    return first, second


def _float_if_bool(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == bool:
        return arr.astype(np.float32)
    return arr


def _intensity_scale(arr: np.ndarray) -> float:
    if np.issubdtype(arr.dtype, np.integer):
        return float(np.iinfo(arr.dtype).max)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return 1.0
    maximum = float(finite.max())
    return maximum if maximum > 1.0 else 1.0


def _restore_numeric_dtype(values: np.ndarray, original: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if original.dtype == bool:
        return np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=0.0) > 0
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        safe = np.nan_to_num(
            values,
            nan=0.0,
            posinf=float(info.max),
            neginf=float(info.min),
        )
        rounded = np.rint(safe)
        return np.clip(rounded, info.min, info.max).astype(original.dtype)
    if np.issubdtype(original.dtype, np.floating):
        return values.astype(original.dtype, copy=False)
    return values.astype(original.dtype, copy=False)


@dataclass(frozen=True)
class _UnitIntensityScale:
    offset: float
    span: float


def _restore_unit_float_dtype(
    values: np.ndarray,
    original: np.ndarray,
    intensity_scale: _UnitIntensityScale,
    *,
    operation: str,
) -> np.ndarray:
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError(f"{operation} produced non-finite output intensities.")
    restored = values * intensity_scale.span + intensity_scale.offset
    if original.dtype == bool:
        return restored > 0.5
    if np.issubdtype(original.dtype, np.integer):
        return _restore_numeric_dtype(restored, original)
    if np.issubdtype(original.dtype, np.floating):
        return restored.astype(original.dtype, copy=False)
    return restored.astype(original.dtype, copy=False)


def _to_float_unit(
    arr: np.ndarray,
    *,
    operation: str,
) -> tuple[np.ndarray, _UnitIntensityScale]:
    """Map one complete input to a shared unit scale without modifying it."""
    arr = np.asarray(arr)
    if arr.size == 0:
        raise ValueError(f"{operation} requires non-empty image data.")
    work_dtype = np.float64 if arr.dtype == np.float64 else np.float32
    try:
        values = arr.astype(work_dtype, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{operation} requires numeric image data.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"{operation} requires finite image intensities.")

    if arr.dtype == bool:
        offset = 0.0
        span = 1.0
    elif np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        offset = float(info.min)
        span = float(info.max) - offset
    else:
        minimum = float(np.min(values))
        maximum = float(np.max(values))
        if 0.0 <= minimum and maximum <= 1.0:
            offset = 0.0
            span = 1.0
        else:
            offset = minimum
            span = maximum - minimum if maximum > minimum else 1.0
    if not np.isfinite(offset) or not np.isfinite(span) or span <= 0.0:
        raise ValueError(f"{operation} could not resolve a finite intensity span.")

    values -= offset
    values /= span
    np.clip(values, 0.0, 1.0, out=values)
    return values, _UnitIntensityScale(offset=offset, span=span)


def _validated_filter_channel_axis(
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


def _validated_luma_channel_axis(
    arr: np.ndarray,
    value,
    *,
    operation: str,
) -> int | None:
    axis = _validated_filter_channel_axis(
        value,
        np.asarray(arr).ndim,
        operation=operation,
    )
    if axis is None:
        return None

    arr = np.asarray(arr)
    channel_count = int(arr.shape[axis])
    if channel_count not in {3, 4}:
        raise ValueError(
            f"{operation} channel_axis must contain exactly 3 RGB or 4 RGBA "
            f"channels, not {channel_count}."
        )
    if not (
        arr.dtype == bool
        or np.issubdtype(arr.dtype, np.integer)
        or np.issubdtype(arr.dtype, np.floating)
    ):
        raise ValueError(
            f"{operation} RGB/RGBA conversion requires real-valued boolean, "
            "integer, or floating image data."
        )
    return axis


def _validated_filter_scale(
    value,
    *,
    operation: str,
    name: str,
    minimum: float,
    inclusive: bool = True,
) -> float:
    try:
        resolved = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{operation} {name} must be a finite number.") from exc
    if not np.isfinite(resolved):
        raise ValueError(f"{operation} {name} must be a finite number.")
    valid = resolved >= minimum if inclusive else resolved > minimum
    if not valid:
        comparison = "at least" if inclusive else "greater than"
        raise ValueError(
            f"{operation} {name} must be {comparison} {minimum:g}."
        )
    return resolved


def _validated_filter_integer(
    value,
    *,
    operation: str,
    name: str,
    minimum: int,
) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{operation} {name} must be an integer.")
    resolved = int(value)
    if resolved < minimum:
        raise ValueError(f"{operation} {name} must be at least {minimum}.")
    return resolved


def _validated_odd_filter_size(
    value,
    *,
    operation: str,
    name: str,
    minimum: int,
) -> int:
    resolved = _validated_filter_integer(
        value,
        operation=operation,
        name=name,
        minimum=minimum,
    )
    if resolved % 2 == 0:
        raise ValueError(f"{operation} {name} must be odd; got {resolved}.")
    return resolved


def _composite_channel_to_float(arr: np.ndarray) -> np.ndarray:
    """Apply the explicitly requested legacy lossy display normalization."""
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr.astype(np.float32)
    values = arr.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.float32)
    lo, hi = np.percentile(finite.astype(np.float64, copy=False), [1.0, 99.0])
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        if hi > 0.0:
            return np.ones(values.shape, dtype=np.float32)
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0, 1).astype(np.float32)


def _validated_composite_channel_axis(value, ndim: int) -> int:
    if ndim < 1:
        raise ValueError(
            "Composite → RGB requires array data with an explicit channel axis."
        )
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(
            "Composite → RGB requires an explicit non-negative integer "
            "channel axis."
        )
    axis = int(value)
    if axis < 0 or axis >= ndim:
        raise ValueError(
            f"Composite → RGB channel axis {axis} is out of range for "
            f"{ndim}D input."
        )
    return axis


def _validated_composite_intensity_mapping(value) -> str:
    if value not in {
        COMPOSITE_RGB_PRESERVE_VALUES,
        COMPOSITE_RGB_PERCENTILE_1_99,
    }:
        raise ValueError(
            "Composite → RGB intensity mapping must be "
            f"{COMPOSITE_RGB_PRESERVE_VALUES!r} or "
            f"{COMPOSITE_RGB_PERCENTILE_1_99!r}."
        )
    return str(value)


def _validate_composite_input_values(
    arr: np.ndarray,
    intensity_mapping: str,
) -> None:
    if arr.dtype.kind not in {"b", "i", "u", "f"}:
        raise ValueError(
            "Composite → RGB requires boolean, integer, or real floating-point "
            "image data."
        )
    if arr.dtype.kind == "f" and not np.isfinite(arr).all():
        raise ValueError(
            "Composite → RGB requires finite input values; NaN and infinity "
            "have no defined RGB intensity mapping."
        )
    if intensity_mapping != COMPOSITE_RGB_PRESERVE_VALUES or arr.size == 0:
        return
    if arr.dtype.kind == "u":
        outside_exact_float64 = int(arr.max()) > 2**53
    elif arr.dtype.kind == "i":
        minimum = int(arr.min())
        maximum = int(arr.max())
        outside_exact_float64 = minimum < -(2**53) or maximum > 2**53
    else:
        outside_exact_float64 = False
    if outside_exact_float64:
        raise ValueError(
            "Composite → RGB preserve mode cannot represent integer levels "
            "outside the exact float64 range [-2**53, 2**53]. Rescale or "
            "convert the data explicitly before compositing, or select the "
            "explicitly lossy percentile mapping."
        )


def _validated_composite_axis_semantics(value, *, count: int) -> str:
    if not isinstance(value, str):
        raise ValueError("Composite → RGB channel-axis semantics must be text.")
    semantic = value.strip().casefold()
    if semantic == "rgb" and count != 3:
        raise ValueError(
            "Composite → RGB received rgb axis semantics for an axis that "
            f"contains {count} channels instead of 3."
        )
    if semantic == "rgba" and count != 4:
        raise ValueError(
            "Composite → RGB received rgba axis semantics for an axis that "
            f"contains {count} channels instead of 4."
        )
    return semantic


def _validated_composite_channel_selections(
    selections,
    *,
    count: int,
) -> tuple[int, int, int]:
    resolved: list[int] = []
    for plane, selection in zip("RGB", selections, strict=True):
        if isinstance(selection, (bool, np.bool_)) or not isinstance(
            selection,
            Integral,
        ):
            raise ValueError(
                f"Composite → RGB {plane} channel selection must be an integer."
            )
        index = int(selection)
        if index < -1 or index >= count:
            raise ValueError(
                f"Composite → RGB {plane} channel index {index} is out of "
                f"range for {count} channels; use -1 for the positional default."
            )
        resolved.append(index)
    return resolved[0], resolved[1], resolved[2]


def _composite_output_dtype(arr: np.ndarray, intensity_mapping: str) -> np.dtype:
    if intensity_mapping == COMPOSITE_RGB_PERCENTILE_1_99:
        return np.dtype(np.float32)
    return np.dtype(np.result_type(arr.dtype, np.float32))


def _composite_mapped_channel(
    channel: np.ndarray,
    *,
    intensity_mapping: str,
    output_dtype: np.dtype,
) -> np.ndarray:
    if intensity_mapping == COMPOSITE_RGB_PERCENTILE_1_99:
        return _composite_channel_to_float(channel)
    return np.asarray(channel).astype(output_dtype, copy=True)


def _validated_composite_color_table(
    channel_colors: str | list[int | str] | tuple[int | str, ...],
    count: int,
) -> np.ndarray:
    if isinstance(channel_colors, str):
        values: list[int | str] = (
            [part.strip() for part in channel_colors.split(",")]
            if channel_colors.strip()
            else []
        )
    elif isinstance(channel_colors, (list, tuple)):
        values = list(channel_colors)
    else:
        raise ValueError(
            "Composite → RGB channel colours must be comma-separated text "
            "or a sequence."
        )
    if len(values) > count:
        raise ValueError(
            f"Composite → RGB received {len(values)} channel colours for "
            f"an axis containing {count} channels."
        )

    table = channel_color_table("", count)
    for index, value in enumerate(values):
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        color = color_value_to_rgb(value)
        if color is None:
            raise ValueError(
                "Composite → RGB channel colour "
                f"{value!r} at index {index} is not recognized."
            )
        table[index] = color
    return table


def _coloc_normalized_inputs(
    inputs,
    *,
    intensity_max: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    ch1, ch2, _roi_mask, warnings = _coloc_normalized_inputs_and_mask(
        inputs,
        intensity_max=intensity_max,
    )
    return ch1, ch2, warnings


def _coloc_normalized_inputs_and_mask(
    inputs,
    *,
    intensity_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, tuple[str, ...]]:
    values = list(inputs)
    if len(values) < 2:
        raise ValueError("Colocalization nodes require two image inputs.")
    ch1 = _coloc_reduce_to_intensity(values[0])
    ch2 = _coloc_reduce_to_intensity(values[1])
    ch1, ch2, warnings = _normalize_coloc_channels(
        ch1,
        ch2,
        output_max=intensity_max,
    )
    roi_mask = None
    if len(values) >= 3 and values[2] is not None:
        roi_mask = np.asarray(values[2]) > 0
        if roi_mask.shape != ch1.shape:
            raise ValueError(
                f"ROI mask shape {roi_mask.shape} does not match channels {ch1.shape}."
            )
    return ch1, ch2, roi_mask, warnings


def _coloc_reduce_to_intensity(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.ndim < 2:
        raise ValueError("Colocalization nodes require at least 2D image data.")
    if arr.ndim >= 3 and arr.shape[-1] in RGB_CHANNELS:
        return np.max(arr[..., :3], axis=-1)
    return arr


def _normalize_coloc_channels(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    output_max: float,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    ch1 = np.asarray(channel_1, dtype=np.float32)
    ch2 = np.asarray(channel_2, dtype=np.float32)
    if ch1.shape != ch2.shape:
        raise ValueError(f"Input shapes differ: {ch1.shape} vs {ch2.shape}.")
    if not np.isfinite(ch1).any() or not np.isfinite(ch2).any():
        raise ValueError("Both channels must contain finite values.")

    output_max = max(float(output_max), 1.0)
    warnings: list[str] = []
    max_intensity = float(np.nanmax([np.nanmax(ch1), np.nanmax(ch2)]))
    min_intensity = float(np.nanmin([np.nanmin(ch1), np.nanmin(ch2)]))
    if min_intensity < 0:
        warnings.append("Negative intensities were clipped to zero.")
        ch1 = np.clip(ch1, 0.0, None)
        ch2 = np.clip(ch2, 0.0, None)
        max_intensity = float(np.nanmax([np.nanmax(ch1), np.nanmax(ch2)]))
    if max_intensity <= 0:
        raise ValueError("Both channels are empty or zero-valued.")
    if max_intensity <= 1.0:
        scale = output_max
        warnings.append(
            f"Input appeared normalized; scaled jointly to 0..{output_max:g}."
        )
    elif max_intensity > output_max:
        scale = output_max / max_intensity
        warnings.append(
            f"Input maximum {max_intensity:g} exceeded {output_max:g}; "
            f"scaled jointly to 0..{output_max:g}."
        )
    else:
        scale = 1.0

    ch1 = np.nan_to_num(ch1 * scale, nan=0.0, posinf=output_max, neginf=0.0)
    ch2 = np.nan_to_num(ch2 * scale, nan=0.0, posinf=output_max, neginf=0.0)
    return (
        np.clip(ch1, 0.0, output_max).astype(np.float32, copy=False),
        np.clip(ch2, 0.0, output_max).astype(np.float32, copy=False),
        tuple(warnings),
    )


def _coloc_thresholds(
    ch1: np.ndarray,
    ch2: np.ndarray,
    *,
    threshold_mode: str,
    channel_1_threshold: float,
    channel_2_threshold: float,
    intensity_max: float,
    roi_mask: np.ndarray | None = None,
) -> tuple[float, float, dict[str, float] | None]:
    mode = str(threshold_mode).strip().lower()
    if mode.startswith("costes"):
        threshold_ch1 = ch1[roi_mask] if roi_mask is not None else ch1
        threshold_ch2 = ch2[roi_mask] if roi_mask is not None else ch2
        costes = _costes_thresholds(
            threshold_ch1,
            threshold_ch2,
            intensity_max=float(intensity_max),
        )
        return costes["threshold_1"], costes["threshold_2"], costes
    return (
        _clamp(float(channel_1_threshold), 0.0, float(intensity_max)),
        _clamp(float(channel_2_threshold), 0.0, float(intensity_max)),
        None,
    )


def _spatial_table_context(
    reference: np.ndarray,
    *,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
    axis_names: tuple[str, ...] | None,
    axis_types: tuple[str, ...] | None,
    axis_scales: tuple[float, ...] | None,
    axis_units: tuple[str | None, ...] | None,
) -> _SpatialTableContext:
    spatial_ndim = _resolved_spatial_ndim(
        reference,
        spatial_mode,
        resolved_spatial_ndim,
    )
    axis_names = _measurement_axis_names(reference.ndim, axis_names)
    axis_types = _measurement_axis_types(reference.ndim, axis_types)
    spatial_axes = _measurement_spatial_axes(
        reference.ndim,
        spatial_ndim,
        axis_names,
        axis_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(reference.ndim - spatial_ndim, reference.ndim))
    moved_axis_names = tuple(
        axis_names[index]
        for index in range(reference.ndim)
        if index not in spatial_axes
    ) + tuple(axis_names[index] for index in spatial_axes)
    moved_axis_scales = _reordered_axis_values(
        axis_scales,
        reference.ndim,
        spatial_axes,
    )
    moved_axis_units = _reordered_axis_values(
        axis_units,
        reference.ndim,
        spatial_axes,
    )
    leading_axis_count = reference.ndim - spatial_ndim
    spatial_axis_names = _safe_axis_column_names(
        moved_axis_names[-spatial_ndim:],
        fallback=("z", "y", "x")[-spatial_ndim:],
    )
    leading_axis_names = _safe_axis_column_names(
        moved_axis_names[:leading_axis_count],
        fallback=tuple(f"axis_{index}" for index in range(leading_axis_count)),
    )
    moved_reference = _move_to_spatial_last(reference, spatial_axes, spatial_ndim)
    spatial_scales = _normalized_spatial_scales(
        spatial_ndim,
        moved_axis_scales[-spatial_ndim:],
    )
    unit_values = tuple(
        str(unit).strip()
        for unit in moved_axis_units[-spatial_ndim:]
        if unit not in {None, ""}
    )
    calibrated = any(abs(scale - 1.0) > 1e-12 for scale in spatial_scales) or bool(
        unit_values
    )
    physical_unit = _spatial_length_unit_label(unit_values)
    return _SpatialTableContext(
        spatial_ndim=spatial_ndim,
        spatial_axes=spatial_axes,
        leading_axis_names=leading_axis_names,
        spatial_axis_names=spatial_axis_names,
        leading_shape=moved_reference.shape[:leading_axis_count],
        spatial_scales=spatial_scales,
        physical_unit=physical_unit,
        has_physical_calibration=calibrated,
    )


def _move_to_spatial_last(
    arr: np.ndarray,
    spatial_axes: tuple[int, ...],
    spatial_ndim: int,
) -> np.ndarray:
    target_axes = tuple(range(arr.ndim - spatial_ndim, arr.ndim))
    return np.moveaxis(arr, spatial_axes, target_axes)


def _spatial_length_unit_label(units: Sequence[str]) -> str:
    units = tuple(unit for unit in units if str(unit).strip())
    if not units:
        return "pixels"
    if len(set(units)) == 1:
        return units[0]
    return "physical units"


def _label_pair_inputs_and_context(
    inputs,
    message: str,
    *,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
    axis_names: tuple[str, ...] | None,
    axis_types: tuple[str, ...] | None,
    axis_scales: tuple[float, ...] | None,
    axis_units: tuple[str | None, ...] | None,
) -> tuple[np.ndarray, np.ndarray, _SpatialTableContext]:
    values = list(inputs)
    if len(values) < 2:
        raise ValueError(message)
    first = np.asarray(values[0])
    second = np.asarray(values[1])
    if first.shape != second.shape:
        raise ValueError(
            f"Label-like inputs must have matching shapes; got {first.shape} "
            f"and {second.shape}."
        )
    context = _spatial_table_context(
        first,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
        axis_names=axis_names,
        axis_types=axis_types,
        axis_scales=axis_scales,
        axis_units=axis_units,
    )
    return first, second, context


def _objects_from_mask_or_labels(
    data,
    *,
    context: _SpatialTableContext | None = None,
) -> np.ndarray:
    arr = np.asarray(data)
    if arr.dtype == bool or _is_binary_label_like(arr):
        if context is not None:
            return _label_binary_spatial_blocks(arr > 0, context)
        return measure.label(arr > 0).astype(np.int32, copy=False)
    return _validated_labels(arr)


def _event_objects_from_mask_or_labels(
    data,
    *,
    context: _SpatialTableContext | None = None,
) -> np.ndarray:
    return _objects_from_mask_or_labels(data, context=context)


def _regions_from_mask_or_labels(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.dtype == bool or _is_binary_label_like(arr):
        return (arr > 0).astype(np.int32, copy=False)
    return _validated_labels(arr)


def _is_binary_label_like(arr: np.ndarray) -> bool:
    if not np.issubdtype(arr.dtype, np.integer) or arr.size == 0:
        return False
    values = np.unique(arr)
    return bool(np.all((values == 0) | (values == 1)))


def _label_binary_spatial_blocks(
    mask: np.ndarray,
    context: _SpatialTableContext,
) -> np.ndarray:
    moved = _move_to_spatial_last(
        np.asarray(mask, dtype=bool),
        context.spatial_axes,
        context.spatial_ndim,
    )
    labelled = np.zeros(moved.shape, dtype=np.int32)
    for leading_index in np.ndindex(context.leading_shape or (1,)):
        block_index = () if not context.leading_shape else leading_index
        block = moved[block_index] if context.leading_shape else moved
        block_labels = measure.label(block).astype(np.int32, copy=False)
        if context.leading_shape:
            labelled[block_index] = block_labels
        else:
            labelled = block_labels
    target_axes = tuple(range(mask.ndim - context.spatial_ndim, mask.ndim))
    return np.moveaxis(labelled, target_axes, context.spatial_axes)


def _object_colocalization_columns(
    leading_axis_names: tuple[str, ...],
) -> dict[str, list[object]]:
    columns = _leading_index_columns(leading_axis_names)
    columns.update(
        {
            "label_id": [],
            "threshold_mode": [],
            "channel_1_threshold": [],
            "channel_2_threshold": [],
            "threshold_units": [],
            "object_voxels": [],
            "channel_1_positive_voxels": [],
            "channel_2_positive_voxels": [],
            "colocalized_voxels": [],
            "colocalized_fraction": [],
            "pearson_object": [],
            "pearson_colocalized": [],
            "manders_m1": [],
            "manders_m2": [],
            "overlap_coefficient_object": [],
            "overlap_coefficient_colocalized": [],
            "channel_1_positive_sum": [],
            "channel_2_positive_sum": [],
            "colocalized_channel_1_sum": [],
            "colocalized_channel_2_sum": [],
            "costes_slope": [],
            "costes_intercept": [],
            "costes_pearson_below": [],
            "costes_iterations": [],
            "normalization_warnings": [],
        }
    )
    return columns


def _append_object_colocalization_rows(
    columns: dict[str, list[object]],
    leading_axis_names: tuple[str, ...],
    block_index: tuple[int, ...],
    labels: np.ndarray,
    ch1: np.ndarray,
    ch2: np.ndarray,
    *,
    threshold_mode: str,
    threshold_1: float,
    threshold_2: float,
    intensity_max: float,
    costes: dict[str, float] | None,
    warnings: tuple[str, ...],
) -> None:
    threshold_units = f"normalized_0_{_format_coloc_max(intensity_max)}"
    warning_text = "; ".join(warnings)
    for label_id in _positive_label_ids(labels):
        object_mask = labels == label_id
        object_voxels = int(np.count_nonzero(object_mask))
        positive_1 = object_mask & (ch1 >= threshold_1)
        positive_2 = object_mask & (ch2 >= threshold_2)
        overlap = positive_1 & positive_2
        sum_1_positive = float(np.sum(ch1[positive_1], dtype=np.float64))
        sum_2_positive = float(np.sum(ch2[positive_2], dtype=np.float64))
        sum_1_overlap = float(np.sum(ch1[overlap], dtype=np.float64))
        sum_2_overlap = float(np.sum(ch2[overlap], dtype=np.float64))
        _append_leading_index_values(columns, leading_axis_names, block_index)
        columns["label_id"].append(int(label_id))
        columns["threshold_mode"].append(str(threshold_mode))
        columns["channel_1_threshold"].append(float(threshold_1))
        columns["channel_2_threshold"].append(float(threshold_2))
        columns["threshold_units"].append(threshold_units)
        columns["object_voxels"].append(object_voxels)
        columns["channel_1_positive_voxels"].append(int(np.count_nonzero(positive_1)))
        columns["channel_2_positive_voxels"].append(int(np.count_nonzero(positive_2)))
        columns["colocalized_voxels"].append(int(np.count_nonzero(overlap)))
        columns["colocalized_fraction"].append(
            _safe_fraction(np.count_nonzero(overlap), object_voxels)
        )
        columns["pearson_object"].append(_pearson(ch1[object_mask], ch2[object_mask]))
        columns["pearson_colocalized"].append(_pearson(ch1[overlap], ch2[overlap]))
        columns["manders_m1"].append(_safe_fraction(sum_1_overlap, sum_1_positive))
        columns["manders_m2"].append(_safe_fraction(sum_2_overlap, sum_2_positive))
        columns["overlap_coefficient_object"].append(
            _overlap_coefficient(ch1[object_mask], ch2[object_mask])
        )
        columns["overlap_coefficient_colocalized"].append(
            _overlap_coefficient(ch1[overlap], ch2[overlap])
        )
        columns["channel_1_positive_sum"].append(sum_1_positive)
        columns["channel_2_positive_sum"].append(sum_2_positive)
        columns["colocalized_channel_1_sum"].append(sum_1_overlap)
        columns["colocalized_channel_2_sum"].append(sum_2_overlap)
        columns["costes_slope"].append(
            float("nan") if costes is None else float(costes["slope"])
        )
        columns["costes_intercept"].append(
            float("nan") if costes is None else float(costes["intercept"])
        )
        columns["costes_pearson_below"].append(
            float("nan") if costes is None else float(costes["pearson_below"])
        )
        columns["costes_iterations"].append(
            0 if costes is None else int(costes["iterations"])
        )
        columns["normalization_warnings"].append(warning_text)


def _label_overlap_columns(
    leading_axis_names: tuple[str, ...],
) -> dict[str, list[object]]:
    columns = _leading_index_columns(leading_axis_names)
    columns.update(
        {
            "label_id": [],
            "target_label_id": [],
            "reference_voxels": [],
            "target_voxels": [],
            "overlap_voxels": [],
            "reference_overlap_fraction": [],
            "target_overlap_fraction": [],
            "intersection_over_union": [],
        }
    )
    return columns


def _append_label_overlap_rows(
    columns: dict[str, list[object]],
    leading_axis_names: tuple[str, ...],
    block_index: tuple[int, ...],
    reference: np.ndarray,
    target: np.ndarray,
) -> None:
    reference_counts = _positive_label_counts(reference)
    target_counts = _positive_label_counts(target)
    pair_counts = _positive_pair_counts(reference, target)
    for reference_id, target_id in sorted(pair_counts):
        overlap_voxels = int(pair_counts[(reference_id, target_id)])
        reference_voxels = int(reference_counts.get(reference_id, 0))
        target_voxels = int(target_counts.get(target_id, 0))
        union = reference_voxels + target_voxels - overlap_voxels
        _append_leading_index_values(columns, leading_axis_names, block_index)
        columns["label_id"].append(int(reference_id))
        columns["target_label_id"].append(int(target_id))
        columns["reference_voxels"].append(reference_voxels)
        columns["target_voxels"].append(target_voxels)
        columns["overlap_voxels"].append(overlap_voxels)
        columns["reference_overlap_fraction"].append(
            _safe_fraction(overlap_voxels, reference_voxels)
        )
        columns["target_overlap_fraction"].append(
            _safe_fraction(overlap_voxels, target_voxels)
        )
        columns["intersection_over_union"].append(_safe_fraction(overlap_voxels, union))


def _nearest_distance_columns(
    context: _SpatialTableContext,
) -> dict[str, list[object]]:
    columns = _leading_index_columns(context.leading_axis_names)
    columns.update(
        {
            "label_id": [],
            "nearest_label_id": [],
            "centroid_distance_pixels": [],
        }
    )
    if context.has_physical_calibration:
        columns["centroid_distance_physical"] = []
        columns["physical_unit"] = []
    return columns


def _append_nearest_distance_rows(
    columns: dict[str, list[object]],
    context: _SpatialTableContext,
    block_index: tuple[int, ...],
    reference: np.ndarray,
    target: np.ndarray,
) -> None:
    reference_centroids = _label_centroids(reference)
    target_centroids = _label_centroids(target)
    target_ids = sorted(target_centroids)
    target_points = (
        np.asarray([target_centroids[label_id] for label_id in target_ids])
        if target_ids
        else np.empty((0, context.spatial_ndim), dtype=np.float64)
    )
    scales = np.asarray(context.spatial_scales, dtype=np.float64)
    for label_id in sorted(reference_centroids):
        centroid = reference_centroids[label_id]
        if target_points.size:
            deltas = target_points - centroid
            pixel_distances = np.linalg.norm(deltas, axis=1)
            nearest_index = int(np.argmin(pixel_distances))
            nearest_label_id = int(target_ids[nearest_index])
            pixel_distance = float(pixel_distances[nearest_index])
            physical_distance = float(np.linalg.norm(deltas[nearest_index] * scales))
        else:
            nearest_label_id = 0
            pixel_distance = float("nan")
            physical_distance = float("nan")
        _append_leading_index_values(
            columns,
            context.leading_axis_names,
            block_index,
        )
        columns["label_id"].append(int(label_id))
        columns["nearest_label_id"].append(nearest_label_id)
        columns["centroid_distance_pixels"].append(pixel_distance)
        if context.has_physical_calibration:
            columns["centroid_distance_physical"].append(physical_distance)
            columns["physical_unit"].append(context.physical_unit)


def _event_localization_columns(
    leading_axis_names: tuple[str, ...],
) -> dict[str, list[object]]:
    columns = _leading_index_columns(leading_axis_names)
    columns.update(
        {
            "event_id": [],
            "event_voxels": [],
            "region_label_id": [],
            "overlap_voxels": [],
            "event_overlap_fraction": [],
            "in_region": [],
        }
    )
    return columns


def _append_event_localization_rows(
    columns: dict[str, list[object]],
    leading_axis_names: tuple[str, ...],
    block_index: tuple[int, ...],
    events: np.ndarray,
    regions: np.ndarray,
) -> None:
    event_counts = _positive_label_counts(events)
    overlaps_by_event: dict[int, list[tuple[int, int]]] = {}
    for (event_id, region_id), count in _positive_pair_counts(events, regions).items():
        overlaps_by_event.setdefault(int(event_id), []).append(
            (int(region_id), int(count))
        )
    for event_id in sorted(event_counts):
        overlaps = sorted(
            overlaps_by_event.get(event_id, ()),
            key=lambda item: (-item[1], item[0]),
        )
        if overlaps:
            region_label_id, overlap_voxels = overlaps[0]
        else:
            region_label_id, overlap_voxels = 0, 0
        event_voxels = int(event_counts[event_id])
        _append_leading_index_values(columns, leading_axis_names, block_index)
        columns["event_id"].append(int(event_id))
        columns["event_voxels"].append(event_voxels)
        columns["region_label_id"].append(int(region_label_id))
        columns["overlap_voxels"].append(int(overlap_voxels))
        columns["event_overlap_fraction"].append(
            _safe_fraction(overlap_voxels, event_voxels)
        )
        columns["in_region"].append(bool(overlap_voxels > 0))


def _leading_index_columns(
    leading_axis_names: tuple[str, ...],
) -> dict[str, list[object]]:
    return {f"{axis_name}_index": [] for axis_name in leading_axis_names}


def _append_leading_index_values(
    columns: dict[str, list[object]],
    leading_axis_names: tuple[str, ...],
    block_index: tuple[int, ...],
) -> None:
    for axis_position, axis_name in enumerate(leading_axis_names):
        columns[f"{axis_name}_index"].append(int(block_index[axis_position]))


def _positive_label_counts(block: np.ndarray) -> dict[int, int]:
    positive = np.asarray(block)[np.asarray(block) > 0]
    if positive.size == 0:
        return {}
    values, counts = np.unique(positive, return_counts=True)
    return {int(value): int(count) for value, count in zip(values, counts, strict=True)}


def _positive_pair_counts(
    first: np.ndarray,
    second: np.ndarray,
) -> dict[tuple[int, int], int]:
    mask = (first > 0) & (second > 0)
    if not np.any(mask):
        return {}
    pairs = zip(
        first[mask].astype(int).flat,
        second[mask].astype(int).flat,
        strict=True,
    )
    return dict(Counter((int(left), int(right)) for left, right in pairs))


def _label_centroids(block: np.ndarray) -> dict[int, np.ndarray]:
    mask = block > 0
    if not np.any(mask):
        return {}
    coords = np.argwhere(mask).astype(np.float64)
    label_values = block[mask].astype(int)
    centroids: dict[int, np.ndarray] = {}
    for label_id in sorted(set(int(value) for value in label_values.flat)):
        centroids[int(label_id)] = coords[label_values == label_id].mean(axis=0)
    return centroids


def _costes_thresholds(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    intensity_max: float,
    max_iterations: int = 100,
    tolerance: float = 1.0,
) -> dict[str, float]:
    if channel_1.size < 2:
        raise ValueError("Costes thresholding requires at least two voxels.")
    mean_x, mean_y, var_xx, var_yy, var_xy = _channel_moments(
        channel_1,
        channel_2,
    )
    slope = _costes_regression_slope(var_xx, var_yy, var_xy)
    intercept = mean_y - slope * mean_x

    max_x = float(np.nanmax(channel_1))
    max_y = float(np.nanmax(channel_2))
    min_x = float(np.nanmin(channel_1))
    min_y = float(np.nanmin(channel_2))

    if -1.0 < slope < 1.0:
        threshold = abs(max_x + min_x) * 0.5
        previous_threshold = max_x

        def map_threshold(value: float) -> tuple[float, float]:
            return value, value * slope + intercept

    else:
        threshold = abs(max_y + min_y) * 0.5
        previous_threshold = max_y

        def map_threshold(value: float) -> tuple[float, float]:
            return (value - intercept) / slope, value

    diff = abs(threshold - previous_threshold)
    pearson_below = float("nan")
    iterations = 0
    threshold_1 = threshold_2 = 0.0
    while iterations <= max_iterations and diff >= tolerance:
        threshold_1, threshold_2 = map_threshold(threshold)
        threshold_1 = _clamp(threshold_1, 0.0, intensity_max)
        threshold_2 = _clamp(threshold_2, 0.0, intensity_max)
        pearson_below = _pearson_below_threshold(
            channel_1,
            channel_2,
            threshold_1,
            threshold_2,
        )
        previous_threshold, old_diff = threshold, diff
        if np.isfinite(pearson_below) and pearson_below > 0:
            threshold -= old_diff * 0.5
        else:
            threshold += old_diff * 0.5
        diff = abs(threshold - previous_threshold)
        iterations += 1

    threshold_1, threshold_2 = map_threshold(threshold)
    threshold_1 = float(_round_and_clamp(threshold_1, 0.0, intensity_max))
    threshold_2 = float(_round_and_clamp(threshold_2, 0.0, intensity_max))
    pearson_below = _pearson_below_threshold(
        channel_1,
        channel_2,
        threshold_1,
        threshold_2,
    )
    return {
        "threshold_1": threshold_1,
        "threshold_2": threshold_2,
        "slope": float(slope),
        "intercept": float(intercept),
        "pearson_below": float(pearson_below),
        "iterations": int(iterations),
    }


def _channel_moments(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
) -> tuple[float, float, float, float, float]:
    n = float(channel_1.size)
    sum_x = float(np.sum(channel_1, dtype=np.float64))
    sum_y = float(np.sum(channel_2, dtype=np.float64))
    mean_x = sum_x / n
    mean_y = sum_y / n
    sum_xx = float(np.sum(channel_1 * channel_1, dtype=np.float64))
    sum_yy = float(np.sum(channel_2 * channel_2, dtype=np.float64))
    sum_xy = float(np.sum(channel_1 * channel_2, dtype=np.float64))
    var_xx = max(sum_xx / n - mean_x * mean_x, 0.0)
    var_yy = max(sum_yy / n - mean_y * mean_y, 0.0)
    var_xy = sum_xy / n - mean_x * mean_y
    return mean_x, mean_y, var_xx, var_yy, var_xy


def _costes_regression_slope(var_xx: float, var_yy: float, var_xy: float) -> float:
    eps = np.finfo(float).eps
    if abs(var_xy) <= eps:
        if var_xx > eps and var_yy > eps:
            return float(np.sqrt(var_yy / var_xx))
        return 1.0
    delta = var_yy - var_xx
    root = float(np.sqrt(delta * delta + 4.0 * var_xy * var_xy))
    slope = (delta + root) / (2.0 * var_xy)
    if not np.isfinite(slope) or abs(slope) <= eps:
        raise ValueError("Could not fit a Costes regression slope.")
    return float(slope)


def _pearson_below_threshold(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    threshold_1: float,
    threshold_2: float,
) -> float:
    mask = (channel_1 < threshold_1) | (channel_2 < threshold_2)
    count = int(np.count_nonzero(mask))
    if count < 2:
        return float("nan")
    inv_count = 1.0 / count
    sum_x = float(np.sum(channel_1, where=mask, dtype=np.float64))
    sum_y = float(np.sum(channel_2, where=mask, dtype=np.float64))
    sum_xx = float(np.sum(channel_1 * channel_1, where=mask, dtype=np.float64))
    sum_yy = float(np.sum(channel_2 * channel_2, where=mask, dtype=np.float64))
    sum_xy = float(np.sum(channel_1 * channel_2, where=mask, dtype=np.float64))
    numerator = sum_xy - sum_x * sum_y * inv_count
    denom_x = sum_xx - sum_x * sum_x * inv_count
    denom_y = sum_yy - sum_y * sum_y * inv_count
    denominator = denom_x * denom_y
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(numerator / np.sqrt(denominator))


def _pearson(channel_1: np.ndarray, channel_2: np.ndarray) -> float:
    x = np.asarray(channel_1, dtype=np.float64).ravel()
    y = np.asarray(channel_2, dtype=np.float64).ravel()
    if x.size < 2 or y.size < 2:
        return float("nan")
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= np.finfo(float).eps:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _overlap_coefficient(channel_1: np.ndarray, channel_2: np.ndarray) -> float:
    x = np.asarray(channel_1, dtype=np.float64).ravel()
    y = np.asarray(channel_2, dtype=np.float64).ravel()
    if x.size == 0 or y.size == 0:
        return float("nan")
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    if denom <= np.finfo(float).eps:
        return float("nan")
    return float(np.sum(x * y) / denom)


def _safe_fraction(numerator, denominator) -> float:
    denominator = float(denominator)
    if denominator <= np.finfo(float).eps:
        return float("nan")
    return float(numerator) / denominator


def _format_coloc_max(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{float(value):g}"


def _coloc_voxel_overlay(
    ch1: np.ndarray,
    ch2: np.ndarray,
    *,
    threshold_1: float,
    threshold_2: float,
    display_mode: str,
    channel_1_color: str,
    channel_2_color: str,
    intensity_max: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    if roi_mask is None:
        roi_mask = np.ones(ch1.shape, dtype=bool)
    mask = (ch1 >= threshold_1) & (ch2 >= threshold_2) & roi_mask
    ch1_unit = np.clip(ch1 / float(intensity_max), 0.0, 1.0)
    ch2_unit = np.clip(ch2 / float(intensity_max), 0.0, 1.0)
    color_1 = _coloc_color(channel_1_color, fallback="red")
    color_2 = _coloc_color(channel_2_color, fallback="green")
    colored = (
        ch1_unit[..., np.newaxis] * color_1
        + ch2_unit[..., np.newaxis] * color_2
    )
    colored = np.clip(colored, 0.0, 1.0).astype(np.float32, copy=False)
    colored[~roi_mask] = 0.0

    mode = str(display_mode).strip().lower()
    if "white on black" in mode:
        output = np.zeros(ch1.shape + (3,), dtype=np.float32)
        output[mask] = (1.0, 1.0, 1.0)
        return output
    if "channel colors only" in mode:
        output = np.zeros(ch1.shape + (3,), dtype=np.float32)
        output[mask] = colored[mask]
        return output

    output = colored.copy()
    output[mask] = (1.0, 1.0, 1.0)
    return output


def _coloc_color(value: str, *, fallback: str) -> np.ndarray:
    color = color_value_to_rgb(value)
    if color is None:
        color = color_value_to_rgb(fallback)
    if color is None:
        color = np.ones(3, dtype=np.float32)
    return np.asarray(color, dtype=np.float32)


def _coloc_scatter_plot_image(
    ch1: np.ndarray,
    ch2: np.ndarray,
    *,
    threshold_1: float,
    threshold_2: float,
    bins: int,
    log_counts: bool,
    intensity_max: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    bins = int(np.clip(int(bins), 32, 512))
    values_1 = ch1[roi_mask] if roi_mask is not None else np.ravel(ch1)
    values_2 = ch2[roi_mask] if roi_mask is not None else np.ravel(ch2)
    hist, _, _ = np.histogram2d(
        np.ravel(values_1),
        np.ravel(values_2),
        bins=bins,
        range=((0.0, float(intensity_max)), (0.0, float(intensity_max))),
    )
    counts = hist.T
    if bool(log_counts):
        counts = np.log1p(counts)
    max_count = float(np.max(counts))
    values = counts / max_count if max_count > 0 else counts
    values = np.flipud(values)

    image = np.zeros((bins, bins, 3), dtype=np.float32)
    image[..., 2] = 0.20 + 0.65 * values
    image[..., 1] = 0.08 + 0.90 * values
    image[..., 0] = 0.75 * np.sqrt(values)
    image[values <= 0] = (0.02, 0.03, 0.07)

    x_pos = int(
        np.clip(
            round(threshold_1 / float(intensity_max) * (bins - 1)),
            0,
            bins - 1,
        )
    )
    y_pos = bins - 1 - int(
        np.clip(round(threshold_2 / float(intensity_max) * (bins - 1)), 0, bins - 1)
    )
    image[:, x_pos : x_pos + 1] = (1.0, 0.20, 0.20)
    image[y_pos : y_pos + 1, :] = (0.20, 1.0, 0.20)
    image[max(y_pos - 1, 0) : y_pos + 2, max(x_pos - 1, 0) : x_pos + 2] = (
        1.0,
        1.0,
        1.0,
    )

    if image.shape[:2] != (512, 512):
        image = transform.resize(
            image,
            (512, 512, 3),
            order=0,
            preserve_range=True,
            anti_aliasing=False,
        )
    return image.astype(np.float32, copy=False)


def _racc_index_image(
    ch1: np.ndarray,
    ch2: np.ndarray,
    *,
    threshold_1: float,
    threshold_2: float,
    theta_degrees: float,
    include_percentile: float,
    intensity_max: float,
    roi_mask: np.ndarray | None = None,
) -> np.ndarray:
    theta_degrees = float(theta_degrees)
    include_percentile = float(include_percentile)
    if not 0 <= theta_degrees < 90:
        raise ValueError("RACC theta must satisfy 0 <= theta < 90.")
    if not 0 < include_percentile <= 100:
        raise ValueError("RACC included percentile must satisfy 0 < p <= 100.")

    if roi_mask is None:
        roi_mask = np.ones(ch1.shape, dtype=bool)
    mask = (ch1 >= threshold_1) & (ch2 >= threshold_2) & roi_mask
    overlap_voxels = int(np.count_nonzero(mask))
    if overlap_voxels < 2:
        raise ValueError(
            "Fewer than two voxels pass both channel thresholds; RACC is undefined."
        )

    x = ch1[mask].astype(np.float64, copy=False)
    y = ch2[mask].astype(np.float64, copy=False)
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    covariance = np.cov(x, y)
    var_xx = float(covariance[0, 0])
    var_yy = float(covariance[1, 1])
    var_xy = float(covariance[0, 1])
    slope = _positive_deming_slope(var_xx, var_yy, var_xy)
    intercept = mean_y - slope * mean_x

    p0 = _line_threshold_intersection(threshold_1, threshold_2, slope, intercept)
    p1 = _line_intensity_intersection(float(intensity_max), slope, intercept)
    points = np.column_stack((x, y))
    t_full = _projection_fraction(points, p0, p1)
    distances = _normalized_line_distance(points, p0, p1, float(intensity_max))

    percentile_fraction = include_percentile / 100.0
    t_max = float(np.quantile(t_full, percentile_fraction))
    if not np.isfinite(t_max):
        raise ValueError("Could not calculate RACC pmax from the overlap population.")
    t_max = max(t_max, np.finfo(float).eps)
    pmax = (p0[0] + t_max * (p1[0] - p0[0]), p0[1] + t_max * (p1[1] - p0[1]))
    distance_threshold = float(np.quantile(distances, percentile_fraction))
    return _calculate_racc_index(
        ch1,
        ch2,
        mask,
        p0,
        pmax,
        distance_threshold,
        theta_degrees,
        float(intensity_max),
    )


def _positive_deming_slope(var_xx: float, var_yy: float, var_xy: float) -> float:
    eps = np.finfo(float).eps
    if abs(var_xy) <= eps:
        if var_xx > eps and var_yy > eps:
            return float(np.sqrt(var_yy / var_xx))
        return 1.0
    delta = var_yy - var_xx
    root = float(np.sqrt(delta * delta + 4.0 * var_xy * var_xy))
    if var_xy >= 0:
        slope = (delta + root) / (2.0 * var_xy)
    else:
        slope = (delta - root) / (2.0 * var_xy)
    if not np.isfinite(slope) or slope <= 0:
        raise ValueError("Could not fit a positive RACC regression slope.")
    return float(slope)


def _line_threshold_intersection(
    threshold_1: float,
    threshold_2: float,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    y_at_threshold_1 = threshold_1 * slope + intercept
    if threshold_2 <= y_at_threshold_1:
        return (float(threshold_1), float(y_at_threshold_1))
    return (float((threshold_2 - intercept) / slope), float(threshold_2))


def _line_intensity_intersection(
    intensity_max: float,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    if intercept >= intensity_max * (1.0 - slope):
        return (float((intensity_max - intercept) / slope), float(intensity_max))
    return (float(intensity_max), float(intensity_max * slope + intercept))


def _projection_fraction(
    points: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float64)
    vector = np.asarray(p1, dtype=np.float64) - p0_arr
    denom = float(np.dot(vector, vector))
    if denom <= np.finfo(float).eps:
        raise ValueError("Regression line has degenerate endpoints.")
    return ((points - p0_arr) @ vector) / denom


def _normalized_line_distance(
    points: np.ndarray,
    p0: tuple[float, float],
    p1: tuple[float, float],
    intensity_max: float,
) -> np.ndarray:
    p0_arr = np.asarray(p0, dtype=np.float64)
    vector = np.asarray(p1, dtype=np.float64) - p0_arr
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        raise ValueError("Regression line has degenerate endpoints.")
    relative = points - p0_arr
    distance = np.abs(vector[0] * relative[:, 1] - vector[1] * relative[:, 0]) / norm
    return distance / intensity_max


def _calculate_racc_index(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    mask: np.ndarray,
    p0: tuple[float, float],
    pmax: tuple[float, float],
    distance_threshold: float,
    theta_degrees: float,
    intensity_max: float,
) -> np.ndarray:
    output = np.zeros(channel_1.shape, dtype=np.float32)
    if not np.any(mask):
        return output
    x = channel_1[mask].astype(np.float64, copy=False)
    y = channel_2[mask].astype(np.float64, copy=False)
    points = np.column_stack((x, y))
    t = _projection_fraction(points, p0, pmax)
    distances = _normalized_line_distance(points, p0, pmax, intensity_max)
    theta = np.deg2rad(theta_degrees)
    values = np.minimum(t, 1.0) - distances * np.tan(theta)
    values[(t <= 0.0) | (distances > distance_threshold)] = 0.0
    values = np.clip(values, 0.0, 1.0)
    output[mask] = values.astype(np.float32, copy=False)
    return output


def _clamp(value: float, lower: float, upper: float) -> float:
    return float(min(max(value, lower), upper))


def _round_and_clamp(value: float, lower: float, upper: float) -> int:
    return int(_clamp(np.floor(value + 0.5), lower, upper))


def _default_rgb_channel_indices(
    count: int,
    *,
    channel_axis_semantics: str = "",
) -> tuple[int | None, int | None, int | None]:
    if _is_true_rgb_channel_axis(
        count,
        channel_axis_semantics=channel_axis_semantics,
    ):
        return (0, 1, 2)
    if count >= 3:
        return (2, 1, 0)
    if count == 2:
        return (None, 1, 0)
    return (0, 0, 0)


def _is_true_rgb_channel_axis(
    count: int,
    *,
    channel_axis_semantics: str = "",
) -> bool:
    semantic = str(channel_axis_semantics or "").lower()
    return bool(
        (semantic == "rgb" and count == 3)
        or (semantic == "rgba" and count == 4)
    )


def _composite_to_rgb_by_color_table(
    moved: np.ndarray,
    color_table: np.ndarray,
    *,
    intensity_mapping: str,
    output_dtype: np.dtype,
) -> np.ndarray:
    _validate_composite_integer_blend_range(
        moved,
        color_table,
        intensity_mapping=intensity_mapping,
    )
    additive_components = np.count_nonzero(color_table[:, :3], axis=0) > 1
    accumulator_dtype = output_dtype
    if (
        intensity_mapping == COMPOSITE_RGB_PRESERVE_VALUES
        and bool(np.any(additive_components))
    ):
        accumulator_dtype = np.dtype(np.result_type(output_dtype, np.float64))
    rgb = np.zeros(moved.shape[:-1] + (3,), dtype=accumulator_dtype)
    try:
        with np.errstate(over="raise", invalid="raise"):
            for channel in range(moved.shape[-1]):
                values = _composite_mapped_channel(
                    moved[..., channel],
                    intensity_mapping=intensity_mapping,
                    output_dtype=accumulator_dtype,
                )
                for component, weight in enumerate(color_table[channel, :3]):
                    if float(weight) != 0.0:
                        rgb[..., component] += values * float(weight)
    except FloatingPointError as exc:
        raise ValueError(
            "Composite → RGB arithmetic overflowed the available finite "
            "floating-point range. Rescale the input explicitly before "
            "compositing."
        ) from exc
    if not np.isfinite(rgb).all():
        raise ValueError(
            "Composite → RGB produced a non-finite value. Rescale the input "
            "explicitly before compositing."
        )
    if intensity_mapping == COMPOSITE_RGB_PERCENTILE_1_99:
        return np.clip(rgb, 0, 1).astype(np.float32, copy=False)
    return rgb


def _validate_composite_integer_blend_range(
    moved: np.ndarray,
    color_table: np.ndarray,
    *,
    intensity_mapping: str,
) -> None:
    if (
        intensity_mapping != COMPOSITE_RGB_PRESERVE_VALUES
        or moved.dtype.kind not in {"i", "u"}
        or moved.size == 0
    ):
        return
    maxima: list[int] = []
    for channel in range(moved.shape[-1]):
        values = moved[..., channel]
        minimum = int(values.min())
        maximum = int(values.max())
        maxima.append(max(abs(minimum), abs(maximum)))
    for component in range(3):
        bound = math.fsum(
            maxima[channel] * abs(float(color_table[channel, component]))
            for channel in range(moved.shape[-1])
        )
        if bound > 2**53:
            raise ValueError(
                "Composite → RGB preserve-mode additive mixing could exceed "
                "the exact float64 integer range. Rescale or convert the data "
                "explicitly before compositing."
            )
