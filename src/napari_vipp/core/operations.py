"""Prototype image-processing operations."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage import (
    feature,
    filters,
    measure,
    morphology,
    restoration,
    segmentation,
    transform,
)

from napari_vipp.core.channel_colors import channel_color_table
from napari_vipp.core.io import write_image
from napari_vipp.core.tables import TableData, table_from_columns

RGB_CHANNELS = (3, 4)


def crop_stack(
    data,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
) -> np.ndarray:
    """Crop the last two spatial axes of an image or stack."""
    arr = np.asarray(data)
    if arr.ndim < 2:
        return arr.copy()

    y_axis, x_axis = _xy_axes(arr)
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
        return arr.copy()
    return np.ascontiguousarray(np.transpose(arr, indices))


def linear_scale_offset(data, alpha: float = 3.0, beta: float = 1.0) -> np.ndarray:
    """Apply a linear scale-and-offset contrast operation."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return arr.copy()
    scaled = arr.astype(np.float32, copy=False) * float(alpha) + float(beta)
    return _restore_numeric_dtype(scaled, arr)


def gamma_correction(data, gamma: float = 0.5) -> np.ndarray:
    """Apply power-law gamma correction."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return arr.copy()

    gamma = max(float(gamma), 1e-6)
    scale = _intensity_scale(arr)
    corrected = np.power(np.clip(arr.astype(np.float32) / scale, 0, 1), gamma) * scale
    return _restore_numeric_dtype(corrected, arr)


def average_blur(data, size: int = 5) -> np.ndarray:
    """Apply a slice-wise mean blur over the x/y plane."""
    arr = _float_if_bool(np.asarray(data))
    size = max(int(size), 1)
    filter_size = [1] * arr.ndim
    for axis in _xy_axes(arr):
        filter_size[axis] = size
    return ndi.uniform_filter(arr, size=filter_size)


def gaussian_blur(data, sigma: float = 1.0) -> np.ndarray:
    """Apply slice-wise Gaussian blur while preserving channel and z axes."""
    arr = _float_if_bool(np.asarray(data))
    sigma = max(float(sigma), 0.0)
    if sigma == 0:
        return arr.copy()

    sigma_by_axis = [0.0] * arr.ndim
    for axis in _xy_axes(arr):
        sigma_by_axis[axis] = sigma
    return ndi.gaussian_filter(arr, sigma=sigma_by_axis)


def gaussian_blur_3d(
    data,
    sigma_z: float = 2.0,
    sigma_y: float = 2.0,
    sigma_x: float = 2.0,
    lock_xy: bool = True,
) -> np.ndarray:
    """Apply Gaussian blur across the z/y/x volume axes."""
    del lock_xy  # UI convenience flag; sigma_y/sigma_x carry the actual values.
    arr = _float_if_bool(np.asarray(data))
    spatial_axes = _spatial_axes(arr)
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


def median_filter(data, size: int = 5) -> np.ndarray:
    """Apply a slice-wise median filter over the x/y plane."""
    arr = np.asarray(data)
    size = _odd_size(size, minimum=1)
    filter_size = [1] * arr.ndim
    for axis in _xy_axes(arr):
        filter_size[axis] = size
    return ndi.median_filter(arr, size=filter_size)


def bilateral_filter(
    data,
    diameter: int = 5,
    sigma_color: float = 5.0,
    sigma_space: float = 5.0,
) -> np.ndarray:
    """Apply a slice-wise bilateral denoising filter."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        arr = arr.astype(np.float32)
    sigma_color = _scaled_sigma_color(arr, sigma_color)
    sigma_space = max(float(sigma_space), 0.01)

    def filter_plane(plane: np.ndarray) -> np.ndarray:
        plane_float = _to_float_unit(plane)
        channel_axis = -1 if _has_channel_axis(plane_float) else None
        return restoration.denoise_bilateral(
            plane_float,
            win_size=_odd_size(diameter, minimum=3),
            sigma_color=sigma_color,
            sigma_spatial=sigma_space,
            channel_axis=channel_axis,
        ).astype(np.float32)

    return _restore_unit_float_dtype(_apply_plane_wise(arr, filter_plane), arr)


def difference_of_gaussians_filter(
    data,
    low_sigma: float = 1.0,
    high_sigma: float = 3.0,
) -> np.ndarray:
    """Enhance structures between two Gaussian blur scales slice-wise."""
    original = np.asarray(data)
    dtype = original.dtype if np.issubdtype(original.dtype, np.floating) else np.float32
    arr = _float_if_bool(original).astype(dtype, copy=False)
    low_sigma = max(float(low_sigma), 0.0)
    high_sigma = max(float(high_sigma), low_sigma + 1e-6)
    low = gaussian_blur(arr, sigma=low_sigma)
    high = gaussian_blur(arr, sigma=high_sigma)
    return low.astype(dtype, copy=False) - high.astype(dtype, copy=False)


def unsharp_mask_filter(
    data,
    radius: float = 1.0,
    amount: float = 1.0,
) -> np.ndarray:
    """Sharpen image detail using skimage's unsharp mask per plane."""
    arr = np.asarray(data)
    radius = max(float(radius), 0.0)
    amount = max(float(amount), 0.0)

    def sharpen_plane(plane: np.ndarray) -> np.ndarray:
        plane_float = _to_float_unit(plane)
        channel_axis = -1 if _has_channel_axis(plane_float) else None
        return filters.unsharp_mask(
            plane_float,
            radius=radius,
            amount=amount,
            channel_axis=channel_axis,
            preserve_range=False,
        ).astype(np.float32)

    return _restore_unit_float_dtype(_apply_plane_wise(arr, sharpen_plane), arr)


def rolling_ball_background(
    data,
    radius: float = 50.0,
    light_background: bool = False,
    disable_smoothing: bool = False,
    spatial_mode: str = "2D YX",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Estimate smooth background with a Fiji/ImageJ-style rolling ball."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return np.zeros_like(arr)
    background = _estimate_rolling_ball_background(
        arr,
        radius=radius,
        light_background=light_background,
        disable_smoothing=disable_smoothing,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
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
) -> np.ndarray:
    """Subtract a rolling-ball background estimate while preserving dtype."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return arr.copy()
    background = _estimate_rolling_ball_background(
        arr,
        radius=radius,
        light_background=light_background,
        disable_smoothing=disable_smoothing,
        spatial_mode=spatial_mode,
        resolved_spatial_ndim=resolved_spatial_ndim,
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
) -> np.ndarray:
    """Denoise image planes using skimage non-local means."""
    arr = np.asarray(data)
    patch_size = _odd_size(patch_size, minimum=3)
    patch_distance = max(int(patch_distance), 1)
    h = max(float(h), 0.0)

    def denoise_plane(plane: np.ndarray) -> np.ndarray:
        plane_float = _to_float_unit(plane)
        channel_axis = -1 if _has_channel_axis(plane_float) else None
        return restoration.denoise_nl_means(
            plane_float,
            patch_size=patch_size,
            patch_distance=patch_distance,
            h=h,
            fast_mode=bool(fast_mode),
            channel_axis=channel_axis,
        ).astype(np.float32)

    return _restore_unit_float_dtype(_apply_plane_wise(arr, denoise_plane), arr)


def sobel_filter(data) -> np.ndarray:
    """Return the slice-wise Sobel edge magnitude of a grayscale image."""
    original = np.asarray(data)
    arr = _to_grayscale(original)
    result = _apply_plane_wise(
        arr,
        lambda plane: filters.sobel(plane.astype(np.float32, copy=False)),
    )
    return _restore_numeric_dtype(result, original)


def laplace_filter(data, kernel_size: int = 3) -> np.ndarray:
    """Return a slice-wise Laplace edge/detail response."""
    original = np.asarray(data)
    arr = _to_grayscale(original)
    kernel_size = _odd_size(kernel_size, minimum=3)
    result = _apply_plane_wise(
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
) -> np.ndarray:
    """Return a slice-wise Canny edge mask."""
    arr = _to_grayscale(np.asarray(data))
    low, high = _ordered_threshold_pair(low_quantile, high_quantile)
    low = float(np.clip(low, 0.0, 1.0))
    high = float(np.clip(high, 0.0, 1.0))
    sigma = max(float(sigma), 0.0)

    def canny_plane(plane: np.ndarray) -> np.ndarray:
        values = plane.astype(np.float32, copy=False)
        return feature.canny(
            values,
            sigma=sigma,
            low_threshold=low,
            high_threshold=high,
            use_quantiles=True,
        ).astype(bool)

    return _apply_plane_wise(arr, canny_plane).astype(bool)


def hysteresis_threshold(
    data,
    low_threshold: float = 0.25,
    high_threshold: float = 0.7,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Return a binary mask from connected low/high intensity thresholds."""
    arr = _to_grayscale(np.asarray(data))
    low, high = _ordered_threshold_pair(low_threshold, high_threshold)
    spatial_ndim = _resolved_spatial_ndim(
        arr,
        spatial_mode,
        resolved_spatial_ndim,
    )

    def threshold_block(block: np.ndarray) -> np.ndarray:
        values = block.astype(np.float32, copy=False)
        return filters.apply_hysteresis_threshold(values, low, high).astype(bool)

    return _apply_spatial_blocks(
        arr,
        spatial_ndim,
        threshold_block,
        dtype=bool,
    )


def otsu_threshold(data, threshold_scope: str = "Stack histogram") -> np.ndarray:
    """Return a binary mask from an Otsu threshold."""
    arr = _to_grayscale(np.asarray(data))
    return _global_threshold(arr, _otsu_value, threshold_scope=threshold_scope)


def triangle_threshold(data, threshold_scope: str = "Stack histogram") -> np.ndarray:
    """Return a binary mask from a triangle threshold."""
    arr = _to_grayscale(np.asarray(data))
    return _global_threshold(arr, _triangle_value, threshold_scope=threshold_scope)


def li_threshold(data, threshold_scope: str = "Stack histogram") -> np.ndarray:
    """Return a binary mask from a Li threshold."""
    arr = _to_grayscale(np.asarray(data))
    return _global_threshold(
        arr,
        lambda plane: _safe_threshold(plane, filters.threshold_li),
        threshold_scope=threshold_scope,
    )


def yen_threshold(data, threshold_scope: str = "Stack histogram") -> np.ndarray:
    """Return a binary mask from a Yen threshold."""
    arr = _to_grayscale(np.asarray(data))
    return _global_threshold(
        arr,
        lambda plane: _safe_threshold(plane, filters.threshold_yen),
        threshold_scope=threshold_scope,
    )


def isodata_threshold(data, threshold_scope: str = "Stack histogram") -> np.ndarray:
    """Return a binary mask from an Isodata threshold."""
    arr = _to_grayscale(np.asarray(data))
    return _global_threshold(
        arr,
        lambda plane: _safe_threshold(plane, filters.threshold_isodata),
        threshold_scope=threshold_scope,
    )


def minimum_threshold(data, threshold_scope: str = "Stack histogram") -> np.ndarray:
    """Return a binary mask from a minimum threshold."""
    arr = _to_grayscale(np.asarray(data))
    return _global_threshold(
        arr,
        lambda plane: _safe_threshold(plane, filters.threshold_minimum),
        threshold_scope=threshold_scope,
    )


def automatic_threshold_value(data, operation_id: str) -> float | None:
    """Return the scalar threshold value used by a global threshold operation."""
    threshold_func = _automatic_threshold_function(operation_id)
    if threshold_func is None:
        return None
    arr = _to_grayscale(np.asarray(data))
    return float(threshold_func(arr))


def binary_threshold(data, threshold: float = 0.5) -> np.ndarray:
    """Return a binary mask using a fixed intensity threshold."""
    arr = _to_grayscale(np.asarray(data))
    return arr > float(threshold)


def adaptive_mean_threshold(
    data,
    block_size: int = 11,
    c: float = 2.0,
) -> np.ndarray:
    """Return a binary mask using local mean thresholding."""
    return _adaptive_threshold(data, block_size=block_size, c=c, method="mean")


def adaptive_gaussian_threshold(
    data,
    block_size: int = 11,
    c: float = 2.0,
) -> np.ndarray:
    """Return a binary mask using local Gaussian thresholding."""
    return _adaptive_threshold(data, block_size=block_size, c=c, method="gaussian")


def sauvola_threshold(
    data,
    window_size: int = 15,
    k: float = 0.2,
    dynamic_range: float = 0.0,
) -> np.ndarray:
    """Return a binary mask using local Sauvola thresholding."""
    arr = _to_grayscale(np.asarray(data))

    def threshold_plane(plane: np.ndarray) -> np.ndarray:
        window = _odd_size(window_size, minimum=3, maximum=min(plane.shape[-2:]))
        if window < 3:
            return plane > float(np.mean(plane))
        values = plane.astype(np.float32, copy=False)
        r = float(dynamic_range)
        if r <= 0.0:
            finite = values[np.isfinite(values)]
            r = float(np.ptp(finite) / 2.0) if finite.size else 1.0
            r = max(r, 1e-6)
        local = filters.threshold_sauvola(values, window_size=window, k=float(k), r=r)
        return values > local

    return _apply_plane_wise(arr, threshold_plane).astype(bool)


def niblack_threshold(
    data,
    window_size: int = 15,
    k: float = 0.2,
) -> np.ndarray:
    """Return a binary mask using local Niblack thresholding."""
    arr = _to_grayscale(np.asarray(data))

    def threshold_plane(plane: np.ndarray) -> np.ndarray:
        window = _odd_size(window_size, minimum=3, maximum=min(plane.shape[-2:]))
        if window < 3:
            return plane > float(np.mean(plane))
        values = plane.astype(np.float32, copy=False)
        local = filters.threshold_niblack(values, window_size=window, k=float(k))
        return values > local

    return _apply_plane_wise(arr, threshold_plane).astype(bool)


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
        finite = values[np.isfinite(values)]
        if finite.size == 0 or float(finite.min()) == float(finite.max()):
            return np.zeros(values.shape, dtype=np.int32)
        safe = np.nan_to_num(
            values,
            nan=float(finite.min()),
            posinf=float(finite.max()),
            neginf=float(finite.min()),
        )
        if height > 0:
            peaks = morphology.h_maxima(safe, height)
        else:
            peaks = morphology.local_maxima(safe)
        peaks &= safe > float(finite.min())
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
        return morphology.skeletonize(block, method=method_value).astype(bool)

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
    iterations: int = 1,
    remove_isolated: bool = True,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
) -> np.ndarray:
    """Remove short terminal skeleton branches within each spatial block."""
    mask = _to_bool_mask(data)
    spatial_ndim = _resolved_spatial_ndim(
        mask,
        spatial_mode,
        resolved_spatial_ndim,
    )

    def prune_block(block: np.ndarray) -> np.ndarray:
        return _prune_skeleton_branches_block(
            block,
            min_branch_length=float(min_branch_length),
            iterations=int(iterations),
            remove_isolated=bool(remove_isolated),
        )

    return _apply_spatial_blocks(mask, spatial_ndim, prune_block, dtype=bool)


def measure_objects(
    data,
    spatial_mode: str = "Auto from axes",
    measurement_set: str = "Basic morphology",
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
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
        spatial_ndim=spatial_ndim,
    )
    for leading_index in np.ndindex(leading_shape or (1,)):
        block_index = () if not leading_shape else leading_index
        block = labels_for_measure[block_index] if leading_shape else labels_for_measure
        block_columns = _measure_label_block(
            block,
            spatial_axis_names,
            spatial_ndim,
            include_shape_descriptors=include_shape_descriptors,
            include_axis_descriptors=include_axis_descriptors,
            include_2d_boundary_descriptors=include_2d_boundary_descriptors,
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
            include_shape_descriptors=include_shape_descriptors,
            include_axis_descriptors=include_axis_descriptors,
            include_2d_boundary_descriptors=include_2d_boundary_descriptors,
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
        spatial_ndim=spatial_ndim,
    )
    return table_from_columns(
        columns,
        name="Object intensity measurements",
        table_kind=table_kind,
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
    spatial_axes = _measurement_spatial_axes(mask.ndim, spatial_ndim, axis_types)
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
    spatial_axes = _measurement_spatial_axes(mask.ndim, spatial_ndim, axis_types)
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
    requested = _parse_table_column_list(columns)
    if not requested:
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
    if not selected:
        raise ValueError("Select Table Columns would remove every table column.")
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


def extract_channel(data, channel: int = 0) -> np.ndarray:
    """Extract a single channel from a channel-last image or stack."""
    arr = np.asarray(data)
    return _extract_channel(
        arr,
        channel=channel,
        channel_axis=_default_channel_axis(arr),
    )


def combine_channels(
    inputs,
    input_count: int = 2,
    channel_axis: int = -1,
    channel_colors: str = "",
) -> np.ndarray:
    """Combine multiple same-shaped inputs into a multichannel image."""
    arrays = [np.asarray(item) for item in inputs if item is not None]
    input_count = int(np.clip(int(input_count), 1, len(arrays))) if arrays else 0
    arrays = arrays[:input_count]
    if not arrays:
        raise ValueError("Combine Channels needs at least one connected input.")

    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("Combine Channels inputs must have matching shapes.")

    axis = int(channel_axis)
    if axis < 0:
        axis = 0
    axis = int(np.clip(axis, 0, arrays[0].ndim))
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
    """Create a new float image from weighted same-shaped inputs."""
    arrays = [np.asarray(item) for item in inputs if item is not None]
    input_count = int(np.clip(int(input_count), 1, len(arrays))) if arrays else 0
    arrays = arrays[:input_count]
    if not arrays:
        raise ValueError("Image Calculator needs at least one connected input.")

    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError("Image Calculator inputs must have matching shapes.")

    parsed_weights = _parse_float_list(weights)
    if len(parsed_weights) < len(arrays):
        parsed_weights.extend([1.0] * (len(arrays) - len(parsed_weights)))

    result = np.zeros(shape, dtype=np.float32)
    for array, weight in zip(arrays, parsed_weights, strict=False):
        result += array.astype(np.float32, copy=False) * float(weight)
    result += float(offset)
    return result


def composite_to_rgb(
    data,
    channel_axis: int = -1,
    red_channel: int = -1,
    green_channel: int = -1,
    blue_channel: int = -1,
    channel_axis_semantics: str = "",
    channel_colors: str | list[int | str] | tuple[int | str, ...] = "",
) -> np.ndarray:
    """Convert a multichannel composite into a channel-last RGB image.

    By default (all parameters ``-1``) the channel axis is detected automatically.
    True channel-last RGB/RGBA inputs keep RGB order. Other multichannel inputs
    use VIPP's fluorescence display order so channel 0 maps to blue, channel 1
    maps to green, and channel 2 maps to red. A single channel is written to all
    three (white/grayscale) and extra channels are ignored.

    The mapping is configurable: set ``channel_axis`` to pick the channel axis
    explicitly, and set ``red_channel``/``green_channel``/``blue_channel`` to
    choose which channel index feeds each RGB plane. Any selection that is
    ``-1`` falls back to the positional default, and selections outside the
    available channel range leave that plane blank.
    """
    arr = np.asarray(data)
    if int(channel_axis) < 0:
        axis = _default_channel_axis(arr)
    else:
        axis = _normalize_axis(int(channel_axis), arr.ndim)
    if axis is None:
        gray = _composite_channel_to_float(arr)
        return np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    moved = np.moveaxis(arr, axis, -1)
    count = moved.shape[-1]
    if count == 1:
        gray = _composite_channel_to_float(moved[..., 0])
        return np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    manual_mapping = any(
        int(selection) >= 0 for selection in (red_channel, green_channel, blue_channel)
    )
    true_rgb = _is_true_rgb_channel_axis(
        arr,
        axis,
        count,
        channel_axis_semantics=channel_axis_semantics,
    )
    if not manual_mapping and not true_rgb:
        color_table = channel_color_table(channel_colors, count)
        return _composite_to_rgb_by_color_table(moved, color_table)

    defaults = _default_rgb_channel_indices(
        arr,
        axis,
        count,
        channel_axis_semantics=channel_axis_semantics,
    )
    selections = (red_channel, green_channel, blue_channel)
    blank = np.zeros(moved.shape[:-1], dtype=np.float32)
    channels = []
    for position, selection in enumerate(selections):
        index = defaults[position] if int(selection) < 0 else int(selection)
        if 0 <= index < count:
            channels.append(_composite_channel_to_float(moved[..., index]))
        else:
            channels.append(blank)
    return np.stack(channels, axis=-1).astype(np.float32)


def split_channels(data, preview_channel: int = 0) -> list[np.ndarray]:
    """Split a multi-channel image into one output array per channel.

    The channel axis is detected automatically and each output is a single
    channel with the spatial shape of the input. The number of outputs equals
    the true number of channels in the image. A grayscale image with no
    detected channel axis yields a single output.

    ``preview_channel`` is a UI-only hint used by the graph thumbnail and does
    not affect the returned channel outputs.
    """
    arr = np.asarray(data)
    axis = _default_channel_axis(arr)
    if axis is None:
        return [arr.copy()]

    moved = np.moveaxis(arr, axis, -1)
    count = moved.shape[-1]
    return [moved[..., index].copy() for index in range(count)]



def rescale_intensity(
    data,
    in_low_percentile: float = 0.0,
    in_high_percentile: float = 100.0,
    out_min: float = 0.0,
    out_max: float = 1.0,
    in_low_value: float | None = None,
    in_high_value: float | None = None,
) -> np.ndarray:
    """Rescale intensity from input cutoffs to a requested output range."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return arr.copy()
    values = arr[np.isfinite(arr)]
    if values.size == 0:
        return np.zeros_like(arr)
    value_cutoffs = _rescale_value_cutoffs(in_low_value, in_high_value)
    if value_cutoffs is None:
        low_p = float(np.clip(in_low_percentile, 0.0, 100.0))
        high_p = float(np.clip(in_high_percentile, 0.0, 100.0))
        if low_p > high_p:
            low_p, high_p = high_p, low_p
        low, high = np.percentile(values, [low_p, high_p])
    else:
        low, high = value_cutoffs
    if high == low:
        return _cast_rescaled_intensity(
            np.full(arr.shape, float(out_min), dtype=np.float64),
            arr.dtype,
        )
    scaled = (arr.astype(np.float64, copy=False) - float(low)) / float(high - low)
    scaled = np.clip(scaled, 0.0, 1.0)
    output = scaled * (float(out_max) - float(out_min)) + float(out_min)
    return _cast_rescaled_intensity(output, arr.dtype)


def _rescale_value_cutoffs(low, high) -> tuple[float, float] | None:
    if low is None or high is None:
        return None
    try:
        low = float(low)
        high = float(high)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(low) or not np.isfinite(high):
        return None
    if low > high:
        low, high = high, low
    return low, high


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
) -> np.ndarray:
    """Clip intensities while preserving the incoming dtype where possible."""
    arr = np.asarray(data)
    if arr.dtype == bool:
        return arr.copy()
    low = float(minimum)
    high = float(maximum)
    if low > high:
        low, high = high, low
    clipped = np.clip(arr, low, high)
    if np.issubdtype(arr.dtype, np.integer):
        return clipped.astype(arr.dtype)
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
) -> np.ndarray:
    """Apply a binary mask to an image, filling outside-mask pixels."""
    arrays = [np.asarray(item) for item in inputs if item is not None]
    if len(arrays) < 2:
        raise ValueError("Mask Image needs an image input and a mask input.")

    image = arrays[0]
    mask = _to_bool_mask(arrays[1])
    if str(invert_mask).lower() == "yes":
        mask = ~mask
    mask = _broadcast_mask_to_image(mask, image.shape)
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
    input_count = int(np.clip(int(input_count), 1, len(arrays))) if arrays else 0
    arrays = arrays[:input_count]
    if not arrays:
        raise ValueError(f"{operation_name} needs at least one connected input.")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError(f"{operation_name} inputs must have matching shapes.")
    return arrays


def _broadcast_mask_to_image(
    mask: np.ndarray,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    """Return a boolean mask broadcast to an image shape."""
    mask = np.asarray(mask, dtype=bool)
    if mask.shape == image_shape:
        return mask
    if mask.ndim > len(image_shape):
        raise ValueError(
            "Mask Image mask shape must match the image shape or be broadcastable "
            "to it."
        )

    priority_shapes: list[tuple[int, ...]] = []
    if len(image_shape) >= 3 and image_shape[-1] in RGB_CHANNELS:
        priority_shapes.append(mask.shape + (1,))
    priority_shapes.append((1,) * (len(image_shape) - mask.ndim) + mask.shape)

    for expanded_shape in priority_shapes:
        if len(expanded_shape) != len(image_shape):
            continue
        try:
            return np.broadcast_to(mask.reshape(expanded_shape), image_shape)
        except ValueError:
            continue

    inserted_axes = len(image_shape) - mask.ndim
    for singleton_axes in combinations(range(len(image_shape)), inserted_axes):
        singleton_axes_set = set(singleton_axes)
        expanded_shape: list[int] = []
        mask_axis = 0
        for axis in range(len(image_shape)):
            if axis in singleton_axes_set:
                expanded_shape.append(1)
            else:
                expanded_shape.append(mask.shape[mask_axis])
                mask_axis += 1
        try:
            return np.broadcast_to(mask.reshape(expanded_shape), image_shape)
        except ValueError:
            continue

    raise ValueError(
        "Mask Image mask shape must match the image shape or be broadcastable "
        "to it."
    )


def _extract_channel(
    arr: np.ndarray,
    channel: int = 0,
    channel_axis: int | None = None,
) -> np.ndarray:
    if channel_axis is None:
        return arr.copy()
    channel_axis = _normalize_axis(channel_axis, arr.ndim)
    channel = int(np.clip(channel, 0, arr.shape[channel_axis] - 1))
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

    if output_dtype == "bool":
        if arr.dtype == bool:
            return arr.copy()
        return _to_grayscale(arr) > 0

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
    if arr.ndim <= 2:
        return arr.copy()
    axis = int(axis)
    if axis < 0 or axis >= arr.ndim:
        axis = 0
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


def _adaptive_threshold(data, block_size: int, c: float, method: str) -> np.ndarray:
    arr = _to_grayscale(np.asarray(data))

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

    return _apply_plane_wise(arr, threshold_plane).astype(bool)


def _global_threshold(
    arr: np.ndarray,
    threshold_func: Callable[[np.ndarray], float],
    *,
    threshold_scope: str = "Stack histogram",
) -> np.ndarray:
    if str(threshold_scope).strip().lower().startswith("stack"):
        return arr > threshold_func(arr)

    def threshold_plane(plane: np.ndarray) -> np.ndarray:
        return plane > threshold_func(plane)

    return _apply_plane_wise(arr, threshold_plane).astype(bool)


def _to_grayscale(arr: np.ndarray) -> np.ndarray:
    if _has_channel_axis(arr):
        rgb = arr[..., :3].astype(np.float32)
        return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    return arr.astype(np.float32, copy=False)


def _to_bool_mask(data) -> np.ndarray:
    arr = np.asarray(data)
    if arr.dtype == bool:
        return arr.copy()
    return _to_grayscale(arr) > 0


def _target_dtype(name: str) -> np.dtype:
    choices = {
        "uint8": np.dtype(np.uint8),
        "uint16": np.dtype(np.uint16),
        "float32": np.dtype(np.float32),
    }
    return choices.get(name, np.dtype(np.uint8))


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
        scaled = values
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


def _otsu_value(arr: np.ndarray) -> float:
    values = arr[np.isfinite(arr)].astype(np.float64, copy=False)
    if values.size == 0:
        return 0.0
    if values.min() == values.max():
        return float(values.min())

    hist, edges = np.histogram(values, bins=256)
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * centers) / np.maximum(weight1, 1)
    mean2 = (
        np.cumsum((hist * centers)[::-1]) / np.maximum(weight2[::-1], 1)
    )[::-1]
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if variance12.size == 0:
        return float(values.mean())
    return float(centers[:-1][np.argmax(variance12)])


def _triangle_value(arr: np.ndarray) -> float:
    return _safe_threshold(arr, filters.threshold_triangle)


def _ordered_threshold_pair(low, high) -> tuple[float, float]:
    try:
        low_value = float(low)
    except Exception:
        low_value = 0.0
    try:
        high_value = float(high)
    except Exception:
        high_value = low_value
    if not np.isfinite(low_value):
        low_value = 0.0
    if not np.isfinite(high_value):
        high_value = low_value
    if high_value < low_value:
        low_value, high_value = high_value, low_value
    return low_value, high_value


def _safe_threshold(
    arr: np.ndarray,
    threshold_func: Callable[[np.ndarray], float],
) -> float:
    values = arr[np.isfinite(arr)].astype(np.float64, copy=False)
    if values.size == 0:
        return 0.0
    if values.min() == values.max():
        return float(values.min())
    try:
        return float(threshold_func(values))
    except Exception:
        return float(values.mean())


def _automatic_threshold_function(
    operation_id: str,
) -> Callable[[np.ndarray], float] | None:
    return {
        "otsu_threshold": _otsu_value,
        "triangle_threshold": _triangle_value,
        "li_threshold": lambda arr: _safe_threshold(arr, filters.threshold_li),
        "yen_threshold": lambda arr: _safe_threshold(arr, filters.threshold_yen),
        "isodata_threshold": lambda arr: _safe_threshold(
            arr,
            filters.threshold_isodata,
        ),
        "minimum_threshold": lambda arr: _safe_threshold(
            arr,
            filters.threshold_minimum,
        ),
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


def _apply_spatial_blocks(
    arr: np.ndarray,
    spatial_ndim: int,
    func: Callable[[np.ndarray], np.ndarray],
    *,
    dtype,
) -> np.ndarray:
    """Apply ``func`` independently over leading non-spatial dimensions."""
    spatial_ndim = int(np.clip(spatial_ndim, 1, max(arr.ndim, 1)))
    if arr.ndim <= spatial_ndim:
        return np.asarray(func(arr), dtype=dtype)

    result = np.empty(arr.shape, dtype=dtype)
    leading_shape = arr.shape[: arr.ndim - spatial_ndim]
    for index in np.ndindex(leading_shape):
        result[index] = func(arr[index])
    return result


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
    mode = str(spatial_mode).strip().lower()
    if mode.startswith("2d"):
        requested = 2
    elif mode.startswith("3d"):
        requested = 3
    elif resolved_spatial_ndim is not None:
        requested = int(resolved_spatial_ndim)
    else:
        requested = 3 if arr.ndim >= 3 else 2
    return int(np.clip(requested, 1, max(arr.ndim, 1)))


def _estimate_rolling_ball_background(
    arr: np.ndarray,
    *,
    radius: float,
    light_background: bool,
    disable_smoothing: bool,
    spatial_mode: str,
    resolved_spatial_ndim: int | None,
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
        return _rolling_ball_background_block(
            block,
            radius_pixels=radius_pixels,
            light_background=bool(light_background),
            disable_smoothing=bool(disable_smoothing),
            output_dtype=output_dtype,
        )

    if _has_channel_axis(arr):
        channels = [
            _apply_spatial_blocks(
                arr[..., channel],
                min(spatial_ndim, max(arr.ndim - 1, 1)),
                estimate,
                dtype=output_dtype,
            )
            for channel in range(arr.shape[-1])
        ]
        return np.stack(channels, axis=-1).astype(output_dtype, copy=False)
    return _apply_spatial_blocks(arr, spatial_ndim, estimate, dtype=output_dtype)


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
    unit_label: str
    column_units: dict[str, str]


@dataclass(frozen=True)
class _SkeletonUnits:
    length_column: str
    physical_column: str
    scales: tuple[float, ...]
    unit_label: str
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
) -> np.ndarray:
    skeleton = np.asarray(block, dtype=bool).copy()
    minimum = max(float(min_branch_length), 0.0)
    iterations = max(int(iterations), 1)
    for _iteration in range(iterations):
        coords, adjacency, degrees, _edge_count, _pixel_length, _physical_length = (
            _skeleton_adjacency(skeleton)
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
            if float(edge_count) >= minimum:
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
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
) -> dict[str, list[object]]:
    properties = _regionprops_measurement_properties(
        spatial_ndim,
        include_shape_descriptors=include_shape_descriptors,
        include_axis_descriptors=include_axis_descriptors,
        include_2d_boundary_descriptors=include_2d_boundary_descriptors,
    )
    raw = measure.regionprops_table(block, properties=properties)
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
    if intensity_block is not None:
        _add_intensity_measurements(result, block, intensity_block)
    return result


def _regionprops_measurement_properties(
    spatial_ndim: int,
    *,
    include_shape_descriptors: bool,
    include_axis_descriptors: bool,
    include_2d_boundary_descriptors: bool,
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
    return tuple(dict.fromkeys(properties))


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


def _size_descriptor_column(prefix: str, spatial_ndim: int) -> str:
    if spatial_ndim >= 3:
        return f"{prefix}_volume_voxels"
    if spatial_ndim == 2:
        return f"{prefix}_area_pixels"
    return f"{prefix}_length_pixels"


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


def _measurement_empty_columns(
    leading_axis_names: tuple[str, ...],
    spatial_axis_names: tuple[str, ...],
    units: _MeasurementUnits,
    *,
    include_intensity: bool = False,
    include_shape_descriptors: bool = False,
    include_axis_descriptors: bool = False,
    include_2d_boundary_descriptors: bool = False,
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
    columns["extent"] = []
    columns["euler_number"] = []
    if include_shape_descriptors:
        columns[_size_descriptor_column("bbox", spatial_ndim)] = []
        columns[_size_descriptor_column("filled", spatial_ndim)] = []
        if spatial_ndim == 2:
            columns["convex_area_pixels"] = []
            columns["solidity"] = []
            columns["feret_diameter_max_pixels"] = []
    if include_axis_descriptors and spatial_ndim >= 2:
        suffix = "voxels" if spatial_ndim >= 3 else "pixels"
        columns[f"major_axis_length_{suffix}"] = []
        columns[f"minor_axis_length_{suffix}"] = []
        for axis_index in range(spatial_ndim):
            columns[f"inertia_tensor_eigval_{axis_index}"] = []
        if spatial_ndim == 2:
            columns["eccentricity"] = []
            columns["orientation_radians"] = []
    if include_2d_boundary_descriptors and spatial_ndim == 2:
        columns["perimeter_pixels"] = []
        columns["perimeter_crofton_pixels"] = []
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
        length_unit = "voxels" if spatial_ndim >= 3 else "pixels"
        squared_unit = "voxels^2" if spatial_ndim >= 3 else "pixels^2"
        column_units[f"major_axis_length_{suffix}"] = length_unit
        column_units[f"minor_axis_length_{suffix}"] = length_unit
        for axis_index in range(spatial_ndim):
            column_units[f"inertia_tensor_eigval_{axis_index}"] = squared_unit
    column_units["solidity"] = "ratio"
    column_units["eccentricity"] = "ratio"
    column_units["orientation_radians"] = "radians"
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
    spatial_ndim: int,
) -> str:
    extras: list[str] = []
    if include_shape_descriptors:
        extras.append("shape descriptors")
    if include_axis_descriptors:
        extras.append("axis/inertia descriptors")
    if include_2d_boundary_descriptors and spatial_ndim == 2:
        extras.append("2D boundary descriptors")
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
    axis_types: tuple[str, ...],
) -> tuple[int, ...]:
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


def _xy_axes(arr: np.ndarray) -> tuple[int, int]:
    spatial_axes = _spatial_axes(arr)
    if len(spatial_axes) >= 2:
        return spatial_axes[-2], spatial_axes[-1]
    if len(spatial_axes) == 1:
        return spatial_axes[0], spatial_axes[0]
    return 0, 0


def _spatial_axes(arr: np.ndarray) -> list[int]:
    axes = list(range(arr.ndim))
    if _has_channel_axis(arr):
        axes.pop()
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


def _normalize_axis(axis: int, ndim: int) -> int:
    if ndim <= 0:
        return 0
    axis = int(axis)
    if axis < 0:
        axis += ndim
    return int(np.clip(axis, 0, ndim - 1))


def _slice_selections(
    arr: np.ndarray,
    axis: int,
    index: int,
    axes,
    indices,
    *,
    use_default: bool = True,
) -> dict[int, int]:
    selected_axes = _parse_int_list(axes)
    selected_indices = _parse_int_list(indices)
    if not selected_axes:
        if not use_default:
            return {}
        selected_axes = [axis]
        selected_indices = [index]

    selections: dict[int, int] = {}
    for position, axis_value in enumerate(selected_axes):
        axis_index = _normalize_axis(axis_value, arr.ndim)
        slice_index = (
            selected_indices[position]
            if position < len(selected_indices)
            else 0
        )
        selections[axis_index] = int(
            np.clip(int(slice_index), 0, max(arr.shape[axis_index] - 1, 0))
        )
    return selections


def _axis_range_selection(arr: np.ndarray, ranges) -> np.ndarray:
    parsed = _parse_axis_ranges(ranges, arr.shape)
    if not parsed:
        return arr.copy()
    slices = [slice(None)] * arr.ndim
    for axis, (start, end) in parsed.items():
        axis_index = _normalize_axis(axis, arr.ndim)
        maximum = max(arr.shape[axis_index] - 1, 0)
        start = int(np.clip(start, 0, maximum))
        end = int(np.clip(end, 0, maximum))
        if start > end:
            start, end = end, start
        slices[axis_index] = slice(start, end + 1)
    return np.ascontiguousarray(arr[tuple(slices)])


def _parse_axis_ranges(value, shape: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    if not isinstance(value, str) or not value.strip():
        return {}
    ranges: dict[int, tuple[int, int]] = {}
    for part in value.split(";"):
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) != 3:
            continue
        try:
            axis = _normalize_axis(int(pieces[0]), len(shape))
            start = int(pieces[1])
            end = int(pieces[2])
        except ValueError:
            continue
        ranges[axis] = (start, end)
    return ranges


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
    if ndim <= 2:
        return ()
    tokens = _projection_axis_tokens(axes)
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
        if index is not None and index not in parsed:
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
    if isinstance(value, (int, np.integer)):
        return f"axis:{int(value)}"
    text = str(value).strip().lower()
    if text in {"", "auto", "non_yx_spatial"}:
        return text
    if text.startswith(("axis:", "name:")):
        return text
    return ""


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
            return _normalize_axis(int(normalized.removeprefix("axis:")), ndim)
        except ValueError:
            return None
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
    aliases = {
        "µm": "micrometer",
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


def _parse_float_list(value) -> list[float]:
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
            parsed.append(float(part))
        except (TypeError, ValueError):
            continue
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
    first = max(int(first), 0)
    second = max(int(second), 0)
    if axis_size <= 1:
        return 0, 0
    if first >= axis_size:
        first = axis_size - 1
    max_second = axis_size - first - 1
    return first, min(second, max_second)


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


def _restore_unit_float_dtype(values: np.ndarray, original: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(
        np.clip(np.asarray(values), 0.0, 1.0),
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    if original.dtype == bool:
        return values > 0.5
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        scaled = values * float(info.max)
        return _restore_numeric_dtype(scaled, original)
    if np.issubdtype(original.dtype, np.floating):
        return values.astype(original.dtype, copy=False)
    return values.astype(original.dtype, copy=False)


def _to_float_unit(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == bool:
        return arr.astype(np.float32)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(np.float32) / float(np.iinfo(arr.dtype).max)
    values = arr.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        lo = float(finite.min())
        hi = float(finite.max())
        if hi > lo:
            values = (values - lo) / (hi - lo)
    return np.clip(values, 0, 1)


def _scaled_sigma_color(arr: np.ndarray, sigma_color: float) -> float:
    sigma = max(float(sigma_color), 0.001)
    if np.issubdtype(arr.dtype, np.integer):
        return sigma / float(np.iinfo(arr.dtype).max)
    return sigma


def _composite_channel_to_float(arr: np.ndarray) -> np.ndarray:
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


def _default_rgb_channel_indices(
    arr: np.ndarray,
    channel_axis: int,
    count: int,
    *,
    channel_axis_semantics: str = "",
) -> tuple[int, int, int]:
    if _is_true_rgb_channel_axis(
        arr,
        channel_axis,
        count,
        channel_axis_semantics=channel_axis_semantics,
    ):
        return (0, 1, 2)
    if count >= 3:
        return (2, 1, 0)
    if count == 2:
        return (-1, 1, 0)
    return (0, 0, 0)


def _is_true_rgb_channel_axis(
    arr: np.ndarray,
    channel_axis: int,
    count: int,
    *,
    channel_axis_semantics: str = "",
) -> bool:
    semantic = str(channel_axis_semantics or "").lower()
    is_declared_rgb = semantic in {"rgb", "rgba"}
    is_unlabelled_channel_last_rgb = (
        not semantic
        and channel_axis == arr.ndim - 1
        and count in RGB_CHANNELS
    )
    return bool(is_declared_rgb or is_unlabelled_channel_last_rgb)


def _composite_to_rgb_by_color_table(
    moved: np.ndarray,
    color_table: np.ndarray,
) -> np.ndarray:
    rgb = np.zeros(moved.shape[:-1] + (3,), dtype=np.float32)
    for channel in range(moved.shape[-1]):
        normalized = _composite_channel_to_float(moved[..., channel])
        rgb += normalized[..., None] * color_table[channel, :3]
    return np.clip(rgb, 0, 1).astype(np.float32)
