"""Prototype image-processing operations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy import ndimage as ndi
from skimage import filters, restoration

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


def contrast_stretch(data, alpha: float = 3.0, beta: float = 1.0) -> np.ndarray:
    """Apply a linear scale-and-offset contrast operation."""
    arr = np.asarray(data)
    scaled = arr.astype(np.float32, copy=False) * float(alpha) + float(beta)
    return np.clip(scaled, 0, 255).astype(np.uint8)


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
) -> np.ndarray:
    """Apply Gaussian blur across the z/y/x volume axes."""
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

    return _apply_plane_wise(arr, filter_plane)


def otsu_threshold(data) -> np.ndarray:
    """Return a binary mask from a slice-wise Otsu threshold."""
    arr = _to_grayscale(np.asarray(data))
    return _global_threshold(arr, _otsu_value)


def triangle_threshold(data) -> np.ndarray:
    """Return a binary mask from a slice-wise triangle threshold."""
    arr = _to_grayscale(np.asarray(data))

    def threshold(plane: np.ndarray) -> float:
        values = plane[np.isfinite(plane)]
        if values.size == 0 or values.min() == values.max():
            return float(values.mean()) if values.size else 0.0
        try:
            return float(filters.threshold_triangle(values))
        except Exception:
            return float(values.mean())

    return _global_threshold(arr, threshold)


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


def fill_holes(data) -> np.ndarray:
    """Fill holes in a binary mask volume."""
    return ndi.binary_fill_holes(_to_bool_mask(data))


def volume_filter(data, min_volume: int = 10) -> np.ndarray:
    """Remove connected components below a minimum voxel/pixel count."""
    mask = _to_bool_mask(data)
    labeled, count = ndi.label(mask)
    if count == 0:
        return mask.copy()
    sizes = np.bincount(labeled.ravel())
    keep = sizes >= max(int(min_volume), 1)
    keep[0] = False
    return keep[labeled]


def extract_channel(data, channel: int = 0) -> np.ndarray:
    """Extract a single channel from a channel-last image or stack."""
    arr = np.asarray(data)
    return _extract_channel(
        arr,
        channel=channel,
        channel_axis=_default_channel_axis(arr),
    )


def channel_composite(
    data,
    channel_axis: int = 0,
    red_channel: int = 2,
    green_channel: int = 1,
    blue_channel: int = 0,
) -> np.ndarray:
    """Create a channel-last RGB composite from a multi-channel image."""
    arr = np.asarray(data)
    if arr.ndim < 3:
        gray = _to_float_unit(arr)
        return np.stack([gray, gray, gray], axis=-1).astype(np.float32)

    channel_axis = _normalize_axis(channel_axis, arr.ndim)
    channels = [
        _extract_channel(arr, channel=red_channel, channel_axis=channel_axis),
        _extract_channel(arr, channel=green_channel, channel_axis=channel_axis),
        _extract_channel(arr, channel=blue_channel, channel_axis=channel_axis),
    ]
    return np.stack(
        [_composite_channel_to_float(channel) for channel in channels],
        axis=-1,
    ).astype(np.float32)


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


def max_intensity_projection(data, axis: int = 0) -> np.ndarray:
    """Project an image along one axis using maximum intensity."""
    arr = np.asarray(data)
    if arr.ndim <= 2:
        return arr.copy()
    axis = int(axis)
    if axis < 0 or axis >= arr.ndim:
        axis = 0
    return np.max(arr, axis=axis)


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
) -> np.ndarray:
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
    if np.issubdtype(original.dtype, np.integer):
        info = np.iinfo(original.dtype)
        return np.clip(values, info.min, info.max).astype(original.dtype)
    return values.astype(np.float32)


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
    lo = float(finite.min())
    hi = float(finite.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0, 1).astype(np.float32)
