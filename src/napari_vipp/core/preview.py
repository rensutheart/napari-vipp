"""Preview reduction and thumbnail helpers."""

from __future__ import annotations

import numpy as np

from napari_vipp.core.channel_colors import (
    NAMED_CHANNEL_COLORS,
    channel_color_table,
)
from napari_vipp.core.metadata import ImageState

RGB_CHANNELS = (3, 4)
THUMBNAIL_PERCENTILE_RANGE = (0.5, 99.9)
MONOCHROME_COLORMAPS = (
    "Gray",
    "Viridis",
    "Magma",
    "Inferno",
    "Plasma",
    "Cividis",
    "Green",
    "Magenta",
    "Cyan",
    "Yellow",
    "Red",
    "Blue",
)
THUMBNAIL_CONTRAST_MODES = ("Percentile", "Min-max", "Raw")
THUMBNAIL_CONTRAST_SCOPES = ("Stack", "Slice")


def make_preview(
    data,
    mode: str = "slice",
    current_step=None,
    current_step_nsteps=None,
    state: ImageState | None = None,
    channel_colors: str | list[str] | tuple[str, ...] | None = None,
    contrast_mode: str = "Percentile",
    contrast_scope: str = "Slice",
    contrast_limits=None,
) -> np.ndarray | None:
    """Reduce arbitrary image-like data to a 2D or RGB thumbnail source array."""
    if data is None or mode.lower() == "off":
        return None

    arr = np.asarray(data)
    if arr.size == 0:
        return None

    mode = mode.lower()
    if state is not None:
        state_preview = _state_aware_preview(
            arr,
            mode,
            current_step,
            current_step_nsteps,
            state,
            channel_colors=channel_colors,
            contrast_mode=contrast_mode,
            contrast_scope=contrast_scope,
            contrast_limits=contrast_limits,
        )
        if state_preview is not None:
            return state_preview
    if mode == "mip":
        return _mip(arr)
    return _slice(
        arr,
        current_step=current_step,
        current_step_nsteps=current_step_nsteps,
    )


def normalize_thumbnail(data, size: tuple[int, int] = (180, 110)) -> np.ndarray | None:
    """Convert preview data to uint8 RGB for display."""
    return normalize_thumbnail_with_colormap(
        data,
        size=size,
        colormap="Gray",
        contrast_mode="Percentile",
    )


def normalize_thumbnail_with_colormap(
    data,
    size: tuple[int, int] = (180, 110),
    *,
    colormap: str = "Gray",
    contrast_mode: str = "Percentile",
    contrast_reference=None,
    contrast_limits=None,
    data_kind: str = "image",
) -> np.ndarray | None:
    """Convert preview data to uint8 RGB for display."""
    if data is None:
        return None

    arr = np.asarray(data)
    if arr.size == 0:
        return None

    if str(data_kind or "").lower() in {"label", "labels", "label image"}:
        arr = _slice(arr)
        if arr.ndim != 2:
            return None
        return _resize_nearest(_label_thumbnail_rgb(arr), size)

    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255

    if arr.ndim == 2:
        gray = _normalize_uint8(
            arr,
            contrast_mode=contrast_mode,
            reference=contrast_reference,
            contrast_limits=contrast_limits,
        )
        rgb = _apply_monochrome_colormap(gray, colormap)
    elif arr.ndim == 3 and arr.shape[-1] in RGB_CHANNELS:
        rgb = _normalize_rgb(
            arr[..., :3],
            contrast_mode=contrast_mode,
            reference=contrast_reference,
            contrast_limits=contrast_limits,
        )
    else:
        gray = _normalize_uint8(
            _slice(arr),
            contrast_mode=contrast_mode,
            reference=contrast_reference,
            contrast_limits=contrast_limits,
        )
        rgb = _apply_monochrome_colormap(gray, colormap)

    return _resize_nearest(rgb, size)


def thumbnail_contrast_limits(
    data,
    *,
    contrast_mode: str = "Percentile",
    data_kind: str = "image",
) -> tuple[float, float] | None:
    """Return reusable thumbnail contrast limits for an image-like array."""
    if data is None or str(data_kind or "").lower() in {
        "label",
        "labels",
        "label image",
        "table",
    }:
        return None
    arr = np.asarray(data)
    if arr.size == 0:
        return (0.0, 0.0)

    mode = _contrast_mode_key(contrast_mode)
    if mode == "raw":
        if arr.dtype == bool or np.issubdtype(arr.dtype, np.integer):
            return None
        values = arr.astype(np.float32, copy=False)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 0.0)
        return (float(finite.min()), float(finite.max()))

    values = arr.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 0.0)
    if float(finite.min()) < 0.0:
        values = np.clip(values, 0.0, None)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return (0.0, 0.0)
    if mode == "minmax":
        lo = float(finite.min())
        hi = float(finite.max())
    else:
        lo, hi = (
            float(value)
            for value in np.percentile(finite, THUMBNAIL_PERCENTILE_RANGE)
        )
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    return (lo, hi)


def thumbnail_channel_contrast_limits(
    data,
    *,
    channel_axis: int,
    channel_count: int | None = None,
    contrast_mode: str = "Percentile",
    data_kind: str = "image",
) -> tuple[tuple[float, float] | None, ...]:
    """Return reusable per-channel thumbnail contrast limits."""
    arr = np.asarray(data)
    if arr.size == 0 or arr.ndim == 0:
        return ()
    axis = int(channel_axis)
    if axis < 0:
        axis += arr.ndim
    if axis < 0 or axis >= arr.ndim:
        return ()
    count = arr.shape[axis] if channel_count is None else int(channel_count)
    count = max(0, min(count, arr.shape[axis]))
    return tuple(
        thumbnail_contrast_limits(
            np.take(arr, channel, axis=axis),
            contrast_mode=contrast_mode,
            data_kind=data_kind,
        )
        for channel in range(count)
    )


def _slice(arr: np.ndarray, current_step=None, current_step_nsteps=None) -> np.ndarray:
    arr = _strip_rgb_safe_singletons(arr)
    while arr.ndim > 3 or (arr.ndim == 3 and arr.shape[-1] not in RGB_CHANNELS):
        axis = 0
        index = _axis_index(
            axis,
            arr.shape[axis],
            current_step,
            current_step_nsteps=current_step_nsteps,
        )
        arr = np.take(arr, index, axis=axis)

    if arr.ndim == 3 and arr.shape[-1] not in RGB_CHANNELS:
        index = _axis_index(
            0,
            arr.shape[0],
            current_step,
            current_step_nsteps=current_step_nsteps,
        )
        arr = arr[index]
    return arr


def _state_aware_preview(
    arr: np.ndarray,
    mode: str,
    current_step,
    current_step_nsteps,
    state: ImageState,
    *,
    channel_colors: str | list[str] | tuple[str, ...] | None = None,
    contrast_mode: str = "Percentile",
    contrast_scope: str = "Slice",
    contrast_limits=None,
) -> np.ndarray | None:
    if len(state.axes) != arr.ndim:
        return None

    channel_axis = _axis_index_by_type(state, "channel")
    y_axis = _axis_index_by_name(state, "y")
    x_axis = _axis_index_by_name(state, "x")
    if y_axis is None or x_axis is None:
        return None

    if channel_axis is not None:
        reduced = _reduce_to_axes(
            arr,
            state,
            keep_axes={channel_axis, y_axis, x_axis},
            mode=mode,
            current_step=current_step,
            current_step_nsteps=current_step_nsteps,
        )
        if reduced.ndim != 3:
            return None
        remaining_axes = _remaining_axis_indices(
            state,
            keep_axes={channel_axis, y_axis, x_axis},
        )
        local_channel = remaining_axes.index(channel_axis)
        local_y = remaining_axes.index(y_axis)
        local_x = remaining_axes.index(x_axis)
        reduced = np.moveaxis(reduced, [local_y, local_x, local_channel], [0, 1, 2])
        axis_name = state.axes[channel_axis].name.lower()
        if axis_name in {"rgb", "rgba"} and not channel_colors:
            return reduced
        metadata_colors = tuple(channel.color for channel in state.channels)
        reference = (
            reduced
            if _contrast_scope_key(contrast_scope) == "slice"
            else None
            if contrast_limits is not None
            else _channel_last_reference(arr, channel_axis)
        )
        return _fluorescence_composite(
            reduced,
            channel_colors=channel_colors,
            metadata_colors=metadata_colors,
            contrast_mode=contrast_mode,
            reference=reference,
            contrast_limits=contrast_limits,
        )

    reduced = _reduce_to_axes(
        arr,
        state,
        keep_axes={y_axis, x_axis},
        mode=mode,
        current_step=current_step,
        current_step_nsteps=current_step_nsteps,
    )
    if reduced.ndim != 2:
        return None
    remaining_axes = _remaining_axis_indices(state, keep_axes={y_axis, x_axis})
    local_y = remaining_axes.index(y_axis)
    local_x = remaining_axes.index(x_axis)
    return np.moveaxis(reduced, [local_y, local_x], [0, 1])


def _reduce_to_axes(
    arr: np.ndarray,
    state: ImageState,
    *,
    keep_axes: set[int],
    mode: str,
    current_step,
    current_step_nsteps,
) -> np.ndarray:
    result = arr
    remaining = list(range(arr.ndim))
    for original_axis in reversed(range(arr.ndim)):
        if original_axis in keep_axes:
            continue
        local_axis = remaining.index(original_axis)
        axis = state.axes[original_axis]
        if mode == "mip" and axis.type == "space":
            result = np.max(result, axis=local_axis)
        else:
            step_axis = _current_step_axis(state, original_axis, current_step)
            index = _axis_index(
                step_axis,
                result.shape[local_axis],
                current_step,
                current_step_nsteps=current_step_nsteps,
            )
            result = np.take(result, index, axis=local_axis)
        remaining.pop(local_axis)
    return result


def _remaining_axis_indices(state: ImageState, *, keep_axes: set[int]) -> list[int]:
    return [index for index in range(len(state.axes)) if index in keep_axes]


def _axis_index_by_type(state: ImageState, axis_type: str) -> int | None:
    for index, axis in enumerate(state.axes):
        if axis.type == axis_type:
            return index
    return None


def _axis_index_by_name(state: ImageState, name: str) -> int | None:
    for index, axis in enumerate(state.axes):
        if axis.name.lower() == name:
            return index
    return None


def _current_step_axis(
    state: ImageState,
    axis_index: int,
    current_step,
) -> int:
    """Map a derived output axis back to the viewer axis that controls it."""
    try:
        current_ndim = len(tuple(current_step))
    except Exception:
        current_ndim = 0
    try:
        source_axis = state.axes[axis_index].source_axis
    except Exception:
        source_axis = None
    if source_axis is not None and 0 <= int(source_axis) < current_ndim:
        return int(source_axis)
    state_ndim = len(getattr(state, "axes", ()))
    if current_ndim == state_ndim and current_ndim > 0:
        # When viewer dims already match this state, map positionally.
        return axis_index
    return axis_index


def _fluorescence_composite(
    arr: np.ndarray,
    *,
    channel_colors: str | list[str] | tuple[str, ...] | None = None,
    metadata_colors: tuple[int | None, ...] = (),
    contrast_mode: str = "Percentile",
    reference=None,
    contrast_limits=None,
) -> np.ndarray:
    color_table = channel_color_table(
        channel_colors,
        arr.shape[-1],
        metadata_colors=metadata_colors,
    )
    channels = []
    for channel in range(arr.shape[-1]):
        channel_reference = _channel_reference(reference, channel)
        normalized = (
            _normalize_uint8(
                arr[..., channel],
                contrast_mode=contrast_mode,
                reference=channel_reference,
                contrast_limits=_channel_contrast_limits(contrast_limits, channel),
            ).astype(
                np.float32,
            )
            / 255.0
        )
        color = color_table[channel]
        channels.append(normalized[..., None] * color)
    if not channels:
        return np.zeros(arr.shape[:2] + (3,), dtype=np.float32)
    return np.clip(np.sum(channels, axis=0), 0, 1)


def _mip(arr: np.ndarray) -> np.ndarray:
    arr = _strip_rgb_safe_singletons(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[-1] in RGB_CHANNELS:
        return arr
    if arr.ndim == 3:
        return np.max(arr, axis=0)
    if arr.ndim >= 4 and arr.shape[-1] in RGB_CHANNELS:
        axes = tuple(range(arr.ndim - 3))
        if axes:
            arr = np.take(arr, 0, axis=0) if len(axes) > 1 else arr
        while arr.ndim > 3:
            arr = np.max(arr, axis=0)
        return arr
    while arr.ndim > 2:
        arr = np.max(arr, axis=0)
    return arr


def _axis_index(
    axis: int,
    axis_size: int,
    current_step=None,
    *,
    current_step_nsteps=None,
) -> int:
    if current_step is None:
        return axis_size // 2
    try:
        step = int(tuple(current_step)[axis])
    except Exception:
        step = axis_size // 2
    if current_step_nsteps is not None:
        try:
            source_nsteps = int(tuple(current_step_nsteps)[axis])
        except Exception:
            source_nsteps = 0
        # Keep exact napari index when source and target axis sizes match.
        if source_nsteps == int(axis_size):
            return max(0, min(axis_size - 1, step))
        source_max = max(source_nsteps - 1, 0)
        target_max = max(int(axis_size) - 1, 0)
        if source_max > 0 and target_max > 0:
            ratio = float(np.clip(step, 0, source_max)) / float(source_max)
            step = int(round(ratio * target_max))
    return max(0, min(axis_size - 1, step))


def _strip_rgb_safe_singletons(arr: np.ndarray) -> np.ndarray:
    while arr.ndim > 2:
        if arr.ndim >= 3 and arr.shape[-1] in RGB_CHANNELS:
            break
        singleton_axes = [i for i, size in enumerate(arr.shape) if size == 1]
        if not singleton_axes:
            break
        arr = np.squeeze(arr, axis=singleton_axes[0])
    return arr


def _normalize_uint8(
    arr: np.ndarray,
    *,
    contrast_mode: str = "Percentile",
    reference=None,
    contrast_limits=None,
) -> np.ndarray:
    mode = _contrast_mode_key(contrast_mode)
    if mode == "raw":
        return _raw_uint8(arr, reference=reference, contrast_limits=contrast_limits)

    values = arr.astype(np.float32, copy=False)
    limits = _coerce_contrast_limits(contrast_limits)
    if limits is not None:
        lo, hi = limits
    else:
        reference_values = (
            values
            if reference is None
            else np.asarray(reference).astype(np.float32, copy=False)
        )
        finite = reference_values[np.isfinite(reference_values)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.uint8)
        # Signed difference images (e.g. Subtract) carry negative values around a
        # zero background. Stretching from the negative minimum would render that
        # background as mid-grey and saturate positive features to white. Anchor
        # the black point at zero so the thumbnail matches the inspector view,
        # where the zero background renders black.
        if float(finite.min()) < 0.0:
            values = np.clip(values, 0.0, None)
            reference_values = np.clip(reference_values, 0.0, None)
            finite = reference_values[np.isfinite(reference_values)]
        if mode == "minmax":
            lo = float(finite.min())
            hi = float(finite.max())
        else:
            lo, hi = (
                float(value)
                for value in np.percentile(finite, THUMBNAIL_PERCENTILE_RANGE)
            )
        if hi <= lo:
            hi = float(finite.max())
            lo = float(finite.min())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lo) / (hi - lo)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _normalize_rgb(
    arr: np.ndarray,
    *,
    contrast_mode: str = "Percentile",
    reference=None,
    contrast_limits=None,
) -> np.ndarray:
    values = arr.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if (
        contrast_limits is None
        and finite.size
        and float(finite.min()) >= 0.0
        and float(finite.max()) <= 1.0
    ):
        return (np.clip(values, 0, 1)[..., :3] * 255).astype(np.uint8)

    channels = [
        _normalize_uint8(
            arr[..., i],
            contrast_mode=contrast_mode,
            reference=_rgb_channel_reference(reference, i),
            contrast_limits=_channel_contrast_limits(contrast_limits, i),
        )
        for i in range(min(3, arr.shape[-1]))
    ]
    while len(channels) < 3:
        channels.append(channels[-1] if channels else np.zeros(arr.shape[:2], np.uint8))
    return np.stack(channels[:3], axis=-1)


def _contrast_mode_key(contrast_mode: str) -> str:
    text = str(contrast_mode or "").strip().lower()
    if text in {"min-max", "minmax", "minimum-maximum", "minimum maximum"}:
        return "minmax"
    if text == "raw":
        return "raw"
    return "percentile"


def _contrast_scope_key(contrast_scope: str) -> str:
    text = str(contrast_scope or "").strip().lower()
    return "slice" if text.startswith("slice") else "stack"


def _channel_last_reference(arr: np.ndarray, channel_axis: int) -> np.ndarray:
    if channel_axis == arr.ndim - 1:
        return arr
    return np.moveaxis(arr, channel_axis, -1)


def _channel_reference(reference, channel: int):
    if reference is None:
        return None
    values = np.asarray(reference)
    if values.ndim >= 3 and values.shape[-1] > channel:
        return values[..., channel]
    return None


def _rgb_channel_reference(reference, channel: int):
    if reference is None:
        return None
    values = np.asarray(reference)
    if (
        values.ndim >= 3
        and values.shape[-1] in RGB_CHANNELS
        and values.shape[-1] > channel
    ):
        return values[..., channel]
    return None


def _channel_contrast_limits(contrast_limits, channel: int):
    if contrast_limits is None:
        return None
    if _coerce_contrast_limits(contrast_limits) is not None:
        return contrast_limits
    try:
        return contrast_limits[channel]
    except Exception:
        return None


def _coerce_contrast_limits(contrast_limits) -> tuple[float, float] | None:
    if contrast_limits is None:
        return None
    try:
        lo = float(contrast_limits[0])
        hi = float(contrast_limits[1])
    except Exception:
        return None
    if not np.isfinite(lo) or not np.isfinite(hi):
        return None
    return (lo, hi)


def _raw_uint8(
    arr: np.ndarray,
    *,
    reference=None,
    contrast_limits=None,
) -> np.ndarray:
    source = np.asarray(arr)
    if source.dtype == bool:
        return source.astype(np.uint8) * 255
    if np.issubdtype(source.dtype, np.integer):
        values = source.astype(np.float32, copy=False)
        values = np.nan_to_num(values, nan=0.0, posinf=255.0, neginf=0.0)
        info = np.iinfo(source.dtype)
        if info.min < 0:
            scale = float(info.max - info.min)
            if scale <= 0:
                return np.zeros(values.shape, dtype=np.uint8)
            scaled = (values - float(info.min)) / scale
        else:
            scaled = values / float(info.max)
        return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)
    values = source.astype(np.float32, copy=False)
    limits = _coerce_contrast_limits(contrast_limits)
    if limits is None:
        reference_values = (
            values
            if reference is None
            else np.asarray(reference).astype(np.float32, copy=False)
        )
        finite = reference_values[np.isfinite(reference_values)]
        if finite.size == 0:
            return np.zeros(values.shape, dtype=np.uint8)
        finite_min = float(finite.min())
        finite_max = float(finite.max())
    else:
        finite_min, finite_max = limits
    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=finite_max,
        neginf=finite_min,
    )
    if finite_min >= 0.0:
        if finite_max <= 0.0:
            return np.zeros(values.shape, dtype=np.uint8)
        scaled = values / finite_max
    else:
        scale = finite_max - finite_min
        if scale <= 0.0:
            return np.zeros(values.shape, dtype=np.uint8)
        scaled = (values - finite_min) / scale
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _label_thumbnail_rgb(arr: np.ndarray) -> np.ndarray:
    labels = np.asarray(arr)
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    mask = labels != 0
    if not np.any(mask):
        return rgb

    label_values = np.abs(labels[mask].astype(np.int64, copy=False))
    rgb[..., 0][mask] = ((label_values * 53) % 200 + 55).astype(np.uint8)
    rgb[..., 1][mask] = ((label_values * 97) % 200 + 55).astype(np.uint8)
    rgb[..., 2][mask] = ((label_values * 193) % 200 + 55).astype(np.uint8)
    return rgb


def _apply_monochrome_colormap(gray: np.ndarray, colormap: str) -> np.ndarray:
    values = gray.astype(np.float32, copy=False) / 255.0
    name = str(colormap or "Gray").strip().lower()
    if name in {"gray", "grey"}:
        return np.stack([gray, gray, gray], axis=-1)
    if name in NAMED_CHANNEL_COLORS:
        color = NAMED_CHANNEL_COLORS[name]
        rgb = values[..., None] * color
        return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    stops = _COLORMAP_STOPS.get(name, _COLORMAP_STOPS["viridis"])
    positions = stops[:, 0]
    channels = [
        np.interp(values, positions, stops[:, channel]).astype(np.float32)
        for channel in (1, 2, 3)
    ]
    rgb = np.stack(channels, axis=-1)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


_COLORMAP_STOPS = {
    "viridis": np.asarray(
        [
            [0.0, 0.267, 0.005, 0.329],
            [0.25, 0.230, 0.322, 0.546],
            [0.5, 0.128, 0.567, 0.551],
            [0.75, 0.369, 0.789, 0.383],
            [1.0, 0.993, 0.906, 0.144],
        ],
        dtype=np.float32,
    ),
    "magma": np.asarray(
        [
            [0.0, 0.001, 0.000, 0.014],
            [0.25, 0.316, 0.071, 0.486],
            [0.5, 0.716, 0.215, 0.475],
            [0.75, 0.986, 0.535, 0.382],
            [1.0, 0.987, 0.991, 0.749],
        ],
        dtype=np.float32,
    ),
    "inferno": np.asarray(
        [
            [0.0, 0.001, 0.000, 0.014],
            [0.25, 0.342, 0.062, 0.429],
            [0.5, 0.735, 0.216, 0.330],
            [0.75, 0.978, 0.557, 0.034],
            [1.0, 0.988, 0.998, 0.645],
        ],
        dtype=np.float32,
    ),
    "plasma": np.asarray(
        [
            [0.0, 0.050, 0.030, 0.528],
            [0.25, 0.495, 0.012, 0.658],
            [0.5, 0.798, 0.280, 0.470],
            [0.75, 0.973, 0.586, 0.252],
            [1.0, 0.940, 0.975, 0.131],
        ],
        dtype=np.float32,
    ),
    "cividis": np.asarray(
        [
            [0.0, 0.000, 0.135, 0.305],
            [0.25, 0.264, 0.307, 0.423],
            [0.5, 0.489, 0.485, 0.471],
            [0.75, 0.736, 0.681, 0.424],
            [1.0, 0.996, 0.909, 0.218],
        ],
        dtype=np.float32,
    ),
}


def _resize_nearest(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    target_w, target_h = size
    h, w = arr.shape[:2]
    if h <= 0 or w <= 0:
        return arr
    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    ys = np.linspace(0, h - 1, new_h).astype(np.intp)
    xs = np.linspace(0, w - 1, new_w).astype(np.intp)
    return arr[ys][:, xs]
