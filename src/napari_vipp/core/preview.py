"""Preview reduction and thumbnail helpers."""

from __future__ import annotations

import numpy as np

from napari_vipp.core.metadata import ImageState

RGB_CHANNELS = (3, 4)
FLUORESCENCE_COLORS = np.array(
    [
        [0.1, 0.35, 1.0],
        [0.1, 1.0, 0.2],
        [1.0, 0.15, 0.05],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
    ],
    dtype=np.float32,
)
NAMED_CHANNEL_COLORS = {
    "red": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "green": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "blue": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "magenta": np.array([1.0, 0.0, 1.0], dtype=np.float32),
    "cyan": np.array([0.0, 1.0, 1.0], dtype=np.float32),
    "yellow": np.array([1.0, 1.0, 0.0], dtype=np.float32),
}


def make_preview(
    data,
    mode: str = "slice",
    current_step=None,
    state: ImageState | None = None,
    channel_colors: str | list[str] | tuple[str, ...] | None = None,
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
            state,
            channel_colors=channel_colors,
        )
        if state_preview is not None:
            return state_preview
    if mode == "mip":
        return _mip(arr)
    return _slice(arr, current_step=current_step)


def normalize_thumbnail(data, size: tuple[int, int] = (180, 110)) -> np.ndarray | None:
    """Convert preview data to uint8 RGB for display."""
    if data is None:
        return None

    arr = np.asarray(data)
    if arr.size == 0:
        return None

    if arr.dtype == bool:
        arr = arr.astype(np.uint8) * 255

    if arr.ndim == 2:
        gray = _normalize_uint8(arr)
        rgb = np.stack([gray, gray, gray], axis=-1)
    elif arr.ndim == 3 and arr.shape[-1] in RGB_CHANNELS:
        rgb = _normalize_rgb(arr[..., :3])
    else:
        gray = _normalize_uint8(_slice(arr))
        rgb = np.stack([gray, gray, gray], axis=-1)

    return _resize_nearest(rgb, size)


def _slice(arr: np.ndarray, current_step=None) -> np.ndarray:
    arr = _strip_rgb_safe_singletons(arr)
    while arr.ndim > 3 or (arr.ndim == 3 and arr.shape[-1] not in RGB_CHANNELS):
        axis = 0
        index = _axis_index(axis, arr.shape[axis], current_step)
        arr = np.take(arr, index, axis=axis)

    if arr.ndim == 3 and arr.shape[-1] not in RGB_CHANNELS:
        index = _axis_index(0, arr.shape[0], current_step)
        arr = arr[index]
    return arr


def _state_aware_preview(
    arr: np.ndarray,
    mode: str,
    current_step,
    state: ImageState,
    *,
    channel_colors: str | list[str] | tuple[str, ...] | None = None,
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
        if channel_axis == arr.ndim - 1 and not channel_colors:
            return reduced
        return _fluorescence_composite(reduced, channel_colors=channel_colors)

    reduced = _reduce_to_axes(
        arr,
        state,
        keep_axes={y_axis, x_axis},
        mode=mode,
        current_step=current_step,
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
            index = _axis_index(step_axis, result.shape[local_axis], current_step)
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
    if current_ndim <= len(state.axes):
        return axis_index
    try:
        source_axis = state.axes[axis_index].source_axis
    except Exception:
        source_axis = None
    if source_axis is None:
        return axis_index
    return int(source_axis)


def _fluorescence_composite(
    arr: np.ndarray,
    *,
    channel_colors: str | list[str] | tuple[str, ...] | None = None,
) -> np.ndarray:
    color_table = _channel_color_table(channel_colors, arr.shape[-1])
    channels = []
    for channel in range(arr.shape[-1]):
        normalized = _normalize_uint8(arr[..., channel]).astype(np.float32) / 255.0
        color = color_table[channel]
        channels.append(normalized[..., None] * color)
    if not channels:
        return np.zeros(arr.shape[:2] + (3,), dtype=np.float32)
    return np.clip(np.sum(channels, axis=0), 0, 1)


def _channel_color_table(
    channel_colors: str | list[str] | tuple[str, ...] | None,
    count: int,
) -> np.ndarray:
    names = _channel_color_names(channel_colors)
    colors = []
    for channel in range(count):
        color = None
        if channel < len(names):
            color = NAMED_CHANNEL_COLORS.get(names[channel].lower())
        if color is None:
            color = FLUORESCENCE_COLORS[channel % len(FLUORESCENCE_COLORS)]
        colors.append(color)
    return np.asarray(colors, dtype=np.float32)


def _channel_color_names(
    channel_colors: str | list[str] | tuple[str, ...] | None,
) -> list[str]:
    if channel_colors is None:
        return []
    if isinstance(channel_colors, str):
        return [part.strip() for part in channel_colors.split(",") if part.strip()]
    return [str(part).strip() for part in channel_colors if str(part).strip()]


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


def _axis_index(axis: int, axis_size: int, current_step=None) -> int:
    if current_step is None:
        return axis_size // 2
    try:
        step = int(tuple(current_step)[axis])
    except Exception:
        step = axis_size // 2
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


def _normalize_uint8(arr: np.ndarray) -> np.ndarray:
    values = arr.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    # Signed difference images (e.g. Subtract) carry negative values around a
    # zero background. Stretching from the negative minimum would render that
    # background as mid-grey and saturate positive features to white. Anchor the
    # black point at zero so the thumbnail matches the inspector view, where the
    # zero background renders black.
    if float(finite.min()) < 0.0:
        values = np.clip(values, 0.0, None)
        finite = values[np.isfinite(values)]
    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        hi = float(finite.max())
        lo = float(finite.min())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lo) / (hi - lo)
    return (np.clip(scaled, 0, 1) * 255).astype(np.uint8)


def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    values = arr.astype(np.float32, copy=False)
    finite = values[np.isfinite(values)]
    if finite.size and float(finite.min()) >= 0.0 and float(finite.max()) <= 1.0:
        return (np.clip(values, 0, 1)[..., :3] * 255).astype(np.uint8)

    channels = [_normalize_uint8(arr[..., i]) for i in range(min(3, arr.shape[-1]))]
    while len(channels) < 3:
        channels.append(channels[-1] if channels else np.zeros(arr.shape[:2], np.uint8))
    return np.stack(channels[:3], axis=-1)


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
