"""Preview reduction and thumbnail helpers."""

from __future__ import annotations

import numpy as np

RGB_CHANNELS = (3, 4)


def make_preview(data, mode: str = "slice", current_step=None) -> np.ndarray | None:
    """Reduce arbitrary image-like data to a 2D or RGB thumbnail source array."""
    if data is None or mode.lower() == "off":
        return None

    arr = np.asarray(data)
    if arr.size == 0:
        return None

    mode = mode.lower()
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
