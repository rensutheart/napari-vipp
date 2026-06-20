"""Shared channel colour helpers for metadata, previews, and RGB conversion."""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_CHANNEL_COLOR_NAMES = ("Blue", "Green", "Red", "Magenta", "Yellow", "Cyan")
CHANNEL_COLOR_CHOICES = ("Red", "Green", "Blue", "Magenta", "Cyan", "Yellow")

NAMED_CHANNEL_COLORS = {
    "red": np.array([1.0, 0.0, 0.0], dtype=np.float32),
    "green": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    "blue": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    "magenta": np.array([1.0, 0.0, 1.0], dtype=np.float32),
    "cyan": np.array([0.0, 1.0, 1.0], dtype=np.float32),
    "yellow": np.array([1.0, 1.0, 0.0], dtype=np.float32),
}

CHANNEL_COLOR_HEX = {
    "red": "#ef4444",
    "green": "#22c55e",
    "blue": "#60a5fa",
    "magenta": "#d946ef",
    "cyan": "#06b6d4",
    "yellow": "#eab308",
}

FLUORESCENCE_COLORS = np.asarray(
    [NAMED_CHANNEL_COLORS[name.lower()] for name in DEFAULT_CHANNEL_COLOR_NAMES],
    dtype=np.float32,
)


def channel_color_names(
    channel_colors: str | list[Any] | tuple[Any, ...] | None,
) -> list[str]:
    if channel_colors is None:
        return []
    if isinstance(channel_colors, str):
        return [part.strip() for part in channel_colors.split(",") if part.strip()]
    return [str(part).strip() for part in channel_colors if str(part).strip()]


def channel_color_int(name: Any) -> int | None:
    rgb = color_value_to_rgb(name)
    if rgb is None:
        return None
    values = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    return (int(values[0]) << 16) | (int(values[1]) << 8) | int(values[2])


def color_value_to_rgb(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, np.integer):
        value = int(value)
    if isinstance(value, int):
        return _int_to_rgb(value)
    text = str(value).strip()
    if not text:
        return None
    if text.lower() in NAMED_CHANNEL_COLORS:
        return NAMED_CHANNEL_COLORS[text.lower()]
    if text.startswith("#"):
        return _hex_to_rgb(text[1:])
    try:
        base = 16 if any(c in text.lower() for c in "abcdef") else 10
        return _int_to_rgb(int(text, base))
    except Exception:
        return None


def channel_color_table(
    channel_colors: str | list[Any] | tuple[Any, ...] | None,
    count: int,
    *,
    metadata_colors: tuple[int | None, ...] = (),
) -> np.ndarray:
    explicit = channel_color_names(channel_colors)
    colors = []
    for index in range(int(count)):
        color = None
        if index < len(explicit):
            color = color_value_to_rgb(explicit[index])
        if color is None and index < len(metadata_colors):
            color = color_value_to_rgb(metadata_colors[index])
        if color is None:
            color = FLUORESCENCE_COLORS[index % len(FLUORESCENCE_COLORS)]
        colors.append(color)
    return np.asarray(colors, dtype=np.float32)


def channel_color_labels_from_metadata(
    metadata_colors: tuple[int | None, ...],
    count: int,
) -> list[str]:
    labels = []
    for index in range(int(count)):
        color = metadata_colors[index] if index < len(metadata_colors) else None
        name = channel_color_name_from_int(color)
        if name is None:
            name = DEFAULT_CHANNEL_COLOR_NAMES[index % len(DEFAULT_CHANNEL_COLOR_NAMES)]
        labels.append(name)
    return labels


def channel_color_name_from_int(color: int | None) -> str | None:
    if color is None:
        return None
    rgb = color_value_to_rgb(color)
    if rgb is None:
        return None
    for name, named_rgb in NAMED_CHANNEL_COLORS.items():
        if np.allclose(rgb, named_rgb, atol=1 / 255):
            return name.title()
    distances = {
        name: float(np.linalg.norm(rgb - named_rgb))
        for name, named_rgb in NAMED_CHANNEL_COLORS.items()
    }
    return min(distances, key=distances.get).title()


def _int_to_rgb(value: int) -> np.ndarray:
    value = int(value) & 0xFFFFFF
    return np.asarray(
        [
            ((value >> 16) & 0xFF) / 255.0,
            ((value >> 8) & 0xFF) / 255.0,
            (value & 0xFF) / 255.0,
        ],
        dtype=np.float32,
    )


def _hex_to_rgb(value: str) -> np.ndarray | None:
    text = value.strip()
    if len(text) == 8:
        text = text[:6]
    if len(text) != 6:
        return None
    try:
        return _int_to_rgb(int(text, 16))
    except ValueError:
        return None
