"""Image metadata summaries for graph nodes and inspector panels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RGB_CHANNELS = (3, 4)
MAX_METADATA_VALUES = 500_000


@dataclass(frozen=True)
class ImageMetadata:
    kind: str
    shape: str
    axes: str
    dimensions: str
    channels: str
    timepoints: str
    z_slices: str
    dtype: str
    bit_depth: str
    value_range: str
    value_pattern: str
    memory: str


def summarize_image(data) -> ImageMetadata | None:
    """Return a compact, inferred metadata model for array-like image data."""
    if data is None:
        return None

    arr = np.asarray(data)
    axes = infer_axes(arr)
    return ImageMetadata(
        kind=_kind_label(arr),
        shape=_shape_label(arr.shape),
        axes=axes,
        dimensions=_dimensions_label(arr, axes),
        channels=_channels_label(arr, axes),
        timepoints=_axis_count_label(arr, axes, "T"),
        z_slices=_axis_count_label(arr, axes, "Z"),
        dtype=arr.dtype.name,
        bit_depth=_bit_depth_label(arr.dtype),
        value_range=_value_range_label(arr),
        value_pattern=_value_pattern_label(arr),
        memory=_memory_label(arr.nbytes),
    )


def format_compact_metadata(data) -> str:
    """Two-line metadata summary suitable for a small graph node."""
    metadata = summarize_image(data)
    if metadata is None:
        return "No output"

    first = f"{metadata.axes} {metadata.shape} | {metadata.dtype}"
    second_parts = [metadata.kind, metadata.bit_depth, f"range {metadata.value_range}"]
    if metadata.value_pattern:
        second_parts.insert(1, metadata.value_pattern)
    return first + "\n" + " | ".join(second_parts)


def format_detailed_metadata(data) -> str:
    """Multi-line metadata summary for the selected node inspector."""
    metadata = summarize_image(data)
    if metadata is None:
        return "No output yet."

    lines = [
        f"Kind: {metadata.kind}",
        f"Shape: {metadata.shape}",
        f"Axes: {metadata.axes} (inferred)",
        f"Dimensions: {metadata.dimensions}",
        f"Channels: {metadata.channels}",
        f"Timepoints: {metadata.timepoints}",
        f"Z slices: {metadata.z_slices}",
        f"Dtype: {metadata.dtype}",
        f"Bit depth: {metadata.bit_depth}",
        f"Value range: {metadata.value_range}",
    ]
    if metadata.value_pattern:
        lines.append(f"Value pattern: {metadata.value_pattern}")
    lines.append(f"Memory: {metadata.memory}")
    return "\n".join(lines)


def infer_axes(arr: np.ndarray) -> str:
    """Infer common bioimage axes from shape alone."""
    if arr.ndim == 0:
        return "scalar"
    if arr.ndim == 1:
        return "X"

    has_channels = _has_channel_axis(arr)
    if has_channels:
        spatial_ndim = arr.ndim - 1
        if spatial_ndim == 2:
            return "YXC"
        if spatial_ndim == 3:
            return "ZYXC"
        if spatial_ndim == 4:
            return "TZYXC"
        return _fallback_axes(spatial_ndim - 2) + "YXC"

    if arr.ndim == 2:
        return "YX"
    if arr.ndim == 3:
        return "ZYX"
    if arr.ndim == 4:
        return "TZYX"
    return _fallback_axes(arr.ndim - 2) + "YX"


def _kind_label(arr: np.ndarray) -> str:
    if arr.dtype == bool:
        return "binary mask"
    if _has_channel_axis(arr):
        return "RGBA image" if arr.shape[-1] == 4 else "RGB image"
    if np.issubdtype(arr.dtype, np.number):
        return "intensity image"
    return "array"


def _shape_label(shape: tuple[int, ...]) -> str:
    return " x ".join(str(size) for size in shape) if shape else "scalar"


def _dimensions_label(arr: np.ndarray, axes: str) -> str:
    if len(axes) != arr.ndim:
        return _shape_label(arr.shape)
    return ", ".join(
        f"{axis}={size}" for axis, size in zip(axes, arr.shape, strict=True)
    )


def _channels_label(arr: np.ndarray, axes: str) -> str:
    if "C" not in axes or len(axes) != arr.ndim:
        return "none inferred"
    count = arr.shape[axes.index("C")]
    if count == 3:
        return "RGB (3)"
    if count == 4:
        return "RGBA (4)"
    return str(count)


def _axis_count_label(arr: np.ndarray, axes: str, axis: str) -> str:
    if axis not in axes or len(axes) != arr.ndim:
        return "none inferred"
    return str(arr.shape[axes.index(axis)])


def _bit_depth_label(dtype: np.dtype) -> str:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(bool):
        return "1-bit logical"
    if np.issubdtype(dtype, np.integer):
        return f"{np.iinfo(dtype).bits}-bit integer"
    if np.issubdtype(dtype, np.floating):
        return f"{dtype.itemsize * 8}-bit float"
    return f"{dtype.itemsize * 8}-bit"


def _value_range_label(arr: np.ndarray) -> str:
    if arr.size == 0:
        return "empty"
    if arr.dtype == bool:
        return f"{bool(arr.min())} to {bool(arr.max())}"

    values = _finite_sample(arr)
    if values.size == 0:
        return "no finite values"
    return f"{_format_number(values.min())} to {_format_number(values.max())}"


def _value_pattern_label(arr: np.ndarray) -> str:
    if arr.size == 0:
        return ""
    if arr.dtype == bool:
        return "binary values"
    if not np.issubdtype(arr.dtype, np.number):
        return ""

    values = _finite_sample(arr)
    if values.size == 0:
        return ""
    unique = np.unique(values)
    if unique.size <= 2:
        return "binary-valued"
    return ""


def _finite_sample(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr).ravel()
    if values.size > MAX_METADATA_VALUES:
        stride = int(np.ceil(values.size / MAX_METADATA_VALUES))
        values = values[::stride]
    if not np.issubdtype(values.dtype, np.number):
        return np.array([], dtype=np.float32)
    try:
        return values[np.isfinite(values)]
    except TypeError:
        return values


def _format_number(value) -> str:
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:.4g}"


def _memory_label(nbytes: int) -> str:
    value = float(nbytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def _fallback_axes(prefix_count: int) -> str:
    if prefix_count <= 0:
        return ""
    labels = ["P", "T", "Z"]
    if prefix_count <= len(labels):
        return "".join(labels[-prefix_count:])
    extra = "".join(f"D{index}" for index in range(prefix_count - len(labels)))
    return extra + "".join(labels)


def _has_channel_axis(arr: np.ndarray) -> bool:
    return arr.ndim >= 3 and arr.shape[-1] in RGB_CHANNELS
