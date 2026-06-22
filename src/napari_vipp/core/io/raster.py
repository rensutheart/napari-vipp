"""Common raster image support through imageio/Pillow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np

from napari_vipp.core.io.model import (
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
)
from napari_vipp.core.metadata import (
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)

RASTER_SUFFIXES = frozenset(
    {
        ".bmp",
        ".dib",
        ".gif",
        ".jpeg",
        ".jpg",
        ".jpe",
        ".jfif",
        ".pbm",
        ".pgm",
        ".png",
        ".pnm",
        ".ppm",
        ".tga",
        ".webp",
    }
)
ANIMATED_SUFFIXES = {".gif", ".webp"}
RASTER_WRITE_FORMATS = frozenset(
    {
        "bmp",
        "gif",
        "jpeg",
        "pbm",
        "pgm",
        "png",
        "pnm",
        "ppm",
        "tga",
        "webp",
    }
)


def inspect_raster(path: Path) -> SourceInspection:
    """Inspect an ordinary raster image without normalizing scientific metadata."""
    props = iio.improps(path)
    shape = tuple(int(size) for size in props.shape)
    series = (
        ImageSeriesInfo(
            index=0,
            key="image",
            name=path.stem,
            shape=shape,
            dtype=np.dtype(props.dtype).name,
            axes=_axis_order(shape, path.suffix.lower()),
        ),
    )
    return SourceInspection(
        str(path),
        raster_format(path),
        series,
        original_metadata=_imageio_metadata(path),
    )


def read_raster(path: Path, series_index: int = 0) -> ImageDataset:
    """Read one common raster image as a VIPP image dataset."""
    inspection = inspect_raster(path)
    selected = _selected_series(inspection, series_index)
    data = np.asarray(iio.imread(path))
    axes = _axis_metadata(tuple(int(size) for size in data.shape), path.suffix.lower())
    source = SourceMetadata(
        uri=str(path),
        format=inspection.format,
        series_index=selected.index,
        series_name=selected.name,
    )
    state = image_state_from_array(
        data,
        source_name=selected.name or path.name,
        axes=axes,
        metadata_source="common raster image metadata",
        source=source,
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    return ImageDataset(
        data,
        state,
        inspection,
        selected,
        original_metadata=inspection.original_metadata,
        provenance={"reader": "napari-vipp", "source_uri": str(path)},
    )


def write_raster(
    data,
    path: Path,
    *,
    format: str = "auto",
    image_state=None,
) -> Path:
    """Write a 2D display raster image.

    Raster formats are intentionally limited to 2D intensity or 2D RGB/RGBA
    arrays. Stacks should be saved through OME-TIFF, ImageJ TIFF, TIFF,
    OME-Zarr, or NPY so axes and pixel values remain meaningful.
    """
    selected = raster_write_format(path, format)
    arr = _raster_writable_array(data, selected, image_state)
    iio.imwrite(path, arr, extension=_write_extension(path, selected))
    return path


def raster_format(path: Path) -> str:
    """Return the normalized raster format label for a file path."""
    suffix = path.suffix.lower().removeprefix(".")
    if suffix in {"jpg", "jpe", "jfif"}:
        return "jpeg"
    if suffix == "dib":
        return "bmp"
    return suffix or "raster"


def raster_write_format(path: Path, format: str = "auto") -> str:
    """Resolve a requested raster write format."""
    selected = str(format or "auto").lower()
    if selected == "auto":
        selected = raster_format(path)
    aliases = {
        "jpe": "jpeg",
        "jfif": "jpeg",
        "jpg": "jpeg",
        "dib": "bmp",
    }
    selected = aliases.get(selected, selected)
    if selected not in RASTER_WRITE_FORMATS:
        raise ValueError(f"Unsupported raster save format: {format}")
    return selected


def _axis_metadata(
    shape: tuple[int, ...],
    suffix: str,
) -> tuple[AxisMetadata, ...]:
    ndim = len(shape)
    if ndim == 0:
        return ()

    channel_count = shape[-1] if ndim >= 3 and shape[-1] in {2, 3, 4} else None
    spatial_ndim = ndim - 1 if channel_count is not None else ndim
    axes: list[AxisMetadata] = []
    axes.extend(_leading_axes(max(spatial_ndim - 2, 0), suffix))
    if spatial_ndim >= 2:
        axes.extend((AxisMetadata("y", "space"), AxisMetadata("x", "space")))
    elif spatial_ndim == 1:
        axes.append(AxisMetadata("x", "space"))

    if channel_count is not None:
        if channel_count == 3:
            channel_name = "rgb"
        elif channel_count == 4:
            channel_name = "rgba"
        else:
            channel_name = "c"
        axes.append(AxisMetadata(channel_name, "channel"))

    if len(axes) != ndim:
        return tuple(AxisMetadata(f"d{index}", "unknown") for index in range(ndim))
    return tuple(axes)


def _raster_writable_array(
    arr: np.ndarray,
    format: str,
    image_state,
) -> np.ndarray:
    arr = _materialized_array(arr)
    if arr.ndim == 2:
        return _writable_gray(arr, format, image_state)
    if arr.ndim == 3 and arr.shape[-1] in {3, 4}:
        if format == "jpeg" and arr.shape[-1] == 4:
            raise ValueError(
                "JPEG cannot represent alpha. Save RGB/RGBA data as PNG, WebP, "
                "TIFF, or remove the alpha channel first."
            )
        return _writable_color(arr, format)
    raise ValueError(
        "Raster export supports only 2D intensity arrays or 2D RGB/RGBA arrays. "
        "Use TIFF, OME-TIFF, OME-Zarr, or NPY for stacks."
    )


def _materialized_array(arr: np.ndarray) -> np.ndarray:
    if not isinstance(arr, np.ndarray) and hasattr(arr, "compute"):
        arr = arr.compute()
    return np.asarray(arr)


def _writable_gray(
    arr: np.ndarray,
    format: str,
    image_state,
) -> np.ndarray:
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if _is_label_state(image_state):
        return _label_raster_array(arr, format)
    if np.issubdtype(arr.dtype, np.floating):
        return _float_to_uint8(arr)
    if np.issubdtype(arr.dtype, np.integer):
        if format in {"png", "pgm", "pnm"}:
            minimum, maximum = _integer_range(arr)
            if minimum >= 0 and maximum <= np.iinfo(np.uint16).max:
                if maximum <= np.iinfo(np.uint8).max:
                    return arr.astype(np.uint8, copy=False)
                return arr.astype(np.uint16, copy=False)
        return _integer_to_uint8(arr)
    raise ValueError(f"Cannot save dtype {arr.dtype} as a raster image.")


def _writable_color(arr: np.ndarray, _format: str) -> np.ndarray:
    if arr.dtype == bool:
        return arr.astype(np.uint8) * 255
    if np.issubdtype(arr.dtype, np.floating):
        return _float_to_uint8(arr)
    if np.issubdtype(arr.dtype, np.integer):
        if arr.dtype == np.uint8:
            return arr
        return _integer_to_uint8(arr)
    raise ValueError(f"Cannot save dtype {arr.dtype} as a raster image.")


def _label_raster_array(arr: np.ndarray, format: str) -> np.ndarray:
    if not np.issubdtype(arr.dtype, np.integer):
        return _writable_gray(arr, format, None)
    minimum, maximum = _integer_range(arr)
    if minimum < 0:
        raise ValueError("Label raster export requires non-negative label IDs.")
    if format in {"png", "pgm", "pnm"} and maximum <= np.iinfo(np.uint16).max:
        if maximum <= np.iinfo(np.uint8).max:
            return arr.astype(np.uint8, copy=False)
        return arr.astype(np.uint16, copy=False)
    if maximum <= np.iinfo(np.uint8).max:
        return arr.astype(np.uint8, copy=False)
    raise ValueError(
        "This raster format cannot preserve the label IDs in this 2D label "
        "image. Use PNG for labels <= 65535, or TIFF/OME-TIFF/NPY for larger "
        "label IDs."
    )


def _float_to_uint8(arr: np.ndarray) -> np.ndarray:
    values = np.asarray(arr, dtype=np.float32)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    minimum = float(finite.min())
    maximum = float(finite.max())
    if minimum >= 0.0 and maximum <= 1.0:
        scaled = np.clip(values, 0.0, 1.0) * 255.0
    elif maximum > minimum:
        scaled = (values - minimum) / (maximum - minimum) * 255.0
    else:
        scaled = np.zeros(values.shape, dtype=np.float32)
    return np.nan_to_num(scaled, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)


def _integer_to_uint8(arr: np.ndarray) -> np.ndarray:
    minimum, maximum = _integer_range(arr)
    if minimum >= 0 and maximum <= np.iinfo(np.uint8).max:
        return arr.astype(np.uint8, copy=False)
    if maximum <= minimum:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - float(minimum)) / float(maximum - minimum)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def _integer_range(arr: np.ndarray) -> tuple[int, int]:
    if arr.size == 0:
        return 0, 0
    return int(np.min(arr)), int(np.max(arr))


def _is_label_state(image_state) -> bool:
    if image_state is None:
        return False
    if isinstance(image_state, dict):
        return str(image_state.get("kind", "")).lower() == "label image"
    return str(getattr(image_state, "kind", "")).lower() == "label image"


def _write_extension(path: Path, format: str) -> str:
    suffix = path.suffix.lower()
    if suffix:
        normalized = raster_format(path)
        if normalized == format:
            return suffix
    if format == "jpeg":
        return ".jpg"
    return f".{format}"


def _leading_axes(count: int, suffix: str) -> list[AxisMetadata]:
    if count <= 0:
        return []
    if count == 1:
        name = "t" if suffix in ANIMATED_SUFFIXES else "z"
        return [AxisMetadata(name, _axis_type(name))]
    if count == 2:
        return [
            AxisMetadata("t", "time"),
            AxisMetadata("z", "space"),
        ]
    extra = [
        AxisMetadata(f"d{index}", "unknown")
        for index in range(count - 2)
    ]
    extra.extend(
        [
            AxisMetadata("t", "time"),
            AxisMetadata("z", "space"),
        ]
    )
    return extra


def _axis_order(shape: tuple[int, ...], suffix: str) -> str:
    axes = _axis_metadata(shape, suffix)
    if not axes:
        return "scalar"
    labels = [axis.short_label for axis in axes]
    if all(len(label) == 1 for label in labels):
        return "".join(labels)
    return ",".join(labels)


def _axis_type(name: str) -> str:
    if name == "t":
        return "time"
    if name in {"x", "y", "z"}:
        return "space"
    if name in {"c", "rgb", "rgba"}:
        return "channel"
    return "unknown"


def _imageio_metadata(path: Path) -> dict[str, Any] | None:
    try:
        metadata = dict(iio.immeta(path))
    except Exception:
        return None
    return metadata or None


def _selected_series(
    inspection: SourceInspection,
    series_index: int,
) -> ImageSeriesInfo:
    if not inspection.series:
        raise ValueError(f"No image found in {inspection.uri}")
    index = int(series_index)
    if index < 0 or index >= len(inspection.series):
        raise IndexError(
            f"Series index {index} is outside 0..{len(inspection.series) - 1}"
        )
    return inspection.series[index]
