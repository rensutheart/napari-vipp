"""Format detection and shared image reader/writer dispatch."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from napari_vipp.core.io.model import ImageDataset, SourceInspection
from napari_vipp.core.io.numpy_io import inspect_numpy, read_numpy, write_numpy
from napari_vipp.core.io.ome_zarr import (
    inspect_ome_zarr,
    read_ome_zarr,
    write_ome_zarr,
)
from napari_vipp.core.io.raster import (
    RASTER_SUFFIXES,
    RASTER_WRITE_FORMATS,
    inspect_raster,
    raster_format,
    read_raster,
    write_raster,
)
from napari_vipp.core.io.tiff import inspect_tiff, read_tiff, write_tiff
from napari_vipp.core.metadata import ImageState

WRITE_FORMATS = (
    "auto",
    "ome-zarr",
    "ome-zarr-0.5",
    "ome-tiff",
    "imagej-tiff",
    "tiff",
    "npy",
    "png",
    "jpeg",
    "bmp",
    "gif",
    "webp",
    "tga",
    "pnm",
)


def inspect_image_source(path: str | Path) -> SourceInspection:
    """Inspect a supported local image source."""
    source_path = _source_path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".zarr":
        return inspect_ome_zarr(source_path)
    if suffix in {".npy", ".npz"}:
        return inspect_numpy(source_path)
    if suffix in {".tif", ".tiff"}:
        return inspect_tiff(source_path)
    if suffix in RASTER_SUFFIXES:
        return inspect_raster(source_path)
    raise ValueError(f"Unsupported image source: {source_path}")


def read_image(
    path: str | Path,
    *,
    series_index: int = 0,
) -> ImageDataset:
    """Read one selected image item from a supported source."""
    source_path = _source_path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".zarr":
        return read_ome_zarr(source_path, series_index)
    if suffix in {".npy", ".npz"}:
        return read_numpy(source_path, series_index)
    if suffix in {".tif", ".tiff"}:
        return read_tiff(source_path, series_index)
    if suffix in RASTER_SUFFIXES:
        return read_raster(source_path, series_index)
    raise ValueError(f"Unsupported image source: {source_path}")


def write_image(
    data,
    path: str | Path,
    *,
    format: str = "auto",
    overwrite: bool = True,
    image_state: ImageState | dict[str, Any] | None = None,
) -> Path:
    """Write an image through the shared format registry."""
    if data is None:
        raise ValueError("No node output is available to save.")
    raw_path = str(path).strip()
    if not raw_path:
        raise ValueError("A save path is required.")
    output_path = Path(raw_path).expanduser()
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected = _resolve_write_format(output_path, format, image_state)
    if selected in {"ome-zarr", "ome-zarr-0.4", "ome-zarr-0.5"}:
        if _state_kind(image_state).lower() == "label image":
            raise ValueError(
                "Standalone label arrays are not written as ordinary OME-Zarr "
                "images. Use TIFF/OME-TIFF until Export OME Analysis Dataset "
                "is available."
            )
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
        version = "0.5" if selected.endswith("0.5") else "0.4"
        return write_ome_zarr(
            data,
            output_path,
            version=version,
            image_state=image_state,
        )
    if selected == "npy":
        return write_numpy(data, output_path)
    if selected in {"ome-tiff", "imagej-tiff", "tiff"}:
        return write_tiff(
            data,
            output_path,
            format=selected,
            image_state=image_state,
        )
    if selected in RASTER_WRITE_FORMATS:
        return write_raster(
            data,
            output_path,
            format=selected,
            image_state=image_state,
        )
    raise ValueError(f"Unsupported save format: {format}")


def _resolve_write_format(
    path: Path,
    format: str,
    image_state: ImageState | dict[str, Any] | None,
) -> str:
    selected = str(format or "auto").lower()
    aliases = {
        "zarr": "ome-zarr",
        "ome-tif": "ome-tiff",
        "imagej": "imagej-tiff",
        "ij-tiff": "imagej-tiff",
        "jpg": "jpeg",
        "jpe": "jpeg",
        "jfif": "jpeg",
        "dib": "bmp",
        "tif": "tiff",
    }
    selected = aliases.get(selected, selected)
    if selected != "auto":
        return selected
    lower_name = path.name.lower()
    if path.suffix.lower() == ".npy":
        return "npy"
    if path.suffix.lower() == ".zarr":
        return "ome-zarr"
    if path.suffix.lower() in RASTER_SUFFIXES:
        return raster_format(path)
    if lower_name.endswith((".ome.tif", ".ome.tiff")):
        return "ome-tiff"
    kind = _state_kind(image_state)
    return "tiff" if kind.lower() == "label image" else "ome-tiff"


def _source_path(path: str | Path) -> Path:
    source_path = Path(path).expanduser()
    if not source_path.exists():
        raise FileNotFoundError(f"Image source not found: {source_path}")
    return source_path


def _state_kind(image_state: ImageState | dict[str, Any] | None) -> str:
    if isinstance(image_state, ImageState):
        return image_state.kind
    if isinstance(image_state, dict):
        return str(image_state.get("kind", ""))
    return ""
