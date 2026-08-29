"""Format detection and shared image reader/writer dispatch."""

from __future__ import annotations

import shutil
from dataclasses import replace
from importlib.metadata import PackageNotFoundError, version
from math import prod
from pathlib import Path
from typing import Any

import numpy as np

from napari_vipp.core.io.microscope import (
    MICROSCOPE_SUFFIXES,
    image_state_from_microscope_inspection,
    inspect_microscope,
    read_microscope,
)
from napari_vipp.core.io.model import ImageDataset, ImageSeriesInfo, SourceInspection
from napari_vipp.core.io.numpy_io import inspect_numpy, read_numpy, write_numpy
from napari_vipp.core.io.ome_zarr import (
    image_state_from_ome_zarr_inspection,
    inspect_ome_zarr,
    read_ome_zarr,
    read_ome_zarr_exact_window,
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
from napari_vipp.core.io.tiff import (
    image_state_from_tiff_inspection,
    inspect_tiff,
    read_tiff,
    write_tiff,
)
from napari_vipp.core.metadata import (
    ImageState,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.source_window import (
    SourceWindowControl,
    SourceWindowRequest,
    SourceWindowResult,
)

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
        return _annotated_inspection(inspect_ome_zarr(source_path), suffix=suffix)
    if suffix in MICROSCOPE_SUFFIXES:
        return _annotated_inspection(inspect_microscope(source_path), suffix=suffix)
    if suffix in {".npy", ".npz"}:
        return _annotated_inspection(inspect_numpy(source_path), suffix=suffix)
    if suffix in {".tif", ".tiff"}:
        return _annotated_inspection(inspect_tiff(source_path), suffix=suffix)
    if suffix in RASTER_SUFFIXES:
        return _annotated_inspection(inspect_raster(source_path), suffix=suffix)
    raise ValueError(f"Unsupported image source: {source_path}")


def inspect_image_state(
    path: str | Path,
    *,
    inspection: SourceInspection | None = None,
    series_index: int = 0,
) -> ImageState:
    """Normalize one series' metadata without loading its pixel values."""
    source_path = _source_path(path)
    resolved_inspection = inspection or inspect_image_source(source_path)
    index = int(series_index)
    if index < 0 or index >= len(resolved_inspection.series):
        raise IndexError(
            f"Series index {index} is outside 0.."
            f"{len(resolved_inspection.series) - 1}"
        )
    if source_path.suffix.lower() == ".zarr":
        return image_state_from_ome_zarr_inspection(
            source_path,
            resolved_inspection,
            index,
        )
    if source_path.suffix.lower() in MICROSCOPE_SUFFIXES:
        return image_state_from_microscope_inspection(
            source_path,
            resolved_inspection,
            index,
        )
    if source_path.suffix.lower() in {".tif", ".tiff"}:
        return image_state_from_tiff_inspection(
            source_path,
            resolved_inspection,
            index,
        )

    selected = resolved_inspection.series[index]
    metadata = (
        None
        if resolved_inspection.format in {"npy", "npz"}
        else {"axes": selected.axes}
    )
    state = image_state_from_array(
        _MetadataOnlyArray(selected.shape, selected.dtype),
        layer_metadata=metadata,
        source_name=selected.name or source_path.name,
        metadata_source=(
            "NumPy array container"
            if resolved_inspection.format in {"npy", "npz"}
            else "common raster image metadata"
        ),
        source=SourceMetadata(
            uri=str(source_path),
            format=resolved_inspection.format,
            series_index=selected.index,
            series_name=selected.name,
        ),
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {source_path}")
    if selected.kind == "labels":
        state = replace(state, kind="label image")
    return state


def read_image(
    path: str | Path,
    *,
    series_index: int = 0,
) -> ImageDataset:
    """Read one selected image item from a supported source."""
    source_path = _source_path(path)
    suffix = source_path.suffix.lower()
    if suffix == ".zarr":
        dataset = read_ome_zarr(source_path, series_index)
    elif suffix in MICROSCOPE_SUFFIXES:
        dataset = read_microscope(source_path, series_index)
    elif suffix in {".npy", ".npz"}:
        dataset = read_numpy(source_path, series_index)
    elif suffix in {".tif", ".tiff"}:
        dataset = read_tiff(source_path, series_index)
    elif suffix in RASTER_SUFFIXES:
        dataset = read_raster(source_path, series_index)
    else:
        raise ValueError(f"Unsupported image source: {source_path}")
    inspection = _annotated_inspection(dataset.inspection, suffix=suffix)
    selected = next(
        (
            item
            for item in inspection.series
            if item.key == dataset.selected_series.key
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            "Reader contract mismatch: the selected item is absent from the "
            "annotated source inspection."
        )
    return replace(dataset, inspection=inspection, selected_series=selected)


def read_image_exact_window(
    path: str | Path,
    *,
    request: SourceWindowRequest,
    series_index: int = 0,
    control: SourceWindowControl | None = None,
) -> SourceWindowResult:
    """Read an exact scientific level-0 window through a capable local reader.

    Exact region reads are deliberately opt-in.  The registry currently
    dispatches only local OME-Zarr 0.4/0.5; unsupported formats fail visibly
    instead of falling back to a full materialization that could exceed RAM.
    """
    source_path = _source_path(path)
    suffix = source_path.suffix.lower()
    if suffix != ".zarr":
        raise ValueError(
            "Exact scientific source-window reads currently support local "
            "OME-Zarr 0.4/0.5 sources only."
        )
    result = read_ome_zarr_exact_window(
        source_path,
        series_index,
        request=request,
        control=control,
    )
    inspection = _annotated_inspection(result.inspection, suffix=suffix)
    selected = next(
        (
            item
            for item in inspection.series
            if item.key == result.selected_series.key
        ),
        None,
    )
    if selected is None:
        raise ValueError(
            "Reader contract mismatch: the exact-window item is absent from "
            "the annotated source inspection."
        )
    identity = replace(
        result.identity,
        reader_key=selected.reader_key or result.identity.reader_key,
        reader_version=selected.reader_version or result.identity.reader_version,
    )
    return replace(
        result,
        inspection=inspection,
        selected_series=selected,
        identity=identity,
    )


def _annotated_inspection(
    inspection: SourceInspection,
    *,
    suffix: str,
) -> SourceInspection:
    return replace(
        inspection,
        series=tuple(
            _annotated_series(item, inspection.format, suffix=suffix)
            for item in inspection.series
        ),
    )


def _annotated_series(
    item: ImageSeriesInfo,
    source_format: str,
    *,
    suffix: str,
) -> ImageSeriesInfo:
    implementation, package = _reader_implementation(item, suffix=suffix)
    # Reader-specific adapters historically used human-facing hyphenated
    # capability labels, while the canonical SourceItem contract uses field
    # identifiers.  Normalize at the registry boundary so a truthful native
    # lazy reader cannot be persisted as eager merely because of punctuation.
    capabilities = {
        str(capability).strip().lower().replace("-", "_")
        for capability in item.capabilities
        if str(capability).strip()
    }
    capabilities.add("pixel_lazy_inspection")
    if suffix == ".zarr":
        capabilities.update(
            {
                "lazy_data",
                "level_enumeration",
                "chunked_read",
            }
        )
    if suffix in {".vsi", ".oif"}:
        capabilities.add("companion_discovery")
    estimated = item.estimated_decoded_bytes
    if estimated is None:
        try:
            estimated = int(prod(item.shape) * np.dtype(item.dtype).itemsize)
        except (TypeError, ValueError):
            estimated = None
    if estimated is not None:
        capabilities.add("decoded_size_estimate")
    return replace(
        item,
        reader_key=item.reader_key or implementation,
        reader_version=item.reader_version or _package_version(package),
        capabilities=tuple(sorted(capabilities)),
        estimated_decoded_bytes=estimated,
        level_shapes=item.level_shapes or (item.shape,),
    )


def _reader_implementation(
    item: ImageSeriesInfo,
    *,
    suffix: str,
) -> tuple[str, str]:
    if item.reader_key:
        package = {
            "bioio": "bioio",
            "oiffile": "oiffile",
            "nd2": "nd2",
            "liffile": "liffile",
            "czifile": "czifile",
            "oirfile": "oirfile",
        }.get(item.reader_key, item.reader_key)
        return item.reader_key, package
    if suffix == ".zarr":
        return "ome-zarr", "ome-zarr"
    if suffix in {".npy", ".npz"}:
        return "numpy", "numpy"
    if suffix in {".tif", ".tiff"}:
        return "tifffile", "tifffile"
    if suffix in RASTER_SUFFIXES:
        return "imageio", "imageio"
    return source_format_reader_key(source_format="microscope"), "napari-vipp"


def source_format_reader_key(*, source_format: str) -> str:
    """Return a stable fallback reader key for internal adapters."""

    normalized = str(source_format or "napari-vipp").strip().lower()
    return "-".join(normalized.split()) or "napari-vipp"


def _package_version(package: str) -> str:
    try:
        return version(package)
    except (PackageNotFoundError, ValueError):
        return "unknown"


class _MetadataOnlyArray:
    """Shape/dtype carrier recognized as lazy by metadata normalization."""

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = tuple(int(size) for size in shape)
        self.dtype = np.dtype(dtype)

    def compute(self):
        raise RuntimeError("A metadata-only image cannot load pixels.")


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
    selected = _resolve_write_format(output_path, format, image_state)
    output_path = _normalized_output_path(output_path, selected)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if selected in {"ome-zarr", "ome-zarr-0.4", "ome-zarr-0.5"}:
        if _state_kind(image_state).lower() == "label image":
            raise ValueError(
                "Standalone label arrays are not written as ordinary OME-Zarr "
                "images. Use TIFF/OME-TIFF for a standalone label, or Export "
                "OME Analysis Dataset to preserve labels in an image-linked "
                "OME-Zarr store."
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


def _normalized_output_path(path: Path, format: str) -> Path:
    if format == "npy" and path.suffix.lower() != ".npy":
        return Path(f"{path}.npy")
    return path


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
