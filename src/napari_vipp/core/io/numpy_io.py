"""NumPy array container readers and writers."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from napari_vipp.core.io.model import (
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
)
from napari_vipp.core.metadata import SourceMetadata, image_state_from_array


def inspect_numpy(path: Path) -> SourceInspection:
    """Inspect an NPY or NPZ source with minimal materialization."""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        data = np.load(path, mmap_mode="r")
        series = (_series_info(0, path.stem, data),)
    elif suffix == ".npz":
        with np.load(path) as archive:
            series = tuple(
                _series_info(index, key, archive[key])
                for index, key in enumerate(archive.files)
            )
    else:
        raise ValueError(f"Unsupported NumPy source: {path}")
    return SourceInspection(str(path), suffix.removeprefix("."), series)


def read_numpy(path: Path, series_index: int = 0) -> ImageDataset:
    """Read one array from an NPY or NPZ source."""
    inspection = inspect_numpy(path)
    selected = _selected_series(inspection, series_index)
    if path.suffix.lower() == ".npy":
        data = np.load(path)
    else:
        with np.load(path) as archive:
            data = archive[selected.key].copy()
    state = image_state_from_array(
        data,
        source_name=selected.name,
        metadata_source="NumPy array container",
        source=SourceMetadata(
            uri=str(path),
            format=inspection.format,
            series_index=selected.index,
            series_name=selected.name,
        ),
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    return ImageDataset(data, state, inspection, selected)


def write_numpy(data, path: Path) -> Path:
    """Write one array to NPY."""
    with path.open("wb") as handle:
        np.save(handle, np.asarray(data))
    return path


def _series_info(index: int, key: str, data) -> ImageSeriesInfo:
    arr = np.asarray(data)
    axes = image_state_from_array(arr)
    return ImageSeriesInfo(
        index=index,
        key=key,
        name=key,
        shape=tuple(int(size) for size in arr.shape),
        dtype=arr.dtype.name,
        axes=axes.axis_order if axes is not None else "",
    )


def _selected_series(
    inspection: SourceInspection,
    series_index: int,
) -> ImageSeriesInfo:
    if not inspection.series:
        raise ValueError(f"No arrays found in {inspection.uri}")
    index = int(series_index)
    if index < 0 or index >= len(inspection.series):
        raise IndexError(
            f"Series index {index} is outside 0..{len(inspection.series) - 1}"
        )
    return inspection.series[index]
