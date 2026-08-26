"""NumPy array container readers and writers."""

from __future__ import annotations

import zipfile
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
        series = _inspect_npz_members(path)
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


def _inspect_npz_members(path: Path) -> tuple[ImageSeriesInfo, ...]:
    """Inspect only each embedded NPY header, never its complete pixel payload."""

    members: list[ImageSeriesInfo] = []
    keys: set[str] = set()
    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.endswith(".npy"):
                    continue
                key = info.filename[:-4]
                if key in keys:
                    raise ValueError(
                        f"NPZ source contains duplicate member key {key!r}."
                    )
                keys.add(key)
                with archive.open(info, mode="r") as stream:
                    major, minor = np.lib.format.read_magic(stream)
                    if (major, minor) == (1, 0):
                        header = np.lib.format.read_array_header_1_0(stream)
                    elif (major, minor) in {(2, 0), (3, 0)}:
                        # Versions 2 and 3 share the 4-byte header-length
                        # layout. Structured/object dtypes are refused below,
                        # so their encoding distinction cannot affect accepted
                        # image arrays.
                        header = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError(
                            f"NPZ member {key!r} uses unsupported NPY format "
                            f"{major}.{minor}."
                        )
                    shape, _fortran_order, dtype = header
                dtype = np.dtype(dtype)
                if dtype.hasobject:
                    raise ValueError(
                        f"NPZ member {key!r} contains Python objects; VIPP only "
                        "accepts numeric or Boolean image arrays."
                    )
                members.append(
                    _series_info_from_header(
                        len(members),
                        key,
                        tuple(int(size) for size in shape),
                        dtype,
                    )
                )
    except zipfile.BadZipFile as exc:
        raise ValueError(f"NPZ source is corrupt or truncated: {path}") from exc
    if not members:
        raise ValueError(f"No NumPy arrays found in NPZ source: {path}")
    return tuple(members)


def _series_info_from_header(
    index: int,
    key: str,
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> ImageSeriesInfo:
    state = image_state_from_array(_MetadataOnlyArray(shape, dtype))
    return ImageSeriesInfo(
        index=index,
        key=key,
        name=key,
        shape=shape,
        dtype=dtype.name,
        axes=state.axis_order if state is not None else "",
    )


class _MetadataOnlyArray:
    def __init__(self, shape: tuple[int, ...], dtype: np.dtype) -> None:
        self.shape = shape
        self.dtype = dtype

    def compute(self):
        raise RuntimeError("A metadata-only NumPy array cannot load pixels.")


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
