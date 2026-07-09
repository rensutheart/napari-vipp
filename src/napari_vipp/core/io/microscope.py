"""Optional proprietary microscope-format readers.

The core I/O registry imports this module without importing any heavy optional
dependencies. Individual readers are imported only when their file suffix is
selected.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass, replace
from importlib import import_module
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import numpy as np

from napari_vipp.core.channel_colors import channel_color_int
from napari_vipp.core.io.model import (
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
)
from napari_vipp.core.io.tiff import inspect_tiff, read_tiff
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    ChannelMetadata,
    SourceMetadata,
    image_state_from_array,
)

MICROSCOPE_SUFFIXES = frozenset(
    {
        ".czi",
        ".lif",
        ".lof",
        ".lsm",
        ".nd2",
        ".oib",
        ".oif",
        ".oir",
        ".vsi",
        ".xlif",
    }
)
MICROSCOPE_FILE_FILTER = (
    "Microscope files (*.nd2 *.czi *.lsm *.lif *.lof *.xlif "
    "*.oir *.oib *.oif *.vsi)"
)

_BIOIO_FALLBACK_SUFFIXES = {
    ".lif",
    ".lof",
    ".oib",
    ".oif",
    ".oir",
    ".vsi",
    ".xlif",
}
_FORMAT_BY_SUFFIX = {
    ".czi": "zeiss-czi",
    ".lif": "leica-lif",
    ".lof": "leica-lof",
    ".lsm": "zeiss-lsm",
    ".nd2": "nikon-nd2",
    ".oib": "olympus-oib",
    ".oif": "olympus-oif",
    ".oir": "olympus-oir",
    ".vsi": "olympus-vsi",
    ".xlif": "leica-xlif",
}
_NATIVE_INSTALL_COMMANDS = {
    ".czi": 'pip install "napari-vipp[czi]"',
    ".nd2": 'pip install "napari-vipp[nd2]"',
    ".lif": 'pip install "napari-vipp[microscope]"',
    ".lof": 'pip install "napari-vipp[microscope]"',
    ".oib": 'pip install "napari-vipp[microscope]"',
    ".oif": 'pip install "napari-vipp[microscope]"',
    ".oir": 'pip install "napari-vipp[microscope]"',
    ".xlif": 'pip install "napari-vipp[microscope]"',
}
_BIOIO_INSTALL_COMMAND = 'pip install "napari-vipp[bioformats]"'
_DECONVOLUTION_TERMS = (
    "richardson-lucy",
    "richardson lucy",
    "huygens",
    "deconvolutionlab",
    "widefield decon",
    "blind decon",
    "deconvolution",
    "deconvolved",
)
_DECONVOLUTION_NEGATIONS = (
    "no deconvolution",
    "not deconvolved",
    "deconvolution=false",
    "deconvolution: false",
    "deconvolution applied=false",
)


class OptionalMicroscopeReaderError(ImportError):
    """Raised when a microscope reader needs an optional dependency."""

    def __init__(
        self,
        message: str,
        *,
        suffix: str = "",
        format_name: str = "",
        module_name: str = "",
        install_command: str = "",
        fallback_install_command: str = "",
        restart_required: bool = True,
    ) -> None:
        super().__init__(message)
        self.suffix = str(suffix or "").lower()
        self.format_name = str(format_name or "")
        self.module_name = str(module_name or "")
        self.install_command = str(install_command or "")
        self.fallback_install_command = str(fallback_install_command or "")
        self.restart_required = bool(restart_required)

    @property
    def reader_label(self) -> str:
        if self.suffix:
            return f"{self.suffix.lstrip('.').upper()} reader"
        if self.format_name:
            return f"{self.format_name} reader"
        return "Microscope reader"


def is_microscope_source(path: str | Path) -> bool:
    """Return whether a file path looks like a microscope acquisition source."""
    return Path(path).suffix.lower() in MICROSCOPE_SUFFIXES


def microscope_format_for_path(path: str | Path) -> str:
    """Return VIPP's normalized microscope format label for a path."""
    suffix = Path(path).suffix.lower()
    return _FORMAT_BY_SUFFIX.get(suffix, suffix.removeprefix("."))


def inspect_microscope(path: Path) -> SourceInspection:
    """Inspect a microscope acquisition source through optional readers."""
    suffix = path.suffix.lower()
    if suffix == ".lsm":
        return _inspect_lsm(path)
    if suffix == ".nd2":
        try:
            return _inspect_nd2(path)
        except OptionalMicroscopeReaderError:
            return _inspect_bioio(path, microscope_format_for_path(path))
    if suffix == ".czi":
        try:
            return _inspect_czi(path)
        except OptionalMicroscopeReaderError:
            return _inspect_bioio(path, microscope_format_for_path(path))
    if suffix in {".lif", ".lof", ".xlif"}:
        try:
            return _inspect_lif(path)
        except OptionalMicroscopeReaderError:
            return _inspect_bioio(path, microscope_format_for_path(path))
    if suffix == ".oir":
        try:
            return _inspect_oir(path)
        except OptionalMicroscopeReaderError:
            return _inspect_bioio(path, microscope_format_for_path(path))
    if suffix in {".oib", ".oif"}:
        try:
            return _inspect_oif(path)
        except OptionalMicroscopeReaderError:
            return _inspect_bioio(path, microscope_format_for_path(path))
    if suffix in _BIOIO_FALLBACK_SUFFIXES:
        return _inspect_bioio(path, microscope_format_for_path(path))
    raise ValueError(f"Unsupported microscope source: {path}")


def read_microscope(path: Path, series_index: int = 0) -> ImageDataset:
    """Read one selected microscope acquisition image."""
    suffix = path.suffix.lower()
    if suffix == ".lsm":
        return _read_lsm(path, series_index)
    if suffix == ".nd2":
        try:
            return _read_nd2(path, series_index)
        except OptionalMicroscopeReaderError:
            return _read_bioio(path, series_index, microscope_format_for_path(path))
    if suffix == ".czi":
        try:
            return _read_czi(path, series_index)
        except OptionalMicroscopeReaderError:
            return _read_bioio(path, series_index, microscope_format_for_path(path))
    if suffix in {".lif", ".lof", ".xlif"}:
        try:
            return _read_lif(path, series_index)
        except OptionalMicroscopeReaderError:
            return _read_bioio(path, series_index, microscope_format_for_path(path))
    if suffix == ".oir":
        try:
            return _read_oir(path, series_index)
        except OptionalMicroscopeReaderError:
            return _read_bioio(path, series_index, microscope_format_for_path(path))
    if suffix in {".oib", ".oif"}:
        try:
            return _read_oif(path, series_index)
        except OptionalMicroscopeReaderError:
            return _read_bioio(path, series_index, microscope_format_for_path(path))
    if suffix in _BIOIO_FALLBACK_SUFFIXES:
        return _read_bioio(path, series_index, microscope_format_for_path(path))
    raise ValueError(f"Unsupported microscope source: {path}")


def detect_deconvolution_metadata(metadata: Any) -> tuple[bool | None, str]:
    """Conservatively detect upstream deconvolution from raw metadata text."""
    text = _metadata_text(metadata)
    if not text:
        return None, ""
    lowered = text.lower()
    if any(term in lowered for term in _DECONVOLUTION_NEGATIONS):
        return False, ""
    for term in _DECONVOLUTION_TERMS:
        if term in lowered:
            return True, _deconvolution_label(term)
    return None, ""


def _inspect_nd2(path: Path) -> SourceInspection:
    nd2 = _optional_import("nd2", path.suffix)
    with nd2.ND2File(str(path)) as nd_file:
        shape = tuple(int(size) for size in getattr(nd_file, "shape", ()))
        dtype = np.dtype(getattr(nd_file, "dtype", np.float32)).name
        axes = _nd2_axis_order(nd_file, shape)
        original = _nd2_original_metadata(nd_file)
    series = (
        ImageSeriesInfo(
            index=0,
            key="0",
            name=path.stem,
            shape=shape,
            dtype=dtype,
            axes=axes,
        ),
    )
    return SourceInspection(str(path), "nikon-nd2", series, original)


def _read_nd2(path: Path, series_index: int = 0) -> ImageDataset:
    inspection = _inspect_nd2(path)
    selected = _selected_series(inspection, series_index)
    nd2 = _optional_import("nd2", path.suffix)
    data = _nd2_array(nd2, path)
    with nd2.ND2File(str(path)) as nd_file:
        axes = _nd2_axes(nd_file, tuple(int(size) for size in data.shape))
        channels = _nd2_channels(nd_file)
        original = _nd2_original_metadata(nd_file)
        acquisition = _acquisition_from_metadata(original)
    source = SourceMetadata(
        uri=str(path),
        format="nikon-nd2",
        series_index=selected.index,
        series_name=selected.name,
    )
    state = image_state_from_array(
        data,
        source_name=selected.name or path.name,
        axes=axes,
        metadata_source="Nikon ND2 metadata",
        channels=channels,
        acquisition=acquisition,
        source=source,
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    return ImageDataset(
        data,
        state,
        inspection,
        selected,
        original_metadata=original,
        provenance={"reader": "nd2", "source_uri": str(path)},
    )


def _inspect_czi(path: Path) -> SourceInspection:
    czifile = _optional_import("czifile", path.suffix)
    with czifile.CziFile(str(path)) as czi:
        series = _czi_series(czi, path)
        original = _safe_call(czi, "metadata")
    return SourceInspection(str(path), "zeiss-czi", series, original)


def _read_czi(path: Path, series_index: int = 0) -> ImageDataset:
    inspection = _inspect_czi(path)
    selected = _selected_series(inspection, series_index)
    czifile = _optional_import("czifile", path.suffix)
    with czifile.CziFile(str(path)) as czi:
        scene = _czi_scene(czi, selected)
        data, axes, channels, attrs = _scene_payload(scene)
        original = _safe_call(czi, "metadata")
    acquisition = _acquisition_from_metadata({"metadata": original, "attrs": attrs})
    source = SourceMetadata(
        uri=str(path),
        format="zeiss-czi",
        series_index=selected.index,
        series_name=selected.name,
    )
    state = image_state_from_array(
        data,
        source_name=selected.name or path.name,
        axes=axes,
        metadata_source="Zeiss CZI metadata",
        channels=channels,
        acquisition=acquisition,
        source=source,
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    return ImageDataset(
        data,
        state,
        inspection,
        selected,
        original_metadata=original,
        provenance={"reader": "czifile", "source_uri": str(path)},
    )


def _inspect_lif(path: Path) -> SourceInspection:
    liffile = _optional_import("liffile", path.suffix)
    with liffile.LifFile(str(path)) as lif:
        images = _scene_items(getattr(lif, "images", None))
        series = tuple(
            _series_from_generic_image(index, key, image)
            for index, (key, image) in enumerate(images)
        )
        original = getattr(lif, "xml", None) or getattr(lif, "metadata", None)
    if not series:
        raise ValueError(f"No Leica image series found in {path}")
    return SourceInspection(
        str(path),
        microscope_format_for_path(path),
        series,
        original,
    )


def _read_lif(path: Path, series_index: int = 0) -> ImageDataset:
    inspection = _inspect_lif(path)
    selected = _selected_series(inspection, series_index)
    liffile = _optional_import("liffile", path.suffix)
    with liffile.LifFile(str(path)) as lif:
        image = _container_image(getattr(lif, "images", None), selected)
        data, axes, channels, attrs = _scene_payload(image)
        original = getattr(lif, "xml", None) or getattr(lif, "metadata", None)
    return _microscope_dataset(
        data,
        path,
        inspection,
        selected,
        axes,
        channels,
        {"metadata": original, "attrs": attrs},
        metadata_source="Leica metadata",
        reader="liffile",
    )


def _inspect_oir(path: Path) -> SourceInspection:
    oirfile = _optional_import("oirfile", path.suffix)
    with oirfile.OirFile(str(path)) as oir:
        series = (_series_from_generic_image(0, "0", oir, fallback_name=path.stem),)
        original = getattr(oir, "xml_metadata", None)
    return SourceInspection(
        str(path),
        microscope_format_for_path(path),
        series,
        original,
    )


def _read_oir(path: Path, series_index: int = 0) -> ImageDataset:
    inspection = _inspect_oir(path)
    selected = _selected_series(inspection, series_index)
    oirfile = _optional_import("oirfile", path.suffix)
    with oirfile.OirFile(str(path)) as oir:
        data, axes, channels, attrs = _scene_payload(oir)
        original = getattr(oir, "xml_metadata", None)
    return _microscope_dataset(
        data,
        path,
        inspection,
        selected,
        axes,
        channels,
        {"metadata": original, "attrs": attrs},
        metadata_source="Olympus OIR metadata",
        reader="oirfile",
    )


def _inspect_oif(path: Path) -> SourceInspection:
    oiffile = _optional_import("oiffile", path.suffix)
    with oiffile.OifFile(str(path)) as oif:
        series = (_series_from_generic_image(0, "0", oif, fallback_name=path.stem),)
        original = getattr(oif, "mainfile", None)
    return SourceInspection(
        str(path),
        microscope_format_for_path(path),
        series,
        original,
    )


def _read_oif(path: Path, series_index: int = 0) -> ImageDataset:
    inspection = _inspect_oif(path)
    selected = _selected_series(inspection, series_index)
    oiffile = _optional_import("oiffile", path.suffix)
    with oiffile.OifFile(str(path)) as oif:
        data, axes, channels, attrs = _scene_payload(oif)
        original = getattr(oif, "mainfile", None)
    return _microscope_dataset(
        data,
        path,
        inspection,
        selected,
        axes,
        channels,
        {"metadata": original, "attrs": attrs},
        metadata_source="Olympus OIF/OIB metadata",
        reader="oiffile",
    )


def _inspect_lsm(path: Path) -> SourceInspection:
    inspection = inspect_tiff(path)
    return SourceInspection(
        inspection.uri,
        "zeiss-lsm",
        inspection.series,
        inspection.original_metadata,
    )


def _read_lsm(path: Path, series_index: int = 0) -> ImageDataset:
    dataset = read_tiff(path, series_index)
    source = replace(dataset.image_state.source, format="zeiss-lsm")
    state = replace(
        dataset.image_state,
        metadata_source="Zeiss LSM TIFF metadata",
        source=source,
    )
    inspection = SourceInspection(
        dataset.inspection.uri,
        "zeiss-lsm",
        dataset.inspection.series,
        dataset.inspection.original_metadata,
    )
    return ImageDataset(
        dataset.data,
        state,
        inspection,
        dataset.selected_series,
        original_metadata=dataset.original_metadata,
        multiscale_levels=dataset.multiscale_levels,
        associated_labels=dataset.associated_labels,
        provenance={"reader": "tifffile", "source_uri": str(path)},
    )


def _inspect_bioio(path: Path, format_hint: str) -> SourceInspection:
    bioio = _optional_bioio(path.suffix)
    image = bioio.BioImage(str(path))
    series: list[ImageSeriesInfo] = []
    for index, scene in enumerate(tuple(getattr(image, "scenes", ()) or (0,))):
        _bioio_set_scene(image, scene)
        series.append(
            ImageSeriesInfo(
                index=index,
                key=str(scene),
                name=str(scene) or f"Series {index + 1}",
                shape=tuple(int(size) for size in getattr(image, "shape", ())),
                dtype=np.dtype(getattr(image, "dtype", np.float32)).name,
                axes=str(getattr(getattr(image, "dims", None), "order", "")),
            )
        )
    if not series:
        raise ValueError(f"No image series found in {path}")
    return SourceInspection(
        str(path),
        f"{format_hint}+bioio",
        tuple(series),
        _bioio_metadata(image),
    )


def _read_bioio(path: Path, series_index: int, format_hint: str) -> ImageDataset:
    inspection = _inspect_bioio(path, format_hint)
    selected = _selected_series(inspection, series_index)
    bioio = _optional_bioio(path.suffix)
    image = bioio.BioImage(str(path))
    _bioio_set_scene(image, selected.key)
    data = getattr(image, "dask_data", None)
    if data is None:
        data = image.data
    metadata = _bioio_metadata(image)
    axes = _bioio_axes(image, tuple(int(size) for size in data.shape))
    channels = _bioio_channels(image)
    acquisition = _acquisition_from_metadata(metadata)
    source_format = f"{format_hint}+bioio"
    source = SourceMetadata(
        uri=str(path),
        format=source_format,
        series_index=selected.index,
        series_name=selected.name,
    )
    state = image_state_from_array(
        data,
        source_name=selected.name or path.name,
        axes=axes,
        metadata_source="BioIO reader metadata",
        channels=channels,
        acquisition=acquisition,
        source=source,
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    return ImageDataset(
        data,
        state,
        inspection,
        selected,
        original_metadata=metadata,
        provenance={"reader": "bioio", "source_uri": str(path)},
    )


def _optional_import(module_name: str, suffix: str):
    try:
        return import_module(module_name)
    except ImportError as error:
        normalized_suffix = str(suffix or "").lower()
        command = _NATIVE_INSTALL_COMMANDS.get(
            normalized_suffix,
            _BIOIO_INSTALL_COMMAND,
        )
        raise OptionalMicroscopeReaderError(
            f"Reading {suffix} files requires optional dependency "
            f"{module_name!r}. Install it with: {command}",
            suffix=normalized_suffix,
            format_name=_FORMAT_BY_SUFFIX.get(normalized_suffix, ""),
            module_name=module_name,
            install_command=command,
        ) from error


def _optional_bioio(suffix: str = ""):
    try:
        return import_module("bioio")
    except ImportError as error:
        normalized_suffix = str(suffix or "").lower()
        native_command = _NATIVE_INSTALL_COMMANDS.get(normalized_suffix)
        command = native_command or _BIOIO_INSTALL_COMMAND
        hint = (
            f"Install the format-specific extra with: {native_command}. "
            f"For BioIO/Bio-Formats fallback, install: {_BIOIO_INSTALL_COMMAND}"
            if native_command
            else f"Install the fallback extra with: {_BIOIO_INSTALL_COMMAND}"
        )
        raise OptionalMicroscopeReaderError(
            "This microscope format requires an optional dependency: "
            "a BioIO reader plugin. "
            f"{hint}",
            suffix=normalized_suffix,
            format_name=_FORMAT_BY_SUFFIX.get(normalized_suffix, ""),
            module_name="bioio",
            install_command=command,
            fallback_install_command=(
                _BIOIO_INSTALL_COMMAND if native_command else ""
            ),
        ) from error


def _selected_series(
    inspection: SourceInspection,
    series_index: int,
) -> ImageSeriesInfo:
    if not inspection.series:
        raise ValueError(f"No image series found in {inspection.uri}")
    index = int(series_index)
    if index < 0 or index >= len(inspection.series):
        raise IndexError(
            f"Series index {index} is outside 0..{len(inspection.series) - 1}"
        )
    return inspection.series[index]


def _nd2_array(nd2, path: Path):
    try:
        return nd2.imread(str(path), dask=True)
    except Exception:
        return nd2.imread(str(path))


def _nd2_axis_order(nd_file, shape: tuple[int, ...]) -> str:
    sizes = getattr(nd_file, "sizes", None)
    if isinstance(sizes, dict) and len(sizes) == len(shape):
        labels = [str(label) for label in sizes]
        return _axis_order_label(labels)
    return _fallback_axis_order(shape)


def _nd2_axes(nd_file, shape: tuple[int, ...]) -> tuple[AxisMetadata, ...]:
    labels = _split_axis_order(_nd2_axis_order(nd_file, shape))
    voxel = _safe_call(nd_file, "voxel_size")
    scales = {
        "X": _optional_float(getattr(voxel, "x", None)),
        "Y": _optional_float(getattr(voxel, "y", None)),
        "Z": _optional_float(getattr(voxel, "z", None)),
    }
    axes: list[AxisMetadata] = []
    for label in labels:
        unit = "micrometer" if label in {"X", "Y", "Z"} and scales.get(label) else None
        axes.append(
            AxisMetadata(
                name=_axis_name(label),
                type=_axis_type(label),
                unit=unit,
                scale=scales.get(label) or 1.0,
            )
        )
    if len(axes) != len(shape):
        return _axes_from_order(_fallback_axis_order(shape), shape)
    return tuple(axes)


def _nd2_channels(nd_file) -> tuple[ChannelMetadata, ...]:
    metadata = getattr(nd_file, "metadata", None)
    channels = getattr(metadata, "channels", None) or ()
    records: list[ChannelMetadata] = []
    for item in channels:
        channel = getattr(item, "channel", item)
        color = getattr(channel, "colorRGBA", None)
        records.append(
            ChannelMetadata(
                name=str(getattr(channel, "name", "") or ""),
                color=_optional_int(color),
                excitation_wavelength=_optional_float(
                    getattr(channel, "excitationLambdaNm", None)
                ),
                excitation_wavelength_unit=(
                    "nm" if getattr(channel, "excitationLambdaNm", None) else ""
                ),
                emission_wavelength=_optional_float(
                    getattr(channel, "emissionLambdaNm", None)
                ),
                emission_wavelength_unit=(
                    "nm" if getattr(channel, "emissionLambdaNm", None) else ""
                ),
            )
        )
    return tuple(records)


def _nd2_original_metadata(nd_file) -> dict[str, Any]:
    raw = {
        "attributes": getattr(nd_file, "attributes", None),
        "metadata": getattr(nd_file, "metadata", None),
        "experiment": getattr(nd_file, "experiment", None),
        "text_info": getattr(nd_file, "text_info", None),
    }
    try:
        raw["unstructured"] = nd_file.unstructured_metadata()
    except Exception:
        pass
    return raw


def _czi_series(czi, path: Path) -> tuple[ImageSeriesInfo, ...]:
    scenes = getattr(czi, "scenes", None)
    scene_items = _scene_items(scenes)
    if not scene_items:
        shape = tuple(int(size) for size in getattr(czi, "shape", ()))
        dtype = np.dtype(getattr(czi, "dtype", np.float32)).name
        axes = _axis_order_label(getattr(czi, "dims", ()))
        axes = axes or _fallback_axis_order(shape)
        return (
            ImageSeriesInfo(
                index=0,
                key="0",
                name=path.stem,
                shape=shape,
                dtype=dtype,
                axes=axes,
            ),
        )
    records: list[ImageSeriesInfo] = []
    for index, (key, scene) in enumerate(scene_items):
        shape = tuple(int(size) for size in getattr(scene, "shape", ()))
        dtype = np.dtype(getattr(scene, "dtype", np.float32)).name
        axes = _axis_order_label(getattr(scene, "dims", ()))
        name = str(getattr(scene, "name", "") or f"Scene {key}")
        records.append(
            ImageSeriesInfo(
                index=index,
                key=str(key),
                name=name,
                shape=shape,
                dtype=dtype,
                axes=axes,
            )
        )
    return tuple(records)


def _scene_items(scenes) -> list[tuple[Any, Any]]:
    if scenes is None:
        return []
    if hasattr(scenes, "items"):
        try:
            return list(scenes.items())
        except Exception:
            return []
    try:
        return list(enumerate(scenes))
    except Exception:
        return []


def _czi_scene(czi, selected: ImageSeriesInfo):
    scenes = getattr(czi, "scenes", None)
    if scenes is None:
        return czi
    key: Any = selected.key
    try:
        key = int(key)
    except Exception:
        pass
    try:
        return scenes[key]
    except Exception:
        try:
            return list(scenes.values())[selected.index]
        except Exception:
            return list(scenes)[selected.index]


def _scene_payload(
    scene,
) -> tuple[
    Any,
    tuple[AxisMetadata, ...],
    tuple[ChannelMetadata, ...],
    dict[str, Any],
]:
    xarray = _safe_call(scene, "asxarray")
    if xarray is not None:
        data = getattr(xarray, "data", xarray)
        dims = tuple(str(dim) for dim in getattr(xarray, "dims", ()))
        attrs = dict(getattr(xarray, "attrs", {}) or {})
        axes = _axes_from_xarray(xarray, dims, tuple(int(size) for size in data.shape))
        channels = _channels_from_xarray(xarray)
        return data, axes, channels, attrs

    data = _safe_call(scene, "asarray")
    if data is None:
        data = np.asarray(scene)
    dims = _axis_order_label(getattr(scene, "dims", ()) or getattr(scene, "axes", ()))
    axes = _axes_from_order(dims or _fallback_axis_order(data.shape), data.shape)
    channels = _channels_from_labels(getattr(scene, "channels", ()))
    attrs = dict(getattr(scene, "attrs", {}) or {})
    return data, axes, channels, attrs


def _series_from_generic_image(
    index: int,
    key,
    image,
    *,
    fallback_name: str = "",
) -> ImageSeriesInfo:
    shape = tuple(int(size) for size in getattr(image, "shape", ()) or ())
    if not shape and isinstance(getattr(image, "sizes", None), dict):
        shape = tuple(int(size) for size in image.sizes.values())
    axes = _axis_order_label(
        getattr(image, "dims", ())
        or getattr(image, "axes", ())
        or (
            tuple(image.sizes.keys())
            if isinstance(getattr(image, "sizes", None), dict)
            else ()
        )
    )
    dtype = np.dtype(getattr(image, "dtype", np.float32)).name
    name = (
        str(getattr(image, "name", "") or "")
        or str(getattr(image, "path", "") or "")
        or fallback_name
        or f"Series {index + 1}"
    )
    return ImageSeriesInfo(
        index=index,
        key=str(key),
        name=name,
        shape=shape,
        dtype=dtype,
        axes=axes,
    )


def _container_image(container, selected: ImageSeriesInfo):
    if container is None:
        raise ValueError(f"Series {selected.index} is not available.")
    key: Any = selected.key
    try:
        key = int(key)
    except Exception:
        pass
    try:
        return container[key]
    except Exception:
        pass
    try:
        return list(container.values())[selected.index]
    except Exception:
        return list(container)[selected.index]


def _microscope_dataset(
    data,
    path: Path,
    inspection: SourceInspection,
    selected: ImageSeriesInfo,
    axes: tuple[AxisMetadata, ...],
    channels: tuple[ChannelMetadata, ...],
    metadata: Any,
    *,
    metadata_source: str,
    reader: str,
) -> ImageDataset:
    acquisition = _acquisition_from_metadata(metadata)
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
        metadata_source=metadata_source,
        channels=channels,
        acquisition=acquisition,
        source=source,
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    return ImageDataset(
        data,
        state,
        inspection,
        selected,
        original_metadata=metadata,
        provenance={"reader": reader, "source_uri": str(path)},
    )


def _bioio_set_scene(image, scene) -> None:
    try:
        image.set_scene(scene)
    except Exception:
        try:
            image.set_scene(int(scene))
        except Exception:
            pass


def _bioio_metadata(image) -> dict[str, Any]:
    return {
        "metadata": getattr(image, "metadata", None),
        "current_scene": getattr(image, "current_scene", ""),
    }


def _bioio_axes(image, shape: tuple[int, ...]) -> tuple[AxisMetadata, ...]:
    order = str(getattr(getattr(image, "dims", None), "order", ""))
    sizes = getattr(image, "physical_pixel_sizes", None)
    scales = {
        "X": _optional_float(getattr(sizes, "X", None)),
        "Y": _optional_float(getattr(sizes, "Y", None)),
        "Z": _optional_float(getattr(sizes, "Z", None)),
    }
    axes: list[AxisMetadata] = []
    for label in _split_axis_order(order):
        scale = scales.get(label) or 1.0
        unit = "micrometer" if label in scales and scales.get(label) else None
        axes.append(
            AxisMetadata(
                name=_axis_name(label),
                type=_axis_type(label),
                unit=unit,
                scale=scale,
            )
        )
    if len(axes) != len(shape):
        return _axes_from_order(_fallback_axis_order(shape), shape)
    return tuple(axes)


def _bioio_channels(image) -> tuple[ChannelMetadata, ...]:
    return _channels_from_labels(getattr(image, "channel_names", None) or ())


def _axes_from_xarray(
    xarray,
    dims: tuple[str, ...],
    shape: tuple[int, ...],
) -> tuple[AxisMetadata, ...]:
    attrs = dict(getattr(xarray, "attrs", {}) or {})
    scales = attrs.get("coord_scales", {}) or attrs.get("mpp", {}) or {}
    units = attrs.get("coord_units", {}) or {}
    axes: list[AxisMetadata] = []
    for dim in dims:
        label = dim.upper()
        scale = _optional_float(_mapping_value(scales, dim, label))
        if scale is None:
            scale = _scale_from_coord(getattr(xarray, "coords", {}), dim)
        scale, inferred_unit = _normalized_axis_scale_and_unit(
            label,
            scale,
            _mapping_value(units, dim, label),
        )
        axes.append(
            AxisMetadata(
                name=_axis_name(label),
                type=_axis_type(label),
                unit=inferred_unit,
                scale=scale,
            )
        )
    if len(axes) != len(shape):
        return _axes_from_order(_fallback_axis_order(shape), shape)
    return tuple(axes)


def _scale_from_coord(coords, dim: str) -> float | None:
    try:
        coord = coords[dim]
        values = np.asarray(getattr(coord, "values", coord))
    except Exception:
        return None
    if values.size < 2:
        return None
    try:
        delta = float(values.reshape(-1)[1] - values.reshape(-1)[0])
    except Exception:
        return None
    delta = abs(delta)
    return delta if np.isfinite(delta) and delta > 0 else None


def _normalized_axis_scale_and_unit(
    label: str,
    scale: float | None,
    unit,
) -> tuple[float, str | None]:
    axis_type = _axis_type(label)
    if axis_type == "channel":
        return 1.0, None
    value = float(scale) if scale is not None else 1.0
    text = str(unit or "").strip().lower()
    if axis_type == "space":
        if text in {"meter", "metre", "meters", "metres", "m"}:
            return value * 1_000_000.0, "micrometer"
        if text in {"nanometer", "nanometre", "nanometers", "nanometres", "nm"}:
            return value / 1_000.0, "micrometer"
        if text in {
            "micrometer",
            "micrometre",
            "micrometers",
            "micrometres",
            "um",
            "\u00b5m",
            "\u03bcm",
        }:
            return value, "micrometer"
        return value, str(unit) if unit else None
    if axis_type == "time":
        if text in {"millisecond", "milliseconds", "ms"}:
            return value / 1_000.0, "second"
        if text in {"second", "seconds", "sec", "s"}:
            return value, "second"
        inferred_unit = "second" if scale is not None else None
        return value, str(unit) if unit else inferred_unit
    return value, str(unit) if unit else None


def _channel_labels_from_xarray(xarray) -> tuple[str, ...]:
    labels = []
    coords = getattr(xarray, "coords", {})
    for key in ("C", "c", "S", "s"):
        try:
            values = np.asarray(coords[key].values)
        except Exception:
            continue
        labels = [str(value) for value in values.reshape(-1)]
        break
    return tuple(labels)


def _channels_from_xarray(xarray) -> tuple[ChannelMetadata, ...]:
    attrs = dict(getattr(xarray, "attrs", {}) or {})
    channel_records = attrs.get("channels")
    records: list[ChannelMetadata] = []
    if isinstance(channel_records, dict):
        for key, item in channel_records.items():
            if not isinstance(item, dict):
                records.append(ChannelMetadata(name=str(key)))
                continue
            excitation = _optional_float(
                _mapping_value(item, "ExcitationWavelength", "excitation")
            )
            emission = _optional_float(
                _mapping_value(item, "EmissionWavelength", "emission")
            )
            if emission is None:
                emission = _wavelength_midpoint(
                    _mapping_value(item, "DetectionWavelength", "detection")
                )
            name = _mapping_value(item, "Fluor", "DyeName", "Name", "ChannelName")
            records.append(
                ChannelMetadata(
                    name=str(name or key),
                    color=channel_color_int(_mapping_value(item, "Color", "color")),
                    excitation_wavelength=excitation,
                    excitation_wavelength_unit="nm" if excitation else "",
                    emission_wavelength=emission,
                    emission_wavelength_unit="nm" if emission else "",
                )
            )
    if records:
        return tuple(records)
    return _channels_from_labels(_channel_labels_from_xarray(xarray))


def _wavelength_midpoint(value) -> float | None:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        first = _optional_float(value[0])
        second = _optional_float(value[1])
        if first is not None and second is not None:
            return (first + second) / 2.0
    return _optional_float(value)


def _channels_from_labels(labels) -> tuple[ChannelMetadata, ...]:
    records = []
    for label in labels or ():
        text = str(label)
        if not text:
            continue
        records.append(ChannelMetadata(name=text))
    return tuple(records)


def _acquisition_from_metadata(metadata: Any) -> AcquisitionMetadata:
    deconvolved, method = detect_deconvolution_metadata(metadata)
    return AcquisitionMetadata(
        objective=_first_text(
            metadata,
            (
                "objective",
                "objectiveName",
                "objective_name",
                "ObjectiveName",
            ),
        ),
        instrument=_first_text(metadata, ("instrument", "microscope", "system")),
        detector=_first_text(metadata, ("detector", "camera", "Detector")),
        objective_na=_first_number(
            metadata,
            (
                "objectiveNumericalAperture",
                "numericalAperture",
                "lensNA",
                "lens_na",
                "na",
            ),
        ),
        objective_magnification=_first_number(
            metadata,
            (
                "objectiveMagnification",
                "nominalMagnification",
                "magnification",
            ),
        ),
        objective_immersion=_first_text(
            metadata,
            ("immersion", "immersionType", "objectiveImmersion"),
        ),
        refractive_index=_first_number(
            metadata,
            (
                "refractiveIndex",
                "refractive_index",
                "immersionRefractiveIndex",
            ),
        ),
        deconvolution_applied=deconvolved,
        deconvolution_method=method,
    )


def _first_text(metadata: Any, keys: tuple[str, ...]) -> str:
    for key in keys:
        value = _find_metadata_value(metadata, key)
        if not _is_empty_metadata_value(value) and not isinstance(
            value,
            (Mapping, list, tuple, set),
        ):
            return str(value)
    return ""


def _first_number(metadata: Any, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _find_metadata_value(metadata, key)
        number = _optional_float(value)
        if number is not None:
            return number
    return None


def _find_metadata_value(value: Any, target_key: str, depth: int = 0) -> Any:
    if value is None or depth > 10:
        return None
    normalized_target = _normalized_key(target_key)
    if isinstance(value, str):
        return _find_xml_metadata_value(value, normalized_target)
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _normalized_key(str(key)) == normalized_target:
                return item
        for item in value.values():
            found = _find_metadata_value(item, target_key, depth + 1)
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple, set)):
        for item in value:
            found = _find_metadata_value(item, target_key, depth + 1)
            if found is not None:
                return found
        return None
    if hasattr(value, target_key):
        return getattr(value, target_key)
    for key, item in _metadata_object_items(value):
        if _normalized_key(str(key)) == normalized_target:
            return item
    for _key, item in _metadata_object_items(value):
        found = _find_metadata_value(item, target_key, depth + 1)
        if found is not None:
            return found
    return None


def _metadata_object_items(value: Any) -> tuple[tuple[str, Any], ...]:
    if is_dataclass(value) and not isinstance(value, type):
        try:
            return tuple(asdict(value).items())
        except Exception:
            return ()
    if hasattr(value, "_asdict"):
        try:
            return tuple(value._asdict().items())
        except Exception:
            return ()
    attrs = getattr(value, "__dict__", None)
    if isinstance(attrs, Mapping):
        return tuple((str(key), item) for key, item in attrs.items())
    return ()


def _find_xml_metadata_value(text: str, normalized_target: str) -> Any:
    if "<" not in text or ">" not in text:
        return None
    try:
        root = ElementTree.fromstring(text)
    except ElementTree.ParseError:
        return _find_xml_metadata_value_regex(text, normalized_target)
    for element in root.iter():
        tag = _xml_local_name(element.tag)
        if _normalized_key(tag) == normalized_target:
            if element.text and element.text.strip():
                return element.text.strip()
            for item in element.attrib.values():
                if str(item).strip():
                    return item
        for key, item in element.attrib.items():
            if _normalized_key(_xml_local_name(key)) == normalized_target:
                return item
    return None


def _find_xml_metadata_value_regex(text: str, normalized_target: str) -> str | None:
    pattern = (
        r"<(?P<tag>[A-Za-z0-9_:.-]+)(?:\s[^>]*)?>"
        r"(?P<value>[^<]+)</(?P=tag)>"
    )
    for match in re.finditer(pattern, text):
        if _normalized_key(match.group("tag").split(":")[-1]) == normalized_target:
            value = match.group("value").strip()
            if value:
                return value
    return None


def _xml_local_name(value: Any) -> str:
    text = str(value)
    if "}" in text:
        text = text.rsplit("}", 1)[-1]
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return text


def _metadata_text(metadata: Any) -> str:
    try:
        text = repr(metadata)
    except Exception:
        return ""
    if len(text) > 250_000:
        return text[:250_000]
    return text


def _deconvolution_label(term: str) -> str:
    mapping = {
        "deconvolution": "metadata mentions deconvolution",
        "deconvolved": "metadata marks image as deconvolved",
        "richardson-lucy": "Richardson-Lucy",
        "richardson lucy": "Richardson-Lucy",
        "huygens": "Huygens",
        "deconvolutionlab": "DeconvolutionLab",
        "widefield decon": "widefield deconvolution",
        "blind decon": "blind deconvolution",
    }
    return mapping.get(term, term)


def _axis_order_label(labels) -> str:
    if not labels:
        return ""
    return "".join(str(label).upper() for label in labels)


def _split_axis_order(order: str) -> tuple[str, ...]:
    text = str(order or "")
    if "," in text:
        return tuple(part.strip().upper() for part in text.split(",") if part.strip())
    return tuple(char.upper() for char in text if char.strip())


def _axes_from_order(order: str, shape: tuple[int, ...]) -> tuple[AxisMetadata, ...]:
    labels = _split_axis_order(order)
    if len(labels) != len(shape):
        labels = _split_axis_order(_fallback_axis_order(shape))
    return tuple(
        AxisMetadata(_axis_name(label), _axis_type(label))
        for label in labels
    )


def _fallback_axis_order(shape: tuple[int, ...]) -> str:
    ndim = len(tuple(shape))
    if ndim == 0:
        return ""
    if ndim == 1:
        return "X"
    if ndim == 2:
        return "YX"
    if ndim == 3:
        if shape[-1] in {3, 4}:
            return "YXS"
        return "ZYX"
    if ndim == 4:
        return "CZYX"
    if ndim == 5:
        return "TCZYX"
    prefix = "".join(f"D{index}" for index in range(ndim - 5))
    return prefix + "TCZYX"


def _axis_name(label: str) -> str:
    label = str(label).strip()
    mapping = {
        "C": "c",
        "H": "scene",
        "M": "m",
        "P": "position",
        "S": "rgb",
        "T": "t",
        "X": "x",
        "Y": "y",
        "Z": "z",
    }
    return mapping.get(label.upper(), label.lower() or "d")


def _axis_type(label: str) -> str:
    normalized = str(label).upper()
    if normalized == "T":
        return "time"
    if normalized in {"X", "Y", "Z"}:
        return "space"
    if normalized in {"C", "S"}:
        return "channel"
    return "unknown"


def _mapping_value(mapping, *keys: str):
    if not isinstance(mapping, dict):
        return None
    normalized = {_normalized_key(str(key)): value for key, value in mapping.items()}
    for key in keys:
        value = normalized.get(_normalized_key(key))
        if not _is_empty_metadata_value(value):
            return value
    return None


def _normalized_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _safe_call(obj, name: str):
    try:
        method = getattr(obj, name)
    except Exception:
        return None
    try:
        return method()
    except Exception:
        return None


def _optional_float(value) -> float | None:
    if _is_empty_metadata_value(value):
        return None
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _optional_int(value) -> int | None:
    if _is_empty_metadata_value(value):
        return None
    try:
        return int(value)
    except Exception:
        return None


def _is_empty_metadata_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and value == "")
