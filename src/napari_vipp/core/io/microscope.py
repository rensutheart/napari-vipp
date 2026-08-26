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
from tifffile import TiffFile

from napari_vipp.core.channel_colors import channel_color_int
from napari_vipp.core.io.errors import ImageSourceError, ImageSourceErrorCode
from napari_vipp.core.io.model import (
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
)
from napari_vipp.core.io.tiff import (
    image_state_from_tiff_inspection,
    inspect_tiff,
    read_tiff,
)
from napari_vipp.core.metadata import (
    AXIS_CONFIDENCE_INFERRED,
    AcquisitionMetadata,
    AxisMetadata,
    ChannelMetadata,
    ImageState,
    SourceMetadata,
    image_state_from_array,
)

MICROSCOPE_SUFFIXES = frozenset(
    {
        ".czi",
        ".ims",
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
    "Microscope files (*.nd2 *.czi *.ims *.lsm *.lif *.lof *.xlif "
    "*.oir *.oib *.oif *.vsi)"
)

_BIOIO_FALLBACK_SUFFIXES = {
    ".ims",
    ".lif",
    ".lof",
    ".oib",
    ".oif",
    ".oir",
    ".vsi",
    ".xlif",
}
_BIOFORMATS_REQUIRED_SUFFIXES = {".ims", ".oib", ".oif", ".oir", ".vsi"}
_FORMAT_BY_SUFFIX = {
    ".czi": "zeiss-czi",
    ".ims": "imaris-ims",
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
_EAGER_READER_CAPABILITIES = (
    "pixel-lazy-inspection",
    "eager-data",
    "decoded-size-estimate",
)
_LAZY_READER_CAPABILITIES = (
    "pixel-lazy-inspection",
    "lazy-data",
    "decoded-size-estimate",
)
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


def image_state_from_microscope_inspection(
    path: Path,
    inspection: SourceInspection,
    series_index: int = 0,
) -> ImageState:
    """Build axis-safe microscope metadata without loading image pixels."""
    selected = _selected_series(inspection, series_index)
    if selected.image_state is not None:
        return replace(
            selected.image_state,
            source_name=selected.name or path.name,
            source=replace(
                selected.image_state.source,
                uri=str(path),
                format=inspection.format,
                series_index=selected.index,
                series_name=selected.name,
            ),
        )

    if path.suffix.lower() == ".lsm":
        tiff_inspection = inspect_tiff(path)
        state = image_state_from_tiff_inspection(
            path,
            tiff_inspection,
            selected.index,
        )
        return replace(
            state,
            metadata_source="Zeiss LSM TIFF metadata",
            source=replace(state.source, format="zeiss-lsm"),
        )

    state = image_state_from_array(
        _MetadataOnlyMicroscopeArray(selected.shape, selected.dtype),
        source_name=selected.name or path.name,
        axes=_axes_from_order(selected.axes, selected.shape),
        metadata_source=f"{inspection.format} source inspection",
        source=SourceMetadata(
            uri=str(path),
            format=inspection.format,
            series_index=selected.index,
            series_name=selected.name,
        ),
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    if selected.kind == "labels":
        state = replace(state, kind="label image")
    return state


class _MetadataOnlyMicroscopeArray:
    """Lazy shape/dtype carrier for microscope inspection metadata."""

    def __init__(self, shape: tuple[int, ...], dtype: str) -> None:
        self.shape = tuple(int(size) for size in shape)
        self.dtype = np.dtype(dtype)

    def compute(self):
        raise RuntimeError("A microscope metadata-only image cannot load pixels.")


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
        selected = ImageSeriesInfo(
            index=0,
            key="0",
            name=path.stem,
            shape=shape,
            dtype=dtype,
            axes=axes,
        )
        selected = _series_with_inspection_state(
            selected,
            path=path,
            format_name="nikon-nd2",
            axes=_nd2_axes(nd_file, shape),
            channels=_nd2_channels(nd_file),
            acquisition=_acquisition_from_metadata(original),
            metadata_source="Nikon ND2 metadata",
            reader_key="nd2",
            reader_version=_reader_version(nd2),
            capabilities=_LAZY_READER_CAPABILITIES,
        )
    series = (selected,)
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
    return _microscope_dataset(
        data,
        path,
        inspection,
        selected,
        axes,
        channels,
        original,
        metadata_source="Nikon ND2 metadata",
        reader="nd2",
    )


def _inspect_czi(path: Path) -> SourceInspection:
    czifile = _optional_import("czifile", path.suffix)
    with czifile.CziFile(str(path)) as czi:
        original = _safe_call(czi, "metadata")
        series = tuple(
            _series_with_reader_metadata(
                selected,
                image=scene,
                path=path,
                format_name="zeiss-czi",
                original_metadata=original,
                metadata_source="Zeiss CZI metadata",
                reader_key="czifile",
                reader_version=_reader_version(czifile),
                capabilities=_EAGER_READER_CAPABILITIES,
            )
            for selected, scene in _czi_series_and_scenes(czi, path)
        )
    return SourceInspection(str(path), "zeiss-czi", series, original)


def _read_czi(path: Path, series_index: int = 0) -> ImageDataset:
    inspection = _inspect_czi(path)
    selected = _selected_series(inspection, series_index)
    czifile = _optional_import("czifile", path.suffix)
    with czifile.CziFile(str(path)) as czi:
        scene = _czi_scene(czi, selected)
        data, axes, channels, attrs = _scene_payload(scene)
        original = _safe_call(czi, "metadata")
    return _microscope_dataset(
        data,
        path,
        inspection,
        selected,
        axes,
        channels,
        {"metadata": original, "attrs": attrs},
        metadata_source="Zeiss CZI metadata",
        reader="czifile",
    )


def _inspect_lif(path: Path) -> SourceInspection:
    liffile = _optional_import("liffile", path.suffix)
    with liffile.LifFile(str(path)) as lif:
        images = _scene_items(getattr(lif, "images", None))
        original = _lif_original_metadata(lif)
        series = tuple(
            _series_with_reader_metadata(
                _series_from_generic_image(index, key, image),
                image=image,
                path=path,
                format_name=microscope_format_for_path(path),
                original_metadata=original,
                metadata_source="Leica metadata",
                reader_key="liffile",
                reader_version=_reader_version(liffile),
                capabilities=_EAGER_READER_CAPABILITIES,
            )
            for index, (key, image) in enumerate(images)
        )
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
        original = _lif_original_metadata(lif)
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
        original = getattr(oir, "xml_metadata", None)
        selected = _series_from_generic_image(
            0,
            "0",
            oir,
            fallback_name=path.name,
        )
        series = (
            _series_with_reader_metadata(
                selected,
                image=oir,
                path=path,
                format_name=microscope_format_for_path(path),
                original_metadata=original,
                metadata_source="Olympus OIR metadata",
                reader_key="oirfile",
                reader_version=_reader_version(oirfile),
                capabilities=_EAGER_READER_CAPABILITIES,
            ),
        )
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
        original = _plain_metadata_mapping(getattr(oif, "mainfile", None))
        series = tuple(
            _oif_series_info(
                oif,
                path=path,
                index=index,
                series_item=series_item,
                metadata=original,
                reader_version=_reader_version(oiffile),
            )
            for index, series_item in enumerate(tuple(getattr(oif, "series", ()) or ()))
        )
        if not series:
            base = _series_from_generic_image(0, "0", oif, fallback_name=path.stem)
            axes = _oif_axes(original, base.axes, base.shape)
            series = (
                _series_with_inspection_state(
                    base,
                    path=path,
                    format_name=microscope_format_for_path(path),
                    axes=axes,
                    channels=_oif_channels(original, base.shape, base.axes),
                    acquisition=_acquisition_from_metadata(original),
                    metadata_source="Olympus OIF/OIB metadata",
                    reader_key="oiffile",
                    reader_version=_reader_version(oiffile),
                ),
            )
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
        data = oif.asarray(selected.index)
        axes = selected.image_state.axes if selected.image_state is not None else ()
        channels = (
            selected.image_state.channels if selected.image_state is not None else ()
        )
        original = _plain_metadata_mapping(getattr(oif, "mainfile", None))
    return _microscope_dataset(
        data,
        path,
        inspection,
        selected,
        axes,
        channels,
        original,
        metadata_source="Olympus OIF/OIB metadata",
        reader="oiffile",
    )


def _inspect_lsm(path: Path) -> SourceInspection:
    tiff_inspection = inspect_tiff(path)
    tifffile = import_module("tifffile")
    with TiffFile(path) as tif:
        lsm_metadata = dict(getattr(tif, "lsm_metadata", None) or {})
        series = tuple(
            _series_with_inspection_state(
                selected,
                path=path,
                format_name="zeiss-lsm",
                axes=_lsm_axes(
                    selected,
                    tif.series[selected.index],
                    lsm_metadata,
                ),
                channels=_lsm_channels(selected, lsm_metadata),
                acquisition=_lsm_acquisition(lsm_metadata),
                metadata_source="Zeiss LSM metadata",
                reader_key="tifffile",
                reader_version=_reader_version(tifffile),
                capabilities=_EAGER_READER_CAPABILITIES,
            )
            for selected in tiff_inspection.series
        )
    return SourceInspection(
        tiff_inspection.uri,
        "zeiss-lsm",
        series,
        lsm_metadata,
    )


def _read_lsm(path: Path, series_index: int = 0) -> ImageDataset:
    inspection = _inspect_lsm(path)
    selected = _selected_series(inspection, series_index)
    tiff_dataset = read_tiff(path, series_index)
    dataset = _microscope_dataset(
        tiff_dataset.data,
        path,
        inspection,
        selected,
        selected.image_state.axes if selected.image_state is not None else (),
        selected.image_state.channels if selected.image_state is not None else (),
        inspection.original_metadata,
        metadata_source="Zeiss LSM metadata",
        reader="tifffile",
    )
    return ImageDataset(
        dataset.data,
        dataset.image_state,
        inspection,
        selected,
        original_metadata=inspection.original_metadata,
        multiscale_levels=tiff_dataset.multiscale_levels,
        associated_labels=tiff_dataset.associated_labels,
        provenance={"reader": "tifffile", "source_uri": str(path)},
    )


def _inspect_bioio(path: Path, format_hint: str) -> SourceInspection:
    bioio = _optional_bioio(path.suffix)
    image = _new_bioio_image(
        bioio,
        path,
        format_hint=format_hint,
        stage="inspect",
    )
    series: list[ImageSeriesInfo] = []
    original_by_scene: dict[str, Any] = {}
    for index, scene in enumerate(tuple(getattr(image, "scenes", ()) or (0,))):
        _bioio_set_scene(image, scene)
        shape = tuple(int(size) for size in getattr(image, "shape", ()))
        dtype = np.dtype(getattr(image, "dtype", np.float32)).name
        order = str(getattr(getattr(image, "dims", None), "order", ""))
        metadata = _bioio_metadata(image)
        key = str(scene)
        original_by_scene[key] = metadata
        selected = ImageSeriesInfo(
            index=index,
            key=key,
            name=key or f"Series {index + 1}",
            shape=shape,
            dtype=dtype,
            axes=order,
        )
        series.append(
            _series_with_inspection_state(
                selected,
                path=path,
                format_name=f"{format_hint}+bioio",
                axes=_bioio_axes(image, shape),
                channels=_bioio_channels(image),
                acquisition=_acquisition_from_metadata(metadata),
                metadata_source="BioIO reader metadata",
                reader_key=_bioio_reader_key(path.suffix),
                reader_version=_bioio_reader_version(bioio, path.suffix),
                capabilities=_bioio_capabilities(image),
            )
        )
    if not series:
        raise ValueError(f"No image series found in {path}")
    return SourceInspection(
        str(path),
        f"{format_hint}+bioio",
        tuple(series),
        original_by_scene,
    )


def _read_bioio(path: Path, series_index: int, format_hint: str) -> ImageDataset:
    inspection = _inspect_bioio(path, format_hint)
    selected = _selected_series(inspection, series_index)
    bioio = _optional_bioio(path.suffix)
    image = _new_bioio_image(
        bioio,
        path,
        format_hint=format_hint,
        stage="read",
    )
    _bioio_set_scene(image, selected.key)
    data = getattr(image, "dask_data", None)
    if data is None:
        data = image.data
    metadata = _bioio_metadata(image)
    axes = _bioio_axes(image, tuple(int(size) for size in data.shape))
    channels = _bioio_channels(image)
    source_format = f"{format_hint}+bioio"
    dataset = _microscope_dataset(
        data,
        path,
        inspection,
        selected,
        axes,
        channels,
        metadata,
        metadata_source="BioIO reader metadata",
        reader=_bioio_reader_key(path.suffix),
    )
    if dataset.image_state.source.format != source_format:
        raise ValueError(
            "BioIO source format changed between inspection and read: "
            f"{dataset.image_state.source.format!r} versus {source_format!r}."
        )
    return dataset


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
    normalized_suffix = str(suffix or "").lower()
    try:
        bioio = import_module("bioio")
    except ImportError as error:
        raise _bioio_optional_dependency_error(
            error,
            normalized_suffix,
            module_name="bioio",
        ) from error
    if normalized_suffix in _BIOFORMATS_REQUIRED_SUFFIXES:
        try:
            import_module("bioio_bioformats")
        except ImportError as error:
            raise _bioio_optional_dependency_error(
                error,
                normalized_suffix,
                module_name="bioio_bioformats",
            ) from error
    return bioio


def _bioio_optional_dependency_error(
    _error: ImportError,
    suffix: str,
    *,
    module_name: str,
) -> OptionalMicroscopeReaderError:
    native_command = _NATIVE_INSTALL_COMMANDS.get(suffix)
    command = native_command or _BIOIO_INSTALL_COMMAND
    hint = (
        f"Install the format-specific extra with: {native_command}. "
        f"For BioIO/Bio-Formats fallback, install: {_BIOIO_INSTALL_COMMAND}"
        if native_command
        else f"Install the fallback extra with: {_BIOIO_INSTALL_COMMAND}"
    )
    return OptionalMicroscopeReaderError(
        "This microscope format requires an optional dependency: "
        f"{module_name!r}. {hint}",
        suffix=suffix,
        format_name=_FORMAT_BY_SUFFIX.get(suffix, ""),
        module_name=module_name,
        install_command=command,
        fallback_install_command=(_BIOIO_INSTALL_COMMAND if native_command else ""),
    )


def _bioio_reader_key(suffix: str) -> str:
    return (
        "bioio-bioformats"
        if str(suffix or "").lower() in _BIOFORMATS_REQUIRED_SUFFIXES
        else "bioio"
    )


def _bioio_reader_version(bioio, suffix: str) -> str:
    bioio_version = _reader_version(bioio)
    if str(suffix or "").lower() not in _BIOFORMATS_REQUIRED_SUFFIXES:
        return bioio_version
    try:
        plugin_version = _reader_version(import_module("bioio_bioformats"))
    except ImportError:
        plugin_version = ""
    if bioio_version and plugin_version:
        return f"bioio {bioio_version}; bioio-bioformats {plugin_version}"
    return plugin_version or bioio_version


def _bioio_capabilities(image) -> tuple[str, ...]:
    descriptor = getattr(type(image), "dask_data", None)
    instance_values = getattr(image, "__dict__", {})
    has_lazy_data = descriptor is not None or (
        isinstance(instance_values, Mapping) and "dask_data" in instance_values
    )
    return _LAZY_READER_CAPABILITIES if has_lazy_data else _EAGER_READER_CAPABILITIES


def _new_bioio_image(
    bioio,
    path: Path,
    *,
    format_hint: str,
    stage: str,
):
    """Construct a BioIO image with an actionable Java/runtime boundary."""
    try:
        return bioio.BioImage(str(path))
    except Exception as error:
        if not _is_java_or_bioformats_readiness_error(error):
            raise
        raise ImageSourceError(
            ImageSourceErrorCode.JAVA_BIOFORMATS_READINESS,
            f"Bio-Formats could not initialize its Java runtime for {path.name}.",
            stage=stage,
            path=path,
            format=f"{format_hint}+bioio",
            backend="bioio-bioformats",
            remediation=(
                "Verify the Bio-Formats extra and a supported 64-bit Java runtime, "
                "restart VIPP, and retry."
            ),
        ) from error


def _is_java_or_bioformats_readiness_error(error: Exception) -> bool:
    text = f"{type(error).__name__}: {error}".casefold()
    return any(
        signal in text
        for signal in (
            "java",
            "jvm",
            "scyjava",
            "javabridge",
            "classnotfound",
            "noclassdeffound",
            "bioformats",
            "bio-formats",
        )
    )


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
    return nd2.imread(str(path), dask=True)


def _nd2_axis_order(nd_file, shape: tuple[int, ...]) -> str:
    sizes = getattr(nd_file, "sizes", None)
    if not isinstance(sizes, Mapping) or len(sizes) != len(shape):
        return _fallback_axis_order(shape)

    labels: list[str] = []
    declared_shape: list[int] = []
    for raw_label, raw_size in sizes.items():
        if not isinstance(raw_label, str):
            return _fallback_axis_order(shape)
        label = raw_label.strip().upper()
        if len(label) != 1 or label in labels:
            return _fallback_axis_order(shape)
        if (
            isinstance(raw_size, (bool, np.bool_))
            or not isinstance(raw_size, (int, np.integer))
            or int(raw_size) < 1
        ):
            return _fallback_axis_order(shape)
        labels.append(label)
        declared_shape.append(int(raw_size))

    if tuple(declared_shape) != tuple(shape):
        return _fallback_axis_order(shape)
    if labels:
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
        return _axes_from_order("", shape)
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


def _czi_series_and_scenes(
    czi,
    path: Path,
) -> tuple[tuple[ImageSeriesInfo, Any], ...]:
    scenes = getattr(czi, "scenes", None)
    scene_items = _scene_items(scenes)
    if not scene_items:
        shape = tuple(int(size) for size in getattr(czi, "shape", ()))
        dtype = np.dtype(getattr(czi, "dtype", np.float32)).name
        axes = _axis_order_label(getattr(czi, "dims", ()))
        axes = axes or _fallback_axis_order(shape)
        return (
            (
                ImageSeriesInfo(
                    index=0,
                    key="0",
                    name=path.stem,
                    shape=shape,
                    dtype=dtype,
                    axes=axes,
                ),
                czi,
            ),
        )
    records: list[tuple[ImageSeriesInfo, Any]] = []
    for index, (key, scene) in enumerate(scene_items):
        shape = tuple(int(size) for size in getattr(scene, "shape", ()))
        dtype = np.dtype(getattr(scene, "dtype", np.float32)).name
        axes = _axis_order_label(getattr(scene, "dims", ()))
        name = str(getattr(scene, "name", "") or f"Scene {key}")
        records.append(
            (
                ImageSeriesInfo(
                    index=index,
                    key=str(key),
                    name=name,
                    shape=shape,
                    dtype=dtype,
                    axes=axes,
                ),
                scene,
            )
        )
    return tuple(records)


def _czi_series(czi, path: Path) -> tuple[ImageSeriesInfo, ...]:
    """Return CZI item records without losing their stable scene keys."""
    return tuple(selected for selected, _scene in _czi_series_and_scenes(czi, path))


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
    except Exception as error:
        raise ValueError(
            f"CZI scene {selected.key!r} is no longer available; "
            "VIPP will not substitute another scene by position."
        ) from error


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
    axes = _axes_from_order(dims, data.shape)
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
    axes = axes or _fallback_axis_order(shape)
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


def _series_with_reader_metadata(
    selected: ImageSeriesInfo,
    *,
    image,
    path: Path,
    format_name: str,
    original_metadata: Any,
    metadata_source: str,
    reader_key: str,
    reader_version: str,
    capabilities: tuple[str, ...],
) -> ImageSeriesInfo:
    """Attach one metadata contract without decoding the image payload."""
    metadata = _reader_metadata_payload(image, original_metadata)
    return _series_with_inspection_state(
        selected,
        path=path,
        format_name=format_name,
        axes=_reader_axes(image, selected),
        channels=_reader_channels(image, selected),
        acquisition=_acquisition_from_reader_metadata(image, metadata),
        metadata_source=metadata_source,
        reader_key=reader_key,
        reader_version=reader_version,
        capabilities=capabilities,
    )


def _reader_metadata_payload(image, original_metadata: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for name in ("attrs", "objective", "scanner"):
        try:
            value = getattr(image, name)
        except Exception:
            continue
        if value is not None:
            payload[name] = value
    payload["metadata"] = original_metadata
    return payload


def _acquisition_from_reader_metadata(
    image,
    metadata: Mapping[str, Any],
) -> AcquisitionMetadata:
    acquisition = _acquisition_from_metadata(metadata)
    attrs = _reader_mapping(image, "attrs")
    objective = _reader_value(image, "objective")
    if not isinstance(objective, Mapping):
        objective = _mapping_value(attrs, "objective")
    if not isinstance(objective, Mapping):
        return acquisition

    name = _mapping_value(objective, "Name", "Model", "ObjectiveName")
    if not name:
        manufacturer = _mapping_value(objective, "Manufacturer")
        if isinstance(manufacturer, Mapping):
            name = _mapping_value(manufacturer, "Model", "Name")
    magnification = _optional_float(
        _mapping_value(
            objective,
            "NominalMagnification",
            "CalibratedMagnification",
            "Magnification",
        )
    )
    numerical_aperture = _optional_float(
        _mapping_value(objective, "LensNA", "NumericalAperture", "NA")
    )
    immersion = _mapping_value(objective, "Immersion", "ImmersionType")
    refractive_index = _optional_float(
        _mapping_value(
            objective,
            "ImmersionRefractiveIndex",
            "RefractiveIndex",
            "RefractionIndex",
        )
    )
    return replace(
        acquisition,
        objective=str(name or acquisition.objective),
        objective_magnification=(
            magnification
            if magnification is not None
            else acquisition.objective_magnification
        ),
        objective_na=(
            numerical_aperture
            if numerical_aperture is not None
            else acquisition.objective_na
        ),
        objective_immersion=str(immersion or acquisition.objective_immersion),
        refractive_index=(
            refractive_index
            if refractive_index is not None
            else acquisition.refractive_index
        ),
    )


def _reader_axes(
    image,
    selected: ImageSeriesInfo,
) -> tuple[AxisMetadata, ...]:
    labels = _split_axis_order(selected.axes)
    confidence = "explicit"
    if len(labels) != len(selected.shape):
        labels = _split_axis_order(_fallback_axis_order(selected.shape))
        confidence = AXIS_CONFIDENCE_INFERRED

    attrs = _reader_mapping(image, "attrs")
    scales = _merged_reader_mapping(image, attrs, "coord_scales")
    units = _merged_reader_mapping(image, attrs, "coord_units")
    offsets = _merged_reader_mapping(image, attrs, "coord_offsets")
    coords = _reader_mapping(image, "coords")
    dimension_units = _reader_dimension_units(image)

    axes: list[AxisMetadata] = []
    for label in labels:
        scale = _optional_float(_mapping_value(scales, label, label.lower()))
        if scale is None:
            scale = _scale_from_coord(coords, label)
        unit = _mapping_value(units, label, label.lower())
        if not unit:
            unit = dimension_units.get(label)
        normalized_scale, normalized_unit = _normalized_axis_scale_and_unit(
            label,
            scale,
            unit,
        )
        translation = _optional_float(
            _mapping_value(offsets, label, label.lower())
        )
        if translation is None:
            translation = _coord_origin(coords, label)
        if translation is None:
            translation = 0.0
        normalized_translation, _ = _normalized_axis_scale_and_unit(
            label,
            translation,
            unit,
        )
        if _axis_type(label) == "channel":
            normalized_translation = 0.0
        axes.append(
            AxisMetadata(
                name=_axis_name(label),
                type=_axis_type(label),
                unit=normalized_unit,
                scale=normalized_scale,
                translation=normalized_translation,
                confidence=confidence,
            )
        )
    return tuple(axes)


def _reader_channels(
    image,
    selected: ImageSeriesInfo,
) -> tuple[ChannelMetadata, ...]:
    labels = _split_axis_order(selected.axes)
    if "S" in labels and "C" not in labels:
        return ()
    channel_count = selected.shape[labels.index("C")] if "C" in labels else 1
    coords = _reader_mapping(image, "coords")
    names = _string_coord_values(coords, "C")
    if len(names) != channel_count:
        names = _reader_channel_names(image)

    channel_records = _reader_channel_records(image)
    if len(names) != channel_count and len(channel_records) == channel_count:
        names = tuple(channel_records)
    if "C" not in labels and len(names) != 1:
        return ()
    if len(names) != channel_count:
        names = tuple(f"Channel {index + 1}" for index in range(channel_count))

    records: list[ChannelMetadata] = []
    for name in names:
        item = channel_records.get(name)
        if item is None:
            folded_name = str(name).casefold()
            item = next(
                (
                    record
                    for key, record in channel_records.items()
                    if key.casefold() == folded_name
                ),
                None,
            )
        records.append(_channel_from_reader_record(str(name), item))
    return tuple(records)


def _reader_channel_records(image) -> dict[str, Any]:
    records: dict[str, Any] = {}
    attrs = _reader_mapping(image, "attrs")
    sources = (
        _mapping_value(attrs, "channels"),
        _reader_value(image, "channels"),
    )
    for source in sources:
        if isinstance(source, Mapping):
            for key, item in source.items():
                name = str(key)
                records.setdefault(name, item)
        elif isinstance(source, (list, tuple)):
            for item in source:
                name = str(getattr(item, "name", "") or "")
                if name:
                    records.setdefault(name, item)
    return records


def _channel_from_reader_record(name: str, item: Any) -> ChannelMetadata:
    if isinstance(item, Mapping):
        excitation = _optional_float(
            _mapping_value(
                item,
                "ExcitationWavelength",
                "excitation",
                "DyeMaxExcitation",
            )
        )
        emission = _optional_float(
            _mapping_value(
                item,
                "EmissionWavelength",
                "emission",
                "DyeMaxEmission",
            )
        )
        if emission is None:
            emission = _wavelength_midpoint(
                _mapping_value(item, "DetectionWavelength", "detection")
            )
        fluor = str(_mapping_value(item, "Fluor", "DyeName") or "")
        color = channel_color_int(_mapping_value(item, "Color", "color"))
    else:
        excitation = _optional_float(getattr(item, "excitation_wavelength", None))
        start = _optional_float(getattr(item, "start_wavelength", None))
        end = _optional_float(getattr(item, "end_wavelength", None))
        emission = (
            (start + end) / 2.0
            if start is not None and end is not None
            else None
        )
        fluor = str(getattr(item, "fluor", "") or "")
        color = channel_color_int(getattr(item, "color", None))
    return ChannelMetadata(
        name=name,
        color=color,
        fluor=fluor,
        excitation_wavelength=excitation,
        excitation_wavelength_unit="nm" if excitation is not None else "",
        emission_wavelength=emission,
        emission_wavelength_unit="nm" if emission is not None else "",
    )


def _reader_channel_names(image) -> tuple[str, ...]:
    for name in ("_channel_names", "channel_names"):
        value = _reader_value(image, name)
        if value:
            return tuple(str(item) for item in value)
    return ()


def _reader_dimension_units(image) -> dict[str, str]:
    dimensions = _reader_value(image, "_dimensions") or ()
    return {
        str(getattr(item, "label", "")).upper(): str(
            getattr(item, "unit", "") or ""
        )
        for item in dimensions
        if str(getattr(item, "label", "")).strip()
    }


def _merged_reader_mapping(
    image,
    attrs: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    merged: dict[str, Any] = {}
    attr_value = _mapping_value(attrs, name)
    if isinstance(attr_value, Mapping):
        merged.update(attr_value)
    direct = _reader_value(image, name)
    if isinstance(direct, Mapping):
        merged.update(direct)
    return merged


def _reader_mapping(image, name: str) -> Mapping[str, Any]:
    value = _reader_value(image, name)
    return value if isinstance(value, Mapping) else {}


def _reader_value(image, name: str) -> Any:
    try:
        return getattr(image, name)
    except Exception:
        return None


def _string_coord_values(coords: Mapping[str, Any], label: str) -> tuple[str, ...]:
    value = _mapping_value(coords, label, label.lower())
    if value is None:
        return ()
    try:
        values = np.asarray(getattr(value, "values", value)).reshape(-1)
    except Exception:
        return ()
    if values.dtype.kind not in {"O", "S", "U"}:
        return ()
    return tuple(str(item) for item in values)


def _coord_origin(coords: Mapping[str, Any], label: str) -> float | None:
    value = _mapping_value(coords, label, label.lower())
    if value is None:
        return None
    try:
        values = np.asarray(getattr(value, "values", value)).reshape(-1)
        if not values.size:
            return None
        return _optional_float(values[0])
    except Exception:
        return None


def _lif_original_metadata(lif) -> Any:
    for name in ("xml_header", "xml", "metadata"):
        value = _reader_value(lif, name)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        if value is not None and not (isinstance(value, str) and not value):
            return value
    return None


def _container_image(container, selected: ImageSeriesInfo):
    if container is None:
        raise ValueError(f"Series {selected.index} is not available.")
    key: Any = selected.key
    try:
        key = int(key)
    except Exception:
        pass
    if isinstance(container, Mapping):
        for candidate in (selected.key, key):
            try:
                return container[candidate]
            except (KeyError, IndexError, TypeError):
                continue
        raise ValueError(
            f"Series {selected.key!r} is no longer available; "
            "VIPP will not substitute another item by position."
        )
    try:
        return container[key]
    except Exception:
        try:
            return list(container)[selected.index]
        except Exception as error:
            raise ValueError(f"Series {selected.key!r} is not available.") from error


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
    observed_shape = tuple(int(size) for size in getattr(data, "shape", ()))
    raw_dtype = getattr(data, "dtype", None)
    if raw_dtype is None:
        raw_dtype = np.asarray(data).dtype
    observed_dtype = np.dtype(raw_dtype).name
    if observed_shape != selected.shape or observed_dtype != selected.dtype:
        raise ValueError(
            "Microscope reader contract mismatch for "
            f"{selected.name or selected.key!r}: inspection declared "
            f"{selected.axes} {selected.shape} {selected.dtype}, but {reader} "
            f"returned {observed_shape} {observed_dtype}."
        )

    contract_state = selected.image_state
    if contract_state is not None:
        axes = contract_state.axes
        channels = contract_state.channels
        acquisition = contract_state.acquisition
        metadata_source = contract_state.metadata_source
    else:
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


def _series_with_inspection_state(
    selected: ImageSeriesInfo,
    *,
    path: Path,
    format_name: str,
    axes: tuple[AxisMetadata, ...],
    channels: tuple[ChannelMetadata, ...],
    acquisition: AcquisitionMetadata,
    metadata_source: str,
    reader_key: str,
    reader_version: str,
    capabilities: tuple[str, ...] = _EAGER_READER_CAPABILITIES,
) -> ImageSeriesInfo:
    if len(axes) != len(selected.shape):
        raise ValueError(
            "Microscope inspection axis contract mismatch for "
            f"{selected.name or selected.key!r}: {len(axes)} axes describe "
            f"{len(selected.shape)} dimensions."
        )
    state = image_state_from_array(
        _MetadataOnlyMicroscopeArray(selected.shape, selected.dtype),
        source_name=selected.name or path.name,
        axes=axes,
        metadata_source=metadata_source,
        channels=channels,
        acquisition=acquisition,
        source=SourceMetadata(
            uri=str(path),
            format=format_name,
            series_index=selected.index,
            series_name=selected.name,
        ),
    )
    if state is None:
        raise ValueError(f"Could not build image metadata for {path}")
    if selected.kind == "labels":
        state = replace(state, kind="label image")
    estimated_decoded_bytes = int(
        np.prod(selected.shape, dtype=np.int64) * np.dtype(selected.dtype).itemsize
    )
    return replace(
        selected,
        image_state=state,
        reader_key=reader_key,
        reader_version=reader_version,
        capabilities=tuple(capabilities),
        estimated_decoded_bytes=estimated_decoded_bytes,
    )


def _reader_version(module: Any) -> str:
    for name in ("__version__", "version", "VERSION"):
        value = getattr(module, name, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _lsm_axes(
    selected: ImageSeriesInfo,
    tif_series,
    metadata: Mapping[str, Any],
) -> tuple[AxisMetadata, ...]:
    axes = list(_reader_axes(tif_series, selected))
    if selected.index != 0:
        return tuple(axes)
    voxel_scales = {
        "X": _optional_float(metadata.get("VoxelSizeX")),
        "Y": _optional_float(metadata.get("VoxelSizeY")),
        "Z": _optional_float(metadata.get("VoxelSizeZ")),
    }
    origins = {
        "X": _optional_float(metadata.get("OriginX")),
        "Y": _optional_float(metadata.get("OriginY")),
        "Z": _optional_float(metadata.get("OriginZ")),
    }
    labels = _split_axis_order(selected.axes)
    for index, label in enumerate(labels):
        if label not in voxel_scales or axes[index].unit:
            continue
        scale = voxel_scales[label]
        if scale is None or scale <= 0.0:
            continue
        normalized_scale, unit = _normalized_axis_scale_and_unit(
            label,
            scale,
            "meter",
        )
        translation, _ = _normalized_axis_scale_and_unit(
            label,
            origins[label] or 0.0,
            "meter",
        )
        axes[index] = replace(
            axes[index],
            scale=normalized_scale,
            translation=translation,
            unit=unit,
        )
    return tuple(axes)


def _lsm_channels(
    selected: ImageSeriesInfo,
    metadata: Mapping[str, Any],
) -> tuple[ChannelMetadata, ...]:
    labels = _split_axis_order(selected.axes)
    if "C" not in labels:
        return ()
    count = selected.shape[labels.index("C")]
    channel_colors = metadata.get("ChannelColors", {})
    if not isinstance(channel_colors, Mapping):
        channel_colors = {}
    names = tuple(str(value) for value in channel_colors.get("ColorNames", ()) or ())
    colors = tuple(channel_colors.get("Colors", ()) or ())
    scan_info = metadata.get("ScanInformation", {})
    detections: list[Mapping[str, Any]] = []
    if isinstance(scan_info, Mapping):
        for track in scan_info.get("Tracks", ()) or ():
            if not isinstance(track, Mapping):
                continue
            detections.extend(
                item
                for item in track.get("DetectionChannels", ()) or ()
                if isinstance(item, Mapping)
            )

    records: list[ChannelMetadata] = []
    for index in range(count):
        detection = detections[index] if index < len(detections) else {}
        name = (
            names[index]
            if index < len(names)
            else str(detection.get("ChannelName", "") or f"Channel {index + 1}")
        )
        start = _optional_float(detection.get("SpiWavelengthStart"))
        stop = _optional_float(detection.get("SpiWavelengthStop"))
        emission = (
            (start + stop) / 2.0
            if start is not None and stop is not None
            else None
        )
        records.append(
            ChannelMetadata(
                name=name,
                color=(
                    _lsm_color_int(colors[index]) if index < len(colors) else None
                ),
                fluor=str(detection.get("DyeName", "") or ""),
                emission_wavelength=emission,
                emission_wavelength_unit="nm" if emission is not None else "",
            )
        )
    return tuple(records)


def _lsm_color_int(value: Any) -> int | None:
    try:
        channels = tuple(int(item) for item in value)
    except Exception:
        return channel_color_int(value)
    if len(channels) < 3 or any(item < 0 or item > 255 for item in channels[:3]):
        return None
    return (channels[0] << 16) | (channels[1] << 8) | channels[2]


def _lsm_acquisition(metadata: Mapping[str, Any]) -> AcquisitionMetadata:
    scan_info = metadata.get("ScanInformation", {})
    if not isinstance(scan_info, Mapping):
        return AcquisitionMetadata()
    objective = str(scan_info.get("Objective", "") or "")
    match = re.search(
        r"(?P<magnification>\d+(?:\.\d+)?)\s*x\s*/\s*(?P<na>\d+(?:\.\d+)?)",
        objective,
        flags=re.IGNORECASE,
    )
    magnification = None
    numerical_aperture = None
    if match is not None:
        magnification = float(match.group("magnification"))
        numerical_aperture = float(match.group("na"))
    deconvolved, method = detect_deconvolution_metadata(scan_info)
    return AcquisitionMetadata(
        description=str(scan_info.get("Description", "") or ""),
        objective=objective,
        instrument=str(scan_info.get("Name", "") or ""),
        objective_magnification=magnification,
        objective_na=numerical_aperture,
        deconvolution_applied=deconvolved,
        deconvolution_method=method,
    )


def _oif_series_info(
    oif,
    *,
    path: Path,
    index: int,
    series_item,
    metadata: Mapping[str, Any],
    reader_version: str,
) -> ImageSeriesInfo:
    size_by_axis = _oif_axis_sizes(metadata)
    labels = list(_split_axis_order(getattr(series_item, "axes", "")))
    shape = list(tuple(int(size) for size in getattr(series_item, "shape", ()) or ()))
    if len(labels) != len(shape):
        labels = []
        shape = []
    for label in ("Y", "X"):
        size = size_by_axis.get(label)
        if label not in labels and size is not None and size > 0:
            labels.append(label)
            shape.append(size)
    if not labels:
        fallback_labels = _split_axis_order(str(getattr(oif, "axes", "")))
        fallback_shape = tuple(int(size) for size in getattr(oif, "shape", ()) or ())
        if len(fallback_labels) != len(fallback_shape):
            raise ValueError(f"Could not determine Olympus OIF/OIB axes for {path}")
        labels = list(fallback_labels)
        shape = list(fallback_shape)
    axes_text = _axis_order_label(labels)
    selected = ImageSeriesInfo(
        index=index,
        key=str(index),
        name=path.stem if index == 0 else f"{path.stem} [{index + 1}]",
        shape=tuple(shape),
        dtype=np.dtype(getattr(oif, "dtype", np.float32)).name,
        axes=axes_text,
    )
    return _series_with_inspection_state(
        selected,
        path=path,
        format_name=microscope_format_for_path(path),
        axes=_oif_axes(metadata, axes_text, selected.shape),
        channels=_oif_channels(metadata, selected.shape, axes_text),
        acquisition=_acquisition_from_metadata(metadata),
        metadata_source="Olympus OIF/OIB metadata",
        reader_key="oiffile",
        reader_version=reader_version,
    )


def _oif_axis_sizes(metadata: Mapping[str, Any]) -> dict[str, int]:
    sizes: dict[str, int] = {}
    for key, values in metadata.items():
        if not re.fullmatch(r"Axis \d+ Parameters Common", str(key)):
            continue
        if not isinstance(values, Mapping):
            continue
        label = str(values.get("AxisCode", "")).strip().upper()
        size = _optional_int(values.get("MaxSize"))
        if label and size is not None and size > 0:
            sizes[label] = size
    return sizes


def _oif_axes(
    metadata: Mapping[str, Any],
    order: str,
    shape: tuple[int, ...],
) -> tuple[AxisMetadata, ...]:
    labels = _split_axis_order(order)
    if len(labels) != len(shape):
        return _axes_from_order(order, shape)
    sections: dict[str, Mapping[str, Any]] = {}
    for key, values in metadata.items():
        if not re.fullmatch(r"Axis \d+ Parameters Common", str(key)):
            continue
        if isinstance(values, Mapping):
            label = str(values.get("AxisCode", "")).strip().upper()
            if label:
                sections[label] = values
    reference = metadata.get("Reference Image Parameter", {})
    if not isinstance(reference, Mapping):
        reference = {}

    axes: list[AxisMetadata] = []
    for label, size in zip(labels, shape, strict=True):
        section = sections.get(label, {})
        scale: float | None = None
        unit: Any = None
        if label == "X":
            scale = _oif_number(reference.get("WidthConvertValue"))
            unit = reference.get("WidthUnit")
        elif label == "Y":
            scale = _oif_number(reference.get("HeightConvertValue"))
            unit = reference.get("HeightUnit")
        if scale is None:
            scale = _oif_number(section.get("Interval"))
            unit = unit or section.get("PixUnit") or section.get("UnitName")
        if scale is None and size > 1:
            start = _oif_number(section.get("StartPosition"))
            end = _oif_number(section.get("EndPosition"))
            if start is not None and end is not None and end != start:
                scale = abs(end - start) / float(size - 1)
                unit = unit or section.get("PixUnit") or section.get("UnitName")
        normalized_scale, normalized_unit = _normalized_axis_scale_and_unit(
            label,
            scale,
            unit,
        )
        axes.append(
            AxisMetadata(
                name=_axis_name(label),
                type=_axis_type(label),
                unit=normalized_unit,
                scale=normalized_scale,
            )
        )
    return tuple(axes)


def _oif_channels(
    metadata: Mapping[str, Any],
    shape: tuple[int, ...],
    order: str,
) -> tuple[ChannelMetadata, ...]:
    labels = _split_axis_order(order)
    try:
        channel_count = shape[labels.index("C")]
    except (ValueError, IndexError):
        return ()
    sections: list[tuple[int, Mapping[str, Any]]] = []
    for key, values in metadata.items():
        match = re.fullmatch(r"Channel (\d+) Parameters", str(key))
        if match is None or not isinstance(values, Mapping):
            continue
        sections.append((int(match.group(1)), values))
    sections.sort(key=lambda item: item[0])
    records: list[ChannelMetadata] = []
    for _index, values in sections[:channel_count]:
        dye = str(values.get("DyeName", "") or "").strip()
        if dye.casefold() == "none":
            dye = ""
        name = dye or str(values.get("CH Name", "") or "").strip()
        excitation = _oif_number(values.get("ExcitationWavelength"))
        emission = _oif_number(values.get("EmissionWavelength"))
        records.append(
            ChannelMetadata(
                name=name,
                fluor=dye,
                excitation_wavelength=excitation,
                excitation_wavelength_unit="nm" if excitation is not None else "",
                emission_wavelength=emission,
                emission_wavelength_unit="nm" if emission is not None else "",
            )
        )
    return tuple(records)


def _oif_number(value: Any) -> float | None:
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"[-+]?\d+,\d+\.0", text):
            text = text[:-2].replace(",", ".")
        elif "," in text and "." not in text:
            text = text.replace(",", ".")
        value = text
    return _optional_float(value)


def _plain_metadata_mapping(value: Any, depth: int = 0) -> Any:
    if depth > 12:
        return repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): _plain_metadata_mapping(item, depth + 1)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return tuple(_plain_metadata_mapping(item, depth + 1) for item in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return value


def _bioio_set_scene(image, scene) -> None:
    first_error: Exception | None = None
    try:
        image.set_scene(scene)
        return
    except Exception as exc:
        first_error = exc
        try:
            image.set_scene(int(scene))
            return
        except Exception as second_error:
            raise ValueError(f"BioIO could not select scene {scene!r}.") from (
                first_error or second_error
            )


def _bioio_metadata(image) -> dict[str, Any]:
    metadata = {
        "metadata": getattr(image, "metadata", None),
        "current_scene": getattr(image, "current_scene", ""),
    }
    reader = getattr(image, "reader", None)
    biofile = getattr(reader, "_bf", None)
    if biofile is None:
        return metadata
    try:
        with biofile.ensure_open():
            global_metadata = biofile.global_metadata()
    except Exception:
        return metadata
    if isinstance(global_metadata, Mapping):
        metadata["global_metadata"] = dict(global_metadata)
    return metadata


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
        return _axes_from_order("", shape)
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
        return _axes_from_order("", shape)
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
    for key in ("C", "c"):
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
    objective = _first_text(
        metadata,
        (
            "objective",
            "objectiveName",
            "objective_name",
            "ObjectiveName",
            "objectiveLens",
            "objectiveModel",
        ),
    )
    parsed_magnification, parsed_na = _objective_numbers(objective)
    lens_power = _first_text(metadata, ("lensPower",))
    lens_magnification, lens_na = _objective_numbers(lens_power)
    if lens_magnification is not None and lens_na is not None:
        if not objective:
            objective = lens_power
        if parsed_magnification is None:
            parsed_magnification = lens_magnification
        if parsed_na is None:
            parsed_na = lens_na
    objective_na = _first_number(
        metadata,
        (
            "objectiveNumericalAperture",
            "numericalAperture",
            "lensNA",
            "lens_na",
            "naValue",
            "na",
        ),
    )
    objective_magnification = _first_number(
        metadata,
        (
            "objectiveMagnification",
            "nominalMagnification",
            "magnification",
        ),
    )
    return AcquisitionMetadata(
        description=_first_text(metadata, ("description", "notes", "comment")),
        acquisition_date=_first_text(
            metadata,
            (
                "acquisitionDate",
                "AcquisitionDateAndTime",
                "datetime",
                "creationDate",
            ),
        ),
        objective=objective,
        instrument=_first_text(
            metadata,
            ("instrument", "microscope", "system", "systemName"),
        ),
        detector=_first_text(metadata, ("detector", "camera", "Detector")),
        objective_na=(objective_na if objective_na is not None else parsed_na),
        objective_magnification=(
            objective_magnification
            if objective_magnification is not None
            else parsed_magnification
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
                "refractionIndex",
                "refraction",
            ),
        ),
        deconvolution_applied=deconvolved,
        deconvolution_method=method,
    )


def _objective_numbers(value: str) -> tuple[float | None, float | None]:
    """Parse conservative magnification/NA fallbacks from an objective label."""
    text = str(value or "").strip()
    magnification_match = re.search(
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*[x\u00d7](?=\s|[,;/]|$)",
        text,
        flags=re.IGNORECASE,
    )
    aperture_match = re.search(
        r"(?<![\w.])(\d+(?:\.\d+)?)\s*NA\b",
        text,
        flags=re.IGNORECASE,
    )
    if aperture_match is None:
        aperture_match = re.search(
            r"[x\u00d7]\s*/\s*(\d+(?:\.\d+)?)\b",
            text,
            flags=re.IGNORECASE,
        )
    return (
        float(magnification_match.group(1))
        if magnification_match is not None
        else None,
        float(aperture_match.group(1)) if aperture_match is not None else None,
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
    normalized = tuple(
        str(label).strip().upper() for label in labels if str(label).strip()
    )
    if any(len(label) > 1 for label in normalized):
        return ",".join(normalized)
    return "".join(normalized)


def _split_axis_order(order: str) -> tuple[str, ...]:
    text = str(order or "")
    if "," in text:
        return tuple(part.strip().upper() for part in text.split(",") if part.strip())
    return tuple(char.upper() for char in text if char.strip())


def _axes_from_order(order: str, shape: tuple[int, ...]) -> tuple[AxisMetadata, ...]:
    labels = _split_axis_order(order)
    confidence = "explicit"
    if len(labels) != len(shape):
        labels = _split_axis_order(_fallback_axis_order(shape))
        confidence = AXIS_CONFIDENCE_INFERRED
    return tuple(
        AxisMetadata(
            _axis_name(label),
            _axis_type(label),
            confidence=confidence,
        )
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
    return ",".join(
        (*(f"D{index}" for index in range(ndim - 5)), "T", "C", "Z", "Y", "X")
    )


def _axis_name(label: str) -> str:
    label = str(label).strip()
    mapping = {
        "C": "c",
        "H": "h",
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
    if not isinstance(mapping, Mapping):
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
