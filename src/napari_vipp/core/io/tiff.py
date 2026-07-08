"""OME-TIFF, ImageJ TIFF, and conventional TIFF support."""

from __future__ import annotations

import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np
from ome_types import from_xml
from tifffile import TiffFile, imwrite

from napari_vipp.core.io.model import (
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
)
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    ChannelMetadata,
    ImageState,
    SourceMetadata,
    image_state_from_array,
)

TIFF_FORMATS = {"ome-tiff", "imagej-tiff", "tiff"}
IMAGEJ_AXES = "TZCYXS"


def inspect_tiff(path: Path) -> SourceInspection:
    """Inspect TIFF series and metadata without loading all pixel data."""
    with TiffFile(path) as tif:
        source_format = _tiff_format(tif)
        ome_xml = tif.ome_metadata if source_format == "ome-tiff" else None
        ome = _parse_ome(ome_xml)
        series = tuple(
            _series_info(item, index, ome)
            for index, item in enumerate(tif.series)
        )
        original = (
            ome_xml
            if ome_xml
            else dict(tif.imagej_metadata or {})
            if source_format == "imagej-tiff"
            else None
        )
    return SourceInspection(str(path), source_format, series, original)


def read_tiff(path: Path, series_index: int = 0) -> ImageDataset:
    """Read one TIFF series and normalize its scientific metadata."""
    inspection = inspect_tiff(path)
    selected = _selected_series(inspection, series_index)
    with TiffFile(path) as tif:
        tif_series = tif.series[selected.index]
        data = tif_series.asarray()
        ome_xml = tif.ome_metadata if inspection.format == "ome-tiff" else None
        ome = _parse_ome(ome_xml)
        imagej = dict(tif.imagej_metadata or {})
        resolution = _tiff_resolution(tif, imagej)
        axes = _axis_metadata(
            selected.axes,
            data.shape,
            ome=ome,
            series_index=selected.index,
            imagej=imagej,
            resolution=resolution,
        )
        channels = _channel_metadata(ome, selected.index)
        acquisition = _acquisition_metadata(ome, selected.index)
        source = SourceMetadata(
            uri=str(path),
            format=inspection.format,
            series_index=selected.index,
            series_name=selected.name,
            source_uuid=str(getattr(ome, "uuid", "") or ""),
        )
        state = image_state_from_array(
            data,
            source_name=selected.name or path.name,
            axes=axes,
            metadata_source=_metadata_source_label(inspection.format),
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
        original_metadata=inspection.original_metadata,
        provenance={"reader": "napari-vipp", "source_uri": str(path)},
    )


def write_tiff(
    data,
    path: Path,
    *,
    format: str,
    image_state: ImageState | dict[str, Any] | None = None,
) -> Path:
    """Write one explicit TIFF family format."""
    selected = str(format).lower()
    if selected not in TIFF_FORMATS:
        raise ValueError(f"Unsupported TIFF format: {format}")
    state = _coerce_state(image_state)
    if selected == "ome-tiff":
        _write_ome_tiff(data, path, state)
    elif selected == "imagej-tiff":
        _write_imagej_tiff(data, path, state)
    else:
        _write_conventional_tiff(data, path, state)
    return path


def _write_ome_tiff(data, path: Path, state: ImageState | None) -> None:
    arr = _tiff_writable_array(np.asarray(data), binary_values=False)
    axes = _axes_for_array(arr, state, ome=True)
    metadata = _ome_write_metadata(state, axes, path)
    photometric = (
        "rgb" if axes.endswith("S") and arr.shape[-1] in (3, 4) else "minisblack"
    )
    imwrite(
        path,
        arr,
        ome=True,
        bigtiff=arr.nbytes >= 4_000_000_000,
        metadata=metadata,
        photometric=photometric,
    )


def _write_imagej_tiff(data, path: Path, state: ImageState | None) -> None:
    arr = np.asarray(data)
    if state is not None and state.kind == "label image" and arr.dtype.itemsize > 2:
        raise ValueError(
            "ImageJ TIFF cannot safely represent 32-bit label IDs; "
            "use TIFF or OME-TIFF."
        )
    writable, axes = _imagej_payload(arr, state)
    metadata: dict[str, Any] = {"axes": axes}
    z_axis = _axis_by_name(state, "z")
    t_axis = _axis_by_name(state, "t")
    if z_axis is not None:
        metadata["spacing"] = z_axis.scale
        if z_axis.unit:
            metadata["unit"] = _imagej_unit(z_axis.unit)
    if t_axis is not None:
        metadata["finterval"] = t_axis.scale
    resolution = _xy_resolution(state)
    imwrite(
        path,
        writable,
        imagej=True,
        metadata=metadata,
        resolution=resolution,
    )


def _write_conventional_tiff(
    data,
    path: Path,
    state: ImageState | None,
) -> None:
    arr = _tiff_writable_array(np.asarray(data), binary_values=False)
    axes = _axes_for_array(arr, state, ome=False)
    imwrite(
        path,
        arr,
        metadata={"axes": axes},
        resolution=_xy_resolution(state),
        photometric=(
            "rgb" if axes.endswith("S") and arr.shape[-1] in (3, 4) else "minisblack"
        ),
    )


def _ome_write_metadata(
    state: ImageState | None,
    axes: str,
    path: Path,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "axes": axes,
        "Name": state.source_name if state and state.source_name else path.stem,
        "Creator": _creator(),
    }
    if state is None:
        return metadata

    for name, key, unit_key in (
        ("x", "PhysicalSizeX", "PhysicalSizeXUnit"),
        ("y", "PhysicalSizeY", "PhysicalSizeYUnit"),
        ("z", "PhysicalSizeZ", "PhysicalSizeZUnit"),
    ):
        axis = _axis_by_name(state, name)
        if axis is not None:
            metadata[key] = axis.scale
            if axis.unit:
                metadata[unit_key] = axis.unit
    time_axis = _axis_by_name(state, "t")
    if time_axis is not None:
        metadata["TimeIncrement"] = time_axis.scale
        if time_axis.unit:
            metadata["TimeIncrementUnit"] = time_axis.unit

    if state.acquisition.description:
        metadata["Description"] = state.acquisition.description
    if state.acquisition.acquisition_date:
        metadata["AcquisitionDate"] = state.acquisition.acquisition_date

    channel_count = _axis_size(state, "c")
    if channel_count is None:
        channel_count = 1
    channels = list(state.channels[:channel_count])
    while len(channels) < channel_count:
        channels.append(ChannelMetadata(name=f"Channel {len(channels) + 1}"))
    channel_metadata = _ome_channel_write_metadata(channels)
    if channel_metadata:
        metadata["Channel"] = channel_metadata

    provenance = {
        "software": _creator(),
        "history": json.dumps(list(state.history)),
        "source_uri": state.source.uri,
        "source_format": state.source.format,
        "source_series": state.source.series_name,
    }
    metadata["MapAnnotation"] = {
        "Namespace": "https://github.com/rensutheart/napari-vipp/provenance/1",
        "Value": {key: value for key, value in provenance.items() if value},
    }
    return metadata


def _ome_channel_write_metadata(
    channels: list[ChannelMetadata],
) -> list[dict[str, Any]]:
    field_map = {
        "Name": "name",
        "Color": "color",
        "Fluor": "fluor",
        "ExcitationWavelength": "excitation_wavelength",
        "ExcitationWavelengthUnit": "excitation_wavelength_unit",
        "EmissionWavelength": "emission_wavelength",
        "EmissionWavelengthUnit": "emission_wavelength_unit",
    }
    return [
        {
            ome_name: value
            for ome_name, field_name in field_map.items()
            if (value := getattr(channel, field_name)) not in {"", None}
        }
        for channel in channels
    ]


def _series_info(series, index: int, ome) -> ImageSeriesInfo:
    name = str(getattr(series, "name", "") or "")
    if ome is not None and index < len(ome.images):
        name = str(ome.images[index].name or name)
    return ImageSeriesInfo(
        index=index,
        key=str(index),
        name=name or f"Series {index + 1}",
        shape=tuple(int(size) for size in series.shape),
        dtype=np.dtype(series.dtype).name,
        axes=str(series.axes),
    )


def _axis_metadata(
    axes: str,
    shape: tuple[int, ...],
    *,
    ome,
    series_index: int,
    imagej: dict[str, Any],
    resolution: tuple[float, float, str | None] | None,
) -> tuple[AxisMetadata, ...]:
    if len(axes) != len(shape):
        state = image_state_from_array(np.empty((1,) * len(shape)))
        return state.axes if state is not None else ()

    pixels = None
    if ome is not None and series_index < len(ome.images):
        pixels = ome.images[series_index].pixels
    axis_records: list[AxisMetadata] = []
    for label in axes:
        name, axis_type = _axis_identity(label)
        scale = 1.0
        unit = None
        if pixels is not None:
            scale, unit = _ome_axis_scale(pixels, label)
        elif imagej:
            scale, unit = _imagej_axis_scale(imagej, label, resolution)
        elif resolution is not None and label in {"X", "Y"}:
            scale = resolution[0] if label == "X" else resolution[1]
            unit = resolution[2]
        axis_records.append(
            AxisMetadata(name=name, type=axis_type, unit=unit, scale=scale)
        )
    return tuple(axis_records)


def _channel_metadata(ome, series_index: int) -> tuple[ChannelMetadata, ...]:
    if ome is None or series_index >= len(ome.images):
        return ()
    return tuple(
        ChannelMetadata(
            name=str(channel.name or ""),
            color=_optional_int(channel.color),
            fluor=str(channel.fluor or ""),
            excitation_wavelength=_optional_float(channel.excitation_wavelength),
            excitation_wavelength_unit=_model_value(
                channel.excitation_wavelength_unit
            ),
            emission_wavelength=_optional_float(channel.emission_wavelength),
            emission_wavelength_unit=_model_value(channel.emission_wavelength_unit),
        )
        for channel in ome.images[series_index].pixels.channels
    )


def _acquisition_metadata(ome, series_index: int) -> AcquisitionMetadata:
    if ome is None or series_index >= len(ome.images):
        return AcquisitionMetadata()
    image = ome.images[series_index]
    instrument = ""
    objective = ""
    detector = ""
    objective_record = None
    if image.instrument_ref is not None:
        instrument = str(image.instrument_ref.id)
    if image.objective_settings is not None:
        objective = str(image.objective_settings.id)
        objective_record = _objective_record(
            ome,
            instrument,
            image.objective_settings.id,
        )
    pixels = image.pixels
    for channel in pixels.channels:
        if channel.detector_settings is not None:
            detector = str(channel.detector_settings.id)
            break
    return AcquisitionMetadata(
        description=str(image.description or ""),
        acquisition_date=str(image.acquisition_date or ""),
        objective=objective,
        instrument=instrument,
        detector=detector,
        objective_na=_optional_float(getattr(objective_record, "lens_na", None)),
        objective_magnification=_optional_float(
            getattr(objective_record, "nominal_magnification", None),
        ),
        objective_immersion=_model_value(getattr(objective_record, "immersion", "")),
        refractive_index=_optional_float(
            getattr(image.objective_settings, "refractive_index", None)
            if image.objective_settings is not None
            else None
        ),
    )


def _objective_record(ome, instrument_id: str, objective_id: str):
    if not objective_id:
        return None
    for instrument in getattr(ome, "instruments", ()) or ():
        if instrument_id and str(getattr(instrument, "id", "")) != instrument_id:
            continue
        for objective in getattr(instrument, "objectives", ()) or ():
            if str(getattr(objective, "id", "")) == str(objective_id):
                return objective
    return None


def _ome_axis_scale(pixels, label: str) -> tuple[float, str | None]:
    attributes = {
        "X": ("physical_size_x", "physical_size_x_unit"),
        "Y": ("physical_size_y", "physical_size_y_unit"),
        "Z": ("physical_size_z", "physical_size_z_unit"),
        "T": ("time_increment", "time_increment_unit"),
    }
    names = attributes.get(label)
    if names is None:
        return 1.0, None
    value = getattr(pixels, names[0], None)
    unit = getattr(pixels, names[1], None)
    return _optional_float(value) or 1.0, _model_value(unit) or None


def _imagej_axis_scale(
    metadata: dict[str, Any],
    label: str,
    resolution: tuple[float, float, str | None] | None,
) -> tuple[float, str | None]:
    if label == "Z":
        unit = str(metadata.get("unit", "")) or None
        return float(metadata.get("spacing", 1.0)), unit
    if label == "T":
        return float(metadata.get("finterval", 1.0)), None
    if label in {"X", "Y"}:
        unit = str(metadata.get("unit", "")) or None
        if resolution is not None:
            scale = resolution[0] if label == "X" else resolution[1]
            return scale, unit or resolution[2]
        return 1.0, unit
    return 1.0, None


def _tiff_format(tif: TiffFile) -> str:
    if tif.ome_metadata:
        return "ome-tiff"
    if tif.is_imagej:
        return "imagej-tiff"
    return "tiff"


def _metadata_source_label(format: str) -> str:
    return {
        "ome-tiff": "OME-XML",
        "imagej-tiff": "ImageJ TIFF metadata",
        "tiff": "TIFF series metadata",
    }[format]


def _parse_ome(xml: str | None):
    if not xml:
        return None
    return from_xml(xml, validate=False, warn_on_schema_update=False)


def _selected_series(
    inspection: SourceInspection,
    series_index: int,
) -> ImageSeriesInfo:
    index = int(series_index)
    if index < 0 or index >= len(inspection.series):
        raise IndexError(
            f"Series index {index} is outside 0..{len(inspection.series) - 1}"
        )
    return inspection.series[index]


def _axis_identity(label: str) -> tuple[str, str]:
    return {
        "T": ("t", "time"),
        "C": ("c", "channel"),
        "S": ("rgb", "channel"),
        "Z": ("z", "space"),
        "Y": ("y", "space"),
        "X": ("x", "space"),
    }.get(label, (label.lower(), "unknown"))


def _axes_for_array(
    arr: np.ndarray,
    state: ImageState | None,
    *,
    ome: bool,
) -> str:
    labels = _axis_labels_from_state(state, arr.ndim)
    allowed = "TZCYXS" if ome else "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if labels is None or any(label not in allowed for label in labels):
        labels = _fallback_axes(arr)
    return labels


def _imagej_payload(
    arr: np.ndarray,
    state: ImageState | None,
) -> tuple[np.ndarray, str]:
    writable = _tiff_writable_array(arr, binary_values=True)
    axes = _axis_labels_from_state(state, writable.ndim) or _fallback_axes(writable)
    if len(axes) != writable.ndim or any(label not in IMAGEJ_AXES for label in axes):
        axes = _fallback_axes(writable)
    order = [axes.index(axis) for axis in IMAGEJ_AXES if axis in axes]
    if order != list(range(len(axes))):
        writable = np.transpose(writable, order)
        axes = "".join(axes[index] for index in order)
    return np.ascontiguousarray(writable), axes


def _axis_labels_from_state(
    state: ImageState | None,
    ndim: int,
) -> str | None:
    if state is None or len(state.axes) != ndim:
        return None
    labels: list[str] = []
    for axis in state.axes:
        if axis.name == "t" or axis.type == "time":
            labels.append("T")
        elif axis.name == "z":
            labels.append("Z")
        elif axis.name in {"c", "channel"}:
            labels.append("C")
        elif axis.name == "y":
            labels.append("Y")
        elif axis.name == "x":
            labels.append("X")
        elif axis.name in {"rgb", "rgba"}:
            labels.append("S")
        elif axis.type == "channel":
            labels.append("C")
        else:
            return None
    return "".join(labels)


def _fallback_axes(arr: np.ndarray) -> str:
    if arr.ndim <= 2:
        return "YX"
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return "YXS"
    return {
        3: "ZYX",
        4: "ZCYX",
        5: "TZCYX",
        6: "TZCYXS",
    }.get(arr.ndim, "YX")


def _tiff_writable_array(
    arr: np.ndarray,
    *,
    binary_values: bool,
) -> np.ndarray:
    if arr.dtype == bool and binary_values:
        arr = arr.astype(np.uint8) * np.uint8(255)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        return arr.reshape(1, arr.shape[0])
    return np.ascontiguousarray(arr)


def _axis_by_name(state: ImageState | None, name: str) -> AxisMetadata | None:
    if state is None:
        return None
    return next((axis for axis in state.axes if axis.name == name), None)


def _axis_size(state: ImageState, name: str) -> int | None:
    for index, axis in enumerate(state.axes):
        if axis.name == name:
            return state.shape[index]
    return None


def _xy_resolution(state: ImageState | None) -> tuple[float, float] | None:
    x_axis = _axis_by_name(state, "x")
    y_axis = _axis_by_name(state, "y")
    if x_axis is None or y_axis is None or x_axis.scale <= 0 or y_axis.scale <= 0:
        return None
    return 1.0 / x_axis.scale, 1.0 / y_axis.scale


def _tiff_resolution(
    tif: TiffFile,
    imagej: dict[str, Any],
) -> tuple[float, float, str | None] | None:
    if not tif.pages:
        return None
    tags = tif.pages[0].tags
    x_tag = tags.get("XResolution")
    y_tag = tags.get("YResolution")
    if x_tag is None or y_tag is None:
        return None
    x_resolution = _rational_value(x_tag.value)
    y_resolution = _rational_value(y_tag.value)
    if x_resolution <= 0 or y_resolution <= 0:
        return None
    unit = str(imagej.get("unit", "")) or None
    if unit is None:
        resolution_unit = tags.get("ResolutionUnit")
        code = int(resolution_unit.value) if resolution_unit is not None else 1
        unit = {2: "inch", 3: "cm"}.get(code)
    return 1.0 / x_resolution, 1.0 / y_resolution, unit


def _rational_value(value) -> float:
    if isinstance(value, tuple) and len(value) == 2:
        return float(value[0]) / float(value[1])
    return float(value)


def _coerce_state(value: ImageState | dict[str, Any] | None) -> ImageState | None:
    if isinstance(value, ImageState):
        return value
    if isinstance(value, dict):
        return ImageState.from_dict(value)
    return None


def _creator() -> str:
    try:
        package_version = version("napari-vipp")
    except PackageNotFoundError:
        package_version = "development"
    return f"napari-vipp {package_version}"


def _model_value(value) -> str:
    if value is None:
        return ""
    return str(getattr(value, "value", value))


def _imagej_unit(value: str) -> str:
    return {
        "µm": "micron",
        "μm": "micron",
        "um": "micron",
    }.get(value, value.encode("ascii", errors="ignore").decode() or "pixel")


def _optional_float(value) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
