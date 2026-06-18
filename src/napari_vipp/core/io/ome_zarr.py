"""OME-Zarr 0.4/0.5 image reading and local image writing."""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from ome_zarr.format import FormatV04, FormatV05
from ome_zarr.io import parse_url
from ome_zarr.reader import Label, Reader
from ome_zarr.writer import add_metadata
from ome_zarr.writer import (
    write_image as write_ome_zarr_image,
)
from ome_zarr.writer import (
    write_labels as write_ome_zarr_labels,
)

from napari_vipp.core.io.model import (
    AnalysisLabel,
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
)
from napari_vipp.core.metadata import (
    AxisMetadata,
    ChannelMetadata,
    ImageState,
    SourceMetadata,
    image_state_from_array,
)

_DEFAULT_CHANNEL_COLORS = (
    "FFFFFF",
    "00FF00",
    "FF00FF",
    "00FFFF",
    "FF0000",
    "FFFF00",
    "0000FF",
)


def inspect_ome_zarr(path: Path) -> SourceInspection:
    """Discover image and label groups in an OME-Zarr store."""
    location, nodes = _readable_nodes(path)
    series = tuple(_series_info(node, index) for index, node in enumerate(nodes))
    return SourceInspection(
        str(path),
        f"ome-zarr-{location.version}",
        series,
        original_metadata=location.root_attrs,
    )


def read_ome_zarr(path: Path, series_index: int = 0) -> ImageDataset:
    """Read one OME-Zarr image group as a lazy Dask-backed dataset."""
    location, nodes = _readable_nodes(path)
    inspection = SourceInspection(
        str(path),
        f"ome-zarr-{location.version}",
        tuple(_series_info(node, index) for index, node in enumerate(nodes)),
        original_metadata=location.root_attrs,
    )
    selected = _selected_series(inspection, series_index)
    node = nodes[selected.index]
    data = node.data[0]
    axes = _node_axes(node)
    channels = _node_channels(node, location.root_attrs)
    state = image_state_from_array(
        data,
        source_name=selected.name,
        axes=axes,
        metadata_source=f"OME-Zarr {location.version} metadata",
        channels=channels,
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
    labels = tuple(
        item.name for item in inspection.series if item.kind == "labels"
    )
    return ImageDataset(
        data,
        state,
        inspection,
        selected,
        original_metadata=location.root_attrs,
        multiscale_levels=tuple(node.data),
        associated_labels=labels,
        provenance={"reader": "napari-vipp", "source_uri": str(path)},
    )


def write_ome_zarr(
    data,
    path: Path,
    *,
    version: str = "0.4",
    image_state: ImageState | dict[str, Any] | None = None,
) -> Path:
    """Write one local OME-Zarr image using version 0.4 or 0.5."""
    state = _coerce_state(image_state)
    arr, axes = _canonical_payload(data, state)
    fmt = FormatV04() if version == "0.4" else FormatV05()
    axis_records = [
        {
            key: value
            for key, value in {
                "name": axis.name,
                "type": axis.type,
                "unit": _ngff_unit(axis.unit),
            }.items()
            if value
        }
        for axis in axes
    ]
    scale = {axis.name: axis.scale for axis in axes}
    units = {
        axis.name: unit
        for axis in axes
        if (unit := _ngff_unit(axis.unit)) is not None
    }
    omero_metadata = _omero_metadata(state, axes, arr.shape, np.dtype(arr.dtype))
    write_ome_zarr_image(
        arr,
        str(path),
        fmt=fmt,
        axes=axis_records,
        axes_units=units or None,
        scale=scale,
        scale_factors=(),
        name=state.source_name if state and state.source_name else path.stem,
        omero=omero_metadata,
    )
    add_metadata(str(path), {"vipp": _vipp_metadata(state)}, fmt=fmt)
    return path


def write_ome_zarr_analysis_dataset(
    image_data,
    path: Path,
    *,
    labels: tuple[AnalysisLabel, ...],
    version: str = "0.4",
    image_state: ImageState | dict[str, Any] | None = None,
) -> Path:
    """Write a reference image plus label outputs as one OME-Zarr store."""
    if not labels:
        raise ValueError("At least one label image is required.")
    state = _coerce_state(image_state)
    write_ome_zarr(image_data, path, version=version, image_state=state)

    fmt = _format(version)
    image_arr, image_axes = _canonical_payload(image_data, state)
    used_names: set[str] = set()
    for label in labels:
        label_state = _coerce_state(label.image_state)
        if label_state is not None and label_state.kind != "label image":
            raise ValueError(f"{label.name!r} is not a label image output.")
        label_arr, label_axes = _canonical_payload(label.data, label_state)
        _validate_label_matches_reference(
            label.name,
            label_arr.shape,
            label_axes,
            image_arr.shape,
            image_axes,
        )
        label_name = _unique_name(_safe_label_name(label.name), used_names)
        axis_records = _axis_records(label_axes)
        units = _axis_units(label_axes)
        write_ome_zarr_labels(
            label_arr,
            str(path),
            name=label_name,
            fmt=fmt,
            axes=axis_records,
            axes_units=units or None,
            scale=_axis_scale(label_axes),
            scale_factors=(),
            scaler=None,
            label_metadata={
                "source": {"image": "../../"},
                "vipp": {
                    "software": "napari-vipp",
                    "source_node_id": label.source_node_id,
                    "label_name": label.name,
                    "history": (
                        list(label_state.history) if label_state is not None else []
                    ),
                    "source": (
                        label_state.source.to_dict()
                        if label_state is not None
                        else {}
                    ),
                },
            },
        )
    return path


def _readable_nodes(path: Path):
    location = parse_url(path, fmt=_detect_format(path))
    if location is None:
        raise FileNotFoundError(f"OME-Zarr source not found: {path}")
    nodes = tuple(node for node in Reader(location)() if node.data)
    if not nodes:
        raise ValueError(f"No image groups found in OME-Zarr source: {path}")
    return location, nodes


def _series_info(node, index: int) -> ImageSeriesInfo:
    data = node.data[0]
    axes = "".join(
        str(axis.get("name", "?")).upper()
        for axis in node.metadata.get("axes", ())
    )
    is_label = node.first(Label) is not None
    name = str(node.metadata.get("name") or node.zarr.basename())
    return ImageSeriesInfo(
        index=index,
        key=node.zarr.path,
        name=name,
        shape=tuple(int(size) for size in data.shape),
        dtype=np.dtype(data.dtype).name,
        axes=axes,
        kind="labels" if is_label else "image",
    )


def _node_axes(node) -> tuple[AxisMetadata, ...]:
    raw_axes = node.metadata.get("axes", ())
    scales = [1.0] * len(raw_axes)
    translations = [0.0] * len(raw_axes)
    transforms = node.metadata.get("coordinateTransformations", ())
    if transforms:
        for transform in transforms[0] or ():
            if transform.get("type") == "scale":
                scales = [float(value) for value in transform.get("scale", scales)]
            elif transform.get("type") == "translation":
                translations = [
                    float(value) for value in transform.get("translation", translations)
                ]
    return tuple(
        AxisMetadata(
            name=str(axis.get("name", f"d{index}")),
            type=str(axis.get("type", "unknown")),
            unit=str(axis["unit"]) if axis.get("unit") else None,
            scale=scales[index] if index < len(scales) else 1.0,
            translation=translations[index] if index < len(translations) else 0.0,
        )
        for index, axis in enumerate(raw_axes)
    )


def _node_channels(node, root_attrs: Any) -> tuple[ChannelMetadata, ...]:
    names = node.metadata.get("channel_names", ())
    if names:
        return tuple(ChannelMetadata(name=str(name)) for name in names)

    root_metadata = _normalised_root_metadata(root_attrs)
    omero = root_metadata.get("omero", {})
    channel_records = omero.get("channels", ()) if isinstance(omero, dict) else ()
    return tuple(
        ChannelMetadata(
            name=str(channel.get("label", "")),
            color=_parse_channel_color(channel.get("color")),
        )
        for channel in channel_records
        if isinstance(channel, dict)
    )


def _normalised_root_metadata(root_attrs: Any) -> dict[str, Any]:
    if not isinstance(root_attrs, dict):
        return {}
    if isinstance(root_attrs.get("ome"), dict):
        return root_attrs["ome"]
    return root_attrs


def _canonical_payload(
    data,
    state: ImageState | None,
) -> tuple[Any, tuple[AxisMetadata, ...]]:
    ndim = len(data.shape)
    if ndim < 2 or ndim > 5:
        raise ValueError("OME-Zarr image writing supports 2D through 5D arrays.")
    if state is not None and len(state.axes) == ndim:
        axes = tuple(
            replace(axis, name="c", type="channel")
            if axis.name in {"rgb", "rgba"}
            else axis
            for axis in state.axes
        )
    else:
        inferred = image_state_from_array(data)
        if inferred is None:
            raise ValueError("Could not infer OME-Zarr axes.")
        axes = inferred.axes

    desired = ("t", "c", "z", "y", "x")
    order = [
        index
        for name in desired
        for index, axis in enumerate(axes)
        if axis.name == name
    ]
    if len(order) != ndim:
        raise ValueError(
            "OME-Zarr writing requires semantic T/C/Z/Y/X axes with no duplicates."
        )
    if order != list(range(ndim)):
        data = data.transpose(order)
        axes = tuple(axes[index] for index in order)
    return data, axes


def _format(version: str):
    return FormatV05() if version == "0.5" else FormatV04()


def _detect_format(path: Path):
    try:
        attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    except Exception:
        return FormatV05()
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        return _format(str(ome.get("version", "0.5")))
    multiscales = attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        return _format(str(multiscales[0].get("version", "0.4")))
    return FormatV05()


def _axis_records(axes: tuple[AxisMetadata, ...]) -> list[dict[str, str]]:
    return [
        {
            key: value
            for key, value in {
                "name": axis.name,
                "type": axis.type,
                "unit": _ngff_unit(axis.unit),
            }.items()
            if value
        }
        for axis in axes
    ]


def _axis_units(axes: tuple[AxisMetadata, ...]) -> dict[str, str]:
    return {
        axis.name: unit
        for axis in axes
        if (unit := _ngff_unit(axis.unit)) is not None
    }


def _axis_scale(axes: tuple[AxisMetadata, ...]) -> dict[str, float]:
    return {axis.name: axis.scale for axis in axes}


def _validate_label_matches_reference(
    name: str,
    label_shape: tuple[int, ...],
    label_axes: tuple[AxisMetadata, ...],
    image_shape: tuple[int, ...],
    image_axes: tuple[AxisMetadata, ...],
) -> None:
    reference = {
        axis.name: int(size)
        for axis, size in zip(image_axes, image_shape, strict=False)
        if axis.name != "c"
    }
    for axis, size in zip(label_axes, label_shape, strict=False):
        if axis.name == "c":
            raise ValueError(f"Label image {name!r} must not contain a channel axis.")
        if axis.name in reference and int(size) != reference[axis.name]:
            raise ValueError(
                f"Label image {name!r} axis {axis.name!r} has size {size}, "
                f"but the reference image has size {reference[axis.name]}."
            )


def _omero_metadata(
    state: ImageState | None,
    axes: tuple[AxisMetadata, ...],
    shape: tuple[int, ...],
    dtype: np.dtype,
) -> dict[str, Any]:
    channel_count = 1
    for index, axis in enumerate(axes):
        if axis.name == "c":
            channel_count = int(shape[index])
            break
    channels = list(state.channels[:channel_count]) if state else []
    while len(channels) < channel_count:
        channels.append(ChannelMetadata(name=f"Channel {len(channels) + 1}"))
    window = _default_channel_window(dtype)
    return {
        "channels": [
            {
                "label": channel.name or f"Channel {index + 1}",
                "color": (
                    _channel_color(channel.color)
                    or _DEFAULT_CHANNEL_COLORS[index % len(_DEFAULT_CHANNEL_COLORS)]
                ),
                "active": True,
                "window": dict(window),
            }
            for index, channel in enumerate(channels)
        ],
        "rdefs": {"model": "color"},
    }


def _vipp_metadata(state: ImageState | None) -> dict[str, Any]:
    if state is None:
        return {"software": "napari-vipp"}
    return {
        "software": "napari-vipp",
        "history": list(state.history),
        "source": state.source.to_dict(),
        "acquisition": state.acquisition.to_dict(),
        "metadata_source": state.metadata_source,
    }


def _channel_color(color: int | None) -> str | None:
    if color is None:
        return None
    return f"{int(color) & 0xFFFFFF:06X}"


def _default_channel_window(dtype: np.dtype) -> dict[str, float]:
    if np.issubdtype(dtype, np.bool_):
        maximum: float = 1
    elif np.issubdtype(dtype, np.integer):
        maximum = float(np.iinfo(dtype).max)
    else:
        maximum = 1.0
    return {"min": 0, "start": 0, "max": maximum, "end": maximum}


def _parse_channel_color(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip().removeprefix("#")
    try:
        return int(text, 16) if len(text) == 6 else None
    except ValueError:
        return None


def _ngff_unit(unit: str | None) -> str | None:
    if not unit:
        return None
    return {
        "µm": "micrometer",
        "μm": "micrometer",
        "um": "micrometer",
        "nm": "nanometer",
        "s": "second",
        "ms": "millisecond",
    }.get(unit, unit)


def _safe_label_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return name.strip("._-") or "labels"


def _unique_name(name: str, used: set[str]) -> str:
    candidate = name
    suffix = 2
    while candidate in used:
        candidate = f"{name}_{suffix}"
        suffix += 1
    used.add(candidate)
    return candidate


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


def _coerce_state(value: ImageState | dict[str, Any] | None) -> ImageState | None:
    if isinstance(value, ImageState):
        return value
    if isinstance(value, dict):
        return ImageState.from_dict(value)
    return None


def provenance_json(dataset: ImageDataset) -> str:
    """Return a stable JSON representation useful to batch manifests."""
    return json.dumps(dataset.provenance, sort_keys=True)
