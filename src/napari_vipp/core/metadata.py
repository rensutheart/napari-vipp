"""OME-NGFF-inspired image state metadata for graph execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from napari_vipp.core.channel_colors import channel_color_int, channel_color_names
from napari_vipp.core.tables import (
    TableState,
    is_table_data,
    table_state_from_data,
)

RGB_CHANNELS = (3, 4)
MAX_METADATA_VALUES = 500_000
CHANNEL_COLLAPSE_OPERATIONS = {
    "otsu_threshold",
    "triangle_threshold",
    "li_threshold",
    "yen_threshold",
    "isodata_threshold",
    "minimum_threshold",
    "binary_threshold",
    "hysteresis_threshold",
    "canny_edges",
    "adaptive_mean_threshold",
    "adaptive_gaussian_threshold",
    "sauvola_threshold",
    "niblack_threshold",
}
LABEL_OPERATIONS = {
    "label_connected_components",
    "filter_labels_by_volume",
    "relabel_sequential",
}
KIND_PRESERVING_OPERATIONS = {
    "clear_border_objects",
}


@dataclass(frozen=True)
class AxisMetadata:
    """Single array axis with OME-NGFF-like semantics."""

    name: str
    type: str
    unit: str | None = None
    scale: float = 1.0
    translation: float = 0.0
    source_axis: int | None = None

    @property
    def short_label(self) -> str:
        return self.name.upper() if len(self.name) == 1 else self.name

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "name": self.name,
            "type": self.type,
            "scale": self.scale,
            "translation": self.translation,
        }
        if self.unit:
            data["unit"] = self.unit
        if self.source_axis is not None:
            data["source_axis"] = self.source_axis
        return data

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AxisMetadata:
        name = str(data.get("name", "d"))
        return cls(
            name=name,
            type=str(data.get("type", _axis_type_for_name(name))),
            unit=str(data["unit"]) if data.get("unit") else None,
            scale=_safe_float(data.get("scale"), 1.0),
            translation=_safe_float(data.get("translation"), 0.0),
            source_axis=_optional_int(data.get("source_axis")),
        )


@dataclass(frozen=True)
class ChannelMetadata:
    """Normalized metadata for one acquisition channel."""

    name: str = ""
    color: int | None = None
    fluor: str = ""
    excitation_wavelength: float | None = None
    excitation_wavelength_unit: str = ""
    emission_wavelength: float | None = None
    emission_wavelength_unit: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "name": self.name,
                "color": self.color,
                "fluor": self.fluor,
                "excitation_wavelength": self.excitation_wavelength,
                "excitation_wavelength_unit": self.excitation_wavelength_unit,
                "emission_wavelength": self.emission_wavelength,
                "emission_wavelength_unit": self.emission_wavelength_unit,
            }.items()
            if value not in {"", None}
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ChannelMetadata:
        return cls(
            name=str(data.get("name", "")),
            color=_optional_int(data.get("color")),
            fluor=str(data.get("fluor", "")),
            excitation_wavelength=_optional_float(
                data.get("excitation_wavelength")
            ),
            excitation_wavelength_unit=str(
                data.get("excitation_wavelength_unit", "")
            ),
            emission_wavelength=_optional_float(data.get("emission_wavelength")),
            emission_wavelength_unit=str(data.get("emission_wavelength_unit", "")),
        )


@dataclass(frozen=True)
class AcquisitionMetadata:
    """Acquisition facts that remain meaningful after image processing."""

    description: str = ""
    acquisition_date: str = ""
    objective: str = ""
    instrument: str = ""
    detector: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "description": self.description,
                "acquisition_date": self.acquisition_date,
                "objective": self.objective,
                "instrument": self.instrument,
                "detector": self.detector,
            }.items()
            if value
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> AcquisitionMetadata:
        return cls(
            description=str(data.get("description", "")),
            acquisition_date=str(data.get("acquisition_date", "")),
            objective=str(data.get("objective", "")),
            instrument=str(data.get("instrument", "")),
            detector=str(data.get("detector", "")),
        )


@dataclass(frozen=True)
class SourceMetadata:
    """Stable identity and format information for the imported source item."""

    uri: str = ""
    format: str = ""
    series_index: int = 0
    series_name: str = ""
    source_uuid: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            key: value
            for key, value in {
                "uri": self.uri,
                "format": self.format,
                "series_index": self.series_index,
                "series_name": self.series_name,
                "source_uuid": self.source_uuid,
            }.items()
            if value not in {"", None}
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SourceMetadata:
        return cls(
            uri=str(data.get("uri", "")),
            format=str(data.get("format", "")),
            series_index=int(data.get("series_index", 0)),
            series_name=str(data.get("series_name", "")),
            source_uuid=str(data.get("source_uuid", "")),
        )


@dataclass(frozen=True)
class ImageState:
    """Array metadata carried alongside every pipeline node output."""

    shape: tuple[int, ...]
    dtype: str
    kind: str
    axes: tuple[AxisMetadata, ...]
    bit_depth: str
    value_range: str
    value_pattern: str
    memory: str
    metadata_source: str
    source_name: str = ""
    history: tuple[str, ...] = ()
    channels: tuple[ChannelMetadata, ...] = ()
    acquisition: AcquisitionMetadata = AcquisitionMetadata()
    source: SourceMetadata = SourceMetadata()

    @property
    def axis_order(self) -> str:
        if not self.axes:
            return "scalar"
        labels = [axis.short_label for axis in self.axes]
        if all(len(label) == 1 for label in labels):
            return "".join(labels)
        return ",".join(labels)

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "kind": self.kind,
            "axes": [axis.to_dict() for axis in self.axes],
            "bit_depth": self.bit_depth,
            "value_range": self.value_range,
            "value_pattern": self.value_pattern,
            "memory": self.memory,
            "metadata_source": self.metadata_source,
            "source_name": self.source_name,
            "history": list(self.history),
            "channels": [channel.to_dict() for channel in self.channels],
            "acquisition": self.acquisition.to_dict(),
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> ImageState | None:
        axes_data = data.get("axes")
        if not isinstance(axes_data, list):
            return None
        try:
            axes = tuple(
                AxisMetadata.from_dict(axis)
                for axis in axes_data
                if isinstance(axis, dict)
            )
            channels_data = data.get("channels", ())
            channels = tuple(
                ChannelMetadata.from_dict(channel)
                for channel in channels_data
                if isinstance(channel, dict)
            )
            acquisition_data = data.get("acquisition")
            acquisition = (
                AcquisitionMetadata.from_dict(acquisition_data)
                if isinstance(acquisition_data, dict)
                else AcquisitionMetadata()
            )
            source_data = data.get("source")
            source = (
                SourceMetadata.from_dict(source_data)
                if isinstance(source_data, dict)
                else SourceMetadata()
            )
            return cls(
                shape=tuple(int(value) for value in data.get("shape", ())),
                dtype=str(data.get("dtype", "")),
                kind=str(data.get("kind", "")),
                axes=axes,
                bit_depth=str(data.get("bit_depth", "")),
                value_range=str(data.get("value_range", "")),
                value_pattern=str(data.get("value_pattern", "")),
                memory=str(data.get("memory", "")),
                metadata_source=str(data.get("metadata_source", "VIPP carried state")),
                source_name=str(data.get("source_name", "")),
                history=tuple(str(step) for step in data.get("history", ())),
                channels=channels,
                acquisition=acquisition,
                source=source,
            )
        except Exception:
            return None


@dataclass(frozen=True)
class MetadataRow:
    label: str
    value: str


def image_state_from_array(
    data,
    *,
    layer_metadata: dict | None = None,
    source_name: str = "",
    axes: tuple[AxisMetadata, ...] | None = None,
    metadata_source: str | None = None,
    history: tuple[str, ...] = (),
    channels: tuple[ChannelMetadata, ...] | None = None,
    acquisition: AcquisitionMetadata | None = None,
    source: SourceMetadata | None = None,
) -> ImageState | None:
    """Create a carried image state from array data and optional metadata."""
    if data is None:
        return None

    assign_default_source_axes = axes is not None and all(
        axis.source_axis is None for axis in axes
    )
    lazy = _is_lazy_array(data)
    if lazy:
        shape = tuple(int(size) for size in data.shape)
        dtype = np.dtype(data.dtype)
        arr = None
    else:
        arr = np.asarray(data)
        shape = tuple(int(size) for size in arr.shape)
        dtype = arr.dtype
    carried_state = _carried_image_state(layer_metadata)
    if axes is None:
        axes, parsed_source, parsed_history = _axes_from_layer_metadata(
            layer_metadata,
            shape,
        )
        assign_default_source_axes = parsed_source != "VIPP carried state"
        if parsed_history:
            history = parsed_history + history
        metadata_source = metadata_source or parsed_source
    else:
        metadata_source = metadata_source or "VIPP transformed metadata"

    if len(axes) != len(shape):
        axes = infer_axis_metadata_from_shape(shape)
        axes = _with_default_source_axes(axes)
        metadata_source = "inferred from array shape"
    elif assign_default_source_axes:
        axes = _with_default_source_axes(axes)

    if carried_state is not None:
        channels = channels if channels is not None else carried_state.channels
        acquisition = acquisition or carried_state.acquisition
        source = source or carried_state.source

    return ImageState(
        shape=shape,
        dtype=dtype.name,
        kind=_lazy_kind_label(dtype, shape, axes) if lazy else _kind_label(arr, axes),
        axes=axes,
        bit_depth=_bit_depth_label(dtype),
        value_range="not computed (lazy)" if lazy else _value_range_label(arr),
        value_pattern="" if lazy else _value_pattern_label(arr),
        memory=_memory_label(int(np.prod(shape, dtype=np.int64)) * dtype.itemsize),
        metadata_source=metadata_source or "inferred from array shape",
        source_name=source_name,
        history=history,
        channels=channels or (),
        acquisition=acquisition or AcquisitionMetadata(),
        source=source or SourceMetadata(),
    )


def transform_image_state(
    data,
    input_state: ImageState | None,
    *,
    operation_id: str,
    operation_title: str,
    params: dict[str, Any],
) -> ImageState | None:
    """Transform carried metadata after an operation has produced output data."""
    if data is None:
        return None
    if input_state is None:
        state = image_state_from_array(
            data,
            metadata_source=f"inferred after {operation_title}",
            history=(f"{operation_title}: metadata reconstructed from output shape",),
        )
        return _with_operation_kind(state, operation_id)

    arr = np.asarray(data)
    axes = _transformed_axes(
        input_state,
        arr,
        operation_id=operation_id,
        params=params,
    )
    metadata_source = input_state.metadata_source
    if len(axes) != arr.ndim:
        axes = infer_axis_metadata(arr)
        metadata_source = f"inferred after {operation_title}"

    state = image_state_from_array(
        arr,
        axes=axes,
        metadata_source=metadata_source,
        source_name=input_state.source_name,
        history=input_state.history
        + (_operation_history(input_state, operation_id, operation_title, params),),
        channels=_transformed_channels(input_state, operation_id, params),
        acquisition=input_state.acquisition,
        source=input_state.source,
    )
    if operation_id in KIND_PRESERVING_OPERATIONS:
        state = replace(state, kind=input_state.kind)
    return _with_operation_kind(state, operation_id)


def transform_multi_input_image_state(
    data,
    input_states: list[ImageState | None],
    *,
    operation_id: str,
    operation_title: str,
    params: dict[str, Any],
) -> ImageState | None:
    """Transform metadata for operations that consume several upstream images."""
    if data is None:
        return None
    states = [state for state in input_states if state is not None]
    if not states:
        return image_state_from_array(
            data,
            metadata_source=f"inferred after {operation_title}",
            history=(f"{operation_title}: metadata reconstructed from output shape",),
        )

    arr = np.asarray(data)
    first = states[0]
    axes = _multi_input_axes(first.axes, arr, operation_id=operation_id, params=params)
    metadata_source = first.metadata_source
    if len(axes) != arr.ndim:
        axes = infer_axis_metadata(arr)
        metadata_source = f"inferred after {operation_title}"

    return image_state_from_array(
        arr,
        axes=axes,
        metadata_source=metadata_source,
        source_name=first.source_name,
        history=first.history
        + (_multi_input_history(states, operation_id, operation_title, params),),
        channels=_multi_input_channels(states, operation_id, params),
        acquisition=first.acquisition,
        source=first.source,
    )


def transform_split_output_state(
    data,
    input_state: ImageState | None,
    *,
    operation_id: str,
    operation_title: str,
    port_name: str,
    params: dict[str, Any],
) -> ImageState | None:
    """Transform metadata for a single output port of a channel-splitting node."""
    if data is None:
        return None
    label = f"{operation_title} ({port_name})"
    if input_state is None:
        return image_state_from_array(
            data,
            metadata_source=f"inferred after {label}",
            history=(f"{label}: metadata reconstructed from output shape",),
        )

    arr = np.asarray(data)
    axes = _split_output_axes(input_state.axes, arr.ndim)
    metadata_source = input_state.metadata_source
    if len(axes) != arr.ndim:
        axes = infer_axis_metadata(arr)
        metadata_source = f"inferred after {label}"

    return image_state_from_array(
        arr,
        axes=axes,
        metadata_source=metadata_source,
        source_name=input_state.source_name,
        history=input_state.history + (f"{label}: extracted {port_name}",),
        channels=_split_output_channels(input_state.channels, port_name),
        acquisition=input_state.acquisition,
        source=input_state.source,
    )


def format_compact_metadata(state_or_data) -> str:
    """Two-line metadata summary suitable for a small graph node."""
    table_state = _coerce_table_state(state_or_data)
    if table_state is not None:
        return (
            f"TABLE: {table_state.row_count} rows x "
            f"{table_state.column_count} columns\n{table_state.kind}"
        )

    state = _coerce_state(state_or_data)
    if state is None:
        return "No output"

    first = f"{state.axis_order}: {_dimensions_compact_label(state)} | {state.dtype}"
    second_parts = [state.kind, state.bit_depth]
    if "inferred" in state.metadata_source:
        second_parts.append("axes inferred")
    return first + "\n" + " | ".join(second_parts)


def metadata_table_rows(state_or_data) -> list[MetadataRow]:
    """Return display rows for the selected-node metadata table."""
    table_state = _coerce_table_state(state_or_data)
    if table_state is not None:
        rows = [
            MetadataRow("Kind", table_state.kind),
            MetadataRow("Rows", str(table_state.row_count)),
            MetadataRow("Columns", str(table_state.column_count)),
            MetadataRow("Measurement set", table_state.table_kind),
            MetadataRow("Column names", ", ".join(table_state.columns) or "none"),
            MetadataRow("Metadata source", table_state.metadata_source),
        ]
        if table_state.column_units:
            units = ", ".join(
                f"{column}: {unit}" for column, unit in table_state.column_units
            )
            rows.append(MetadataRow("Units", units))
        if table_state.source_name:
            rows.append(MetadataRow("Source", table_state.source_name))
        return rows

    state = _coerce_state(state_or_data)
    if state is None:
        return [MetadataRow("Status", "No output yet.")]

    rows = [
        MetadataRow("Kind", state.kind),
        MetadataRow("Shape", _shape_label(state.shape)),
        MetadataRow("Axes", _axes_detail_label(state)),
        MetadataRow("Dimensions", _dimensions_label(state)),
        MetadataRow("Physical scale", _scale_label(state)),
        MetadataRow("Origin", _origin_label(state)),
        MetadataRow("Channels", _axis_count_label(state, "channel")),
        MetadataRow("Timepoints", _axis_count_label(state, "time")),
        MetadataRow("Z slices", _named_axis_count_label(state, "z")),
        MetadataRow("Dtype", state.dtype),
        MetadataRow("Bit depth", state.bit_depth),
        MetadataRow("Value range", state.value_range),
    ]
    if state.value_pattern:
        rows.append(MetadataRow("Value pattern", state.value_pattern))
    rows.extend(
        [
            MetadataRow("Memory", state.memory),
            MetadataRow("Metadata source", state.metadata_source),
        ]
    )
    if state.source_name:
        rows.append(MetadataRow("Source", state.source_name))
    if state.channels:
        names = [channel.name for channel in state.channels if channel.name]
        if names:
            rows.append(MetadataRow("Channel names", ", ".join(names)))
    if state.source.format:
        rows.append(MetadataRow("Source format", state.source.format))
    if state.source.series_name:
        rows.append(MetadataRow("Source series", state.source.series_name))
    return rows


def metadata_history_items(state_or_data) -> list[str]:
    """Return operation history entries for inspector display."""
    table_state = _coerce_table_state(state_or_data)
    if table_state is not None:
        return list(table_state.history)

    state = _coerce_state(state_or_data)
    if state is None:
        return []
    return list(state.history)


def format_detailed_metadata(state_or_data) -> str:
    """Multi-line metadata summary for the selected node inspector."""
    table_state = _coerce_table_state(state_or_data)
    if table_state is not None:
        lines = [
            f"Kind: {table_state.kind}",
            f"Rows: {table_state.row_count}",
            f"Columns: {table_state.column_count}",
            f"Measurement set: {table_state.table_kind}",
            "Column names: " + (", ".join(table_state.columns) or "none"),
            f"Metadata source: {table_state.metadata_source}",
        ]
        if table_state.source_name:
            lines.append(f"Source: {table_state.source_name}")
        if table_state.history:
            lines.append("History: " + " -> ".join(table_state.history[-4:]))
        return "\n".join(lines)

    state = _coerce_state(state_or_data)
    if state is None:
        return "No output yet."

    lines = [
        f"Kind: {state.kind}",
        f"Shape: {_shape_label(state.shape)}",
        f"Axes: {_axes_detail_label(state)}",
        f"Dimensions: {_dimensions_label(state)}",
        f"Physical scale: {_scale_label(state)}",
        f"Origin: {_origin_label(state)}",
        f"Channels: {_axis_count_label(state, 'channel')}",
        f"Timepoints: {_axis_count_label(state, 'time')}",
        f"Z slices: {_named_axis_count_label(state, 'z')}",
        f"Dtype: {state.dtype}",
        f"Bit depth: {state.bit_depth}",
        f"Value range: {state.value_range}",
    ]
    if state.value_pattern:
        lines.append(f"Value pattern: {state.value_pattern}")
    lines.extend(
        [
            f"Memory: {state.memory}",
            f"Metadata source: {state.metadata_source}",
        ]
    )
    if state.source_name:
        lines.append(f"Source: {state.source_name}")
    if state.history:
        lines.append("History: " + " -> ".join(state.history[-4:]))
    return "\n".join(lines)


def infer_axis_metadata(arr: np.ndarray) -> tuple[AxisMetadata, ...]:
    """Infer common bioimage axes from shape when no explicit metadata exists."""
    names = infer_axes(arr)
    return _axis_metadata_from_order(names)


def infer_axis_metadata_from_shape(shape: tuple[int, ...]) -> tuple[AxisMetadata, ...]:
    """Infer axes from shape without allocating a same-shaped temporary array."""
    names = _infer_axes_from_shape(shape)
    return _axis_metadata_from_order(names)


def _axis_metadata_from_order(names: str) -> tuple[AxisMetadata, ...]:
    if names == "scalar":
        return ()
    return tuple(
        AxisMetadata(name=name.lower(), type=_axis_type_for_name(name))
        for name in _split_axis_order(names)
    )


def infer_axes(arr: np.ndarray) -> str:
    """Infer common bioimage axes from shape alone."""
    return _infer_axes_from_shape(tuple(arr.shape))


def _infer_axes_from_shape(shape: tuple[int, ...]) -> str:
    ndim = len(shape)
    if ndim == 0:
        return "scalar"
    if ndim == 1:
        return "X"

    has_channels = len(shape) >= 3 and shape[-1] in RGB_CHANNELS
    if has_channels:
        spatial_ndim = ndim - 1
        if spatial_ndim == 2:
            return "YXC"
        if spatial_ndim == 3:
            return "ZYXC"
        if spatial_ndim == 4:
            return "TZYXC"
        return _fallback_axes(spatial_ndim - 2) + "YXC"

    if ndim == 2:
        return "YX"
    if ndim == 3:
        return "ZYX"
    if ndim == 4:
        return "TZYX"
    return _fallback_axes(ndim - 2) + "YX"


def _axes_from_layer_metadata(
    layer_metadata: dict | None,
    shape: tuple[int, ...],
) -> tuple[tuple[AxisMetadata, ...], str, tuple[str, ...]]:
    if not isinstance(layer_metadata, dict):
        return infer_axis_metadata_from_shape(shape), "inferred from array shape", ()

    carried = layer_metadata.get("vipp_image_state")
    if isinstance(carried, dict):
        state = ImageState.from_dict(carried)
        if state is not None and len(state.axes) == len(shape):
            return state.axes, "VIPP carried state", state.history

    ome = layer_metadata.get("ome")
    if isinstance(ome, dict):
        axes = _axes_from_multiscales(ome.get("multiscales"), shape)
        if axes is not None:
            return axes, "OME-NGFF multiscales", ()

    axes = _axes_from_multiscales(layer_metadata.get("multiscales"), shape)
    if axes is not None:
        return axes, "OME-NGFF multiscales", ()

    axes_value = (
        layer_metadata.get("axes")
        or layer_metadata.get("axis_order")
        or layer_metadata.get("vipp_axis_order")
    )
    axes = _axes_from_value(axes_value, shape)
    if axes is not None:
        return axes, "napari layer axes metadata", ()

    return infer_axis_metadata_from_shape(shape), "inferred from array shape", ()


def _carried_image_state(layer_metadata: dict | None) -> ImageState | None:
    if not isinstance(layer_metadata, dict):
        return None
    carried = layer_metadata.get("vipp_image_state")
    if not isinstance(carried, dict):
        return None
    return ImageState.from_dict(carried)


def _axes_from_multiscales(value, shape: tuple[int, ...]):
    if not isinstance(value, list) or not value:
        return None
    multiscale = value[0]
    if not isinstance(multiscale, dict):
        return None
    axes = _axes_from_value(multiscale.get("axes"), shape)
    if axes is None:
        return None

    datasets = multiscale.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        return axes
    dataset = datasets[0]
    if not isinstance(dataset, dict):
        return axes

    scales: list[float] | None = None
    translations: list[float] | None = None
    transforms = dataset.get("coordinateTransformations")
    if isinstance(transforms, list):
        for transform in transforms:
            if not isinstance(transform, dict):
                continue
            values = transform.get("scale")
            if transform.get("type") == "scale" and isinstance(values, list):
                scales = [_safe_float(value, 1.0) for value in values]
            values = transform.get("translation")
            if transform.get("type") == "translation" and isinstance(values, list):
                translations = [_safe_float(value, 0.0) for value in values]

    if scales is None and translations is None:
        return axes
    return tuple(
        replace(
            axis,
            scale=scales[index] if scales and index < len(scales) else axis.scale,
            translation=(
                translations[index]
                if translations and index < len(translations)
                else axis.translation
            ),
        )
        for index, axis in enumerate(axes)
    )


def _axes_from_value(value, shape: tuple[int, ...]):
    if isinstance(value, str):
        names = _split_axis_order(value)
        if len(names) != len(shape):
            return None
        return tuple(
            AxisMetadata(name=name.lower(), type=_axis_type_for_name(name))
            for name in names
        )
    if isinstance(value, list):
        if len(value) != len(shape):
            return None
        axes: list[AxisMetadata] = []
        for index, axis in enumerate(value):
            if isinstance(axis, str):
                axes.append(
                    AxisMetadata(
                        name=axis.lower(),
                        type=_axis_type_for_name(axis),
                    )
                )
            elif isinstance(axis, dict):
                axis_name = str(axis.get("name", f"d{index}"))
                axes.append(
                    AxisMetadata(
                        name=axis_name,
                        type=str(axis.get("type", _axis_type_for_name(axis_name))),
                        unit=str(axis["unit"]) if axis.get("unit") else None,
                    )
                )
            else:
                return None
        return tuple(axes)
    return None


def _with_default_source_axes(
    axes: tuple[AxisMetadata, ...],
) -> tuple[AxisMetadata, ...]:
    """Mark source axes so derived arrays can follow napari's global sliders.

    napari's viewer dimensions are shared across all layers. When a VIPP node
    removes an axis, such as Split Channels removing C from TCZYX, the remaining
    output axes still need to know their original viewer-axis positions so
    thumbnails and histograms follow the same Z/T sliders as the inspect layer.
    """
    return tuple(
        axis if axis.source_axis is not None else replace(axis, source_axis=index)
        for index, axis in enumerate(axes)
    )


def _transformed_axes(
    input_state: ImageState,
    arr: np.ndarray,
    *,
    operation_id: str,
    params: dict[str, Any],
) -> tuple[AxisMetadata, ...]:
    axes = input_state.axes
    if operation_id == "crop_stack":
        axes = _crop_shifted_axes(axes, params)
    if operation_id == "reorder_axes":
        return _reordered_axes(axes, params, arr.ndim)
    if operation_id == "composite_to_rgb":
        return _composite_to_rgb_axes(axes, arr.ndim)

    if operation_id == "select_axis_slice" and params.get("range_mode", False):
        return _range_and_removed_axes(axes, params)

    if arr.ndim == len(axes):
        return axes

    if operation_id == "select_axis_slice":
        axis_indices = _selected_axis_indices(params, len(axes))
        if axis_indices and arr.ndim == len(axes) - len(axis_indices):
            return tuple(
                axis for index, axis in enumerate(axes) if index not in axis_indices
            )

    if operation_id == "mip" and arr.ndim == len(axes) - 1:
        axis_index = _clamped_axis(params.get("axis", 0), len(axes))
        return _remove_axis(axes, axis_index)

    if operation_id == "extract_channel" and arr.ndim == len(axes) - 1:
        channel_index = _channel_axis_index(axes)
        if channel_index is not None:
            return _remove_axis(axes, channel_index)

    if (
        operation_id in CHANNEL_COLLAPSE_OPERATIONS | {"convert_dtype"}
        and arr.ndim == len(axes) - 1
    ):
        channel_index = _channel_axis_index(axes)
        if channel_index is not None:
            return _remove_axis(axes, channel_index)

    return axes


def _crop_shifted_axes(
    axes: tuple[AxisMetadata, ...],
    params: dict[str, Any],
) -> tuple[AxisMetadata, ...]:
    spatial = [index for index, axis in enumerate(axes) if axis.type == "space"]
    if len(spatial) < 2:
        return axes

    y_index, x_index = spatial[-2], spatial[-1]
    top = _safe_float(params.get("top"), 0.0)
    left = _safe_float(params.get("left"), 0.0)
    shifted = list(axes)
    shifted[y_index] = _translated_axis(shifted[y_index], top)
    shifted[x_index] = _translated_axis(shifted[x_index], left)
    return tuple(shifted)


def _range_shifted_axes(
    axes: tuple[AxisMetadata, ...],
    params: dict[str, Any],
) -> tuple[AxisMetadata, ...]:
    ranges = _parse_axis_ranges(params.get("ranges"), len(axes))
    if not ranges:
        return axes
    shifted = list(axes)
    for axis_index, (start, _end) in ranges.items():
        shifted[axis_index] = _translated_axis(shifted[axis_index], start)
    return tuple(shifted)


def _range_and_removed_axes(
    axes: tuple[AxisMetadata, ...],
    params: dict[str, Any],
) -> tuple[AxisMetadata, ...]:
    shifted = _range_shifted_axes(axes, params)
    removed = _selected_axis_indices(
        params,
        len(shifted),
        axes_key="remove_axes",
        default_axis=False,
    )
    if not removed:
        return shifted
    return tuple(axis for index, axis in enumerate(shifted) if index not in removed)


def _reordered_axes(
    axes: tuple[AxisMetadata, ...],
    params: dict[str, Any],
    output_ndim: int,
) -> tuple[AxisMetadata, ...]:
    if len(axes) != output_ndim:
        return axes
    if not _axis_order_tokens(params.get("order", "")):
        return axes
    indices = _axis_order_indices(
        params.get("order", ""),
        len(axes),
        [axis.name for axis in axes],
    )
    if indices is None:
        return axes
    ordered = [axes[index] for index in indices]
    spatial_output_positions = [
        output_index
        for output_index, input_index in enumerate(indices)
        if axes[input_index].type == "space"
    ]
    spatial_semantics = [axis for axis in axes if axis.type == "space"]
    if len(spatial_output_positions) == len(spatial_semantics):
        for output_index, semantic_axis in zip(
            spatial_output_positions,
            spatial_semantics,
            strict=False,
        ):
            moved_axis = ordered[output_index]
            ordered[output_index] = replace(
                moved_axis,
                name=semantic_axis.name,
                type=semantic_axis.type,
                source_axis=output_index,
            )
    return tuple(
        replace(axis, source_axis=output_index)
        for output_index, axis in enumerate(ordered)
    )


def _multi_input_axes(
    first_axes: tuple[AxisMetadata, ...],
    arr: np.ndarray,
    *,
    operation_id: str,
    params: dict[str, Any],
) -> tuple[AxisMetadata, ...]:
    if operation_id == "combine_channels":
        channel_index = _clamped_insert_axis(
            params.get("channel_axis", 0),
            len(first_axes),
        )
        axes = list(first_axes)
        axes.insert(channel_index, AxisMetadata(name="c", type="channel"))
        return tuple(axes)
    return first_axes if arr.ndim == len(first_axes) else infer_axis_metadata(arr)


def _composite_to_rgb_axes(
    axes: tuple[AxisMetadata, ...],
    output_ndim: int,
) -> tuple[AxisMetadata, ...]:
    rgb_axis = AxisMetadata(name="rgb", type="channel")
    leading_count = max(output_ndim - 1, 0)
    spatial = [axis for axis in axes if axis.type == "space"]
    leading = spatial[-leading_count:] if leading_count else []
    if len(leading) == leading_count:
        return tuple(leading) + (rgb_axis,)
    # Fall back to a channel-axis removal when spatial axes are insufficient.
    channel_index = _channel_axis_index(axes)
    if channel_index is not None:
        return _remove_axis(axes, channel_index) + (rgb_axis,)
    return axes + (rgb_axis,)


def _split_output_axes(
    axes: tuple[AxisMetadata, ...],
    output_ndim: int,
) -> tuple[AxisMetadata, ...]:
    """Axes for one channel extracted from a multi-channel image."""
    spatial = [axis for axis in axes if axis.type == "space"]
    if len(spatial) >= output_ndim:
        return tuple(spatial[-output_ndim:])
    channel_index = _channel_axis_index(axes)
    if channel_index is not None:
        reduced = _remove_axis(axes, channel_index)
        if len(reduced) == output_ndim:
            return reduced
    return axes


def _transformed_channels(
    input_state: ImageState,
    operation_id: str,
    params: dict[str, Any],
) -> tuple[ChannelMetadata, ...]:
    channels = input_state.channels
    if operation_id == "extract_channel" and channels:
        index = int(np.clip(int(params.get("channel", 0)), 0, len(channels) - 1))
        return (channels[index],)
    if operation_id == "assign_channel_colors":
        return _channels_with_colors(channels, params.get("channel_colors", ""))
    if operation_id in CHANNEL_COLLAPSE_OPERATIONS:
        return ()
    return channels


def _multi_input_channels(
    states: list[ImageState],
    operation_id: str,
    params: dict[str, Any],
) -> tuple[ChannelMetadata, ...]:
    if operation_id != "combine_channels":
        return states[0].channels
    channels: list[ChannelMetadata] = []
    for index, state in enumerate(states):
        if state.channels:
            channels.append(state.channels[0])
        else:
            channels.append(ChannelMetadata(name=f"Channel {index + 1}"))
    return _channels_with_colors(tuple(channels), params.get("channel_colors", ""))


def with_channel_colors(
    state: ImageState | None,
    channel_colors: str | list[str] | tuple[str, ...],
) -> ImageState | None:
    if state is None:
        return None
    colors = channel_color_names(channel_colors)
    if not colors:
        return state
    count = max(_state_channel_count(state), len(colors), len(state.channels))
    channels = _channels_with_colors(state.channels, colors, count=count)
    return replace(state, channels=channels)


def _channels_with_colors(
    channels: tuple[ChannelMetadata, ...],
    channel_colors: str | list[str] | tuple[str, ...],
    *,
    count: int | None = None,
) -> tuple[ChannelMetadata, ...]:
    colors = channel_color_names(channel_colors)
    if not colors:
        return channels
    target_count = max(count or 0, len(colors), len(channels))
    updated = list(channels[:target_count])
    while len(updated) < target_count:
        updated.append(ChannelMetadata(name=f"Channel {len(updated) + 1}"))
    for index, color_name in enumerate(colors[:target_count]):
        color = channel_color_int(color_name)
        if color is not None:
            updated[index] = replace(updated[index], color=color)
    return tuple(updated)


def _state_channel_count(state: ImageState) -> int:
    channel_index = _channel_axis_index(state.axes)
    if channel_index is None or channel_index >= len(state.shape):
        return len(state.channels)
    return int(state.shape[channel_index])


def _split_output_channels(
    channels: tuple[ChannelMetadata, ...],
    port_name: str,
) -> tuple[ChannelMetadata, ...]:
    if not channels:
        return ()
    digits = "".join(character for character in port_name if character.isdigit())
    index = max(int(digits or "1") - 1, 0)
    index = min(index, len(channels) - 1)
    return (channels[index],)


def _translated_axis(axis: AxisMetadata, pixels: float) -> AxisMetadata:
    return replace(axis, translation=axis.translation + pixels * axis.scale)


def _remove_axis(
    axes: tuple[AxisMetadata, ...],
    axis_index: int,
) -> tuple[AxisMetadata, ...]:
    return tuple(axis for index, axis in enumerate(axes) if index != axis_index)


def _operation_history(
    input_state: ImageState,
    operation_id: str,
    operation_title: str,
    params: dict[str, Any],
) -> str:
    if operation_id == "mip":
        axis = _axis_label(input_state.axes, params.get("axis", 0))
        return f"{operation_title}: projected {axis}"
    if operation_id == "select_axis_slice":
        if params.get("range_mode", False):
            ranges = _parse_axis_ranges(params.get("ranges"), len(input_state.axes))
            range_text = ", ".join(
                f"{_axis_label(input_state.axes, axis)}[{start}..{end}]"
                for axis, (start, end) in sorted(ranges.items())
            )
            removals = _selected_axis_index_pairs(
                input_state.axes,
                params,
                axes_key="remove_axes",
                indices_key="remove_indices",
                default_axis=False,
            )
            removal_text = ", ".join(
                f"{_axis_label(input_state.axes, axis)}[{index}]"
                for axis, index in removals
            )
            parts = []
            if range_text:
                parts.append(f"kept {range_text}")
            if removal_text:
                parts.append(f"removed {removal_text}")
            if not parts:
                return f"{operation_title}: kept full input axes"
            return f"{operation_title}: {'; '.join(parts)}"
        selections = _selected_axis_index_pairs(input_state.axes, params)
        selected = ", ".join(
            f"{_axis_label(input_state.axes, axis)}[{index}]"
            for axis, index in selections
        )
        return f"{operation_title}: selected {selected}"
    if operation_id == "reorder_axes":
        if not str(params.get("order", "")).strip():
            return f"{operation_title}: kept input order"
        indices = _axis_order_indices(
            params.get("order", ""),
            len(input_state.axes),
            [axis.name for axis in input_state.axes],
        )
        if indices is None:
            return f"{operation_title}: kept input order"
        data_order = "".join(input_state.axes[index].short_label for index in indices)
        effective_axes = _reordered_axes(
            input_state.axes,
            params,
            len(input_state.axes),
        )
        effective_order = "".join(axis.short_label for axis in effective_axes)
        if effective_order != data_order:
            return (
                f"{operation_title}: transposed data to {data_order}; "
                f"effective axes {effective_order}"
            )
        return f"{operation_title}: reordered axes to {data_order}"
    if operation_id == "extract_channel":
        return f"{operation_title}: selected channel {int(params.get('channel', 0))}"
    if operation_id == "composite_to_rgb":
        return f"{operation_title}: mapped channels to RGB"
    if operation_id == "crop_stack":
        return (
            f"{operation_title}: cropped top={int(params.get('top', 0))}, "
            f"bottom={int(params.get('bottom', 0))}, "
            f"left={int(params.get('left', 0))}, right={int(params.get('right', 0))}"
        )
    if operation_id == "convert_dtype":
        return (
            f"{operation_title}: {params.get('output_dtype', 'uint8')} "
            f"via {params.get('scaling', 'rescale')}"
        )
    return operation_title


def _multi_input_history(
    states: list[ImageState],
    operation_id: str,
    operation_title: str,
    params: dict[str, Any],
) -> str:
    count = len(states)
    if operation_id == "combine_channels":
        return f"{operation_title}: combined {count} inputs as channels"
    if operation_id == "calculate_weighted_image":
        weights = str(params.get("weights", "")).strip() or "1"
        offset = _safe_float(params.get("offset"), 0.0)
        return (
            f"{operation_title}: weighted {count} inputs "
            f"(weights {weights}, offset {_format_number(offset)})"
        )
    return f"{operation_title}: combined {count} inputs"


def _axis_label(axes: tuple[AxisMetadata, ...], axis_value) -> str:
    if not axes:
        return "axis 0"
    axis_index = _clamped_axis(axis_value, len(axes))
    axis = axes[axis_index]
    return f"{axis.name} axis ({axis_index})"


def _selected_axis_indices(
    params: dict[str, Any],
    ndim: int,
    *,
    axes_key: str = "axes",
    default_axis: bool = True,
) -> set[int]:
    axes = _parse_int_list(params.get(axes_key))
    if not axes:
        if not default_axis:
            return set()
        axes = [_clamped_axis(params.get("axis", 0), ndim)]
    return {_clamped_axis(axis, ndim) for axis in axes}


def _parse_axis_ranges(value, ndim: int) -> dict[int, tuple[int, int]]:
    if not isinstance(value, str) or not value.strip():
        return {}
    ranges: dict[int, tuple[int, int]] = {}
    for part in value.split(";"):
        pieces = [piece.strip() for piece in part.split(":")]
        if len(pieces) != 3:
            continue
        try:
            axis = _clamped_axis(int(pieces[0]), ndim)
            start = int(pieces[1])
            end = int(pieces[2])
        except ValueError:
            continue
        if start > end:
            start, end = end, start
        ranges[axis] = (start, end)
    return ranges


def _selected_axis_index_pairs(
    axes: tuple[AxisMetadata, ...],
    params: dict[str, Any],
    *,
    axes_key: str = "axes",
    indices_key: str = "indices",
    default_axis: bool = True,
) -> list[tuple[int, int]]:
    axis_values = _parse_int_list(params.get(axes_key))
    index_values = _parse_int_list(params.get(indices_key))
    if not axis_values:
        if not default_axis:
            return []
        axis_values = [params.get("axis", 0)]
        index_values = [params.get("index", 0)]
    pairs = []
    for position, axis_value in enumerate(axis_values):
        axis_index = _clamped_axis(axis_value, len(axes))
        index = index_values[position] if position < len(index_values) else 0
        pairs.append((axis_index, int(index)))
    return sorted(pairs)


def _parse_int_list(value) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        parts = [value]
    parsed = []
    for part in parts:
        try:
            parsed.append(int(part))
        except (TypeError, ValueError):
            continue
    return parsed


def _axis_order_indices(
    order,
    ndim: int,
    axis_names: list[str],
) -> tuple[int, ...] | None:
    if ndim <= 1:
        return tuple(range(ndim))
    tokens = _axis_order_tokens(order)
    if not tokens:
        return tuple(range(ndim))
    if len(tokens) != ndim:
        return None

    indices = _numeric_axis_order(tokens, ndim)
    if indices is not None:
        return indices
    return _named_axis_order(tokens, ndim, axis_names)


def _axis_order_tokens(order) -> list[str]:
    if order is None:
        return []
    if isinstance(order, (list, tuple)):
        return [str(part).strip() for part in order if str(part).strip()]
    text = str(order).strip()
    if not text:
        return []
    if any(separator in text for separator in (",", ";", " ")):
        normalized = text.replace(";", ",").replace(" ", ",")
        return [part.strip() for part in normalized.split(",") if part.strip()]
    return list(text)


def _numeric_axis_order(tokens: list[str], ndim: int) -> tuple[int, ...] | None:
    indices: list[int] = []
    try:
        for token in tokens:
            axis = int(token)
            if axis < 0:
                axis += ndim
            indices.append(axis)
    except ValueError:
        return None
    if sorted(indices) != list(range(ndim)):
        return None
    return tuple(indices)


def _named_axis_order(
    tokens: list[str],
    ndim: int,
    axis_names: list[str],
) -> tuple[int, ...] | None:
    names = [str(name).strip().lower() for name in axis_names]
    if len(names) != ndim:
        return None
    indices: list[int] = []
    used: set[int] = set()
    for token in tokens:
        target = str(token).strip().lower()
        matches = [
            index
            for index, name in enumerate(names)
            if name == target and index not in used
        ]
        if not matches:
            return None
        index = matches[0]
        indices.append(index)
        used.add(index)
    return tuple(indices)


def _clamped_axis(value, ndim: int) -> int:
    if ndim <= 0:
        return 0
    try:
        axis = int(value)
    except Exception:
        axis = 0
    return min(max(axis, 0), ndim - 1)


def _clamped_insert_axis(value, ndim: int) -> int:
    try:
        axis = int(value)
    except Exception:
        axis = 0
    return min(max(axis, 0), ndim)


def _channel_axis_index(axes: tuple[AxisMetadata, ...]) -> int | None:
    for index, axis in enumerate(axes):
        if axis.type == "channel" or axis.name.lower() == "c":
            return index
    return None


def _coerce_state(state_or_data) -> ImageState | None:
    if state_or_data is None:
        return None
    if isinstance(state_or_data, ImageState):
        return state_or_data
    return image_state_from_array(state_or_data)


def _coerce_table_state(state_or_data) -> TableState | None:
    if state_or_data is None:
        return None
    if isinstance(state_or_data, TableState):
        return state_or_data
    if is_table_data(state_or_data):
        return table_state_from_data(state_or_data)
    return None


def _with_operation_kind(
    state: ImageState | None,
    operation_id: str,
) -> ImageState | None:
    if state is not None and operation_id in LABEL_OPERATIONS:
        return replace(state, kind="label image")
    return state


def _kind_label(arr: np.ndarray, axes: tuple[AxisMetadata, ...]) -> str:
    if arr.dtype == bool:
        return "binary mask"
    channel_axis = _channel_axis_index(axes)
    if channel_axis is not None:
        channel_count = arr.shape[channel_axis]
        if channel_axis == arr.ndim - 1 and channel_count == 3:
            return "RGB image"
        if channel_axis == arr.ndim - 1 and channel_count == 4:
            return "RGBA image"
        return "multi-channel image"
    if np.issubdtype(arr.dtype, np.number):
        return "intensity image"
    return "array"


def _shape_label(shape: tuple[int, ...]) -> str:
    return " x ".join(str(size) for size in shape) if shape else "scalar"


def _axes_detail_label(state: ImageState) -> str:
    if not state.axes:
        return "scalar"
    return ", ".join(
        f"{axis.name}({axis.type})" for axis in state.axes
    )


def _dimensions_label(state: ImageState) -> str:
    if len(state.axes) != len(state.shape):
        return _shape_label(state.shape)
    return ", ".join(
        f"{axis.name}={size}"
        for axis, size in zip(state.axes, state.shape, strict=True)
    )


def _dimensions_compact_label(state: ImageState) -> str:
    return _shape_label(state.shape)


def _scale_label(state: ImageState) -> str:
    axes = [axis for axis in state.axes if axis.type in {"space", "time"}]
    if not axes:
        return "not specified"
    return ", ".join(
        f"{axis.name}={_format_number(axis.scale)} {axis.unit or 'pixel'}"
        for axis in axes
    )


def _origin_label(state: ImageState) -> str:
    axes = [axis for axis in state.axes if axis.type in {"space", "time"}]
    if not axes:
        return "not specified"
    return ", ".join(
        f"{axis.name}={_format_number(axis.translation)} {axis.unit or 'pixel'}"
        for axis in axes
    )


def _axis_count_label(state: ImageState, axis_type: str) -> str:
    for axis, size in zip(state.axes, state.shape, strict=False):
        if axis.type == axis_type:
            return str(size)
    return "none"


def _named_axis_count_label(state: ImageState, name: str) -> str:
    for axis, size in zip(state.axes, state.shape, strict=False):
        if axis.name.lower() == name:
            return str(size)
    return "none"


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


def _split_axis_order(value: str) -> list[str]:
    value = value.strip()
    if "," in value:
        return [part.strip() for part in value.split(",") if part.strip()]
    return [part for part in value if not part.isspace()]


def _axis_type_for_name(name: str) -> str:
    normalized = name.lower()
    if normalized == "t":
        return "time"
    if normalized == "c":
        return "channel"
    if normalized in {"x", "y", "z"}:
        return "space"
    return "unknown"


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


def _is_lazy_array(data) -> bool:
    return (
        not isinstance(data, np.ndarray)
        and hasattr(data, "shape")
        and hasattr(data, "dtype")
        and hasattr(data, "compute")
    )


def _lazy_kind_label(
    dtype: np.dtype,
    shape: tuple[int, ...],
    axes: tuple[AxisMetadata, ...],
) -> str:
    if dtype == np.dtype(bool):
        return "binary mask"
    channel_axes = [
        (index, axis) for index, axis in enumerate(axes) if axis.type == "channel"
    ]
    if channel_axes:
        index, axis = channel_axes[-1]
        if axis.name in {"rgb", "rgba"} or shape[index] in RGB_CHANNELS:
            return "RGB image"
        return "multi-channel image"
    return "intensity image"


def _safe_float(value, default: float) -> float:
    try:
        number = float(value)
    except Exception:
        return default
    if not np.isfinite(number):
        return default
    return number


def _optional_float(value) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value) -> int | None:
    if value in {None, ""}:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
