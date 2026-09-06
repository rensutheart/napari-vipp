"""Packed host-finalization contract for GPU skeleton-network analysis.

The accelerator computes one fixed-width numeric record per connected skeleton
component.  Public :class:`~napari_vipp.core.tables.TableData` construction is
kept on the host because the table contains both numeric values and an optional
physical-unit string.  The private boundary is a self-describing, versioned
``uint8`` payload so malformed or mismatched accelerator output cannot silently
be interpreted as a scientific result.

Only ``Analyze Skeleton`` with an already-skeletonized boolean input is covered
by this contract.  Skeletonization itself remains an authoritative CPU region.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from math import prod

import numpy as np

from napari_vipp.core.connected_components import resolve_spatial_ndim
from napari_vipp.core.tables import TableData, table_from_columns

SKELETON_PAYLOAD_MAGIC = b"VIPPSKL1"
SKELETON_PAYLOAD_VERSION = 1
SKELETON_RECORD_BASE_WORDS = 14

# version, header bytes, spatial ndim, leading ndim, row count, record words,
# directory bytes, and total bytes.
_HEADER_VALUE_COUNT = 8
SKELETON_PAYLOAD_HEADER_BYTES = len(SKELETON_PAYLOAD_MAGIC) + 8 * _HEADER_VALUE_COUNT


@dataclass(frozen=True, slots=True)
class SkeletonAnalysisLayout:
    """Frozen axis, calibration, and record layout for one analysis call."""

    input_shape: tuple[int, ...]
    spatial_axes: tuple[int, ...]
    leading_axes: tuple[int, ...]
    permutation: tuple[int, ...]
    leading_shape: tuple[int, ...]
    spatial_shape: tuple[int, ...]
    leading_axis_names: tuple[str, ...]
    units: object

    @property
    def spatial_ndim(self) -> int:
        return len(self.spatial_shape)

    @property
    def leading_ndim(self) -> int:
        return len(self.leading_shape)

    @property
    def block_count(self) -> int:
        return int(prod(self.leading_shape)) if self.leading_shape else 1

    @property
    def record_words(self) -> int:
        return self.leading_ndim + SKELETON_RECORD_BASE_WORDS


@dataclass(frozen=True, slots=True)
class _SkeletonPayloadRecord:
    leading_index: tuple[int, ...]
    component_id: int
    component_count: int
    total_voxel_count: int
    voxel_count: int
    endpoint_count: int
    junction_count: int
    isolated_count: int
    branch_count: int
    graph_node_count: int
    graph_edge_count: int
    voxel_edge_count: int
    cycle_count: int
    pixel_length: float
    physical_length: float


def skeleton_analysis_layout(
    shape: Sequence[int],
    *,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_names: Sequence[str] | None = None,
    axis_types: Sequence[str] | None = None,
    axis_scales: Sequence[float | None] | None = None,
    axis_units: Sequence[str | None] | None = None,
) -> SkeletonAnalysisLayout:
    """Resolve the exact CPU table layout without inspecting image values."""

    # Keep scientific imports behind provider/finalizer use.  In particular,
    # importing this ABI must not trigger an accelerator array probe.
    from napari_vipp.core import operations

    input_shape = _validated_shape(shape)
    ndim = len(input_shape)
    spatial_ndim = resolve_spatial_ndim(
        ndim,
        spatial_mode,
        resolved_spatial_ndim,
    )
    if spatial_ndim not in {2, 3}:
        raise ValueError("GPU skeleton analysis requires resolved 2D or 3D blocks.")

    normalized_names = operations._measurement_axis_names(ndim, axis_names)
    normalized_types = operations._measurement_axis_types(ndim, axis_types)
    spatial_axes = operations._measurement_spatial_axes(
        ndim,
        spatial_ndim,
        normalized_names,
        normalized_types,
    )
    if len(spatial_axes) != spatial_ndim:
        spatial_axes = tuple(range(ndim - spatial_ndim, ndim))
    leading_axes = tuple(index for index in range(ndim) if index not in spatial_axes)
    permutation = leading_axes + spatial_axes
    moved_names = tuple(normalized_names[index] for index in permutation)
    moved_scales = operations._reordered_axis_values(
        axis_scales,
        ndim,
        spatial_axes,
    )
    moved_units = operations._reordered_axis_values(
        axis_units,
        ndim,
        spatial_axes,
    )
    leading_names = operations._safe_axis_column_names(
        moved_names[:-spatial_ndim],
        fallback=tuple(f"axis_{index}" for index in range(ndim - spatial_ndim)),
    )
    units = operations._skeleton_units(
        spatial_ndim,
        moved_scales[-spatial_ndim:],
        moved_units[-spatial_ndim:],
    )
    return SkeletonAnalysisLayout(
        input_shape=input_shape,
        spatial_axes=tuple(spatial_axes),
        leading_axes=leading_axes,
        permutation=permutation,
        leading_shape=tuple(input_shape[index] for index in leading_axes),
        spatial_shape=tuple(input_shape[index] for index in spatial_axes),
        leading_axis_names=tuple(leading_names),
        units=units,
    )


def skeleton_payload_header(
    *,
    layout: SkeletonAnalysisLayout,
    row_count: int,
) -> np.ndarray:
    """Return the canonical fixed header as a native one-dimensional byte array."""

    rows = _validated_nonnegative_integer(row_count, "row_count")
    maximum_rows = int(prod(layout.input_shape)) if layout.input_shape else 0
    if rows > maximum_rows:
        raise ValueError("Skeleton payload rows cannot exceed input elements.")
    directory_bytes = rows * layout.record_words * 8
    total_bytes = SKELETON_PAYLOAD_HEADER_BYTES + directory_bytes
    values = np.asarray(
        (
            SKELETON_PAYLOAD_VERSION,
            SKELETON_PAYLOAD_HEADER_BYTES,
            layout.spatial_ndim,
            layout.leading_ndim,
            rows,
            layout.record_words,
            directory_bytes,
            total_bytes,
        ),
        dtype="<u8",
    )
    header = np.empty(SKELETON_PAYLOAD_HEADER_BYTES, dtype=np.uint8)
    header[: len(SKELETON_PAYLOAD_MAGIC)] = np.frombuffer(
        SKELETON_PAYLOAD_MAGIC,
        dtype=np.uint8,
    )
    header[len(SKELETON_PAYLOAD_MAGIC) :] = values.view(np.uint8)
    return header


def finalize_analyze_skeleton_table(
    payload,
    *,
    layout: SkeletonAnalysisLayout,
    source_name: str = "",
) -> TableData:
    """Validate a packed accelerator payload and construct exact public output."""

    from napari_vipp.core import operations

    records = _parse_skeleton_payload(payload, layout=layout)
    columns = operations._skeleton_empty_columns(
        layout.leading_axis_names,
        layout.units,
    )
    for record in records:
        for axis_position, axis_name in enumerate(layout.leading_axis_names):
            columns[f"{axis_name}_index"].append(
                int(record.leading_index[axis_position])
            )
        columns["component_id"].append(record.component_id)
        columns["component_count_in_block"].append(record.component_count)
        columns["component_voxel_fraction"].append(
            float(record.voxel_count / record.total_voxel_count)
        )
        columns["skeleton_voxel_count"].append(record.voxel_count)
        columns["endpoint_voxel_count"].append(record.endpoint_count)
        columns["junction_voxel_count"].append(record.junction_count)
        columns["isolated_node_count"].append(record.isolated_count)
        columns["branch_count"].append(record.branch_count)
        columns["graph_node_count"].append(record.graph_node_count)
        columns["graph_edge_count"].append(record.graph_edge_count)
        columns["voxel_graph_edge_count"].append(record.voxel_edge_count)
        columns["cycle_count"].append(record.cycle_count)
        columns[layout.units.length_column].append(record.pixel_length)
        if layout.units.physical_column:
            columns[layout.units.physical_column].append(record.physical_length)
            columns["physical_unit"].append(layout.units.unit_label)

    return table_from_columns(
        columns,
        name="Skeleton network measurements",
        table_kind="Skeleton network",
        source_name=str(source_name),
        column_units=layout.units.column_units,
    )


def finalize_analyze_skeleton_outputs(host_outputs, *, call):
    """Finalize the one-output resident provider ABI after device cleanup."""

    payloads = tuple(host_outputs)
    if len(payloads) != 1:
        raise ValueError(
            "Analyze Skeleton host finalization requires exactly one packed "
            f"payload; received {len(payloads)}."
        )
    operation_id = str(getattr(call, "operation_id", "")).strip()
    if operation_id != "analyze_skeleton":
        raise ValueError(
            "Analyze Skeleton host finalization cannot handle operation "
            f"{operation_id!r}."
        )
    kwargs = dict(getattr(call, "kwargs", {}))
    _validate_already_skeletonized(kwargs.get("input_mode", "Already skeletonized"))
    layout = skeleton_analysis_layout(
        _prepared_skeleton_input_shape(call),
        spatial_mode=kwargs.get("spatial_mode", "Auto from axes"),
        resolved_spatial_ndim=kwargs.get("resolved_spatial_ndim"),
        axis_names=kwargs.get("axis_names"),
        axis_types=kwargs.get("axis_types"),
        axis_scales=kwargs.get("axis_scales"),
        axis_units=kwargs.get("axis_units"),
    )
    return finalize_analyze_skeleton_table(
        payloads[0],
        layout=layout,
        source_name=str(kwargs.get("source_name", "")),
    )


def _parse_skeleton_payload(
    payload,
    *,
    layout: SkeletonAnalysisLayout,
) -> tuple[_SkeletonPayloadRecord, ...]:
    raw = np.asarray(payload)
    if raw.dtype != np.dtype(np.uint8) or not raw.dtype.isnative:
        raise TypeError(
            "Packed skeleton payloads must use native uint8 storage; "
            f"received {raw.dtype}."
        )
    if raw.ndim != 1:
        raise ValueError(
            "Packed skeleton payloads must be one-dimensional; "
            f"received shape {raw.shape}."
        )
    raw = np.ascontiguousarray(raw)
    if raw.size < SKELETON_PAYLOAD_HEADER_BYTES:
        raise ValueError("Packed skeleton payload has a truncated header.")
    if raw[: len(SKELETON_PAYLOAD_MAGIC)].tobytes() != SKELETON_PAYLOAD_MAGIC:
        raise ValueError("Packed skeleton payload has an invalid magic value.")
    header = np.frombuffer(
        raw[len(SKELETON_PAYLOAD_MAGIC) : SKELETON_PAYLOAD_HEADER_BYTES],
        dtype="<u8",
        count=_HEADER_VALUE_COUNT,
    )
    (
        version,
        header_bytes,
        spatial_ndim,
        leading_ndim,
        row_count,
        record_words,
        directory_bytes,
        total_bytes,
    ) = (int(value) for value in header)
    if version != SKELETON_PAYLOAD_VERSION:
        raise ValueError(f"Unsupported skeleton payload version {version}.")
    if header_bytes != SKELETON_PAYLOAD_HEADER_BYTES:
        raise ValueError("Packed skeleton payload has an invalid header size.")
    if spatial_ndim != layout.spatial_ndim:
        raise ValueError(
            "Packed skeleton payload spatial rank does not match the call."
        )
    if leading_ndim != layout.leading_ndim:
        raise ValueError(
            "Packed skeleton payload leading rank does not match the call."
        )
    if record_words != layout.record_words:
        raise ValueError(
            "Packed skeleton payload record width does not match the call."
        )
    maximum_rows = int(prod(layout.input_shape)) if layout.input_shape else 0
    if row_count > maximum_rows:
        raise ValueError("Packed skeleton payload has more rows than input elements.")
    expected_directory_bytes = row_count * record_words * 8
    expected_total_bytes = SKELETON_PAYLOAD_HEADER_BYTES + expected_directory_bytes
    if (
        directory_bytes != expected_directory_bytes
        or total_bytes != expected_total_bytes
    ):
        raise ValueError("Packed skeleton payload sizes are inconsistent.")
    if total_bytes != raw.size:
        raise ValueError(
            "Packed skeleton payload byte count does not match its header."
        )

    directory = np.frombuffer(
        raw[SKELETON_PAYLOAD_HEADER_BYTES:],
        dtype="<u8",
        count=row_count * record_words,
    ).reshape(row_count, record_words)
    return _validated_records(directory, layout=layout)


def _validated_records(
    directory: np.ndarray,
    *,
    layout: SkeletonAnalysisLayout,
) -> tuple[_SkeletonPayloadRecord, ...]:
    records: list[_SkeletonPayloadRecord] = []
    previous_key: tuple[tuple[int, ...], int] | None = None
    lead = layout.leading_ndim
    for row in directory:
        leading_index = tuple(int(value) for value in row[:lead])
        cursor = lead
        integer_values = [int(value) for value in row[cursor : cursor + 12]]
        (
            component_id,
            component_count,
            total_voxel_count,
            voxel_count,
            endpoint_count,
            junction_count,
            isolated_count,
            branch_count,
            graph_node_count,
            graph_edge_count,
            voxel_edge_count,
            cycle_count,
        ) = integer_values
        pixel_length = _word_to_float(row[cursor + 12])
        physical_length = _word_to_float(row[cursor + 13])

        if any(
            value >= layout.leading_shape[index]
            for index, value in enumerate(leading_index)
        ):
            raise ValueError("Packed skeleton leading index is out of bounds.")
        if component_id <= 0 or component_count <= 0 or component_id > component_count:
            raise ValueError("Packed skeleton component IDs/counts are invalid.")
        if total_voxel_count <= 0 or not 0 < voxel_count <= total_voxel_count:
            raise ValueError("Packed skeleton voxel counts are invalid.")
        if endpoint_count + junction_count + isolated_count > voxel_count:
            raise ValueError("Packed skeleton node counts exceed component voxels.")
        if graph_node_count != endpoint_count + junction_count + isolated_count:
            raise ValueError("Packed skeleton graph-node count is inconsistent.")
        if graph_edge_count != branch_count:
            raise ValueError("Packed skeleton graph-edge count is inconsistent.")
        expected_cycles = voxel_edge_count - voxel_count + 1
        if cycle_count != expected_cycles or cycle_count < 0:
            raise ValueError("Packed skeleton cycle count is inconsistent.")
        if voxel_count == 1 and (
            endpoint_count != 0
            or junction_count != 0
            or isolated_count != 1
            or branch_count != 0
            or voxel_edge_count != 0
        ):
            raise ValueError("Packed singleton skeleton metrics are inconsistent.")
        if not np.isfinite(pixel_length) or pixel_length < 0.0:
            raise ValueError(
                "Packed skeleton pixel length must be finite and nonnegative."
            )
        if not np.isfinite(physical_length) or physical_length < 0.0:
            raise ValueError(
                "Packed skeleton physical length must be finite and nonnegative."
            )

        key = (leading_index, component_id)
        if previous_key is not None and key <= previous_key:
            raise ValueError(
                "Packed skeleton records must use block-major component order."
            )
        previous_key = key
        records.append(
            _SkeletonPayloadRecord(
                leading_index=leading_index,
                component_id=component_id,
                component_count=component_count,
                total_voxel_count=total_voxel_count,
                voxel_count=voxel_count,
                endpoint_count=endpoint_count,
                junction_count=junction_count,
                isolated_count=isolated_count,
                branch_count=branch_count,
                graph_node_count=graph_node_count,
                graph_edge_count=graph_edge_count,
                voxel_edge_count=voxel_edge_count,
                cycle_count=cycle_count,
                pixel_length=pixel_length,
                physical_length=physical_length,
            )
        )

    for leading_index, grouped in groupby(records, key=lambda item: item.leading_index):
        block_records = tuple(grouped)
        component_count = block_records[0].component_count
        total_voxel_count = block_records[0].total_voxel_count
        if component_count != len(block_records):
            raise ValueError(
                "Packed skeleton block "
                f"{leading_index} component count is inconsistent."
            )
        if tuple(record.component_id for record in block_records) != tuple(
            range(1, component_count + 1)
        ):
            raise ValueError(
                f"Packed skeleton block {leading_index} component IDs are not dense."
            )
        if any(
            record.component_count != component_count
            or record.total_voxel_count != total_voxel_count
            for record in block_records
        ):
            raise ValueError(
                f"Packed skeleton block {leading_index} summaries disagree."
            )
        if sum(record.voxel_count for record in block_records) != total_voxel_count:
            raise ValueError(
                f"Packed skeleton block {leading_index} voxel total is inconsistent."
            )
    return tuple(records)


def _word_to_float(value: np.uint64) -> float:
    word = np.asarray([value], dtype="<u8")
    return float(word.view("<f8")[0])


def _validate_already_skeletonized(input_mode: object) -> None:
    normalized = str(input_mode).strip().casefold()
    if normalized != "already skeletonized":
        raise ValueError(
            "GPU Analyze Skeleton supports only 'Already skeletonized'; "
            "skeletonization remains on CPU."
        )


def _prepared_skeleton_input_shape(call) -> tuple[int, ...]:
    input_states = tuple(getattr(call, "input_states", ()))
    if input_states:
        shape = getattr(input_states[0], "shape", None)
        if shape is not None:
            return _validated_shape(shape)
    inputs = tuple(getattr(call, "inputs", ()))
    if inputs and inputs[0] is not None:
        shape = getattr(inputs[0], "shape", None)
        if shape is not None:
            return _validated_shape(shape)
    raise ValueError(
        "Analyze Skeleton host finalization requires the original input shape "
        "in call.input_states[0].shape."
    )


def _validated_shape(shape: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in tuple(shape):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError("Skeleton analysis shapes require non-negative integers.")
        size = int(value)
        if size < 0:
            raise ValueError("Skeleton analysis shapes require non-negative integers.")
        result.append(size)
    return tuple(result)


def _validated_nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a non-negative integer.")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer.")
    return normalized


__all__ = [
    "SKELETON_PAYLOAD_HEADER_BYTES",
    "SKELETON_PAYLOAD_MAGIC",
    "SKELETON_PAYLOAD_VERSION",
    "SKELETON_RECORD_BASE_WORDS",
    "SkeletonAnalysisLayout",
    "finalize_analyze_skeleton_outputs",
    "finalize_analyze_skeleton_table",
    "skeleton_analysis_layout",
    "skeleton_payload_header",
]
