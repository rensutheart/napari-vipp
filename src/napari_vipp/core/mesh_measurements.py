"""Packed host-finalization contract for 3D mesh morphology accelerators.

The expensive label discovery and crop extraction can run on an accelerator,
but VIPP's authoritative marching-cubes and convex-hull implementations remain
host algorithms.  A device provider therefore returns one opaque ``uint8``
payload containing sorted object metadata and compact, bounding-box-local mask
encodings.  Dense crops use bit packing; pathological sparse crops use uint32
flat indices so encoded data is always bounded by four bytes per positive
voxel.  This module validates that payload before reconstructing the exact
mixed-type :class:`~napari_vipp.core.tables.TableData` exposed by the CPU path.

The payload is deliberately self-describing and versioned.  Every integer is
little-endian and unsigned.  After the fixed header comes a rectangular record
directory, followed by tightly concatenated mask bytes.  Records for objects
below ``minimum_voxel_count`` contain no mask data at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import numpy as np

from napari_vipp.core.connected_components import resolve_spatial_ndim
from napari_vipp.core.tables import TableData, table_from_columns

MESH_PAYLOAD_MAGIC = b"VIPPMSH1"
MESH_PAYLOAD_VERSION = 1
MESH_PAYLOAD_SPATIAL_NDIM = 3

# magic + version, header bytes, spatial ndim, leading ndim, row count,
# record width, directory bytes, mask-data bytes, and total bytes.
_HEADER_VALUE_COUNT = 9
MESH_PAYLOAD_HEADER_BYTES = len(MESH_PAYLOAD_MAGIC) + 8 * _HEADER_VALUE_COUNT
MESH_RECORD_BASE_WORDS = 12
MESH_ENCODING_NONE = 0
MESH_ENCODING_BITMASK = 1
MESH_ENCODING_SPARSE_UINT32 = 2


@dataclass(frozen=True, slots=True)
class MeshMorphologyLayout:
    """Frozen axis and public-table layout for one mesh-measurement call."""

    input_shape: tuple[int, ...]
    spatial_axes: tuple[int, ...]
    leading_axes: tuple[int, ...]
    permutation: tuple[int, ...]
    leading_shape: tuple[int, ...]
    spatial_shape: tuple[int, int, int]
    leading_axis_names: tuple[str, ...]
    spatial_axis_names: tuple[str, str, str]
    units: object
    include_convex_hull_metrics: bool

    @property
    def leading_ndim(self) -> int:
        return len(self.leading_shape)

    @property
    def block_count(self) -> int:
        return int(prod(self.leading_shape)) if self.leading_shape else 1

    @property
    def record_words(self) -> int:
        return self.leading_ndim + MESH_RECORD_BASE_WORDS


@dataclass(frozen=True, slots=True)
class _MeshPayloadRecord:
    leading_index: tuple[int, ...]
    label_id: int
    voxel_count: int
    bbox_minimum: tuple[int, int, int]
    bbox_maximum: tuple[int, int, int]
    encoding: int
    data_offset: int
    data_nbytes: int
    data_count: int


def mesh_morphology_layout(
    shape: Sequence[int],
    *,
    spatial_mode: str = "Auto from axes",
    resolved_spatial_ndim: int | None = None,
    axis_names: Sequence[str] | None = None,
    axis_types: Sequence[str] | None = None,
    axis_scales: Sequence[float | None] | None = None,
    axis_units: Sequence[str | None] | None = None,
    include_convex_hull_metrics: bool = True,
) -> MeshMorphologyLayout:
    """Resolve the CPU-identical 3D mesh layout without inspecting values."""

    # ``operations`` imports scikit-image, which may probe accelerator array
    # namespaces.  Keep that import behind actual provider/finalizer use so
    # merely discovering this contract never imports CuPy.
    from napari_vipp.core import operations

    normalized_shape = _validated_shape(shape)
    ndim = len(normalized_shape)
    spatial_ndim = resolve_spatial_ndim(
        ndim,
        spatial_mode,
        resolved_spatial_ndim,
    )
    if spatial_ndim != MESH_PAYLOAD_SPATIAL_NDIM:
        raise ValueError("Measure 3D Mesh Morphology requires true 3D labels.")

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
    moved_scales = operations._reordered_axis_values(axis_scales, ndim, spatial_axes)
    moved_units = operations._reordered_axis_values(axis_units, ndim, spatial_axes)
    spatial_names = operations._safe_axis_column_names(
        moved_names[-spatial_ndim:],
        fallback=("z", "y", "x"),
    )
    leading_names = operations._safe_axis_column_names(
        moved_names[:-spatial_ndim],
        fallback=tuple(f"axis_{index}" for index in range(ndim - spatial_ndim)),
    )
    units = operations._mesh_units(
        moved_scales[-spatial_ndim:],
        moved_units[-spatial_ndim:],
        spatial_names,
        include_convex_hull_metrics=bool(include_convex_hull_metrics),
    )
    return MeshMorphologyLayout(
        input_shape=normalized_shape,
        spatial_axes=tuple(spatial_axes),
        leading_axes=leading_axes,
        permutation=permutation,
        leading_shape=tuple(normalized_shape[index] for index in leading_axes),
        spatial_shape=tuple(normalized_shape[index] for index in spatial_axes),
        leading_axis_names=tuple(leading_names),
        spatial_axis_names=tuple(spatial_names),
        units=units,
        include_convex_hull_metrics=bool(include_convex_hull_metrics),
    )


def mesh_payload_header(
    *,
    layout: MeshMorphologyLayout,
    row_count: int,
    mask_data_bytes: int,
) -> np.ndarray:
    """Build the canonical fixed header for a device payload.

    Providers may upload this small host constant before concatenating their
    device directory and mask data.  The returned value is always native
    ``uint8`` and one-dimensional.
    """

    rows = _validated_nonnegative_integer(row_count, "row_count")
    mask_bytes = _validated_nonnegative_integer(mask_data_bytes, "mask_data_bytes")
    directory_bytes = rows * layout.record_words * 8
    total_bytes = MESH_PAYLOAD_HEADER_BYTES + directory_bytes + mask_bytes
    values = np.asarray(
        (
            MESH_PAYLOAD_VERSION,
            MESH_PAYLOAD_HEADER_BYTES,
            MESH_PAYLOAD_SPATIAL_NDIM,
            layout.leading_ndim,
            rows,
            layout.record_words,
            directory_bytes,
            mask_bytes,
            total_bytes,
        ),
        dtype="<u8",
    )
    header = np.empty(MESH_PAYLOAD_HEADER_BYTES, dtype=np.uint8)
    header[: len(MESH_PAYLOAD_MAGIC)] = np.frombuffer(
        MESH_PAYLOAD_MAGIC,
        dtype=np.uint8,
    )
    header[len(MESH_PAYLOAD_MAGIC) :] = values.view(np.uint8)
    return header


def finalize_mesh_morphology_table(
    payload,
    *,
    layout: MeshMorphologyLayout,
    minimum_voxel_count: int = 16,
    source_name: str = "",
    progress=None,
) -> TableData:
    """Validate and finalize one packed mesh payload into exact ``TableData``."""

    from scipy.spatial import QhullError

    from napari_vipp.core import operations

    _finalization_check_cancelled(progress)
    minimum = max(int(minimum_voxel_count), 1)
    raw, records, mask_data = _parse_mesh_payload(payload, layout=layout)
    del raw
    _finalization_check_cancelled(progress)
    progress_total = max(len(records), 1)
    _finalization_report(
        progress,
        0,
        progress_total,
        "3D mesh morphology: starting exact CPU finalization",
    )

    columns = operations._mesh_morphology_empty_columns(
        layout.leading_axis_names,
        layout.spatial_axis_names,
        include_convex_hull_metrics=layout.include_convex_hull_metrics,
    )
    for record_index, record in enumerate(records, start=1):
        _finalization_check_cancelled(progress)
        for axis_position, axis_name in enumerate(layout.leading_axis_names):
            columns[f"{axis_name}_index"].append(
                int(record.leading_index[axis_position])
            )
        base_values = {
            "label_id": record.label_id,
            "voxel_count": record.voxel_count,
            "voxel_volume_physical": (
                float(record.voxel_count) * layout.units.voxel_volume
            ),
        }
        if record.voxel_count < minimum:
            if record.encoding != MESH_ENCODING_NONE or record.data_nbytes:
                raise ValueError(
                    "Objects below minimum_voxel_count must not carry mesh data."
                )
            operations._append_mesh_row(
                columns,
                layout.spatial_axis_names,
                include_convex_hull_metrics=layout.include_convex_hull_metrics,
                **base_values,
                mesh_status="skipped_too_few_voxels",
                mesh_error=(
                    f"voxel_count {record.voxel_count} is below minimum {minimum}"
                ),
            )
            columns["physical_unit"].append(layout.units.length_unit)
            _finalization_record_done(
                progress,
                record_index,
                progress_total,
                record.label_id,
            )
            continue

        if record.encoding == MESH_ENCODING_NONE or not record.data_nbytes:
            raise ValueError(
                "Objects meeting minimum_voxel_count must carry mesh data."
            )

        mask = _record_mask(record, mask_data)
        _finalization_check_cancelled(progress)
        try:
            metrics = operations._mesh_metrics_for_label_mask(
                mask,
                layout.spatial_axis_names,
                layout.units,
            )
        except Exception as exc:
            _finalization_check_cancelled(progress)
            operations._append_mesh_row(
                columns,
                layout.spatial_axis_names,
                include_convex_hull_metrics=layout.include_convex_hull_metrics,
                **base_values,
                mesh_status="mesh_failed",
                mesh_error=str(exc),
            )
            columns["physical_unit"].append(layout.units.length_unit)
            _finalization_record_done(
                progress,
                record_index,
                progress_total,
                record.label_id,
            )
            continue
        _finalization_check_cancelled(progress)

        if layout.include_convex_hull_metrics:
            try:
                metrics.update(
                    operations._convex_hull_metrics(metrics["mesh_vertices"])
                )
                metrics["solidity_3d"] = operations._nan_ratio(
                    metrics["mesh_volume_physical"],
                    metrics["convex_hull_volume_physical"],
                )
                metrics["surface_area_to_convex_hull_area"] = operations._nan_ratio(
                    metrics["mesh_surface_area_physical"],
                    metrics["convex_hull_surface_area_physical"],
                )
                mesh_status = "ok"
                mesh_error = ""
            except (QhullError, ValueError) as exc:
                mesh_status = "partial_convex_hull_failed"
                mesh_error = str(exc)
        else:
            mesh_status = "ok"
            mesh_error = ""
        _finalization_check_cancelled(progress)
        metrics.pop("mesh_vertices", None)
        operations._append_mesh_row(
            columns,
            layout.spatial_axis_names,
            include_convex_hull_metrics=layout.include_convex_hull_metrics,
            **base_values,
            **metrics,
            mesh_status=mesh_status,
            mesh_error=mesh_error,
        )
        columns["physical_unit"].append(layout.units.length_unit)
        _finalization_record_done(
            progress,
            record_index,
            progress_total,
            record.label_id,
        )

    if not records:
        _finalization_report(
            progress,
            1,
            progress_total,
            "3D mesh morphology: exact CPU finalization found no objects",
        )

    return table_from_columns(
        columns,
        name="3D mesh morphology measurements",
        table_kind="3D mesh morphology",
        source_name=str(source_name),
        column_units=layout.units.column_units,
    )


def finalize_mesh_morphology_outputs(host_outputs, *, call):
    """Finalize the one-output resident mesh provider ABI after device cleanup."""

    payloads = tuple(host_outputs)
    if len(payloads) != 1:
        raise ValueError(
            "3D mesh morphology host finalization requires exactly one packed "
            f"payload; received {len(payloads)}."
        )
    operation_id = str(getattr(call, "operation_id", "")).strip()
    if operation_id != "measure_3d_mesh_morphology":
        raise ValueError(
            "3D mesh morphology host finalization cannot handle operation "
            f"{operation_id!r}."
        )
    kwargs = dict(getattr(call, "kwargs", {}))
    layout = mesh_morphology_layout(
        _prepared_mesh_input_shape(call),
        spatial_mode=kwargs.get("spatial_mode", "Auto from axes"),
        resolved_spatial_ndim=kwargs.get("resolved_spatial_ndim"),
        axis_names=kwargs.get("axis_names"),
        axis_types=kwargs.get("axis_types"),
        axis_scales=kwargs.get("axis_scales"),
        axis_units=kwargs.get("axis_units"),
        include_convex_hull_metrics=kwargs.get(
            "include_convex_hull_metrics",
            True,
        ),
    )
    return finalize_mesh_morphology_table(
        payloads[0],
        layout=layout,
        minimum_voxel_count=kwargs.get("minimum_voxel_count", 16),
        source_name=str(kwargs.get("source_name", "")),
        progress=kwargs.get("progress"),
    )


def _finalization_check_cancelled(progress) -> None:
    if progress is not None:
        progress.check_cancelled()


def _finalization_report(
    progress,
    current: int,
    total: int,
    message: str,
) -> None:
    if progress is None:
        return
    progress.report(int(current), int(total), str(message))
    progress.check_cancelled()


def _finalization_record_done(
    progress,
    current: int,
    total: int,
    label_id: int,
) -> None:
    _finalization_report(
        progress,
        current,
        total,
        f"3D mesh morphology: exact CPU finalization for label {int(label_id)}",
    )


def _parse_mesh_payload(
    payload,
    *,
    layout: MeshMorphologyLayout,
) -> tuple[np.ndarray, tuple[_MeshPayloadRecord, ...], np.ndarray]:
    raw = np.asarray(payload)
    if raw.dtype != np.dtype(np.uint8) or not raw.dtype.isnative:
        raise TypeError(
            "Packed mesh morphology payloads must use native uint8 storage; "
            f"received {raw.dtype}."
        )
    if raw.ndim != 1:
        raise ValueError(
            "Packed mesh morphology payloads must be one-dimensional; "
            f"received shape {raw.shape}."
        )
    raw = np.ascontiguousarray(raw)
    if raw.size < MESH_PAYLOAD_HEADER_BYTES:
        raise ValueError("Packed mesh morphology payload has a truncated header.")
    if raw[: len(MESH_PAYLOAD_MAGIC)].tobytes() != MESH_PAYLOAD_MAGIC:
        raise ValueError("Packed mesh morphology payload has an invalid magic value.")
    header = np.frombuffer(
        raw[len(MESH_PAYLOAD_MAGIC) : MESH_PAYLOAD_HEADER_BYTES],
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
        mask_data_bytes,
        total_bytes,
    ) = (int(value) for value in header)
    if version != MESH_PAYLOAD_VERSION:
        raise ValueError(f"Unsupported mesh morphology payload version {version}.")
    if header_bytes != MESH_PAYLOAD_HEADER_BYTES:
        raise ValueError("Packed mesh morphology payload has an invalid header size.")
    if spatial_ndim != MESH_PAYLOAD_SPATIAL_NDIM:
        raise ValueError("Packed mesh morphology payload is not a 3D payload.")
    if leading_ndim != layout.leading_ndim:
        raise ValueError(
            "Packed mesh morphology payload leading rank does not match the call."
        )
    if record_words != layout.record_words:
        raise ValueError(
            "Packed mesh morphology payload record width does not match the call."
        )
    expected_directory_bytes = row_count * record_words * 8
    expected_total_bytes = (
        MESH_PAYLOAD_HEADER_BYTES + expected_directory_bytes + mask_data_bytes
    )
    if (
        directory_bytes != expected_directory_bytes
        or total_bytes != expected_total_bytes
    ):
        raise ValueError("Packed mesh morphology payload sizes are inconsistent.")
    if total_bytes != raw.size:
        raise ValueError(
            "Packed mesh morphology payload byte count does not match its header."
        )

    directory_end = MESH_PAYLOAD_HEADER_BYTES + directory_bytes
    directory = np.frombuffer(
        raw[MESH_PAYLOAD_HEADER_BYTES:directory_end],
        dtype="<u8",
        count=row_count * record_words,
    ).reshape(row_count, record_words)
    mask_data = raw[directory_end:]
    records = _validated_records(directory, mask_data, layout=layout)
    return raw, records, mask_data


def _validated_records(
    directory: np.ndarray,
    mask_data: np.ndarray,
    *,
    layout: MeshMorphologyLayout,
) -> tuple[_MeshPayloadRecord, ...]:
    records: list[_MeshPayloadRecord] = []
    next_data_offset = 0
    previous_key: tuple[tuple[int, ...], int] | None = None
    lead = layout.leading_ndim
    for row in directory:
        leading_index = tuple(int(value) for value in row[:lead])
        cursor = lead
        label_id = int(row[cursor])
        voxel_count = int(row[cursor + 1])
        minimum = tuple(int(value) for value in row[cursor + 2 : cursor + 5])
        maximum = tuple(int(value) for value in row[cursor + 5 : cursor + 8])
        encoding = int(row[cursor + 8])
        data_offset = int(row[cursor + 9])
        data_nbytes = int(row[cursor + 10])
        data_count = int(row[cursor + 11])

        if any(
            value < 0 or value >= layout.leading_shape[index]
            for index, value in enumerate(leading_index)
        ):
            raise ValueError("Packed mesh morphology leading index is out of bounds.")
        if label_id <= 0 or label_id > np.iinfo(np.int32).max:
            raise ValueError("Packed mesh morphology label IDs must be positive int32.")
        if voxel_count <= 0:
            raise ValueError("Packed mesh morphology voxel counts must be positive.")
        if any(
            lo < 0 or hi <= lo or hi > layout.spatial_shape[index]
            for index, (lo, hi) in enumerate(zip(minimum, maximum, strict=True))
        ):
            raise ValueError("Packed mesh morphology bounding boxes are invalid.")
        bbox_bits = int(prod(hi - lo for lo, hi in zip(minimum, maximum, strict=True)))
        if voxel_count > bbox_bits:
            raise ValueError(
                "Packed mesh morphology voxel count exceeds its bounding box."
            )
        if encoding == MESH_ENCODING_NONE:
            if data_nbytes != 0 or data_count != 0:
                raise ValueError(
                    "Data-free mesh records must have zero data size and count."
                )
            if data_offset != next_data_offset:
                raise ValueError(
                    "Packed mesh morphology data offsets are not contiguous."
                )
        elif encoding == MESH_ENCODING_BITMASK:
            minimum_bytes = (bbox_bits + 7) // 8
            expected_bytes = _aligned_bytes(minimum_bytes, alignment=4)
            if data_count != bbox_bits:
                raise ValueError(
                    "Packed mesh morphology bit count does not match its bounding box."
                )
            if data_nbytes != expected_bytes:
                raise ValueError(
                    "Packed mesh morphology bitmask byte count is invalid."
                )
            if data_nbytes > voxel_count * 4:
                raise ValueError(
                    "Packed mesh morphology bitmasks must obey the linear "
                    "four-bytes-per-voxel bound."
                )
            if data_offset != next_data_offset:
                raise ValueError(
                    "Packed mesh morphology data offsets are not contiguous."
                )
            if data_offset + data_nbytes > mask_data.size:
                raise ValueError(
                    "Packed mesh morphology data extends beyond the payload."
                )
            packed = mask_data[data_offset : data_offset + data_nbytes]
            significant_remainder = bbox_bits % 8
            if significant_remainder and int(packed[minimum_bytes - 1]) >> (
                significant_remainder
            ):
                raise ValueError(
                    "Packed mesh morphology bitmask has nonzero padding bits."
                )
            if np.any(packed[minimum_bytes:]):
                raise ValueError(
                    "Packed mesh morphology bitmask has nonzero alignment padding."
                )
            next_data_offset += data_nbytes
        elif encoding == MESH_ENCODING_SPARSE_UINT32:
            if data_count != voxel_count or data_nbytes != voxel_count * 4:
                raise ValueError(
                    "Packed sparse mesh data must contain one uint32 per voxel."
                )
            if data_offset % 4 or data_offset != next_data_offset:
                raise ValueError(
                    "Packed sparse mesh data must be aligned and contiguous."
                )
            if data_offset + data_nbytes > mask_data.size:
                raise ValueError(
                    "Packed mesh morphology data extends beyond the payload."
                )
            next_data_offset += data_nbytes
        else:
            raise ValueError(
                f"Packed mesh morphology record has unknown encoding {encoding}."
            )

        key = (leading_index, label_id)
        if previous_key is not None and key <= previous_key:
            raise ValueError(
                "Packed mesh morphology records must use block-major, "
                "sorted-label order."
            )
        previous_key = key
        records.append(
            _MeshPayloadRecord(
                leading_index=leading_index,
                label_id=label_id,
                voxel_count=voxel_count,
                bbox_minimum=minimum,
                bbox_maximum=maximum,
                encoding=encoding,
                data_offset=data_offset,
                data_nbytes=data_nbytes,
                data_count=data_count,
            )
        )
    if next_data_offset != mask_data.size:
        raise ValueError("Packed mesh morphology payload has unreferenced data bytes.")
    return tuple(records)


def _record_mask(record: _MeshPayloadRecord, mask_data: np.ndarray) -> np.ndarray:
    encoded = mask_data[record.data_offset : record.data_offset + record.data_nbytes]
    shape = tuple(
        hi - lo
        for lo, hi in zip(
            record.bbox_minimum,
            record.bbox_maximum,
            strict=True,
        )
    )
    if record.encoding == MESH_ENCODING_BITMASK:
        unpacked = np.unpackbits(
            encoded,
            bitorder="little",
            count=record.data_count,
        )
        if int(np.count_nonzero(unpacked)) != record.voxel_count:
            raise ValueError(
                "Packed mesh morphology bitmask does not match its voxel count."
            )
        return unpacked.reshape(shape).astype(bool, copy=False)
    if record.encoding == MESH_ENCODING_SPARSE_UINT32:
        indexes = np.frombuffer(encoded, dtype="<u4", count=record.data_count)
        bbox_size = int(prod(shape))
        if np.any(indexes >= bbox_size):
            raise ValueError(
                "Packed sparse mesh indices must be inside the bounding box."
            )
        mask = np.zeros(bbox_size, dtype=bool)
        mask[indexes] = True
        if int(np.count_nonzero(mask)) != record.voxel_count:
            raise ValueError("Packed sparse mesh indices must be unique.")
        return mask.reshape(shape)
    raise ValueError("A mesh record without encoded data cannot be finalized.")


def _aligned_bytes(size: int, *, alignment: int) -> int:
    return ((int(size) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _prepared_mesh_input_shape(call) -> tuple[int, ...]:
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
        "3D mesh morphology host finalization requires the original label "
        "shape in call.input_states[0].shape."
    )


def _validated_shape(shape: Sequence[int]) -> tuple[int, ...]:
    result: list[int] = []
    for value in tuple(shape):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise ValueError("Mesh morphology shapes require non-negative integers.")
        size = int(value)
        if size < 0:
            raise ValueError("Mesh morphology shapes require non-negative integers.")
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
    "MESH_PAYLOAD_HEADER_BYTES",
    "MESH_PAYLOAD_MAGIC",
    "MESH_PAYLOAD_SPATIAL_NDIM",
    "MESH_PAYLOAD_VERSION",
    "MESH_ENCODING_BITMASK",
    "MESH_ENCODING_NONE",
    "MESH_ENCODING_SPARSE_UINT32",
    "MESH_RECORD_BASE_WORDS",
    "MeshMorphologyLayout",
    "finalize_mesh_morphology_outputs",
    "finalize_mesh_morphology_table",
    "mesh_morphology_layout",
    "mesh_payload_header",
]
