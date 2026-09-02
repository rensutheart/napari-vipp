from __future__ import annotations

from dataclasses import replace
from math import prod
from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.operations as operations_module
from napari_vipp.core.measurements import measurement_table_parity
from napari_vipp.core.mesh_measurements import (
    MESH_ENCODING_BITMASK,
    MESH_ENCODING_NONE,
    MESH_ENCODING_SPARSE_UINT32,
    MESH_PAYLOAD_HEADER_BYTES,
    finalize_mesh_morphology_outputs,
    finalize_mesh_morphology_table,
    mesh_morphology_layout,
    mesh_payload_header,
)
from napari_vipp.core.node_execution import PreparedNodeCall
from napari_vipp.core.operations import measure_3d_mesh_morphology


class _Progress:
    def __init__(self) -> None:
        self.cancelled = False
        self.reports: list[tuple[int, int, str]] = []

    def report(self, current: int, total: int, message: str) -> None:
        self.reports.append((int(current), int(total), str(message)))

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise RuntimeError("cancelled during host finalization")


def _labels() -> np.ndarray:
    labels = np.zeros((14, 2, 16, 18), dtype=np.int32)
    labels[1:9, 0, 2:11, 3:13] = 17
    labels[3:7, 0, 5:8, 6:9] = 0
    labels[10:13, 0, 12:15, 14:17] = 9001
    labels[2:11, 1, 3:13, 4:15] = np.iinfo(np.int32).max
    labels[12:14, 1, 1:3, 1:4] = 44
    return labels


def _kwargs(*, include_hull: bool = True) -> dict[str, object]:
    return {
        "spatial_mode": "3D ZYX",
        "axis_names": ("z", "t", "y", "x"),
        "axis_types": ("space", "time", "space", "space"),
        "axis_scales": (2.0, 7.0, 0.5, 0.25),
        "axis_units": ("um", None, "um", "um"),
        "minimum_voxel_count": 16,
        "include_convex_hull_metrics": include_hull,
        "source_name": "packed-mesh-fixture",
    }


def _host_payload(
    labels: np.ndarray,
    *,
    kwargs: dict[str, object],
) -> tuple[np.ndarray, object]:
    layout = mesh_morphology_layout(
        labels.shape,
        **{
            name: value
            for name, value in kwargs.items()
            if name not in {"minimum_voxel_count", "source_name"}
        },
    )
    minimum = max(int(kwargs.get("minimum_voxel_count", 16)), 1)
    moved = np.ascontiguousarray(np.transpose(labels, layout.permutation))
    records: list[list[int]] = []
    masks: list[np.ndarray] = []
    mask_offset = 0
    indexes = np.ndindex(layout.leading_shape) if layout.leading_shape else ((),)
    for leading_index in indexes:
        block = moved[leading_index] if layout.leading_shape else moved
        for label_id in (int(value) for value in np.unique(block) if int(value) > 0):
            coordinates = np.argwhere(block == label_id)
            voxel_count = int(coordinates.shape[0])
            minimums = coordinates.min(axis=0)
            maximums = coordinates.max(axis=0) + 1
            slices = tuple(
                slice(int(low), int(high))
                for low, high in zip(minimums, maximums, strict=True)
            )
            if voxel_count < minimum:
                encoded = np.empty((0,), dtype=np.uint8)
                encoding = MESH_ENCODING_NONE
                data_count = 0
            else:
                local = block[slices] == label_id
                minimum_bitmask_bytes = (int(local.size) + 7) // 8
                aligned_bitmask_bytes = ((minimum_bitmask_bytes + 3) // 4) * 4
                if aligned_bitmask_bytes <= voxel_count * 4:
                    encoding = MESH_ENCODING_BITMASK
                    encoded = np.zeros(aligned_bitmask_bytes, dtype=np.uint8)
                    packed = np.packbits(local.reshape(-1), bitorder="little")
                    encoded[: packed.size] = packed
                    data_count = int(local.size)
                else:
                    encoding = MESH_ENCODING_SPARSE_UINT32
                    indexes = np.flatnonzero(local).astype("<u4", copy=False)
                    encoded = indexes.view(np.uint8)
                    data_count = voxel_count
            records.append(
                [
                    *leading_index,
                    label_id,
                    voxel_count,
                    *(int(value) for value in minimums),
                    *(int(value) for value in maximums),
                    encoding,
                    mask_offset,
                    int(encoded.size),
                    data_count,
                ]
            )
            masks.append(encoded)
            mask_offset += int(encoded.size)
    directory = np.asarray(records, dtype="<u8").reshape(-1, layout.record_words)
    mask_data = (
        np.concatenate(masks).astype(np.uint8, copy=False)
        if masks
        else np.empty((0,), dtype=np.uint8)
    )
    header = mesh_payload_header(
        layout=layout,
        row_count=len(records),
        mask_data_bytes=int(mask_data.size),
    )
    payload = np.concatenate((header, directory.view(np.uint8).reshape(-1), mask_data))
    return payload, layout


@pytest.mark.parametrize("include_hull", (False, True))
def test_packed_finalizer_matches_cpu_for_leading_blocks_sparse_ids_and_units(
    include_hull: bool,
) -> None:
    labels = _labels()
    kwargs = _kwargs(include_hull=include_hull)
    payload, layout = _host_payload(labels, kwargs=kwargs)

    actual = finalize_mesh_morphology_table(
        payload,
        layout=layout,
        minimum_voxel_count=int(kwargs["minimum_voxel_count"]),
        source_name=str(kwargs["source_name"]),
    )
    expected = measure_3d_mesh_morphology(labels, **kwargs)

    parity = measurement_table_parity(expected, actual, exact_float_columns=True)
    assert parity.passed, parity.detail
    assert layout.permutation == (1, 0, 2, 3)
    assert layout.leading_axis_names == ("t",)
    assert [row[0:2] for row in actual.rows] == [
        (0, 17),
        (0, 9001),
        (1, 44),
        (1, np.iinfo(np.int32).max),
    ]
    hull_columns = {
        "convex_hull_volume_physical",
        "convex_hull_surface_area_physical",
        "solidity_3d",
        "surface_area_to_convex_hull_area",
    }
    assert bool(hull_columns.intersection(actual.columns)) is include_hull


def test_skipped_records_have_no_mask_and_threshold_equality_keeps_a_mask() -> None:
    labels = np.zeros((6, 6, 6), dtype=np.int32)
    labels[0, 0, 0:3] = 5
    labels[2:4, 2:4, 2:4] = 8
    kwargs = {
        "spatial_mode": "3D ZYX",
        "minimum_voxel_count": 8,
        "include_convex_hull_metrics": False,
    }
    payload, layout = _host_payload(labels, kwargs=kwargs)
    directory = np.frombuffer(
        payload[MESH_PAYLOAD_HEADER_BYTES : MESH_PAYLOAD_HEADER_BYTES + 2 * 12 * 8],
        dtype="<u8",
    ).reshape(2, 12)

    assert tuple(int(value) for value in directory[:, -4]) == (
        MESH_ENCODING_NONE,
        MESH_ENCODING_BITMASK,
    )
    assert tuple(int(value) for value in directory[:, -2]) == (0, 4)
    assert tuple(int(value) for value in directory[:, -1]) == (0, 8)
    actual = finalize_mesh_morphology_table(
        payload,
        layout=layout,
        minimum_voxel_count=8,
    )
    expected = measure_3d_mesh_morphology(labels, **kwargs)
    parity = measurement_table_parity(expected, actual, exact_float_columns=True)
    assert parity.passed, parity.detail


def test_empty_payload_finalizes_to_the_exact_empty_cpu_schema() -> None:
    labels = np.zeros((0, 7, 9), dtype=np.int32)
    kwargs = {
        "spatial_mode": "3D ZYX",
        "include_convex_hull_metrics": True,
    }
    payload, layout = _host_payload(labels, kwargs=kwargs)
    actual = finalize_mesh_morphology_table(payload, layout=layout)
    expected = measure_3d_mesh_morphology(labels, **kwargs)

    assert actual == expected
    assert payload.size == MESH_PAYLOAD_HEADER_BYTES


def test_host_finalizer_recovers_shape_after_inputs_are_sanitized() -> None:
    labels = _labels()
    kwargs = _kwargs()
    payload, _layout = _host_payload(labels, kwargs=kwargs)
    call = PreparedNodeCall(
        node_id="mesh-node",
        operation_id="measure_3d_mesh_morphology",
        cpu_function=lambda *_args, **_kwargs: None,
        inputs=(labels,),
        input_states=(SimpleNamespace(shape=labels.shape),),
        kwargs=kwargs,
    )
    call = replace(call, inputs=(None,))

    actual = finalize_mesh_morphology_outputs((payload,), call=call)
    expected = measure_3d_mesh_morphology(labels, **kwargs)

    parity = measurement_table_parity(expected, actual, exact_float_columns=True)
    assert parity.passed, parity.detail


def test_host_finalizer_reports_each_exact_cpu_record_from_call_progress() -> None:
    labels = _labels()
    kwargs = _kwargs(include_hull=False)
    payload, _layout = _host_payload(labels, kwargs=kwargs)
    progress = _Progress()
    call = PreparedNodeCall(
        node_id="mesh-node",
        operation_id="measure_3d_mesh_morphology",
        cpu_function=lambda *_args, **_kwargs: None,
        inputs=(None,),
        input_states=(SimpleNamespace(shape=labels.shape),),
        kwargs={**kwargs, "progress": progress},
    )

    actual = finalize_mesh_morphology_outputs((payload,), call=call)

    assert actual.row_count == 4
    assert [(current, total) for current, total, _message in progress.reports] == [
        (current, 4) for current in range(5)
    ]
    assert "starting exact CPU finalization" in progress.reports[0][2]
    assert all(
        "exact CPU finalization for label" in message
        for _current, _total, message in progress.reports[1:]
    )


def test_host_finalizer_checks_cancellation_after_exact_mesh_geometry(
    monkeypatch,
) -> None:
    labels = np.zeros((7, 8, 9), dtype=np.int32)
    labels[1:6, 1:7, 1:8] = 17
    kwargs = {
        "spatial_mode": "3D ZYX",
        "minimum_voxel_count": 1,
        "include_convex_hull_metrics": False,
    }
    payload, _layout = _host_payload(labels, kwargs=kwargs)
    progress = _Progress()
    original = operations_module._mesh_metrics_for_label_mask

    def cancel_after_geometry(*args, **inner_kwargs):
        result = original(*args, **inner_kwargs)
        progress.cancelled = True
        return result

    monkeypatch.setattr(
        operations_module,
        "_mesh_metrics_for_label_mask",
        cancel_after_geometry,
    )
    call = PreparedNodeCall(
        node_id="mesh-node",
        operation_id="measure_3d_mesh_morphology",
        cpu_function=lambda *_args, **_kwargs: None,
        inputs=(None,),
        input_states=(SimpleNamespace(shape=labels.shape),),
        kwargs={**kwargs, "progress": progress},
    )

    with pytest.raises(RuntimeError, match="cancelled during host finalization"):
        finalize_mesh_morphology_outputs((payload,), call=call)

    assert [(current, total) for current, total, _message in progress.reports] == [
        (0, 1)
    ]


def test_payload_validation_rejects_wrong_type_header_sizes_and_mask_semantics() -> (
    None
):
    labels = np.zeros((5, 6, 7), dtype=np.int32)
    labels[1:4, 1:5, 1:6] = 7
    kwargs = {
        "spatial_mode": "3D ZYX",
        "minimum_voxel_count": 1,
        "include_convex_hull_metrics": False,
    }
    payload, layout = _host_payload(labels, kwargs=kwargs)

    with pytest.raises(TypeError, match="uint8"):
        finalize_mesh_morphology_table(payload.astype(np.uint16), layout=layout)
    with pytest.raises(ValueError, match="one-dimensional"):
        finalize_mesh_morphology_table(payload[None, :], layout=layout)
    with pytest.raises(ValueError, match="truncated header"):
        finalize_mesh_morphology_table(payload[:10], layout=layout)
    malformed = payload.copy()
    malformed[0] ^= np.uint8(1)
    with pytest.raises(ValueError, match="magic"):
        finalize_mesh_morphology_table(malformed, layout=layout)
    with pytest.raises(ValueError, match="byte count"):
        finalize_mesh_morphology_table(payload[:-1], layout=layout)

    # Voxel count is directory word 1 for a layout with no leading axes.
    malformed = payload.copy()
    directory = np.frombuffer(
        malformed[MESH_PAYLOAD_HEADER_BYTES : MESH_PAYLOAD_HEADER_BYTES + 12 * 8],
        dtype="<u8",
    )
    directory[1] = np.uint64(prod(labels.shape))
    with pytest.raises(ValueError, match="voxel count"):
        finalize_mesh_morphology_table(malformed, layout=layout)


def test_finalizer_rejects_a_threshold_that_disagrees_with_mask_presence() -> None:
    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 4
    packed_as_skipped, layout = _host_payload(
        labels,
        kwargs={"spatial_mode": "3D ZYX", "minimum_voxel_count": 9},
    )
    with pytest.raises(ValueError, match="must carry mesh data"):
        finalize_mesh_morphology_table(
            packed_as_skipped,
            layout=layout,
            minimum_voxel_count=8,
        )

    packed_as_eligible, layout = _host_payload(
        labels,
        kwargs={"spatial_mode": "3D ZYX", "minimum_voxel_count": 8},
    )
    with pytest.raises(ValueError, match="must not carry mesh data"):
        finalize_mesh_morphology_table(
            packed_as_eligible,
            layout=layout,
            minimum_voxel_count=9,
        )
