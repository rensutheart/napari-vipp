from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.skeleton_measurements import (
    SKELETON_PAYLOAD_HEADER_BYTES,
    finalize_analyze_skeleton_outputs,
    finalize_analyze_skeleton_table,
    skeleton_analysis_layout,
    skeleton_payload_header,
)


def _float_word(value: float) -> int:
    return int(np.asarray([value], dtype="<f8").view("<u8")[0])


def _payload(layout, rows: list[list[int]]) -> np.ndarray:
    directory = np.asarray(rows, dtype="<u8").reshape(-1, layout.record_words)
    return np.concatenate(
        (
            skeleton_payload_header(layout=layout, row_count=len(rows)),
            directory.reshape(-1).view(np.uint8),
        )
    )


def _layout():
    return skeleton_analysis_layout(
        (2, 5, 6),
        spatial_mode="2D YX",
        axis_names=("t", "y", "x"),
        axis_types=("time", "space", "space"),
        axis_scales=(1.0, 2.0, 0.5),
        axis_units=(None, "um", "um"),
    )


def _records() -> list[list[int]]:
    # leading index; component id/count; total/component voxels; endpoint,
    # junction, isolated; branch, graph nodes/edges, voxel edges, cycles;
    # bit-preserving float64 pixel/physical lengths.
    return [
        [0, 1, 2, 3, 1, 0, 0, 1, 0, 1, 0, 0, 0, _float_word(0.0), _float_word(0.0)],
        [0, 2, 2, 3, 2, 2, 0, 0, 1, 2, 1, 1, 0, _float_word(1.0), _float_word(0.5)],
    ]


def test_host_finalizer_builds_exact_public_schema_and_metadata() -> None:
    layout = _layout()
    table = finalize_analyze_skeleton_table(
        _payload(layout, _records()),
        layout=layout,
        source_name="host-fixture",
    )

    assert table.name == "Skeleton network measurements"
    assert table.table_kind == "Skeleton network"
    assert table.source_name == "host-fixture"
    assert table.columns == (
        "t_index",
        "component_id",
        "component_count_in_block",
        "component_voxel_fraction",
        "skeleton_voxel_count",
        "endpoint_voxel_count",
        "junction_voxel_count",
        "isolated_node_count",
        "branch_count",
        "graph_node_count",
        "graph_edge_count",
        "voxel_graph_edge_count",
        "cycle_count",
        "skeleton_length_pixels",
        "skeleton_length_physical",
        "physical_unit",
    )
    assert table.rows == (
        (0, 1, 2, 1.0 / 3.0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0.0, 0.0, "um"),
        (0, 2, 2, 2.0 / 3.0, 2, 2, 0, 0, 1, 2, 1, 1, 0, 1.0, 0.5, "um"),
    )
    assert table.unit_for("skeleton_length_pixels") == "pixels"
    assert table.unit_for("skeleton_length_physical") == "um"


def test_call_finalizer_rejects_skeletonization_and_uses_input_state_shape() -> None:
    layout = _layout()
    packed = _payload(layout, _records())
    state = SimpleNamespace(shape=(2, 5, 6))
    call = SimpleNamespace(
        operation_id="analyze_skeleton",
        input_states=(state,),
        inputs=(),
        kwargs={
            "spatial_mode": "2D YX",
            "input_mode": "Already skeletonized",
            "axis_names": ("t", "y", "x"),
            "axis_types": ("time", "space", "space"),
            "axis_scales": (1.0, 2.0, 0.5),
            "axis_units": (None, "um", "um"),
            "source_name": "call-fixture",
        },
    )

    table = finalize_analyze_skeleton_outputs((packed,), call=call)

    assert table.source_name == "call-fixture"
    call.kwargs["input_mode"] = "Skeletonize first"
    with pytest.raises(ValueError, match="Already skeletonized"):
        finalize_analyze_skeleton_outputs((packed,), call=call)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda payload, _layout: payload.__setitem__(0, 0), "magic"),
        (
            lambda payload, _layout: payload.__setitem__(
                SKELETON_PAYLOAD_HEADER_BYTES + 8,
                3,
            ),
            "component IDs/counts|component IDs are not dense",
        ),
        (
            lambda payload, layout: payload.__setitem__(
                SKELETON_PAYLOAD_HEADER_BYTES + (layout.record_words + 3) * 8,
                4,
            ),
            "summaries disagree|voxel total",
        ),
    ),
)
def test_host_finalizer_rejects_corrupt_payloads(mutation, message: str) -> None:
    layout = _layout()
    packed = _payload(layout, _records())

    mutation(packed, layout)

    with pytest.raises(ValueError, match=message):
        finalize_analyze_skeleton_table(packed, layout=layout)
