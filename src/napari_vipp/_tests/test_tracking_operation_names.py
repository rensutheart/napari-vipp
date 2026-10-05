"""Consistent canonical identities for the unreleased tracking operations."""

from copy import deepcopy

import numpy as np
import pytest

from napari_vipp.core import (
    time_detection,
    time_detection_nodes,
    tracking,
    tracking_nodes,
)
from napari_vipp.core.compute_specs import compute_specs_for, cpu_implementation_id
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.graph_fragments import (
    GraphFragmentError,
    GraphFragmentNode,
    prepare_paste_values,
)
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import (
    NODE_LIBRARY,
    NODE_LIBRARY_BY_ID,
    PALETTE_NODE_LIBRARY,
    PrototypePipeline,
)
from napari_vipp.core.snapshots import GraphSnapshot, NodeSnapshot
from napari_vipp.core.workflow import (
    deserialize_workflow,
    serialize_workflow,
    workflow_document_from_snapshot,
    workflow_snapshot_from_document,
)

RENAMES = (
    ("detect_over_time", "detect_spots_per_frame", "Detect Spots per Frame"),
    ("link_tracks", "build_tracks", "Build Tracks"),
)


def _graph():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    detector = pipeline.add_node("detect_spots_per_frame")
    tracks = pipeline.add_node("build_tracks")
    summary = pipeline.add_node("select_table_columns")
    assert pipeline.connect("input", detector.id).success
    assert pipeline.connect(detector.id, tracks.id).success
    assert pipeline.connect(tracks.id, summary.id, source_port=1).success
    pipeline.set_param(tracks.id, "maximum_gap", 0)
    return pipeline, detector, tracks


def _document():
    pipeline, detector, tracks = _graph()
    names = {detector.id: "My spot candidates", tracks.id: "My tracked objects"}
    return serialize_workflow(
        pipeline,
        positions={detector.id: (13.0, 27.0), tracks.id: (413.0, 27.0)},
        notes=[{
            "id": "review", "text": "Review my custom tracks", "width": 300.0,
            "position": (413.0, -100.0), "attached_node": tracks.id,
        }],
        metadata={"vipp": {"node_names": names}},
    )


@pytest.mark.parametrize("former,canonical,label", RENAMES)
def test_registry_and_compute_identity_have_only_canonical_nodes(
    former, canonical, label,
):
    spec = NODE_LIBRARY_BY_ID[canonical]
    assert spec.title == label
    assert spec.function.__name__ == f"{canonical}_node"
    assert sum(item.id == canonical for item in NODE_LIBRARY) == 1
    assert sum(item.id == canonical for item in PALETTE_NODE_LIBRARY) == 1
    assert former not in NODE_LIBRARY_BY_ID
    assert not any(item.id == former for item in NODE_LIBRARY)
    compute = compute_specs_for(canonical)[0]
    assert compute.operation_id == canonical
    assert compute.callable_ref.endswith(f":{canonical}_node")
    assert compute.implementation_id == f"cpu-{canonical}-v1"
    assert compute.implementation_id != f"cpu-{former}-v1"
    assert cpu_implementation_id(canonical) == compute.implementation_id
    with pytest.raises(KeyError, match="Unknown operation"):
        compute_specs_for(former)


@pytest.mark.parametrize("former,canonical,label", RENAMES)
def test_node_authoring_uses_canonical_name_and_counter(former, canonical, label):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    first = pipeline.add_node(canonical)
    second = pipeline.add_node(canonical)
    assert first.operation_id == second.operation_id == canonical
    assert first.title == second.title == label
    assert first.id == f"{canonical}_1"
    assert second.id == f"{canonical}_2"
    with pytest.raises(KeyError):
        pipeline.add_node(former)


def test_workflow_round_trip_preserves_authored_state_with_canonical_operations():
    document = _document()
    original = deepcopy(document)
    snapshot = workflow_snapshot_from_document(document)
    emitted = workflow_document_from_snapshot(snapshot)
    assert document == original
    assert emitted == document


@pytest.mark.parametrize("former,canonical,label", RENAMES)
def test_former_unreleased_operation_ids_are_not_persisted_aliases(
    former, canonical, label,
):
    document = _document()
    node = next(n for n in document["nodes"] if n["operation_id"] == canonical)
    node["operation_id"] = former
    with pytest.raises(ValueError, match="unknown operation"):
        deserialize_workflow(document)


@pytest.mark.parametrize("former,canonical,label", RENAMES)
def test_canonical_snapshots_and_clipboard_nodes_preserve_user_names(
    former, canonical, label,
):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(canonical)
    snapshot = NodeSnapshot(node.id, canonical, node.params)
    restored = GraphSnapshot((snapshot,)).to_pipeline().nodes[node.id]
    assert restored.operation_id == canonical
    assert restored.title == label
    assert restored.params == node.params
    fragment = GraphFragmentNode("n0", canonical, node.params, custom_name="My node")
    assert fragment.operation_id == canonical
    assert fragment.custom_name == "My node"
    assert prepare_paste_values(fragment, canonical) == node.params
    with pytest.raises(GraphFragmentError, match="Unknown operation"):
        GraphFragmentNode("n0", former, node.params)


def test_python_export_emits_only_canonical_operation_ids():
    pipeline, _detector, _tracks = _graph()
    code = export_pipeline_to_python(pipeline)
    assert "detect_spots_per_frame" in code
    assert "build_tracks" in code
    assert "detect_over_time" not in code
    assert "link_tracks" not in code
    compile(code, "tracking-canonical-export.py", "exec")


def test_canonical_domain_and_adapter_names_preserve_known_answer_behavior():
    assert time_detection.detect_spots_per_frame.__name__ == "detect_spots_per_frame"
    assert tracking.build_tracks.__name__ == "build_tracks"
    assert time_detection_nodes.detect_spots_per_frame_node.__name__ == (
        "detect_spots_per_frame_node"
    )
    assert tracking_nodes.build_tracks_node.__name__ == "build_tracks_node"
    assert not hasattr(time_detection, "detect_over_time")
    assert not hasattr(tracking, "link_tracks")
    assert not hasattr(time_detection_nodes, "detect_over_time_node")
    assert not hasattr(tracking_nodes, "link_tracks_node")
    data = np.zeros((3, 7, 9), dtype=np.float32)
    data[0, 3, 2] = data[2, 3, 4] = 10
    data.setflags(write=False)
    before = data.copy()
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("t", "time"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    observations = time_detection.detect_spots_per_frame(data, image_state=state)
    assert observations.rows == ((0, 1, 3, 2, 10.0), (2, 1, 3, 4, 10.0))
    assert observations.observation_metadata.frame_populations[1].retained_count == 0
    tracked, summary = tracking.build_tracks(observations, maximum_gap=0)
    assert tracked.row_count == summary.row_count == 2
    tracked, summary = tracking.build_tracks(observations, maximum_gap=1)
    assert tracked.row_count == 2 and summary.row_count == 1
    assert np.array_equal(data, before)
