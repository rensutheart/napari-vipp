from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from napari_vipp.core.batch import scientific_workflow_hash
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.pipeline import (
    GraphConnection,
    OutputTunnel,
    PrototypePipeline,
)
from napari_vipp.core.snapshots import (
    GraphSnapshot,
    NodeSnapshot,
    WorkflowNoteSnapshot,
    WorkflowSnapshot,
)
from napari_vipp.core.workflow import (
    deserialize_workflow,
    serialize_workflow,
    workflow_document_from_snapshot,
    workflow_snapshot_from_document,
    workflow_snapshot_from_pipeline,
)

_EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"
_EXAMPLE_FILENAMES = tuple(
    path.name for path in sorted(_EXAMPLE_DIR.glob("*.json"))
)


def _node(snapshot: GraphSnapshot, node_id: str) -> NodeSnapshot:
    return next(node for node in snapshot.nodes if node.id == node_id)


def _established_document_round_trip(document: dict[str, Any]) -> dict[str, Any]:
    restored = deserialize_workflow(document)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        restored["nodes"],
        restored["connections"],
        restored["output_tunnels"],
    )
    return serialize_workflow(
        pipeline,
        positions=restored["positions"],
        notes=restored["notes"],
        metadata=restored["metadata"],
        compute_request=restored["compute_request"],
    )


def _expected_saved_metadata(document: dict[str, Any]) -> dict[str, Any] | None:
    """Explicit schema-1 display defaults, independent of production loading."""
    metadata = deepcopy(document.get("metadata"))
    if metadata is None:
        return None
    reviews = (
        metadata.get("vipp", {}).get("inspector", {}).get("image_reviews", {})
    )
    for identifier, raw in reviews.items():
        recipe = {
            "version": 1,
            "mode": "side-by-side",
            "left": "a",
            "right": "overlay",
            "ndisplay": 2,
            "orientation": "oblique",
            "show_axes": True,
            "show_scale_bar": True,
            "link_navigation": True,
            "link_contrast": False,
        }
        for key, colormap, opacity in (("a", "gray", 1.0), ("b", "green", 0.5)):
            style = {
                "contrast_limits": None,
                "colormap": colormap,
                "opacity": opacity,
                "visible": True,
                "mask_color": "#00FF00",
                "threshold": None,
                "lock_contrast": False,
                "rendering": "mip",
                "attenuation": 0.05,
                "iso_threshold": None,
            }
            style.update(raw.get(key, {}))
            style["mask_color"] = style["mask_color"].upper()
            recipe[key] = style
        recipe.update(
            {key: value for key, value in raw.items() if key not in ("a", "b")}
        )
        reviews[identifier] = recipe
    return metadata


def test_graph_snapshot_deeply_isolates_nested_params_in_both_directions():
    pipeline = PrototypePipeline()
    nested = {
        "channels": [
            {"index": 0, "calibration": {"scale": [0.4, 0.2, 0.2]}},
        ]
    }
    pipeline.set_param("gaussian", "_vipp_nested_state", nested)

    snapshot = GraphSnapshot.from_pipeline(pipeline)
    nested["channels"][0]["calibration"]["scale"][0] = 99.0
    pipeline.nodes["gaussian"].params["_vipp_nested_state"]["channels"].append(
        {"index": 1}
    )

    saved = _node(snapshot, "gaussian").params["_vipp_nested_state"]
    assert saved == {
        "channels": [
            {"index": 0, "calibration": {"scale": [0.4, 0.2, 0.2]}},
        ]
    }

    saved["channels"][0]["calibration"]["scale"][1] = 88.0
    first = snapshot.to_pipeline()
    second = snapshot.to_pipeline()
    first.nodes["gaussian"].params["_vipp_nested_state"]["channels"][0][
        "calibration"
    ]["scale"][2] = 77.0

    assert second.nodes["gaussian"].params["_vipp_nested_state"] == {
        "channels": [
            {"index": 0, "calibration": {"scale": [0.4, 0.2, 0.2]}},
        ]
    }
    assert _node(snapshot, "gaussian").params["_vipp_nested_state"] == {
        "channels": [
            {"index": 0, "calibration": {"scale": [0.4, 0.2, 0.2]}},
        ]
    }


def test_workflow_snapshot_isolates_metadata_and_preserves_order():
    pipeline = PrototypePipeline()
    positions = {
        "threshold": (30.0, 40.0),
        "input": (10.0, 20.0),
        "gaussian": (20.0, 30.0),
    }
    notes = [
        {
            "id": "note-b",
            "text": "Second spatially, first in declaration order.",
            "position": [50.0, 60.0],
            "width": 280.0,
            "attached_node": "threshold",
        },
        {
            "id": "note-a",
            "text": "Keep this order.",
            "position": [5.0, 6.0],
            "width": 220.0,
        },
    ]
    metadata = {
        "vipp": {
            "inspector": {
                "selected_node_id": "threshold",
                "right_panel_visible": True,
            },
            "thumbnails": {"disabled_node_ids": ["gaussian"]},
        }
    }

    snapshot = workflow_snapshot_from_pipeline(
        pipeline,
        positions,
        notes,
        metadata,
    )
    positions["threshold"] = (999.0, 999.0)
    notes[0]["position"][0] = 999.0
    metadata["vipp"]["thumbnails"]["disabled_node_ids"].append("threshold")

    assert tuple(node.id for node in snapshot.graph.nodes) == tuple(pipeline.nodes)
    assert tuple(node_id for node_id, _position in snapshot.positions) == (
        "threshold",
        "input",
        "gaussian",
    )
    assert tuple(note.id for note in snapshot.notes) == ("note-b", "note-a")
    assert snapshot.notes[0].position == (50.0, 60.0)
    assert snapshot.metadata["vipp"]["thumbnails"]["disabled_node_ids"] == [
        "gaussian"
    ]

    exposed_metadata = snapshot.metadata
    exposed_metadata["vipp"]["thumbnails"]["disabled_node_ids"].clear()
    document = workflow_document_from_snapshot(snapshot)
    document["metadata"]["vipp"]["thumbnails"]["disabled_node_ids"].clear()

    assert snapshot.metadata["vipp"]["thumbnails"]["disabled_node_ids"] == [
        "gaussian"
    ]
    assert snapshot.positions[0] == ("threshold", (30.0, 40.0))


def test_workflow_snapshot_preserves_detached_compute_intent():
    pipeline = PrototypePipeline()
    request = ComputeRequest(
        mode="custom",
        node_preferences={"gaussian": "library:cupyx"},
        fallback_policy="strict",
    )

    snapshot = workflow_snapshot_from_pipeline(
        pipeline,
        compute_request=request,
    )
    document = workflow_document_from_snapshot(snapshot)
    restored = workflow_snapshot_from_document(document)

    assert snapshot.compute_request == request
    assert restored.compute_request == request
    assert document["execution"]["compute"]["node_preferences"] == {
        "gaussian": "library:cupyx"
    }


def test_workflow_snapshot_preserves_prefer_gpu_intent():
    pipeline = PrototypePipeline()
    request = ComputeRequest(mode="prefer_gpu")

    snapshot = workflow_snapshot_from_pipeline(
        pipeline,
        compute_request=request,
    )
    document = workflow_document_from_snapshot(snapshot)
    restored = workflow_snapshot_from_document(document)

    assert snapshot.compute_request == request
    assert restored.compute_request == request
    assert document["execution"]["compute"]["mode"] == "prefer_gpu"
    assert document["execution"]["compute"]["fallback_policy"] == "visible"


def test_graph_snapshot_preserves_pipeline_collection_order():
    pipeline = PrototypePipeline()
    pipeline.add_output_tunnel("Z output", "gaussian")
    pipeline.add_output_tunnel("A output", "threshold")

    snapshot = GraphSnapshot.from_pipeline(pipeline)

    assert tuple(node.id for node in snapshot.nodes) == tuple(pipeline.nodes)
    assert snapshot.connections == tuple(pipeline.connections)
    assert tuple(tunnel.name for tunnel in snapshot.output_tunnels) == (
        "A output",
        "Z output",
    )


def test_snapshot_types_have_value_equality_and_are_not_hashable():
    node_a = NodeSnapshot("node", "gaussian_blur", {"sigma": 1.25})
    node_b = NodeSnapshot("node", "gaussian_blur", {"sigma": 1.25})
    graph_a = GraphSnapshot((node_a,))
    graph_b = GraphSnapshot((node_b,))
    note_a = WorkflowNoteSnapshot("note", "Text", (1.0, 2.0))
    note_b = WorkflowNoteSnapshot("note", "Text", (1.0, 2.0))
    workflow_a = WorkflowSnapshot(
        graph_a,
        (("node", (3.0, 4.0)),),
        (note_a,),
        {"vipp": {}},
    )
    workflow_b = WorkflowSnapshot(
        graph_b,
        (("node", (3.0, 4.0)),),
        (note_b,),
        {"vipp": {}},
    )

    assert node_a == node_b
    assert graph_a == graph_b
    assert note_a == note_b
    assert workflow_a == workflow_b
    for snapshot in (node_a, graph_a, note_a, workflow_a):
        with pytest.raises(TypeError):
            hash(snapshot)
    with pytest.raises(FrozenInstanceError):
        node_a.id = "changed"  # type: ignore[misc]


def test_pipeline_snapshot_supports_empty_transient_graph_without_weakening_v2():
    pipeline = PrototypePipeline()
    pipeline.restore_graph((), ())

    snapshot = workflow_snapshot_from_pipeline(pipeline)
    document = workflow_document_from_snapshot(snapshot)

    assert snapshot.graph.nodes == ()
    assert document["nodes"] == []
    with pytest.raises(ValueError, match="non-empty nodes list"):
        workflow_snapshot_from_document(document)


def _document_with_invalid_node(case: str) -> dict[str, Any]:
    document = serialize_workflow(PrototypePipeline())
    gaussian = next(node for node in document["nodes"] if node["id"] == "gaussian")
    if case == "operation":
        gaussian["operation_id"] = "missing_operation"
    elif case == "missing parameter":
        gaussian["params"].pop("sigma")
    elif case == "unknown parameter":
        gaussian["params"]["silent_simplification"] = True
    else:
        threshold = next(
            node for node in document["nodes"] if node["id"] == "threshold"
        )
        threshold["params"]["threshold_scope"] = "Undocumented shortcut"
    return document


def _raw_graph_snapshot(document: dict[str, Any]) -> GraphSnapshot:
    return GraphSnapshot(
        (
            NodeSnapshot(node["id"], node["operation_id"], node["params"])
            for node in document["nodes"]
        ),
        (
            GraphConnection(
                connection["source"],
                connection["target"],
                connection["target_port"],
                connection["source_port"],
                connection.get("tunnel", ""),
            )
            for connection in document["connections"]
        ),
        (
            OutputTunnel(tunnel["name"], tunnel["source"], tunnel["source_port"])
            for tunnel in document.get("tunnels", [])
        ),
    )


@pytest.mark.parametrize(
    "case",
    ("operation", "missing parameter", "unknown parameter", "invalid choice"),
)
def test_graph_snapshot_uses_exact_persisted_node_validation(case):
    document = _document_with_invalid_node(case)

    with pytest.raises(ValueError) as persisted_error:
        deserialize_workflow(document)
    with pytest.raises(ValueError) as snapshot_error:
        _raw_graph_snapshot(document).to_pipeline()

    assert str(snapshot_error.value) == str(persisted_error.value)


def _cycle_document() -> dict[str, Any]:
    source = PrototypePipeline()
    first = source.add_node("reorder_axes")
    second = source.add_node("set_pixel_size")
    document = serialize_workflow(source)
    document["nodes"] = [
        node for node in document["nodes"] if node["id"] in {first.id, second.id}
    ]
    document["connections"] = [
        {
            "source": first.id,
            "target": second.id,
            "target_port": 0,
            "source_port": 0,
        },
        {
            "source": second.id,
            "target": first.id,
            "target_port": 0,
            "source_port": 0,
        },
    ]
    document["positions"] = {}
    return document


@pytest.mark.parametrize("case", ("source port", "target port", "cycle"))
def test_document_snapshot_matches_executable_graph_rejection(case):
    document = (
        _cycle_document()
        if case == "cycle"
        else serialize_workflow(PrototypePipeline())
    )
    if case == "source port":
        document["connections"][0]["source_port"] = 999
    elif case == "target port":
        document["connections"][0]["target_port"] = 999

    with pytest.raises(ValueError) as established_error:
        _established_document_round_trip(document)
    with pytest.raises(ValueError) as snapshot_error:
        workflow_snapshot_from_document(document)

    assert str(snapshot_error.value) == str(established_error.value)


def test_graph_snapshot_rejects_duplicate_nodes_and_boolean_ports():
    node = NodeSnapshot(
        "node",
        "gaussian_blur",
        {"sigma": 1.0, "channel_axis": -1},
    )
    duplicate_nodes = GraphSnapshot((node, node))
    boolean_port = GraphSnapshot(
        (
            NodeSnapshot("input", "input", PrototypePipeline().nodes["input"].params),
            node,
        ),
        (GraphConnection("input", "node", target_port=False),),
    )

    with pytest.raises(ValueError, match="duplicate node ids"):
        duplicate_nodes.to_pipeline()
    with pytest.raises(ValueError, match="non-integer port"):
        boolean_port.to_pipeline()


@pytest.mark.parametrize("filename", _EXAMPLE_FILENAMES)
def test_example_snapshot_adapters_match_existing_boundary(filename):
    document = json.loads((_EXAMPLE_DIR / filename).read_text(encoding="utf-8"))
    original = deepcopy(document)
    expected = _established_document_round_trip(document)

    snapshot = workflow_snapshot_from_document(document)
    actual = workflow_document_from_snapshot(snapshot)

    assert actual == expected
    assert ("metadata" in actual) is ("metadata" in expected)
    assert actual.get("metadata") == _expected_saved_metadata(document)
    assert scientific_workflow_hash(actual) == scientific_workflow_hash(document)
    assert document == original


def test_review_snapshot_fills_display_defaults_without_mutating_raw_recipe():
    path = _EXAMPLE_DIR / "synthetic-image-review-mask-3d.json"
    document = json.loads(
        path.read_text(encoding="utf-8")
    )
    document["metadata"]["vipp"]["inspector"]["image_reviews"]["review"] = {
        "version": 1,
        "show_axes": False,
        "b": {"opacity": 0.4, "mask_color": "#00ff66"},
    }
    original = deepcopy(document)
    snapshot = workflow_snapshot_from_document(document)
    expected_metadata = _expected_saved_metadata(document)

    assert snapshot.metadata == expected_metadata
    exposed = snapshot.metadata
    exposed["vipp"]["inspector"]["image_reviews"]["review"]["b"]["opacity"] = 0.9
    actual = workflow_document_from_snapshot(snapshot)
    assert snapshot.metadata == expected_metadata
    assert actual["metadata"] == expected_metadata
    assert scientific_workflow_hash(actual) == scientific_workflow_hash(document)
    assert document == original


@pytest.mark.parametrize(
    "recipe",
    ({"version": 1, "show_axes": "false"}, {"version": 1, "b": {"silent_cast": True}}),
)
def test_review_snapshot_rejects_malformed_recipe_without_mutating_document(recipe):
    path = _EXAMPLE_DIR / "synthetic-image-review-mask-3d.json"
    document = json.loads(
        path.read_text(encoding="utf-8")
    )
    document["metadata"]["vipp"]["inspector"]["image_reviews"]["review"] = recipe
    original = deepcopy(document)

    with pytest.raises(ValueError):
        workflow_snapshot_from_document(document)
    assert document == original
