from __future__ import annotations

import json

import numpy as np
import pytest

from napari_vipp.core.pipeline import GraphConnection, PrototypePipeline
from napari_vipp.core.workflow import (
    WORKFLOW_TYPE,
    deserialize_workflow,
    load_workflow,
    save_workflow,
    serialize_workflow,
)


def _build_pipeline() -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.set_param("gaussian", "sigma", 2.5)
    median = pipeline.add_node("median_filter")
    pipeline.connect("gaussian", median.id)
    return pipeline


def test_serialize_roundtrip_preserves_graph(tmp_path):
    pipeline = _build_pipeline()
    positions = {
        "input": (0.0, 20.0),
        "gaussian": (330.0, 20.0),
        "threshold": (660.0, 20.0),
    }

    document = serialize_workflow(pipeline, positions)
    assert document["type"] == WORKFLOW_TYPE
    assert document["version"] == 2

    path = tmp_path / "workflow.json"
    saved = save_workflow(path, pipeline, positions)
    assert saved.exists()

    workflow = load_workflow(saved)
    ids = {node.id for node in workflow["nodes"]}
    assert "gaussian" in ids
    assert "median_filter_1" in ids

    by_id = {node.id: node for node in workflow["nodes"]}
    assert by_id["gaussian"].params["sigma"] == 2.5

    connection_pairs = {
        (connection.source_id, connection.target_id)
        for connection in workflow["connections"]
    }
    assert ("gaussian", "median_filter_1") in connection_pairs
    assert workflow["positions"]["gaussian"] == (330.0, 20.0)


@pytest.mark.parametrize(
    "operation_id",
    [
        "otsu_threshold",
        "triangle_threshold",
        "yen_threshold",
        "isodata_threshold",
        "minimum_threshold",
    ],
)
def test_workflow_roundtrip_preserves_float_histogram_bins(operation_id):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    pipeline.set_param(node.id, "histogram_bins", 4_096)

    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert serialized["params"]["histogram_bins"] == 4_096
    assert restored_node.params["histogram_bins"] == 4_096


def test_rescale_cutoff_mode_roundtrips():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("rescale_intensity")
    assert node.params["cutoff_mode"] == "Percentiles"
    pipeline.set_param(node.id, "cutoff_mode", "Percentiles")

    document = serialize_workflow(pipeline)
    explicit = deserialize_workflow(document)
    explicit_node = next(item for item in explicit["nodes"] if item.id == node.id)
    assert explicit_node.params["cutoff_mode"] == "Percentiles"


def test_workflow_roundtrip_preserves_exact_wide_integer_cutoffs():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("rescale_intensity")
    base = 2**60
    pipeline.set_param(node.id, "cutoff_mode", "Values")
    pipeline.set_param(node.id, "in_low_value", base + 1)
    pipeline.set_param(node.id, "in_high_value", base + 2)

    document = json.loads(json.dumps(serialize_workflow(pipeline)))
    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert restored_node.params["in_low_value"] == base + 1
    assert restored_node.params["in_high_value"] == base + 2
    assert isinstance(restored_node.params["in_low_value"], int)


def test_clip_cutoff_mode_roundtrips():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("clip_intensity")
    assert node.params["cutoff_mode"] == "Data range"

    document = serialize_workflow(pipeline)
    explicit = deserialize_workflow(document)
    explicit_node = next(item for item in explicit["nodes"] if item.id == node.id)
    assert explicit_node.params["cutoff_mode"] == "Data range"


@pytest.mark.parametrize(
    ("operation_id", "parameter"),
    [
        ("otsu_threshold", "histogram_bins"),
        ("minimum_threshold", "max_iterations"),
        ("rescale_intensity", "cutoff_mode"),
        ("clip_intensity", "cutoff_mode"),
    ],
)
def test_current_scientific_workflow_parameters_are_required(
    operation_id,
    parameter,
):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"].pop(parameter)

    with pytest.raises(ValueError, match=f"missing required parameters: {parameter}"):
        deserialize_workflow(document)


@pytest.mark.parametrize(
    ("operation_id", "parameter", "value", "message"),
    [
        (
            "otsu_threshold",
            "threshold_scope",
            "banana",
            "must be one of: 'Stack histogram', 'Slice histogram'",
        ),
        (
            "otsu_threshold",
            "histogram_bins",
            1,
            "must be between 2 and 65,536",
        ),
        (
            "rescale_intensity",
            "cutoff_mode",
            "Estimated",
            "must be one of: 'Percentiles', 'Values'",
        ),
        (
            "clip_intensity",
            "cutoff_mode",
            "Automatic",
            "must be one of: 'Data range', 'Values'",
        ),
        (
            "rescale_intensity",
            "out_min",
            float("nan"),
            "must be a finite number",
        ),
    ],
)
def test_workflow_rejects_malformed_scientific_parameters(
    operation_id,
    parameter,
    value,
    message,
):
    pipeline = PrototypePipeline()
    node = pipeline.add_node(operation_id)
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"][parameter] = value

    with pytest.raises(ValueError, match=message):
        deserialize_workflow(document)


def test_set_param_rejects_invalid_choice_before_execution():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("otsu_threshold")

    with pytest.raises(ValueError, match="must be one of"):
        pipeline.set_param(node.id, "threshold_scope", "banana")


def test_set_param_rejects_unknown_public_parameter():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("otsu_threshold")

    with pytest.raises(ValueError, match="has no public parameter 'histogram_bin'"):
        pipeline.set_param(node.id, "histogram_bin", 256)


def test_workflow_rejects_unknown_public_parameter_but_keeps_vipp_state():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("otsu_threshold")
    document = serialize_workflow(pipeline)
    serialized = next(item for item in document["nodes"] if item["id"] == node.id)
    serialized["params"]["unexpected"] = 1

    with pytest.raises(ValueError, match="unknown parameters: 'unexpected'"):
        deserialize_workflow(document)

    serialized["params"].pop("unexpected")
    serialized["params"]["_vipp_review_state"] = {"expanded": True}
    restored = deserialize_workflow(document)
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)
    assert restored_node.params["_vipp_review_state"] == {"expanded": True}


def test_workflow_preserves_valid_runtime_derived_parameter():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("hysteresis_threshold")
    pipeline.set_param(node.id, "resolved_spatial_ndim", 3)

    restored = deserialize_workflow(serialize_workflow(pipeline))
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert restored_node.params["resolved_spatial_ndim"] == 3


def test_workflow_preserves_supported_optional_ui_parameters():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("rescale_axes")
    pipeline.set_param(node.id, "resize_mode", "Output size")
    pipeline.set_param(node.id, "x_size", 12)

    restored = deserialize_workflow(serialize_workflow(pipeline))
    restored_node = next(item for item in restored["nodes"] if item.id == node.id)

    assert restored_node.params["resize_mode"] == "Output size"
    assert restored_node.params["x_size"] == 12


def test_dynamic_choice_parameters_still_require_nonempty_text():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("split_axis")

    with pytest.raises(ValueError, match="non-empty choice value"):
        pipeline.set_param(node.id, "axis", 1)
    with pytest.raises(ValueError, match="non-empty choice value"):
        pipeline.set_param(node.id, "axis", "")


def test_workflow_rejects_blank_paths_and_unknown_positions(tmp_path):
    pipeline = _build_pipeline()

    with pytest.raises(ValueError, match="save path.*blank"):
        save_workflow("", pipeline)
    with pytest.raises(ValueError, match="path.*blank"):
        load_workflow("")
    with pytest.raises(ValueError, match="positions reference unknown"):
        save_workflow(tmp_path / "invalid.json", pipeline, {"ghost": (0, 0)})


def test_restore_graph_runs_after_load(tmp_path):
    pipeline = _build_pipeline()
    path = save_workflow(tmp_path / "wf.json", pipeline, {})

    workflow = load_workflow(path)
    restored = PrototypePipeline()
    restored.restore_graph(workflow["nodes"], workflow["connections"])

    assert set(restored.nodes) == set(pipeline.nodes)
    outputs = restored.run(np.random.rand(4, 8, 8).astype(np.float32))
    assert outputs["threshold"] is not None


def test_workflow_preserves_multi_input_target_ports(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    composite = pipeline.add_node("combine_channels")
    pipeline.connect("input", composite.id, target_port=1)
    pipeline.connect("gaussian", composite.id, target_port=0)

    path = save_workflow(tmp_path / "ports.json", pipeline, {})
    workflow = load_workflow(path)

    ports = {
        (connection.source_id, connection.target_id): connection.target_port
        for connection in workflow["connections"]
    }

    assert ports[("input", composite.id)] == 1
    assert ports[("gaussian", composite.id)] == 0


def test_workflow_preserves_named_tunnels(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    median = pipeline.add_node("median_filter")
    tunnel = pipeline.add_output_tunnel("Raw reference", "input", 0)
    result = pipeline.connect_to_tunnel(tunnel.name, median.id)
    assert result.success

    path = save_workflow(tmp_path / "tunnels.json", pipeline, {})
    document = serialize_workflow(pipeline)

    assert document["tunnels"] == [
        {"name": "Raw reference", "source": "input", "source_port": 0}
    ]
    assert any(
        connection.get("tunnel") == "Raw reference"
        and connection["source"] == "input"
        and connection["target"] == median.id
        for connection in document["connections"]
    )

    workflow = load_workflow(path)
    restored = PrototypePipeline()
    restored.restore_graph(
        workflow["nodes"],
        workflow["connections"],
        workflow["output_tunnels"],
    )

    assert restored.output_tunnel("Raw reference") == tunnel
    assert restored.tunnel_connection_for_input(median.id, 0) is not None
    outputs = restored.run(np.random.rand(4, 8, 8).astype(np.float32))
    assert outputs[median.id] is not None


def test_workflow_preserves_graph_notes(tmp_path):
    pipeline = _build_pipeline()
    notes = [
        {
            "id": "note_1",
            "text": "Check threshold before batch.",
            "position": [42.0, 84.0],
            "width": 260.0,
            "attached_node": "gaussian",
        }
    ]

    path = save_workflow(tmp_path / "notes.json", pipeline, {}, notes)
    document = serialize_workflow(pipeline, {}, notes)

    assert document["notes"] == notes

    workflow = load_workflow(path)

    assert workflow["notes"] == [
        {
            "id": "note_1",
            "text": "Check threshold before batch.",
            "position": (42.0, 84.0),
            "width": 260.0,
            "attached_node": "gaussian",
        }
    ]


def test_workflow_preserves_vipp_metadata(tmp_path):
    pipeline = _build_pipeline()
    metadata = {
        "vipp": {
            "inspector": {
                "selected_node_id": "gaussian",
                "right_panel_visible": False,
            },
            "thumbnails": {
                "disabled_node_ids": ["median_filter_1"],
            },
        }
    }

    document = serialize_workflow(pipeline, metadata=metadata)
    assert document["metadata"] == metadata

    path = save_workflow(tmp_path / "metadata.json", pipeline, metadata=metadata)
    workflow = load_workflow(path)

    assert workflow["metadata"] == metadata


def test_workflow_loads_without_thumbnail_metadata():
    pipeline = _build_pipeline()
    metadata = {
        "vipp": {
            "inspector": {
                "selected_node_id": "gaussian",
                "right_panel_visible": True,
            },
        }
    }
    document = serialize_workflow(pipeline, metadata=metadata)

    workflow = deserialize_workflow(document)

    assert workflow["metadata"] == metadata


def test_workflow_metadata_node_references_must_exist():
    document = serialize_workflow(_build_pipeline())
    document["metadata"] = {
        "vipp": {
            "inspector": {
                "selected_node_id": "ghost",
            },
        }
    }

    with pytest.raises(ValueError, match="references missing node"):
        deserialize_workflow(document)


def test_unknown_operation_is_rejected():
    pipeline = _build_pipeline()
    document = serialize_workflow(pipeline)
    document["nodes"].append(
        {"id": "ghost", "operation_id": "does_not_exist", "params": {}}
    )

    with pytest.raises(ValueError, match="unknown operation"):
        deserialize_workflow(document)


def test_wrong_workflow_version_is_rejected():
    document = serialize_workflow(_build_pipeline())
    document["version"] = 1

    with pytest.raises(ValueError, match="Unsupported workflow version"):
        deserialize_workflow(document)


def test_dangling_connection_is_rejected():
    document = serialize_workflow(_build_pipeline())
    document["connections"].append(
        {
            "source": "ghost",
            "target": "threshold",
            "target_port": 0,
            "source_port": 0,
        }
    )

    with pytest.raises(ValueError, match="references a missing node"):
        deserialize_workflow(document)


def test_missing_node_parameter_is_rejected():
    document = serialize_workflow(_build_pipeline())
    gaussian = next(node for node in document["nodes"] if node["id"] == "gaussian")
    gaussian["params"].pop("sigma")

    with pytest.raises(ValueError, match="missing required parameters: sigma"):
        deserialize_workflow(document)


def test_non_integer_connection_port_is_rejected():
    document = serialize_workflow(_build_pipeline())
    document["connections"][0]["target_port"] = "0"

    with pytest.raises(ValueError, match="target_port.*must be an integer"):
        deserialize_workflow(document)


def test_duplicate_tunnel_names_are_rejected():
    document = serialize_workflow(_build_pipeline())
    document["tunnels"] = [
        {"name": "Ch1", "source": "input", "source_port": 0},
        {"name": "ch1", "source": "gaussian", "source_port": 0},
    ]

    with pytest.raises(ValueError, match="duplicate tunnel name"):
        deserialize_workflow(document)


def test_unknown_tunnel_connection_is_rejected():
    document = serialize_workflow(_build_pipeline())
    document["connections"][0]["tunnel"] = "Missing"

    with pytest.raises(ValueError, match="unknown tunnel"):
        deserialize_workflow(document)


def test_duplicate_graph_note_ids_are_rejected():
    document = serialize_workflow(_build_pipeline())
    document["notes"] = [
        {"id": "note_1", "text": "A", "position": [0, 0], "width": 240},
        {"id": "NOTE_1", "text": "B", "position": [10, 10], "width": 240},
    ]

    with pytest.raises(ValueError, match="duplicate note id"):
        deserialize_workflow(document)


def test_graph_note_attached_node_must_exist():
    document = serialize_workflow(_build_pipeline())
    document["notes"] = [
        {
            "id": "note_1",
            "text": "A",
            "position": [0, 0],
            "width": 240,
            "attached_node": "ghost",
        }
    ]

    with pytest.raises(ValueError, match="references missing attached node"):
        deserialize_workflow(document)


def test_restore_graph_rejects_dangling_connection():
    pipeline = _build_pipeline()
    original_connections = list(pipeline.connections)

    with pytest.raises(ValueError, match="references a missing node"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [GraphConnection("ghost", "threshold", 0, 0)],
        )

    assert pipeline.connections == original_connections


def test_restore_graph_rejects_incompatible_typed_input_connection():
    pipeline = PrototypePipeline()
    measurements = pipeline.add_node("measure_objects_intensity")

    with pytest.raises(ValueError, match="image output to labels input"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [GraphConnection("input", measurements.id, 0, 0)],
        )


@pytest.mark.parametrize(
    "connection",
    [
        GraphConnection("input", "gaussian", -1, 0),
        GraphConnection("input", "gaussian", 0, -1),
    ],
)
def test_restore_graph_rejects_negative_ports(connection):
    pipeline = PrototypePipeline()

    with pytest.raises(ValueError, match="negative port"):
        pipeline.restore_graph(pipeline.nodes.values(), [connection])


def test_restore_graph_rejects_duplicate_target_slots():
    pipeline = PrototypePipeline()

    with pytest.raises(ValueError, match="multiple connections"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [
                GraphConnection("input", "gaussian", 0, 0),
                GraphConnection("threshold", "gaussian", 0, 0),
            ],
        )


def test_restore_graph_rejects_cycles_without_replacing_live_graph():
    pipeline = PrototypePipeline()
    original_connections = list(pipeline.connections)

    with pytest.raises(ValueError, match="containing a cycle"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [
                GraphConnection("gaussian", "threshold", 0, 0),
                GraphConnection("threshold", "gaussian", 0, 0),
            ],
        )

    assert pipeline.connections == original_connections


def test_restore_graph_rejects_connections_into_source_nodes():
    pipeline = PrototypePipeline()
    original_connections = list(pipeline.connections)

    with pytest.raises(ValueError, match="connection to source node"):
        pipeline.restore_graph(
            pipeline.nodes.values(),
            [GraphConnection("gaussian", "input")],
        )

    assert pipeline.connections == original_connections
