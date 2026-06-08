from __future__ import annotations

import numpy as np

from napari_vipp.core.pipeline import PrototypePipeline
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
    assert document["version"] == 1

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
    composite = pipeline.add_node("channel_composite")
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


def test_unknown_operation_is_skipped():
    pipeline = _build_pipeline()
    document = serialize_workflow(pipeline)
    document["nodes"].append({"id": "ghost", "operation_id": "does_not_exist"})

    workflow = deserialize_workflow(document)
    ids = {node.id for node in workflow["nodes"]}
    assert "ghost" not in ids
