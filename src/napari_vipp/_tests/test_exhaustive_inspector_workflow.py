from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from napari_vipp.core.pipeline import PALETTE_NODE_LIBRARY
from napari_vipp.core.workflow import (
    WORKFLOW_TYPE,
    WORKFLOW_VERSION,
    canonical_workflow_document,
    workflow_document_from_snapshot,
    workflow_snapshot_from_document,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKFLOW_PATH = REPO_ROOT / "examples" / "manual" / "exhaustive-inspector-showcase.json"


def _showcase_document() -> dict[str, object]:
    return json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))


def test_exhaustive_inspector_showcase_is_current_and_canonical():
    document = _showcase_document()

    assert document["type"] == WORKFLOW_TYPE
    assert document["version"] == WORKFLOW_VERSION

    snapshot = workflow_snapshot_from_document(document)
    assert workflow_document_from_snapshot(snapshot) == document

    canonical = canonical_workflow_document(document)
    assert canonical == document
    assert canonical_workflow_document(canonical) == canonical


def test_exhaustive_inspector_showcase_covers_every_palette_operation_once():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    operation_counts = Counter(node.operation_id for node in snapshot.graph.nodes)
    palette_ids = {operation.id for operation in PALETTE_NODE_LIBRARY}

    assert set(operation_counts) == palette_ids
    assert operation_counts["input"] >= 1
    assert {
        operation_id: count
        for operation_id, count in operation_counts.items()
        if operation_id != "input" and count != 1
    } == {}


def test_exhaustive_inspector_showcase_places_and_connects_every_node():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    pipeline = snapshot.graph.to_pipeline()
    node_ids = set(pipeline.nodes)
    positions = snapshot.positions_dict()

    assert set(positions) == node_ids
    assert all(
        math.isfinite(coordinate)
        for position in positions.values()
        for coordinate in position
    )

    declared_order = list(pipeline.nodes)
    topological_order = pipeline.topological_order()
    assert declared_order == topological_order
    topological_rank = {
        node_id: index for index, node_id in enumerate(topological_order)
    }

    for connection in pipeline.connections:
        assert (
            topological_rank[connection.source_id]
            < topological_rank[connection.target_id]
        )
        assert (
            0
            <= connection.source_port
            < len(pipeline.output_ports(connection.source_id))
        )
        assert (
            0
            <= connection.target_port
            < pipeline.input_port_count(connection.target_id)
        )

    for node_id, node in pipeline.nodes.items():
        required_connections = pipeline._required_input_connections(node_id)
        assert required_connections is not None, (
            f"{node_id!r} ({node.operation_id}) has an unconnected required input"
        )
        assert len(required_connections) == (
            0
            if not pipeline.operation_spec(node.operation_id).has_input
            else pipeline._required_inputs_for(node)
        )


def test_exhaustive_inspector_showcase_uses_modern_time_slice_parameters():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    selector = next(
        node
        for node in snapshot.graph.nodes
        if node.operation_id == "select_axis_slice"
    )

    params = selector.params
    assert set(params) == {
        "axis",
        "index",
        "axes",
        "indices",
        "ranges",
        "range_mode",
        "remove_axes",
        "remove_indices",
    }
    assert params["range_mode"] is True
    assert params["ranges"] == ""
    assert params["axis"] == 0
    assert params["axes"] == params["remove_axes"] == "0"

    selected_index = params["index"]
    assert isinstance(selected_index, int)
    assert selected_index >= 0
    assert params["indices"] == params["remove_indices"] == str(selected_index)


def test_exhaustive_inspector_showcase_uses_tunnels_selectively():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    pipeline = snapshot.graph.to_pipeline()

    expected_tunnels = {
        "Born-Wolf PSF": ("born_wolf_psf_1", 0, 1),
        "Expanded labels": ("expand_labels_1", 0, 1),
        "Green channel": ("split_channels_1", 1, 9),
        "Object labels": ("relabel_sequential_1", 0, 4),
        "ROI mask": ("binary_threshold_1", 0, 11),
        "Raw volume": ("input_2", 0, 4),
        "Red channel": ("split_channels_1", 0, 10),
        "Skeleton mask": ("skeletonize_1", 0, 5),
        "Watershed labels": ("auto_watershed_from_mask_1", 0, 2),
    }
    actual_tunnels = {
        tunnel.name: (tunnel.source_id, tunnel.source_port)
        for tunnel in pipeline.output_tunnel_list()
    }
    assert actual_tunnels == {
        name: (source_id, source_port)
        for name, (source_id, source_port, _) in expected_tunnels.items()
    }

    tunnel_counts = Counter(
        connection.tunnel_name
        for connection in pipeline.connections
        if connection.tunnel_name
    )
    assert tunnel_counts == Counter(
        {
            name: subscriber_count
            for name, (*_, subscriber_count) in expected_tunnels.items()
        }
    )
    assert sum(tunnel_counts.values()) == 47
    assert sum(not connection.tunnel_name for connection in pipeline.connections) == 100

    for connection in pipeline.connections:
        if not connection.tunnel_name:
            continue
        tunnel = pipeline.output_tunnel(connection.tunnel_name)
        assert tunnel is not None
        assert (connection.source_id, connection.source_port) == (
            tunnel.source_id,
            tunnel.source_port,
        )

    # The busy fan-outs still retain direct local wires for the main lane paths.
    for tunnel_name in (
        "Raw volume",
        "Red channel",
        "Green channel",
        "ROI mask",
        "Object labels",
        "Skeleton mask",
    ):
        tunnel = pipeline.output_tunnel(tunnel_name)
        assert tunnel is not None
        assert any(
            connection.source_id == tunnel.source_id
            and connection.source_port == tunnel.source_port
            and not connection.tunnel_name
            for connection in pipeline.connections
        )


def test_exhaustive_inspector_showcase_cannot_auto_save_to_disk():
    snapshot = workflow_snapshot_from_document(_showcase_document())
    save_nodes = [
        node for node in snapshot.graph.nodes if node.operation_id == "save_output"
    ]

    assert len(save_nodes) == 1
    assert save_nodes[0].params["enabled"] == "off"
    assert save_nodes[0].params["path"] == ""
