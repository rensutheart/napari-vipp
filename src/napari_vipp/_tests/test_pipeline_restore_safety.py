from __future__ import annotations

import pytest

from napari_vipp.core.pipeline import (
    GraphConnection,
    OutputTunnel,
    PrototypePipeline,
)


def test_restore_graph_rejects_preserved_type_cycle_without_recursing():
    source = PrototypePipeline()
    first = source.add_node("reorder_axes")
    second = source.add_node("set_pixel_size")
    nodes = (source.nodes[first.id], source.nodes[second.id])
    connections = (
        GraphConnection(first.id, second.id),
        GraphConnection(second.id, first.id),
    )
    target = PrototypePipeline()
    original_node_ids = tuple(target.nodes)
    original_connections = tuple(target.connections)

    with pytest.raises(ValueError, match="graph containing a cycle"):
        target.restore_graph(nodes, connections)

    assert tuple(target.nodes) == original_node_ids
    assert tuple(target.connections) == original_connections


def test_restore_graph_canonicalizes_tunnel_references_for_later_removal():
    source = PrototypePipeline()
    nodes = (source.nodes["input"], source.nodes["gaussian"])
    target = PrototypePipeline()

    target.restore_graph(
        nodes,
        (GraphConnection("input", "gaussian", tunnel_name=" RAW data "),),
        (OutputTunnel("  Raw   Data  ", "input"),),
    )

    assert target.output_tunnel_list() == (OutputTunnel("Raw Data", "input"),)
    assert target.connections[0].tunnel_name == "Raw Data"

    removed = target.remove_output_tunnel("raw data")

    assert removed == (GraphConnection("input", "gaussian", 0, 0, "Raw Data"),)
    assert target.connections == []
