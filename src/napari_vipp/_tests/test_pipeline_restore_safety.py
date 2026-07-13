from __future__ import annotations

import pytest

from napari_vipp.core.pipeline import GraphConnection, PrototypePipeline


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
