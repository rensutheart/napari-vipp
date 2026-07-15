from __future__ import annotations

from napari_vipp.core.graph_layout import (
    LayoutEdge,
    LayoutNode,
    layout_layered_dag,
)


def test_layered_layout_places_dependencies_left_to_right():
    positions = layout_layered_dag(
        [
            LayoutNode("input", 100, 60),
            LayoutNode("branch_a", 120, 70),
            LayoutNode("branch_b", 120, 70),
            LayoutNode("output", 110, 80),
        ],
        [
            LayoutEdge("input", "branch_a"),
            LayoutEdge("input", "branch_b"),
            LayoutEdge("branch_a", "output"),
            LayoutEdge("branch_b", "output"),
        ],
        current_positions={
            "branch_a": (0, 300),
            "branch_b": (0, 0),
        },
    )

    assert positions["input"][0] < positions["branch_a"][0]
    assert positions["input"][0] < positions["branch_b"][0]
    assert positions["branch_a"][0] < positions["output"][0]
    assert positions["branch_b"][0] < positions["output"][0]
    assert positions["branch_b"][1] < positions["branch_a"][1]
    assert positions["branch_a"][1] - positions["branch_b"][1] >= 70


def test_layered_layout_places_disconnected_components_in_lanes():
    positions = layout_layered_dag(
        [
            LayoutNode("input", 100, 60),
            LayoutNode("threshold", 100, 60),
            LayoutNode("loose", 100, 60),
        ],
        [LayoutEdge("input", "threshold")],
        component_gap=120,
    )

    assert positions["input"][0] < positions["threshold"][0]
    assert positions["loose"][0] == positions["input"][0]
    assert positions["loose"][1] > positions["input"][1] + 60


def test_layered_layout_keeps_expanded_multi_port_nodes_disjoint():
    nodes = [
        LayoutNode("source", 300, 180),
        LayoutNode("tall_a", 420, 348),
        LayoutNode("tall_b", 360, 420),
        LayoutNode("output", 310, 180),
    ]
    positions = layout_layered_dag(
        nodes,
        [
            LayoutEdge("source", "tall_a"),
            LayoutEdge("source", "tall_b"),
            LayoutEdge("tall_a", "output"),
            LayoutEdge("tall_b", "output"),
        ],
        vertical_gap=90,
    )
    sizes = {node.id: (node.width, node.height) for node in nodes}

    for index, first in enumerate(nodes):
        first_x, first_y = positions[first.id]
        first_w, first_h = sizes[first.id]
        for second in nodes[index + 1 :]:
            second_x, second_y = positions[second.id]
            second_w, second_h = sizes[second.id]
            separated = (
                first_x + first_w <= second_x
                or second_x + second_w <= first_x
                or first_y + first_h <= second_y
                or second_y + second_h <= first_y
            )
            assert separated, f"{first.id} overlaps {second.id}"

    tall_nodes = sorted(
        (
            (positions["tall_a"][1], "tall_a"),
            (positions["tall_b"][1], "tall_b"),
        )
    )
    top_y, top_id = tall_nodes[0]
    bottom_y, _bottom_id = tall_nodes[1]
    assert bottom_y - top_y >= sizes[top_id][1] + 90
