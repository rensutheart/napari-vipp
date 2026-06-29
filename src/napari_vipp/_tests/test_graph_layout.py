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
