from __future__ import annotations

import pytest
from qtpy.QtCore import QPointF

from napari_vipp._graph import PipelineGraphView, _wire_path
from napari_vipp._tests.test_graph_wire_routing import (
    _assert_clear,
    _assert_connection_endpoints,
    _assert_no_shared_runs,
)
from napari_vipp.core.pipeline import PrototypePipeline


def _detour_view(qtbot, *, multi_output=False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source = (
        pipeline.add_node("split_channels")
        if multi_output else pipeline.nodes["input"]
    )
    other_source = pipeline.add_node("median_filter")
    blocker = pipeline.add_node("gaussian_blur")
    targets = [pipeline.add_node("normalize_image") for _ in range(3)]
    view = PipelineGraphView()
    qtbot.addWidget(view)
    view.build_graph(
        [source, other_source, blocker, *targets], (),
        positions={
            source.id: QPointF(0, 0),
            other_source.id: QPointF(0, 300),
            blocker.id: QPointF(350, 0),
            targets[0].id: QPointF(1000, 0),
            targets[1].id: QPointF(1300, 300),
            targets[2].id: QPointF(1600, 0),
        },
    )
    # A tall multi-port card forces these independent routes onto the same
    # upper detour when they do not reserve other source groups' wire lanes.
    view.set_node_input_ports(blocker.id, 40)
    return view, source, other_source, targets


def _ordinary_path(view, connection):
    return _wire_path(
        connection.source.port_scene_pos("output", connection.source_port),
        connection.target.port_scene_pos("input", connection.target_port),
        obstacles=view.connection_obstacle_rects(connection),
        endpoint_rects=(
            connection.source.sceneBoundingRect(),
            connection.target.sceneBoundingRect(),
        ),
    )


def _assert_all_cards_clear(view):
    for connection in view._connections:
        _assert_connection_endpoints(connection)
        # Routing may touch the outer boundary of the card's 24-pixel buffer;
        # that still leaves the rendered wire well clear of the card itself.
        _assert_clear(connection.path(), [
            rect.adjusted(0.001, 0.001, -0.001, -0.001)
            for rect in view.connection_obstacle_rects(connection)
        ])


def test_different_source_nodes_keep_their_obstacle_detours_separate(qtbot):
    view, source, other_source, targets = _detour_view(qtbot)
    # Align independent output ports while keeping their cards separate. Both
    # ordinary routes then use the same near upper gap around the tall blocker.
    view.apply_node_positions({other_source.id: QPointF(-300, 0)})
    view.add_connection(source.id, targets[0].id)
    view.add_connection(other_source.id, targets[1].id)
    first, second = view._connections
    with pytest.raises(AssertionError):
        _assert_no_shared_runs([
            _ordinary_path(view, first), _ordinary_path(view, second)
        ])

    _assert_no_shared_runs([first.path(), second.path()])
    _assert_all_cards_clear(view)


def test_distinct_output_ports_on_one_node_keep_their_detours_separate(qtbot):
    view, source, _, targets = _detour_view(qtbot, multi_output=True)
    view.add_connection(source.id, targets[0].id, source_port=0)
    view.add_connection(source.id, targets[1].id, source_port=1)
    first, second = view._connections
    with pytest.raises(AssertionError):
        _assert_no_shared_runs([
            _ordinary_path(view, first), _ordinary_path(view, second)
        ])

    _assert_no_shared_runs([first.path(), second.path()])
    _assert_all_cards_clear(view)


def test_shared_source_trunks_remain_natural_and_exclude_other_sources(qtbot):
    results = []
    for reverse in (False, True):
        view, source, other_source, targets = _detour_view(qtbot)
        edges = [
            (source.id, targets[0].id),
            (source.id, targets[1].id),
            (other_source.id, targets[2].id),
        ]
        for source_id, target_id in reversed(edges) if reverse else edges:
            view.add_connection(source_id, target_id)
        connections = {
            (item.source_id, item.target_id): item
            for item in view._connections
        }
        first, second, independent = [connections[edge] for edge in edges]
        assert first.path() == _ordinary_path(view, first)
        assert second.path() == _ordinary_path(view, second)
        with pytest.raises(AssertionError):
            _assert_no_shared_runs([first.path(), second.path()])
        for branch in (first, second):
            _assert_no_shared_runs([branch.path(), independent.path()])
        _assert_all_cards_clear(view)
        expected = {edge: item.path() for edge, item in connections.items()}
        view.reroute_connections()
        assert expected == {edge: item.path() for edge, item in connections.items()}
        results.append(expected)

    assert results[0] == results[1]
