from __future__ import annotations

import pytest
from qtpy.QtCore import QPointF, QRectF

from napari_vipp._graph import (
    PipelineGraphView,
    _flatten_wire_path,
    _route_collision_penalty,
    _wire_path,
)
from napari_vipp.core.pipeline import PrototypePipeline


def _assert_clear(path, rects) -> None:
    points = _flatten_wire_path(path)
    assert _route_collision_penalty(points, tuple(rects)) == 0
    # Independently inspect the rendered curve, not just its control polygon.
    assert not any(
        rect.contains(path.pointAtPercent(index / 1000))
        for rect in rects
        for index in range(1001)
    )


def _assert_endpoint_route(path, start, end, rects) -> None:
    assert (path.pointAtPercent(0) - start).manhattanLength() < 0.01
    assert (path.pointAtPercent(1) - end).manhattanLength() < 0.01
    assert path.pointAtPercent(0.001).x() > start.x()
    assert path.pointAtPercent(0.999).x() < end.x()
    _assert_clear(path, [rect.adjusted(0.75, 0.75, -0.75, -0.75) for rect in rects])


@pytest.mark.parametrize(
    ("source", "target", "target_port_y"),
    [
        (QRectF(350, 0, 220, 217), QRectF(350, 460, 220, 217), 568.5),
        (QRectF(350, 460, 220, 217), QRectF(350, 0, 220, 217), 108.5),
        (QRectF(500, 0, 220, 217), QRectF(0, 0, 220, 217), 108.5),
        (QRectF(500, 360, 220, 217), QRectF(0, 0, 220, 217), 108.5),
        # Reported Dilation -> Logical OR route: wide, multi-input target.
        (QRectF(350, 40, 220, 218), QRectF(80, 355, 292, 211), 394),
        (QRectF(350, 40, 220, 218), QRectF(80, 355, 292, 411), 718),
    ],
    ids=[
        "same-column-down",
        "same-column-up",
        "backward-row",
        "backward-up",
        "logical-or-upper-port",
        "expanded-lower-port",
    ],
)
def test_backward_routes_never_cross_endpoint_cards(source, target, target_port_y):
    start = QPointF(source.right(), source.center().y())
    end = QPointF(target.left(), target_port_y)

    path = _wire_path(start, end, endpoint_rects=(source, target))

    _assert_endpoint_route(path, start, end, (source, target))


def test_crowded_route_candidate_selection_never_drops_endpoint_protection():
    source = QRectF(350, 0, 220, 217)
    target = QRectF(350, 460, 220, 217)
    start = QPointF(source.right(), source.center().y())
    end = QPointF(target.left(), target.center().y())
    # More obstacles than the bounded candidate-lane catalogue retains.
    obstacles = tuple(QRectF(110 + i * 24, 255, 18, 100) for i in range(20))

    path = _wire_path(start, end, obstacles=obstacles, endpoint_rects=(source, target))

    _assert_endpoint_route(path, start, end, (source, target))


def test_staggered_obstacles_use_clear_alternate_lanes(monkeypatch):
    import napari_vipp._graph as graph

    start, end = QPointF(0, 0), QPointF(600, 0)
    obstacles = (
        QRectF(20, -120, 50, 100), QRectF(20, 20, 50, 100),
        QRectF(250, -50, 100, 100),
    )
    calls = []
    original = graph._wire_visibility_detour

    def traced(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(graph, "_wire_visibility_detour", traced)
    path = _wire_path(start, end, obstacles=obstacles)

    assert calls
    _assert_clear(path, obstacles)
    assert path.pointAtPercent(0.001).x() > start.x()
    assert path.pointAtPercent(0.999).x() < end.x()
    # Repetition also checks ownership of points flattened from temporary Qt
    # polygons: borrowed storage caused intermittent collisions/different paths.
    for _ in range(30):
        repeated = _wire_path(start, end, obstacles=obstacles)
        _assert_clear(repeated, obstacles)
        assert path == repeated


def test_backward_wire_uses_narrow_gap_between_cards():
    source, target = QRectF(350, 0, 220, 217), QRectF(240, 267, 292, 217)
    start, end = QPointF(570, 108.5), QPointF(240, 309)

    path = _wire_path(start, end, endpoint_rects=(source, target))

    _assert_endpoint_route(path, start, end, (source, target))
    assert path.boundingRect().top() >= start.y()
    assert path.boundingRect().bottom() <= end.y()


def test_detours_are_checked_outside_the_original_port_corridor():
    start, end = QPointF(0, 0), QPointF(600, 0)
    obstacles = (
        QRectF(240, -80, 120, 160),
        # Above the original corridor; it blocks the first upper detour.
        QRectF(100, -190, 400, 75),
    )

    path = _wire_path(start, end, obstacles=obstacles)

    _assert_clear(path, obstacles)


def test_distant_cards_do_not_trigger_individual_segment_checks(monkeypatch):
    import napari_vipp._graph as graph

    source, target = QRectF(350, 0, 220, 217), QRectF(350, 460, 220, 217)
    distant = tuple(QRectF(2000 + 280 * i, 0, 220, 217) for i in range(400))
    original = graph._polyline_rect_penalty
    distant_checks = []

    def traced(points, rect):
        if rect.left() >= 2000:
            distant_checks.append(rect)
        return original(points, rect)

    monkeypatch.setattr(graph, "_polyline_rect_penalty", traced)
    path = _wire_path(
        QPointF(source.right(), source.center().y()),
        QPointF(target.left(), target.center().y()),
        endpoint_rects=(source, target), obstacles=distant,
    )

    assert not distant_checks
    _assert_clear(path, distant)


def _build_view(qtbot, source_pos=None, target_pos=None):
    source_pos = QPointF(350, 20) if source_pos is None else source_pos
    target_pos = QPointF(350, 460) if target_pos is None else target_pos
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    qtbot.addWidget(view)
    view.build_graph(
        pipeline.nodes.values(),
        [],
        positions={
            "input": source_pos,
            "gaussian": QPointF(-800, -500),
            "threshold": target_pos,
        },
    )
    view.add_connection("input", "threshold")
    return view, view._connections[-1]


def _assert_connection_endpoints(connection):
    _assert_endpoint_route(
        connection.path(),
        connection.source.port_scene_pos("output", connection.source_port),
        connection.target.port_scene_pos("input", connection.target_port),
        (connection.source.sceneBoundingRect(), connection.target.sceneBoundingRect()),
    )


def test_connected_drag_keeps_endpoint_protection_without_scene_obstacle_scan(
    qtbot, monkeypatch
):
    view, connection = _build_view(qtbot)
    calls = []
    original_obstacles = view.connection_obstacle_rects

    def tracked_obstacles(*args, **kwargs):
        calls.append(True)
        return original_obstacles(*args, **kwargs)

    monkeypatch.setattr(view, "connection_obstacle_rects", tracked_obstacles)
    starts = {"threshold": QPointF(view._proxies["threshold"].pos())}
    for delta in (QPointF(-250, 60), QPointF(0, -800), QPointF(-320, -450)):
        view._move_selected_nodes_during_drag(starts, delta)
        assert view.node_drag_in_progress()
        assert not calls
        _assert_connection_endpoints(connection)

    view._finish_selected_node_drag(starts)

    assert not view.node_drag_in_progress()
    assert calls
    _assert_connection_endpoints(connection)


def test_expanded_multi_input_target_updates_route_and_preserves_body_clearance(qtbot):
    view, first_connection = _build_view(qtbot)
    old_height = view.node_scene_rect("threshold").height()

    view.set_node_input_ports(
        "threshold", 12, [f"Input {index + 1}" for index in range(12)]
    )
    view.add_connection("input", "threshold", target_port=11)

    assert view.node_scene_rect("threshold").height() > old_height
    _assert_connection_endpoints(first_connection)
    _assert_connection_endpoints(view._connections[-1])


def test_moving_unrelated_card_onto_wire_reroutes_without_crossing_cards(qtbot):
    view, connection = _build_view(qtbot, QPointF(0, 20), QPointF(750, 20))
    original_path = connection.path()

    view.center_node_on("gaussian", original_path.pointAtPercent(0.5))

    obstacle = view.node_scene_rect("gaussian")
    margin = view.WIRE_OBSTACLE_MARGIN
    _assert_clear(
        connection.path(), [obstacle.adjusted(-margin, -margin, margin, margin)]
    )
    _assert_connection_endpoints(connection)
    assert connection.path() != original_path

    view.center_node_on("gaussian", QPointF(-800, -500))

    _assert_connection_endpoints(connection)
    assert connection.path() == original_path


def test_thin_obstacle_is_not_skipped_between_curve_samples():
    start = QPointF(0, 0)
    end = QPointF(600, 0)
    obstacle = QRectF(271.125, -40, 0.2, 80)

    path = _wire_path(start, end, obstacles=(obstacle,))

    _assert_clear(path, (obstacle,))
