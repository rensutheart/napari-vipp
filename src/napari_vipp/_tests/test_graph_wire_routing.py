from __future__ import annotations

import pytest
from qtpy.QtCore import QPointF, QRectF
from qtpy.QtGui import QPainterPath

from napari_vipp._graph import (
    PipelineGraphView,
    _flatten_wire_path,
    _route_collision_penalty,
    _wire_path,
)
from napari_vipp.core.pipeline import GraphConnection, PrototypePipeline


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


def _assert_no_shared_runs(paths):
    """Independently compare segments of the rendered curves, including bends."""
    for index, path in enumerate(paths):
        for other in paths[:index]:
            first_points = _flatten_wire_path(path)
            other_points = _flatten_wire_path(other)
            for a, b in zip(first_points, first_points[1:], strict=False):
                direction = b - a
                norm = direction.x() ** 2 + direction.y() ** 2
                if norm < 1e-12:
                    continue
                for c, d in zip(other_points, other_points[1:], strict=False):
                    relative, other_direction = c - a, d - c
                    if abs(direction.x() * other_direction.y()
                           - direction.y() * other_direction.x()) > 1e-6:
                        continue
                    if abs(direction.x() * relative.y()
                           - direction.y() * relative.x()) > 1e-6:
                        continue
                    projections = [
                        ((p - a).x() * direction.x() + (p - a).y() * direction.y())
                        / norm for p in (c, d)
                    ]
                    overlap = min(1.0, max(projections)) - max(0.0, min(projections))
                    assert overlap < 1e-6


@pytest.mark.parametrize("backward", [False, True])
def test_shared_output_fans_out_without_shared_stubs(qtbot, backward):
    pipeline = PrototypePipeline()
    nodes = [pipeline.nodes["input"]]
    targets = [pipeline.add_node("gaussian_blur") for _ in range(5)]
    nodes.extend(targets)
    view = PipelineGraphView()
    qtbot.addWidget(view)
    source_x, target_x = (650, 0) if backward else (0, 650)
    view.build_graph(nodes, [], positions={
        "input": QPointF(source_x, 200),
        **{node.id: QPointF(target_x, i * 300) for i, node in enumerate(targets)},
    })
    for node in targets:
        view.add_connection("input", node.id)

    _assert_no_shared_runs([item.path() for item in view._connections])
    for item in view._connections:
        _assert_connection_endpoints(item)
    paths = [item.path() for item in view._connections]
    for _ in range(3):
        view.reroute_connections()
        assert paths == [item.path() for item in view._connections]


def test_aligned_independent_wires_and_obstacle_detours_use_distinct_lanes():
    obstacle = QRectF(240, -80, 120, 160)
    paths = []
    for start, end in ((QPointF(0, 0), QPointF(600, 0)),
                       (QPointF(-60, 0), QPointF(700, 0)),
                       (QPointF(-120, 0), QPointF(800, 0))):
        path = _wire_path(start, end, obstacles=(obstacle,),
                          peer_paths=tuple(_flatten_wire_path(p) for p in paths))
        _assert_clear(path, (obstacle,))
        paths.append(path)
    _assert_no_shared_runs(paths)


def test_distant_peer_wires_do_not_expand_local_fanout_lanes():
    start, end = QPointF(0, 0), QPointF(600, 30)
    expected = _wire_path(start, end, start_lane=6)
    distant = tuple(
        _flatten_wire_path(_wire_path(QPointF(0, y), QPointF(600, y)))
        for y in range(2000, 4000, 100)
    )
    actual = _wire_path(start, end, start_lane=6, peer_paths=distant)
    assert actual == expected


def test_buried_port_skips_impossible_obstacle_but_preserves_other_clearance(
    monkeypatch,
):
    import napari_vipp._graph as graph

    start, end = QPointF(0, 0), QPointF(600, 0)
    buried = QRectF(580, -40, 100, 80)
    avoidable = QRectF(240, -80, 120, 160)
    observed = []
    original = graph._wire_middle_candidates

    def candidates(first, last, blockers):
        observed.extend(blockers)
        yield from original(first, last, blockers)

    monkeypatch.setattr(graph, "_wire_middle_candidates", candidates)
    peer = _flatten_wire_path(_wire_path(start, end, obstacles=(avoidable,)))
    path = _wire_path(start, end, obstacles=(buried, avoidable), peer_paths=(peer,))

    assert buried not in observed
    _assert_clear(path, (avoidable,))
    _assert_no_shared_runs([path, _wire_path(start, end, obstacles=(avoidable,))])


def test_peer_lane_search_preserves_staggered_card_clearance():
    obstacles = (
        QRectF(20, -120, 50, 100), QRectF(20, 20, 50, 100),
        QRectF(250, -50, 100, 100),
    )
    paths = []
    for start_x in (0, -30, -60):
        path = _wire_path(QPointF(start_x, 0), QPointF(600 - start_x, 0),
                          obstacles=obstacles,
                          peer_paths=tuple(_flatten_wire_path(p) for p in paths))
        _assert_clear(path, obstacles)
        paths.append(path)
    _assert_no_shared_runs(paths)


def test_deleting_branch_restores_single_wire_route(qtbot):
    view, first = _build_view(qtbot, QPointF(0, 20), QPointF(750, 20))
    original = first.path()
    view.add_connection("input", "gaussian")
    _assert_no_shared_runs([item.path() for item in view._connections])
    view.delete_connection_item(view._connections[-1], notify=False)
    assert first.path() == original


def test_fanout_drag_retains_distinct_routes_without_card_scan(qtbot, monkeypatch):
    view, _ = _build_view(qtbot)
    view.add_connection("input", "gaussian")
    calls = []
    monkeypatch.setattr(view, "connection_obstacle_rects",
                        lambda *args: calls.append(True) or ())
    starts = {"input": QPointF(view._proxies["input"].pos())}
    view._move_selected_nodes_during_drag(starts, QPointF(100, 80))
    assert not calls
    _assert_no_shared_runs([item.path() for item in view._connections])
    for item in view._connections:
        _assert_connection_endpoints(item)


def test_isolated_wire_crossing_keeps_the_direct_route():
    crossing = QPainterPath(QPointF(300, -200))
    crossing.lineTo(QPointF(300, 200))
    start, end = QPointF(0, 0), QPointF(600, 0)
    direct = _wire_path(start, end)

    routed = _wire_path(
        start, end, peer_paths=(_flatten_wire_path(crossing),)
    )

    assert routed == direct
    assert routed.intersects(crossing)
    _assert_no_shared_runs([routed, crossing])


@pytest.mark.parametrize("mode", ["build", "incremental"])
def test_connection_order_does_not_change_assigned_wire_routes(qtbot, mode):
    pipeline = PrototypePipeline()
    last_target = pipeline.add_node("median_filter")
    positions = {
        "input": QPointF(0, 300),
        "gaussian": QPointF(650, 0),
        "threshold": QPointF(650, 300),
        last_target.id: QPointF(650, 600),
    }
    connections = (
        GraphConnection("input", "gaussian"),
        GraphConnection("input", "threshold"),
        GraphConnection("input", last_target.id),
    )
    results = []
    for order in (connections, tuple(reversed(connections))):
        view = PipelineGraphView()
        qtbot.addWidget(view)
        view.build_graph(
            pipeline.nodes.values(),
            order if mode == "build" else (),
            positions=positions,
        )
        if mode == "incremental":
            for connection in order:
                view.add_connection(
                    connection.source_id,
                    connection.target_id,
                    connection.target_port,
                    connection.source_port,
                )
        _assert_no_shared_runs([item.path() for item in view._connections])
        for item in view._connections:
            _assert_connection_endpoints(item)
        results.append({
            (item.source_id, item.target_id, item.target_port, item.source_port):
            item.path()
            for item in view._connections
        })

    assert results[0] == results[1]


def test_independent_output_ports_do_not_share_the_same_obstacle_detour():
    source = QRectF(-220, -100, 220, 300)
    first_target = QRectF(600, -100, 220, 300)
    second_target = QRectF(950, -50, 220, 300)
    blocker = QRectF(220, -80, 160, 600)
    first_start, second_start = QPointF(0, 0), QPointF(0, 50)
    first_end, second_end = QPointF(600, 0), QPointF(950, 50)
    first = _wire_path(
        first_start, first_end,
        obstacles=(blocker, second_target),
        endpoint_rects=(source, first_target),
    )
    ordinary_second = _wire_path(
        second_start, second_end,
        obstacles=(blocker, first_target),
        endpoint_rects=(source, second_target),
    )
    # This geometry actually produces a shared upper detour when edges are
    # considered independently, although their output ports are distinct.
    with pytest.raises(AssertionError):
        _assert_no_shared_runs([first, ordinary_second])

    second = _wire_path(
        second_start, second_end,
        obstacles=(blocker, first_target),
        endpoint_rects=(source, second_target),
        peer_paths=(_flatten_wire_path(first),),
    )

    _assert_no_shared_runs([first, second])
    _assert_endpoint_route(first, first_start, first_end, (source, first_target))
    _assert_endpoint_route(second, second_start, second_end, (source, second_target))
    _assert_clear(first, (blocker, second_target))
    _assert_clear(second, (blocker, first_target))


def test_impossible_other_card_clearance_keeps_endpoint_protection_and_separation():
    source = QRectF(350, 0, 220, 217)
    target = QRectF(350, 200, 220, 217)
    # The endpoint cards overlap at their edges, but neither port is buried in
    # the other's body. A third card enclosing both ports cannot be avoided.
    enclosing_card = QRectF(-1000, -1000, 3000, 3000)
    start = QPointF(source.right(), source.center().y())
    first_end = QPointF(target.left(), 280)
    second_end = QPointF(target.left(), 340)
    first = _wire_path(
        start, first_end,
        obstacles=(enclosing_card,),
        endpoint_rects=(source, target),
    )

    second = _wire_path(
        start, second_end,
        obstacles=(enclosing_card,),
        endpoint_rects=(source, target),
        peer_paths=(_flatten_wire_path(first),),
        start_lane=12,
    )

    _assert_no_shared_runs([first, second])
    _assert_endpoint_route(first, start, first_end, (source, target))
    _assert_endpoint_route(second, start, second_end, (source, target))


def test_dense_backward_fanout_uses_clear_candidate_lanes_without_visibility_search(
    monkeypatch,
):
    import napari_vipp._graph as graph

    def unexpected_search(*_args, **_kwargs):
        pytest.fail("This backward fanout should fit the ordinary routing candidates.")

    monkeypatch.setattr(graph, "_wire_visibility_detour", unexpected_search)
    source = QRectF(650, 200, 220, 217)
    targets = [QRectF(0, index * 300, 220, 217) for index in range(20)]
    start = QPointF(source.right(), source.center().y())
    paths = []
    reservations = []
    for index, target in enumerate(targets):
        end = QPointF(target.left(), target.center().y())
        obstacles = tuple(
            other.adjusted(-24, -24, 24, 24)
            for other_index, other in enumerate(targets)
            if other_index != index
        )
        path = _wire_path(
            start, end,
            obstacles=obstacles,
            endpoint_rects=(source, target),
            peer_paths=tuple(reservations),
            start_lane=(index - 9.5) * 12,
        )
        _assert_endpoint_route(path, start, end, (source, target))
        _assert_clear(path, obstacles)
        paths.append(path)
        reservations.append(_flatten_wire_path(path))

    _assert_no_shared_runs(paths)


def test_partial_reroute_repairs_peer_changed_before_the_global_pass(
    qtbot, monkeypatch
):
    import napari_vipp._graph as graph

    pipeline = PrototypePipeline()
    later_source = pipeline.add_node("median_filter")
    view = PipelineGraphView()
    qtbot.addWidget(view)
    view.build_graph(
        pipeline.nodes.values(), (),
        positions={
            "input": QPointF(0, 20),
            "gaussian": QPointF(750, 20),
            later_source.id: QPointF(0, 720),
            "threshold": QPointF(750, 720),
        },
    )
    view.add_connection("input", "gaussian")
    view.add_connection(later_source.id, "threshold")
    earlier, later = view._connections
    original_later = later.path()
    start = earlier.source.port_scene_pos("output", earlier.source_port)
    end = earlier.target.port_scene_pos("input", earlier.target_port)
    later_start = later.source.port_scene_pos("output", later.source_port)
    later_end = later.target.port_scene_pos("input", later.target_port)
    assert abs(later_start.y() - later_end.y()) < 0.01
    left, right = start.x() + 100, end.x() - 100
    changed = QPainterPath(start)
    for point in (
        QPointF(left, start.y()),
        QPointF(left, later_start.y()),
        QPointF(right, later_start.y()),
        QPointF(right, end.y()),
        end,
    ):
        changed.lineTo(point)
    _assert_endpoint_route(
        changed, start, end,
        (earlier.source.sceneBoundingRect(), earlier.target.sceneBoundingRect()),
    )
    _assert_clear(changed, view.connection_obstacle_rects(earlier))

    original_router = graph._wire_path

    def changed_earlier_route(route_start, route_end, **kwargs):
        if route_start == start and route_end == end:
            return changed
        return original_router(route_start, route_end, **kwargs)

    monkeypatch.setattr(graph, "_wire_path", changed_earlier_route)
    view._mark_graph_geometry_changed()
    # Port refreshes can update one wire before the spatial rerouting pass.
    # The valid new detour now occupies part of the distant lower wire.
    earlier.update_path()
    with pytest.raises(AssertionError):
        _assert_no_shared_runs([earlier.path(), later.path()])
    affected = earlier.source.sceneBoundingRect()
    margin = view.WIRE_OBSTACLE_MARGIN * 2
    assert not view._connection_route_rect(later).intersects(
        affected.adjusted(-margin, -margin, margin, margin)
    )

    view.reroute_connections(affected_rect=affected)

    assert earlier.path() == changed
    assert later.path() != original_later
    _assert_no_shared_runs([earlier.path(), later.path()])
    for connection in (earlier, later):
        _assert_connection_endpoints(connection)
        _assert_clear(connection.path(), view.connection_obstacle_rects(connection))


@pytest.mark.parametrize("mode", ("build", "update"))
def test_fanout_routes_clear_tunnel_badge_on_another_output_port(qtbot, mode):
    pipeline = PrototypePipeline()
    source = pipeline.add_node("skeleton_graph_tables")
    targets = [pipeline.add_node("merge_tables") for _ in range(2)]
    nodes = [source, *targets]
    connections = [GraphConnection(source.id, target.id) for target in targets]
    positions = {
        source.id: QPointF(650, 200),
        targets[0].id: QPointF(0, 0),
        targets[1].id: QPointF(0, 650),
    }
    tunnel = pipeline.add_output_tunnel("Measurement branches", source.id, 1)
    view = PipelineGraphView()
    qtbot.addWidget(view)
    view.build_graph(
        nodes, connections, positions=positions,
        output_tunnels=(tunnel,) if mode == "build" else (),
    )
    original_paths = [item.path() for item in view._connections]
    if mode == "update":
        view.set_port_tunnels((tunnel,), connections)
    badge = view._proxies[source.id].output_ports[1]._tunnel_badge
    rect = badge.sceneBoundingRect()
    if mode == "update":
        # This is an actual route obstruction outside the endpoint card,
        # rather than a badge that the existing wires never approached.
        assert any(
            _route_collision_penalty(_flatten_wire_path(path), (rect,))
            for path in original_paths
        )
    _assert_no_shared_runs([item.path() for item in view._connections])
    for item in view._connections:
        _assert_connection_endpoints(item)
        _assert_clear(item.path(), (rect,))

    wider = pipeline.rename_output_tunnel(
        tunnel.name, "Measurement branches with additional output details"
    )
    view.set_port_tunnels((wider,), connections)
    assert badge.sceneBoundingRect().width() > rect.width()
    for item in view._connections:
        _assert_clear(item.path(), (badge.sceneBoundingRect(),))
    _assert_no_shared_runs([item.path() for item in view._connections])

    view.set_port_tunnels((), connections)
    assert not badge.isVisible()
    if mode == "update":
        assert original_paths == [item.path() for item in view._connections]


def test_direct_tunnel_source_wire_keeps_its_intentional_badge_contact(qtbot):
    view, connection = _build_view(qtbot, QPointF(0, 20), QPointF(750, 20))
    pipeline = PrototypePipeline()
    tunnel = pipeline.add_output_tunnel("Raw output", "input", 0)
    original = connection.path()

    view.set_port_tunnels((tunnel,), ())

    badge = view._proxies["input"].output_ports[0]._tunnel_badge
    assert badge.isVisible()
    assert connection.path() == original
    assert _route_collision_penalty(
        _flatten_wire_path(connection.path()), (badge.sceneBoundingRect(),)
    )


def test_moving_long_tunnel_badge_repairs_wire_outside_its_parent_card_area(qtbot):
    view, connection = _build_view(qtbot)
    pipeline = PrototypePipeline()
    tunnel = pipeline.add_output_tunnel("Remote tunnel " * 35, "gaussian", 0)
    view.set_port_tunnels((tunnel,), ())
    original = connection.path()
    middle = original.pointAtPercent(0.5)
    proxy = view._proxies["gaussian"]
    badge = proxy.output_ports[0]._tunnel_badge

    view.center_node_on("gaussian", QPointF(-1500, middle.y()))

    margin = view.WIRE_OBSTACLE_MARGIN * 2
    assert not view._connection_route_rect(connection).intersects(
        proxy.sceneBoundingRect().adjusted(-margin, -margin, margin, margin)
    )
    assert _route_collision_penalty(
        _flatten_wire_path(original), (badge.sceneBoundingRect(),)
    )
    assert connection.path() != original
    _assert_connection_endpoints(connection)
    _assert_clear(connection.path(), (badge.sceneBoundingRect(),))
