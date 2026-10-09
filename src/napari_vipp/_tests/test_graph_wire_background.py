from __future__ import annotations

import threading
from dataclasses import FrozenInstanceError

import pytest
from qtpy.compat import isalive
from qtpy.QtCore import QPointF, QTimer

import napari_vipp._graph as graph
from napari_vipp._graph import PipelineGraphView
from napari_vipp.core.pipeline import GraphConnection, PrototypePipeline


def _simple_graph(qtbot):
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    qtbot.addWidget(view)
    connections = (
        GraphConnection("input", "gaussian"),
        GraphConnection("input", "threshold"),
    )
    positions = {
        "input": QPointF(0, 300),
        "gaussian": QPointF(700, 0),
        "threshold": QPointF(700, 600),
    }
    view.build_graph(pipeline.nodes.values(), connections, positions=positions)
    return view, pipeline, connections, positions


def _wait_for_routes(qtbot, view):
    qtbot.waitUntil(lambda: not view.wire_routing_pending, timeout=15000)


@pytest.fixture
def paused_worker(monkeypatch):
    """Pause the first real worker without holding up the Qt event loop."""
    entered = threading.Event()
    release = threading.Event()
    workers = []
    complete_stale = False
    original = graph._WireRoutingWorker.run

    def run(worker):
        workers.append(worker)
        if len(workers) == 1:
            if complete_stale:
                # Model work that completed just before cancellation became
                # observable. The GUI must reject its completed old routes.
                worker.cancelled = lambda: False
            entered.set()
            assert release.wait(10), "Test did not release its routing worker."
        original(worker)

    def allow_stale_completion():
        nonlocal complete_stale
        complete_stale = True

    monkeypatch.setattr(graph._WireRoutingWorker, "run", run)
    yield entered, release, workers, allow_stale_completion
    release.set()


def test_background_wire_search_leaves_qt_heartbeat_and_endpoint_preview_live(
    qtbot, paused_worker,
):
    view, _pipeline, _connections, _positions = _simple_graph(qtbot)
    entered, release, workers, _allow_stale = paused_worker
    ticks = []
    timer = QTimer(view)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(True))
    timer.start()
    view.set_background_wire_routing(True)
    try:
        qtbot.waitUntil(entered.is_set)
        assert view.wire_routing_pending
        initial_ticks = len(ticks)
        qtbot.waitUntil(lambda: len(ticks) >= initial_ticks + 4)
        starts = {"input": QPointF(view._proxies["input"].pos())}
        revision = view.route_revision
        view._move_selected_nodes_during_drag(starts, QPointF(125, 80))
        assert view.route_revision > revision
        assert workers[0].cancel_event.is_set()
        for wire in view._connections:
            assert wire.path().pointAtPercent(0) == wire.source.port_scene_pos(
                "output", wire.source_port,
            )
        view._finish_selected_node_drag(starts)
    finally:
        release.set()
        timer.stop()
        _wait_for_routes(qtbot, view)


def test_drag_after_completed_full_layout_updates_every_pointer_move(qtbot):
    view, _pipeline, _connections, _positions = _simple_graph(qtbot)
    view.set_background_wire_routing(True)
    _wait_for_routes(qtbot, view)
    applied_revision = view.route_revision
    assert view._wire_routing_applied_revision == applied_revision
    assert view._wire_routing_applied_full
    starts = {"input": QPointF(view._proxies["input"].pos())}
    for delta in (QPointF(30, 20), QPointF(75, 50), QPointF(140, 85)):
        previous_revision = view.route_revision
        view._move_selected_nodes_during_drag(starts, delta)
        assert view.route_revision > previous_revision
        assert view.wire_routing_pending
        for wire in view._connections:
            assert wire.path().pointAtPercent(0) == wire.source.port_scene_pos(
                "output", wire.source_port,
            )
    view._finish_selected_node_drag(starts)
    _wait_for_routes(qtbot, view)
    assert view._wire_routing_applied_full
    assert view._wire_routing_applied_revision == view.route_revision


@pytest.mark.parametrize("edit", ("move", "drag", "delete", "rebuild"))
def test_completed_stale_routes_never_replace_current_graph(
    qtbot, paused_worker, edit,
):
    view, pipeline, connections, positions = _simple_graph(qtbot)
    entered, release, workers, allow_stale = paused_worker
    allow_stale()
    view.set_background_wire_routing(True)
    try:
        qtbot.waitUntil(entered.is_set)
        first = workers[0]
        observed = []

        def finished(result):
            observed.append((
                result,
                tuple((
                    graph._wire_connection_order(wire), wire.path(),
                    wire.target.port_scene_pos("input", wire.target_port),
                ) for wire in view._connections),
            ))

        first.signals.finished.connect(finished)
        if edit == "move":
            view._proxies["threshold"].setPos(QPointF(950, 950))
        elif edit == "drag":
            starts = {"threshold": QPointF(view._proxies["threshold"].pos())}
            view._move_selected_nodes_during_drag(starts, QPointF(250, 350))
        elif edit == "delete":
            view.delete_connection_item(view._connections[-1])
        else:
            replacement = {key: point + QPointF(450, 300)
                           for key, point in positions.items()}
            view.build_graph(pipeline.nodes.values(), connections,
                             positions=replacement)
        expected_keys = tuple(graph._wire_connection_order(wire)
                              for wire in view._connections)
        release.set()
        qtbot.waitUntil(lambda: bool(observed))
        result, displayed = observed[0]
        assert result.routes and not result.cancelled
        assert result.generation != view._wire_routing_generation
        assert tuple(key for key, _path, _end in displayed) == expected_keys
        for _key, path, current_end in displayed:
            assert path.pointAtPercent(1) == current_end
        _wait_for_routes(qtbot, view)
        assert tuple(graph._wire_connection_order(wire)
                     for wire in view._connections) == expected_keys
        assert view._wire_routing_applied_revision == view.route_revision
        if edit == "drag":
            # The old full result was rejected before releasing the pointer.
            assert not view._wire_routing_applied_full
            view._finish_selected_node_drag(starts)
            _wait_for_routes(qtbot, view)
            assert view._wire_routing_applied_full
    finally:
        release.set()
        _wait_for_routes(qtbot, view)


def test_repeated_drag_edits_cancel_one_worker_and_coalesce_to_final_geometry(
    qtbot, paused_worker,
):
    view, _pipeline, _connections, _positions = _simple_graph(qtbot)
    entered, release, workers, _allow_stale = paused_worker
    view.set_background_wire_routing(True)
    try:
        qtbot.waitUntil(entered.is_set)
        starts = {"input": QPointF(view._proxies["input"].pos())}
        for index in range(1, 8):
            view._move_selected_nodes_during_drag(starts, QPointF(20 * index, index))
        view._finish_selected_node_drag(starts)
        assert len(workers) == 1
        assert workers[0].cancel_event.is_set()
        assert view._wire_routing_pending_revision == view.route_revision
        assert view._wire_routing_pending_mode is True
        release.set()
        _wait_for_routes(qtbot, view)
        assert len(workers) == 2
        assert workers[-1].revision == view.route_revision
        assert workers[-1].obstacle_aware
        for wire in view._connections:
            assert wire.path().pointAtPercent(0) == wire.source.port_scene_pos(
                "output", wire.source_port,
            )
        assert view._wire_routing_applied_revision == view.route_revision
    finally:
        release.set()
        _wait_for_routes(qtbot, view)


def _parity_graph(qtbot, mode, *, background):
    pipeline = PrototypePipeline()
    source = pipeline.add_node("skeleton_graph_tables")
    targets = [pipeline.add_node("merge_tables") for _ in range(3)]
    blocker = pipeline.nodes["gaussian"]
    nodes = [source, *targets, blocker]
    connections = [GraphConnection(source.id, target.id) for target in targets]
    positions = {
        source.id: QPointF(650, 250),
        targets[0].id: QPointF(0, 0),
        targets[1].id: QPointF(0, 650),
        targets[2].id: QPointF(1200, 500),
        blocker.id: QPointF(1100, 50),
    }
    tunnels = ()
    if mode == "independent-ports":
        connections[1] = GraphConnection(source.id, targets[1].id, source_port=1)
    elif mode == "independent-sources":
        other = pipeline.add_node("skeleton_graph_tables")
        nodes.append(other)
        positions[other.id] = QPointF(650, 850)
        connections[1] = GraphConnection(other.id, targets[1].id)
    elif mode == "tunnel-clearance":
        tunnels = (pipeline.add_output_tunnel(
            "Measurement branches with additional output details", source.id, 1,
        ),)
    view = PipelineGraphView()
    qtbot.addWidget(view)
    view.set_background_wire_routing(background)
    view.build_graph(nodes, connections, positions=positions, output_tunnels=tunnels)
    if background:
        _wait_for_routes(qtbot, view)
    return view


@pytest.mark.parametrize("mode", (
    "shared-output", "independent-ports", "independent-sources", "tunnel-clearance",
))
def test_background_routes_equal_synchronous_card_clear_routes(qtbot, mode):
    synchronous = _parity_graph(qtbot, mode, background=False)
    background = _parity_graph(qtbot, mode, background=True)
    expected = {graph._wire_connection_order(wire): wire.path()
                for wire in synchronous._connections}
    actual = {graph._wire_connection_order(wire): wire.path()
              for wire in background._connections}
    assert actual == expected
    for wire in background._connections:
        rects = background.connection_obstacle_rects(wire) + (
            wire.source.sceneBoundingRect().adjusted(0.75, 0.75, -0.75, -0.75),
            wire.target.sceneBoundingRect().adjusted(0.75, 0.75, -0.75, -0.75),
        )
        assert graph._route_collision_penalty(wire._route_points, rects) == 0
        assert not any(rect.contains(wire.path().pointAtPercent(index / 500))
                       for rect in rects for index in range(501))
        peers, _, _ = background.connection_routing_context(wire)
        assert not graph._wire_lane_conflict(
            wire._route_points, tuple(peer._route_points for peer in peers),
        )


def test_worker_snapshot_has_frozen_owned_numeric_geometry(qtbot):
    view, _pipeline, _connections, _positions = _simple_graph(qtbot)
    snapshot = view._wire_route_snapshot(True)
    original_end = snapshot[-1].end
    view._proxies["threshold"].setPos(QPointF(950, 950))
    assert snapshot[-1].end == original_end
    assert view._wire_route_snapshot(True)[-1].end != original_end
    for request in snapshot:
        assert isinstance(request.start, tuple)
        assert isinstance(request.end, tuple)
        assert all(isinstance(value, float) for value in request.start + request.end)
        assert all(isinstance(rect, tuple) and len(rect) == 4
                   for rect in request.endpoint_rects + request.obstacles)
        with pytest.raises(FrozenInstanceError):
            request.start = (0.0, 0.0)


@pytest.mark.parametrize("action", ("close", "disable-background"))
def test_closing_or_switching_to_sync_cancels_pending_background_routes(
    qtbot, paused_worker, action,
):
    view, _pipeline, _connections, _positions = _simple_graph(qtbot)
    entered, release, workers, _allow_stale = paused_worker
    view.set_background_wire_routing(True)
    try:
        qtbot.waitUntil(entered.is_set)
        results = []
        workers[0].signals.finished.connect(results.append)
        if action == "close":
            view.close()
        else:
            view.set_background_wire_routing(False)
        expected = [wire.path() for wire in view._connections]
        assert workers[0].cancel_event.is_set()
        release.set()
        _wait_for_routes(qtbot, view)
        assert results and results[0].cancelled
        assert [wire.path() for wire in view._connections] == expected
        assert len(workers) == 1
    finally:
        release.set()
        _wait_for_routes(qtbot, view)


def test_destroyed_view_cancels_worker_without_accessing_scene(qtbot, paused_worker):
    view, _pipeline, _connections, _positions = _simple_graph(qtbot)
    entered, release, workers, _allow_stale = paused_worker
    view.set_background_wire_routing(True)
    try:
        qtbot.waitUntil(entered.is_set)
        worker = workers[0]
        results = []
        worker.signals.finished.connect(results.append)
        view.deleteLater()
        qtbot.waitUntil(lambda: not isalive(view))
        assert worker.shutdown.is_set()
        release.set()
        qtbot.waitUntil(lambda: bool(results))
        assert results[0].cancelled
        assert not results[0].routes
        assert not results[0].error
    finally:
        release.set()
