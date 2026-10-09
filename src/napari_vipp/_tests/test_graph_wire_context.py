from __future__ import annotations

import pytest
from qtpy.QtCore import QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QPainterPath

from napari_vipp._graph import PipelineGraphView
from napari_vipp.core.pipeline import PrototypePipeline


def _build_view(qtbot):
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    return view, pipeline


def _set_horizontal_path(connection, y, *, start_x=-200, end_x=900):
    path = QPainterPath(QPointF(start_x, y))
    path.lineTo(QPointF(end_x, y))
    connection.setPath(path)


@pytest.mark.parametrize("zoom", (20, 100, 400))
def test_wire_right_click_uses_screen_tolerance_and_requests_tunnel(
    qtbot, monkeypatch, zoom,
):
    view, pipeline = _build_view(qtbot)
    view.show()
    qtbot.waitExposed(view)
    view.set_zoom_percent(zoom)
    view.scene.setSceneRect(QRectF(-3000, -3000, 6000, 6000))
    connection = view._connections[0]
    _set_horizontal_path(connection, 1000)
    _set_horizontal_path(view._connections[1], 1800)
    view.centerOn(QPointF(377, 1000))
    view.select_node("threshold")
    requests = []
    menu_labels = []
    view.connection_tunnel_requested.connect(requests.append)

    def choose_tunnel(menu, _pos):
        menu_labels[:] = [action.text() for action in menu.actions()]
        return next(
            action for action in menu.actions()
            if action.text() == "Replace with tunnel..."
        )

    monkeypatch.setattr("napari_vipp._graph._exec_menu", choose_tunnel)
    wire_point = view.mapFromScene(QPointF(377, 1000))
    point = wire_point + QPoint(0, 7)
    original_connections = tuple(pipeline.connections)

    qtbot.mouseClick(view.viewport(), Qt.RightButton, pos=point)

    assert menu_labels == [
        "Info", "Insert node here...", "Replace with tunnel...", "Delete"
    ]
    assert requests == [("input", "gaussian", 0, 0)]
    assert connection.isSelected()
    assert view.selected_node_ids() == ()
    assert tuple(pipeline.connections) == original_connections
    assert view._connection_at_view_pos(wire_point + QPoint(0, 12)) is None


def test_right_click_picks_nearest_wire_when_hit_areas_intersect(qtbot, monkeypatch):
    view, _pipeline = _build_view(qtbot)
    view.show()
    qtbot.waitExposed(view)
    view.scene.setSceneRect(QRectF(-3000, -3000, 6000, 6000))
    first, second = view._connections
    _set_horizontal_path(first, 1000)
    _set_horizontal_path(second, 1008)
    view.centerOn(QPointF(377, 1000))
    requests = []
    view.connection_tunnel_requested.connect(requests.append)
    monkeypatch.setattr(
        "napari_vipp._graph._exec_menu",
        lambda menu, _pos: next(
            action for action in menu.actions()
            if action.text() == "Replace with tunnel..."
        ),
    )
    point = view.mapFromScene(QPointF(377, 1002))
    # The later wire wins scene stacking, but the first wire is visibly nearer.
    assert view.itemAt(point) is second

    qtbot.mouseClick(view.viewport(), Qt.RightButton, pos=point)

    assert requests == [(first.source_id, first.target_id, 0, 0)]
    assert first.isSelected()
    assert not second.isSelected()


@pytest.mark.parametrize("kind", ("node", "port"))
def test_wire_hit_area_preserves_node_and_port_context(qtbot, monkeypatch, kind):
    view, _pipeline = _build_view(qtbot)
    view.show()
    qtbot.waitExposed(view)
    proxy = view._proxies["gaussian"]
    port = proxy.output_ports[0]
    scene_point = (
        proxy.sceneBoundingRect().center()
        if kind == "node" else port.sceneBoundingRect().center()
    )
    _set_horizontal_path(view._connections[0], scene_point.y())
    point = view.mapFromScene(scene_point)
    requested = []
    view.port_context_requested.connect(
        lambda *_args: requested.append("port")
    )
    monkeypatch.setattr(
        view, "_show_node_context_menu", lambda *_args: requested.append("node")
    )
    monkeypatch.setattr(
        "napari_vipp._graph._exec_menu",
        lambda *_args: pytest.fail("wire menu displaced node or port context"),
    )

    qtbot.mouseClick(view.viewport(), Qt.RightButton, pos=point)

    assert requested == [kind]


def test_wire_distance_covers_long_segments_and_corner_endpoints(qtbot):
    view, _pipeline = _build_view(qtbot)
    connection = view._connections[0]
    path = QPainterPath(QPointF(0, 0))
    path.lineTo(QPointF(10000, 0))
    path.lineTo(QPointF(10000, 10000))
    connection.setPath(path)

    assert view._distance_to_connection(connection, QPointF(101, 3)) == 9
    assert view._distance_to_connection(connection, QPointF(-3, 4)) == 25
    assert view._distance_to_connection(connection, QPointF(10004, 101)) == 16


def test_replace_with_tunnel_request_preserves_both_port_indices(qtbot, monkeypatch):
    view, pipeline = _build_view(qtbot)
    source = pipeline.add_node("skeleton_graph_tables")
    target = pipeline.add_node("merge_tables")
    view.add_node(source, QPointF(0, 1000))
    view.add_node(target, QPointF(800, 1000))
    view.add_connection(source.id, target.id, target_port=1, source_port=1)
    requests = []
    view.connection_tunnel_requested.connect(requests.append)
    monkeypatch.setattr(
        "napari_vipp._graph._exec_menu",
        lambda menu, _pos: next(
            action for action in menu.actions()
            if action.text() == "Replace with tunnel..."
        ),
    )

    view._connections[-1].show_context_menu(QPointF(500, 1000), QPoint(0, 0))

    assert requests == [(source.id, target.id, 1, 1)]
