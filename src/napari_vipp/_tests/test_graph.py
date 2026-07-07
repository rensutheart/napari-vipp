from __future__ import annotations

from qtpy.QtCore import QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QPainterPath

from napari_vipp._graph import PipelineGraphView, _wire_path
from napari_vipp._theme import category_color, category_tint
from napari_vipp.core.pipeline import PrototypePipeline


def _build_view() -> tuple[PipelineGraphView, PrototypePipeline]:
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    return view, pipeline


def test_graph_node_can_be_dragged_with_mouse(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    proxy = view._proxies["gaussian"]
    before = QPointF(proxy.pos())
    connection = view._connections[0]
    path_before = connection.path().elementAt(3).x

    start = view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(90, 45)
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    assert proxy.pos() != before
    assert connection.path().elementAt(3).x != path_before


def test_dragging_node_is_translucent_until_released(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    proxy = view._proxies["gaussian"]
    start = view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(90, 45)

    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)

    assert 0.0 < proxy.opacity() < 1.0

    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    assert proxy.opacity() == 1.0


def test_dragging_loose_node_does_not_reroute_wires_until_release(qtbot):
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        positions={
            "input": QPointF(0, 20),
            "gaussian": QPointF(330, 360),
            "threshold": QPointF(660, 20),
        },
    )
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    view.add_connection("input", "threshold")
    node = pipeline.add_node("median_filter")
    view.add_node(node, QPointF(330, -260))
    proxy = view._proxies[node.id]
    connection = next(
        item
        for item in view._connections
        if item.source_id == "input" and item.target_id == "threshold"
    )
    path_before = QPainterPath(connection.path())
    scene_target = connection.path().pointAtPercent(0.5)
    start = view.mapFromScene(proxy.sceneBoundingRect().center())
    end = view.mapFromScene(scene_target)
    view.set_connection_insert_validator(
        lambda _operation_id, _key: ("incompatible", "Not for this test.")
    )

    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)

    assert _paths_equal(connection.path(), path_before)
    assert _path_intersects_rect(
        connection.path(),
        proxy.sceneBoundingRect().adjusted(-4, -4, 4, 4),
    )

    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    assert not _paths_equal(connection.path(), path_before)
    assert not _path_intersects_rect(
        connection.path(),
        proxy.sceneBoundingRect().adjusted(-4, -4, 4, 4),
    )


def test_clicking_node_selects_it_without_inspect_button(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    selected = []
    view.node_selected.connect(selected.append)

    card = view._cards["gaussian"]
    preview_center = QPointF(card.preview.geometry().center())
    start = view.mapFromScene(
        view._proxies["gaussian"].mapToScene(preview_center)
    )
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=start)

    assert selected[-1] == "gaussian"
    assert not hasattr(card, "inspect_button")


def test_pin_button_is_not_shown_on_node_cards(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    assert not view._cards["input"].pin_button.isVisible()
    assert not view._cards["gaussian"].pin_button.isVisible()
    assert not view._cards["threshold"].pin_button.isVisible()
    assert view._cards["input"]._can_pin
    assert view._cards["gaussian"]._can_pin
    assert view._cards["threshold"]._can_pin


def test_node_context_menu_emits_requested_action(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    labels = []

    def fake_exec(menu, _pos):
        labels[:] = [
            action.text()
            for action in menu.actions()
            if not action.isSeparator()
        ]
        return next(action for action in menu.actions() if action.text() == "Delete")

    deleted = []
    view.node_delete_requested.connect(deleted.append)
    monkeypatch.setattr("napari_vipp._graph._exec_menu", fake_exec)

    view._show_node_context_menu("threshold", QPoint(0, 0))

    assert labels == ["Delete", "Inspect Code", "Duplicate Node", "Add note", "Pin"]
    assert deleted == ["threshold"]


def test_node_context_menu_can_request_attached_note(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    def fake_exec(menu, _pos):
        return next(action for action in menu.actions() if action.text() == "Add note")

    requested = []
    view.node_note_requested.connect(requested.append)
    monkeypatch.setattr("napari_vipp._graph._exec_menu", fake_exec)

    view._show_node_context_menu("threshold", QPoint(0, 0))

    assert requested == ["threshold"]


def test_selecting_graph_note_clears_node_selection(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    view.add_note(
        "note_1",
        "Check blur",
        QPointF(180, 60),
        attached_node="gaussian",
    )
    view.select_node("gaussian")
    assert view._proxies["gaussian"].isSelected()

    view.select_note("note_1")

    assert view._notes["note_1"].isSelected()
    assert not view._proxies["gaussian"].isSelected()
    assert not view._cards["gaussian"]._selected


def test_node_context_menu_uses_unpin_label_for_pinned_nodes(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.set_pinned_node("threshold")

    def fake_exec(menu, _pos):
        return next(action for action in menu.actions() if action.text() == "Unpin")

    pinned = []
    view.pin_requested.connect(pinned.append)
    monkeypatch.setattr("napari_vipp._graph._exec_menu", fake_exec)

    view._show_node_context_menu("threshold", QPoint(0, 0))

    assert pinned == ["threshold"]
    assert "border: 4px solid #facc15" in view._cards["threshold"].styleSheet()


def test_connection_context_menu_can_request_insert(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    labels = []

    def fake_exec(menu, _pos):
        labels[:] = [
            action.text()
            for action in menu.actions()
            if not action.isSeparator()
        ]
        return next(
            action
            for action in menu.actions()
            if action.text() == "Insert node here..."
        )

    class FakeContextMenuEvent:
        def screenPos(self):
            return QPoint(0, 0)

        def scenePos(self):
            return QPointF(123, 45)

    requests = []
    view.connection_insert_requested.connect(
        lambda connection_key, position: requests.append(
            (tuple(connection_key), QPointF(position))
        )
    )
    monkeypatch.setattr("napari_vipp._graph._exec_menu", fake_exec)

    view._connections[0].contextMenuEvent(FakeContextMenuEvent())

    assert labels == ["Info", "Insert node here...", "Delete"]
    assert requests == [(("input", "gaussian", 0, 0), QPointF(123, 45))]


def test_releasing_loose_node_on_connection_requests_splice(qtbot):
    view, pipeline = _build_view()
    qtbot.addWidget(view)
    node = pipeline.add_node("median_filter")
    view.add_node(node, QPointF(180, 180))
    connection = view._connections[0]
    scene_pos = connection.path().pointAtPercent(0.5)
    old_pos = QPointF(view.node_position(node.id))
    new_pos = old_pos + QPointF(10, 15)
    requests = []

    view.set_connection_insert_validator(lambda _operation_id, _key: ("full", "drop"))
    view.node_splice_requested.connect(
        lambda node_id, key, old, new: requests.append(
            (node_id, tuple(key), QPointF(old), QPointF(new))
        )
    )

    view.update_existing_node_insert_preview(node.id, scene_pos)

    assert view._highlighted_connection is connection
    assert view._highlighted_connection_state == "full"
    assert view.release_existing_node_insert(node.id, old_pos, new_pos, scene_pos)
    assert requests == [
        (node.id, ("input", "gaussian", 0, 0), old_pos, new_pos)
    ]
    assert view._highlighted_connection is None


def test_releasing_connected_node_on_connection_does_not_request_splice(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    connection = view._connections[0]
    scene_pos = connection.path().pointAtPercent(0.5)
    old_pos = QPointF(view.node_position("gaussian"))
    requests = []

    view.node_splice_requested.connect(
        lambda node_id, key, old, new: requests.append((node_id, tuple(key)))
    )

    assert not view.release_existing_node_insert(
        "gaussian",
        old_pos,
        old_pos + QPointF(10, 10),
        scene_pos,
    )
    assert requests == []


def test_graph_cards_use_category_colors(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    gaussian = view._cards["gaussian"]

    assert gaussian._category_color == category_color("Filtering")
    assert gaussian._category_tint == category_tint("Filtering")


def test_graph_zoom_can_be_set_and_reset(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    initial = view.transform().m11()
    assert view.zoom_percent == PipelineGraphView.DEFAULT_ZOOM

    view.set_zoom_percent(150)

    assert view.zoom_percent == 150
    assert view.transform().m11() > initial
    assert abs(view.transform().m11() / initial - 1.5) < 1e-6

    view.reset_zoom()

    assert view.zoom_percent == PipelineGraphView.DEFAULT_ZOOM
    assert view.transform().m11() == initial


def test_graph_view_can_apply_absolute_node_positions(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    assert view.apply_node_positions(
        {
            "input": QPointF(40, 50),
            "gaussian": (360, 120),
            "missing": (1, 1),
        }
    )

    assert view.node_position("input") == QPointF(40, 50)
    assert view.node_position("gaussian") == QPointF(360, 120)
    assert not view.apply_node_positions({"input": QPointF(40, 50)})


def test_connection_routes_around_intermediate_node(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    view.add_connection("input", "threshold")
    connection = next(
        item
        for item in view._connections
        if item.source_id == "input" and item.target_id == "threshold"
    )
    obstacle = view.node_scene_rect("gaussian")
    assert obstacle is not None
    margin = view.WIRE_OBSTACLE_MARGIN
    inflated = obstacle.adjusted(-margin, -margin, margin, margin)

    assert not _path_intersects_rect(connection.path(), inflated)


def test_routed_connection_keeps_port_tangents(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    view.add_connection("input", "threshold")
    connection = next(
        item
        for item in view._connections
        if item.source_id == "input" and item.target_id == "threshold"
    )
    start = connection.source.port_scene_pos("output", connection.source_port)
    end = connection.target.port_scene_pos("input", connection.target_port)
    path = connection.path()

    assert path.pointAtPercent(0.01).x() > start.x()
    assert path.pointAtPercent(0.99).x() < end.x()


def test_close_port_routing_does_not_create_horizontal_loop():
    start = QPointF(0, 0)
    end = QPointF(45, 65)
    obstacle = QRectF(20, 16, 12, 34)

    path = _wire_path(start, end, obstacles=(obstacle,))
    points = [path.pointAtPercent(index / 100.0) for index in range(101)]

    assert min(point.x() for point in points) >= start.x() - 1.0
    assert max(point.x() for point in points) <= end.x() + 1.0
    assert path.pointAtPercent(0.01).x() > start.x()
    assert path.pointAtPercent(0.99).x() < end.x()


def test_local_obstacle_uses_compact_curve_instead_of_deep_u():
    start = QPointF(0, 100)
    end = QPointF(170, 118)
    obstacle = QRectF(55, 45, 90, 160)

    path = _wire_path(start, end, obstacles=(obstacle,))
    points = [path.pointAtPercent(index / 100.0) for index in range(101)]

    assert max(point.y() for point in points) <= end.y() + 1.0
    assert min(point.x() for point in points) >= start.x() - 1.0
    assert max(point.x() for point in points) <= end.x() + 1.0


def test_adding_node_over_existing_wire_reroutes_connection(qtbot):
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        positions={
            "input": QPointF(0, 20),
            "gaussian": QPointF(330, 360),
            "threshold": QPointF(660, 20),
        },
    )
    qtbot.addWidget(view)
    view.add_connection("input", "threshold")
    connection = next(
        item
        for item in view._connections
        if item.source_id == "input" and item.target_id == "threshold"
    )

    inserted = pipeline.add_node("median_filter")
    view.add_node(inserted, QPointF(330, 20))
    obstacle = view.node_scene_rect(inserted.id)
    assert obstacle is not None
    margin = view.WIRE_OBSTACLE_MARGIN
    inflated = obstacle.adjusted(-margin, -margin, margin, margin)

    assert not _path_intersects_rect(connection.path(), inflated)


def test_ports_grow_for_hover_and_pending_connection_feedback(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    source = view._proxies["input"].output_port
    target = view._proxies["gaussian"].input_port

    assert source is not None
    assert target is not None
    assert source.rect().width() == source.radius * 2

    source.set_active(True)

    assert source.rect().width() == source.target_radius * 2

    source.set_active(False)
    view.begin_connection(source, source.mapToScene(QPointF(0, 0)))
    view.update_pending_connection(target.mapToScene(QPointF(0, 0)), dragging=True)

    assert source._active
    assert source.rect().width() == source.target_radius * 2
    assert target._drop_state == "compatible"
    assert target.rect().width() == target.target_radius * 2

    view._cancel_pending_connection()

    assert not source._active
    assert target._drop_state is None


def _path_intersects_rect(path: QPainterPath, rect: QRectF) -> bool:
    return any(
        rect.contains(path.pointAtPercent(index / 120.0))
        for index in range(121)
    )


def _paths_equal(first: QPainterPath, second: QPainterPath) -> bool:
    if first.elementCount() != second.elementCount():
        return False
    for index in range(first.elementCount()):
        first_element = first.elementAt(index)
        second_element = second.elementAt(index)
        if (
            abs(first_element.x - second_element.x) > 1e-6
            or abs(first_element.y - second_element.y) > 1e-6
        ):
            return False
    return True


def test_clear_border_input_accepts_mask_and_labels_but_rejects_image(qtbot):
    pipeline = PrototypePipeline()
    labels = pipeline.add_node("label_connected_components")
    cleared = pipeline.add_node("clear_border_objects")
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)

    target = view._proxies[cleared.id].input_port
    assert target is not None

    states = {}
    for name, source_id in (
        ("image", "input"),
        ("mask", "threshold"),
        ("labels", labels.id),
    ):
        source = view._proxies[source_id].output_port
        assert source is not None
        view.begin_connection(source, source.mapToScene(QPointF(0, 0)))
        view.update_pending_connection(
            target.mapToScene(QPointF(0, 0)),
            dragging=True,
        )
        states[name] = target._drop_state
        view._cancel_pending_connection()

    assert states == {
        "image": "incompatible",
        "mask": "compatible",
        "labels": "compatible",
    }


def test_dragging_node_keeps_viewport_stationary(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    proxy = view._proxies["input"]
    scene_rect_before = view.scene.sceneRect()
    h_scroll_before = view.horizontalScrollBar().value()
    v_scroll_before = view.verticalScrollBar().value()
    center_before = view.mapToScene(view.viewport().rect().center())

    start = view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(0, -90)
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    center_after = view.mapToScene(view.viewport().rect().center())
    center_delta = center_after - center_before

    assert view.scene.sceneRect() == scene_rect_before
    assert view.horizontalScrollBar().value() == h_scroll_before
    assert view.verticalScrollBar().value() == v_scroll_before
    assert center_delta.manhattanLength() < 0.01


def test_delete_key_requests_selected_node_deletion(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    deleted = []
    view.node_delete_requested.connect(deleted.append)
    view.select_node("gaussian")

    qtbot.keyClick(view, Qt.Key_Delete)

    assert deleted == ["gaussian"]


def test_delete_key_prefers_selected_connection_over_selected_node(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    deleted_nodes = []
    removed_connections = []
    view.node_delete_requested.connect(deleted_nodes.append)
    view.connection_removed.connect(
        lambda source, target, port: removed_connections.append(
            (source, target, port)
        )
    )

    view._proxies["gaussian"].setSelected(True)
    view._connections[0].setSelected(True)

    qtbot.keyClick(view, Qt.Key_Delete)

    assert deleted_nodes == []
    assert removed_connections == [("input", "gaussian", 0)]
    assert "gaussian" in view._proxies
    assert len(view._connections) == 1


def test_removing_node_removes_related_connections(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    view.remove_node("gaussian")

    assert "gaussian" not in view._proxies
    assert "gaussian" not in view._cards
    assert not view._connections


def test_scene_expands_when_node_moves_near_right_edge(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    scene_rect_before = view.scene.sceneRect()
    proxy = view._proxies["threshold"]
    move_x = (
        scene_rect_before.right()
        - proxy.boundingRect().width()
        - (PipelineGraphView.SCENE_EDGE_MARGIN * 0.4)
    )
    proxy.setPos(QPointF(move_x, proxy.pos().y()))

    scene_rect_after = view.scene.sceneRect()
    assert scene_rect_after.right() > scene_rect_before.right()


def test_scene_expands_when_viewport_reaches_edge(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    scene_rect_before = view.scene.sceneRect()
    near_edge_rect = QRectF(
        scene_rect_before.right() - 32.0,
        scene_rect_before.center().y() - 32.0,
        64.0,
        64.0,
    )
    view._ensure_scene_space_for_rect(near_edge_rect)

    scene_rect_after = view.scene.sceneRect()
    assert scene_rect_after.right() > scene_rect_before.right()
