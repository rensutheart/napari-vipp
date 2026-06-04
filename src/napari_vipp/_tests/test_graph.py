from __future__ import annotations

from qtpy.QtCore import QPoint, QPointF, Qt

from napari_vipp._graph import PipelineGraphView
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


def test_pin_button_only_shows_for_mask_nodes(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    assert not view._cards["input"].pin_button.isVisible()
    assert not view._cards["gaussian"].pin_button.isVisible()
    assert view._cards["threshold"].pin_button.isVisible()


def test_graph_cards_use_category_colors(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    gaussian = view._cards["gaussian"]

    assert gaussian._category_color == category_color("Filtering")
    assert gaussian._category_tint == category_tint("Filtering")


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
