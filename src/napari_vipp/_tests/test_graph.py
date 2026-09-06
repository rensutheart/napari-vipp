from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QMimeData, QPoint, QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QImage, QPainter, QPainterPath, QPalette
from qtpy.QtWidgets import QApplication, QVBoxLayout, QWidget

from napari_vipp._graph import (
    BYPASSED_NODE_OPACITY,
    BYPASSED_NODE_OUTLINE,
    BYPASSED_NODE_PASS_THROUGH,
    ISOLATED_TUNING_ACCENT,
    OPERATION_MIME,
    STALE_EXECUTION_ACCENT,
    ComputeBadgeKind,
    PipelineGraphView,
    PortLabelMode,
    _wire_path,
)
from napari_vipp._theme import (
    category_color,
    category_foreground,
    category_tint,
    graph_theme,
)
from napari_vipp.core.pipeline import (
    EXECUTION_BLOCKED,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
)


def _build_view() -> tuple[PipelineGraphView, PrototypePipeline]:
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    return view, pipeline


def _graph_palette(*, dark: bool) -> QPalette:
    palette = QPalette()
    surface = "#20242b" if dark else "#ffffff"
    window = "#151922" if dark else "#f5f7fb"
    text = "#f3f4f6" if dark else "#172033"
    palette.setColor(QPalette.Base, QColor(surface))
    palette.setColor(QPalette.Window, QColor(window))
    palette.setColor(QPalette.Text, QColor(text))
    palette.setColor(QPalette.Button, QColor(surface))
    palette.setColor(QPalette.ButtonText, QColor(text))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#64748b"))
    return palette


def _dominant_widget_color(widget) -> str:
    image = widget.grab().toImage().convertToFormat(QImage.Format_RGBA8888)
    counts: dict[str, int] = {}
    for y in range(3, max(image.height() - 3, 3)):
        for x in range(3, max(image.width() - 3, 3)):
            name = image.pixelColor(x, y).name()
            counts[name] = counts.get(name, 0) + 1
    assert counts
    return max(counts, key=counts.get)


def test_graph_canvas_and_items_follow_runtime_palette_changes(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()

    view.setPalette(_graph_palette(dark=False))
    view._apply_palette_theme()

    card = view._cards["gaussian"]
    connection = view._connections[0]
    port = view._proxies["gaussian"].input_ports[0]
    assert view.backgroundBrush().color().name() == "#f2f5f9"
    assert "background: #ffffff" in card.styleSheet()
    assert "color: #172033" in card.styleSheet()
    assert "background: #e8edf4" in card.preview.styleSheet()
    assert connection.pen().color().name() == "#64748b"
    assert port.pen().color().name() == "#ffffff"

    view.setPalette(_graph_palette(dark=True))
    view._apply_palette_theme()

    assert view.backgroundBrush().color().name() == "#151922"
    assert "background: #20242b" in card.styleSheet()
    assert "color: #f3f4f6" in card.styleSheet()
    assert "background: #111827" in card.preview.styleSheet()
    assert "background-color: #334155" in card.calculate_button.styleSheet()
    assert "color: #f8fafc" in card.calculate_button.styleSheet()
    assert connection.pen().color().name() == "#8aa0c8"
    assert port.pen().color().name() == "#111827"


def test_card_added_after_dark_graph_theme_ignores_light_application_palette(qtbot):
    application = QApplication.instance()
    assert application is not None
    original_palette = QPalette(application.palette())

    try:
        application.setPalette(_graph_palette(dark=False))
        view, pipeline = _build_view()
        qtbot.addWidget(view)
        view.setPalette(_graph_palette(dark=True))
        view._apply_palette_theme()

        node = pipeline.add_node("binary_threshold")
        view.add_node(node, QPointF(990, 20))

        card = view._cards[node.id]
        assert card.palette().color(QPalette.Base).name() == "#20242b"
        assert "background: #20242b" in card.styleSheet()
        assert "color: #f3f4f6" in card.styleSheet()
        assert "background: #111827" in card.preview.styleSheet()
        assert "background-color: #334155" in card.calculate_button.styleSheet()
        assert "color: #f8fafc" in card.calculate_button.styleSheet()
        assert card._bypass_overlay._graph_theme.is_dark
        assert card.processing_badge._graph_theme.is_dark
    finally:
        application.setPalette(original_palette)
        QApplication.processEvents()


def test_card_added_after_light_graph_theme_ignores_dark_application_palette(qtbot):
    application = QApplication.instance()
    assert application is not None
    original_palette = QPalette(application.palette())

    try:
        application.setPalette(_graph_palette(dark=True))
        view, pipeline = _build_view()
        qtbot.addWidget(view)
        view.setPalette(_graph_palette(dark=False))
        view._apply_palette_theme()

        node = pipeline.add_node("binary_threshold")
        view.add_node(node, QPointF(990, 20))

        card = view._cards[node.id]
        assert card.palette().color(QPalette.Base).name() == "#ffffff"
        assert "background: #ffffff" in card.styleSheet()
        assert "color: #172033" in card.styleSheet()
        assert "background: #e8edf4" in card.preview.styleSheet()
        assert "background-color: #e2e8f0" in card.calculate_button.styleSheet()
        assert "color: #172033" in card.calculate_button.styleSheet()
        assert not card._bypass_overlay._graph_theme.is_dark
        assert not card.processing_badge._graph_theme.is_dark
    finally:
        application.setPalette(original_palette)
        QApplication.processEvents()


def test_manual_card_button_follows_real_napari_qss_theme(qtbot):
    from napari._qt.qt_resources import get_stylesheet

    host = QWidget()
    layout = QVBoxLayout(host)
    view, _pipeline = _build_view()
    layout.addWidget(view)
    qtbot.addWidget(host)
    host.show()

    host.setStyleSheet(
        get_stylesheet("light", extra_variables={"font_size": "9pt"})
    )
    qtbot.waitUntil(
        lambda: view.palette().color(QPalette.Base).lightnessF() > 0.5
    )
    view._apply_palette_theme()
    card = view._cards["gaussian"]
    card.set_execution_state("not_calculated", manual=True)
    QApplication.processEvents()
    button = card.calculate_button
    assert button.palette().color(QPalette.Button).lightnessF() > 0.5
    assert button.palette().color(QPalette.ButtonText).lightnessF() < 0.5
    assert _dominant_widget_color(button) == "#e2e8f0"

    card.set_execution_state(EXECUTION_BLOCKED, manual=True)
    QApplication.processEvents()
    assert _dominant_widget_color(button) == "#e5e7eb"

    host.setStyleSheet(
        get_stylesheet("dark", extra_variables={"font_size": "9pt"})
    )
    qtbot.waitUntil(
        lambda: view.palette().color(QPalette.Base).lightnessF() < 0.5
    )
    view._apply_palette_theme()
    assert button.palette().color(QPalette.Button).lightnessF() < 0.5
    assert button.palette().color(QPalette.ButtonText).lightnessF() > 0.5
    card.set_execution_state("not_calculated", manual=True)
    QApplication.processEvents()
    assert _dominant_widget_color(button) == "#334155"

    card.set_execution_state(EXECUTION_BLOCKED, manual=True)
    QApplication.processEvents()
    assert _dominant_widget_color(button) == "#2b3038"


class _OperationDropEvent:
    """Small binding-neutral operation drag/drop event used by view tests."""

    def __init__(self, operation_id: str, position: QPoint):
        self._mime = QMimeData()
        self._mime.setData(OPERATION_MIME, operation_id.encode())
        self._position = QPointF(position)
        self.accepted = False

    def mimeData(self):  # noqa: N802
        return self._mime

    def position(self):
        return QPointF(self._position)

    def acceptProposedAction(self):  # noqa: N802
        self.accepted = True


def test_node_subtitle_is_elided_but_keeps_complete_binding_tooltip(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    binding = "very-long-acquisition-name-" * 12 + "MAGENTA.ome.tif"

    view.set_node_subtitle("input", binding)

    label = view._cards["input"].subtitle_label
    assert label._full_text == binding
    assert label.toolTip() == binding
    assert label.isVisible()
    assert label.text() != binding
    assert "…" in label.text()


def test_node_card_shows_and_updates_authored_bypass_badge(qtbot):
    pipeline = PrototypePipeline()
    source = pipeline.add_node("input")
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect(source.id, crop.id).success
    pipeline.set_node_execution_mode(crop.id, "bypass")
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    view.setPalette(_graph_palette(dark=True))
    view._apply_palette_theme()

    card = view._cards[crop.id]
    assert card._bypassed
    assert card._compute_badge_kind is ComputeBadgeKind.BYPASSED
    assert card.compute_badge.text() == "Bypassed"
    assert not card.compute_badge.isHidden()
    assert not card._bypass_overlay.isHidden()
    assert card._bypass_overlay.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert card._bypass_overlay.focusPolicy() == Qt.NoFocus
    assert view._proxies[crop.id].opacity() == pytest.approx(
        BYPASSED_NODE_OPACITY
    )
    assert "exact primary input" in card.compute_badge.toolTip()
    outline_pen = card._bypass_outline_pen()
    assert outline_pen.color() == QColor(BYPASSED_NODE_OUTLINE)
    assert outline_pen.style() == Qt.DotLine
    assert outline_pen.widthF() == pytest.approx(3.0)
    assert outline_pen.isCosmetic()
    pass_through_pen = card._bypass_pass_through_pen()
    assert pass_through_pen.color().name() == BYPASSED_NODE_PASS_THROUGH
    assert pass_through_pen.color().alpha() < 255
    assert pass_through_pen.style() == Qt.SolidLine
    assert pass_through_pen.isCosmetic()
    card.resize(card.sizeHint())
    card.show()
    rendered = QImage(card.size(), QImage.Format_ARGB32)
    rendered.fill(QColor("#000000"))
    card.render(rendered)
    outline_rgb = QColor(BYPASSED_NODE_OUTLINE).rgb()
    outline_pixels = sum(
        QColor.fromRgb(rendered.pixel(x, y)).rgb() == outline_rgb
        for y in range(rendered.height())
        for x in range(rendered.width())
    )
    assert outline_pixels > 50
    center_y = rendered.height() // 2
    pass_through_pixels = sum(
        (
            QColor.fromRgb(rendered.pixel(x, y)).green()
            - QColor.fromRgb(rendered.pixel(x, y)).red()
            > 35
        )
        and (
            QColor.fromRgb(rendered.pixel(x, y)).blue()
            - QColor.fromRgb(rendered.pixel(x, y)).red()
            > 35
        )
        for y in range(center_y - 1, center_y + 2)
        for x in range(rendered.width())
    )
    assert pass_through_pixels > rendered.width()

    view.set_node_bypassed(crop.id, False)

    assert not card._bypassed
    assert card.compute_badge.isHidden()
    assert card._bypass_overlay.isHidden()
    assert view._proxies[crop.id].opacity() == pytest.approx(1.0)


def test_bypass_fade_preserves_selected_pinned_and_error_cues(qtbot):
    pipeline = PrototypePipeline()
    source = pipeline.add_node("input")
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect(source.id, crop.id).success
    pipeline.set_node_execution_mode(crop.id, "bypass")
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    view.setPalette(_graph_palette(dark=True))
    view._apply_palette_theme()
    card = view._cards[crop.id]
    proxy = view._proxies[crop.id]

    card.set_selected(True)
    assert "#60a5fa" in card.styleSheet()
    assert proxy.opacity() == pytest.approx(BYPASSED_NODE_OPACITY)

    card.set_pinned(True)
    assert "#facc15" in card.styleSheet()
    assert proxy.opacity() == pytest.approx(BYPASSED_NODE_OPACITY)

    card.set_pinned(False)
    card.set_execution_state("error", manual=True, message="Expected test error")
    assert "#ef4444" in card.styleSheet()
    assert proxy.opacity() == pytest.approx(BYPASSED_NODE_OPACITY)


def test_incrementally_added_bypassed_node_is_faded_immediately(qtbot):
    pipeline = PrototypePipeline()
    source = pipeline.add_node("input")
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect(source.id, crop.id).success
    pipeline.set_node_execution_mode(crop.id, "bypass")

    view.add_node(crop, QPointF(330, -220))

    assert view._cards[crop.id]._bypassed
    assert view._proxies[crop.id].opacity() == pytest.approx(
        BYPASSED_NODE_OPACITY
    )


def test_node_context_menu_toggles_reviewed_bypass_mode(qtbot, monkeypatch):
    pipeline = PrototypePipeline()
    source = pipeline.add_node("input")
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect(source.id, crop.id).success
    pipeline.add_output_tunnel("Crop result", crop.id, 0)
    view = PipelineGraphView()
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        output_tunnels=pipeline.output_tunnel_list(),
    )
    qtbot.addWidget(view)
    requested = []
    menu_states = []
    menu_enabled = []
    view.node_bypass_requested.connect(
        lambda node_id, bypassed: requested.append((node_id, bypassed))
    )

    def fake_exec(menu, _pos):
        action = next(
            action for action in menu.actions() if action.text() == "Bypass node"
        )
        menu_states.append(action.isChecked())
        menu_enabled.append(action.isEnabled())
        return action

    monkeypatch.setattr("napari_vipp._graph._exec_menu", fake_exec)

    view._show_node_context_menu(crop.id, QPoint(0, 0))
    view.set_node_bypassed(crop.id, True)
    view._show_node_context_menu(crop.id, QPoint(0, 0))

    assert menu_states == [False, True]
    assert menu_enabled == [True, True]
    assert requested == [(crop.id, True), (crop.id, False)]


def test_node_context_bypass_is_disabled_at_terminal_but_can_clear(qtbot):
    pipeline = PrototypePipeline()
    source = pipeline.add_node("input")
    median = pipeline.add_node("median_filter")
    assert pipeline.connect(source.id, median.id).success
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    operation = NODE_LIBRARY_BY_ID[median.operation_id]

    visible, enabled, tooltip = view._node_bypass_action_state(
        median.id,
        operation,
        bypassed=False,
    )

    assert visible
    assert not enabled
    assert "no downstream connection or output tunnel" in tooltip.casefold()

    view.set_node_bypassed(median.id, True)
    _visible, enabled, tooltip = view._node_bypass_action_state(
        median.id,
        operation,
        bypassed=True,
    )
    assert enabled
    assert "clear bypass" in tooltip.casefold()


def test_dragging_output_tunnel_badge_requests_source_reroute(qtbot):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    median = pipeline.add_node("median_filter")
    pipeline.add_output_tunnel("Raw", "input", 0)
    assert pipeline.connect_to_tunnel("Raw", "gaussian", 0).success
    view = PipelineGraphView()
    view.resize(980, 520)
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        output_tunnels=pipeline.output_tunnel_list(),
    )
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    requests = []
    view.tunnel_reroute_requested.connect(
        lambda name, node_id, port: requests.append((name, node_id, port))
    )

    source = view._proxies["input"].output_port_at(0)
    target = view._proxies[median.id].output_port_at(0)
    badge = source._tunnel_badge
    start = view.mapFromScene(badge.mapToScene(badge.boundingRect().center()))
    end = view.mapFromScene(target.mapToScene(QPointF(0, 0)))
    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)

    assert view._pending_tunnel_wire is not None
    assert target._drop_state == "compatible"

    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    assert requests == [("Raw", median.id, 0)]
    assert view._pending_tunnel_wire is None
    assert target._drop_state is None


def test_tunnel_drag_marks_cycle_producing_output_incompatible(qtbot):
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    pipeline.add_output_tunnel("Raw", "input", 0)
    assert pipeline.connect_to_tunnel("Raw", "gaussian", 0).success
    view = PipelineGraphView()
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        output_tunnels=pipeline.output_tunnel_list(),
    )
    qtbot.addWidget(view)

    def validate(name, node_id, port_index):
        try:
            pipeline.validate_output_tunnel_reroute(name, node_id, port_index)
        except ValueError as exc:
            return "incompatible", str(exc)
        return "compatible", ""

    view.set_tunnel_reroute_validator(validate)
    requests = []
    messages = []
    view.tunnel_reroute_requested.connect(
        lambda name, node_id, port: requests.append((name, node_id, port))
    )
    view.status_message.connect(messages.append)
    source = view._proxies["input"].output_port_at(0)
    target = view._proxies["threshold"].output_port_at(0)
    target_pos = target.mapToScene(QPointF(0, 0))

    view.begin_tunnel_reroute("Raw", source.mapToScene(QPointF(0, 0)))
    view.update_pending_tunnel_reroute(target_pos, dragging=True)

    assert target._drop_state == "incompatible"
    assert "cycle" in messages[-1]

    view.release_tunnel_reroute(target_pos)

    assert requests == []
    assert pipeline.output_tunnel("Raw").source_id == "input"


def test_center_graph_recovers_far_canvas_position_without_changing_zoom(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    view.set_zoom_percent(175)
    transform_before = view.transform()
    graph_center = view._graph_content_rect().center()

    view.centerOn(view.sceneRect().bottomRight())
    far_center = view.mapToScene(view.viewport().rect().center())
    assert abs(far_center.x() - graph_center.x()) > 100

    assert view.center_graph()

    focused_center = view.mapToScene(view.viewport().rect().center())
    assert abs(focused_center.x() - graph_center.x()) <= 1.0
    assert abs(focused_center.y() - graph_center.y()) <= 1.0
    assert view.zoom_percent == 175
    assert view.transform() == transform_before


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


def test_releasing_dragged_loose_node_commits_green_wire_target_before_reroute(
    qtbot,
):
    pipeline = PrototypePipeline()
    view = PipelineGraphView()
    view.resize(980, 620)
    view.build_graph(
        pipeline.nodes.values(),
        pipeline.connections,
        positions={
            "input": QPointF(0, 20),
            "gaussian": QPointF(660, 20),
            "threshold": QPointF(1000, 20),
        },
    )
    node = pipeline.add_node("median_filter")
    view.add_node(node, QPointF(330, -250))
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    connection = next(
        item
        for item in view._connections
        if item.source_id == "input" and item.target_id == "gaussian"
    )
    path_before = QPainterPath(connection.path())
    scene_target = connection.path().pointAtPercent(0.5)
    proxy = view._proxies[node.id]
    start = view.mapFromScene(proxy.sceneBoundingRect().center())
    end = view.mapFromScene(scene_target)
    requests = []
    moves = []
    view.set_connection_insert_validator(
        lambda _operation_id, _key: ("full", "Drop to splice.")
    )
    view.node_splice_requested.connect(
        lambda node_id, key, old, new: requests.append(
            (node_id, tuple(key), QPointF(old), QPointF(new))
        )
    )
    view.node_moved.connect(
        lambda node_id, old, new: moves.append((node_id, QPointF(old), QPointF(new)))
    )

    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)

    assert connection._insert_preview_state == "full"
    assert _paths_equal(connection.path(), path_before)

    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    assert len(requests) == 1
    assert requests[0][:2] == (node.id, ("input", "gaussian", 0, 0))
    assert moves == []
    assert view._highlighted_connection is None


def test_releasing_dragged_loose_node_in_free_space_is_only_a_layout_move(qtbot):
    view, pipeline = _build_view()
    node = pipeline.add_node("median_filter")
    view.add_node(node, QPointF(260, -180))
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    proxy = view._proxies[node.id]
    start = view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(70, -45)
    requests = []
    moves = []
    view.node_splice_requested.connect(lambda *args: requests.append(args))
    view.node_moved.connect(lambda *args: moves.append(args))

    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    qtbot.mouseMove(view.viewport(), pos=end)
    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    assert requests == []
    assert len(moves) == 1
    assert moves[0][0] == node.id
    assert view._highlighted_connection is None


def test_clicking_node_selects_it_without_inspect_button(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)

    selected = []
    view.node_selected.connect(selected.append)

    card = view._cards["gaussian"]
    preview_center = QPointF(card.preview.geometry().center())
    start = view.mapFromScene(view._proxies["gaussian"].mapToScene(preview_center))
    qtbot.mouseClick(view.viewport(), Qt.LeftButton, pos=start)

    assert selected[-1] == "gaussian"
    assert not hasattr(card, "inspect_button")


def test_pressing_selected_node_exposes_press_boundary_to_receivers(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    selected = []
    press_boundaries = []
    selection_changes = []
    view.node_selected.connect(
        lambda node_id: (
            selected.append(node_id),
            press_boundaries.append(view.node_press_dispatch_active()),
        )
    )
    view.node_selection_changed.connect(
        lambda node_ids, primary: selection_changes.append(
            (tuple(node_ids), primary)
        )
    )

    view.select_node("gaussian")
    selected.clear()
    press_boundaries.clear()
    selection_changes.clear()

    view._handle_node_press("gaussian", Qt.NoModifier)

    assert selected == ["gaussian"]
    assert press_boundaries == [True]
    assert selection_changes == [(('gaussian',), "gaussian")]

    # Programmatic same-node selection remains distinguishable as an explicit
    # refresh boundary rather than a pointer press.
    view.select_node("gaussian")
    assert selected == ["gaussian", "gaussian"]
    assert press_boundaries == [True, False]
    assert selection_changes == [
        (('gaussian',), "gaussian"),
        (('gaussian',), "gaussian"),
    ]


def test_connected_node_drag_defers_obstacle_routing_until_release(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    view.select_node("gaussian")
    proxy = view._proxies["gaussian"]
    connection = next(
        item for item in view._connections if item.target_id == "gaussian"
    )
    start = view.mapFromScene(proxy.sceneBoundingRect().center())
    end = start + QPoint(70, -45)
    obstacle_calls = []
    original_obstacles = view.connection_obstacle_rects

    def tracked_obstacles(*args, **kwargs):
        obstacle_calls.append(True)
        return original_obstacles(*args, **kwargs)

    view.connection_obstacle_rects = tracked_obstacles
    old_target = QPointF(connection.path().pointAtPercent(1.0))

    qtbot.mousePress(view.viewport(), Qt.LeftButton, pos=start)
    obstacle_calls.clear()
    qtbot.mouseMove(view.viewport(), pos=end)

    assert view.node_drag_in_progress()
    assert obstacle_calls == []
    assert connection.path().pointAtPercent(1.0) != old_target

    qtbot.mouseRelease(view.viewport(), Qt.LeftButton, pos=end)

    assert not view.node_drag_in_progress()
    assert obstacle_calls
    assert connection.path().pointAtPercent(1.0) == proxy.port_scene_pos("input")


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


def test_unchanged_node_card_states_do_not_refresh_styles(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["gaussian"]
    refreshes = []
    monkeypatch.setattr(card, "_refresh_style", lambda: refreshes.append(True))

    card.set_selected(False)
    card.set_pinned(False)
    card.set_search_highlight(False)
    card.set_can_pin(True)

    assert refreshes == []

    card.set_selected(True)
    card.set_selected(True)
    card.set_pinned(True)
    card.set_pinned(True)
    card.set_search_highlight(True)
    card.set_search_highlight(True)
    card.set_can_pin(False)
    card.set_can_pin(False)

    assert len(refreshes) == 4
    assert not card._pinned
    assert card.pin_button.text() == "Pin"


def test_compute_badge_is_hidden_until_an_accepted_identity_is_supplied(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    card = view._cards["gaussian"]

    assert card.compute_badge.isHidden()
    assert card._compute_badge_kind is None
    assert card.card_layout.indexOf(card.title_row) >= 0
    assert card.title_label.parent() is card.title_row


def test_compute_badge_renders_supported_cpu_and_gpu_identities(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["gaussian"]
    cases = (
        (ComputeBadgeKind.CPU, "CPU"),
        (ComputeBadgeKind.CUPY, "GPU · CuPy"),
        (ComputeBadgeKind.CUCIM, "GPU · cuCIM"),
        (ComputeBadgeKind.HYBRID, "GPU + CPU"),
        (ComputeBadgeKind.CPU_FALLBACK, "CPU fallback"),
    )

    for kind, label in cases:
        view.set_node_compute_badge(
            "gaussian",
            kind,
            tooltip=f"Actual implementation for {label}",
        )

        assert not card.compute_badge.isHidden()
        assert card.compute_badge.text() == label
        assert card.compute_badge.toolTip() == f"Actual implementation for {label}"
        assert card.compute_badge.accessibleName() == f"Compute used: {label}"

    assert "#f59e0b" in card.compute_badge.styleSheet()


def test_compute_badge_hover_tooltip_is_reachable(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["gaussian"]
    view.set_node_compute_badge(
        "gaussian",
        "cpu_fallback",
        tooltip="cuCIM dependency unavailable.",
    )

    assert not card.compute_badge.testAttribute(Qt.WA_TransparentForMouseEvents)
    assert "cuCIM dependency unavailable" in card.compute_badge.toolTip()
    assert card.title_row.toolTip() == card.compute_badge.toolTip()


def test_stale_compute_badge_keeps_identity_but_is_visibly_muted(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["gaussian"]

    view.set_node_compute_badge(
        "gaussian",
        "cucim",
        tooltip="Used cucim-subtract-background-v2 on cuda:0.",
        stale=True,
    )

    assert card.compute_badge.text() == "GPU · cuCIM"
    assert card._compute_badge_stale is True
    assert "#a8a29e" in card.compute_badge.styleSheet()
    assert "Previous result (stale)." in card.compute_badge.toolTip()
    assert "cuda:0" in card.compute_badge.toolTip()
    assert "stale previous result" in card.compute_badge.accessibleName()


def test_compute_badge_clear_api_supports_one_node_or_the_whole_graph(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.set_node_compute_badge("input", "cpu")
    view.set_node_compute_badge("gaussian", "cupy")

    view.clear_node_compute_badges(("input",))

    assert view._cards["input"].compute_badge.isHidden()
    assert not view._cards["gaussian"].compute_badge.isHidden()

    view.clear_node_compute_badges()

    assert all(card.compute_badge.isHidden() for card in view._cards.values())


def test_compute_badge_does_not_replace_preview_processing_badge(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["gaussian"]

    view.set_node_compute_badge("gaussian", "cupy")
    view.set_node_processing("gaussian", True)

    assert not card.compute_badge.isHidden()
    assert not card.processing_badge.isHidden()
    assert card.compute_badge.parent() is card.title_row
    assert card.processing_badge.parent() is card


def test_compute_badge_rejects_unknown_presentation_identity(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    with pytest.raises(ValueError, match="Unknown compute badge kind"):
        view.set_node_compute_badge("gaussian", "mystery-accelerator")


def test_unchanged_compute_badge_skips_card_geometry_refresh(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["gaussian"]
    proxy = view._proxies["gaussian"]
    view.set_node_compute_badge(
        "gaussian",
        "cpu",
        tooltip="Used cpu-numpy.",
    )
    refreshes = []
    monkeypatch.setattr(card, "adjustSize", lambda: refreshes.append("card"))
    monkeypatch.setattr(proxy, "refresh_ports", lambda: refreshes.append("ports"))

    view.set_node_compute_badge(
        "gaussian",
        ComputeBadgeKind.CPU,
        tooltip=" Used cpu-numpy. ",
    )

    assert refreshes == []

    view.set_node_compute_badge(
        "gaussian",
        "cpu",
        tooltip="Used cpu-numpy on another environment.",
    )

    assert refreshes == ["card", "ports"]


def test_gpu_optimization_hint_is_subtle_explanatory_and_independent(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.setPalette(_graph_palette(dark=True))
    view._apply_palette_theme()
    card = view._cards["gaussian"]

    assert card.optimization_badge.isHidden()

    view.set_node_compute_badge("gaussian", "cpu")
    view.set_node_optimization_hint(
        "gaussian",
        "This node could use GPU after an exact dtype conversion. Select it to review.",
    )

    assert card.optimization_badge.text() == "GPU tip"
    assert not card.optimization_badge.isHidden()
    assert not card.compute_badge.isHidden()
    assert "exact dtype conversion" in card.optimization_badge.toolTip()
    assert "GPU eligibility tip" in card.optimization_badge.accessibleName()
    assert "#422006" in card.optimization_badge.styleSheet()

    view.clear_node_optimization_hints()

    assert card.optimization_badge.isHidden()
    assert not card.compute_badge.isHidden()


def test_thumbnail_retains_full_render_detail_for_device_aware_painting(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["gaussian"]
    card.preview.setFixedSize(180, 110)
    thumbnail = np.zeros((440, 720, 3), dtype=np.uint8)
    thumbnail[:, ::2] = 255

    view.set_thumbnail("gaussian", thumbnail)

    source = card.preview.source_pixmap()
    assert (source.width(), source.height()) == (720, 440)
    assert card.preview.has_source_pixmap()
    # QLabel's built-in pixmap path is intentionally empty: it would reduce the
    # source to the current screen size before QGraphicsView applies graph zoom.
    displayed = card.preview.pixmap()
    assert displayed is None or displayed.isNull()

    rendered = QImage(720, 440, QImage.Format_RGB888)
    rendered.fill(Qt.black)
    painter = QPainter(rendered)
    painter.scale(4.0, 4.0)
    card.preview.render(painter, QPoint())
    painter.end()
    scanline = np.fromiter(
        (rendered.pixelColor(x, 220).red() for x in range(720)),
        dtype=np.uint8,
        count=720,
    )
    # The alternating source pixels survive a 4x graph render. A copy first
    # reduced to the 180-pixel card would lose these 719 transitions.
    assert np.count_nonzero(np.diff(scanline.astype(np.int16))) == 719


def test_thumbnail_pending_preserves_complete_pixels_and_marks_first_load(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["gaussian"]

    view.set_thumbnail_pending("gaussian")

    assert not view.node_has_thumbnail("gaussian")
    assert card.preview.text() == "Calculating preview…"
    assert "not available yet" in card.preview.accessibleDescription()

    view.set_thumbnail(
        "gaussian",
        np.full((24, 32, 3), 73, dtype=np.uint8),
    )
    committed_key = card.preview.source_pixmap().cacheKey()
    view.set_thumbnail_pending("gaussian")

    assert view.node_has_thumbnail("gaussian")
    assert card.preview.source_pixmap().cacheKey() == committed_key
    assert card.preview.text() == ""
    assert card.preview.accessibleDescription() == ""

    view.set_thumbnail("gaussian", None)
    view.set_thumbnail_pending(
        "gaussian",
        "Preview unavailable",
        accessible_description="Exact thumbnail statistics failed: test failure.",
    )

    assert card.preview.text() == "Preview unavailable"
    assert card.preview.accessibleDescription() == (
        "Exact thumbnail statistics failed: test failure."
    )
    assert "waiting" not in card.preview.accessibleDescription().casefold()


def test_thumbnail_statistics_detail_is_nonvisual_and_keeps_card_compact(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    card = view._cards["gaussian"]
    view.set_thumbnail(
        "gaussian",
        np.full((24, 32, 3), 127, dtype=np.uint8),
    )
    assert view.node_has_thumbnail("gaussian")
    assert not view.node_has_thumbnail("missing-node")
    card.adjustSize()
    before = view._proxies["gaussian"].sceneBoundingRect()
    detail = (
        "Thumbnail presentation only. Exact uint16 histogram. "
        "Does not affect pipeline data."
    )

    view.set_node_thumbnail_stats_tooltip("gaussian", detail)

    after = view._proxies["gaussian"].sceneBoundingRect()
    assert not hasattr(card, "thumbnail_stats_row")
    assert not hasattr(card, "thumbnail_stats_badge")
    assert card.preview.toolTip() == detail
    assert card.preview.accessibleDescription() == ""
    assert card.preview.accessibleName() == "Gaussian Blur thumbnail preview"
    assert before.size() == after.size()

    view.set_thumbnail("gaussian", None)
    assert not view.node_has_thumbnail("gaussian")


def test_thumbnail_statistics_tooltip_clear_api_supports_one_node_or_all(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    for node_id in ("input", "gaussian"):
        view.set_node_thumbnail_stats_tooltip(
            node_id,
            f"Presentation-only detail for {node_id}.",
        )

    view.clear_node_thumbnail_stats_tooltips(("input",))

    assert view._cards["input"].preview.toolTip() == ""
    assert view._cards["gaussian"].preview.toolTip()

    view.clear_node_thumbnail_stats_tooltips()

    assert all(not card.preview.toolTip() for card in view._cards.values())


def test_automatic_stale_node_is_visibly_amber(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    card = view._cards["threshold"]

    view.set_node_execution_state(
        "threshold",
        "stale",
        manual=False,
        message="Downstream propagation is paused.",
    )

    assert STALE_EXECUTION_ACCENT in card.styleSheet()
    assert not card.execution_label.isHidden()
    assert card.execution_label.text() == "Stale; downstream paused"
    assert "paused" in card.toolTip().lower()


@pytest.mark.parametrize("dark", [False, True])
def test_isolated_tuning_has_distinct_active_mode_treatment(qtbot, dark):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.setPalette(_graph_palette(dark=dark))
    view._apply_palette_theme()
    card = view._cards["gaussian"]

    view.set_node_execution_state(
        "gaussian",
        "stale",
        manual=False,
        message="Parameters changed while downstream is paused.",
    )
    view.set_isolated_tuning_node("gaussian")

    assert ISOLATED_TUNING_ACCENT in card.styleSheet()
    assert STALE_EXECUTION_ACCENT not in card.styleSheet()
    assert graph_theme(card.palette()).tuning_surface in card.styleSheet()
    assert card.execution_label.text() == "Tuning in isolation"
    assert (
        "#c4b5fd" if dark else "#6d28d9"
    ) in card.execution_label.styleSheet()
    stale_category_colors = (
        ("#78350f", "#fde68a")
        if dark
        else ("#fef3c7", "#92400e")
    )
    assert all(
        color not in card.category_label.styleSheet()
        for color in stale_category_colors
    )

    card.set_processing(True, queued=True)

    assert ISOLATED_TUNING_ACCENT in card.styleSheet()
    assert STALE_EXECUTION_ACCENT not in card.styleSheet()
    assert graph_theme(card.palette()).tuning_surface in card.styleSheet()

    card.set_processing(False)
    view.set_isolated_tuning_node(None)

    assert ISOLATED_TUNING_ACCENT not in card.styleSheet()
    assert STALE_EXECUTION_ACCENT in card.styleSheet()


def test_node_context_menu_emits_requested_action(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    labels = []

    def fake_exec(menu, _pos):
        labels[:] = [
            action.text() for action in menu.actions() if not action.isSeparator()
        ]
        return next(action for action in menu.actions() if action.text() == "Delete")

    deleted = []
    view.node_delete_requested.connect(deleted.append)
    monkeypatch.setattr("napari_vipp._graph._exec_menu", fake_exec)

    view._show_node_context_menu("threshold", QPoint(0, 0))

    assert labels == [
        "Copy node",
        "Paste values",
        "Delete",
            "Inspect Code",
            "Duplicate Node",
            "Add note",
            "Bypass node",
            "Tune node in isolation",
        "Pin",
    ]
    assert deleted == ["threshold"]


def test_node_context_menu_toggles_isolated_tuning(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)

    def fake_exec(menu, _pos):
        return next(
            action
            for action in menu.actions()
            if action.text() == "Tune node in isolation"
        )

    requested = []
    view.node_isolation_requested.connect(requested.append)
    monkeypatch.setattr("napari_vipp._graph._exec_menu", fake_exec)

    view._show_node_context_menu("gaussian", QPoint(0, 0))
    view.set_isolated_tuning_node("gaussian")
    checked = []

    def inspect_exec(menu, _pos):
        action = next(
            action
            for action in menu.actions()
            if action.text() == "Tune node in isolation"
        )
        checked.append(action.isChecked())
        return None

    monkeypatch.setattr("napari_vipp._graph._exec_menu", inspect_exec)
    view._show_node_context_menu("gaussian", QPoint(0, 0))

    assert requested == ["gaussian"]
    assert checked == [True]
    assert view._cards["gaussian"]._isolated_tuning


def test_node_context_menu_uses_isolation_state_resolver(qtbot, monkeypatch):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    resolved: list[str] = []
    menus: list[dict[str, tuple[bool, str]]] = []

    def resolve(node_id: str) -> tuple[bool, bool, str]:
        resolved.append(node_id)
        if node_id == "threshold":
            return False, False, ""
        return True, False, "Calculate the graph before isolated tuning."

    def inspect_exec(menu, _pos):
        menus.append(
            {
                action.text(): (action.isEnabled(), action.toolTip())
                for action in menu.actions()
                if not action.isSeparator()
            }
        )
        return None

    view.set_node_isolation_state_resolver(resolve)
    monkeypatch.setattr("napari_vipp._graph._exec_menu", inspect_exec)

    view._show_node_context_menu("gaussian", QPoint(0, 0))
    view._show_node_context_menu("threshold", QPoint(0, 0))

    assert resolved == ["gaussian", "threshold"]
    assert menus[0]["Tune node in isolation"] == (
        False,
        "Calculate the graph before isolated tuning.",
    )
    assert "Tune node in isolation" not in menus[1]


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
    view.setPalette(_graph_palette(dark=True))
    view._apply_palette_theme()
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
            action.text() for action in menu.actions() if not action.isSeparator()
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


def test_palette_drop_on_terminal_node_appends_from_its_only_output(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    source_proxy = view._proxies["threshold"]
    source_port = source_proxy.output_ports[0]
    position = view.mapFromScene(source_proxy.sceneBoundingRect().center())
    event = _OperationDropEvent("median_filter", position)
    validation_calls = []
    append_requests = []
    free_create_requests = []
    view.set_node_append_validator(
        lambda operation_id, node_id, port_index: (
            validation_calls.append((operation_id, node_id, port_index))
            or ("compatible", "Drop to append.")
        )
    )
    view.node_append_requested.connect(
        lambda operation_id, node_id, port_index, scene_pos: append_requests.append(
            (operation_id, node_id, port_index, QPointF(scene_pos))
        )
    )
    view.node_create_requested.connect(
        lambda operation_id, scene_pos: free_create_requests.append(
            (operation_id, QPointF(scene_pos))
        )
    )

    view.dragMoveEvent(event)

    assert validation_calls == [("median_filter", "threshold", 0)]
    assert view._highlighted_append_port is source_port
    assert source_port._drop_state == "compatible"
    assert view._cards["threshold"]._append_drop_state == "compatible"

    view.dropEvent(event)

    assert event.accepted
    assert len(append_requests) == 1
    assert append_requests[0][:3] == ("median_filter", "threshold", 0)
    assert free_create_requests == []
    assert view._highlighted_append_port is None
    assert source_port._drop_state is None
    assert view._cards["threshold"]._append_drop_state is None


def test_palette_drop_on_occupied_output_is_incompatible_and_not_free_created(
    qtbot,
):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    source_proxy = view._proxies["gaussian"]
    source_port = source_proxy.output_ports[0]
    position = view.mapFromScene(source_proxy.sceneBoundingRect().center())
    event = _OperationDropEvent("median_filter", position)
    append_requests = []
    free_create_requests = []
    status_messages = []

    def unexpected_validator(*_args):
        raise AssertionError("occupied outputs must be rejected before validation")

    view.set_node_append_validator(unexpected_validator)
    view.node_append_requested.connect(lambda *args: append_requests.append(args))
    view.node_create_requested.connect(lambda *args: free_create_requests.append(args))
    view.status_message.connect(status_messages.append)

    view.dragMoveEvent(event)

    assert view._highlighted_append_port is source_port
    assert source_port._drop_state == "incompatible"
    assert view._cards["gaussian"]._append_drop_state == "incompatible"
    assert "already feeds" in status_messages[-1]

    view.dropEvent(event)

    assert event.accepted
    assert append_requests == []
    assert free_create_requests == []
    assert view._highlighted_append_port is None
    assert source_port._drop_state is None
    assert view._cards["gaussian"]._append_drop_state is None


def test_multi_output_node_requires_an_exact_port_for_palette_append(qtbot):
    view, pipeline = _build_view()
    node = pipeline.add_node("skeleton_graph_tables")
    view.add_node(node, QPointF(360, 320))
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    proxy = view._proxies[node.id]
    assert len(proxy.output_ports) == 2
    view.set_node_append_validator(
        lambda _operation_id, _node_id, _port_index: (
            "compatible",
            "Drop to append.",
        )
    )
    append_requests = []
    view.node_append_requested.connect(
        lambda operation_id, node_id, port_index, scene_pos: append_requests.append(
            (operation_id, node_id, port_index, QPointF(scene_pos))
        )
    )

    card_event = _OperationDropEvent(
        "merge_tables",
        view.mapFromScene(proxy.sceneBoundingRect().center()),
    )
    view.dragMoveEvent(card_event)

    assert view._highlighted_append_port is None

    second_output = proxy.output_ports[1]
    port_event = _OperationDropEvent(
        "merge_tables",
        view.mapFromScene(second_output.mapToScene(QPointF(0, 0))),
    )
    view.dragMoveEvent(port_event)

    assert view._highlighted_append_port is second_output
    assert second_output._drop_state == "compatible"

    view.dropEvent(port_event)

    assert len(append_requests) == 1
    assert append_requests[0][:3] == ("merge_tables", node.id, 1)
    assert view._highlighted_append_port is None
    assert second_output._drop_state is None


def test_wire_insert_preview_takes_over_and_clears_terminal_append_feedback(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.show()
    qtbot.waitExposed(view)
    terminal_proxy = view._proxies["threshold"]
    terminal_port = terminal_proxy.output_ports[0]
    view.set_node_append_validator(
        lambda _operation_id, _node_id, _port_index: (
            "compatible",
            "Drop to append.",
        )
    )
    view.set_connection_insert_validator(
        lambda _operation_id, _connection_key: (
            "full",
            "Drop to splice.",
        )
    )
    terminal_event = _OperationDropEvent(
        "median_filter",
        view.mapFromScene(terminal_proxy.sceneBoundingRect().center()),
    )
    view.dragMoveEvent(terminal_event)
    assert view._highlighted_append_port is terminal_port

    connection = view._connections[0]
    wire_event = _OperationDropEvent(
        "median_filter",
        view.mapFromScene(connection.path().pointAtPercent(0.5)),
    )
    view.dragMoveEvent(wire_event)

    assert view._highlighted_append_port is None
    assert terminal_port._drop_state is None
    assert view._cards["threshold"]._append_drop_state is None
    assert view._highlighted_connection is connection
    assert connection._insert_preview_state == "full"

    view.dropEvent(wire_event)

    assert view._highlighted_connection is None
    assert connection._insert_preview_state is None


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
    assert requests == [(node.id, ("input", "gaussian", 0, 0), old_pos, new_pos)]
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


@pytest.mark.parametrize(
    ("state", "state_accent", "surface_attribute"),
    (
        ("ready", "#22c55e", "ready_surface"),
        ("not_calculated", STALE_EXECUTION_ACCENT, "stale_surface"),
        ("error", "#ef4444", "error_surface"),
    ),
)
def test_manual_colocalization_card_keeps_category_identity(
    qtbot,
    state,
    state_accent,
    surface_attribute,
):
    view, pipeline = _build_view()
    qtbot.addWidget(view)
    view.setPalette(_graph_palette(dark=True))
    view._apply_palette_theme()
    node = pipeline.add_node("colocalization_metrics")
    view.add_node(node, QPointF(990, 20))

    view.set_node_execution_state(node.id, state, manual=True)

    card = view._cards[node.id]
    card_style = card.styleSheet()
    category_style = card.category_label.styleSheet()
    category = "Colocalization & Spatial Analysis"
    theme = graph_theme(card.palette())
    assert f"border: 2px solid {state_accent};" in card_style
    assert f"background: {getattr(theme, surface_attribute)};" in card_style
    assert f"background: {category_color(category)};" in card_style
    assert f"background: {category_tint(category, card.palette())};" in category_style
    assert f"color: {category_foreground(category, card.palette())};" in category_style


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


def test_port_label_modes_show_only_the_requested_labels(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    gaussian = view._proxies["gaussian"]
    card = view._cards["gaussian"]
    positions_before = view.node_positions()
    compact_width = gaussian.sceneBoundingRect().width()

    assert view.port_label_mode == PortLabelMode.AMBIGUOUS_ONLY
    assert not gaussian.input_port.label_item.isVisible()
    assert not gaussian.output_port.label_item.isVisible()

    view.set_port_label_mode("Show all")

    assert view.port_label_mode == PortLabelMode.SHOW_ALL
    assert gaussian.input_port.label_item.isVisible()
    assert gaussian.output_port.label_item.isVisible()
    assert gaussian.sceneBoundingRect().width() > compact_width
    content_rect = gaussian.mapRectToScene(QRectF(card.card_layout.contentsRect()))
    assert not content_rect.intersects(
        gaussian.input_port.label_item.sceneBoundingRect()
    )
    assert not content_rect.intersects(
        gaussian.output_port.label_item.sceneBoundingRect()
    )
    assert view.node_positions() == positions_before

    view.set_port_label_mode(PortLabelMode.HIDE_ALL)

    assert not gaussian.input_port.label_item.isVisible()
    assert not gaussian.output_port.label_item.isVisible()
    assert gaussian.sceneBoundingRect().width() == compact_width
    assert view.node_positions() == positions_before


def test_ambiguous_port_labels_use_declared_deconvolution_input_names(qtbot):
    pipeline = PrototypePipeline()
    nodes = [
        pipeline.add_node("richardson_lucy_deconvolution"),
        pipeline.add_node("richardson_lucy_tv_deconvolution"),
    ]
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)

    for node in nodes:
        proxy = view._proxies[node.id]
        expanded_width = proxy.sceneBoundingRect().width()
        assert [port.label for port in proxy.input_ports] == ["Image", "PSF"]
        assert [port.label_item.toPlainText() for port in proxy.input_ports] == [
            "Image",
            "PSF",
        ]
        assert all(port.label_item.isVisible() for port in proxy.input_ports)

        view.set_port_label_mode(PortLabelMode.HIDE_ALL)
        assert proxy.sceneBoundingRect().width() < expanded_width
        view.set_port_label_mode(PortLabelMode.AMBIGUOUS_ONLY)
        assert proxy.sceneBoundingRect().width() == expanded_width


def test_label_expansion_reports_overlap_without_moving_nodes(qtbot):
    view, _pipeline = _build_view()
    qtbot.addWidget(view)
    view.apply_node_positions(
        {
            "input": (-300.0, 0.0),
            "gaussian": (0.0, 0.0),
            "threshold": (235.0, 0.0),
        }
    )
    positions_before = view.node_positions()

    assert view.overlapping_node_pairs() == []

    view.set_port_label_mode(PortLabelMode.SHOW_ALL)

    assert ("gaussian", "threshold") in view.overlapping_node_pairs()
    assert view.node_positions() == positions_before


def test_resolved_dynamic_output_labels_are_drawn_in_ambiguous_mode(qtbot):
    pipeline = PrototypePipeline()
    node = pipeline.add_node("split_channels")
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    ports = pipeline.output_ports(node.id)

    view.set_node_output_ports(
        node.id,
        len(ports),
        [port.label for port in ports],
        data_types=[port.output_type for port in ports],
    )

    drawn = view._proxies[node.id].output_ports
    assert [port.label for port in drawn] == [port.label for port in ports]
    assert [port.label_item.toPlainText() for port in drawn] == [
        port.label for port in ports
    ]
    assert all(port.label_item.isVisible() for port in drawn)


def test_long_port_labels_are_elided_with_full_tooltips(qtbot):
    pipeline = PrototypePipeline()
    node = pipeline.add_node("richardson_lucy_tv_deconvolution")
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    long_label = "Observed fluorescence image with a deliberately long name"

    view.set_node_input_ports(node.id, 2, [long_label, "PSF"])

    proxy = view._proxies[node.id]
    port = proxy.input_ports[0]
    assert port.label_item.toPlainText() != long_label
    assert port.label_item.toPlainText().endswith("…")
    assert port.label_item.toolTip() == long_label
    assert proxy.sceneBoundingRect().contains(port.label_item.sceneBoundingRect())
    card = view._cards[node.id]
    content_rect = proxy.mapRectToScene(QRectF(card.card_layout.contentsRect()))
    assert not content_rect.intersects(port.label_item.sceneBoundingRect())


def test_expanded_port_rows_keep_wires_attached_and_feed_layout_sizes(qtbot):
    pipeline = PrototypePipeline()
    target = pipeline.add_node("combine_channels")
    view = PipelineGraphView()
    view.build_graph(pipeline.nodes.values(), pipeline.connections)
    qtbot.addWidget(view)
    view.add_connection("input", target.id, target_port=0)
    connection = view._connections[-1]
    height_before = view.node_scene_rect(target.id).height()

    view.set_node_input_ports(
        target.id,
        12,
        [f"Channel {index + 1}" for index in range(12)],
    )

    proxy = view._proxies[target.id]
    expected_end = proxy.port_scene_pos("input", 0)
    actual_end = connection.path().pointAtPercent(1.0)
    assert view.node_scene_rect(target.id).height() > height_before
    assert view.node_card_sizes()[target.id][1] >= 348.0
    assert (actual_end - expected_end).manhattanLength() < 0.01


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
        rect.contains(path.pointAtPercent(index / 120.0)) for index in range(121)
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
        lambda source, target, port: removed_connections.append((source, target, port))
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
