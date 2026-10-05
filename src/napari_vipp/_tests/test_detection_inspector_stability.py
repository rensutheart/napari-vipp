"""Live detection tuning must not transiently tear down unchanged context."""

import pytest
from qtpy.QtCore import QEvent, QObject, QPoint
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QApplication

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_array_output,
    _select,
    _widget,
)


class _GeometryEvents(QObject):
    def __init__(self, watched, control, panel):
        super().__init__()
        self.samples = []
        self.control = control
        self.panel = panel
        for widget in watched:
            widget.installEventFilter(self)

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() in {QEvent.Move, QEvent.Resize}:
            self.samples.append((
                watched,
                watched.geometry().getRect(),
                self.control.mapTo(self.panel.viewport(), QPoint(0, 0)),
                self.panel.verticalScrollBar().value(),
            ))
        return False


def test_execution_refresh_relayouts_only_changed_isolation_controls(
    qtbot, monkeypatch
):
    widget = _widget(qtbot)
    _select(widget, "input")
    widget._sync_isolated_tuning_ui()
    refreshes = []
    monkeypatch.setattr(
        widget, "_sync_inspector_responsive_layout", lambda: refreshes.append(True)
    )
    widget._sync_isolated_tuning_ui()
    assert refreshes == []

    # The source doesn't support bypass. Restore its deliberately disturbed
    # visibility and still refresh the layout when the presentation changes.
    assert widget.node_bypass_checkbox.isHidden()
    widget.node_bypass_checkbox.show()
    widget._sync_isolated_tuning_ui()
    assert widget.node_bypass_checkbox.isHidden()
    assert refreshes == [True]
    widget._sync_isolated_tuning_ui()
    assert refreshes == [True]


@pytest.mark.parametrize("width", (340, 800))
@pytest.mark.parametrize("theme", ("dark", "light"))
@pytest.mark.parametrize("font_size", (10, 14))
def test_detection_cap_scrubbing_keeps_input_card_and_control_position(
    qtbot, qapp, width, theme, font_size
):
    import numpy as np
    from napari._qt.qt_resources import get_stylesheet

    previous_font = qapp.font()
    qapp.setFont(QFont("Segoe UI", font_size))
    data = np.zeros((7, 72, 96), dtype=np.float32)
    widget = _widget(qtbot, data, axes="TYX")
    widget.setStyleSheet(
        get_stylesheet(theme, extra_variables={"font_size": f"{font_size}pt"})
    )
    detector = widget.add_node_from_palette("detect_spots_per_frame")
    connected = widget.pipeline.connect("input", detector.id)
    assert connected.success
    widget._apply_connection_result_to_graph(connected)
    _publish_array_output(widget, "input", data, axes="TYX")
    _select(widget, detector.id)
    # Use the real scroll surface, with enough content to expose transient
    # shrinking as a scroll/control movement instead of hiding it in slack.
    panel = widget.inspector_panel
    panel.setParent(None)
    qtbot.addWidget(panel)
    panel.setStyleSheet(widget.styleSheet())
    panel.resize(width, 350)
    panel.show()
    control = widget._parameter_widgets["maximum_detections"]
    panel.ensureWidgetVisible(control)
    qtbot.wait(50)
    assert panel.verticalScrollBar().maximum() > 0
    row = widget.connected_inputs_panel.rows[0]
    watched = (
        widget.connected_inputs_panel, widget.parameter_group,
        widget.parameter_form_widget, widget.inspector_content, control,
    )
    scroll = panel.verticalScrollBar().value()
    position = control.mapTo(panel.viewport(), QPoint(0, 0))
    events = _GeometryEvents(watched, control, panel)

    try:
        for value in (1, 2, 843, 10000, 1, 843):
            control.slider.setValue(control._to_slider(value))
            QApplication.processEvents()
            qtbot.wait(5)
            assert detector.params["maximum_detections"] == control.value()
            assert widget.connected_inputs_panel.rows == [row]
            assert control is widget._parameter_widgets["maximum_detections"]
            assert panel.verticalScrollBar().value() == scroll
            assert control.mapTo(panel.viewport(), QPoint(0, 0)) == position
            label = widget.node_name_editor.summary_label
            assert label.text() == widget._node_presentation(detector.id).summary
            assert label.height() >= label.heightForWidth(label.width())
            assert widget.node_name_editor.rect().contains(label.geometry())
        # Catch intermediate movement as ancestor layouts resize, not just the
        # final position after Qt has settled. Harmless parent size-hint changes
        # need not be frozen when they leave the control/scroll position alone.
        assert all(
            pos == position and value == scroll
            for _, _, pos, value in events.samples
        ), "\n".join(
            f"{item.objectName()}: {rect}; {position} -> {pos}, "
            f"scroll {scroll} -> {value}"
            for item, rect, pos, value in events.samples
        )
    finally:
        qapp.setFont(previous_font)
        widget._permit_incomplete_startup_discard()
