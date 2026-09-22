"""Scrolling the inspector must not silently change scientific parameters."""

import pytest
from qtpy.QtCore import QEvent, QObject, QPoint, QPointF, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QApplication, QComboBox, QVBoxLayout, QWidget

from napari_vipp._tests.test_ui_inspector_widget_integration import _widget


def _wheel(target, *, angle=-120, pixels=0, native=True):
    position = target.rect().center()
    window = target.window()
    event = QWheelEvent(
        QPointF(target.mapTo(window, position) if native else position),
        QPointF(target.mapToGlobal(position)),
        QPoint(0, pixels),
        QPoint(0, angle),
        Qt.NoButton,
        Qt.NoModifier,
        Qt.ScrollUpdate if pixels else Qt.NoScrollPhase,
        False,
    )
    # Enter through QWindow like a real mouse/touchpad event, so Qt performs
    # hit testing and parent propagation instead of a direct widget-only call.
    QApplication.sendEvent(window.windowHandle() if native else target, event)


def _scrollable_inspector(qtbot, *, editable=False):
    widget = _widget(qtbot)
    # Create after the application's event filter is installed, like a newly
    # selected node. Include a wrapper and editable child to exercise nesting.
    wrapper = QWidget()
    layout = QVBoxLayout(wrapper)
    combo = QComboBox()
    combo.setEditable(editable)
    combo.addItems([f"Value {index}" for index in range(40)])
    combo.setMaxVisibleItems(6)
    combo.setCurrentIndex(3)
    layout.addWidget(combo)
    spacer = QWidget()
    spacer.setMinimumHeight(1200)
    layout.addWidget(spacer)
    widget._inspector_layout.insertWidget(0, wrapper)
    widget.resize(1100, 480)
    widget.show()
    qtbot.waitUntil(lambda: widget.inspector_panel.verticalScrollBar().maximum() > 100)
    widget.inspector_panel.verticalScrollBar().setValue(0)
    return widget, combo


@pytest.mark.parametrize("focused", (False, True))
@pytest.mark.parametrize("editable", (False, True))
@pytest.mark.parametrize("pixels", (0, -45))
def test_closed_dropdown_keeps_value_and_scrolls_inspector(
    qtbot, focused, editable, pixels,
):
    widget, combo = _scrollable_inspector(qtbot, editable=editable)
    if focused:
        combo.setFocus(Qt.TabFocusReason)
    else:
        widget.graph_search_edit.setFocus()
    changes = []
    combo.currentIndexChanged.connect(changes.append)
    scroll = widget.inspector_panel.verticalScrollBar()
    target = combo.lineEdit() if editable else combo
    _wheel(target, pixels=pixels)
    assert scroll.value() > 0
    assert combo.currentIndex() == 3
    assert changes == []


@pytest.mark.parametrize("at_bottom", (False, True))
def test_closed_focused_dropdown_remains_safe_at_inspector_scroll_limits(
    qtbot, at_bottom,
):
    widget, combo = _scrollable_inspector(qtbot)
    scroll = widget.inspector_panel.verticalScrollBar()
    end = scroll.maximum() if at_bottom else scroll.minimum()
    scroll.setValue(end)
    combo.setFocus()
    for _ in range(3):
        _wheel(combo, angle=-120 if at_bottom else 120, native=False)
    assert scroll.value() == end
    assert combo.currentIndex() == 3


def test_open_dropdown_list_scrolls_without_scrolling_inspector(qtbot):
    widget, combo = _scrollable_inspector(qtbot)
    combo.showPopup()
    qtbot.waitUntil(lambda: combo.view().isVisible())
    popup_scroll = combo.view().verticalScrollBar()
    popup_scroll.setValue(popup_scroll.minimum())
    inspector_position = widget.inspector_panel.verticalScrollBar().value()
    _wheel(combo.view().viewport())
    assert popup_scroll.value() > popup_scroll.minimum()
    assert widget.inspector_panel.verticalScrollBar().value() == inspector_position
    combo.hidePopup()


def test_dropdown_keyboard_selection_and_popup_selection_still_work(qtbot):
    _owner, combo = _scrollable_inspector(qtbot)
    combo.setFocus(Qt.TabFocusReason)
    qtbot.keyClick(combo, Qt.Key_Down)
    assert combo.currentIndex() == 4
    qtbot.keyClick(combo, Qt.Key_F4)
    qtbot.waitUntil(lambda: combo.view().isVisible())
    index = combo.model().index(5, 0)
    combo.view().scrollTo(index)
    qtbot.mouseClick(
        combo.view().viewport(), Qt.LeftButton,
        pos=combo.view().visualRect(index).center(),
    )
    assert combo.currentIndex() == 5
    assert not combo.view().isVisible()


def test_wheel_behavior_outside_inspector_is_unchanged(qtbot):
    _owner, _combo = _scrollable_inspector(qtbot)
    outside = QComboBox()
    qtbot.addWidget(outside)
    outside.addItems(["First", "Second", "Third"])
    outside.show()
    outside.setCurrentIndex(1)
    _wheel(outside)
    assert outside.currentIndex() == 2


def test_wheel_over_real_parameter_does_not_edit_workflow_or_undo_history(
    qtbot, monkeypatch,
):
    widget = _widget(qtbot)
    # Adding the fixture node dirties this shown workflow; avoid a save prompt
    # during teardown without bypassing the parameter/undo assertions below.
    monkeypatch.setattr(widget, "_confirm_close_dirty_workflow_tabs", lambda: True)
    node = widget.add_node_from_palette("subtract_background")
    widget._select_node(node.id)
    combo = widget._parameter_widgets["spatial_mode"].combo
    widget.resize(1100, 480)
    widget.show()
    combo.setFocus()
    widget._debounce_timer.stop()
    params = dict(node.params)
    undo_count = len(widget._undo_stack)
    index = combo.currentIndex()
    _wheel(combo, native=False)
    assert combo.currentIndex() == index
    assert node.params == params
    assert len(widget._undo_stack) == undo_count
    assert not widget._debounce_timer.isActive()


@pytest.mark.parametrize("inverted", (False, True))
@pytest.mark.parametrize("phase", (Qt.ScrollBegin, Qt.ScrollUpdate, Qt.ScrollEnd))
def test_forwarded_touchpad_event_retains_deltas_phase_and_direction_once(
    qtbot, phase, inverted,
):
    widget, combo = _scrollable_inspector(qtbot, editable=True)
    received = []

    class Recorder(QObject):
        def eventFilter(self, watched, event):  # noqa: N802
            if event.type() == QEvent.Wheel:
                received.append((
                    event.pixelDelta(), event.angleDelta(), event.phase(),
                    event.inverted(), event.modifiers(),
                    event.pointingDevice(),
                ))
            return False

    recorder = Recorder(widget)
    widget.inspector_panel.viewport().installEventFilter(recorder)
    event = QWheelEvent(
        QPointF(2, 2), QPointF(combo.mapToGlobal(QPoint(2, 2))),
        QPoint(4, -45), QPoint(0, 0), Qt.NoButton, Qt.ShiftModifier,
        phase, inverted,
    )
    QApplication.sendEvent(combo.lineEdit(), event)
    assert received == [(
        QPoint(4, -45), QPoint(0, 0), phase, inverted, Qt.ShiftModifier,
        event.pointingDevice(),
    )]
    assert combo.currentIndex() == 3
