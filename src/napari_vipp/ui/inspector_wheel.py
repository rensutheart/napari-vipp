"""Keep inspector scrolling separate from deliberate parameter edits."""

from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QScrollBar,
    QWidget,
)


def route_inspector_control_wheel(scope: QWidget, watched, event) -> bool:
    """Route a parameter control's wheel to its closest enclosing scroll area.

    An application event filter returning True stops Qt's normal propagation,
    even for an ignored event. Forward once with the original deltas and phase
    instead; never synthesize selection changes or alter keyboard focus policy.
    """
    if (
        event.type() != QEvent.Wheel
        or not isinstance(watched, QWidget)
        or not scope.isAncestorOf(watched)
    ):
        return False
    control = watched
    while control is not None and control is not scope:
        if isinstance(control, QScrollBar):
            return False  # This is already a scrolling surface, not a parameter.
        if isinstance(control, (QAbstractSlider, QAbstractSpinBox, QComboBox)):
            break
        control = control.parentWidget()
    else:
        return False
    if isinstance(control, QComboBox) and control.view().isVisible():
        return False  # The user opened the list; let it scroll normally.

    parent = control.parentWidget()
    while parent is not None and not isinstance(parent, QAbstractScrollArea):
        parent = parent.parentWidget()
    if parent is not None:
        viewport = parent.viewport()
        forwarded = QWheelEvent(
            QPointF(viewport.mapFromGlobal(event.globalPosition().toPoint())),
            event.globalPosition(),
            event.pixelDelta(),
            event.angleDelta(),
            event.buttons(),
            event.modifiers(),
            event.phase(),
            event.inverted(),
            Qt.MouseEventSynthesizedByApplication,
            event.pointingDevice(),
        )
        QApplication.sendEvent(viewport, forwarded)
    # Consume even at a scroll limit: there must be no fallback value edit.
    event.accept()
    return True
