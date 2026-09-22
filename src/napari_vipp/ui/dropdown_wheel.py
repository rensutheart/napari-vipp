"""Keep inspector scrolling separate from deliberate drop-down selection."""

from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QWheelEvent
from qtpy.QtWidgets import QAbstractScrollArea, QApplication, QComboBox, QWidget


def route_closed_dropdown_wheel(scope: QWidget, watched, event) -> bool:
    """Route a closed selector's wheel to its closest enclosing scroll area.

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
    combo = watched
    while combo is not None and combo is not scope:
        if isinstance(combo, QComboBox):
            break
        combo = combo.parentWidget()
    if not isinstance(combo, QComboBox) or combo.view().isVisible():
        return False

    parent = combo.parentWidget()
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
