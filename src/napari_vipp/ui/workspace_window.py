"""Standard window controls for retained Batch and Results workspaces."""

from __future__ import annotations

import logging
import re
from weakref import WeakSet

from qtpy.compat import isalive
from qtpy.QtCore import QEvent, QObject, QPoint, Qt, QTimer
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QAbstractButton,
    QAbstractScrollArea,
    QAbstractSlider,
    QAbstractSpinBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QMenu,
    QTabBar,
    QWidget,
)

from napari_vipp.ui.floating_dock import make_window_independent


def _interactive_widget(widget: QWidget) -> bool:
    """Keep controls and selectable/link labels out of passive-header gestures."""
    if isinstance(
        widget,
        (
            QAbstractButton,
            QAbstractScrollArea,
            QAbstractSlider,
            QAbstractSpinBox,
            QComboBox,
            QLineEdit,
            QMenu,
            QTabBar,
        ),
    ):
        return True
    if widget.focusPolicy() != Qt.NoFocus:
        return True
    if isinstance(widget, QLabel):
        interaction = widget.textInteractionFlags()
        if interaction & (Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard):
            return True
        if (
            interaction & Qt.LinksAccessibleByMouse
            and widget.textFormat() != Qt.PlainText
            and re.search(r"<a(?:\s|>)", widget.text(), re.IGNORECASE)
        ):
            return True
    return False


def _interactive_hit(surface: QWidget, position: QPoint) -> bool:
    target = surface.childAt(position) or surface
    while target is not None:
        if _interactive_widget(target):
            return True
        if target is surface:
            break
        target = target.parentWidget()
    return False


class WorkspaceWindowController(QObject):
    """Configure native chrome while retaining the workspace's QObject parent.

    Create before the window is first shown. Native ownership is repaired after
    Qt's show/state/handle events, never by rebuilding a visible window or
    changing its geometry, visibility or minimized/maximized state.
    """

    def __init__(self, window: QWidget):
        super().__init__(window)
        self._window = window
        self._toolbar_surfaces: WeakSet[QWidget] = WeakSet()
        self._owner_repair_warning = ""
        self._owner_repair_timer = QTimer(self)
        self._owner_repair_timer.setSingleShot(True)
        self._owner_repair_timer.timeout.connect(self._repair_owner)

        flags = (window.windowFlags() & ~Qt.WindowType_Mask) | Qt.Window
        flags &= ~(
            Qt.FramelessWindowHint
            | Qt.MSWindowsFixedSizeDialogHint
            | Qt.WindowContextHelpButtonHint
        )
        flags |= (
            Qt.WindowTitleHint
            | Qt.WindowSystemMenuHint
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        if flags != window.windowFlags():
            window.setWindowFlags(flags)
        window.setWindowModality(Qt.NonModal)
        window.installEventFilter(self)

    def add_toolbar_surface(self, widget: QWidget) -> None:
        """Enable maximize/restore on this explicitly registered passive area."""
        if widget in self._toolbar_surfaces:
            return
        if widget is not self._window and not self._window.isAncestorOf(widget):
            raise ValueError("A workspace header must belong to its window.")
        self._toolbar_surfaces.add(widget)
        if widget is not self._window:
            widget.installEventFilter(self)

    def _schedule_owner_repair(self) -> None:
        self._owner_repair_timer.start(0)

    def _repair_owner(self) -> None:
        if not isalive(self._window):
            return
        try:
            make_window_independent(self._window)
            self._owner_repair_warning = ""
        except Exception as exc:
            warning = (
                f"Could not configure the {self._window.windowTitle()} window: {exc}"
            )
            if warning != self._owner_repair_warning:
                logging.getLogger(__name__).warning(warning)
                self._owner_repair_warning = warning

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self._window and event.type() in {
            QEvent.Show,
            QEvent.WinIdChange,
            QEvent.WindowStateChange,
        }:
            self._schedule_owner_repair()
        # Windows already handles the native title bar. Only client-area events
        # on explicitly registered passive surfaces receive a second gesture.
        if (
            watched in self._toolbar_surfaces
            and event.type() == QEvent.MouseButtonDblClick
            and isinstance(event, QMouseEvent)
            and event.button() == Qt.LeftButton
            and not _interactive_hit(watched, event.pos())
        ):
            if self._window.isMaximized():
                self._window.showNormal()
            else:
                self._window.showMaximized()
            event.accept()
            return True
        return super().eventFilter(watched, event)


def show_workspace_window(window: QWidget) -> None:
    """Reveal a workspace, restoring only its minimized state when necessary."""
    state = window.windowState()
    if state & Qt.WindowMinimized:
        window.setWindowState(state & ~Qt.WindowMinimized)
    window.show()
    window.raise_()
    window.activateWindow()
