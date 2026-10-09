"""Restore this viewer's existing VIPP dock when its plugin action is chosen."""

from __future__ import annotations

from weakref import ref

from qtpy.compat import isalive
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QMenu, QWidget

VIPP_DOCK_NAME = "VIPP Workflow (napari-vipp)"
_PLUGINS_MENU_NAME = "napari/plugins"
_VIPP_ACTION_NAME = "napari-vipp.make_widget"
_active_restorer_ref = None


def restore_plugin_dock(dock: QDockWidget, *, window_state=None) -> bool:
    """Show a surviving floating dock while retaining its normal/maximized state."""
    if not isalive(dock) or not dock.isFloating():
        return False
    state = dock.windowState() if window_state is None else window_state
    dock.setWindowState(state & ~Qt.WindowMinimized)
    dock.show()
    dock.raise_()
    dock.activateWindow()
    return True


class _PluginDockRestorer(QObject):
    def __init__(self, owner, dock, window, menu, lifecycle):
        super().__init__(owner)
        self._owner_ref = ref(owner)
        self._dock_ref = ref(dock)
        self._window_ref = ref(window)
        self._capture = None
        self._generation = 0
        lifecycle.connect(menu.aboutToShow, self._capture_minimized_dock)
        lifecycle.connect(menu.aboutToHide, self._expire_menu_capture)
        lifecycle.connect(menu.triggered, self._on_menu_triggered)
        lifecycle.install_event_filter(dock, self)

    def _live_dock(self):
        owner = self._owner_ref()
        dock = self._dock_ref()
        window = self._window_ref()
        if (
            not isalive(self)
            or owner is None
            or not isalive(owner)
            or getattr(owner, "_closing", False)
            or dock is None
            or not isalive(dock)
            or window is None
            or not isalive(window)
            or not dock.isFloating()
            or not dock.isAncestorOf(owner)
        ):
            return None
        parent = dock.parentWidget()
        while parent is not None and parent is not window:
            parent = parent.parentWidget()
        return dock if parent is window else None

    def _capture_minimized_dock(self):
        global _active_restorer_ref
        previous = _active_restorer_ref() if _active_restorer_ref is not None else None
        if previous is not None and previous is not self and isalive(previous):
            previous._invalidate_capture()
        self._invalidate_capture()
        _active_restorer_ref = ref(self)
        dock = self._live_dock()
        self._capture = (
            (self._generation, dock.windowState())
            if dock is not None and dock.isMinimized()
            else None
        )

    def _invalidate_capture(self):
        self._generation += 1
        self._capture = None

    def _active_host_is_ours(self):
        window = self._window_ref()
        if window is None or not isalive(window):
            return False
        active = QApplication.activeWindow()
        while active is not None:
            if active is window:
                return True
            active = active.parentWidget()
        return False

    def _expire_menu_capture(self):
        # Qt hides a menu before triggering its chosen action. Let that action's
        # synchronous callbacks finish before expiring a canceled opening.
        generation = self._generation
        QTimer.singleShot(0, lambda: self._clear_capture(generation))

    def _clear_capture(self, generation):
        if isalive(self) and generation == self._generation:
            self._capture = None

    def _on_menu_triggered(self, action):
        if not self._active_host_is_ours():
            self._invalidate_capture()
            return
        capture = self._capture
        self._capture = None
        active = _active_restorer_ref() if _active_restorer_ref is not None else None
        if (
            active is not self
            or capture is None
            or action.objectName() != _VIPP_ACTION_NAME
        ):
            return
        self._generation += 1
        generation = self._generation
        _opening_generation, state = capture
        # Napari's existing action toggles visibility. Restore only after that
        # action has completed, without intercepting unrelated hide events.
        QTimer.singleShot(0, lambda: self._restore_capture(generation, state))

    def _restore_capture(self, generation, state):
        if (
            not isalive(self)
            or generation != self._generation
            or not self._active_host_is_ours()
        ):
            return
        dock = self._live_dock()
        if dock is not None:
            restore_plugin_dock(dock, window_state=state)

    def eventFilter(self, watched, event):  # noqa: N802
        if event.type() == QEvent.Close:
            self._invalidate_capture()
        return False


def install_plugin_dock_restorer(
    owner: QWidget, dock: QDockWidget, window: QMainWindow, lifecycle
) -> _PluginDockRestorer | None:
    """Bind only the Plugins menu belonging to this dock's Qt main window."""
    menu_bar = window.menuWidget()
    if menu_bar is None:
        return None
    for action in menu_bar.actions():
        menu = action.menu()
        if isinstance(menu, QMenu) and menu.objectName() == _PLUGINS_MENU_NAME:
            return _PluginDockRestorer(owner, dock, window, menu, lifecycle)
    return None


__all__ = [
    "VIPP_DOCK_NAME",
    "install_plugin_dock_restorer",
    "restore_plugin_dock",
]
