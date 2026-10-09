from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
from app_model import Action, Application
from app_model.backends.qt import QModelMenu
from app_model.types import ToggleRule
from qtpy.compat import isalive
from qtpy.QtCore import QEvent, Qt
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QVBoxLayout, QWidget

from napari_vipp.ui.plugin_dock import (
    VIPP_DOCK_NAME,
    install_plugin_dock_restorer,
    restore_plugin_dock,
)


class _Connections:
    def __init__(self):
        self.connections = []
        self.filters = []

    def connect(self, signal, callback):
        signal.connect(callback)
        self.connections.append((signal, callback))

    def install_event_filter(self, watched, event_filter):
        watched.installEventFilter(event_filter)
        self.filters.append((watched, event_filter))

    def shutdown(self):
        for signal, callback in self.connections:
            signal.disconnect(callback)
        for watched, event_filter in self.filters:
            watched.removeEventFilter(event_filter)
        self.connections.clear()
        self.filters.clear()


class _Dock(QDockWidget):
    def inner_widget(self):
        return self.widget()


def _dock_case(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    owner = QWidget()
    owner._closing = False
    owner.pipeline = object()
    owner.outputs = {"input": np.arange(16).reshape(4, 4)}
    dock = _Dock(VIPP_DOCK_NAME, window)
    dock.setWidget(owner)
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)
    window.show()
    dock.setFloating(True)
    return window, dock, owner


@pytest.fixture
def application():
    name = f"vipp-dock-test-{uuid4()}"
    app = Application(name)
    yield app
    Application.destroy(name)


def _plugin_menu(application, window, dock, monkeypatch, active_viewer=None):
    from napari._qt._qplugins import _qnpe2

    host = SimpleNamespace(_wrapped_dock_widgets={VIPP_DOCK_NAME: dock})
    viewer = SimpleNamespace(window=host)
    active = active_viewer or [viewer]
    if active[0] is None:
        active[0] = viewer
    monkeypatch.setattr(_qnpe2, "_provide_viewer_or_raise", lambda **_kwargs: active[0])
    if "napari-vipp.make_widget" not in application.commands:
        application.register_action(
            Action(
                id="napari-vipp.make_widget",
                title=VIPP_DOCK_NAME,
                callback=partial(
                    _qnpe2._toggle_or_get_widget,
                    plugin="napari-vipp",
                    widget_name="VIPP Workflow",
                    full_name=VIPP_DOCK_NAME,
                ),
                menus=[{"id": "napari/plugins"}],
                toggled=ToggleRule(
                    get_current=lambda: (
                        (
                            target_dock := active[0].window._wrapped_dock_widgets.get(
                                VIPP_DOCK_NAME
                            )
                        )
                        is not None
                        and target_dock.isVisible()
                    )
                ),
            )
        )
    menu = QModelMenu("napari/plugins", application, parent=window)
    window.menuBar().addMenu(menu)
    # Emitting aboutToShow directly models an opening in this viewer. A real
    # mouse opening activates its host window before the action is selected.
    menu.aboutToShow.connect(lambda: QApplication.setActiveWindow(window))
    action = menu.findAction("napari-vipp.make_widget")
    assert action is not None and action.isEnabled()
    assert action.objectName() == "napari-vipp.make_widget"
    return menu, action, viewer


@pytest.mark.parametrize("maximized", (False, True))
def test_plugin_menu_restores_same_minimized_dock_and_cached_session(
    qtbot, monkeypatch, application, maximized
):
    window, dock, owner = _dock_case(qtbot)
    menu, action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    connections = _Connections()
    restorer = install_plugin_dock_restorer(owner, dock, window, connections)
    assert restorer is not None
    pipeline = owner.pipeline
    cached_output = owner.outputs["input"]
    state = Qt.WindowMinimized | (Qt.WindowMaximized if maximized else Qt.WindowNoState)
    dock.setWindowState(state)
    assert dock.isMinimized()

    menu.aboutToShow.emit()
    menu.aboutToHide.emit()
    action.trigger()

    qtbot.waitUntil(lambda: dock.isVisible() and not dock.isMinimized())
    assert dock.isMaximized() == maximized
    assert dock.widget() is owner
    assert owner.pipeline is pipeline
    assert owner.outputs["input"] is cached_output
    assert dock.parentWidget() is window
    assert not owner._closing


def test_plugin_menu_restores_hidden_minimized_dock(qtbot, monkeypatch, application):
    window, dock, owner = _dock_case(qtbot)
    menu, action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    install_plugin_dock_restorer(owner, dock, window, _Connections())
    dock.showMinimized()
    dock.hide()
    assert dock.isMinimized() and not dock.isVisible()

    menu.aboutToShow.emit()
    action.trigger()

    qtbot.waitUntil(lambda: dock.isVisible() and not dock.isMinimized())


def test_visible_plugin_dock_keeps_napari_normal_toggle(
    qtbot, monkeypatch, application
):
    window, dock, owner = _dock_case(qtbot)
    menu, action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    install_plugin_dock_restorer(owner, dock, window, _Connections())

    menu.aboutToShow.emit()
    action.trigger()
    QApplication.processEvents()

    assert not dock.isVisible()
    menu.aboutToShow.emit()
    action.trigger()
    QApplication.processEvents()
    assert dock.isVisible() and not dock.isMinimized()


def test_direct_hide_and_frame_reconfiguration_keep_minimized_state(
    qtbot, monkeypatch, application
):
    window, dock, owner = _dock_case(qtbot)
    _menu, _action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    install_plugin_dock_restorer(owner, dock, window, _Connections())
    dock.showMinimized()
    dock.hide()
    QApplication.processEvents()

    assert dock.isMinimized() and not dock.isVisible()
    dock.show()
    QApplication.processEvents()
    assert dock.isMinimized()


def test_canceled_menu_does_not_leave_a_restore_capture(
    qtbot, monkeypatch, application
):
    window, dock, owner = _dock_case(qtbot)
    menu, action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    install_plugin_dock_restorer(owner, dock, window, _Connections())
    dock.showMinimized()
    menu.aboutToShow.emit()
    menu.aboutToHide.emit()
    QApplication.processEvents()

    action.trigger()
    QApplication.processEvents()

    assert dock.isMinimized() and not dock.isVisible()


@pytest.mark.parametrize("closing", ("owner", "dock", "destroyed"))
def test_pending_restore_does_not_reopen_closed_or_destroyed_session(
    qtbot, monkeypatch, application, closing
):
    window, dock, owner = _dock_case(qtbot)
    menu, action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    install_plugin_dock_restorer(owner, dock, window, _Connections())
    dock.showMinimized()
    menu.aboutToShow.emit()
    action.trigger()
    if closing == "owner":
        owner._closing = True
    elif closing == "dock":
        dock.close()
    else:
        window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)

    QApplication.processEvents()

    if closing == "destroyed":
        assert not isalive(dock) and not isalive(owner)
    else:
        assert not dock.isVisible()


@pytest.mark.parametrize(
    "first_menu_state", ("expired", "expiry_pending", "restore_pending")
)
def test_another_viewers_plugin_menu_does_not_restore_this_viewer(
    qtbot, monkeypatch, application, first_menu_state
):
    first_window, first_dock, first_owner = _dock_case(qtbot)
    second_window, second_dock, second_owner = _dock_case(qtbot)
    active_viewer = [None]
    first_menu, first_action, first_viewer = _plugin_menu(
        application, first_window, first_dock, monkeypatch, active_viewer
    )
    second_menu, second_action, second_viewer = _plugin_menu(
        application, second_window, second_dock, monkeypatch, active_viewer
    )
    install_plugin_dock_restorer(first_owner, first_dock, first_window, _Connections())
    install_plugin_dock_restorer(
        second_owner, second_dock, second_window, _Connections()
    )
    first_dock.showMinimized()
    second_dock.showMinimized()
    active_viewer[0] = first_viewer
    first_menu.aboutToShow.emit()
    first_menu.aboutToHide.emit()
    if first_menu_state == "expired":
        QApplication.processEvents()
    elif first_menu_state == "restore_pending":
        first_action.trigger()
    active_viewer[0] = second_viewer

    second_menu.aboutToShow.emit()
    second_action.trigger()

    qtbot.waitUntil(lambda: not second_dock.isMinimized())
    assert first_dock.isMinimized()
    assert first_dock.widget() is first_owner
    assert second_dock.widget() is second_owner


def test_other_plugin_action_does_not_restore_minimized_vipp(
    qtbot, monkeypatch, application
):
    window, dock, owner = _dock_case(qtbot)
    menu, _action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    other_action = menu.addAction("Another plugin")
    other_action.setObjectName("another-plugin.make_widget")
    install_plugin_dock_restorer(owner, dock, window, _Connections())
    dock.showMinimized()

    menu.aboutToShow.emit()
    other_action.trigger()
    QApplication.processEvents()

    assert dock.isVisible() and dock.isMinimized()


@pytest.mark.parametrize("first_menu_state", ("expiry_pending", "restore_pending"))
def test_viewer_without_vipp_does_not_restore_another_viewers_pending_dock(
    qtbot, monkeypatch, application, first_menu_state
):
    first_window, first_dock, first_owner = _dock_case(qtbot)
    second_window, second_dock, _second_owner = _dock_case(qtbot)
    active_viewer = [None]
    first_menu, first_action, first_viewer = _plugin_menu(
        application, first_window, first_dock, monkeypatch, active_viewer
    )
    second_menu, second_action, second_viewer = _plugin_menu(
        application, second_window, second_dock, monkeypatch, active_viewer
    )
    second_viewer.window._wrapped_dock_widgets.clear()
    second_dock.hide()
    # The second viewer creates its first editor only when its action runs.
    # Returning None isolates that command from OpenGL and startup imports.
    monkeypatch.setattr(
        "napari._qt._qplugins._qnpe2.get_widget_contribution",
        lambda *_args: (lambda: None, "VIPP Workflow"),
    )
    install_plugin_dock_restorer(first_owner, first_dock, first_window, _Connections())
    first_dock.showMinimized()
    active_viewer[0] = first_viewer
    first_menu.aboutToShow.emit()
    first_menu.aboutToHide.emit()
    if first_menu_state == "restore_pending":
        first_action.trigger()
    active_viewer[0] = second_viewer

    second_menu.aboutToShow.emit()
    second_action.trigger()
    QApplication.processEvents()

    assert first_dock.isMinimized()
    assert first_dock.widget() is first_owner


def test_queued_restore_is_canceled_when_another_host_becomes_active(
    qtbot, monkeypatch, application
):
    window, dock, owner = _dock_case(qtbot)
    menu, action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    other_window = QMainWindow()
    qtbot.addWidget(other_window)
    other_window.show()
    install_plugin_dock_restorer(owner, dock, window, _Connections())
    dock.showMinimized()
    menu.aboutToShow.emit()
    action.trigger()

    QApplication.setActiveWindow(other_window)
    QApplication.processEvents()

    assert dock.isMinimized() and not dock.isVisible()


def test_actual_popup_mouse_selection_restores_the_owning_dock(
    qtbot, monkeypatch, application
):
    window, dock, owner = _dock_case(qtbot)
    menu, action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    install_plugin_dock_restorer(owner, dock, window, _Connections())
    dock.showMinimized()
    menu.popup(window.mapToGlobal(window.rect().center()))
    qtbot.waitUntil(menu.isVisible)

    qtbot.mouseClick(menu, Qt.LeftButton, pos=menu.actionGeometry(action).center())

    qtbot.waitUntil(lambda: dock.isVisible() and not dock.isMinimized())


def test_canonical_direct_dock_is_reused_by_napari_plugin_api(qtbot, monkeypatch):
    from napari._qt.qt_main_window import Window
    from napari.plugins import _npe2

    _window, dock, owner = _dock_case(qtbot)
    host = SimpleNamespace(_wrapped_dock_widgets={VIPP_DOCK_NAME: dock})

    class NewEditor(QWidget):
        def __init__(self):
            pytest.fail("The existing direct editor must be reused.")

    monkeypatch.setattr(
        _npe2,
        "get_widget_contribution",
        lambda *_args: (NewEditor, "VIPP Workflow"),
    )

    returned_dock, returned_widget = Window.add_plugin_dock_widget(
        host, "napari-vipp", "VIPP Workflow"
    )

    assert returned_dock is dock
    assert returned_widget is owner
    assert dock.widget() is owner
    dock.showMinimized()
    assert restore_plugin_dock(returned_dock)
    assert dock.isVisible() and not dock.isMinimized()


def test_restore_helper_leaves_docked_and_deleted_widgets_alone(qtbot):
    window, dock, _owner = _dock_case(qtbot)
    dock.setFloating(False)
    assert not restore_plugin_dock(dock)
    window.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert not restore_plugin_dock(dock)


@pytest.mark.parametrize("wrapped", (False, True))
def test_widget_installs_menu_hook_once_and_disconnects_on_shutdown(
    qtbot, monkeypatch, application, wrapped
):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    window = QMainWindow()
    qtbot.addWidget(window)
    widget = VippWidget(_Viewer(), defer_initial_run=True)
    dock = _Dock(VIPP_DOCK_NAME, window)
    if wrapped:
        host = QWidget()
        QVBoxLayout(host).addWidget(widget)
        dock.setWidget(host)
    else:
        dock.setWidget(widget)
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)
    menu, action, _viewer = _plugin_menu(application, window, dock, monkeypatch)
    widget._ensure_dock_widget_chrome()
    restorer = widget._plugin_dock_restorer
    assert restorer is not None
    connection_count = len(widget._lifecycle._connections)
    widget._ensure_dock_widget_chrome()
    assert widget._plugin_dock_restorer is restorer
    assert len(widget._lifecycle._connections) == connection_count
    pipeline = widget.pipeline
    cached_output = np.arange(16).reshape(4, 4)
    widget.pipeline.outputs["input"] = cached_output
    dock.setFloating(True)
    dock.showMinimized()
    menu.aboutToShow.emit()
    action.trigger()
    qtbot.waitUntil(lambda: dock.isVisible() and not dock.isMinimized())
    assert widget.pipeline is pipeline
    assert widget.pipeline.outputs["input"] is cached_output
    dock.showMinimized()
    menu.aboutToShow.emit()

    widget._lifecycle.shutdown()
    action.trigger()
    QApplication.processEvents()

    assert widget._lifecycle._connections == []
    assert widget._lifecycle._event_filters == []
    assert widget._closing
    assert dock.isMinimized() and not dock.isVisible()
