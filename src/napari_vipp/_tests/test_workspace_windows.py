"""Workspace window controls preserve analysis and their owning workflow."""

from __future__ import annotations

from dataclasses import replace

import pytest
from qtpy.compat import isalive
from qtpy.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import QApplication, QDialog, QDockWidget, QMainWindow, QWidget

from napari_vipp._tests.test_floating_dock import _FakeOwnerApi, _mock_native_windows
from napari_vipp._tests.test_results_workspace_dialog import _table
from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.ui import floating_dock, workspace_window
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.results_workspace import ResultsWorkspaceDialog


def _dialog(qtbot, tmp_path, kind, *, parent=None):
    calls = []
    if kind == "batch":
        plan = _preview_result(tmp_path)
        actions = replace(
            _actions(plan, calls),
            preview_batch=lambda *_args: calls.append("check") or plan,
        )
        dialog = CollectionBatchDialog(parent, actions=actions)
        dialog.apply_preview_result(plan, preview_representative=False)
        dialog.preview_table.selectRow(1)
        dialog.preview_table.item(0, 0).setCheckState(Qt.Unchecked)
        dialog.runRequested.connect(lambda *_args: calls.append("run"))
    else:
        dialog = ResultsWorkspaceDialog(parent)
        dialog.set_workflows([("workflow-a", "Measurements")], "workflow-a")
        dialog.set_data(_table(), node_id="measure", title="Measure Objects")
        dialog.set_choices(
            data_sources=[(("measure", 0), "Measurements")],
            data_source=("measure", 0),
            summaries=[("summary", "Statistics")],
            summary_id="summary",
            plots=[("plot", "Plot Results")],
            plot_id="plot",
            plot_sources=[("measure", "Original measurements")],
            plot_source_id="measure",
        )
        dialog.data_panel.table_view.selectRow(1)
        dialog.recalculate_requested.connect(calls.append)
        dialog.workflow_selected.connect(calls.append)
        dialog.data_selected.connect(lambda *_args: calls.append("data"))
    qtbot.addWidget(dialog)
    dialog.show()
    QApplication.processEvents()
    return dialog, calls


def _state(dialog, kind):
    if kind == "batch":
        return (
            dialog.values(),
            dialog._preview_result,
            set(dialog._checked_items),
            dialog._current_item,
            tuple(index.row() for index in (
                dialog.preview_table.selectionModel().selectedRows()
            )),
        )
    return (
        dialog._data_table,
        dialog.workflow_selector.currentData(),
        dialog.data_selector.currentData(),
        dialog.summary_selector.currentData(),
        dialog.plot_selector.currentData(),
        tuple(index.row() for index in (
            dialog.data_panel.table_view.selectionModel().selectedRows()
        )),
    )


def _double_click(widget, *, position=None, button=Qt.LeftButton,
                  event_type=QEvent.MouseButtonDblClick):
    point = widget.rect().center() if position is None else position
    event = QMouseEvent(
        event_type,
        QPointF(point),
        QPointF(widget.mapToGlobal(point)),
        button,
        button,
        Qt.NoModifier,
    )
    QApplication.sendEvent(widget, event)


@pytest.mark.parametrize("kind", ["batch", "results"])
def test_actual_workspaces_have_native_nonmodal_window_controls(
    qtbot, tmp_path, kind
):
    parent = QMainWindow()
    qtbot.addWidget(parent)
    dialog, _calls = _dialog(qtbot, tmp_path, kind, parent=parent)
    flags = dialog.windowFlags()
    assert flags & Qt.WindowType_Mask == Qt.Window
    for hint in (
        Qt.WindowTitleHint,
        Qt.WindowSystemMenuHint,
        Qt.WindowMinimizeButtonHint,
        Qt.WindowMaximizeButtonHint,
        Qt.WindowCloseButtonHint,
    ):
        assert flags & hint
    assert not flags & Qt.WindowContextHelpButtonHint
    assert not flags & Qt.FramelessWindowHint
    assert not dialog.isModal()
    assert dialog.parentWidget() is parent


@pytest.mark.parametrize("kind", ["batch", "results"])
def test_toolbar_double_click_maximizes_and_restores_without_analysis(
    qtbot, tmp_path, kind
):
    dialog, calls = _dialog(qtbot, tmp_path, kind)
    before = _state(dialog, kind)
    surface = (
        dialog.config_name_label if kind == "batch" else dialog.connection_bar
    )
    position = surface.rect().center() if kind == "batch" else QPoint(2, 2)
    assert surface.childAt(position) is None
    normal_size = dialog.size()

    _double_click(surface, position=position)
    qtbot.waitUntil(dialog.isMaximized)
    assert not dialog.isFullScreen()
    _double_click(surface, position=position)
    qtbot.waitUntil(lambda: not dialog.isMaximized())

    assert dialog.size() == normal_size
    assert _state(dialog, kind) == before
    assert calls == []


@pytest.mark.parametrize("kind", ["batch", "results"])
def test_toolbar_interactive_children_and_right_double_click_do_not_maximize(
    qtbot, tmp_path, kind
):
    dialog, _calls = _dialog(qtbot, tmp_path, kind)
    surface = dialog.config_row if kind == "batch" else dialog.connection_bar
    control = (
        dialog.load_config_button if kind == "batch"
        else dialog.workflow_selector
    )
    hit = control.mapTo(surface, control.rect().center())
    assert surface.childAt(hit) is not None
    _double_click(surface, position=hit)
    assert not dialog.isMaximized()
    _double_click(surface, position=QPoint(2, 2), button=Qt.RightButton)
    assert not dialog.isMaximized()


@pytest.mark.parametrize("kind", ["batch", "results"])
@pytest.mark.parametrize("maximized", [False, True])
def test_native_title_bar_double_click_is_left_to_native_window_manager(
    qtbot, tmp_path, monkeypatch, kind, maximized
):
    dialog, _calls = _dialog(qtbot, tmp_path, kind)
    if maximized:
        dialog.showMaximized()
    state = dialog.windowState()
    toggles = []
    monkeypatch.setattr(dialog, "showMaximized", lambda: toggles.append("max"))
    monkeypatch.setattr(dialog, "showNormal", lambda: toggles.append("normal"))

    _double_click(dialog, event_type=QEvent.NonClientAreaMouseButtonDblClick)
    QApplication.sendEvent(
        dialog, QEvent(QEvent.NonClientAreaMouseButtonDblClick)
    )

    assert toggles == []
    assert dialog.windowState() == state


@pytest.mark.parametrize("kind", ["batch", "results"])
@pytest.mark.parametrize("maximized", [False, True])
def test_reopening_minimized_workspace_restores_same_config_and_selection(
    qtbot, tmp_path, kind, maximized
):
    dialog, calls = _dialog(qtbot, tmp_path, kind)
    before = _state(dialog, kind)
    dialog.setWindowState(
        Qt.WindowMinimized | (Qt.WindowMaximized if maximized else Qt.WindowNoState)
    )

    workspace_window.show_workspace_window(dialog)

    assert dialog.isVisible()
    assert not dialog.isMinimized()
    assert dialog.isMaximized() is maximized
    assert _state(dialog, kind) == before
    assert calls == []


def test_owner_repairs_coalesce_then_become_idle(qtbot, monkeypatch):
    dialog = QDialog()
    qtbot.addWidget(dialog)
    controls = workspace_window.WorkspaceWindowController(dialog)
    dialog.show()
    QApplication.processEvents()
    repaired = []
    monkeypatch.setattr(
        workspace_window, "make_window_independent",
        lambda window: repaired.append(window) or True,
    )

    for event_type in (QEvent.Show, QEvent.WinIdChange, QEvent.WindowStateChange):
        controls.eventFilter(dialog, QEvent(event_type))
    qtbot.waitUntil(lambda: bool(repaired))
    qtbot.wait(80)

    assert repaired == [dialog]


@pytest.mark.parametrize("state", [
    Qt.WindowNoState,
    Qt.WindowMinimized,
    Qt.WindowMaximized,
    Qt.WindowMinimized | Qt.WindowMaximized,
])
@pytest.mark.parametrize("visible", [False, True])
def test_native_owner_repair_preserves_qt_parent_state_and_visibility(
    qtbot, monkeypatch, state, visible
):
    parent = QMainWindow()
    qtbot.addWidget(parent)
    dialog = QDialog(parent)
    qtbot.addWidget(dialog)
    controls = workspace_window.WorkspaceWindowController(dialog)
    dialog.setWindowState(state)
    dialog.setVisible(visible)
    QApplication.processEvents()
    repaired = []
    monkeypatch.setattr(
        workspace_window, "make_window_independent",
        lambda window: repaired.append(window) or True,
    )

    controls._schedule_owner_repair()
    qtbot.waitUntil(lambda: bool(repaired))

    assert repaired == [dialog]
    assert dialog.windowState() == state
    assert dialog.isVisible() is visible
    assert dialog.parentWidget() is parent


def test_queued_owner_repair_is_discarded_when_workflow_parent_is_destroyed(
    qtbot, monkeypatch
):
    parent = QMainWindow()
    qtbot.addWidget(parent)
    dialog = QDialog(parent)
    qtbot.addWidget(dialog)
    controls = workspace_window.WorkspaceWindowController(dialog)
    repaired = []
    monkeypatch.setattr(
        workspace_window, "make_window_independent",
        lambda window: repaired.append(window) or True,
    )
    controls._schedule_owner_repair()

    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()

    assert not isalive(parent)
    assert not isalive(dialog)
    assert not isalive(controls)
    assert repaired == []


@pytest.mark.parametrize("kind", ["batch", "results"])
def test_actual_workspaces_and_queued_repairs_die_with_workflow_parent(
    qtbot, tmp_path, kind
):
    parent = QMainWindow()
    qtbot.addWidget(parent)
    dialog, calls = _dialog(qtbot, tmp_path, kind, parent=parent)
    controls = dialog._window_controls
    controls._schedule_owner_repair()

    parent.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()

    assert not isalive(parent)
    assert not isalive(dialog)
    assert not isalive(controls)
    assert calls == []


@pytest.mark.parametrize("cached_root_owner", [False, True])
def test_native_owner_repair_accepts_only_actual_qt_window_ancestors(
    qtbot, monkeypatch, cached_root_owner
):
    host = QMainWindow()
    qtbot.addWidget(host)
    dock = QDockWidget("VIPP", host)
    content = QWidget()
    dock.setWidget(content)
    host.addDockWidget(Qt.BottomDockWidgetArea, dock)
    dialog = QDialog(content)
    qtbot.addWidget(dialog)
    workspace_window.WorkspaceWindowController(dialog)
    dock.setFloating(True)
    assert dialog.parentWidget().window() is dock
    assert dock.parentWidget() is host
    owner = host if cached_root_owner else dock
    api = _FakeOwnerApi(int(owner.winId()))
    _mock_native_windows(monkeypatch, api)

    assert floating_dock.make_window_independent(dialog)

    assert api.cleared == [int(dialog.winId())]
    assert dialog.parentWidget() is content
    assert dock.parentWidget() is host


def test_native_owner_repair_rejects_unrelated_window_even_in_same_process(
    qtbot, monkeypatch
):
    host = QMainWindow()
    unrelated = QMainWindow()
    qtbot.addWidget(host)
    qtbot.addWidget(unrelated)
    dialog = QDialog(host)
    qtbot.addWidget(dialog)
    workspace_window.WorkspaceWindowController(dialog)
    api = _FakeOwnerApi(int(unrelated.winId()))
    _mock_native_windows(monkeypatch, api)

    with pytest.raises(RuntimeError, match="unexpected native owner"):
        floating_dock.make_window_independent(dialog)

    assert api.cleared == []
    assert dialog.parentWidget() is host
