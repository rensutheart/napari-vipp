"""Tab location actions target saved workflows without changing editor state."""

from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import QPoint

from napari_vipp._tests.test_workflow_tabs import _session
from napari_vipp.ui import file_reveal, workflow_tabs
from napari_vipp.ui.workflow_tabs import WorkflowTabBar, WorkflowTabModel


@pytest.mark.parametrize(
    "platform,label",
    [
        ("win32", "Open in File Explorer"),
        ("darwin", "Open in Finder"),
        ("linux", "Open containing folder"),
    ],
)
def test_reveal_menu_targets_clicked_tab_and_tracks_saves(
    qtbot, monkeypatch, tmp_path, platform, label
):
    monkeypatch.setattr(workflow_tabs, "sys", SimpleNamespace(platform=platform))
    model = WorkflowTabModel()
    target = _session(session_id="target", title="Custom tab title")
    model.add(target)
    model.add(_session(session_id="active"))
    bar = WorkflowTabBar()
    qtbot.addWidget(bar)
    bar.resize(650, 40)
    bar.sync_from_model(model)
    bar.show()
    requested = []
    bar.revealTabRequested.connect(requested.append)
    seen = []

    def choose_reveal(menu, _position):
        action = next(action for action in menu.actions() if action.text() == label)
        seen.append((action.isEnabled(), action.toolTip()))
        return action

    monkeypatch.setattr(workflow_tabs.QMenu, "exec", choose_reveal)
    bar._open_context_menu(bar.tabRect(0).center())
    assert seen[-1] == (
        False,
        "Save this workflow first to give it a location on disk.",
    )
    assert requested == []

    # Save/Save As updates the action even with a custom title and pending edits.
    for filename in ("saved workflow.json", "renamed workflow.json"):
        target.mark_saved(tmp_path / filename)
        target.mark_dirty()
        bar.refresh_session(0, target)
        bar._open_context_menu(bar.tabRect(0).center())
        assert seen[-1][0]
        assert str(target.path) in seen[-1][1]
        assert requested[-1] == "target"
        assert bar.currentIndex() == 1
        assert target.dirty

    bar.moveTab(0, 1)
    bar._open_context_menu(bar.tabRect(1).center())
    assert requested[-1] == "target"
    target.detach_path()
    bar.refresh_session(1, target)
    bar._open_context_menu(bar.tabRect(1).center())
    assert not seen[-1][0]
    assert len(requested) == 3


def test_empty_tab_bar_context_has_no_file_action(qtbot, monkeypatch):
    bar = WorkflowTabBar()
    qtbot.addWidget(bar)

    def dismiss(menu, _position):
        assert [action.text() for action in menu.actions()] == ["New workflow tab"]
        return None

    monkeypatch.setattr(workflow_tabs.QMenu, "exec", dismiss)
    bar._open_context_menu(QPoint(10, 10))


def test_widget_reveals_inactive_tab_and_reports_unavailable_files(
    qtbot, monkeypatch, tmp_path
):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((8, 8), np.float32)))
    qtbot.addWidget(widget)
    active = widget._workflow_tabs.current
    target = widget._workflow_tabs.create_blank(make_current=False)
    path = tmp_path / "workflow with spaces, é.json"
    path.touch()
    target.mark_saved(path)
    target.mark_dirty()
    widget.workflow_tab_bar.sync_from_model(widget._workflow_tabs)
    launched = []
    monkeypatch.setattr(file_reveal, "_launch", launched.append)
    widget.workflow_tab_bar.revealTabRequested.emit(target.session_id)
    assert len(launched) == 1
    assert launched[0][-1] in (str(path), str(path.parent))
    assert widget._workflow_tabs.current is active
    assert widget.pipeline is active.pipeline
    assert target.dirty
    assert "Requested selection" in widget.status_label.text() or (
        "Opened containing folder" in widget.status_label.text()
    )

    # Resolve the session's current Save As path again on activation.
    target.mark_saved(tmp_path / "missing.json")
    widget.workflow_tab_bar.revealTabRequested.emit(target.session_id)
    assert len(launched) == 1
    assert "missing or unavailable" in widget.status_label.text()
    target.detach_path()
    widget.workflow_tab_bar.revealTabRequested.emit(target.session_id)
    assert "Save this workflow first" in widget.status_label.text()
    widget.workflow_tab_bar.revealTabRequested.emit("closed-session")
    assert len(launched) == 1
