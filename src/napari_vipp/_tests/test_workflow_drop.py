"""Workflow URL drops use the normal validated, retained-tab Open action."""

from pathlib import Path

import pytest
from qtpy.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QUrl
from qtpy.QtGui import QDragEnterEvent, QDragMoveEvent, QDropEvent
from qtpy.QtWidgets import QApplication, QDialog, QWidget

from napari_vipp._tests.test_widget import VippWidget, _Viewer
from napari_vipp.core.workflow import save_workflow
from napari_vipp.ui.reproduction import ReproductionOpenCancelled
from napari_vipp.ui.workflow_drop import WorkflowFileDropHandler, workflow_drop_path


def _mime(*urls):
    mime = QMimeData()
    mime.setUrls(list(urls))
    return mime


def _send_drop(target, path, *, actions=Qt.CopyAction | Qt.MoveAction):
    mime = _mime(QUrl.fromLocalFile(str(path)))
    enter = QDragEnterEvent(QPoint(5, 5), actions, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(target, enter)
    assert enter.isAccepted()
    assert enter.dropAction() == Qt.CopyAction
    move = QDragMoveEvent(QPoint(5, 5), actions, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(target, move)
    assert move.isAccepted()
    drop = QDropEvent(QPointF(5, 5), actions, mime, Qt.LeftButton, Qt.NoModifier)
    QApplication.sendEvent(target, drop)
    assert drop.isAccepted()
    assert drop.dropAction() == Qt.CopyAction


@pytest.fixture
def widget(qtbot, monkeypatch):
    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    monkeypatch.setattr(widget, "run_pipeline", lambda *args, **kwargs: None)
    return widget


def test_single_local_json_url_is_recognized_without_reading(tmp_path, monkeypatch):
    path = tmp_path / "Workflow with spaces Δ.JSON"

    def no_read(*args, **kwargs):
        pytest.fail("Drag recognition must not read or stat the file")

    monkeypatch.setattr(Path, "open", no_read)
    monkeypatch.setattr(Path, "stat", no_read)
    assert workflow_drop_path(_mime(QUrl.fromLocalFile(str(path)))) == path


@pytest.mark.parametrize("kind", ["empty", "text", "image", "remote", "multiple"])
def test_unrelated_payloads_are_not_workflow_drops(kind, tmp_path):
    url = QUrl.fromLocalFile(str(tmp_path / "workflow.json"))
    mime = {
        "empty": QMimeData(),
        "text": QMimeData(),
        "image": _mime(QUrl.fromLocalFile(str(tmp_path / "image.tif"))),
        "remote": _mime(QUrl("https://example.org/workflow.json")),
        "multiple": _mime(url, url),
    }[kind]
    if kind == "text":
        mime.setText(str(tmp_path / "workflow.json"))
    assert workflow_drop_path(mime) is None


@pytest.mark.parametrize("surface", ["panel", "canvas", "tabs", "inspector", "search"])
def test_drop_opens_new_tab_and_preserves_unsaved_work(
    widget, qtbot, tmp_path, surface
):
    path = save_workflow(tmp_path / "My workflow Δ.JSON", widget.pipeline)
    old = widget._workflow_tabs.current
    old_pipeline = widget.pipeline
    old_pipeline.set_param("gaussian", "sigma", 3.25)
    old_pipeline.outputs["input"] = "retained cache"
    widget._sync_current_workflow_tab_state()
    assert old.dirty
    targets = {
        "panel": widget,
        "canvas": widget.graph_view.viewport(),
        "tabs": widget.workflow_tab_bar,
        "inspector": widget.inspector_viewport,
        "search": widget.palette_search,
    }

    _send_drop(targets[surface], path)
    # No file reads, tab changes or modal prompts inside the OS drag callback.
    assert len(widget._workflow_tabs) == 1
    qtbot.waitUntil(lambda: len(widget._workflow_tabs) == 2)

    assert widget._workflow_tabs[0] is old
    assert old.pipeline is old_pipeline
    assert old.dirty
    assert old.pipeline.nodes["gaussian"].params["sigma"] == 3.25
    assert old.pipeline.outputs["input"] == "retained cache"
    assert widget._workflow_tabs.current.path == path.resolve()
    assert widget._workflow_tabs.current.title == "My workflow Δ"
    assert widget.pipeline.nodes["gaussian"].params["sigma"] != 3.25
    assert path.is_file()  # Copy/open semantics, never a move.


@pytest.mark.parametrize("kind", ["malformed", "other-json", "missing", "directory"])
def test_invalid_drop_reports_failure_without_changing_tabs(
    widget, qtbot, tmp_path, kind
):
    path = tmp_path / "invalid.json"
    if kind == "malformed":
        path.write_text("{")
    elif kind == "other-json":
        path.write_text('{"something": "else"}')
    elif kind == "directory":
        path.mkdir()
    old = widget._workflow_tabs.current
    pipeline = widget.pipeline
    _send_drop(widget.graph_view.viewport(), path)
    qtbot.waitUntil(lambda: "Load failed" in widget.status_label.text())
    assert len(widget._workflow_tabs) == 1
    assert widget._workflow_tabs.current is old
    assert widget.pipeline is pipeline


def test_cancelled_reproduction_choice_keeps_current_tab(
    widget, qtbot, tmp_path, monkeypatch
):
    path = save_workflow(tmp_path / "recorded.json", widget.pipeline)

    def cancel(workflow):
        raise ReproductionOpenCancelled()

    monkeypatch.setattr(widget, "_choose_workflow_reproduction", cancel)
    old = widget._workflow_tabs.current
    _send_drop(widget, path)
    qtbot.waitUntil(lambda: "cancelled" in widget.status_label.text())
    assert len(widget._workflow_tabs) == 1
    assert widget._workflow_tabs.current is old


def test_busy_drop_uses_existing_open_guard(widget, qtbot, tmp_path, monkeypatch):
    monkeypatch.setattr(
        widget, "_workflow_tab_switch_block_reason", lambda: "the source load finishes"
    )
    _send_drop(widget, tmp_path / "workflow.json")
    qtbot.waitUntil(lambda: "source load finishes" in widget.status_label.text())
    assert len(widget._workflow_tabs) == 1


def test_queued_drop_does_not_open_after_close(widget, qtbot, tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("A closing widget must not open workflows")

    monkeypatch.setattr(widget, "load_workflow_file", unexpected)
    _send_drop(widget, tmp_path / "workflow.json")
    widget._closing = True
    qtbot.waitUntil(lambda: widget._workflow_file_drop._pending_path is None)


def test_handler_ignores_other_windows_and_move_only_drops(qtbot, tmp_path):
    host = QWidget()
    other = QWidget()
    dialog = QDialog(host)
    child = QWidget(dialog)
    for window in (host, other):
        qtbot.addWidget(window)
    handler = WorkflowFileDropHandler(host)
    mime = _mime(QUrl.fromLocalFile(str(tmp_path / "workflow.json")))
    for target in (other, dialog, child):
        event = QDragEnterEvent(
            QPoint(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier
        )
        assert not handler.handle_event(target, event)
    event = QDragEnterEvent(QPoint(), Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
    assert handler.handle_event(host, event)
    assert not event.isAccepted()
    assert not handler.handle_event(host, QEvent(QEvent.Resize))


def test_node_and_image_drags_are_left_to_existing_handlers(widget):
    for mime in (
        _mime(QUrl.fromLocalFile("C:/images/test.tif")),
        QMimeData(),
    ):
        if not mime.hasUrls():
            mime.setData("application/x-napari-vipp-operation", b"gaussian_blur")
        event = QDragEnterEvent(
            QPoint(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier
        )
        assert not widget._workflow_file_drop.handle_event(
            widget.graph_view.viewport(), event
        )


def test_sibling_workflow_panel_does_not_claim_drop(qtbot, tmp_path):
    window = QWidget()
    qtbot.addWidget(window)
    first, second = QWidget(window), QWidget(window)
    handler = WorkflowFileDropHandler(first)
    mime = _mime(QUrl.fromLocalFile(str(tmp_path / "workflow.json")))
    event = QDragEnterEvent(QPoint(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
    assert not handler.handle_event(second, event)


def test_pending_drop_is_not_replaced_and_host_deletion_cancels_it(qtbot, tmp_path):
    host = QWidget()
    qtbot.addWidget(host)
    handler = WorkflowFileDropHandler(host)
    opened = []
    handler.openRequested.connect(opened.append)
    for i in range(2):
        mime = _mime(QUrl.fromLocalFile(str(tmp_path / f"workflow-{i}.json")))
        event = QDropEvent(QPointF(), Qt.CopyAction, mime, Qt.LeftButton, Qt.NoModifier)
        assert handler.handle_event(host, event)
        assert event.isAccepted() is (i == 0)
    assert handler._pending_path.name == "workflow-0.json"
    host.deleteLater()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    QApplication.processEvents()
    assert opened == []
