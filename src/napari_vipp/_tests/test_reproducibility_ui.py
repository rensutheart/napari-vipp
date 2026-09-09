"""Review authorization, detached evidence, offline resources and task lifetime."""

from __future__ import annotations

import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.compat import isalive
from qtpy.QtCore import QCoreApplication, QEvent, QThread, QThreadPool, QUrl

from napari_vipp.ui import reproducibility as module


def _package(title="Prepared evidence"):
    return SimpleNamespace(
        report_data={
            "privacy": {"paths_redacted": True, "anonymise_filenames": False},
            "omissions": ["No raw images"],
            "changes": ["Folders hidden"],
            "limitations": ["Recipe is not a past run"],
            "files": [{"path": "report.html", "description": "Offline report"}],
        },
        members={
            "report.html": f"<h1>{title}</h1><p>Exact prepared contents.</p>".encode(),
            "workflow.json": b'{"description": "review me"}',
        },
    )


@pytest.fixture
def service(qapp, monkeypatch):
    service = module._PackageTasks(qapp)
    monkeypatch.setattr(module, "_task_service", lambda: service)
    yield service
    for task_id in tuple(service._jobs):
        service.cancel(task_id)
    assert QThreadPool.globalInstance().waitForDone(5000)
    qapp.processEvents()
    service.deleteLater()


def _dialog(qtbot, **kwargs):
    dialog = module.ReproducibilityDialog(**kwargs)
    qtbot.addWidget(dialog)
    return dialog


def _prepare(dialog, qtbot):
    dialog.prepare_report()
    qtbot.waitUntil(lambda: dialog.prepared_package is not None)


def test_requires_exactly_one_evidence_source(qtbot, service):
    with pytest.raises(ValueError, match="Choose one"):
        _dialog(qtbot)
    with pytest.raises(ValueError, match="Choose one"):
        _dialog(qtbot, workflow={}, manifest_path="manifest.json")


def test_review_is_required_and_input_is_detached_off_gui_thread(
    qtbot, monkeypatch, service
):
    calls = []
    source = {"nodes": [{"name": "Opening snapshot"}]}
    package = _package()
    gui_thread = QThread.currentThread()

    def build(**kwargs):
        calls.append((kwargs, QThread.currentThread()))
        return package

    monkeypatch.setattr(module, "build_reproducibility_package", build)
    dialog = _dialog(qtbot, workflow=source, title="Reviewed title")
    assert not calls
    assert "not a verified record" in dialog.source_label.text()
    assert not dialog.export_button.isEnabled()
    assert not dialog.review_checkbox.isEnabled()
    source["nodes"][0]["name"] = "Later edit"
    dialog.notes_edit.setPlainText("Optional note")
    dialog.anonymise_checkbox.setChecked(True)
    _prepare(dialog, qtbot)
    assert calls[0][1] != gui_thread
    arguments = calls[0][0]
    assert arguments == {
        "workflow": {"nodes": [{"name": "Opening snapshot"}]},
        "manifest_path": None,
        "title": "Reviewed title",
        "anonymise_filenames": True,
        "notes": "Optional note",
    }
    assert dialog.prepared_package is package
    assert "Exact prepared contents." in dialog.report_browser.toPlainText()
    assert dialog.contents_list.count() == 2
    assert "Recipe is not a past run" in dialog.privacy_browser.toPlainText()
    assert "Folders hidden" in dialog.privacy_browser.toPlainText()
    assert "Nothing is uploaded" in dialog.privacy_label.text()
    assert "scientific labels" in dialog.privacy_label.text()
    assert "Workflow notes are included" in dialog.privacy_label.text()
    privacy_text = dialog.privacy_browser.toPlainText()
    assert "Original filenames are included." in privacy_text
    assert "Deliberately left out\n• No raw images" in privacy_text
    assert "Changes made for sharing\n• Folders hidden" in privacy_text
    assert "Limits to keep in mind\n• Recipe is not a past run" in privacy_text
    assert "paths_redacted" not in privacy_text
    assert dialog.review_checkbox.isEnabled()
    assert not dialog.export_button.isEnabled()
    dialog.review_checkbox.setChecked(True)
    assert dialog.export_button.isEnabled()


def test_manifest_source_never_reads_open_graph(qtbot, monkeypatch, service, tmp_path):
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return _package()

    monkeypatch.setattr(module, "build_reproducibility_package", build)
    manifest = tmp_path / "manifest.json"
    dialog = _dialog(qtbot, manifest_path=manifest)
    assert "Archived batch evidence" in dialog.source_label.text()
    _prepare(dialog, qtbot)
    assert calls[0]["workflow"] is None
    assert calls[0]["manifest_path"] == manifest


@pytest.mark.parametrize("field", ["title", "notes", "anonymise"])
def test_editing_any_package_input_invalidates_preview_and_review(
    qtbot, monkeypatch, service, field
):
    monkeypatch.setattr(
        module, "build_reproducibility_package", lambda **kw: _package()
    )
    dialog = _dialog(qtbot, workflow={})
    _prepare(dialog, qtbot)
    dialog.review_checkbox.setChecked(True)
    if field == "title":
        dialog.title_edit.setText("New title")
    elif field == "notes":
        dialog.notes_edit.setPlainText("New note")
    else:
        dialog.anonymise_checkbox.setChecked(True)
    assert dialog.prepared_package is None
    assert not dialog.review_checkbox.isChecked()
    assert not dialog.review_checkbox.isEnabled()
    assert not dialog.export_button.isEnabled()
    assert dialog.contents_list.count() == 0


def test_stale_preparation_cannot_restore_preview(qtbot, monkeypatch, service):
    entered, release = threading.Event(), threading.Event()
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        entered.set()
        assert release.wait(4)
        return _package(kwargs["title"])

    monkeypatch.setattr(module, "build_reproducibility_package", build)
    dialog = _dialog(qtbot, workflow={})
    dialog.prepare_report()
    try:
        qtbot.waitUntil(entered.is_set)
        dialog.title_edit.setText("New generation")
        dialog.prepare_report()
        assert len(calls) == 1  # No unbounded queue of obsolete preparations.
        assert not dialog.export_button.isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: dialog._active_task is None)
    assert dialog.prepared_package is None
    assert "Prepare a new report" in dialog.status_label.text()
    _prepare(dialog, qtbot)
    assert "New generation" in dialog.report_browser.toPlainText()


def test_queued_completion_is_rejected_after_invalidation(
    qtbot, monkeypatch, service
):
    monkeypatch.setattr(
        module, "build_reproducibility_package", lambda **kw: _package()
    )
    dialog = _dialog(qtbot, workflow={})
    dialog.prepare_report()
    assert QThreadPool.globalInstance().waitForDone(5000)
    dialog.notes_edit.setPlainText("Changed before signal delivery")
    qtbot.waitUntil(lambda: dialog._active_task is None)
    assert dialog.prepared_package is None
    assert not dialog.export_button.isEnabled()


@pytest.mark.parametrize("delete_native", [False, True])
def test_close_or_native_deletion_during_preparation_never_uses_dead_widgets(
    qtbot, monkeypatch, service, delete_native
):
    entered, release = threading.Event(), threading.Event()

    def build(**kwargs):
        entered.set()
        assert release.wait(4)
        return _package()

    monkeypatch.setattr(module, "build_reproducibility_package", build)
    dialog = _dialog(qtbot, workflow={})
    dialog.prepare_report()
    try:
        qtbot.waitUntil(entered.is_set)
        if delete_native:
            dialog.deleteLater()
            QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
            assert not isalive(dialog)
        else:
            dialog.reject()
            assert dialog._closed
            assert dialog.prepared_package is None
        assert service._jobs  # Closing does not join/block the background job.
    finally:
        release.set()
    qtbot.waitUntil(lambda: not service._jobs)
    if not delete_native:
        assert dialog.prepared_package is None


def test_preparation_errors_never_authorize_export(qtbot, monkeypatch, service):
    def fail(**kwargs):
        raise ValueError("Archived evidence is incomplete")

    monkeypatch.setattr(module, "build_reproducibility_package", fail)
    dialog = _dialog(qtbot, manifest_path="manifest.json")
    dialog.prepare_report()
    qtbot.waitUntil(lambda: dialog._active_task is None)
    assert "Archived evidence is incomplete" in dialog.status_label.text()
    assert not dialog.review_checkbox.isEnabled()
    assert not dialog.export_button.isEnabled()


def test_report_blocks_links_and_resource_loading(qtbot, service):
    dialog = _dialog(qtbot, workflow={})
    browser = dialog.report_browser
    assert not browser.openLinks()
    assert not browser.openExternalLinks()
    for name in ("https://example.invalid/private.png", "file:///private/image.png"):
        assert bytes(browser.loadResource(2, QUrl(name))) == b""
    browser.setHtml('<h1>Safe text</h1><img src="https://example.invalid/private.png">')
    assert "Safe text" in browser.toPlainText()


def test_only_explicit_official_installation_links_can_open(
    qtbot, monkeypatch, service
):
    from napari_vipp.core.reproducibility_install import INSTALLATION_GUIDE_URL
    from napari_vipp.core.updates import RELEASES_URL

    opened = []
    monkeypatch.setattr(
        module.QDesktopServices, "openUrl",
        lambda url: opened.append(url.toString()) or True,
    )
    package = _package()
    package.report_data.update({
        "package_kind": "recorded_batch_run",
        "environment": {
            "run": {"packages": {"napari-vipp": "0.15.0a2"}},
            "export": {"packages": {"napari-vipp": "0.16.0"}},
        },
    })
    dialog = _dialog(qtbot, manifest_path="manifest.json")
    release = RELEASES_URL + "/tag/v0.15.0a2"
    dialog.report_browser.anchorClicked.emit(QUrl(release))
    assert not opened  # No prepared evidence yet.
    dialog._show_package(package)
    assert not opened  # Rendering never fetches anything.
    for target in (release, INSTALLATION_GUIDE_URL):
        dialog.report_browser.anchorClicked.emit(QUrl(target))
    assert opened == [release, INSTALLATION_GUIDE_URL]
    for target in (
        "file:///private/workflow.json", "../workflow.json", "javascript:alert(1)",
        "https://bad.test/", release + "?private=sample", release + "/anything",
        RELEASES_URL + "/tag/v0.16.0", "https://github.com.evil.test/", "//bad.test/",
    ):
        dialog.report_browser.anchorClicked.emit(QUrl(target))
    assert opened == [release, INSTALLATION_GUIDE_URL]
    dialog.report_browser.anchorClicked.emit(QUrl("workflow.json"))
    assert "Export and unzip" in dialog.status_label.text()
    assert opened == [release, INSTALLATION_GUIDE_URL]
    dialog._invalidate()
    dialog.report_browser.anchorClicked.emit(QUrl(release))
    assert opened == [release, INSTALLATION_GUIDE_URL]


def test_export_guards_cancel_and_existing_destination(
    qtbot, monkeypatch, service, tmp_path
):
    calls = []
    monkeypatch.setattr(
        module, "build_reproducibility_package", lambda **kw: _package()
    )
    monkeypatch.setattr(
        module, "export_reproducibility_package", lambda **kw: calls.append(kw)
    )
    chosen = []

    def choose(*args, **kwargs):
        chosen.append(1)
        return "", ""

    monkeypatch.setattr(module.QFileDialog, "getSaveFileName", choose)
    dialog = _dialog(qtbot, workflow={})
    dialog.export_package()
    _prepare(dialog, qtbot)
    dialog.export_package()
    assert not chosen
    dialog.review_checkbox.setChecked(True)
    dialog.export_package()
    assert chosen and not calls
    assert dialog.export_button.isEnabled()
    destination = tmp_path / "existing.zip"
    destination.write_bytes(b"original package")
    monkeypatch.setattr(
        module.QFileDialog, "getSaveFileName", lambda *a, **kw: (str(destination), "")
    )
    dialog.export_package()
    assert not calls
    assert destination.read_bytes() == b"original package"
    assert "not overwritten" in dialog.status_label.text()


def test_export_writes_exact_reviewed_snapshot_off_thread_and_holds_close(
    qtbot, monkeypatch, service, tmp_path
):
    package = _package()
    calls = []
    entered, release = threading.Event(), threading.Event()
    destination = tmp_path / "reviewed.zip"
    gui_thread = QThread.currentThread()
    monkeypatch.setattr(module, "build_reproducibility_package", lambda **kw: package)
    monkeypatch.setattr(
        module.QFileDialog, "getSaveFileName", lambda *a, **kw: (str(destination), "")
    )

    def write(**kwargs):
        calls.append((kwargs, QThread.currentThread()))
        entered.set()
        assert release.wait(4)
        return Path(kwargs["path"])

    monkeypatch.setattr(module, "export_reproducibility_package", write)
    dialog = _dialog(qtbot, workflow={})
    _prepare(dialog, qtbot)
    dialog.review_checkbox.setChecked(True)
    exported = []
    dialog.exported.connect(exported.append)
    dialog.export_package()
    try:
        qtbot.waitUntil(entered.is_set)
        assert calls[0][0]["package"] is package
        assert calls[0][0]["overwrite"] is False
        assert calls[0][1] != gui_thread
        assert not dialog.title_edit.isEnabled()
        assert not dialog.close_button.isEnabled()
        dialog.reject()
        assert not dialog._closed
    finally:
        release.set()
    qtbot.waitUntil(lambda: bool(exported))
    assert exported == [str(destination)]
    assert dialog.close_button.isEnabled()
    assert "saved locally" in dialog.status_label.text()


def test_export_error_keeps_reviewed_snapshot_available(
    qtbot, monkeypatch, service, tmp_path
):
    monkeypatch.setattr(
        module, "build_reproducibility_package", lambda **kw: _package()
    )
    monkeypatch.setattr(
        module.QFileDialog, "getSaveFileName",
        lambda *a, **kw: (str(tmp_path / "package.zip"), ""),
    )

    def fail(**kwargs):
        raise PermissionError("Destination is read-only")

    monkeypatch.setattr(module, "export_reproducibility_package", fail)
    dialog = _dialog(qtbot, workflow={})
    _prepare(dialog, qtbot)
    dialog.review_checkbox.setChecked(True)
    dialog.export_package()
    qtbot.waitUntil(lambda: dialog._active_task is None)
    assert "Destination is read-only" in dialog.status_label.text()
    assert dialog.prepared_package is not None
    assert dialog.export_button.isEnabled()
    assert dialog.close_button.isEnabled()


def test_narrow_report_preview_wraps_technical_parameters(
    qtbot, service
):
    from napari_vipp.core.reproducibility_report import render_reproducibility_report

    data = {
        "title": "Reviewed synthetic analysis",
        "package_kind": "recorded_batch_run",
        "summary": {"nodes": 2, "items": 1, "completed": 1, "saved_outputs": 1},
        "workflow": [
            {
                "title": "Set Pixel Size / Units",
                "operation": "set_pixel_size",
                "parameters": {
                    "voxel_size_x": 0.123456789,
                    "unit": "micrometer",
                    "axes": {
                        "Z": {"spacing": 0.8, "origin": 1.25},
                        "Y": {"spacing": 0.2, "origin": 100.125},
                    },
                },
            },
        ],
        "sources": [{"name": "image.tif", "sha256": "a" * 64}],
        "environment": {"export": {"python": "3.12", "packages": {"numpy": "2.5"}}},
    }
    report = render_reproducibility_report(data)
    assert "<pre>" not in report
    assert "0.123456789" in report
    dialog = _dialog(qtbot, manifest_path="not-read.json")
    dialog.resize(900, 720)
    dialog._show_package(
        SimpleNamespace(report_data=data, members={"report.html": report.encode()})
    )
    dialog.show()
    qtbot.waitUntil(lambda: dialog.report_browser.viewport().width() > 0)
    dialog.resize(700, 720)
    # Resizing schedules the browser's reflow timer for the next event turn.
    qtbot.waitUntil(
        lambda: dialog.report_browser.horizontalScrollBar().maximum() == 0
    )
    assert "0.123456789" in dialog.report_browser.toPlainText()
