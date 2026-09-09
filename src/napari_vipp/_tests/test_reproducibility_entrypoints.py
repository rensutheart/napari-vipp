"""Exports use the requested evidence, never the later edited batch graph."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QDialog

from napari_vipp._tests.test_batch_results import _preview, _record, _result
from napari_vipp._tests.test_ui_batch import _actions
from napari_vipp.core.batch import BatchStatus
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_results import BatchResultsPanel


def _dialog_spy(monkeypatch):
    from napari_vipp.ui import reproducibility

    calls = []

    class Dialog(QDialog):
        exported = Signal(str)

        def __init__(self, parent=None, **kwargs):
            super().__init__(parent)
            self.arguments = kwargs
            self.prepared = False
            calls.append(self)

        def prepare_report(self):
            self.prepared = True

    monkeypatch.setattr(reproducibility, "ReproducibilityDialog", Dialog)
    return calls


def test_batch_entry_uses_archived_path_and_requires_finished_evidence(
    qtbot, monkeypatch, tmp_path
):
    calls = _dialog_spy(monkeypatch)
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel._sync_artifact_buttons()
    assert not panel.export_package_button.isEnabled()
    manifest = tmp_path / "archived.json"
    manifest.write_text("{}", encoding="utf-8")
    panel._report_path = manifest
    panel._running = True
    panel._sync_artifact_buttons()
    assert not panel.export_package_button.isEnabled()
    panel._export_package()
    assert not calls
    panel._running = False
    panel._sync_artifact_buttons()
    assert panel.export_package_button.isEnabled()
    panel.export_package_button.click()
    assert calls[0].arguments["manifest_path"] == manifest
    assert "workflow" not in calls[0].arguments
    assert calls[0].prepared


@pytest.mark.parametrize("automatic", [False, True])
def test_missing_archived_manifest_syncs_report_and_footer_without_rechecking(
    qtbot, monkeypatch, tmp_path, automatic
):
    calls = _dialog_spy(monkeypatch)
    plan = _preview(tmp_path, 1)
    checked, previewed, run = [], [], []
    actions = replace(
        _actions(plan, previewed), check_batch=lambda *_: checked.append(1)
    )
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    result = _result(plan, (_record(plan.items[0], BatchStatus.COMPLETED),))
    result.manifest_path.parent.mkdir()
    result.manifest_path.touch()
    archived = result.manifest_path.with_name("archived-run.json")
    result = replace(result, manifest_archive_path=archived)
    dialog.apply_preview_result(plan, preview_representative=False)
    dialog.runRequested.connect(run.append)
    dialog.begin_run(1)
    dialog.finish_run(result)
    dialog.show()
    panel = dialog.results_panel
    report = panel.run_report
    assert panel.has_run_report
    assert dialog.next_button.text() == "Export package…"
    # The latest-manifest alias exists, but it must never replace this archive.
    assert result.manifest_path.is_file()
    assert not panel.export_package_button.isEnabled()
    assert not dialog.next_button.isEnabled()
    report_text = {name: label.text() for name, label in report.fields.items()}
    if automatic:
        qtbot.waitUntil(lambda: bool(panel._file_watcher.directories()))

    for available in (True, False, True):
        archived.touch() if available else archived.unlink()
        if not automatic:
            panel.refresh_files_button.click()
        qtbot.waitUntil(
            lambda expected=available: (
                panel.export_package_button.isEnabled() is expected
            ),
            timeout=3000,
        )
        assert dialog.next_button.isEnabled() is available
        assert panel.has_run_report
        assert {
            name: label.text() for name, label in report.fields.items()
        } == report_text
        assert checked == [] and previewed == [] and run == [] and calls == []

    # A click also rechecks the file in case the watcher has not fired yet.
    archived.unlink()
    panel._export_package()
    assert not dialog.next_button.isEnabled()
    assert not panel.export_package_button.isEnabled()
    assert calls == []
    for section in (1, 2):
        dialog.tabs.setCurrentIndex(section)
        assert dialog.next_button.text() == "View run report"
        assert dialog.next_button.isEnabled()
    dialog.tabs.setCurrentIndex(0)
    assert dialog.next_button.text() == "Check batch"
    assert checked == [] and previewed == [] and run == []


def test_workflow_entry_snapshots_current_recipe_without_a_manifest(qtbot, monkeypatch):
    from napari_vipp._widget import VippWidget

    calls = _dialog_spy(monkeypatch)
    parent = QDialog()
    qtbot.addWidget(parent)
    commits = []
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import workflow_snapshot_from_pipeline

    workflow = workflow_snapshot_from_pipeline(PrototypePipeline())
    parent._commit_crop_draft = lambda **kwargs: commits.append(kwargs)
    parent._current_history_snapshot = lambda: SimpleNamespace(workflow=workflow)
    VippWidget._export_reproducibility_package_dialog(parent)
    assert commits == [{"schedule_run": False}]
    document = calls[0].arguments["workflow"]
    assert isinstance(document, dict)
    assert {node["id"] for node in document["nodes"]} == {
        node.id for node in workflow.graph.nodes
    }
    assert "manifest_path" not in calls[0].arguments
    assert calls[0].prepared
