"""Resume is explicit, detached from open settings, and honest about reuse."""

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QTimer

from napari_vipp._tests.test_batch_results import _preview, _record, _result
from napari_vipp._tests.test_batch_workers import _cancelled_result
from napari_vipp._tests.test_ui_batch import _actions
from napari_vipp._widget import VippWidget
from napari_vipp.core.batch import BatchStatus
from napari_vipp.core.export import export_batch_runner_to_python
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_results import BatchResultsPanel
from napari_vipp.ui.batch_workers import (
    CollectionBatchResumeRequest,
    CollectionBatchWorker,
)


@pytest.mark.parametrize("confirmed", (True, False))
def test_resume_confirmation_uses_archive_not_open_settings(
    qtbot,
    monkeypatch,
    tmp_path,
    confirmed,
):
    from napari_vipp.ui import batch_resume

    dialog = CollectionBatchDialog(actions=_actions(_preview(tmp_path, 2), []))
    qtbot.addWidget(dialog)
    before = dialog.values()
    selected = tmp_path / "vipp_batch_manifest_previous.json"
    monkeypatch.setattr(
        batch_resume.QFileDialog,
        "getOpenFileName",
        lambda *_args: (str(selected), ""),
    )
    confirmations = []

    def confirm(box):
        confirmations.append(box.informativeText())
        button = next(
            item
            for item in box.buttons()
            if item.text() == ("Verify and resume" if confirmed else "Cancel")
        )
        button.click()

    monkeypatch.setattr(batch_resume.QMessageBox, "exec", confirm)
    requests = []
    dialog.runRequested.connect(requests.append)
    dialog._request_resume()
    assert requests == ([{"resume_manifest_path": str(selected)}] if confirmed else [])
    assert "not the currently open workflow" in confirmations[0]
    assert dialog.values() == before
    assert not selected.exists()  # No I/O or interpretation in the dialog.


def test_resume_view_is_separate_from_current_plan(qtbot, tmp_path):
    dialog = CollectionBatchDialog(actions=_actions(_preview(tmp_path, 2), []))
    qtbot.addWidget(dialog)
    before = dialog.values()
    selected = tmp_path / "vipp_batch_manifest_previous.json"
    dialog.begin_resume_run(selected)
    assert dialog.values() == before
    assert not dialog.results_panel.resume_button.isEnabled()
    assert "archive's workflow" in dialog.run_recap_label.text()
    assert not dialog.results_panel._item_review_available
    assert dialog.results_panel.review_item_button.isHidden()
    assert dialog.cancel_run_button.isEnabled()
    assert dialog.compute_summary_label.text() == "Saved run · verified resume"
    assert "saved run's outputs" in dialog.results_panel.items_hint.text()


def test_resume_worker_verification_is_responsive_and_cancellable(
    qtbot,
    monkeypatch,
    tmp_path,
):
    from napari_vipp.core import batch_resume

    entered, release = threading.Event(), threading.Event()
    captured = []

    def verify(path, **kwargs):
        captured.append((path, kwargs))
        entered.set()
        assert release.wait(5)
        assert kwargs["cancel_event"].is_set()
        return _cancelled_result(tmp_path)

    monkeypatch.setattr(batch_resume, "run_batch_resume_from_manifest", verify)
    worker = CollectionBatchWorker(
        CollectionBatchResumeRequest(37, "origin-tab", tmp_path / "manifest.json")
    )
    outcomes = []
    worker.signals.finished.connect(outcomes.append)
    thread = threading.Thread(target=worker.run)
    thread.start()
    responsive = []
    QTimer.singleShot(0, lambda: responsive.append(True))
    try:
        qtbot.waitUntil(lambda: entered.is_set() and bool(responsive))
        worker.cancel()
    finally:
        release.set()
        thread.join(5)
    qtbot.waitUntil(lambda: bool(outcomes))
    assert len(outcomes) == 1 and not outcomes[0].error
    assert outcomes[0].job_id == 37
    assert outcomes[0].origin_session_id == "origin-tab"
    assert outcomes[0].result.cancelled
    assert captured[0][0] == tmp_path / "manifest.json"


def test_host_resume_never_prepares_the_open_workflow(qtbot, tmp_path):
    dialog = CollectionBatchDialog(actions=_actions(_preview(tmp_path, 2), []))
    qtbot.addWidget(dialog)
    scheduled = []

    def forbidden(**_kwargs):
        pytest.fail("Resume must not prepare or persist the open workflow.")

    host = SimpleNamespace(
        _workflow_tabs=SimpleNamespace(current=SimpleNamespace(session_id="tab")),
        _compute_runtime_quarantined_reason="",
        _collection_batch_running=False,
        _active_thumbnail_contrast_run_id=None,
        _queued_thumbnail_contrast_limit_requests=(),
        _pending_thumbnail_contrast_limit_keys=(),
        _collection_batch_job_serial=0,
        _prepare_collection_batch_run=forbidden,
        _collection_batch_workers={},
        _sync_compute_policy_editability=lambda: None,
        _collection_batch_thread_pool=SimpleNamespace(start=scheduled.append),
        batch_navigator=SimpleNamespace(
            set_navigation_enabled=lambda _enabled: None,
            begin_batch_progress=lambda *_args: None,
        ),
    )
    for suffix in (
        "progress",
        "operation_progress",
        "preparation_progress",
        "finished",
    ):
        setattr(host, "_on_collection_batch_worker_" + suffix, lambda _update: None)
    VippWidget._start_collection_batch_worker(
        host,
        dialog,
        total=0,
        expected_items=(),
        resume_manifest_path=tmp_path / "manifest.json",
    )
    assert len(scheduled) == 1
    assert isinstance(scheduled[0].prepared, CollectionBatchResumeRequest)
    assert host._active_collection_batch_job.validation_config_path is None
    assert host._collection_batch_running


def test_report_distinguishes_verified_reuse_from_new_writes(qtbot, tmp_path):
    plan = _preview(tmp_path, 2)
    reused = replace(
        _record(plan.items[0], BatchStatus.COMPLETED),
        resumed_from_run_id="original-run",
    )
    written = _record(plan.items[1], BatchStatus.COMPLETED)
    result = replace(
        _result(plan, (reused, written)),
        saved_paths=(Path(written.outputs[0].path),),
    )
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.finish_run(result)
    assert panel.items_table.item(0, 1).text() == "Verified reuse"
    assert "verified" in panel.items_table.item(0, 2).text()
    assert panel.items_table.item(0, 3).text() != "00:10"
    panel.select_item(0)
    assert panel.output_table.item(0, 1).text().startswith("Verified reuse")
    assert "not rewritten" in panel.output_table.item(0, 1).toolTip()
    assert "1 verified reused" in panel.run_report.fields["Items"].text()
    assert panel.run_report.fields["Output files"].text() == (
        "1 saved this run · 1 verified from previous run"
    )


def _runner(tmp_path):
    namespace = {"__name__": "resume_runner", "__file__": str(tmp_path / "run.py")}
    exec(
        compile(export_batch_runner_to_python(), namespace["__file__"], "exec"),
        namespace,
    )
    return namespace


def test_generated_runner_resume_does_not_load_mutable_companions(tmp_path):
    runner = _runner(tmp_path)
    calls = []

    def resume(path, **kwargs):
        calls.append((path, kwargs))
        return SimpleNamespace(
            summary=dict(completed=1, partial=0, skipped=0, cancelled=0, failed=0),
            saved_paths=(),
            manifest_path=tmp_path / "continuation.json",
            cancelled=False,
            has_failures=False,
        )

    runner["run_batch_resume_from_manifest"] = resume
    runner["_load_run_inputs"] = lambda _args: pytest.fail("Mutable config was read")
    assert runner["main"](["--resume", "previous.json", "--progress"]) == 0
    assert calls[0][0] == "previous.json"
    assert calls[0][1]["cancel_event"].is_set() is False


@pytest.mark.parametrize(
    "override",
    (
        ["--config", "different.json"],
        ["--workflow", "different.json"],
        ["--compute-mode", "cpu"],
        ["--fallback-policy", "visible"],
        ["--node-preference", "node=cpu"],
    ),
)
def test_generated_runner_resume_rejects_parameter_overrides(tmp_path, override):
    runner = _runner(tmp_path)
    with pytest.raises(SystemExit) as exc:
        runner["main"](["--resume", "previous.json", *override])
    assert exc.value.code == 2
