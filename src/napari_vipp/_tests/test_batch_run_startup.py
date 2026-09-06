"""Run startup is observable, cancellable, and never silently changes its plan."""

import threading
from dataclasses import replace
from types import MethodType, SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import QTimer

from napari_vipp._tests.test_batch_redesign_host import _check_host
from napari_vipp._tests.test_batch_redesign_host import batch_case as _case
from napari_vipp._widget import VippWidget
from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_SCRIPT_FILENAME,
    BatchPreflightProgress,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.source_identity import capture_local_source_identity
from napari_vipp.ui import batch_workers
from napari_vipp.ui.batch_workers import (
    CollectionBatchRunRequest,
    CollectionBatchWorker,
)


@pytest.fixture
def batch_case(tmp_path):
    return _case.__wrapped__(tmp_path)


def _run_host(batch_case):
    host, dialog, scheduled, _, failures, _ = _check_host(batch_case)
    continued = []
    dialog._preview_result = batch_case[0].preview(
        **batch_case[1], compute_request=ComputeRequest()
    )
    dialog._run_preparing = False
    dialog.begin_background_run_preparation = lambda: setattr(
        dialog, "_run_preparing", True
    )
    dialog.end_background_run_preparation = lambda: setattr(
        dialog, "_run_preparing", False
    )
    dialog.show_run_preparation_progress = dialog._check_progress.append
    dialog.show_preflight_error = lambda message, **_kw: failures.append(message)
    dialog.show_plan_refresh_required = failures.append
    host._reviewed_batch_source_identities = lambda _items: ()
    host._run_collection_batch_from_workspace = lambda *args, **kwargs: (
        continued.append((args, kwargs))
    )
    for name in ("_start_batch_run_preflight", "_present_batch_run_preflight"):
        setattr(host, name, MethodType(getattr(VippWidget, name), host))
    return host, dialog, scheduled, continued, failures


def test_run_preflight_keeps_qt_responsive_before_processing(
    qtbot, monkeypatch, batch_case
):
    host, dialog, scheduled, continued, failures = _run_host(batch_case)
    release = threading.Event()
    original = batch_workers.execute_prepared_collection_batch_preview

    def slow(prepared, **kwargs):
        kwargs["progress_callback"](BatchPreflightProgress("checking", total=2))
        assert release.wait(5)
        return original(prepared, **kwargs)

    monkeypatch.setattr(
        batch_workers, "execute_prepared_collection_batch_preview", slow
    )
    host._start_batch_run_preflight(dialog, dialog.values())
    assert dialog._run_preparing
    assert not continued
    thread = threading.Thread(target=scheduled[0].run)
    thread.start()
    heartbeat = []
    QTimer.singleShot(0, lambda: heartbeat.append(True))
    try:
        qtbot.waitUntil(lambda: bool(dialog._check_progress) and bool(heartbeat))
        assert not continued
        assert not batch_case[1]["output_dir"].exists()
    finally:
        release.set()
        thread.join(5)
    qtbot.waitUntil(lambda: bool(continued) or bool(failures))
    assert failures == []
    assert len(continued) == 1
    assert continued[0][1]["_fresh_preview"].total_items == 2
    assert not dialog._run_preparing


@pytest.mark.parametrize("change", ["settings", "workflow", "tab", "cancel"])
def test_changed_or_cancelled_preparation_never_starts_run(qtbot, batch_case, change):
    host, dialog, scheduled, continued, failures = _run_host(batch_case)
    host._start_batch_run_preflight(dialog, dialog.values())
    if change == "settings":
        batch_case[1]["pattern"] = "field_0.npy"
    elif change == "workflow":
        batch_case[2]["nodes"][-1]["params"]["tag"] = "changed"
    elif change == "tab":
        host._workflow_tab_is_active = lambda _id: False
    else:
        host._cancel_attached_batch_workspace_preview(dialog)
    scheduled[0].run()
    assert not continued
    assert not dialog._run_preparing
    assert not batch_case[1]["output_dir"].exists()
    assert failures or change == "cancel"


def _request(batch_case):
    controller, values, _ = batch_case
    snapshot = controller.prepare_preview(**values)
    reviewed = controller.preview(**values)
    output = values["output_dir"]
    return CollectionBatchRunRequest(
        job_id=5,
        origin_session_id="origin",
        workflow=snapshot.workflow,
        config=snapshot.config,
        workflow_path=snapshot.workflow_path,
        config_path=output / BATCH_CONFIG_FILENAME,
        script_path=output / BATCH_SCRIPT_FILENAME,
        expected_items=reviewed.items,
    )


def test_worker_reports_preparation_before_first_item_and_saves_real_outputs(
    qtbot, batch_case
):
    request = _request(batch_case)
    worker = CollectionBatchWorker(request)
    events, outcomes = [], []
    worker.signals.preparation_progress.connect(
        lambda update: events.append(update.progress)
    )
    worker.signals.progress.connect(lambda update: events.append(update.status))
    worker.signals.finished.connect(outcomes.append)
    worker.run()
    assert len(outcomes) == 1 and not outcomes[0].error
    result = outcomes[0].result
    assert result.summary["completed"] == 2
    assert len(result.saved_paths) == 2
    assert all(path.exists() for path in result.saved_paths)
    first_running = events.index("running")
    stages = [
        event.phase for event in events[:first_running] if not isinstance(event, str)
    ]
    assert "checking" in stages and "contract" in stages and "pipelines" in stages
    assert "artifacts" in stages


@pytest.mark.parametrize("cancel", [True, False])
def test_worker_cancel_or_changed_source_stops_before_artifacts(
    qtbot, batch_case, cancel
):
    request = _request(batch_case)
    worker = CollectionBatchWorker(request)
    outcomes, items = [], []
    worker.signals.finished.connect(outcomes.append)
    worker.signals.progress.connect(items.append)
    if cancel:
        worker.cancel()
    else:
        np.save(batch_case[1]["input_dir"] / "field_0.npy", np.ones((3, 4)))
    worker.run()
    assert not items
    assert not batch_case[1]["output_dir"].exists()
    assert outcomes[0].cancelled_before_start is cancel
    assert bool(outcomes[0].error) is not cancel


def test_reviewed_identity_verification_runs_inside_worker(
    qtbot, monkeypatch, batch_case
):
    host, dialog, scheduled, continued, failures = _run_host(batch_case)
    path = batch_case[1]["input_dir"] / "field_0.npy"
    identity = capture_local_source_identity(path)
    host._reviewed_batch_source_identities = lambda _items: ((path, identity),)
    host._start_batch_run_preflight(dialog, dialog.values())
    assert not dialog._check_progress
    monkeypatch.setattr(
        "napari_vipp.ui.batch_controller.verify_local_source_identity",
        lambda *_a, **_kw: pytest.fail(
            "The final scan already hashes collection sources"
        ),
    )
    scheduled[0].run()
    assert not failures
    assert continued
    assert any(
        event.phase == "checking" and event.byte_total
        for event in dialog._check_progress
    )


def test_reviewed_source_changed_during_preparation_requires_refresh(qtbot, batch_case):
    host, dialog, scheduled, continued, failures = _run_host(batch_case)
    path = batch_case[1]["input_dir"] / "field_0.npy"
    identity = capture_local_source_identity(path)
    host._reviewed_batch_source_identities = lambda _items: ((path, identity),)
    invalidations = []
    dialog.invalidate_for_source_change = lambda *args, **kw: invalidations.append(kw)
    host.batch_navigator = SimpleNamespace(set_session_stale=lambda *a, **kw: None)
    host.status_label = SimpleNamespace(setText=lambda text: failures.append(text))
    host._start_batch_run_preflight(dialog, dialog.values())
    np.save(path, np.ones((4, 5), dtype=np.uint8))
    scheduled[0].run()
    assert not continued
    assert invalidations == [{"before_run_started": True}]
    assert host._interactive_collection_batch_plan_stale
    assert "Press Refresh" in failures[-1]
    assert not batch_case[1]["output_dir"].exists()


def test_byte_progress_is_throttled_but_new_stages_and_final_bytes_are_delivered(
    monkeypatch,
):
    monkeypatch.setattr(batch_workers.time, "monotonic", lambda: 100.0)
    gate = batch_workers._PreparationProgressGate()
    event = BatchPreflightProgress("checking", byte_current=1, byte_total=100)
    assert gate.accepts(event)
    assert not gate.accepts(replace(event, byte_current=2))
    assert gate.accepts(replace(event, byte_current=100))
    assert gate.accepts(BatchPreflightProgress("contract"))
