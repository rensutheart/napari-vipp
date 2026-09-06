"""Progressive batch checks stay responsive, read-only, and request-owned."""

from __future__ import annotations

import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from napari_vipp._tests.test_batch_redesign_host import (
    _check_host,
)
from napari_vipp._tests.test_batch_redesign_host import (
    batch_case as _small_file_batch_case,
)
from napari_vipp.core import batch as batch_core
from napari_vipp.core.batch import BatchPreflightProgress
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.ui import batch_workers
from napari_vipp.ui.batch_workers import BatchWorkspacePreviewProgress


@pytest.fixture(name="batch_case")
def _batch_case(tmp_path):
    return _small_file_batch_case.__wrapped__(tmp_path)


def test_inventory_reaches_items_view_before_slow_checks_finish(
    qtbot, monkeypatch, batch_case
):
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    started = threading.Event()
    release = threading.Event()
    original_execute = batch_workers.execute_prepared_collection_batch_preview

    def execute(prepared, *, progress_callback, cancel_callback):
        def publish_progress(progress):
            progress_callback(progress)
            if progress.phase == "discovered":
                started.set()
                assert release.wait(timeout=5)

        return original_execute(
            prepared,
            progress_callback=publish_progress,
            cancel_callback=cancel_callback,
        )

    def forbid_pixels(*_args, **_kwargs):
        raise AssertionError("Check must never decode scientific image pixels")

    monkeypatch.setattr(
        batch_workers, "execute_prepared_collection_batch_preview", execute
    )
    monkeypatch.setattr(batch_core, "read_image", forbid_pixels)
    assert host._check_collection_batch(dialog.values())
    thread = threading.Thread(target=scheduled[0].run)
    thread.start()
    try:
        qtbot.waitUntil(lambda: started.is_set() and bool(dialog._check_progress))
        progress, options = dialog._check_progress[0]
        assert progress.phase == "discovered"
        assert progress.total == 2
        assert sum(len(paths) for _node, _title, paths in progress.source_paths) == 2
        assert options == {"reveal_items": True}
        assert dialog._checking_plan
        assert dialog._preview_result is None
        assert published == failures == []
        assert host._interactive_collection_batch_items == ()
        assert not batch_case[1]["output_dir"].exists()
    finally:
        release.set()
        thread.join(timeout=5)
    assert not thread.is_alive()
    qtbot.waitUntil(lambda: bool(published) or bool(failures))
    assert len(published) == 1
    assert failures == []
    assert not batch_case[1]["output_dir"].exists()


def test_worker_stops_publishing_progress_when_cancelled_mid_check(
    qtbot, monkeypatch, batch_case
):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    worker = scheduled[0]
    updates = []
    outcomes = []
    worker.signals.progress.connect(updates.append)
    worker.signals.finished.connect(outcomes.append)

    def execute(_prepared, *, progress_callback, cancel_callback):
        progress_callback(BatchPreflightProgress("checking", total=2))
        worker.cancel()
        assert cancel_callback()
        progress_callback(BatchPreflightProgress("checked", current=1, total=2))
        raise OperationCancelled("Cancelled between source chunks")

    monkeypatch.setattr(
        batch_workers, "execute_prepared_collection_batch_preview", execute
    )
    worker.run()
    assert len(updates) == 1
    assert updates[0].request_id == worker.spec.request_id
    assert updates[0].origin_session_id == "origin"
    assert len(outcomes) == 1
    assert outcomes[0].cancelled
    assert outcomes[0].error is None
    assert outcomes[0].result is None
    assert published == failures == []
    assert not batch_case[1]["output_dir"].exists()


@pytest.mark.parametrize(
    "unavailable",
    [
        "superseded",
        "closed",
        "wrong-owner",
        "replaced-dialog",
        "missing-tab",
        "inactive-tab",
    ],
)
def test_stale_progress_never_repopulates_items(qtbot, batch_case, unavailable):
    del qtbot
    host, dialog, scheduled, _published, _failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    spec = scheduled[0].spec
    update = BatchWorkspacePreviewProgress(
        spec.request_id,
        spec.origin_session_id,
        BatchPreflightProgress("checking", total=2),
    )
    if unavailable == "superseded":
        assert host._check_collection_batch(dialog.values())
    elif unavailable == "closed":
        host._closing = True
    elif unavailable == "wrong-owner":
        update = replace(update, origin_session_id="other-tab")
    elif unavailable == "replaced-dialog":
        host._active_collection_batch_dialog = SimpleNamespace()
    elif unavailable == "missing-tab":
        host._workflow_tab_session = lambda _id: None
    elif unavailable == "inactive-tab":
        host._workflow_tab_is_active = lambda _id: False
    host._on_batch_workspace_preview_progress(update)
    assert dialog._check_progress == []


def test_late_progress_cannot_replace_finished_plan(qtbot, batch_case):
    del qtbot
    host, dialog, scheduled, published, failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    worker = scheduled[0]
    worker.run()
    assert len(published) == 1
    assert failures == []
    completed_progress = tuple(dialog._check_progress)
    host._on_batch_workspace_preview_progress(
        BatchWorkspacePreviewProgress(
            worker.spec.request_id,
            worker.spec.origin_session_id,
            BatchPreflightProgress("checking", total=2),
        )
    )
    assert tuple(dialog._check_progress) == completed_progress
    assert dialog._preview_result is published[0]
    assert not dialog._checking_plan


@pytest.mark.parametrize("purpose", ["restore", "items"])
def test_background_restore_and_item_recheck_do_not_force_tab_navigation(
    qtbot, batch_case, purpose
):
    del qtbot
    host, dialog, scheduled, _published, _failures, _rechecks = _check_host(batch_case)
    assert host._check_collection_batch(dialog.values())
    spec = scheduled[0].spec
    context = host._batch_workspace_preview_contexts[spec.request_id]
    host._batch_workspace_preview_contexts[spec.request_id] = replace(
        context, purpose=purpose
    )
    host._on_batch_workspace_preview_progress(
        BatchWorkspacePreviewProgress(
            spec.request_id,
            spec.origin_session_id,
            BatchPreflightProgress("checking", total=2),
        )
    )
    assert dialog._check_progress[0][1] == {"reveal_items": False}
