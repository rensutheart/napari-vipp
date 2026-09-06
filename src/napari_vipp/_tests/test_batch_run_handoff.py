"""One final inventory scan is handed to execution, without trusting stale files."""

import os
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_batch_redesign_host import batch_case as _case
from napari_vipp._tests.test_batch_run_startup import _request
from napari_vipp.core import batch
from napari_vipp.core.batch import BatchPlan
from napari_vipp.core.source_identity import local_source_identity_from_bundle
from napari_vipp.ui.batch_controller import execute_prepared_collection_batch_preview
from napari_vipp.ui.batch_workers import CollectionBatchWorker


@pytest.fixture
def case(tmp_path):
    return _case.__wrapped__(tmp_path)


def _handoff(request):
    return replace(
        request,
        preflight_plan=BatchPlan(
            request.config,
            request.expected_items,
            request.config.resolve_path(request.config.output_dir),
        ),
    )


def test_run_scans_collection_once_then_hands_exact_plan_to_worker(
    qtbot, monkeypatch, case
):
    controller, values, _ = case
    request = _request(case)
    prepared = controller.prepare_preview(**values)
    prepared = replace(
        prepared,
        reviewed_identities=tuple(
            (
                item.source_paths[node_id],
                local_source_identity_from_bundle(source.container),
            )
            for item in request.expected_items
            for node_id, source in item.source_items.items()
        ),
    )
    scanned = []
    expand = batch._expand_source_items

    def track(paths, **kwargs):
        scanned.extend(paths)
        return expand(paths, **kwargs)

    monkeypatch.setattr(batch, "_expand_source_items", track)
    fresh = execute_prepared_collection_batch_preview(prepared)
    assert len(scanned) == 2
    monkeypatch.setattr(
        "napari_vipp.ui.batch_workers.preflight_batch",
        lambda *_a, **_kw: pytest.fail("Execution repeated the final collection scan"),
    )
    worker = CollectionBatchWorker(
        _handoff(replace(request, expected_items=fresh.items))
    )
    progress, outcomes = [], []
    worker.signals.preparation_progress.connect(
        lambda event: progress.append(event.progress.phase)
    )
    worker.signals.finished.connect(outcomes.append)
    worker.run()
    assert len(scanned) == 2
    assert len(outcomes) == 1 and not outcomes[0].error
    assert outcomes[0].result.summary["completed"] == 2
    assert "handoff" in progress and "pipelines" in progress
    assert not ({"discovered", "checking", "checked"} & set(progress))


@pytest.mark.parametrize(
    "change", ["settings", "workflow", "outputs", "items", "cancel"]
)
def test_invalid_or_cancelled_handoff_writes_no_artifacts(qtbot, case, change):
    request = _handoff(_request(case))
    if change == "settings":
        request = replace(
            request, config=replace(request.config, continue_on_error=False)
        )
    elif change == "workflow":
        request.workflow["nodes"][-1]["params"]["tag"] = "changed"
    elif change == "outputs":
        path = request.expected_items[0].outputs[0].path
        path.parent.mkdir()
        path.write_bytes(b"A new external output must not be overwritten")
    elif change == "items":
        request = replace(request, expected_items=request.expected_items[:1])
    outcomes = []
    worker = CollectionBatchWorker(request)
    worker.signals.finished.connect(outcomes.append)
    if change == "cancel":
        worker.cancel()
    worker.run()
    assert len(outcomes) == 1
    assert outcomes[0].error or outcomes[0].cancelled_before_start
    assert not request.config_path.exists()
    assert not request.workflow_path.exists()
    assert not request.script_path.exists()
    if change == "outputs":
        assert path.read_bytes() == b"A new external output must not be overwritten"


def test_changed_later_source_is_not_used_even_if_size_and_mtime_match(qtbot, case):
    request = _handoff(_request(case))
    path = request.expected_items[1].source_paths["input"]
    before = path.stat()
    np.save(path, np.full((4, 5), 99, dtype=np.uint8))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert path.stat().st_size == before.st_size
    assert path.stat().st_mtime_ns == before.st_mtime_ns
    outcomes = []
    worker = CollectionBatchWorker(request)
    worker.signals.finished.connect(outcomes.append)
    worker.run()
    assert len(outcomes) == 1 and not outcomes[0].error
    assert outcomes[0].result.summary["completed"] == 1
    assert outcomes[0].result.summary["failed"] == 1
    assert request.expected_items[0].outputs[0].path.exists()
    assert not request.expected_items[1].outputs[0].path.exists()


def test_reviewed_fixed_source_outside_collection_is_still_verified(case, monkeypatch):
    controller, values, _ = case
    prepared = controller.prepare_preview(**values)
    reviewed = controller.preview(**values)
    source = reviewed.items[0].source_items["input"]
    fixed_path = values["input_dir"].parent / "fixed-reference.npy"
    fixed_path.write_bytes(b"not part of the collection")
    prepared = replace(
        prepared,
        reviewed_identities=(
            (
                fixed_path,
                local_source_identity_from_bundle(source.container),
            ),
        ),
    )
    verified = []
    monkeypatch.setattr(
        "napari_vipp.ui.batch_controller.verify_local_source_identity",
        lambda path, _identity, **_kwargs: verified.append(path),
    )
    execute_prepared_collection_batch_preview(prepared)
    assert verified == [fixed_path]
