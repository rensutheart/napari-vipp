from __future__ import annotations

from pathlib import Path

from napari_vipp.core.batch import (
    BatchExecutionProgress,
    BatchItemRecord,
    BatchManifest,
    BatchRunResult,
    BatchStatus,
)
from napari_vipp.ui.batch_controller import PreparedCollectionBatchPreview
from napari_vipp.ui.batch_workers import (
    BatchWorkspacePreviewWorker,
    BatchWorkspacePreviewWorkerSpec,
    CollectionBatchWorker,
    PreparedCollectionBatchRun,
)


def _cancelled_result(tmp_path: Path) -> BatchRunResult:
    item = BatchItemRecord(
        index=1,
        batch_id="sample",
        sources=(),
        outputs=(),
        status=BatchStatus.CANCELLED,
        error_type="OperationCancelled",
        error_message="cancelled",
    )
    manifest = BatchManifest(
        run_id="run-1",
        started_at="2026-01-01T00:00:00Z",
        workflow_sha256="workflow",
        config_sha256="config",
        effective_config_sha256="effective",
        workflow_file="workflow.json",
        config_file="config.json",
        output_dir=str(tmp_path),
        runtime={},
        workflow_document={},
        config_document={},
        compute={"runtime_cleanup_succeeded": True},
        items=(item,),
    )
    return BatchRunResult(
        manifest=manifest,
        manifest_path=tmp_path / "manifest.json",
        saved_paths=(),
    )


def test_worker_carries_cancel_and_both_progress_channels(
    qtbot,
    monkeypatch,
    tmp_path,
):
    del qtbot
    captured_cancel_event = None

    def fake_run_batch(*_args, **kwargs):
        nonlocal captured_cancel_event
        captured_cancel_event = kwargs["cancel_event"]
        assert captured_cancel_event.is_set()
        kwargs["progress_callback"](1, 1, "sample", "running")
        kwargs["execution_progress_callback"](
            BatchExecutionProgress(
                item_index=1,
                item_total=1,
                batch_id="sample",
                node_id="gaussian",
                operation_id="gaussian_blur",
                current=2,
                total=4,
                message="GPU tile 2 of 4",
            )
        )
        return _cancelled_result(tmp_path)

    monkeypatch.setattr(
        "napari_vipp.ui.batch_workers.run_batch",
        fake_run_batch,
    )
    prepared = PreparedCollectionBatchRun(
        job_id=17,
        origin_session_id="tab-a",
        workflow={},
        config=object(),
        workflow_path=tmp_path / "workflow.json",
        config_path=tmp_path / "config.json",
        plan=object(),
        artifact_paths=(tmp_path / "workflow.json",),
    )
    worker = CollectionBatchWorker(prepared)
    coarse = []
    nested = []
    outcomes = []
    worker.signals.progress.connect(coarse.append)
    worker.signals.operation_progress.connect(nested.append)
    worker.signals.finished.connect(outcomes.append)

    worker.cancel()
    worker.cancel()
    worker.run()

    assert captured_cancel_event is not None
    assert worker.cancellation_requested
    assert [(update.job_id, update.batch_id) for update in coarse] == [(17, "sample")]
    assert len(nested) == 1
    assert nested[0].job_id == 17
    assert nested[0].progress.operation_id == "gaussian_blur"
    assert nested[0].progress.current == 2
    assert len(outcomes) == 1
    assert outcomes[0].job_id == 17
    assert outcomes[0].error == ""
    assert outcomes[0].result is not None
    assert outcomes[0].result.cancelled
    assert outcomes[0].result.artifact_paths == (tmp_path / "workflow.json",)


def test_workspace_preview_worker_suppresses_a_cancelled_request(tmp_path):
    prepared = PreparedCollectionBatchPreview(
        workflow={},
        config=object(),
        workflow_path=tmp_path / "workflow.json",
        preview_limit=25,
        explicit_outputs=False,
    )
    worker = BatchWorkspacePreviewWorker(
        BatchWorkspacePreviewWorkerSpec(7, "tab-a", prepared)
    )
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.cancel()
    worker.run()

    assert worker.cancellation_requested
    assert len(outcomes) == 1
    assert outcomes[0].request_id == 7
    assert outcomes[0].origin_session_id == "tab-a"
    assert outcomes[0].cancelled
    assert outcomes[0].result is None
    assert outcomes[0].error is None


def test_workspace_preview_worker_preserves_the_verification_error(
    monkeypatch,
    tmp_path,
):
    prepared = PreparedCollectionBatchPreview(
        workflow={},
        config=object(),
        workflow_path=tmp_path / "workflow.json",
        preview_limit=25,
        explicit_outputs=False,
    )
    worker = BatchWorkspacePreviewWorker(
        BatchWorkspacePreviewWorkerSpec(8, "tab-b", prepared)
    )
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    def fail(_prepared):
        raise ValueError("source revision changed")

    monkeypatch.setattr(
        "napari_vipp.ui.batch_workers.execute_prepared_collection_batch_preview",
        fail,
    )
    worker.run()

    assert len(outcomes) == 1
    assert isinstance(outcomes[0].error, ValueError)
    assert str(outcomes[0].error) == "source revision changed"
    assert outcomes[0].result is None
    assert not outcomes[0].cancelled
