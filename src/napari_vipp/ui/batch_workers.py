"""Qt-free collection batch execution wrapped in a small Qt runnable."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.batch import (
    BatchConfig,
    BatchExecutionProgress,
    BatchItemPlan,
    BatchPlan,
    BatchPreflightProgress,
    BatchRunResult,
    atomic_write_json,
    atomic_write_text,
    bind_batch_plan_source_items,
    preflight_batch,
    run_batch,
    save_batch_config,
    validate_batch_config,
)
from napari_vipp.core.export import export_batch_runner_to_python
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.ui.batch import BatchPreviewResult
from napari_vipp.ui.batch_controller import (
    PreparedCollectionBatchPreview,
    execute_prepared_collection_batch_preview,
)


@dataclass(frozen=True, slots=True)
class BatchWorkspacePreviewWorkerSpec:
    """Generation-owned immutable inputs for restoring one saved workspace."""

    request_id: int
    origin_session_id: str
    prepared: PreparedCollectionBatchPreview


@dataclass(frozen=True, slots=True)
class BatchWorkspacePreviewWorkerOutcome:
    """Terminal metadata-only verification result returned to the GUI thread."""

    request_id: int
    origin_session_id: str
    result: BatchPreviewResult | None = None
    error: Exception | None = None
    cancelled: bool = False


@dataclass(frozen=True, slots=True)
class BatchWorkspacePreviewProgress:
    """Read-only check progress tied to one dialog request and workflow tab."""

    request_id: int
    origin_session_id: str
    progress: BatchPreflightProgress


class _BatchWorkspacePreviewWorkerSignals(QObject):
    progress = Signal(object)
    finished = Signal(object)


class _PreparationProgressGate:
    """Keep large-file byte updates responsive without flooding Qt's event queue."""

    def __init__(self):
        self._key = None
        self._last_emit = 0.0

    def accepts(self, event):
        key = (event.phase, event.current, event.total, event.path)
        now = time.monotonic()
        if (
            key == self._key
            and event.byte_total
            and event.byte_current < event.byte_total
            and now - self._last_emit < 0.1
        ):
            return False
        self._key, self._last_emit = key, now
        return True


class BatchWorkspacePreviewWorker(QRunnable):
    """Verify an attached batch workspace without reading scientific pixels."""

    def __init__(self, spec: BatchWorkspacePreviewWorkerSpec) -> None:
        super().__init__()
        self.spec = spec
        self._progress_gate = _PreparationProgressGate()
        self.signals = _BatchWorkspacePreviewWorkerSignals()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Suppress publication of a queued or superseded verification."""

        self._cancel_event.set()

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel_event.is_set()

    def _report_progress(self, progress: BatchPreflightProgress) -> None:
        if not self._cancel_event.is_set() and self._progress_gate.accepts(progress):
            self.signals.progress.emit(
                BatchWorkspacePreviewProgress(
                    self.spec.request_id, self.spec.origin_session_id, progress
                )
            )

    def run(self) -> None:
        spec = self.spec
        if self._cancel_event.is_set():
            self.signals.finished.emit(
                BatchWorkspacePreviewWorkerOutcome(
                    spec.request_id,
                    spec.origin_session_id,
                    cancelled=True,
                )
            )
            return
        try:
            result = execute_prepared_collection_batch_preview(
                spec.prepared,
                progress_callback=self._report_progress,
                cancel_callback=self._cancel_event.is_set,
            )
        except Exception as exc:
            outcome = BatchWorkspacePreviewWorkerOutcome(
                spec.request_id,
                spec.origin_session_id,
                error=None if self._cancel_event.is_set() else exc,
                cancelled=self._cancel_event.is_set(),
            )
        else:
            outcome = BatchWorkspacePreviewWorkerOutcome(
                spec.request_id,
                spec.origin_session_id,
                result=result,
                cancelled=self._cancel_event.is_set(),
            )
        self.signals.finished.emit(outcome)


@dataclass(frozen=True, slots=True)
class PreparedCollectionBatchRun:
    """Verified inputs and persisted configuration ready for batch execution."""

    job_id: int
    origin_session_id: str
    workflow: dict
    config: BatchConfig
    workflow_path: Path
    config_path: Path
    plan: BatchPlan
    artifact_paths: tuple[Path, ...]
    performance_history_path: Path | None = None


@dataclass(frozen=True, slots=True)
class CollectionBatchRunRequest:
    """GUI snapshot; all source reads and artifact writes happen in the worker."""

    job_id: int
    origin_session_id: str
    workflow: dict
    config: BatchConfig
    workflow_path: Path
    config_path: Path
    script_path: Path
    expected_items: tuple[BatchItemPlan, ...] | None = None
    performance_history_path: Path | None = None
    # Only the just-completed Run preflight supplies this, never the older
    # displayed review. Execution still verifies each source before using it.
    preflight_plan: BatchPlan | None = None


def prepare_collection_batch_run(
    request,
    *,
    progress_callback=None,
    cancel_callback=None,
):
    """Finish the Run preflight, without repeating an already validated scan."""
    config = request.config
    if cancel_callback is not None and cancel_callback():
        raise OperationCancelled("Batch preparation cancelled before saving artifacts.")
    if request.preflight_plan is None:
        # Direct/headless callers have not passed through the desktop preflight.
        plan = preflight_batch(
            request.workflow,
            config,
            workflow_path=request.workflow_path,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
    else:
        plan = request.preflight_plan
        if plan.config != config or plan.output_dir != config.resolve_path(
            config.output_dir
        ):
            raise ValueError(
                "The validated run plan does not match the current settings."
            )
        validate_batch_config(
            request.workflow, config, workflow_path=request.workflow_path
        )
        if plan.has_collisions:
            raise FileExistsError("The validated run plan still has output collisions.")
        # A user may have spent time on the overwrite prompt. Recheck cheap
        # destination presence before any artifact writes, but do not rescan
        # and rehash the whole collection. run_batch verifies exact source
        # contents before each item and atomically guards output publication.
        for item in plan.items:
            if cancel_callback is not None and cancel_callback():
                raise OperationCancelled(
                    "Batch preparation cancelled before saving artifacts."
                )
            for output in item.outputs:
                if output.path.exists() != output.exists:
                    raise ValueError(
                        "Output presence changed during run startup. "
                        "Check batch and review the output choices again."
                    )
        plan = replace(plan, config=config)
        if progress_callback is not None:
            progress_callback(BatchPreflightProgress("handoff"))
    if request.expected_items is not None and plan.items != request.expected_items:
        raise RuntimeError(
            "The batch plan changed during run startup. No batch item was "
            "run; click Run batch to refresh the displayed plan, review it, "
            "then run again."
        )
    config = bind_batch_plan_source_items(config, plan)
    plan = replace(plan, config=config)
    if cancel_callback is not None and cancel_callback():
        raise OperationCancelled("Batch preparation cancelled before saving artifacts.")
    if progress_callback is not None:
        progress_callback(BatchPreflightProgress("artifacts"))
    request.workflow_path.parent.mkdir(parents=True, exist_ok=True)
    artifacts = [atomic_write_json(request.workflow_path, request.workflow)]
    if config.save_python_script:
        atomic_write_text(request.script_path, export_batch_runner_to_python())
        artifacts.append(request.script_path)
    save_batch_config(request.config_path, config)
    return PreparedCollectionBatchRun(
        job_id=request.job_id,
        origin_session_id=request.origin_session_id,
        workflow=request.workflow,
        config=config,
        workflow_path=request.workflow_path,
        config_path=request.config_path,
        plan=plan,
        artifact_paths=tuple(artifacts),
        performance_history_path=request.performance_history_path,
    )


@dataclass(frozen=True, slots=True)
class CollectionBatchProgress:
    """Progress tagged with the job that emitted it."""

    job_id: int
    index: int
    total: int
    batch_id: str
    status: str


@dataclass(frozen=True, slots=True)
class CollectionBatchOperationProgress:
    """Nested execution progress tagged with the worker job that emitted it."""

    job_id: int
    progress: BatchExecutionProgress


@dataclass(frozen=True, slots=True)
class CollectionBatchPreparationProgress:
    job_id: int
    progress: BatchPreflightProgress


@dataclass(frozen=True, slots=True)
class CollectionBatchWorkerOutcome:
    """Terminal result tagged with its originating tab and job."""

    job_id: int
    origin_session_id: str
    result: BatchRunResult | None = None
    error: str = ""
    cancelled_before_start: bool = False


class _CollectionBatchWorkerSignals(QObject):
    progress = Signal(object)
    operation_progress = Signal(object)
    preparation_progress = Signal(object)
    finished = Signal(object)


class CollectionBatchWorker(QRunnable):
    """Run only the headless batch engine away from the Qt GUI thread."""

    def __init__(
        self,
        prepared: PreparedCollectionBatchRun | CollectionBatchRunRequest,
    ) -> None:
        super().__init__()
        self.prepared = prepared
        self.signals = _CollectionBatchWorkerSignals()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Request cooperative cancellation at the next safe checkpoint."""

        self._cancel_event.set()

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel_event.is_set()

    def run(self) -> None:
        prepared = self.prepared
        progress_gate = _PreparationProgressGate()

        def emit_progress(
            index: int,
            total: int,
            batch_id: str,
            status: str,
        ) -> None:
            self.signals.progress.emit(
                CollectionBatchProgress(
                    job_id=prepared.job_id,
                    index=int(index),
                    total=int(total),
                    batch_id=str(batch_id),
                    status=str(status),
                )
            )

        def emit_operation_progress(progress: BatchExecutionProgress) -> None:
            self.signals.operation_progress.emit(
                CollectionBatchOperationProgress(
                    job_id=prepared.job_id,
                    progress=progress,
                )
            )

        def emit_preparation_progress(progress):
            if not self._cancel_event.is_set() and progress_gate.accepts(progress):
                self.signals.preparation_progress.emit(
                    CollectionBatchPreparationProgress(prepared.job_id, progress)
                )

        try:
            if isinstance(prepared, CollectionBatchRunRequest):
                prepared = prepare_collection_batch_run(
                    prepared,
                    progress_callback=emit_preparation_progress,
                    cancel_callback=self._cancel_event.is_set,
                )
            result = run_batch(
                prepared.workflow,
                prepared.config,
                workflow_path=prepared.workflow_path,
                config_path=prepared.config_path,
                plan=prepared.plan,
                cancel_event=self._cancel_event,
                progress_callback=emit_progress,
                execution_progress_callback=emit_operation_progress,
                preparation_progress_callback=emit_preparation_progress,
                performance_history_path=prepared.performance_history_path,
            )
            outcome = CollectionBatchWorkerOutcome(
                job_id=prepared.job_id,
                origin_session_id=prepared.origin_session_id,
                result=replace(
                    result,
                    artifact_paths=prepared.artifact_paths,
                ),
            )
        except OperationCancelled:
            outcome = CollectionBatchWorkerOutcome(
                job_id=prepared.job_id,
                origin_session_id=prepared.origin_session_id,
                cancelled_before_start=True,
            )
        except Exception as exc:
            outcome = CollectionBatchWorkerOutcome(
                job_id=prepared.job_id,
                origin_session_id=prepared.origin_session_id,
                error=str(exc),
            )
        self.signals.finished.emit(outcome)


__all__ = [
    "BatchWorkspacePreviewProgress",
    "BatchWorkspacePreviewWorker",
    "BatchWorkspacePreviewWorkerOutcome",
    "BatchWorkspacePreviewWorkerSpec",
    "CollectionBatchProgress",
    "CollectionBatchOperationProgress",
    "CollectionBatchPreparationProgress",
    "CollectionBatchRunRequest",
    "CollectionBatchWorker",
    "CollectionBatchWorkerOutcome",
    "PreparedCollectionBatchRun",
    "prepare_collection_batch_run",
]
