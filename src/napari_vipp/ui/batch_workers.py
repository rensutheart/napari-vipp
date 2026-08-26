"""Qt-free collection batch execution wrapped in a small Qt runnable."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.batch import (
    BatchConfig,
    BatchExecutionProgress,
    BatchPlan,
    BatchRunResult,
    run_batch,
)
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


class _BatchWorkspacePreviewWorkerSignals(QObject):
    finished = Signal(object)


class BatchWorkspacePreviewWorker(QRunnable):
    """Verify an attached batch workspace without reading scientific pixels."""

    def __init__(self, spec: BatchWorkspacePreviewWorkerSpec) -> None:
        super().__init__()
        self.spec = spec
        self.signals = _BatchWorkspacePreviewWorkerSignals()
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        """Suppress publication of a queued or superseded verification."""

        self._cancel_event.set()

    @property
    def cancellation_requested(self) -> bool:
        return self._cancel_event.is_set()

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
            result = execute_prepared_collection_batch_preview(spec.prepared)
        except Exception as exc:
            outcome = BatchWorkspacePreviewWorkerOutcome(
                spec.request_id,
                spec.origin_session_id,
                error=exc,
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
    """Immutable inputs prepared on the GUI thread before a batch starts."""

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
class CollectionBatchWorkerOutcome:
    """Terminal result tagged with its originating tab and job."""

    job_id: int
    origin_session_id: str
    result: BatchRunResult | None = None
    error: str = ""


class _CollectionBatchWorkerSignals(QObject):
    progress = Signal(object)
    operation_progress = Signal(object)
    finished = Signal(object)


class CollectionBatchWorker(QRunnable):
    """Run only the headless batch engine away from the Qt GUI thread."""

    def __init__(self, prepared: PreparedCollectionBatchRun) -> None:
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

        try:
            result = run_batch(
                prepared.workflow,
                prepared.config,
                workflow_path=prepared.workflow_path,
                config_path=prepared.config_path,
                plan=prepared.plan,
                cancel_event=self._cancel_event,
                progress_callback=emit_progress,
                execution_progress_callback=emit_operation_progress,
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
        except Exception as exc:
            outcome = CollectionBatchWorkerOutcome(
                job_id=prepared.job_id,
                origin_session_id=prepared.origin_session_id,
                error=str(exc),
            )
        self.signals.finished.emit(outcome)


__all__ = [
    "BatchWorkspacePreviewWorker",
    "BatchWorkspacePreviewWorkerOutcome",
    "BatchWorkspacePreviewWorkerSpec",
    "CollectionBatchProgress",
    "CollectionBatchOperationProgress",
    "CollectionBatchWorker",
    "CollectionBatchWorkerOutcome",
    "PreparedCollectionBatchRun",
]
