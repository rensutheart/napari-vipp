"""Qt-free collection batch execution wrapped in a small Qt runnable."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.batch import (
    BatchConfig,
    BatchPlan,
    BatchRunResult,
    run_batch,
)


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


@dataclass(frozen=True, slots=True)
class CollectionBatchProgress:
    """Progress tagged with the job that emitted it."""

    job_id: int
    index: int
    total: int
    batch_id: str
    status: str


@dataclass(frozen=True, slots=True)
class CollectionBatchWorkerOutcome:
    """Terminal result tagged with its originating tab and job."""

    job_id: int
    origin_session_id: str
    result: BatchRunResult | None = None
    error: str = ""


class _CollectionBatchWorkerSignals(QObject):
    progress = Signal(object)
    finished = Signal(object)


class CollectionBatchWorker(QRunnable):
    """Run only the headless batch engine away from the Qt GUI thread."""

    def __init__(self, prepared: PreparedCollectionBatchRun) -> None:
        super().__init__()
        self.prepared = prepared
        self.signals = _CollectionBatchWorkerSignals()

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

        try:
            result = run_batch(
                prepared.workflow,
                prepared.config,
                workflow_path=prepared.workflow_path,
                config_path=prepared.config_path,
                plan=prepared.plan,
                progress_callback=emit_progress,
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
    "CollectionBatchProgress",
    "CollectionBatchWorker",
    "CollectionBatchWorkerOutcome",
    "PreparedCollectionBatchRun",
]
