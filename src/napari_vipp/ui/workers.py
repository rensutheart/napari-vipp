"""Thin Qt runnables for headless VIPP execution services."""

from __future__ import annotations

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.execution import (
    PipelineRunRequest,
    execute_pipeline_request,
)


class PipelineRunSignals(QObject):
    node_started = Signal(object)
    node_finished = Signal(object)
    progress = Signal(object)
    finished = Signal(object)


class PipelineRunWorker(QRunnable):
    """Run a detached pipeline request and emit its typed result."""

    def __init__(self, request: PipelineRunRequest):
        super().__init__()
        self.request = request
        self.signals = PipelineRunSignals()

    def run(self) -> None:
        result = execute_pipeline_request(
            self.request,
            node_started_callback=self._emit_node_started,
            node_finished_callback=self.signals.node_finished.emit,
            progress_callback=self._emit_progress,
        )
        self.signals.finished.emit(result)

    def _emit_node_started(self, node_id: str) -> None:
        self.signals.node_started.emit((self.request.run_id, node_id))

    def _emit_progress(
        self,
        node_id: str,
        current: int,
        total: int,
        message: str,
    ) -> None:
        self.signals.progress.emit(
            (self.request.run_id, node_id, int(current), int(total), str(message))
        )


__all__ = ["PipelineRunSignals", "PipelineRunWorker"]
