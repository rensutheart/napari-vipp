"""Thin Qt runnables for headless VIPP execution services."""

from __future__ import annotations

import math
from dataclasses import dataclass

from qtpy.QtCore import QObject, QRunnable, Signal

from napari_vipp.core.execution import (
    PipelineRunRequest,
    execute_pipeline_request,
)


@dataclass(frozen=True, slots=True)
class PipelineWorkerStarted:
    """Worker-thread start observation safe to deliver through queued signals."""

    run_id: int
    started_monotonic_seconds: float | None


@dataclass(frozen=True, slots=True)
class PipelineWorkerTerminal:
    """Worker-thread terminal time before queued UI result delivery."""

    run_id: int
    terminal_monotonic_seconds: float


class PipelineRunSignals(QObject):
    started = Signal(object)
    terminal = Signal(object)
    node_started = Signal(object)
    node_finished = Signal(object)
    presentation_shadow_finished = Signal(object)
    progress = Signal(object)
    finished = Signal(object)


class PipelineRunWorker(QRunnable):
    """Run a detached pipeline request and emit its typed result."""

    def __init__(self, request: PipelineRunRequest):
        super().__init__()
        self.request = request
        self.signals = PipelineRunSignals()

    def run(self) -> None:
        started = _worker_started_timestamp(self.request)
        if started is not None:
            self.signals.started.emit(
                PipelineWorkerStarted(
                    self.request.run_id,
                    started,
                )
            )
        result = execute_pipeline_request(
            self.request,
            node_started_callback=self._emit_node_started,
            node_finished_callback=self.signals.node_finished.emit,
            presentation_shadow_callback=(
                self.signals.presentation_shadow_finished.emit
            ),
            progress_callback=self._emit_progress,
        )
        terminal = _worker_started_timestamp(self.request)
        if terminal is not None:
            self.signals.terminal.emit(
                PipelineWorkerTerminal(self.request.run_id, terminal)
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


def _worker_started_timestamp(request: PipelineRunRequest) -> float | None:
    telemetry = request.device_execution_telemetry
    if telemetry is None:
        return None
    clock = telemetry.clock
    try:
        sampled = clock()
    except Exception:
        return None
    if (
        isinstance(sampled, bool)
        or not isinstance(sampled, (int, float))
        or not math.isfinite(float(sampled))
    ):
        return None
    return float(sampled)


__all__ = [
    "PipelineRunSignals",
    "PipelineRunWorker",
    "PipelineWorkerStarted",
    "PipelineWorkerTerminal",
]
