from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qtpy.QtCore import Qt

from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.execution_telemetry import DeviceExecutionTelemetryConfig
from napari_vipp.ui import workers
from napari_vipp.ui.diagnostic_workers import (
    ThumbnailContrastLimitRequest,
    ThumbnailContrastLimitWorker,
    ThumbnailContrastStarted,
    ThumbnailContrastTerminal,
)


class _FakeClock:
    def __init__(self) -> None:
        self.value = 20.0

    def __call__(self) -> float:
        return self.value


def test_pipeline_worker_started_payload_keeps_worker_thread_timestamp(
    qtbot,
    monkeypatch,
) -> None:
    clock = _FakeClock()
    request = SimpleNamespace(
        run_id=73,
        device_execution_telemetry=DeviceExecutionTelemetryConfig(clock=clock),
    )
    terminal = object()

    def execute(_request, **_callbacks):
        clock.value = 25.0
        return terminal

    monkeypatch.setattr(workers, "execute_pipeline_request", execute)
    worker = workers.PipelineRunWorker(request)
    started = []
    observed_terminal = []
    finished = []
    worker.signals.started.connect(started.append, Qt.QueuedConnection)
    worker.signals.terminal.connect(observed_terminal.append, Qt.QueuedConnection)
    worker.signals.finished.connect(finished.append)

    worker.run()

    assert finished == [terminal]
    assert started == []
    assert observed_terminal == []
    qtbot.waitUntil(lambda: bool(started) and bool(observed_terminal))
    assert started[0] == workers.PipelineWorkerStarted(73, 20.0)
    assert observed_terminal[0] == workers.PipelineWorkerTerminal(73, 25.0)
    assert clock.value == 25.0


def test_pipeline_worker_emits_no_timing_when_tracing_is_disabled(
    qtbot,
    monkeypatch,
) -> None:
    terminal = object()
    request = SimpleNamespace(run_id=74, device_execution_telemetry=None)
    monkeypatch.setattr(
        workers,
        "execute_pipeline_request",
        lambda _request, **_callbacks: terminal,
    )
    worker = workers.PipelineRunWorker(request)
    started = []
    observed_terminal = []
    finished = []
    worker.signals.started.connect(started.append)
    worker.signals.terminal.connect(observed_terminal.append)
    worker.signals.finished.connect(finished.append)

    worker.run()

    assert started == []
    assert observed_terminal == []
    assert finished == [terminal]


def _thumbnail_worker(*, clock=None) -> ThumbnailContrastLimitWorker:
    request = ThumbnailContrastLimitRequest(
        ("scalar", "gaussian", 1),
        "gaussian",
        np.arange(16, dtype=np.float32).reshape(4, 4),
        None,
        "Min-max",
        "image",
    )
    return ThumbnailContrastLimitWorker(
        81,
        (request,),
        calculate_scalar=lambda *_args, **_kwargs: (0.0, 15.0),
        calculate_channel=lambda *_args, **_kwargs: (),
        started_clock=clock,
    )


def test_thumbnail_worker_emits_true_start_and_terminal_timestamps(qtbot) -> None:
    clock = _FakeClock()
    worker = _thumbnail_worker(clock=clock)
    started = []
    terminal = []
    finished = []
    worker.signals.started.connect(started.append)
    worker.signals.terminal.connect(terminal.append)
    worker.signals.finished.connect(finished.append)

    def advance_during_calculation(*_args, **_kwargs):
        clock.value = 22.5
        return (0.0, 15.0)

    worker._calculate_scalar = advance_during_calculation
    worker.run()

    assert started == [ThumbnailContrastStarted(81, 20.0)]
    assert terminal == [ThumbnailContrastTerminal(81, 22.5)]
    assert len(finished) == 1


def test_thumbnail_worker_emits_no_timing_when_tracing_is_disabled(qtbot) -> None:
    worker = _thumbnail_worker()
    started = []
    terminal = []
    finished = []
    worker.signals.started.connect(started.append)
    worker.signals.terminal.connect(terminal.append)
    worker.signals.finished.connect(finished.append)

    worker.run()

    assert started == []
    assert terminal == []
    assert len(finished) == 1


def test_thumbnail_worker_preserves_explicit_device_for_selection_and_execution(
    qtbot,
) -> None:
    observed_select = []
    observed_calculate = []

    class _StatisticsEngine:
        def select(self, request):
            observed_select.append(request)
            return SimpleNamespace(
                scanned_values=int(np.asarray(request.data).size),
                backend=SimpleNamespace(value="gpu-cupy"),
                reason="qualified",
            )

        def calculate(self, request, *, progress):
            observed_calculate.append(request)
            progress.report(1, 1, "ready")
            return SimpleNamespace(
                limits=(0.0, 15.0),
                actual_backend=SimpleNamespace(value="gpu-cupy"),
                used_fallback=False,
            )

    request = ThumbnailContrastLimitRequest(
        ("scalar", "gaussian", 1),
        "gaussian",
        np.arange(16, dtype=np.float32).reshape(4, 4),
        None,
        "Min-max",
        "image",
    )
    worker = ThumbnailContrastLimitWorker(
        82,
        (request,),
        statistics_engine=_StatisticsEngine(),
        compute_mode=ComputeMode.PREFER_GPU,
        device_id="cuda:1",
    )
    finished = []
    worker.signals.finished.connect(finished.append)

    worker.run()

    assert len(observed_select) == 2
    assert len(observed_calculate) == 1
    assert all(item.device_id == "cuda:1" for item in observed_select)
    assert observed_calculate[0].device_id == "cuda:1"
    assert len(finished) == 1
    assert not finished[0].error
