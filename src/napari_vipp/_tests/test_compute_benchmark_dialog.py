from __future__ import annotations

from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt

from napari_vipp.core.compute import (
    BenchmarkCandidateResult,
    NodeComputePreference,
    NodePreferenceKind,
)
from napari_vipp.core.compute_benchmark import (
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
)
from napari_vipp.core.compute_benchmark_coordinator import (
    NodeBenchmarkProgress,
    NodeBenchmarkUnavailable,
)
from napari_vipp.ui.compute_benchmark_dialog import (
    NodeBenchmarkDialog,
    NodeBenchmarkWorker,
    NodeBenchmarkWorkerOutcome,
)


class _ClosingRegistry:
    def __init__(self) -> None:
        self.close_calls = 0

    def close(self) -> None:
        self.close_calls += 1


class _CapturingThreadPool:
    def __init__(self) -> None:
        self.workers = []

    def start(self, worker) -> None:
        self.workers.append(worker)


def _renderable_result():
    candidates = (
        BenchmarkCandidateResult(
            "cpu-scipy",
            parity_passed=True,
            cold_seconds=0.08,
            warm_seconds=(0.04, 0.06),
            peak_memory_bytes=1024,
        ),
        BenchmarkCandidateResult(
            "cuda-cupy",
            parity_passed=True,
            cold_seconds=0.02,
            warm_seconds=(0.01, 0.02),
            peak_memory_bytes=2 * 1024**2,
        ),
        BenchmarkCandidateResult(
            "cuda-cucim",
            parity_passed=False,
            cold_seconds=None,
            warm_seconds=(),
            error="cuCIM rejected this exact dtype",
        ),
    )
    reference = SimpleNamespace(implementation_id="cpu-scipy")
    request = SimpleNamespace(reference=reference)
    registered = SimpleNamespace(request=request)
    return SimpleNamespace(
        plan=SimpleNamespace(registered=registered),
        record=SimpleNamespace(
            candidates=candidates,
            accepted_implementation_id="cuda-cupy",
        ),
        winner_preference=NodeComputePreference(
            NodePreferenceKind.LIBRARY,
            "cupy",
        ),
    )


def _worker(
    tmp_path,
    *,
    registry_factory,
    coordinator_factory,
) -> NodeBenchmarkWorker:
    return NodeBenchmarkWorker(
        pipeline=object(),
        node_id="gaussian-1",
        store_path=tmp_path / "benchmarks.json",
        allow_experimental=True,
        time_budget_seconds=8.5,
        registry_factory=registry_factory,
        coordinator_factory=coordinator_factory,
    )


def test_worker_emits_progress_and_success_then_closes_registry(tmp_path):
    registry = _ClosingRegistry()
    result = object()
    progress = NodeBenchmarkProgress(
        "benchmarking",
        3,
        4,
        "Timing eligible implementations.",
    )
    calls = {}

    class Coordinator:
        def benchmark(self, pipeline, node_id, **kwargs):
            calls["benchmark"] = (pipeline, node_id, kwargs)
            kwargs["progress"](progress)
            return result

    def coordinator_factory(received_registry, store_path):
        calls["coordinator"] = (received_registry, store_path)
        return Coordinator()

    worker = _worker(
        tmp_path,
        registry_factory=lambda: registry,
        coordinator_factory=coordinator_factory,
    )
    observed_progress = []
    outcomes = []
    worker.signals.progress.connect(observed_progress.append)
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert calls["coordinator"] == (registry, tmp_path / "benchmarks.json")
    pipeline, node_id, kwargs = calls["benchmark"]
    assert pipeline is worker.pipeline
    assert node_id == "gaussian-1"
    assert kwargs["allow_experimental"] is True
    assert kwargs["time_budget_seconds"] == 8.5
    assert kwargs["cancelled"]() is False
    assert observed_progress == [progress]
    assert outcomes == [NodeBenchmarkWorkerOutcome("gaussian-1", result=result)]
    assert registry.close_calls == 1


def test_worker_cleanup_failure_withdraws_saved_result(tmp_path):
    discarded = []
    result = SimpleNamespace(record=SimpleNamespace(key="exact-key"))

    class FailingRegistry:
        @staticmethod
        def close():
            raise RuntimeError("device cleanup failed")

    class Store:
        @staticmethod
        def discard(key):
            discarded.append(key)

    class Coordinator:
        store = Store()

        @staticmethod
        def benchmark(*_args, **_kwargs):
            return result

    worker = _worker(
        tmp_path,
        registry_factory=FailingRegistry,
        coordinator_factory=lambda _registry, _path: Coordinator(),
    )
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert discarded == ["exact-key"]
    assert outcomes[0].result is None
    assert outcomes[0].reason_code == "benchmark_failed"
    assert "cleanup failed" in outcomes[0].error


@pytest.mark.parametrize(
    ("failure", "reason_code", "cancelled", "expected_error"),
    [
        pytest.param(
            BenchmarkCancelled("stopped by user"),
            "cancelled",
            True,
            "stopped by user",
            id="cancelled",
        ),
        pytest.param(
            NodeBenchmarkUnavailable("no eligible GPU candidate"),
            "unavailable",
            False,
            "no eligible GPU candidate",
            id="unavailable",
        ),
        pytest.param(
            BenchmarkBudgetExceeded("timing budget expired"),
            "budget_exceeded",
            False,
            "timing budget expired",
            id="budget-exceeded",
        ),
        pytest.param(
            RuntimeError("driver failed"),
            "benchmark_failed",
            False,
            "RuntimeError: driver failed",
            id="unexpected-error",
        ),
    ],
)
def test_worker_maps_terminal_failures_and_closes_registry(
    tmp_path,
    failure,
    reason_code,
    cancelled,
    expected_error,
):
    registry = _ClosingRegistry()

    class Coordinator:
        def benchmark(self, _pipeline, _node_id, **kwargs):
            assert kwargs["cancelled"]() is cancelled
            raise failure

    worker = _worker(
        tmp_path,
        registry_factory=lambda: registry,
        coordinator_factory=lambda _registry, _path: Coordinator(),
    )
    if cancelled:
        worker.cancel()
    outcomes = []
    worker.signals.finished.connect(outcomes.append)

    worker.run()

    assert outcomes == [
        NodeBenchmarkWorkerOutcome(
            "gaussian-1",
            error=expected_error,
            reason_code=reason_code,
            cancelled=cancelled,
        )
    ]
    assert registry.close_calls == 1


def test_dialog_renders_progress_and_completed_evidence(qtbot):
    dialog = NodeBenchmarkDialog("Gaussian blur")
    qtbot.addWidget(dialog)
    result = _renderable_result()
    outcome = NodeBenchmarkWorkerOutcome("gaussian-1", result=result)
    finished = []
    dialog.benchmark_finished.connect(finished.append)

    dialog._on_progress(
        NodeBenchmarkProgress(
            "benchmarking",
            3,
            4,
            "Timing eligible implementations.",
        )
    )

    assert dialog.progress_bar.value() == 3
    assert dialog.progress_label.text() == "Timing eligible implementations."

    dialog._on_finished(outcome)

    assert dialog.outcome is outcome
    assert not dialog.running
    assert dialog.progress_bar.value() == 4
    assert dialog.progress_label.text() == "Benchmark complete."
    assert not dialog.result_table.isHidden()
    assert dialog.result_table.rowCount() == 3
    assert dialog.result_table.item(0, 0).text() == "cpu-scipy"
    assert dialog.result_table.item(0, 1).text() == "50.0 ms"
    assert dialog.result_table.item(0, 2).text() == "1.00×"
    assert dialog.result_table.item(0, 3).text() == "Passed"
    assert dialog.result_table.item(0, 4).text() == "1.0 KiB"
    assert dialog.result_table.item(1, 0).text() == "✓ cuda-cupy"
    assert dialog.result_table.item(1, 1).text() == "15.0 ms"
    assert dialog.result_table.item(1, 2).text() == "3.33×"
    assert dialog.result_table.item(1, 4).text() == "2.0 MiB"
    assert dialog.result_table.item(2, 1).text() == "—"
    assert dialog.result_table.item(2, 3).text() == "Failed"
    assert (
        dialog.result_table.item(2, 0).toolTip()
        == "cuCIM rejected this exact dtype"
    )
    assert "Fastest qualified implementation: cuda-cupy" in dialog.result_label.text()
    assert "library:cupy" in dialog.result_label.text()
    assert dialog.apply_button.isEnabled()
    assert dialog.close_button.isEnabled()
    assert dialog.cancel_button.isHidden()
    assert finished == [outcome]


def test_success_requires_an_explicit_apply_action(qtbot):
    dialog = NodeBenchmarkDialog("Gaussian blur")
    qtbot.addWidget(dialog)
    result = _renderable_result()
    applied = []
    dialog.apply_requested.connect(applied.append)

    dialog._on_finished(NodeBenchmarkWorkerOutcome("gaussian-1", result=result))

    assert applied == []

    qtbot.mouseClick(dialog.apply_button, Qt.LeftButton)

    assert applied == [result]

    qtbot.mouseClick(dialog.close_button, Qt.LeftButton)

    assert applied == [result]


@pytest.mark.parametrize("close_method", ["reject", "window-close"])
def test_closing_while_running_requests_cancellation(
    qtbot,
    tmp_path,
    close_method,
):
    worker = _worker(
        tmp_path,
        registry_factory=_ClosingRegistry,
        coordinator_factory=lambda _registry, _path: object(),
    )
    pool = _CapturingThreadPool()
    dialog = NodeBenchmarkDialog("Gaussian blur")
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.start(worker, pool)

    if close_method == "reject":
        dialog.reject()
    else:
        assert dialog.close() is False

    assert dialog.running
    assert dialog.isVisible()
    assert worker.cancel_event.is_set()
    assert not dialog.cancel_button.isEnabled()
    assert dialog.progress_label.text() == (
        "Cancel requested; finishing the current call…"
    )
    assert pool.workers == [worker]

    worker.signals.finished.emit(
        NodeBenchmarkWorkerOutcome(
            "gaussian-1",
            error="stopped by user",
            reason_code="cancelled",
            cancelled=True,
        )
    )

    assert not dialog.running
    assert dialog.progress_label.text() == (
        "Benchmark cancelled. No preference changed."
    )
    assert not dialog.apply_button.isEnabled()
    assert dialog.close_button.isEnabled()


@pytest.mark.parametrize(
    "outcome",
    [
        NodeBenchmarkWorkerOutcome(
            "gaussian-1",
            error="stopped by user",
            reason_code="cancelled",
            cancelled=True,
        ),
        NodeBenchmarkWorkerOutcome(
            "gaussian-1",
            error="no eligible GPU candidate",
            reason_code="unavailable",
        ),
    ],
)
def test_unsuccessful_outcomes_never_offer_or_emit_apply(qtbot, outcome):
    dialog = NodeBenchmarkDialog("Gaussian blur")
    qtbot.addWidget(dialog)
    applied = []
    dialog.apply_requested.connect(applied.append)

    dialog._on_finished(outcome)
    dialog._apply_result()

    assert applied == []
    assert not dialog.apply_button.isEnabled()
    assert dialog.close_button.isEnabled()
    if outcome.cancelled:
        assert dialog.result_label.text() == ""
    else:
        assert dialog.result_label.text() == outcome.error
