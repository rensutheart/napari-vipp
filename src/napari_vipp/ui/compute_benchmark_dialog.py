"""Qt worker and review dialog for exact selected-node benchmarks."""

from __future__ import annotations

import statistics
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.compute_benchmark import (
    BenchmarkBudgetExceeded,
    BenchmarkCancelled,
)
from napari_vipp.core.compute_benchmark_coordinator import (
    ApplicationNodeBenchmarkCoordinator,
    ApplicationNodeBenchmarkResult,
    NodeBenchmarkProgress,
    NodeBenchmarkUnavailable,
)
from napari_vipp.core.compute_registry import ComputeRegistry


@dataclass(frozen=True, slots=True)
class NodeBenchmarkWorkerOutcome:
    """Terminal worker result; expected failures remain ordinary data."""

    node_id: str
    result: ApplicationNodeBenchmarkResult | None = None
    error: str = ""
    reason_code: str = ""
    cancelled: bool = False


class _NodeBenchmarkWorkerSignals(QObject):
    progress = Signal(object)
    finished = Signal(object)


class NodeBenchmarkWorker(QRunnable):
    """Run production benchmark preparation and timing off the Qt thread."""

    def __init__(
        self,
        pipeline,
        node_id: str,
        store_path: str | Path,
        *,
        allow_experimental: bool,
        time_budget_seconds: float = 120.0,
        registry_factory: Callable[[], ComputeRegistry] = ComputeRegistry,
        coordinator_factory: Callable[
            [ComputeRegistry, str | Path], ApplicationNodeBenchmarkCoordinator
        ] = ApplicationNodeBenchmarkCoordinator,
    ) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.node_id = str(node_id).strip()
        self.store_path = Path(store_path)
        self.allow_experimental = bool(allow_experimental)
        self.time_budget_seconds = float(time_budget_seconds)
        self.registry_factory = registry_factory
        self.coordinator_factory = coordinator_factory
        self.cancel_event = threading.Event()
        self.signals = _NodeBenchmarkWorkerSignals()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        registry: ComputeRegistry | None = None
        try:
            registry = self.registry_factory()
            coordinator = self.coordinator_factory(registry, self.store_path)
            result = coordinator.benchmark(
                self.pipeline,
                self.node_id,
                allow_experimental=self.allow_experimental,
                time_budget_seconds=self.time_budget_seconds,
                cancelled=self.cancel_event.is_set,
                progress=self.signals.progress.emit,
            )
            outcome = NodeBenchmarkWorkerOutcome(self.node_id, result=result)
        except BenchmarkCancelled as exc:
            outcome = NodeBenchmarkWorkerOutcome(
                self.node_id,
                error=str(exc),
                reason_code="cancelled",
                cancelled=True,
            )
        except NodeBenchmarkUnavailable as exc:
            outcome = NodeBenchmarkWorkerOutcome(
                self.node_id,
                error=str(exc),
                reason_code="unavailable",
            )
        except BenchmarkBudgetExceeded as exc:
            outcome = NodeBenchmarkWorkerOutcome(
                self.node_id,
                error=str(exc),
                reason_code="budget_exceeded",
            )
        except Exception as exc:
            outcome = NodeBenchmarkWorkerOutcome(
                self.node_id,
                error=f"{type(exc).__name__}: {exc}",
                reason_code="benchmark_failed",
            )
        finally:
            if registry is not None:
                try:
                    registry.close()
                except Exception:
                    pass
        self.signals.finished.emit(outcome)


class NodeBenchmarkDialog(QDialog):
    """Show benchmark progress, evidence, and an explicit apply action."""

    apply_requested = Signal(object)
    benchmark_finished = Signal(object)

    def __init__(self, node_title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Benchmark node · {node_title}")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumSize(700, 380)
        self._worker: NodeBenchmarkWorker | None = None
        self._outcome: NodeBenchmarkWorkerOutcome | None = None
        self._running = False

        self.summary_label = QLabel(
            "VIPP will compare the current node input and parameters on CPU and "
            "every scientifically eligible GPU implementation."
        )
        self.summary_label.setWordWrap(True)
        self.progress_label = QLabel("Waiting to start.")
        self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%v/%m")
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(
            ("Implementation", "Warm median", "CPU speedup", "Parity", "Peak memory")
        )
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.result_table.setVisible(False)
        self.result_table.horizontalHeader().setStretchLastSection(True)
        self.apply_button = QPushButton("Use fastest for this node")
        self.apply_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel benchmark")
        self.close_button = QPushButton("Close")
        self.close_button.setEnabled(False)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.apply_button)
        buttons.addWidget(self.cancel_button)
        buttons.addWidget(self.close_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_table, 1)
        layout.addLayout(buttons)

        self.apply_button.clicked.connect(self._apply_result)
        self.cancel_button.clicked.connect(self.cancel)
        self.close_button.clicked.connect(self.accept)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def outcome(self) -> NodeBenchmarkWorkerOutcome | None:
        return self._outcome

    def start(self, worker: NodeBenchmarkWorker, thread_pool: QThreadPool) -> None:
        if self._running or self._worker is not None:
            raise RuntimeError("This benchmark dialog has already been started.")
        self._worker = worker
        self._running = True
        self.progress_label.setText(
            "Preparing exact current inputs. You can cancel at any time."
        )
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        thread_pool.start(worker)

    def cancel(self) -> None:
        if not self._running or self._worker is None:
            return
        self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.progress_label.setText("Cancel requested; finishing the current call…")

    def reject(self) -> None:
        if self._running:
            self.cancel()
            return
        super().reject()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._running:
            self.cancel()
            event.ignore()
            return
        super().closeEvent(event)

    def _on_progress(self, progress: NodeBenchmarkProgress) -> None:
        self.progress_bar.setValue(progress.completed)
        self.progress_label.setText(progress.message)

    def _on_finished(self, outcome: NodeBenchmarkWorkerOutcome) -> None:
        self._running = False
        self._outcome = outcome
        self.cancel_button.setVisible(False)
        self.close_button.setEnabled(True)
        if outcome.result is not None:
            self.progress_bar.setValue(4)
            self.progress_label.setText("Benchmark complete.")
            self._render_result(outcome.result)
            self.apply_button.setEnabled(True)
        elif outcome.cancelled:
            self.progress_label.setText("Benchmark cancelled. No preference changed.")
            self.result_label.setText("")
        else:
            self.progress_label.setText("Benchmark could not be completed.")
            self.result_label.setText(outcome.error)
            self.result_label.setStyleSheet("color: #fca5a5;")
        self.benchmark_finished.emit(outcome)

    def _render_result(self, result: ApplicationNodeBenchmarkResult) -> None:
        record = result.record
        winner = record.accepted_implementation_id
        reference_id = result.plan.registered.request.reference.implementation_id
        reference = next(
            item for item in record.candidates if item.implementation_id == reference_id
        )
        cpu_median = (
            statistics.median(reference.warm_seconds)
            if reference.warm_seconds
            else None
        )
        self.result_table.setRowCount(len(record.candidates))
        for row, candidate in enumerate(record.candidates):
            warm_median = (
                statistics.median(candidate.warm_seconds)
                if candidate.warm_seconds
                else None
            )
            name = candidate.implementation_id
            if name == winner:
                name = f"✓ {name}"
            speedup = (
                cpu_median / warm_median
                if cpu_median is not None and warm_median not in {None, 0}
                else None
            )
            values = (
                name,
                _format_seconds(warm_median),
                "—" if speedup is None else f"{speedup:.2f}×",
                (
                    "Passed"
                    if candidate.parity_passed and not candidate.error
                    else "Failed"
                ),
                _format_bytes(candidate.peak_memory_bytes),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if candidate.error:
                    item.setToolTip(candidate.error)
                self.result_table.setItem(row, column, item)
        self.result_table.resizeColumnsToContents()
        self.result_table.setVisible(True)
        preference = result.winner_preference
        preference_text = preference.kind.value
        if preference.value:
            preference_text += f":{preference.value}"
        self.result_label.setStyleSheet("color: #86efac; font-weight: 650;")
        self.result_label.setText(
            f"Fastest qualified implementation: {winner}. Applying this result "
            f"will set the portable preference {preference_text}."
        )

    def _apply_result(self) -> None:
        if self._outcome is None or self._outcome.result is None:
            return
        self.apply_requested.emit(self._outcome.result)


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "—"
    if value < 0.001:
        return f"{value * 1_000_000:.1f} µs"
    if value < 1.0:
        return f"{value * 1_000:.1f} ms"
    return f"{value:.3f} s"


def _format_bytes(value: int) -> str:
    amount = float(max(value, 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024.0
    return "0 B"


__all__ = [
    "NodeBenchmarkDialog",
    "NodeBenchmarkWorker",
    "NodeBenchmarkWorkerOutcome",
]
