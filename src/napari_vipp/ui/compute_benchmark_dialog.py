"""Qt worker and review dialog for exact selected-node benchmarks."""

from __future__ import annotations

import statistics
import threading
from collections.abc import Callable
from copy import deepcopy
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

from napari_vipp.core.benchmark_store_quarantine import (
    ensure_benchmark_store_ready,
    quarantine_benchmark_store,
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
        self.pipeline = _snapshot_prototype_pipeline(pipeline)
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
        coordinator: ApplicationNodeBenchmarkCoordinator | None = None
        result: ApplicationNodeBenchmarkResult | None = None
        benchmark_store = None
        original_store_put = None
        measured_record_rollbacks: dict[
            str,
            tuple[object, object | None, object],
        ] = {}
        try:
            store_path = ensure_benchmark_store_ready(self.store_path)
            registry = self.registry_factory()
            coordinator = self.coordinator_factory(registry, store_path)
            benchmark_store = getattr(coordinator, "store", None)
            if (
                benchmark_store is not None
                and callable(getattr(benchmark_store, "get", None))
                and callable(getattr(benchmark_store, "put", None))
            ):
                original_store_put = benchmark_store.put
                original_store_get = benchmark_store.get

                def track_benchmark_record(record) -> None:
                    digest = str(getattr(record.key, "digest", record.key))
                    first_write = digest not in measured_record_rollbacks
                    previous = (
                        original_store_get(record.key) if first_write else None
                    )
                    original_store_put(record)
                    if first_write:
                        measured_record_rollbacks[digest] = (
                            record.key,
                            previous,
                            record,
                        )

                benchmark_store.put = track_benchmark_record
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
                except Exception as exc:
                    rollback_failures: list[str] = []
                    if benchmark_store is not None and measured_record_rollbacks:
                        for key, previous, written in reversed(
                            tuple(measured_record_rollbacks.values())
                        ):
                            try:
                                if benchmark_store.get(key) != written:
                                    continue
                                if previous is None:
                                    benchmark_store.discard(key)
                                elif original_store_put is not None:
                                    original_store_put(previous)
                            except Exception as rollback_exc:
                                rollback_failures.append(
                                    f"{type(rollback_exc).__name__}: {rollback_exc}"
                                )
                    elif coordinator is not None and result is not None:
                        try:
                            coordinator.store.discard(result.record.key)
                        except Exception as rollback_exc:
                            rollback_failures.append(
                                f"{type(rollback_exc).__name__}: {rollback_exc}"
                            )
                    detail = f"{type(exc).__name__}: {exc}"
                    if rollback_failures:
                        detail += (
                            "; invalid benchmark evidence could not be fully "
                            "rolled back: " + "; ".join(rollback_failures)
                        )
                        quarantine = quarantine_benchmark_store(
                            self.store_path,
                            reason=(
                                "Node benchmark runtime cleanup and record "
                                "rollback both failed."
                            ),
                        )
                        if quarantine.safe_for_restart:
                            detail += (
                                "; the complete local benchmark store was moved "
                                "away from its active path and will not be reused"
                            )
                            if quarantine.quarantined_path is not None:
                                detail += f" ({quarantine.quarantined_path})"
                        elif quarantine.marker_present:
                            detail += (
                                "; a durable quarantine marker prevents the store "
                                f"from reopening ({quarantine.marker_path}): "
                                f"{quarantine.error}"
                            )
                        else:
                            detail += (
                                "; suspect evidence may remain at "
                                f"{self.store_path}: {quarantine.error}; rename or "
                                "delete it before restarting VIPP"
                            )
                    elif result is not None:
                        detail += "; newly measured benchmark evidence was rolled back"
                    else:
                        detail += "; no newly measured evidence was retained"
                    outcome = NodeBenchmarkWorkerOutcome(
                        self.node_id,
                        error=f"Benchmark cleanup failed: {detail}",
                        reason_code="cleanup_failed",
                    )
                finally:
                    if benchmark_store is not None and original_store_put is not None:
                        benchmark_store.put = original_store_put
        self.signals.finished.emit(outcome)


def _snapshot_prototype_pipeline(pipeline):
    """Freeze graph/runtime ownership without copying large result arrays.

    A selected-node benchmark is dispatched after the dialog opens.  Retaining
    the live pipeline here would let graph edits, parameter edits, or a later
    calculation change the evidence before the runnable starts.  The snapshot
    therefore owns its graph containers, parameters, and runtime bookkeeping,
    while intentionally retaining references to the current output values.
    The benchmark coordinator performs the potentially large, read-only array
    detachment later on the worker thread.

    Non-production objects are returned unchanged so lightweight test doubles
    and embedders keep the established worker contract.
    """

    # Keep the operation catalog import behind construction of an actual
    # production worker.  Importing it can inspect optional scientific
    # packages, while this Qt module and its test doubles remain import-safe.
    from napari_vipp.core.pipeline import GraphNode, PrototypePipeline

    if not isinstance(pipeline, PrototypePipeline):
        return pipeline

    nodes = tuple(
        GraphNode(
            node.id,
            node.operation_id,
            node.title,
            node.category,
            node.input_type,
            node.output_type,
            deepcopy(node.params),
            node.max_inputs,
        )
        for node in pipeline.nodes.values()
    )
    snapshot = PrototypePipeline()
    snapshot.restore_graph(
        nodes,
        tuple(pipeline.connections),
        pipeline.output_tunnel_list(),
    )
    node_ids = set(snapshot.nodes)

    # Copy containers, not payloads.  Replacing outputs or mutating an output
    # list on the live graph can no longer affect the worker, and no image stack
    # is copied while the UI thread is dispatching the benchmark.
    snapshot.outputs = {
        node_id: pipeline.outputs.get(node_id) for node_id in snapshot.nodes
    }
    snapshot.output_states = {
        node_id: pipeline.output_states.get(node_id) for node_id in snapshot.nodes
    }
    snapshot.node_outputs = {
        node_id: list(pipeline.node_outputs.get(node_id, ()))
        for node_id in snapshot.nodes
    }
    snapshot.node_output_states = {
        node_id: list(pipeline.node_output_states.get(node_id, ()))
        for node_id in snapshot.nodes
    }
    snapshot.completed_node_ids = set(pipeline.completed_node_ids) & node_ids
    snapshot.node_compute_provenance = {
        node_id: provenance
        for node_id, provenance in pipeline.node_compute_provenance.items()
        if node_id in node_ids
    }
    snapshot.node_execution_states = {
        node_id: pipeline.node_execution_states.get(
            node_id,
            snapshot.node_execution_states[node_id],
        )
        for node_id in snapshot.nodes
    }
    snapshot.node_execution_messages = {
        node_id: pipeline.node_execution_messages.get(node_id, "")
        for node_id in snapshot.nodes
    }
    return snapshot


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
        self._shutdown = False

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
        if self._shutdown:
            raise RuntimeError("This benchmark dialog is shut down.")
        if self._running or self._worker is not None:
            raise RuntimeError("This benchmark dialog has already been started.")
        self._worker = worker
        self._running = True
        self.progress_label.setText(
            "Preparing exact current inputs. You can cancel at any time."
        )
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        try:
            thread_pool.start(worker)
        except Exception:
            for signal, slot in (
                (worker.signals.progress, self._on_progress),
                (worker.signals.finished, self._on_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
            self._worker = None
            self._running = False
            self.cancel_button.setEnabled(False)
            self.close_button.setEnabled(True)
            self.progress_label.setText("Benchmark could not be dispatched.")
            raise

    def cancel(self) -> None:
        if not self._running or self._worker is None:
            return
        self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.progress_label.setText("Cancel requested; finishing the current call…")

    def shutdown(self) -> None:
        """Make owner teardown terminal even when a runnable never starts."""

        if self._shutdown:
            return
        self._shutdown = True
        worker = self._worker
        if worker is not None:
            worker.cancel()
            for signal, slot in (
                (worker.signals.progress, self._on_progress),
                (worker.signals.finished, self._on_finished),
            ):
                try:
                    signal.disconnect(slot)
                except (RuntimeError, TypeError):
                    pass
        self._worker = None
        self._running = False
        self.apply_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.setWindowModality(Qt.NonModal)
        self.close()

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
        if self._shutdown:
            return
        self.progress_bar.setValue(progress.completed)
        self.progress_label.setText(progress.message)

    def _on_finished(self, outcome: NodeBenchmarkWorkerOutcome) -> None:
        if self._shutdown:
            return
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
        if (
            self._shutdown
            or self._outcome is None
            or self._outcome.result is None
        ):
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
