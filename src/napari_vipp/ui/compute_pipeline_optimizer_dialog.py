"""Review-first Qt surface for whole-pipeline compute optimization."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

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

from napari_vipp.core.compute_pipeline_optimizer import (
    PipelineOptimizationCancelled,
    PipelineOptimizationDeadlineExceeded,
    PipelineOptimizationEvidenceIncomplete,
    PipelineOptimizationNotBeneficial,
    PipelineOptimizationProposal,
    PipelineValidationWinner,
)


@dataclass(frozen=True, slots=True)
class PipelineOptimizerProgress:
    """One application-level progress update suitable for queued Qt delivery."""

    completed: int
    total: int
    message: str

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.completed, self.total)
        ):
            raise ValueError("optimizer progress values must be non-negative integers")
        if self.total < 1 or self.completed > self.total:
            raise ValueError("optimizer progress must fit inside its declared total")
        message = str(self.message).strip()
        if not message:
            raise ValueError("optimizer progress message must not be empty")
        object.__setattr__(self, "message", message)


@dataclass(frozen=True, slots=True)
class PipelineOptimizerWorkerOutcome:
    """Terminal optimizer result; expected refusals remain ordinary data."""

    result: object | None = None
    error: str = ""
    reason_code: str = ""
    cancelled: bool = False


class _PipelineOptimizerWorkerSignals(QObject):
    progress = Signal(object)
    finished = Signal(object)


class PipelineOptimizerWorker(QRunnable):
    """Run a caller-supplied, Qt-free optimizer transaction off the GUI thread."""

    def __init__(
        self,
        optimize: Callable[
            [Callable[[], bool], Callable[[PipelineOptimizerProgress], None]],
            object,
        ],
    ) -> None:
        super().__init__()
        if not callable(optimize):
            raise TypeError("optimize must be callable")
        self.optimize = optimize
        self.cancel_event = threading.Event()
        self.signals = _PipelineOptimizerWorkerSignals()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        try:
            result = self.optimize(
                self.cancel_event.is_set,
                self.signals.progress.emit,
            )
            outcome = PipelineOptimizerWorkerOutcome(result=result)
        except PipelineOptimizationCancelled as exc:
            outcome = PipelineOptimizerWorkerOutcome(
                error=str(exc),
                reason_code="cancelled",
                cancelled=True,
            )
        except PipelineOptimizationEvidenceIncomplete as exc:
            outcome = PipelineOptimizerWorkerOutcome(
                error=str(exc),
                reason_code="evidence_incomplete",
            )
        except PipelineOptimizationNotBeneficial as exc:
            outcome = PipelineOptimizerWorkerOutcome(
                error=str(exc),
                reason_code="not_beneficial",
            )
        except PipelineOptimizationDeadlineExceeded as exc:
            outcome = PipelineOptimizerWorkerOutcome(
                error=str(exc),
                reason_code="deadline_exceeded",
            )
        except Exception as exc:
            outcome = PipelineOptimizerWorkerOutcome(
                error=f"{type(exc).__name__}: {exc}",
                reason_code="optimizer_failed",
            )
        self.signals.finished.emit(outcome)


class PipelineOptimizerDialog(QDialog):
    """Collect intent, show evidence, and request one explicit atomic apply."""

    analyze_requested = Signal()
    apply_requested = Signal(object)
    optimizer_finished = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        locked_node_count: int = 0,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find fastest pipeline")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumSize(860, 480)
        self._worker: PipelineOptimizerWorker | None = None
        self._outcome: PipelineOptimizerWorkerOutcome | None = None
        self._running = False
        self._shutdown = False

        self.summary_label = QLabel(
            "<b>Find the fastest scientifically valid pipeline.</b>"
            "<ul>"
            "<li><b>Search:</b> Compare every scientifically eligible CPU and "
            "GPU implementation for each unlocked node.</li>"
            "<li><b>Locks:</b> The current backend is only the starting choice; "
            "an explicit optimizer lock preserves it.</li>"
            "<li><b>Control:</b> Nothing changes until you review and apply the "
            "result.</li>"
            "<li><b>Timing:</b> CPU uses paired warm medians; GPU uses "
            "resident-compute medians, with transfers modeled across the whole "
            "pipeline.</li>"
            "</ul>"
        )
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        self.progress_label = QLabel(
            f"Ready. {locked_node_count} explicitly locked node(s) will be preserved."
        )
        self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_table = QTableWidget(0, 7)
        self.result_table.setHorizontalHeaderLabels(
            (
                "Node",
                "Current",
                "Tested",
                "Selected winner",
                "Saved preference",
                "Modeled timings",
                "Status",
            )
        )
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.result_table.setVisible(False)
        self.result_table.horizontalHeader().setStretchLastSection(True)

        self.analyze_button = QPushButton("Find fastest")
        self.apply_button = QPushButton("Apply measured assignment")
        self.apply_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel analysis")
        self.cancel_button.setVisible(False)
        self.close_button = QPushButton("Close")

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.analyze_button)
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

        self.analyze_button.clicked.connect(self._request_analysis)
        self.apply_button.clicked.connect(self._apply_result)
        self.cancel_button.clicked.connect(self.cancel)
        self.close_button.clicked.connect(self.accept)

    @property
    def running(self) -> bool:
        return self._running

    @property
    def outcome(self) -> PipelineOptimizerWorkerOutcome | None:
        return self._outcome

    def start(
        self,
        worker: PipelineOptimizerWorker,
        thread_pool: QThreadPool,
    ) -> None:
        if self._shutdown:
            raise RuntimeError("The pipeline optimizer dialog is shut down")
        if self._running:
            raise RuntimeError("A pipeline optimization is already running")
        self._worker = worker
        self._outcome = None
        self._running = True
        self.analyze_button.setEnabled(False)
        self.apply_button.setEnabled(False)
        self.close_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.result_table.setVisible(False)
        self.result_label.setText("")
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText(
            "Capturing exact pipeline evidence. You can cancel at any time."
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
            self.analyze_button.setEnabled(True)
            self.close_button.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.cancel_button.setVisible(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_label.setText("Analysis could not be dispatched.")
            raise

    def cancel(self) -> None:
        if not self._running or self._worker is None:
            return
        self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.progress_label.setText(
            "Cancel requested; finishing the current synchronized call…"
        )

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
        self.analyze_button.setEnabled(False)
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

    def _request_analysis(self) -> None:
        if self._running or self._shutdown:
            return
        self.analyze_requested.emit()

    def _on_progress(self, progress: PipelineOptimizerProgress) -> None:
        if self._shutdown:
            return
        self.progress_bar.setRange(0, progress.total)
        self.progress_bar.setValue(progress.completed)
        self.progress_label.setText(progress.message)

    def _on_finished(self, outcome: PipelineOptimizerWorkerOutcome) -> None:
        if self._shutdown:
            return
        self._running = False
        self._outcome = outcome
        self._worker = None
        self.analyze_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        if outcome.result is not None:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_label.setText("Pipeline analysis complete.")
            self._render_result(outcome.result)
            proposal = _proposal_from_result(outcome.result)
            self.apply_button.setEnabled(
                any(
                    row.current_preference != row.proposed_preference
                    for row in proposal.rows
                )
            )
        else:
            self.apply_button.setEnabled(False)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            if outcome.cancelled:
                self.progress_label.setText(
                    "Analysis cancelled. No node preference changed."
                )
                self.result_label.setStyleSheet("")
            elif outcome.reason_code in {
                "evidence_incomplete",
                "not_beneficial",
                "deadline_exceeded",
            }:
                self.progress_label.setText(
                    "No safe pipeline-wide change is recommended."
                )
                self.result_label.setStyleSheet("color: #fcd34d;")
            else:
                self.progress_label.setText("Pipeline analysis failed.")
                self.result_label.setStyleSheet("color: #fca5a5;")
            self.result_label.setText(outcome.error)
        self.optimizer_finished.emit(outcome)

    def _render_result(self, result: object) -> None:
        proposal = _proposal_from_result(result)
        evidence = getattr(result, "evidence", {})
        reused = set(getattr(result, "reused_node_ids", ()))
        measured = set(getattr(result, "measured_node_ids", ()))
        rows = tuple(proposal.rows)
        tested_by_node = dict(proposal.tested_assignment)
        self.result_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            preference = row.proposed_preference.kind.value
            if row.proposed_preference.value:
                preference += f":{row.proposed_preference.value}"
            status = "Switch backend" if row.changed else "Keep measured backend"
            if row.locked:
                status = "Locked"
            elif not row.eligible:
                status = "Fixed / excluded"
            elif (
                proposal.validation_winner is PipelineValidationWinner.CURRENT
                and tested_by_node.get(row.node_id)
                != row.current_implementation_id
            ):
                status = "Current won final validation"
            timing = _candidate_timing_text(evidence.get(row.node_id))
            if row.node_id in reused:
                timing = f"{timing} [reused]" if timing else "Exact evidence reused"
            elif row.node_id in measured:
                timing = f"{timing} [measured]" if timing else "Measured now"
            values = (
                row.node_id,
                row.current_implementation_id,
                tested_by_node.get(row.node_id, row.proposed_implementation_id),
                row.proposed_implementation_id,
                preference,
                timing,
                status,
            )
            for column, value in enumerate(values):
                self.result_table.setItem(
                    row_index,
                    column,
                    QTableWidgetItem(str(value)),
                )
        self.result_table.resizeColumnsToContents()
        self.result_table.setVisible(True)
        changed = sum(
            1
            for row in rows
            if row.current_preference != row.proposed_preference
        )
        self.result_label.setStyleSheet("color: #86efac; font-weight: 650;")
        if proposal.pipeline_validation_performed:
            current = proposal.validated_current_seconds
            tested = proposal.validated_proposed_seconds
            if proposal.validation_winner is PipelineValidationWinner.CURRENT:
                speedup = tested / current if current > 0 else float("inf")
                self.result_label.setText(
                    "The current assignment won the final paired validation: "
                    f"{_format_seconds(current)} versus "
                    f"{_format_seconds(tested)} for the model-selected "
                    f"alternative ({speedup:.2f}× faster; lower confidence "
                    "bound "
                    f"{proposal.validated_current_speedup_lower_confidence_bound:.2f}×;"
                    " "
                    f"{proposal.validation_measurement_rounds} paired rounds). "
                    f"{changed} measured preference(s) can still be saved; "
                    f"{len(reused)} node benchmark(s) reused exact saved evidence."
                )
            else:
                speedup = current / tested if tested > 0 else float("inf")
                self.result_label.setText(
                    f"Validated proposal: {_format_seconds(current)} → "
                    f"{_format_seconds(tested)} ({speedup:.2f}×; lower confidence "
                    "bound "
                    f"{proposal.validated_speedup_lower_confidence_bound:.2f}×; "
                    f"{proposal.validation_measurement_rounds} paired validation "
                    "rounds). "
                    f"{changed} measured node preference(s) would change; "
                    f"{len(reused)} node benchmark(s) reused exact saved evidence."
                )
        else:
            self.result_label.setText(
                "The current exact backend assignment won the measured global "
                "comparison, so no alternative pipeline timing run was needed. "
                f"{changed} measured preference(s) can still be saved; "
                f"{len(reused)} node benchmark(s) reused exact saved evidence."
            )

    def _apply_result(self) -> None:
        if (
            self._shutdown
            or self._outcome is None
            or self._outcome.result is None
        ):
            return
        self.apply_requested.emit(self._outcome.result)


def _format_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1_000_000:.1f} µs"
    if value < 1.0:
        return f"{value * 1_000:.1f} ms"
    return f"{value:.3f} s"


def _proposal_from_result(result: object) -> PipelineOptimizationProposal:
    proposal = getattr(result, "proposal", result)
    if not isinstance(proposal, PipelineOptimizationProposal):
        raise TypeError("optimizer worker returned no pipeline proposal")
    return proposal


def _candidate_timing_text(evidence: object) -> str:
    record = getattr(evidence, "record", None)
    candidates = getattr(record, "candidates", ())
    values: list[str] = []
    for candidate in candidates:
        error = str(getattr(candidate, "error", "") or "").strip()
        if error or not bool(getattr(candidate, "parity_passed", False)):
            detail = error or "scientific parity failed"
            values.append(f"{candidate.implementation_id} excluded ({detail})")
            continue
        resident = tuple(getattr(candidate, "warm_resident_seconds", ()))
        warm = tuple(getattr(candidate, "warm_seconds", ()))
        modeled = resident or warm
        if not modeled:
            continue
        ordered = sorted(float(value) for value in modeled)
        middle = len(ordered) // 2
        median = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
        metric = "GPU resident" if resident else "CPU/warm end-to-end"
        values.append(
            f"{candidate.implementation_id} {_format_seconds(median)} ({metric})"
        )
    return "; ".join(values)


__all__ = [
    "PipelineOptimizerDialog",
    "PipelineOptimizerProgress",
    "PipelineOptimizerWorker",
    "PipelineOptimizerWorkerOutcome",
]
