"""Review-first Qt surface for whole-pipeline compute optimization."""

from __future__ import annotations

import html
import threading
from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
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
    PipelineOptimizationTimeoutReport,
    PipelineValidationWinner,
)


@dataclass(frozen=True, slots=True)
class PipelineOptimizerProgress:
    """One application-level progress update suitable for queued Qt delivery."""

    completed: int
    total: int
    message: str
    phase: str = ""
    operation_completed: int = 0
    operation_total: int = 0
    operation_message: str = ""
    node_id: str = ""
    node_title: str = ""
    implementation_id: str = ""
    measurement_phase: str = ""

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.completed, self.total)
        ):
            raise ValueError("optimizer progress values must be non-negative integers")
        if self.total < 1 or self.completed > self.total:
            raise ValueError("optimizer progress must fit inside its declared total")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.operation_completed, self.operation_total)
        ):
            raise ValueError("operation progress values must be non-negative integers")
        if self.operation_completed > self.operation_total:
            raise ValueError("operation progress must fit inside its declared total")
        message = str(self.message).strip()
        if not message:
            raise ValueError("optimizer progress message must not be empty")
        object.__setattr__(self, "message", message)
        for name in (
            "phase",
            "operation_message",
            "node_id",
            "node_title",
            "implementation_id",
            "measurement_phase",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if self.operation_total and not self.operation_message:
            raise ValueError("operation progress requires a message")


@dataclass(frozen=True, slots=True)
class PipelineOptimizerWorkerOutcome:
    """Terminal optimizer result; expected refusals remain ordinary data."""

    result: object | None = None
    error: str = ""
    reason_code: str = ""
    cancelled: bool = False
    timeout_report: PipelineOptimizationTimeoutReport | None = None


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
                timeout_report=exc.report,
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
            "<li><b>Manual nodes:</b> The detached analysis calculates manual "
            "nodes so their output and cost are included; your live cached "
            "results are not changed.</li>"
            "<li><b>Control:</b> Nothing changes until you review and apply the "
            "result.</li>"
            "<li><b>Timing:</b> CPU uses paired warm medians; GPU uses "
            "resident-compute medians, with transfers modeled across the whole "
            "pipeline.</li>"
            "</ul>"
        )
        self.summary_label.setTextFormat(Qt.RichText)
        self.summary_label.setWordWrap(True)
        locked_node_label = "node" if locked_node_count == 1 else "nodes"
        self.time_limit_combo = QComboBox()
        self.time_limit_combo.setAccessibleName("Pipeline optimization time limit")
        for label, seconds in (
            ("5 minutes", 300.0),
            ("15 minutes", 900.0),
            ("30 minutes", 1_800.0),
            ("60 minutes", 3_600.0),
        ):
            self.time_limit_combo.addItem(label, seconds)
        self.time_limit_combo.setToolTip(
            "Maximum wall-clock analysis time. This is not a RAM or VRAM "
            "limit. Completed exact node evidence is reused on a later retry."
        )
        time_limit_label = QLabel("Time limit")
        time_limit_label.setBuddy(self.time_limit_combo)
        time_limit_note = QLabel(
            "Completed exact node results are reused if you retry."
        )
        time_limit_note.setWordWrap(True)
        time_limit_row = QHBoxLayout()
        time_limit_row.addWidget(time_limit_label)
        time_limit_row.addWidget(self.time_limit_combo)
        time_limit_row.addWidget(time_limit_note, 1)

        self.overall_progress_label = QLabel(
            f"Ready. {locked_node_count} explicitly locked {locked_node_label} "
            "will be preserved."
        )
        self.overall_progress_label.setWordWrap(True)
        self.overall_progress_bar = QProgressBar()
        self.overall_progress_bar.setAccessibleName(
            "Overall pipeline optimization progress"
        )
        self.overall_progress_bar.setRange(0, 1)
        self.overall_progress_bar.setValue(0)
        self.overall_progress_bar.setFormat("Overall %p%")
        self.operation_progress_label = QLabel(
            "Current operation: waiting for analysis."
        )
        self.operation_progress_label.setWordWrap(True)
        self.operation_progress_bar = QProgressBar()
        self.operation_progress_bar.setAccessibleName(
            "Current operation benchmark progress"
        )
        self.operation_progress_bar.setRange(0, 1)
        self.operation_progress_bar.setValue(0)
        self.operation_progress_bar.setFormat("Waiting")
        # Compatibility aliases for callers written against the original
        # single-progress dialog surface.
        self.progress_label = self.overall_progress_label
        self.progress_bar = self.overall_progress_bar
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
        layout.addLayout(time_limit_row)
        layout.addWidget(self.overall_progress_label)
        layout.addWidget(self.overall_progress_bar)
        layout.addWidget(self.operation_progress_label)
        layout.addWidget(self.operation_progress_bar)
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

    @property
    def time_budget_seconds(self) -> float:
        value = self.time_limit_combo.currentData()
        try:
            seconds = float(value)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Pipeline optimizer time limit is invalid") from exc
        if seconds <= 0:
            raise RuntimeError("Pipeline optimizer time limit must be positive")
        return seconds

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
        self.time_limit_combo.setEnabled(False)
        self.cancel_button.setVisible(True)
        self.cancel_button.setEnabled(True)
        self.result_table.setVisible(False)
        self.result_label.setText("")
        self.overall_progress_bar.setRange(0, 0)
        self.overall_progress_label.setText(
            "Overall pipeline: capturing exact evidence. You can cancel at any "
            "time."
        )
        self.operation_progress_bar.setRange(0, 1)
        self.operation_progress_bar.setValue(0)
        self.operation_progress_bar.setFormat("Waiting")
        self.operation_progress_label.setText(
            "Current operation: waiting for the first benchmark stage."
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
            self.time_limit_combo.setEnabled(True)
            self.cancel_button.setEnabled(False)
            self.cancel_button.setVisible(False)
            self.overall_progress_bar.setRange(0, 1)
            self.overall_progress_bar.setValue(0)
            self.overall_progress_label.setText(
                "Overall pipeline: analysis could not be dispatched."
            )
            self.operation_progress_bar.setRange(0, 1)
            self.operation_progress_bar.setValue(0)
            self.operation_progress_bar.setFormat("Not started")
            self.operation_progress_label.setText(
                "Current operation: no benchmark was started."
            )
            raise

    def cancel(self) -> None:
        if not self._running or self._worker is None:
            return
        self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.overall_progress_label.setText(
            "Overall pipeline: cancel requested."
        )
        self.operation_progress_label.setText(
            "Current operation: finishing the current synchronized call before "
            "cancelling…"
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
        self.time_limit_combo.setEnabled(False)
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
        self.overall_progress_bar.setRange(0, progress.total)
        self.overall_progress_bar.setValue(progress.completed)
        self.overall_progress_bar.setFormat("Overall %p%")
        self.overall_progress_label.setText(
            f"Overall pipeline: {progress.message}"
        )
        if progress.operation_total:
            self.operation_progress_bar.setRange(0, progress.operation_total)
            self.operation_progress_bar.setValue(progress.operation_completed)
            self.operation_progress_bar.setFormat("Current %p%")
            self.operation_progress_label.setText(
                f"Current operation: {progress.operation_message}"
            )
        else:
            self.operation_progress_bar.setRange(0, 1)
            self.operation_progress_bar.setValue(0)
            self.operation_progress_bar.setFormat("Waiting")
            self.operation_progress_label.setText(
                "Current operation: waiting for a node or validation stage."
            )

    def _on_finished(self, outcome: PipelineOptimizerWorkerOutcome) -> None:
        if self._shutdown:
            return
        self._running = False
        self._outcome = outcome
        self._worker = None
        self.analyze_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.time_limit_combo.setEnabled(True)
        self.cancel_button.setVisible(False)
        if outcome.result is not None:
            self.overall_progress_bar.setValue(
                self.overall_progress_bar.maximum()
            )
            self.overall_progress_bar.setFormat("Overall 100%")
            self.overall_progress_label.setText(
                "Overall pipeline: analysis complete."
            )
            self.operation_progress_bar.setRange(0, 1)
            self.operation_progress_bar.setValue(1)
            self.operation_progress_bar.setFormat("Current 100%")
            self.operation_progress_label.setText(
                "Current operation: final validation complete."
            )
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
            if outcome.cancelled:
                self.overall_progress_label.setText(
                    "Overall pipeline: analysis cancelled. No node preference "
                    "changed."
                )
                self.overall_progress_bar.setFormat("Cancelled at %p%")
                self.operation_progress_bar.setFormat("Cancelled at %p%")
                self.result_label.setStyleSheet("")
                self.result_label.setTextFormat(Qt.PlainText)
                self.result_label.setText(outcome.error)
            elif outcome.reason_code == "deadline_exceeded":
                self.overall_progress_label.setText(
                    "Overall pipeline: analysis stopped before a winner could "
                    "be determined."
                )
                self.overall_progress_bar.setFormat("Stopped at %p%")
                self.operation_progress_bar.setFormat("Stopped at %p%")
                self.result_label.setStyleSheet("color: #fcd34d;")
                self.result_label.setTextFormat(Qt.RichText)
                self.result_label.setText(
                    _timeout_result_html(
                        outcome.error,
                        outcome.timeout_report,
                        selected_budget_seconds=self.time_budget_seconds,
                    )
                )
            elif outcome.reason_code in {
                "evidence_incomplete",
                "not_beneficial",
            }:
                self.overall_progress_label.setText(
                    "Overall pipeline: no safe pipeline-wide change is "
                    "recommended."
                )
                self.result_label.setStyleSheet("color: #fcd34d;")
                self.result_label.setTextFormat(Qt.PlainText)
                self.result_label.setText(outcome.error)
            else:
                self.overall_progress_label.setText(
                    "Overall pipeline: analysis failed."
                )
                self.result_label.setStyleSheet("color: #fca5a5;")
                self.result_label.setTextFormat(Qt.PlainText)
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


def _format_duration(value: float) -> str:
    seconds = max(0.0, float(value))
    if seconds < 60:
        return f"{seconds:.1f} seconds"
    minutes = seconds / 60
    if minutes.is_integer():
        return f"{int(minutes)} minutes"
    return f"{minutes:.1f} minutes"


def _timeout_result_html(
    error: str,
    report: PipelineOptimizationTimeoutReport | None,
    *,
    selected_budget_seconds: float,
) -> str:
    """Render an actionable timeout without implying an optimal result."""

    def escaped(value: object) -> str:
        return html.escape(str(value), quote=True)

    budget = selected_budget_seconds
    elapsed: float | None = None
    stage = "The selected analysis time limit was reached."
    progress_text = "The analysis stopped before all required evidence was collected."
    evidence_text = (
        "Any complete exact node benchmarks remain reusable; incomplete timings "
        "from the interrupted operation are not used."
    )
    if report is not None:
        if report.budget_seconds is not None:
            budget = report.budget_seconds
        elapsed = report.elapsed_seconds
        stage_detail = report.stage_message or report.stage.replace("-", " ")
        if report.node_title:
            stage = f"{report.node_title} — {stage_detail}"
        else:
            stage = stage_detail
        progress_parts: list[str] = []
        if report.node_total:
            progress_parts.append(
                f"node {report.node_index} of {report.node_total}"
            )
        progress_parts.append(
            f"overall step {report.overall_completed} of {report.overall_total}"
        )
        if report.operation_total:
            progress_parts.append(
                "current operation "
                f"{report.operation_completed} of {report.operation_total}"
            )
        progress_text = "; ".join(progress_parts) + "."
        completed_count = len(report.completed_node_ids)
        reused_count = len(report.reused_node_ids)
        evidence_parts = [
            f"{completed_count} complete exact node benchmark"
            f"{'s' if completed_count != 1 else ''} retained"
        ]
        if reused_count:
            evidence_parts.append(f"{reused_count} reused from an earlier run")
        if report.partial_node_discarded:
            evidence_parts.append(
                "partial timings for the interrupted node were discarded"
            )
        evidence_text = "; ".join(evidence_parts) + "."

    if elapsed is None:
        time_text = f"Selected wall-clock limit: {_format_duration(budget)}."
    else:
        time_text = (
            f"Stopped after {_format_duration(elapsed)} of the "
            f"{_format_duration(budget)} wall-clock limit."
        )
    error_text = str(error).strip()
    if error_text:
        stage = stage or error_text

    items = (
        ("Stopped stage", stage),
        ("Progress", progress_text),
        ("Time", f"{time_text} This is a time limit, not a RAM or VRAM limit."),
        ("Saved evidence", evidence_text),
        (
            "Conclusion",
            "No fastest assignment was determined. The current pipeline was "
            "not proven fastest, and no settings changed.",
        ),
        (
            "Next step",
            "Retry with a longer time limit. Complete exact node benchmarks "
            "will be reused when the workload and environment are unchanged.",
        ),
    )
    rendered = "".join(
        f"<li><b>{escaped(label)}:</b> {escaped(value)}</li>"
        for label, value in items
    )
    return (
        "<p><b>Analysis timed out before a winner was determined.</b></p>"
        f"<ul>{rendered}</ul>"
    )


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
