"""Review-first Qt surface for whole-pipeline compute optimization."""

from __future__ import annotations

import html
import statistics
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtGui import QBrush, QColor, QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QHeaderView,
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
    PipelineOptimizationSelectionBasis,
    PipelineOptimizationTimeoutReport,
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


@dataclass(frozen=True, slots=True)
class PipelineOptimizerApplyRequest:
    """Explicit acceptance of one measured near-parity optimizer result."""

    result: object
    parity_review_digest: str = ""

    def __post_init__(self) -> None:
        digest = self.parity_review_digest
        if not isinstance(digest, str) or not digest.strip():
            raise ValueError("parity review digest must not be blank")
        proposal = _proposal_from_result(self.result)
        if not proposal.requires_parity_review:
            raise ValueError("optimizer result does not require parity review")
        if digest != proposal.parity_review_digest:
            raise ValueError("parity review digest does not match the proposal")


class PipelineOptimizerCleanupError(RuntimeError):
    """The optimizer's private accelerator runtime did not close safely."""

    cleanup_succeeded = False


def _subtle_group_brush(widget: QWidget) -> QBrush:
    """Return a dark/light-theme-safe tint with deliberately low contrast."""

    base = widget.palette().base().color()
    text = widget.palette().text().color()
    fraction = _GROUP_TINT_FRACTION
    color = QColor(
        round(base.red() * (1.0 - fraction) + text.red() * fraction),
        round(base.green() * (1.0 - fraction) + text.green() * fraction),
        round(base.blue() * (1.0 - fraction) + text.blue() * fraction),
        base.alpha(),
    )
    return QBrush(color)


_PRIMARY_RESULT_COLUMNS = (
    "Node",
    "Implementation",
    "Total time",
    "Scientific check",
    "Result",
)
_DETAIL_RESULT_COLUMNS = (
    "Compute",
    "Data movement",
    "First run",
    "Peak memory",
    "Evidence",
)
_DETAIL_COLUMN_START = len(_PRIMARY_RESULT_COLUMNS)
_RESULT_COLUMN_COUNT = len(_PRIMARY_RESULT_COLUMNS) + len(_DETAIL_RESULT_COLUMNS)
_GROUP_TINT_FRACTION = 0.06


class _PipelineOptimizerWorkerSignals(QObject):
    progress = Signal(object)
    finished = Signal(object)


class _CopyableResultTable(QTableWidget):
    """Read-only result table whose visible selection can be copied as TSV."""

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.matches(QKeySequence.Copy):
            self.copy_selection()
            event.accept()
            return
        super().keyPressEvent(event)

    def copy_selection(self) -> None:
        indexes = tuple(self.selectedIndexes())
        if not indexes:
            return
        rows = range(
            min(index.row() for index in indexes),
            max(index.row() for index in indexes) + 1,
        )
        columns = tuple(
            column
            for column in range(
                min(index.column() for index in indexes),
                max(index.column() for index in indexes) + 1,
            )
            if not self.isColumnHidden(column)
        )
        selected = {(index.row(), index.column()) for index in indexes}
        lines: list[str] = []
        for row in rows:
            values: list[str] = []
            for column in columns:
                item = self.item(row, column)
                values.append(
                    item.text()
                    if (row, column) in selected and item is not None
                    else ""
                )
            lines.append("\t".join(values))
        QApplication.clipboard().setText("\n".join(lines))


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
        except PipelineOptimizerCleanupError as exc:
            outcome = PipelineOptimizerWorkerOutcome(
                error=str(exc),
                reason_code="cleanup_failed",
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
        node_titles: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Find fastest pipeline")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumSize(860, 480)
        self._worker: PipelineOptimizerWorker | None = None
        self._outcome: PipelineOptimizerWorkerOutcome | None = None
        self._running = False
        self._shutdown = False
        self._node_titles = {
            str(node_id): str(title).strip() or str(node_id)
            for node_id, title in dict(node_titles or {}).items()
        }

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
            "pipeline. A decisively slower cooperative CPU attempt may stop "
            "early and is shown as a lower bound, never an exact timing; the "
            "chosen assignment still receives final whole-pipeline "
            "validation.</li>"
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
        self.result_table = _CopyableResultTable(0, _RESULT_COLUMN_COUNT)
        self.result_table.setHorizontalHeaderLabels(
            _PRIMARY_RESULT_COLUMNS + _DETAIL_RESULT_COLUMNS
        )
        self.result_table.setAccessibleName("Per-node CPU and GPU pipeline comparison")
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionBehavior(QAbstractItemView.SelectItems)
        self.result_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.result_table.setWordWrap(True)
        self.result_table.setAlternatingRowColors(False)
        self.result_table.verticalHeader().setVisible(False)
        self.result_table.setVisible(False)
        header = self.result_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.Interactive)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)
        for column in range(_DETAIL_COLUMN_START, _RESULT_COLUMN_COUNT):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
            self.result_table.setColumnHidden(column, True)
        self.result_table.setColumnWidth(0, 190)
        header_tooltips = (
            "Workflow node. One cell groups every implementation tested for it.",
            "Friendly CPU or GPU name. Hover for the exact implementation ID.",
            "Median measured time after warm-up. Hover for timing scope.",
            "Whether this implementation reproduced the reference result.",
            "How this implementation relates to the reviewed pipeline decision.",
            "Resident compute time where available; otherwise the measured call time.",
            "Measured transfer and host-materialization time.",
            "The first measured call, before typical repeated timings.",
            "Largest recorded memory use for this implementation.",
            "Measured rounds and whether evidence was measured now or reused.",
        )
        for column, tooltip in enumerate(header_tooltips):
            item = self.result_table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tooltip)

        self.details_button = QPushButton("Show timing details")
        self.details_button.setAccessibleName("Show detailed pipeline timing columns")
        self.details_button.setCheckable(True)
        self.details_button.setVisible(False)
        self.details_button.toggled.connect(self._set_timing_details_visible)

        self.parity_review_checkbox = QCheckBox(
            "I reviewed and accept this measured CPU/GPU difference for this "
            "exact workflow and input."
        )
        self.parity_review_checkbox.setAccessibleName(
            "Accept the measured CPU and GPU output difference"
        )
        self.parity_review_checkbox.setToolTip(
            "This acceptance applies only to the workflow, input, and exact "
            "CPU/GPU backends measured in this result."
        )
        self.parity_review_checkbox.setVisible(False)

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
        layout.addWidget(self.details_button, 0, Qt.AlignLeft)
        layout.addWidget(self.result_table, 1)
        layout.addWidget(self.parity_review_checkbox)
        layout.addLayout(buttons)

        self.analyze_button.clicked.connect(self._request_analysis)
        self.apply_button.clicked.connect(self._apply_result)
        self.parity_review_checkbox.toggled.connect(self._sync_apply_enabled)
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
        self._reset_parity_review()
        self._set_review_mode(False)
        self.details_button.setChecked(False)
        self.details_button.setVisible(False)
        self.result_table.setVisible(False)
        self.result_label.setText("")
        self.result_label.setToolTip("")
        self.overall_progress_bar.setRange(0, 0)
        self.overall_progress_label.setText(
            "Overall pipeline: capturing exact evidence. You can cancel at any time."
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
        self.overall_progress_label.setText("Overall pipeline: cancel requested.")
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
        self._reset_parity_review()
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

    def _set_review_mode(self, reviewing: bool) -> None:
        """Give completed results the space previously used by progress prose."""

        for widget in (
            self.summary_label,
            self.overall_progress_label,
            self.overall_progress_bar,
            self.operation_progress_label,
            self.operation_progress_bar,
        ):
            widget.setVisible(not reviewing)

    def _reset_parity_review(self) -> None:
        """Clear acceptance so it can never carry across optimizer results."""

        self.parity_review_checkbox.setChecked(False)
        self.parity_review_checkbox.setVisible(False)

    def _sync_apply_enabled(self, _checked: bool | None = None) -> None:
        if self._shutdown or self._running:
            self.apply_button.setEnabled(False)
            return
        outcome = self._outcome
        if outcome is None or outcome.result is None:
            self.apply_button.setEnabled(False)
            return
        proposal = _proposal_from_result(outcome.result)
        changed = any(
            row.current_preference != row.proposed_preference for row in proposal.rows
        )
        enabled = _enum_value(proposal.validation_winner) != "inconclusive" and changed
        if proposal.requires_parity_review:
            enabled = enabled and self.parity_review_checkbox.isChecked()
        self.apply_button.setEnabled(enabled)

    def _set_timing_details_visible(self, visible: bool) -> None:
        for column in range(_DETAIL_COLUMN_START, _RESULT_COLUMN_COUNT):
            self.result_table.setColumnHidden(column, not visible)
        self.details_button.setText(
            "Hide timing details" if visible else "Show timing details"
        )
        if self.result_table.isVisible():
            self.result_table.resizeRowsToContents()

    def _on_progress(self, progress: PipelineOptimizerProgress) -> None:
        if self._shutdown:
            return
        self.overall_progress_bar.setRange(0, progress.total)
        self.overall_progress_bar.setValue(progress.completed)
        self.overall_progress_bar.setFormat("Overall %p%")
        self.overall_progress_label.setText(f"Overall pipeline: {progress.message}")
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
        if outcome.result is None:
            self._reset_parity_review()
            self.result_label.setToolTip("")
        if outcome.result is not None:
            self.overall_progress_bar.setValue(self.overall_progress_bar.maximum())
            self.overall_progress_bar.setFormat("Overall 100%")
            self.overall_progress_label.setText("Overall pipeline: analysis complete.")
            self.operation_progress_bar.setRange(0, 1)
            self.operation_progress_bar.setValue(1)
            self.operation_progress_bar.setFormat("Current 100%")
            self.operation_progress_label.setText(
                "Current operation: final validation complete."
            )
            self._render_result(outcome.result)
            self._sync_apply_enabled()
        else:
            self.apply_button.setEnabled(False)
            if outcome.cancelled:
                self.overall_progress_label.setText(
                    "Overall pipeline: analysis cancelled. No node preference changed."
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
                    "Overall pipeline: no safe pipeline-wide change is recommended."
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
            self.details_button.setVisible(False)
        self.optimizer_finished.emit(outcome)

    def _render_result(self, result: object) -> None:
        proposal = _proposal_from_result(result)
        self._reset_parity_review()
        evidence = getattr(result, "evidence", {})
        reused = set(getattr(result, "reused_node_ids", ()))
        measured = set(getattr(result, "measured_node_ids", ()))
        rows = tuple(proposal.rows)
        tested_by_node = dict(proposal.tested_assignment)
        winner = _enum_value(proposal.validation_winner)

        groups: list[tuple[object, tuple[str, ...], Mapping[str, object]]] = []
        for row in rows:
            node_evidence = evidence.get(row.node_id)
            record = getattr(node_evidence, "record", None)
            candidates = tuple(getattr(record, "candidates", ()))
            candidate_by_id = {
                str(candidate.implementation_id): candidate for candidate in candidates
            }
            ordered_ids = _ordered_implementation_ids(
                row,
                tested_by_node.get(
                    row.node_id,
                    row.proposed_implementation_id,
                ),
                candidate_by_id,
            )
            groups.append((row, ordered_ids, candidate_by_id))

        self.result_table.clearSpans()
        self.result_table.setRowCount(
            sum(len(implementation_ids) for _, implementation_ids, _ in groups)
        )
        table_row = 0
        alternate_brush = _subtle_group_brush(self.result_table)
        for group_index, (row, implementation_ids, candidate_by_id) in enumerate(
            groups
        ):
            first_row = table_row
            node_title = self._node_titles.get(row.node_id, row.node_id)
            node_item = QTableWidgetItem(node_title)
            node_item.setToolTip(
                f"Workflow node: {node_title}\nInternal ID: {row.node_id}"
            )
            node_item.setTextAlignment(int(Qt.AlignLeft | Qt.AlignVCenter))
            node_font = node_item.font()
            node_font.setBold(True)
            node_item.setFont(node_font)
            self.result_table.setItem(first_row, 0, node_item)

            tested_id = tested_by_node.get(
                row.node_id,
                row.proposed_implementation_id,
            )
            evidence_source = _evidence_source_text(
                row.node_id,
                measured=measured,
                reused=reused,
                has_record=record is not None,
            )
            for implementation_id in implementation_ids:
                candidate = candidate_by_id.get(implementation_id)
                timing = _candidate_timing_display(
                    candidate,
                    evidence_source=evidence_source,
                )
                scientific_text, scientific_tooltip = _scientific_check(candidate)
                result_text = _implementation_result_text(
                    row,
                    implementation_id,
                    tested_id=tested_id,
                    winner=winner,
                    candidate=candidate,
                )
                values = (
                    "" if table_row != first_row else node_title,
                    _friendly_implementation_name(implementation_id),
                    timing.total,
                    scientific_text,
                    result_text,
                    timing.compute,
                    timing.data_movement,
                    timing.first_run,
                    timing.peak_memory,
                    timing.evidence,
                )
                for column, value in enumerate(values):
                    if column == 0 and table_row == first_row:
                        item = node_item
                    else:
                        item = QTableWidgetItem(str(value))
                        self.result_table.setItem(table_row, column, item)
                    if group_index % 2:
                        item.setBackground(alternate_brush)
                implementation_item = self.result_table.item(table_row, 1)
                implementation_item.setToolTip(
                    "Exact implementation ID: " + implementation_id
                )
                total_item = self.result_table.item(table_row, 2)
                total_item.setToolTip(timing.total_tooltip)
                scientific_item = self.result_table.item(table_row, 3)
                scientific_item.setToolTip(scientific_tooltip)
                movement_item = self.result_table.item(table_row, 6)
                movement_item.setToolTip(timing.data_movement_tooltip)
                evidence_item = self.result_table.item(table_row, 9)
                evidence_item.setToolTip(timing.evidence_tooltip)
                table_row += 1

            if len(implementation_ids) > 1:
                self.result_table.setSpan(
                    first_row,
                    0,
                    len(implementation_ids),
                    1,
                )

        self.details_button.setChecked(False)
        self.details_button.setVisible(True)
        self.result_table.setVisible(True)
        self.result_table.resizeRowsToContents()
        self.result_table.scrollToTop()
        self._set_review_mode(True)
        changed = sum(
            1 for row in rows if row.current_preference != row.proposed_preference
        )
        self.result_label.setTextFormat(Qt.PlainText)
        if proposal.pipeline_validation_performed:
            current = proposal.validated_current_seconds
            tested = proposal.validated_proposed_seconds
            rounds = proposal.validation_measurement_rounds
            rounds_text = f"{rounds} paired round{'s' if rounds != 1 else ''}"
            if winner == "inconclusive":
                self.result_label.setStyleSheet("font-weight: 650;")
                self.result_label.setText(
                    "No clear winner—current settings kept. Whole-pipeline "
                    f"totals: current {_format_seconds(current)}; tested "
                    f"{_format_seconds(tested)} ({rounds_text}). The difference "
                    "was not large and certain enough to justify a change."
                )
                self.result_label.setToolTip(
                    "How this was decided: VIPP changes the pipeline only when "
                    "the measured improvement is greater than 5% or 10 ms, "
                    "whichever is larger, and the paired lower confidence "
                    "bound is above 1.0."
                )
            elif winner == "current":
                self.result_label.setToolTip("")
                self.result_label.setStyleSheet("color: #86efac; font-weight: 650;")
                self.result_label.setText(
                    "Current settings were faster in final validation: "
                    f"{_format_seconds(current)} versus "
                    f"{_format_seconds(tested)} for the tested setup "
                    f"({rounds_text}). {changed} measured preference"
                    f"{'s' if changed != 1 else ''} can be saved."
                )
            else:
                self.result_label.setToolTip("")
                self.result_label.setStyleSheet("color: #86efac; font-weight: 650;")
                self.result_label.setText(
                    "Faster pipeline validated: "
                    f"{_format_seconds(current)} current → "
                    f"{_format_seconds(tested)} tested ({rounds_text}). "
                    f"{changed} node preference{'s' if changed != 1 else ''} "
                    "would change."
                )
        else:
            self.result_label.setToolTip("")
            self.result_label.setStyleSheet("color: #86efac; font-weight: 650;")
            basis_type = PipelineOptimizationSelectionBasis
            conservative_basis = basis_type.CONSERVATIVE_BOUND_RETAINED_CURRENT
            if proposal.selection_basis is conservative_basis:
                explanation = (
                    "The current GPU assignment was retained because every "
                    "competing CPU measurement had already crossed a decisive "
                    "slower lower bound. Those losing CPU timings stopped early, "
                    "so a redundant alternative pipeline run was not needed. "
                )
            else:
                explanation = (
                    "The current exact backend assignment won the measured global "
                    "comparison, so no alternative pipeline timing run was needed. "
                )
            self.result_label.setText(
                explanation + f"Estimated whole-pipeline total: "
                f"{_format_seconds(proposal.estimated_current_seconds)}. "
                f"{changed} measured preference{'s' if changed != 1 else ''} "
                "can be saved."
            )

        if proposal.requires_parity_review:
            optimization_summary = self.result_label.text()
            review_text = _parity_review_text(proposal, self._node_titles)
            self.result_label.setToolTip(
                "The measured difference is within VIPP's review limit, but "
                "only you can decide whether it is acceptable for this analysis."
            )
            self.result_label.setStyleSheet("color: #fcd34d; font-weight: 650;")
            self.result_label.setText(
                f"{review_text}\n\n{optimization_summary}"
                if optimization_summary
                else review_text
            )
            self.parity_review_checkbox.setVisible(True)

        candidate_refusals = tuple(getattr(result, "candidate_refusals", ()))
        if candidate_refusals:
            existing_summary = self.result_label.text()
            refusal_text = _candidate_refusal_text(
                candidate_refusals,
                self._node_titles,
            )
            self.result_label.setStyleSheet("color: #fcd34d; font-weight: 650;")
            self.result_label.setText(
                f"{existing_summary}\n\n{refusal_text}"
                if existing_summary
                else refusal_text
            )

    def _apply_result(self) -> None:
        if self._shutdown or self._outcome is None or self._outcome.result is None:
            return
        result = self._outcome.result
        proposal = _proposal_from_result(result)
        if proposal.requires_parity_review:
            if not self.parity_review_checkbox.isChecked():
                return
            request = PipelineOptimizerApplyRequest(
                result,
                proposal.parity_review_digest,
            )
            self.apply_requested.emit(request)
            return
        self.apply_requested.emit(result)


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
            progress_parts.append(f"node {report.node_index} of {report.node_total}")
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
        f"<li><b>{escaped(label)}:</b> {escaped(value)}</li>" for label, value in items
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


@dataclass(frozen=True, slots=True)
class _CandidateTimingDisplay:
    total: str = "—"
    compute: str = "—"
    data_movement: str = "—"
    first_run: str = "—"
    peak_memory: str = "—"
    evidence: str = "Not measured"
    total_tooltip: str = "No complete timing was recorded."
    data_movement_tooltip: str = "No separate data-movement timing was recorded."
    evidence_tooltip: str = "No complete timing evidence was recorded."


def _enum_value(value: object) -> str:
    return str(getattr(value, "value", value)).strip().lower()


def _format_fraction_percent(value: object) -> str:
    percent = float(value) * 100.0
    rendered = f"{percent:.6f}".rstrip("0").rstrip(".")
    return f"{rendered or '0'}%"


def _parity_review_text(
    proposal: PipelineOptimizationProposal,
    node_titles: Mapping[str, str],
) -> str:
    deviations = proposal.reviewable_deviations
    lines = [
        "Measured CPU/GPU output difference needs your review. VIPP verified "
        "that the exact requested CPU and GPU backends ran before comparing "
        "their outputs."
    ]
    for deviation in deviations:
        node_id = deviation.node_id
        node_title = node_titles.get(node_id, node_id or "Workflow node")
        output_number = deviation.output_port_index + 1
        metric = _enum_value(deviation.metric)
        threshold = _format_fraction_percent(deviation.acceptance_threshold)
        if metric == "differing-value-fraction":
            differing = deviation.differing_values
            total = deviation.total_values
            differing_percent = _format_fraction_percent(deviation.differing_fraction)
            summary = (
                f"{differing_percent} "
                f"of values differed ({differing:,} of {total:,}); review limit "
                f"{threshold}; largest absolute change "
                f"{deviation.maximum_absolute_error:.6g}."
            )
        elif metric == "normalized-rmse":
            normalized_rmse = _format_fraction_percent(
                deviation.normalized_root_mean_square_error
            )
            normalized_maximum = _format_fraction_percent(
                deviation.normalized_maximum_absolute_error
            )
            summary = (
                f"normalized RMSE {normalized_rmse} (RMSE review limit "
                f"{threshold}); normalized maximum error {normalized_maximum}."
            )
        else:
            detail = deviation.detail
            summary = detail or "a bounded output difference was measured."
        lines.append(f"• {node_title} — output {output_number}: {summary}")
    lines.append(
        "Nothing changes until you select the acceptance box below and apply "
        "this result."
    )
    return "\n".join(lines)


def _candidate_refusal_text(
    refusals: tuple[object, ...],
    node_titles: Mapping[str, str],
) -> str:
    lines = [
        "VIPP skipped an unavailable backend and continued with the remaining "
        "safe choices."
    ]
    for refusal in refusals:
        node_id = str(getattr(refusal, "node_id", "")).strip()
        node_title = node_titles.get(node_id, node_id or "Pipeline")
        code = str(getattr(refusal, "code", "")).strip().lower()
        if "_planning_assignment_mismatch" in code:
            category = "planning identity problem"
        elif "_device_segment_mismatch" in code:
            category = "device-plan identity problem"
        elif "_actual_assignment_mismatch" in code:
            category = "execution identity problem"
        else:
            category = "backend unavailable"
        message = str(getattr(refusal, "message", "")).strip()
        lines.append(f"• {node_title} — {category}: {message}")
    return "\n".join(lines)


def _ordered_implementation_ids(
    row: object,
    tested_id: str,
    candidates: Mapping[str, object],
) -> tuple[str, ...]:
    ordered: list[str] = []
    for implementation_id in (
        getattr(row, "current_implementation_id", ""),
        tested_id,
        getattr(row, "proposed_implementation_id", ""),
        *candidates,
    ):
        normalized = str(implementation_id).strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return tuple(ordered)


def _is_gpu_implementation(implementation_id: str) -> bool:
    value = str(implementation_id).lower()
    return any(token in value for token in ("cuda", "cupy", "cupyx", "cucim"))


def _friendly_implementation_name(implementation_id: str) -> str:
    value = str(implementation_id).strip()
    lowered = value.lower()
    if "cucim" in lowered:
        return "GPU — cuCIM"
    if any(token in lowered for token in ("cupy", "cupyx", "cuda")):
        return "GPU — CuPy"
    if lowered.startswith("cpu"):
        return "CPU — built in"
    return value


def _evidence_source_text(
    node_id: str,
    *,
    measured: set[str],
    reused: set[str],
    has_record: bool,
) -> str:
    if not has_record:
        return "not measured"
    if node_id in reused:
        return "reused exact evidence"
    if node_id in measured:
        return "measured now"
    return "saved exact evidence"


def _scientific_check(candidate: object | None) -> tuple[str, str]:
    if candidate is None:
        return "Not measured", "No scientific comparison was required or recorded."
    error = str(getattr(candidate, "error", "") or "").strip()
    parity_passed = bool(getattr(candidate, "parity_passed", False))
    failure_kind = _enum_value(getattr(candidate, "failure_kind", ""))
    if failure_kind == "scientific-parity":
        return (
            "Did not match",
            error
            or "This implementation did not reproduce the reference scientific result.",
        )
    if failure_kind == "transient-runtime" or error:
        return "Could not verify", error
    if not parity_passed:
        return (
            "Did not match",
            "This implementation did not reproduce the reference scientific result.",
        )
    return "Matches", "This implementation reproduced the reference scientific result."


def _implementation_result_text(
    row: object,
    implementation_id: str,
    *,
    tested_id: str,
    winner: str,
    candidate: object | None,
) -> str:
    error = str(getattr(candidate, "error", "") or "").strip()
    parity_failed = candidate is not None and not bool(
        getattr(candidate, "parity_passed", False)
    )
    if error or parity_failed:
        return "Excluded"

    current_id = str(getattr(row, "current_implementation_id", ""))
    proposed_id = str(getattr(row, "proposed_implementation_id", ""))
    if bool(getattr(row, "locked", False)) and implementation_id == current_id:
        return "Locked"
    if not bool(getattr(row, "eligible", True)) and implementation_id == current_id:
        return "Fixed"

    censored = bool(getattr(candidate, "timing_censored", False))
    if censored and implementation_id != current_id:
        prefix = "Tested · " if implementation_id == tested_id else ""
        return prefix + "stopped early — already slower"

    if winner == "inconclusive":
        if implementation_id == current_id == tested_id:
            return "Current · compared"
        if implementation_id == current_id:
            return "Current · kept"
        if implementation_id == tested_id:
            return "Tested · no clear winner"
        return "Compared"
    if winner == "proposed":
        if implementation_id == proposed_id:
            return "Would use"
        if implementation_id == current_id:
            return "Current"
        if implementation_id == tested_id:
            return "Tested"
        return "Compared"
    if winner == "current":
        if implementation_id == current_id:
            return "Current · kept"
        if implementation_id == tested_id:
            return "Tested alternative"
        return "Compared"
    if implementation_id == current_id:
        return "Current · kept"
    if implementation_id == tested_id:
        return "Tested"
    return "Compared"


def _median(values: object) -> float | None:
    series = tuple(values or ())
    if not series:
        return None
    return float(statistics.median(float(value) for value in series))


def _format_optional_seconds(value: float | None) -> str:
    return "—" if value is None else _format_seconds(value)


def _format_bytes(value: int) -> str:
    amount = float(max(int(value), 0))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024.0 or unit == "TiB":
            return f"{amount:.0f} {unit}" if unit == "B" else f"{amount:.1f} {unit}"
        amount /= 1024.0
    return "0 B"


def _candidate_timing_display(
    candidate: object | None,
    *,
    evidence_source: str,
) -> _CandidateTimingDisplay:
    if candidate is None:
        return _CandidateTimingDisplay(evidence=evidence_source.capitalize())

    implementation_id = str(getattr(candidate, "implementation_id", ""))
    error = str(getattr(candidate, "error", "") or "").strip()
    parity_passed = bool(getattr(candidate, "parity_passed", False))
    valid = parity_passed and not error
    warm = tuple(getattr(candidate, "warm_seconds", ()) or ())
    resident = tuple(getattr(candidate, "warm_resident_seconds", ()) or ())
    transfer = tuple(getattr(candidate, "warm_transfer_seconds", ()) or ())
    host = tuple(getattr(candidate, "warm_host_materialization_seconds", ()) or ())
    transfers_included = bool(getattr(candidate, "transfers_included", False))
    timing_censored = bool(getattr(candidate, "timing_censored", False))
    lower_bound = getattr(candidate, "timing_lower_bound_seconds", None)

    warm_median = _median(warm) if valid else None
    if timing_censored and lower_bound is not None:
        total = f">{_format_seconds(float(lower_bound))}"
        total_tooltip = (
            "A safe lower bound only. Timing stopped once this implementation "
            "was already decisively slower."
        )
    elif warm_median is None:
        total = "—"
        total_tooltip = "No complete valid repeated timing was recorded."
    else:
        total = _format_seconds(warm_median)
        if _is_gpu_implementation(implementation_id) and not transfers_included:
            total += " (compute only)"
            total_tooltip = (
                "Median GPU implementation time after warm-up. Transfers were "
                "not included, so this is compute-only rather than an additive "
                "whole-pipeline total."
            )
        elif transfers_included:
            total_tooltip = (
                "Median measured call after warm-up, including transfers for "
                "this isolated node benchmark. Whole-pipeline transfers are "
                "modeled once across runtime boundaries."
            )
        else:
            total_tooltip = "Median measured call after warm-up."

    if valid:
        resident_median = _median(resident)
        compute_median = resident_median
        if compute_median is None and not transfers_included:
            compute_median = warm_median
    else:
        compute_median = None

    movement_values: tuple[float, ...] = ()
    if valid and (transfer or host):
        count = max(len(transfer), len(host))
        movement_values = tuple(
            (float(transfer[index]) if index < len(transfer) else 0.0)
            + (float(host[index]) if index < len(host) else 0.0)
            for index in range(count)
        )
    movement_median = _median(movement_values)
    movement_tooltip = (
        "Median data transfer plus host-output materialization time. The "
        "whole-pipeline model counts shared runtime-boundary transfers only once."
        if movement_median is not None
        else "No separate data-movement timing was recorded."
    )

    first_run = getattr(candidate, "cold_seconds", None) if valid else None
    peak_memory = int(getattr(candidate, "peak_memory_bytes", 0) or 0)
    if timing_censored:
        evidence_text = f"Lower bound · {evidence_source}"
    elif warm:
        count = len(warm)
        evidence_text = f"{count} round{'s' if count != 1 else ''} · {evidence_source}"
    elif error or not parity_passed:
        evidence_text = "Excluded"
    else:
        evidence_text = evidence_source.capitalize()
    censor_reason = str(getattr(candidate, "timing_censor_reason", "") or "").strip()
    evidence_tooltip = censor_reason or (
        f"Timing evidence was {evidence_source}."
        if valid
        else error or "No valid timing evidence was accepted."
    )
    return _CandidateTimingDisplay(
        total=total,
        compute=_format_optional_seconds(compute_median),
        data_movement=_format_optional_seconds(movement_median),
        first_run=_format_optional_seconds(
            None if first_run is None else float(first_run)
        ),
        peak_memory=_format_bytes(peak_memory),
        evidence=evidence_text,
        total_tooltip=total_tooltip,
        data_movement_tooltip=movement_tooltip,
        evidence_tooltip=evidence_tooltip,
    )


def _candidate_timing_text(evidence: object) -> str:
    record = getattr(evidence, "record", None)
    candidates = getattr(record, "candidates", ())
    values: list[str] = []
    for candidate in candidates:
        if bool(getattr(candidate, "timing_censored", False)):
            lower_bound = getattr(candidate, "timing_lower_bound_seconds", None)
            incumbent = str(
                getattr(candidate, "timing_censor_incumbent_id", "") or ""
            ).strip()
            reason = str(getattr(candidate, "timing_censor_reason", "") or "").strip()
            if lower_bound is not None:
                detail = (
                    f">{_format_seconds(float(lower_bound))} "
                    "(censored CPU/warm lower bound; stopped early"
                )
                if incumbent:
                    detail += f" versus {incumbent}"
                detail += ")"
                if reason:
                    detail += f" — {reason}"
                values.append(f"{candidate.implementation_id} {detail}")
                continue
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
    "PipelineOptimizerApplyRequest",
    "PipelineOptimizerDialog",
    "PipelineOptimizerCleanupError",
    "PipelineOptimizerProgress",
    "PipelineOptimizerWorker",
    "PipelineOptimizerWorkerOutcome",
]
