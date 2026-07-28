"""Review-first Qt surface for whole-pipeline compute optimization."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
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

    analyze_requested = Signal(bool)
    apply_requested = Signal(object)
    optimizer_finished = Signal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optimize pipeline compute")
        self.setWindowModality(Qt.WindowModal)
        self.setMinimumSize(860, 480)
        self._worker: PipelineOptimizerWorker | None = None
        self._outcome: PipelineOptimizerWorkerOutcome | None = None
        self._running = False

        self.summary_label = QLabel(
            "Measure the exact current pipeline and propose one globally faster "
            "CPU/GPU assignment. Nothing changes until you review and apply it."
        )
        self.summary_label.setWordWrap(True)
        self.override_authored_checkbox = QCheckBox(
            "Allow optimizer to replace explicit per-node choices"
        )
        self.override_authored_checkbox.setChecked(False)
        self.override_authored_checkbox.setToolTip(
            "Off preserves CPU, GPU-library, and exact implementation choices. "
            "Turn this on only when you want the measured proposal to replace them."
        )
        self.progress_label = QLabel(
            "Ready. Authored per-node choices will be preserved."
        )
        self.progress_label.setWordWrap(True)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_table = QTableWidget(0, 5)
        self.result_table.setHorizontalHeaderLabels(
            ("Node", "Current", "Proposed", "Applied preference", "Status")
        )
        self.result_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.result_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.result_table.setVisible(False)
        self.result_table.horizontalHeader().setStretchLastSection(True)

        self.analyze_button = QPushButton("Analyze pipeline")
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
        layout.addWidget(self.override_authored_checkbox)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.result_label)
        layout.addWidget(self.result_table, 1)
        layout.addLayout(buttons)

        self.analyze_button.clicked.connect(self._request_analysis)
        self.apply_button.clicked.connect(self._apply_result)
        self.cancel_button.clicked.connect(self.cancel)
        self.close_button.clicked.connect(self.accept)
        self.override_authored_checkbox.toggled.connect(
            self._on_override_scope_changed
        )

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
        self.override_authored_checkbox.setEnabled(False)
        self.result_table.setVisible(False)
        self.result_label.setText("")
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText(
            "Capturing exact pipeline evidence. You can cancel at any time."
        )
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_finished)
        thread_pool.start(worker)

    def cancel(self) -> None:
        if not self._running or self._worker is None:
            return
        self._worker.cancel()
        self.cancel_button.setEnabled(False)
        self.progress_label.setText(
            "Cancel requested; finishing the current synchronized call…"
        )

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
        if self._running:
            return
        self.analyze_requested.emit(self.override_authored_checkbox.isChecked())

    def _on_override_scope_changed(self, _checked: bool) -> None:
        """Require fresh evidence when the optimizer's constraint scope changes."""

        if self._running or self._outcome is None or self._outcome.result is None:
            return
        self._outcome = None
        self.apply_button.setEnabled(False)
        self.result_table.setVisible(False)
        self.result_label.setStyleSheet("")
        self.result_label.setText(
            "The override scope changed. Analyze the pipeline again before "
            "applying a measured assignment."
        )
        self.progress_label.setText("Ready for a fresh pipeline analysis.")
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)

    def _on_progress(self, progress: PipelineOptimizerProgress) -> None:
        self.progress_bar.setRange(0, progress.total)
        self.progress_bar.setValue(progress.completed)
        self.progress_label.setText(progress.message)

    def _on_finished(self, outcome: PipelineOptimizerWorkerOutcome) -> None:
        self._running = False
        self._outcome = outcome
        self._worker = None
        self.analyze_button.setEnabled(True)
        self.close_button.setEnabled(True)
        self.cancel_button.setVisible(False)
        self.override_authored_checkbox.setEnabled(True)
        if outcome.result is not None:
            self.progress_bar.setValue(self.progress_bar.maximum())
            self.progress_label.setText("Pipeline analysis complete.")
            self._render_result(_proposal_from_result(outcome.result))
            self.apply_button.setEnabled(True)
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

    def _render_result(self, proposal: PipelineOptimizationProposal) -> None:
        rows = tuple(proposal.rows)
        self.result_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            preference = row.proposed_preference.kind.value
            if row.proposed_preference.value:
                preference += f":{row.proposed_preference.value}"
            status = "Change" if row.changed else "Keep"
            if not row.eligible:
                status = "Fixed / excluded"
            values = (
                row.node_id,
                row.current_implementation_id,
                row.proposed_implementation_id,
                preference,
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
        current = proposal.validated_current_seconds
        proposed = proposal.validated_proposed_seconds
        speedup = current / proposed if proposed > 0 else float("inf")
        changed = sum(1 for row in rows if row.changed)
        self.result_label.setStyleSheet("color: #86efac; font-weight: 650;")
        self.result_label.setText(
            f"Validated proposal: {_format_seconds(current)} → "
            f"{_format_seconds(proposed)} ({speedup:.2f}×; lower confidence "
            f"bound {proposal.validated_speedup_lower_confidence_bound:.2f}×). "
            f"{changed} node preference(s) would change."
        )

    def _apply_result(self) -> None:
        if self._outcome is None or self._outcome.result is None:
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


__all__ = [
    "PipelineOptimizerDialog",
    "PipelineOptimizerProgress",
    "PipelineOptimizerWorker",
    "PipelineOptimizerWorkerOutcome",
]
