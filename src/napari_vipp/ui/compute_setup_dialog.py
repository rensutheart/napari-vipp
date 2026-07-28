"""Nonblocking Qt surface for optional GPU setup and memory diagnostics."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.compute_diagnostics import (
    ComputeDoctorReport,
    DoctorStatus,
    collect_compute_diagnostics,
)
from napari_vipp.ui.compute_setup import (
    ComputeSetupActionKind,
    ComputeSetupPresentation,
    ComputeSetupTone,
    HostMemorySnapshot,
    compute_setup_checking,
    compute_setup_not_checked,
    present_compute_setup,
)


@dataclass(frozen=True, slots=True)
class ComputeSetupCheckResult:
    """One stale-safe result emitted by the background doctor worker."""

    serial: int
    report: ComputeDoctorReport


class _ComputeSetupWorkerSignals(QObject):
    finished = Signal(object)


class ComputeSetupWorker(QRunnable):
    """Run the headless compute doctor without blocking the Qt thread."""

    def __init__(
        self,
        serial: int,
        *,
        track: str,
        refresh: bool,
        doctor: Callable[..., ComputeDoctorReport] = collect_compute_diagnostics,
    ) -> None:
        super().__init__()
        self.serial = int(serial)
        self.track = str(track)
        self.refresh = bool(refresh)
        self.doctor = doctor
        self.signals = _ComputeSetupWorkerSignals()

    def run(self) -> None:
        try:
            report = self.doctor(track=self.track, refresh=self.refresh)
            if not isinstance(report, ComputeDoctorReport):
                raise TypeError("compute doctor returned an invalid report")
        except Exception as exc:
            report = _failed_doctor_report(exc, track=self.track)
        self.signals.finished.emit(ComputeSetupCheckResult(self.serial, report))


class ComputeSetupDialog(QDialog):
    """Inspect optional acceleration and copy safe setup guidance."""

    presentation_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        thread_pool: QThreadPool | None = None,
        doctor: Callable[..., ComputeDoctorReport] = collect_compute_diagnostics,
        host_memory_provider: Callable[[], HostMemorySnapshot] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compute setup and memory")
        self.setMinimumWidth(560)
        self.setModal(False)
        self._thread_pool = thread_pool or QThreadPool.globalInstance()
        self._doctor = doctor
        self._host_memory_provider = host_memory_provider
        self._serial = 0
        self._active_serial: int | None = None
        self._last_report: ComputeDoctorReport | None = None
        self._presentation = compute_setup_not_checked(host_memory=self._host_memory())

        self.track_combo = QComboBox()
        self.track_combo.addItem("Automatic", "auto")
        self.track_combo.addItem("CUDA 13", "cuda13")
        self.track_combo.addItem("CUDA 12", "cuda12")
        self.track_combo.setToolTip(
            "Choose the CUDA package track to verify. Automatic uses the "
            "installed track or VIPP's current recommended track."
        )
        self.verify_button = QPushButton("Verify GPU setup")
        self.verify_button.setAccessibleName("Verify GPU compute setup")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.title_label = QLabel()
        self.title_label.setStyleSheet("font-size: 14px; font-weight: 650;")
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.memory_widget = QWidget()
        self.memory_form = QFormLayout(self.memory_widget)
        self.memory_form.setContentsMargins(0, 0, 0, 0)
        self.memory_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        self.command_edit.setAccessibleName("GPU setup command")
        self.command_edit.setPlaceholderText("No setup command is required.")
        self.copy_button = QPushButton("Copy setup command")
        self.copy_button.setAccessibleName("Copy GPU setup command")
        self.copy_button.setEnabled(False)
        self.close_buttons = QDialogButtonBox(QDialogButtonBox.Close)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Package track"))
        controls.addWidget(self.track_combo)
        controls.addWidget(self.verify_button)
        controls.addWidget(self.progress)
        controls.addStretch(1)
        command_row = QHBoxLayout()
        command_row.addWidget(self.command_edit, 1)
        command_row.addWidget(self.copy_button)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.title_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.details_label)
        layout.addWidget(self.memory_widget)
        layout.addLayout(command_row)
        layout.addWidget(self.close_buttons)

        self.verify_button.clicked.connect(self.verify)
        self.copy_button.clicked.connect(self.copy_setup_command)
        self.close_buttons.rejected.connect(self.close)
        self._apply_presentation(self._presentation)

    @property
    def presentation(self) -> ComputeSetupPresentation:
        return self._presentation

    @property
    def checking(self) -> bool:
        return self._active_serial is not None

    def verify(self) -> None:
        """Queue one refreshed diagnostic run; repeated clicks never overlap."""
        if self.checking:
            return
        self._serial += 1
        serial = self._serial
        self._active_serial = serial
        track = str(self.track_combo.currentData() or "auto")
        self._apply_presentation(
            compute_setup_checking(track=track, host_memory=self._host_memory())
        )
        worker = ComputeSetupWorker(
            serial,
            track=track,
            refresh=True,
            doctor=self._doctor,
        )
        worker.signals.finished.connect(self._on_check_finished)
        self._thread_pool.start(worker)

    def copy_setup_command(self) -> None:
        """Copy the displayed command; never execute it."""
        command = self.command_edit.text().strip()
        if not command:
            return
        QApplication.clipboard().setText(command)

    def _on_check_finished(self, result: ComputeSetupCheckResult) -> None:
        if result.serial != self._active_serial:
            return
        self._active_serial = None
        self._last_report = result.report
        self._apply_presentation(
            present_compute_setup(result.report, host_memory=self._host_memory())
        )

    def _apply_presentation(self, presentation: ComputeSetupPresentation) -> None:
        self._presentation = presentation
        self.title_label.setText(presentation.title)
        self.summary_label.setText(presentation.summary)
        self.summary_label.setStyleSheet(_summary_style(presentation.tone))
        self.details_label.setText("\n".join(presentation.details))
        self.details_label.setVisible(bool(presentation.details))
        self.progress.setVisible(presentation.busy)
        self.track_combo.setEnabled(not presentation.busy)

        verify_action = next(
            (
                action
                for action in presentation.actions
                if action.kind is ComputeSetupActionKind.VERIFY
            ),
            None,
        )
        self.verify_button.setText(
            verify_action.label if verify_action is not None else "Verify again"
        )
        self.verify_button.setEnabled(
            bool(verify_action is not None and verify_action.enabled)
        )
        copy_action = next(
            (
                action
                for action in presentation.actions
                if action.kind is ComputeSetupActionKind.COPY_COMMAND
            ),
            None,
        )
        command = copy_action.command if copy_action is not None else ""
        self.command_edit.setText(command)
        self.command_edit.setVisible(bool(command))
        self.copy_button.setVisible(bool(command))
        self.copy_button.setEnabled(bool(command))
        _replace_memory_rows(self.memory_form, presentation)
        self.presentation_changed.emit(presentation)

    def _host_memory(self) -> HostMemorySnapshot | None:
        if self._host_memory_provider is None:
            return None
        try:
            snapshot = self._host_memory_provider()
        except Exception:
            return None
        return snapshot if isinstance(snapshot, HostMemorySnapshot) else None


def _replace_memory_rows(
    form: QFormLayout,
    presentation: ComputeSetupPresentation,
) -> None:
    while form.rowCount():
        form.removeRow(0)
    for row in presentation.memory_rows:
        value = QLabel(row.value)
        value.setWordWrap(True)
        value.setToolTip(row.detail)
        form.addRow(row.label, value)


def _summary_style(tone: ComputeSetupTone) -> str:
    color = {
        ComputeSetupTone.NEUTRAL: "#cbd5e1",
        ComputeSetupTone.INFO: "#93c5fd",
        ComputeSetupTone.SUCCESS: "#86efac",
        ComputeSetupTone.WARNING: "#fde68a",
        ComputeSetupTone.ERROR: "#fca5a5",
    }[tone]
    return f"color: {color};"


def _failed_doctor_report(exc: Exception, *, track: str) -> ComputeDoctorReport:
    return ComputeDoctorReport(
        status=DoctorStatus.UNAVAILABLE,
        reason_code="diagnostic_worker_failed",
        summary=f"GPU setup verification failed: {type(exc).__name__}: {exc}",
        platform=sys.platform,
        execution_mode="native",
        python=(
            f"{platform.python_implementation()} "
            f"{sys.version_info.major}.{sys.version_info.minor}"
        ),
        packages=(),
        track=str(track).strip() or "auto",
    )


__all__ = [
    "ComputeSetupCheckResult",
    "ComputeSetupDialog",
    "ComputeSetupWorker",
]
