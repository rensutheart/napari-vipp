"""Nonblocking Qt surface for optional GPU setup and memory diagnostics."""

from __future__ import annotations

import platform
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import QObject, QRunnable, QSignalBlocker, Qt, QThreadPool, Signal
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.compute import ExecutionReport
from napari_vipp.core.compute_diagnostics import (
    ComputeDoctorReport,
    DoctorStatus,
    collect_compute_diagnostics,
    write_compute_support_bundle,
)
from napari_vipp.ui.compute_setup import (
    ComputeDeviceOption,
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
    device_selection_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        thread_pool: QThreadPool | None = None,
        doctor: Callable[..., ComputeDoctorReport] = collect_compute_diagnostics,
        host_memory_provider: Callable[[], HostMemorySnapshot] | None = None,
        recent_execution_provider: Callable[[], ExecutionReport | None] | None = None,
        support_writer: Callable[..., Path] = write_compute_support_bundle,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compute setup and memory")
        self.setMinimumWidth(560)
        self.setModal(False)
        self._thread_pool = thread_pool or QThreadPool.globalInstance()
        self._doctor = doctor
        self._host_memory_provider = host_memory_provider
        self._recent_execution_provider = recent_execution_provider
        self._support_writer = support_writer
        self._serial = 0
        self._active_serial: int | None = None
        self._last_report: ComputeDoctorReport | None = None
        self._device_selection_editable = True
        self._presentation = compute_setup_not_checked(host_memory=self._host_memory())

        self.device_combo = QComboBox()
        self.device_combo.setAccessibleName("Compute device")
        self.device_combo.setToolTip(
            "Choose Automatic or one exact accelerator for this workflow session."
        )
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
        self.next_step_label = QLabel()
        self.next_step_label.setWordWrap(True)
        self.next_step_label.setStyleSheet("font-weight: 600;")
        self.check_widget = QWidget()
        self.check_form = QFormLayout(self.check_widget)
        self.check_form.setContentsMargins(0, 4, 0, 4)
        self.check_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.advanced_button = QPushButton("Show advanced details")
        self.advanced_button.setCheckable(True)
        self.advanced_button.setChecked(False)
        self.advanced_widget = QWidget()
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
        self.export_button = QPushButton("Save privacy-redacted support report…")
        self.export_button.setAccessibleName("Save privacy-redacted support report")
        self.export_button.setEnabled(False)
        self.save_status_label = QLabel()
        self.save_status_label.setWordWrap(True)
        self.close_buttons = QDialogButtonBox(QDialogButtonBox.Close)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("GPU for this workflow session"))
        controls.addWidget(self.device_combo)
        controls.addWidget(QLabel("Package track"))
        controls.addWidget(self.track_combo)
        controls.addWidget(self.verify_button)
        controls.addWidget(self.progress)
        controls.addStretch(1)
        command_row = QHBoxLayout()
        command_row.addWidget(self.command_edit, 1)
        command_row.addWidget(self.copy_button)
        advanced_layout = QVBoxLayout(self.advanced_widget)
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.addWidget(self.details_label)
        advanced_layout.addWidget(self.memory_widget)
        advanced_layout.addLayout(command_row)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.title_label)
        layout.addWidget(self.summary_label)
        layout.addWidget(self.next_step_label)
        layout.addWidget(self.check_widget)
        layout.addWidget(self.advanced_button)
        layout.addWidget(self.advanced_widget)
        layout.addWidget(self.export_button)
        layout.addWidget(self.save_status_label)
        layout.addWidget(self.close_buttons)

        self.verify_button.clicked.connect(self.verify)
        self.device_combo.currentIndexChanged.connect(self._on_device_selection_changed)
        self.copy_button.clicked.connect(self.copy_setup_command)
        self.export_button.clicked.connect(self.save_support_report)
        self.advanced_button.toggled.connect(self._set_advanced_visible)
        self.close_buttons.rejected.connect(self.close)
        self._apply_presentation(self._presentation)
        self._set_advanced_visible(False)

    @property
    def presentation(self) -> ComputeSetupPresentation:
        return self._presentation

    @property
    def checking(self) -> bool:
        return self._active_serial is not None

    @property
    def device_selection(self) -> ComputeDeviceOption:
        """Return the exact Automatic or explicit device selection."""

        option = self.device_combo.currentData()
        if isinstance(option, ComputeDeviceOption):
            return option
        return ComputeDeviceOption("", "", "Automatic (runtime default)")

    def set_device_selection(
        self,
        runtime_id: str = "",
        device_id: str = "",
        display_name: str = "",
    ) -> None:
        """Select exact IDs without emitting a user-selection signal.

        An explicit device that is absent from the latest verification result is
        retained as an unavailable choice instead of being silently retargeted.
        """

        runtime_id = str(runtime_id).strip()
        device_id = str(device_id).strip()
        display_name = str(display_name).strip()
        if bool(runtime_id) != bool(device_id):
            raise ValueError(
                "runtime_id and device_id must either both be set or both be blank."
            )
        target_index = _find_device_option_index(
            self.device_combo,
            runtime_id=runtime_id,
            device_id=device_id,
        )
        blocker = QSignalBlocker(self.device_combo)
        try:
            if target_index < 0:
                option = ComputeDeviceOption(
                    runtime_id=runtime_id,
                    device_id=device_id,
                    display_name=display_name or device_id,
                    available=False,
                )
                self.device_combo.addItem(_device_option_label(option), option)
                target_index = self.device_combo.count() - 1
                _set_device_option_tooltip(
                    self.device_combo,
                    target_index,
                    option,
                )
            self.device_combo.setCurrentIndex(target_index)
        finally:
            del blocker

    def set_device_selection_editable(self, enabled: bool) -> None:
        """Enable or disable user device changes outside verification."""

        if not isinstance(enabled, bool):
            raise TypeError("enabled must be a boolean.")
        self._device_selection_editable = enabled
        self.device_combo.setEnabled(enabled and not self._presentation.busy)

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

    def save_support_report(self) -> None:
        """Ask for a target and atomically save the redacted support document."""

        report = self._last_report
        if report is None:
            return
        selected, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Save privacy-redacted support report",
            "vipp-compute-support.json",
            "JSON files (*.json)",
        )
        if not selected:
            return
        target = Path(selected)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        try:
            self._support_writer(
                target,
                report,
                recent_execution=self._recent_execution(),
            )
        except Exception as exc:
            self.save_status_label.setText(
                f"The support report could not be saved: {type(exc).__name__}: {exc}"
            )
            self.save_status_label.setStyleSheet(
                _summary_style(ComputeSetupTone.WARNING)
            )
            return
        self.save_status_label.setText(
            f"Saved privacy-redacted support report: {target.name}"
        )
        self.save_status_label.setStyleSheet(_summary_style(ComputeSetupTone.SUCCESS))

    def _on_check_finished(self, result: ComputeSetupCheckResult) -> None:
        if result.serial != self._active_serial:
            return
        self._active_serial = None
        self._last_report = result.report
        self._apply_presentation(
            present_compute_setup(result.report, host_memory=self._host_memory())
        )

    def _on_device_selection_changed(self, index: int) -> None:
        if index < 0:
            return
        option = self.device_combo.itemData(index)
        if isinstance(option, ComputeDeviceOption):
            self.device_selection_changed.emit(option)

    def _apply_presentation(self, presentation: ComputeSetupPresentation) -> None:
        self._presentation = presentation
        self.title_label.setText(presentation.title)
        self.summary_label.setText(presentation.summary)
        self.summary_label.setStyleSheet(_summary_style(presentation.tone))
        self.next_step_label.setText(presentation.next_step)
        self.next_step_label.setVisible(bool(presentation.next_step))
        self.details_label.setText("\n".join(presentation.details))
        self.details_label.setVisible(bool(presentation.details))
        self.progress.setVisible(presentation.busy)
        self.track_combo.setEnabled(not presentation.busy)
        self._replace_device_options(presentation.device_options)
        self.device_combo.setEnabled(
            self._device_selection_editable and not presentation.busy
        )

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
        export_action = next(
            (
                action
                for action in presentation.actions
                if action.kind is ComputeSetupActionKind.EXPORT_SUPPORT
            ),
            None,
        )
        self.export_button.setVisible(export_action is not None)
        self.export_button.setEnabled(
            bool(export_action is not None and export_action.enabled)
        )
        _replace_check_rows(self.check_form, presentation)
        _replace_memory_rows(self.memory_form, presentation)
        self.presentation_changed.emit(presentation)

    def _replace_device_options(
        self,
        reported_options: tuple[ComputeDeviceOption, ...],
    ) -> None:
        current = self.device_selection
        automatic = ComputeDeviceOption("", "", "Automatic (runtime default)")
        options = list(reported_options)
        automatic_index = next(
            (
                index
                for index, option in enumerate(options)
                if not option.runtime_id and not option.device_id
            ),
            -1,
        )
        if automatic_index < 0:
            options.insert(0, automatic)
        elif automatic_index:
            options.insert(0, options.pop(automatic_index))

        current_key = (current.runtime_id, current.device_id)
        option_keys = {(option.runtime_id, option.device_id) for option in options}
        if current.runtime_id and current_key not in option_keys:
            options.append(
                ComputeDeviceOption(
                    runtime_id=current.runtime_id,
                    device_id=current.device_id,
                    display_name=current.display_name,
                    total_memory_bytes=current.total_memory_bytes,
                    available=False,
                )
            )

        blocker = QSignalBlocker(self.device_combo)
        try:
            self.device_combo.clear()
            selected_index = 0
            for index, option in enumerate(options):
                self.device_combo.addItem(_device_option_label(option), option)
                _set_device_option_tooltip(self.device_combo, index, option)
                if (option.runtime_id, option.device_id) == current_key:
                    selected_index = index
            self.device_combo.setCurrentIndex(selected_index)
        finally:
            del blocker

    def _set_advanced_visible(self, visible: bool) -> None:
        self.advanced_widget.setVisible(bool(visible))
        self.advanced_button.setText(
            "Hide advanced details" if visible else "Show advanced details"
        )

    def _host_memory(self) -> HostMemorySnapshot | None:
        if self._host_memory_provider is None:
            return None
        try:
            snapshot = self._host_memory_provider()
        except Exception:
            return None
        return snapshot if isinstance(snapshot, HostMemorySnapshot) else None

    def _recent_execution(self) -> ExecutionReport | None:
        if self._recent_execution_provider is None:
            return None
        try:
            report = self._recent_execution_provider()
        except Exception:
            return None
        return report if isinstance(report, ExecutionReport) else None


def _replace_check_rows(
    form: QFormLayout,
    presentation: ComputeSetupPresentation,
) -> None:
    while form.rowCount():
        form.removeRow(0)
    for row in presentation.check_rows:
        value = QLabel(row.value)
        value.setWordWrap(True)
        value.setToolTip(row.detail)
        value.setStyleSheet(_summary_style(row.tone))
        form.addRow(row.label, value)


def _find_device_option_index(
    combo: QComboBox,
    *,
    runtime_id: str,
    device_id: str,
) -> int:
    for index in range(combo.count()):
        option = combo.itemData(index)
        if isinstance(option, ComputeDeviceOption) and (
            option.runtime_id,
            option.device_id,
        ) == (runtime_id, device_id):
            return index
    return -1


def _device_option_label(option: ComputeDeviceOption) -> str:
    if not option.runtime_id:
        return option.display_name
    label = f"{option.display_name} ({option.device_id})"
    return label if option.available else f"{label} — Unavailable"


def _set_device_option_tooltip(
    combo: QComboBox,
    index: int,
    option: ComputeDeviceOption,
) -> None:
    if not option.runtime_id:
        tooltip = "Let the runtime choose its default device."
    elif option.available:
        tooltip = f"{option.runtime_id} · {option.device_id}"
    else:
        tooltip = (
            f"{option.runtime_id} · {option.device_id} is not available in the "
            "latest verification result."
        )
    combo.setItemData(index, tooltip, Qt.ToolTipRole)


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
