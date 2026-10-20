"""Nonblocking Qt surface for optional GPU setup and memory diagnostics."""

from __future__ import annotations

import platform
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from qtpy.QtCore import (
    QEvent,
    QObject,
    QRunnable,
    QSignalBlocker,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from qtpy.QtGui import QFontDatabase, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
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
    ComputeSetupState,
    ComputeSetupTone,
    HostMemorySnapshot,
    compute_setup_checking,
    compute_setup_not_checked,
    present_compute_setup,
)
from napari_vipp.ui.compute_setup_style import (
    SetupDisclosureButton as _DisclosureButton,
)
from napari_vipp.ui.compute_setup_style import setup_stylesheet
from napari_vipp.ui.dialog_buttons import DialogButtonBox as QDialogButtonBox
from napari_vipp.ui.palette_roles import ThemeColors, theme_colors
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon

_COMPUTE_TONE_PROPERTY = "vippComputeTone"


class _CommandPreview(QPlainTextEdit):
    """A bounded, wrapped command with the former read-only field API."""

    def text(self) -> str:
        return self.toPlainText()

    def setText(self, text: str) -> None:  # noqa: N802
        self.setPlainText(text)


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
        gpu_setup_factory: Callable[..., QDialog] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Compute setup and memory")
        self.setObjectName("ComputeSetupDialog")
        self.setMinimumSize(560, 380)
        self.setModal(False)
        self._thread_pool = thread_pool or QThreadPool.globalInstance()
        self._doctor = doctor
        self._host_memory_provider = host_memory_provider
        self._recent_execution_provider = recent_execution_provider
        self._support_writer = support_writer
        self._gpu_setup_factory = gpu_setup_factory
        self._gpu_setup_dialog: QDialog | None = None
        self._serial = 0
        self._active_serial: int | None = None
        self._last_report: ComputeDoctorReport | None = None
        self._device_selection_editable = True
        self._refreshing_tone_styles = False
        self._presentation = compute_setup_not_checked(host_memory=self._host_memory())
        self._content_fit_timer = QTimer(self)
        self._content_fit_timer.setSingleShot(True)
        self._content_fit_timer.timeout.connect(self._fit_content)

        self.device_combo = QComboBox()
        self.device_combo.setSizeAdjustPolicy(
            QComboBox.AdjustToMinimumContentsLengthWithIcon
        )
        self.device_combo.setMinimumContentsLength(24)
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
        self.verify_button = ToolbarCommandButton("Verify GPU setup")
        self.verify_button.setAccessibleName("Verify GPU compute setup")
        self.setup_button = ToolbarCommandButton("Set up GPU…")
        self.setup_button.setObjectName("ComputePrimaryAction")
        self.setup_button.setAccessibleName("Set up or repair GPU support")
        self.setup_button.setToolTip(
            "Review and install a separate GPU-enabled VIPP installation. "
            "Your current workflows stay open."
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setVisible(False)
        self.status_heading = QLabel()
        self.status_heading.setObjectName("ComputeStatusHeading")
        self.status_heading.setWordWrap(True)
        self.status_description = QLabel()
        self.status_description.setWordWrap(True)
        self.title_label = QLabel()
        self.title_label.setObjectName("ComputePlatform")
        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.next_step_label = QLabel()
        self.next_step_label.setWordWrap(True)
        self.check_widget = QWidget()
        self.check_widget.setObjectName("ComputeCardBody")
        self.check_form = QFormLayout(self.check_widget)
        self.check_form.setContentsMargins(0, 4, 0, 4)
        self.check_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.check_form.setHorizontalSpacing(20)
        self.check_form.setVerticalSpacing(8)
        self.advanced_button = _DisclosureButton("Advanced details")
        self.advanced_widget = QWidget()
        self.advanced_widget.setObjectName("ComputeCardBody")
        self.details_label = QLabel()
        self.details_label.setWordWrap(True)
        self.details_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.memory_widget = QWidget()
        self.memory_widget.setObjectName("ComputeCardBody")
        self.memory_form = QFormLayout(self.memory_widget)
        self.memory_form.setContentsMargins(0, 0, 0, 0)
        self.memory_form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.command_widget = QWidget()
        self.command_widget.setObjectName("ComputeCardBody")
        self.command_title = QLabel("Manual command")
        self.command_title.setObjectName("ComputeSectionTitle")
        self.command_help = QLabel()
        self.command_help.setWordWrap(True)
        self.command_edit = _CommandPreview()
        self.command_edit.setReadOnly(True)
        command_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        command_font.setFixedPitch(True)
        self.command_edit.setFont(command_font)
        self.command_edit.setFixedHeight(
            self.command_edit.fontMetrics().lineSpacing() * 4 + 22
        )
        self.command_edit.setAccessibleName("Manual GPU command (copy only)")
        self.command_title.setBuddy(self.command_edit)
        self.copy_button = ToolbarCommandButton("Copy command")
        self.copy_button.setAccessibleName("Copy GPU setup command")
        self.copy_button.setEnabled(False)
        self.export_button = ToolbarCommandButton("Save support report…")
        self.export_button.setAccessibleName("Save privacy-redacted support report")
        self.export_button.setToolTip(
            "Save a privacy-redacted report for troubleshooting."
        )
        self.export_button.setEnabled(False)
        self.save_status_label = QLabel()
        self.save_status_label.setWordWrap(True)
        self.save_status_label.hide()
        self.close_buttons = QDialogButtonBox(QDialogButtonBox.Close)

        heading = QLabel("GPU setup and memory")
        heading.setObjectName("ComputeWindowHeading")
        intro = QLabel("Check acceleration, choose a device, or set up GPU support.")
        intro.setWordWrap(True)
        intro.setObjectName("ComputeMuted")

        self.session_widget = QFrame()
        self.session_widget.setObjectName("ComputeCard")
        controls = QVBoxLayout(self.session_widget)
        controls.setContentsMargins(16, 14, 16, 14)
        controls.setSpacing(8)
        device_label = QLabel("GPU for this workflow session")
        device_label.setObjectName("ComputeSectionTitle")
        device_label.setBuddy(self.device_combo)
        controls.addWidget(device_label)
        device_row = QHBoxLayout()
        device_row.setSpacing(10)
        device_row.addWidget(self.device_combo, 1)
        device_row.addWidget(self.verify_button)
        controls.addLayout(device_row)
        device_help = QLabel("Automatic lets VIPP choose the runtime’s default device.")
        device_help.setWordWrap(True)
        device_help.setObjectName("ComputeMuted")
        controls.addWidget(device_help)

        track_label = QLabel("Package track")
        track_label.setBuddy(self.track_combo)
        track_help = QLabel(
            "Choose which CUDA package track to verify. This does not install "
            "packages or change the GPU setup wizard’s recommended track."
        )
        track_help.setWordWrap(True)
        track_help.setObjectName("ComputeMuted")
        self.progress.setFixedHeight(4)
        command_layout = QVBoxLayout(self.command_widget)
        command_layout.setContentsMargins(0, 6, 0, 0)
        command_layout.setSpacing(8)
        command_layout.addWidget(self.command_title)
        command_layout.addWidget(self.command_help)
        command_layout.addWidget(self.command_edit)
        command_layout.addWidget(self.copy_button, 0, Qt.AlignLeft)
        advanced_layout = QVBoxLayout(self.advanced_widget)
        advanced_layout.setContentsMargins(16, 2, 16, 16)
        advanced_layout.setSpacing(10)
        advanced_layout.addWidget(track_label)
        advanced_layout.addWidget(self.track_combo)
        advanced_layout.addWidget(track_help)
        diagnostic_title = QLabel("Verification details")
        diagnostic_title.setObjectName("ComputeSectionTitle")
        advanced_layout.addWidget(diagnostic_title)
        advanced_layout.addWidget(self.title_label)
        advanced_layout.addWidget(self.summary_label)
        advanced_layout.addWidget(self.details_label)
        advanced_layout.addWidget(self.memory_widget)
        advanced_layout.addWidget(self.command_widget)

        self.status_widget = QFrame()
        self.status_widget.setObjectName("ComputeStatusCard")
        status_layout = QVBoxLayout(self.status_widget)
        status_layout.setContentsMargins(18, 16, 18, 16)
        status_layout.setSpacing(10)
        status_layout.addWidget(self.status_heading)
        status_layout.addWidget(self.status_description)
        status_layout.addWidget(self.check_widget)
        self.action_widget = QWidget()
        self.action_widget.setObjectName("ComputeCardBody")
        action_layout = QVBoxLayout(self.action_widget)
        action_layout.setContentsMargins(0, 8, 0, 0)
        action_layout.setSpacing(10)
        action_layout.addWidget(self.next_step_label)
        action_layout.addWidget(self.setup_button, 0, Qt.AlignLeft)
        status_layout.addWidget(self.action_widget)

        for label in (
            self.status_heading,
            self.status_description,
            self.title_label,
            self.summary_label,
            self.next_step_label,
            self.details_label,
            self.save_status_label,
        ):
            label.setTextFormat(Qt.PlainText)
            label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
            policy = label.sizePolicy()
            policy.setVerticalPolicy(QSizePolicy.Preferred)
            label.setSizePolicy(policy)
        for widget in (self.status_widget, self.advanced_widget):
            policy = widget.sizePolicy()
            policy.setHeightForWidth(True)
            widget.setSizePolicy(policy)

        advanced_card = QFrame()
        advanced_card.setObjectName("ComputeCard")
        disclosure_layout = QVBoxLayout(advanced_card)
        disclosure_layout.setContentsMargins(0, 0, 0, 0)
        disclosure_layout.setSpacing(0)
        disclosure_layout.addWidget(self.advanced_button)
        disclosure_layout.addWidget(self.advanced_widget)
        content = QWidget()
        content.setObjectName("ComputeContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 18, 20, 18)
        content_layout.setSpacing(14)
        content_layout.addWidget(heading)
        content_layout.addWidget(intro)
        content_layout.addWidget(self.progress)
        content_layout.addWidget(self.status_widget)
        content_layout.addWidget(self.session_widget)
        content_layout.addWidget(advanced_card)
        content_layout.addWidget(self.save_status_label)
        content_layout.addStretch(1)
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(content)

        self.footer_widget = QFrame()
        self.footer_widget.setObjectName("ComputeFooter")
        footer = QHBoxLayout(self.footer_widget)
        footer.setContentsMargins(20, 12, 20, 12)
        footer.addWidget(self.export_button)
        footer.addStretch(1)
        footer.addWidget(self.close_buttons)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(self.footer_widget)

        self.verify_button.clicked.connect(self.verify)
        self.setup_button.clicked.connect(self.open_gpu_setup)
        self.device_combo.currentIndexChanged.connect(self._on_device_selection_changed)
        self.copy_button.clicked.connect(self.copy_setup_command)
        self.export_button.clicked.connect(self.save_support_report)
        self.advanced_button.toggled.connect(self._set_advanced_visible)
        self.close_buttons.rejected.connect(self.close)
        self._apply_presentation(self._presentation)
        self._set_advanced_visible(False)
        self._refresh_surface_style()
        self.resize(720, 660)

    @property
    def presentation(self) -> ComputeSetupPresentation:
        return self._presentation

    @property
    def checking(self) -> bool:
        return self._active_serial is not None

    @property
    def gpu_setup_running(self) -> bool:
        """Keep terminal app shutdown separate from ordinary tab switching."""

        return bool(
            self._gpu_setup_dialog is not None
            and getattr(self._gpu_setup_dialog, "running", False)
        )

    def request_gpu_setup_cancel(self) -> None:
        """Ask the installer to cancel and finish its rollback safely."""

        if self._gpu_setup_dialog is not None:
            self._gpu_setup_dialog.cancel_setup()

    def open_gpu_setup(self) -> None:
        """Open one review window; never execute the copied shell command."""

        if self.checking:
            return
        if not any(
            action.kind is ComputeSetupActionKind.SETUP_GPU and action.enabled
            for action in self._presentation.actions
        ):
            return
        if self._gpu_setup_dialog is None:
            if self._gpu_setup_factory is None:
                from napari_vipp.ui.gpu_setup_dialog import GpuSetupDialog

                factory = GpuSetupDialog
            else:
                factory = self._gpu_setup_factory
            try:
                self._gpu_setup_dialog = factory(parent=self)
            except Exception as exc:
                self.save_status_label.setText(
                    f"GPU setup could not open: {type(exc).__name__}: {exc}"
                )
                self.save_status_label.show()
                return
        self._gpu_setup_dialog.show()
        self._gpu_setup_dialog.raise_()
        self._gpu_setup_dialog.activateWindow()

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
            self.save_status_label.show()
            _set_tone_style(
                self.save_status_label,
                ComputeSetupTone.WARNING,
                _effective_palette(self),
            )
            return
        self.save_status_label.setText(
            f"Saved privacy-redacted support report: {target.name}"
        )
        self.save_status_label.show()
        _set_tone_style(
            self.save_status_label,
            ComputeSetupTone.SUCCESS,
            _effective_palette(self),
        )

    def _on_check_finished(self, result: ComputeSetupCheckResult) -> None:
        if result.serial != self._active_serial:
            return
        self._active_serial = None
        self._last_report = result.report
        self._apply_presentation(
            present_compute_setup(result.report, host_memory=self._host_memory())
        )

    def changeEvent(self, event) -> None:  # noqa: N802
        """Refresh semantic text after napari changes its live theme."""

        super().changeEvent(event)
        if event.type() not in (QEvent.PaletteChange, QEvent.StyleChange):
            return
        if getattr(self, "_refreshing_tone_styles", False):
            return
        self._refreshing_tone_styles = True
        try:
            if hasattr(self, "status_widget"):
                self._refresh_surface_style()
            _refresh_tone_labels(self)
        finally:
            self._refreshing_tone_styles = False

    def event(self, event) -> bool:
        result = super().event(event)
        # Qt resolves wrapped labels after showing/changing their text. Fit on
        # the next event turn, including newly visible support/error messages.
        if event.type() == QEvent.LayoutRequest and hasattr(self, "_content_fit_timer"):
            self._content_fit_timer.start(0)
        return result

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if event.size().width() != event.oldSize().width() and hasattr(
            self, "_content_fit_timer"
        ):
            self._content_fit_timer.start(0)

    def _on_device_selection_changed(self, index: int) -> None:
        if index < 0:
            return
        option = self.device_combo.itemData(index)
        if isinstance(option, ComputeDeviceOption):
            self.device_selection_changed.emit(option)

    def _apply_presentation(self, presentation: ComputeSetupPresentation) -> None:
        self._presentation = presentation
        self.status_heading.setText(
            {
                ComputeSetupState.NOT_CHECKED: "Ready to check your GPU",
                ComputeSetupState.CHECKING: "Checking GPU support…",
                ComputeSetupState.AVAILABLE: "GPU support is ready",
                ComputeSetupState.DEGRADED: "GPU support needs attention",
                ComputeSetupState.UNAVAILABLE: "GPU support is not ready",
                ComputeSetupState.UNSUPPORTED: "CPU processing is available",
                ComputeSetupState.MISCONFIGURED: "GPU setup needs repair",
            }[presentation.state]
        )
        self.status_description.setText(
            "GPU components are missing in this VIPP environment. "
            "You can keep working on the CPU."
            if presentation.reason_code == "cupy_missing"
            else {
                ComputeSetupState.NOT_CHECKED: (
                    "Verify GPU setup to see what is available. "
                    "Nothing is installed by this check."
                ),
                ComputeSetupState.CHECKING: (
                    "Testing packages and hardware in the background. "
                    "VIPP stays responsive."
                ),
                ComputeSetupState.AVAILABLE: (
                    "VIPP can use GPU acceleration for supported operations. "
                    "Other operations still use the CPU."
                ),
                ComputeSetupState.DEGRADED: (
                    "Some GPU capabilities are not ready. "
                    "Review the checks and next step below."
                ),
                ComputeSetupState.UNAVAILABLE: (
                    "This environment could not start GPU acceleration. "
                    "CPU processing remains available."
                ),
                ComputeSetupState.UNSUPPORTED: presentation.summary,
                ComputeSetupState.MISCONFIGURED: (
                    "The GPU components do not match this environment. "
                    "Review the next step below."
                ),
            }[presentation.state]
        )
        self.title_label.setText(presentation.title)
        self.summary_label.setText(presentation.summary)
        _set_tone_style(
            self.summary_label,
            presentation.tone,
            _effective_palette(self),
        )
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
        setup_action = next(
            (
                action
                for action in presentation.actions
                if action.kind is ComputeSetupActionKind.SETUP_GPU
            ),
            None,
        )
        self.setup_button.setVisible(setup_action is not None)
        self.setup_button.setEnabled(
            bool(setup_action is not None and setup_action.enabled)
        )
        if setup_action is not None:
            self.setup_button.setText(setup_action.label)
            self.next_step_label.setText(
                "Review a separate GPU-enabled installation. Your work stays open; "
                "nothing is installed until you confirm in the setup wizard."
            )
        self.action_widget.setVisible(bool(setup_action or presentation.next_step))
        if presentation.reason_code == "diagnostic_worker_failed":
            self.status_heading.setText("GPU check could not finish")
            self.status_description.setText(
                "Verification stopped unexpectedly. Choose Verify GPU setup to "
                "try again. If it keeps failing, save a support report; "
                "the error is under Advanced details."
            )
            self.action_widget.hide()
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
        self.command_widget.setVisible(bool(command))
        self.command_edit.setVisible(bool(command))
        self.copy_button.setVisible(bool(command))
        self.copy_button.setEnabled(bool(command))
        manual_setup = bool(
            re.search(r"\bpip(?:3)?\s+install\b|(?:^|\s)-m\s+venv(?:\s|$)", command)
        )
        verification_only = not manual_setup and (
            "napari_vipp.core.compute_diagnostics" in command
            or "vipp-compute-doctor" in command
        )
        if manual_setup:
            self.command_help.setText(
                "For manual setup in a terminal: this command creates or updates "
                "GPU components and may also verify them. The Python path identifies "
                "the environment it targets. Prefer the setup wizard above when "
                "available. Copying does not run or install anything."
            )
            self.copy_button.setText("Copy setup command")
        elif verification_only:
            self.command_help.setText(
                "For manual verification in a terminal: this checks GPU support; "
                "it does not install it. The Python path identifies the environment "
                "being checked. Copying does not run anything."
            )
            self.copy_button.setText("Copy verification command")
        else:
            self.command_help.setText(
                "Optional troubleshooting command supplied by the diagnostic report. "
                "Review it before using it in a terminal. The Python path, if shown, "
                "identifies the target environment. Copying does not run anything."
            )
            self.copy_button.setText("Copy command")
        self.copy_button.setAccessibleName(self.copy_button.text())
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
        _replace_check_rows(
            self.check_form,
            presentation,
            palette=_effective_palette(self),
        )
        _replace_memory_rows(self.memory_form, presentation)
        self._refresh_surface_style()
        self._fit_content()
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
        self._fit_content()

    def _fit_content(self, *, shrink: bool = False) -> None:
        """Relayout within the viewport; never grow beyond the user's screen."""

        del shrink
        self.layout().activate()

    def _refresh_surface_style(self) -> None:
        stylesheet = setup_stylesheet(
            _effective_palette(self),
            dialog_name=self.objectName(),
            tone=self._presentation.tone.value,
        )
        # Keep styles off the dialog itself: Qt otherwise replaces an explicit
        # caller palette with the stylesheet's old resolved palette on switches.
        self.scroll_area.setStyleSheet(stylesheet)
        self.footer_widget.setStyleSheet(stylesheet)
        self.advanced_button.refresh_disclosure()
        self.verify_button.setIcon(toolbar_icon("refresh", _effective_palette(self)))
        # The primary icon needs the same high contrast as its label.
        primary_palette = QPalette(_effective_palette(self))
        primary_palette.setColor(QPalette.ButtonText, Qt.white)
        self.setup_button.setIcon(toolbar_icon("compute", primary_palette))
        self.export_button.setIcon(toolbar_icon("save", _effective_palette(self)))

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
    *,
    palette: QPalette,
) -> None:
    while form.rowCount():
        form.removeRow(0)
    for row in presentation.check_rows:
        value = QLabel(row.value)
        value.setWordWrap(True)
        value.setToolTip(row.detail)
        _set_tone_style(value, row.tone, palette)
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


def _effective_palette(widget: QWidget) -> QPalette:
    parent = widget.parentWidget()
    return QWidget.palette(parent) if parent is not None else QWidget.palette(widget)


def _tone_color(tone: ComputeSetupTone, colors: ThemeColors):
    return {
        ComputeSetupTone.NEUTRAL: colors.muted_text,
        ComputeSetupTone.INFO: colors.info.foreground,
        ComputeSetupTone.SUCCESS: colors.success.foreground,
        ComputeSetupTone.WARNING: colors.warning.foreground,
        ComputeSetupTone.ERROR: colors.error.foreground,
    }[tone]


def _summary_style(tone: ComputeSetupTone, palette: QPalette) -> str:
    return f"color: {_tone_color(tone, theme_colors(palette)).name()};"


def _set_tone_style(
    label: QLabel,
    tone: ComputeSetupTone,
    palette: QPalette,
) -> None:
    label.setProperty(_COMPUTE_TONE_PROPERTY, tone.value)
    label.setStyleSheet(_summary_style(tone, palette))


def _refresh_tone_labels(widget: QWidget) -> None:
    palette = _effective_palette(widget)
    for label in widget.findChildren(QLabel):
        raw_tone = label.property(_COMPUTE_TONE_PROPERTY)
        try:
            tone = ComputeSetupTone(str(raw_tone))
        except ValueError:
            continue
        _set_tone_style(label, tone, palette)


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
