"""Qt review and progress for a separate, verified GPU installation."""

from __future__ import annotations

from collections.abc import Callable

from qtpy.QtCore import QEvent, QObject, Qt, QTimer, Signal
from qtpy.QtGui import QFontDatabase, QPalette
from qtpy.QtWidgets import (
    QDialog,
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

from napari_vipp.installer.frontend import (
    InstallerBackend,
    InstallerController,
    InstallerScreen,
    InstallerSelection,
    InstallerViewState,
    TargetKind,
    TrackChoice,
)
from napari_vipp.ui.compute_setup_style import SetupDisclosureButton, setup_stylesheet
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


class _SetupSignals(QObject):
    state = Signal(object)
    console = Signal(str)


def _plain_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
    policy = label.sizePolicy()
    policy.setVerticalPolicy(QSizePolicy.Maximum)
    label.setSizePolicy(policy)
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _size(value: int) -> str:
    if value >= 1024**3:
        return f"{value / 1024**3:.1f} GiB"
    return f"{value / 1024**2:.0f} MiB"


class GpuSetupDialog(QDialog):
    """Review first, then install without modifying the running environment.

    The installer controller owns worker cancellation, stale results and rollback.
    Closing this window during work requests cancellation and waits for that
    rollback; the application also consults ``running`` before terminal shutdown.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        backend: InstallerBackend | None = None,
        backend_factory: Callable[[], InstallerBackend] | None = None,
        worker_factory: Callable | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("GpuSetupDialog")
        self.setWindowTitle("Set up GPU support — VIPP")
        self.setModal(False)
        self.resize(720, 650)
        self.setMinimumSize(560, 380)
        self._tone = "info"
        self._updating_theme = False
        self._backend = backend
        self._backend_factory = backend_factory
        self._worker_factory = worker_factory
        self._controller: InstallerController | None = None
        self._close_when_idle = False
        host = parent
        while host is not None and host.parentWidget() is not None:
            host = host.parentWidget()
        self._host_window = host
        if host is not None:
            host.installEventFilter(self)
        if parent is not None and parent is not host:
            parent.installEventFilter(self)
        self._signals = _SetupSignals(self)
        self._signals.state.connect(self._render, Qt.QueuedConnection)
        self._signals.console.connect(self._append_console, Qt.QueuedConnection)

        self.heading = _plain_label("Checking GPU setup…")
        self.heading.setObjectName("ComputeStatusHeading")
        self.description = _plain_label(
            "Install NVIDIA GPU support (CUDA 13) in a separate copy of this "
            "VIPP build. Setup checks your hardware and shows the location "
            "before you choose Install. Internet access is required."
        )
        self.message = _plain_label()
        self.location = _plain_label()
        self.location.setAccessibleName("GPU installation folder")
        self.location.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        path_font = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        path_font.setFixedPitch(True)
        self.location.setFont(path_font)
        self.size_label = _plain_label()
        self.status = _plain_label()
        self.status.setObjectName("ComputeMuted")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(6)
        self.work_note = _plain_label(
            "Your current VIPP window and workflows stay open. When setup "
            "finishes, Open VIPP with GPU starts a new window. Save and reopen "
            "any workflows you want to use there."
        )
        self.work_note.setObjectName("ComputeMuted")
        self.details_button = SetupDisclosureButton("Technical details")
        self.details_button.toggled.connect(self._show_details)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(500)
        self.details.setFont(path_font)
        self.details.setFixedHeight(180)
        self.details.hide()
        self.log_path = _plain_label()
        self.log_path.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.log_path.hide()
        self.action_button = ToolbarCommandButton("Checking…")
        self.action_button.setObjectName("ComputePrimaryAction")
        self.action_button.setEnabled(False)
        self.action_button.setAutoDefault(False)
        self.action_button.clicked.connect(self._action)
        self.cancel_button = ToolbarCommandButton("Cancel setup")
        self.cancel_button.clicked.connect(self.cancel_setup)
        self.close_button = ToolbarCommandButton("Close")
        self.close_button.clicked.connect(self.close)

        self.footer_widget = QFrame()
        self.footer_widget.setObjectName("ComputeFooter")
        actions = QHBoxLayout(self.footer_widget)
        actions.setContentsMargins(20, 12, 20, 12)
        actions.setSpacing(8)
        actions.addStretch(1)
        add_dialog_buttons(
            actions,
            actions=(self.action_button,),
            stop_actions=(self.cancel_button,),
            dismiss=self.close_button,
        )
        content = QWidget()
        content.setObjectName("ComputeContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(20, 18, 20, 18)
        content_layout.setSpacing(14)
        title = _plain_label("Set up GPU support")
        title.setObjectName("ComputeWindowHeading")
        content_layout.addWidget(title)
        content_layout.addWidget(self.description)

        self.status_widget = QFrame()
        self.status_widget.setObjectName("ComputeStatusCard")
        status_layout = QVBoxLayout(self.status_widget)
        status_layout.setContentsMargins(18, 16, 18, 16)
        status_layout.setSpacing(10)
        for widget in (self.heading, self.message, self.status, self.progress):
            status_layout.addWidget(widget)
        content_layout.addWidget(self.status_widget)

        self.summary_widget = QFrame()
        self.summary_widget.setObjectName("ComputeCard")
        summary_layout = QVBoxLayout(self.summary_widget)
        summary_layout.setContentsMargins(16, 14, 16, 14)
        summary_layout.setSpacing(10)
        self.summary_title = _plain_label("Installation location")
        self.summary_title.setObjectName("ComputeSectionTitle")
        self.summary_description = _plain_label(
            "The separate folder for GPU-enabled VIPP and its components, "
            "not your images or workflow files."
        )
        self.summary_description.setObjectName("ComputeMuted")
        for widget in (
            self.summary_title,
            self.location,
            self.summary_description,
            self.size_label,
        ):
            summary_layout.addWidget(widget)
        content_layout.addWidget(self.summary_widget)
        content_layout.addWidget(self.work_note)

        details_card = QFrame()
        details_card.setObjectName("ComputeCard")
        details_layout = QVBoxLayout(details_card)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(0)
        details_layout.addWidget(self.details_button)
        self.details_widget = QWidget()
        self.details_widget.setObjectName("ComputeCardBody")
        log_layout = QVBoxLayout(self.details_widget)
        log_layout.setContentsMargins(16, 0, 16, 16)
        log_layout.setSpacing(10)
        log_help = _plain_label(
            "Setup checks and progress for troubleshooting. "
            "Opening this section does not start an installation."
        )
        log_help.setObjectName("ComputeMuted")
        log_layout.addWidget(log_help)
        log_layout.addWidget(self.details)
        log_layout.addWidget(self.log_path)
        details_layout.addWidget(self.details_widget)
        self.details_widget.hide()
        content_layout.addWidget(details_card)
        content_layout.addStretch(1)

        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(content)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(self.footer_widget)
        self._refresh_style()
        self._start()

    @property
    def running(self) -> bool:
        return self._controller is not None and self._controller.busy

    def cancel_setup(self) -> None:
        """Request cooperative cancellation without interrupting rollback."""

        if self._controller is not None:
            self._controller.cancel()
            self._render(self._controller.state)

    def _start(self) -> None:
        if self.running:
            return
        self._close_when_idle = False
        try:
            if self._backend is None:
                if self._backend_factory is None:
                    from napari_vipp.gpu_setup import create_gpu_setup_backend

                    factory = create_gpu_setup_backend
                else:
                    factory = self._backend_factory
                self._backend = factory()
            kwargs = {}
            if self._worker_factory is not None:
                kwargs["worker_factory"] = self._worker_factory
            self._controller = InstallerController(
                self._backend,
                self._signals.state.emit,
                console_listener=self._signals.console.emit,
                **kwargs,
            )
            self._controller.start(
                InstallerSelection(
                    track=TrackChoice.CUDA13,
                    create_desktop_shortcut=False,
                )
            )
            self._render(self._controller.state)
        except Exception as exc:
            self._controller = None
            self._tone = "error"
            self.heading.setText("GPU setup could not start")
            self.message.setText(str(exc))
            self.status.setText("Your current VIPP installation is unchanged.")
            self.progress.hide()
            self.cancel_button.hide()
            self.action_button.setText("Try again")
            self.action_button.setEnabled(True)
            self.close_button.setEnabled(True)
            self.details.setPlainText(f"{type(exc).__name__}: {exc}")
            self._update_optional_rows()
            self._refresh_style()

    def _action(self) -> None:
        controller = self._controller
        if controller is None:
            self._start()
            return
        screen = controller.state.screen
        if screen is InstallerScreen.READY:
            controller.confirm()
        elif screen is InstallerScreen.SUCCESS:
            controller.open_vipp()
        elif screen is InstallerScreen.CURRENT:
            # The GPU backend only returns CURRENT after checking the exact
            # installed wheel and completing a fresh GPU verification process.
            if controller.state.target_kind is TargetKind.CURRENT:
                controller.open_vipp()
        elif screen in {
            InstallerScreen.BLOCKED,
            InstallerScreen.CANCELLED,
            InstallerScreen.FAILED,
        }:
            controller.retry()
        self._render(controller.state)

    def _render(self, state: InstallerViewState) -> None:
        controller = self._controller
        # Queued progress can arrive after a click already advanced the controller.
        if controller is None or state is not controller.state:
            return
        screen = state.screen
        busy = controller.busy
        verified = screen is InstallerScreen.SUCCESS or (
            screen is InstallerScreen.CURRENT
            and state.target_kind is TargetKind.CURRENT
        )
        self._tone = (
            "success"
            if verified
            else "error"
            if screen is InstallerScreen.FAILED
            else "warning"
            if screen in {InstallerScreen.BLOCKED, InstallerScreen.CURRENT}
            else "neutral"
            if screen is InstallerScreen.CANCELLED
            else "info"
        )
        self.heading.setText(
            {
                InstallerScreen.CHECKING: "Checking GPU setup…",
                InstallerScreen.READY: "Ready to set up GPU support",
                InstallerScreen.WORKING: "Setting up GPU support…",
                InstallerScreen.CANCELLING: "Cancelling safely…",
                InstallerScreen.SUCCESS: "GPU setup is ready",
                InstallerScreen.CURRENT: (
                    "GPU-enabled VIPP is ready" if verified else state.headline
                ),
            }.get(screen, state.headline)
        )
        self.description.setText(
            "Open the verified GPU-enabled copy of VIPP in a new window. "
            "No installation is needed."
            if verified
            else "Review the issue below before trying setup again. "
            "Your current VIPP window stays open."
            if screen in {InstallerScreen.FAILED, InstallerScreen.BLOCKED}
            else "Set up NVIDIA acceleration (CUDA 13) in a separate VIPP copy. "
            "Review the checks and location before choosing Install. "
            "Internet access is required for setup."
        )
        self.message.setText(
            "GPU checks passed for this exact VIPP build. It is ready to open."
            if verified
            else state.message.replace("Advanced details", "Technical details")
        )
        self.work_note.setText(
            "Your current VIPP window and workflows stay open. "
            + (
                "Open VIPP with GPU starts a new window. "
                if verified
                else "When setup finishes, Open VIPP with GPU starts a new window. "
            )
            + "Save and reopen any workflows you want to use there."
        )
        self.location.setText(str(state.target) if state.target else "")
        self.location.setToolTip(self.location.text())
        estimate = state.size_estimate
        self.size_label.setText(
            f"Estimated download: {_size(estimate.download_bytes)} · "
            f"Installed size: {_size(estimate.installed_bytes)} · "
            f"Temporary space: {_size(estimate.peak_temporary_bytes)}"
            if estimate is not None and not verified
            else ""
        )
        self.status.setText(state.status_message)
        self._update_optional_rows()
        self.progress.setVisible(busy)
        if state.progress_fraction is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(round(state.progress_fraction * 1000))
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(state.cancel_enabled)
        self.close_button.setEnabled(not busy)
        label = state.primary_label
        if screen is InstallerScreen.READY:
            label = (
                "Repair GPU setup"
                if state.target_kind is TargetKind.REPAIR
                else "Install GPU support"
            )
        elif screen is InstallerScreen.SUCCESS:
            label = "Open VIPP with GPU"
        elif screen is InstallerScreen.CURRENT:
            label = "Open VIPP with GPU"
        elif screen is InstallerScreen.BLOCKED:
            label = "Check again"
        self.action_button.setText(label)
        self.action_button.setEnabled(
            state.primary_enabled
            and not (
                screen is InstallerScreen.CURRENT
                and state.target_kind is not TargetKind.CURRENT
            )
        )
        self._refresh_style()
        if state.technical_details and screen in {
            InstallerScreen.READY,
            InstallerScreen.SUCCESS,
            InstallerScreen.BLOCKED,
            InstallerScreen.CANCELLED,
            InstallerScreen.FAILED,
        }:
            self.details.setPlainText(state.technical_details[-32000:])
        self.log_path.setText(f"Setup log: {state.log_path}" if state.log_path else "")
        self.log_path.setVisible(
            bool(state.log_path) and self.details_button.isChecked()
        )
        if self._close_when_idle and not busy:
            QTimer.singleShot(0, self.close)

    def _append_console(self, text: str) -> None:
        self.details.appendPlainText(text[-32000:].rstrip("\r\n"))

    def _update_optional_rows(self) -> None:
        for label in (self.message, self.location, self.size_label, self.status):
            label.setVisible(bool(label.text()))
        self.summary_widget.setVisible(
            bool(self.location.text() or self.size_label.text())
        )

    def _show_details(self, visible: bool) -> None:
        self.details_widget.setVisible(visible)
        self.details.setVisible(visible)
        self.log_path.setVisible(visible and bool(self.log_path.text()))

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if (
            event.type() in {QEvent.PaletteChange, QEvent.StyleChange}
            and hasattr(self, "scroll_area")
            and not self._updating_theme
        ):
            self._updating_theme = True
            try:
                self._refresh_style()
            finally:
                self._updating_theme = False

    def _refresh_style(self) -> None:
        # The parent/root palette is not painted by our local stylesheet, so a
        # live theme change remains an uncontaminated source of semantic colors.
        parent = self.parentWidget()
        palette = QWidget.palette(parent if parent is not None else self)
        stylesheet = setup_stylesheet(
            palette, dialog_name=self.objectName(), tone=self._tone
        )
        self.scroll_area.setStyleSheet(stylesheet)
        self.footer_widget.setStyleSheet(stylesheet)
        self.details_button.refresh_disclosure()
        primary_palette = QPalette(palette)
        primary_palette.setColor(QPalette.ButtonText, Qt.white)
        self.action_button.setIcon(toolbar_icon("compute", primary_palette))
        self.cancel_button.setIcon(toolbar_icon("stop", palette))

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.running:
            self._close_when_idle = True
            self.cancel_setup()
            event.ignore()
            return
        event.accept()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if (
            watched in (self.parentWidget(), self._host_window)
            and event.type() in {QEvent.PaletteChange, QEvent.StyleChange}
            and hasattr(self, "scroll_area")
        ):
            # Qt finishes propagating the parent's palette after this filter.
            QTimer.singleShot(0, self._refresh_style)
        # A dock's rejected close event does not necessarily veto its host's
        # shutdown. Protect the actual top-level viewer until rollback finishes.
        if (
            watched is self._host_window
            and event.type() == QEvent.Close
            and self.running
        ):
            event.ignore()
            self.show()
            self.raise_()
            self.activateWindow()
            self.message.setText(
                "Wait for GPU setup to finish, or cancel setup before closing VIPP."
            )
            self.message.show()
            return True
        return super().eventFilter(watched, event)

    def reject(self) -> None:
        # Escape follows the same cooperative cancellation as the window X.
        self.close()
