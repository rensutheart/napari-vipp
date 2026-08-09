"""Branded, reusable startup view for the standalone VIPP launcher."""

from __future__ import annotations

from importlib import resources
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtGui import QCloseEvent, QPixmap, QShowEvent
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.startup import (
    PROFILE_SPECS,
    LaunchProfile,
    StartupPhase,
    StartupSnapshot,
)


def _branding_pixmap() -> QPixmap | None:
    """Load the packaged wordmark without assuming a filesystem install."""
    try:
        asset = resources.files("napari_vipp").joinpath(
            "assets",
            "branding",
            "vipp-logo-dark.svg",
        )
        payload = asset.read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError, TypeError):
        return None
    pixmap = QPixmap()
    return pixmap if pixmap.loadFromData(payload) else None


class StartupSplash(QWidget):
    """Show authenticated child milestones while the full application imports."""

    keep_waiting_requested = Signal()
    hide_requested = Signal()
    close_requested = Signal()
    open_log_requested = Signal()

    def __init__(
        self,
        *,
        profile: LaunchProfile | str,
        version: str,
        log_path: Path,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = LaunchProfile.parse(profile)
        self.profile_spec = PROFILE_SPECS[self.profile]
        self.log_path = Path(log_path)
        self._allow_close = False
        self._build_ui(version)

    def _build_ui(self, version: str) -> None:
        self.setObjectName("VippStandaloneSplash")
        self.setWindowTitle("Starting VIPP")
        self.setWindowFlags(Qt.SplashScreen | Qt.WindowStaysOnTopHint)
        self.setFixedSize(640, 390)
        accent = self.profile_spec.accent
        self.setStyleSheet(
            "QWidget#VippStandaloneSplash { background: #08151F; "
            "border: 1px solid #334155; }"
            "QLabel { color: #F8FAFC; background: transparent; }"
            "QFrame#VippProfileBadge { background: #111E2A; "
            f"border: 1px solid {accent}; border-radius: 11px; }}"
            f"QLabel#VippProfileText {{ color: {accent}; font-size: 10px; "
            "font-weight: 700; }}"
            "QProgressBar { background: #172432; border: 1px solid #334155; "
            "border-radius: 6px; min-height: 12px; max-height: 12px; }"
            f"QProgressBar::chunk {{ background: {accent}; border-radius: 5px; }}"
            "QPushButton { background: #172432; color: #E2E8F0; "
            "border: 1px solid #475569; border-radius: 5px; padding: 7px 14px; }"
            f"QPushButton:hover {{ border-color: {accent}; }}"
        )

        root = QVBoxLayout(self)
        root.setContentsMargins(42, 22, 42, 24)
        root.setSpacing(10)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        pixmap = _branding_pixmap()
        if pixmap is None:
            self.logo_label.setText("VIPP")
            self.logo_label.setStyleSheet(
                "font-size: 46px; font-weight: 800; color: #F8FAFC;"
            )
            self.logo_label.setFixedHeight(112)
        else:
            self.logo_label.setPixmap(
                pixmap.scaledToWidth(500, Qt.SmoothTransformation)
            )
            self.logo_label.setFixedHeight(146)
        root.addWidget(self.logo_label)

        identity = QHBoxLayout()
        identity.addStretch(1)
        self.version_label = QLabel(f"VIPP {version}")
        self.version_label.setStyleSheet("color: #94A3B8; font-size: 11px;")
        identity.addWidget(self.version_label)
        identity.addSpacing(10)
        badge = QFrame()
        badge.setObjectName("VippProfileBadge")
        badge_layout = QHBoxLayout(badge)
        badge_layout.setContentsMargins(10, 3, 10, 3)
        profile_label = QLabel(self.profile_spec.label.upper())
        profile_label.setObjectName("VippProfileText")
        badge_layout.addWidget(profile_label)
        identity.addWidget(badge)
        identity.addStretch(1)
        root.addLayout(identity)

        self.status_label = QLabel("Starting the VIPP application")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 14px; font-weight: 650;")
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        root.addWidget(self.progress_bar)

        stats = QHBoxLayout()
        self.stage_label = QLabel("Step 0 of 6")
        self.stage_label.setStyleSheet("color: #94A3B8; font-size: 10px;")
        stats.addWidget(self.stage_label)
        stats.addStretch(1)
        self.elapsed_label = QLabel("Elapsed 0:00")
        self.elapsed_label.setStyleSheet("color: #94A3B8; font-size: 10px;")
        stats.addWidget(self.elapsed_label)
        root.addLayout(stats)

        self.profile_description = QLabel(self.profile_spec.description)
        self.profile_description.setAlignment(Qt.AlignCenter)
        self.profile_description.setWordWrap(True)
        self.profile_description.setStyleSheet("color: #A8B7C8; font-size: 10px;")
        root.addWidget(self.profile_description)

        self.detail_label = QLabel()
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_label.setStyleSheet("color: #FCA5A5; font-size: 10px;")
        self.detail_label.hide()
        root.addWidget(self.detail_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.keep_waiting_button = QPushButton("Keep waiting")
        self.keep_waiting_button.clicked.connect(self.keep_waiting_requested)
        actions.addWidget(self.keep_waiting_button)
        self.hide_button = QPushButton("Hide splash")
        self.hide_button.clicked.connect(self.hide_requested)
        actions.addWidget(self.hide_button)
        self.open_log_button = QPushButton("Open diagnostic log")
        self.open_log_button.clicked.connect(self.open_log_requested)
        actions.addWidget(self.open_log_button)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_requested)
        actions.addWidget(self.close_button)
        actions.addStretch(1)
        root.addLayout(actions)

        self.keep_waiting_button.hide()
        self.hide_button.hide()
        self.open_log_button.hide()
        self.close_button.hide()

        notes = {
            LaunchProfile.AUTO: (
                "Optional GPU libraries initialize only when a workflow needs them."
            ),
            LaunchProfile.CPU: (
                "CPU safe mode keeps workflow execution on the scientifically "
                "authoritative CPU path."
            ),
            LaunchProfile.PREFER_GPU: (
                "The first CUDA start can take longer while kernels and caches "
                "are prepared."
            ),
        }
        self.note_label = QLabel(notes[self.profile])
        self.note_label.setAlignment(Qt.AlignCenter)
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("color: #64748B; font-size: 9px;")
        root.addWidget(self.note_label)

    def update_snapshot(self, snapshot: StartupSnapshot) -> None:
        """Render one immutable launcher-state snapshot."""
        self.status_label.setText(snapshot.message)
        self.progress_bar.setValue(snapshot.progress_percent)
        self.stage_label.setText(
            f"Step {snapshot.step} of {snapshot.total_steps}"
        )
        timed_out = snapshot.phase is StartupPhase.TIMED_OUT
        failed = snapshot.phase is StartupPhase.FAILED
        self.keep_waiting_button.setVisible(timed_out)
        self.hide_button.setVisible(timed_out)
        self.open_log_button.setVisible(failed)
        self.close_button.setVisible(failed)
        self.detail_label.setVisible(failed and bool(snapshot.error))
        self.detail_label.setText(snapshot.error)
        if snapshot.phase is StartupPhase.READY:
            self.status_label.setStyleSheet(
                "font-size: 14px; font-weight: 650; color: #86EFAC;"
            )
        elif failed:
            self.status_label.setStyleSheet(
                "font-size: 14px; font-weight: 650; color: #FCA5A5;"
            )
        else:
            self.status_label.setStyleSheet("font-size: 14px; font-weight: 650;")

    def update_elapsed(self, seconds: float) -> None:
        """Display stable elapsed time without implying synthetic progress."""
        total_seconds = max(0, int(seconds))
        minutes, remainder = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            rendered = f"{hours:d}:{minutes:02d}:{remainder:02d}"
        else:
            rendered = f"{minutes:d}:{remainder:02d}"
        self.elapsed_label.setText(f"Elapsed {rendered}")

    def permit_close(self) -> None:
        """Allow the controller to close the splash without treating it as hide."""
        self._allow_close = True

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        screen = QApplication.primaryScreen()
        if screen is not None:
            frame = self.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self.move(frame.topLeft())

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._allow_close:
            super().closeEvent(event)
            return
        event.ignore()
        self.hide_requested.emit()


__all__ = ["StartupSplash"]
