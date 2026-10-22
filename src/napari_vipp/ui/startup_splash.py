"""Branded, reusable startup view for the standalone VIPP launcher."""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from threading import Event, Thread

from qtpy.QtCore import QPoint, QRect, Qt, QTimer, Signal
from qtpy.QtGui import (
    QCloseEvent,
    QColor,
    QFontDatabase,
    QMouseEvent,
    QPainter,
    QPaintEvent,
    QPixmap,
    QShowEvent,
)
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.startup import (
    PROFILE_SPECS,
    STARTUP_STAGES,
    LaunchProfile,
    StartupPhase,
    StartupSnapshot,
)
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.startup_text import (
    PreparedStatusLabel,
    PreparedText,
    PreparedTypeface,
)


def _branding_pixmap() -> QPixmap | None:
    """Load the packaged wordmark without assuming a filesystem install."""
    prepared = _bootstrap_pixmap("vipp-startup-logo.png")
    if prepared is not None:
        prepared.setDevicePixelRatio(4)
        return prepared
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


def _first_frame_pixmap() -> QPixmap | None:
    return _bootstrap_pixmap("vipp-startup-first-frame.png")


def _bootstrap_pixmap(name: str) -> QPixmap | None:
    try:
        payload = (
            resources.files("napari_vipp")
            .joinpath("assets", "branding", name)
            .read_bytes()
        )
    except (FileNotFoundError, ModuleNotFoundError, OSError, TypeError):
        return None
    pixmap = QPixmap()
    return pixmap if pixmap.loadFromData(payload) else None


def _prepare_font_database(done: Event, cancelled: Event) -> None:
    # QFontDatabase functions are documented as thread-safe. Do not construct
    # widgets here; font-file scanning must not hold up the initial frame.
    try:
        from napari_vipp.core.font_prefetch import prefetch_windows_fonts

        prefetch_windows_fonts(cancelled)
        if not cancelled.is_set():
            QFontDatabase.families()
    except Exception:
        # Fall back to ordinary GUI-thread font initialization if warming fails.
        pass
    finally:
        done.set()


class StartupSplash(QWidget):
    """Show authenticated child milestones while the full application imports."""

    keep_waiting_requested = Signal()
    hide_requested = Signal()
    close_requested = Signal()
    open_log_requested = Signal()
    first_painted = Signal()

    def __init__(
        self,
        *,
        profile: LaunchProfile | str,
        version: str,
        log_path: Path,
        defer_font_loading: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profile = LaunchProfile.parse(profile)
        self.profile_spec = PROFILE_SPECS[self.profile]
        self.log_path = Path(log_path)
        self._allow_close = False
        self._positioned = False
        self._paint_reported = False
        self._drag_offset: QPoint | None = None
        self._version = version
        self._elapsed_text = "0:00"
        self._snapshot: StartupSnapshot | None = None
        self._font_worker: Thread | None = None
        self._font_done = Event()
        self._font_cancelled = Event()
        self._first_frame = _first_frame_pixmap() if defer_font_loading else None
        typeface = PreparedTypeface.load()
        self._prepared_typeface = typeface
        if typeface is None:
            self._first_frame = None
        self._ui_ready = self._first_frame is None
        self.setObjectName("VippStandaloneSplash")
        self.setWindowTitle("Starting VIPP")
        # Keep normal taskbar/Dock and stacking behavior, without a title bar.
        # Qt.SplashScreen itself can imply special stacking on some platforms.
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)
        self.setFixedSize(640, 390)
        if self._ui_ready:
            self._build_ui(version)
        else:
            self._bootstrap_labels = []

            def text(name, value, rect, pixels, alignment=Qt.AlignCenter, **kwargs):
                label = PreparedText(
                    typeface, self, pixels=pixels, alignment=alignment, **kwargs
                )
                label.setAccessibleName(name)
                label.setText(value)
                label.setGeometry(*rect)
                self._bootstrap_labels.append(label)
                return label

            self._bootstrap_version = text(
                "VIPP version", f"VIPP {version}", (42, 183, 556, 24), 12
            )
            if self.profile is not LaunchProfile.AUTO:
                self._bootstrap_version.setText(
                    f"VIPP {version}    {self.profile_spec.label.upper()}"
                )
                text(
                    "Compute mode",
                    self.profile_spec.description,
                    (42, 319, 556, 20),
                    11,
                    color="#A8B7C8",
                )
            self._bootstrap_status = text(
                "Startup status",
                STARTUP_STAGES[0].message,
                (42, 220, 556, 38),
                14,
                style="status",
                color="#F8FAFC",
            )
            self._bootstrap_stage = text(
                "Startup step",
                f"Step 0 of {len(STARTUP_STAGES)}",
                (42, 295, 270, 24),
                11,
                Qt.AlignLeft,
            )
            self._bootstrap_clock = text(
                "Elapsed startup time", "", (320, 295, 278, 24), 11, Qt.AlignRight
            )
            self._bootstrap_clock.display(self._elapsed_text)
            text(
                "Startup note",
                self._startup_note(),
                (42, 338, 556, 30),
                10,
                color="#64748B",
            )
            self._font_poll = QTimer(self)
            self._font_poll.setInterval(20)
            self._font_poll.timeout.connect(self._finish_font_preparation)
            self.first_painted.connect(
                lambda: QTimer.singleShot(0, self._start_font_preparation)
            )

    def _start_font_preparation(self) -> None:
        if self._font_worker is not None:
            return
        self._font_worker = Thread(
            target=_prepare_font_database,
            args=(self._font_done, self._font_cancelled),
            name="vipp-splash-fonts",
            daemon=False,
        )
        try:
            self._font_worker.start()
        except RuntimeError:
            self._font_worker = None
            self._font_done.set()
        self._font_poll.start()

    def _finish_font_preparation(self) -> None:
        if not self._font_done.is_set():
            self.update()
            return
        self._font_poll.stop()
        for label in self._bootstrap_labels:
            label.hide()
            label.deleteLater()
        self._bootstrap_labels.clear()
        self._build_ui(self._version)
        self._ui_ready = True
        self._first_frame = None
        self.elapsed_label.setText(f"Elapsed {self._elapsed_text}")
        if self._snapshot is not None:
            self.update_snapshot(self._snapshot)
        self.update()

    def finish_font_worker(self) -> None:
        """Keep QApplication alive until font work ends, after its loop exits."""
        self._font_cancelled.set()
        if self._font_worker is not None:
            self._font_worker.join()

    def _build_ui(self, version: str) -> None:
        accent = self.profile_spec.accent
        self.setStyleSheet(
            "QWidget#VippStandaloneSplash { background: #08151F; "
            "border: 1px solid #334155; }"
            "QLabel { color: #F8FAFC; background: transparent; }"
            "QFrame#VippProfileBadge { background: #111E2A; "
            f"border: 1px solid {accent}; border-radius: 11px; }}"
            f"QLabel#VippProfileText {{ color: {accent}; font-size: 10px; "
            "font-weight: 700; }"
            "QProgressBar { background: #172432; border: 1px solid #334155; "
            "border-radius: 6px; min-height: 12px; max-height: 12px; }"
            f"QProgressBar::chunk {{ background: {accent}; border-radius: 5px; }}"
            "QPushButton { background: #172432; color: #E2E8F0; "
            "border: 1px solid #475569; border-radius: 5px; padding: 7px 14px; }"
            f"QPushButton:hover {{ border-color: {accent}; }}"
            "QToolButton#VippSplashMinimize { background: transparent; "
            "color: #A8B7C8; border: 1px solid transparent; border-radius: 5px; "
            "font-size: 22px; padding: 0; }"
            "QToolButton#VippSplashMinimize:hover { background: #172432; "
            "color: #F8FAFC; }"
            f"QToolButton#VippSplashMinimize:focus {{ border-color: {accent}; }}"
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
                pixmap
                if pixmap.devicePixelRatioF() > 1
                else pixmap.scaledToWidth(500, Qt.SmoothTransformation)
            )
            self.logo_label.setFixedHeight(146)
        root.addWidget(self.logo_label)

        identity = QHBoxLayout()
        identity.addStretch(1)
        self.version_label = QLabel(f"VIPP {version}")
        self.version_label.setStyleSheet("color: #94A3B8; font-size: 12px;")
        identity.addWidget(self.version_label)
        # The installed application has one identity: VIPP. Only an explicit
        # diagnostic CLI override needs a mode badge; Auto is the normal path.
        explicit_profile = self.profile is not LaunchProfile.AUTO
        if explicit_profile:
            identity.addSpacing(10)
            badge = QFrame()
            badge.setObjectName("VippProfileBadge")
            badge_layout = QHBoxLayout(badge)
            badge_layout.setContentsMargins(10, 3, 10, 3)
            profile_label = QLabel(self.profile_spec.label.upper())
            profile_label.setObjectName("VippProfileText")
            badge_layout.addWidget(profile_label)
            badge.setAttribute(Qt.WA_TransparentForMouseEvents)
            identity.addWidget(badge)
        identity.addStretch(1)
        root.addLayout(identity)

        self.status_label = (
            PreparedStatusLabel(self._prepared_typeface, STARTUP_STAGES[0].message)
            if self._prepared_typeface is not None
            else QLabel(STARTUP_STAGES[0].message)
        )
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(
            "font-size: 14px; font-weight: 500; color: #F8FAFC;"
        )
        root.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        root.addWidget(self.progress_bar)

        stats = QHBoxLayout()
        self.stage_label = QLabel("Step 0 of 6")
        self.stage_label.setStyleSheet("color: #94A3B8; font-size: 11px;")
        stats.addWidget(self.stage_label)
        stats.addStretch(1)
        self.elapsed_label = QLabel("Elapsed 0:00")
        self.elapsed_label.setStyleSheet("color: #94A3B8; font-size: 11px;")
        stats.addWidget(self.elapsed_label)
        root.addLayout(stats)

        self.profile_description = QLabel(
            self.profile_spec.description if explicit_profile else ""
        )
        self.profile_description.setAlignment(Qt.AlignCenter)
        self.profile_description.setWordWrap(True)
        self.profile_description.setStyleSheet("color: #A8B7C8; font-size: 10px;")
        root.addWidget(self.profile_description)
        self.profile_description.setVisible(explicit_profile)

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
        self.hide_button = QPushButton("Hide splash")
        self.hide_button.clicked.connect(self.hide_requested)
        self.open_log_button = QPushButton("Open diagnostic log")
        self.open_log_button.clicked.connect(self.open_log_requested)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close_requested)
        add_dialog_buttons(
            actions, actions=(self.keep_waiting_button,), dismiss=self.hide_button
        )
        add_dialog_buttons(
            actions, actions=(self.open_log_button,), dismiss=self.close_button
        )
        actions.addStretch(1)
        root.addLayout(actions)

        self.keep_waiting_button.hide()
        self.hide_button.hide()
        self.open_log_button.hide()
        self.close_button.hide()

        self.note_label = QLabel(self._startup_note())
        self.note_label.setAlignment(Qt.AlignCenter)
        self.note_label.setWordWrap(True)
        self.note_label.setStyleSheet("color: #64748B; font-size: 10px;")
        root.addWidget(self.note_label)

        # Decorative content is part of the drag surface; diagnostic text stays
        # selectable and action buttons keep their ordinary mouse handling.
        for label in self.findChildren(QLabel):
            if label is not self.detail_label:
                label.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.progress_bar.setAttribute(Qt.WA_TransparentForMouseEvents)

        # Overlay the existing top margin, rather than adding a toolbar row.
        self.minimize_button = QToolButton(self)
        self.minimize_button.setObjectName("VippSplashMinimize")
        self.minimize_button.setText("\N{MINUS SIGN}")
        self.minimize_button.setAccessibleName("Minimize startup window")
        self.minimize_button.setToolTip(
            "Minimize — VIPP will keep starting in the background."
        )
        self.minimize_button.setFocusPolicy(Qt.StrongFocus)
        self.minimize_button.setFixedSize(28, 28)
        self.minimize_button.move(self.width() - 38, 10)
        self.minimize_button.clicked.connect(self.showMinimized)

    def _startup_note(self) -> str:
        return {
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
        }[self.profile]

    def _start_system_move(self) -> bool:
        """Prefer the window manager's drag support (including Wayland)."""
        handle = self.windowHandle()
        start_move = getattr(handle, "startSystemMove", None)
        return bool(start_move()) if callable(start_move) else False

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            if not self._ui_ready and QRect(602, 10, 28, 28).contains(event.pos()):
                self.showMinimized()
                event.accept()
                return
            self._drag_offset = None
            if not self._start_system_move():
                self._drag_offset = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if self._drag_offset is not None and event.buttons() & Qt.LeftButton:
            self.move(event.globalPos() - self._drag_offset)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._drag_offset = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def update_snapshot(self, snapshot: StartupSnapshot) -> None:
        """Render one immutable launcher-state snapshot."""
        self._snapshot = snapshot
        status_color = {
            StartupPhase.READY: "#86EFAC",
            StartupPhase.FAILED: "#FCA5A5",
        }.get(snapshot.phase, "#F8FAFC")
        if not self._ui_ready:
            self._bootstrap_status.setText(snapshot.message)
            self._bootstrap_status.set_color(status_color)
            self._bootstrap_stage.setText(
                f"Step {snapshot.step} of {snapshot.total_steps}"
            )
            self.update()
            return
        self.status_label.setText(snapshot.message)
        self.progress_bar.setValue(snapshot.progress_percent)
        self.stage_label.setText(f"Step {snapshot.step} of {snapshot.total_steps}")
        timed_out = snapshot.phase is StartupPhase.TIMED_OUT
        failed = snapshot.phase is StartupPhase.FAILED
        self.keep_waiting_button.setVisible(timed_out)
        self.hide_button.setVisible(timed_out)
        self.open_log_button.setVisible(failed)
        self.close_button.setVisible(failed)
        self.detail_label.setVisible(failed and bool(snapshot.error))
        self.detail_label.setText(snapshot.error)
        self.status_label.setStyleSheet(
            f"font-size: 14px; font-weight: 500; color: {status_color};"
        )

    def update_elapsed(self, seconds: float) -> None:
        """Display stable elapsed time without implying synthetic progress."""
        total_seconds = max(0, int(seconds))
        minutes, remainder = divmod(total_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            rendered = f"{hours:d}:{minutes:02d}:{remainder:02d}"
        else:
            rendered = f"{minutes:d}:{remainder:02d}"
        self._elapsed_text = rendered
        if not self._ui_ready:
            self._bootstrap_clock.display(rendered)
            return
        self.elapsed_label.setText(f"Elapsed {rendered}")

    def permit_close(self) -> None:
        """Allow the controller to close the splash without treating it as hide."""
        self._allow_close = True

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        if not self._ui_ready:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.SmoothPixmapTransform)
            painter.drawPixmap(self.rect(), self._first_frame)
            if self._snapshot is not None:
                width = round(552 * self._snapshot.progress_percent / 100)
                painter.fillRect(
                    QRect(44, 267, width, 8), QColor(self.profile_spec.accent)
                )
            painter.end()
        if not self._paint_reported:
            self._paint_reported = True
            self.first_painted.emit()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802
        super().showEvent(event)
        if self._positioned:
            return
        screen = QApplication.primaryScreen()
        if screen is not None:
            frame = self.frameGeometry()
            frame.moveCenter(screen.availableGeometry().center())
            self.move(frame.topLeft())
            self._positioned = True

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self._allow_close:
            super().closeEvent(event)
            return
        event.ignore()
        self.hide_requested.emit()


__all__ = ["StartupSplash"]
