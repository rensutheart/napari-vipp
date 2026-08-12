"""Lightweight, staged napari host for the full VIPP workflow editor.

This module deliberately imports only Qt and Python standard-library modules.
The large widget composition root is imported by a worker only after napari has
painted this host at least once; QWidget construction remains on the GUI thread.
"""

from __future__ import annotations

import traceback
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from importlib import import_module, resources
from typing import TYPE_CHECKING

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from qtpy.QtGui import QCloseEvent, QPaintEvent, QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from napari_vipp import __version__

if TYPE_CHECKING:
    import napari


class StartupState(StrEnum):
    """Observable states of the staged plugin startup host."""

    IDLE = "idle"
    LOADING = "loading"
    BUILDING = "building"
    PREPARING = "preparing"
    READY = "ready"
    ERROR = "error"


@dataclass(frozen=True)
class _PreloadOutcome:
    generation: int
    widget_class: type | None = None
    message: str = ""
    details: str = ""


class _PreloadSignals(QObject):
    finished = Signal(object)


class _WidgetPreloadWorker(QRunnable):
    """Import the composition root without constructing any QWidget."""

    def __init__(
        self,
        generation: int,
        loader: Callable[[], type],
    ) -> None:
        super().__init__()
        self.generation = int(generation)
        self.loader = loader
        self.signals = _PreloadSignals()

    def run(self) -> None:
        try:
            widget_class = self.loader()
        except Exception as exc:  # pragma: no cover - exercised through outcome
            outcome = _PreloadOutcome(
                generation=self.generation,
                message=f"{type(exc).__name__}: {exc}",
                details=traceback.format_exc(),
            )
        else:
            outcome = _PreloadOutcome(
                generation=self.generation,
                widget_class=widget_class,
            )
        self.signals.finished.emit(outcome)


def _load_vipp_widget_class() -> type:
    """Import and return the full widget class without creating an instance."""
    module = import_module("napari_vipp._widget")
    return module.VippWidget


def _branding_pixmap() -> QPixmap | None:
    """Load packaged branding while retaining a text-only fallback."""
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
    if not pixmap.loadFromData(payload):
        return None
    return pixmap


class VippStartupWidget(QWidget):
    """Paint a responsive loading host before constructing :class:`VippWidget`."""

    state_changed = Signal(object)

    def __init__(
        self,
        viewer: napari.viewer.Viewer,
        parent=None,
        *,
        initial_compute_mode: object | None = None,
    ) -> None:
        super().__init__(parent)
        self.viewer = viewer
        self._initial_compute_mode = initial_compute_mode
        self._state = StartupState.IDLE
        self._generation = 0
        self._closed = False
        self._start_scheduled = False
        self._workers: dict[int, _WidgetPreloadWorker] = {}
        self._real_widget: QWidget | None = None
        self._error_details = ""

        self.setObjectName("VippStartupWidget")
        self.setMinimumSize(440, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._stack = QStackedLayout(self)
        self._stack.setContentsMargins(0, 0, 0, 0)
        self._splash = self._build_splash()
        self._stack.addWidget(self._splash)
        self._stack.setCurrentWidget(self._splash)
        self._sync_presentation()

    @property
    def startup_state(self) -> StartupState:
        """Return the current startup state for diagnostics and tests."""
        return self._state

    @property
    def real_widget(self) -> QWidget | None:
        """Return the fully constructed editor once startup reaches ready."""
        return self._real_widget

    @property
    def error_details(self) -> str:
        """Return the traceback captured for the most recent startup failure."""
        return self._error_details

    def _build_splash(self) -> QWidget:
        splash = QFrame(self)
        splash.setObjectName("VippStartupSplash")
        splash.setStyleSheet(
            "QFrame#VippStartupSplash { background: #101722; color: #e5eef8; }"
            "QLabel { color: #e5eef8; }"
            "QProgressBar { background: #263242; border: 1px solid #40516a; "
            "border-radius: 5px; min-height: 10px; max-height: 10px; }"
            "QProgressBar::chunk { background: #24b7d9; border-radius: 4px; }"
            "QPushButton { padding: 6px 14px; }"
        )
        layout = QVBoxLayout(splash)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(12)
        layout.addStretch(1)

        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        pixmap = _branding_pixmap()
        if pixmap is None:
            self.logo_label.setText("VIPP")
            self.logo_label.setStyleSheet(
                "font-size: 38px; font-weight: 750; color: #f8fafc;"
            )
        else:
            self.logo_label.setPixmap(
                pixmap.scaledToWidth(410, Qt.SmoothTransformation)
            )
        layout.addWidget(self.logo_label)

        self.version_label = QLabel(f"VIPP {__version__} · napari plugin")
        self.version_label.setAlignment(Qt.AlignCenter)
        self.version_label.setStyleSheet("color: #9fb1c7; font-size: 11px;")
        layout.addWidget(self.version_label)

        self.status_label = QLabel()
        self.status_label.setObjectName("VippStartupStatus")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("font-size: 13px; font-weight: 600;")
        layout.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

        self.detail_label = QLabel()
        self.detail_label.setObjectName("VippStartupDetail")
        self.detail_label.setAlignment(Qt.AlignCenter)
        self.detail_label.setWordWrap(True)
        self.detail_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.detail_label.setStyleSheet("color: #fca5a5; font-size: 11px;")
        layout.addWidget(self.detail_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.retry_button = QPushButton("Retry")
        self.retry_button.clicked.connect(self.retry)
        actions.addWidget(self.retry_button)
        self.close_button = QPushButton("Close VIPP panel")
        self.close_button.clicked.connect(self.close)
        actions.addWidget(self.close_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        self.accelerator_note = QLabel(
            "Optional GPU libraries are initialized only when a workflow needs them."
        )
        self.accelerator_note.setAlignment(Qt.AlignCenter)
        self.accelerator_note.setWordWrap(True)
        self.accelerator_note.setStyleSheet("color: #8192a8; font-size: 10px;")
        layout.addWidget(self.accelerator_note)
        layout.addStretch(1)
        return splash

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802
        super().paintEvent(event)
        if (
            self._state is StartupState.IDLE
            and not self._start_scheduled
            and not self._closed
        ):
            # Scheduling from the first completed paint guarantees that napari
            # has shown useful feedback before the expensive imports begin.
            self._start_scheduled = True
            QTimer.singleShot(0, self.start)

    def start(self) -> bool:
        """Begin background preloading once; return whether work was started."""
        self._start_scheduled = False
        if self._closed or self._state not in {
            StartupState.IDLE,
            StartupState.ERROR,
        }:
            return False
        self._generation += 1
        generation = self._generation
        self._error_details = ""
        self._set_state(StartupState.LOADING)

        worker = _WidgetPreloadWorker(generation, _load_vipp_widget_class)
        worker.signals.finished.connect(
            self._on_preload_finished,
            Qt.QueuedConnection,
        )
        self._workers[generation] = worker
        QThreadPool.globalInstance().start(worker)
        return True

    def retry(self) -> bool:
        """Retry a failed startup without duplicating an active attempt."""
        if self._state is not StartupState.ERROR or self._closed:
            return False
        return self.start()

    def _on_preload_finished(self, outcome: _PreloadOutcome) -> None:
        self._workers.pop(outcome.generation, None)
        if self._closed or outcome.generation != self._generation:
            return
        if outcome.widget_class is None:
            self._show_error(outcome.message, outcome.details)
            return
        self._set_state(StartupState.BUILDING)
        QTimer.singleShot(
            0,
            lambda generation=outcome.generation, widget_class=outcome.widget_class: (
                self._build_real_widget(generation, widget_class)
            ),
        )

    def _build_real_widget(self, generation: int, widget_class: type) -> None:
        if self._closed or generation != self._generation:
            return
        try:
            widget = widget_class(
                self.viewer,
                self,
                defer_initial_run=True,
                initial_compute_mode=self._initial_compute_mode,
            )
            if not isinstance(widget, QWidget):
                raise TypeError("The VIPP plugin command did not return a QWidget.")
        except Exception as exc:
            self._show_error(
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )
            return
        if self._closed or generation != self._generation:
            widget.close()
            widget.deleteLater()
            return

        self._real_widget = widget
        self._stack.addWidget(widget)
        self._set_state(StartupState.PREPARING)
        # Force the status update to paint before the next synchronous workflow
        # calculation occupies the GUI thread.
        self._splash.repaint()
        QTimer.singleShot(
            0,
            lambda generation=generation: self._run_initial_pipeline(generation),
        )

    def _run_initial_pipeline(self, generation: int) -> None:
        widget = self._real_widget
        if self._closed or generation != self._generation or widget is None:
            return
        try:
            run_once = widget.run_initial_pipeline_once
            run_once()
        except Exception as exc:
            self._real_widget = None
            self._stack.removeWidget(widget)
            permit_discard = getattr(
                widget,
                "_permit_incomplete_startup_discard",
                None,
            )
            if callable(permit_discard):
                permit_discard()
            widget.close()
            widget.deleteLater()
            self._show_error(
                f"{type(exc).__name__}: {exc}",
                traceback.format_exc(),
            )
            return
        self._stack.setCurrentWidget(widget)
        self._set_state(StartupState.READY)

    def _set_state(self, state: StartupState) -> None:
        if state is self._state:
            return
        self._state = state
        self._sync_presentation()
        self.state_changed.emit(state)

    def _sync_presentation(self) -> None:
        messages = {
            StartupState.IDLE: "Preparing the VIPP workflow environment…",
            StartupState.LOADING: "Loading VIPP scientific modules…",
            StartupState.BUILDING: "Building the workflow interface…",
            StartupState.PREPARING: "Preparing the initial workflow…",
            StartupState.READY: "VIPP is ready.",
            StartupState.ERROR: "VIPP could not start.",
        }
        self.status_label.setText(messages[self._state])
        busy = self._state in {
            StartupState.LOADING,
            StartupState.BUILDING,
            StartupState.PREPARING,
        }
        self.progress_bar.setVisible(busy)
        failed = self._state is StartupState.ERROR
        self.detail_label.setVisible(failed)
        self.retry_button.setVisible(failed)
        self.close_button.setVisible(failed)

    def _show_error(self, message: str, details: str) -> None:
        self._error_details = details
        self.detail_label.setText(message or "Unknown startup error.")
        self.detail_label.setToolTip(details)
        self._stack.setCurrentWidget(self._splash)
        self._set_state(StartupState.ERROR)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        widget = self._real_widget
        if widget is not None and self._state is not StartupState.READY:
            permit_discard = getattr(
                widget,
                "_permit_incomplete_startup_discard",
                None,
            )
            if callable(permit_discard):
                permit_discard()
        if widget is not None and not widget.close():
            event.ignore()
            return
        self._closed = True
        self._generation += 1
        self._real_widget = None
        super().closeEvent(event)


__all__ = ["StartupState", "VippStartupWidget"]
