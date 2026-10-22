"""Schedule reader imports only after a supported launch reports UI readiness."""

from __future__ import annotations

import logging

from qtpy.QtCore import QObject, QTimer
from qtpy.QtWidgets import QApplication

from napari_vipp.core.reader_warmup import ReaderLibraryWarmup

_WARMUP_DELAY_MS = 500
_APPLICATION_ATTRIBUTE = "_vipp_reader_warmup"
_LOGGER = logging.getLogger(__name__)


class _ReaderWarmupScheduler(QObject):
    """One application-owned timer and worker, shared by all VIPP docks."""

    def __init__(self, application: QApplication) -> None:
        super().__init__(application)
        self.warmup = ReaderLibraryWarmup()
        self._stopping = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._start)
        application.aboutToQuit.connect(self.cancel)
        self.destroyed.connect(self.warmup.cancel)
        self._timer.start(_WARMUP_DELAY_MS)

    def _start(self) -> None:
        if not self._stopping:
            try:
                self.warmup.start()
            except Exception:
                # Optional preparation must not turn a usable app into a
                # startup failure; actual file operations keep their errors.
                _LOGGER.exception("Could not start the reader warm-up worker")
                self.cancel()

    def cancel(self) -> None:
        self._stopping = True
        self._timer.stop()
        self.warmup.cancel()


def schedule_reader_warmup() -> None:
    """Return immediately and schedule once per QApplication, on its GUI thread."""
    application = QApplication.instance()
    if application is None or hasattr(application, _APPLICATION_ATTRIBUTE):
        return
    scheduler = _ReaderWarmupScheduler(application)
    setattr(application, _APPLICATION_ATTRIBUTE, scheduler)


def reader_warmup_pending(module: str) -> bool:
    """Keep a file request off the UI thread while its import could be locked."""
    application = QApplication.instance()
    scheduler = getattr(application, _APPLICATION_ATTRIBUTE, None)
    if scheduler is None or scheduler._stopping:
        return False
    return scheduler.warmup.pending(module)
