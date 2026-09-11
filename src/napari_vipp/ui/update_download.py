"""Owned background transfer state for an explicitly requested update.

The GUI chooses whether to hand off a ready file. This controller never starts
a transfer on a timer, closes the application or applies an installation.
"""

from __future__ import annotations

import threading
import weakref

from qtpy.QtCore import QObject, Signal

from napari_vipp.core.update_install import (
    DownloadProgress,
    InstallerDownloadRequest,
    UpdateDownloadCancelled,
    discard_verified_installer,
    download_installer,
    launch_verified_installer,
)


class UpdateDownloadController(QObject):
    changed = Signal()
    _progress_received = Signal(object)
    _finished = Signal(object)

    def __init__(self, parent=None, *, downloader=None, launcher=None, discard=None):
        super().__init__(parent)
        self.busy = False
        self.phase = ""
        self.received_bytes = 0
        self.total_bytes = None
        self.error = ""
        self.verified = None
        self.request = None
        self._downloader = downloader or download_installer
        self._launcher = launcher or launch_verified_installer
        self._discard = discard or discard_verified_installer
        self._cancellation = threading.Event()
        self._closed = threading.Event()
        self._generation = 0
        self._thread = None
        self._artifacts = {}
        self._progress_received.connect(self._on_progress)
        self._finished.connect(self._on_finished)
        # Do not attach a Python callback to QObject.destroyed: native teardown
        # across many PyQt dialogs can outlive its callback proxy. Owners call
        # shutdown explicitly; Python finalization is the non-Qt fallback.
        # This closure must not retain self.
        closed, cancelled = self._closed, self._cancellation
        artifacts, discard_artifact = self._artifacts, self._discard

        def cleanup():
            closed.set()
            cancelled.set()
            for key in tuple(artifacts):
                item = artifacts.pop(key, None)
                if item is not None:
                    discard_artifact(item)

        self._finalizer = weakref.finalize(self, cleanup)

    def _discard_generation(self, generation) -> None:
        artifact = self._artifacts.pop(generation, None)
        if artifact is not None:
            self._discard(artifact)

    def start(self, request: InstallerDownloadRequest) -> None:
        if self.busy or self._closed.is_set():
            return
        self._discard_generation(self._generation)
        self._generation += 1
        generation = self._generation
        self.request = request
        self.verified = None
        self.error = ""
        self.received_bytes = 0
        self.total_bytes = None
        self.phase = "downloading"
        self.busy = True
        self._cancellation.clear()
        self.changed.emit()
        self._thread = threading.Thread(
            target=self._run,
            args=(generation, request),
            name="VIPP update download",
            daemon=True,
        )
        self._thread.start()

    def _run(self, generation: int, request: InstallerDownloadRequest) -> None:
        def report(progress: DownloadProgress) -> None:
            if not self._closed.is_set():
                try:
                    self._progress_received.emit((generation, progress))
                except RuntimeError:
                    self._cancellation.set()

        try:
            verified = self._downloader(
                request,
                progress=report,
                cancelled=self._cancellation.is_set,
            )
            self._artifacts[generation] = verified
            was_cancelled = self._cancellation.is_set() or self._closed.is_set()
            if was_cancelled:
                self._discard_generation(generation)
                verified = None
            result = (generation, verified, "", was_cancelled)
        except UpdateDownloadCancelled:
            result = (generation, None, "", True)
        except Exception as exc:
            result = (generation, None, str(exc), self._cancellation.is_set())
        if not self._closed.is_set():
            try:
                self._finished.emit(result)
            except RuntimeError:
                self._cancellation.set()
                self._discard_generation(generation)

    def _on_progress(self, payload) -> None:
        generation, progress = payload
        if self._closed.is_set() or generation != self._generation or not self.busy:
            return
        self.phase = progress.phase
        self.received_bytes = progress.received_bytes
        self.total_bytes = progress.total_bytes
        self.changed.emit()

    def _on_finished(self, payload) -> None:
        generation, verified, error, cancelled = payload
        if self._closed.is_set() or generation != self._generation:
            self._discard_generation(generation)
            return
        if not self.busy:
            return
        self.busy = False
        if cancelled or self._cancellation.is_set():
            self._discard_generation(generation)
            self.phase = "cancelled"
        elif error:
            self.phase = "failed"
            self.error = error
        else:
            self.phase = "ready"
            self.verified = verified
        self.changed.emit()

    def cancel(self) -> None:
        self._cancellation.set()
        if not self.busy and self.phase == "ready":
            self._discard_generation(self._generation)
            self.verified = None
            self.phase = "cancelled"
            self.changed.emit()

    def open_installer(self) -> None:
        if self._closed.is_set() or self.busy or self.phase != "ready":
            return
        if self.verified is None or self.verified.request != self.request:
            self.phase = "failed"
            self.error = "The prepared update changed. Check for updates again."
            self.changed.emit()
            return
        # Mark before calling out: changed/reentrant UI callbacks cannot hand
        # the same installer off twice, even when a launch fails.
        self.phase = "opened"
        try:
            self._launcher(self.verified)
            # The external setup may still need its executable. Never remove
            # a handed-off file when this application/dialog closes.
            self._artifacts.pop(self._generation, None)
        except Exception as exc:
            self.phase = "failed"
            self.error = str(exc)
        self.changed.emit()

    def shutdown(self) -> None:
        self._closed.set()
        self._cancellation.set()
        for generation in tuple(self._artifacts):
            self._discard_generation(generation)
        if self.verified is not None and self.phase != "opened":
            self.verified = None
