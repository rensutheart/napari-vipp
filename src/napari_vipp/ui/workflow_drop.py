"""Route local workflow-file drops without changing image or node drags."""

from __future__ import annotations

from pathlib import Path

from qtpy.QtCore import QEvent, QMimeData, QObject, Qt, QTimer, Signal
from qtpy.QtWidgets import QWidget


def workflow_drop_path(mime: QMimeData) -> Path | None:
    """Recognize one local JSON URL without reading the file during a drag."""
    urls = mime.urls()
    if len(urls) != 1 or not urls[0].isLocalFile():
        return None
    path = Path(urls[0].toLocalFile())
    return path if path.suffix.lower() == ".json" else None


class WorkflowFileDropHandler(QObject):
    """Handle drops only inside one workflow panel, not other napari windows.

    The host forwards its application-level event filter here so accepting
    children (graph viewports and text fields included) cannot consume JSON
    files first. Other MIME payloads keep their normal handlers.
    """

    openRequested = Signal(object)

    def __init__(self, host: QWidget) -> None:
        super().__init__(host)
        self._host = host
        host.setAcceptDrops(True)
        self._pending_path: Path | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._open_pending)

    def handle_event(self, watched: QObject, event: QEvent) -> bool:
        if event.type() not in {QEvent.DragEnter, QEvent.DragMove, QEvent.Drop}:
            return False
        if not isinstance(watched, QWidget) or not (
            watched is self._host or self._host.isAncestorOf(watched)
        ):
            return False
        # Child dialogs have their own purpose (batch config, file selection,
        # etc.); a drop there must not change the underlying workflow tab.
        if watched.window() is not self._host.window():
            return False
        path = workflow_drop_path(event.mimeData())
        if path is None:
            return False
        if not event.possibleActions() & Qt.CopyAction or self._pending_path:
            event.ignore()
            return True
        # Opening a workflow never moves/deletes the original JSON, even when
        # Explorer proposes a move. Never accept a move-only source.
        event.setDropAction(Qt.CopyAction)
        event.accept()
        if event.type() == QEvent.Drop:
            self._pending_path = path
            # Finish the OS drag before the normal Open action can show a
            # modal reproducibility prompt. Parent ownership cancels on delete.
            self._timer.start(0)
        return True

    def _open_pending(self) -> None:
        path, self._pending_path = self._pending_path, None
        if path is not None:
            self.openRequested.emit(path)
