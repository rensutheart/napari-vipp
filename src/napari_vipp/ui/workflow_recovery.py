"""Workflow working copies and recovery with guarded automatic calculations."""

from __future__ import annotations

import json
import os
from pathlib import Path

from qtpy.QtCore import (
    QObject,
    QRunnable,
    QSettings,
    QStandardPaths,
    QThreadPool,
    QTimer,
    Signal,
)
from qtpy.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from napari_vipp.core.compute import canonical_digest
from napari_vipp.core.workflow_recovery import RecoveryStore

RECOVERY_ROOT_ENV = "VIPP_WORKFLOW_RECOVERY_ROOT"


def recovery_root():
    override = os.environ.get(RECOVERY_ROOT_ENV)
    if override:
        return Path(override).expanduser().absolute()
    settings = QSettings("napari-vipp", "napari-vipp")
    chosen = settings.value("workflow/recovery-root-v1", "")
    if chosen:
        return Path(str(chosen)).expanduser().absolute()
    return (
        Path(QStandardPaths.writableLocation(QStandardPaths.GenericDataLocation))
        / "VIPP"
        / "workflow-recovery"
    )


class _Signals(QObject):
    done = Signal(int, object, str)


class _Save(QRunnable):
    def __init__(self, store, tabs, active, reason, sequence, signals):
        super().__init__()
        self.args = store, tabs, active, reason, sequence, signals

    def run(self):
        store, tabs, active, reason, sequence, signals = self.args
        result, error = None, ""
        try:
            result = store.checkpoint(tabs, active, reason=reason)
        except Exception:
            error = "Recovery copy could not be saved. Check disk space and the folder."
        signals.done.emit(sequence, result, error)


class WorkflowRecoveryController(QObject):
    """Capture on Qt's thread; write stable detached documents on one worker."""

    def __init__(self, parent, *, capture, restore, announce):
        super().__init__(parent)
        self.capture, self.restore, self.announce = capture, restore, announce
        self.store = None
        self.pending_recovery = False
        self.summary = "Recovery is starting…"
        self.last_checkpoint = None
        self.capturing = False
        self.closed = False
        self._writing = False
        self._pending = False
        self._digest = ""
        self._sequence = 0
        self._offer_shown = False
        self._dialog = None
        self._restored = set()
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._signals = _Signals(self)
        self._signals.done.connect(self._saved)
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.autosave)
        self.periodic = QTimer(self)
        self.periodic.setInterval(30_000)
        self.periodic.timeout.connect(self._periodic)
        try:
            self.store = RecoveryStore(recovery_root())
            self.pending_recovery = bool(self.store.recoverable())
            self.summary = "Recovery ready · working copies are separate from Save"
            if self.store.issues:
                self.summary = "Some recovery copies need attention · open Recovery"
            self.periodic.start()
        except Exception:
            self.summary = "Recovery unavailable · choose another recovery folder"

    def changed(self):
        if self.closed or self.capturing or self.store is None:
            return
        self._pending = True
        self.timer.start()

    def _periodic(self):
        if self._pending:
            self.autosave()

    def _capture(self):
        self.capturing = True
        try:
            tabs, active = self.capture()
            # Canonical plain values: workers never receive widget/cache aliases.
            tabs = json.loads(json.dumps(tabs, allow_nan=False))
            return tabs, active, canonical_digest({"tabs": tabs, "active": active})
        finally:
            self.capturing = False

    def autosave(self):
        if self.closed or self.store is None or self._writing:
            return
        self.timer.stop()
        try:
            tabs, active, digest = self._capture()
            if not tabs or digest == self._digest:
                self._pending = False
                return
        except Exception:
            self.summary = "Recovery copy pending · current edit is not yet saveable"
            return
        self._sequence += 1
        self._writing = True
        self._pending = False
        self._submitted_digest = digest
        self._pool.start(
            _Save(self.store, tabs, active, "autosave", self._sequence, self._signals)
        )

    def _saved(self, sequence, result, error):
        if self.closed or sequence != self._sequence:
            return
        self._writing = False
        if error:
            self.summary = error
            self._pending = True
            self.announce(error)
        else:
            self.last_checkpoint = result
            self._digest = self._submitted_digest
            self.summary = f"Recovery checkpoint · {result.created_utc}"
        if self._pending:
            self.timer.start()

    def checkpoint_now(self, reason="before edit", *, required=False):
        """Durable pre-edit boundary; failure must abort a requested AI Apply."""
        if self.closed or self.store is None:
            if required:
                raise ValueError("Recovery is unavailable. Choose a recovery folder.")
            return None
        self.timer.stop()
        if not self._pool.waitForDone(5000):
            raise ValueError("A recovery write is still running. Try again shortly.")
        self._sequence += 1  # Queued older UI receipts cannot replace this result.
        self._writing = False
        try:
            tabs, active, digest = self._capture()
            result = self.store.checkpoint(tabs, active, reason=reason)
        except Exception as error:
            self.summary = "Recovery checkpoint failed · no edit was applied"
            if required:
                raise ValueError(
                    "Cannot protect this edit: the recovery checkpoint failed. "
                    "Check disk space or choose another recovery folder."
                ) from error
            return None
        self.last_checkpoint = result
        self._digest = digest
        self._pending = False
        self.summary = f"Recovery checkpoint · {result.created_utc}"
        return result

    def offer_once(self):
        if self._offer_shown or self.closed or self.store is None:
            return
        self._offer_shown = True
        self.show_recovery(quiet=True)

    def show_recovery(self, *, quiet=False):
        if self.store is None:
            self.announce(self.summary)
            return
        try:
            checkpoints = [
                checkpoint
                for checkpoint in self.store.recoverable()
                if checkpoint.checkpoint_id not in self._restored
            ]
        except Exception:
            self.announce(
                "Recovery copies could not be checked. Check the recovery folder."
            )
            return
        if self.store.issues:
            self.announce(
                "Some recovery copies could not be verified. "
                "They were kept unchanged in the recovery folder."
            )
        if not checkpoints:
            if not quiet and not self.store.issues:
                self.announce("No interrupted VIPP workflows are waiting for recovery.")
            return
        if self._dialog is not None:
            self._dialog.raise_()
            return
        dialog = QDialog(self.parent())
        dialog.setWindowTitle("Recover interrupted workflows")
        dialog.resize(640, 340)
        layout = QVBoxLayout(dialog)
        heading = QLabel(
            "VIPP found working copies from an interrupted session. "
            "Recover opens new unsaved tabs; original files stay untouched. "
            "Automatic previews are rebuilt. Manual steps and file-writing "
            "nodes stay paused; nothing is sent to an assistant."
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)
        choices = QListWidget()
        for checkpoint in checkpoints:
            titles = ", ".join(tab["title"] for tab in checkpoint.tabs)
            suffix = " · previous valid copy" if checkpoint.fallback_used else ""
            choices.addItem(f"{checkpoint.created_utc}{suffix}\n{titles}")
        choices.setCurrentRow(0)
        layout.addWidget(choices, 1)
        actions = QHBoxLayout()
        recover = QPushButton("Recover selected")
        later = QPushButton("Keep for later")
        discard = QPushButton("Discard selected recovery…")
        actions.addWidget(recover)
        actions.addWidget(discard)
        actions.addStretch()
        actions.addWidget(later)
        layout.addLayout(actions)

        def apply():
            index = choices.currentRow()
            if not 0 <= index < len(checkpoints):
                return
            checkpoint = checkpoints[index]
            try:
                self.restore(checkpoint)
            except Exception:
                self.announce(
                    "Recovery did not finish. The recovery copy was kept; "
                    "wait for current work to stop and try again."
                )
                return
            self._restored.add(checkpoint.checkpoint_id)
            try:
                self.checkpoint_now("recovered workspace", required=True)
                self.store.mark_recovered(checkpoint.checkpoint_id)
            except Exception:
                self.announce(
                    "Recovered tabs are open. Their previous recovery copy was "
                    "kept because the new checkpoint could not be confirmed. "
                    "Save the recovered workflows to protect your edits."
                )
            self.changed()
            dialog.accept()

        def remove():
            index = choices.currentRow()
            if not 0 <= index < len(checkpoints):
                return
            if (
                QMessageBox.question(
                    dialog,
                    "Discard recovery copy?",
                    "Stop offering this interrupted session for recovery? "
                    "Your ordinary saved workflow files are not changed.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                == QMessageBox.Yes
            ):
                self.store.dismiss(checkpoints[index].checkpoint_id)
                dialog.accept()

        recover.clicked.connect(apply)
        discard.clicked.connect(remove)
        later.clicked.connect(dialog.reject)
        dialog.finished.connect(lambda *_: setattr(self, "_dialog", None))
        self._dialog = dialog
        dialog.show()

    def choose_folder(self):
        selected = QFileDialog.getExistingDirectory(
            self.parent(), "Choose VIPP recovery folder", str(recovery_root())
        )
        if not selected:
            return
        self.checkpoint_now("recovery folder changed")
        if not self._pool.waitForDone(5000):
            self.announce(
                "Wait for the recovery write to finish before changing folders."
            )
            return
        try:
            replacement = RecoveryStore(Path(selected).absolute())
            tabs, active, digest = self._capture()
            checkpoint = replacement.checkpoint(
                tabs, active, reason="recovery folder changed"
            )
        except Exception:
            if "replacement" in locals():
                replacement.close(clean=False)
            self.announce(
                "That recovery folder is unavailable. The current folder is kept."
            )
            return
        previous, self.store = self.store, replacement
        self._sequence += 1
        self._digest = digest
        self.last_checkpoint = checkpoint
        self._writing = False
        self._pending = False
        if previous is not None:
            try:
                previous.close(clean=True)
            except Exception:
                # New protection is already durable. A stale old copy is safer
                # than losing either copy because cleanup failed.
                self.announce("Recovery folder changed; an older copy was retained.")
        settings = QSettings("napari-vipp", "napari-vipp")
        settings.setValue("workflow/recovery-root-v1", str(Path(selected).absolute()))
        settings.sync()
        self.summary = f"Recovery checkpoint · {checkpoint.created_utc}"

    def close(self):
        if self.closed:
            return
        self.timer.stop()
        self.periodic.stop()
        self._pool.waitForDone(5000)
        self.closed = True
        if self.store is not None:
            try:
                self.store.close(clean=True)
            except Exception:
                # OS lease release is in the store's finally block. Keep the
                # previous copy discoverable rather than blocking app shutdown.
                self.summary = "An older recovery copy was retained on close"
        if self._dialog is not None:
            self._dialog.close()
