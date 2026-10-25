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
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
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
        dialog.resize(700, 420)
        layout = QVBoxLayout(dialog)
        heading = QLabel(
            "VIPP found working copies from an interrupted session. "
            "Recover opens new unsaved tabs; original files stay untouched. "
            "Automatic previews are rebuilt. Manual steps and file-writing "
            "nodes stay paused; nothing is sent to an assistant."
        )
        heading.setWordWrap(True)
        layout.addWidget(heading)
        hint = QLabel(
            "Select several sessions with Ctrl-click (Command-click on macOS), "
            "or a range with Shift-click."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        selection_tools = QHBoxLayout()
        selection_summary = QLabel()
        selection_summary.setObjectName("recoverySelectionSummary")
        select_all = QPushButton("Select all")
        select_all.setObjectName("recoverySelectAll")
        clear_selection = QPushButton("Clear selection")
        clear_selection.setObjectName("recoveryClearSelection")
        selection_tools.addWidget(selection_summary)
        selection_tools.addStretch()
        selection_tools.addWidget(select_all)
        selection_tools.addWidget(clear_selection)
        layout.addLayout(selection_tools)
        choices = QListWidget()
        choices.setObjectName("recoverySessions")
        choices.setAccessibleName("Interrupted workflow sessions")
        choices.setSelectionMode(QAbstractItemView.ExtendedSelection)
        for index, checkpoint in enumerate(checkpoints):
            titles = ", ".join(tab["title"] for tab in checkpoint.tabs)
            suffix = " · previous valid copy" if checkpoint.fallback_used else ""
            item = QListWidgetItem(f"{checkpoint.created_utc}{suffix}\n{titles}")
            item.setData(Qt.UserRole, index)
            choices.addItem(item)
        choices.setCurrentRow(0)
        layout.addWidget(choices, 1)
        actions = QHBoxLayout()
        recover = QPushButton("Recover selected")
        recover.setObjectName("recoverSelected")
        later = QPushButton("Keep for later")
        discard = QPushButton("Discard selected recovery…")
        discard.setObjectName("discardSelectedRecovery")
        actions.addWidget(recover)
        actions.addWidget(discard)
        actions.addStretch()
        actions.addWidget(later)
        layout.addLayout(actions)
        processing = False

        def counted(count, singular, plural=None):
            return f"{count} {singular if count == 1 else (plural or singular + 's')}"

        def selected():
            # Items retain their checkpoint identity as successful rows disappear.
            return [
                (item, checkpoints[item.data(Qt.UserRole)])
                for item in sorted(choices.selectedItems(), key=choices.row)
            ]

        def refresh():
            count = len(choices.selectedItems())
            selection_summary.setText(f"{count} of {choices.count()} sessions selected")
            recover.setText(f"Recover selected ({count})")
            discard.setText(f"Discard selected ({count})…")
            recover.setEnabled(bool(count) and not processing)
            discard.setEnabled(bool(count) and not processing)
            select_all.setEnabled(count < choices.count() and not processing)
            clear_selection.setEnabled(bool(count) and not processing)
            choices.setEnabled(not processing)
            later.setEnabled(not processing)

        def finish(failed):
            nonlocal processing
            # takeItem can select an adjacent row. Never turn an unselected copy
            # into an implied next action after the batch completes.
            choices.clearSelection()
            for item in failed:
                item.setSelected(True)
            processing = False
            refresh()
            if choices.count() == 0:
                dialog.accept()

        def apply():
            nonlocal processing
            batch = selected()
            if processing or not batch:
                return
            processing = True
            refresh()
            opened, retained = 0, 0
            failed = []
            # Stay in this Qt callback: queued automatic previews cannot start
            # halfway through opening the other selected workspaces.
            for item, checkpoint in batch:
                try:
                    self.restore(checkpoint)
                except Exception:
                    failed.append(item)
                    continue
                opened += 1
                self._restored.add(checkpoint.checkpoint_id)
                try:
                    self.checkpoint_now("recovered workspace", required=True)
                    self.store.mark_recovered(checkpoint.checkpoint_id)
                except Exception:
                    retained += 1
                choices.takeItem(choices.row(item))
            if opened:
                self.changed()
            if retained:
                self.announce(
                    f"Recovered tabs are open from {counted(opened, 'session')}. "
                    "Recovery bookkeeping could not be confirmed for "
                    f"{counted(retained, 'session')}. "
                    "Save the recovered workflows to protect your edits."
                    + (
                        f" {counted(len(failed), 'selected session')} could not be "
                        "recovered and remain available; wait for current work "
                        "to stop and try again."
                        if failed
                        else ""
                    )
                )
            elif failed:
                self.announce(
                    f"Recovered {counted(opened, 'session')}. "
                    f"{counted(len(failed), 'selected session')} could not be "
                    "recovered and remain available; "
                    "wait for current work to stop and try again."
                )
            else:
                self.announce(
                    f"Recovered {counted(opened, 'session')} into new unsaved tabs. "
                    "Original files are unchanged. Automatic previews rebuild "
                    "when a recovered tab is active; manual steps and "
                    "file-writing nodes stay paused."
                )
            finish(failed)

        def remove():
            nonlocal processing
            batch = selected()
            if processing or not batch:
                return
            count = len(batch)
            noun = "copy" if count == 1 else "copies"
            processing = True
            refresh()
            if (
                QMessageBox.question(
                    dialog,
                    f"Discard {count} recovery {noun}?",
                    f"The {counted(count, 'selected interrupted session')} "
                    "will no longer be offered for recovery. "
                    "Your ordinary saved workflow files are not changed.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                != QMessageBox.Yes
            ):
                processing = False
                refresh()
                return
            failed = []
            for item, checkpoint in batch:
                try:
                    self.store.dismiss(checkpoint.checkpoint_id)
                except Exception:
                    failed.append(item)
                    continue
                choices.takeItem(choices.row(item))
            dismissed = counted(count - len(failed), "recovery copy", "recovery copies")
            unconfirmed = counted(len(failed), "selected copy", "selected copies")
            self.announce(
                f"Discarded {dismissed}. "
                "Ordinary saved workflow files are unchanged."
                + (
                    f" Discard could not be confirmed for {unconfirmed}; "
                    "these remain listed. Check the recovery folder before retrying."
                    if failed
                    else ""
                )
            )
            finish(failed)

        choices.itemSelectionChanged.connect(refresh)
        select_all.clicked.connect(choices.selectAll)
        clear_selection.clicked.connect(choices.clearSelection)
        recover.clicked.connect(apply)
        discard.clicked.connect(remove)
        later.clicked.connect(dialog.reject)
        dialog.finished.connect(lambda *_: setattr(self, "_dialog", None))
        self._dialog = dialog
        refresh()
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
