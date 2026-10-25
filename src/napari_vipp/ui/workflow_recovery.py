"""Workflow working copies and recovery with guarded automatic calculations."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

from qtpy.QtCore import (
    QObject,
    QPersistentModelIndex,
    QRunnable,
    QSettings,
    QSize,
    QStandardPaths,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
)
from qtpy.QtGui import QFontMetrics, QPalette
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
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QVBoxLayout,
)

from napari_vipp.core.compute import canonical_digest
from napari_vipp.core.workflow_recovery import RecoveryStore
from napari_vipp.ui.design import apply_text_role, install_window_policy
from napari_vipp.ui.dialog_buttons import add_dialog_buttons

RECOVERY_ROOT_ENV = "VIPP_WORKFLOW_RECOVERY_ROOT"
_SESSION_TITLE = Qt.UserRole + 1
_SESSION_DETAILS = Qt.UserRole + 2


class _RecoverySessionDelegate(QStyledItemDelegate):
    """Keep native accessible checkboxes with a readable two-line session row."""

    def style_option(self, index):
        option = QStyleOptionViewItem()
        option.initFrom(self.parent())
        self.initStyleOption(option, index)
        option.rect = self.parent().visualRect(index)
        return option

    def paint(self, painter, option, index):
        option = QStyleOptionViewItem(option)
        self.initStyleOption(option, index)
        style = option.widget.style()
        text_rect = style.subElementRect(
            QStyle.SE_ItemViewItemText, option, option.widget
        ).adjusted(8, 0, -12, 0)
        option.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, option, painter, option.widget)
        painter.save()
        painter.setClipRect(text_rect)
        title_font = option.font
        title_font.setBold(True)
        title_metrics = QFontMetrics(title_font)
        details_metrics = QFontMetrics(option.font)
        height = title_metrics.height() + details_metrics.height() + 4
        top = text_rect.center().y() - height // 2
        group = (
            QPalette.Active
            if option.state & QStyle.State_Enabled
            else QPalette.Disabled
        )
        painter.setPen(option.palette.color(group, QPalette.Text))
        painter.setFont(title_font)
        painter.drawText(
            text_rect.left(),
            top + title_metrics.ascent(),
            title_metrics.elidedText(
                index.data(_SESSION_TITLE), Qt.ElideRight, text_rect.width()
            ),
        )
        secondary = option.palette.color(group, QPalette.Text)
        secondary.setAlpha(190)
        painter.setPen(secondary)
        painter.setFont(option.font)
        painter.drawText(
            text_rect.left(),
            top + title_metrics.height() + 4 + details_metrics.ascent(),
            details_metrics.elidedText(
                index.data(_SESSION_DETAILS), Qt.ElideRight, text_rect.width()
            ),
        )
        painter.restore()
        if index.row() + 1 < index.model().rowCount():
            painter.save()
            separator = option.palette.color(group, QPalette.Text)
            separator.setAlpha(35)
            painter.setPen(separator)
            painter.drawLine(
                option.rect.left() + 10,
                option.rect.bottom(),
                option.rect.right() - 10,
                option.rect.bottom(),
            )
            painter.restore()

    def sizeHint(self, option, index):
        return QSize(0, QFontMetrics(option.font).height() * 2 + 28)


class _RecoverySessions(QListWidget):
    """Row clicks choose one; native checkbox/Space toggles choose several."""

    def __init__(self):
        super().__init__()
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setItemDelegate(_RecoverySessionDelegate(self))
        self.setUniformItemSizes(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._pressed = None

    def check_indicator_rect(self, index):
        option = self.itemDelegate().style_option(index)
        return self.style().subElementRect(
            QStyle.SE_ItemViewItemCheckIndicator, option, self
        )

    def set_checked(self, checked):
        for row in range(self.count()):
            self.item(row).setCheckState(Qt.Checked if checked else Qt.Unchecked)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)
        self._pressed = None
        index = self.indexAt(event.pos())
        if self.isEnabled() and index.isValid():
            self.setFocus(Qt.MouseFocusReason)
            self.setCurrentIndex(index)
            self._pressed = (
                QPersistentModelIndex(index),
                self.check_indicator_rect(index).contains(event.pos()),
            )
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)
        pressed, self._pressed = self._pressed, None
        index = self.indexAt(event.pos())
        if (
            self.isEnabled()
            and pressed
            and pressed[0].isValid()
            and index.isValid()
            and pressed[0] == index
        ):
            on_checkbox = self.check_indicator_rect(index).contains(event.pos())
            if on_checkbox == pressed[1]:
                item = self.itemFromIndex(index)
                if on_checkbox:
                    item.setCheckState(
                        Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                    )
                else:
                    self.set_checked(False)
                    item.setCheckState(Qt.Checked)
        event.accept()

    def mouseDoubleClickEvent(self, event):
        # A second click still only chooses; it never starts recovery.
        self.mousePressEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Space and self.currentItem() is not None:
            if not event.isAutoRepeat():
                item = self.currentItem()
                item.setCheckState(
                    Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked
                )
            event.accept()
        else:
            super().keyPressEvent(event)


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

    def __init__(self, parent, *, capture, restore, announce, announce_warning=None):
        super().__init__(parent)
        self.capture, self.restore, self.announce = capture, restore, announce
        self.announce_warning = announce_warning or announce
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
            self.announce_warning(error)
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
            self.announce_warning(self.summary)
            return
        try:
            checkpoints = [
                checkpoint
                for checkpoint in self.store.recoverable()
                if checkpoint.checkpoint_id not in self._restored
            ]
        except Exception:
            self.announce_warning(
                "Recovery copies could not be checked. Check the recovery folder."
            )
            return
        if self.store.issues:
            self.announce_warning(
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
        install_window_policy(dialog, profile="chooser")
        dialog.setWindowTitle("Recover interrupted workflows")
        dialog.resize(740, 540)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(12)
        heading = QLabel("Pick up where you left off")
        apply_text_role(heading, "title")
        heading.setWordWrap(True)
        layout.addWidget(heading)
        description = QLabel(
            "Choose interrupted sessions to open their workflows as new, unsaved tabs. "
            "Your original files stay untouched."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        hint = QLabel(
            "Click a row to choose just one session. "
            "Use the checkboxes to choose several."
        )
        hint.setWordWrap(True)
        selection_tools = QHBoxLayout()
        selection_summary = QLabel()
        selection_summary.setObjectName("recoverySelectionSummary")
        summary_font = selection_summary.font()
        summary_font.setBold(True)
        selection_summary.setFont(summary_font)
        select_all = QPushButton("Select all")
        select_all.setObjectName("recoverySelectAll")
        clear_selection = QPushButton("Clear selection")
        clear_selection.setObjectName("recoveryClearSelection")
        select_all.setAutoDefault(False)
        clear_selection.setAutoDefault(False)
        selection_tools.addWidget(selection_summary)
        selection_tools.addStretch()
        selection_tools.addWidget(select_all)
        selection_tools.addWidget(clear_selection)
        layout.addLayout(selection_tools)
        choices = _RecoverySessions()
        choices.setObjectName("recoverySessions")
        choices.setAccessibleName("Interrupted workflow sessions")
        choices.setStyleSheet("QListWidget#recoverySessions { padding: 6px; }")
        choices.setAccessibleDescription(
            "Click a row to choose only that session. Check boxes to choose several. "
            "Use arrow keys to move and Space to toggle the focused session."
        )
        for index, checkpoint in enumerate(checkpoints):
            titles = " · ".join(tab["title"] for tab in checkpoint.tabs)
            tab_count = len(checkpoint.tabs)
            try:
                when = datetime.fromisoformat(checkpoint.created_utc).astimezone()
                date = when.strftime("%d %b %Y, %H:%M")
            except (ValueError, TypeError, OverflowError):
                date = checkpoint.created_utc
            details = (
                f"{date} · {tab_count} workflow {'tab' if tab_count == 1 else 'tabs'}"
            )
            if checkpoint.fallback_used:
                details += " · Previous valid copy"
            item = QListWidgetItem(f"{titles}\n{details}")
            item.setData(Qt.UserRole, index)
            item.setData(_SESSION_TITLE, titles)
            item.setData(_SESSION_DETAILS, details)
            item.setFlags(
                Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable
            )
            item.setCheckState(Qt.Unchecked)
            item.setToolTip(f"{titles}\n{details}\nSaved: {checkpoint.created_utc}")
            choices.addItem(item)
        choices.setCurrentRow(0)
        layout.addWidget(choices, 1)
        layout.addWidget(hint)
        safety = QLabel(
            "Automatic previews rebuild when a recovered tab is active. "
            "Manual steps and file-writing nodes stay paused; "
            "nothing is sent to an assistant."
        )
        safety.setWordWrap(True)
        layout.addWidget(safety)
        layout.addSpacing(4)
        actions = QHBoxLayout()
        recover = QPushButton("Recover selected")
        recover.setObjectName("recoverSelected")
        later = QPushButton("Keep for later")
        discard = QPushButton("Discard selected recovery…")
        discard.setObjectName("discardSelectedRecovery")
        for button in (recover, discard, later):
            button.setAutoDefault(False)
        recover.setStyleSheet("font-weight: 600; padding: 6px 12px;")
        actions.addWidget(discard)
        actions.addStretch()
        discard.setProperty("vippActionRole", "destructive")
        add_dialog_buttons(actions, actions=(recover,), dismiss=later)
        layout.addLayout(actions)
        processing = False

        def counted(count, singular, plural=None):
            return f"{count} {singular if count == 1 else (plural or singular + 's')}"

        def selected():
            # Items retain their checkpoint identity as successful rows disappear.
            return [
                (item, checkpoints[item.data(Qt.UserRole)])
                for row in range(choices.count())
                if (item := choices.item(row)).checkState() == Qt.Checked
            ]

        def refresh():
            count = len(selected())
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
            # Focus moving to a neighbor never authorizes another recovery.
            choices.set_checked(False)
            for item in failed:
                item.setCheckState(Qt.Checked)
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
                self.announce_warning(
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
                self.announce_warning(
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
                    f"Discard the recovery {noun} for "
                    f"{counted(count, 'selected session')}? "
                    "You won't be able to recover them here. "
                    "Your saved workflow files stay untouched.",
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
            (self.announce_warning if failed else self.announce)(
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

        choices.itemChanged.connect(refresh)
        select_all.clicked.connect(lambda: choices.set_checked(True))
        clear_selection.clicked.connect(lambda: choices.set_checked(False))
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
            self.announce_warning(
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
            self.announce_warning(
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
                self.announce_warning(
                    "Recovery folder changed; an older copy was retained."
                )
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
