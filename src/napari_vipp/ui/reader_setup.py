"""Separate setup window: review, wait for the application to close, then add."""

from __future__ import annotations

import sys

from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from napari_vipp import reader_setup as setup
from napari_vipp.core.reader_support import probe_reader, reader_spec


class _SetupWorker(QThread):
    done = Signal(object)
    failed = Signal(str)
    log = Signal(str)

    def __init__(self, function, parent):
        super().__init__(parent)
        self.function = function

    def run(self):
        try:
            self.done.emit(self.function(self.log.emit))
        except Exception as error:
            self.failed.emit(str(error))


class ReaderSetupDialog(QDialog):
    def __init__(self, reader, directory, parent_pid, parent_created):
        super().__init__()
        self.spec = reader_spec(reader)
        self.directory = directory
        self.parent_pid = parent_pid
        self.parent_created = parent_created
        self.plan = None
        self.worker = None
        self.phase = "review"
        self.setWindowTitle(f"Set up {self.spec.name} — VIPP")
        self.resize(660, 500)
        layout = QVBoxLayout(self)
        self.description = QLabel(
            f"Add {self.spec.name} to this VIPP / napari installation.\n\n"
            f"Environment: {sys.prefix}\n\n"
            "First, review and download the required packages from PyPI. "
            "Nothing is installed until you approve the plan and close VIPP / napari. "
            "Existing packages will not be upgraded or replaced."
            + (
                "\n\nBio-Formats may download Java and its runtime on first use; "
                "an internet connection is needed then too."
                if self.spec.optional
                else ""
            )
        )
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.status = QLabel("Ready to check the required packages.")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.details = QPlainTextEdit()
        self.details.setReadOnly(True)
        self.details.setMaximumBlockCount(250)
        layout.addWidget(self.details, 1)
        actions = QHBoxLayout()
        self.action = QPushButton("Review packages")
        self.action.clicked.connect(self._action)
        self.cancel = QPushButton("Close")
        self.cancel.clicked.connect(self.close)
        actions.addWidget(self.action)
        actions.addWidget(self.cancel)
        layout.addLayout(actions)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._wait_for_exit)
        if reason := setup.setup_blocker():
            self.status.setText(reason)
            self.action.setEnabled(False)

    def _start(self, function, done):
        self.action.setEnabled(False)
        self.cancel.setEnabled(False)
        self.progress.show()
        self.worker = _SetupWorker(function, self)
        self.worker.log.connect(self.details.appendPlainText)
        self.worker.done.connect(done)
        self.worker.failed.connect(self._failed)
        self.worker.finished.connect(self._worker_finished)
        self.worker.start()

    def _worker_finished(self):
        self.progress.hide()
        self.cancel.setEnabled(True)
        self.action.setEnabled(self.phase in {"review", "approve"})

    def _action(self):
        if self.phase == "review":
            self.status.setText(
                "Checking and downloading packages. This can take a few minutes; "
                "VIPP can stay open."
            )
            self._start(
                lambda log: setup.prepare(self.spec.key, self.directory, log),
                self._prepared,
            )
        elif self.phase == "approve":
            self.phase = "waiting"
            self.action.setEnabled(False)
            self.status.setText(
                "Save your work and close VIPP / napari normally. Also close other "
                "Python sessions in this environment. Setup will continue here; "
                "nothing will be closed for you."
            )
            self.timer.start()

    def _prepared(self, plan):
        self.plan = plan
        self.phase = "approve"
        self.details.setPlainText("Packages to ADD:\n" + "\n".join(plan.packages))
        self.status.setText(
            "Review this exact package list. Existing packages stay unchanged. "
            "Approve below, then save your work and close VIPP / napari."
        )
        self.action.setText("Install after VIPP / napari closes")

    def _wait_for_exit(self):
        try:
            if setup.environment_users(self.parent_pid, self.parent_created):
                return
        except Exception as error:
            self.timer.stop()
            self._failed(str(error))
            return
        self.timer.stop()
        self.phase = "installing"
        self.status.setText(
            "Installing the approved packages. "
            "Keep VIPP / napari closed until setup finishes."
        )
        self._start(self._install, self._installed)

    def _install(self, log):
        setup.install(self.plan, self.parent_pid, self.parent_created, log)
        # This helper is disposable and the target application has closed.
        return probe_reader(self.spec.key)

    def _installed(self, status):
        self.phase = "finished"
        if status.state == "ready":
            self.status.setText(
                "Reader packages installed and import check passed. "
                "Reopen VIPP / napari and retry your image. "
                + (
                    "Java/Bio-Formats is checked when opening a file."
                    if self.spec.optional
                    else ""
                )
            )
        else:
            self.status.setText(
                "Packages were installed, but the reader check needs attention: "
                + status.detail
            )
        self.action.setText("Setup finished")

    def _failed(self, message):
        if self.phase == "installing":
            message = (
                "Some packages may already have been added. Keep the details "
                "below and check Reader support after restarting. " + message
            )
        self.phase = "failed"
        self.status.setText("Setup stopped. " + message)
        self.action.setEnabled(False)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            return
        self.timer.stop()
        event.accept()

    def reject(self):
        # Escape must not destroy a running installer thread.
        self.close()
