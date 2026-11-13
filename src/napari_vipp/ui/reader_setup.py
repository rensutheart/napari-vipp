"""Separate setup window: review, wait for the application to close, then add."""

from __future__ import annotations

import sys
import time

from qtpy.QtCore import QEvent, Qt, QThread, QTimer, Signal
from qtpy.QtGui import QPalette
from qtpy.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_vipp import reader_setup as setup
from napari_vipp.core.reader_support import probe_reader, reader_spec
from napari_vipp.ui.compute_setup_style import setup_stylesheet
from napari_vipp.ui.design import apply_text_role, install_window_policy
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.setup_presentation import SetupElapsed, SetupLogPanel


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
        self._started_at = None
        self._tone = "info"
        self._updating_theme = False
        self.setObjectName("ReaderSetupDialog")
        self.setWindowTitle(f"Set up {self.spec.name} — VIPP")
        self.resize(720, 650)
        install_window_policy(self, profile="chooser")
        self.content = QWidget()
        self.content.setObjectName("ComputeContent")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(24, 24, 24, 24)
        content_layout.setSpacing(16)
        self.heading = QLabel(f"Set up {self.spec.name}")
        self.heading.setWordWrap(True)
        apply_text_role(self.heading, "title")
        content_layout.addWidget(self.heading)
        self.description = QLabel(
            "Review and download the required packages from PyPI. "
            "Nothing is installed until you approve the plan and close VIPP / napari. "
            "Existing packages will not be upgraded or replaced."
            + (
                "\n\nBio-Formats may download Java and its runtime on first use; "
                "an internet connection is needed then too."
                if self.spec.optional
                else ""
            )
        )
        self.description.setTextFormat(Qt.PlainText)
        self.description.setWordWrap(True)
        apply_text_role(self.description, "body")
        content_layout.addWidget(self.description)
        environment_title = QLabel("VIPP / napari environment")
        apply_text_role(environment_title, "section")
        self.environment = QLabel(sys.prefix)
        self.environment.setTextFormat(Qt.PlainText)
        self.environment.setWordWrap(True)
        self.environment.setTextInteractionFlags(Qt.TextSelectableByMouse)
        apply_text_role(self.environment, "code")
        content_layout.addWidget(environment_title)
        content_layout.addWidget(self.environment)
        self.stage = QLabel("Step 1 of 3 · Review packages")
        apply_text_role(self.stage, "section")
        content_layout.addWidget(self.stage)
        self.status = QLabel("Ready to check the required packages.")
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        self.status.setTextInteractionFlags(Qt.TextSelectableByMouse)
        content_layout.addWidget(self.status)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.progress.setTextVisible(False)
        self.progress.setMinimumHeight(12)
        content_layout.addWidget(self.progress)
        self.elapsed = SetupElapsed()
        content_layout.addWidget(self.elapsed)
        self.package_title = QLabel("Exact packages to add")
        apply_text_role(self.package_title, "section")
        self.package_plan = QPlainTextEdit()
        self.package_plan.setReadOnly(True)
        self.package_plan.setMinimumHeight(140)
        apply_text_role(self.package_plan, "code")
        self.package_title.hide()
        self.package_plan.hide()
        content_layout.addWidget(self.package_title)
        content_layout.addWidget(self.package_plan)
        self.activity = SetupLogPanel()
        self.details = self.activity.output
        content_layout.addWidget(self.activity)
        content_layout.addStretch(1)
        self.scroll_area = QScrollArea()
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(self.content)
        self.footer = QFrame()
        self.footer.setObjectName("ComputeFooter")
        actions = QHBoxLayout(self.footer)
        actions.setContentsMargins(24, 16, 24, 16)
        actions.setSpacing(8)
        self.action = QPushButton("Review packages")
        self.action.setObjectName("ComputePrimaryAction")
        self.action.clicked.connect(self._action)
        self.cancel = QPushButton("Close")
        self.cancel.clicked.connect(self.close)
        actions.addStretch(1)
        add_dialog_buttons(actions, actions=(self.action,), dismiss=self.cancel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.scroll_area, 1)
        layout.addWidget(self.footer)
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._wait_for_exit)
        if reason := setup.setup_blocker():
            self._tone = "warning"
            self.status.setText(reason)
            self.action.setEnabled(False)
        self._refresh_style()

    def _refresh_style(self):
        style = setup_stylesheet(
            QPalette(self.palette()),
            dialog_name=self.objectName(),
            tone=self._tone,
            font_widget=self,
        )
        self.scroll_area.setStyleSheet(style)
        self.footer.setStyleSheet(style)

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if (
            event.type()
            in {QEvent.PaletteChange, QEvent.StyleChange, QEvent.FontChange}
            and hasattr(self, "footer")
            and not self._updating_theme
        ):
            self._updating_theme = True
            try:
                self._refresh_style()
            finally:
                self._updating_theme = False

    def _timing(self, *, finished=False):
        self.elapsed.update_timing(
            phase=self.phase,
            started=self._started_at,
            finished=time.monotonic() if finished else None,
        )

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
            self._started_at = time.monotonic()
            self._timing()
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
            self.stage.setText("Step 2 of 3 · Close VIPP / napari")
            self._timing()
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
        self.package_plan.setPlainText("\n".join(plan.packages))
        self.package_title.setText(f"Exact packages to add · {len(plan.packages):,}")
        self.package_title.show()
        self.package_plan.show()
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
        self.stage.setText("Step 3 of 3 · Install and check reader")
        self._timing()
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
        self._timing(finished=True)
        if status.state == "ready":
            self._tone = "success"
            self.heading.setText(f"{self.spec.name} is ready to retry")
            self.stage.setText("3 of 3 setup steps complete")
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
            self._tone = "warning"
            self.stage.setText("Reader check needs attention")
            self.status.setText(
                "Packages were installed, but the reader check needs attention: "
                + status.detail
            )
        self.action.setText("Setup finished")
        self._refresh_style()

    def _failed(self, message):
        if self.phase == "installing":
            message = (
                "Some packages may already have been added. Keep the details "
                "below and check Reader support after restarting. " + message
            )
        self.phase = "failed"
        self._tone = "error"
        self.stage.setText("Setup stopped before completion")
        self._timing(finished=True)
        self.status.setText("Setup stopped. " + message)
        self.action.setEnabled(False)
        self._refresh_style()

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            event.ignore()
            return
        self.timer.stop()
        event.accept()

    def reject(self):
        # Escape must not destroy a running installer thread.
        self.close()
