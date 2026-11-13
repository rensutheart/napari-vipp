"""Presentation-only progress and stable output for setup windows.

These widgets never discover environments, start work or interpret output as a
successful operation. A caller supplies authoritative state and timing.
"""

from __future__ import annotations

import time

from qtpy.QtCore import QTimer
from qtpy.QtGui import QFontDatabase
from qtpy.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.design import apply_text_role

INSTALLATION_STAGES = (
    ("preparing", "Prepare"),
    ("creating_environment", "Environment"),
    ("installing", "Packages"),
    ("verifying", "Checks"),
    ("creating_shortcuts", "Shortcuts"),
    ("committing", "Finish"),
)


def duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return (
        f"{hours}h {minutes:02}m {seconds:02}s"
        if hours
        else f"{minutes}m {seconds:02}s"
    )


def measured_progress(state) -> str:
    """Render only backend-provided quantities; never imply overall time progress."""
    if state.progress_stage == "rolling_back":
        return "Restoring the previous state — installation has not completed."
    completed, total = state.progress_completed, state.progress_total
    unit = getattr(state.progress_unit, "value", state.progress_unit)
    if (
        type(completed) is int
        and type(total) is int
        and 0 <= completed <= total
        and total > 0
        and unit in {"files", "packages", "checks", "bytes"}
    ):
        percent = 100 if completed == total else min(99, round(100 * completed / total))
        if unit == "bytes":
            count = f"{completed / 1024**2:,.1f} of {total / 1024**2:,.1f} MiB"
        else:
            count = f"{completed:,} of {total:,} {unit}"
        return f"{count} · {percent}% of this task"
    return "This activity does not report a measurable total."


class SetupElapsed(QLabel):
    """A small stable clock, separate from the text the user is reading."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._started = None
        self._finished = None
        self._phase = None
        self._phase_started = None
        apply_text_role(self, "secondary")
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.hide()

    def update_timing(self, *, phase, started=None, finished=None):
        if started != self._started:
            self._phase = None
            self._phase_started = None
        self._started, self._finished = started, finished
        if phase != self._phase:
            self._phase, self._phase_started = phase, time.monotonic()
        self.setVisible(started is not None)
        if started is not None and finished is None:
            self.timer.start()
        else:
            self.timer.stop()
        self.refresh()

    def refresh(self):
        if self._started is None:
            self.clear()
            return
        now = self._finished if self._finished is not None else time.monotonic()
        text = f"Total elapsed: {duration(now - self._started)}"
        if self._finished is None and self._phase_started is not None:
            text += f" · Current stage: {duration(now - self._phase_started)}"
        self.setText(text)


class StableSetupLog(QPlainTextEdit):
    """Keep a manual reading position; follow only while at the bottom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFontDatabase.systemFont(QFontDatabase.FixedFont))
        self.setMaximumBlockCount(1000)
        self.setMinimumHeight(180)
        self._following = True
        self._updating = False
        self.verticalScrollBar().valueChanged.connect(self._scroll_changed)

    def _scroll_changed(self, value):
        if not self._updating:
            self._following = value >= self.verticalScrollBar().maximum() - 2

    def appendPlainText(self, text):  # noqa: N802
        bar = self.verticalScrollBar()
        value, horizontal = bar.value(), self.horizontalScrollBar().value()
        following = self._following
        self._updating = True
        try:
            super().appendPlainText(text)
            bar.setValue(bar.maximum() if following else value)
            self.horizontalScrollBar().setValue(horizontal)
        finally:
            self._updating = False

    def set_snapshot(self, text):
        if text == self.toPlainText():
            return
        bar = self.verticalScrollBar()
        value, horizontal = bar.value(), self.horizontalScrollBar().value()
        self._updating = True
        try:
            self.setPlainText(text)
            bar.setValue(min(value, bar.maximum()))
            self.horizontalScrollBar().setValue(horizontal)
        finally:
            self._updating = False

    def follow_latest(self):
        self._following = True
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())


class SetupLogPanel(QWidget):
    """Optional activity output, never the authoritative approval package list."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        actions = QHBoxLayout()
        self.toggle = QCheckBox("Show activity output")
        self.latest = QPushButton("Jump to latest")
        self.latest.setAutoDefault(False)
        actions.addWidget(self.toggle)
        actions.addStretch(1)
        actions.addWidget(self.latest)
        layout.addLayout(actions)
        self.help = QLabel(
            "Recent output for troubleshooting (up to 1,000 lines). "
            "Scroll up to pause following; select text to copy."
        )
        self.help.setWordWrap(True)
        apply_text_role(self.help, "secondary")
        self.output = StableSetupLog()
        layout.addWidget(self.help)
        layout.addWidget(self.output, 1)
        self.toggle.toggled.connect(self._show_output)
        self.latest.clicked.connect(self.output.follow_latest)
        self._show_output(False)

    def _show_output(self, visible):
        self.output.setVisible(visible)
        self.help.setVisible(visible)
        self.latest.setVisible(visible)


__all__ = [
    "INSTALLATION_STAGES",
    "SetupElapsed",
    "SetupLogPanel",
    "StableSetupLog",
    "duration",
    "measured_progress",
]
