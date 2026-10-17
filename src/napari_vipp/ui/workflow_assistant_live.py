"""Local opt-in live MCP lease; every UI capture stays on the GUI thread."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from dataclasses import dataclass, field

from qtpy.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from napari_vipp.core.workflow_agent_live import WorkflowAgentLiveBridge
from napari_vipp.ui.workflow_assistant_style import AssistantButton


@dataclass
class _Pending:
    call: object
    epoch: int
    session: object
    done: threading.Event = field(default_factory=threading.Event)
    response: object = None
    rejected: bool = False


class _Signals(QObject):
    begin = Signal(object)
    ready = Signal(object, object)
    disconnected = Signal(int)
    tool_result = Signal(int, object, object)


class _Work(QRunnable):
    def __init__(self, pending, signals):
        super().__init__()
        self.pending, self.signals = pending, signals

    def run(self):
        try:
            result = self.pending.call()
        except Exception:
            result = None
        self.signals.ready.emit(self.pending, result)


class AssistantLiveAgentController(QObject):
    """Attach only a current external-owned investigation selected in VIPP.

    The composition root supplies ``start(external=True)``, ``_guard`` and the
    ordinary source capture. Nothing here imports the widget or accepts a path
    from an MCP tool. Public trial/nomination feedback rejoins native UI history.
    """

    def __init__(self, dialog, investigation):
        super().__init__(dialog)
        self.dialog, self.investigation = dialog, investigation
        self.bridge = None
        self._epoch = 0
        self._pending = None
        self._session = None
        self._pool = QThreadPool(dialog)
        self._pool.setMaxThreadCount(1)
        self._signals = _Signals(dialog)
        self._signals.begin.connect(self._begin, Qt.QueuedConnection)
        self._signals.ready.connect(self._ready, Qt.QueuedConnection)
        self._signals.disconnected.connect(self._disconnected, Qt.QueuedConnection)
        self._signals.tool_result.connect(self._tool_result, Qt.QueuedConnection)
        self.card = QFrame()
        self.card.setObjectName("AssistantInvestigationOptions")
        layout = QVBoxLayout(self.card)
        title = QLabel("Connect an external assistant to this VIPP session")
        title.setWordWrap(True)
        layout.addWidget(title)
        self.status_label = QLabel(
            "Off. Enable only when you want an external MCP client to inspect "
            "and test this goal."
        )
        self.status_label.setWordWrap(True)
        self.status_label.setTextFormat(Qt.PlainText)
        layout.addWidget(self.status_label)
        buttons = QHBoxLayout()
        self.connect_button = AssistantButton("Enable live connection")
        self.disconnect_button = AssistantButton("Disconnect and pause")
        self.copy_button = AssistantButton("Copy connection command")
        for button in (self.connect_button, self.disconnect_button, self.copy_button):
            button.setAutoDefault(False)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.command_edit = QLineEdit()
        self.command_edit.setReadOnly(True)
        self.command_edit.setAccessibleName("Explicit live MCP stdio command")
        self.command_edit.setPlaceholderText(
            "A private local connection command appears after you enable it."
        )
        layout.addWidget(self.command_edit)
        privacy = QLabel(
            "Uses this selected goal, source, sharing choices and remaining "
            "limits. No automatic client setup. Trials are detached; Keep, Add "
            "and Run remain your actions in VIPP."
        )
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        self.connect_button.clicked.connect(self.start)
        self.disconnect_button.clicked.connect(
            lambda: self.revoke("External assistant disconnected.")
        )
        self.copy_button.clicked.connect(
            lambda: QApplication.clipboard().setText(self.command_edit.text())
        )
        self.timer = QTimer(dialog)
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.sync)
        self.timer.start()
        self.sync()

    def _guard(self):
        i = self.investigation
        if (
            self.dialog._closed
            or not i.active
            or not getattr(i, "_external", False)
            or i.session is not self._session
            or i.agent_loop is None
            or i.agent_loop.session is not self._session
        ):
            raise ValueError("The selected live assistant owner changed.")
        i._guard(check_budget=False)

    @Slot()
    def start(self):
        try:
            if self.bridge is not None and not self.bridge.closed:
                return
            i = self.investigation
            if not i.active:
                self.dialog.consent_checkbox.setChecked(True)
                i.start(external=True)
            elif not getattr(i, "_external", False):
                raise ValueError(
                    "Pause the built-in assistant before choosing an external owner."
                )
            self.attach()
        except Exception:
            self.status_label.setText(
                "Connection was not enabled. Select a loaded input and goal, "
                "confirm sharing choices, and stop any other active assistant first."
            )

    def attach(self):
        """Called on GUI thread after the composition root's explicit capture."""
        self._session = self.investigation.session
        self._guard()
        if getattr(self._session, "paused", False):
            self.status_label.setText(
                "Paused. Resume this investigation in VIPP before enabling "
                "a new live connection."
            )
            return
        self._epoch += 1
        epoch = self._epoch
        self.bridge = WorkflowAgentLiveBridge(
            self.investigation.agent_loop.tools,
            enabled=True,
            dispatch=lambda call, disconnected: self._dispatch(
                call, disconnected, epoch
            ),
            on_disconnect=lambda: self._signals.disconnected.emit(epoch),
            on_tool_result=lambda public, session=self._session: (
                self._signals.tool_result.emit(epoch, session, public)
            ),
        )
        self.command_edit.setText(
            subprocess.list2cmdline(
                [
                    sys.executable,
                    "-m",
                    "napari_vipp.core.workflow_agent_mcp_cli",
                    "--enable-mcp",
                    "--live",
                    str(self.bridge.descriptor_path),
                ]
            )
        )
        self.status_label.setText(
            "Ready for one external client. Copy this stdio command into your "
            "MCP client; nothing is registered automatically."
        )
        self.sync()

    def _dispatch(self, call, disconnected, epoch):
        pending = _Pending(call, epoch, self._session)
        self._signals.begin.emit(pending)
        while not pending.done.wait(0.05):
            if disconnected.is_set() or epoch != self._epoch:
                pending.rejected = True
                self._signals.disconnected.emit(epoch)
                return None
        if pending.rejected:
            raise ValueError(
                "The live application owner no longer authorizes this request."
            )
        return pending.response

    @Slot(object)
    def _begin(self, pending):
        try:
            self._guard()
            if pending.epoch != self._epoch or self._pending is not None:
                raise ValueError("Another live request owns this session.")
            self._pending = pending
            self.status_label.setText(
                "External assistant is inspecting or testing. You can "
                "disconnect and pause; Add and Run are not automatic."
            )
            self._pool.start(_Work(pending, self._signals))
        except Exception:
            pending.rejected = True
            pending.done.set()

    @Slot(object, object)
    def _ready(self, pending, response):
        if self._pending is pending:
            self._pending = None
        if (
            pending.session is not self._session
            or self.investigation.session is not pending.session
        ):
            pending.rejected = True
            pending.done.set()
            return
        try:
            self._guard()
            if pending.epoch != self._epoch or pending.rejected:
                pending.rejected = (
                    True  # Record a completed local step, but never deliver it late.
                )
            else:
                pending.response = response
        except Exception:
            pending.rejected = True
            self.revoke(
                "Live authority changed. No late result was shared.", pause=False
            )
        finally:
            pending.done.set()
        self._show_pause_if_ready()
        self.sync()

    @Slot(int, object, object)
    def _tool_result(self, epoch, session, public):
        # This local receipt may arrive after a safe disconnect/pause. Preserve
        # its history for the SAME owner, but never transmit it from this slot.
        if session is not self._session or session is not self.investigation.session:
            return
        try:
            self._guard()
        except Exception:
            return
        handler = getattr(self.investigation, "external_result", None)
        if callable(handler):
            handler(public)
        else:
            self.dialog.trace_edit.setPlainText(
                json.dumps(public, indent=2, ensure_ascii=False)
            )
            self.investigation._event(
                "External assistant: "
                + public.get("action", "inspection").replace("_", " "),
                "Recorded in this same investigation.",
            )
            if session.pending_finish_review is not None:
                self.investigation._show_finish_review()
        self._show_pause_if_ready()

    @Slot(int)
    def _disconnected(self, epoch):
        if epoch != self._epoch:
            return
        self.revoke(
            "External client disconnected. This investigation is paused with "
            "its history and remaining limits kept."
        )

    def _show_pause_if_ready(self):
        if (
            self._session is not None
            and self._session is self.investigation.session
            and getattr(self._session, "paused", False)
        ):
            show = getattr(self.investigation, "_show_pause", None)
            if callable(show):
                show()

    def revoke(self, reason="Live connection disabled.", *, pause=True):
        """GUI owner revocation; no join, graph mutation or budget reset."""
        self._epoch += 1
        bridge, self.bridge = self.bridge, None
        if bridge is not None:
            bridge.close()
        if self._pending is not None:
            self._pending.rejected = True
            self._pending.done.set()
        self.command_edit.clear()
        if pause and self._session is not None:
            request_pause = getattr(self._session, "request_pause", None)
            if (
                callable(request_pause)
                and not self._session.finished
                and not self._session.stop_reason
            ):
                request_pause("External client disconnected.")
        self._show_pause_if_ready()
        self.status_label.setText(reason)
        self.sync(check_owner=False)

    def close(self):
        self.timer.stop()
        self.revoke("Live connection closed.")

    def sync(self, *, check_owner=True):
        bridge = self.bridge
        enabled = bridge is not None and not bridge.closed
        if enabled and check_owner:
            try:
                self._guard()
            except Exception:
                self.revoke(
                    "The input, workflow, sharing choice or owner changed. "
                    "Enable a new connection only after reviewing it.",
                    pause=False,
                )
                return
        self.disconnect_button.setEnabled(enabled)
        self.copy_button.setEnabled(enabled and bool(self.command_edit.text()))
        self.connect_button.setEnabled(
            not enabled
            and self._pending is None
            and not self.dialog._closed
            and (
                not self.investigation.active
                or getattr(self.investigation, "_external", False)
            )
        )
