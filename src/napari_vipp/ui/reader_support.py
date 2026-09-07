"""Collapsed Image Source reader diagnostics and explicit recovery actions."""

from __future__ import annotations

import json
from html import escape

from qtpy.QtCore import QEvent, QObject, QProcess, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QPalette
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.reader_support import (
    READERS,
    ReaderStatus,
    dependency_status,
    reader_for_path,
    reader_spec,
)
from napari_vipp.reader_setup import launch_setup, python_executable
from napari_vipp.ui.palette_roles import palette_is_dark, theme_colors

READER_HELP = (
    "https://rensutheart.github.io/vipp-mkdocs/nightly/getting-started/reader-support/"
)


class _ReaderSupportSession(QObject):
    """One lazy diagnostic queue/cache per application, independent of nodes."""

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.statuses = {}
        self.process = None
        self._queue = []
        self._closed = False
        if isinstance(parent, QApplication):
            parent.aboutToQuit.connect(self.close)

    def check(self, keys):
        if self._closed:
            return
        for key in keys:
            reader_spec(key)
            if key not in self._queue and (
                self.process is None or self.process.property("reader") != key
            ):
                self._queue.append(key)
        if self.process is None:
            self._next_check()

    def record(self, status):
        self.statuses[status.key] = status
        self.changed.emit()

    def close(self):
        self._closed = True
        self._queue.clear()
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(1000)

    def _next_check(self):
        if self._closed or not self._queue:
            return
        key = self._queue.pop(0)
        # Application ownership lets checks finish even while source controls
        # are rebuilt, deselected or deleted. Never import readers on the UI thread.
        process = QProcess(self)
        process.setProperty("reader", key)
        self.process = process
        timer = QTimer(process)
        timer.setSingleShot(True)
        timer.timeout.connect(process.kill)
        timer.start(30000)
        finished = False

        def done(*_args):
            nonlocal finished
            if finished:
                return
            finished = True
            timer.stop()
            raw = bytes(process.readAllStandardOutput()).decode(
                "utf-8", errors="replace"
            )[-16000:]
            self.process = None
            if not self._closed:
                try:
                    data = json.loads(raw.strip().splitlines()[-1])
                    if (
                        process.exitCode()
                        or process.exitStatus() != QProcess.NormalExit
                        or data["key"] != key
                        or data["state"]
                        not in {"ready", "missing", "incompatible", "broken"}
                    ):
                        raise ValueError("Invalid reader check result")
                    status = ReaderStatus(
                        key, data["state"], str(data["detail"])[:1000]
                    )
                except (ValueError, KeyError, IndexError, TypeError):
                    status = ReaderStatus(
                        key,
                        "broken",
                        "Reader check could not finish (timeout, launch failure "
                        "or native-library error). Retry the check or "
                        "open Reader help.",
                    )
                self.record(status)
                self._next_check()
            process.deleteLater()

        process.finished.connect(done)
        process.errorOccurred.connect(
            lambda error: done() if error == QProcess.FailedToStart else None
        )
        process.setProgram(python_executable())
        process.setArguments(["-m", "napari_vipp.core.reader_support", key])
        self.changed.emit()
        process.start()


def _session_cache():
    app = QApplication.instance()
    cache = getattr(app, "_vipp_reader_support_session", None)
    if cache is None:
        cache = _ReaderSupportSession(app)
        app._vipp_reader_support_session = cache
    return cache


class _ReaderStatusLabel(QLabel):
    """Reserve wrapped text height through nested inspector/scroll layouts."""

    def _fit_height(self):
        if self.wordWrap():
            height = self.heightForWidth(max(self.width(), 1))
            if height >= 0 and height != self.minimumHeight():
                self.setMinimumHeight(height)

    def setText(self, text):  # noqa: N802
        super().setText(text)
        self._fit_height()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._fit_height()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.FontChange, QEvent.StyleChange):
            self._fit_height()


def _label(text=""):
    label = _ReaderStatusLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return label


class _ReaderRow(_ReaderStatusLabel):
    """Neutral bold format name, coloured status, and quieter escaped details."""

    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.status = None
        self.checking = False
        self.setTextFormat(Qt.RichText)
        self.setWordWrap(True)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.refresh(None)

    def refresh(self, status, checking=False):
        self.status, self.checking = status, checking
        # QLabel/napari styles set Window/WindowText, not the editor Base/Text
        # roles. Normalize these before choosing contrast-safe semantic colours.
        palette = self.palette()
        surface = palette.color(QPalette.Window)
        if surface.alpha() == 0 and self.parentWidget() is not None:
            surface = self.parentWidget().palette().color(QPalette.Window)
        palette.setColor(QPalette.Base, surface)
        palette.setColor(QPalette.Text, palette.color(QPalette.WindowText))
        colors = theme_colors(palette)
        state = "checking" if checking else status.state if status else "unchecked"
        tone = {
            "ready": colors.success,
            "missing": colors.warning,
            "incompatible": colors.warning,
            "broken": colors.error,
        }.get(state)
        color = (
            (tone.accent if palette_is_dark(palette)
             and state not in {"missing", "incompatible"} else tone.foreground)
            if tone else colors.text
        )
        name = {
            "ready": "Reader loads",
            "missing": "Not installed",
            "incompatible": "Needs update",
            "broken": "Could not load",
            "checking": "Checking…",
            "unchecked": "Not checked",
        }[state]
        detail = (
            "Checking reader loading…" if checking
            else status.detail if status
            else "Optional reader" if self.spec.optional else "Included reader"
        )
        self.setText(
            f'<b style="color: {colors.text.name()}">{escape(self.spec.name)}</b>'
            f' · <span style="color: {color.name()}">{escape(name)}</span><br>'
            f'<span style="color: {colors.muted_text.name()}">'
            f'{escape(detail).replace(chr(10), "<br>")}</span>'
        )
        self.setAccessibleName(f"{self.spec.name}: {name}. {detail}")

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.StyleChange):
            self.refresh(self.status, self.checking)


class ReaderSupportControl(QWidget):
    retryRequested = Signal()
    layoutChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.path = ""
        self.error = ""
        self._session = _session_cache()
        self.content_host = None
        self._lifetime = {"closed": False}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(6)
        layout.setAlignment(Qt.AlignTop)
        self.failure = QWidget()
        self.failure.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        failure_layout = QVBoxLayout(self.failure)
        failure_layout.setContentsMargins(0, 0, 0, 6)
        failure_layout.setAlignment(Qt.AlignTop)
        self.failure_label = _label()
        self.failure_action = QPushButton()
        self.failure_action.clicked.connect(self._failure_action)
        self.retry = QPushButton("Retry opening this image")
        self.retry.clicked.connect(self.retryRequested.emit)
        failure_layout.addWidget(self.failure_label)
        failure_layout.addWidget(self.failure_action)
        failure_layout.addWidget(self.retry)
        self.failure.hide()
        layout.addWidget(self.failure)
        self.toggle = QToolButton()
        self.toggle.setText("Reader support")
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.RightArrow)
        self.toggle.setCheckable(True)
        self.toggle.toggled.connect(self._toggle)
        layout.addWidget(self.toggle)
        self.content = QWidget()
        self.content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(12, 0, 0, 0)
        content_layout.setSpacing(8)
        content_layout.setAlignment(Qt.AlignTop)
        content_layout.addWidget(
            _label(
                "Reader availability for this VIPP session, shared by all "
                "Image Source nodes. Results are cached; recheck after changes.\n"
                "Checks test reader loading, not the current image."
            )
        )
        self.rows = {}
        self.actions = {}
        for spec in READERS:
            row = _ReaderRow(spec)
            content_layout.addWidget(row)
            action = QPushButton(f"Check {spec.name}")
            action.clicked.connect(
                lambda _checked=False, key=spec.key: self._reader_action(key)
            )
            content_layout.addWidget(action)
            action.hide()
            self.rows[spec.key] = row
            self.actions[spec.key] = action
        self.check_all = QPushButton("Check reader support")
        self.check_all.clicked.connect(lambda: self.check([s.key for s in READERS]))
        content_layout.addWidget(self.check_all)
        self.help = QPushButton("Reader help")
        self.help.clicked.connect(lambda: QDesktopServices.openUrl(QUrl(READER_HELP)))
        content_layout.addWidget(self.help)
        self.content.hide()
        layout.addWidget(self.content)
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lifetime = self._lifetime
        self.destroyed.connect(lambda: lifetime.update(closed=True))
        self._session.changed.connect(self._refresh_statuses)
        self._refresh_statuses()

    @property
    def statuses(self):
        return self._session.statuses

    @property
    def process(self):
        return self._session.process

    def _toggle(self, expanded):
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.content.setVisible(expanded)
        if expanded:
            self.check_if_needed()
        self.layoutChanged.emit()

    def check_if_needed(self):
        missing = [s.key for s in READERS if s.key not in self.statuses]
        if missing:
            self.check(missing)

    def set_content_host(self, host=None):
        """Host diagnostics outside the fixed-height form; keep failure CTAs inline.

        Restore with ``None`` before deleting the owning Image Source control.
        Moving the panel itself does not start checks or modify source settings.
        """
        if host is self.content_host:
            return
        if host is not None and host.layout() is None:
            raise ValueError("Reader support host must have a layout")
        old_parent = self.content.parentWidget()
        old_parent.layout().removeWidget(self.content)
        self.content_host = host
        target = self if host is None else host
        target.layout().addWidget(self.content)
        self.content.setVisible(host is not None or self.toggle.isChecked())
        self.toggle.setVisible(host is None)
        self.layoutChanged.emit()

    def set_source(self, path: str, error: str | None = None):
        if path != self.path:
            self.path = path
            self.error = ""
        if error is not None:
            self.error = str(error)
        self._refresh_failure()

    def _failure_key(self):
        spec = reader_for_path(self.path)
        if spec is None:
            return None
        # A native reader is the first repair target if its dependencies are
        # missing. Only then consider the optional broad fallback mentioned by
        # a classified load failure. Never execute text from an exception.
        if dependency_status(spec.key).state in {"missing", "incompatible"}:
            return spec.key
        if any(
            word in self.error.casefold()
            for word in ("bioio", "bio-formats", "bioformats", "java")
        ):
            return "bioformats"
        return spec.key

    def _refresh_failure(self):
        was_visible = not self.failure.isHidden()
        key = self._failure_key() if self.error else None
        self.failure.setVisible(bool(key))
        if key is None:
            if was_visible:
                self.layoutChanged.emit()
            return
        spec = reader_spec(key)
        status = self.statuses.get(key) or dependency_status(key)
        self.failure_label.setText(
            f"Trouble opening this image? Check {spec.name} support here. "
            "Reader setup does not change the image or its saved axes."
        )
        if status.state == "missing":
            self.failure_action.setText(f"Install {spec.name}…")
        elif status.state == "incompatible":
            self.failure_action.setText("Update / repair this installation…")
        else:
            self.failure_action.setText(f"Retry {spec.name} reader check")
        self.failure_action.setToolTip(status.detail)
        self.layoutChanged.emit()

    def _failure_action(self):
        if key := self._failure_key():
            self._reader_action(key)

    def _reader_action(self, key):
        status = self.statuses.get(key) or dependency_status(key)
        if status.state == "missing":
            try:
                launch_setup(key)
            except Exception as error:
                self._show_status(
                    ReaderStatus(key, "broken", f"Could not open reader setup: {error}")
                )
        elif status.state == "incompatible":
            QDesktopServices.openUrl(QUrl(READER_HELP))
        else:
            self.check([key])

    def check(self, keys):
        self._session.check(keys)

    def _show_status(self, status):
        self._session.record(status)

    def _refresh_statuses(self):
        active = self.process.property("reader") if self.process else None
        for spec in READERS:
            key = spec.key
            status = self.statuses.get(key)
            self.rows[key].refresh(status, checking=key == active)
            state = status.state if status else "unchecked"
            self.actions[key].setText(
                f"Install {spec.name}…"
                if state == "missing"
                else "Update / repair help…"
                if state == "incompatible"
                else f"Retry {spec.name} check"
            )
            self.actions[key].setEnabled(key != active)
            self.actions[key].setVisible(
                state not in {"ready", "unchecked"} and key != active
            )
        self.check_all.setText(
            "Checking reader support…" if active
            else "Recheck reader support" if self.statuses
            else "Check reader support"
        )
        self.check_all.setEnabled(active is None)
        self._refresh_failure()
        self.layoutChanged.emit()
