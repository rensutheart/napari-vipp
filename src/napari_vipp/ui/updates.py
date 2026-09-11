"""Quiet, bounded release checks and an explicitly opened update dialog.

Network replies, timers, and dialogs have Qt owners. No worker touches scientific
state. Installer downloads and handoffs require an explicit dialog action.
"""

from __future__ import annotations

import json
import os
import re
import time

from packaging.version import Version
from qtpy.QtCore import QEvent, QObject, QSettings, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices
from qtpy.QtNetwork import (
    QNetworkAccessManager,
    QNetworkReply,
    QNetworkRequest,
    QSslSocket,
)
from qtpy.QtWidgets import QMenu, QPushButton

from napari_vipp.core.updates import (
    MAX_RESPONSE_BYTES,
    RELEASES_API,
    current_release_url,
    newest_release,
    parse_releases,
)
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.update_dialog import UpdateDialog as UpdateDialog
from napari_vipp.ui.update_dialog import _WrappedLabel as _WrappedLabel

CHECK_INTERVAL = 24 * 60 * 60
DISABLE_AUTO_ENV = "VIPP_DISABLE_UPDATE_CHECKS"


def _settings():
    return QSettings("napari-vipp", "napari-vipp")


def _bool(value) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


class UpdateController(QObject):
    changed = Signal()

    def __init__(
        self, current_version: str, parent=None, *, settings=None, manager=None
    ):
        super().__init__(parent)
        self.current_version = Version(current_version)
        self.settings = settings if settings is not None else _settings()
        self.manager = manager if manager is not None else QNetworkAccessManager(self)
        self.reply = None
        self.releases = ()
        self.checking = False
        self.error = ""
        self.error_kind = ""
        self.error_details = ""
        self.checked = False
        self.checked_this_session = False
        self.last_success = None
        self._closed = False
        self._last_attempt = None
        self._attempts = 0
        self._timed_out = False
        self._finishing = False
        self._ssl_errors = []
        self._data = bytearray()
        self._too_large = False
        self.deadline = QTimer(self)
        self.deadline.setSingleShot(True)
        self.deadline.timeout.connect(self._timeout)
        try:
            self.releases = parse_releases(
                json.loads(self.settings.value("updates/cache-v1", "[]"))
            )
            self.checked = bool(self.releases)
        except (ValueError, TypeError):
            pass
        try:
            saved_success = float(self.settings.value("updates/last-success", 0))
            if 0 < saved_success <= time.time():
                self.last_success = saved_success
        except (ValueError, TypeError):
            pass
        self.retry_timer = QTimer(self)
        self.retry_timer.setSingleShot(True)
        self.retry_timer.timeout.connect(self._begin_request)
        self.start_timer = QTimer(self)
        self.start_timer.setSingleShot(True)
        self.start_timer.timeout.connect(self.check_if_due)
        self.start_timer.start(8000)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.check_if_due)
        self.timer.start(60 * 60 * 1000)

    @property
    def automatic(self) -> bool:
        return _bool(self.settings.value("updates/automatic", True))

    @property
    def include_prereleases(self) -> bool:
        return _bool(
            self.settings.value(
                "updates/prereleases", self.current_version.is_prerelease
            )
        )

    @property
    def latest(self):
        return newest_release(
            self.releases, include_prereleases=self.include_prereleases
        )

    @property
    def available(self) -> bool:
        return self.latest is not None and self.latest.version > self.current_version

    def set_automatic(self, enabled: bool):
        self.settings.setValue("updates/automatic", enabled)
        self.settings.sync()
        self.changed.emit()

    def set_prereleases(self, enabled: bool):
        self.settings.setValue("updates/prereleases", enabled)
        self.settings.sync()
        self.changed.emit()

    def check_if_due(self):
        if not self.automatic or _bool(os.environ.get(DISABLE_AUTO_ENV, "0")):
            return
        # A previous launch's cached success/failure must not suppress this
        # launch. Only repeated checks within this controller are throttled.
        if self._last_attempt is None or (
            time.monotonic() - self._last_attempt >= CHECK_INTERVAL
        ):
            self.check()

    def check(self):
        if self._closed or self.checking:
            return
        self._last_attempt = time.monotonic()
        self.settings.setValue("updates/last-attempt", time.time())
        self.settings.sync()
        self.checking, self.error = True, ""
        self.error_kind = self.error_details = ""
        self.checked_this_session = False
        self._attempts = 0
        self._timed_out = False
        self.deadline.start(20000)
        self._begin_request()
        self.changed.emit()

    def _begin_request(self):
        if self._closed or not self.checking or self.reply is not None:
            return
        self._attempts += 1
        self._data = bytearray()
        self._too_large = False
        self._ssl_errors = []
        request = QNetworkRequest(QUrl(RELEASES_API))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"User-Agent", b"napari-vipp-update-check")
        request.setAttribute(
            QNetworkRequest.Attribute.RedirectPolicyAttribute,
            QNetworkRequest.RedirectPolicy.ManualRedirectPolicy,
        )
        request.setTransferTimeout(15000)
        reply = self.manager.get(request)
        self.reply = reply
        # A late signal from an aborted attempt cannot drain/finish its retry.
        reply.readyRead.connect(lambda: self._read(reply))
        reply.finished.connect(lambda: self._finished(reply))
        if hasattr(reply, "sslErrors"):
            reply.sslErrors.connect(
                lambda errors: self._record_ssl_errors(reply, errors)
            )

    def _record_ssl_errors(self, reply, errors):
        if reply is self.reply:
            self._ssl_errors = [
                self._safe_detail(error.errorString()) for error in errors[:4]
            ]

    @staticmethod
    def _safe_detail(value):
        detail = " ".join(str(value).split())
        # Diagnostics stay local, but do not display proxy URL credentials.
        detail = re.sub(r"(?i)(https?://)[^/\s@]+@", r"\1[redacted]@", detail)
        return detail[:300]

    def _timeout(self):
        self._timed_out = True
        self.retry_timer.stop()
        if self.reply is not None:
            self.reply.abort()
        elif self.checking:
            self._set_error("timeout", "The GitHub update check timed out. Try again.")
            self._end_check()

    def _read(self, reply=None):
        reply = self.reply if reply is None else reply
        if reply is None or reply is not self.reply or self._too_large:
            return
        remaining = MAX_RESPONSE_BYTES - len(self._data)
        # PyQt6 may return None (not an empty QByteArray) after the final drain.
        self._data.extend(bytes(reply.read(remaining + 1) or b""))
        if len(self._data) > MAX_RESPONSE_BYTES:
            self._too_large = True
            reply.abort()

    def _finished(self, reply=None):
        reply = self.reply if reply is None else reply
        if reply is None or reply is not self.reply or self._finishing:
            return
        self._finishing = True
        retry = False
        try:
            # abort() while draining an oversized reply can emit finished again.
            self._read(reply)
            if self._closed:
                return
            status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
            if self._too_large:
                self._set_error(
                    "response_size",
                    "The release response was too large. Try the GitHub release page.",
                )
            elif self._timed_out or (
                status != 200 or reply.error() != QNetworkReply.NetworkError.NoError
            ):
                self._network_error(reply, status)
                retry = (
                    self._attempts < 2 and not self._timed_out
                    and self.deadline.isActive()
                    and self.error_kind in {"connection", "dns", "timeout", "server"}
                )
            else:
                releases = parse_releases(json.loads(bytes(self._data)))
                if not releases:
                    raise ValueError("No published VIPP releases were returned.")
                self.releases = releases
                self.checked = self.checked_this_session = True
                self.last_success = time.time()
                self.settings.setValue(
                    "updates/cache-v1", json.dumps([r.cache_record() for r in releases])
                )
                self.settings.setValue("updates/last-success", self.last_success)
                self.settings.sync()
        except (ValueError, TypeError, UnicodeError) as exc:
            self._set_error(
                "invalid_response",
                "GitHub returned an invalid release response. Try again later.",
                self._safe_detail(exc),
            )
        finally:
            self.reply = None
            self._data.clear()
            reply.deleteLater()
            self._finishing = False
            if retry:
                self.error = self.error_kind = self.error_details = ""
                self.retry_timer.start(1000)
            else:
                self._end_check()

    def _set_error(self, kind, message, details=""):
        self.error_kind, self.error, self.error_details = kind, message, details

    def _end_check(self):
        self.deadline.stop()
        self.retry_timer.stop()
        self.checking = False
        if not self._closed:
            self.changed.emit()

    def _network_error(self, reply, status):
        code = reply.error()
        errors = QNetworkReply.NetworkError
        name = getattr(code, "name", str(code))
        detail = self._safe_detail(
            reply.errorString() if hasattr(reply, "errorString") else ""
        )
        details = f"Qt: {name}" + (f"; HTTP {status}" if status is not None else "")
        if detail and detail != "Unknown error":
            details += f"; {detail}"
        if self._ssl_errors:
            details += "; " + "; ".join(self._ssl_errors)
        if self._timed_out or code == errors.TimeoutError:
            kind, message = "timeout", "The GitHub update check timed out. Try again."
        elif code == errors.SslHandshakeFailedError or self._ssl_errors:
            kind = "tls"
            message = (
                "Could not establish a verified HTTPS connection to GitHub. "
                "Check your system clock, proxy or security software certificates."
            )
        elif code == errors.ProtocolUnknownError and not QSslSocket.supportsSsl():
            kind, message = "tls", (
                "This installation's Qt HTTPS support is unavailable. "
                "Use the GitHub release page or repair the VIPP installation."
            )
        elif code in {
            errors.ProxyConnectionRefusedError, errors.ProxyConnectionClosedError,
            errors.ProxyNotFoundError, errors.ProxyTimeoutError,
            errors.ProxyAuthenticationRequiredError, errors.UnknownProxyError,
        }:
            kind, message = "proxy", (
                "The network proxy could not connect to GitHub. "
                "Check the system proxy settings or ask your network administrator."
            )
        elif status in {403, 429}:
            # HTTP 403 also means access denied, not just rate limiting.
            rate_limited = status == 429 or b"rate limit" in bytes(self._data).lower()
            if hasattr(reply, "rawHeader"):
                rate_limited |= bytes(reply.rawHeader(b"X-RateLimit-Remaining")) == b"0"
            kind = "rate_limit" if rate_limited else "access"
            message = (
                "GitHub's request limit was reached. Please try again later."
                if rate_limited else
                "GitHub denied the update request (HTTP 403). "
                "Check your network policy or try the GitHub release page."
            )
        elif status is not None and 300 <= status < 400:
            kind, message = "redirect", (
                "GitHub redirected the release request unexpectedly. "
                "Check your network sign-in or use the GitHub release page."
            )
        elif status is not None and status >= 500:
            kind, message = "server", (
                "GitHub is temporarily unavailable. Try again later."
            )
        elif code == errors.HostNotFoundError:
            kind, message = "dns", (
                "Could not find api.github.com. Check your internet or DNS connection."
            )
        elif code in {
            errors.ConnectionRefusedError, errors.RemoteHostClosedError,
            errors.TemporaryNetworkFailureError, errors.NetworkSessionFailedError,
        }:
            kind, message = "connection", (
                "The connection to GitHub was interrupted. "
                "Check your connection, proxy or firewall and try again."
            )
        else:
            kind, message = "network", (
                "Could not check GitHub for updates. "
                "Check your connection or use the GitHub release page."
            )
        self._set_error(kind, message, details)

    def shutdown(self):
        self._closed = True
        self.start_timer.stop()
        self.timer.stop()
        self.retry_timer.stop()
        self.deadline.stop()
        if self.reply is not None:
            self.reply.abort()
        else:
            self.checking = False




class VersionBadge(QPushButton):
    """Always actionable; automatic detection changes appearance only."""

    def __init__(self, version: str, parent=None, *, controller=None):
        super().__init__(parent)
        self.version = version
        self.controller = controller or UpdateController(version, self)
        self.dialog = None
        self.setObjectName("VippVersionBadge")
        self.setCursor(Qt.PointingHandCursor)
        self.setAutoDefault(False)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._context_menu)
        self.clicked.connect(self.show_updates)
        self.controller.changed.connect(self.refresh)
        self.refresh()

    def refresh(self):
        available = self.controller.available
        self.setProperty("updateAvailable", available)
        self.setText(f"VIPP {self.version}" + (" ↑" if available else ""))
        status = (
            f"Update available: VIPP {self.controller.latest.version}. "
            if available
            else ""
        )
        self.setToolTip(
            status + "Click for updates. Right-click for release notes or to check now."
        )
        self.setAccessibleName(status + f"VIPP {self.version}. Open update dialog")
        self.refresh_theme()

    def refresh_theme(self, palette=None):
        if getattr(self, "_applying_theme", False):
            return
        self._applying_theme = True
        colors = theme_colors(palette or self.palette())
        if self.controller.available:
            foreground, background, border = (
                colors.info.foreground,
                colors.info.surface,
                colors.info.border,
            )
        else:
            foreground, background, border = (
                colors.muted_text,
                colors.alternate_surface,
                colors.border,
            )
        self.setStyleSheet(
            "QPushButton#VippVersionBadge { font-size: 11px; font-weight: 600;"
            f" color: {foreground.name()}; background: {background.name()};"
            f" border: 1px solid {border.name()};"
            " border-radius: 8px; padding: 4px 8px; }"
            "QPushButton#VippVersionBadge:hover, QPushButton#VippVersionBadge:focus {"
            f" border: 1px solid {colors.info.accent.name()};"
            f" color: {colors.text.name()}; }}"
        )
        self._applying_theme = False

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
        ) and hasattr(self, "controller"):
            self.refresh_theme()

    def show_updates(self, *, check=False):
        if self.dialog is None:
            self.dialog = UpdateDialog(self.controller, self.window())
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        if check or not self.controller.checked_this_session:
            self.controller.check()

    def make_context_menu(self):
        menu = QMenu(self)
        action = menu.addAction("Release notes on GitHub")
        action.triggered.connect(
            lambda: QDesktopServices.openUrl(QUrl(current_release_url(self.version)))
        )
        action = menu.addAction("Check for updates…")
        action.triggered.connect(lambda: self.show_updates(check=True))
        return menu

    def _context_menu(self, position):
        menu = self.make_context_menu()
        menu.exec_(self.mapToGlobal(position))
        menu.deleteLater()

    def shutdown(self):
        self.controller.shutdown()
        if self.dialog is not None:
            self.dialog.shutdown()
