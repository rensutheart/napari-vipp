"""Quiet, bounded release checks and an explicitly opened update dialog.

Network replies, timers, and dialogs have Qt owners. No worker touches scientific
state and no path installs packages or executes a downloaded file.
"""

from __future__ import annotations

import json
import os
import platform
import time
from importlib.resources import files

from packaging.version import Version
from qtpy.QtCore import QEvent, QObject, QSettings, Qt, QTimer, QUrl, Signal
from qtpy.QtGui import QDesktopServices, QPixmap
from qtpy.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from qtpy.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMenu,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.updates import (
    MAX_RESPONSE_BYTES,
    RELEASES_API,
    RELEASES_URL,
    current_release_url,
    installer_asset,
    newest_release,
    parse_releases,
)
from napari_vipp.ui.palette_roles import theme_colors

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
        self.checked = False
        self._closed = False
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
        try:
            last = float(self.settings.value("updates/last-attempt", 0))
        except (TypeError, ValueError):
            last = 0
        if not 0 <= time.time() - last < CHECK_INTERVAL:
            self.check()

    def check(self):
        if self._closed or self.checking:
            return
        self.settings.setValue("updates/last-attempt", time.time())
        self.settings.sync()
        self.checking, self.error = True, ""
        self._data = bytearray()
        self._too_large = False
        request = QNetworkRequest(QUrl(RELEASES_API))
        request.setRawHeader(b"Accept", b"application/vnd.github+json")
        request.setRawHeader(b"User-Agent", b"napari-vipp-update-check")
        request.setAttribute(
            QNetworkRequest.Attribute.RedirectPolicyAttribute,
            QNetworkRequest.RedirectPolicy.ManualRedirectPolicy,
        )
        request.setTransferTimeout(15000)
        self.reply = self.manager.get(request)
        self.reply.readyRead.connect(self._read)
        self.reply.finished.connect(self._finished)
        self.deadline.start(20000)
        self.changed.emit()

    def _timeout(self):
        if self.reply is not None:
            self.reply.abort()

    def _read(self):
        if self.reply is None or self._too_large:
            return
        remaining = MAX_RESPONSE_BYTES - len(self._data)
        # PyQt6 may return None (not an empty QByteArray) after the final drain.
        self._data.extend(bytes(self.reply.read(remaining + 1) or b""))
        if len(self._data) > MAX_RESPONSE_BYTES:
            self._too_large = True
            self.reply.abort()

    def _finished(self):
        reply = self.reply
        if reply is None:
            return
        # readyRead normally drains the reply; include any final buffered bytes.
        self._read()
        if self.reply is None:  # abort() can synchronously emit finished again
            return
        status = reply.attribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute)
        try:
            if self._too_large:
                raise ValueError(
                    "The release response was too large. Try the GitHub release page."
                )
            if status in {403, 429}:
                raise ValueError(
                    "GitHub's request limit was reached. Please try again later."
                )
            if status != 200 or reply.error() != QNetworkReply.NetworkError.NoError:
                raise ValueError(
                    "Could not reach GitHub. Check your connection and try again."
                )
            releases = parse_releases(json.loads(bytes(self._data)))
            if not releases:
                raise ValueError(
                    "No published VIPP releases were returned. Try again later."
                )
            self.releases = releases
            self.checked = True
            self.settings.setValue(
                "updates/cache-v1", json.dumps([r.cache_record() for r in releases])
            )
            self.settings.setValue("updates/last-success", time.time())
            self.settings.sync()
        except (ValueError, TypeError, UnicodeError) as exc:
            self.error = str(exc)
        finally:
            self.deadline.stop()
            self.reply = None
            self.checking = False
            self._data.clear()
            reply.deleteLater()
        if not self._closed:
            self.changed.emit()

    def shutdown(self):
        self._closed = True
        self.start_timer.stop()
        self.timer.stop()
        self.deadline.stop()
        if self.reply is not None:
            self.reply.abort()


class _WrappedLabel(QLabel):
    """Reserve the actual wrapped height inside a resizable scroll area."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.PlainText)

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

    def _fit_height(self):
        if self.wordWrap():
            height = self.heightForWidth(max(1, self.width()))
            if height >= 0 and self.minimumHeight() != height:
                self.setMinimumHeight(height)


class UpdateDialog(QDialog):
    def __init__(self, controller: UpdateController, parent=None, *, open_url=None):
        super().__init__(parent)
        self.controller = controller
        self.open_url = open_url or (lambda url: QDesktopServices.openUrl(QUrl(url)))
        self.setWindowTitle("VIPP updates")
        self.setMinimumWidth(420)
        self.resize(560, 420)
        self._initial_size_fitted = False
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(self.scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        layout.setSpacing(10)
        layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)
        header = QHBoxLayout()
        header.setSpacing(16)
        self.logo = QLabel()
        self.logo.setAccessibleName("VIPP logo")
        self.logo.setFixedSize(185, 55)
        self.logo.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap()
        try:
            payload = files("napari_vipp").joinpath(
                "assets", "branding", "vipp-logo-dark.svg"
            ).read_bytes()
            pixmap.loadFromData(payload)
        except (FileNotFoundError, ModuleNotFoundError, OSError, TypeError):
            pass
        if pixmap.isNull():
            self.logo.setText("VIPP")
        else:
            ratio = self.devicePixelRatioF()
            pixmap = pixmap.scaledToWidth(
                round(185 * ratio), Qt.SmoothTransformation
            )
            pixmap.setDevicePixelRatio(ratio)
            self.logo.setPixmap(pixmap)
        header.addWidget(self.logo)
        self.heading = _WrappedLabel()
        self.heading.setStyleSheet("font-size: 16px; font-weight: 650;")
        self.heading.setWordWrap(True)
        header.addWidget(self.heading, 1)
        layout.addLayout(header)
        self.summary = _WrappedLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.PlainText)
        layout.addWidget(self.summary)
        self.instructions = _WrappedLabel()
        self.instructions.setWordWrap(True)
        self.instructions.setTextFormat(Qt.PlainText)
        layout.addWidget(self.instructions)
        row = QHBoxLayout()
        self.download = QPushButton("Download installer")
        self.download.clicked.connect(self._download)
        row.addWidget(self.download)
        self.checksums = QPushButton("Download checksums")
        self.checksums.clicked.connect(self._checksums)
        row.addWidget(self.checksums)
        self.download_row = row
        self.scroll.viewport().installEventFilter(self)
        self.notes = QPushButton("Release notes on GitHub")
        self.notes.clicked.connect(self._notes)
        layout.addLayout(row)
        layout.addWidget(self.notes, alignment=Qt.AlignLeft)
        self.manual = QPushButton("Installation / update instructions")
        self.manual.clicked.connect(
            lambda: self._open(
                "https://rensutheart.github.io/vipp-mkdocs/stable/getting-started/installation/"
            )
        )
        layout.addWidget(self.manual, alignment=Qt.AlignLeft)
        layout.addSpacing(6)
        self.prereleases = QCheckBox("Include pre-release versions")
        self.prereleases.setToolTip(
            "Include alpha, beta and release-candidate versions."
        )
        self.prereleases.setChecked(controller.include_prereleases)
        self.prereleases.toggled.connect(controller.set_prereleases)
        layout.addWidget(self.prereleases)
        self.automatic = QCheckBox("Check automatically once a day")
        self.automatic.setChecked(controller.automatic)
        self.automatic.toggled.connect(controller.set_automatic)
        layout.addWidget(self.automatic)
        privacy = _WrappedLabel(
            "Checks contact GitHub for public release information. No images, "
            "workflows, file paths or hardware details are sent. Installers and "
            "packages are never downloaded or installed automatically."
        )
        privacy.setWordWrap(True)
        layout.addWidget(privacy)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.check_button = buttons.addButton(
            "Check for updates", QDialogButtonBox.ActionRole
        )
        self.check_button.clicked.connect(controller.check)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.buttons = buttons
        controller.changed.connect(self.refresh)
        self.refresh()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if not self._initial_size_fitted:
            self._initial_size_fitted = True
            # Measure after host fonts and wrapped labels have their real widths.
            # Subsequent status updates preserve the user's chosen window size.
            QTimer.singleShot(0, self._fit_initial_size)

    def _fit_initial_size(self):
        if not self.isVisible():
            return
        self._reflow_downloads()
        body_layout = self.scroll.widget().layout()
        body_layout.activate()
        content_height = body_layout.totalHeightForWidth(self.scroll.viewport().width())
        if content_height < 0:
            content_height = body_layout.sizeHint().height()
        margins = self.layout().contentsMargins()
        height = (
            content_height + margins.top() + margins.bottom()
            + self.layout().spacing() + self.buttons.sizeHint().height()
            + 2 * self.scroll.frameWidth()
        )
        # Longer release guidance remains scrollable on a smaller display.
        self.resize(
            self.width(), min(height, self.screen().availableGeometry().height() - 80)
        )

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.scroll.viewport() and event.type() == QEvent.Resize:
            self._reflow_downloads()
        return super().eventFilter(watched, event)

    def _reflow_downloads(self):
        # Native font metrics can make two readable buttons wider than the
        # viewport. Stack them instead of clipping text or requiring sideways
        # scrolling; the viewport also accounts for the vertical scrollbar.
        required = (
            max(self.download.minimumWidth(), self.download.minimumSizeHint().width())
            + max(
                self.checksums.minimumWidth(), self.checksums.minimumSizeHint().width()
            )
            + max(0, self.download_row.spacing())
        )
        direction = (
            QBoxLayout.TopToBottom
            if required > self.scroll.viewport().width()
            else QBoxLayout.LeftToRight
        )
        if self.download_row.direction() != direction:
            self.download_row.setDirection(direction)

    def refresh(self):
        controller = self.controller
        latest = controller.latest
        if controller.checking:
            title = "Checking for updates…"
        elif controller.available:
            title = f"VIPP {latest.version} is available"
        elif controller.error:
            title = "Update check unavailable"
        elif controller.checked and latest:
            title = "You're up to date"
        elif controller.checked:
            title = "No stable release found"
        else:
            title = "Check for a newer VIPP version"
        self.heading.setText(title)
        detail = f"Installed: {controller.current_version}"
        if latest:
            channel = "pre-release" if latest.prerelease else "release"
            detail += f" · Latest {channel}: {latest.version}"
        if controller.error:
            detail += "\n" + controller.error
            if latest:
                detail += (
                    "\nVersion information above is from the last successful check."
                )
        self.summary.setText(detail)
        offer = (
            installer_asset(latest, platform.system(), platform.machine())
            if controller.available
            else None
        )
        self.download.setVisible(offer is not None)
        self.checksums.setVisible(offer is not None)
        if offer:
            self.instructions.setText(
                "1. Download the installer and its checksums from GitHub.\n"
                "2. Verify the download, save your workflows, and close VIPP/napari.\n"
                "3. Run setup and review the installation options.\n\n"
                "Using pip, conda or a source checkout? Follow your environment's "
                "update instructions instead. The installer may create a "
                "separate installation."
                + (
                    "\n\nThis alpha installer is unsigned. See the installation "
                    "guide for verification and platform security warnings."
                    if "-UNSIGNED" in offer[0]
                    else ""
                )
            )
        else:
            self.instructions.setText(
                "Save your workflows and close VIPP/napari before updating. "
                "Follow the installation guide for your installer, pip/conda "
                "environment, or source checkout."
                + (
                    "\n\nNo matching installer with checksums is available "
                    "for this platform in that release."
                    if controller.available
                    else ""
                )
            )
        self.check_button.setEnabled(not controller.checking)
        for checkbox, value in (
            (self.automatic, controller.automatic),
            (self.prereleases, controller.include_prereleases),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(value)
            checkbox.blockSignals(False)
        self._reflow_downloads()

    def _open(self, url):
        if not self.open_url(url):
            self.summary.setText(
                "The browser could not be opened. Copy this address:\n" + url
            )
            self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def _notes(self):
        latest = self.controller.latest
        self._open(latest.notes_url if latest else RELEASES_URL)

    def _download(self):
        self._open_asset(0)

    def _checksums(self):
        self._open_asset(1)

    def _open_asset(self, index):
        latest = self.controller.latest
        if latest and self.controller.available:
            offer = installer_asset(latest, platform.system(), platform.machine())
            if offer:
                self._open(latest.asset_url(offer[index]))


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
        if check or not self.controller.checked:
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
            self.dialog.close()
