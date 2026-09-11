"""Readable release status and an explicit, verified installer handoff."""

from __future__ import annotations

import platform
from datetime import datetime
from importlib.resources import files

from qtpy.QtCore import QEvent, Qt, QTimer, QUrl
from qtpy.QtGui import QDesktopServices, QPixmap
from qtpy.QtWidgets import (
    QBoxLayout,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLayout,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.update_install import (
    UpdateInstallError,
    installer_download_request,
    managed_update_target,
)
from napari_vipp.core.updates import RELEASES_URL, installer_asset
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.update_download import UpdateDownloadController


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


def _bold(label, *, scale=1):
    font = label.font()
    font.setBold(True)
    if scale != 1 and font.pointSizeF() > 0:
        font.setPointSizeF(font.pointSizeF() * scale)
    label.setFont(font)


class UpdateDialog(QDialog):
    def __init__(
        self,
        controller,
        parent=None,
        *,
        open_url=None,
        download_controller=None,
        target_provider=None,
    ):
        super().__init__(parent)
        self.controller = controller
        self.open_url = open_url or (lambda url: QDesktopServices.openUrl(QUrl(url)))
        self.target_provider = target_provider or managed_update_target
        self.transfer = download_controller or UpdateDownloadController(self)
        self._handoff_requested = False
        self._closing = False
        self._confirming_close = False
        self._initial_size_fitted = False
        self.setWindowTitle("VIPP updates")
        self.setMinimumWidth(420)
        self.resize(560, 520)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(20, 20, 20, 16)
        outer.setSpacing(16)
        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        body = QWidget(self.scroll)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 4, 0)
        layout.setSizeConstraint(QLayout.SetMinAndMaxSize)
        layout.setSpacing(14)
        layout.setAlignment(Qt.AlignTop)
        self.scroll.setWidget(body)
        outer.addWidget(self.scroll, 1)

        header = QHBoxLayout()
        header.setSpacing(12)
        self.logo = QLabel()
        self.logo.setAccessibleName("VIPP logo")
        self.logo.setFixedSize(40, 40)
        pixmap = QPixmap()
        try:
            pixmap.loadFromData(
                files("napari_vipp")
                .joinpath("assets", "branding", "vipp-mark.svg")
                .read_bytes()
            )
        except (FileNotFoundError, ModuleNotFoundError, OSError, TypeError):
            pass
        if pixmap.isNull():
            self.logo.setText("VIPP")
        else:
            ratio = self.devicePixelRatioF()
            pixmap = pixmap.scaledToWidth(round(40 * ratio), Qt.SmoothTransformation)
            pixmap.setDevicePixelRatio(ratio)
            self.logo.setPixmap(pixmap)
        header.addWidget(self.logo)
        title = QLabel("VIPP updates")
        _bold(title, scale=1.4)
        header.addWidget(title, 1)
        layout.addLayout(header)

        self.status = QFrame()
        self.status.setObjectName("UpdateStatus")
        status_layout = QVBoxLayout(self.status)
        status_layout.setContentsMargins(14, 12, 14, 12)
        status_layout.setSpacing(6)
        self.heading = _WrappedLabel()
        _bold(self.heading, scale=1.15)
        self.summary = _WrappedLabel()
        status_layout.addWidget(self.heading)
        status_layout.addWidget(self.summary)
        layout.addWidget(self.status)

        versions = QHBoxLayout()
        versions.setSpacing(16)
        for caption, attr in (("Installed", "installed"), ("Latest release", "latest")):
            column = QVBoxLayout()
            column.setSpacing(4)
            label = QLabel(caption)
            setattr(self, attr + "_caption", label)
            value = QLabel()
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            _bold(value, scale=1.3)
            setattr(self, attr + "_version", value)
            column.addWidget(label)
            column.addWidget(value)
            versions.addLayout(column, 1)
        layout.addLayout(versions)

        self.update_section = QWidget()
        update_layout = QVBoxLayout(self.update_section)
        update_layout.setContentsMargins(0, 0, 0, 0)
        update_layout.setSpacing(10)
        self.update_button = QPushButton("Download && open update")
        self.update_button.setAccessibleName("Download and open update")
        self.update_button.clicked.connect(self._start_update)
        _bold(self.update_button)
        update_layout.addWidget(self.update_button, alignment=Qt.AlignLeft)
        self.instructions = _WrappedLabel()
        update_layout.addWidget(self.instructions)
        layout.addWidget(self.update_section)

        self.progress_section = QWidget()
        progress_layout = QVBoxLayout(self.progress_section)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(6)
        self.progress_label = _WrappedLabel()
        _bold(self.progress_label)
        progress_layout.addWidget(self.progress_label)
        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        progress_row.addWidget(self.progress, 1)
        self.cancel_download = QPushButton("Cancel download")
        self.cancel_download.clicked.connect(self._cancel_download)
        progress_row.addWidget(self.cancel_download)
        progress_layout.addLayout(progress_row)
        self.transfer_detail = _WrappedLabel()
        progress_layout.addWidget(self.transfer_detail)
        layout.addWidget(self.progress_section)

        self.download = QPushButton("Download in browser")
        self.download.clicked.connect(lambda: self._open_asset(0))
        self.checksums = QPushButton("Checksums")
        self.checksums.setToolTip("Open the published SHA-256 checksums on GitHub.")
        self.checksums.clicked.connect(lambda: self._open_asset(1))
        self.download_row = QHBoxLayout()
        self.download_row.addWidget(self.download)
        self.download_row.addWidget(self.checksums)
        self.browser_toggle = QPushButton("Other download options")
        self.browser_toggle.setCheckable(True)
        self.browser_toggle.toggled.connect(lambda _: self.refresh())
        layout.addWidget(self.browser_toggle, alignment=Qt.AlignLeft)
        layout.addLayout(self.download_row)

        self.notes = QPushButton("Release notes ↗")
        self.notes.clicked.connect(self._notes)
        self.manual = QPushButton("Update help ↗")
        self.manual.clicked.connect(
            lambda: self._open(
                "https://rensutheart.github.io/vipp-mkdocs/stable/getting-started/updating/"
            )
        )
        self.links_row = QHBoxLayout()
        self.links_row.addWidget(self.notes)
        self.links_row.addWidget(self.manual)
        layout.addLayout(self.links_row)

        self.technical_toggle = QPushButton("Show connection details")
        self.technical_toggle.setCheckable(True)
        self.technical_toggle.toggled.connect(self._show_details)
        layout.addWidget(self.technical_toggle, alignment=Qt.AlignLeft)
        self.technical_details = _WrappedLabel()
        self.technical_details.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.technical_details.hide()
        layout.addWidget(self.technical_details)

        self.separator = QFrame()
        self.separator.setFrameShape(QFrame.HLine)
        layout.addWidget(self.separator)
        self.preferences = QWidget()
        prefs = QVBoxLayout(self.preferences)
        prefs.setContentsMargins(0, 0, 0, 0)
        prefs.setSpacing(8)
        self.automatic = QCheckBox("Check for updates on startup")
        self.automatic.setToolTip(
            "Check on every launch, and daily while VIPP stays open. "
            "An administrator can disable automatic checks."
        )
        self.automatic.toggled.connect(controller.set_automatic)
        prefs.addWidget(self.automatic)
        self.prereleases = QCheckBox("Include pre-release versions")
        self.prereleases.setToolTip(
            "Include alpha, beta and release-candidate versions."
        )
        self.prereleases.toggled.connect(controller.set_prereleases)
        prefs.addWidget(self.prereleases)
        layout.addWidget(self.preferences)
        self.privacy = _WrappedLabel(
            "Checks contact GitHub. No images, workflows or file paths are sent. "
            "Downloading and opening setup requires your click."
        )
        layout.addWidget(self.privacy)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.check_button = buttons.addButton(
            "Check again", QDialogButtonBox.ActionRole
        )
        self.check_button.clicked.connect(controller.check)
        buttons.rejected.connect(self.reject)
        outer.addWidget(buttons)
        self.buttons = buttons
        self.scroll.viewport().installEventFilter(self)
        controller.changed.connect(self.refresh)
        self.transfer.changed.connect(self._transfer_changed)
        self.refresh()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self.refresh()
        if not self._initial_size_fitted:
            self._initial_size_fitted = True
            QTimer.singleShot(0, self._fit_initial_size)

    def _fit_initial_size(self):
        if not self.isVisible():
            return
        self._reflow_downloads()
        body_layout = self.scroll.widget().layout()
        body_layout.activate()
        height = body_layout.totalHeightForWidth(self.scroll.viewport().width())
        if height < 0:
            height = body_layout.sizeHint().height()
        margins = self.layout().contentsMargins()
        height += (
            margins.top()
            + margins.bottom()
            + self.layout().spacing()
            + self.buttons.sizeHint().height()
            + 2 * self.scroll.frameWidth()
        )
        self.resize(
            self.width(), min(height, self.screen().availableGeometry().height() - 80)
        )

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.scroll.viewport() and event.type() == QEvent.Resize:
            self._reflow_downloads()
        return super().eventFilter(watched, event)

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in (QEvent.PaletteChange, QEvent.ApplicationPaletteChange):
            if hasattr(self, "buttons"):
                self.refresh()

    def _reflow_downloads(self):
        for row, buttons in (
            (self.download_row, (self.download, self.checksums)),
            (self.links_row, (self.notes, self.manual)),
        ):
            required = sum(
                max(b.minimumWidth(), b.minimumSizeHint().width()) for b in buttons
            ) + max(0, row.spacing())
            row.setDirection(
                QBoxLayout.TopToBottom
                if required > (self.scroll.viewport().width() - 4)
                else QBoxLayout.LeftToRight
            )

    def refresh(self):
        c = self.controller
        latest = c.latest
        fresh = c.checked_this_session and not c.error and not c.checking
        tone = "info"
        if c.checking:
            title, detail = "Checking GitHub…", "Looking for the newest VIPP release."
        elif c.error:
            title, detail, tone = "Couldn’t check for updates", c.error, "warning"
            if latest:
                detail += " The version below is from a previous check."
        elif not fresh:
            title = "Ready to check for updates"
            detail = "Check again to get the latest release information."
        elif c.available:
            title, detail = (
                "An update is available",
                "See what’s new in the release notes.",
            )
        elif latest:
            title, detail, tone = (
                "You’re up to date",
                "No newer release was found for your settings.",
                "success",
            )
        else:
            title, detail = (
                "No stable release found",
                "Enable pre-release versions to see alpha builds.",
            )
        if self.transfer.phase == "opened":
            title = "Setup is open"
            detail = "Review setup to finish updating, then restart VIPP."
        self.heading.setText(title)
        if c.last_success and not c.checking and self.transfer.phase != "opened":
            stamp = datetime.fromtimestamp(c.last_success).strftime("%d %b %Y, %H:%M")
            detail += f"\nLast successful check: {stamp}"
        self.summary.setText(detail)
        self.installed_version.setText(str(c.current_version))
        self.latest_caption.setText("Latest release" if fresh else "Last known release")
        self.latest_version.setText(str(latest.version) if latest else "—")
        offer = (
            installer_asset(latest, platform.system(), platform.machine())
            if c.available
            else None
        )
        target = self.target_provider() if offer else None
        managed = target is not None
        self.update_section.setVisible(
            (bool(offer) or c.available)
            and not self.transfer.busy and self.transfer.phase != "opened"
        )
        self.update_button.setVisible(bool(offer) and managed)
        self.update_button.setEnabled(
            fresh and not self.transfer.busy and self.transfer.phase != "opened"
        )
        self.update_button.setToolTip(
            "Download, verify SHA-256, then open setup. "
            "Installation still needs your approval."
            if fresh
            else "Check again successfully before downloading an update."
        )
        if managed:
            instructions = (
                "VIPP will verify the download and open setup for this installation. "
                "Save your work first. Review setup, then restart VIPP after updating."
            )
        elif platform.system() == "Windows":
            instructions = (
                "This session isn’t a managed VIPP desktop installation. "
                "Use Update help for pip, conda or source installs. "
                "The desktop installer may create a separate installation."
            )
        else:
            instructions = (
                "Download the installer, save your work and close VIPP "
                "before running setup. "
                "For pip or conda, follow Update help instead."
            )
        if offer and "-UNSIGNED" in offer[0]:
            instructions += (
                " This installer is unsigned; your system may ask for approval."
            )
        elif c.available and not offer:
            instructions += " No matching installer with checksums is published."
        self.instructions.setText(instructions)
        self.browser_toggle.setVisible(bool(offer) and managed)
        show_browser = offer is not None and (
            not managed or self.browser_toggle.isChecked()
        )
        self.download.setVisible(show_browser)
        self.checksums.setVisible(show_browser)
        self.download.setEnabled(not self.transfer.busy)
        self.check_button.setEnabled(not c.checking and not self.transfer.busy)
        self.prereleases.setEnabled(not self.transfer.busy)
        for checkbox, value in (
            (self.automatic, c.automatic),
            (self.prereleases, c.include_prereleases),
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(value)
            checkbox.blockSignals(False)
        self.technical_toggle.setVisible(bool(c.error_details))
        self.technical_details.setText(c.error_details)
        self.technical_details.setVisible(
            bool(c.error_details) and self.technical_toggle.isChecked()
        )
        colors = theme_colors(self.palette())
        role = getattr(colors, tone)
        self.status.setStyleSheet(
            f"QFrame#UpdateStatus {{ background: {role.surface.name()}; "
            f"border: 1px solid {role.border.name()}; border-radius: 8px; }}"
        )
        for label in (self.heading, self.summary):
            label.setStyleSheet(
                f"color: {role.foreground.name()}; background: transparent;"
            )
        for label in (self.privacy, self.installed_caption, self.latest_caption):
            label.setStyleSheet(f"color: {colors.muted_text.name()};")
        self.separator.setStyleSheet(f"background: {colors.border.name()};")
        self.separator.setFixedHeight(1)
        self.update_button.setStyleSheet(
            "QPushButton { font-weight: 600; padding: 9px 14px; "
            f"border: 1px solid {colors.info.accent.name()}; border-radius: 4px; }}"
            "QPushButton:enabled {"
            f"background: {colors.info.surface.name()}; "
            f"color: {colors.info.foreground.name()}; }}"
        )
        self._render_transfer()
        self._reflow_downloads()

    def _show_details(self, checked):
        self.technical_toggle.setText(
            "Hide connection details" if checked else "Show connection details"
        )
        self.technical_details.setVisible(
            checked and bool(self.controller.error_details)
        )

    def _start_update(self):
        c = self.controller
        if (
            self.transfer.busy
            or self.transfer.phase == "opened"
            or c.checking
            or c.error
            or not c.checked_this_session
        ):
            return
        if not c.available:
            return
        try:
            target = self.target_provider()
            if target is None:
                raise ValueError(
                    "This session is no longer a managed desktop installation."
                )
            request = installer_download_request(c.latest, target)
        except (UpdateInstallError, ValueError, OSError) as exc:
            self.summary.setText(str(exc))
            return
        self._handoff_requested = True
        self.transfer.start(request)

    def _transfer_changed(self):
        # Only completion of this explicit click can open setup, never a release
        # check, cached result, dialog reopening, or duplicate worker signal.
        if (
            self.transfer.phase == "ready"
            and self._handoff_requested
            and not self._closing
            and not self._confirming_close
        ):
            self._handoff_requested = False
            self.transfer.open_installer()
        self.refresh()

    def _render_transfer(self):
        t = self.transfer
        phase = t.phase
        self.progress_section.setVisible(phase not in ("", "idle"))
        self.progress.setVisible(t.busy)
        self.cancel_download.setVisible(t.busy)
        titles = {
            "downloading": "Downloading update…",
            "verifying": "Verifying download…",
            "ready": "Download verified",
            "opened": "Setup is open",
            "cancelled": "Download cancelled",
            "failed": "Update couldn’t be prepared",
        }
        self.progress_label.setText(titles.get(phase, "Preparing update…"))
        if t.busy and t.total_bytes:
            self.progress.setRange(0, 1000)
            self.progress.setValue(
                min(1000, int(1000 * t.received_bytes / t.total_bytes))
            )
        else:
            self.progress.setRange(0, 0)
        detail = ""
        if phase == "opened":
            detail = (
                "The download passed its checksum check. Review the setup window "
                "to continue. Your current session is still open; "
                "save your work before restarting VIPP."
            )
        elif phase == "failed":
            detail = t.error + " No installation was changed by the updater."
        elif phase == "cancelled":
            detail = "No installer was opened. You can try again when ready."
        elif t.busy:
            detail = f"{t.received_bytes / 1048576:.1f} MB downloaded"
            if t.total_bytes:
                detail += f" of {t.total_bytes / 1048576:.1f} MB"
            detail += ". You can keep working while VIPP prepares setup."
        self.transfer_detail.setText(detail)

    def _cancel_download(self):
        self._handoff_requested = False
        self.transfer.cancel()

    def _can_close(self):
        if self.transfer.busy:
            # The question runs a nested event loop. Completion may arrive
            # while the user is deciding; never open setup behind the prompt.
            self._confirming_close = True
            try:
                answer = QMessageBox.question(
                    self,
                    "Cancel the update download?",
                    "Closing this window will cancel the download. "
                    "Your installation won’t change.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
            finally:
                self._confirming_close = False
            if answer != QMessageBox.Yes:
                self._transfer_changed()
                return False
            self._cancel_download()
        return True

    def reject(self):
        if self._can_close():
            super().reject()

    def closeEvent(self, event):  # noqa: N802
        if self._can_close():
            event.accept()
        else:
            event.ignore()

    def shutdown(self):
        self._closing = True
        self._handoff_requested = False
        self.transfer.shutdown()
        self.hide()

    def _open(self, url):
        if not self.open_url(url):
            self.summary.setText(
                "The browser could not be opened. Copy this address:\n" + url
            )
            self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)

    def _notes(self):
        latest = self.controller.latest
        self._open(latest.notes_url if latest else RELEASES_URL)

    def _open_asset(self, index):
        latest = self.controller.latest
        if latest and self.controller.available:
            offer = installer_asset(latest, platform.system(), platform.machine())
            if offer:
                self._open(latest.asset_url(offer[index]))
