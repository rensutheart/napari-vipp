"""Explicit intent and software-version review before opening a shared run."""

from __future__ import annotations

from qtpy.QtCore import QEvent, Qt, QUrl
from qtpy.QtGui import QDesktopServices
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QStyle,
    QVBoxLayout,
)

from napari_vipp.core.reproducibility_install import (
    INSTALLATION_GUIDE_URL,
    installation_guidance,
)
from napari_vipp.core.reproduction import versions_match
from napari_vipp.ui.palette_roles import blend_colors, theme_colors


def _label(text: str, *, bold: bool = False) -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    if bold:
        font = label.font()
        font.setBold(True)
        label.setFont(font)
    return label


class _ChoiceCard(QFrame):
    """One radio choice with an aligned explanation and a generous click target."""

    def __init__(self, title: str, description: str):
        super().__init__()
        self.setObjectName("ReproductionChoiceCard")
        self.setCursor(Qt.PointingHandCursor)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(5)
        self.radio = QRadioButton(title)
        self.radio.setObjectName("ReproductionChoiceTitle")
        self.radio.setAccessibleDescription(description)
        font = self.radio.font()
        font.setBold(True)
        self.radio.setFont(font)
        layout.addWidget(self.radio)
        self.description = _label(description)
        self.description.setObjectName("ReproductionChoiceDescription")
        self.description.installEventFilter(self)
        explanation = QHBoxLayout()
        explanation.setContentsMargins(0, 0, 0, 0)
        explanation.setSpacing(0)
        # Match the native indicator and text spacing, including high DPI.
        style = self.radio.style()
        inset = style.pixelMetric(
            QStyle.PM_ExclusiveIndicatorWidth
        ) + style.pixelMetric(QStyle.PM_RadioButtonLabelSpacing)
        explanation.addSpacing(inset)
        explanation.addWidget(self.description)
        layout.addLayout(explanation)

    def _select(self):
        self.radio.setFocus(Qt.MouseFocusReason)
        self.radio.setChecked(True)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton and self.rect().contains(event.pos()):
            self._select()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def eventFilter(self, watched, event):  # noqa: N802
        if (
            watched is self.description
            and event.type() == QEvent.MouseButtonRelease
            and event.button() == Qt.LeftButton
        ):
            self._select()
            return True
        return super().eventFilter(watched, event)

    def refresh_theme(self, colors):
        selected = self.radio.isChecked()
        self.setProperty("selected", selected)
        fill = colors.info.surface if selected else colors.alternate_surface
        border = colors.info.accent if selected else colors.border
        self.setStyleSheet(
            "QFrame#ReproductionChoiceCard {"
            f"background: {fill.name()}; border: 2px solid {border.name()};"
            "border-radius: 8px; }"
            "QFrame#ReproductionChoiceCard:hover {"
            f"border-color: {colors.info.accent.name()}; }}"
            "QRadioButton#ReproductionChoiceTitle {"
            f"color: {colors.text.name()}; background: transparent;"
            "font-weight: 600; }"
            "QLabel#ReproductionChoiceDescription {"
            f"color: {colors.text.name()}; background: transparent; }}"
        )


class ReproductionOpenCancelled(Exception):
    """The user left the existing workspace unchanged."""


class ReproductionChoiceDialog(QDialog):
    """Opening a package never silently authorizes a different-version run."""

    def __init__(self, recorded_version: str, current_version: str, parent=None):
        super().__init__(parent)
        # A dialog is a window: opt into the host's font/palette propagation.
        self.setAttribute(Qt.WA_WindowPropagation, True)
        self.setWindowTitle("Open reproducibility workflow")
        self.setMinimumWidth(440)
        self.recorded_version = recorded_version
        self.current_version = current_version
        self.versions_match = versions_match(recorded_version, current_version)
        self._guidance = installation_guidance(
            {"run": {"packages": {"napari-vipp": recorded_version}}},
            recorded=True,
        )
        self._versions_known = versions_match(
            recorded_version, recorded_version
        ) and versions_match(current_version, current_version)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        self.heading = _label("Choose how to use this workflow", bold=True)
        self.heading.setObjectName("ReproductionHeading")
        font = self.heading.font()
        font.setPointSizeF(max(10, font.pointSizeF()) + 2)
        self._heading_point_size = font.pointSizeF()
        self.heading.setFont(font)
        layout.addWidget(self.heading)

        self.reproduce_card = _ChoiceCard(
            "Reproduce original run",
            "Use the original images. VIPP checks that their contents match "
            "before you can run the batch.",
        )
        self.new_data_card = _ChoiceCard(
            "Use workflow on new data",
            "Use the same analysis on different images. No comparison with "
            "the original inputs; normal checks still apply.",
        )
        self.reproduce_radio = self.reproduce_card.radio
        self.new_data_radio = self.new_data_card.radio
        # Cards have separate parents, so native autoExclusive alone is insufficient.
        self.choices = QButtonGroup(self)
        self.choices.addButton(self.reproduce_radio)
        self.choices.addButton(self.new_data_radio)
        self.reproduce_radio.installEventFilter(self)
        self.new_data_radio.installEventFilter(self)
        self.reproduce_radio.setChecked(True)
        layout.addWidget(self.reproduce_card)
        layout.addWidget(self.new_data_card)
        layout.addSpacing(2)

        self.version_panel = QFrame()
        self.version_panel.setObjectName("ReproductionVersionPanel")
        version_layout = QVBoxLayout(self.version_panel)
        version_layout.setContentsMargins(14, 12, 14, 12)
        version_layout.setSpacing(7)
        self.version_heading = _label("", bold=True)
        self.version_heading.setObjectName("ReproductionVersionHeading")
        version_layout.addWidget(self.version_heading)
        self.version_label = _label(
            f"Original run: VIPP {recorded_version or 'version not recorded'}\n"
            f"Installed here: VIPP {current_version or 'version not recorded'}"
        )
        version_layout.addWidget(self.version_label)
        self.version_notice = _label("")
        version_layout.addWidget(self.version_notice)

        self.release_button = QPushButton()
        self.release_button.setObjectName("ReproductionHelpLink")
        self.release_button.setFlat(True)
        exact_release = "/releases/tag/" in self._guidance["url"]
        release_version = self._guidance["version"]
        self.release_button.setText(
            f"View VIPP {release_version} release on GitHub"
            if exact_release and len(release_version) < 24
            else "View original VIPP release on GitHub"
            if exact_release
            else "Browse VIPP releases on GitHub"
        )
        self.release_button.setToolTip(
            "Opens the original release page in your browser, where you can choose "
            "an installer. Nothing is downloaded or installed by this button."
            if exact_release
            else "Opens VIPP's release list in your browser. Ask the author for "
            "the original version or development build."
        )
        self.release_button.clicked.connect(self._open_release)
        self.install_button = QPushButton("Read the VIPP installation guide")
        self.install_button.setObjectName("ReproductionHelpLink")
        self.install_button.setFlat(True)
        self.install_button.setToolTip(
            "Opens the official instructions for installing VIPP on your "
            "operating system. This does not change your installation."
        )
        self.install_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl(INSTALLATION_GUIDE_URL))
        )
        for button in (self.release_button, self.install_button):
            button.setCursor(Qt.PointingHandCursor)
            button.setAutoDefault(False)
            version_layout.addWidget(button, alignment=Qt.AlignLeft)
        self.release_help = _label(
            "Links open your browser; nothing is downloaded or installed."
        )
        self.release_help.setObjectName("ReproductionQuietText")
        version_layout.addWidget(self.release_help)
        self.override_checkbox = QCheckBox("Use this version anyway")
        self.override_checkbox.setToolTip(
            "This explicit exception is recorded for this VIPP version only. "
            "Input-hash checks remain required. This is not a verified "
            "same-version reproduction."
        )
        version_layout.addWidget(self.override_checkbox)
        layout.addWidget(self.version_panel)

        self.disclaimer = _label(
            "Matching inputs and VIPP versions do not guarantee identical results. "
            "Other software, hardware and analysis settings can also matter."
        )
        self.disclaimer.setObjectName("ReproductionQuietText")
        quiet_font = self.disclaimer.font()
        quiet_font.setPointSizeF(max(9, quiet_font.pointSizeF() - 1))
        self.disclaimer.setFont(quiet_font)
        self.release_help.setFont(quiet_font)
        layout.addWidget(self.disclaimer)

        self.footer_rule = QFrame()
        self.footer_rule.setFixedHeight(1)
        layout.addWidget(self.footer_rule)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        self.open_button = buttons.addButton(
            "Continue to batch setup", QDialogButtonBox.AcceptRole
        )
        self.open_button.setObjectName("ReproductionContinue")
        self.open_button.setDefault(True)
        self.open_button.setToolTip(
            "Choose input and output folders in Batch Setup. "
            "Continuing does not check files or start an analysis."
        )
        buttons.button(QDialogButtonBox.Cancel).setAutoDefault(False)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
        self.reproduce_radio.toggled.connect(self._refresh)
        self.new_data_radio.toggled.connect(self._refresh)
        self.override_checkbox.toggled.connect(self._refresh)
        self.ensurePolished()
        self._refresh()
        # Content sets the height; no large, empty stretch below the decision.
        self.resize(640, self.sizeHint().height())

    @property
    def mode(self):
        return "reproduce" if self.reproduce_radio.isChecked() else "new-data"

    @property
    def version_override_accepted(self):
        return (
            self.mode == "reproduce"
            and not self.versions_match
            and self.override_checkbox.isChecked()
        )

    def _refresh(self, *_args):
        reproduction = self.mode == "reproduce"
        needs_override = reproduction and not self.versions_match
        self.override_checkbox.setVisible(needs_override)
        self.open_button.setEnabled(
            not needs_override or self.override_checkbox.isChecked()
        )
        for control in (self.release_button, self.install_button, self.release_help):
            control.setVisible(needs_override)
        if self.versions_match:
            self.version_heading.setText("VIPP versions match")
            message = "The VIPP version matches the recorded run."
        else:
            self.version_heading.setText(
                "Different VIPP versions"
                if self._versions_known
                else "Version could not be verified"
            )
            if reproduction:
                message = (
                    "Install the original version, or accept the difference below. "
                    if "/releases/tag/" in self._guidance["url"]
                    else "Ask the author for the original build, or accept "
                    "this unverified version below. "
                )
                message += "This exception will be recorded."
            else:
                message = "Results may differ, but this does not block a new analysis."
        self.version_notice.setText(message)
        self.version_notice.setVisible(not self.versions_match)
        self.refresh_theme()

    def refresh_theme(self):
        if getattr(self, "_applying_theme", False):
            return
        self._applying_theme = True
        try:
            colors = theme_colors(self.palette())
            tone = colors.success if self.versions_match else colors.warning
            self.version_panel.setProperty(
                "status", "match" if self.versions_match else "warning"
            )
            highlight = colors.info.surface
            highlight_text = colors.info.foreground
            hover = blend_colors(highlight, colors.info.accent, 0.15)
            self.setStyleSheet(
                "QLabel#ReproductionHeading {"
                f"font-size: {self._heading_point_size:g}pt; font-weight: 600; }}"
                "QFrame#ReproductionVersionPanel {"
                f"background: {tone.surface.name()}; border-radius: 7px;"
                f"border-left: 3px solid {tone.accent.name()}; }}"
                "QFrame#ReproductionVersionPanel QLabel, "
                "QFrame#ReproductionVersionPanel QCheckBox {"
                f"color: {colors.text.name()}; background: transparent; }}"
                "QFrame#ReproductionVersionPanel QLabel#ReproductionVersionHeading {"
                f"color: {tone.foreground.name()}; font-weight: 600; }}"
                "QLabel#ReproductionQuietText, "
                "QFrame#ReproductionVersionPanel QLabel#ReproductionQuietText {"
                f"color: {colors.muted_text.name()}; background: transparent; }}"
                "QPushButton#ReproductionHelpLink {"
                f"color: {colors.text.name()}; background: transparent;"
                "border: none; padding: 2px 0; text-align: left;"
                "text-decoration: underline; }"
                "QPushButton#ReproductionHelpLink:hover {"
                f"color: {colors.info.foreground.name()}; }}"
                "QPushButton#ReproductionContinue {"
                f"background: {highlight.name()}; color: {highlight_text.name()};"
                f"border: 2px solid {colors.info.accent.name()}; border-radius: 4px;"
                "padding: 8px 14px; font-weight: 600; }"
                "QPushButton#ReproductionContinue:hover {"
                f"background: {hover.name()}; }}"
                "QPushButton#ReproductionContinue:disabled {"
                f"background: {colors.raised_surface.name()};"
                f"color: {colors.muted_text.name()};"
                f"border-color: {colors.border.name()}; }}"
            )
            self.footer_rule.setStyleSheet(f"background: {colors.border.name()};")
            self.reproduce_card.refresh_theme(colors)
            self.new_data_card.refresh_theme(colors)
        finally:
            self._applying_theme = False

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if hasattr(self, "footer_rule") and event.type() in (
            QEvent.PaletteChange,
            QEvent.ApplicationPaletteChange,
        ):
            self.refresh_theme()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        # Stylesheet-based hosts can finish propagating their palette at show.
        self.refresh_theme()

    def eventFilter(self, watched, event):  # noqa: N802
        # Qt's spatial radio navigation cannot cross the two card parents.
        if (
            watched in (self.reproduce_radio, self.new_data_radio)
            and event.type() == QEvent.KeyPress
            and event.key() in (Qt.Key_Up, Qt.Key_Down, Qt.Key_Left, Qt.Key_Right)
            and event.modifiers() == Qt.NoModifier
        ):
            other = (
                self.new_data_radio
                if watched is self.reproduce_radio
                else self.reproduce_radio
            )
            other.setFocus(Qt.TabFocusReason)
            other.setChecked(True)
            return True
        return super().eventFilter(watched, event)

    def _open_release(self):
        # Derive this link from a validated version, never a package-supplied URL.
        QDesktopServices.openUrl(QUrl(self._guidance["url"]))

    def accept(self):
        if self.open_button.isEnabled():
            super().accept()
