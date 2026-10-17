"""Presentation-only chrome for the staged workflow assistant."""

from __future__ import annotations

import math

from qtpy.QtCore import QEvent, QRectF, QSize, Qt, QTimer
from qtpy.QtGui import QColor, QIcon, QPainter, QPalette, QPen, QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QPlainTextEdit,
    QScrollArea,
    QSizePolicy,
    QTabBar,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
)

from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.iconography import interface_icon
from napari_vipp.ui.palette_roles import blend_colors, theme_colors
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


class _StageTabBar(QTabBar):
    def tabSizeHint(self, index):  # noqa: N802
        hint = super().tabSizeHint(index)
        if self.count():
            hint.setWidth(max(1, self.width() // self.count()))
        hint.setHeight(max(44, self.fontMetrics().height() + 24))
        return hint

    def minimumTabSizeHint(self, index):  # noqa: N802
        return self.tabSizeHint(index)


class AssistantStageTabs(QTabWidget):
    """Keep native tab navigation; allocate the whole strip equally."""

    def __init__(self, parent=None):
        super().__init__(parent)
        bar = _StageTabBar(self)
        self.setTabBar(bar)
        bar.setObjectName("AssistantStageBar")
        self.setDocumentMode(True)
        bar.setExpanding(True)
        bar.setUsesScrollButtons(False)
        self.setIconSize(QSize(18, 18))

    def resizeEvent(self, event):  # noqa: N802
        self.tabBar().setFixedWidth(self.width())
        super().resizeEvent(event)


class AssistantButton(ToolbarCommandButton):
    """Assistant-local icon gap; labels and action/accessibility text stay literal."""

    _ICON_TEXT_SPACER = "\u2002"

    def _toolbar_style_option(self):
        option = super()._toolbar_style_option()
        if (
            option.text
            and not option.icon.isNull()
            and option.direction == Qt.RightToLeft
        ):
            option.text = self.text() + self._ICON_TEXT_SPACER
        return option


class AssistantDisclosureButton(AssistantButton):
    """Passive, keyboard-accessible section header using VIPP's chevrons."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("AssistantDisclosure")
        self.setCheckable(True)
        self.setAutoDefault(False)
        self.setProperty("assistantAction", "link")
        self.setIconSize(QSize(18, 18))
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMinimumHeight(max(34, self.fontMetrics().height() + 16))
        self.toggled.connect(self.refresh_disclosure)
        self.refresh_disclosure()

    def refresh_disclosure(self, *_):
        self.setIcon(
            interface_icon(
                "chevron-down" if self.isChecked() else "chevron-right",
                self.palette(),
                18,
            )
        )
        action = "Collapse" if self.isChecked() else "Expand"
        self.setAccessibleName(f"{action} {self.text()}")
        self.setToolTip(f"{action} {self.text().lower()}")

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in {
            QEvent.FontChange,
            QEvent.PaletteChange,
            QEvent.StyleChange,
        }:
            self.setMinimumHeight(max(34, self.fontMetrics().height() + 16))
            self.refresh_disclosure()

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in {Qt.Key_Return, Qt.Key_Enter}:
            self.click()
            event.accept()
        else:
            super().keyPressEvent(event)


class AssistantSendButton(AssistantButton):
    """A send glyph becomes a local spinner; it never owns request authority."""

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self._sending = False
        self._idle_icon = QIcon()
        self._spinner_angle = 0
        self._spinner_timer = QTimer(self)
        self._spinner_timer.setInterval(80)
        self._spinner_timer.timeout.connect(self._advance_spinner)
        self.setMinimumHeight(max(34, self.fontMetrics().height() + 16))

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in {QEvent.FontChange, QEvent.StyleChange}:
            self.setMinimumHeight(max(34, self.fontMetrics().height() + 16))

    def set_idle_icon(self, icon):
        self._idle_icon = QIcon(icon)
        if self._sending:
            self._paint_spinner()
        else:
            self.setIcon(self._idle_icon)

    def set_sending(self, sending):
        self._sending = bool(sending)
        self.setProperty("sending", self._sending)
        if self._sending:
            self._paint_spinner()
            if self.isVisible():
                self._spinner_timer.start()
        else:
            self._spinner_timer.stop()
            self.setIcon(self._idle_icon)

    def _advance_spinner(self):
        self._spinner_angle = (self._spinner_angle + 30) % 360
        self._paint_spinner()

    def _paint_spinner(self):
        icon = QIcon()
        for mode, group in (
            (QIcon.Normal, QPalette.Active),
            (QIcon.Disabled, QPalette.Disabled),
        ):
            pixmap = QPixmap(24, 24)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(self.palette().color(group, QPalette.ButtonText), 2.2)
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawArc(QRectF(4, 4, 16, 16), self._spinner_angle * 16, 250 * 16)
            painter.end()
            icon.addPixmap(pixmap, mode)
        self.setIcon(icon)

    def hideEvent(self, event):  # noqa: N802
        self._spinner_timer.stop()
        super().hideEvent(event)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        if self._sending:
            self._spinner_timer.start()


class DiscussionTranscript(QTextBrowser):
    """Read-only message history sized to its document, not an inner scroller.

    The Discussion page owns scrolling. Deferred fitting lets Qt finish word
    wrapping after a page, font or width change without recursive layout edits.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setOpenExternalLinks(False)
        self.setOpenLinks(False)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_document)
        self.document().documentLayout().documentSizeChanged.connect(self.queue_fit)

    def queue_fit(self, *_):
        self._fit_timer.start(0)

    def _fit_document(self):
        self.document().setTextWidth(max(1, self.viewport().width()))
        chrome = self.height() - self.viewport().height()
        height = max(48, math.ceil(self.document().size().height()) + chrome + 2)
        if self.height() != height:
            self.setFixedHeight(height)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.queue_fit()

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self.queue_fit()

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() in {QEvent.FontChange, QEvent.StyleChange} and hasattr(
            self, "_fit_timer"
        ):
            self.queue_fit()


class ResponsiveAssistantEditor(QPlainTextEdit):
    """Give a short editor bounded space without changing its text or cursor."""

    def __init__(self, *, minimum, maximum, lines, chrome, ratio, parent=None):
        super().__init__(parent)
        self._compact_minimum = minimum
        self._compact_maximum = maximum
        self._minimum_lines = lines
        self._editor_chrome = chrome
        self._height_ratio = ratio
        self._height_anchor = None
        self._height_timer = QTimer(self)
        self._height_timer.setSingleShot(True)
        self._height_timer.timeout.connect(self._fit_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._fit_height()

    def bind_height_to(self, widget):
        self._height_anchor = widget
        widget.installEventFilter(self)
        self._height_timer.start(0)

    def height_bounds(self):
        minimum = max(
            self._compact_minimum,
            self.fontMetrics().lineSpacing() * self._minimum_lines
            + self._editor_chrome,
        )
        return minimum, max(minimum, self._compact_maximum)

    def _fit_height(self):
        minimum, maximum = self.height_bounds()
        available = self._height_anchor.height() if self._height_anchor else 0
        target = max(minimum, min(maximum, round(available * self._height_ratio)))
        if self.height() != target or self.minimumHeight() != target:
            self.setFixedHeight(target)

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self._height_anchor and event.type() == QEvent.Resize:
            self._height_timer.start(0)
        return super().eventFilter(watched, event)

    def changeEvent(self, event):  # noqa: N802
        super().changeEvent(event)
        if event.type() == QEvent.FontChange and hasattr(self, "_height_timer"):
            self._height_timer.start(0)


class SharingPreviewDialog(QDialog):
    """Fit short request content; bound long content to a clear scrolling card."""

    def __init__(self, parent, palette):
        super().__init__(parent)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_content)
        self.setObjectName("SharedRequestPreview")
        self.setWindowTitle("Preview what is shared — nothing has been sent")
        self.resize(700, 420)
        self.setMinimumWidth(360)
        self.setStyleSheet(sharing_preview_stylesheet(palette))
        outer = QVBoxLayout(self)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)
        self.scroll = QScrollArea(self)
        self.scroll.setObjectName("SharedRequestScroll")
        self.scroll.viewport().setObjectName("SharedRequestViewport")
        self.scroll.setWidgetResizable(True)
        outer.addWidget(self.scroll, 1)
        self.close_button = AssistantButton("Close preview")
        self.close_button.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addStretch()
        add_dialog_buttons(footer, actions=(), dismiss=self.close_button)
        outer.addLayout(footer)

    def set_content(self, panel):
        panel.setObjectName("SharedRequestContent")
        panel.installEventFilter(self)
        self.scroll.setWidget(panel)
        self.queue_fit()

    def queue_fit(self, *_):
        self._fit_timer.start(0)

    def _fit_content(self):
        panel = self.scroll.widget()
        if panel is None:
            return
        self.ensurePolished()
        self.layout().activate()
        width = max(1, self.scroll.viewport().width())
        height = panel.heightForWidth(width)
        if height < 0:
            height = panel.sizeHint().height()
        chrome = 32 + 12 + self.close_button.sizeHint().height() + 4
        screen = self.screen()
        cap = min(740, screen.availableGeometry().height() - 80) if screen else 740
        target = min(max(240, height + chrome), max(240, cap))
        if self.height() != target:
            self.resize(self.width(), target)

    def eventFilter(self, watched, event):  # noqa: N802
        if watched is self.scroll.widget() and event.type() == QEvent.LayoutRequest:
            self.queue_fit()
        return super().eventFilter(watched, event)

    def showEvent(self, event):  # noqa: N802
        super().showEvent(event)
        self.queue_fit()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if event.oldSize().width() != event.size().width():
            self.queue_fit()


def sharing_preview_stylesheet(palette):
    colors, panel, raised, input_surface, _ = assistant_surfaces(palette)
    border = blend_colors(panel, colors.text, 0.28).name()
    return f"""
QDialog#SharedRequestPreview {{
    background: {panel.name()}; color: {colors.text.name()};
}}
QScrollArea#SharedRequestScroll {{
    background: {raised.name()}; border: 1px solid {border}; border-radius: 6px;
}}
QWidget#SharedRequestViewport, QWidget#SharedRequestContent {{
    background: {raised.name()}; color: {colors.text.name()};
}}
QDialog#SharedRequestPreview QLabel, QDialog#SharedRequestPreview QCheckBox {{
    background: transparent; color: {colors.text.name()};
}}
QPlainTextEdit#SharedRequestJSON {{
    background: {input_surface.name()}; color: {colors.text.name()};
    border: 1px solid {border}; border-radius: 5px; padding: 8px;
}}
"""


def assistant_surfaces(palette):
    """Derive product surfaces from the active theme, not a fixed dark skin."""
    colors = theme_colors(palette)
    panel = QColor(palette.color(QPalette.Window))
    raised = blend_colors(panel, colors.text, 0.055)
    input_surface = blend_colors(panel, colors.surface, 0.55)
    user_surface = blend_colors(panel, colors.info.accent, 0.42)
    return colors, panel, raised, input_surface, user_surface


def assistant_connection_pixmap(tone, palette):
    """A small status mark; it reports UI state, not model availability."""
    colors = theme_colors(palette)
    pixmap = QPixmap(24, 24)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    pen = QPen(getattr(colors, tone, colors.info).foreground, 1.8)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.drawEllipse(QRectF(3, 3, 18, 18))
    if tone == "success":
        painter.drawLine(7, 12, 10, 15)
        painter.drawLine(10, 15, 17, 8)
    else:
        painter.drawLine(12, 7, 12, 13)
        painter.drawPoint(12, 17)
    painter.end()
    return pixmap


def assistant_settings_stylesheet(palette):
    """Compact native settings cards with the same visual language as the assistant."""
    colors, panel, raised, input_surface, _ = assistant_surfaces(palette)
    border = blend_colors(panel, colors.text, 0.22).name()
    text = colors.text.name()
    accent = colors.info.accent.name()
    muted = colors.muted_text.name()
    primary = colors.info.accent.darker(145).name()
    return f"""
QDialog#WorkflowAssistantSettings {{ background: {panel.name()}; color: {text}; }}
QFrame#AssistantSettingsHeader, QFrame#AssistantSettingsFooter {{
    background: {raised.name()}; border: none;
}}
QFrame#AssistantSettingsHeader {{ border-bottom: 1px solid {border}; }}
QFrame#AssistantSettingsFooter {{ border-top: 1px solid {border}; }}
QScrollArea#AssistantSettingsScroll, QWidget#AssistantSettingsViewport,
QWidget#AssistantSettingsBody {{ background: {panel.name()}; border: none; }}
QDialog#WorkflowAssistantSettings QLabel {{
    background: transparent; color: {text}; border: none;
}}
QDialog#WorkflowAssistantSettings QLabel#SettingsHeading {{
    font-size: 18px; font-weight: 600;
}}
QDialog#WorkflowAssistantSettings QLabel#SettingsCardHeading {{ font-weight: 600; }}
QDialog#WorkflowAssistantSettings QLabel#SettingsMuted {{ color: {muted}; }}
QFrame#AssistantSettingsCard, QFrame#AssistantConnectionCard {{
    background: {raised.name()}; border: 1px solid {border}; border-radius: 8px;
}}
QWidget#AssistantSettingsAdvanced, QWidget#AssistantSettingsCodexOptions,
QWidget#AssistantSettingsConnectionDetails {{
    background: transparent; border: none;
}}
QFrame#AssistantConnectionCard[connectionTone="success"] {{
    border-color: {colors.success.accent.name()};
}}
QFrame#AssistantConnectionCard[connectionTone="error"] {{
    border-color: {colors.error.accent.name()};
}}
QFrame#AssistantConnectionCard[connectionTone="warning"] {{
    border-color: {colors.warning.accent.name()};
}}
QDialog#WorkflowAssistantSettings QLineEdit,
QDialog#WorkflowAssistantSettings QComboBox,
QDialog#WorkflowAssistantSettings QSpinBox {{
    background: {input_surface.name()}; color: {text}; border: 1px solid {border};
    border-radius: 5px; padding: 7px 9px; min-height: 20px;
}}
QDialog#WorkflowAssistantSettings QComboBox QAbstractItemView {{
    background: {input_surface.name()}; color: {text};
    selection-background-color: {primary};
    selection-color: white;
}}
QDialog#WorkflowAssistantSettings QPushButton[assistantAction="primary"] {{
    background: {primary}; color: white; border: 1px solid {primary};
    border-radius: 6px; padding: 8px 12px;
}}
QDialog#WorkflowAssistantSettings QPushButton[assistantAction="secondary"] {{
    background: transparent; color: {text}; border: 1px solid {border};
    border-radius: 6px; padding: 8px 12px;
}}
QDialog#WorkflowAssistantSettings QPushButton#AssistantDisclosure {{
    background: transparent; color: {accent}; border: none;
    padding: 7px 0px; text-align: left; font-weight: 600;
}}
QDialog#WorkflowAssistantSettings QPushButton[assistantAction]:hover,
QDialog#WorkflowAssistantSettings QPushButton[assistantAction]:focus {{
    border-color: {accent};
}}
QDialog#WorkflowAssistantSettings QPushButton[assistantAction]:disabled {{
    background: transparent; color: {muted}; border-color: {border};
}}
"""


def assistant_chat_surfaces(palette):
    """Outlined messages and blue-tinted quotations, independent of authority."""
    colors, panel, _raised, _input_surface, user_surface = assistant_surfaces(palette)
    quote_surface = blend_colors(user_surface, panel, 0.55)
    return (
        panel,
        blend_colors(panel, colors.text, 0.26),
        quote_surface,
        blend_colors(quote_surface, colors.info.accent, 0.65),
    )


def assistant_stylesheet(palette):
    colors, panel, raised, input_surface, _ = assistant_surfaces(palette)
    border = blend_colors(panel, colors.text, 0.22).name()
    text = colors.text.name()
    muted = colors.muted_text.name()
    accent = colors.info.accent.name()
    # Filled actions need readable text in either theme.
    primary = colors.info.accent.darker(145).name()
    primary_text = QColor("white").name()
    return f"""
QDialog#WorkflowAssistantWindow {{ background: {panel.name()}; color: {text}; }}
QFrame#AssistantHeader, QFrame#AssistantFooter, QFrame#AssistantBottomNote,
QFrame#AssistantPageActions {{
    background: {raised.name()}; border: none;
}}
QFrame#AssistantHeader {{ border-bottom: 1px solid {border}; }}
QFrame#AssistantFooter, QFrame#AssistantBottomNote, QFrame#AssistantPageActions {{
    border-top: 1px solid {border};
}}
QWidget#AssistantGoalPage, QWidget#AssistantDiscussionPage,
QWidget#AssistantPlansPage, QWidget#AssistantGoalBody,
QWidget#AssistantDiscussionBody, QWidget#AssistantDiscussionViewport {{
    background: {panel.name()}; color: {text};
}}
QDialog#WorkflowAssistantWindow QLabel, QDialog#WorkflowAssistantWindow QCheckBox {{
    color: {text};
}}
QTabWidget#WorkflowAssistantStages::pane {{ border: none; background: {panel.name()}; }}
QTabBar#AssistantStageBar {{ background: {raised.name()}; border: none; }}
QTabBar#AssistantStageBar::tab {{
    background: {raised.name()}; color: {muted}; border: none;
    border-right: 1px solid {border}; border-bottom: 1px solid {border};
    border-radius: 0px; padding: 0px; margin: 0px;
}}
QTabBar#AssistantStageBar::tab:last {{ border-right: none; }}
QTabBar#AssistantStageBar::tab:selected {{
    background: {panel.name()}; color: {accent}; border-bottom: 3px solid {accent};
}}
QTabBar#AssistantStageBar::tab:hover {{ color: {text}; }}
QFrame#AssistantSourceCard, QFrame#AssistantPlanSummary, QFrame#AssistantInitiative,
QFrame#AssistantOriginalGoal, QFrame#AssistantConversation,
QFrame#AssistantQuestions, QFrame#AssistantAssumptionsSection,
QFrame#AssistantChecksSection, QFrame#AssistantEvidenceCard,
QFrame#AssistantAssessmentCard, QFrame#AssistantRefinementOptions,
QFrame#AssistantRefinementCard, QFrame#AssistantInvestigationOptions,
QFrame#AssistantInvestigationCard {{
    background: {raised.name()}; border: 1px solid {border}; border-radius: 7px;
}}
QFrame#AssistantSourceCard QLabel, QFrame#AssistantPlanSummary QLabel,
QFrame#AssistantInitiative QLabel, QWidget#AssistantInitiativeControls,
QWidget#AssistantInitiativeControls QLabel,
QFrame#AssistantOriginalGoal QLabel, QFrame#AssistantConversation QLabel,
QFrame#AssistantQuestions QLabel,
QFrame#AssistantAssumptionsSection QLabel, QFrame#AssistantChecksSection QLabel,
QFrame#AssistantEvidenceCard QLabel, QFrame#AssistantAssessmentCard QLabel,
QFrame#AssistantRefinementOptions QLabel, QFrame#AssistantRefinementCard QLabel,
QFrame#AssistantInvestigationOptions QLabel, QFrame#AssistantInvestigationCard QLabel,
QWidget#AssistantReviewDisclosureContent {{
    background: transparent; border: none;
}}
QFrame#AssistantEvidenceCard {{ border-left: 3px solid {accent}; }}
QCheckBox#AssistantEvidenceOption {{
    background: transparent; font-weight: 600; spacing: 10px; padding: 4px 0px;
}}
QLabel#AssistantSectionTitle {{
    font-size: 15px; font-weight: 600; background: transparent;
}}
QPlainTextEdit#AssistantGoalInput, QFrame#AssistantComposer {{
    background: {input_surface.name()}; color: {text};
    border: 1px solid {border}; border-radius: 7px;
}}
QPlainTextEdit#AssistantGoalInput {{ padding: 10px; }}
QPlainTextEdit#AssistantMessageInput {{
    background: transparent; color: {text}; border: none; padding: 0px;
}}
QTextBrowser#WorkflowAssistantConversation {{
    background: transparent; color: {text}; border: none; padding: 0px;
}}
QFrame#AssistantQuestion {{
    background: {panel.name()}; border: 1px solid {border}; border-radius: 6px;
}}
QFrame#AssistantQuestion QLabel {{ background: transparent; border: none; }}
QLineEdit#AssistantAnswerInput {{
    background: {input_surface.name()}; color: {text};
    border: 1px solid {border}; border-radius: 5px; padding: 8px;
}}
QLabel#AssistantQuestionNumber {{
    color: {accent}; background: transparent; font-weight: 600;
}}
QLabel#AssistantInitiativeDescription {{
    color: {text}; background: transparent; font-size: 12px;
}}
QSlider#AssistantInitiativeSlider::groove:horizontal {{
    height: 5px; background: {border}; border-radius: 2px;
}}
QSlider#AssistantInitiativeSlider::sub-page:horizontal {{
    background: {accent}; border-radius: 2px;
}}
QSlider#AssistantInitiativeSlider::handle:horizontal {{
    width: 16px; margin: -6px 0px; border-radius: 8px;
    background: {accent}; border: 1px solid {panel.name()};
}}
QScrollArea#AssistantDiscussionScroll {{ background: {panel.name()}; border: none; }}
QScrollArea#AssistantDiscussionScroll QScrollBar:vertical {{ width: 14px; }}
QFrame#AssistantComposer QLabel {{ background: transparent; border: none; }}
QProgressBar#AssistantRequestProgress {{
    background: {raised.name()}; border: 1px solid {border}; border-radius: 3px;
}}
QProgressBar#AssistantRequestProgress::chunk {{
    background: {accent}; border-radius: 2px;
}}
QLabel#AssistantMuted, QLabel#AssistantStageCaption {{
    color: {muted}; background: transparent; font-size: 12px;
}}
QLabel#AssistantSectionHeading {{
    font-size: 17px; font-weight: 600; background: transparent;
}}
QLabel#AssistantHeaderTitle {{
    font-size: 16px; font-weight: 600; background: transparent;
}}
QPushButton[assistantAction="secondary"], QPushButton#AssistantSettings {{
    background: transparent; color: {text}; border: 1px solid {border};
    border-radius: 6px; padding: 7px 11px;
}}
QPushButton[assistantAction="primary"] {{
    background: {primary}; color: {primary_text}; border: 1px solid {primary};
    border-radius: 6px; padding: 7px 11px;
}}
QPushButton[assistantAction="link"] {{
    background: transparent; color: {accent}; border: none; padding: 4px 0px;
    text-align: left;
}}
QPushButton#AssistantDisclosure {{
    font-weight: 600; padding: 7px 0px; text-align: left;
}}
QPushButton[assistantAction]:hover, QPushButton#AssistantSettings:hover {{
    border-color: {accent};
}}
QPushButton[assistantAction]:focus, QPushButton#AssistantSettings:focus {{
    border-color: {accent};
}}
QPushButton[assistantAction]:disabled {{
    background: transparent; color: {muted}; border-color: {border};
}}
QLabel#AssistantStepBadge {{
    color: {accent}; background: {colors.info.surface.name()}; border-radius: 4px;
}}
QFrame#WorkflowPlanStep {{
    background: transparent; border: none; border-bottom: 1px solid {border};
}}
QFrame#WorkflowPlanStep QWidget, QFrame#WorkflowPlanStep QLabel {{
    background: transparent; border: none;
}}
QFrame#WorkflowPlanStep QLabel#AssistantStepBadge {{
    color: {accent}; background: {colors.info.surface.name()}; border-radius: 4px;
}}
"""


def assistant_action_icon(kind, palette, *, primary=False):
    """Keep enabled glyphs readable, independent of a button's disabled palette."""
    icon_palette = QPalette(palette)
    if primary:
        icon_palette.setColor(QPalette.Active, QPalette.ButtonText, QColor("white"))
        icon_palette.setColor(QPalette.Inactive, QPalette.ButtonText, QColor("white"))
    return toolbar_icon(kind, icon_palette)
