"""Severity-aware status messaging for the VIPP user interface."""

from __future__ import annotations

from enum import StrEnum

from qtpy.QtCore import QEvent, Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.palette_roles import SemanticToneColors, theme_colors


class MessageSeverity(StrEnum):
    """Presentation levels supported by :class:`StatusMessageStrip`."""

    NEUTRAL = "neutral"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class StatusMessageStrip(QLabel):
    """A drop-in ``QLabel`` with explicit message severity.

    Existing callers may continue to use :meth:`setText`; that legacy path always
    presents a neutral, non-actionable message.  New callers should use
    :meth:`show_message` when a semantic severity is known.

    Only an actionable :attr:`MessageSeverity.ERROR` receives a filled, bordered
    alert treatment.  All other messages remain lightweight text/accent status so
    routine progress and success feedback do not compete with failures.
    """

    message_changed = Signal()

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("", parent)
        self._severity = MessageSeverity.NEUTRAL
        self._actionable = False
        self._applying_theme_style = False
        self.setObjectName("VippStatusMessageStrip")
        self.setTextFormat(Qt.PlainText)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.show_message(text)

    @property
    def severity(self) -> MessageSeverity:
        """The semantic severity of the currently displayed message."""

        return self._severity

    @property
    def actionable(self) -> bool:
        """Whether the current message requires user action."""

        return self._actionable

    def setText(self, text: str) -> None:  # noqa: N802
        """Set legacy status text and reset presentation to neutral."""

        self.show_message(text)

    def show_message(
        self,
        text: str,
        *,
        severity: MessageSeverity | str = MessageSeverity.NEUTRAL,
        actionable: bool = False,
        detail: str = "",
    ) -> None:
        """Display concise ``text`` with optional hover/accessibility detail."""

        resolved_severity = MessageSeverity(severity)
        resolved_actionable = bool(actionable)
        resolved_detail = str(detail).strip()
        full_width_alert = (
            resolved_severity is MessageSeverity.ERROR and resolved_actionable
        )

        self._severity = resolved_severity
        self._actionable = resolved_actionable
        self.setProperty("messageSeverity", resolved_severity.value)
        self.setProperty("messageActionable", resolved_actionable)
        self.setProperty("fullWidthAlert", full_width_alert)
        self.setAccessibleName(str(text))
        self.setAccessibleDescription(
            f"{resolved_severity.value} status"
            + (" requiring action" if resolved_actionable else "")
            + (f". Details: {resolved_detail}" if resolved_detail else "")
        )
        self.setToolTip(resolved_detail)
        self._apply_theme_style()
        super().setText(str(text))
        self.message_changed.emit()

    def changeEvent(self, event) -> None:  # noqa: N802
        """Refresh palette-derived colors after a live host-theme change."""

        super().changeEvent(event)
        if not self._applying_theme_style and event.type() in (
            QEvent.PaletteChange,
            QEvent.StyleChange,
        ):
            self._apply_theme_style()

    def refresh_theme(self) -> None:
        """Re-resolve colors after the host applies a new theme stylesheet."""

        self._apply_theme_style()

    def _apply_theme_style(self) -> None:
        if self._applying_theme_style:
            return
        self._applying_theme_style = True
        try:
            parent = self.parentWidget()
            palette = (
                QWidget.palette(parent) if parent is not None else QWidget.palette(self)
            )
            colors = theme_colors(palette)
            full_width_alert = (
                self._severity is MessageSeverity.ERROR and self._actionable
            )
            if full_width_alert:
                tone = colors.error
                style = (
                    f"color: {tone.foreground.name()};"
                    f" background-color: {tone.surface.name()};"
                    f" border: 1px solid {tone.border.name()};"
                    f" border-left: 4px solid {tone.accent.name()};"
                    " border-radius: 4px;"
                    " padding: 6px 8px;"
                    " font-weight: 600;"
                )
            elif self._severity is MessageSeverity.NEUTRAL:
                style = (
                    f"color: {colors.muted_text.name()};"
                    " background: transparent; border: none; padding: 3px 7px;"
                )
            else:
                tone = self._semantic_tone(colors)
                style = (
                    f"color: {tone.foreground.name()};"
                    " background: transparent;"
                    " border: none;"
                    f" border-left: 3px solid {tone.accent.name()};"
                    " padding: 3px 7px;"
                )
            self.setStyleSheet(style)
        finally:
            self._applying_theme_style = False

    def _semantic_tone(self, colors) -> SemanticToneColors:
        return {
            MessageSeverity.INFO: colors.info,
            MessageSeverity.SUCCESS: colors.success,
            MessageSeverity.WARNING: colors.warning,
            MessageSeverity.ERROR: colors.error,
        }[self._severity]


class StatusMessageActions(QWidget):
    """Compact, keyboard-accessible actions beside a workflow status strip.

    Dismissing only clears presentation; it never changes execution state or
    acknowledges a scientific error. Details are a snapshot in a reusable,
    nonmodal window, so even a long traceback cannot enlarge the workflow dock.
    """

    def __init__(self, strip: StatusMessageStrip, parent: QWidget | None = None):
        super().__init__(parent)
        self._strip = strip
        self._active = True
        self._details_dialog: QDialog | None = None
        self._details_text: QPlainTextEdit | None = None
        self.setSizePolicy(QSizePolicy.Maximum, QSizePolicy.Fixed)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.details_button = QPushButton("Details…", self)
        self.details_button.setAccessibleName("Show status details")
        self.details_button.setToolTip("View and copy the full technical message.")
        self.details_button.clicked.connect(self._show_details)
        self.dismiss_button = QPushButton("Dismiss", self)
        self.dismiss_button.setAccessibleName("Dismiss status message")
        self.dismiss_button.setToolTip(
            "Hide this message. Calculated results and node errors are unchanged."
        )
        self.dismiss_button.clicked.connect(lambda: strip.setText(""))
        layout.addWidget(self.details_button)
        layout.addWidget(self.dismiss_button)
        strip.message_changed.connect(self._sync)
        self._sync()

    def set_active(self, active: bool) -> None:
        """Hide alongside the status strip while the progress row is shown."""
        self._active = bool(active)
        self._sync()

    def _sync(self) -> None:
        self.details_button.setVisible(bool(self._strip.toolTip()))
        self.setVisible(
            self._active
            and bool(self._strip.text())
            and self._strip.severity in (MessageSeverity.WARNING, MessageSeverity.ERROR)
        )

    def _show_details(self) -> None:
        if not self._strip.toolTip():
            return
        if self._details_dialog is None:
            dialog = QDialog(self)
            dialog.setWindowTitle("VIPP message details")
            dialog.resize(680, 400)
            layout = QVBoxLayout(dialog)
            details = QPlainTextEdit(dialog)
            details.setReadOnly(True)
            details.setAccessibleName("Technical message; select text to copy")
            layout.addWidget(details, 1)
            buttons = QDialogButtonBox(QDialogButtonBox.Close, parent=dialog)
            buttons.rejected.connect(dialog.close)
            layout.addWidget(buttons)
            self._details_dialog = dialog
            self._details_text = details
        self._details_text.setPlainText(
            f"{self._strip.text()}\n\n{self._strip.toolTip()}"
        )
        self._details_dialog.show()
        self._details_dialog.raise_()
        self._details_dialog.activateWindow()


__all__ = ["MessageSeverity", "StatusMessageActions", "StatusMessageStrip"]
