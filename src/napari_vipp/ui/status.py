"""Severity-aware status messaging for the VIPP user interface."""

from __future__ import annotations

from enum import StrEnum

from qtpy.QtCore import QEvent, Signal
from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget

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


__all__ = ["MessageSeverity", "StatusMessageStrip"]
