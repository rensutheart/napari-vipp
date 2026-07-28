"""Severity-aware status messaging for the VIPP user interface."""

from __future__ import annotations

from enum import StrEnum

from qtpy.QtWidgets import QLabel, QSizePolicy, QWidget


class MessageSeverity(StrEnum):
    """Presentation levels supported by :class:`StatusMessageStrip`."""

    NEUTRAL = "neutral"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


_ACCENT_STYLES = {
    MessageSeverity.NEUTRAL: (
        "color: #cbd5e1; background: transparent; border: none; padding: 3px 7px;"
    ),
    MessageSeverity.INFO: (
        "color: #bfdbfe;"
        " background: transparent;"
        " border: none;"
        " border-left: 3px solid #3b82f6;"
        " padding: 3px 7px;"
    ),
    MessageSeverity.SUCCESS: (
        "color: #bbf7d0;"
        " background: transparent;"
        " border: none;"
        " border-left: 3px solid #22c55e;"
        " padding: 3px 7px;"
    ),
    MessageSeverity.WARNING: (
        "color: #fde68a;"
        " background: transparent;"
        " border: none;"
        " border-left: 3px solid #f59e0b;"
        " padding: 3px 7px;"
    ),
    MessageSeverity.ERROR: (
        "color: #fca5a5;"
        " background: transparent;"
        " border: none;"
        " border-left: 3px solid #ef4444;"
        " padding: 3px 7px;"
    ),
}

_ACTIONABLE_ERROR_STYLE = (
    "color: #fecaca;"
    " background-color: #450a0a;"
    " border: 1px solid #ef4444;"
    " border-left: 4px solid #ef4444;"
    " border-radius: 4px;"
    " padding: 6px 8px;"
    " font-weight: 600;"
)


class StatusMessageStrip(QLabel):
    """A drop-in ``QLabel`` with explicit message severity.

    Existing callers may continue to use :meth:`setText`; that legacy path always
    presents a neutral, non-actionable message.  New callers should use
    :meth:`show_message` when a semantic severity is known.

    Only an actionable :attr:`MessageSeverity.ERROR` receives a filled, bordered
    alert treatment.  All other messages remain lightweight text/accent status so
    routine progress and success feedback do not compete with failures.
    """

    def __init__(
        self,
        text: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__("", parent)
        self._severity = MessageSeverity.NEUTRAL
        self._actionable = False
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
    ) -> None:
        """Display ``text`` with semantic, severity-aware presentation."""

        resolved_severity = MessageSeverity(severity)
        resolved_actionable = bool(actionable)
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
        )
        self.setStyleSheet(
            _ACTIONABLE_ERROR_STYLE
            if full_width_alert
            else _ACCENT_STYLES[resolved_severity]
        )
        super().setText(str(text))


__all__ = ["MessageSeverity", "StatusMessageStrip"]
