"""Shared platform-aware ordering for VIPP dialog actions.

Button placement is presentation only: callers retain all click, rejection,
worker-cancellation and safe-default behavior. Native file/message dialogs are
left to Qt. The explicit Windows/macOS policy also works with napari's Fusion
theme, which must not change the operating system's button convention.
"""

from __future__ import annotations

import sys
from collections.abc import Iterable

from qtpy.QtWidgets import (
    QApplication,
    QDialogButtonBox,
    QHBoxLayout,
    QPushButton,
    QStyle,
)


def _layout_number(value) -> int:
    return int(getattr(value, "value", value))


def dialog_button_layout(platform: str | None = None) -> int:
    """Resolve the OS convention, retaining the desktop style on Linux."""
    platform = sys.platform if platform is None else platform
    if platform == "win32":
        return _layout_number(QDialogButtonBox.WinLayout)
    if platform == "darwin":
        return _layout_number(QDialogButtonBox.MacLayout)
    style = QApplication.style()
    return (
        int(style.styleHint(QStyle.SH_DialogButtonLayout))
        if style is not None
        else _layout_number(QDialogButtonBox.WinLayout)
    )


class DialogButtonBox(QDialogButtonBox):
    """A normal Qt button box with the same OS policy as custom footers."""

    def __init__(self, *args, platform: str | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        policy = dialog_button_layout(platform)
        self.setStyleSheet(f"QDialogButtonBox {{ button-layout: {policy}; }}")


def add_dialog_buttons(
    layout: QHBoxLayout,
    *,
    actions: Iterable[QPushButton],
    dismiss: QPushButton,
    stop_actions: Iterable[QPushButton] = (),
    platform: str | None = None,
) -> tuple[QPushButton, ...]:
    """Append an action/dismiss group without changing what any button does.

    Callers place hints, auxiliary utilities and a stretch before this group.
    Stop controls are separate from dismissal, and can be hidden independently.
    The return value is the visual order, useful for responsive footer sizing.
    """
    actions = tuple(actions)
    stops = tuple(stop_actions)
    buttons = (*stops, *actions, dismiss)
    if len({id(button) for button in buttons}) != len(buttons):
        raise ValueError("A dialog button must appear in only one role.")
    dismiss.setAutoDefault(False)
    for button in stops:
        button.setAutoDefault(False)
        button.setDefault(False)
        layout.addWidget(button)
    dismiss_first = dialog_button_layout(platform) in (
        _layout_number(QDialogButtonBox.MacLayout),
        _layout_number(QDialogButtonBox.GnomeLayout),
    )
    ordered = (dismiss, *actions) if dismiss_first else (*actions, dismiss)
    for button in ordered:
        layout.addWidget(button)
    return (*stops, *ordered)
