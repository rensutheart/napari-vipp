"""Shared, palette-aware presentation for the GPU check and setup windows.

This module supplies visual chrome only. Expanding details never starts a
diagnostic check or installation.
"""

from __future__ import annotations

from qtpy.QtGui import QPalette

from napari_vipp.ui.iconography import interface_icon
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton


class SetupDisclosureButton(ToolbarCommandButton):
    """Quiet, keyboard-accessible disclosure; expansion is not an action."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self.setObjectName("ComputeDisclosure")
        self.setCheckable(True)
        self.setAutoDefault(False)
        self.toggled.connect(self.refresh_disclosure)
        self.refresh_disclosure()

    def refresh_disclosure(self, *_args) -> None:
        expanded = self.isChecked()
        self.setIcon(
            interface_icon(
                "chevron-down" if expanded else "chevron-right", self.palette()
            )
        )
        action = "Collapse" if expanded else "Expand"
        self.setAccessibleName(f"{action} {self.text().lower()}")


def setup_stylesheet(palette: QPalette, *, dialog_name: str, tone: str) -> str:
    """Build matching setup chrome without owning a dialog or its state.

    Apply this to the content scroll area and footer, not the root dialog: a
    root stylesheet can make Qt retain an old resolved palette after a theme
    switch. ``dialog_name`` is the root dialog's application-owned object name.
    """

    colors = theme_colors(palette)
    border = {
        "neutral": colors.muted_text,
        "info": colors.info.foreground,
        "success": colors.success.foreground,
        "warning": colors.warning.foreground,
        "error": colors.error.foreground,
    }.get(tone, colors.muted_text).name()
    root = f"QDialog#{dialog_name}"
    return f"""
        QWidget#ComputeContent {{
            background: {colors.surface.name()}; color: {colors.text.name()};
        }}
        {root} QLabel {{
            background: transparent; color: {colors.text.name()};
        }}
        QWidget#ComputeCardBody {{ background: transparent; border: none; }}
        QFrame#ComputeCard, QFrame#ComputeStatusCard {{
            background: {colors.raised_surface.name()};
            border: 1px solid {colors.border.name()}; border-radius: 9px;
        }}
        QFrame#ComputeStatusCard {{ border-left: 3px solid {border}; }}
        QLabel#ComputeWindowHeading {{ font-size: 21px; font-weight: 650; }}
        QLabel#ComputeStatusHeading {{ font-size: 17px; font-weight: 650; }}
        QLabel#ComputeSectionTitle {{ font-weight: 600; }}
        {root} QLabel#ComputeMuted,
        {root} QLabel#ComputePlatform {{
            color: {colors.muted_text.name()};
        }}
        {root} QPushButton {{
            padding: 8px 14px; border: 1px solid {colors.border.name()};
            border-radius: 5px; background: transparent;
            color: {colors.text.name()};
        }}
        {root} QPushButton:hover {{
            background: {colors.alternate_surface.name()};
        }}
        {root} QPushButton:focus {{
            border-color: {colors.info.accent.name()};
        }}
        {root} QPushButton#ComputePrimaryAction {{
            background: {colors.info.accent.darker(145).name()}; color: white;
            border-color: {colors.info.accent.darker(145).name()}; font-weight: 600;
        }}
        {root} QPushButton#ComputePrimaryAction:hover {{
            background: {colors.info.accent.darker(165).name()};
        }}
        {root} QPushButton:disabled,
        {root} QPushButton#ComputePrimaryAction:disabled {{
            color: {colors.muted_text.name()};
            background: {colors.alternate_surface.name()};
            border-color: {colors.border.name()};
        }}
        {root} QPushButton#ComputeDisclosure,
        {root} QPushButton#ComputeDisclosure:checked {{
            text-align: left; color: {colors.info.foreground.name()};
            border: none; padding: 11px 14px; background: transparent;
        }}
        {root} QComboBox,
        {root} QPlainTextEdit {{
            background: {colors.surface.name()}; color: {colors.text.name()};
            border: 1px solid {colors.border.name()};
            border-radius: 5px; padding: 6px;
        }}
        {root} QProgressBar {{
            background: {colors.alternate_surface.name()};
            border: none; border-radius: 2px;
            color: {colors.text.name()}; text-align: center;
        }}
        {root} QProgressBar::chunk {{
            background: {colors.info.accent.name()}; border-radius: 2px;
        }}
        QFrame#ComputeFooter {{
            background: {colors.raised_surface.name()};
            border-top: 1px solid {colors.border.name()};
        }}
    """


__all__ = ["SetupDisclosureButton", "setup_stylesheet"]
