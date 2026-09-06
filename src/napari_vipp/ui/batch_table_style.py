"""Explicit batch table colours, independent of native alternate-row defaults."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import QTableView

from napari_vipp.ui.palette_roles import palette_is_dark


@dataclass(frozen=True)
class BatchTableColors:
    background: str
    alternate: str
    text: str
    selected: str
    selected_text: str
    header: str
    border: str
    hover: str
    disabled_text: str


_DARK = BatchTableColors(
    background="#262b34",
    alternate="#303743",
    text="#eef2f6",
    selected="#234963",
    selected_text="#ffffff",
    header="#37414f",
    border="#465260",
    hover="#37485c",
    disabled_text="#aebaca",
)
_LIGHT = BatchTableColors(
    background="#ffffff",
    alternate="#f1f5f9",
    text="#1e293b",
    selected="#cfe2f5",
    selected_text="#102a43",
    header="#e2e8f0",
    border="#cbd5e1",
    hover="#e2eef9",
    disabled_text="#526477",
)


def batch_table_colors(palette: QPalette) -> BatchTableColors:
    """Choose by the owning surface's theme, never its AlternateBase colour."""
    return _DARK if palette_is_dark(palette) else _LIGHT


def apply_batch_table_style(
    table: QTableView,
    palette: QPalette,
    *,
    font: QFont | None = None,
    horizontal_padding: int | None = None,
) -> None:
    """Pin normal, alternate, selected and inactive colours for one batch view.

    The explicit stylesheet beats inherited napari/native QSS; palette roles
    also cover delegates and embedded editors. Pass the owning panel palette,
    not the previously styled table, so runtime theme switches remain live.
    """
    colors = batch_table_colors(palette)
    resolved = QPalette(palette)
    for group in (QPalette.Active, QPalette.Inactive, QPalette.Disabled):
        for role, value in (
            (QPalette.Base, colors.background),
            (QPalette.AlternateBase, colors.alternate),
            (QPalette.Window, colors.background),
            (QPalette.Text, colors.text),
            (QPalette.WindowText, colors.text),
            (QPalette.Highlight, colors.selected),
            (QPalette.HighlightedText, colors.selected_text),
        ):
            resolved.setColor(group, role, QColor(value))
    resolved.setColor(QPalette.Disabled, QPalette.Text, QColor(colors.disabled_text))
    if table.palette() != resolved:
        table.setPalette(resolved)
    stylesheet = (
        "QTableView {"
        f"background: {colors.background}; color: {colors.text};"
        f"alternate-background-color: {colors.alternate};"
        f"selection-background-color: {colors.selected};"
        f"selection-color: {colors.selected_text};"
        f"gridline-color: {colors.border};"
        f"border: 1px solid {colors.border};"
        "}"
        + (
            f"QTableView::item {{ padding: 0px {horizontal_padding}px; }}"
            if horizontal_padding is not None
            else ""
        )
        + "QTableView::item:alternate {"
        f"background: {colors.alternate};"
        "}"
        "QTableView::item:hover, QTableView::item:alternate:hover {"
        f"background: {colors.hover};"
        "}"
        "QTableView::item:selected, QTableView::item:alternate:selected,"
        "QTableView::item:selected:hover, QTableView::item:alternate:selected:hover {"
        f"background: {colors.selected}; color: {colors.selected_text};"
        "}"
        "QHeaderView::section, QTableCornerButton::section {"
        f"background: {colors.header}; color: {colors.text};"
        f"border: 0; border-right: 1px solid {colors.border};"
        f"border-bottom: 1px solid {colors.border};"
        f"padding: 2px {4 if horizontal_padding is None else horizontal_padding}px;"
        "}"
        "QTableView:disabled {"
        f"color: {colors.disabled_text};"
        "}"
    )
    if table.styleSheet() != stylesheet:
        table.setStyleSheet(stylesheet)
    # A local QSS establishes a font boundary in Qt. Keep the host's font and
    # multiline header sizing live instead of freezing the application default.
    if font is not None:
        for widget in (table, table.horizontalHeader(), table.verticalHeader()):
            if widget.font() != font:
                widget.setFont(font)
    table.setAlternatingRowColors(True)


__all__ = ["BatchTableColors", "apply_batch_table_style", "batch_table_colors"]
