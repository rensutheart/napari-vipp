"""Batch tables must not inherit unreadable alternate rows from host palettes."""

from __future__ import annotations

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QLineEdit,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_table_style import (
    apply_batch_table_style,
    batch_table_colors,
)


def _palette(dark: bool) -> QPalette:
    palette = QPalette()
    background = QColor("#25282f" if dark else "#ffffff")
    foreground = QColor("#eef2f6" if dark else "#202530")
    for role in (QPalette.Base, QPalette.Window, QPalette.Button):
        palette.setColor(role, background)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(role, foreground)
    # Reproduce a native alternate role left behind by the opposite theme.
    palette.setColor(QPalette.AlternateBase, QColor("#ffffff" if dark else "#25282f"))
    return palette


def _contrast(first: str, second: str) -> float:
    def luminance(value: str) -> float:
        color = QColor(value)
        channels = (color.redF(), color.greenF(), color.blueF())
        linear = [
            channel / 12.92
            if channel <= 0.04045
            else ((channel + 0.055) / 1.055) ** 2.4
            for channel in channels
        ]
        return sum(
            weight * channel
            for weight, channel in zip((0.2126, 0.7152, 0.0722), linear, strict=True)
        )

    lighter, darker = sorted((luminance(first), luminance(second)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


@pytest.mark.parametrize("dark", [False, True])
def test_review_column_padding_is_included_in_content_sizing(qtbot, dark):
    table = QTableWidget(1, 4)
    qtbot.addWidget(table)
    table.setHorizontalHeaderLabels(["Sources", "Outputs", "Checks", "Run result"])
    for column, text in enumerate(["1 source", "4 planned", "Ready", "Not run"]):
        table.setItem(0, column, QTableWidgetItem(text))
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
    apply_batch_table_style(table, _palette(dark), horizontal_padding=4)
    table.show()
    QApplication.processEvents()
    compact = [table.columnWidth(column) for column in range(4)]
    apply_batch_table_style(table, _palette(dark), horizontal_padding=12)
    table.resizeColumnsToContents()
    QApplication.processEvents()
    for column, width in enumerate(compact):
        assert table.columnWidth(column) >= width + 16


@pytest.mark.parametrize("dark", [False, True])
def test_batch_table_colors_repair_opposite_theme_alternate_base(dark):
    palette = _palette(dark)
    colors = batch_table_colors(palette)

    assert (QColor(colors.background).lightness() < 128) is dark
    assert colors.alternate != palette.color(QPalette.AlternateBase).name()
    assert colors.alternate != colors.background
    for background in (colors.background, colors.alternate, colors.header):
        assert _contrast(colors.text, background) >= 4.5
    assert _contrast(colors.selected_text, colors.selected) >= 4.5
    assert QColor(colors.border).isValid()
    # Resolving safe colors must not mutate the host's palette.
    assert palette.color(QPalette.AlternateBase).name() == (
        "#ffffff" if dark else "#25282f"
    )


@pytest.mark.parametrize("start_dark", [False, True])
def test_every_batch_table_rethemes_in_both_directions(qtbot, start_dark):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    tables = {
        "items": dialog.preview_table,
        "overrides": dialog.parameter_override_editor.table,
        "frozen identity": dialog.parameter_override_editor.table.frozen_identity,
        "result items": dialog.results_panel.items_table,
        "result outputs": dialog.results_panel.output_table,
    }
    seen = []

    for dark in (start_dark, not start_dark, start_dark):
        palette = _palette(dark)
        dialog.setPalette(palette)
        QApplication.processEvents()
        colors = batch_table_colors(palette)
        styles = tuple(table.styleSheet() for table in tables.values())
        for (name, table), style in zip(tables.items(), styles, strict=True):
            assert table.alternatingRowColors()
            assert f"alternate-background-color: {colors.alternate}" in style, name
            assert f"selection-background-color: {colors.selected}" in style
            assert f"selection-color: {colors.selected_text}" in style
            assert colors.background in style
            assert colors.text in style
            assert colors.header in style
            assert colors.border in style
            for group in (QPalette.Active, QPalette.Inactive):
                for role, expected in (
                    (QPalette.Base, colors.background),
                    (QPalette.AlternateBase, colors.alternate),
                    (QPalette.Text, colors.text),
                    (QPalette.Highlight, colors.selected),
                    (QPalette.HighlightedText, colors.selected_text),
                ):
                    assert table.palette().color(group, role).name() == expected, name
        seen.append(styles)

    assert seen[0] != seen[1]
    assert seen[0] == seen[2]


def _row_pixel(table: QTableWidget, row: int) -> str:
    rect = table.visualRect(table.model().index(row, 0))
    image = table.viewport().grab().toImage()
    ratio = image.devicePixelRatio()
    return image.pixelColor(
        int((rect.right() - 12) * ratio), int(rect.center().y() * ratio)
    ).name()


def _assert_selected_pixel(table: QTableWidget, color: str, row: int = 2) -> None:
    actual = QColor(_row_pixel(table, row))
    expected = QColor(color)
    # Native focus painting can shade a selected cell slightly; it must stay
    # close to the explicit color rather than reverting to native selection.
    assert (
        max(
            abs(actual.red() - expected.red()),
            abs(actual.green() - expected.green()),
            abs(actual.blue() - expected.blue()),
        )
        <= 6
    )


@pytest.mark.parametrize("start_dark", [False, True])
def test_batch_rows_and_unfocused_selection_render_under_napari_qss(qtbot, start_dark):
    from napari._qt.qt_resources import get_stylesheet

    host = QWidget()
    qtbot.addWidget(host)
    layout = QVBoxLayout(host)
    table = QTableWidget(3, 1)
    table.setSelectionBehavior(QAbstractItemView.SelectRows)
    table.setSelectionMode(QAbstractItemView.SingleSelection)
    table.setEditTriggers(QAbstractItemView.NoEditTriggers)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    for row in range(3):
        table.setItem(row, 0, QTableWidgetItem(""))
        table.setRowHeight(row, 36)
    other_control = QLineEdit()
    layout.addWidget(table)
    layout.addWidget(other_control)
    host.resize(440, 260)
    host.show()
    host.activateWindow()

    for dark in (start_dark, not start_dark, start_dark):
        host.setStyleSheet(
            get_stylesheet(
                "dark" if dark else "light", extra_variables={"font_size": "10pt"}
            )
        )
        palette = _palette(dark)
        host.setPalette(palette)
        apply_batch_table_style(table, palette)
        colors = batch_table_colors(palette)
        table.setCurrentCell(2, 0)
        table.selectRow(2)
        table.setFocus(Qt.OtherFocusReason)
        qtbot.waitUntil(table.hasFocus)
        QApplication.processEvents()

        assert _row_pixel(table, 0) == colors.background
        assert _row_pixel(table, 1) == colors.alternate
        _assert_selected_pixel(table, colors.selected)

        other_control.setFocus(Qt.OtherFocusReason)
        qtbot.waitUntil(other_control.hasFocus)
        QApplication.processEvents()
        assert not table.hasFocus()
        assert table.selectionModel().isRowSelected(2, table.rootIndex())
        _assert_selected_pixel(table, colors.selected)

        # Alternate-row styling must not win over selection when focus is away.
        table.selectRow(1)
        QApplication.processEvents()
        _assert_selected_pixel(table, colors.selected, row=1)
