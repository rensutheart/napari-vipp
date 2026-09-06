"""Consistent result tables and explicit selected-item output context."""

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QStyleOptionViewItem, QVBoxLayout, QWidget

from napari_vipp._tests.test_batch_results import _preview
from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp.ui.batch_results import BatchResultsPanel


def _cell_font(table, row, column):
    option = QStyleOptionViewItem()
    option.initFrom(table)
    # QAbstractItemView supplies the view font before invoking its delegate.
    option.font = table.font()
    table.itemDelegate().initStyleOption(option, table.model().index(row, column))
    # PySide exposes this property as a borrowed wrapper owned by the option.
    # Copy the value before the temporary style option is released.
    return QFont(option.font)


@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("width", [560, 1080])
def test_results_tables_share_gutters_font_and_padding(qtbot, tmp_path, dark, width):
    from napari.qt import get_stylesheet

    host = QWidget()
    qtbot.addWidget(host)
    host.setPalette(_palette(dark))
    host.setStyleSheet(get_stylesheet("dark" if dark else "light"))
    panel = BatchResultsPanel()
    QVBoxLayout(host).addWidget(panel)
    panel.setFont(QFont("Segoe UI", 11))
    plan = _preview(tmp_path)
    # Existing-file links and planned-file plain text must use identical fonts.
    plan.items[0].outputs[0].path.parent.mkdir(parents=True)
    plan.items[0].outputs[0].path.touch()
    panel.set_plan(plan)
    host.resize(width, 1100)
    host.show()
    qtbot.wait(20)
    assert host.width() == width
    tables = (panel.items_table, panel.output_table)
    left = [table.mapTo(panel, QPoint()).x() for table in tables]
    assert left[0] == left[1]
    assert tables[0].width() == tables[1].width()
    assert tables[0].font() == tables[1].font()
    assert tables[0].rowHeight(0) == tables[1].rowHeight(0)
    assert tables[0].rowHeight(0) >= tables[0].fontMetrics().height() + 14
    for table in tables:
        assert "padding: 0px 12px" in table.styleSheet()
        fonts = [_cell_font(table, 0, column) for column in range(table.columnCount())]
        assert all(font.pointSizeF() == table.font().pointSizeF() for font in fonts), (
            table.font().toString(), [font.toString() for font in fonts]
        )
        assert all(font.family() == table.font().family() for font in fonts)
    assert _cell_font(panel.output_table, 0, 0).underline()

    panel.select_item(1)
    assert not _cell_font(panel.output_table, 0, 0).underline()
    panel.setFont(QFont("Segoe UI", 13))
    qtbot.wait(10)
    assert _cell_font(panel.items_table, 0, 0).pointSizeF() == 13
    assert _cell_font(panel.output_table, 0, 0).pointSizeF() == 13


def test_output_context_follows_selection_without_requesting_preview(qtbot, tmp_path):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    plan = _preview(tmp_path, count=125)
    panel.set_plan(plan)
    requested = []
    panel.itemRequested.connect(requested.append)
    assert panel.select_item(119)
    assert panel.selected_item_label.text() == plan.items[119].batch_id
    assert panel.selected_item_meta.text() == "Item 120 of 125 · 1 output · Not run"
    assert panel.output_table.item(0, 0).data(Qt.UserRole) == str(
        plan.items[119].outputs[0].path
    )
    assert requested == []
    panel.set_plan(None)
    assert panel.selected_item_meta.text() == ""
    assert panel.output_table.rowCount() == 0
