"""Consistent result tables and explicit selected-item output context."""

import pytest
from qtpy.QtCore import QPoint, QRect, Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QPushButton, QStyleOptionViewItem, QVBoxLayout, QWidget

from napari_vipp._tests.test_batch_results import _preview, _record, _result
from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp.core.batch import BatchStatus
from napari_vipp.ui.batch import CollectionBatchDialog
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


@pytest.mark.parametrize("dark", [True, False])
@pytest.mark.parametrize("width, font_size", [(640, 10), (1080, 12), (1200, 16)])
def test_finished_actions_fit_one_toolbar_row_and_export_stays_in_report(
    qtbot, tmp_path, dark, width, font_size
):
    from napari.qt import get_stylesheet

    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    dialog.setPalette(_palette(dark))
    dialog.setStyleSheet(get_stylesheet("dark" if dark else "light"))
    dialog.setFont(QFont("Segoe UI", font_size))
    plan = _preview(tmp_path, 1)
    result = _result(plan, (_record(plan.items[0], BatchStatus.COMPLETED),))
    result.manifest_path.parent.mkdir()
    result.manifest_path.touch()
    dialog.apply_preview_result(plan, preview_representative=False)
    dialog.begin_run(1)
    dialog.finish_run(result)
    dialog.resize(width, 1000)
    dialog.show()
    qtbot.wait(20)

    assert dialog.width() == width
    panel = dialog.results_panel
    toolbar = panel.artifact_toolbar
    buttons = [
        panel.resume_button,
        panel.output_folder_button,
        panel.refresh_files_button,
    ]
    assert toolbar.findChildren(QPushButton) == buttons
    assert [button.text() for button in buttons] == [
        "Resume saved run…", "Output folder", "Refresh file status"
    ]
    rectangles = [
        QRect(button.mapTo(toolbar, QPoint()), button.size()) for button in buttons
    ]
    assert all(button.isVisible() and not button.icon().isNull() for button in buttons)
    assert all(toolbar.rect().contains(rectangle) for rectangle in rectangles), (
        toolbar.rect(), rectangles,
    )
    assert max(rect.center().y() for rect in rectangles) - min(
        rect.center().y() for rect in rectangles
    ) <= 2
    assert all(
        left.right() < right.left()
        for left, right in zip(rectangles, rectangles[1:], strict=False)
    )
    separator = panel.artifact_separator
    separator_rect = QRect(separator.mapTo(toolbar, QPoint()), separator.size())
    assert separator.isVisible() and separator.width() == 1
    assert toolbar.rect().contains(separator_rect)
    assert rectangles[0].right() < separator_rect.left()
    assert separator_rect.right() < rectangles[1].left()
    for button in buttons:
        assert button.width() >= button.minimumSizeHint().width()
    assert toolbar.geometry().bottom() < dialog.run_scroll.geometry().top()

    report = panel.run_report
    export = report.export_package_button
    assert panel.export_package_button is export
    assert export.parentWidget() is report
    assert export.text() == "Export reproducibility package…"
    assert export.isVisible() and export.isEnabled()
    assert export.width() >= export.minimumSizeHint().width()
    assert report.rect().contains(QRect(export.mapTo(report, QPoint()), export.size()))
    assert not export.icon().isNull()
    assert not hasattr(report, "manifest_button")
    assert all("manifest JSON" not in button.text() for button in panel.findChildren(
        QPushButton
    ))
    assert dialog.next_button.text() == "Export package…"
    assert not dialog.next_button.icon().isNull()
