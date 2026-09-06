"""Text-only links, persistent actions, and read-only file availability updates."""

from dataclasses import replace

import pytest
from qtpy.QtCore import QEvent, QPoint, Qt
from qtpy.QtWidgets import QApplication

from napari_vipp._tests.test_batch_results import _preview, _record, _result
from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp.core.batch import BatchStatus
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_results import BatchResultsPanel


@pytest.mark.parametrize("dark", [True, False])
def test_only_text_not_cell_padding_or_blank_space_activates_item(
    qtbot, tmp_path, dark
):
    from napari.qt import get_stylesheet

    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.setPalette(_palette(dark))
    panel.setStyleSheet(get_stylesheet("dark" if dark else "light"))
    preview = _preview(tmp_path)
    panel.set_plan(preview)
    panel.resize(1080, 1100)
    panel.show()
    qtbot.wait(10)
    table = panel.items_table
    index = table.model().index(1, 0)
    cell = table.visualRect(index)
    link = table.link_rect(index)
    requested = []
    panel.itemRequested.connect(requested.append)
    assert link.left() >= cell.left() + 12
    assert link.right() < cell.right() - 20
    blank = QPoint(cell.right() - 16, cell.center().y())
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, pos=blank)
    assert panel.selected_item_label.text() == preview.items[1].batch_id
    assert requested == []
    qtbot.mouseDClick(table.viewport(), Qt.LeftButton, pos=blank)
    qtbot.mouseClick(
        table.viewport(), Qt.LeftButton, pos=QPoint(cell.left() + 2, cell.center().y())
    )
    assert requested == []
    qtbot.mouseMove(table.viewport(), pos=link.center())
    assert table.viewport().cursor().shape() == Qt.PointingHandCursor
    qtbot.mouseMove(table.viewport(), pos=blank)
    assert table.viewport().cursor().shape() != Qt.PointingHandCursor
    qtbot.mousePress(table.viewport(), Qt.LeftButton, pos=blank)
    qtbot.mouseRelease(table.viewport(), Qt.LeftButton, pos=link.center())
    assert requested == []
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, pos=link.center())
    assert requested == [1]
    table.setFocus()
    qtbot.keyClick(table, Qt.Key_Return)
    assert requested == [1, 1]


def test_file_links_have_same_hit_target_and_preserve_exact_path(
    qtbot, tmp_path, monkeypatch
):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    preview = _preview(tmp_path)
    path = preview.items[0].outputs[0].path
    path.parent.mkdir()
    path.touch()
    panel.set_plan(preview)
    panel.resize(1080, 1100)
    panel.show()
    qtbot.wait(10)
    revealed = []
    monkeypatch.setattr(panel, "_reveal_path", revealed.append)
    table = panel.output_table
    cell = table.visualRect(table.model().index(0, 0))
    link = table.link_rect(table.model().index(0, 0))
    qtbot.mouseClick(
        table.viewport(),
        Qt.LeftButton,
        pos=QPoint(cell.right() - 20, cell.center().y()),
    )
    assert revealed == []
    qtbot.mouseClick(table.viewport(), Qt.LeftButton, pos=link.center())
    assert revealed == [path]
    # Planned/missing filenames are plain text, not invisible links.
    panel.select_item(1)
    assert table.link_rect(table.model().index(0, 0)).isEmpty()


def test_directory_changes_update_existence_without_rewriting_run_outcome(
    qtbot, tmp_path
):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    preview = _preview(tmp_path, 1)
    path = preview.items[0].outputs[0].path
    path.parent.mkdir()
    path.touch()
    result = _result(preview, (_record(preview.items[0], BatchStatus.COMPLETED),))
    result.manifest_path.touch()
    panel.set_plan(preview)
    panel.finish_run(result)
    panel.show()
    qtbot.waitUntil(lambda: bool(panel._file_watcher.directories()))
    before = panel.summary_label.text()
    path.unlink()
    qtbot.waitUntil(
        lambda: panel.output_table.item(0, 1).text() == "Saved · file missing",
        timeout=3000,
    )
    assert not panel.reveal_button.isEnabled()
    assert panel.items_table.item(0, 1).text() == "Completed"
    assert panel.summary_label.text() == before
    result.manifest_path.unlink()
    qtbot.waitUntil(
        lambda: not panel.run_report.manifest_button.isEnabled(), timeout=3000
    )
    assert panel.has_run_report
    panel.hide()
    QApplication.processEvents()
    assert panel._file_watcher.directories() == []
    assert not panel._file_refresh_timer.isActive()
    path.touch()
    panel.show()
    qtbot.waitUntil(
        lambda: panel.output_table.item(0, 1).text() == "Saved", timeout=3000
    )


def test_refresh_preserves_output_selection_and_scrolling(qtbot, tmp_path):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    preview = _preview(tmp_path, 1)
    item = preview.items[0]
    preview = replace(
        preview,
        items=(
            replace(
                item,
                outputs=tuple(
                    replace(
                        item.outputs[0],
                        path=item.outputs[0].path.with_name(f"output-{i}.npy"),
                    )
                    for i in range(25)
                ),
            ),
        ),
    )
    panel.set_plan(preview)
    panel.resize(1080, 1100)
    panel.show()
    qtbot.wait(10)
    panel.output_table.selectRow(22)
    panel.output_table.scrollToItem(panel.output_table.item(22, 0))
    before = panel.output_table.verticalScrollBar().value()
    panel.refresh_files()
    assert panel.output_table.currentRow() == 22
    assert panel.output_table.verticalScrollBar().value() == before


def test_returning_to_window_rechecks_when_directory_notifications_are_unavailable(
    qtbot, tmp_path, monkeypatch
):
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    preview = _preview(tmp_path, 1)
    path = preview.items[0].outputs[0].path
    path.parent.mkdir()
    path.touch()
    panel.set_plan(preview)
    panel.show()
    qtbot.waitUntil(lambda: bool(panel._file_watcher.directories()))
    panel._file_refresh_timer.stop()
    panel._file_watcher.removePaths(panel._file_watcher.directories())
    path.unlink()
    monkeypatch.setattr(panel, "isActiveWindow", lambda: True)
    QApplication.sendEvent(panel, QEvent(QEvent.ActivationChange))
    qtbot.waitUntil(lambda: not panel.reveal_button.isEnabled(), timeout=3000)
    assert panel.output_table.item(0, 1).text() == "To create"


def test_run_actions_remain_above_scrolling_content(qtbot, tmp_path):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(_preview(tmp_path, 14), preview_representative=False)
    dialog.resize(736, 700)
    dialog.tabs.setCurrentIndex(3)
    dialog.show()
    qtbot.wait(10)
    toolbar = dialog.results_panel.artifact_toolbar
    assert toolbar.parentWidget() is dialog.run_tab
    before = toolbar.mapTo(dialog, QPoint())
    dialog.run_scroll.verticalScrollBar().setValue(
        dialog.run_scroll.verticalScrollBar().maximum()
    )
    QApplication.processEvents()
    assert toolbar.mapTo(dialog, QPoint()) == before
    assert toolbar.isVisible()
    assert toolbar.geometry().bottom() < dialog.run_scroll.geometry().top()
