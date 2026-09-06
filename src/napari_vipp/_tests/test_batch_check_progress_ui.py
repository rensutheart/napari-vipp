"""Early inventory remains responsive and cannot authorize execution."""

from dataclasses import replace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import QHeaderView

from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.core.batch import BatchPlan, BatchPreflightProgress
from napari_vipp.ui.batch import CollectionBatchDialog


def _dialog(qtbot, tmp_path, count=3):
    plan = _preview_result(tmp_path, count=count)
    actions = replace(_actions(plan, []), check_batch=lambda _values, _limit: True)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog._check_batch()
    paths = tuple(item.primary_source for item in plan.items)
    event = BatchPreflightProgress(
        "discovered",
        total=count,
        source_paths=(("input", "Image Source", paths),),
    )
    return dialog, plan, event


def test_inventory_appears_before_metadata_and_is_not_a_runnable_plan(qtbot, tmp_path):
    dialog, _plan, event = _dialog(qtbot, tmp_path)
    dialog.show_check_progress(event)
    assert dialog.tabs.currentIndex() == 1
    assert dialog.preview_table.rowCount() == 3
    assert dialog.preview_table.horizontalHeaderItem(1).text() == "Source file"
    assert dialog.preview_table.item(0, 1).text() == "field-1.npy"
    assert dialog.preview_table.item(0, 4).text() == "Waiting"
    assert not dialog.preview_table.item(0, 0).flags() & Qt.ItemIsUserCheckable
    assert dialog.check_count_label.text() == "0 of 3 files checked"
    assert dialog._preview_result is None
    assert dialog._display_plan is None
    for button in (
        dialog.preview_item_button,
        dialog.load_overrides_button,
        dialog.run_button,
        dialog.recheck_item_button,
        dialog.next_button,
    ):
        assert not button.isEnabled()
    assert not dialog.tabs.isTabEnabled(2)


def test_restore_inventory_does_not_steal_selected_tab(qtbot, tmp_path):
    dialog, _plan, event = _dialog(qtbot, tmp_path)
    dialog.show_check_progress(event, reveal_items=False)
    assert dialog.tabs.currentIndex() == 0
    assert dialog.tabs.isTabEnabled(1)
    assert dialog.preview_table.rowCount() == 3


def test_active_file_animates_and_counter_is_determinate(qtbot, tmp_path):
    dialog, plan, event = _dialog(qtbot, tmp_path)
    dialog.show_check_progress(event)
    active = BatchPreflightProgress(
        "checking",
        total=3,
        source_node_id="input",
        path=plan.items[0].primary_source,
        byte_current=40,
        byte_total=100,
    )
    dialog.show_check_progress(active)
    cell = dialog.preview_table.item(0, 4)
    assert cell.text() == "Checking…"
    assert not cell.icon().isNull()
    assert dialog._check_spinner_timer.isActive()
    before = cell.icon().cacheKey()
    dialog._animate_check_spinner()
    assert cell.icon().cacheKey() != before
    assert "reading file 40%" in dialog.batch_activity_status.text()
    assert dialog.source_detection_progress.maximum() == 3
    dialog.show_check_progress(replace(active, phase="checked", current=1))
    assert cell.text() == "Checked"
    assert cell.icon().isNull()
    assert not dialog._check_spinner_timer.isActive()
    assert dialog.check_count_label.text() == "1 of 3 files checked"
    assert dialog.source_detection_progress.value() == 1
    assert dialog.preview_table.item(1, 4).text() == "Waiting"


def test_inventory_paging_and_search_work_while_checks_run(qtbot, tmp_path):
    dialog, _plan, event = _dialog(qtbot, tmp_path, count=1200)
    dialog.show_check_progress(event)
    assert dialog.preview_table.rowCount() == 50
    dialog.items_next_button.click()
    assert dialog.preview_table.item(0, 1).text() == "field-51.npy"
    dialog.item_search.setText("field-1200.npy")
    assert dialog.preview_table.rowCount() == 1
    assert dialog.preview_table.item(0, 1).text() == "field-1200.npy"
    assert dialog.items_previous_button.isHidden()
    assert dialog.items_next_button.isHidden()


def test_exact_samples_still_wait_for_authoritative_success(qtbot, tmp_path):
    dialog, result, event = _dialog(qtbot, tmp_path)
    dialog.show_check_progress(event)
    core_plan = BatchPlan(result.config, result.items, result.config.output_dir)
    dialog.show_check_progress(
        BatchPreflightProgress("planning", current=3, total=3, plan=core_plan)
    )
    assert dialog.preview_table.horizontalHeaderItem(1).text() == "Batch item"
    assert dialog.preview_table.item(0, 1).text() == result.items[0].batch_id
    assert "finalizing" in dialog.check_count_label.text()
    assert dialog._preview_result is None
    assert not dialog.run_button.isEnabled()
    dialog.show_check_progress(
        BatchPreflightProgress(
            "contract",
            current=3,
            total=3,
            source_node_id="input",
            path=result.items[0].primary_source,
        )
    )
    assert dialog.preview_table.item(0, 4).text() == "Validating…"
    assert dialog._check_spinner_timer.isActive()
    assert "3 of 3 files checked" in dialog.check_count_label.text()
    dialog.show_check_progress(BatchPreflightProgress("complete", current=3, total=3))
    assert not dialog._check_spinner_timer.isActive()
    assert dialog._preview_result is None
    assert not dialog.run_button.isEnabled()
    dialog.apply_preview_result(result, preview_representative=False)
    assert dialog._check_rows is None
    assert dialog.check_count_label.isHidden()
    assert dialog._preview_result is result
    assert dialog.preview_table.item(0, 4).text() == "Ready"
    assert dialog.next_button.isEnabled()


def test_warning_is_not_shown_as_success(qtbot, tmp_path):
    dialog, plan, event = _dialog(qtbot, tmp_path)
    dialog.show_check_progress(event)
    dialog.show_check_progress(
        BatchPreflightProgress(
            "checked",
            current=1,
            total=3,
            source_node_id="input",
            path=plan.items[0].primary_source,
            warning=True,
        )
    )
    assert dialog.preview_table.item(0, 4).text() == "Needs attention"


def test_failure_retains_inventory_stops_spinner_and_leaves_recheck(qtbot, tmp_path):
    dialog, plan, event = _dialog(qtbot, tmp_path)
    dialog.show_check_progress(event)
    dialog.show_check_progress(
        BatchPreflightProgress(
            "checking",
            total=3,
            source_node_id="input",
            path=plan.items[0].primary_source,
        )
    )
    dialog._show_preview_failure("Image axes need attention")
    assert not dialog._check_spinner_timer.isActive()
    assert dialog.preview_table.rowCount() == 3
    assert dialog.preview_table.item(0, 4).text() == "Needs attention"
    assert dialog.preview_table.item(1, 4).text() == "Not checked"
    assert dialog.next_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert not dialog.load_overrides_button.isEnabled()
    assert dialog._preview_result is None


def test_setting_change_clears_stale_inventory_and_animation(qtbot, tmp_path):
    dialog, _plan, event = _dialog(qtbot, tmp_path)
    dialog.show_check_progress(event)
    dialog.pattern_edit.setText("*.tif")
    assert dialog._check_rows is None
    assert not dialog._check_spinner_timer.isActive()
    assert dialog.check_count_label.isHidden()
    assert dialog._preview_result is None


@pytest.mark.parametrize("dark", [False, True])
def test_spinner_and_fixed_footer_work_in_both_themes(qtbot, tmp_path, dark):
    dialog, plan, event = _dialog(qtbot, tmp_path)
    palette = QPalette()
    for role in (QPalette.Window, QPalette.Base):
        palette.setColor(role, QColor("#20242c" if dark else "#ffffff"))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(role, QColor("#edf2f8" if dark else "#20242c"))
    dialog.setPalette(palette)
    dialog.resize(1080, 800)
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    height = dialog.footer.height()
    dialog.show_check_progress(event)
    dialog.show_check_progress(
        BatchPreflightProgress(
            "checking",
            total=3,
            source_node_id="input",
            path=plan.items[0].primary_source,
        )
    )
    assert dialog.footer.height() == height
    assert not dialog.preview_table.item(0, 4).icon().isNull()
    assert dialog.check_count_label.isVisible()


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [760, 1480])
def test_check_inventory_gives_names_space_without_tall_wrapped_rows(
    qtbot, tmp_path, dark, width
):
    from napari.qt import get_stylesheet

    from napari_vipp._tests.test_batch_table_theme import _palette

    dialog, plan, event = _dialog(qtbot, tmp_path, count=14)
    dialog.setStyleSheet(get_stylesheet("dark" if dark else "light"))
    dialog.setPalette(_palette(dark))
    dialog.setFont(QFont("Segoe UI", 12))
    paths = tuple(
        item.primary_source.with_name(
            f"MMV558 30 hpt MitoTracker_{index:03d}-Airyscan Processing.czi"
        )
        for index, item in enumerate(plan.items, 1)
    )
    event = replace(event, source_paths=(("input", "Image Source", paths),))
    # Inventory can arrive before the splitter has its final on-screen width.
    dialog.show_check_progress(event)
    dialog.resize(width, 800)
    dialog.show()
    qtbot.wait(10)
    table = dialog.preview_table
    assert not table.wordWrap()
    heights = [table.rowHeight(row) for row in range(table.rowCount())]
    assert max(heights) <= max(28, table.fontMetrics().height() + 10)
    assert table.columnWidth(1) > table.columnWidth(4)
    if width == 1480:
        assert table.columnWidth(1) > table.viewport().width() * 0.4
    assert table.item(0, 1).text() == paths[0].name
    assert table.item(0, 1).toolTip() == str(paths[0])
    assert table.item(0, 2).text() == "1 source"
    assert table.item(0, 2).toolTip() == "Image Source"
    widths = [table.columnWidth(column) for column in range(6)]

    active = BatchPreflightProgress(
        "checking", total=14, source_node_id="input", path=paths[0]
    )
    dialog.show_check_progress(active)
    assert not table.item(0, 4).icon().isNull()
    dialog.show_check_progress(replace(active, phase="checked", warning=True))
    assert table.item(0, 4).toolTip() == "Needs attention"
    assert [table.columnWidth(column) for column in range(6)] == widths
    assert [table.rowHeight(row) for row in range(table.rowCount())] == heights

    # Leave the existing post-check layout intact, including wrapped names.
    dialog.apply_preview_result(plan, preview_representative=False)
    assert table.wordWrap()
    assert table.horizontalHeader().sectionResizeMode(4) == QHeaderView.ResizeToContents
    assert table.item(0, 4).text() == "Ready"
