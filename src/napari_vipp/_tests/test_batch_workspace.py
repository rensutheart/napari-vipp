"""Approved batch-window navigation, scaling, and safety regressions."""

from dataclasses import replace

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QPushButton

from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.ui.batch import CollectionBatchDialog


def test_check_then_review_then_run_never_previews_implicitly(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    previewed = []
    dialog = CollectionBatchDialog(actions=_actions(plan, previewed))
    qtbot.addWidget(dialog)
    assert dialog.tabs.currentIndex() == 0
    assert dialog.next_button.text() == "Check batch"
    assert not dialog.run_button.isEnabled()
    dialog.next_button.click()
    assert dialog.tabs.currentIndex() == 1
    assert dialog._preview_result is plan
    assert previewed == []
    assert dialog.next_button.text() == "Continue to run"
    dialog.next_button.click()
    assert dialog.tabs.currentIndex() == 3
    assert dialog.run_button.text() == "Run 3 items"
    calls = []
    dialog.runRequested.connect(calls.append)
    dialog.run_button.click()
    assert len(calls) == 1
    assert dialog.result() == 0


def test_only_explicit_selected_preview_calculates(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    previewed = []
    dialog = CollectionBatchDialog(actions=_actions(plan, previewed))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog.select_preview_item(2)
    dialog.preview_item_button.click()
    assert previewed == [2]


def test_items_paging_uses_full_inventory_not_preview_row_limit(qtbot, tmp_path):
    full = _preview_result(tmp_path, count=1200)
    plan = replace(full, rows=full.rows[:25])
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    assert dialog.preview_table.rowCount() == 50
    assert not dialog.items_next_button.isHidden()
    assert not dialog.items_previous_button.isEnabled()
    dialog.items_next_button.click()
    assert dialog.preview_table.item(0, 0).data(Qt.UserRole) == 50
    dialog.item_search.setText("1200_field-1200")
    assert dialog.preview_table.rowCount() == 1
    assert dialog.items_next_button.isHidden()
    assert dialog.items_previous_button.isHidden()
    dialog.item_search.clear()
    assert not dialog.items_next_button.isHidden()
    dialog._review_result_item(1199)
    assert dialog.tabs.currentIndex() == 1
    assert dialog._current_item == 1199
    assert dialog.preview_table.item(49, 0).data(Qt.UserRole) == 1199


def test_one_page_keeps_count_and_hides_both_buttons(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    assert "1–3 of 3" in dialog.item_range_label.text()
    assert dialog.items_next_button.isHidden()
    assert dialog.items_previous_button.isHidden()
    dialog.item_search.setText("no such sample")
    assert dialog.preview_table.rowCount() == 0
    assert dialog.items_next_button.isHidden()


def test_async_check_is_busy_and_failure_leaves_recheck_available(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    calls = []
    actions = replace(
        _actions(plan, []), check_batch=lambda values, limit: calls.append(limit)
    )
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.next_button.click()
    assert calls == [50]
    assert dialog._checking_plan
    assert not dialog.next_button.isEnabled()
    dialog._check_batch()
    assert len(calls) == 1
    dialog._show_preview_failure("Missing source folder")
    assert not dialog._checking_plan
    assert dialog.next_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert dialog.tabs.currentIndex() == 1
    assert dialog.tabs.isTabEnabled(1)


def test_failed_run_preflight_cannot_reenable_old_run_plan(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog.tabs.setCurrentIndex(3)
    assert dialog.run_button.isEnabled()
    dialog.show_preflight_error("Source revision changed")
    assert dialog._preview_result is None
    assert dialog._display_plan is plan
    assert not dialog.run_button.isEnabled()
    assert dialog.next_button.text() == "Check batch"


def test_changed_settings_keep_historical_table_but_disable_preview(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog.pattern_edit.setText("*.tif")
    assert dialog._preview_result is None
    assert dialog.preview_table.rowCount() == 3
    assert dialog.preview_table.item(0, 4).text() == "Needs recheck"
    assert not dialog.preview_item_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert dialog.next_button.text() == "Check batch"


def test_same_path_reordered_inventory_does_not_retarget_checked_identity(
    qtbot, tmp_path
):
    plan = _preview_result(tmp_path)
    items = tuple(
        replace(item, parameter_override_source_item_key=str(i) * 64)
        for i, item in enumerate(plan.items, 1)
    )
    plan = replace(plan, items=items)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    dialog._checked_items = {2}
    dialog._current_item = 2
    reordered = replace(
        plan,
        items=tuple(
            replace(item, index=index)
            for index, item in enumerate((items[2], items[0], items[1]), 1)
        ),
        rows=(),
    )
    dialog.apply_preview_result(reordered, preview_representative=False)
    assert dialog._checked_items == {0}
    assert dialog._current_item == 0


def test_override_sample_on_another_page_is_the_next_preview_target(qtbot, tmp_path):
    plan = _preview_result(tmp_path, count=60)
    plan = replace(
        plan,
        items=tuple(
            replace(item, parameter_override_source_item_key=f"key-{index}")
            for index, item in enumerate(plan.items)
        ),
    )
    previewed = []
    dialog = CollectionBatchDialog(actions=_actions(plan, previewed))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog.tabs.setCurrentIndex(2)
    dialog.parameter_override_editor.sourceSelected.emit("key-52", 52)
    dialog.tabs.setCurrentIndex(1)
    assert dialog._item_page == 1
    dialog.preview_item_button.click()
    assert previewed == [52]


def test_double_click_cannot_preview_during_check_or_pending_preview(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    previewed = []
    dialog = CollectionBatchDialog(actions=_actions(plan, previewed))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    for checking, pending in ((True, False), (False, True)):
        dialog._checking_plan = checking
        dialog._representative_pending = pending
        dialog._preview_table_item_double_clicked(dialog.preview_table.item(0, 0))
    assert previewed == []


def test_error_status_remains_correct_after_paging(qtbot, tmp_path):
    plan = _preview_result(tmp_path, count=60)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog.begin_run(60)
    dialog.update_run_progress(53, 60, plan.items[52].batch_id, "running")
    dialog.show_run_error("Example failure")
    dialog._review_result_item(52)
    assert dialog.preview_table.item(2, 5).text() == "Failed"
    assert dialog.preview_table.item(3, 5).text() == "Not run"
    dialog.apply_preview_result(plan, preview_representative=False)
    assert dialog.footer_elapsed_label.text() == ""


def test_failed_check_invalidates_visible_historical_evidence(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog._show_preview_failure("Source revision changed")
    assert dialog.preview_table.rowCount() == 3
    assert dialog.preview_table.item(0, 4).text() == "Needs recheck"
    assert "Ready for" not in dialog.results_panel.summary_label.text()


def test_items_to_overrides_uses_exact_keys_not_editor_row_order(qtbot, tmp_path):
    from napari_vipp._tests.test_ui_batch_parameter_overrides import (
        _parameter,
        _source_item,
    )
    from napari_vipp.core.batch_parameters import batch_source_item_override_key
    from napari_vipp.ui.batch_overrides import BatchOverrideSourceItem

    plan = _preview_result(tmp_path)
    sources = [_source_item(str(i)) for i in range(3)]
    keys = [batch_source_item_override_key("input", source) for source in sources]
    plan = replace(
        plan,
        items=tuple(
            replace(item, parameter_override_source_item_key=key)
            for item, key in zip(plan.items, keys, strict=True)
        ),
    )
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog.configure_parameter_overrides(
        [BatchOverrideSourceItem("input", str(i), sources[i]) for i in (2, 1, 0)],
        [_parameter("threshold", "threshold", "float", 0.0, 100.0)],
        overrides=(),
    )
    assert dialog.parameter_override_editor.selected_source_keys() == (keys[0],)
    dialog.preview_table.item(0, 0).setCheckState(Qt.Unchecked)
    dialog.preview_table.item(2, 0).setCheckState(Qt.Checked)
    assert dialog.parameter_override_editor.selected_source_keys() == (keys[2],)
    dialog.resize(1080, 800)
    dialog.show()
    qtbot.wait(20)
    assert dialog.footer_overrides_button.x() < dialog.next_button.x()
    dialog.footer_overrides_button.click()
    assert dialog.tabs.currentIndex() == 2
    assert dialog.parameter_override_editor.selected_source_keys() == (keys[2],)


def test_demo_is_in_more_menu_and_enter_in_setup_does_not_run(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.show()
    calls = []
    dialog.runRequested.connect(calls.append)
    assert dialog.demo_config_button.isHidden()
    assert dialog.demo_action in dialog.more_menu.actions()
    dialog.input_edit.setFocus()
    qtbot.keyClick(dialog.input_edit, Qt.Key_Return)
    assert not calls
    assert all(not button.autoDefault() for button in dialog.findChildren(QPushButton))


def test_footer_height_stable_during_check_and_run(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.resize(1080, 800)
    dialog.show()
    qtbot.wait(20)
    idle = dialog.footer.height()
    dialog.show_workspace_activity("Checking…", indeterminate=True, state="working")
    qtbot.wait(20)
    assert dialog.footer.height() == idle
    dialog._check_batch()
    dialog.begin_run(3)
    qtbot.wait(20)
    assert dialog.footer.height() == idle
    assert not dialog.cancel_run_button.isHidden()
    dialog.show_run_error("Stopped for test")
    qtbot.wait(20)
    assert dialog.footer.height() == idle


def test_native_theme_switch_updates_visible_workspace(qtbot, tmp_path):
    plan = _preview_result(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    colors = []
    for background, foreground in (("#25282f", "#eef2f6"), ("#ffffff", "#202530")):
        palette = QPalette(dialog.palette())
        palette.setColor(QPalette.Base, QColor(background))
        palette.setColor(QPalette.Text, QColor(foreground))
        palette.setColor(QPalette.AlternateBase, QColor(background))
        dialog.setPalette(palette)
        colors.append(dialog.item_details.styleSheet())
        assert foreground in dialog.item_details.styleSheet()
    assert colors[0] != colors[1]
