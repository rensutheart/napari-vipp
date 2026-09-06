"""A stale run plan has an adjacent check action, separate from disk refresh."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtGui import QFont

from napari_vipp._tests.test_batch_results import _preview, _record, _result
from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_ui_batch import _actions
from napari_vipp.core.batch import BatchStatus
from napari_vipp.ui.batch import CollectionBatchDialog


def _stale_dialog(qtbot, tmp_path, calls, previewed):
    plan = _preview(tmp_path)
    actions = replace(
        _actions(plan, previewed),
        check_batch=lambda values, limit: calls.append((values, limit)),
    )
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    result = _result(plan, (_record(plan.items[0], BatchStatus.COMPLETED),))
    dialog.results_panel.finish_run(result)
    dialog.pattern_edit.setText("*.tif")
    dialog.tabs.setCurrentIndex(3)
    return dialog, plan


@pytest.mark.parametrize("tab", [1, 3])
def test_warning_button_checks_batch_not_files_and_waits_for_success(
    qtbot, tmp_path, tab,
):
    calls, previewed, run = [], [], []
    dialog, plan = _stale_dialog(qtbot, tmp_path, calls, previewed)
    dialog.runRequested.connect(run.append)
    dialog.tabs.setCurrentIndex(tab)
    panel = dialog.results_panel
    check = dialog.review_check_button if tab == 1 else panel.check_batch_button
    assert check.text() == "Check batch again"
    assert not check.isHidden() and check.isEnabled()
    assert check.parentWidget() is (
        dialog.review_banner if tab == 1 else panel.summary_banner
    )
    assert not check.icon().isNull()
    assert dialog.run_recap_label.isHidden()
    assert not dialog.run_button.isEnabled()
    assert panel.items_table.item(0, 1).text() == "Completed"

    panel.refresh_files_button.click()
    assert panel.refresh_files_button.text() == "Refresh file status"
    assert "does not validate the batch" in panel.refresh_files_button.toolTip()
    assert calls == []
    assert dialog._preview_result is None
    assert check.isEnabled()
    assert panel.items_table.item(0, 1).text() == "Completed"

    check.click()
    assert len(calls) == 1
    assert dialog._checking_plan
    assert not check.isEnabled()
    assert check.text() == "Checking…"
    assert panel.items_table.item(0, 1).text() == "Completed"
    check.click()
    assert len(calls) == 1
    assert dialog._preview_result is None

    dialog.apply_preview_result(plan, preview_representative=False)
    assert dialog.tabs.currentIndex() == 1
    assert panel.check_batch_button.isHidden()
    assert dialog.review_check_button.isHidden()
    assert dialog._preview_result is plan
    assert not dialog.run_recap_label.isHidden()
    assert previewed == [] and run == []


def test_failed_check_retains_results_and_offers_retry(qtbot, tmp_path):
    calls = []
    dialog, _plan = _stale_dialog(qtbot, tmp_path, calls, [])
    panel = dialog.results_panel
    panel.check_batch_button.click()
    dialog._show_preview_failure("Source folder is unavailable.")
    dialog.tabs.setCurrentIndex(3)
    assert panel.summary_label.text() == "Source folder is unavailable."
    assert panel.items_table.item(0, 1).text() == "Completed"
    assert panel.check_batch_button.isEnabled()
    assert panel.check_batch_button.text() == "Check batch again"
    assert not dialog.run_button.isEnabled()
    panel.check_batch_button.click()
    assert len(calls) == 2


@pytest.mark.parametrize(
    "state", ["_run_in_progress", "_run_preparing", "_representative_pending"]
)
def test_check_action_cannot_interrupt_active_work(qtbot, tmp_path, state):
    calls = []
    dialog, _plan = _stale_dialog(qtbot, tmp_path, calls, [])
    setattr(dialog, state, True)
    dialog._sync_workspace()
    check = dialog.results_panel.check_batch_button
    assert check.isHidden() or not check.isEnabled()
    check = dialog.review_check_button
    assert check.isHidden() or not check.isEnabled()
    # The command handler is guarded too, not just its button.
    assert dialog._check_batch() is False
    assert calls == []


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("width", [736, 1100])
@pytest.mark.parametrize("tab", [1, 3])
def test_action_and_wrapped_warning_fit_one_banner(qtbot, tmp_path, dark, width, tab):
    from napari.qt import get_stylesheet

    from napari_vipp.ui.palette_roles import theme_colors

    dialog, _plan = _stale_dialog(qtbot, tmp_path, [], [])
    dialog.tabs.setCurrentIndex(tab)
    dialog.setFont(QFont("Segoe UI", 10))
    dialog.setPalette(_palette(dark))
    dialog.setStyleSheet(get_stylesheet("dark" if dark else "light"))
    dialog.resize(width, 700)
    dialog.show()
    qtbot.wait(10)
    panel = dialog.results_panel
    banner = dialog.review_banner if tab == 1 else panel.summary_banner
    text = dialog.preview_status if tab == 1 else panel.summary_label
    check = dialog.review_check_button if tab == 1 else panel.check_batch_button
    table = dialog.preview_table if tab == 1 else panel.items_table
    assert check.isVisible()
    assert banner.rect().contains(check.geometry())
    assert banner.rect().contains(text.geometry())
    assert text.geometry().right() + 8 <= check.geometry().left()
    assert text.height() >= text.heightForWidth(text.width())
    assert (
        check.mapTo(dialog, QPoint()).y()
        < table.mapTo(dialog, QPoint()).y()
    )
    colors = theme_colors(panel.palette())
    assert colors.warning.surface.name() in banner.styleSheet()
    assert colors.warning.foreground.name() in text.styleSheet()


@pytest.mark.parametrize("kind", ["parameter", "node_execution"])
@pytest.mark.parametrize("preview_was_running", [False, True])
def test_override_warning_allows_check_without_forcing_image_preview(
    qtbot, tmp_path, kind, preview_was_running,
):
    from napari_vipp._widget import VippWidget

    calls, previewed, abandoned, stale = [], [], [], []
    dialog, plan = _stale_dialog(qtbot, tmp_path, calls, previewed)
    dialog.tabs.setCurrentIndex(1)
    dialog.set_representative_pending(preview_was_running)
    host = SimpleNamespace(
        _active_collection_batch_dialog=dialog,
        _active_pipeline_run_id=7 if preview_was_running else None,
        _interactive_collection_batch_requested_index=0 if preview_was_running else -1,
        _abandon_background_pipeline_run=lambda: abandoned.append(True),
        pipeline=SimpleNamespace(
            nodes={}, mark_manual_descendants_stale=lambda _: None,
        ),
        batch_navigator=SimpleNamespace(
            set_session_stale=lambda value, **kwargs: stale.append(value),
        ),
        _refresh_batch_effective_parameter_panel=lambda: None,
        _sync_execution_ui=lambda: None,
        _sync_current_workflow_tab_state=lambda: None,
    )
    handler = getattr(VippWidget, f"_collection_batch_{kind}_overrides_changed")
    handler(host, dialog, ())
    assert abandoned == ([True] if preview_was_running else [])
    assert stale == [True]
    assert host._interactive_collection_batch_plan_stale
    assert host._interactive_collection_batch_index == -1
    assert not dialog._representative_pending
    assert "Check batch again" in dialog.preview_status.text()
    assert "Preview batch" not in dialog.preview_status.text()
    assert "check batch again" in dialog.batch_activity_status.text()
    assert "optional after checks pass" in dialog.graph_preview_status.text()
    assert dialog.review_check_button.isEnabled()
    assert not dialog.preview_item_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    dialog.review_check_button.click()
    assert len(calls) == 1
    assert previewed == []
    assert not dialog.review_check_button.isEnabled()
    dialog.apply_preview_result(plan, preview_representative=False)
    assert dialog.review_check_button.isHidden()
    # Checking validates the plan; it never labels the old pixels as updated.
    assert host._interactive_collection_batch_plan_stale
    assert previewed == []
