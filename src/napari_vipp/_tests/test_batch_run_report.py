"""Finished runs lead to human-readable evidence, not accidental revalidation."""

from dataclasses import replace

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QPushButton

from napari_vipp._tests.test_batch_results import _preview, _record, _result
from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_ui_batch import _actions
from napari_vipp.core.batch import BatchStatus
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.batch_results import BatchResultsPanel


def _kept(item):
    record = _record(item, BatchStatus.SKIPPED)
    return replace(
        record,
        outputs=tuple(
            replace(output, existed_at_preflight=True) for output in record.outputs
        ),
    )


def test_report_explains_all_skipped_without_claiming_new_results(qtbot, tmp_path):
    plan = _preview(tmp_path, 14)
    result = _result(plan, tuple(_kept(item) for item in plan.items))
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(plan)
    panel.finish_run(result)
    report = panel.run_report
    assert not report.isHidden()
    assert panel.summary_banner.isHidden() and panel.progress_group.isHidden()
    assert "No samples needed processing" in report.outcome_label.text()
    assert report.fields["Items"].text() == "14 total · 14 skipped"
    assert report.fields["Output files"].text() == "0 saved this run · 14 existing kept"
    assert report.elapsed_label.text() == "Total elapsed 01:02"
    assert report.fields["Saved to"].text() == str(plan.config.output_dir)
    assert report.issues_label.isHidden()
    assert panel.has_run_report  # No JSON file required.
    assert not report.manifest_button.isEnabled()
    assert report.details_toggle.isHidden()


def test_mixed_stopped_report_counts_output_records_and_reports_issues(qtbot, tmp_path):
    plan = _preview(tmp_path, 4)
    saved = _record(plan.items[0], BatchStatus.COMPLETED)
    saved = replace(
        saved, outputs=(replace(saved.outputs[0], overwrote_existing=True),)
    )
    records = (
        saved,
        _kept(plan.items[1]),
        _record(plan.items[2], BatchStatus.FAILED, error="Disk is full."),
        _record(plan.items[3], BatchStatus.CANCELLED),
    )
    result = _result(plan, records)
    result = replace(
        result,
        manifest=replace(
            result.manifest,
            compute={
                "runtime_cleanup_succeeded": False,
                "warnings": ["Device did not respond."],
            },
        ),
    )
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.set_plan(plan)
    panel.finish_run(result)
    report = panel.run_report
    assert "Stopped" in report.outcome_label.text()
    assert report.fields["Output files"].text() == (
        "1 saved this run (1 overwritten) · 1 existing kept · 1 failed · 1 cancelled"
    )
    assert "1 cancelled" in report.fields["Items"].text()
    assert "Restart VIPP" in report.issues_label.text()
    assert "Disk is full." in report.issues_label.text()
    panel.invalidate_plan()
    assert not report.isHidden()  # Changes affect the next run, not this evidence.
    assert not panel.summary_banner.isHidden()
    panel.set_plan(plan)
    assert report.isHidden() and not panel.has_run_report


def test_report_handles_missing_timing(qtbot, tmp_path):
    plan = _preview(tmp_path, 1)
    result = _result(
        plan, (_record(plan.items[0], BatchStatus.FAILED, timing=False),), timing=False
    )
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.finish_run(result)
    assert panel.run_report.fields["Finished"].text() == "Not reported"
    assert panel.run_report.elapsed_label.text() == "Total elapsed —"


@pytest.mark.parametrize("dark", [True, False])
def test_finished_primary_scrolls_to_report_and_new_run_is_explicit(
    qtbot, tmp_path, dark
):
    from napari.qt import get_stylesheet

    plan = _preview(tmp_path, 14)
    checked, previewed, run = [], [], []
    actions = replace(
        _actions(plan, previewed), check_batch=lambda *_: checked.append(True)
    )
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.setPalette(_palette(dark))
    dialog.setStyleSheet(get_stylesheet("dark" if dark else "light"))
    dialog.setFont(QFont("Segoe UI", 12))
    dialog.resize(1100, 760)
    dialog.show()
    dialog.apply_preview_result(plan, preview_representative=False)
    dialog.runRequested.connect(run.append)
    dialog.begin_run(14)
    dialog.finish_run(_result(plan, tuple(_kept(item) for item in plan.items)))
    qtbot.wait(20)
    assert dialog.next_button.text() == "View run report"
    assert dialog.next_button.isEnabled()
    assert not dialog.run_button.isEnabled() and dialog.run_button.isHidden()
    assert dialog._preview_result is None
    assert dialog.tabs.currentIndex() == 3
    panel = dialog.results_panel
    assert panel.run_report.isVisible()
    # The report must use the same readable body size as the result tables,
    # even across napari's styled tab/card boundaries.
    for label in panel.run_report.fields.values():
        assert label.font().pointSizeF() == panel.items_table.font().pointSizeF()
    assert [
        button.text() for button in panel.artifact_toolbar.findChildren(QPushButton)
    ] == [
        "Output folder",
        "Refresh file status",
    ]
    dialog.run_scroll.verticalScrollBar().setValue(
        dialog.run_scroll.verticalScrollBar().maximum()
    )
    dialog.next_button.click()
    qtbot.wait(20)
    top = panel.run_report.mapTo(dialog.run_scroll.viewport(), QPoint()).y()
    assert 0 <= top <= 20
    dialog.tabs.setCurrentIndex(1)
    dialog.next_button.click()
    assert dialog.tabs.currentIndex() == 3
    assert checked == [] and previewed == [] and run == []
    dialog.tabs.setCurrentIndex(0)
    assert dialog.next_button.text() == "Check batch"
    assert panel.has_run_report
    assert checked == []  # Just returning to Setup never starts another check.
    dialog.next_button.click()
    assert checked == [True]
    assert previewed == [] and run == []
    assert dialog.next_button.text() == "Checking…"
    assert not dialog.next_button.isEnabled()
    dialog.apply_preview_result(plan, preview_representative=False)
    assert panel.run_report.isHidden()
    assert dialog.tabs.currentIndex() == 1
    assert dialog._preview_result is plan


def test_file_failures_have_explicit_counts_and_expandable_reasons(qtbot, tmp_path):
    plan = _preview(tmp_path, 5)
    records = tuple(
        # A partial item can have multiple failed output files, so these counts
        # must not be inferred from the number of failed items.
        replace(
            _record(item, BatchStatus.PARTIAL, output_status=BatchStatus.FAILED),
            outputs=tuple(
                replace(
                    _record(item, BatchStatus.FAILED).outputs[0],
                    node_title=f"Output {index}",
                    path=f"{item.index}-{index}.tif",
                    error_message=f"Output failure {item.index}-{index}.",
                )
                for index in (1, 2)
            ),
        )
        for item in plan.items
    )
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.finish_run(_result(plan, records))
    report = panel.run_report
    assert "5 partial" in report.fields["Items"].text()
    assert report.fields["Output files"].text() == "0 saved this run · 10 failed"
    assert report.issues_label.text().startswith("Output failure 1-1.")
    assert "Output failure 5-2." not in report.issues_label.text()
    assert not report.details_toggle.isHidden()
    report.details_toggle.click()
    assert "Output failure 5-2." in report.issues_label.text()
    panel.select_item(4)
    assert "Output 2: Output failure 5-2." in panel.item_error_label.text()
    assert not panel.item_error_label.isHidden()


def test_long_reason_is_readable_without_a_tooltip_or_json(qtbot, tmp_path):
    plan = _preview(tmp_path, 1)
    reason = (
        "Cannot save this output. " + "Further detail. " * 30 + "Final explanation."
    )
    result = _result(plan, (_record(plan.items[0], BatchStatus.FAILED, error=reason),))
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.finish_run(result)
    report = panel.run_report
    assert "Final explanation." not in report.issues_label.text()
    report.details_toggle.click()
    assert reason in report.issues_label.text()
    assert not report.manifest_button.isEnabled()


def test_missing_failure_reason_is_not_invented(qtbot, tmp_path):
    plan = _preview(tmp_path, 1)
    panel = BatchResultsPanel()
    qtbot.addWidget(panel)
    panel.finish_run(_result(plan, (_record(plan.items[0], BatchStatus.FAILED),)))
    assert "1 failed" in panel.run_report.fields["Items"].text()
    assert "1 failed" in panel.run_report.fields["Output files"].text()
    assert "No failure reason was recorded." in panel.run_report.issues_label.text()
    assert "No failure reason was recorded." in panel.item_error_label.text()
