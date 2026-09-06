"""Setup follows the reviewed layout without overstating unchecked evidence."""

from dataclasses import replace

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QFormLayout

from napari_vipp._tests.test_batch_table_theme import _palette
from napari_vipp._tests.test_ui_batch import _actions, _preview_result
from napari_vipp.core.batch import BatchOutputPlan, ExistingFilePolicy
from napari_vipp.ui.batch import CollectionBatchDialog
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon


def _output_plan(tmp_path):
    result = _preview_result(tmp_path)
    items = tuple(
        replace(
            item,
            outputs=tuple(
                BatchOutputPlan(
                    node_id=f"output-{index}",
                    node_title=f"Batch Output {index}",
                    tag=f"result-{index}",
                    kind="image",
                    format="npy",
                    path=tmp_path / "outputs" / f"{item.batch_id}-{index}.npy",
                    existing_file_policy=ExistingFilePolicy.ERROR,
                )
                for index in (1, 2)
            ),
        )
        for item in result.items
    )
    return replace(result, items=items)


def test_setup_identifies_live_workflow_without_claiming_a_checked_plan(
    qtbot, tmp_path
):
    plan = _output_plan(tmp_path)
    summary = ["mitotracker-workflow", str(tmp_path / "mitotracker-workflow.json")]
    actions = replace(_actions(plan, []), workflow_summary=lambda: tuple(summary))
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)

    assert "mitotracker-workflow" in dialog.workflow_summary_label.text()
    assert summary[1] in dialog.workflow_summary_label.toolTip()
    assert "not checked" in dialog.setup_state_label.text().lower()
    assert "1 source" in dialog.sources_summary_label.text()
    assert "check batch" in dialog.outputs_summary_label.text().lower()

    summary[:] = ["renamed-workflow *", "Current workflow has unsaved changes"]
    dialog._sync_workspace()
    assert "renamed-workflow" in dialog.workflow_summary_label.text()
    assert "unsaved changes" in dialog.workflow_summary_label.toolTip()


def test_setup_check_counts_are_cleared_after_settings_change(qtbot, tmp_path):
    plan = _output_plan(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()

    assert dialog.setup_state_label.text() == "Checked"
    assert dialog.outputs_summary_label.text() == "2 files"
    assert "3 files" in dialog._source_rows[0]["count_label"].text()
    assert "workflow" in dialog.workflow_summary_label.text().lower()

    dialog.pattern_edit.setText("*.tif")
    assert dialog._preview_result is None
    assert dialog.setup_state_label.text() != "Checked"
    assert "check batch" in dialog.outputs_summary_label.text().lower()
    assert "3 files" not in dialog._source_rows[0]["count_label"].text()


def test_setup_pending_and_failed_checks_do_not_reuse_old_counts(qtbot, tmp_path):
    plan = _output_plan(tmp_path)
    actions = replace(_actions(plan, []), check_batch=lambda _values, _limit: True)
    dialog = CollectionBatchDialog(actions=actions)
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)

    dialog._check_batch()
    assert "checking" in dialog.setup_state_label.text().lower()
    dialog._show_preview_failure("The source folder is missing")
    assert "needs attention" in dialog.setup_state_label.text().lower()
    assert "check batch" in dialog.outputs_summary_label.text().lower()
    assert "3 files" not in dialog._source_rows[0]["count_label"].text()


def test_setup_conflicts_are_attention_not_checked(qtbot, tmp_path):
    plan = replace(_output_plan(tmp_path), collision_count=1)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    assert "needs attention" in dialog.setup_state_label.text().lower()


def test_setup_counts_distinguish_files_from_multiple_series(qtbot, tmp_path):
    plan = _output_plan(tmp_path)
    path = tmp_path / "inputs" / "multi-series.czi"
    plan = replace(
        plan,
        items=tuple(
            replace(
                item,
                primary_source=path,
                source_paths={"input": path},
                source_series_indices={"input": index},
                source_series_names={"input": f"Series {index + 1}"},
            )
            for index, item in enumerate(plan.items)
        ),
    )
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog.apply_preview_result(plan, preview_representative=False)
    count = dialog._source_rows[0]["count_label"].text()
    assert "1 file" in count
    assert "3 source items" in count
    assert "3 files" not in count


def test_artifact_disclosure_preserves_saved_settings_and_checked_plan(qtbot, tmp_path):
    plan = _output_plan(tmp_path)
    dialog = CollectionBatchDialog(actions=_actions(plan, []))
    qtbot.addWidget(dialog)
    dialog._check_batch()
    dialog.tabs.setCurrentIndex(0)
    dialog.resize(1080, 800)
    dialog.show()
    qtbot.wait(10)

    assert dialog.artifacts_content.isHidden()
    assert not dialog.artifacts_toggle.isChecked()
    assert dialog.artifacts_content.isAncestorOf(dialog.workflow_checkbox)
    assert dialog.artifacts_content.isAncestorOf(dialog.script_checkbox)
    before = dialog.values()
    dialog.artifacts_toggle.click()
    assert not dialog.artifacts_content.isHidden()
    assert dialog._preview_result is plan
    assert dialog.workflow_checkbox.isChecked()
    assert not dialog.workflow_checkbox.isEnabled()
    dialog.artifacts_toggle.click()
    assert dialog.artifacts_content.isHidden()
    assert dialog._preview_result is plan
    assert dialog.values() == before

    dialog.artifacts_toggle.click()
    dialog.script_checkbox.setChecked(False)
    dialog.artifacts_toggle.click()
    dialog.artifacts_toggle.click()
    assert not dialog.script_checkbox.isChecked()
    assert dialog.values()["save_python_script"] is False


def test_setup_section_icons_follow_dark_and_light_theme(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    images = []
    for dark in (True, False, True):
        dialog.setPalette(_palette(dark))
        qtbot.wait(1)
        icons = (
            dialog.source_section_icon.pixmap(),
            dialog.destination_section_icon.pixmap(),
        )
        assert all(pixmap is not None and not pixmap.isNull() for pixmap in icons)
        images.append(tuple(pixmap.toImage() for pixmap in icons))
    assert images[0] != images[1]
    assert images[0] == images[2]


def test_source_fields_use_mockup_columns_and_shared_folder_button(qtbot):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    dialog.resize(1080, 800)
    dialog.show()
    qtbot.wait(10)
    row = dialog._source_rows[0]
    card = row["widget"]
    folder = row["folder"].mapTo(card, QPoint(0, 0))
    pattern = row["pattern"].mapTo(card, QPoint(0, 0))
    axes = row["axis_declaration"].mapTo(card, QPoint(0, 0))
    folder_label = row["folder_label"].mapTo(card, QPoint(0, 0))
    assert folder_label.y() < folder.y() < pattern.y()
    assert abs(pattern.y() - axes.y()) <= 3
    assert axes.x() > pattern.x() + row["pattern"].width()
    button = row["browse_button"]
    assert isinstance(button, ToolbarCommandButton)
    assert button.text() == ""
    assert "folder" in button.accessibleName().lower()
    assert button.icon().pixmap(18, 18).toImage() == (
        toolbar_icon("open", dialog.palette()).pixmap(18, 18).toImage()
    )


@pytest.mark.parametrize("width", [1080, 736, 560])
def test_setup_cards_reflow_without_clipping_fields(qtbot, width):
    dialog = CollectionBatchDialog()
    qtbot.addWidget(dialog)
    dialog.resize(width, 800)
    dialog.show()
    qtbot.wait(20)

    assert dialog.width() == width
    source = dialog.source_card.geometry()
    destination = dialog.destination_group.geometry()
    if width >= 960:
        assert source.top() == destination.top()
        assert destination.left() > source.right()
    else:
        assert source.left() == destination.left()
        assert destination.top() > source.bottom()
    assert isinstance(dialog.destination_group.layout(), QFormLayout)
    assert dialog.destination_group.layout().rowWrapPolicy() == QFormLayout.WrapAllRows

    for field in (dialog.input_edit, dialog.pattern_edit, dialog.output_edit):
        left = field.mapTo(dialog.setup_page, QPoint(0, 0)).x()
        assert left >= 0
        assert left + field.width() <= dialog.setup_page.width()
        assert field.width() >= 60
    assert dialog.content_scroll.horizontalScrollBar().maximum() == 0
