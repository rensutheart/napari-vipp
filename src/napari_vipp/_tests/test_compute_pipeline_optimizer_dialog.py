from napari_vipp.ui.compute_pipeline_optimizer_dialog import (
    PipelineOptimizerDialog,
    PipelineOptimizerWorkerOutcome,
)


def test_optimizer_override_change_invalidates_reviewed_result(qtbot):
    dialog = PipelineOptimizerDialog()
    qtbot.addWidget(dialog)
    dialog._outcome = PipelineOptimizerWorkerOutcome(result=object())
    dialog.apply_button.setEnabled(True)
    dialog.result_table.setVisible(True)

    dialog.override_authored_checkbox.setChecked(True)

    assert dialog.outcome is None
    assert not dialog.apply_button.isEnabled()
    assert dialog.result_table.isHidden()
    assert "Analyze the pipeline again" in dialog.result_label.text()
