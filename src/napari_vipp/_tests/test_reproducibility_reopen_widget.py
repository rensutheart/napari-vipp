"""An exported batch opens in VIPP and accepts folder choices without JSON edits."""

import numpy as np

from napari_vipp._tests.test_batch import _batch_config, _batch_workflow, _write_arrays
from napari_vipp._tests.test_widget import VippWidget, _Viewer
from napari_vipp.core.reproducibility import build_reproducibility_package


def test_open_packaged_workflow_then_choose_folders_and_check(qtbot, tmp_path):
    workflow, outputs = _batch_workflow()
    original_input = tmp_path / "original-input"
    config = _batch_config(workflow, original_input, tmp_path / "old-output", outputs)
    workflow["batch_config"] = config.to_dict()
    package = build_reproducibility_package(workflow)
    shared = tmp_path / "shared"
    shared.mkdir()
    target = shared / "workflow.json"
    target.write_bytes(package.members["workflow.json"])
    recipient_input = tmp_path / "recipient-input"
    _write_arrays(recipient_input, field=np.arange(20, dtype=np.uint16).reshape(4, 5))
    recipient_output = tmp_path / "recipient-output"

    widget = VippWidget(_Viewer())
    qtbot.addWidget(widget)
    widget.load_workflow_file(target)
    qtbot.waitUntil(lambda: not widget._batch_workspace_preview_workers, timeout=10_000)
    dialog = widget._active_collection_batch_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert dialog._preview_result is None  # Redacted paths are not a checked plan.
    assert not dialog.run_button.isEnabled()
    dialog.input_edit.setText(str(recipient_input))
    dialog.output_edit.setText(str(recipient_output))
    assert dialog._check_batch()
    qtbot.waitUntil(lambda: not dialog._checking_plan, timeout=10_000)
    plan = dialog._preview_result
    assert plan is not None, dialog.preview_status.text()
    assert len(plan.items) == 1
    assert plan.config.sources[0].input_dir == recipient_input
    assert plan.config.output_dir == recipient_output
    assert plan.config.outputs[0].node_id == outputs[0]
    assert plan.items[0].outputs[0].path.parent == recipient_output
    assert dialog.run_button.isEnabled()
    assert not recipient_output.exists()  # Checking never processes or saves images.
