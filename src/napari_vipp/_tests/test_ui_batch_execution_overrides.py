from pathlib import Path

import pytest

from napari_vipp.core.batch import (
    BatchConfig,
    BatchNodeExecutionMode,
    BatchNodeExecutionOverride,
    BatchOutputConfig,
    BatchSourceConfig,
)
from napari_vipp.core.batch_execution import BatchNodeExecutionSpec
from napari_vipp.ui.batch import CollectionBatchDialog


def _spec():
    return BatchNodeExecutionSpec(
        node_id="crop_stack_1",
        title="Crop Stack",
        operation_id="crop_stack",
        workflow_mode=BatchNodeExecutionMode.RUN,
    )


def _config(*, node_id="crop_stack_1"):
    return BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256="0" * 64,
        output_dir=Path("outputs"),
        sources=(
            BatchSourceConfig(
                "input",
                "Image Source",
                Path("inputs"),
                "*.npy",
            ),
        ),
        outputs=(
            BatchOutputConfig(
                "batch_output_1",
                "Batch Output",
                "result",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
        node_execution_overrides=(BatchNodeExecutionOverride(node_id, "bypass"),),
    )


def test_batch_node_behavior_is_dynamic_and_use_workflow_is_omitted(qtbot):
    dialog = CollectionBatchDialog(execution_nodes=(_spec(),))
    qtbot.addWidget(dialog)
    combo = dialog._node_execution_combos["crop_stack_1"]

    assert not dialog.node_execution_group.isHidden()
    assert combo.itemText(0) == "Use workflow (currently Run)"
    assert "node_execution_overrides" not in dialog.values()

    with qtbot.waitSignal(dialog.nodeExecutionOverridesChanged):
        combo.setCurrentIndex(combo.findData("bypass"))
    assert dialog.values()["node_execution_overrides"] == (
        BatchNodeExecutionOverride("crop_stack_1", "bypass"),
    )
    assert dialog._preview_result is None

    combo.setCurrentIndex(combo.findData("run"))
    assert dialog.node_execution_overrides() == (
        BatchNodeExecutionOverride("crop_stack_1", "run"),
    )

    combo.setCurrentIndex(combo.findData(None))
    assert "node_execution_overrides" not in dialog.values()

    empty = CollectionBatchDialog()
    qtbot.addWidget(empty)
    assert empty.node_execution_group.isHidden()


def test_loaded_batch_node_behavior_restores_without_editing_signal(qtbot):
    dialog = CollectionBatchDialog(execution_nodes=(_spec(),))
    qtbot.addWidget(dialog)
    emissions = []
    dialog.nodeExecutionOverridesChanged.connect(emissions.append)

    dialog._apply_config(_config())

    combo = dialog._node_execution_combos["crop_stack_1"]
    assert combo.currentData() == "bypass"
    assert dialog.node_execution_overrides() == (
        BatchNodeExecutionOverride("crop_stack_1", "bypass"),
    )
    assert emissions == []


def test_loaded_batch_node_behavior_rejects_missing_reviewed_node(qtbot):
    dialog = CollectionBatchDialog(execution_nodes=(_spec(),))
    qtbot.addWidget(dialog)

    with pytest.raises(ValueError, match="not available"):
        dialog._apply_config(_config(node_id="missing"))
