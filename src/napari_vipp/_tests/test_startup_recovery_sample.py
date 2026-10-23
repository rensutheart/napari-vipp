"""The real standalone starter remains independent of an interrupted recipe."""

import numpy as np
from napari.components import ViewerModel

from napari_vipp._widget import VippWidget
from napari_vipp.app import _configure_initial_workflow
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.operations import gaussian_blur, otsu_threshold
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow
from napari_vipp.core.workflow_recovery import RecoveryStore


def test_standalone_starter_and_recovered_source_with_real_interrupted_copy(
    qtbot, monkeypatch, tmp_path
):
    root = tmp_path / "recovery"
    monkeypatch.setenv("VIPP_WORKFLOW_RECOVERY_ROOT", str(root))
    interrupted = PrototypePipeline()
    interrupted.nodes["input"].params.update(
        source_mode="sample", sample_name="VIPP synthetic volume"
    )
    writer = interrupted.add_node("save_output")
    assert interrupted.connect("threshold", writer.id).success
    destination = tmp_path / "not-automatically-written" / "result.npy"
    writer.params.update(path=str(destination), enabled="on")
    tab = {
        "tab_id": "interrupted-tab",
        "title": "Interrupted analysis",
        "path": None,
        "base_dir": str(tmp_path),
        "workflow": serialize_workflow(interrupted),
        "dirty": True,
    }
    crashed = RecoveryStore(root)
    checkpoint = crashed.checkpoint([tab], tab["tab_id"], "interrupted edit")
    crashed.close(clean=False)
    original_bytes = {
        path: path.read_bytes() for path in root.rglob("*") if path.is_file()
    }

    widget = VippWidget(
        ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    widget._discard_incomplete_startup_on_close = True
    assert widget._workflow_recovery.pending_recovery
    _configure_initial_workflow(widget)
    expected = widget._sample_payloads()["VIPP synthetic volume"].data
    expected_gaussian = gaussian_blur(expected, sigma=1.2)
    expected_mask = otsu_threshold(expected_gaussian)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail("input")
        and not widget._workflow_tab_switch_block_reason(),
        timeout=10_000,
    )
    assert widget._selected_node_id == "input"
    layer = widget.viewer.layers["VIPP Inspect"]
    assert layer.metadata["node_id"] == "input"
    np.testing.assert_array_equal(layer.data, expected)
    np.testing.assert_allclose(widget.pipeline.outputs["gaussian"], expected_gaussian)
    np.testing.assert_array_equal(widget.pipeline.outputs["threshold"], expected_mask)
    assert widget._initial_pipeline_run_completed
    assert all(path.read_bytes() == data for path, data in original_bytes.items())
    assert not destination.parent.exists()

    widget._restore_recovery_workspace(checkpoint)
    qtbot.waitUntil(
        lambda: widget.pipeline.outputs.get("threshold") is not None
        and widget._active_source_load_id is None
        and widget._active_pipeline_run_id is None,
        timeout=10_000,
    )
    preview = widget.viewer.layers["VIPP Inspect"]
    np.testing.assert_array_equal(preview.data, expected)
    assert not preview.data.flags.writeable
    np.testing.assert_allclose(widget.pipeline.outputs["gaussian"], expected_gaussian)
    np.testing.assert_array_equal(widget.pipeline.outputs["threshold"], expected_mask)
    assert {"input", "gaussian", "threshold"} <= widget.pipeline.completed_node_ids
    assert widget.pipeline.outputs.get(writer.id) is None
    assert writer.id not in widget.pipeline.completed_node_ids
    assert widget._recovery_preview_restricted
    assert widget.pipeline.nodes[writer.id].params["enabled"] == "on"
    assert not destination.parent.exists()
    assert all(path.read_bytes() == data for path, data in original_bytes.items())
