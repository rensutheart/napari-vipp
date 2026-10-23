"""Recovery rebuilds safe previews without writing files or granting authority."""

from pathlib import Path

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.workflow import save_workflow
from napari_vipp.core.workflow_recovery import RecoveryStore


@pytest.fixture
def recovery_widget(qtbot):
    viewer = ViewerModel()
    widget = VippWidget(
        viewer, defer_initial_run=True, initial_compute_mode=ComputeMode.CPU
    )
    qtbot.addWidget(widget)
    widget.pipeline.reset_empty_graph()
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    yield widget
    widget._discard_incomplete_startup_on_close = True


def _edit(widget, threshold=0.63):
    node = widget.pipeline.add_node("binary_threshold")
    assert widget.pipeline.connect("input", node.id).success
    node.params["threshold"] = threshold
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    return node


def _wait_for_recovery_preview(widget, qtbot):
    qtbot.waitUntil(
        lambda: not widget._recovery_preview_pending
        and widget._active_source_load_id is None
        and widget._active_pipeline_run_id is None,
        timeout=15_000,
    )


def test_normal_autosave_keeps_original_file_and_dirty_state(
    recovery_widget, tmp_path, qtbot
):
    widget = recovery_widget
    node = _edit(widget, 0.25)
    original = save_workflow(tmp_path / "original.json", widget.pipeline)
    before = original.read_bytes()
    widget._workflow_tabs.current.mark_saved(original)
    node.params["threshold"] = 0.63
    widget._sync_current_workflow_tab_state()
    controller = widget._workflow_recovery
    assert controller.timer.interval() == 2000
    qtbot.waitUntil(lambda: controller.last_checkpoint is not None, timeout=15_000)
    checkpoint = controller.last_checkpoint
    assert checkpoint.tabs[0]["workflow"]["nodes"][-1]["params"]["threshold"] == 0.63
    assert widget._workflow_tabs.current.dirty
    assert original.read_bytes() == before
    assert not widget.pipeline.completed_node_ids
    assert controller.store.recoverable() == []  # Never recover a live owner.


def test_recovered_copy_retains_graph_and_calculates_only_safe_automatic_nodes(
    recovery_widget, qtbot, tmp_path
):
    widget = recovery_widget
    node = _edit(widget)
    widget.pipeline.nodes["input"].params.update(
        source_mode="sample", sample_name="VIPP synthetic volume"
    )
    manual = widget.pipeline.add_node("intensity_histogram")
    widget.pipeline.set_node_auto_recalculate(manual.id, True)
    assert widget.pipeline.connect(node.id, manual.id).success
    downstream = widget.pipeline.add_node("select_table_columns")
    assert widget.pipeline.connect(manual.id, downstream.id).success
    writer = widget.pipeline.add_node("save_output")
    output_path = tmp_path / "must-not-be-published.npy"
    writer.params.update(enabled="on", path=str(output_path))
    assert widget.pipeline.connect(node.id, writer.id).success
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    positions = widget.graph_view.node_positions()
    original = save_workflow(tmp_path / "source.json", widget.pipeline)
    widget._workflow_tabs.current.mark_saved(original)
    before = original.read_bytes()
    checkpoint = widget._workflow_recovery.checkpoint_now(required=True)
    ids = widget._restore_recovery_workspace(checkpoint)
    assert len(ids) == 1
    _wait_for_recovery_preview(widget, qtbot)
    recovered = widget._workflow_tabs.current
    assert recovered.title.startswith("Recovered") and recovered.path is None
    assert recovered.dirty and recovered.session_id == ids[0]
    assert node.id in widget.pipeline.nodes
    assert widget.pipeline.nodes[node.id].params["threshold"] == 0.63
    assert widget.graph_view.node_positions() == positions
    source_data = widget._sample_payloads()["VIPP synthetic volume"].data
    np.testing.assert_array_equal(widget.pipeline.outputs["input"], source_data)
    np.testing.assert_array_equal(widget.pipeline.outputs[node.id], source_data > 0.63)
    assert widget.pipeline.node_execution_states[node.id] == EXECUTION_READY
    assert {"input", node.id} <= widget.pipeline.completed_node_ids
    for paused in (manual.id, downstream.id, writer.id):
        assert paused not in widget.pipeline.completed_node_ids
        assert widget.pipeline.outputs.get(paused) is None
        assert widget.pipeline.node_execution_states[paused] != EXECUTION_READY
    assert widget.pipeline.nodes[manual.id].params["_vipp_auto_recalculate"] is True
    assert widget.pipeline.nodes[writer.id].params["enabled"] == "on"
    assert not output_path.exists()
    assert not recovered.history.can_undo
    assert original.read_bytes() == before


def test_failed_preedit_checkpoint_aborts_without_graph_changes(
    recovery_widget, monkeypatch
):
    widget = recovery_widget
    _edit(widget)
    controller = widget._workflow_recovery
    before = widget._capture_recovery_workspace()

    def denied(*_, **__):
        raise PermissionError("Synthetic disk failure")

    monkeypatch.setattr(controller.store, "checkpoint", denied)
    with pytest.raises(ValueError, match="checkpoint failed"):
        controller.checkpoint_now("before AI patch", required=True)
    assert widget._capture_recovery_workspace() == before


def test_recovery_base_folder_is_not_changed_to_checkpoint_directory(
    recovery_widget, tmp_path
):
    widget = recovery_widget
    tab = widget._capture_recovery_workspace()[0][0]
    assert Path(tab["base_dir"]).is_absolute()
    assert Path(tab["base_dir"]) != Path(widget._workflow_recovery.store.root)


def test_relative_image_binding_survives_changed_working_directory(
    recovery_widget, tmp_path, monkeypatch, qtbot
):
    widget = recovery_widget
    original = tmp_path / "image-folder"
    original.mkdir()
    monkeypatch.chdir(original)
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path="image.ome.tif"
    )
    checkpoint = widget._workflow_recovery.checkpoint_now(required=True)
    monkeypatch.chdir(tmp_path)
    widget._restore_recovery_workspace(checkpoint)
    _wait_for_recovery_preview(widget, qtbot)
    assert widget.pipeline.nodes["input"].params["file_path"] == str(
        original / "image.ome.tif"
    )
    assert not widget.pipeline.completed_node_ids  # The source file does not exist.
    assert widget._last_pipeline_source_signature is None


def test_opened_recovery_not_duplicated_when_bookkeeping_fails(
    recovery_widget, tmp_path, monkeypatch
):
    widget = recovery_widget
    controller = widget._workflow_recovery
    tabs, active = widget._capture_recovery_workspace()
    crashed = RecoveryStore(controller.store.root)
    crashed.checkpoint(tabs, active)
    crashed.close(clean=False)

    def denied(*_, **__):
        raise PermissionError("Synthetic disk full after opening recovered tabs")

    monkeypatch.setattr(controller, "checkpoint_now", denied)
    controller.show_recovery()
    dialog = controller._dialog
    from qtpy.QtWidgets import QPushButton

    button = next(
        b for b in dialog.findChildren(QPushButton) if b.text() == "Recover selected"
    )
    before = len(widget._workflow_tabs)
    button.click()
    assert len(widget._workflow_tabs) == before + 1
    assert "Recovered tabs are open" in widget.status_label.text()
    assert controller._dialog is None
    controller.show_recovery()
    assert controller._dialog is None
    assert len(widget._workflow_tabs) == before + 1
    assert len(controller.store.recoverable()) == 1  # Old copy retained, not erased.


def test_folder_change_requires_durable_new_copy(
    recovery_widget, tmp_path, monkeypatch
):
    import napari_vipp.ui.workflow_recovery as ui

    widget = recovery_widget
    controller = widget._workflow_recovery
    original = controller.store
    selected = tmp_path / "new-recovery-folder"
    selected.mkdir()
    monkeypatch.setattr(
        ui.QFileDialog, "getExistingDirectory", lambda *a: str(selected)
    )
    real_checkpoint = RecoveryStore.checkpoint

    def checkpoint(store, *args, **kwargs):
        if store.root == selected:
            raise PermissionError("Synthetic unavailable destination")
        return real_checkpoint(store, *args, **kwargs)

    monkeypatch.setattr(RecoveryStore, "checkpoint", checkpoint)
    controller.choose_folder()
    assert controller.store is original
    assert controller.last_checkpoint is not None
    assert "current folder is kept" in widget.status_label.text()
    assert original.checkpoint(*widget._capture_recovery_workspace())


@pytest.mark.parametrize("legacy_relative_copy", [False, True])
def test_portable_recipe_recovery_keeps_original_bindings_and_source_proofs(
    recovery_widget, tmp_path, monkeypatch, legacy_relative_copy, qtbot
):
    """A new unsaved recovered tab must not use the recovery folder as its base."""
    import hashlib
    import shutil

    from napari_vipp._tests.test_workflow_paths import _project
    from napari_vipp.core.workflow import load_workflow
    from napari_vipp.core.workflow_paths import transform_workflow_paths

    pipeline, original, _item = _project(tmp_path / "original-project")
    save_workflow(original, pipeline)
    moved = tmp_path / "relocated-project"
    shutil.copytree(original.parent.parent, moved)
    saved_path = moved / "workflows" / original.name
    before_file = saved_path.read_bytes()
    source = moved / "data" / "cells.npy"
    dataset = moved / "data" / "measurements.json"
    before_source, before_dataset = source.read_bytes(), dataset.read_bytes()
    unrelated = tmp_path / "unrelated-working-directory"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    loaded = load_workflow(saved_path)
    widget = recovery_widget
    widget.pipeline.restore_graph(
        loaded["nodes"], loaded["connections"], loaded["output_tunnels"]
    )
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    widget._workflow_tabs.current.mark_saved(saved_path)
    controller = widget._workflow_recovery
    if legacy_relative_copy:
        # Working copies predating portable-path integration may retain
        # relative bindings together with their original containing folder.
        tabs, active = widget._capture_recovery_workspace()
        tabs[0]["workflow"] = transform_workflow_paths(
            tabs[0]["workflow"], saved_path.parent, relative=True
        )
        checkpoint = controller.store.checkpoint(tabs, active)
    else:
        checkpoint = controller.checkpoint_now(required=True)
    ids = widget._restore_recovery_workspace(checkpoint)
    assert len(ids) == 1
    _wait_for_recovery_preview(widget, qtbot)
    params = widget.pipeline.nodes["input"].params
    assert Path(params["file_path"]) == source
    proof = params["_vipp_source_item"]["container"]
    assert Path(proof["uri"]) == source
    expected_proof = loaded["nodes"][0].params["_vipp_source_item"]["container"]
    assert proof == expected_proof
    table = next(
        node
        for node in widget.pipeline.nodes.values()
        if node.operation_id == "table_source"
    )
    output = next(
        node
        for node in widget.pipeline.nodes.values()
        if node.operation_id == "save_output"
    )
    assert Path(table.params["dataset_path"]) == dataset
    assert table.params["dataset_sha256"] == hashlib.sha256(before_dataset).hexdigest()
    assert Path(output.params["path"]) == moved / "results" / "mask.tif"
    assert output.params["enabled"] == "off"
    assert widget._workflow_tabs.current.path is None
    assert widget._workflow_tabs.current.dirty
    # This provenance-only fixture deliberately contains non-NPY marker bytes.
    # Automatic preview must reject those bytes rather than invent a result.
    assert not widget.pipeline.completed_node_ids
    assert widget._last_pipeline_source_signature is None
    assert output.id not in widget.pipeline.completed_node_ids
    assert widget.pipeline.outputs.get(output.id) is None
    assert saved_path.read_bytes() == before_file
    assert source.read_bytes() == before_source
    assert dataset.read_bytes() == before_dataset
    assert not (moved / "results").exists()


def test_new_relative_recipe_bindings_are_frozen_before_cwd_changes(
    recovery_widget, tmp_path, monkeypatch, qtbot
):
    widget = recovery_widget
    authored = tmp_path / "authoring-folder"
    authored.mkdir()
    monkeypatch.chdir(authored)
    widget.pipeline.nodes["input"].params.update(
        source_mode="file path", file_path="input.ome.tif"
    )
    table = widget.pipeline.add_node("table_source")
    table.params.update(dataset_path="measurements.json", dataset_sha256="a" * 64)
    output = widget.pipeline.add_node("save_output")
    output.params["path"] = "results/mask.tif"
    widget._build_graph_from_pipeline()
    widget._sync_current_workflow_tab_state()
    checkpoint = widget._workflow_recovery.checkpoint_now(required=True)
    monkeypatch.chdir(tmp_path)
    widget._restore_recovery_workspace(checkpoint)
    _wait_for_recovery_preview(widget, qtbot)
    assert Path(widget.pipeline.nodes["input"].params["file_path"]) == (
        authored / "input.ome.tif"
    )
    assert Path(widget.pipeline.nodes[table.id].params["dataset_path"]) == (
        authored / "measurements.json"
    )
    assert widget.pipeline.nodes[table.id].params["dataset_sha256"] == "a" * 64
    assert Path(widget.pipeline.nodes[output.id].params["path"]) == (
        authored / "results" / "mask.tif"
    )
    assert not widget.pipeline.completed_node_ids  # No source files were created.
    assert widget._last_pipeline_source_signature is None
