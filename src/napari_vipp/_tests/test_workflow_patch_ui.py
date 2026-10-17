"""Composition-root patch transactions preserve identity and do not execute."""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp._tests.test_workflow_assessment import commit_cache, state_for
from napari_vipp._tests.test_workflow_investigation_finish_review import nominate
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_assessment import capture_assessment_snapshot
from napari_vipp.core.workflow_investigation import capture_completed_candidate


@pytest.fixture
def patch_editor(qtbot, monkeypatch):
    pipeline, baseline, _ = accepted(mesh=True)
    mask = next(
        n for n in pipeline.nodes.values() if n.operation_id == "binary_threshold"
    )
    mesh = pipeline.nodes[baseline.candidate_node_ids[1]]
    other = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", other.id).success
    data = np.array(pipeline.outputs["input"], copy=True)
    data.setflags(write=False)
    commit_cache(pipeline, other.id, data, state_for(data))
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (mask.id, mesh.id), "patch-test"
    )
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        [
            {"role": "band_mask", "node": mask.id, "port": 0},
            {"role": "band_mesh", "node": mesh.id, "port": 0},
        ],
    )
    session = engine_for(pipeline, bundle, review_before_finish=True)
    result = update(session, 80)
    assert result["status"] == "ok", result
    nominate(session, "candidate1")
    session.accept_finish_review()
    widget = VippWidget(
        ViewerModel(), defer_initial_run=True, initial_compute_mode=ComputeMode.AUTO
    )
    qtbot.addWidget(widget)
    widget._discard_incomplete_startup_on_close = True
    widget.pipeline = pipeline
    widget._workflow_tabs.current.pipeline = pipeline
    widget._build_graph_from_pipeline()
    widget._history.clear()
    widget._sync_current_workflow_tab_state()
    context, owner = widget._capture_workflow_proposal_context("input")
    patch = widget._prepare_workflow_patch(context, owner, bundle, session)
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda **_: calls.append("run"))
    return widget, context, owner, patch, other.id, calls


def test_patch_is_one_nonrunning_edit_and_preserves_unaffected_cache(patch_editor):
    widget, context, owner, patch, other, calls = patch_editor
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    source = widget.pipeline.nodes["input"]
    source_data = widget.pipeline.outputs["input"]
    other_node = widget.pipeline.nodes[other]
    other_data = widget.pipeline.outputs[other]
    positions = widget.graph_view.node_positions()
    widget._apply_workflow_patch(patch, context, owner)
    assert widget._workflow_recovery.last_checkpoint.reason == "before assistant patch"
    assert len(widget._history.undo_stack) == 1
    assert widget.pipeline.nodes["input"] is source
    assert widget.pipeline.outputs["input"] is source_data
    assert widget.pipeline.nodes[other] is other_node
    assert widget.pipeline.outputs[other] is other_data
    assert other in widget.pipeline.completed_node_ids
    assert widget.graph_view.node_positions() == positions
    assert not set(patch.affected_node_ids) & widget.pipeline.completed_node_ids
    assert all(widget.pipeline.outputs.get(i) is None for i in patch.affected_node_ids)
    assert calls == [] and not widget._debounce_timer.isActive()
    after = GraphSnapshot.from_pipeline(widget.pipeline)
    widget.undo()
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before and calls == []
    widget.redo()
    assert GraphSnapshot.from_pipeline(widget.pipeline) == after and calls == []


def test_selected_parameters_previews_and_view_remain_current(patch_editor):
    widget, context, owner, patch, other, calls = patch_editor
    changed = patch.changed_node_ids[0]
    widget._select_node(changed)
    # Selection is presentation-only, so recapture the GUI owner for this edit.
    context, owner = widget._capture_workflow_proposal_context("input")
    widget.graph_view.scale(1.15, 1.15)
    transform = widget.graph_view.transform()
    calls.clear()
    ids = widget._apply_workflow_patch(patch, context, owner)
    assert calls == []
    assert widget._selected_node_id == changed
    assert widget._parameter_widgets["threshold"].value() == 80
    assert widget.graph_view.transform() == transform
    assert set(ids) == set(patch._baseline_bundle.snapshot.candidate_node_ids)
    revision = widget._workflow_assistant_run_revision
    assert revision.edit_kind == "patch"
    assert revision.changed_node_ids == patch.changed_node_ids
    assert set(revision.added_node_ids) > set(revision.changed_node_ids)
    assert widget.pipeline.outputs[other] is not None


def test_disk_failure_blocks_patch_without_mutation(patch_editor, monkeypatch):
    widget, context, owner, patch, _, calls = patch_editor
    before = GraphSnapshot.from_pipeline(widget.pipeline)

    def denied(*_, **__):
        raise PermissionError("Synthetic disk full")

    monkeypatch.setattr(widget._workflow_recovery.store, "checkpoint", denied)
    with pytest.raises(ValueError, match="checkpoint failed"):
        widget._apply_workflow_patch(patch, context, owner)
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert not widget._history.can_undo and calls == []


def test_patch_graphics_failure_rolls_back_identity_cache_and_history(
    patch_editor, monkeypatch
):
    widget, context, owner, patch, other, calls = patch_editor
    before = GraphSnapshot.from_pipeline(widget.pipeline)
    nodes = dict(widget.pipeline.nodes)
    outputs = dict(widget.pipeline.outputs)
    original = widget._build_graph_from_pipeline
    attempts = []

    def fail_once(**kwargs):
        if not attempts:
            attempts.append(1)
            raise RuntimeError("Synthetic graphics failure")
        return original(**kwargs)

    monkeypatch.setattr(widget, "_build_graph_from_pipeline", fail_once)
    with pytest.raises(ValueError, match="rolled back"):
        widget._apply_workflow_patch(patch, context, owner)
    assert GraphSnapshot.from_pipeline(widget.pipeline) == before
    assert all(widget.pipeline.nodes[k] is v for k, v in nodes.items())
    assert all(widget.pipeline.outputs[k] is v for k, v in outputs.items())
    assert other in widget.pipeline.completed_node_ids
    assert not widget._history.can_undo and calls == []
