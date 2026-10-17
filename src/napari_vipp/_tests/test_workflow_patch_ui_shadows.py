"""Patch edits invalidate would-run shadows without changing bypass aliases."""

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp._tests.test_workflow_assessment import commit_cache
from napari_vipp._tests.test_workflow_investigation_finish_review import nominate
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp._widget import VippWidget
from napari_vipp.core.execution import PipelinePresentationShadowResult
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_assessment import capture_assessment_snapshot
from napari_vipp.core.workflow_investigation import capture_completed_candidate


@pytest.fixture
def shadow_patch_editor(qtbot, monkeypatch):
    pipeline, baseline, _ = accepted()
    changed = baseline.candidate_node_ids[0]
    bypass = pipeline.add_node("mask_image")
    assert pipeline.connect("input", bypass.id, target_port=0).success
    assert pipeline.connect(changed, bypass.id, target_port=1).success
    assert pipeline.set_node_execution_mode(bypass.id, "bypass")
    downstream = pipeline.add_node("gaussian_blur")
    assert pipeline.connect(bypass.id, downstream.id).success
    unrelated = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", unrelated.id).success
    assert pipeline.set_node_execution_mode(unrelated.id, "bypass")
    source, source_state = pipeline.outputs["input"], pipeline.output_states["input"]
    for node in (bypass, downstream, unrelated):
        commit_cache(pipeline, node.id, source, source_state)
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (changed,), "shadow-patch"
    )
    bundle = capture_completed_candidate(
        pipeline, snapshot, [{"role": "mask", "node": changed, "port": 0}]
    )
    session = engine_for(pipeline, bundle)
    assert update(session, 80)["status"] == "ok"
    nominate(session, "candidate1")
    widget = VippWidget(ViewerModel(), defer_initial_run=True)
    qtbot.addWidget(widget)
    widget._discard_incomplete_startup_on_close = True
    widget.pipeline = pipeline
    widget._workflow_tabs.current.pipeline = pipeline
    widget._build_graph_from_pipeline()
    widget._history.clear()
    widget._sync_current_workflow_tab_state()
    context, owner = widget._capture_workflow_proposal_context("input")
    patch = widget._prepare_workflow_patch(context, owner, bundle, session)
    old_shadows = {}
    for node in (bypass, unrelated):
        shadow_data = np.array(source + 1, copy=True)
        shadow_data.setflags(write=False)
        shadow = PipelinePresentationShadowResult(
            run_id=1,
            node_id=node.id,
            operation_id=node.operation_id,
            output=shadow_data,
            output_state=source_state,
            node_outputs=(shadow_data,),
            node_output_states=(source_state,),
        )
        widget._bypass_shadow_results[node.id] = old_shadows[node.id] = shadow
        widget._bypass_shadow_errors[node.id] = "Previous diagnostic"
        widget._bypass_shadow_pending_node_ids.add(node.id)
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda **_: calls.append("run"))
    return (
        widget,
        context,
        owner,
        patch,
        bypass.id,
        downstream.id,
        unrelated.id,
        old_shadows,
        calls,
    )


def test_secondary_patch_clears_only_affected_shadows_and_preserves_alias_caches(
    shadow_patch_editor,
):
    widget, context, owner, patch, bypass, downstream, unrelated, shadows, calls = (
        shadow_patch_editor
    )
    assert bypass not in patch.affected_node_ids
    assert downstream not in patch.affected_node_ids
    assert bypass in patch.presentation_affected_node_ids
    assert unrelated not in patch.presentation_affected_node_ids
    source = widget.pipeline.outputs["input"]
    assert widget._node_thumbnail_display_payload(bypass)[0] is shadows[bypass].output
    completed = set(widget.pipeline.completed_node_ids)
    widget._apply_workflow_patch(patch, context, owner)
    assert bypass not in widget._bypass_shadow_results
    assert bypass not in widget._bypass_shadow_errors
    assert bypass not in widget._bypass_shadow_pending_node_ids
    assert widget._bypass_shadow_results[unrelated] is shadows[unrelated]
    assert widget._bypass_shadow_errors[unrelated] == "Previous diagnostic"
    assert unrelated in widget._bypass_shadow_pending_node_ids
    assert widget.pipeline.outputs[bypass] is source
    assert widget.pipeline.outputs[downstream] is source
    assert widget.pipeline.outputs[unrelated] is source
    assert {bypass, downstream, unrelated} <= widget.pipeline.completed_node_ids
    assert widget.pipeline.completed_node_ids == completed - set(
        patch.affected_node_ids
    )
    assert widget._node_thumbnail_display_payload(bypass)[0] is source
    assert (
        widget._node_thumbnail_display_payload(unrelated)[0]
        is shadows[unrelated].output
    )
    assert calls == [] and not widget._debounce_timer.isActive()


def test_failed_secondary_patch_restores_exact_shadow_objects_and_pending_state(
    shadow_patch_editor,
    monkeypatch,
):
    widget, context, owner, patch, bypass, _, unrelated, shadows, calls = (
        shadow_patch_editor
    )
    graph = GraphSnapshot.from_pipeline(widget.pipeline)
    outputs = dict(widget.pipeline.outputs)
    errors = dict(widget._bypass_shadow_errors)
    pending = set(widget._bypass_shadow_pending_node_ids)
    original = widget._build_graph_from_pipeline
    attempts = []

    def fail_once(**kwargs):
        if not attempts:
            attempts.append(1)
            raise RuntimeError("Synthetic graphics failure after shadow invalidation")
        return original(**kwargs)

    monkeypatch.setattr(widget, "_build_graph_from_pipeline", fail_once)
    with pytest.raises(ValueError, match="rolled back"):
        widget._apply_workflow_patch(patch, context, owner)
    assert GraphSnapshot.from_pipeline(widget.pipeline) == graph
    assert all(widget.pipeline.outputs[key] is value for key, value in outputs.items())
    assert widget._bypass_shadow_results[bypass] is shadows[bypass]
    assert widget._bypass_shadow_results[unrelated] is shadows[unrelated]
    assert widget._bypass_shadow_errors == errors
    assert widget._bypass_shadow_pending_node_ids == pending
    assert not widget._history.can_undo and calls == []
