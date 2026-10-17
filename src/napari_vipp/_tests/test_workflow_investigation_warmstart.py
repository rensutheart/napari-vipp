"""Accepted cache import, detached micro-trials and exact retention authority."""

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_assessment import commit_cache, fixture, state_for
from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp._tests.test_workflow_investigation_experiment_session import plan
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.metadata import AxisMetadata
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID
from napari_vipp.core.workflow_assessment import capture_assessment_snapshot
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationLimits,
    InvestigationSession,
    InvestigationStopped,
    capture_completed_candidate,
)
from napari_vipp.core.workflow_refinement import capture_preservation_constraint


def accepted(*, protection=False, mesh=False):
    pipeline, ids = fixture(mesh=mesh)
    source = pipeline.outputs["input"]
    data = source > 100
    data.setflags(write=False)
    pipeline.nodes[ids[0]].params["threshold"] = 100
    commit_cache(pipeline, ids[0], data, state_for(data, kind="binary mask"))
    if mesh:
        from napari_vipp.core.meshes import mask_to_3d_mesh

        geometry = mask_to_3d_mesh(data, image_state=pipeline.output_states[ids[0]])
        commit_cache(pipeline, ids[1], geometry, geometry.state)
    snapshot = capture_assessment_snapshot(pipeline, "input", ids, "owned-baseline")
    roles = [{"role": "band_mask", "node": ids[0], "port": 0}]
    if mesh:
        roles.append({"role": "band_mesh", "node": ids[1], "port": 0})
    constraint = (
        capture_preservation_constraint(snapshot, (ids[0], 0), output_sharing=True)
        if protection
        else None
    )
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        roles,
        preservation_constraint=constraint,
        protected_role="band_mask" if protection else None,
    )
    return pipeline, snapshot, bundle


def engine_for(pipeline, bundle, **kwargs):
    return InvestigationSession.from_completed(
        pipeline,
        bundle,
        "Preserve the accepted structure and improve the result.",
        compute_request=ComputeRequest(),
        include_output_images=True,
        assert_current=lambda: None,
        **kwargs,
    )


def update(engine, threshold):
    key = engine.candidates[0].output_roles[0]["node"]
    return act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        edits=[
            {"kind": "set_parameters", "node": key, "params": {"threshold": threshold}}
        ],
    ).public_result()


def test_import_is_no_run_and_zero_trial_cost_with_exact_explicit_roles(monkeypatch):
    pipeline, snapshot, bundle = accepted()
    import napari_vipp.core.workflow_investigation as module

    monkeypatch.setattr(
        module, "execute_pipeline_request", lambda *_: pytest.fail("baseline was rerun")
    )
    engine = engine_for(pipeline, bundle, limits=InvestigationLimits(max_attempts=1))
    assert engine.attempted_trials == 0
    assert engine.retained_candidate_id == "candidate0"
    assert engine.public_request()["budgets"]["attempts_remaining"] == 1
    candidate = engine.selected_candidate
    assert candidate.run_id == snapshot.run_id
    assert candidate.key_to_node == bundle.key_to_node
    assert candidate.graph == snapshot.baseline
    np.testing.assert_array_equal(
        candidate._outputs[0][1].data, pipeline.outputs[snapshot.candidate_node_ids[0]]
    )
    assert not candidate._outputs[0][1].data.flags.writeable
    assert "PRIVATE" not in str(engine.public_request())


def test_parameter_microtrial_preserves_keys_roles_wires_and_accepted_graph():
    pipeline, snapshot, bundle = accepted()
    before = snapshot.baseline
    engine = engine_for(pipeline, bundle, limits=InvestigationLimits(max_attempts=1))
    result = update(engine, 150)
    assert result["status"] == "ok", result
    assert result["result"]["status"] == "completed"
    candidate = engine.candidates[-1]
    assert candidate.recipe["connections"] == engine.candidates[0].recipe["connections"]
    assert candidate.output_roles == engine.candidates[0].output_roles
    assert candidate.evidence["edit_receipt"]["parent_candidate_id"] == "candidate0"
    assert engine.retained_candidate_id == "candidate0"
    assert engine.attempted_trials == 1
    assert update(engine, 160)["status"] == "error"
    snapshot.assert_current(pipeline)
    assert snapshot.baseline == before
    np.testing.assert_array_equal(
        candidate._outputs[0][1].data, pipeline.outputs["input"] > 150
    )


def test_operation_replacement_uses_ordinary_validation_and_explicit_all_parameters():
    pipeline, _, bundle = accepted()
    engine = engine_for(pipeline, bundle)
    params = {
        p.name: p.default for p in NODE_LIBRARY_BY_ID["otsu_threshold"].parameters
    }
    key = bundle.output_roles[0]["node"]
    result = act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        edits=[
            {
                "kind": "replace_operation",
                "node": key,
                "operation_id": "otsu_threshold",
                "params": params,
            }
        ],
    ).public_result()
    assert result["status"] == "ok", result
    assert engine.candidates[-1].recipe["nodes"][0]["operation_id"] == "otsu_threshold"


def test_lost_protected_region_rejects_without_repair_and_keeps_baseline():
    pipeline, _, bundle = accepted(protection=True)
    engine = engine_for(pipeline, bundle)
    result = update(engine, 150)
    assert result["status"] == "ok", result
    assert result["result"]["status"] == "rejected"
    assert not result["result"]["preservation_check"]["passed"]
    assert engine.selected_candidate.candidate_id == "candidate0"
    assert (
        act(
            engine, "retain_candidate", candidate_id="candidate1", reason="Pick latest"
        ).public_result()["status"]
        == "error"
    )
    assert (
        act(
            engine, "finish", candidate_id="candidate1", summary="Done", limitations=[]
        ).public_result()["status"]
        == "error"
    )
    inspect = act(
        engine, "inspect_candidate", candidate_id="candidate1", role="band_mask"
    ).public_result()
    assert inspect["status"] == "ok"
    np.testing.assert_array_equal(
        engine.candidates[-1]._outputs[0][1].data, pipeline.outputs["input"] > 150
    )


def test_protected_mesh_trial_uses_actual_lineage_and_same_source():
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle)
    result = update(engine, 80)
    assert result["status"] == "ok", result
    check = result["result"]["preservation_check"]
    assert check["passed"]
    assert check["mesh_creators"] and check["mesh_deliverables"]
    assert engine.candidates[-1].graph != bundle.snapshot.baseline


@pytest.mark.parametrize(
    "mode", ["graph", "source", "role", "unsafe", "consent", "empty_role"]
)
def test_warm_start_fails_on_stale_unsafe_or_ambiguous_inputs(mode):
    pipeline, snapshot, bundle = accepted()
    if mode == "graph":
        pipeline.nodes[snapshot.candidate_node_ids[0]].params["threshold"] = 99
    elif mode == "source":
        changed = pipeline.outputs["input"].copy()
        changed[0, 0, 0] += 1
        changed.setflags(write=False)
        commit_cache(pipeline, "input", changed)
    elif mode == "role":
        with pytest.raises(ValueError):
            capture_completed_candidate(
                pipeline, snapshot, [{"role": "band", "node": "missing", "port": 0}]
            )
        return
    elif mode == "unsafe":
        pipeline.nodes[snapshot.candidate_node_ids[0]].operation_id = "save_output"
    elif mode == "empty_role":
        with pytest.raises(ValueError):
            capture_completed_candidate(pipeline, snapshot, [])
        return
    with pytest.raises(ValueError):
        if mode == "consent":
            InvestigationSession.from_completed(
                pipeline,
                bundle,
                "Goal",
                compute_request=ComputeRequest(),
                include_output_images=False,
                assert_current=lambda: None,
            )
        else:
            engine_for(pipeline, bundle)


def test_capture_and_session_guards_cover_unselected_role_replacement_and_expiry():
    pipeline, snapshot, bundle = accepted()
    now = [0.0]
    engine = engine_for(pipeline, bundle, clock=lambda: now[0])
    now[0] = 1801
    with pytest.raises(InvestigationStopped):
        engine.public_request()
    data = pipeline.outputs[snapshot.candidate_node_ids[0]].copy()
    commit_cache(pipeline, snapshot.candidate_node_ids[0], data)
    with pytest.raises(ValueError):
        bundle.assert_current(pipeline)


def test_failed_edit_consumes_attempt_without_running_or_changing_baseline(monkeypatch):
    pipeline, _, bundle = accepted()
    engine = engine_for(pipeline, bundle)
    import napari_vipp.core.workflow_investigation as module

    monkeypatch.setattr(
        module, "execute_pipeline_request", lambda *_: pytest.fail("invalid edit ran")
    )
    result = act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        edits=[
            {"kind": "set_parameters", "node": "missing", "params": {"threshold": 80}}
        ],
    ).public_result()
    assert result["status"] == "error"
    assert engine.attempted_trials == 1
    assert engine.retained_candidate_id == "candidate0"
    assert engine.candidates[-1].status == "failed"


def test_detached_source_only_cache_bundle_still_has_actual_accepted_outputs():
    pipeline, snapshot, bundle = accepted()
    detached = snapshot.baseline.to_pipeline()
    commit_cache(
        detached, "input", pipeline.outputs["input"], pipeline.output_states["input"]
    )
    engine = engine_for(detached, bundle)
    assert (
        engine.candidates[0]._outputs[0][1].fingerprint
        == bundle.snapshot._outputs[0].binding.fingerprint
    )
    assert all(
        node not in detached.completed_node_ids for node in snapshot.candidate_node_ids
    )
    assert (
        engine.public_request()["accepted_candidate"]["execution"]["new_calculation"]
        is False
    )


def test_protection_can_name_actual_mesh_mask_while_constraint_uses_equal_labels():
    pipeline, snapshot, _ = accepted(mesh=True)
    labels = pipeline.add_node("label_connected_components")
    pipeline.nodes[labels.id].params["spatial_mode"] = "3D ZYX"
    assert pipeline.connect(snapshot.candidate_node_ids[0], labels.id).success
    data = pipeline.outputs[snapshot.candidate_node_ids[0]].astype(np.uint16) * 7
    commit_cache(pipeline, labels.id, data, state_for(data, kind="label image"))
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (*snapshot.candidate_node_ids, labels.id), "baseline-labels"
    )
    constraint = capture_preservation_constraint(
        snapshot, (labels.id, 0), output_sharing=True
    )
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        [
            {"role": "band_mask", "node": snapshot.candidate_node_ids[0], "port": 0},
            {"role": "band_mesh", "node": snapshot.candidate_node_ids[1], "port": 0},
        ],
        preservation_constraint=constraint,
        protected_role="band_mask",
    )
    engine = engine_for(pipeline, bundle)
    assert engine.candidates[0].output_roles[0]["node"] == "protected_structure"
    assert len(engine.candidates[0]._outputs) == 2
    assert any(output.node_id == labels.id for output in bundle.snapshot._outputs)
    result = update(engine, 80)
    assert result["status"] == "ok", result
    assert result["result"]["preservation_check"]["passed"]


def test_prospective_experiment_still_required_and_failed_request_does_not_bypass_it():
    pipeline, _, bundle = accepted()
    engine = engine_for(pipeline, bundle, require_experiment_plan=True)
    assert update(engine, 80)["status"] == "error"
    assert engine.attempted_trials == 0
    engine.record_model_call(request=engine.public_request())
    receipt = plan(engine).public_result()
    assert receipt["status"] == "ok", receipt
    experiment_id = receipt["result"]["experiment_id"]
    result = act(
        engine,
        "trial_edit",
        candidate_id="candidate0",
        experiment_id=experiment_id,
        edits=[
            {
                "kind": "set_parameters",
                "node": bundle.output_roles[0]["node"],
                "params": {"threshold": 80},
            }
        ],
    ).public_result()
    assert result["status"] == "ok", result
    assert (
        act(
            engine,
            "finish",
            candidate_id="candidate0",
            summary="Keep prior",
            limitations=[],
        ).public_result()["status"]
        == "error"
    )


def test_accepted_grid_changes_and_cancel_are_not_repaired():
    pipeline, snapshot, bundle = accepted()
    changed = snapshot.baseline.to_pipeline()
    state = pipeline.output_states["input"]
    commit_cache(
        changed,
        "input",
        pipeline.outputs["input"],
        replace(state, axes=(replace(state.axes[0], scale=2), *state.axes[1:])),
    )
    with pytest.raises(InvestigationError):
        engine_for(changed, bundle)
    engine = engine_for(pipeline, bundle)
    engine.cancel()
    with pytest.raises(InvestigationStopped):
        engine.record_model_call(request=engine.public_request())


def test_review_only_nodes_are_preserved_without_fabricated_completed_cache():
    pipeline, snapshot, _ = accepted()
    review = pipeline.add_node("review_images")
    assert pipeline.connect(snapshot.candidate_node_ids[0], review.id).success
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (*snapshot.candidate_node_ids, review.id), "with-review"
    )
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        [{"role": "band_mask", "node": snapshot.candidate_node_ids[0], "port": 0}],
    )
    engine = engine_for(pipeline, bundle)
    result = update(engine, 80)
    assert result["status"] == "ok", result
    assert any(
        node["operation_id"] == "review_images"
        for node in engine.candidates[-1].recipe["nodes"]
    )
    assert result["result"]["execution"]["presentation_exceptions"]


def test_detached_protection_rejects_forged_baseline_nomination_and_source():
    pipeline, snapshot, bundle = accepted(protection=True)
    constraint = bundle.preservation_constraint
    _, other_snapshot, _ = accepted()
    with pytest.raises(ValueError, match="Only the imported original"):
        constraint.check_detached(
            bundle.snapshot,
            other_snapshot,
            (other_snapshot.candidate_node_ids[0], 0),
            accepted_baseline=True,
        )
    with pytest.raises(ValueError, match="exact nominated"):
        constraint.check_detached(
            bundle.snapshot, other_snapshot, (other_snapshot.candidate_node_ids[0], 0)
        )
    changed = pipeline.outputs["input"].copy()
    changed[0, 0, 0] += 1
    changed.setflags(write=False)
    from napari_vipp.core.workflow_assessment import _capture_binding

    wrong_source = replace(
        other_snapshot,
        _source=_capture_binding(
            "input", 0, changed, pipeline.output_states["input"], [0]
        ),
    )
    with pytest.raises(ValueError, match="identical source"):
        constraint.check_detached(
            bundle.snapshot, wrong_source, (other_snapshot.candidate_node_ids[0], 0)
        )
    snapshot.assert_current(pipeline)


def test_full_recipe_cannot_substitute_another_protected_role():
    pipeline, _, bundle = accepted(protection=True)
    engine = engine_for(pipeline, bundle)
    proposal = engine.candidates[0].recipe
    proposal["nodes"][0]["key"] = "decoy"
    proposal["connections"][0]["target"] = "decoy"
    result = act(
        engine,
        "trial_candidate",
        proposal=proposal,
        output_roles=[{"role": "band_mask", "node": "decoy", "port": 0}],
    ).public_result()
    assert result["status"] == "error"
    assert engine.retained_candidate_id == "candidate0"
    assert engine.candidates[-1].status == "failed"


@pytest.mark.parametrize(
    "operation,parameter",
    [
        ("add_metadata_columns", "metadata_columns"),
        ("plot_results", "title"),
        ("select_table_columns", "columns"),
    ],
)
def test_existing_arbitrary_private_text_is_not_exposed_or_erased(operation, parameter):
    pipeline, snapshot, _ = accepted()
    visible = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", visible.id).success
    commit_cache(pipeline, visible.id, pipeline.outputs[snapshot.candidate_node_ids[0]])
    node = pipeline.nodes[snapshot.candidate_node_ids[0]]
    node.operation_id = operation
    node.params.clear()
    node.params.update(
        {p.name: p.default for p in NODE_LIBRARY_BY_ID[operation].parameters}
    )
    node.params[parameter] = "PRIVATE_PATIENT_123"
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (*snapshot.candidate_node_ids, visible.id), "private-text"
    )
    with pytest.raises(InvestigationError, match="free-text") as error:
        capture_completed_candidate(
            pipeline, snapshot, [{"role": "output", "node": visible.id, "port": 0}]
        )
    assert "PRIVATE" not in str(error.value)
    assert node.params[parameter] == "PRIVATE_PATIENT_123"


@pytest.mark.parametrize("mode", ["axes", "units", "dtype"])
def test_explicit_otherwise_unselected_intermediate_is_validated(mode):
    pipeline, snapshot, _ = accepted()
    first = snapshot.candidate_node_ids[0]
    last = pipeline.add_node("binary_threshold")
    assert pipeline.connect(first, last.id).success
    valid = pipeline.outputs[first].copy()
    commit_cache(pipeline, last.id, valid, state_for(valid, kind="binary mask"))
    if mode == "axes":
        data = valid[np.newaxis].copy()
        state = state_for(
            data,
            kind="binary mask",
            axes=(
                AxisMetadata("c", "channel", None),
                *pipeline.output_states[first].axes,
            ),
        )
    else:
        data = valid
        state = state_for(data, kind="binary mask")
        if mode == "units":
            state = replace(
                state,
                axes=(replace(state.axes[0], unit="PRIVATE_UNIT"), *state.axes[1:]),
            )
        else:
            state = replace(state, dtype="float32")
    commit_cache(pipeline, first, data, state)
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (first, last.id), "unselected-invalid"
    )
    assert all(out.node_id != first for out in snapshot._outputs)
    with pytest.raises(ValueError):
        capture_completed_candidate(
            pipeline, snapshot, [{"role": "intermediate", "node": first, "port": 0}]
        )


def test_private_worker_constraint_does_not_read_original_gui_buffers():
    pipeline, snapshot, bundle = accepted(protection=True)
    detached = snapshot.baseline.to_pipeline()
    copied_source = pipeline.outputs["input"].copy()
    copied_source.setflags(write=False)
    commit_cache(detached, "input", copied_source, pipeline.output_states["input"])
    engine = engine_for(detached, bundle)
    assert engine._accepted_bundle._live_snapshot is None
    for output in engine._accepted_bundle.snapshot._outputs:
        assert output.binding.live is output.binding.data
        assert output.binding.data is not pipeline.outputs[output.node_id]
    original = pipeline.outputs[snapshot.candidate_node_ids[0]]
    original.setflags(write=True)
    original[:] = False
    original.setflags(write=False)
    # A private worker still has the exact accepted frozen data; only GUI
    # adoption authority can (and must) reject the changed live cache.
    assert engine.public_request()["preservation_constraint"]["protected_count"] > 0
    with pytest.raises(ValueError):
        bundle.assert_current(pipeline)


def test_accepted_retained_pool_uses_aggregate_source_plus_output_budget(monkeypatch):
    pipeline, snapshot, _ = accepted()
    import napari_vipp.core.workflow_investigation as module

    monkeypatch.setattr(module, "MAX_SNAPSHOT_BYTES", pipeline.outputs["input"].nbytes)
    with pytest.raises(InvestigationError, match="64 MiB"):
        capture_completed_candidate(
            pipeline,
            snapshot,
            [{"role": "band", "node": snapshot.candidate_node_ids[0], "port": 0}],
        )


def test_protected_trial_cannot_use_undisclosed_mesh_output():
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle)
    result = act(
        engine,
        "trial_candidate",
        proposal=engine.candidates[0].recipe,
        output_roles=[engine.candidates[0].output_roles[0]],
    ).public_result()
    assert result["status"] == "error"
    assert engine.retained_candidate_id == "candidate0"
    assert engine.candidates[-1].status == "failed"
