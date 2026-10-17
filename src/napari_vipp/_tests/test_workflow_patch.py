"""Tested exact-node patches stage no live edits, file writes or calculations."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_assessment import commit_cache, state_for
from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp._tests.test_workflow_investigation_finish_review import nominate
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp.core import workflow_investigation as engine_module
from napari_vipp.core.pipeline import (
    MANUAL_AUTO_RECALCULATE_PARAM,
    NODE_LIBRARY_BY_ID,
)
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_assessment import capture_assessment_snapshot
from napari_vipp.core.workflow_investigation import capture_completed_candidate
from napari_vipp.core.workflow_patch import (
    WorkflowPatchError,
    prepare_investigation_patch,
)
from napari_vipp.core.workflow_proposals import capture_proposal_context


def completed(*, protection=False, mesh=False, threshold=80, review=True):
    pipeline, snapshot, bundle = accepted(protection=protection, mesh=mesh)
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle, review_before_finish=review)
    result = update(session, threshold)
    assert result["status"] == "ok" and result["result"]["status"] == "completed", (
        result
    )
    nominate(session, "candidate1")
    if review:
        session.accept_finish_review()
    return pipeline, context, snapshot, bundle, session


def prepare(pipeline, context, baseline, session):
    return prepare_investigation_patch(
        pipeline, context=context, baseline=baseline, session=session
    )


def test_exact_parameter_patch_retains_ids_wires_tunnels_and_all_live_cache_objects(
    monkeypatch,
):
    pipeline, context, snapshot, bundle, session = completed(mesh=True)
    nodes = dict(pipeline.nodes)
    outputs = dict(pipeline.outputs)
    arrays = {
        key: value.copy() for key, value in outputs.items() if type(value) is np.ndarray
    }
    graph = GraphSnapshot.from_pipeline(pipeline)
    usage = (session.actions_used, session.model_calls_used, session.attempted_trials)
    notebook = session.export_notebook()
    monkeypatch.setattr(
        engine_module,
        "execute_pipeline_request",
        lambda *_: pytest.fail("Patch preparation ran science"),
    )
    patch = prepare(pipeline, context, bundle, session)
    original_id = bundle.key_to_node[bundle.output_roles[0]["node"]]
    assert patch.can_apply and patch.changed_node_ids == (original_id,)
    assert patch.source_id == "input" and patch.selected_candidate_id == "candidate1"
    assert patch.session_id == session.session_id
    assert patch.baseline == graph and GraphSnapshot.from_pipeline(pipeline) == graph
    assert patch.candidate.connections == graph.connections
    assert patch.candidate.output_tunnels == graph.output_tunnels
    assert tuple(node.id for node in patch.candidate.nodes) == tuple(pipeline.nodes)
    assert patch.candidate.to_pipeline().nodes[original_id].params["threshold"] == 80
    assert pipeline.nodes[original_id].params["threshold"] == 100
    assert all(pipeline.nodes[key] is value for key, value in nodes.items())
    assert all(pipeline.outputs[key] is value for key, value in outputs.items())
    for key, value in arrays.items():
        np.testing.assert_array_equal(pipeline.outputs[key], value)
    assert patch.affected_node_ids == snapshot.candidate_node_ids
    assert patch.presentation_affected_node_ids == snapshot.candidate_node_ids
    assert usage == (
        session.actions_used,
        session.model_calls_used,
        session.attempted_trials,
    )
    assert session.export_notebook() == notebook
    patch.assert_current(pipeline)


def test_diff_is_defensive_frozen_local_ui_data_with_exact_before_after_values():
    pipeline, context, _, bundle, session = completed()
    patch = prepare(pipeline, context, bundle, session)
    diff = patch.public_diff()
    assert diff["version"] == 1
    change = diff["changes"][0]
    assert change["before_operation"] == change["after_operation"] == "binary_threshold"
    assert change["parameters"] == [
        {
            "name": "threshold",
            "before_present": True,
            "before": 100,
            "after_present": True,
            "after": 80,
        }
    ]
    diff["changes"].clear()
    assert patch.public_diff()["changes"]
    assert "PRIVATE" not in str(patch.public_diff())
    with pytest.raises(FrozenInstanceError):
        patch.source_id = "another"


def test_protected_alias_maps_back_to_original_id_and_not_detached_trial_id():
    pipeline, context, _, bundle, session = completed(protection=True, mesh=True)
    patch = prepare(pipeline, context, bundle, session)
    key = bundle.output_roles[0]["node"]
    assert key == "protected_structure"
    assert patch.changed_node_ids == (bundle.key_to_node[key],)
    assert session.selected_candidate.key_to_node[key] != bundle.key_to_node[key]
    assert session.selected_candidate.evidence["preservation_check"]["passed"]


def test_compatible_operation_replacement_requires_full_ordinary_validated_trial():
    pipeline, _, bundle = accepted()
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle)
    params = {
        p.name: p.default for p in NODE_LIBRARY_BY_ID["otsu_threshold"].parameters
    }
    key = bundle.output_roles[0]["node"]
    result = act(
        session,
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
    nominate(session, "candidate1")
    patch = prepare(pipeline, context, bundle, session)
    original_id = bundle.key_to_node[key]
    assert (
        patch.candidate.to_pipeline().nodes[original_id].operation_id
        == "otsu_threshold"
    )
    assert pipeline.nodes[original_id].operation_id == "binary_threshold"
    assert patch.public_diff()["changes"][0]["after_operation"] == "otsu_threshold"


@pytest.mark.parametrize(
    "kind",
    [
        "graph",
        "source",
        "metadata",
        "baseline_output",
        "selection_output",
        "permission",
        "cancel",
    ],
)
def test_prepared_patch_rejects_stale_source_graph_or_evidence(kind):
    pipeline, context, snapshot, bundle, session = completed()
    patch = prepare(pipeline, context, bundle, session)
    if kind == "graph":
        pipeline.nodes[snapshot.candidate_node_ids[0]].params["threshold"] = 99
    elif kind == "source":
        data = pipeline.outputs["input"].copy()
        data.flat[0] += 1
        commit_cache(pipeline, "input", data)
    elif kind == "metadata":
        pipeline.output_states["input"] = replace(
            pipeline.output_states["input"], kind="label image"
        )
        pipeline.node_output_states["input"][0] = pipeline.output_states["input"]
    elif kind == "baseline_output":
        data = pipeline.outputs[snapshot.candidate_node_ids[0]].copy()
        data.flat[0] = ~data.flat[0]
        commit_cache(
            pipeline,
            snapshot.candidate_node_ids[0],
            data,
            state_for(data, kind="binary mask"),
        )
    elif kind == "selection_output":
        data = session.selected_candidate._outputs[0][1].data
        data.setflags(write=True)
        data.flat[0] = ~data.flat[0]
        data.setflags(write=False)
    elif kind == "permission":
        session.include_output_images = False
    else:
        session.cancel()
    with pytest.raises(ValueError):
        patch.assert_current(pipeline)


def test_nomination_waiting_for_human_review_and_baseline_are_not_patches():
    pipeline, _, bundle = accepted()
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle, review_before_finish=True)
    assert update(session, 80)["status"] == "ok"
    nominate(session, "candidate1")
    with pytest.raises(WorkflowPatchError, match="Finish and accept"):
        prepare(pipeline, context, bundle, session)
    session.accept_finish_review()
    assert prepare(pipeline, context, bundle, session).can_apply
    original = engine_for(pipeline, bundle)
    nominate(original, "candidate0")
    with pytest.raises(WorkflowPatchError, match="completed alternative"):
        prepare(pipeline, context, bundle, original)


def test_worker_bundle_and_another_baseline_have_no_live_patch_authority():
    pipeline, context, _, bundle, session = completed()
    with pytest.raises(ValueError, match="worker bundle"):
        prepare(pipeline, context, session._accepted_bundle, session)
    other_pipeline, _, other_bundle = accepted()
    with pytest.raises(WorkflowPatchError):
        prepare(other_pipeline, context, other_bundle, session)


def test_changed_recipe_wires_or_roles_cannot_be_guessed_onto_existing_nodes():
    pipeline, context, _, bundle, session = completed()
    selected = session.selected_candidate
    recipe = selected.recipe
    recipe["nodes"][0]["key"] = "new_key"
    altered = replace(selected, _recipe=engine_module._json(recipe))
    session._candidates[-1] = altered
    with pytest.raises(WorkflowPatchError, match="changes nodes"):
        prepare(pipeline, context, bundle, session)


def test_local_manual_preference_and_unchanged_executor_fields_are_preserved():
    pipeline, _, _ = accepted()
    node = pipeline.nodes[next(node for node in pipeline.nodes if node != "input")]
    node.params[MANUAL_AUTO_RECALCULATE_PARAM] = False
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (node.id,), "local-policy"
    )
    bundle = capture_completed_candidate(
        pipeline, snapshot, [{"role": "mask", "node": node.id, "port": 0}]
    )
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle)
    assert update(session, 80)["status"] == "ok"
    nominate(session, "candidate1")
    patch = prepare(pipeline, context, bundle, session)
    assert (
        patch.candidate.to_pipeline()
        .nodes[node.id]
        .params[MANUAL_AUTO_RECALCULATE_PARAM]
        is False
    )


def test_branch_patch_preserves_unrelated_nodes_and_existing_output_tunnels():
    pipeline, _, _ = accepted()
    first = next(node_id for node_id in pipeline.nodes if node_id != "input")
    other = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", other.id).success
    other.params["threshold"] = 150
    data = pipeline.outputs["input"] > 150
    data.setflags(write=False)
    commit_cache(pipeline, other.id, data, state_for(data, kind="binary mask"))
    pipeline.add_output_tunnel("Retained branch", other.id)
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (first, other.id), "branches"
    )
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        [
            {"role": "changed_mask", "node": first, "port": 0},
            {"role": "retained_mask", "node": other.id, "port": 0},
        ],
    )
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle)
    assert update(session, 80)["status"] == "ok"
    nominate(session, "candidate1")
    old_other = pipeline.nodes[other.id]
    patch = prepare(pipeline, context, bundle, session)
    assert patch.changed_node_ids == patch.affected_node_ids == (first,)
    assert other.id not in patch.presentation_affected_node_ids
    assert pipeline.nodes[other.id] is old_other and pipeline.outputs[other.id] is data
    assert patch.candidate.output_tunnels == (
        pipeline.output_tunnel("Retained branch"),
    )
    assert patch.candidate.to_pipeline().nodes[other.id].params == old_other.params


def test_only_changed_nodes_reset_executor_derived_fields():
    pipeline, _, _ = accepted()
    first = next(node_id for node_id in pipeline.nodes if node_id != "input")
    holes = pipeline.add_node("fill_holes")
    holes.params.update(spatial_mode="3D ZYX", resolved_spatial_ndim=3)
    assert pipeline.connect(first, holes.id).success
    commit_cache(
        pipeline, holes.id, pipeline.outputs[first], pipeline.output_states[first]
    )
    snapshot = capture_assessment_snapshot(
        pipeline, "input", (first, holes.id), "derived"
    )
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        [
            {"role": "mask", "node": first, "port": 0},
            {"role": "filled", "node": holes.id, "port": 0},
        ],
    )
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle)
    key = next(key for key, value in bundle.key_to_node.items() if value == holes.id)
    result = act(
        session,
        "trial_edit",
        candidate_id="candidate0",
        edits=[
            {
                "kind": "set_parameters",
                "node": key,
                "params": {"max_hole_size": 1},
            }
        ],
    ).public_result()
    assert result["status"] == "ok", result
    nominate(session, "candidate1")
    patch = prepare(pipeline, context, bundle, session)
    after = patch.candidate.to_pipeline()
    assert "resolved_spatial_ndim" not in after.nodes[holes.id].params
    assert pipeline.nodes[holes.id].params["resolved_spatial_ndim"] == 3
    assert patch.public_diff()["changes"][0]["local_fields_reset"] == [
        "resolved_spatial_ndim"
    ]
    # A separate threshold edit must retain the untouched morphology node's
    # local derived field; native cache invalidation, not stripping every node,
    # communicates that downstream output must be recalculated later.
    other_session = engine_for(pipeline, bundle)
    assert update(other_session, 80)["status"] == "ok"
    nominate(other_session, "candidate1")
    other_patch = prepare(pipeline, context, bundle, other_session)
    assert (
        other_patch.candidate.to_pipeline()
        .nodes[holes.id]
        .params["resolved_spatial_ndim"]
        == 3
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("changed_node_ids", ()),
        ("affected_node_ids", ()),
        ("presentation_affected_node_ids", ()),
        ("source_id", "wrong-source"),
    ],
)
def test_prepared_patch_metadata_cannot_be_detached_from_its_exact_diff(field, value):
    pipeline, context, _, bundle, session = completed()
    patch = prepare(pipeline, context, bundle, session)
    altered = replace(patch, **{field: value})
    with pytest.raises(WorkflowPatchError, match="exact tested"):
        altered.assert_current(pipeline)


def test_cold_session_cannot_create_an_in_place_patch():
    from napari_vipp._tests.test_workflow_investigation import session as cold

    pipeline, _, bundle = accepted()
    context = capture_proposal_context(pipeline, "input")
    with pytest.raises(WorkflowPatchError, match="GUI-owned"):
        prepare(pipeline, context, bundle, cold())


def test_completed_replacement_with_different_output_contract_remains_a_branch():
    pipeline, _, bundle = accepted()
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle)
    key = bundle.output_roles[0]["node"]
    result = act(
        session,
        "trial_edit",
        candidate_id="candidate0",
        edits=[
            {
                "kind": "replace_operation",
                "node": key,
                "operation_id": "gaussian_blur",
                "params": {
                    p.name: p.default
                    for p in NODE_LIBRARY_BY_ID["gaussian_blur"].parameters
                },
            }
        ],
    ).public_result()
    assert result["status"] == "ok" and result["result"]["status"] == "completed", (
        result
    )
    nominate(session, "candidate1")
    with pytest.raises(WorkflowPatchError, match="input/output ports"):
        prepare(pipeline, context, bundle, session)


@pytest.mark.parametrize("change", ["insertion", "roles", "no_op"])
def test_actual_completed_topology_role_and_no_op_trials_are_not_in_place_edits(change):
    pipeline, _, bundle = accepted()
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle)
    recipe = session.candidates[0].recipe
    roles = session.candidates[0].output_roles
    if change != "no_op":
        recipe["nodes"][0]["params"]["threshold"] = 80
    if change == "insertion":
        recipe["nodes"].append(
            {
                "key": "extra",
                "operation_id": "binary_threshold",
                "params": dict(recipe["nodes"][0]["params"]),
                "reason": "An explicit extra threshold branch.",
            }
        )
        recipe["connections"].append(
            {
                "source": "input",
                "source_port": 0,
                "target": "extra",
                "target_port": 0,
            }
        )
    elif change == "roles":
        roles[0]["role"] = "new_role"
    result = act(
        session, "trial_candidate", proposal=recipe, output_roles=roles
    ).public_result()
    assert result["status"] == "ok" and result["result"]["status"] == "completed", (
        result
    )
    nominate(session, "candidate1")
    message = "no in-place scientific changes" if change == "no_op" else "changes nodes"
    with pytest.raises(WorkflowPatchError, match=message):
        prepare(pipeline, context, bundle, session)


def test_nine_completed_node_edits_exceed_one_patch_even_when_trial_is_valid():
    pipeline, _, _ = accepted()
    node_ids = [next(node_id for node_id in pipeline.nodes if node_id != "input")]
    for _ in range(8):
        node = pipeline.add_node("binary_threshold")
        node.params["threshold"] = 100
        assert pipeline.connect("input", node.id).success
        data = pipeline.outputs["input"] > 100
        data.setflags(write=False)
        commit_cache(pipeline, node.id, data, state_for(data, kind="binary mask"))
        node_ids.append(node.id)
    snapshot = capture_assessment_snapshot(
        pipeline, "input", tuple(node_ids), "many-nodes"
    )
    bundle = capture_completed_candidate(
        pipeline,
        snapshot,
        [
            {"role": f"mask_{index}", "node": node_id, "port": 0}
            for index, node_id in enumerate(node_ids[:8])
        ],
    )
    context = capture_proposal_context(pipeline, "input")
    session = engine_for(pipeline, bundle)
    recipe = session.candidates[0].recipe
    for node in recipe["nodes"]:
        node["params"]["threshold"] = 80
    result = act(
        session,
        "trial_candidate",
        proposal=recipe,
        output_roles=session.candidates[0].output_roles,
    ).public_result()
    assert result["status"] == "ok" and result["result"]["status"] == "completed", (
        result
    )
    nominate(session, "candidate1")
    with pytest.raises(WorkflowPatchError, match="at most eight"):
        prepare(pipeline, context, bundle, session)
