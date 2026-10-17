"""Passive restart checkpoints, with real bounded CPU trials but no provider."""

from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_investigation import (
    act,
    proposal,
    session,
    source_pipeline,
    trial,
)
from napari_vipp._tests.test_workflow_investigation_experiment_session import (
    assessment,
    linked_trial,
    plan,
)
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_assistant_session import (
    load_assistant_session,
    make_checkpoint,
    save_assistant_session,
)
from napari_vipp.core.workflow_investigation import InvestigationLimits
from napari_vipp.core.workflow_investigation_persistence import (
    AssistantCheckpointError,
    checkpoint_investigation,
    restore_investigation,
)


def restore(checkpoint, pipeline, **kwargs):
    return restore_investigation(
        checkpoint,
        pipeline,
        "input",
        compute_request=kwargs.pop("compute_request", ComputeRequest()),
        assert_current=kwargs.pop("assert_current", lambda: None),
        include_input_images=kwargs.pop("include_input_images", False),
        include_output_images=kwargs.pop("include_output_images", True),
        **kwargs,
    )


def altered(checkpoint, change):
    document = checkpoint.document
    change(document)
    return make_checkpoint(
        document, {name: checkpoint.blob_bytes(name) for name in checkpoint.blob_names}
    )


def test_cold_roundtrip_keeps_exact_evidence_clock_history_and_resumes(
    tmp_path, monkeypatch
):
    now = [100.0]
    pipeline = source_pipeline()
    engine = session(
        pipeline,
        include_input_images=True,
        continuation_enabled=True,
        clock=lambda: now[0],
        limits=InvestigationLimits(max_attempts=2),
    )
    engine.record_model_call(request=engine.public_request())
    assert trial(engine).public_result()["status"] == "ok"
    inspect = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        views=[{"axis": "Z", "indices": [1], "overlay_source": True}],
    )
    assert inspect.public_result()["status"] == "ok"
    engine.record_model_call(request=engine.public_request())
    now[0] += 13
    act(engine, "question", question="Keep this starting convention?")
    assert engine.paused
    before_graph = GraphSnapshot.from_pipeline(pipeline)
    before_notebook = engine.export_notebook()
    before_output = engine.candidates[0]._outputs[0][1]
    checkpoint = checkpoint_investigation(
        engine,
        ui_state={
            "chat": [["You", "my literal message"]],
            "stage": 1,
            "pause_answer_draft": "  yes, go  ",
        },
    )
    path = tmp_path / "saved.vipp-assistant"
    receipt = save_assistant_session(path, checkpoint)
    assert receipt.byte_count == path.stat().st_size
    checkpoint = load_assistant_session(path)
    assert checkpoint.document["ui_state"]["pause_answer_draft"] == "  yes, go  "
    import napari_vipp.core.workflow_investigation as investigation

    execute = investigation.execute_pipeline_request
    monkeypatch.setattr(
        investigation,
        "execute_pipeline_request",
        lambda *_: pytest.fail("restore calculated a pipeline"),
    )
    monkeypatch.setattr(
        investigation.InvestigationSession,
        "_render_views",
        lambda *_: pytest.fail("restore rendered a preview"),
    )
    now[0] = 10000.0
    result = restore(
        checkpoint, pipeline, include_input_images=True, clock=lambda: now[0]
    )
    assert result.resumable, result.reason
    resumed = result.session
    assert resumed is not engine and resumed.session_id == engine.session_id
    assert resumed.paused and not resumed.finished
    assert resumed.pending_question == engine.pending_question
    assert resumed.export_notebook() == before_notebook
    assert resumed.actions_used == engine.actions_used
    assert resumed.model_calls_used == engine.model_calls_used == 2
    assert resumed._image_bytes == engine._image_bytes > 0
    assert resumed.attempted_trials == engine.attempted_trials == 1
    assert resumed.continuation_state["active_seconds_used"] == 13
    assert resumed._sent_numeric == resumed._sent_previews == set()
    assert resumed._sent_preview_owners == {}
    assert resumed._last_transmission is None
    assert resumed._agent_tool_lock is not engine._agent_tool_lock
    assert resumed._authority is not engine._authority
    cached = resumed.candidates[0]._outputs[0][1]
    assert cached.fingerprint == before_output.fingerprint
    np.testing.assert_array_equal(cached.data, before_output.data)
    assert not cached.data.flags.writeable
    assert cached.data is not before_output.data
    assert [p.manifest() for p in resumed.preview_images()] == [
        p.manifest() for p in engine.preview_images()
    ]
    assert GraphSnapshot.from_pipeline(pipeline) == before_graph
    now[0] += 100
    assert resumed.continuation_state["active_seconds_used"] == 13
    resumed.resume("  yes, go  ")
    assert resumed.dialogue.clarifications[-1] == (
        "Keep this starting convention?",
        "  yes, go  ",
    )
    assert resumed.continuation_state["active_seconds_used"] == 13
    monkeypatch.setattr(investigation, "execute_pipeline_request", execute)
    outcome = act(
        resumed,
        "trial_edit",
        candidate_id="candidate1",
        edits=[
            {
                "kind": "set_parameters",
                "node": "mask",
                "params": {"threshold": 150},
            }
        ],
    ).public_result()
    assert outcome["status"] == "ok", outcome
    assert resumed.attempted_trials == 2
    np.testing.assert_array_equal(
        resumed.candidates[-1]._outputs[0][1].data, pipeline.outputs["input"] > 150
    )
    assert GraphSnapshot.from_pipeline(pipeline) == before_graph


def test_loading_without_new_sharing_is_history_only(tmp_path):
    engine = session(include_input_images=True, continuation_enabled=True)
    engine.request_pause()
    checkpoint = checkpoint_investigation(engine)
    path = tmp_path / "private.vipp-assistant"
    save_assistant_session(path, checkpoint)
    loaded = load_assistant_session(path)
    assert loaded.document["dialogue"]["objective"] == engine.dialogue.objective
    result = restore(
        loaded,
        engine._pipeline,
        include_input_images=False,
        include_output_images=False,
    )
    assert not result.resumable and result.session is None
    assert "approve" in result.reason and "Nothing was sent" in result.reason
    assert engine.model_calls_used == 0


@pytest.mark.parametrize("change", ["pixels", "grid", "graph", "compute"])
def test_fresh_source_graph_and_compute_binding_fail_closed(change):
    pipeline = source_pipeline()
    engine = session(pipeline, continuation_enabled=True)
    engine.request_pause()
    checkpoint = checkpoint_investigation(engine)
    kwargs = {}
    if change == "pixels":
        data = pipeline.outputs["input"].copy()
        data.flat[0] += 1
        data.flags.writeable = False
        pipeline.outputs["input"] = data
    elif change == "grid":
        state = pipeline.output_states["input"]
        pipeline.output_states["input"] = replace(
            state, axes=(replace(state.axes[0], scale=3), *state.axes[1:])
        )
    elif change == "graph":
        pipeline.add_node("binary_threshold")
    else:
        from napari_vipp.core.compute import ComputeMode

        kwargs["compute_request"] = ComputeRequest(mode=ComputeMode.CPU)
        if kwargs["compute_request"].as_dict() == engine.compute_request.as_dict():
            kwargs["compute_request"] = ComputeRequest(mode=ComputeMode.AUTO)
    result = restore(checkpoint, pipeline, **kwargs)
    assert result.session is None and not result.resumable
    assert engine.model_calls_used == 0


def test_busy_or_running_checkpoint_is_rejected_without_pausing_or_resetting():
    engine = session(continuation_enabled=True)
    with pytest.raises(AssistantCheckpointError, match="Pause"):
        checkpoint_investigation(engine)
    assert not engine.paused
    engine.request_pause()
    engine._active_action = True
    with pytest.raises(AssistantCheckpointError, match="Pause"):
        checkpoint_investigation(engine)
    assert engine.actions_used == engine.model_calls_used == 0


def test_warm_protected_mesh_roundtrip_has_no_live_adoption_and_preserves_middle(
    monkeypatch,
):
    pipeline, _snapshot, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle, continuation_enabled=True)
    engine.request_pause()
    checkpoint = checkpoint_investigation(engine)
    result = restore(checkpoint, pipeline)
    assert result.resumable, result.reason
    resumed = result.session
    assert (
        resumed.attempted_trials == 0 and resumed.retained_candidate_id == "candidate0"
    )
    assert resumed._accepted_bundle._live_snapshot is None
    assert (
        resumed._preservation.public_contract()
        == engine._preservation.public_contract()
    )
    np.testing.assert_array_equal(
        resumed._preservation._protected, engine._preservation._protected
    )
    for (_, old), (_, new) in zip(
        engine.candidates[0]._outputs, resumed.candidates[0]._outputs, strict=True
    ):
        assert old.fingerprint == new.fingerprint
        assert old.state_fingerprint == new.state_fingerprint
    resumed.resume()
    outcome = update(resumed, 90)
    assert outcome["status"] == "ok", outcome
    assert outcome["result"]["status"] == "completed"
    assert outcome["result"]["preservation_check"]["passed"]
    assert resumed.attempted_trials == 1


def test_complete_ledger_is_historical_until_new_exact_transmission():
    engine = session(continuation_enabled=True, require_experiment_plan=True)
    engine.record_model_call(request=engine.public_request())
    assert plan(engine).public_result()["status"] == "ok"
    assert linked_trial(engine, "experiment1").public_result()["status"] == "ok"
    engine.record_model_call(request=engine.public_request())
    assert (
        assessment(engine, "experiment1", "candidate1").public_result()["status"]
        == "ok"
    )
    engine.request_pause()
    result = restore(checkpoint_investigation(engine), engine._pipeline)
    assert result.resumable, result.reason
    resumed = result.session
    assert resumed.experiment_ledger == engine.experiment_ledger
    assert resumed._sent_numeric == set()
    resumed.resume()
    assert plan(resumed, baseline="candidate1").public_result()["status"] == "error"
    request = resumed.public_request()
    fresh_ref = next(
        row["evidence_ref"]
        for row in request["decision_context"]["numeric_evidence"]
        if row["alias"] == "candidate1/measurements/wanted"
    )
    resumed.record_model_call(request=request)
    created = plan(resumed, baseline="candidate1", refs=[fresh_ref]).public_result()
    assert created["status"] == "ok", created


def test_finish_nomination_reopens_for_fresh_review_not_automatic_add(monkeypatch):
    engine = session(continuation_enabled=True, review_before_finish=True)
    original = GraphSnapshot.from_pipeline(engine._pipeline)
    trial(engine)
    act(
        engine,
        "finish",
        candidate_id="candidate1",
        summary="A reviewable starting point.",
        limitations=["No scientific truth."],
    )
    checkpoint = checkpoint_investigation(engine)
    resumed = restore(checkpoint, engine._pipeline).session
    assert resumed.paused and not resumed.finished
    assert resumed.pending_finish_review["candidate_id"] == "candidate1"
    with pytest.raises(ValueError, match="Finish"):
        resumed.final_add_proposal()
    resumed.accept_finish_review()
    assert resumed.finished
    prepared = resumed.final_add_proposal()
    assert prepared.can_apply
    assert GraphSnapshot.from_pipeline(engine._pipeline) == original


def test_finished_checkpoint_requires_fresh_nomination_acceptance():
    engine = session(continuation_enabled=True)
    trial(engine)
    act(
        engine,
        "finish",
        candidate_id="candidate1",
        summary="Choose this candidate.",
        limitations=[],
    )
    assert engine.finished
    result = restore(checkpoint_investigation(engine), engine._pipeline)
    assert result.resumable, result.reason
    assert not result.session.finished
    assert result.session.pending_finish_review is not None


@pytest.mark.parametrize("target", ["output_hash", "notebook", "counters", "protected"])
def test_changed_cache_journal_and_accounting_are_rejected(target):
    if target == "protected":
        pipeline, _, bundle = accepted(protection=True)
        engine = engine_for(pipeline, bundle, continuation_enabled=True)
    else:
        engine = session(continuation_enabled=True)
        trial(engine)
        pipeline = engine._pipeline
    engine.request_pause()
    checkpoint = checkpoint_investigation(engine)

    def mutate(d):
        if target == "output_hash":
            d["scientific"]["candidates"][0]["outputs"][0]["binding"]["sha256"] = (
                "0" * 64
            )
        elif target == "notebook":
            d["notebook"]["records"][0]["payload"]["sha256"] = "0" * 64
        elif target == "counters":
            d["usage"]["attempted_trials"] += 1
        else:
            d["scientific"]["accepted"]["preservation"]["protected_sha256"] = "0" * 64

    with pytest.raises(AssistantCheckpointError):
        restore(altered(checkpoint, mutate), pipeline)


def test_budget_paused_usage_and_grants_are_not_replenished():
    now = [10.0]
    engine = session(
        continuation_enabled=True,
        clock=lambda: now[0],
        limits=InvestigationLimits(
            max_attempts=1, max_actions=10, max_model_calls=20, max_seconds=600
        ),
    )
    trial(engine)
    engine.record_model_call(request=engine.public_request())
    engine.request_pause()
    engine.grant_allowance(
        InvestigationLimits(
            max_attempts=2, max_actions=12, max_model_calls=24, max_seconds=700
        )
    )
    checkpoint = checkpoint_investigation(engine)
    now[0] = 100000
    resumed = restore(checkpoint, engine._pipeline, clock=lambda: now[0]).session
    assert resumed.limits == engine.limits
    assert resumed._allowance_grants == 1
    assert resumed.attempted_trials == 1
    assert resumed.actions_used == engine.actions_used
    assert resumed.model_calls_used == engine.model_calls_used
    assert resumed.deadline == now[0] + 700


def test_no_provider_settings_are_admitted_in_ui_state():
    engine = session(continuation_enabled=True)
    engine.request_pause()
    with pytest.raises(AssistantCheckpointError, match="connection settings"):
        checkpoint_investigation(
            engine, ui_state={"api_key": "no secret should be saved"}
        )
    checkpoint = checkpoint_investigation(engine)
    text = json.dumps(checkpoint.document)
    assert "_authority" not in text and "_live_snapshot" not in text
    assert "api_key" not in text and "bearer" not in text


def test_diagnostic_probe_outputs_survive_restart():
    engine = session(continuation_enabled=True)
    result = act(
        engine,
        "diagnostic_probe",
        input={"kind": "source"},
        roi={"axis_order": "ZYX", "bounds": [[0, 2], [0, 5], [0, 6]]},
        proposal=proposal(engine),
        output_roles=[{"role": "mask", "node": "mask", "port": 0}],
    ).public_result()
    assert result["status"] == "ok", result
    engine.request_pause()
    resumed = restore(checkpoint_investigation(engine), engine._pipeline).session
    assert resumed.attempted_probes == 1 and resumed.attempted_trials == 0
    assert resumed.diagnostic_probes == engine.diagnostic_probes
    np.testing.assert_array_equal(
        resumed._probes[0].outputs[0][1].data, engine._probes[0].outputs[0][1].data
    )


def test_ui_only_conversation_is_passive_and_has_no_scientific_authority(tmp_path):
    state = {
        "version": 1,
        "goal": "My original goal.",
        "chat": [["Assistant", "Which structures?"], ["You", "  both  "]],
        "replies": [
            {
                "index": 1,
                "question": "Which structures?",
                "answer": "  both  ",
                "is_question": True,
            }
        ],
        "clarifications": [["Which structures?", "  both  "]],
        "drafts": [
            {
                "summary": "A starting draft",
                "steps": [["Threshold", "Try this choice"]],
                "assumptions": ["A reviewable choice"],
                "questions": [],
                "proposal_text": "{}",
                "plan_text": "No result is certified.",
            }
        ],
        "draft_index": 0,
        "discussion_draft": "Please refine",
        "pause_answer_draft": "",
        "initiative": 2,
        "source_id": "input",
        "stage": 1,
    }
    checkpoint = checkpoint_investigation(None, ui_state=state)
    path = tmp_path / "conversation.vipp-session"
    save_assistant_session(path, checkpoint)
    loaded = load_assistant_session(path)
    assert loaded.document["ui_state"] == state
    assert loaded.blob_names == ()
    result = restore(loaded, source_pipeline())
    assert result.session is None and not result.resumable
    assert "no paused investigation" in result.reason


def test_plain_measurement_table_cells_and_metadata_roundtrip():
    engine = session(continuation_enabled=True)
    result = act(
        engine,
        "trial_candidate",
        proposal=proposal(engine, manual=True),
        output_roles=[{"role": "metrics", "node": "measure", "port": 0}],
    ).public_result()
    assert result["status"] == "ok", result
    engine.request_pause()
    checkpoint = checkpoint_investigation(engine)
    assert checkpoint.document["resume"]["supported"]
    resumed = restore(checkpoint, engine._pipeline).session
    before = engine.candidates[0]._outputs[0][1]
    after = resumed.candidates[0]._outputs[0][1]
    assert before.fingerprint == after.fingerprint
    assert before.state_fingerprint == after.state_fingerprint
    assert before.data.rows == after.data.rows
    assert before.data.columns == after.data.columns


def test_unsupported_scientific_codec_saves_explicit_history_only():
    engine = session(continuation_enabled=True)
    result = act(
        engine,
        "trial_candidate",
        proposal=proposal(engine, manual=True),
        output_roles=[{"role": "metrics", "node": "measure", "port": 0}],
    ).public_result()
    assert result["status"] == "ok", result
    candidate = engine.candidates[0]
    role, binding = candidate._outputs[0]
    unsupported = replace(binding.data, histogram_metadata=object())
    engine._candidates[0] = replace(
        candidate, _outputs=((role, replace(binding, data=unsupported)),)
    )
    engine.request_pause()
    checkpoint = checkpoint_investigation(engine)
    assert checkpoint.document["resume"]["supported"] is False
    assert "Specialised table" in checkpoint.document["resume"]["reason"]
    assert checkpoint.document["scientific"] is None
    assert checkpoint.document["notebook"] == engine.export_notebook()
    assert checkpoint.blob_names == ()
    assert restore(checkpoint, engine._pipeline).session is None


def test_permission_revocation_cannot_be_smuggled_into_a_new_runtime():
    engine = session(include_input_images=True, continuation_enabled=True)
    trial(engine)
    engine.request_pause()
    engine.include_input_images = False
    engine.include_output_images = False
    checkpoint = checkpoint_investigation(engine)
    assert checkpoint.document["resume"]["supported"] is False
    assert "sharing choices changed" in checkpoint.document["resume"]["reason"]
    result = restore(checkpoint, engine._pipeline, include_output_images=False)
    assert result.session is None


@pytest.mark.parametrize(
    "target", ["graph_parameter", "graph_wire", "action_counter", "candidate_record"]
)
def test_graph_recipe_and_journal_crossbindings_are_not_silently_repaired(target):
    engine = session(continuation_enabled=True)
    trial(engine)
    engine.request_pause()
    checkpoint = checkpoint_investigation(engine)

    def mutate(d):
        row = d["scientific"]["candidates"][0]
        if target == "graph_parameter":
            next(n for n in row["graph"]["nodes"] if n["id"] != "input")["params"][
                "threshold"
            ] = 101
        elif target == "graph_wire":
            row["graph"]["connections"] = []
        elif target == "action_counter":
            d["usage"]["actions_used"] += 1
        else:
            d["runtime"]["record_refs"]["candidates"]["candidate1"] = d["runtime"][
                "record_refs"
            ]["latest_action"]

    with pytest.raises((AssistantCheckpointError, ValueError)):
        restore(altered(checkpoint, mutate), engine._pipeline)


def test_new_goal_draft_is_history_not_old_goal_continuation():
    engine = session(continuation_enabled=True)
    engine.request_pause()
    checkpoint = checkpoint_investigation(engine, ui_state={"goal": "A changed goal."})
    assert checkpoint.document["resume"]["supported"] is False
    assert checkpoint.document["dialogue"]["objective"] == engine.dialogue.objective
    assert restore(checkpoint, engine._pipeline).session is None


@pytest.mark.parametrize(
    "target", ["missing_contract", "extra_node", "changed_nomination"]
)
def test_incomplete_contracts_hidden_nodes_and_unrecorded_finish_fail_closed(target):
    engine = session(continuation_enabled=True, review_before_finish=True)
    trial(engine)
    act(
        engine,
        "finish",
        candidate_id="candidate1",
        summary="Review this candidate.",
        limitations=[],
    )
    checkpoint = checkpoint_investigation(engine)

    def mutate(d):
        if target == "missing_contract":
            d["scientific"]["operation_contracts"].clear()
        elif target == "extra_node":
            node = d["scientific"]["candidates"][0]["graph"]["nodes"][-1].copy()
            node["id"] = "unmapped-extra-node"
            d["scientific"]["candidates"][0]["graph"]["nodes"].append(node)
        else:
            d["runtime"]["pending_finish_review"]["model_interpretation"] = (
                "Unrecorded different nomination."
            )

    with pytest.raises(ValueError):
        restore(altered(checkpoint, mutate), engine._pipeline)
