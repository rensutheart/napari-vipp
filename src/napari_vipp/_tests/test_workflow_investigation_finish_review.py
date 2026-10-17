"""Explicit local completion review never extends a scientific session's limits."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_investigation import act
from napari_vipp._tests.test_workflow_investigation_experiment_session import (
    assessment,
    plan,
)
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
    update,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationSession,
    InvestigationStopped,
)


def review_engine(**kwargs):
    pipeline, snapshot, bundle = accepted(protection=True, mesh=True)
    engine = InvestigationSession.from_completed(
        pipeline,
        bundle,
        "Preserve accepted results; review remaining requirements.",
        compute_request=ComputeRequest(),
        include_output_images=True,
        review_before_finish=True,
        assert_current=kwargs.pop("assert_current", lambda: None),
        **kwargs,
    )
    return pipeline, snapshot, bundle, engine


def nominate(engine, candidate_id="candidate0"):
    result = act(
        engine,
        "finish",
        candidate_id=candidate_id,
        summary="A provisional synthetic preference, not independently validated.",
        limitations=["Known remaining goal requirements may still merit a test."],
    ).public_result()
    assert result["status"] == "ok", result
    return result["result"]


def decisions(engine):
    exported = engine.export_notebook()
    kinds = {row["record_id"]: row["kind"] for row in exported["record_index"]}
    return [
        row["payload"]
        for row in exported["records"]
        if kinds[row["record_id"]] == "user_finish_decision"
    ]


def test_provisional_finish_is_detached_unaccepted_and_blocks_all_model_work():
    pipeline, snapshot, _, engine = review_engine()
    original = pipeline.outputs["input"].copy()
    request = engine.public_request()
    assert {
        key: value
        for key, value in request["completion_review"].items()
        if key != "reservation"
    } == {
        "required": True,
        "pending": False,
        "continue_requests": 0,
        "latest_feedback": None,
    }
    nomination = nominate(engine)
    assert engine.finished is False and engine.retained_candidate_id == "candidate0"
    pending = engine.pending_finish_review
    assert pending["candidate_id"] == "candidate0" and pending["continuation_allowed"]
    pending["limitations"].append("An external mutation.")
    assert "An external mutation." not in engine.pending_finish_review["limitations"]
    assert engine.public_request()["completion_review"]["pending"] is True
    before = (engine.actions_used, engine.model_calls_used, engine._image_bytes)
    notebook = engine.export_notebook()
    with pytest.raises(InvestigationError, match="awaiting.*decision"):
        engine.record_model_call(request=request)
    with pytest.raises(InvestigationError, match="awaiting.*decision"):
        act(engine, "find_nodes", query="threshold")
    assert before == (engine.actions_used, engine.model_calls_used, engine._image_bytes)
    assert engine.export_notebook() == notebook
    with pytest.raises(InvestigationError, match="Finish a completed"):
        engine.final_add_proposal()
    result = engine.accept_finish_review()
    assert result == nomination and engine.finished
    assert decisions(engine)[-1]["decision"] == "accept"
    assert engine.pending_finish_review is None
    np.testing.assert_array_equal(pipeline.outputs["input"], original)
    snapshot.assert_current(pipeline)


def test_continue_preserves_session_limits_cache_protection_and_invalidates_old_send(
    monkeypatch,
):
    import napari_vipp.core.workflow_investigation as module

    _, _, _, engine = review_engine()
    request = engine.public_request()
    engine.record_model_call(request=request)
    nominate(engine)
    before = (
        engine.session_id,
        engine.deadline,
        engine.limits,
        engine.actions_used,
        engine.model_calls_used,
        engine.attempted_trials,
        engine._image_bytes,
        engine.candidates,
        engine._preservation,
        engine._accepted_bundle,
        engine.preview_images(),
        engine.experiment_ledger,
    )
    history = tuple(engine._history)
    records = engine.export_notebook()["records"]
    monkeypatch.setattr(engine, "_render_views", lambda *a, **k: pytest.fail("render"))
    monkeypatch.setattr(
        engine, "_render_mesh_views", lambda *a, **k: pytest.fail("render")
    )
    monkeypatch.setattr(
        module, "execute_pipeline_request", lambda *a, **k: pytest.fail("Run")
    )
    receipt = engine.continue_finish_review()
    assert receipt["decision"] == "continue" and receipt["feedback"].strip()
    assert engine.pending_finish_review is None and not engine.finished
    assert before == (
        engine.session_id,
        engine.deadline,
        engine.limits,
        engine.actions_used,
        engine.model_calls_used,
        engine.attempted_trials,
        engine._image_bytes,
        engine.candidates,
        engine._preservation,
        engine._accepted_bundle,
        engine.preview_images(),
        engine.experiment_ledger,
    )
    assert tuple(engine._history) == history
    assert engine.export_notebook()["records"][: len(records)] == records
    current = engine.public_request()
    assert current["completion_review"]["continue_requests"] == 1
    assert current["completion_review"]["latest_feedback"] == {
        "candidate_id": "candidate0",
        "feedback": receipt["feedback"],
    }
    with pytest.raises(InvestigationError, match="outgoing request"):
        engine.record_model_call(request=request)
    assert engine.model_calls_used == before[4]
    engine.record_model_call(request=current)
    assert engine.model_calls_used == before[4] + 1


def test_continue_allows_a_genuine_second_assessed_trial_without_resetting_baseline():
    pipeline, snapshot, bundle, engine = review_engine(require_experiment_plan=True)

    def supplied(alias):
        request = engine.public_request()
        engine.record_model_call(request=request)
        return next(
            row["evidence_ref"]
            for row in request["decision_context"]["numeric_evidence"]
            if row["alias"] == alias
        )

    def test_trial(parent, threshold, reference):
        planned = plan(engine, baseline=parent, refs=[reference]).public_result()
        assert planned["status"] == "ok", planned
        experiment_id = planned["result"]["experiment_id"]
        trial = act(
            engine,
            "trial_edit",
            candidate_id=parent,
            experiment_id=experiment_id,
            edits=[
                {
                    "kind": "set_parameters",
                    "node": bundle.output_roles[0]["node"],
                    "params": {"threshold": threshold},
                }
            ],
        ).public_result()
        assert (
            trial["status"] == "ok" and trial["result"]["preservation_check"]["passed"]
        )
        return experiment_id, trial["result"]["candidate_id"]

    baseline = supplied("candidate0/measurements/band_mesh")
    first_id, first = test_trial("candidate0", 90, baseline)
    rejected = act(
        engine, "finish", candidate_id=first, summary="Premature", limitations=[]
    ).public_result()
    assert rejected["status"] == "error" and engine.pending_finish_review is None
    first_ref = supplied(f"{first}/preservation")
    assert (
        assessment(engine, first_id, first, refs=[first_ref]).public_result()["status"]
        == "ok"
    )
    nominate(engine, first)
    original_limits, deadline = engine.limits, engine.deadline
    first_ledger = deepcopy(engine.experiment_ledger)
    engine.continue_finish_review("Consider the remaining goal requirements.")
    assert engine.experiment_ledger == first_ledger
    first_ref = supplied(f"{first}/preservation")
    second_id, second = test_trial(first, 80, first_ref)
    second_ref = supplied(f"{second}/preservation")
    assert (
        assessment(engine, second_id, second, refs=[second_ref]).public_result()[
            "status"
        ]
        == "ok"
    )
    result = nominate(engine, second)
    assert engine.retained_candidate_id == "candidate0"  # Still only provisional.
    assert engine.accept_finish_review() == result
    assert engine.selected_candidate.candidate_id == second
    assert engine.attempted_trials == 2 and engine.limits == original_limits
    assert engine.deadline == deadline
    assert all(row["state"] == "assessed" for row in engine.experiment_ledger)
    assert (
        engine.selected_candidate.evidence["preservation_check"]["missing_count"] == 0
    )
    snapshot.assert_current(pipeline)


@pytest.mark.parametrize("feedback", [True, "x" * 1001, "C:/private/source.npy"])
def test_invalid_user_feedback_cannot_change_the_pending_checkpoint(feedback):
    _, _, _, engine = review_engine()
    nominate(engine)
    before = engine.export_notebook()
    with pytest.raises(InvestigationError):
        engine.continue_finish_review(feedback)
    assert engine.export_notebook() == before
    assert engine.pending_finish_review["candidate_id"] == "candidate0"


@pytest.mark.parametrize(
    "budget", ["actions", "calls", "time", "trials", "images", "context"]
)
def test_depleted_original_limits_disable_continue_but_keep_user_acceptance(
    budget, monkeypatch
):
    import napari_vipp.core.workflow_investigation as module

    clock = [100.0]
    _, _, _, engine = review_engine(clock=lambda: clock[0])
    if budget == "trials":
        engine.limits = replace(engine.limits, max_attempts=1)
        assert update(engine, 80)["status"] == "ok"
    result = nominate(engine)
    if budget == "actions":
        engine.actions_used = engine.limits.max_actions - 3
    elif budget == "calls":
        engine.model_calls_used = engine.limits.max_model_calls - 7
    elif budget == "time":
        clock[0] = engine.deadline + 1
    elif budget == "images":
        engine._image_bytes = engine.limits.max_image_bytes
    elif budget == "context":
        monkeypatch.setattr(module, "WORKING_REQUEST_TARGET_BYTES", 1)
    assert engine.pending_finish_review["continuation_allowed"] is False
    before = engine.export_notebook()
    counters = (engine.actions_used, engine.model_calls_used, engine.deadline)
    with pytest.raises(InvestigationError):
        engine.continue_finish_review("Keep improving.")
    assert engine.export_notebook() == before and not engine.stop_reason
    assert counters == (engine.actions_used, engine.model_calls_used, engine.deadline)
    engine.assert_finish_review_current()
    assert engine.accept_finish_review() == result


@pytest.mark.parametrize(
    "change", ["graph", "source", "permissions", "cancelled", "authority", "candidate"]
)
def test_stale_cancelled_or_changed_completed_evidence_cannot_be_accepted(change):
    cancelled, authority = [False], [False]

    def assert_current():
        if authority[0]:
            raise ValueError("Synthetic ownership changed.")

    pipeline, _, _, engine = review_engine(
        cancelled=lambda: cancelled[0], assert_current=assert_current
    )
    assert update(engine, 80)["status"] == "ok"
    nominate(engine, "candidate1")
    if change == "graph":
        pipeline.add_node("invert")
    elif change == "source":
        altered = np.array(pipeline.outputs["input"], copy=True)
        altered.flat[0] += 1
        altered.setflags(write=False)
        pipeline.outputs["input"] = altered
        pipeline.node_outputs["input"] = (altered,)
    elif change == "permissions":
        engine.include_output_images = False
    elif change == "cancelled":
        cancelled[0] = True
    elif change == "authority":
        authority[0] = True
    else:
        binding = engine.candidates[-1]._outputs[0][1]
        object.__setattr__(binding, "fingerprint", "0" * 64)
    with pytest.raises(InvestigationStopped):
        engine.accept_finish_review()
    assert not engine.finished and engine.retained_candidate_id == "candidate0"
    assert decisions(engine) == []


def test_stop_is_a_recorded_cancellation_not_acceptance_and_cannot_reopen():
    _, _, _, engine = review_engine()
    nominate(engine)
    engine.cancel(user_decision=True)
    assert engine.pending_finish_review is None and not engine.finished
    assert decisions(engine)[-1]["decision"] == "stop"
    assert engine.retained_candidate_id == "candidate0"
    for method in (engine.continue_finish_review, engine.accept_finish_review):
        with pytest.raises(InvestigationError, match="No provisional Finish"):
            method()
    with pytest.raises(InvestigationStopped):
        engine.public_request()


def test_legacy_headless_finish_and_closed_sessions_are_unchanged():
    pipeline, _, bundle = accepted()
    engine = engine_for(pipeline, bundle)
    assert engine.review_before_finish is False
    assert "completion_review" not in engine.public_request()
    result = nominate(engine)
    assert engine.finished and result["retained_candidate_id"] == "candidate0"
    assert engine.pending_finish_review is None and decisions(engine) == []
    with pytest.raises(InvestigationError, match="No provisional Finish"):
        engine.continue_finish_review()
    with pytest.raises(InvestigationStopped):
        engine.public_request()


def test_automatic_revocation_is_not_recorded_as_an_explicit_user_stop():
    _, _, _, engine = review_engine()
    nominate(engine)
    engine.cancel()
    assert engine.pending_finish_review is None
    assert decisions(engine) == []
    assert engine.stop_reason and not engine.finished


def test_user_stop_still_cancels_when_the_finite_journal_cannot_record_it(monkeypatch):
    from napari_vipp.core.workflow_investigation_notebook import NotebookError

    _, _, _, engine = review_engine()
    nominate(engine)
    monkeypatch.setattr(
        engine,
        "_finish_decision",
        lambda *a, **k: (_ for _ in ()).throw(NotebookError("Full")),
    )
    engine.cancel(user_decision=True)
    assert engine.pending_finish_review is None and not engine.finished
    assert (
        "cancelled" in engine.stop_reason and "could not record" in engine.stop_reason
    )


@pytest.mark.parametrize("value", [1, None, "yes"])
def test_review_policy_requires_an_explicit_boolean(value):
    pipeline, _, bundle = accepted()
    with pytest.raises(InvestigationError):
        engine_for(pipeline, bundle, review_before_finish=value)


def test_no_new_model_work_can_be_granted_after_elapsed_pending_acceptance():
    clock = [0.0]
    _, _, _, engine = review_engine(clock=lambda: clock[0])
    nominate(engine)
    request = engine.public_request()
    clock[0] = engine.deadline + 1
    engine.assert_finish_review_current()
    published = engine.public_request()
    assert published["completion_review"]["pending"] is True
    assert published["budgets"]["seconds_remaining"] == 0
    assert published["closing_phase"]["active"] is True
    assert not engine.stop_reason
    with pytest.raises(InvestigationError, match="awaiting.*decision"):
        engine.record_model_call(request=request)
    engine.accept_finish_review()
    with pytest.raises(InvestigationStopped):
        engine.record_model_call(request=request)


def test_pending_status_caches_context_but_continue_checks_new_request(monkeypatch):
    clock = [0.0]
    _, _, _, engine = review_engine(clock=lambda: clock[0])
    nominate(engine)
    calls = []
    original = engine._public_request

    def publication(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(engine, "_public_request", publication)
    assert engine.pending_finish_review["continuation_allowed"]
    assert len(calls) == 1
    clock[0] += 20
    assert (
        engine.pending_finish_review["remaining_budgets"]["seconds_remaining"] == 1780
    )
    assert engine.pending_finish_review["continuation_allowed"]
    assert len(calls) == 1  # No repeated full context or PNG-manifest rebuilding.
    engine.continue_finish_review("Consider the remaining requirements.")
    assert len(calls) == 2  # Fresh prospective feedback/user-decision context.
    nominate(engine)
    before = len(calls)
    assert engine.pending_finish_review["continuation_allowed"]
    assert len(calls) == before + 1  # A new pause never reuses the old receipt.
