"""Synthetic recorded-finalization gates; no files, providers, Qt or calculations."""

from copy import deepcopy

import pytest

from napari_vipp.core.workflow_investigation_finalization import RecordedFinalization


def scope(**changes):
    request = {
        "session_id": "s" * 32,
        "image_previews": [],
        "response_schema": {"actions": {"finish": {}, "question": {}}},
        "budgets": {"actions_remaining": 1, "model_calls_remaining": 9},
        "image_permissions": {"output_images": True},
        "experiment_ledger": [
            {"candidate_id": "candidate1", "state": "assessed"},
            {"candidate_id": "candidate2", "state": "assessed"},
        ],
    }
    candidates = {
        identity: {
            "evidence": {"status": "completed"},
            "recipe": {"identity": identity},
        }
        for identity in ("candidate1", "candidate2")
    }
    options = dict(
        request=request,
        candidates=candidates,
        actions_used=31,
        model_calls_used=31,
        limits={"max_actions": 32, "max_model_calls": 40},
        deadline_utc=200.0,
        assert_current=lambda: None,
        clock=lambda: 100.0,
    )
    options.update(changes)
    return RecordedFinalization(**options)


def action(kind="finish", **arguments):
    return {
        "kind": "napari-vipp-investigation-action",
        "version": 1,
        "session_id": "s" * 32,
        "action": kind,
        "arguments": arguments
        or {
            "candidate_id": "candidate1",
            "summary": "Model nomination only.",
            "limitations": ["Not scientific validation."],
        },
    }


def test_explicit_nomination_not_latest_and_detached_authority():
    boundary = scope()
    detached = boundary.candidates
    detached["candidate1"]["recipe"] = {"changed": True}
    boundary.request["experiment_ledger"].clear()
    boundary.reserve_model_call()
    result = boundary.accept(action())
    assert result["result"]["retained_candidate_id"] == "candidate1"
    assert result["result"]["recipe"] == {"identity": "candidate1"}
    assert boundary.model_calls_used == 32 and boundary.actions_used == 32
    with pytest.raises(ValueError, match="one final action"):
        boundary.accept(action())


def test_one_native_turn_even_when_more_original_model_budget_remains():
    boundary = scope()
    boundary.reserve_model_call()
    with pytest.raises(ValueError, match="one actual"):
        boundary.reserve_model_call()


def test_disallowed_action_consumes_the_one_remaining_attempt():
    boundary = scope()
    boundary.reserve_model_call()
    with pytest.raises(ValueError, match="only FINISH or question"):
        boundary.accept(action("find_nodes", query="*"))
    assert boundary.actions_used == 32
    with pytest.raises(ValueError, match="one final action"):
        boundary.accept(action())


def test_every_completed_candidate_requires_assessment_not_just_nominated_one():
    boundary = scope()
    request = boundary.request
    request["experiment_ledger"] = request["experiment_ledger"][:1]
    boundary = scope(request=request)
    boundary.reserve_model_call()
    with pytest.raises(ValueError, match="recorded assessment"):
        boundary.accept(action())


def test_question_does_not_invent_a_selection():
    boundary = scope()
    boundary.reserve_model_call()
    result = boundary.accept(action("question", question="A genuine blocker?"))
    assert "retained_candidate_id" not in result["result"]
    assert boundary.actions_used == 32


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), 100.0, True])
def test_original_deadline_never_resets_or_accepts_nonfinite_values(deadline):
    with pytest.raises(ValueError):
        scope(deadline_utc=deadline)


def test_expiry_or_changed_authority_after_reservation_rejects_finish():
    now = [100.0]
    boundary = scope(clock=lambda: now[0])
    boundary.reserve_model_call()
    now[0] = 200.0
    with pytest.raises(ValueError, match="deadline expired"):
        boundary.accept(action())


def test_wrong_session_and_unknown_candidate_fail_closed():
    for edit in ("session", "candidate"):
        boundary = scope()
        boundary.reserve_model_call()
        candidate = deepcopy(action())
        if edit == "session":
            candidate["session_id"] = "foreign"
        else:
            candidate["arguments"]["candidate_id"] = "missing"
        with pytest.raises(ValueError):
            boundary.accept(candidate)
