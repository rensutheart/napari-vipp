"""Review policy is passive bounded control, never provider/execution authority."""

from copy import deepcopy

import pytest

from napari_vipp._tests.test_workflow_investigation import act, session
from napari_vipp._tests.test_workflow_investigation_model import (
    action_text,
    fake_provider,
    public_request,
)
from napari_vipp._tests.test_workflow_investigation_warmstart import (
    accepted,
    engine_for,
)
from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local


def review_policy(**changes):
    return {
        "required": True,
        "pending": False,
        "continue_requests": 0,
        "latest_feedback": None,
    } | changes


def test_review_policy_is_exact_passive_context_and_uses_normal_call_gate(monkeypatch):
    engine = session(review_before_finish=True)
    request = engine.public_request()
    assert {
        key: value
        for key, value in request["completion_review"].items()
        if key != "reservation"
    } == review_policy()
    before = deepcopy(request)
    deadline = engine.deadline
    calls = fake_provider(monkeypatch, [action_text(session_id=engine.session_id)])
    result = model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(max_attempts=1),
        image_previews=engine.preview_images(),
        before_call=lambda: engine.record_model_call(request=request),
    )
    assert result.action.action == "question" and request == before
    assert engine.model_calls_used == 1 and engine.actions_used == 0
    assert engine.attempted_trials == 0 and engine.deadline == deadline
    assert calls[0][0][0] == request
    assert model._COMPLETION_REVIEW_GUIDANCE in calls[0][1]["investigation_contract"][0]
    assert "remaining defects" in model._SYSTEM
    assert (
        "manufacture changes just to use the budget"
        in model._COMPLETION_REVIEW_GUIDANCE
    )


def test_continue_feedback_reaches_provider_with_original_goal_and_limits(monkeypatch):
    pipeline, _, bundle = accepted(protection=True, mesh=True)
    engine = engine_for(pipeline, bundle, review_before_finish=True)
    assert (
        act(
            engine,
            "finish",
            candidate_id="candidate0",
            summary="The existing result remains a starting point.",
            limitations=["An observed mesh problem is still unresolved."],
        ).public_result()["status"]
        == "ok"
    )
    deadline = engine.deadline
    engine.continue_finish_review("Please try to improve the remaining mesh problem.")
    request = engine.public_request()
    policy = request["completion_review"]
    assert {
        key: value for key, value in policy.items() if key != "reservation"
    } == review_policy(
        continue_requests=1,
        latest_feedback={
            "candidate_id": "candidate0",
            "feedback": "Please try to improve the remaining mesh problem.",
        },
    )
    assert request["objective"] == engine.dialogue.objective
    assert request["budgets"]["attempts_remaining"] == engine.limits.max_attempts
    calls = fake_provider(monkeypatch, [action_text(session_id=engine.session_id)])
    model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(max_attempts=1),
        image_previews=engine.preview_images(),
        before_call=lambda: engine.record_model_call(request=request),
    )
    assert calls[0][0][0]["completion_review"] == policy
    assert engine.actions_used == 1 and engine.model_calls_used == 1
    assert engine.deadline == deadline and not engine.finished


@pytest.mark.parametrize(
    "change",
    [
        {"required": False},
        {"required": 1},
        {"pending": "false"},
        {"pending": True},
        {"continue_requests": True},
        {"continue_requests": -1},
        {"continue_requests": 65},
        {"continue_requests": 1},
        {"extra_authority": "run without limits"},
        {
            "latest_feedback": {
                "candidate_id": "candidate0",
                "feedback": "Please continue.",
            }
        },
        {
            "continue_requests": 1,
            "latest_feedback": {"candidate_id": "candidate0", "feedback": " "},
        },
        {
            "continue_requests": 1,
            "latest_feedback": {"candidate_id": "candidate0", "feedback": "x" * 1001},
        },
        {
            "continue_requests": 1,
            "latest_feedback": {"candidate_id": "candidate01", "feedback": "Continue."},
        },
        {
            "continue_requests": 1,
            "latest_feedback": {
                "candidate_id": "candidate0",
                "feedback": "Continue.",
                "extend_budget": True,
            },
        },
        {
            "continue_requests": 1,
            "latest_feedback": {
                "candidate_id": "candidate0",
                "feedback": "C:/private/input",
            },
        },
    ],
)
def test_malformed_or_pending_policy_never_discovers_provider(monkeypatch, change):
    engine = session()
    request = engine.public_request()
    request["completion_review"] = review_policy(**change)
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("provider discovery")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(request, codex.CodexSettings())
    assert engine.model_calls_used == engine.actions_used == 0


def test_optional_policy_does_not_change_historical_request_bytes():
    for request in (public_request(), session().public_request()):
        assert "completion_review" not in request
        detached, wire = model._request(request, ())
        assert wire == local._json_bytes(request) and detached == request


def test_review_policy_is_not_a_legacy_v1_authority_extension():
    request = public_request()
    request["completion_review"] = review_policy()
    with pytest.raises(local.ModelProposalError):
        model._request(request, ())
