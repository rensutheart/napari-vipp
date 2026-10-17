"""Bounded automatic working context crosses only the normal provider seam."""

from copy import deepcopy

import pytest
from test_workflow_investigation import act, session
from test_workflow_investigation_model import action_text, fake_provider

from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local


def test_automatic_working_context_is_sent_exactly_without_extra_tool_calls(
    monkeypatch,
):
    engine = session(include_input_images=True)
    act(engine, "inspect_source", percentiles=[10, 50, 90])
    request = engine.public_request()
    assert "decision_context" in request
    before = deepcopy(request)
    calls = fake_provider(monkeypatch, [action_text(session_id=engine.session_id)])
    result = model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(max_attempts=1),
        image_previews=engine.preview_images(),
        before_call=lambda: engine.record_model_call(request=request),
    )
    assert result.action.session_id == engine.session_id
    assert request == before and calls[0][0][0] == before
    assert engine.model_calls_used == 1 and engine.actions_used == 1
    instructions = calls[0][1]["investigation_contract"][0]
    assert "Use these inline records first" in instructions
    assert "working_records and any supplied decision_context" in instructions
    assert "disconnected comparison branch" in instructions
    assert "Clone the exact tested parameters and wiring" in instructions
    assert "skipped validation" in instructions
    assert "not a requirement" in instructions
    assert "anatomical accuracy" in instructions


@pytest.mark.parametrize(
    "mutation",
    [
        "extra",
        "session",
        "version",
        "recipes",
        "contracts",
        "numeric",
        "omitted",
        "path",
    ],
)
def test_malformed_working_context_stops_before_provider_discovery(
    monkeypatch, mutation
):
    engine = session()
    request = engine.public_request()
    context = request["decision_context"]
    if mutation == "extra":
        context["execute_shell"] = True
    elif mutation == "session":
        context["session_id"] = "another-session"
    elif mutation == "version":
        context["version"] = True
    elif mutation == "recipes":
        context["recipes"] = [{}] * 3
    elif mutation == "contracts":
        context["node_contracts"] = [{}] * 33
    elif mutation == "numeric":
        context["numeric_evidence"] = [{"alias": "invented"}]
    elif mutation == "omitted":
        context["omitted"] = [{}] * 65
    else:
        context["omitted"] = [
            {"kind": "recipe", "identity": "candidate0", "reason": "C:/private/recipe"}
        ]
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("provider discovery")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(request, codex.CodexSettings())
    assert engine.model_calls_used == 0


def test_historical_v2_without_automatic_context_remains_accepted():
    engine = session()
    request = engine.public_request()
    del request["decision_context"]
    assert model._request(request, ())[0] == request


def test_actual_send_cannot_substitute_forged_automatic_context():
    from napari_vipp.core.workflow_investigation import InvestigationError

    engine = session()
    request = engine.public_request()
    request["decision_context"]["omitted"].append(
        {"kind": "recipe", "identity": "invented", "reason": "not supplied"}
    )
    with pytest.raises(InvestigationError, match="outgoing request"):
        engine.record_model_call(request=request)
    assert engine.model_calls_used == 0 and engine._sent_numeric == set()
