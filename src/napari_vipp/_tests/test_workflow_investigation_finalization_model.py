"""Offline provider contract only; no candidate choice or scientific oracle."""

from __future__ import annotations

import time

import pytest

from napari_vipp._tests.test_workflow_investigation_experiment_model import request
from napari_vipp._tests.test_workflow_investigation_model import (
    action_text,
    fake_provider,
)
from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local


def finalization_request():
    public = request()
    actions = public["response_schema"]["actions"]
    public["response_schema"]["actions"] = {
        name: actions[name] for name in ("finish", "question")
    }
    return public


def test_finalization_uses_one_native_turn_and_preserves_original_request(monkeypatch):
    calls = fake_provider(monkeypatch, [action_text()])
    public = finalization_request()
    reservations = []
    deadline = time.monotonic() + 20
    result = model.generate_codex_finalization_action(
        public,
        codex.CodexSettings(max_attempts=2),
        deadline=deadline,
        before_call=lambda: reservations.append(True),
    )
    assert result.action.action == "question"
    assert len(calls) == len(result.attempts) == len(reservations) == 1
    sent, options = calls[0]
    assert sent[0] == public and sent[1] == ()
    assert sent[2].max_attempts == 1
    assert options["absolute_deadline"] == deadline
    assert "FINALIZATION-ONLY" in options["investigation_contract"][0]
    assert "No candidate is preferred" in options["investigation_contract"][0]
    assert "does not reset any recorded budget" in options["investigation_contract"][0]


@pytest.mark.parametrize(
    "reply",
    ["{}", action_text(action="find_nodes", arguments={"query": "*"})],
)
def test_finalization_rejects_invalid_or_other_actions_without_repair(
    monkeypatch, reply
):
    calls = fake_provider(monkeypatch, [reply])
    with pytest.raises(
        local.ModelProposalError, match="did not pass validation"
    ) as caught:
        model.generate_codex_finalization_action(
            finalization_request(),
            codex.CodexSettings(max_attempts=2),
            deadline=time.monotonic() + 20,
            before_call=lambda: None,
        )
    assert len(calls) == len(caught.value.attempts) == 1
    assert not caught.value.attempts[0].valid


@pytest.mark.parametrize("failure", ["schema", "images", "deadline", "gate", "elapsed"])
def test_finalization_invalid_scope_fails_before_discovery(monkeypatch, failure):
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("Unexpected discovery")
    )
    public = finalization_request()
    deadline, gate = time.monotonic() + 20, lambda: None
    if failure == "schema":
        public = request()
    elif failure == "images":
        public["image_previews"] = [{"sha256": "a" * 64}]
    elif failure == "deadline":
        deadline = float("nan")
    elif failure == "gate":
        gate = None
    elif failure == "elapsed":
        deadline = time.monotonic() - 1
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_finalization_action(
            public, codex.CodexSettings(), deadline=deadline, before_call=gate
        )


def test_finalization_budget_gate_blocks_the_only_actual_turn(monkeypatch):
    calls = fake_provider(monkeypatch, [])

    def exhausted():
        raise ValueError("The original model budget was reached.")

    with pytest.raises(ValueError, match="original model budget"):
        model.generate_codex_finalization_action(
            finalization_request(),
            codex.CodexSettings(),
            deadline=time.monotonic() + 20,
            before_call=exhausted,
        )
    assert calls == []
