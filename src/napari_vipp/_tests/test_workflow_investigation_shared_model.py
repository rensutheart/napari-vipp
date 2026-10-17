"""Lossless shared JSON values reach the restricted provider without tool changes."""

from copy import deepcopy

import pytest

from napari_vipp._tests.test_workflow_investigation_model import (
    action_text,
    fake_provider,
    public_request,
)
from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_investigation_history import (
    compact_history,
    compact_request_values,
    expand_request_history,
)


def shared_request():
    history = [
        {
            "action": "inspect_candidate",
            "status": "ok",
            "result": {
                "meaning": "Synthetic read-only contract evidence. " * 80,
                "values": list(range(150)),
            },
        }
        for _ in range(16)
    ]
    request = public_request()
    request["experiment_ledger"] = []
    request.update(compact_history(history, []))
    return compact_request_values(request), history


def test_shared_values_provider_round_trip_is_exact(monkeypatch):
    request, history = shared_request()
    calls = fake_provider(monkeypatch, [action_text()])
    result = model.generate_codex_investigation_action(request, codex.CodexSettings())
    assert result.action.action == "question"
    sent = calls[0][0][0]
    assert sent == request
    assert expand_request_history(sent) == history
    assert sent["image_previews"] == []
    assert "version 3" in calls[0][1]["investigation_contract"][0]


def test_tampered_shared_value_is_rejected_before_provider(monkeypatch):
    request, _history = shared_request()
    altered = deepcopy(request)
    first = next(iter(altered["history_shared_values"]))
    altered["history_shared_values"][first]["sha256"] = "0" * 64
    calls = fake_provider(monkeypatch, [action_text()])
    with pytest.raises(local.ModelProposalError, match="references are invalid"):
        model.generate_codex_investigation_action(altered, codex.CodexSettings())
    assert calls == []
