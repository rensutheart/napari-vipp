"""The restricted model transport sees the same canonical shared capabilities."""

import json
from copy import deepcopy

import pytest

from napari_vipp._tests.test_workflow_investigation import proposal, session
from napari_vipp._tests.test_workflow_investigation_model import fake_provider
from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_agent_tools import (
    WorkflowAgentTools,
    tool_definitions_for_action_schema,
)


def test_canonical_tools_and_probe_allocation_reach_actual_model_contract(monkeypatch):
    engine = session()
    request = engine.public_request()
    reply = json.dumps(
        dict(
            kind="napari-vipp-investigation-action",
            version=1,
            session_id=engine.session_id,
            action="question",
            arguments={"question": "Offline model-contract check only."},
        )
    )
    calls = fake_provider(monkeypatch, [reply])
    result = model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(),
        before_call=lambda: engine.record_model_call(request=request),
    )
    assert result.action.action == "question" and len(calls) == 1
    public, _ = calls[0][0][:2]
    system, developer, wrapper = calls[0][1]["investigation_contract"]
    assert public["diagnostic_probes"]["maximum_attempts"] == 6
    assert public["diagnostic_probes"]["attempts_remaining"] == 6
    assert "diagnostic_probe" in public["response_schema"]["actions"]
    assert "inspect_probe" in public["response_schema"]["actions"]
    definitions = tool_definitions_for_action_schema(public["response_schema"])
    assert json.dumps(definitions, ensure_ascii=False, allow_nan=False) in developer
    assert "vipp_diagnostic_probe" in developer and "vipp_inspect_probe" in developer
    assert "short ordinary-node" in system and "different" in system
    assert "entire accepted foreground outline" in system
    assert "bounded" in system and "No code, paths, plugins" in system
    assert wrapper["required"] == ["action_json"]
    assert engine.attempted_trials == engine.actions_used == 0
    assert engine.model_calls_used == 1


@pytest.mark.parametrize(
    "change",
    [
        {"maximum_attempts": 7},
        {"attempts_remaining": True},
        {"attempts_remaining": 5},
        {"max_input_voxels": 262145},
        {"max_nodes": 9},
        {"max_outputs": 5},
        {"index": [{"probe_id": "probe99", "status": "completed"}]},
    ],
)
def test_tampered_probe_limits_fail_before_provider(monkeypatch, change):
    engine = session()
    request = deepcopy(engine.public_request())
    request["diagnostic_probes"].update(change)
    calls = fake_provider(monkeypatch, [])
    with pytest.raises(local.ModelProposalError, match="diagnostic-probe"):
        model.generate_codex_investigation_action(request, codex.CodexSettings())
    assert calls == []


def test_local_connector_does_not_claim_unqualified_agent_or_vision_support():
    with pytest.raises(local.ModelProposalError, match="does not support"):
        model.generate_investigation_action({}, local.ModelSettings())


@pytest.mark.parametrize("change", ["valid", "sha256", "record_id", "probe_id"])
def test_probe_index_binds_exact_notebook_record_before_send(change):
    engine = session()
    result = WorkflowAgentTools(engine).call_tool(
        "vipp_diagnostic_probe",
        {
            "input": {"kind": "source"},
            "roi": {"axis_order": "ZYX", "bounds": [[0, 2], [0, 8], [0, 10]]},
            "proposal": proposal(engine, 100),
            "output_roles": [{"role": "diagnostic", "node": "mask", "port": 0}],
        },
    )
    assert result.public_result()["status"] == "ok"
    request = engine.public_request()
    row = request["diagnostic_probes"]["index"][0]
    if change == "valid":
        assert (
            model._request(request, ())[0]["diagnostic_probes"]
            == request["diagnostic_probes"]
        )
        return
    if change == "probe_id":
        # Another valid notebook record is not this probe's provenance.
        descriptor = next(
            item
            for item in request["evidence_notebook"]["record_index"]
            if item["record_id"] == row["full_record"]["record_id"]
        )
        descriptor["owner"]["probe_id"] = "probe2"
    elif change == "record_id":
        other = next(
            item
            for item in request["evidence_notebook"]["record_index"]
            if item["kind"] != "diagnostic_probe"
        )
        row["full_record"] = {key: other[key] for key in ("record_id", "sha256")}
    else:
        row["full_record"]["sha256"] = "0" * 64
    with pytest.raises(local.ModelProposalError):
        model._request(request, ())
