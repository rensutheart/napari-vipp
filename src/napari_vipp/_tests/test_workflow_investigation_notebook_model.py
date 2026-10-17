"""The bounded notebook wire is exact data, never a substitute evidence grant."""

from copy import deepcopy

import pytest
from test_workflow_investigation import act, session
from test_workflow_investigation_experiment_session import plan
from test_workflow_investigation_model import action_text, fake_provider

from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local


def test_real_notebook_request_uses_exact_original_goal_and_bounded_guidance(
    monkeypatch,
):
    engine = session(include_input_images=True)
    request = engine.public_request()
    original = deepcopy(request)
    journal = engine.export_notebook()
    calls = fake_provider(monkeypatch, [action_text(session_id=engine.session_id)])
    result = model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(max_attempts=1),
        image_previews=engine.preview_images(),
        before_call=lambda: engine.record_model_call(request=request),
    )
    assert result.action.session_id == engine.session_id
    assert engine.model_calls_used == 1
    assert engine.actions_used == engine.attempted_trials == 0
    assert request == original and engine.export_notebook() == journal
    sent, kwargs = calls[0]
    assert sent[0] == original
    assert sent[0]["objective"] == engine.dialogue.objective
    assert sent[0]["version"] == 2 and "history" not in sent[0]
    assert sent[1] == engine.preview_images()
    instructions = kwargs["investigation_contract"][0]
    assert "NOT complete inline history" in instructions
    assert "read_notebook" in instructions and "pin_previews" in instructions


@pytest.mark.parametrize(
    "mutation",
    [
        "legacy_history",
        "legacy_encoding",
        "missing_notebook",
        "wrong_session",
        "bool_version",
        "bool_notebook_version",
        "changed_working_record",
        "changed_index_hash",
        "changed_phase_actions",
        "unbounded_target",
        "bool_reserved",
        "phase_extra",
        "missing_phase",
        "bool_hard_cap",
    ],
)
def test_bad_v2_notebook_or_closing_contract_is_rejected_before_discovery(
    monkeypatch, mutation
):
    engine = session()
    act(engine, "find_nodes", query="threshold")
    request = engine.public_request()
    notebook = request["evidence_notebook"]
    phase = request["closing_phase"]
    if mutation == "legacy_history":
        request["history"] = []
    elif mutation == "legacy_encoding":
        request["history_encoding"] = {}
        request["history_preview_manifests"] = {}
    elif mutation == "missing_notebook":
        del request["evidence_notebook"]
    elif mutation == "wrong_session":
        notebook["session_id"] = "another-session"
    elif mutation == "bool_version":
        request["version"] = True
    elif mutation == "bool_notebook_version":
        notebook["version"] = True
    elif mutation == "changed_working_record":
        notebook["working_records"][0]["payload"]["status"] = "invented"
    elif mutation == "changed_index_hash":
        notebook["record_index"][0]["sha256"] = "0" * 64
    elif mutation == "changed_phase_actions":
        phase["allowed_actions"] = ["finish"]
    elif mutation == "unbounded_target":
        phase["request_target_bytes"] = 192 * 1024
    elif mutation == "bool_reserved":
        phase["reserved_actions"] = True
    elif mutation == "phase_extra":
        phase["ignore_guard"] = True
    elif mutation == "missing_phase":
        del request["closing_phase"]
    else:
        phase["hard_request_bytes"] = True
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("provider discovery")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(
            request, codex.CodexSettings(), image_previews=engine.preview_images()
        )
    assert engine.model_calls_used == 0


def test_bounded_format_repair_uses_same_exact_notebook_and_counts_both_calls(
    monkeypatch,
):
    engine = session(include_input_images=True)
    request = engine.public_request()
    calls = fake_provider(
        monkeypatch, ["{}", action_text(session_id=engine.session_id)]
    )
    result = model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(max_attempts=2),
        image_previews=engine.preview_images(),
        before_call=lambda: engine.record_model_call(request=request),
    )
    assert len(result.attempts) == engine.model_calls_used == len(calls) == 2
    assert calls[0][0][0] == calls[1][0][0] == request
    assert calls[0][0][1] == calls[1][0][1] == engine.preview_images()
    assert engine.actions_used == 0


def test_ui_labels_explain_notebook_actions_without_exposing_json():
    from napari_vipp.ui.workflow_assistant_investigation import _ACTION_NAMES

    assert (
        set(("read_notebook", "pin_notebook", "pin_previews")) <= _ACTION_NAMES.keys()
    )
    assert all(
        "JSON" not in _ACTION_NAMES[key]
        for key in ("read_notebook", "pin_notebook", "pin_previews")
    )


def test_readable_authenticated_page_survives_unchanged_transport_path_guard(
    monkeypatch,
):
    engine = session(include_input_images=True)
    descriptor = engine._notebook.append(
        "action_receipt",
        {"help": "A literal \\n or regex \\d is text, not a file path. ~~ μ"},
    )
    result = act(
        engine,
        "read_notebook",
        record_id=descriptor["record_id"],
        sha256=descriptor["sha256"],
        page=0,
    ).public_result()
    assert result["status"] == "ok"
    page = result["result"]
    assert "literal" in page["text"] and "~b" in page["text"]
    request = engine.public_request()
    assert model._request(request, engine.preview_images())[0] == request
    calls = fake_provider(monkeypatch, [action_text(session_id=engine.session_id)])
    model.generate_codex_investigation_action(
        request,
        codex.CodexSettings(max_attempts=1),
        image_previews=engine.preview_images(),
        before_call=lambda: engine.record_model_call(request=request),
    )
    assert calls[0][0][0] == request
    assert ("input", engine._source.fingerprint) in engine._sent_numeric
    assert engine.actions_used == 1 and engine.model_calls_used == 1


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_paired_actions",
        "closing_trial",
        "missing_index",
        "duplicate_experiment",
        "wrong_ledger_pointer",
        "wrong_ledger_owner",
        "invented_check",
        "changed_latest_body",
        "duplicate_check",
        "nonexistent_pending",
        "invalid_state",
        "duplicate_pending",
    ],
)
def test_index_and_phase_tampering_fails_before_provider(monkeypatch, mutation):
    engine = session(require_experiment_plan=True)
    engine.record_model_call(request=engine.public_request())
    assert plan(engine).public_result()["status"] == "ok"
    request = engine.public_request()
    assert model._request(request, engine.preview_images())[0] == request
    row, phase = request["experiment_index"][0], request["closing_phase"]
    if mutation == "unknown_paired_actions":
        phase["allowed_actions"] = ["unknown_external_action"]
        request["response_schema"]["actions"] = {"unknown_external_action": {}}
    elif mutation == "closing_trial":
        phase["active"] = True
    elif mutation == "missing_index":
        del request["experiment_index"]
    elif mutation == "duplicate_experiment":
        request["experiment_index"].append(deepcopy(row))
    elif mutation == "wrong_ledger_pointer":
        row["full_record"]["sha256"] = "0" * 64
    elif mutation == "wrong_ledger_owner":
        descriptor = next(
            item
            for item in request["evidence_notebook"]["record_index"]
            if item["record_id"] == row["full_record"]["record_id"]
        )
        descriptor["owner"]["experiment_id"] = "another-experiment"
    elif mutation == "invented_check":
        row["checks"][0]["outcome"] = "supports"
    elif mutation == "changed_latest_body":
        request["experiment_ledger"][0]["plan"]["hypothesis"] = "Invented change."
    elif mutation == "duplicate_check":
        row["checks"].append(deepcopy(row["checks"][0]))
    elif mutation == "nonexistent_pending":
        phase["pending_experiment_ids"] = ["experiment-other"]
    elif mutation == "invalid_state":
        row["state"] = "complete_and_validated"
    else:
        phase["pending_experiment_ids"] = ["experiment1", "experiment1"]
    monkeypatch.setattr(
        codex, "discover_codex_runtime", lambda *_: pytest.fail("provider discovery")
    )
    with pytest.raises(local.ModelProposalError):
        model.generate_codex_investigation_action(
            request, codex.CodexSettings(), image_previews=engine.preview_images()
        )
    assert engine.model_calls_used == 1
