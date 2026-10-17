"""Independent adversarial coverage of exact bounded decision-context reuse."""

import hashlib
import json
from copy import deepcopy

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_investigation import act, session
from napari_vipp.core.workflow_investigation import (
    InvestigationError,
    InvestigationStopped,
    _json,
)
from napari_vipp.core.workflow_investigation_context import (
    CONTEXT_KIND,
    MAX_CONTEXT_BYTES,
    DecisionContextError,
    build_decision_context,
    validate_decision_context,
)


def empty_context(session_id="review_session"):
    return {
        "kind": CONTEXT_KIND,
        "version": 1,
        "session_id": session_id,
        "recipes": [],
        "node_contracts": [],
        "numeric_evidence": [],
        "omitted": [],
    }


def numeric(alias, payload):
    return {
        "alias": alias,
        "payload": deepcopy(payload),
        "evidence_ref": {
            "kind": "numeric",
            "alias": alias,
            "sha256": hashlib.sha256(_json(payload)).hexdigest(),
        },
    }


def registered_contracts():
    from napari_vipp.core.workflow_investigation import REFINEMENT_SAFE_OPERATION_IDS
    from napari_vipp.core.workflow_proposal_catalogue import build_catalogue

    return [
        row | {"trial_allowed": row["operation_id"] in REFINEMENT_SAFE_OPERATION_IDS}
        for row in build_catalogue()["operations"]
    ]


def test_exact_context_preserves_unknown_finite_values_without_minting_authority():
    context = empty_context()
    original = numeric("candidate1/check", {"values": [None, False, 0, 0.0, "µm"]})
    context["numeric_evidence"] = [deepcopy(original)]
    validated = validate_decision_context(context, session_id="review_session")
    assert validated == context
    validated["numeric_evidence"][0]["payload"]["values"].append("changed")
    assert context["numeric_evidence"] == [original]


@pytest.mark.parametrize(
    "mutation",
    ["bool_version", "wrong_session", "extra", "alias", "hash", "payload", "path"],
)
def test_typed_context_rejects_tampering_and_private_literals(mutation):
    context = empty_context()
    context["numeric_evidence"] = [numeric("candidate1/check", {"measured": 4})]
    if mutation == "bool_version":
        context["version"] = True
    elif mutation == "wrong_session":
        context["session_id"] = "another_session"
    elif mutation == "extra":
        context["authority"] = True
    elif mutation == "alias":
        context["numeric_evidence"][0]["evidence_ref"]["alias"] = "candidate2/check"
    elif mutation == "hash":
        context["numeric_evidence"][0]["evidence_ref"]["sha256"] = "0" * 64
    elif mutation == "payload":
        context["numeric_evidence"][0]["payload"]["measured"] = 5
    else:
        context["numeric_evidence"] = [
            numeric("candidate1/check", {"private": "C:/private/image.tif"})
        ]
    with pytest.raises(DecisionContextError):
        validate_decision_context(context, session_id="review_session")


def test_omission_reserve_counts_actual_bytes_and_never_trims_contracts():
    numbers = [
        numeric(f"candidate1/check{i}", {"i": i, "value": "x" * 900}) for i in range(24)
    ]
    contracts = registered_contracts()
    result = build_decision_context(
        session_id="review_session",
        recipes=[],
        node_contracts=contracts,
        numeric_evidence=numbers,
    )
    assert len(_json(result)) <= MAX_CONTEXT_BYTES
    for contract in result["node_contracts"]:
        assert contract in contracts  # Full exact entry, not a shortened contract.
    assert result["omitted"]  # The finite working view must disclose exclusion.


@pytest.mark.parametrize(
    "field",
    ["inputs", "outputs", "optional_parameters", "axis_contract", "limitations"],
)
def test_incomplete_node_contract_is_not_presented_as_whole(field):
    context = empty_context()
    context["node_contracts"] = [registered_contracts()[0]]
    del context["node_contracts"][0][field]
    with pytest.raises(DecisionContextError):
        validate_decision_context(context, session_id="review_session")


def test_duplicate_alias_or_retyped_parameter_contract_fails_closed():
    context = empty_context()
    first = numeric("candidate1/check", {"value": 1})
    second = numeric("candidate1/check", {"value": 2})
    context["numeric_evidence"] = [first, second]
    with pytest.raises(DecisionContextError):
        validate_decision_context(context, session_id="review_session")
    context = empty_context()
    contract = next(row for row in registered_contracts() if row["parameters"])
    context["node_contracts"] = [contract]
    contract["parameters"][0]["choices"] = "not the registered choices list"
    with pytest.raises(DecisionContextError):
        validate_decision_context(context, session_id="review_session")


def test_exact_recent_numeric_survives_bookkeeping_but_only_actual_send_grants_it():
    engine = session()
    result = act(engine, "inspect_source").public_result()["result"]
    evidence = result["evidence_ref"]
    act(engine, "find_nodes", query="threshold")
    request = engine.public_request()
    wrappers = request["decision_context"]["numeric_evidence"]
    assert any(row["evidence_ref"] == evidence for row in wrappers)
    before = engine.export_notebook()
    engine.public_request()
    assert engine.export_notebook() == before
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([evidence])
    engine.record_model_call(request=request)
    engine._validate_evidence_refs([evidence])


def test_pinned_quoted_wrapper_cannot_become_a_typed_current_measurement():
    engine = session()
    fake = numeric("candidate99/forged", {"claimed": 900})
    descriptor = engine._notebook.append("action_receipt", {"quoted": fake})
    reference = {key: descriptor[key] for key in ("record_id", "sha256")}
    act(engine, "pin_notebook", refs=[reference])
    request = engine.public_request()
    assert fake not in request["decision_context"]["numeric_evidence"]
    engine.record_model_call(request=request)
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([fake["evidence_ref"]])
    assert engine.export_notebook()["pinned_record_refs"] == [reference]


def test_complete_contracts_survive_pins_without_rendering_or_pin_replacement(
    monkeypatch,
):
    engine = session()
    contracts = act(
        engine,
        "describe_nodes",
        operation_ids=["subtract_background", "binary_threshold"],
    ).public_result()["result"]
    description = engine._latest_action_record
    act(engine, "pin_notebook", refs=[description])
    pins = engine.export_notebook()["pinned_record_refs"]
    monkeypatch.setattr(engine, "_render_views", lambda *a, **k: pytest.fail("render"))
    monkeypatch.setattr(
        engine, "_render_mesh_views", lambda *a, **k: pytest.fail("render")
    )
    act(engine, "find_nodes", query="threshold")
    request = engine.public_request()
    assert all(
        row in request["decision_context"]["node_contracts"] for row in contracts
    )
    assert engine.export_notebook()["pinned_record_refs"] == pins
    assert engine.preview_images() == ()


def test_changed_context_payload_or_consent_revokes_actual_send():
    engine = session()
    act(engine, "inspect_source")
    request = engine.public_request()
    changed = deepcopy(request)
    changed["decision_context"]["numeric_evidence"][0]["payload"]["forged"] = 1
    with pytest.raises(InvestigationError, match="outgoing request"):
        engine.record_model_call(request=changed)
    assert engine.model_calls_used == 0 and engine._sent_numeric == set()
    engine.include_output_images = False
    with pytest.raises(InvestigationStopped, match="permissions changed"):
        engine.record_model_call(request=request)


def test_older_untransmitted_roi_is_not_minted_by_latest_alias_carry():
    engine = session()
    replies = []
    for z in (0, 1):
        replies.append(
            act(
                engine,
                "inspect_source",
                roi={"axis_order": "ZYX", "bounds": [[z, z + 1], [0, 8], [0, 10]]},
            ).public_result()["result"]
        )
    act(engine, "find_nodes", query="threshold")
    request = engine.public_request()
    evidence = [
        row["evidence_ref"] for row in request["decision_context"]["numeric_evidence"]
    ]
    assert replies[1]["evidence_ref"] in evidence
    assert replies[0]["evidence_ref"] not in evidence
    engine.record_model_call(request=request)
    engine._validate_evidence_refs([replies[1]["evidence_ref"]])
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([replies[0]["evidence_ref"]])


def test_source_revision_change_blocks_the_previously_captured_context():
    engine = session()
    act(engine, "inspect_source")
    request = engine.public_request()
    changed = np.array(engine._pipeline.outputs["input"], copy=True)
    changed.flat[0] += 1
    engine._pipeline.outputs["input"] = changed
    with pytest.raises(InvestigationStopped, match="revision changed"):
        engine.record_model_call(request=request)
    assert engine.model_calls_used == 0 and engine._sent_numeric == set()


def test_context_omission_does_not_grant_the_unsupplied_cached_numeric(monkeypatch):
    import napari_vipp.core.workflow_investigation as module

    engine = session()
    original = act(engine, "inspect_source").public_result()["result"]
    act(engine, "find_nodes", query="threshold")
    full = engine.public_request()
    mandatory = dict(full)
    mandatory.pop("decision_context")
    monkeypatch.setattr(
        module, "WORKING_REQUEST_TARGET_BYTES", len(_json(mandatory)) + 650
    )
    bounded = engine.public_request()
    supplied = bounded.get("decision_context", {}).get("numeric_evidence", [])
    assert not any(row["evidence_ref"] == original["evidence_ref"] for row in supplied)
    engine.record_model_call(request=bounded)
    with pytest.raises(InvestigationError, match="not transmitted"):
        engine._validate_evidence_refs([original["evidence_ref"]])
    assert module.MAX_REQUEST_BYTES == 192 * 1024


def test_automatic_views_never_evict_a_full_explicit_pin_board(monkeypatch):
    engine = session(include_input_images=True)
    act(engine, "inspect_source", views=[{"axis": "Z", "indices": [0]}])
    result = act(
        engine, "inspect_source", views=[{"axis": "X", "indices": list(range(7))}]
    ).public_result()
    assert result["status"] == "ok" and len(result["image_previews"]) == 7
    view_id = result["result"]["view_id"]
    expected = result["image_previews"]
    monkeypatch.setattr(engine, "_render_views", lambda *a, **k: pytest.fail("render"))
    pinned = act(engine, "pin_previews", view_ids=[view_id]).public_result()
    assert pinned["status"] == "ok"
    assert engine._pinned_view_ids == (view_id,)
    assert [preview.manifest() for preview in engine.preview_images()] == expected
    request = engine.public_request()
    assert request["image_previews"] == expected
    assert sum(row["byte_count"] for row in expected) <= 2 * 1024 * 1024


def test_context_source_opt_out_does_not_invent_previews_or_expose_local_names():
    engine = session(include_input_images=False, include_output_images=False)
    act(engine, "inspect_source")
    request = engine.public_request()
    assert request["decision_context"]["recipes"] == []
    assert request["image_previews"] == [] and engine.preview_images() == ()
    assert "PRIVATE source" not in json.dumps(request)
