from __future__ import annotations

import json
from copy import deepcopy

import pytest

from napari_vipp.core import workflow_investigation_model as model
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_investigation import action_schema


def request():
    return {
        "kind": "napari-vipp-investigation-request",
        "version": 1,
        "session_id": "a" * 32,
        "proposal_context_id": "b" * 32,
        "objective": "Investigate the disconnected foreground.",
        "clarifications": [],
        "deferred_questions": [],
        "source": {"shape": [10, 12], "axes": "YX"},
        "image_permissions": {"input_images": False, "output_images": False},
        "budgets": {},
        "retained_candidate_id": None,
        "response_schema": action_schema(),
        "history": [],
        "image_previews": [],
        "rules": [],
        "node_index": [],
        "experiment_ledger": [],
        "experiment_plan_required": True,
    }


def test_transport_preserves_generic_prospective_ledger_and_requirement():
    public = request()
    public["experiment_ledger"] = [
        {
            "experiment_id": "experiment1",
            "state": "planned",
            "plan": {"hypothesis": "Local connectivity may be recoverable."},
            "candidate_id": None,
            "assessment": None,
        }
    ]
    original = deepcopy(public)
    detached, encoded = model._request(public, ())
    assert detached == original
    assert detached["experiment_plan_required"] is True
    assert len(encoded) < model.MAX_INVESTIGATION_REQUEST_BYTES
    public["experiment_ledger"][0]["plan"].clear()
    assert detached["experiment_ledger"][0]["plan"]["hypothesis"]


def test_legacy_requests_remain_accepted_without_inventing_ledger_state():
    public = request()
    for key in ("experiment_ledger", "experiment_plan_required", "node_index"):
        public.pop(key)
    detached, _ = model._request(public, ())
    assert detached == public and "experiment_ledger" not in detached


@pytest.mark.parametrize(
    "changes",
    [
        {"experiment_ledger": {}},
        {"experiment_ledger": ()},
        {"node_index": {}},
        {"experiment_plan_required": 1},
        {"experiment_plan_required": None},
        {"history_encoding": {}},
        {"history_preview_manifests": {}},
        {"history_encoding": [], "history_preview_manifests": {}},
        {"history_encoding": {}, "history_preview_manifests": []},
        {"reference_recipe": {}},
    ],
)
def test_new_fields_do_not_broaden_the_private_or_malformed_request_contract(changes):
    with pytest.raises(local.ModelProposalError, match="public investigation"):
        model._request(request() | changes, ())


def test_complete_request_history_bound_includes_experiment_ledger():
    public = request()
    public["experiment_ledger"] = [
        {"note": "x" * model.MAX_INVESTIGATION_REQUEST_BYTES}
    ]
    with pytest.raises(local.ModelProposalError, match="192 KiB"):
        model._request(public, ())


def test_lossless_history_validation_does_not_expand_the_transmitted_wire(monkeypatch):
    public = request()
    public["history_encoding"] = {"version": 1}
    public["history_preview_manifests"] = {}
    public["history"] = [
        {"result": {"$ref": "experiment_ledger", "experiment_id": "experiment1"}}
    ]
    validated = []

    def expand(detached):
        validated.append(deepcopy(detached))
        return [{"result": {"complete": "Decoded content remains local."}}]

    monkeypatch.setattr(model, "expand_request_history", expand)
    detached, encoded = model._request(public, ())
    assert validated == [public]
    assert detached == public and json.loads(encoded) == public
    assert "$ref" in encoded.decode() and "Decoded content" not in encoded.decode()
    public["history"][0]["result"].clear()
    assert detached["history"][0]["result"]["$ref"] == "experiment_ledger"


def test_real_codec_references_are_preserved_and_changed_hashes_rejected():
    from napari_vipp._tests.test_workflow_investigation_experiments import (
        plan_args,
        validate,
    )
    from napari_vipp.core.workflow_investigation_experiments import ExperimentLedger
    from napari_vipp.core.workflow_investigation_history import compact_history
    from napari_vipp.core.workflow_proposal_images import ImagePreview

    ledger = ExperimentLedger()
    arguments = plan_args()
    snapshot = ledger.plan(**arguments, validate_evidence_refs=validate)
    preview = ImagePreview(
        "input requested view 1", b"\x89PNG\r\n\x1a\nexample", "Native YX; no crop."
    ).manifest()
    original = [
        {
            "action": "plan_experiment",
            "status": "ok",
            "request": arguments,
            "result": snapshot,
            "image_previews": [preview],
        }
    ]
    public = request() | compact_history(original, [snapshot])
    public["experiment_ledger"] = [snapshot]
    public["image_permissions"]["input_images"] = True
    detached, _ = model._request(public, ())
    assert detached == public
    assert detached["history"][0]["request"]["$ref"] == "experiment_ledger"
    assert detached["history"][0]["image_previews"][0]["$ref"] == (
        "history_preview_manifests"
    )
    assert model.expand_request_history(detached) == original
    public["history"][0]["result"]["sha256"] = "f" * 64
    with pytest.raises(
        local.ModelProposalError, match="history references are invalid"
    ):
        model._request(public, ())


@pytest.mark.parametrize("tamper", [None, "index", "hash"])
def test_real_v2_observation_projection_is_verified_without_wire_expansion(tamper):
    from napari_vipp.core.workflow_investigation_history import compact_history

    arguments = {
        "evidence_refs": [{"kind": "numeric", "alias": "input", "sha256": "a" * 64}],
        "interpretation": "A supplied diagnostic leaves local structure uncertain.",
        "hypothesis": "Test a declared bounded interpretation, not observed truth.",
        "next_check": "Inspect the relevant population before choosing a trial.",
    }
    original = [
        {
            "action": "record_observation",
            "status": "ok",
            "request": arguments,
            "result": {
                "evidence_refs": arguments["evidence_refs"],
                "model_interpretation": arguments["interpretation"],
                "hypothesis_not_observed_fact": arguments["hypothesis"],
                "next_check": arguments["next_check"],
                "meaning": (
                    "Model interpretation only; not a measured fact "
                    "or scientific validation."
                ),
            },
            "image_previews": [],
        }
    ]
    public = request() | compact_history(original, [])
    reference = public["history"][0]["result"]
    assert reference["$ref"] == "history_request"
    if tamper is not None:
        reference["action_index" if tamper == "index" else "sha256"] = (
            1 if tamper == "index" else "f" * 64
        )
        with pytest.raises(
            local.ModelProposalError, match="history references are invalid"
        ):
            model._request(public, ())
        return
    detached, encoded = model._request(public, ())
    assert detached == public and json.loads(encoded) == public
    assert model.expand_request_history(detached) == original
    assert detached["history"][0]["request"] == arguments
    assert detached["history"][0]["result"]["action_index"] == 0


def test_real_v1_history_descriptor_remains_accepted():
    from napari_vipp.core.workflow_investigation_history import HISTORY_ENCODING_V1

    public = request() | {
        "history_encoding": deepcopy(HISTORY_ENCODING_V1),
        "history_preview_manifests": {},
    }
    detached, _ = model._request(public, ())
    assert detached == public


@pytest.mark.parametrize("failure", [ValueError, TypeError, KeyError, IndexError])
def test_invalid_lossless_history_reference_is_rejected_safely(monkeypatch, failure):
    public = request()
    public |= {"history_encoding": {"version": 1}, "history_preview_manifests": {}}

    def expand(_request):
        raise failure("Untrusted decoder failure text.")

    monkeypatch.setattr(model, "expand_request_history", expand)
    with pytest.raises(
        local.ModelProposalError, match="history references are invalid"
    ):
        model._request(public, ())


@pytest.mark.parametrize("violation", ["size", "path"])
def test_compact_request_bound_and_privacy_are_checked_before_history_decode(
    monkeypatch, violation
):
    public = request()
    public |= {"history_encoding": {"version": 1}, "history_preview_manifests": {}}
    public["history_preview_manifests"] = {
        "note": (
            "x" * model.MAX_INVESTIGATION_REQUEST_BYTES
            if violation == "size"
            else "C:/private/image.tif"
        )
    }
    monkeypatch.setattr(
        model, "expand_request_history", lambda _: pytest.fail("Unexpected expansion")
    )
    with pytest.raises(local.ModelProposalError):
        model._request(public, ())


def test_experiment_guidance_requires_evidence_and_distinguishes_inference():
    instructions = model._SYSTEM
    for phrase in (
        "plan_experiment BEFORE",
        "weighing plausible alternatives",
        "most informative next test",
        "predicted geometric change",
        "prior assumptions",
        "predeclared",
        "assess_experiment",
        "EVERY predeclared check",
        "not source-only or other-candidate",
        "unresolved",
        "not a new measurement",
        "Colour, grouping",
        "do not recover missing structure",
        "use best judgment to test",
        "inferred geometry explicitly inferred",
        "Reserve enough remaining",
        "next plan_experiment or assess_experiment",
        "Do not insert record_observation after every inspection",
        "standalone record_observation only when necessary discovery",
        "current visual insight would otherwise be lost",
        "explicit lossless references, not summaries",
        "history_preview_manifests",
        "complete recorded plans, assessments",
        "manifest or caption is not a newly supplied image",
        "table key hashes the FULL manifest JSON",
        "For evidence_refs with kind='preview'",
        "resolved manifest's sha256 (the PNG-byte hash), NEVER the table key",
        "Current image_previews remain unchanged full manifests",
        "a history_request reference losslessly reconstructs",
        "same zero-based action_index's full inline request",
        "not a different observation, a new tool, a new model action",
    ):
        assert phrase in instructions
    assert "not observed" in instructions or "inferred versus observed" in instructions


def test_scientific_uncertainty_does_not_replace_a_useful_bounded_test():
    instructions = model._SYSTEM
    for phrase in (
        "Distinguish scientific risk from execution safety",
        "not by themselves reasons to forbid a reversible detached test",
        "Lack of ground truth alone does not prevent testing",
        "Execution, resource, data and permission guards remain mandatory",
        "changes little or nothing in the target region",
        "informative alternative method",
        "controlled parameter-scale probe",
        "trade-offs declared before trial",
        "Do not switch straight to cosmetics",
        "no supportable useful test remains",
        "never turn an untested possibility into an observed reconstruction",
    ):
        assert phrase in instructions
