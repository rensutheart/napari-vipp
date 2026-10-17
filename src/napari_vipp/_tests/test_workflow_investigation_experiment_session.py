"""Prospective session reasoning guards over unrelated synthetic arrays."""

from __future__ import annotations

import json

import pytest

from napari_vipp._tests.test_workflow_investigation import act, proposal, session
from napari_vipp.core.workflow_investigation import InvestigationLimits
from napari_vipp.core.workflow_proposals import MAX_REQUEST_BYTES


def plan(engine, *, baseline=None, refs=None, visual=False):
    if refs is None:
        if baseline is None:
            refs = [
                {
                    "kind": "numeric",
                    "alias": "input",
                    "sha256": engine._source.fingerprint,
                }
            ]
        else:
            refs = [
                {
                    "kind": "numeric",
                    "alias": f"{baseline}/wanted",
                    "sha256": engine._candidate(baseline)._outputs[0][1].fingerprint,
                }
            ]
    checks = [
        {
            "check_id": "regional_count",
            "kind": "numeric",
            "criterion": "Measure exact foreground in an explicit native region.",
        }
    ]
    if visual:
        checks.append(
            {
                "check_id": "selected_view",
                "kind": "visual",
                "criterion": "Review candidate sections, not inferred unseen pixels.",
            }
        )
    return act(
        engine,
        "plan_experiment",
        baseline_candidate_id=baseline,
        evidence_refs=refs,
        observed_failure="Supplied evidence leaves structure completeness uncertain.",
        hypothesis="An existing-node candidate may change regional foreground.",
        prior_assumptions=["Nonzero signal is not a biological identity."],
        predicted_geometric_change="Native foreground change, not confirmed anatomy.",
        risks=["Wanted signal could be removed or unwanted signal added."],
        intent="reconstruction",
        evaluation_checks=checks,
    )


def linked_trial(engine, experiment_id, **changes):
    return act(
        engine,
        "trial_candidate",
        proposal=proposal(engine, **changes),
        output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
        experiment_id=experiment_id,
    )


def assessment(engine, experiment_id, candidate_id, *, refs=None, visual_ref=None):
    if refs is None:
        refs = [
            {
                "kind": "numeric",
                "alias": f"{candidate_id}/wanted",
                "sha256": engine._candidate(candidate_id)._outputs[0][1].fingerprint,
            }
        ]
    results = [
        {
            "check_id": "regional_count",
            "outcome": "supports",
            "evidence_refs": refs,
            "assessment": "Counts support a numerical prediction, not accuracy.",
        }
    ]
    if visual_ref is not None:
        results.append(
            {
                "check_id": "selected_view",
                "outcome": "supports",
                "evidence_refs": [visual_ref],
                "assessment": "A supplied view leaves unseen regions uncertain.",
            }
        )
    return act(
        engine,
        "assess_experiment",
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        check_results=results,
        unresolved_criteria=["No independent truth or proof of completeness."],
    )


def test_strict_mode_requires_prospective_plan_without_spending_missing_plan_attempt():
    engine = session(require_experiment_plan=True)
    assert engine.require_experiment_plan
    assert engine.public_request()["experiment_plan_required"] is True
    result = act(
        engine,
        "trial_candidate",
        proposal=proposal(engine),
        output_roles=[{"role": "wanted", "node": "mask", "port": 0}],
    ).public_result()
    assert result["status"] == "error"
    assert "prospective plan_experiment" in result["error"]
    assert engine.candidates == ()
    # Exact input identity is not a substitute for already transmitted evidence.
    assert plan(engine).public_result()["status"] == "error"
    engine.record_model_call(request=engine.public_request())
    created = plan(engine).public_result()
    assert created["status"] == "ok", created
    assert created["result"]["state"] == "planned"
    assert linked_trial(engine, "experiment1").public_result()["status"] == "ok"
    assert engine.experiment_ledger[0]["candidate_id"] == "candidate1"
    assert engine.experiment_ledger[0]["state"] == "awaiting_assessment"
    receipt = engine.experiment_ledger[0]
    receipt["plan"]["hypothesis"] = "tampered"
    assert "tampered" not in json.dumps(engine.experiment_ledger)


def test_linked_completed_trial_requires_owned_assessment_before_next_trial_or_finish():
    engine = session(require_experiment_plan=True)
    engine.record_model_call(request=engine.public_request())
    plan(engine)
    linked_trial(engine, "experiment1")
    engine.record_model_call(request=engine.public_request())
    for action, kwargs in [
        ("trial_candidate", {"proposal": proposal(engine), "output_roles": []}),
        (
            "finish",
            {
                "candidate_id": "candidate1",
                "summary": "Not yet evaluated",
                "limitations": [],
            },
        ),
    ]:
        rejected = act(engine, action, **kwargs).public_result()
        assert rejected["status"] == "error"
        assert "Assess the linked completed experiment" in rejected["error"]
    input_ref = {
        "kind": "numeric",
        "alias": "input",
        "sha256": engine._source.fingerprint,
    }
    rejected = assessment(
        engine, "experiment1", "candidate1", refs=[input_ref]
    ).public_result()
    assert rejected["status"] == "error"
    assert "exact experiment candidate" in rejected["error"]
    assert engine.experiment_ledger[0]["state"] == "awaiting_assessment"
    accepted = assessment(engine, "experiment1", "candidate1").public_result()
    assert accepted["status"] == "ok", accepted
    assert engine.experiment_ledger[0]["state"] == "assessed"
    reused = linked_trial(engine, "experiment1").public_result()
    assert reused["status"] == "error"
    assert len(engine.candidates) == 1
    assert plan(engine, baseline="candidate1").public_result()["status"] == "ok"
    assert (
        linked_trial(engine, "experiment2", threshold=150).public_result()["status"]
        == "ok"
    )


def test_failed_linked_trial_is_not_given_fake_assessment_or_allowed_plan_reuse():
    engine = session(require_experiment_plan=True)
    engine.record_model_call(request=engine.public_request())
    plan(engine)
    invalid = proposal(engine)
    invalid["nodes"][0]["operation_id"] = "unknown_node"
    result = act(
        engine,
        "trial_candidate",
        proposal=invalid,
        output_roles=[],
        experiment_id="experiment1",
    ).public_result()
    assert result["status"] == "error"
    assert len(engine.candidates) == 1
    assert engine.candidates[0].status == "failed"
    assert engine.experiment_ledger[0]["candidate_id"] == "candidate1"
    assert linked_trial(engine, "experiment1").public_result()["status"] == "error"
    assert plan(engine).public_result()["status"] == "ok"
    assert linked_trial(engine, "experiment2").public_result()["status"] == "ok"


def test_visual_check_requires_exact_transmitted_candidate_preview_not_source():
    engine = session(require_experiment_plan=True, include_input_images=True)
    source_preview = {
        "kind": "preview",
        "sha256": engine.preview_images()[0].manifest()["sha256"],
    }
    engine.record_model_call(request=engine.public_request())
    plan(engine, visual=True)
    linked_trial(engine, "experiment1")
    engine.record_model_call(request=engine.public_request())
    invalid = assessment(
        engine, "experiment1", "candidate1", visual_ref=source_preview
    ).public_result()
    assert invalid["status"] == "error"
    assert "preview evidence" in invalid["error"]
    result = act(
        engine,
        "inspect_candidate",
        candidate_id="candidate1",
        role="wanted",
        views=[{"axis": "Z", "indices": [1]}],
    )
    candidate_preview = {
        "kind": "preview",
        "sha256": result.image_previews[0].manifest()["sha256"],
    }
    not_sent = assessment(
        engine, "experiment1", "candidate1", visual_ref=candidate_preview
    ).public_result()
    assert not_sent["status"] == "error"
    # The rejected assessment keeps the approved preview, but has not sent it.
    assert candidate_preview["sha256"] in {
        preview.manifest()["sha256"] for preview in engine.preview_images()
    }
    engine.record_model_call(request=engine.public_request())
    accepted = assessment(
        engine, "experiment1", "candidate1", visual_ref=candidate_preview
    ).public_result()
    assert accepted["status"] == "ok", accepted


def test_requested_plan_mode_cannot_be_boolean_coerced():
    with pytest.raises(ValueError):
        session(require_experiment_plan=1)


def test_largest_contract_batch_five_experiments_and_local_measurements_fit_payload():
    from napari_vipp.core.workflow_proposal_catalogue import build_catalogue

    engine = session(
        require_experiment_plan=True,
        limits=InvestigationLimits(max_attempts=5, max_actions=64),
    )
    ids = [
        row["operation_id"]
        for row in sorted(
            build_catalogue()["operations"], key=lambda row: len(json.dumps(row))
        )[-8:]
    ]
    act(engine, "describe_nodes", operation_ids=ids)
    for index in range(5):
        engine.record_model_call(request=engine.public_request())
        baseline = f"candidate{index}" if index else None
        assert plan(engine, baseline=baseline).public_result()["status"] == "ok"
        experiment_id, candidate_id = f"experiment{index + 1}", f"candidate{index + 1}"
        assert (
            linked_trial(engine, experiment_id, threshold=70 + index).public_result()[
                "status"
            ]
            == "ok"
        )
        # A real model would receive the complete trial result before requesting
        # ROI inspection; merely having a candidate index grants no evidence.
        engine.record_model_call(request=engine.public_request())
        result = act(
            engine,
            "inspect_candidate",
            candidate_id=candidate_id,
            role="wanted",
            roi={"axis_order": "ZYX", "bounds": [[0, 2], [1, 4], [2, 7]]},
            histogram=True,
            percentiles=[0, 100],
        ).public_result()["result"]
        engine.record_model_call(request=engine.public_request())
        assert (
            assessment(
                engine, experiment_id, candidate_id, refs=[result["evidence_ref"]]
            ).public_result()["status"]
            == "ok"
        )
    request = engine.public_request()
    assert len(engine.experiment_ledger) == 5
    assert len(request["experiment_ledger"]) == 1
    assert len(request["experiment_index"]) == 5
    assert "history" not in request
    exported = engine.export_notebook()
    receipts = [
        row["payload"]
        for row, descriptor in zip(
            exported["records"], exported["record_index"], strict=True
        )
        if descriptor["kind"] == "action_receipt"
    ]
    assert len(receipts) == 21
    assert receipts == [json.loads(item) for item in engine._history]
    for entry in request["experiment_index"]:
        stored = engine._notebook.lookup(**entry["full_record"])
        assert stored == next(
            original
            for original in engine.experiment_ledger
            if original["experiment_id"] == entry["experiment_id"]
        )
    assert len(json.dumps(request, separators=(",", ":")).encode()) < MAX_REQUEST_BYTES
