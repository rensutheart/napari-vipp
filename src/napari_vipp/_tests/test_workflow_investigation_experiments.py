from __future__ import annotations

from copy import deepcopy

import pytest

from napari_vipp.core import workflow_investigation_experiments as experiments

SOURCE = {"kind": "numeric", "alias": "input", "sha256": "a" * 64}
CANDIDATE = {"kind": "numeric", "alias": "candidate1/mask", "sha256": "b" * 64}
OTHER = {"kind": "numeric", "alias": "candidate2/mask", "sha256": "c" * 64}
PREVIEW = {"kind": "preview", "sha256": "d" * 64}
SOURCE_PREVIEW = {"kind": "preview", "sha256": "e" * 64}
ISSUED = [SOURCE, CANDIDATE, OTHER, PREVIEW, SOURCE_PREVIEW]


def validate(refs, *, candidate_id=None, required_kind="numeric"):
    if any(ref not in ISSUED for ref in refs):
        raise experiments.ExperimentError("Evidence was not transmitted.")
    owners = {
        CANDIDATE["sha256"]: "candidate1",
        OTHER["sha256"]: "candidate2",
        PREVIEW["sha256"]: "candidate1",
    }
    if candidate_id is not None and not any(
        ref["kind"] == required_kind and owners.get(ref["sha256"]) == candidate_id
        for ref in refs
    ):
        raise experiments.ExperimentError("Evidence belongs to another candidate.")


def plan_args(**changes):
    return deepcopy(
        {
            "baseline_candidate_id": None,
            "evidence_refs": [SOURCE],
            "observed_failure": (
                "The inspected region contains disconnected foreground."
            ),
            "hypothesis": "A limited reconstruction might restore continuity.",
            "prior_assumptions": ["Nearby foreground may be part of one structure."],
            "predicted_geometric_change": (
                "One local discontinuity may become connected."
            ),
            "risks": ["An unwanted connection or inflated boundary may be introduced."],
            "intent": "reconstruction",
            "evaluation_checks": [
                {
                    "check_id": "local_count",
                    "kind": "numeric",
                    "criterion": "Compare local components.",
                },
                {
                    "check_id": "boundary",
                    "kind": "visual",
                    "criterion": "Inspect nearby shape distortion.",
                },
            ],
        }
        | changes
    )


def assessment_args(**changes):
    return deepcopy(
        {
            "experiment_id": "experiment1",
            "candidate_id": "candidate1",
            "check_results": [
                {
                    "check_id": "local_count",
                    "outcome": "supports",
                    "evidence_refs": [CANDIDATE],
                    "assessment": (
                        "The supplied component diagnostic agrees with the prediction."
                    ),
                },
                {
                    "check_id": "boundary",
                    "outcome": "unresolved",
                    "evidence_refs": [],
                    "assessment": "No relevant candidate preview was supplied.",
                },
            ],
            "unresolved_criteria": [
                "Shape distortion and true continuity remain unverified."
            ],
        }
        | changes
    )


def bound_ledger():
    ledger = experiments.ExperimentLedger()
    ledger.plan(**plan_args(), validate_evidence_refs=validate)
    ledger.bind_trial("experiment1", "candidate1")
    return ledger


def test_prospective_plan_trial_and_evidence_bound_assessment_are_detached():
    ledger = experiments.ExperimentLedger()
    arguments = plan_args()
    result = ledger.plan(**arguments, validate_evidence_refs=validate)
    arguments["prior_assumptions"].clear()
    result["plan"]["evaluation_checks"].clear()
    assert ledger.get("experiment1")["state"] == "planned"
    assert len(ledger.get("experiment1")["plan"]["evaluation_checks"]) == 2
    linked = ledger.bind_trial("experiment1", "candidate1")
    assert linked["state"] == "awaiting_assessment"
    assert not ledger.all_assessed
    with pytest.raises(experiments.ExperimentError, match="before another trial"):
        ledger.require_assessed("candidate1")
    ledger.require_assessed("candidate2")  # Unlinked legacy trial remains compatible.
    arguments = assessment_args()
    result = ledger.assess(**arguments, validate_evidence_refs=validate)
    arguments["check_results"][0]["evidence_refs"].clear()
    result["assessment"]["check_results"].clear()
    actual = ledger.get("experiment1")
    assert actual["state"] == "assessed" and ledger.all_assessed
    assert actual["assessment"]["unresolved_check_ids"] == ["boundary"]
    assert actual["assessment"]["check_results"][0]["evidence_refs"] == [CANDIDATE]
    assert "not proof" in actual["assessment"]["meaning"]
    assert "Presentation-only" in actual["plan"]["meaning"]
    ledger.require_assessed("candidate1")
    snapshots = ledger.snapshots()
    snapshots[0]["plan"].clear()
    assert ledger.get("experiment1")["plan"]["hypothesis"]


@pytest.mark.parametrize("limit", [0, 6, True, 1.5, "5"])
def test_invalid_experiment_limit_rejected(limit):
    with pytest.raises(experiments.ExperimentError, match="limit"):
        experiments.ExperimentLedger(limit)


def test_plan_limit_fails_without_inserting_or_reusing_ids():
    ledger = experiments.ExperimentLedger(2)
    for number in (1, 2):
        assert (
            ledger.plan(**plan_args(), validate_evidence_refs=validate)["experiment_id"]
            == f"experiment{number}"
        )
    with pytest.raises(experiments.ExperimentError, match="limit"):
        ledger.plan(**plan_args(), validate_evidence_refs=validate)
    assert len(ledger.snapshots()) == 2


@pytest.mark.parametrize(
    "changes",
    [
        {"observed_failure": ""},
        {"hypothesis": "x" * 801},
        {"predicted_geometric_change": None},
        {"prior_assumptions": []},
        {"risks": ["risk"] * 5},
        {"intent": "validated_reconstruction"},
        {"intent": []},
        {"baseline_candidate_id": True},
        {"evidence_refs": []},
        {"evidence_refs": [SOURCE, SOURCE]},
        {"evidence_refs": [{"kind": "numeric", "alias": "input", "sha256": "f" * 64}]},
        {"evidence_refs": [SOURCE | {"sha256": "not-a-hash"}]},
        {"evaluation_checks": []},
        {
            "evaluation_checks": [
                {"check_id": "x", "kind": "numeric", "criterion": "x", "extra": 1}
            ]
        },
        {"evaluation_checks": [{"check_id": "x", "kind": "visual", "criterion": "x"}]},
        {
            "evaluation_checks": [
                {"check_id": "x", "kind": "numeric", "criterion": "x"}
            ]
            * 2
        },
    ],
)
def test_invalid_plan_is_atomic_and_never_consumes_a_slot(changes):
    ledger = experiments.ExperimentLedger()
    with pytest.raises(experiments.ExperimentError):
        ledger.plan(**plan_args(**changes), validate_evidence_refs=validate)
    assert ledger.snapshots() == ()
    assert (
        ledger.plan(**plan_args(), validate_evidence_refs=validate)["experiment_id"]
        == "experiment1"
    )


def test_plan_fields_are_strict_and_baseline_requires_its_issued_evidence():
    ledger = experiments.ExperimentLedger()
    with pytest.raises(experiments.ExperimentError, match="exactly"):
        ledger.plan(
            **plan_args(), extra="new authority", validate_evidence_refs=validate
        )
    with pytest.raises(experiments.ExperimentError, match="another candidate"):
        ledger.plan(
            **plan_args(baseline_candidate_id="candidate1"),
            validate_evidence_refs=validate,
        )
    ledger.plan(
        **plan_args(baseline_candidate_id="candidate1", evidence_refs=[CANDIDATE]),
        validate_evidence_refs=validate,
    )
    with pytest.raises(experiments.ExperimentError, match="baseline"):
        ledger.bind_trial("experiment1", "candidate1")


def test_plan_and_trial_identity_cannot_be_rebound_or_retroactively_reused():
    ledger = bound_ledger()
    with pytest.raises(experiments.ExperimentError, match="only one trial"):
        ledger.bind_trial("experiment1", "candidate2")
    ledger.plan(**plan_args(), validate_evidence_refs=validate)
    with pytest.raises(experiments.ExperimentError, match="retrospectively"):
        ledger.bind_trial("experiment2", "candidate1")
    with pytest.raises(experiments.ExperimentError, match="exact experiment"):
        ledger.bind_trial("experiment3", "candidate3")
    with pytest.raises(experiments.ExperimentError, match="exact prospectively"):
        ledger.assess(
            **assessment_args(experiment_id="experiment2"),
            validate_evidence_refs=validate,
        )
    with pytest.raises(experiments.ExperimentError, match="exact prospectively"):
        ledger.assess(
            **assessment_args(candidate_id="candidate2"),
            validate_evidence_refs=validate,
        )


@pytest.mark.parametrize(
    "change",
    [
        "missing-check",
        "duplicate-check",
        "unknown-check",
        "extra-field",
        "invalid-outcome",
        "empty-assessment",
        "unresolved-without-reason",
        "wrong-kind",
        "no-numeric",
        "other-numeric",
        "unsent-hash",
    ],
)
def test_invalid_assessment_cannot_mark_a_trial_assessed(change):
    ledger = bound_ledger()
    arguments = assessment_args()
    results = arguments["check_results"]
    if change == "missing-check":
        results.pop()
    elif change == "duplicate-check":
        results[1]["check_id"] = "local_count"
    elif change == "unknown-check":
        results[1]["check_id"] = "new_after_trial"
    elif change == "extra-field":
        results[0]["accuracy"] = 1
    elif change == "invalid-outcome":
        results[0]["outcome"] = "proven"
    elif change == "empty-assessment":
        results[0]["assessment"] = ""
    elif change == "unresolved-without-reason":
        arguments["unresolved_criteria"] = []
    elif change == "wrong-kind":
        results[0]["evidence_refs"] = [PREVIEW]
    elif change == "no-numeric":
        results[0]["outcome"] = "unresolved"
        results[0]["evidence_refs"] = []
    elif change == "other-numeric":
        results[0]["evidence_refs"] = [OTHER]
    elif change == "unsent-hash":
        results[0]["evidence_refs"][0]["sha256"] = "f" * 64
    with pytest.raises(experiments.ExperimentError):
        ledger.assess(**arguments, validate_evidence_refs=validate)
    assert ledger.get("experiment1")["state"] == "awaiting_assessment"
    assert ledger.get("experiment1")["assessment"] is None


def test_resolved_visual_requires_an_actual_preview_from_the_exact_trial():
    ledger = bound_ledger()
    arguments = assessment_args()
    arguments["check_results"][1] |= {
        "outcome": "contradicts",
        "evidence_refs": [SOURCE_PREVIEW],
        "assessment": "The supplied preview shows an adverse shape change.",
    }
    with pytest.raises(experiments.ExperimentError, match="another candidate"):
        ledger.assess(**arguments, validate_evidence_refs=validate)
    arguments["check_results"][1]["evidence_refs"] = [PREVIEW]
    result = ledger.assess(**arguments, validate_evidence_refs=validate)
    assert result["assessment"]["check_results"][1]["outcome"] == "contradicts"
    with pytest.raises(experiments.ExperimentError, match="immutable"):
        ledger.assess(**arguments, validate_evidence_refs=validate)


@pytest.mark.parametrize("note", ["plan", "assessment"])
def test_note_byte_bounds_are_fail_closed_and_preserve_state(monkeypatch, note):
    ledger = bound_ledger() if note == "assessment" else experiments.ExperimentLedger()
    monkeypatch.setattr(experiments, f"MAX_EXPERIMENT_{note.upper()}_BYTES", 20)
    with pytest.raises(experiments.ExperimentError, match="byte bound"):
        if note == "plan":
            ledger.plan(**plan_args(), validate_evidence_refs=validate)
        else:
            ledger.assess(**assessment_args(), validate_evidence_refs=validate)
    assert (
        ledger.snapshots() == ()
        if note == "plan"
        else ledger.get("experiment1")["assessment"] is None
    )


def test_presentation_plan_never_receives_structural_repair_or_accuracy_verdict():
    ledger = experiments.ExperimentLedger()
    result = ledger.plan(
        **plan_args(intent="presentation"), validate_evidence_refs=validate
    )
    assert result["plan"]["intent"] == "presentation"
    assert "do not repair structural geometry" in result["plan"]["meaning"]
    assert "success" not in result and "accuracy" not in result
