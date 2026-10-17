"""Previous-run evidence is immutable, comparable, finite and independently gated."""

import json
from dataclasses import FrozenInstanceError, replace

import pytest

from napari_vipp._tests.test_workflow_assessment import fixture
from napari_vipp.core import workflow_assessment_model as model
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.workflow_assessment import (
    AssessmentError,
    build_assessment_context,
    capture_assessment_snapshot,
)
from napari_vipp.core.workflow_assessment_history import (
    MAX_PREVIOUS_ASSESSMENT_BYTES,
    PreviousAssessmentRecord,
    make_previous_assessment,
    with_previous_assessment,
)
from napari_vipp.core.workflow_proposals import ProposalDialogue

GOAL = ProposalDialogue("Separate two structures.", (("Which?", "Use best judgment."),))


def context_for(run_id):
    pipeline, nodes = fixture(mesh=True)
    snapshot = capture_assessment_snapshot(pipeline, "input", nodes, run_id)
    return build_assessment_context(
        snapshot, include_input_images=True, include_output_images=True
    )


def report_for(context, *, comparison=None):
    value = {
        "assessment_id": context.assessment_id,
        "run_id": context.run_id,
        "overall": "partly_achieved",
        "summary": "One observed signal is present; separation remains unresolved.",
        "goal_checks": [
            {
                "criterion": "Separate structures",
                "status": "partly_met",
                "evidence": "One foreground component is measured.",
            }
        ],
        "findings": ["No reference mask is available."],
        "suggested_changes": ["Review the structure boundaries before thresholding."],
        "limitations": ["This is not independent ground truth."],
    }
    if comparison is not None:
        value["comparison"] = comparison
    return value


def prior_pair():
    first, second = context_for("run-1"), context_for("run-2")
    report = model.validate_assessment(json.dumps(report_for(first)), first)
    record = make_previous_assessment(first, GOAL, report)
    return first, with_previous_assessment(second, GOAL, record), record


def comparison_for(first, status="improved"):
    return {
        "status": status,
        "previous_assessment_id": first.assessment_id,
        "explanation": (
            "Foreground component counts can now be compared on the same source "
            "grid; this does not establish correctness."
        ),
    }


def test_record_is_detached_frozen_safe_diagnostics_without_pixels_or_private_owner():
    first, second, record = prior_pair()
    public = second.public_request(GOAL)
    old = public["previous_assessment"]
    assert old["assessment_id"] == first.assessment_id
    assert old["run_id"] == record.run_id == "run-1"
    assert old["selected_outputs"] == first.public_request(GOAL)["selected_outputs"]
    assert old["candidate_graph"] == first.public_request(GOAL)["candidate_graph"]
    assert old["assessment"]["overall"] == "partly_achieved"
    assert "image_previews" not in old and "previous_assessment" not in old
    text = json.dumps(old)
    for private in (
        "PRIVATE",
        "png_bytes",
        "vertices",
        "baseline",
        "_snapshot",
        "file_path",
    ):
        assert private not in text
    assert len(record._payload_bytes) <= MAX_PREVIOUS_ASSESSMENT_BYTES
    old["selected_outputs"].clear()
    assert record.public_record()["selected_outputs"]
    assert first.public_request(GOAL)["selected_outputs"]
    with pytest.raises(FrozenInstanceError):
        record._payload_bytes = b"{}"


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "grid",
        "input_permission",
        "output_permission",
        "goal",
        "answers",
        "deferred",
    ],
)
def test_comparison_rejects_changed_goal_answer_source_grid_and_permissions(change):
    first, _second, record = prior_pair()
    target = context_for("run-3")
    objective = GOAL
    if change == "goal":
        objective = "A different goal."
    elif change == "answers":
        objective = ProposalDialogue(GOAL.objective, (("Which?", "Only the circle."),))
    elif change == "deferred":
        objective = ProposalDialogue(
            GOAL.objective, GOAL.clarifications, ("How much?",)
        )
    else:
        evidence = target.public_request(GOAL)
        if change == "source":
            evidence["source"]["sha256"] = "b" * 64
        elif change == "grid":
            evidence["source"]["grid"]["calibration"][0]["scale"] = 2.0
        else:
            evidence["image_permissions"][
                "input_images" if change == "input_permission" else "output_images"
            ] = False
        target = replace(target, _evidence=evidence)
    with pytest.raises(AssessmentError, match="changed"):
        with_previous_assessment(target, objective, record)
    with pytest.raises(AssessmentError, match="different completed run"):
        with_previous_assessment(first, GOAL, record)


def test_each_new_record_drops_nested_predecessors_but_preserves_relative_report():
    first, second, _record = prior_pair()
    report = model.validate_assessment(
        json.dumps(report_for(second, comparison=comparison_for(first))), second
    )
    latest = make_previous_assessment(second, GOAL, report)
    third = with_previous_assessment(context_for("run-3"), GOAL, latest)
    previous = third.public_request(GOAL)["previous_assessment"]
    assert previous["assessment_id"] == second.assessment_id
    assert "previous_assessment" not in previous
    assert (
        previous["assessment"]["comparison"]["previous_assessment_id"]
        == first.assessment_id
    )
    assert first.assessment_id not in json.dumps(previous["selected_outputs"])


@pytest.mark.parametrize(
    "status", ["improved", "mixed", "worse", "unchanged", "cannot_compare"]
)
def test_relative_status_does_not_promote_partial_goal_to_achieved(status):
    first, second, _record = prior_pair()
    value = report_for(second, comparison=comparison_for(first, status))
    checked = model.validate_assessment(json.dumps(value), second)
    assert checked.comparison.status == status
    assert checked.comparison.previous_assessment_id == first.assessment_id
    assert checked.overall == "partly_achieved"
    assert checked.to_dict()["comparison"] == value["comparison"]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "first_run",
        "wrong_id",
        "null_id",
        "confidence",
        "path",
        "empty",
        "unknown",
    ],
)
def test_comparative_report_requires_exact_bound_comparison_without_extra_authority(
    change,
):
    first, second, _record = prior_pair()
    comparison = comparison_for(first)
    if change == "missing":
        comparison = None
    elif change == "first_run":
        comparison["status"] = "first_run"
    elif change == "wrong_id":
        comparison["previous_assessment_id"] = "other"
    elif change == "null_id":
        comparison["previous_assessment_id"] = None
    elif change == "confidence":
        comparison["confidence"] = 0.9
    elif change == "path":
        comparison["explanation"] = "Read C:/PRIVATE/output.tif"
    elif change == "empty":
        comparison["explanation"] = ""
    else:
        comparison["status"] = "perfect"
    with pytest.raises(local.ModelProposalError):
        model.validate_assessment(
            json.dumps(report_for(second, comparison=comparison)), second
        )


def test_new_first_run_schema_requires_first_run_but_old_saved_report_is_readable():
    context = context_for("run-1")
    assert "comparison" in model.assessment_schema(context)["required"]
    schema = model.assessment_schema(context)["properties"]["comparison"]
    assert schema["properties"]["status"]["enum"] == ["first_run"]
    legacy = model.validate_assessment(json.dumps(report_for(context)), context)
    assert legacy.comparison is None
    first_run = {
        "status": "first_run",
        "previous_assessment_id": None,
        "explanation": "No previous run is supplied.",
    }
    checked = model.validate_assessment(
        json.dumps(report_for(context, comparison=first_run)), context
    )
    assert checked.comparison.status == "first_run"
    with pytest.raises(local.ModelProposalError, match="previous run"):
        model.validate_assessment(
            json.dumps(report_for(context, comparison=comparison_for(context))), context
        )


def test_provider_sends_measured_predecessor_and_same_current_approved_previews(
    monkeypatch,
):
    first, second, record = prior_pair()
    runtime = codex.CodexRuntime("test-codex", "test", True, True, "Ready", "a" * 64)
    monkeypatch.setattr(codex, "discover_codex_runtime", lambda *_: runtime)
    calls = []

    def call(*args, **kwargs):
        calls.append((args, kwargs))
        return json.dumps(report_for(second, comparison=comparison_for(first, "mixed")))

    monkeypatch.setattr(codex, "_call_codex", call)
    result = model.generate_codex_assessment(second, GOAL, codex.CodexSettings())
    request, previews = calls[0][0][:2]
    assert request["previous_assessment"] == record.public_record()
    assert previews == second.image_previews
    assert all(preview in second.image_previews for preview in previews)
    assert (
        request["response_schema"]["properties"]["comparison"]["properties"][
            "previous_assessment_id"
        ]["const"]
        == first.assessment_id
    )
    assert result.assessment.comparison.status == "mixed"
    assert "not automatically improvement" in calls[0][1]["assessment_contract"][0]
    with pytest.raises(local.ModelProposalError, match="no longer matches"):
        model.generate_codex_assessment(
            second, "A changed objective.", codex.CodexSettings()
        )
    assert len(calls) == 1


def test_wrong_report_and_oversized_noncanonical_records_are_not_retained():
    first, second, record = prior_pair()
    wrong_report = model.validate_assessment(json.dumps(report_for(first)), first)
    with pytest.raises(AssessmentError, match="different captured run"):
        make_previous_assessment(second, GOAL, wrong_report)
    with pytest.raises(AssessmentError, match="96 KiB"):
        PreviousAssessmentRecord(b" " * (MAX_PREVIOUS_ASSESSMENT_BYTES + 1), b"{}")
    with pytest.raises(AssessmentError, match="invalid"):
        PreviousAssessmentRecord(record._payload_bytes + b" ", record._binding_bytes)
    public = record.public_record()
    public["unapproved_field"] = "hidden"
    with pytest.raises(AssessmentError, match="invalid"):
        PreviousAssessmentRecord(json.dumps(public).encode(), record._binding_bytes)


def test_attaching_comparison_never_silently_truncates_current_evidence():
    _first, _second, record = prior_pair()
    target = context_for("run-3")
    evidence = target.public_request(GOAL)
    # The current payload fits by itself but not with the measured predecessor.
    base_size = len(json.dumps(evidence).encode())
    evidence["limitations"].append("x" * (192 * 1024 - base_size - 512))
    target = replace(target, _evidence=evidence)
    target.public_request(GOAL)
    with pytest.raises(AssessmentError, match="192 KiB"):
        with_previous_assessment(target, GOAL, record)
    assert "previous_assessment" not in target.public_request(GOAL)


def test_record_can_retain_maximum_accepted_limitations_with_added_caveat():
    context = context_for("run-1")
    value = report_for(context)
    value["limitations"] = [f"Independent limitation {i}." for i in range(16)]
    checked = model.validate_assessment(json.dumps(value), context)
    assert len(checked.limitations) == 17
    record = make_previous_assessment(context, GOAL, checked)
    assert record.public_record()["assessment"]["limitations"] == list(
        checked.limitations
    )
