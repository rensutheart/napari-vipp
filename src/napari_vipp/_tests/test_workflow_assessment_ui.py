"""Explicit assessment/preview/revision UI flow with held offline workers."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from napari_vipp import _widget
from napari_vipp._tests.test_workflow_assessment_widget import (
    assessment_editor as assessment_editor,
)
from napari_vipp.core import workflow_assessment_model
from napari_vipp.core.workflow_assessment import AssessmentError
from napari_vipp.core.workflow_assessment_model import AssessmentReport, GoalCheck
from napari_vipp.core.workflow_proposal_model import SafeModelDiagnostic
from napari_vipp.core.workflow_proposals import ProposalError, reference_proposal
from napari_vipp.ui import workflow_assistant


@pytest.fixture
def review(assessment_editor, qtbot, monkeypatch):
    widget, data, _old_dialog, runs = assessment_editor
    monkeypatch.setattr(
        workflow_assistant.WorkflowAssistantDialog,
        "_refresh_codex_connection",
        lambda *_: None,
    )
    monkeypatch.setattr(_widget, "show_workspace_window", lambda window: window.show())
    widget._workflow_assistant_dialog = None
    widget._open_workflow_assistant()
    dialog = widget._workflow_assistant_dialog
    qtbot.addWidget(dialog)
    dialog._codex_authenticated = True
    dialog._connection_busy = False
    dialog.goal_edit.setPlainText(
        "Find separate bright structures and review their masks."
    )
    dialog._sync_buttons()
    workers, provider_calls, previews = [], [], []
    monkeypatch.setattr(
        workflow_assistant,
        "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=workers.append)),
    )

    def fake_assessment(context, objective, settings, **kwargs):
        provider_calls.append((context, objective, settings))
        return SimpleNamespace(
            assessment=AssessmentReport(
                context.assessment_id,
                context.run_id,
                "partly_achieved",
                "A mask was calculated, but its separation needs review.",
                (
                    GoalCheck(
                        "Separate structures",
                        "cannot_assess",
                        "One connected mask is not proof of accurate separation.",
                    ),
                ),
                ("The completed run contains a labelled mask.",),
                ("Inspect the mask boundary and revise the segmentation if needed.",),
                ("This is not an independently validated reference mask.",),
            )
        )

    monkeypatch.setattr(
        workflow_assessment_model, "generate_codex_assessment", fake_assessment
    )
    monkeypatch.setattr(
        workflow_assessment_model,
        "generate_assessment",
        Mock(side_effect=AssertionError("No unapproved local request")),
    )
    monkeypatch.setattr(
        dialog._assessment,
        "preview_dialog",
        lambda reply: previews.append(reply) or SimpleNamespace(exec=lambda: None),
    )
    return SimpleNamespace(
        widget=widget,
        dialog=dialog,
        data=data,
        runs=runs,
        workers=workers,
        provider_calls=provider_calls,
        previews=previews,
    )


def add_and_run(review):
    d = review.dialog
    d.output_assessment_checkbox.setChecked(True)
    context, owner = review.widget._capture_workflow_proposal_context(
        "input", include_images=d.images_checkbox.isChecked()
    )
    d._context, d._owner = context, owner
    raw = reference_proposal(context)
    raw["nodes"] = [node for node in raw["nodes"] if node["key"] != "measure"]
    raw["connections"] = [
        edge for edge in raw["connections"] if edge["target"] != "measure"
    ]
    d._accept_proposal(raw)
    assert d.apply_button.isEnabled()
    d.apply_button.click()
    assert review.runs == []
    ids = d._added_ids
    d.run_button.click()
    assert len(review.runs) == 1
    assert not d._added_ids  # The assessment retains its separate run authority.
    assert d._assessment.candidate_ids == ids
    assert d._assessment.run_id
    assert d._assessment.assess_button.isEnabled()
    return ids


def finish_worker(review):
    worker = review.workers.pop(0)
    worker.run()
    assert not review.dialog._busy
    return worker


def test_two_prominent_image_options_start_off_and_checking_is_passive(review):
    d = review.dialog
    assert not d.images_checkbox.isChecked()
    assert not d.output_assessment_checkbox.isChecked()
    assert d.images_checkbox.parent() is d.output_assessment_checkbox.parent()
    assert d.images_checkbox.objectName() == "AssistantEvidenceOption"
    assert d.output_assessment_checkbox.objectName() == "AssistantEvidenceOption"
    d.images_checkbox.setChecked(True)
    d.output_assessment_checkbox.setChecked(True)
    assert not review.workers and not review.provider_calls and not review.runs
    assert review.widget._workflow_assistant_assessment_run is None
    assert not d._assessment.assess_button.isEnabled()
    d.stage_tabs.setCurrentIndex(2)
    assert d._assessment.card.isVisible()


def test_successful_ordinary_run_enables_local_preview_without_model_request(review):
    add_and_run(review)
    d = review.dialog
    d._assessment.preview_button.click()
    assert d._busy and d._job_kind == "assessment_preview"
    assert not review.provider_calls
    finish_worker(review)
    assert len(review.previews) == 1 and not review.provider_calls
    context = review.previews[0].context
    request = context.public_request(review.previews[0].objective)
    assert request["run_id"] == d._assessment.run_id
    assert context.image_previews  # Output masks only; input checkbox is off.
    assert request["objective"] == d.goal_edit.toPlainText()
    assert d._assessment.reply is None
    assert not d._assessment.revise_button.isEnabled()
    assert len(review.runs) == 1


def test_explicit_assess_renders_report_and_revise_sends_measured_feedback(
    review, monkeypatch
):
    add_and_run(review)
    d = review.dialog
    d._assessment.assess_button.click()
    assert d._busy and d._job_kind == "assessment"
    finish_worker(review)
    assert len(review.provider_calls) == 1
    assert d._assessment.revise_button.isEnabled()
    assert "Goal partly met" in d._assessment.report.text()
    assert "Separate structures" in d._assessment.report.text()
    assert "Output assessment:" in str(d._chat)
    assert len(review.runs) == 1
    sent = []
    monkeypatch.setattr(
        d, "_send", lambda: sent.append(d._context.public_request(d._request_objective))
    )
    original_goal = d.goal_edit.toPlainText()
    d._assessment.revise_button.click()
    assert len(sent) == 1
    feedback = sent[0]["result_feedback"]
    assert sent[0]["objective"] == original_goal
    assert feedback["run_id"] == d._assessment.run_id
    assert feedback["evidence"]["selected_outputs"]
    assert feedback["assessment"]["overall"] == "partly_achieved"
    assert len(review.provider_calls) == 1 and len(review.runs) == 1


@pytest.mark.parametrize("revoke", ["output", "input", "goal", "cancel", "close"])
def test_pending_or_late_assessment_never_adopts_after_changed_consent(review, revoke):
    add_and_run(review)
    d = review.dialog
    d._assessment.assess_button.click()
    worker = review.workers.pop()
    # Build a response before the user changes anything, then deliver it late.
    reply = worker.call()
    assert len(review.provider_calls) == 1
    if revoke == "output":
        d.output_assessment_checkbox.setChecked(False)
    elif revoke == "input":
        d.images_checkbox.setChecked(True)
    elif revoke == "goal":
        d.goal_edit.setPlainText("A changed objective")
    elif revoke == "cancel":
        d.cancel_request()
    else:
        d.close()
    d._ready(worker.generation, reply)
    assert d._assessment.reply is None
    assert not d._assessment.report.text()
    assert len(review.runs) == 1


def test_withdrawal_before_worker_runs_never_calls_provider(review):
    add_and_run(review)
    d = review.dialog
    d._assessment.assess_button.click()
    d.output_assessment_checkbox.setChecked(False)
    finish_worker(review)
    assert not review.provider_calls
    assert d._assessment.reply is None
    assert not d._assessment.assess_button.isEnabled()


def test_cached_results_are_not_reshared_after_output_permission_off_on(review):
    add_and_run(review)
    d = review.dialog
    d.output_assessment_checkbox.setChecked(False)
    d.output_assessment_checkbox.setChecked(True)
    assert not d._assessment.assess_button.isEnabled()
    assert review.widget._workflow_assistant_assessment_run is None
    assert not review.workers and not review.provider_calls


def test_report_text_is_escaped_not_rendered_as_untrusted_markup(review):
    add_and_run(review)
    d = review.dialog
    d._assessment.assess_button.click()
    worker = review.workers.pop()
    reply = worker.call()
    report = reply.result.assessment
    changed = SimpleNamespace(
        assessment=AssessmentReport(
            report.assessment_id,
            report.run_id,
            report.overall,
            '<img src="https://invalid.example/private"> & <script>bad</script>',
            report.goal_checks,
            report.findings,
            report.suggested_changes,
            report.limitations,
        )
    )
    d._ready(worker.generation, replace(reply, result=changed))
    assert "&lt;img" in d._assessment.report.text()
    assert "<img" not in d._assessment.report.text()
    assert json.loads(d.trace_edit.toPlainText())["overall"] == "partly_achieved"


def assessed_feedback(review):
    add_and_run(review)
    d = review.dialog
    d._assessment.assess_button.click()
    finish_worker(review)
    reply = d._assessment.reply
    context, _owner = review.widget._capture_workflow_proposal_context("input")
    return replace(
        context,
        _result_feedback=reply.context.revision_feedback(
            reply.result.assessment.to_dict()
        ),
        _result_feedback_guard=reply.context.snapshot,
    )


def test_measured_revision_context_keeps_full_feedback_detached(review):
    context = assessed_feedback(review)
    objective = review.dialog.goal_edit.toPlainText()
    request = context.public_request(objective)
    assert request["result_feedback"] == context._result_feedback
    assert request["result_feedback"]["evidence"]["selected_outputs"]
    request["result_feedback"]["evidence"]["selected_outputs"].clear()
    assert context.public_request(objective)["result_feedback"]["evidence"][
        "selected_outputs"
    ]
    context.assert_current(review.widget.pipeline)


def test_feedback_request_exceeds_bound_fails_instead_of_truncating(review):
    context = assessed_feedback(review)
    too_large = dict(context._result_feedback, oversized="x" * 196_608)
    context = replace(context, _result_feedback=too_large)
    with pytest.raises(ProposalError, match="exceeds 192 KiB"):
        context.public_request(review.dialog.goal_edit.toPlainText())
    assert len(context._result_feedback["oversized"]) == 196_608


def test_arbitrary_object_cannot_impersonate_result_guard(review):
    context = assessed_feedback(review)
    with pytest.raises(TypeError, match="actual run snapshot"):
        replace(context, _result_feedback_guard=object())


def test_revised_proposal_freshness_includes_changed_candidate_intermediate(review):
    context = assessed_feedback(review)
    blur = next(
        node
        for node in review.widget.pipeline.nodes.values()
        if node.operation_id == "gaussian_blur"
    )
    pixels = review.widget.pipeline.outputs[blur.id]
    pixels.flags.writeable = True
    pixels.flat[0] += 0.5
    with pytest.raises(AssessmentError, match="changed"):
        context.assert_current(review.widget.pipeline)


def test_output_permission_withdrawal_cancels_pending_measured_revision(review):
    add_and_run(review)
    d = review.dialog
    d._assessment.assess_button.click()
    finish_worker(review)
    d._assessment.revise_button.click()
    assert d._busy and d._job_kind == "proposal"
    assert d._context._result_feedback is not None
    pending = review.workers.pop()
    d.output_assessment_checkbox.setChecked(False)
    assert not d._busy
    assert d._context is None and not d._draft_authorized
    assert not d.apply_button.isEnabled() and not d.run_button.isEnabled()
    # A late proposal delivery is rejected by generation without needing its
    # body to be parsed, added, executed or sent to any provider in this test.
    d._ready(pending.generation, object())
    assert d._prepared is None
    assert len(review.runs) == 1 and len(review.provider_calls) == 1


def test_closed_dialog_cannot_capture_its_retained_run_before_native_deletion(review):
    ids = add_and_run(review)
    d = review.dialog
    snapshot, owner = review.widget._capture_workflow_assessment_snapshot(ids, "input")
    d.close()
    with pytest.raises(ProposalError, match="not enabled"):
        review.widget._assert_workflow_assessment_current(snapshot, owner)


@pytest.mark.parametrize("change", ["answers", "message", "initiative"])
def test_assessment_reply_cannot_adopt_after_dialogue_changed(review, change):
    add_and_run(review)
    d = review.dialog
    d._assessment.assess_button.click()
    worker = review.workers.pop()
    reply = worker.call()
    if change == "answers":
        d._clarifications.append(("Which structures?", "Only the rounded one"))
    elif change == "message":
        d.discussion_edit.setPlainText("Only the rounded structure now")
    else:
        d.initiative_slider.setValue(0)
    d._ready(worker.generation, reply)
    assert d._assessment.reply is None
    assert not d._assessment.report.text()


def test_failed_second_assessment_cannot_resend_previous_feedback_after_opt_out(
    review, monkeypatch
):
    add_and_run(review)
    d = review.dialog
    d._assessment.assess_button.click()
    finish_worker(review)
    sent = []
    monkeypatch.setattr(
        d, "_send", lambda: sent.append(d._context.public_request(d._request_objective))
    )
    d._assessment.revise_button.click()
    assert sent[-1].get("result_feedback") is not None
    # An assessment failure clears its report but must never leave a hidden
    # measured revision attached to the next ordinary Send.
    d._assessment.failed(
        SimpleNamespace(
            attempts=(), diagnostic=SafeModelDiagnostic("codex", "turn", "network")
        )
    )
    assert not d._assessment.revision_requested
    d.output_assessment_checkbox.setChecked(False)
    d.discussion_edit.setPlainText("Please draft a fresh plan without result feedback.")
    d._send_message_clicked()
    assert len(sent) == 2
    assert "result_feedback" not in sent[-1]
    assert not d._context.image_previews
