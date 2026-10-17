"""Bounded refinement through real Add/Run with offline provider adapters."""

from types import SimpleNamespace

import pytest

from napari_vipp._tests.test_workflow_assessment_ui import (
    add_and_run,
    finish_worker,
)
from napari_vipp._tests.test_workflow_assessment_ui import (
    review as review,
)
from napari_vipp._tests.test_workflow_assessment_widget import (
    assessment_editor as assessment_editor,
)
from napari_vipp.core import workflow_assessment_model, workflow_proposal_codex
from napari_vipp.core.workflow_assessment_model import (
    AssessmentComparison,
    AssessmentReport,
    GoalCheck,
)
from napari_vipp.core.workflow_proposals import reference_proposal


@pytest.fixture
def refining(review, monkeypatch):
    # Exercise the retained non-investigative connector/controller path here.
    # Real ChatGPT warm-start execution has separate product CPU acceptance tests.
    monkeypatch.setattr(review.dialog._investigation, "capture", None)
    counts = {"assessment": 0, "revision": 0}
    requests = []

    def assess(context, objective, settings, **kwargs):
        counts["assessment"] += 1
        public = context.public_request(objective)
        requests.append(public)
        prior = public.get("previous_assessment")
        return SimpleNamespace(
            assessment=AssessmentReport(
                context.assessment_id,
                context.run_id,
                "partly_achieved",
                "The mask still needs boundary review.",
                (
                    GoalCheck(
                        "Separate objects", "partly_met", "One mask was calculated."
                    ),
                ),
                ("The mask is connected.",),
                ("Try another smoothing setting.",),
                ("Not a validated reference.",),
                AssessmentComparison(
                    "mixed" if prior else "first_run",
                    prior["assessment_id"] if prior else None,
                    "The boundaries changed, but accuracy is not established."
                    if prior
                    else "No previous run has been assessed.",
                ),
            )
        )

    def propose(context, objective, settings, **kwargs):
        counts["revision"] += 1
        requests.append(context.public_request(objective))
        raw = reference_proposal(context)
        raw["nodes"] = [n for n in raw["nodes"] if n["key"] != "measure"]
        raw["connections"] = [e for e in raw["connections"] if e["target"] != "measure"]
        next(n for n in raw["nodes"] if n["operation_id"] == "gaussian_blur")["params"][
            "sigma"
        ] = 1.0 + counts["revision"]
        return SimpleNamespace(proposal=raw, attempts=())

    monkeypatch.setattr(workflow_assessment_model, "generate_codex_assessment", assess)
    monkeypatch.setattr(workflow_proposal_codex, "generate_codex_proposal", propose)
    review.counts, review.requests = counts, requests
    return review


def complete_job(review, qtbot):
    finish_worker(review)
    qtbot.wait(5)  # Let the guarded, zero-delay continuation run once.


def test_default_manual_slider_changes_are_passive(refining):
    d = refining.dialog
    assert d._refinement.slider.value() == 0
    d._refinement.slider.setValue(3)
    assert not refining.workers and not refining.runs
    assert not d._refinement.start_button.isEnabled()
    add_and_run(refining)
    assert not refining.workers  # Ordinary Run does not start a loop.
    assert d._refinement.start_button.isEnabled()


def test_two_rounds_add_run_assess_with_previous_evidence(refining, qtbot):
    add_and_run(refining)
    d = refining.dialog
    original = d.goal_edit.toPlainText()
    d._refinement.slider.setValue(2)
    d._refinement.start_button.click()
    for _ in range(5):  # initial assessment, then two revision/assessment pairs
        assert refining.workers
        complete_job(refining, qtbot)
    assert not d._refinement.active
    assert d._refinement.loop.completed_rounds == 2
    assert len(refining.runs) == 3
    assert refining.counts == {"assessment": 3, "revision": 2}
    assert not refining.workers
    assert "limit" in d._refinement.note.text()
    assessments = [
        r
        for r in refining.requests
        if r["kind"] == "napari-vipp-output-assessment-request"
    ]
    assert "previous_assessment" not in assessments[0]
    for current, previous in zip(assessments[1:], assessments[:-1], strict=True):
        assert (
            current["previous_assessment"]["assessment_id"] == previous["assessment_id"]
        )
        assert current["previous_assessment"]["selected_outputs"]
        assert "previous_assessment" not in current["previous_assessment"]
    assert len(d._refinement._history) == 3
    assert "Change since the previous run" in d._refinement.history.text()
    assert d.goal_edit.toPlainText() == original
    assert d._clarifications == []  # No machine instruction becomes a user answer.


@pytest.mark.parametrize(
    "action", ["stop", "close", "output_off", "input_change", "deadline"]
)
def test_stop_ignores_late_reply_and_does_not_run(refining, qtbot, action):
    add_and_run(refining)
    d = refining.dialog
    d._refinement.slider.setValue(2)
    d._refinement.start_button.click()
    worker = refining.workers.pop()
    reply = worker.call()
    if action == "stop":
        d._refinement.stop_button.click()
    elif action == "close":
        d.close()
    elif action == "output_off":
        d.output_assessment_checkbox.setChecked(False)
    elif action == "input_change":
        d.images_checkbox.setChecked(True)
    else:
        d._refinement.loop.deadline = 0
        d._refinement._tick()
    d._ready(worker.generation, reply)
    qtbot.wait(5)
    assert not d._refinement.active and not refining.workers
    assert not d._busy
    assert len(refining.runs) == 1


def test_loop_stops_on_questions_without_adding(refining, monkeypatch, qtbot):
    add_and_run(refining)
    d = refining.dialog
    d._refinement.slider.setValue(2)
    d._refinement.start_button.click()
    complete_job(refining, qtbot)
    worker = refining.workers.pop()
    result = worker.call()
    result.proposal["questions"] = ["Which structure should be retained?"]
    d._ready(worker.generation, result)
    qtbot.wait(5)
    assert not d._refinement.active and len(refining.runs) == 1
    assert not d._added_ids and d._prepared.questions


def test_run_denial_stops_before_assessment(refining, monkeypatch, qtbot):
    add_and_run(refining)
    d = refining.dialog
    d._refinement.slider.setValue(1)
    d._refinement.start_button.click()
    complete_job(refining, qtbot)
    monkeypatch.setattr(d, "_run", lambda ids: False)
    complete_job(refining, qtbot)
    assert not d._refinement.active
    assert not refining.workers and len(refining.runs) == 1
    assert "denied" in d._refinement.note.text()


def test_manual_revision_preserves_previous_run_for_new_assessment(refining, qtbot):
    add_and_run(refining)
    d = refining.dialog
    d._assessment.assess_button.click()
    complete_job(refining, qtbot)
    first = d._assessment.reply.context.assessment_id
    d._assessment.revise_button.click()
    complete_job(refining, qtbot)
    assert d.apply_button.isEnabled()
    d.apply_button.click()
    d.run_button.click()
    d._assessment.assess_button.click()
    complete_job(refining, qtbot)
    assert (
        d._assessment.reply.context._evidence["previous_assessment"]["assessment_id"]
        == first
    )
    assert len(refining.runs) == 2
    # Reassessing the latest run retains its predecessor, not itself or first_run.
    d._assessment.assess_button.click()
    complete_job(refining, qtbot)
    assert (
        d._assessment.reply.context._evidence["previous_assessment"]["assessment_id"]
        == first
    )


def test_capture_failure_stops_without_waiting_or_sending(refining, monkeypatch):
    add_and_run(refining)
    d = refining.dialog
    d._refinement.slider.setValue(2)
    original = d._assessment.capture
    calls = []

    def capture(*args):
        calls.append(True)
        if len(calls) > 1:
            raise ValueError("The retained result changed.")
        return original(*args)

    monkeypatch.setattr(d._assessment, "capture", capture)
    d._refinement.start_button.click()
    assert not d._refinement.active
    assert not d._busy and not refining.workers
    assert not refining.provider_calls
    assert len(refining.runs) == 1


def test_existing_assessment_is_reused_and_repeated_draft_is_not_run(
    refining, qtbot, monkeypatch
):
    add_and_run(refining)
    d = refining.dialog
    d._assessment.assess_button.click()
    complete_job(refining, qtbot)
    assert refining.counts["assessment"] == 1
    d._refinement.slider.setValue(3)
    d._refinement.start_button.click()
    assert d._job_kind == "proposal"
    worker = refining.workers.pop()
    result = worker.call()
    # Match the initial ordinary reference candidate, including its blur setting.
    next(n for n in result.proposal["nodes"] if n["operation_id"] == "gaussian_blur")[
        "params"
    ]["sigma"] = 1.2
    d._ready(worker.generation, result)
    qtbot.wait(5)
    assert not d._refinement.active
    assert "unchanged" in d._refinement.note.text()
    assert len(refining.runs) == 1
    assert refining.counts["assessment"] == 1


def test_new_workflow_tab_resets_predecessor_and_visible_history(
    refining, qtbot, monkeypatch
):
    add_and_run(refining)
    d, widget = refining.dialog, refining.widget
    d._assessment.assess_button.click()
    complete_job(refining, qtbot)
    assert d._assessment.previous_record is not None
    assert d._refinement._history
    monkeypatch.setattr(widget, "run_pipeline", lambda **kwargs: None)
    old_session = widget._workflow_tabs.current.session_id
    widget._new_workflow()
    assert widget._workflow_tabs.current.session_id != old_session
    assert d._assessment.previous_record is None
    assert d._assessment.comparison_record is None
    assert not d._refinement._history and not d._refinement.history.text()
    assert not d._assessment.run_id and d._assessment.reply is None
    assert d._context is None and not d._draft_authorized


def test_session_change_cancels_pending_assessment_and_ignores_late_reply(
    refining, qtbot
):
    add_and_run(refining)
    d = refining.dialog
    d._refinement.slider.setValue(2)
    d._refinement.start_button.click()
    worker = refining.workers.pop()
    reply = worker.call()
    d.assessment_session_changed()
    d._ready(worker.generation, reply)
    qtbot.wait(5)
    assert not d._refinement.active and not d._busy
    assert d._assessment.previous_record is None
    assert not refining.workers and len(refining.runs) == 1
