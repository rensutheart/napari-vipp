"""User-visible send/loading/result flow, with held workers and no inference."""

from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui import workflow_assistant


def _navigation_state(dialog):
    return {
        "goal": dialog.goal_edit.toPlainText(),
        "message": dialog.discussion_edit.toPlainText(),
        "answers": tuple(
            (question, field.text()) for question, field in dialog._question_fields
        ),
        "clarifications": tuple(dialog._clarifications),
        "chat": tuple(dialog._chat),
        "drafts": tuple(dialog._drafts),
        "request": dialog.request_edit.toPlainText(),
        "proposal": dialog.json_edit.toPlainText(),
        "objective": dialog._request_objective,
        "context": id(dialog._context),
        "owner": id(dialog._owner),
        "prepared": id(dialog._prepared),
        "busy": dialog._busy,
        "authority": (
            dialog._generation,
            dialog._draft_authorized,
            dialog._added_ids,
            dialog._current_draft_index,
            dialog.draft_combo.currentIndex(),
            dialog.apply_button.isEnabled(),
            dialog.run_button.isEnabled(),
            dialog.consent_checkbox.isChecked(),
            dialog._planning_preferences(),
        ),
    }


def _assert_passive_round_trip(assistant, monkeypatch):
    dialog = assistant.dialog
    actions = []
    for method in ("prepare", "_send", "apply_proposal", "_run_added_plan"):
        monkeypatch.setattr(dialog, method, lambda: actions.append("action"))
    dialog.stage_tabs.setCurrentIndex(0)
    before = _navigation_state(dialog)
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    applies, runs = list(assistant.applies), list(assistant.runs)
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
    for expected, label in ((1, "Next: Discussion"), (2, "View plan")):
        assert dialog.next_button.isVisible() and dialog.next_button.isEnabled()
        assert dialog.next_button.text() == label
        dialog.next_button.click()
        assert dialog.stage_tabs.currentIndex() == expected
    assert dialog.next_button.isHidden() and not dialog.next_button.isEnabled()
    dialog._next_stage_clicked()  # Even a direct invocation cannot advance Plans.
    assert dialog.stage_tabs.currentIndex() == 2
    for expected in (1, 0):
        dialog.back_button.click()
        assert dialog.stage_tabs.currentIndex() == expected
    assert (
        dialog.next_button.isVisible()
        and dialog.next_button.text() == "Next: Discussion"
    )
    assert _navigation_state(dialog) == before
    assert assistant.captures == captures and assistant.assertions == assertions
    assert assistant.applies == applies and assistant.runs == runs
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert not assistant.data.flags.writeable and actions == []


@pytest.fixture
def pending(assistant, monkeypatch):
    dialog = assistant.dialog
    queued = []
    monkeypatch.setattr(
        workflow_assistant,
        "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=queued.append)),
    )
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("offline-synthetic-model")
    dialog.stage_tabs.setCurrentIndex(0)
    dialog.prepare_button.click()
    assert len(queued) == 1  # The held worker never invokes a model.
    assert dialog._busy
    return assistant, queued, dialog._generation


def test_goal_has_one_send_action_and_footer_close(assistant):
    dialog = assistant.dialog
    assert dialog.images_checkbox.text() == "Include input image previews"
    assert not dialog.images_checkbox.isChecked()
    assert dialog.next_button.isHidden()
    assert not dialog.next_button.isEnabled()
    assert not dialog.request_progress.isVisible()
    assert dialog.close_button.parentWidget() is dialog.bottom_frame
    assert not dialog.close_button.autoDefault()
    assert assistant.captures == assistant.applies == assistant.runs == []


def test_pending_goal_stays_on_goal_and_shows_indeterminate_progress(pending):
    assistant, queued, generation = pending
    dialog = assistant.dialog
    assert dialog.stage_tabs.currentIndex() == 0
    assert dialog.prepare_button.text() == "Sending goal…"
    assert not dialog.prepare_button.isEnabled()
    assert dialog.request_progress.isVisible()
    assert dialog.request_progress.minimum() == dialog.request_progress.maximum() == 0
    assert dialog.cancel_button.isEnabled()
    assert dialog.next_button.isHidden()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    dialog._next_stage_clicked()
    dialog.next_button.clicked.emit()
    assert dialog.stage_tabs.currentIndex() == 0
    dialog.prepare_button.click()
    dialog._progress(generation, "Checking plan")
    assert dialog.stage_tabs.currentIndex() == 0
    assert dialog.request_progress.isVisible()
    assert len(queued) == 1
    assert assistant.captures == [("input", False)]
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("questions", [[], ["Which boundary should be preserved?"]])
def test_checked_reply_routes_to_questions_or_plan_and_stops_progress(
    pending, questions
):
    assistant, queued, generation = pending
    dialog = assistant.dialog
    before = GraphSnapshot.from_pipeline(assistant.pipeline)
    proposal = reference_proposal(dialog._context)
    proposal["questions"] = questions
    dialog._ready(generation, SimpleNamespace(proposal=proposal, attempts=()))
    assert dialog.stage_tabs.currentIndex() == (1 if questions else 2)
    assert not dialog.request_progress.isVisible()
    assert dialog.prepare_button.text() == "Send goal"
    assert dialog.apply_button.isEnabled() is (not questions)
    assert not dialog.run_button.isEnabled()
    assert dialog.next_button.isVisible() is bool(questions)
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == before
    assert len(queued) == 1
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("ending", ["cancel", "failure", "close", "invalid"])
def test_no_result_does_not_route_or_leave_loading_indicator(pending, ending):
    assistant, queued, generation = pending
    dialog = assistant.dialog
    proposal = reference_proposal(dialog._context)
    if ending == "cancel":
        dialog.cancel_request()
    elif ending == "failure":
        dialog._failed(generation, RuntimeError("Synthetic connection failure"))
    elif ending == "close":
        dialog.close_button.click()
        assert dialog._closed
    else:
        dialog._ready(generation, SimpleNamespace(proposal={}, attempts=()))
    assert not dialog._busy
    assert not dialog.request_progress.isVisible()
    assert dialog.stage_tabs.currentIndex() == 0
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert not dialog._drafts
    assert dialog.next_button.isHidden() and not dialog.next_button.isEnabled()
    dialog._next_stage_clicked()
    assert dialog.stage_tabs.currentIndex() == 0
    if ending in {"cancel", "close"}:
        dialog._ready(generation, SimpleNamespace(proposal=proposal, attempts=()))
        dialog._progress(generation, "Checking plan")
        assert dialog.stage_tabs.currentIndex() == 0
        assert not dialog.request_progress.isVisible()
        assert dialog._prepared is None
    assert len(queued) == 1
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("stale", ["changed_goal", "historical_draft"])
def test_browsing_draft_history_is_not_a_new_send_or_authorization(
    assistant, monkeypatch, stale
):
    dialog = assistant.dialog
    dialog.prepare()
    dialog.goal_edit.setPlainText("A different goal that needs a new request.")
    if stale == "historical_draft":
        dialog.prepare()
        assert len(dialog._drafts) == 2
        dialog.draft_combo.setCurrentIndex(0)
    assert not dialog.apply_button.isEnabled()
    captures = list(assistant.captures)
    for stage in (0, 1, 2, 0):
        dialog.stage_tabs.setCurrentIndex(stage)
    assert dialog.next_button.isVisible()
    assert dialog.next_button.text() == "Next: Discussion"
    _assert_passive_round_trip(assistant, monkeypatch)
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert assistant.captures == captures
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize(
    "questions",
    [[], ["Which observed boundary should remain?\nKeep this exact question."]],
)
def test_returned_draft_back_to_goal_retains_passive_next_navigation(
    assistant, monkeypatch, questions
):
    dialog = assistant.dialog
    goal = "  Keep the literal original goal.\nPreserve this second line and spaces.  "
    dialog.goal_edit.setPlainText(goal)
    dialog.prepare()  # Fixed offline recipe only.
    raw = reference_proposal(dialog._context)
    raw["questions"] = questions
    dialog._accept_proposal(raw)
    assert dialog.stage_tabs.currentIndex() == (1 if questions else 2)
    if questions:
        dialog._question_fields[0][1].setText("  Preserve the observed boundary.  ")
    dialog.discussion_edit.setPlainText(
        "  Keep my unsent follow-up.\nExact second line.  "
    )
    while dialog.stage_tabs.currentIndex() > 0:
        dialog.back_button.click()
    assert dialog.goal_edit.toPlainText() == goal
    assert (
        dialog.next_button.isVisible()
        and dialog.next_button.text() == "Next: Discussion"
    )
    _assert_passive_round_trip(assistant, monkeypatch)
    assert dialog.apply_button.isEnabled() is (not questions)
    assert not dialog.run_button.isEnabled()


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_direct_next_without_any_returned_history_is_guarded(assistant, stage):
    dialog = assistant.dialog
    dialog.stage_tabs.setCurrentIndex(stage)
    before = GraphSnapshot.from_pipeline(assistant.pipeline)
    goal, message = dialog.goal_edit.toPlainText(), dialog.discussion_edit.toPlainText()
    assert not dialog._drafts
    assert dialog.next_button.isHidden() and not dialog.next_button.isEnabled()
    dialog.next_button.clicked.emit()
    dialog._next_stage_clicked()
    assert dialog.stage_tabs.currentIndex() == stage
    assert (
        dialog.goal_edit.toPlainText() == goal
        and dialog.discussion_edit.toPlainText() == message
    )
    assert dialog._context is dialog._prepared is None
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == before
    assert assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize("latest_failed", [False, True])
def test_history_navigation_is_guarded_during_request_and_passive_after_failure(
    pending, monkeypatch, latest_failed
):
    assistant, queued, generation = pending
    dialog = assistant.dialog
    dialog._ready(
        generation,
        SimpleNamespace(proposal=reference_proposal(dialog._context), attempts=()),
    )
    assert len(dialog._drafts) == 1
    dialog.goal_edit.setPlainText(
        "A fresh goal with old checked history kept for review."
    )
    dialog.stage_tabs.setCurrentIndex(0)
    dialog.prepare_button.click()
    assert len(queued) == 2 and dialog._busy
    if latest_failed:
        dialog._failed(
            dialog._generation, RuntimeError("Synthetic latest request failure")
        )
        assert not dialog._busy
        _assert_passive_round_trip(assistant, monkeypatch)
        assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    else:
        before = _navigation_state(dialog)
        captures, assertions = list(assistant.captures), list(assistant.assertions)
        for stage in (0, 1):
            dialog.stage_tabs.setCurrentIndex(stage)
            assert dialog.next_button.isVisible() and not dialog.next_button.isEnabled()
            assert dialog.next_button.text() == (
                "Next: Discussion" if stage == 0 else "View plan"
            )
            dialog.next_button.clicked.emit()
            dialog._next_stage_clicked()
            assert dialog.stage_tabs.currentIndex() == stage
        assert _navigation_state(dialog) == before
        assert assistant.captures == captures and assistant.assertions == assertions
        assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
        dialog.cancel_request()
    assert len(queued) == 2 and assistant.applies == assistant.runs == []
