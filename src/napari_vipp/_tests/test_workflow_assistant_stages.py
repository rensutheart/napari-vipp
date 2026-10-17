"""Stage navigation and explicit settings/Add/Run boundaries, without inference."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QApplication, QDialog, QPlainTextEdit

from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import (
    ProposalDialogue,
    ProposalError,
    capture_proposal_context,
    reference_proposal,
)
from napari_vipp.ui.workflow_assistant import WorkflowAssistantDialog


@pytest.fixture
def assistant(qtbot, monkeypatch):
    """Use detached synthetic data and spies, never provider or graph execution."""
    monkeypatch.setattr(
        WorkflowAssistantDialog, "_refresh_codex_connection", lambda self, *_: None
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    data = np.arange(20 * 24, dtype=np.float32).reshape(20, 24).copy()
    data.flags.writeable = False
    state = image_state_from_array(
        data, axes=(AxisMetadata("y", "space"), AxisMetadata("x", "space"))
    )
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = state
    owner = object()
    captures, assertions, applies, runs = [], [], [], []
    added_ids = ("added-smoothing", "added-threshold", "added-labels")

    def capture(source_id, *, include_images=False):
        captures.append((source_id, include_images))
        return capture_proposal_context(
            pipeline, source_id, include_images=include_images
        ), owner

    def assert_current(context, recorded_owner):
        assertions.append((context, recorded_owner))
        assert recorded_owner is owner
        context.assert_current(pipeline)

    def apply(proposal, recorded_owner):
        applies.append((proposal, recorded_owner))
        return added_ids

    dialog = WorkflowAssistantDialog(
        None,
        sources=lambda: [("input", "Synthetic test image")],
        capture=capture,
        assert_current=assert_current,
        apply=apply,
        run=lambda ids: runs.append(tuple(ids)),
    )
    qtbot.addWidget(dialog)
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("example"))
    # Existing stage/answer contracts exercise conservative clarification.
    # Dedicated initiative tests independently assert the production default.
    dialog.initiative_slider.setValue(0)
    dialog.show()
    return SimpleNamespace(
        dialog=dialog,
        pipeline=pipeline,
        data=data,
        captures=captures,
        assertions=assertions,
        applies=applies,
        runs=runs,
        added_ids=added_ids,
    )


def _prepare_example(assistant):
    dialog = assistant.dialog
    dialog.prepare()
    assert dialog.apply_button.isEnabled()
    assert dialog._prepared is not None
    return dialog._prepared


def _settings_values(dialog):
    return (
        dialog.provider_combo.currentData(),
        dialog.kind_combo.currentIndex(),
        dialog.address_edit.text(),
        dialog.model_combo.currentText(),
        dialog.codex_model_edit.text(),
        dialog.reasoning_combo.currentText(),
        dialog.attempts_spin.value(),
        dialog.timeout_spin.value(),
    )


def _settings_interaction(dialog, qtbot, action, inspect):
    """Drive the nested modal event loop with a guaranteed local dismissal."""
    observed = []

    def interact():
        settings = dialog._settings_dialog
        try:
            observed.append(settings)
            inspect(settings)
            if action == "accept":
                settings.accept_button.click()
            elif action == "escape":
                qtbot.keyClick(settings, Qt.Key_Escape)
            else:
                settings.cancel_button.click()
        finally:
            # A failing assertion must not leave a modal test hanging.
            if settings.isVisible():
                settings.reject()

    QTimer.singleShot(0, interact)
    dialog.settings_button.click()
    assert len(observed) == 1
    return observed[0]


def test_three_named_stages_and_back_next_are_passive(assistant):
    dialog = assistant.dialog
    before = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    assert [dialog.stage_tabs.tabText(index) for index in range(3)] == [
        "1 · Goal",
        "2 · Discussion",
        "3 · Plans",
    ]
    assert dialog.stage_tabs.currentIndex() == 0
    assert not dialog.back_button.isEnabled()
    assert not dialog.run_button.isEnabled()

    assert dialog.next_button.isHidden()
    assert not dialog.next_button.isEnabled()
    for expected in (1, 2):
        dialog.stage_tabs.setCurrentIndex(expected)
        assert dialog.stage_tabs.currentIndex() == expected
    for expected in (1, 0):
        dialog.back_button.click()
        assert dialog.stage_tabs.currentIndex() == expected
    for index in (2, 0, 1, 2):
        dialog.stage_tabs.setCurrentIndex(index)

    assert assistant.captures == []
    assert assistant.assertions == []
    assert assistant.applies == assistant.runs == []
    assert dialog._context is None
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == before
    assert np.array_equal(assistant.data, pixels)
    assert not assistant.data.flags.writeable


def test_navigation_keeps_goal_and_answers_without_sending(assistant, monkeypatch):
    dialog = assistant.dialog
    _prepare_example(assistant)
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Should touching structures be separated?"]
    dialog._accept_proposal(raw)
    question, answer = dialog._question_fields[0]
    answer.setText("Preserve observed boundaries.")
    original_goal = dialog.goal_edit.toPlainText()
    before = len(assistant.captures)
    sends = []
    monkeypatch.setattr(dialog, "_send", lambda: sends.append(True))

    for index in (0, 1, 2, 1, 0):
        dialog.stage_tabs.setCurrentIndex(index)
    assert dialog.next_button.isVisible()
    assert dialog.next_button.text() == "Next: Discussion"
    dialog.next_button.click()
    assert dialog.stage_tabs.currentIndex() == 1
    dialog.next_button.click()
    assert dialog.stage_tabs.currentIndex() == 2
    dialog.back_button.click()

    assert dialog.goal_edit.toPlainText() == original_goal
    assert dialog._question_fields[0][0] == question
    assert dialog._question_fields[0][1].text() == "Preserve observed boundaries."
    assert len(assistant.captures) == before
    assert assistant.applies == assistant.runs == sends == []


def test_discussion_transcript_is_plain_read_only_text(assistant):
    dialog = assistant.dialog
    assert dialog.transcript_edit.isReadOnly()
    assert dialog.discussion_edit is not dialog.transcript_edit
    assert not dialog.discussion_edit.isReadOnly()
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.discussion_edit.setPlainText("Please keep these as observed structures.")
    dialog.stage_tabs.setCurrentIndex(0)
    dialog.stage_tabs.setCurrentIndex(1)
    assert dialog.discussion_edit.toPlainText() == (
        "Please keep these as observed structures."
    )
    assert assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize("action", ["cancel", "escape"])
def test_settings_cancel_and_escape_preserve_active_configuration_and_plan(
    assistant, qtbot, action
):
    dialog = assistant.dialog
    proposal = _prepare_example(assistant)
    before = _settings_values(dialog)
    generation = dialog._generation
    captures = list(assistant.captures)
    original_goal = dialog.goal_edit.toPlainText()
    history = dialog.draft_combo.count()

    def inspect(settings):
        assert settings.isModal()
        assert settings.provider_combo is not dialog.provider_combo
        settings.provider_combo.setCurrentIndex(
            settings.provider_combo.findData("local")
        )
        settings.kind_combo.setCurrentIndex(1)
        settings.address_edit.setText("http://127.0.0.1:4321")
        settings.model_combo.setCurrentText("synthetic-model")
        settings.codex_model_edit.setText("synthetic-codex-model")
        settings.reasoning_combo.setCurrentText("medium")
        settings.attempts_spin.setValue(1)
        settings.timeout_spin.setValue(45)
        assert _settings_values(dialog) == before
        assert dialog._prepared is proposal

    _settings_interaction(dialog, qtbot, action, inspect)
    assert _settings_values(dialog) == before
    assert dialog._prepared is proposal
    assert dialog.apply_button.isEnabled()
    assert dialog._generation == generation
    assert dialog.goal_edit.toPlainText() == original_goal
    assert dialog.draft_combo.count() == history
    assert assistant.captures == captures
    assert assistant.applies == assistant.runs == []


def test_settings_apply_commits_staged_values_without_sending_or_applying(
    assistant, qtbot, monkeypatch
):
    dialog = assistant.dialog
    _prepare_example(assistant)
    captures = list(assistant.captures)
    original_goal = dialog.goal_edit.toPlainText()
    sends = []
    monkeypatch.setattr(dialog, "_send", lambda: sends.append(True))

    def inspect(settings):
        settings.provider_combo.setCurrentIndex(
            settings.provider_combo.findData("local")
        )
        settings.kind_combo.setCurrentIndex(1)
        settings.address_edit.setText("http://127.0.0.1:4321")
        settings.model_combo.setCurrentText("synthetic-model")
        settings.reasoning_combo.setCurrentText("medium")
        settings.attempts_spin.setValue(1)
        settings.timeout_spin.setValue(45)
        assert dialog._provider() == "example"

    _settings_interaction(dialog, qtbot, "accept", inspect)
    assert dialog._provider() == "local"
    assert dialog.kind_combo.currentIndex() == 1
    assert dialog.address_edit.text() == "http://127.0.0.1:4321"
    assert dialog.model_combo.currentText() == "synthetic-model"
    assert dialog.reasoning_combo.currentText() == "medium"
    assert dialog.attempts_spin.value() == 1
    assert dialog.timeout_spin.value() == 45
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert dialog.goal_edit.toPlainText() == original_goal
    assert assistant.captures == captures
    assert assistant.applies == assistant.runs == sends == []


def test_earlier_draft_is_read_only_and_cannot_apply_or_run(assistant):
    dialog = assistant.dialog
    _prepare_example(assistant)
    dialog.goal_edit.setPlainText("Measure the bright observed objects separately.")
    _prepare_example(assistant)
    assert dialog.draft_combo.count() == 2
    assert dialog.draft_combo.currentIndex() == 1
    assert dialog._current_draft_index == 1

    dialog.draft_combo.setCurrentIndex(0)
    assert dialog.json_edit.isReadOnly()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    dialog.apply_proposal()
    dialog.run_button.click()
    assert assistant.applies == assistant.runs == []

    dialog.draft_combo.setCurrentIndex(1)
    assert not dialog.json_edit.isReadOnly()
    assert dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()


def test_add_never_runs_and_explicit_run_receives_only_added_ids(assistant):
    dialog = assistant.dialog
    proposal = _prepare_example(assistant)
    assert not dialog.run_button.isEnabled()
    dialog.apply_button.click()
    assert len(assistant.applies) == 1
    assert assistant.applies[0][0] is proposal
    assert assistant.runs == []
    assert not dialog.apply_button.isEnabled()
    assert dialog.run_button.isEnabled()

    dialog.run_button.click()
    assert assistant.runs == [assistant.added_ids]
    assert len(assistant.applies) == 1


@pytest.mark.parametrize("change", ["goal", "provider", "json"])
def test_invalidation_revokes_run_after_add(assistant, change):
    dialog = assistant.dialog
    _prepare_example(assistant)
    dialog.apply_button.click()
    assert dialog.run_button.isEnabled()
    if change == "goal":
        dialog.goal_edit.setPlainText("A changed analysis objective.")
    elif change == "provider":
        dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    else:
        dialog.json_edit.setPlainText("{invalid")
    assert not dialog.run_button.isEnabled()
    dialog.run_button.click()
    assert assistant.runs == []
    assert len(assistant.applies) == 1


def test_failed_add_cannot_enable_run(assistant, monkeypatch):
    dialog = assistant.dialog
    _prepare_example(assistant)

    def fail(*_):
        raise ProposalError("Synthetic application failure.")

    monkeypatch.setattr(dialog, "_apply", fail)
    dialog.apply_button.click()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert assistant.applies == assistant.runs == []


def test_questions_and_invalid_json_never_enable_add_or_run(assistant, qtbot):
    dialog = assistant.dialog
    _prepare_example(assistant)
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Please confirm the observed target structures."]
    dialog._accept_proposal(raw)
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    dialog.apply_proposal()
    assert assistant.applies == []
    dialog.json_edit.setPlainText(json.dumps({"not": "a proposal"}))
    qtbot.waitUntil(lambda: dialog.status_icon.property("tone") == "error")
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert assistant.applies == assistant.runs == []


def test_only_explicit_send_click_grants_request_permission(assistant, monkeypatch):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")
    sends = []
    monkeypatch.setattr(
        dialog, "_send", lambda: sends.append(dialog.consent_checkbox.isChecked())
    )
    assert not dialog.consent_checkbox.isChecked()
    assert dialog.consent_checkbox.isHidden()
    dialog.prepare()
    assert assistant.captures == sends == []
    dialog.prepare_button.click()
    assert sends == [True]
    assert len(assistant.captures) == 1
    dialog.goal_edit.setPlainText("Refine the bright-object starting point.")
    assert not dialog.consent_checkbox.isChecked()
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.stage_tabs.setCurrentIndex(2)
    assert sends == [True]
    dialog.prepare_button.click()
    assert sends == [True, True]
    assert len(assistant.captures) == 2
    assert assistant.applies == assistant.runs == []


def test_free_send_preserves_exact_original_goal_and_prior_answers(
    assistant, monkeypatch
):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")
    original_goal = "  Segment the observed wavy structure and central sphere.  "
    dialog.goal_edit.setPlainText(original_goal)
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Which observed boundaries should remain separate?"]
    dialog._accept_proposal(raw)
    dialog._clarifications = [("Earlier question", "  exact earlier answer  ")]
    question, answer = dialog._question_fields[0]
    answer.setText("  Keep the observed boundary; do not reconstruct gaps.  ")
    message = "  Use different colours, and keep the existing calibration.  "
    dialog.discussion_edit.setPlainText(message)
    requests = []

    def send():
        requests.append(dialog._request_objective)
        dialog._busy = True
        dialog._job_kind = "proposal"
        dialog._sync_buttons()

    monkeypatch.setattr(dialog, "_send", send)
    assert dialog.send_button.isEnabled()
    dialog.send_button.click()

    assert len(requests) == 1
    dialogue = requests[0]
    assert isinstance(dialogue, ProposalDialogue)
    assert dialogue.objective == original_goal
    assert dialogue.clarifications == (
        ("Earlier question", "  exact earlier answer  "),
        (question, "  Keep the observed boundary; do not reconstruct gaps.  "),
        ("Additional instructions 1", message),
    )
    assert dialog.goal_edit.toPlainText() == original_goal
    assert dialog._pending_clarifications == list(dialogue.clarifications)
    public = json.loads(dialog.request_edit.toPlainText())
    assert public["objective"] == original_goal
    assert public["clarifications"] == [
        {"question": q, "answer": a} for q, a in dialogue.clarifications
    ]
    assert ("You", f"Additional instructions 1\n{message}") in dialog._chat
    # Rich-text layout may collapse surrounding display whitespace; the request
    # and retained conversation text above must preserve the exact content.
    assert message.strip() in dialog.transcript_edit.toPlainText()
    assert dialog.discussion_edit.toPlainText() == message
    assert assistant.applies == assistant.runs == []


def test_successive_free_messages_remain_separate_from_original_goal(
    assistant, monkeypatch
):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")
    original_goal = "Compare the observed bright structures."
    dialog.goal_edit.setPlainText(original_goal)
    dialog._clarifications = [("Confirmed target", "The two observed structures.")]
    requests = []

    def send():
        requests.append(dialog._request_objective)
        dialog._busy = True
        dialog._job_kind = "proposal"
        dialog._sync_buttons()

    monkeypatch.setattr(dialog, "_send", send)
    for message in ("Keep physical units.", "Use different object colours."):
        dialog.discussion_edit.setPlainText(message)
        dialog.send_button.click()
        assert dialog._busy
        response = SimpleNamespace(
            proposal=reference_proposal(dialog._context), attempts=()
        )
        dialog._ready(dialog._generation, response)
        assert dialog.apply_button.isEnabled()

    assert [request.objective for request in requests] == [original_goal] * 2
    assert requests[1].clarifications == (
        ("Confirmed target", "The two observed structures."),
        ("Additional instructions 1", "Keep physical units."),
        ("Additional instructions 2", "Use different object colours."),
    )
    assert dialog._clarifications == list(requests[1].clarifications)
    assert dialog.goal_edit.toPlainText() == original_goal
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("action", ["cancel", "escape"])
def test_settings_signin_uses_staged_selection_without_mutating_active_plan(
    assistant, qtbot, monkeypatch, action
):
    dialog = assistant.dialog
    proposal = _prepare_example(assistant)
    original_settings = _settings_values(dialog)
    original_generation = dialog._generation
    captures = list(assistant.captures)
    jobs = []

    def start_connection(kind, call, cancelled=None):
        # Retain but never execute the login closure: no browser or auth request.
        jobs.append((kind, call, cancelled))
        dialog._connection_busy = True
        dialog._connection_kind = kind
        if cancelled is not None:
            dialog._connection_cancelled = cancelled
        dialog._sync_buttons()

    monkeypatch.setattr(dialog, "_start_connection", start_connection)

    def inspect(settings):
        settings.provider_combo.setCurrentIndex(
            settings.provider_combo.findData("codex")
        )
        settings.signin_button.click()
        assert len(jobs) == 1
        assert jobs[0][0] == "signin"
        assert callable(jobs[0][1])
        assert _settings_values(dialog) == original_settings
        assert dialog._prepared is proposal
        assert dialog._provider() == "example"
        assert not settings.accept_button.isEnabled()

    _settings_interaction(dialog, qtbot, action, inspect)
    assert _settings_values(dialog) == original_settings
    assert dialog._prepared is proposal
    assert dialog.apply_button.isEnabled()
    assert dialog._generation == original_generation
    assert not dialog._connection_busy
    assert jobs[0][2].is_set()
    assert assistant.captures == captures
    assert assistant.applies == assistant.runs == []


def test_close_reopen_retains_text_and_drafts_but_revokes_all_authority(assistant):
    dialog = assistant.dialog
    _prepare_example(assistant)
    dialog._append_chat("You", "Observed signal only; preserve the missing region.")
    dialog.discussion_edit.setPlainText("An unsent follow-up remains here.")
    original_goal = dialog.goal_edit.toPlainText()
    original_transcript = dialog.transcript_edit.toPlainText()
    original_json = dialog.json_edit.toPlainText()
    original_drafts = tuple(dialog._drafts)
    original_captures = list(assistant.captures)

    dialog.close()
    assert dialog._closed
    assert dialog._context is dialog._owner is dialog._prepared is None
    assert not dialog._draft_authorized
    assert dialog._added_ids == ()
    assert not dialog.consent_checkbox.isChecked()
    assert dialog.request_edit.toPlainText() == ""
    assert dialog.trace_edit.toPlainText() == ""
    dialog.show()

    assert not dialog._closed
    assert dialog.goal_edit.toPlainText() == original_goal
    assert dialog.discussion_edit.toPlainText() == "An unsent follow-up remains here."
    assert dialog.transcript_edit.toPlainText() == original_transcript
    assert tuple(dialog._drafts) == original_drafts
    assert dialog.json_edit.toPlainText() == original_json
    assert dialog.json_edit.isReadOnly()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    dialog.apply_proposal()
    dialog._run_added_plan()
    assert assistant.captures == original_captures
    assert assistant.applies == assistant.runs == []


def test_close_reopen_does_not_reauthorize_added_plan_or_late_response(assistant):
    dialog = assistant.dialog
    _prepare_example(assistant)
    response = SimpleNamespace(
        proposal=reference_proposal(dialog._context), attempts=()
    )
    dialog.apply_button.click()
    assert dialog.run_button.isEnabled()
    old_generation = dialog._generation
    original_transcript = dialog.transcript_edit.toPlainText()
    dialog.close()
    dialog.show()
    dialog._ready(old_generation, response)

    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert dialog._context is dialog._owner is dialog._prepared is None
    assert not dialog._draft_authorized
    assert dialog._added_ids == ()
    assert dialog.transcript_edit.toPlainText() == original_transcript
    dialog._run_added_plan()
    assert len(assistant.applies) == 1
    assert assistant.runs == []


def test_historical_draft_selection_preserves_current_question_answer_fields(
    assistant,
):
    dialog = assistant.dialog
    _prepare_example(assistant)
    raw = reference_proposal(dialog._context)
    raw["questions"] = [
        "Which observed structure should be separate?",
        "Should the unobserved region remain unfilled?",
    ]
    dialog._accept_proposal(raw)
    assert dialog.draft_combo.count() == 2
    latest = dialog._current_draft_index
    fields = tuple(dialog._question_fields)
    answers = ("The central sphere.", "Yes—preserve the acquired boundary.")
    for (_, field), answer in zip(fields, answers, strict=True):
        field.setText(answer)

    for index in (0, latest, 0, latest):
        dialog.draft_combo.setCurrentIndex(index)
        assert tuple(dialog._question_fields) == fields
        assert tuple(field.text() for _, field in fields) == answers
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize(
    ("speaker", "text"),
    [
        (
            '<a href="https://example.invalid">Speaker</a>',
            '<script>alert("not executable")</script>\n'
            '<img src="file:///not-a-real-private-image.png">',
        ),
        (
            "Assistant",
            "<b>Literal model text</b> & symbols\n"
            '<a href="https://example.invalid">Not an active link</a>',
        ),
    ],
)
def test_transcript_displays_provider_and_user_markup_literally(
    assistant, speaker, text
):
    dialog = assistant.dialog
    dialog._append_chat(speaker, text)
    rendered = dialog.transcript_edit.toPlainText()
    assert speaker in rendered
    assert text in rendered
    assert dialog.transcript_edit.isReadOnly()
    assert not dialog.transcript_edit.openExternalLinks()
    document = dialog.transcript_edit.document()
    block = document.begin()
    while block.isValid():
        fragment_iterator = block.begin()
        while not fragment_iterator.atEnd():
            fragment = fragment_iterator.fragment()
            if fragment.isValid():
                assert not fragment.charFormat().isAnchor()
                assert not fragment.charFormat().isImageFormat()
            fragment_iterator += 1
        block = block.next()
    assert assistant.captures == assistant.applies == assistant.runs == []


def test_fresh_incoming_plan_becomes_latest_when_history_is_selected(assistant):
    dialog = assistant.dialog
    _prepare_example(assistant)
    second = reference_proposal(dialog._context)
    second["summary"] = "Second checked synthetic draft."
    dialog._accept_proposal(second)
    assert dialog.draft_combo.count() == 2
    previous_drafts = tuple(dialog._drafts)
    captures = list(assistant.captures)
    dialog.draft_combo.setCurrentIndex(0)
    assert dialog.json_edit.isReadOnly()
    assert not dialog.apply_button.isEnabled()

    incoming = reference_proposal(dialog._context)
    incoming["summary"] = "Newly returned third synthetic draft."
    dialog._accept_proposal(incoming)

    assert dialog.draft_combo.count() == 3
    assert dialog.draft_combo.currentIndex() == dialog._current_draft_index == 2
    assert tuple(dialog._drafts[:2]) == previous_drafts
    assert dialog._prepared.summary == incoming["summary"]
    assert dialog.summary_label.text() == incoming["summary"]
    assert not dialog.json_edit.isReadOnly()
    assert dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert dialog.stage_tabs.currentIndex() == 2
    assert assistant.captures == captures
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("provider", ["local", "example"])
def test_completed_staged_signin_checks_codex_with_another_provider_active(
    assistant, monkeypatch, provider
):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData(provider))
    original_settings = _settings_values(dialog)
    calls, polls, terminations = [], [], []
    process = SimpleNamespace(
        poll=lambda: polls.append(True) or 0,
        terminate=lambda: terminations.append(True),
    )
    dialog._signin_process = process
    dialog._login_holder = {"process": process}
    dialog._connection_busy = True
    dialog._connection_kind = "signin"
    dialog._signin_elapsed.start()
    monkeypatch.setattr(
        dialog, "_check_codex_connection", lambda: calls.append(dialog._provider())
    )

    dialog._poll_signin()
    dialog._poll_signin()

    assert calls == [provider]
    assert polls == [True]
    assert terminations == []
    assert dialog._signin_process is None
    assert dialog._login_holder == {}
    assert not dialog._connection_busy
    assert not dialog._signin_timer.isActive()
    assert _settings_values(dialog) == original_settings
    assert assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_failed_or_cancelled_free_send_keeps_message_for_explicit_retry(
    assistant, monkeypatch, outcome
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalError

    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")
    original_goal = "Keep the observed objects separate."
    dialog.goal_edit.setPlainText(original_goal)
    previous_answers = [("Observed structures?", "The wave and central sphere.")]
    dialog._clarifications = list(previous_answers)
    message = "  Keep this follow-up until the request succeeds.  "
    dialog.discussion_edit.setPlainText(message)
    requests = []

    def send():
        requests.append(dialog._request_objective)
        dialog._generation += 1
        dialog._busy = True
        dialog._job_kind = "proposal"
        dialog._draft_authorized = False
        dialog._sync_buttons()

    monkeypatch.setattr(dialog, "_send", send)
    dialog.send_button.click()
    generation = dialog._generation
    assert dialog._busy
    assert dialog.discussion_edit.toPlainText() == message
    if outcome == "failed":
        dialog._failed(generation, ModelProposalError("Synthetic request failure."))
    else:
        dialog.cancel_request()
        dialog._ready(
            generation,
            SimpleNamespace(proposal=reference_proposal(dialog._context), attempts=()),
        )

    assert not dialog._busy
    assert dialog.discussion_edit.toPlainText() == message
    assert dialog.goal_edit.toPlainText() == original_goal
    assert dialog._clarifications == previous_answers
    assert dialog._pending_clarifications is None
    assert len(requests) == 1
    assert dialog.send_button.isEnabled()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()

    dialog.send_button.click()
    assert len(requests) == 2
    assert requests[1] == requests[0]
    assert dialog.discussion_edit.toPlainText() == message
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("next_action", ["goal", "answers"])
def test_successful_goal_or_answers_does_not_erase_previous_unsent_message(
    assistant, monkeypatch, next_action
):
    from napari_vipp.core.workflow_proposal_model import ModelProposalError

    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Should unobserved gaps remain unfilled?"]
    dialog._accept_proposal(raw)
    dialog._question_fields[0][1].setText("Yes, preserve the observed boundary.")
    message = "  Keep my unsent follow-up for an intentional message retry.  "
    dialog.discussion_edit.setPlainText(message)
    requests = []

    def send():
        requests.append(dialog._request_objective)
        dialog._generation += 1
        dialog._busy = True
        dialog._job_kind = "proposal"
        dialog._draft_authorized = False
        dialog._sync_buttons()

    monkeypatch.setattr(dialog, "_send", send)
    dialog.send_button.click()
    dialog._failed(
        dialog._generation, ModelProposalError("Synthetic message request failure.")
    )
    assert dialog.discussion_edit.toPlainText() == message
    assert len(requests) == 1

    if next_action == "goal":
        dialog.prepare_button.click()
    else:
        dialog.revise_button.click()
    assert len(requests) == 2
    assert isinstance(requests[1], ProposalDialogue)
    assert requests[1].clarifications == (
        (
            "Should unobserved gaps remain unfilled?",
            "Yes, preserve the observed boundary.",
        ),
    )
    assert dialog._submitted_message == ""
    assert dialog.discussion_edit.toPlainText() == message
    dialog._ready(
        dialog._generation,
        SimpleNamespace(proposal=reference_proposal(dialog._context), attempts=()),
    )
    assert dialog.apply_button.isEnabled()
    assert dialog.discussion_edit.toPlainText() == message
    assert dialog._submitted_message == ""
    assert assistant.applies == assistant.runs == []


def test_discussion_preview_includes_unsent_message_without_send_or_recapture(
    assistant, qtbot, monkeypatch
):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")
    original_goal = "Review the observed objects independently."
    dialog.goal_edit.setPlainText(original_goal)
    previous_answers = [("Observed targets", "Wave and central sphere.")]
    dialog._clarifications = list(previous_answers)
    message = "  Keep physical units and use separate colours.  "
    dialog.discussion_edit.setPlainText(message)
    dialog.stage_tabs.setCurrentIndex(1)
    requests, sends = [], []
    monkeypatch.setattr(dialog, "_send", lambda: sends.append(True))

    def inspect_preview():
        preview = QApplication.activeModalWidget()
        try:
            details = preview.findChild(QPlainTextEdit, "SharedRequestJSON")
            assert details is not None
            assert details.isReadOnly()
            requests.append(json.loads(details.toPlainText()))
        finally:
            preview.reject()

    for _ in range(2):
        QTimer.singleShot(0, inspect_preview)
        dialog.preview_button.click()

    assert len(requests) == 2
    assert requests[0] == requests[1]
    assert requests[0]["objective"] == original_goal
    assert requests[0]["clarifications"] == [
        {"question": q, "answer": a}
        for q, a in [*previous_answers, ("Additional instructions 1", message)]
    ]
    assert assistant.captures == [("input", False)]
    assert dialog._clarifications == previous_answers
    assert dialog.discussion_edit.toPlainText() == message
    assert dialog._extra_message == dialog._submitted_message == ""
    assert dialog._pending_clarifications is None
    assert not dialog.consent_checkbox.isChecked()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert sends == assistant.applies == assistant.runs == []


def test_enter_in_question_field_cannot_trigger_an_implicit_send(
    assistant, qtbot, monkeypatch
):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")
    dialog._ensure_context()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["Confirm the observed boundary."]
    dialog._accept_proposal(raw)
    field = dialog._question_fields[0][1]
    field.setText("Preserve it.")
    captures = list(assistant.captures)
    sends = []
    monkeypatch.setattr(dialog, "_send", lambda: sends.append(True))
    for button in (
        dialog.prepare_button,
        dialog.revise_button,
        dialog.send_button,
        dialog.apply_button,
        dialog.run_button,
        dialog.settings_button,
        dialog.back_button,
        dialog.next_button,
    ):
        assert not button.autoDefault(), button.text()
        assert not button.isDefault(), button.text()

    field.setFocus()
    qtbot.keyClick(field, Qt.Key_Return)
    assert dialog.isVisible()
    assert field.text() == "Preserve it."
    assert assistant.captures == captures
    assert sends == assistant.applies == assistant.runs == []


def test_escape_cancels_pending_request_and_ignores_reply_after_reopen(
    assistant, qtbot, monkeypatch
):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")
    original_goal = "Keep the observed structures independent."
    dialog.goal_edit.setPlainText(original_goal)
    message = "An unsent draft must remain available after Escape."
    dialog.discussion_edit.setPlainText(message)
    sends = []

    def send():
        sends.append(dialog._request_objective)
        dialog._generation += 1
        dialog._busy = True
        dialog._job_kind = "proposal"
        dialog._draft_authorized = False
        dialog._sync_buttons()

    monkeypatch.setattr(dialog, "_send", send)
    dialog.send_button.click()
    assert dialog._busy
    assert dialog._pending_clarifications
    generation = dialog._generation
    response = SimpleNamespace(
        proposal=reference_proposal(dialog._context), attempts=()
    )
    transcript = dialog.transcript_edit.toPlainText()
    captures = list(assistant.captures)

    qtbot.keyClick(dialog, Qt.Key_Escape)

    assert dialog._closed
    assert not dialog.isVisible()
    assert not dialog._busy
    assert dialog._cancelled.is_set()
    assert dialog._generation > generation
    assert dialog._context is dialog._owner is dialog._prepared is None
    assert dialog._pending_clarifications is None
    assert dialog._submitted_message == ""
    assert not dialog._draft_authorized
    assert not dialog.consent_checkbox.isChecked()
    assert dialog.request_edit.toPlainText() == ""
    assert dialog.trace_edit.toPlainText() == ""
    dialog._ready(generation, response)
    dialog.show()
    assert not dialog._closed
    dialog._ready(generation, response)

    assert dialog.goal_edit.toPlainText() == original_goal
    assert dialog.discussion_edit.toPlainText() == message
    assert dialog.transcript_edit.toPlainText() == transcript
    assert dialog._context is dialog._owner is dialog._prepared is None
    assert not dialog._draft_authorized
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert assistant.captures == captures
    assert len(sends) == 1
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("dismiss", ["escape", "accept", "done"])
def test_every_dialog_dismissal_revokes_added_plan_but_keeps_text(
    assistant, qtbot, dismiss
):
    dialog = assistant.dialog
    _prepare_example(assistant)
    dialog.apply_button.click()
    assert dialog.run_button.isEnabled()
    dialog.discussion_edit.setPlainText("Preserve this unsent follow-up.")
    goal = dialog.goal_edit.toPlainText()
    transcript = dialog.transcript_edit.toPlainText()
    proposal_text = dialog.json_edit.toPlainText()
    drafts = tuple(dialog._drafts)
    captures = list(assistant.captures)
    before_generation = dialog._generation

    if dismiss == "escape":
        qtbot.keyClick(dialog, Qt.Key_Escape)
    elif dismiss == "accept":
        dialog.accept()
    else:
        dialog.done(QDialog.Accepted)

    assert dialog._closed
    assert not dialog.isVisible()
    assert dialog._generation > before_generation
    assert dialog._context is dialog._owner is dialog._prepared is None
    assert dialog._added_ids == ()
    assert not dialog._draft_authorized
    assert not dialog.consent_checkbox.isChecked()
    after_generation = dialog._generation
    dialog.reject()
    assert dialog._generation == after_generation
    dialog.show()

    assert not dialog._closed
    assert dialog.goal_edit.toPlainText() == goal
    assert dialog.discussion_edit.toPlainText() == "Preserve this unsent follow-up."
    assert dialog.transcript_edit.toPlainText() == transcript
    assert tuple(dialog._drafts) == drafts
    assert dialog.json_edit.toPlainText() == proposal_text
    assert dialog.json_edit.isReadOnly()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    dialog.apply_proposal()
    dialog._run_added_plan()
    assert assistant.captures == captures
    assert len(assistant.applies) == 1
    assert assistant.runs == []
