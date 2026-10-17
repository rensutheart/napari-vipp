"""Client-chosen initiative changes drafting, never sharing or graph authority."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPalette, QTextCursor, QTextDocument
from qtpy.QtWidgets import QLabel

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import ProposalError, reference_proposal
from napari_vipp.ui.workflow_assistant import (
    _INITIATIVE_CHOICES,
    WorkflowAssistantDialog,
)


def _questions(dialog, questions=("Which boundary should remain separate?",)):
    dialog.prepare()
    raw = reference_proposal(dialog._context)
    raw["questions"] = list(questions)
    dialog._accept_proposal(raw)
    assert dialog._prepared is not None
    assert not dialog.apply_button.isEnabled()


def _local(dialog):
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("synthetic-model")


def _initiative_document(dialog):
    explanation = dialog.initiative_explanation
    assert explanation.textFormat() == Qt.RichText
    document = QTextDocument()
    document.setDefaultFont(explanation.font())
    document.setHtml(explanation.text())
    document.setTextWidth(max(1, explanation.width()))
    return document


def _assert_inline_choice(dialog, position):
    _, label, explanation = _INITIATIVE_CHOICES[position]
    assert dialog._initiative_copy() == (label, explanation)
    assert not hasattr(dialog, "initiative_label")
    assert dialog.initiative_card.findChild(QLabel, "AssistantInitiativeChoice") is None
    assert (
        dialog.initiative_explanation.objectName() == "AssistantInitiativeDescription"
    )
    document = _initiative_document(dialog)
    plain = document.toPlainText()
    assert plain.startswith(label) and plain.endswith(explanation)
    mode_cursor = document.find(label)
    body_cursor = document.find(explanation)
    assert not mode_cursor.isNull() and not body_cursor.isNull()
    assert mode_cursor.charFormat().fontWeight() >= QFont.Bold
    assert body_cursor.charFormat().fontWeight() < QFont.Bold
    assert dialog.initiative_slider.accessibleName() == "Assistant initiative"
    assert dialog.initiative_slider.accessibleDescription() == f"{label}. {explanation}"
    assert dialog.initiative_slider.toolTip() == explanation
    assert "<b>" not in dialog.initiative_slider.accessibleDescription()
    assert dialog.initiative_ask_label.text() == "Ask me more"
    assert dialog.initiative_best_label.text() == "Use best judgment"
    return explanation


def test_new_window_defaults_to_best_judgment_without_sending(assistant, qtbot):
    original = assistant.dialog
    dialog = WorkflowAssistantDialog(
        None,
        sources=lambda: [("input", "Synthetic test image")],
        capture=original._capture,
        assert_current=original._assert_current,
        apply=original._apply,
        run=original._run,
    )
    qtbot.addWidget(dialog)
    assert dialog.initiative_slider.value() == 2
    _assert_inline_choice(dialog, 2)
    assert dialog._planning_preferences().max_questions == 1
    assert not dialog.images_checkbox.isChecked()
    assert not dialog.consent_checkbox.isChecked()
    assert assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize(
    ("position", "mode", "label", "cap"),
    [
        (0, "ask_first", "Ask first", 3),
        (1, "balanced", "Balanced", 2),
        (2, "use_best_judgment", "Use best judgment", 1),
    ],
)
def test_slider_sets_typed_immutable_request_and_local_preview(
    assistant, position, mode, label, cap
):
    dialog = assistant.dialog
    dialog.initiative_slider.setValue(position)
    explanation = _assert_inline_choice(dialog, position)
    assert label in dialog.initiative_slider.accessibleDescription()
    request = dialog._ensure_context()
    assert request["planning_preferences"]["mode"] == mode
    assert request["planning_preferences"]["max_questions"] == cap
    assert dialog._context.planning_preferences.mode == mode
    assert json.loads(dialog.request_edit.toPlainText()) == request
    preview = dialog._sharing_preview_dialog(request)
    try:
        policy = preview.findChild(QLabel, "SharedPlanningPreferences")
        assert policy.textFormat() == Qt.PlainText
        assert policy.text() == f"Assistant initiative\n{label}\n{explanation}"
        assert "<b>" not in policy.text()
        assert "<b>" not in json.dumps(request["planning_preferences"])
        assert assistant.applies == assistant.runs == []
    finally:
        preview.close()


@pytest.mark.parametrize("position", [0, 1, 2])
def test_inline_choice_palette_font_and_resize_are_presentation_only(
    assistant, qtbot, monkeypatch, position
):
    dialog = assistant.dialog
    dialog.initiative_slider.setValue(position)
    goal = "  Preserve the literal observed goal.\nKeep its second line and spaces.  "
    message = "  Preserve my unsent message.\nKeep this line and spaces too.  "
    dialog.goal_edit.setPlainText(goal)
    dialog.prepare()  # Fixed offline example; no model or scientific calculation.
    dialog.discussion_edit.setPlainText(message)
    for editor in (dialog.goal_edit, dialog.discussion_edit):
        cursor = editor.textCursor()
        cursor.setPosition(3)
        cursor.setPosition(24, QTextCursor.KeepAnchor)
        editor.setTextCursor(cursor)
    actions = []
    for method in ("_send", "apply_proposal", "_run_added_plan"):
        monkeypatch.setattr(dialog, method, lambda: actions.append("action"))
    context, owner, prepared = dialog._context, dialog._owner, dialog._prepared
    objective = dialog._request_objective
    payload = context.public_request(objective)
    request, proposal = (
        dialog.request_edit.toPlainText(),
        dialog.json_edit.toPlainText(),
    )
    chat, drafts = tuple(dialog._chat), tuple(dialog._drafts)
    cursors = tuple(
        (
            editor.textCursor().position(),
            editor.textCursor().anchor(),
            editor.textCursor().selectedText(),
        )
        for editor in (dialog.goal_edit, dialog.discussion_edit)
    )
    permissions = (
        dialog._generation,
        dialog._draft_authorized,
        dialog._added_ids,
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
        dialog.images_checkbox.isChecked(),
        dialog.consent_checkbox.isChecked(),
        dialog._planning_preferences(),
        dialog.provider_combo.currentData(),
    )
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
    for dark, width, points in ((True, 760, 16), (False, 1040, 12), (True, 760, 12)):
        palette = QPalette(dialog._theme_probe.palette())
        surface, text = ("#202632", "#e9edf4") if dark else ("#f4f1ee", "#202126")
        for role in (QPalette.Window, QPalette.Base, QPalette.Button):
            palette.setColor(role, QColor(surface))
        for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
            palette.setColor(role, QColor(text))
        dialog._theme_probe.setPalette(palette)
        font = QFont(dialog.font())
        font.setPointSize(points)
        dialog.setFont(font)
        dialog.resize(width, 800)
        dialog._refresh_appearance()
        qtbot.waitUntil(lambda: not dialog._appearance_timer.isActive())
        qtbot.waitUntil(lambda: not dialog.goal_edit._height_timer.isActive())
        qtbot.waitUntil(lambda: not dialog.discussion_edit._height_timer.isActive())
        _assert_inline_choice(dialog, position)
    assert (
        dialog._context is context
        and dialog._owner is owner
        and dialog._prepared is prepared
    )
    assert (
        dialog._request_objective == objective
        and context.public_request(objective) == payload
    )
    assert (
        dialog.request_edit.toPlainText() == request
        and dialog.json_edit.toPlainText() == proposal
    )
    assert (
        dialog.goal_edit.toPlainText() == goal
        and dialog.discussion_edit.toPlainText() == message
    )
    assert tuple(dialog._chat) == chat and tuple(dialog._drafts) == drafts
    assert (
        tuple(
            (
                editor.textCursor().position(),
                editor.textCursor().anchor(),
                editor.textCursor().selectedText(),
            )
            for editor in (dialog.goal_edit, dialog.discussion_edit)
        )
        == cursors
    )
    assert (
        dialog._generation,
        dialog._draft_authorized,
        dialog._added_ids,
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
        dialog.images_checkbox.isChecked(),
        dialog.consent_checkbox.isChecked(),
        dialog._planning_preferences(),
        dialog.provider_combo.currentData(),
    ) == permissions
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert not assistant.data.flags.writeable
    assert actions == assistant.applies == assistant.runs == []


def test_keyboard_slider_change_is_passive(assistant, qtbot):
    dialog = assistant.dialog
    before = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    qtbot.keyClick(dialog.initiative_slider, Qt.Key_Right)
    assert dialog.initiative_slider.value() == 1
    assert assistant.captures == assistant.applies == assistant.runs == []
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == before
    np.testing.assert_array_equal(assistant.data, pixels)


def test_policy_change_retains_exact_goal_answers_chat_and_drafts(assistant):
    dialog = assistant.dialog
    goal = "  Separate the circular and wavy structures in this 3D stack.  "
    dialog.goal_edit.setPlainText(goal)
    _questions(dialog, ("Which boundary?", "Which interior convention?"))
    field = dialog._question_fields[0][1]
    field.setText("  two objects; use your best judgment  ")
    dialog._clarifications = [("An earlier question?", "  I don't mind.  ")]
    dialog.discussion_edit.setPlainText("  Keep my unsent message.  ")
    old_context, old_owner = dialog._context, dialog._owner
    old_chat, old_drafts = tuple(dialog._chat), tuple(dialog._drafts)
    old_captures = list(assistant.captures)
    dialog.initiative_slider.setValue(2)
    assert dialog.goal_edit.toPlainText() == goal
    assert dialog._question_fields[0] == ("Which boundary?", field)
    assert field.text() == "  two objects; use your best judgment  "
    assert dialog._clarifications == [("An earlier question?", "  I don't mind.  ")]
    assert dialog.discussion_edit.toPlainText() == "  Keep my unsent message.  "
    assert tuple(dialog._chat) == old_chat and tuple(dialog._drafts) == old_drafts
    assert dialog._context is dialog._owner is None
    assert dialog._image_draft_guard[1] is old_owner
    assert old_context.planning_preferences.mode == "ask_first"
    assert not dialog._draft_authorized
    assert dialog._prepared is None
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert not dialog.consent_checkbox.isChecked()
    assert assistant.captures == old_captures
    request = dialog._ensure_context(dialog._answered_objective())
    assert request["objective"] == goal
    assert request["planning_preferences"]["mode"] == "use_best_judgment"
    assert request["clarifications"][-1]["answer"] == field.text()
    assert request["deferred_questions"] == ["Which interior convention?"]
    assert dialog._context.context_id != old_context.context_id


@pytest.mark.parametrize("position", [0, 1, 2])
def test_blank_answers_depend_on_policy_and_never_clear_questions_locally(
    assistant, monkeypatch, position
):
    dialog = assistant.dialog
    _local(dialog)
    dialog.initiative_slider.setValue(position)
    dialog.consent_checkbox.setChecked(True)
    monkeypatch.setattr(dialog, "_send", lambda: None)
    dialog.prepare()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["A genuinely essential boundary question?"]
    dialog._accept_proposal(raw)
    old_questions = tuple(dialog._question_fields)
    sends = []
    monkeypatch.setattr(
        dialog,
        "_send",
        lambda: sends.append(dialog._context.public_request(dialog._request_objective)),
    )
    assert dialog.revise_button.isEnabled() is (position == 2)
    dialog.discussion_edit.setPlainText("Use your judgment and make a starting draft.")
    assert dialog.send_button.isEnabled() is (position == 2)
    assert not dialog.apply_button.isEnabled()
    assert dialog._prepared.questions == ("A genuinely essential boundary question?",)
    dialog.revise_button.click()
    assert len(sends) == (1 if position == 2 else 0)
    if sends:
        request = sends[0]
        assert request["deferred_questions"] == [old_questions[0][0]]
        assert "clarifications" not in request
        assert not dialog._question_fields[0][1].text()
        assert "best judgment" in dialog._question_fields[0][1].placeholderText()
        assert dialog._chat[-1][0] == "You · Preference"
    assert tuple(dialog._question_fields) == old_questions
    assert assistant.applies == assistant.runs == []


def test_best_judgment_message_keeps_original_and_literal_answers(
    assistant, monkeypatch
):
    dialog = assistant.dialog
    _local(dialog)
    dialog.consent_checkbox.setChecked(True)
    monkeypatch.setattr(dialog, "_send", lambda: None)
    dialog.prepare()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["One choice?", "Another choice?"]
    dialog._accept_proposal(raw)
    dialog._question_fields[0][1].setText("  use your judgment  ")
    goal = dialog.goal_edit.toPlainText()
    dialog.initiative_slider.setValue(2)
    message = "  I don't know the other choice; please draft it.  "
    dialog.discussion_edit.setPlainText(message)
    sends = []
    monkeypatch.setattr(
        dialog,
        "_send",
        lambda: sends.append(dialog._context.public_request(dialog._request_objective)),
    )
    dialog.send_button.click()
    assert len(sends) == 1
    assert sends[0]["objective"] == goal
    assert sends[0]["clarifications"] == [
        {"question": "One choice?", "answer": "  use your judgment  "},
        {"question": "Additional instructions 1", "answer": message},
    ]
    assert sends[0]["deferred_questions"] == ["Another choice?"]
    assert assistant.applies == assistant.runs == []


def test_best_judgment_still_rejects_excess_questions(assistant):
    dialog = assistant.dialog
    dialog.initiative_slider.setValue(2)
    dialog.prepare()
    raw = reference_proposal(dialog._context)
    raw["questions"] = ["One question?", "Another question?"]
    dialog._accept_proposal(raw)
    assert dialog._prepared is None
    assert dialog.status_icon.property("tone") == "error"
    assert not dialog.apply_button.isEnabled()


def test_policy_change_revokes_existing_run_authority(assistant):
    dialog = assistant.dialog
    dialog.prepare()
    dialog.apply_button.click()
    assert dialog.run_button.isEnabled()
    dialog.initiative_slider.setValue(2)
    assert not dialog.run_button.isEnabled()
    dialog.run_button.click()
    assert assistant.runs == []


def test_policy_change_cancels_pending_generation_without_clearing_text(assistant):
    dialog = assistant.dialog
    _questions(dialog)
    field = dialog._question_fields[0][1]
    field.setText("Keep this answer.")
    cancelled = threading.Event()
    dialog._cancelled = cancelled
    dialog._busy = True
    dialog._job_kind = "proposal"
    generation = dialog._generation
    dialog.initiative_slider.setValue(2)  # Programmatic changes also revoke authority.
    assert cancelled.is_set()
    assert dialog._generation > generation
    assert not dialog._busy
    assert field.text() == "Keep this answer."
    assert dialog._pending_clarifications is None


def test_retained_discussion_rejects_changed_source_before_revision(assistant):
    dialog = assistant.dialog
    _questions(dialog)
    dialog.initiative_slider.setValue(2)
    changed = assistant.data.copy()
    changed[0, 0] += 1
    changed.flags.writeable = False
    assistant.pipeline.outputs["input"] = changed
    with pytest.raises(ProposalError, match="input or workflow changed"):
        dialog._ensure_context(dialog._answered_objective())
    assert not dialog.apply_button.isEnabled()
    assert assistant.applies == assistant.runs == []
