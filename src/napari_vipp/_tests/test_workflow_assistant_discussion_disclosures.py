"""Passive disclosures retain complete drafts and avoid duplicate live questions."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QIcon, QPalette
from qtpy.QtWidgets import QLabel

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui.iconography import interface_icon
from napari_vipp.ui.workflow_assistant_style import AssistantDisclosureButton


def _settle(dialog, qtbot):
    qtbot.waitUntil(lambda: not dialog._appearance_timer.isActive())
    qtbot.waitUntil(lambda: not dialog.transcript_edit._fit_timer.isActive())
    qtbot.wait(10)


def _accept_questions(dialog, questions):
    proposal = reference_proposal(dialog._context)
    proposal["questions"] = list(questions)
    dialog._accept_proposal(proposal)
    return proposal


def _frame_margins(frame):
    margins = frame.layout().contentsMargins()
    return margins.left(), margins.top(), margins.right(), margins.bottom()


@pytest.fixture
def question_discussion(assistant, qtbot):
    dialog = assistant.dialog
    dialog.goal_edit.setPlainText("Preserve the observed calibrated structures.")
    dialog.prepare()  # Example is a fixed no-AI recipe.
    questions = (
        "Current question alpha: which observed boundary should remain separate?",
        "Current question beta: which calibrated structure needs explicit review?",
    )
    _accept_questions(dialog, questions)
    dialog.discussion_edit.setPlainText("Keep this unsent follow-up intact.")
    dialog.resize(760, 800)
    dialog.stage_tabs.setCurrentIndex(1)
    _settle(dialog, qtbot)
    return assistant, questions


def test_goal_and_history_disclosures_start_collapsed_and_keep_literal_captions(
    question_discussion, qtbot
):
    assistant, _questions = question_discussion
    dialog = assistant.dialog
    assert dialog.status_label.text() == (
        "Answer the questions in Discussion so the AI can revise the plan. "
        "Nothing can be added yet."
    )
    for button, caption, content in (
        (dialog.original_goal_toggle, "Original goal", dialog.original_goal_label),
        (dialog.conversation_toggle, "Conversation history", dialog.transcript_edit),
    ):
        assert isinstance(button, AssistantDisclosureButton)
        assert button.isCheckable()
        assert not button.isChecked()
        assert not button.autoDefault()
        assert not button.isDefault()
        assert button.text() == button.property("text") == caption
        assert not content.isVisible()
        collapsed_icon = button.icon().pixmap(24, 24).toImage()
        assert not collapsed_icon.isNull()
        assert "expand" in button.accessibleName().lower()
        for expanded in (True, False):
            button.setChecked(expanded)
            _settle(dialog, qtbot)
            assert content.isVisible() is expanded
            assert button.text() == button.property("text") == caption
            assert button._toolbar_style_option().text == "\u2002" + caption
            assert ("collapse" if expanded else "expand") in (
                button.accessibleName().lower()
            )
            current_icon = button.icon().pixmap(24, 24).toImage()
            assert (current_icon != collapsed_icon) is expanded


def test_initial_collapsed_reference_rows_are_as_short_as_their_headers(
    question_discussion
):
    assistant, _questions = question_discussion
    dialog = assistant.dialog
    for frame, header in (
        (dialog.original_goal_frame, dialog.original_goal_toggle),
        (dialog.conversation_frame, dialog.conversation_toggle),
    ):
        assert frame.isVisible()
        assert not header.isChecked()
        assert _frame_margins(frame) == (14, 0, 14, 0)
        assert header.height() >= max(34, header.fontMetrics().height() + 16)
        assert frame.height() <= header.height() + 4


def test_reference_padding_round_trips_under_font_theme_and_width_changes(
    question_discussion, qtbot
):
    assistant, _questions = question_discussion
    dialog = assistant.dialog
    authority = (
        dialog._generation,
        dialog._context,
        dialog._owner,
        dialog._prepared,
        dialog._draft_authorized,
        dialog.consent_checkbox.isChecked(),
    )
    chat, drafts = tuple(dialog._chat), tuple(dialog._drafts)
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    for dark, width, points in ((False, 760, 12), (True, 1040, 16), (False, 760, 14)):
        palette = QPalette(dialog._theme_probe.palette())
        palette.setColor(QPalette.Window, QColor("#202632" if dark else "#f4f6f8"))
        palette.setColor(QPalette.WindowText, QColor("#e9edf4" if dark else "#202530"))
        palette.setColor(QPalette.ButtonText, QColor("#e9edf4" if dark else "#202530"))
        dialog._theme_probe.setPalette(palette)
        font = QFont(dialog.font())
        font.setPointSize(points)
        dialog.setFont(font)
        dialog.resize(width, 800)
        dialog._refresh_appearance()
        _settle(dialog, qtbot)
        for frame, header, roomy_margins in (
            (dialog.original_goal_frame, dialog.original_goal_toggle, (14, 10, 14, 12)),
            (dialog.conversation_frame, dialog.conversation_toggle, (14, 14, 14, 14)),
        ):
            assert not header.isChecked()
            collapsed_height = frame.height()
            assert _frame_margins(frame) == (14, 0, 14, 0)
            assert collapsed_height <= header.height() + 4
            assert header.height() >= max(34, header.fontMetrics().height() + 16)
            header.setChecked(True)
            _settle(dialog, qtbot)
            assert _frame_margins(frame) == roomy_margins
            assert frame.height() > collapsed_height
            header.setChecked(False)
            _settle(dialog, qtbot)
            assert _frame_margins(frame) == (14, 0, 14, 0)
            assert frame.height() == collapsed_height
            assert frame.height() <= header.height() + 4
    assert (
        dialog._generation,
        dialog._context,
        dialog._owner,
        dialog._prepared,
        dialog._draft_authorized,
        dialog.consent_checkbox.isChecked(),
    ) == authority
    assert tuple(dialog._chat) == chat and tuple(dialog._drafts) == drafts
    assert assistant.captures == captures and assistant.assertions == assertions
    assert assistant.applies == assistant.runs == []


def test_live_questions_render_only_in_forms_but_raw_chat_and_drafts_stay_complete(
    question_discussion, qtbot
):
    assistant, questions = question_discussion
    dialog = assistant.dialog
    raw_chat = tuple(dialog._chat)
    drafts = tuple(dialog._drafts)
    raw_proposal = dialog.json_edit.toPlainText()
    assert json.loads(raw_proposal)["questions"] == list(questions)
    assert drafts[-1].questions == questions
    assert json.loads(drafts[-1].proposal_text)["questions"] == list(questions)
    assert all(
        any(speaker == "Assistant" and question in text for speaker, text in raw_chat)
        for question in questions
    )
    for expanded in (False, True, False):
        dialog.conversation_toggle.setChecked(expanded)
        _settle(dialog, qtbot)
        for question in questions:
            assert question not in dialog.transcript_edit.toPlainText()
            assert sum(
                label.text() == question and label.isVisible()
                for label in dialog.questions_group.findChildren(QLabel)
            ) == 1
        assert dialog.questions_group.isVisible()
        assert tuple(dialog._chat) == raw_chat
        assert tuple(dialog._drafts) == drafts
        assert dialog.json_edit.toPlainText() == raw_proposal
    assert assistant.applies == assistant.runs == []


def test_prior_questions_and_answers_return_to_history_after_live_forms_change(
    question_discussion, qtbot
):
    assistant, earlier_questions = question_discussion
    dialog = assistant.dialog
    answer = "Recorded answer: preserve the observed boundary and physical units."
    dialog._append_chat("You", answer)
    later_questions = ("Later question gamma: should the observed gap remain open?",)
    _accept_questions(dialog, later_questions)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)
    history = dialog.transcript_edit.toPlainText()
    assert all(history.count(question) == 1 for question in earlier_questions)
    assert history.count(answer) == 1
    assert later_questions[0] not in history
    assert [question for question, _field in dialog._question_fields] == list(
        later_questions
    )

    _accept_questions(dialog, ())
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)
    history = dialog.transcript_edit.toPlainText()
    assert all(
        history.count(question) == 1
        for question in (*earlier_questions, *later_questions)
    )
    assert history.count(answer) == 1
    assert not dialog._question_fields
    assert not dialog.questions_group.isVisible()
    assert any(draft.questions == earlier_questions for draft in dialog._drafts)
    assert any(draft.questions == later_questions for draft in dialog._drafts)
    assert all(
        any(question in text for _speaker, text in dialog._chat)
        for question in (*earlier_questions, *later_questions)
    )
    assert assistant.applies == assistant.runs == []


def test_browsing_an_earlier_plan_keeps_live_question_deduplication(
    question_discussion, qtbot
):
    assistant, questions = question_discussion
    dialog = assistant.dialog
    raw_chat, drafts = tuple(dialog._chat), tuple(dialog._drafts)
    current_index = dialog._current_draft_index
    fields = tuple(dialog._question_fields)
    assert current_index > 0
    dialog.draft_combo.setCurrentIndex(0)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)
    assert dialog._current_draft_index == current_index
    assert tuple(dialog._question_fields) == fields
    assert all(
        question not in dialog.transcript_edit.toPlainText() for question in questions
    )
    assert tuple(dialog._chat) == raw_chat and tuple(dialog._drafts) == drafts
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
def test_disclosure_icons_use_shared_state_glyphs_and_disabled_palette(
    qtbot, direction
):
    button = AssistantDisclosureButton("Reference")
    qtbot.addWidget(button)
    button.setLayoutDirection(direction)
    palette = QPalette(button.palette())
    palette.setColor(QPalette.ButtonText, QColor("#173f91"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#9a647e"))
    button.setPalette(palette)
    for expanded, kind in ((False, "chevron-right"), (True, "chevron-down")):
        button.setChecked(expanded)
        for mode in (QIcon.Normal, QIcon.Disabled):
            assert button.icon().pixmap(24, 24, mode).toImage() == (
                interface_icon(kind, palette, 18).pixmap(24, 24, mode).toImage()
            )
        expected_text = (
            "Reference\u2002" if direction == Qt.RightToLeft else "\u2002Reference"
        )
        assert button._toolbar_style_option().text == expected_text
        assert button.text() == "Reference"


def test_disclosure_toggles_keyboard_and_appearance_changes_never_send_or_mutate(
    question_discussion, qtbot, monkeypatch
):
    assistant, _questions = question_discussion
    dialog = assistant.dialog
    calls = []
    for method in (
        "prepare", "_send", "revise_plan", "apply_proposal", "_run_added_plan"
    ):
        monkeypatch.setattr(dialog, method, lambda: calls.append("action"))
    authority = (
        dialog._context,
        dialog._owner,
        dialog._prepared,
        dialog._generation,
        dialog._draft_authorized,
        dialog._added_ids,
        dialog._current_draft_index,
        dialog._busy,
        dialog.consent_checkbox.isChecked(),
    )
    enabled = tuple(
        button.isEnabled() for button in (
            dialog.prepare_button,
            dialog.send_button,
            dialog.revise_button,
            dialog.apply_button,
            dialog.run_button,
        )
    )
    chat, drafts = tuple(dialog._chat), tuple(dialog._drafts)
    fields = tuple(dialog._question_fields)
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    for button in (dialog.original_goal_toggle, dialog.conversation_toggle):
        dialog.discussion_scroll.ensureWidgetVisible(button)
        button.setFocus()
        for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter):
            before = button.isChecked()
            qtbot.keyClick(button, key)
            _settle(dialog, qtbot)
            assert button.isChecked() is not before
            assert not dialog._closed

    for dark, width, points in ((True, 760, 14), (False, 1040, 12), (True, 760, 16)):
        palette = QPalette(dialog._theme_probe.palette())
        for role, light, dark_colour in (
            (QPalette.Window, "#f4f6f8", "#202632"),
            (QPalette.WindowText, "#202530", "#e9edf4"),
            (QPalette.Base, "#ffffff", "#151a22"),
            (QPalette.Text, "#202530", "#e9edf4"),
            (QPalette.Button, "#e7ebef", "#303642"),
            (QPalette.ButtonText, "#202530", "#e9edf4"),
        ):
            palette.setColor(role, QColor(dark_colour if dark else light))
        dialog._theme_probe.setPalette(palette)
        font = QFont(dialog.font())
        font.setPointSize(points)
        dialog.setFont(font)
        dialog.resize(width, 800)
        dialog._refresh_appearance()
        for expanded in (True, False):
            dialog.original_goal_toggle.setChecked(expanded)
            dialog.conversation_toggle.setChecked(expanded)
            _settle(dialog, qtbot)
            assert dialog.original_goal_toggle.text() == "Original goal"
            assert dialog.conversation_toggle.text() == "Conversation history"

    assert (
        dialog._context,
        dialog._owner,
        dialog._prepared,
        dialog._generation,
        dialog._draft_authorized,
        dialog._added_ids,
        dialog._current_draft_index,
        dialog._busy,
        dialog.consent_checkbox.isChecked(),
    ) == authority
    assert tuple(
        button.isEnabled() for button in (
            dialog.prepare_button,
            dialog.send_button,
            dialog.revise_button,
            dialog.apply_button,
            dialog.run_button,
        )
    ) == enabled
    assert tuple(dialog._chat) == chat and tuple(dialog._drafts) == drafts
    assert tuple(dialog._question_fields) == fields
    assert dialog.discussion_edit.toPlainText() == "Keep this unsent follow-up intact."
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    assert np.array_equal(assistant.data, pixels) and not assistant.data.flags.writeable
    assert calls == assistant.applies == assistant.runs == []
