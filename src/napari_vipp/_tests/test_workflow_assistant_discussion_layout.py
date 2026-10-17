"""Crowded Discussion content remains readable through one passive outer scroll."""

from __future__ import annotations

import numpy as np
import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QLabel, QScrollArea

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal


def _rect_in(widget, ancestor):
    rectangle = widget.rect()
    rectangle.moveTopLeft(widget.mapTo(ancestor, QPoint(0, 0)))
    return rectangle


def _settle_discussion(dialog, qtbot):
    qtbot.waitUntil(lambda: not dialog.transcript_edit._fit_timer.isActive())
    qtbot.waitUntil(
        lambda: (
            not dialog.transcript_edit.isVisible()
            or dialog.transcript_edit.verticalScrollBar().maximum() == 0
        )
    )
    qtbot.wait(10)


def _reveal_control(scroll, widget, qtbot):
    scroll.ensureWidgetVisible(widget, 0, 8)
    qtbot.wait(10)
    # Verify the complete control, not only the focus rectangle Qt may choose
    # for an editor. The outer scrollbar must be able to expose every edge.
    viewport = scroll.viewport().rect()
    rectangle = _rect_in(widget, scroll.viewport())
    bar = scroll.verticalScrollBar()
    if rectangle.bottom() > viewport.bottom():
        bar.setValue(bar.value() + rectangle.bottom() - viewport.bottom() + 8)
    elif rectangle.top() < viewport.top():
        bar.setValue(bar.value() + rectangle.top() - viewport.top() - 8)
    qtbot.wait(10)
    assert viewport.contains(_rect_in(widget, scroll.viewport()))


@pytest.fixture
def crowded_discussion(assistant, qtbot):
    dialog = assistant.dialog
    goal = (
        "  Original objective: "
        + (
            "Preserve observed boundaries, calibrated units, and separate structures. "
            "Do not infer missing boundaries or treat <literal text> as markup.\n"
        )
        * 17
        + "  End of original objective.  "
    )
    questions = [
        f"Question {index}: "
        + (
            "Which observed boundaries and calibrated structures should remain "
            "separate, and which visible ambiguity requires an explicit decision "
            "instead of an unannounced reconstruction? "
        )
        * 2
        + f" Preserve the exact question {index} and its final punctuation?"
        for index in range(1, 4)
    ]
    assert len(goal) < 4000
    assert all(len(question) < 2000 for question in questions)
    dialog.goal_edit.setPlainText(goal)
    dialog.prepare()  # Fixed no-AI recipe: no provider or graph calculation.
    proposal = reference_proposal(dialog._context)
    proposal["questions"] = questions
    dialog._accept_proposal(proposal)
    messages = [
        "Synthetic assistant history: "
        + "Keep every observation and calibration detail available for review. " * 12
        + "END OF ASSISTANT HISTORY",
        "Synthetic user history: <literal> & unmodified. "
        + "Keep the existing scientific inputs unchanged during layout changes. " * 12
        + "END OF USER HISTORY",
    ]
    dialog._append_chat("Assistant", messages[0])
    dialog._append_chat("You", messages[1])
    dialog.discussion_edit.setPlainText("An unsent follow-up must remain intact.")
    dialog.resize(760, 800)
    dialog.stage_tabs.setCurrentIndex(1)
    # Full-history document sizing is checked only after its disclosure opens.
    dialog.conversation_toggle.setChecked(True)
    _settle_discussion(dialog, qtbot)
    return assistant, goal, tuple(questions), tuple(messages)


def test_three_long_questions_and_composer_are_unclipped_and_outer_reachable(
    crowded_discussion, qtbot
):
    assistant, _goal, questions, messages = crowded_discussion
    dialog = assistant.dialog
    scroll = dialog.discussion_scroll
    assert isinstance(scroll, QScrollArea)
    assert scroll.widgetResizable()
    assert dialog.width() == 760 and dialog.height() == 800
    assert scroll.verticalScrollBar().maximum() > 0
    assert scroll.horizontalScrollBar().maximum() == 0
    assert scroll.widget().isAncestorOf(dialog.questions_group)
    assert scroll.widget().isAncestorOf(dialog.composer_frame)
    assert len(dialog._question_fields) == 3
    minimum, maximum = dialog.discussion_edit.height_bounds()
    assert minimum <= dialog.discussion_edit.height() <= maximum

    rendered = dialog.transcript_edit.toPlainText()
    assert dialog.transcript_edit.isReadOnly()
    assert not dialog.transcript_edit.openExternalLinks()
    assert all(message in rendered for message in messages)
    assert dialog.transcript_edit.verticalScrollBar().maximum() == 0
    assert dialog.transcript_edit.horizontalScrollBar().maximum() == 0
    assert dialog.transcript_edit.viewport().height() + 2 >= (
        dialog.transcript_edit.document().size().height()
    )

    for index, (question, field) in enumerate(dialog._question_fields, 1):
        assert question == questions[index - 1]
        label = next(
            item
            for item in dialog.questions_group.findChildren(QLabel)
            if item.text() == question
        )
        assert label.wordWrap()
        assert label.height() >= label.heightForWidth(label.width())
        assert any(
            item.text() == f"Question {index}"
            for item in dialog.questions_group.findChildren(QLabel)
        )
        for widget in (label, field):
            assert dialog.questions_group.rect().contains(
                _rect_in(widget, dialog.questions_group)
            ), question
            _reveal_control(scroll, widget, qtbot)
        assert field.height() >= field.minimumSizeHint().height()

    _reveal_control(scroll, dialog.discussion_edit, qtbot)
    for widget in (
        dialog.discussion_preview_button,
        dialog.send_button,
        dialog.revise_button,
    ):
        assert dialog.discussion_actions_frame.isAncestorOf(widget)
        assert not scroll.widget().isAncestorOf(widget)
        assert dialog.discussion_actions_frame.rect().contains(
            _rect_in(widget, dialog.discussion_actions_frame)
        )
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert assistant.applies == assistant.runs == []


def test_full_original_goal_is_collapsed_separate_literal_and_expandable(
    crowded_discussion, qtbot
):
    _assistant, goal, _questions, _messages = crowded_discussion
    dialog = _assistant.dialog
    recorded_goal = next(
        text for speaker, text in reversed(dialog._chat) if speaker == "You · Goal"
    )
    assert not dialog.original_goal_toggle.isChecked()
    assert not dialog.original_goal_label.isVisible()
    assert dialog.original_goal_label.text() == recorded_goal
    assert dialog.original_goal_label.textFormat() == Qt.PlainText
    assert dialog.original_goal_label.wordWrap()
    assert recorded_goal not in dialog.transcript_edit.toPlainText()
    assert any(
        speaker == "You · Goal" and text == recorded_goal
        for speaker, text in dialog._chat
    )
    collapsed_range = dialog.discussion_scroll.verticalScrollBar().maximum()

    dialog.original_goal_toggle.setChecked(True)
    _settle_discussion(dialog, qtbot)
    assert dialog.original_goal_label.isVisible()
    assert dialog.original_goal_label.text() == recorded_goal
    assert dialog.original_goal_label.height() >= (
        dialog.original_goal_label.heightForWidth(dialog.original_goal_label.width())
    )
    assert dialog.discussion_scroll.verticalScrollBar().maximum() > collapsed_range
    assert recorded_goal not in dialog.transcript_edit.toPlainText()
    dialog.original_goal_toggle.setChecked(False)
    _settle_discussion(dialog, qtbot)
    assert not dialog.original_goal_label.isVisible()
    assert dialog.goal_edit.toPlainText() == goal

    # Rendering preserves exact recorded text independently of the historical
    # initial plain-string request's whitespace normalization convention.
    raw_recorded_goal = "  Stored <literal goal> & unmodified.\nTrailing spaces.  "
    dialog._append_chat("You · Goal", raw_recorded_goal)
    _settle_discussion(dialog, qtbot)
    assert dialog.original_goal_label.text() == raw_recorded_goal
    assert not dialog.original_goal_toggle.isChecked()
    rendered_history = dialog.transcript_edit.toPlainText()
    assert all(line.strip() in rendered_history for line in recorded_goal.splitlines())
    assert rendered_history.count("<literal text>") == (
        recorded_goal.count("<literal text>")
    )
    assert any(
        speaker == "You · Goal" and text == recorded_goal
        for speaker, text in dialog._chat
    )
    dialog.original_goal_toggle.setChecked(True)
    _settle_discussion(dialog, qtbot)
    assert dialog.original_goal_label.isVisible()
    assert dialog.original_goal_label.text() == raw_recorded_goal
    assert dialog.goal_edit.toPlainText() == goal


def test_returning_to_earlier_goal_updates_reference_without_repeating_revisions(
    assistant, qtbot
):
    dialog = assistant.dialog
    first_goal = "Objective A: keep the observed structures separate."
    second_goal = "Objective B: review calibrated structure measurements."
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    for goal in (first_goal, second_goal, first_goal):
        dialog.goal_edit.setPlainText(goal)
        dialog.prepare()  # Fixed Example recipe: no provider request.
        assert dialog._request_objective == goal
        assert dialog.original_goal_label.text() == goal
        assert not dialog.original_goal_toggle.isChecked()

    goals = [text for speaker, text in dialog._chat if speaker == "You · Goal"]
    assert goals == [first_goal, second_goal, first_goal]
    dialog.stage_tabs.setCurrentIndex(1)
    _settle_discussion(dialog, qtbot)
    rendered = dialog.transcript_edit.toPlainText()
    assert rendered.count(first_goal) == 1
    assert rendered.count(second_goal) == 1
    assert dialog.original_goal_label.text() == first_goal

    for _ in range(2):
        dialog.prepare()
        assert dialog._request_objective == first_goal
        assert dialog.original_goal_label.text() == first_goal
        assert [
            text for speaker, text in dialog._chat if speaker == "You · Goal"
        ] == goals
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    assert assistant.applies == assistant.runs == []


def test_toggle_resize_and_outer_scroll_keep_authority_and_bottom_navigation_passive(
    crowded_discussion, qtbot, monkeypatch
):
    assistant, goal, _questions, _messages = crowded_discussion
    dialog = assistant.dialog
    calls = []
    monkeypatch.setattr(dialog, "_send", lambda: calls.append("send"))
    monkeypatch.setattr(dialog, "apply_proposal", lambda: calls.append("add"))
    monkeypatch.setattr(dialog, "_run_added_plan", lambda: calls.append("run"))
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
    history = tuple(dialog._chat)
    transcript = dialog.transcript_edit.toPlainText()
    drafts = tuple(dialog._drafts)
    fields = tuple(dialog._question_fields)
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    for width, expanded in ((760, False), (1040, True), (760, True), (760, False)):
        dialog.resize(width, 800)
        dialog.original_goal_toggle.setChecked(expanded)
        _settle_discussion(dialog, qtbot)
        footer = _rect_in(dialog.footer_frame, dialog)
        bottom = _rect_in(dialog.bottom_frame, dialog)
        viewport = _rect_in(dialog.discussion_scroll.viewport(), dialog)
        assert viewport.bottom() < footer.top()
        assert footer.bottom() < bottom.top()
        assert dialog.rect().contains(footer)
        assert dialog.rect().contains(bottom)
        assert dialog.back_button.isVisible() and dialog.back_button.isEnabled()
        assert dialog.close_button.isVisible()
        for value in (0, dialog.discussion_scroll.verticalScrollBar().maximum()):
            dialog.discussion_scroll.verticalScrollBar().setValue(value)
            qtbot.wait(10)
            assert _rect_in(dialog.footer_frame, dialog) == footer
            assert _rect_in(dialog.bottom_frame, dialog) == bottom

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
    assert tuple(dialog._chat) == history
    assert dialog.transcript_edit.toPlainText() == transcript
    assert tuple(dialog._drafts) == drafts
    assert tuple(dialog._question_fields) == fields
    assert dialog.goal_edit.toPlainText() == goal
    assert dialog.discussion_edit.toPlainText() == (
        "An unsent follow-up must remain intact."
    )
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    assert np.array_equal(assistant.data, pixels)
    assert not assistant.data.flags.writeable
    assert calls == assistant.applies == assistant.runs == []
