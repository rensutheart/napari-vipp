"""Viewport-sized editors and pinned actions preserve offline authoring state."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import QPoint
from qtpy.QtGui import QFont, QTextCursor

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui import workflow_assistant
from napari_vipp.ui.workflow_assistant_style import ResponsiveAssistantEditor


def _rect_in(widget, ancestor):
    rectangle = widget.rect()
    rectangle.moveTopLeft(widget.mapTo(ancestor, QPoint(0, 0)))
    return rectangle


def _settle(dialog, qtbot):
    timers = (
        dialog._appearance_timer,
        dialog.transcript_edit._fit_timer,
        dialog.goal_edit._height_timer,
        dialog.discussion_edit._height_timer,
    )
    qtbot.waitUntil(lambda: not any(timer.isActive() for timer in timers))
    qtbot.wait(15)


def _assert_editor_height(editor, viewport, *, minimum, maximum, lines, chrome, ratio):
    assert isinstance(editor, ResponsiveAssistantEditor)
    floor = max(minimum, editor.fontMetrics().lineSpacing() * lines + chrome)
    ceiling = max(floor, maximum)
    assert editor.height_bounds() == (floor, ceiling)
    target = max(floor, min(ceiling, round(viewport.height() * ratio)))
    assert editor.height() == editor.minimumHeight() == editor.maximumHeight() == target


def _cursor_state(editor):
    cursor = editor.textCursor()
    return cursor.position(), cursor.anchor(), cursor.selectedText()


def _select_text(editor):
    cursor = editor.textCursor()
    cursor.setPosition(7)
    cursor.setPosition(46, QTextCursor.KeepAnchor)
    editor.setTextCursor(cursor)


def _state(dialog):
    return {
        "goal": dialog.goal_edit.toPlainText(),
        "message": dialog.discussion_edit.toPlainText(),
        "cursors": (
            _cursor_state(dialog.goal_edit),
            _cursor_state(dialog.discussion_edit),
        ),
        "chat": tuple(dialog._chat),
        "replies": dict(dialog._chat_replies),
        "drafts": tuple(dialog._drafts),
        "objective": dialog._request_objective,
        "payload": dialog._context.public_request(dialog._request_objective),
        "json": dialog.json_edit.toPlainText(),
        "request": dialog.request_edit.toPlainText(),
        "policy": dialog._planning_preferences(),
        "provider": dialog.provider_combo.currentData(),
        "previews": dialog.images_checkbox.isChecked(),
        "consent": dialog.consent_checkbox.isChecked(),
        "authority": (
            dialog._generation,
            dialog._draft_authorized,
            dialog._added_ids,
            dialog._current_draft_index,
            dialog._busy,
            dialog.apply_button.isEnabled(),
            dialog.run_button.isEnabled(),
        ),
        "answers": tuple(
            (question, field.text()) for question, field in dialog._question_fields
        ),
    }


@pytest.fixture
def populated_assistant(assistant, qtbot):
    dialog = assistant.dialog
    goal = (
        "  Original literal goal:\n"
        + "Keep observed objects and calibrated units.\n" * 38
        + "  End.  "
    )
    message = (
        "  Unsent literal message:\n"
        + "Keep every line and my own words.\n" * 15
        + "  End.  "
    )
    dialog.goal_edit.setPlainText(goal)
    # Select ordinary send controls, then inject the fixed offline reference:
    # Example mode intentionally hides Preview, so it cannot test that action.
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("offline-inert-model")
    dialog._ensure_context()
    dialog._append_request_chat(dialog._request_objective)
    raw = reference_proposal(dialog._context)
    raw["summary"] = "A conservative, reviewable observed-signal starting graph. " * 28
    raw["assumptions"] = [
        "Preserve observed calibration and never infer missing biology. " * 10
    ]
    dialog._accept_proposal(raw)
    for speaker in ("Assistant", "You", "Assistant"):
        dialog._append_chat(speaker, "Literal history remains available.\n" * 24)
    dialog.discussion_edit.setPlainText(message)
    dialog.original_goal_toggle.setChecked(True)
    dialog.conversation_toggle.setChecked(True)
    dialog.advanced_checkbox.setChecked(True)
    _select_text(dialog.goal_edit)
    _select_text(dialog.discussion_edit)
    _settle(dialog, qtbot)
    return assistant


@pytest.mark.parametrize("size", [(760, 600), (760, 800), (1040, 820), (1440, 1100)])
@pytest.mark.parametrize("points", [12, 16])
def test_all_stage_actions_stay_pinned_at_scroll_extremes_without_state_changes(
    populated_assistant, qtbot, monkeypatch, size, points
):
    assistant = populated_assistant
    dialog = assistant.dialog
    actions = []
    monkeypatch.setattr(dialog, "_send", lambda: actions.append("send"))
    monkeypatch.setattr(dialog, "apply_proposal", lambda: actions.append("add"))
    monkeypatch.setattr(dialog, "_run_added_plan", lambda: actions.append("run"))
    before = _state(dialog)
    context, owner, prepared = dialog._context, dialog._owner, dialog._prepared
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
    font = QFont(dialog.font())
    font.setPointSize(points)
    dialog.setFont(font)
    dialog.resize(*size)
    for stage, page, scroll, strip, buttons in (
        (
            0,
            dialog.goal_page,
            dialog.goal_scroll,
            dialog.goal_actions_frame,
            (dialog.preview_button, dialog.prepare_button),
        ),
        (
            1,
            dialog.discussion_page,
            dialog.discussion_scroll,
            dialog.discussion_actions_frame,
            (dialog.discussion_preview_button, dialog.send_button),
        ),
        (
            2,
            dialog.plans_page,
            dialog.review_scroll,
            dialog.plans_actions_frame,
            (dialog.apply_button, dialog.run_button),
        ),
    ):
        dialog.stage_tabs.setCurrentIndex(stage)
        _settle(dialog, qtbot)
        assert (dialog.width(), dialog.height()) == size
        assert strip.parentWidget() is page
        assert not scroll.widget().isAncestorOf(strip)
        assert strip.objectName() == "AssistantPageActions"
        fixed = _rect_in(strip, dialog)
        assert dialog.rect().contains(fixed)
        assert _rect_in(scroll.viewport(), dialog).bottom() < fixed.top()
        assert fixed.bottom() < _rect_in(dialog.footer_frame, dialog).top()
        assert scroll.horizontalScrollBar().maximum() == 0
        for button in buttons:
            assert button.isVisible() and strip.isAncestorOf(button)
            assert strip.rect().contains(_rect_in(button, strip))
            assert button.height() >= button.fontMetrics().height() + 16
        button_rects = tuple(_rect_in(button, dialog) for button in buttons)
        for value in (0, scroll.verticalScrollBar().maximum()):
            scroll.verticalScrollBar().setValue(value)
            _settle(dialog, qtbot)
            assert _rect_in(strip, dialog) == fixed
            assert tuple(_rect_in(button, dialog) for button in buttons) == button_rects
        if stage == 0:
            _assert_editor_height(
                dialog.goal_edit,
                scroll.viewport(),
                minimum=84,
                maximum=180,
                lines=3,
                chrome=24,
                ratio=0.28,
            )
        elif stage == 1:
            assert scroll.widget().isAncestorOf(dialog.composer_frame)
            assert not dialog.revise_button.isVisible()
            _assert_editor_height(
                dialog.discussion_edit,
                scroll.viewport(),
                minimum=72,
                maximum=140,
                lines=2,
                chrome=16,
                ratio=0.20,
            )
        else:
            assert scroll.widget().isAncestorOf(dialog.advanced_group)
            assert scroll.widget().isAncestorOf(dialog.draft_combo)
            assert scroll.verticalScrollBar().maximum() > 0
    assert _state(dialog) == before
    assert (
        dialog._context is context
        and dialog._owner is owner
        and dialog._prepared is prepared
    )
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert not assistant.data.flags.writeable
    assert actions == assistant.applies == assistant.runs == []


@pytest.mark.parametrize("size", [(760, 600), (1040, 820)])
def test_answer_send_visibility_and_pinning_follow_current_question_fields(
    populated_assistant, qtbot, size
):
    assistant = populated_assistant
    dialog = assistant.dialog
    raw = reference_proposal(dialog._context)
    questions = [
        f"Question {index}: " + "Keep observed boundaries separate? " * 8
        for index in range(1, 4)
    ]
    raw["questions"] = questions
    dialog._accept_proposal(raw)
    for question, field in dialog._question_fields:
        field.setText(f"  Preserve the exact observed choice for {question[:11]}.  ")
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.resize(*size)
    _settle(dialog, qtbot)
    before = _state(dialog)
    button = dialog.revise_button
    strip = dialog.discussion_actions_frame
    assert button.isVisible() and strip.isAncestorOf(button)
    assert not dialog.questions_group.isAncestorOf(button)
    fixed = _rect_in(button, dialog)
    assert strip.rect().contains(_rect_in(button, strip))
    for value in (0, dialog.discussion_scroll.verticalScrollBar().maximum()):
        dialog.discussion_scroll.verticalScrollBar().setValue(value)
        _settle(dialog, qtbot)
        assert _rect_in(button, dialog) == fixed
        assert dialog.rect().contains(fixed)
    assert _state(dialog) == before
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    raw["questions"] = []
    dialog._accept_proposal(raw)
    dialog.stage_tabs.setCurrentIndex(1)
    _settle(dialog, qtbot)
    assert not dialog._question_fields and not button.isVisible()
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("action", ["goal", "message", "answers"])
def test_busy_inline_spinner_remains_visible_in_pinned_action_strip(
    assistant, qtbot, monkeypatch, action
):
    dialog = assistant.dialog
    held = []
    monkeypatch.setattr(
        workflow_assistant,
        "QThreadPool",
        SimpleNamespace(globalInstance=lambda: SimpleNamespace(start=held.append)),
    )
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("local"))
    dialog.model_combo.setCurrentText("offline-held-model")
    dialog._ensure_context()
    if action != "goal":
        raw = reference_proposal(dialog._context)
        if action == "answers":
            raw["questions"] = ["Which observed boundary should remain?"]
        dialog._accept_proposal(raw)
        dialog.stage_tabs.setCurrentIndex(1)
    if action == "goal":
        button, scroll, strip = (
            dialog.prepare_button,
            dialog.goal_scroll,
            dialog.goal_actions_frame,
        )
    elif action == "answers":
        dialog._question_fields[0][1].setText("Keep the observed boundary.")
        button, scroll, strip = (
            dialog.revise_button,
            dialog.discussion_scroll,
            dialog.discussion_actions_frame,
        )
    else:
        dialog.discussion_edit.setPlainText("Keep my unsent observations intact.")
        button, scroll, strip = (
            dialog.send_button,
            dialog.discussion_scroll,
            dialog.discussion_actions_frame,
        )
    dialog.resize(760, 600)
    button.click()
    assert len(held) == 1  # Worker retained but never invoked: no provider calls.
    assert dialog._busy and not button.isEnabled() and button.property("sending")
    _settle(dialog, qtbot)
    before = _state(dialog)
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
    for size in ((760, 600), (1040, 820), (760, 800), (760, 600)):
        dialog.resize(*size)
        _settle(dialog, qtbot)
        assert button.isVisible() and strip.isAncestorOf(button)
        assert strip.rect().contains(_rect_in(button, strip))
        assert dialog.rect().contains(_rect_in(button, dialog))
        assert (
            _rect_in(scroll.viewport(), dialog).bottom() < _rect_in(strip, dialog).top()
        )
        fixed = _rect_in(button, dialog)
        for value in (0, scroll.verticalScrollBar().maximum()):
            scroll.verticalScrollBar().setValue(value)
            _settle(dialog, qtbot)
            assert _rect_in(button, dialog) == fixed
        assert button._spinner_timer.isActive()
        angle = button._spinner_angle
        qtbot.waitUntil(lambda angle=angle: button._spinner_angle != angle)
    assert _state(dialog) == before
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert assistant.applies == assistant.runs == []
    dialog.cancel_request()
    assert not button._spinner_timer.isActive()
