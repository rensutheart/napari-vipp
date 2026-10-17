"""Typed reply attribution changes presentation, not literal records or requests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from qtpy.QtCore import QEvent, QPoint
from qtpy.QtGui import QFont

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import ProposalDialogue, reference_proposal
from napari_vipp.ui.workflow_assistant_style import assistant_surfaces


def _settle(dialog, qtbot):
    qtbot.waitUntil(lambda: not dialog._appearance_timer.isActive())
    qtbot.waitUntil(lambda: not dialog.transcript_edit._fit_timer.isActive())
    qtbot.wait(10)


def _rect_in(widget, ancestor):
    rectangle = widget.rect()
    rectangle.moveTopLeft(widget.mapTo(ancestor, QPoint(0, 0)))
    return rectangle


def _assert_no_links_or_images(document):
    block = document.begin()
    while block.isValid():
        fragments = block.begin()
        while not fragments.atEnd():
            fragment = fragments.fragment()
            if fragment.isValid():
                assert not fragment.charFormat().isAnchor()
                assert not fragment.charFormat().isImageFormat()
            fragments += 1
        block = block.next()


@pytest.fixture
def prepared_assistant(assistant):
    assistant.dialog.prepare()  # Fixed no-AI recipe, no scientific calculation.
    return assistant


def test_typed_reply_separates_question_and_answer_with_literal_nonlink_text(
    prepared_assistant, qtbot
):
    dialog = prepared_assistant.dialog
    question = (
        '  QUESTION_TOKEN <a href="https://example.invalid">literal link</a>\n'
        '<img src="file:///not-a-real-private-image.png"> & details.  '
    )
    answer = "  ANSWER_TOKEN <script>literal</script>\nKeep both lines & spaces.  "
    index = len(dialog._chat)
    dialog._append_chat_reply(question, answer)
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)

    assert dialog._chat[index] == ("You", f"{question}\n{answer}")
    reply = dialog._chat_replies[index]
    assert reply.question == question and reply.answer == answer
    assert reply.is_question
    with pytest.raises(FrozenInstanceError):
        reply.answer = "Changed"
    rendered = dialog.transcript_edit.toPlainText()
    assert "Assistant’s question" in rendered and "Your answer" in rendered
    assert all(
        line.strip() in rendered for line in (question + "\n" + answer).splitlines()
    )
    document = dialog.transcript_edit.document()
    source_cursor = document.find("QUESTION_TOKEN")
    answer_cursor = document.find("ANSWER_TOKEN")
    assert not source_cursor.isNull() and not answer_cursor.isNull()
    colors, *_ = assistant_surfaces(dialog._theme_probe.palette())
    assert source_cursor.charFormat().foreground().color() == colors.text
    assert source_cursor.currentTable() != answer_cursor.currentTable()
    assert source_cursor.currentTable().parentFrame() == answer_cursor.currentTable()
    assert source_cursor.currentTable().format().border() == 1
    assert source_cursor.position() < answer_cursor.position()
    assert dialog.transcript_edit.isReadOnly()
    assert not dialog.transcript_edit.openLinks()
    assert not dialog.transcript_edit.openExternalLinks()
    _assert_no_links_or_images(document)


def test_ordinary_multiline_you_message_is_never_split_as_a_reply(assistant):
    dialog = assistant.dialog
    message = (
        "A question-looking first line?\nMy own second line.\n"
        "Additional instructions 22"
    )
    index = len(dialog._chat)
    dialog._append_chat("You", message)
    assert index not in dialog._chat_replies
    assert dialog._chat[index] == ("You", message)
    rendered = dialog.transcript_edit.toPlainText()
    assert message in rendered
    assert "Assistant’s question" not in rendered and "Your answer" not in rendered


def test_free_message_anchor_is_hidden_only_in_render_and_remains_in_request(
    prepared_assistant,
):
    dialog = prepared_assistant.dialog
    answer = "  Exact reply to the observed boundary question.  "
    dialog._clarifications = [("Real model question?", answer)]
    message = (
        "  My own multiline request.\nAdditional instructions 999 is literal text.  "
    )
    dialog._extra_message = message
    objective = dialog._answered_objective()
    assert isinstance(objective, ProposalDialogue)
    assert dialog._outgoing_message_pair == ("Additional instructions 1", message)
    assert objective.clarifications == (
        ("Real model question?", answer),
        ("Additional instructions 1", message),
    )
    public = dialog._context.public_request(objective)
    assert public["clarifications"][-1] == {
        "question": "Additional instructions 1",
        "answer": message,
    }
    dialog._append_request_chat(objective)
    index = next(
        index
        for index, record in enumerate(dialog._chat)
        if record == ("You", f"Additional instructions 1\n{message}")
    )
    assert not dialog._chat_replies[index].is_question
    assert dialog._chat_replies[index].answer == message
    rendered = dialog.transcript_edit.toPlainText()
    assert "Additional instructions 1\n" not in rendered
    assert "Additional instructions 999 is literal text." in rendered
    assert "My own multiline request." in rendered
    assert dialog._context.public_request(objective) == public

    dialog._clarifications = list(objective.clarifications)
    dialog._extra_message = ""
    revised = dialog._answered_objective()
    assert dialog._outgoing_message_pair is None
    assert revised.clarifications == objective.clarifications
    dialog._update_chat()
    assert "Additional instructions 1\n" not in dialog.transcript_edit.toPlainText()
    assert dialog._chat[index] == ("You", f"Additional instructions 1\n{message}")


def test_reply_metadata_survives_policy_history_browsing_and_close_reopen(
    prepared_assistant, qtbot
):
    assistant = prepared_assistant
    dialog = assistant.dialog
    raw = reference_proposal(dialog._context)
    raw["summary"] = "A distinct synthetic draft for history navigation."
    dialog._accept_proposal(raw)
    dialog._append_chat_reply(
        "Multiline source question?\nSecond question line.", "My answer."
    )
    chat = tuple(dialog._chat)
    replies, drafts = dict(dialog._chat_replies), tuple(dialog._drafts)
    captures = list(assistant.captures)
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    for index in (0, dialog._current_draft_index, 0):
        dialog.draft_combo.setCurrentIndex(index)
        assert dict(dialog._chat_replies) == replies
    dialog.initiative_slider.setValue(2)
    assert not dialog._draft_authorized
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert dict(dialog._chat_replies) == replies
    dialog.close()
    dialog.show()
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)
    assert tuple(dialog._chat) == chat and tuple(dialog._drafts) == drafts
    assert dict(dialog._chat_replies) == replies
    assert "Assistant’s question" in dialog.transcript_edit.toPlainText()
    assert "Your answer" in dialog.transcript_edit.toPlainText()
    assert dialog._context is dialog._owner is dialog._prepared is None
    assert not dialog.consent_checkbox.isChecked()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert assistant.captures == captures
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert assistant.applies == assistant.runs == []


def test_cleared_history_reused_index_does_not_inherit_reply_attribution(assistant):
    dialog = assistant.dialog
    dialog._chat.clear()
    dialog._append_chat_reply("Old question?", "Old answer.")
    assert 0 in dialog._chat_replies
    raw = dialog._chat[0][1]
    dialog._chat.clear()
    dialog._append_chat("You", raw)  # Same bytes, different unstructured origin.
    assert 0 not in dialog._chat_replies
    assert raw in dialog.transcript_edit.toPlainText()
    assert "Assistant’s question" not in dialog.transcript_edit.toPlainText()


@pytest.mark.parametrize("speaker", ["You", "Assistant"])
def test_changed_raw_tuple_is_not_interpreted_using_stale_metadata(assistant, speaker):
    dialog = assistant.dialog
    index = len(dialog._chat)
    dialog._append_chat_reply("Old question?", "Old answer.")
    dialog._chat[index] = (speaker, "Replacement first line.\nReplacement second line.")
    dialog._update_chat()
    rendered = dialog.transcript_edit.toPlainText()
    assert "Replacement first line.\nReplacement second line." in rendered
    assert "Old question?" not in rendered
    assert "Assistant’s question" not in rendered and "Your answer" not in rendered


def test_reply_rendering_does_not_mutate_request_context_proposal_or_source(
    prepared_assistant, qtbot, monkeypatch
):
    assistant = prepared_assistant
    dialog = assistant.dialog
    calls = []
    for method in ("prepare", "_send", "apply_proposal", "_run_added_plan"):
        monkeypatch.setattr(dialog, method, lambda: calls.append("action"))
    context, owner, prepared = dialog._context, dialog._owner, dialog._prepared
    objective = dialog._request_objective
    public = context.public_request(objective)
    proposal = dialog.json_edit.toPlainText()
    request_text = dialog.request_edit.toPlainText()
    generation = dialog._generation
    authorization = dialog._draft_authorized, dialog.consent_checkbox.isChecked()
    enabled = dialog.apply_button.isEnabled(), dialog.run_button.isEnabled()
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    dialog._append_chat_reply("Render-only context?", "Render-only user words.")
    for expanded in (True, False, True):
        dialog.conversation_toggle.setChecked(expanded)
        dialog._refresh_appearance()
        _settle(dialog, qtbot)
    assert dialog._context is context and dialog._owner is owner
    assert dialog._prepared is prepared and dialog._generation == generation
    assert dialog._request_objective == objective
    assert context.public_request(objective) == public
    assert dialog.json_edit.toPlainText() == proposal
    assert dialog.request_edit.toPlainText() == request_text
    assert (dialog._draft_authorized, dialog.consent_checkbox.isChecked()) == (
        authorization
    )
    assert (dialog.apply_button.isEnabled(), dialog.run_button.isEnabled()) == enabled
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert calls == assistant.applies == assistant.runs == []


@pytest.mark.parametrize("width", [760, 1040])
@pytest.mark.parametrize("points", [12, 16])
def test_compact_initiative_slider_endpoints_remain_close_and_unclipped(
    assistant, qtbot, width, points
):
    dialog = assistant.dialog
    font = QFont(dialog.font())
    font.setPointSize(points)
    dialog.setFont(font)
    dialog.resize(width, 800)
    dialog.stage_tabs.setCurrentIndex(0)
    _settle(dialog, qtbot)
    card = dialog.initiative_card
    margins = card.layout().contentsMargins()
    assert margins.top() <= 12 and margins.bottom() <= 12  # Previous card padding.
    assert card.layout().spacing() <= 8  # Previous gap between every card section.
    slider = _rect_in(dialog.initiative_slider, card)
    assert dialog.initiative_slider.minimumHeight() <= 30  # Previous slider minimum.
    assert card.rect().contains(slider)
    for label in (dialog.initiative_ask_label, dialog.initiative_best_label):
        rectangle = _rect_in(label, card)
        assert 0 <= rectangle.top() - slider.bottom() - 1 <= 4
        assert card.rect().contains(rectangle)
        assert label.height() >= label.heightForWidth(label.width())
    assert assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize(
    "event_type",
    [QEvent.StyleChange, QEvent.PaletteChange, QEvent.ApplicationPaletteChange],
)
def test_theme_change_defers_and_coalesces_refresh_without_nested_stylesheets(
    prepared_assistant, qtbot, monkeypatch, event_type
):
    assistant = prepared_assistant
    dialog = assistant.dialog
    dialog._appearance_timer.stop()
    calls = []
    monkeypatch.setattr(dialog, "_refresh_appearance", lambda: calls.append("refresh"))
    dialog._appearance_timer.timeout.disconnect()
    dialog._appearance_timer.timeout.connect(dialog._refresh_appearance)
    authority = (
        dialog._context,
        dialog._prepared,
        dialog._generation,
        dialog._draft_authorized,
    )
    for _ in range(2):
        # Exercise the handler directly: QWidget's native event dispatcher
        # does not forward ApplicationPaletteChange to changeEvent on all Qt builds.
        # The real application-wide stylesheet path is covered by visual-style tests.
        dialog.changeEvent(QEvent(event_type))
        assert not calls
        assert dialog._appearance_timer.isActive()
    qtbot.waitUntil(lambda: calls == ["refresh"])
    assert not dialog._appearance_timer.isActive()
    assert (
        dialog._context,
        dialog._prepared,
        dialog._generation,
        dialog._draft_authorized,
    ) == authority
    assert assistant.applies == assistant.runs == []
