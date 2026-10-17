"""Aligned, readable chat bubbles are presentation only, never request edits."""

from __future__ import annotations

import numpy as np
import pytest
from napari.utils.theme import get_theme
from qtpy.QtCore import Qt
from qtpy.QtGui import (
    QColor,
    QFont,
    QPalette,
    QTextCursor,
    QTextFrameFormat,
    QTextLength,
    QTextTable,
)

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.ui.palette_roles import blend_colors
from napari_vipp.ui.workflow_assistant_style import (
    assistant_chat_surfaces,
    assistant_surfaces,
)


def _settle(dialog, qtbot):
    # Font/style repolishing can queue another width/fit pass after either
    # single-shot timer briefly becomes idle. Wait for the actual no-inner-
    # scrolling contract instead of assuming a fixed 10 ms settles that pass.
    qtbot.waitUntil(
        lambda: (
            not dialog._appearance_timer.isActive()
            and not dialog.transcript_edit._fit_timer.isActive()
            and dialog.transcript_edit.verticalScrollBar().maximum() == 0
        )
    )


def _palette(theme):
    native = get_theme(theme)
    palette = QPalette()
    for role, value in (
        (QPalette.Window, native.background),
        (QPalette.Base, native.foreground),
        (QPalette.Button, native.foreground),
        (QPalette.WindowText, native.text),
        (QPalette.Text, native.text),
        (QPalette.ButtonText, native.text),
    ):
        palette.setColor(role, QColor(*value.as_rgb_tuple()))
    return palette


def _luminance(color):
    def linear(value):
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return sum(
        weight * linear(value)
        for weight, value in zip(
            (0.2126, 0.7152, 0.0722), color.getRgbF()[:3], strict=True
        )
    )


def _contrast(first, second):
    bright, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (bright + 0.05) / (dark + 0.05)


def _color_distance(first, second):
    return sum(
        abs(a - b) for a, b in zip(first.getRgb()[:3], second.getRgb()[:3], strict=True)
    )


def _table_for(document, token):
    cursor = document.find(token)
    assert not cursor.isNull(), token
    table = cursor.currentTable()
    assert isinstance(table, QTextTable), token
    return table, cursor


def _sender_in(document, table, speaker):
    start = QTextCursor(document)
    start.setPosition(table.firstPosition())
    cursor = document.find(speaker, start)
    assert not cursor.isNull()
    assert cursor.currentTable() == table  # Label belongs to the same bubble.
    assert table.firstPosition() <= cursor.position() <= table.lastPosition()
    return cursor


def _assert_literal_nonlinks(document):
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


def _foreground(dialog, cursor):
    brush = cursor.charFormat().foreground()
    return (
        dialog.transcript_edit.palette().color(QPalette.Text)
        if brush.style() == Qt.NoBrush
        else brush.color()
    )


def _assert_solid_border(table, color, background):
    style = table.format()
    assert style.border() >= 1
    assert style.borderStyle() == QTextFrameFormat.BorderStyle_Solid
    assert style.borderBrush().style() != Qt.NoBrush
    assert style.borderBrush().color() == color
    assert _contrast(color, background) >= 1.5


@pytest.mark.parametrize("width", [760, 1040])
@pytest.mark.parametrize("points", [12, 16])
def test_sender_and_message_share_oppositely_aligned_bubbles_with_real_gutters(
    assistant, qtbot, width, points
):
    dialog = assistant.dialog
    font = QFont(dialog.font())
    font.setPointSize(points)
    dialog.setFont(font)
    dialog.resize(width, 800)
    assistant_text = (
        "ASSISTANT_BODY " + "A conservative draft remains reviewable. " * 12
    )
    user_text = "USER_BODY " + "These are my own observations and choices. " * 12
    dialog._append_chat("Assistant", assistant_text)
    dialog._append_chat("You", user_text)
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)

    document = dialog.transcript_edit.document()
    outer = document.rootFrame().childFrames()
    assert len(outer) == 2 and all(isinstance(frame, QTextTable) for frame in outer)
    assistant_table, _ = _table_for(document, "ASSISTANT_BODY")
    user_table, _ = _table_for(document, "USER_BODY")
    _sender_in(document, assistant_table, "Assistant")
    _sender_in(document, user_table, "You")
    for table, percent, alignment in (
        (assistant_table, 82, Qt.AlignLeft),
        (user_table, 74, Qt.AlignRight),
    ):
        width_format = table.format().width()
        assert width_format.type() == QTextLength.PercentageLength
        assert width_format.rawValue() == percent
        horizontal = Qt.AlignLeft | Qt.AlignRight | Qt.AlignHCenter | Qt.AlignJustify
        assert table.format().alignment() & horizontal == alignment

    layout = document.documentLayout()
    available = layout.frameBoundingRect(document.rootFrame())
    assistant_rect = layout.frameBoundingRect(assistant_table)
    user_rect = layout.frameBoundingRect(user_table)
    margin = document.documentMargin()
    # Qt reports the table's content rectangle, inset by native cell padding.
    assistant_inset = margin + assistant_table.format().cellPadding()
    user_inset = margin + user_table.format().cellPadding()
    assert assistant_rect.left() <= available.left() + assistant_inset + 2
    assert available.right() - assistant_rect.right() >= available.width() * 0.12
    assert user_rect.left() - available.left() >= available.width() * 0.20
    assert available.right() - user_rect.right() <= user_inset + 2
    assert user_rect.left() > assistant_rect.left()
    assert 0.70 <= user_rect.width() / available.width() <= 0.78
    assert 0.78 <= assistant_rect.width() / available.width() <= 0.86
    assert (
        user_rect.top() > assistant_rect.bottom()
    )  # Successive bordered bubbles retain a real, positive vertical gap.
    assert dialog.transcript_edit.verticalScrollBar().maximum() == 0
    assert assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_stronger_user_accent_keeps_sender_and_body_readable(assistant, qtbot, theme):
    dialog = assistant.dialog
    palette = _palette(theme)
    dialog._theme_probe.setPalette(palette)
    dialog._refresh_appearance()
    dialog._append_chat("You", "READABLE_USER_BODY")
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)
    colors, panel, raised, _, user_surface = assistant_surfaces(palette)
    previous_tint = blend_colors(panel, colors.info.accent, 0.16)
    assert _color_distance(user_surface, panel) >= 2 * _color_distance(
        previous_tint, panel
    )
    assert user_surface != raised and _contrast(colors.text, user_surface) >= 4.5

    document = dialog.transcript_edit.document()
    table, body = _table_for(document, "READABLE_USER_BODY")
    label = _sender_in(document, table, "You")
    background = table.cellAt(0, 0).format().background().color()
    assert background == user_surface
    for cursor in (label, body):
        foreground = _foreground(dialog, cursor)
        assert _contrast(foreground, background) >= 4.5, (theme, foreground.name())


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_assistant_bubble_is_opaque_neutral_bordered_and_readable(
    assistant, qtbot, theme
):
    dialog = assistant.dialog
    palette = _palette(theme)
    dialog._theme_probe.setPalette(palette)
    dialog._refresh_appearance()
    message = "  NEUTRAL_ASSISTANT_BODY <b>literal</b>\nA second literal line.  "
    index = len(dialog._chat)
    dialog._append_chat("Assistant", message)
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)
    document = dialog.transcript_edit.document()
    table, body = _table_for(document, "NEUTRAL_ASSISTANT_BODY")
    sender = _sender_in(document, table, "Assistant")
    assistant_surface, assistant_border, _, _ = assistant_chat_surfaces(palette)
    colors, panel, _, _, user_surface = assistant_surfaces(palette)
    assert assistant_surface == panel and assistant_surface != user_surface
    assert assistant_border == blend_colors(panel, colors.text, 0.26)
    fill = table.cellAt(0, 0).format().background()
    assert fill.style() != Qt.NoBrush and fill.color() == assistant_surface
    _assert_solid_border(table, assistant_border, assistant_surface)
    for cursor in (sender, body):
        foreground = _foreground(dialog, cursor)
        assert foreground == colors.text
        assert _contrast(foreground, assistant_surface) >= 4.5
    assert table.format().width().rawValue() == 82
    assert dialog._chat[index] == ("Assistant", message)
    assert all(
        line.strip() in dialog.transcript_edit.toPlainText()
        for line in message.splitlines()
    )
    _assert_literal_nonlinks(document)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_quoted_question_has_readable_accent_outline_and_padded_inset(
    assistant, qtbot, theme
):
    dialog = assistant.dialog
    palette = _palette(theme)
    dialog._theme_probe.setPalette(palette)
    dialog._refresh_appearance()
    question = "  INSET_QUESTION <table>literal</table>\nExact question line two.  "
    answer = "  INSET_ANSWER <img src='file:///private'>\nExact answer line two.  "
    index = len(dialog._chat)
    dialog._append_chat_reply(question, answer)
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)
    document = dialog.transcript_edit.document()
    user_table, _ = _table_for(document, "INSET_ANSWER")
    quote_table, quote_body = _table_for(document, "INSET_QUESTION")
    caption = document.find("Assistant’s question")
    assert not caption.isNull() and caption.currentTable() == quote_table
    assert quote_table.parentFrame() == user_table
    _, _, quote_surface, quote_border = assistant_chat_surfaces(palette)
    colors, panel, _, _, user_surface = assistant_surfaces(palette)
    assert quote_surface == blend_colors(user_surface, panel, 0.55)
    assert quote_border == blend_colors(quote_surface, colors.info.accent, 0.65)
    assert quote_surface != user_surface and quote_surface != panel
    fill = quote_table.cellAt(0, 0).format().background()
    assert fill.style() != Qt.NoBrush and fill.color() == quote_surface
    _assert_solid_border(quote_table, quote_border, quote_surface)
    assert quote_table.format().cellPadding() >= 10
    assert quote_table.format().width().rawValue() == 100
    assert user_table.format().width().rawValue() == 74
    assert user_table.format().border() == 0  # Preserve unoutlined blue user bubbles.
    for cursor in (caption, quote_body):
        foreground = _foreground(dialog, cursor)
        assert foreground == colors.text
        assert _contrast(foreground, quote_surface) >= 4.5
    layout = document.documentLayout()
    quote_rect = layout.frameBoundingRect(quote_table)
    user_rect = layout.frameBoundingRect(user_table)
    assert quote_rect.left() > user_rect.left()
    assert quote_rect.right() <= user_rect.right() + 2
    assert quote_rect.top() > user_rect.top()
    assert quote_rect.bottom() < user_rect.bottom()
    assert dialog._chat[index] == ("You", f"{question}\n{answer}")
    assert dialog._chat_replies[index].question == question
    assert dialog._chat_replies[index].answer == answer
    assert all(
        line.strip() in dialog.transcript_edit.toPlainText()
        for line in (question + "\n" + answer).splitlines()
    )
    _assert_literal_nonlinks(document)


def test_typed_question_is_a_separate_neutral_quote_inside_the_user_bubble(
    assistant, qtbot
):
    dialog = assistant.dialog
    question = "  QUOTED_QUESTION <b>literal</b>\nSecond line & details.  "
    answer = '  TYPED_ANSWER <a href="https://example.invalid">literal</a>\nMy words.  '
    index = len(dialog._chat)
    dialog._append_chat_reply(question, answer)
    dialog.stage_tabs.setCurrentIndex(1)
    dialog.conversation_toggle.setChecked(True)
    _settle(dialog, qtbot)
    document = dialog.transcript_edit.document()
    user_table, _ = _table_for(document, "TYPED_ANSWER")
    quote_table, quote = _table_for(document, "QUOTED_QUESTION")
    assert quote_table != user_table and quote_table.parentFrame() == user_table
    _sender_in(document, user_table, "You")
    for caption, table in (
        ("Assistant’s question", quote_table),
        ("Your answer", user_table),
    ):
        caption_cursor = document.find(caption)
        assert not caption_cursor.isNull() and caption_cursor.currentTable() == table
    colors, _, _, _, user_surface = assistant_surfaces(dialog._theme_probe.palette())
    _, _, quote_surface, _ = assistant_chat_surfaces(dialog._theme_probe.palette())
    assert quote_table.cellAt(0, 0).format().background().color() == quote_surface
    assert user_table.cellAt(0, 0).format().background().color() == user_surface
    assert quote.charFormat().foreground().color() == colors.text
    assert dialog._chat[index] == ("You", f"{question}\n{answer}")
    assert dialog._chat_replies[index].question == question
    assert dialog._chat_replies[index].answer == answer
    _assert_literal_nonlinks(document)


@pytest.mark.parametrize("speaker", ["You", "Assistant"])
def test_untyped_bubbles_keep_multiline_markup_literal_without_guessing(
    assistant, speaker
):
    dialog = assistant.dialog
    message = (
        "  LITERAL_BODY <table><tr><td>not a table</td></tr></table>\n"
        '<a href="file:///private">not a link</a> & <img src="file:///private">\n'
        "A question-looking line?\nMy own next line.  "
    )
    index = len(dialog._chat)
    dialog._append_chat(speaker, message)
    document = dialog.transcript_edit.document()
    table, _ = _table_for(document, "LITERAL_BODY")
    _sender_in(document, table, speaker)
    assert len(document.rootFrame().childFrames()) == 1
    assert table.childFrames() == []  # User HTML cannot manufacture a nested table.
    assert dialog._chat[index] == (speaker, message)
    assert index not in dialog._chat_replies
    rendered = dialog.transcript_edit.toPlainText()
    assert all(line.strip() in rendered for line in message.splitlines())
    assert "Assistant’s question" not in rendered and "Your answer" not in rendered
    assert dialog.transcript_edit.isReadOnly()
    assert not dialog.transcript_edit.openLinks()
    assert not dialog.transcript_edit.openExternalLinks()
    _assert_literal_nonlinks(document)


def test_chat_alignment_and_colors_do_not_mutate_authority_payload_plan_or_source(
    assistant, qtbot, monkeypatch
):
    dialog = assistant.dialog
    dialog.prepare()  # Fixed offline recipe only; no calculation/provider.
    actions = []
    for method in ("prepare", "_send", "apply_proposal", "_run_added_plan"):
        monkeypatch.setattr(dialog, method, lambda: actions.append("action"))
    context, prepared, owner = dialog._context, dialog._prepared, dialog._owner
    objective = dialog._request_objective
    payload = context.public_request(objective)
    proposal, request = (
        dialog.json_edit.toPlainText(),
        dialog.request_edit.toPlainText(),
    )
    permissions = (
        dialog._generation,
        dialog._draft_authorized,
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
        dialog.consent_checkbox.isChecked(),
    )
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
    dialog._append_chat("Assistant", "Render-only assistant words.")
    dialog._append_chat_reply("Render-only question?", "Render-only answer.")
    dialog._append_chat_reply(
        "Additional instructions 17", "My own free words.", is_question=False
    )
    raw_chat, replies = tuple(dialog._chat), dict(dialog._chat_replies)
    for theme in ("dark", "light"):
        dialog._theme_probe.setPalette(_palette(theme))
        dialog._refresh_appearance()
        dialog.stage_tabs.setCurrentIndex(1)
        dialog.conversation_toggle.setChecked(True)
        dialog.resize(760, 800)
        _settle(dialog, qtbot)
    assert (
        dialog._context is context
        and dialog._prepared is prepared
        and dialog._owner is owner
    )
    assert dialog._request_objective == objective
    assert context.public_request(objective) == payload
    assert tuple(dialog._chat) == raw_chat and dict(dialog._chat_replies) == replies
    assert (
        dialog.json_edit.toPlainText() == proposal
        and dialog.request_edit.toPlainText() == request
    )
    assert (
        dialog._generation,
        dialog._draft_authorized,
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
        dialog.consent_checkbox.isChecked(),
    ) == permissions
    assert "Additional instructions 17" not in dialog.transcript_edit.toPlainText()
    assert "My own free words." in dialog.transcript_edit.toPlainText()
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert actions == assistant.applies == assistant.runs == []
