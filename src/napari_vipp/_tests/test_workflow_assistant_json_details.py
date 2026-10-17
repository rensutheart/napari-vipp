"""Technical formatting preserves literal text and never authors a plan."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from napari.utils.theme import get_theme
from qtpy.QtCore import QMimeData
from qtpy.QtGui import QColor, QFont, QFontDatabase, QFontInfo, QPalette, QTextCursor
from qtpy.QtWidgets import QApplication, QPlainTextEdit, QVBoxLayout, QWidget

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui.workflow_assistant_details import (
    AssistantDetailsEditor,
    json_detail_formats,
)

_JSON = (
    '  {\n  "emoji": "🧪", "quo\\"ted": '
    '"value: true, null, 42, \\"quote\\" and café",\n'
    '  "n": -12.5e+2, "zero": 0, "yes": true, "no": false, "none": null\n}\n  '
)


@pytest.fixture(autouse=True)
def _installed_fixed_font(qapp):
    """The Windows offscreen fixture otherwise registers only Segoe UI.

    Register an already installed fixed font, never install or replace fonts.
    Keep application-font IDs for the QApplication lifetime, just like the
    existing offscreen font helper: unloading can invalidate Qt font caches.
    """
    if (
        sys.platform == "win32"
        and qapp.platformName().lower() == "offscreen"
        and not any(QFontDatabase.isFixedPitch(f) for f in QFontDatabase.families())
    ):
        fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
        registered = []
        for filename in ("consola.ttf", "cour.ttf"):
            path = fonts / filename
            if path.is_file():
                font_id = QFontDatabase.addApplicationFont(str(path))
                if font_id >= 0:
                    registered.append(font_id)
        qapp._assistant_details_test_fonts = tuple(registered)
    assert any(QFontDatabase.isFixedPitch(f) for f in QFontDatabase.families())


def _settle(editor, qtbot):
    qtbot.waitUntil(lambda: not editor._details_timer.isActive())
    qtbot.wait(10)


def _settle_dialog(dialog, qtbot):
    qtbot.waitUntil(lambda: not dialog._appearance_timer.isActive())
    for editor in (dialog.json_edit, dialog.request_edit, dialog.trace_edit):
        _settle(editor, qtbot)


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


def _contrast(first, second):
    def luminance(color):
        def linear(value):
            return (
                value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            )

        return sum(
            weight * linear(value)
            for weight, value in zip(
                (0.2126, 0.7152, 0.0722), color.getRgbF()[:3], strict=True
            )
        )

    bright, dark = sorted((luminance(first), luminance(second)), reverse=True)
    return (bright + 0.05) / (dark + 0.05)


def _utf16(text):
    return len(text.encode("utf-16-le")) // 2


def _format_at(editor, line, offset):
    """Inspect highlighter layout ranges, not stored document character styles."""
    block = editor.document().findBlockByNumber(line)
    assert block.isValid()
    ranges = block.layout().formats()
    matching = [
        item.format
        for item in ranges
        if item.start <= offset < item.start + item.length
    ]
    assert len(matching) == 1, (line, offset, [(r.start, r.length) for r in ranges])
    return matching[0]


def _assert_token(editor, line, token, kind, *, start=0):
    text = editor.document().findBlockByNumber(line).text()
    index = text.index(token, start)
    expected = editor._json_highlighter.formats[kind]
    for offset in range(_utf16(text[:index]), _utf16(text[: index + len(token)])):
        actual = _format_at(editor, line, offset)
        assert actual.foreground().color() == expected.foreground().color()
        assert actual.fontWeight() == expected.fontWeight()


def _cursor(editor):
    cursor = editor.textCursor()
    return cursor.position(), cursor.anchor(), cursor.selectedText()


def _state(dialog):
    return (
        id(dialog._context),
        id(dialog._owner),
        id(dialog._prepared),
        dialog._request_objective,
        tuple(dialog._clarifications),
        tuple(dialog._drafts),
        dialog._draft_authorized,
        dialog._current_draft_index,
        dialog._live_proposal_text,
        dialog._live_plan_text,
        tuple(dialog._chat),
        dict(dialog._chat_replies),
        dialog.goal_edit.toPlainText(),
        dialog.discussion_edit.toPlainText(),
        tuple(
            editor.toPlainText()
            for editor in (
                dialog.json_edit,
                dialog.request_edit,
                dialog.trace_edit,
                dialog.preview_edit,
            )
        ),
        dialog.initiative_slider.value(),
        dialog.provider_combo.currentData(),
        dialog.images_checkbox.isChecked(),
        dialog.consent_checkbox.isChecked(),
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
        dialog._validation_timer.isActive(),
        tuple(
            editor.isReadOnly()
            for editor in (dialog.json_edit, dialog.request_edit, dialog.trace_edit)
        ),
    )


def test_json_tokens_cover_escaped_quotes_and_utf16_positions_without_rewriting(qtbot):
    assert json.loads(_JSON)["emoji"] == "🧪"
    editor = AssistantDetailsEditor(_JSON, json_syntax=True)
    qtbot.addWidget(editor)
    _settle(editor, qtbot)
    assert editor.toPlainText() == _JSON
    assert editor.document().toPlainText() == _JSON
    _assert_token(editor, 1, '"emoji"', "key")
    _assert_token(editor, 1, '"🧪"', "string")
    # The key follows a supplementary-plane character: Python indices are not
    # Qt's UTF-16 positions. Escaped quotes still belong to one lexical token.
    _assert_token(editor, 1, '"quo\\"ted"', "key")
    _assert_token(
        editor,
        1,
        '"value: true, null, 42, \\"quote\\" and café"',
        "string",
    )
    for token in ("-12.5e+2", "0"):
        _assert_token(editor, 2, token, "number")
    for token in ("true", "false", "null"):
        _assert_token(editor, 2, token, "literal")
    _assert_token(editor, 0, "{", "punctuation")
    _assert_token(editor, 2, ":", "punctuation")
    _assert_token(editor, 3, "}", "punctuation")
    assert not editor.document().isUndoAvailable()


@pytest.mark.parametrize(
    "text",
    [
        '{"unfinished": "literal \\"quote and 🧪',
        '{"number": -1.2e-, "flag": tru, "nothing": null',
        '{"nested": [false, {"key": ',
        'plain truefalse 42.0.1 } <script>\n"unterminated',
    ],
)
def test_partial_or_malformed_json_remains_literal_and_safe_to_edit(qtbot, text):
    editor = AssistantDetailsEditor(text, json_syntax=True)
    qtbot.addWidget(editor)
    _settle(editor, qtbot)
    changes = []
    editor.textChanged.connect(lambda: changes.append(editor.toPlainText()))
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.End)
    editor.setTextCursor(cursor)
    editor.insertPlainText("\n  ")
    expected = text + "\n  "
    assert changes == [expected]
    cursor_before = _cursor(editor)
    modified = editor.document().isModified()
    editor.refresh_highlighting()
    _settle(editor, qtbot)
    assert editor.toPlainText() == expected
    assert _cursor(editor) == cursor_before
    assert editor.document().isModified() == modified
    assert changes == [expected]  # No parser/repair or formatting-triggered edits.
    editor.undo()
    assert editor.toPlainText() == text


@pytest.mark.parametrize("editor_blocked", [False, True])
@pytest.mark.parametrize("document_blocked", [False, True])
def test_highlight_refresh_restores_existing_signal_blocks_and_document_state(
    qtbot, editor_blocked, document_blocked
):
    editor = AssistantDetailsEditor(_JSON, json_syntax=True)
    qtbot.addWidget(editor)
    _settle(editor, qtbot)
    editor.document().setModified(False)
    editor.blockSignals(editor_blocked)
    editor.document().blockSignals(document_blocked)
    changes = []
    editor.textChanged.connect(lambda: changes.append("editor"))
    editor.document().contentsChanged.connect(lambda: changes.append("document"))
    editor.refresh_highlighting()
    assert editor.signalsBlocked() == editor_blocked
    assert editor.document().signalsBlocked() == document_blocked
    assert changes == []
    assert editor.toPlainText() == _JSON
    assert not editor.document().isModified()
    assert not editor.document().isUndoAvailable()
    # Leave test objects with the same normal signal state as ordinary editors.
    editor.blockSignals(False)
    editor.document().blockSignals(False)


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_all_json_token_colors_are_readable_on_native_editor_background(qtbot, theme):
    editor = AssistantDetailsEditor(_JSON, json_syntax=True)
    qtbot.addWidget(editor)
    editor.setPalette(_palette(theme))
    _settle(editor, qtbot)
    formats = json_detail_formats(editor.palette())
    assert set(formats) == {"key", "string", "number", "literal", "punctuation"}
    background = editor.palette().color(QPalette.Base)
    colors = []
    for kind, format_ in formats.items():
        color = format_.foreground().color()
        assert color.isValid() and color.alpha() == 255
        assert _contrast(color, background) >= 4.5, (theme, kind, color.name())
        colors.append(color.name())
        assert editor._json_highlighter.formats[kind].foreground().color() == color
    assert len(set(colors)) == 5
    _assert_token(editor, 1, '"emoji"', "key")
    _assert_token(editor, 2, "-12.5e+2", "number")


def test_fixed_font_and_read_only_modes_are_local_to_technical_details(
    assistant, qtbot
):
    dialog = assistant.dialog
    _settle_dialog(dialog, qtbot)
    for editor, json_syntax, read_only in (
        (dialog.json_edit, True, False),
        (dialog.request_edit, True, True),
        (dialog.trace_edit, False, True),
    ):
        assert isinstance(editor, AssistantDetailsEditor)
        assert QFontInfo(editor.font()).fixedPitch()
        assert editor.fontMetrics().horizontalAdvance(
            "iiii"
        ) == editor.fontMetrics().horizontalAdvance("WWWW")
        assert editor.isReadOnly() == read_only
        assert bool(editor._json_highlighter) == json_syntax
        assert (
            editor.tabStopDistance() == editor.fontMetrics().horizontalAdvance(" ") * 4
        )
    assert not isinstance(dialog.preview_edit, AssistantDetailsEditor)
    assert dialog.preview_edit.isReadOnly()
    assert not isinstance(dialog.goal_edit, AssistantDetailsEditor)
    assert not isinstance(dialog.discussion_edit, AssistantDetailsEditor)
    assert assistant.captures == assistant.applies == assistant.runs == []


def test_details_follow_actual_parent_font_size_in_points_and_pixels(qtbot):
    parent = QWidget()
    layout = QVBoxLayout(parent)
    plain = QPlainTextEdit(_JSON)
    technical = AssistantDetailsEditor(_JSON, json_syntax=True)
    trace = AssistantDetailsEditor("Attempt prose remains literal.")
    for editor in (plain, technical, trace):
        layout.addWidget(editor)
    qtbot.addWidget(parent)
    parent.show()
    _settle(technical, qtbot)
    _settle(trace, qtbot)
    cursor = technical.textCursor()
    cursor.setPosition(2)
    cursor.setPosition(16, QTextCursor.KeepAnchor)
    technical.setTextCursor(cursor)
    cursor_before = _cursor(technical)
    changes = []
    technical.textChanged.connect(lambda: changes.append("editor"))
    technical.document().contentsChanged.connect(lambda: changes.append("document"))
    for unit, value in (("points", 16), ("pixels", 20)):
        font = QFont(parent.font())
        if unit == "points":
            font.setPointSize(value)
        else:
            font.setPixelSize(value)
        parent.setFont(font)
        _settle(technical, qtbot)
        _settle(trace, qtbot)
        for editor in (plain, technical, trace):
            actual = (
                editor.font().pointSize()
                if unit == "points"
                else editor.font().pixelSize()
            )
            assert actual == value
        for editor in (technical, trace):
            assert QFontInfo(editor.font()).fixedPitch()
            assert (
                editor.tabStopDistance()
                == editor.fontMetrics().horizontalAdvance(" ") * 4
            )
        assert technical.toPlainText() == plain.toPlainText() == _JSON
        assert trace.toPlainText() == "Attempt prose remains literal."
        assert _cursor(technical) == cursor_before
        assert changes == []


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_palette_font_and_resize_repaint_preserve_authority_and_cursor(
    assistant, qtbot, theme
):
    dialog = assistant.dialog
    dialog.prepare()  # Fixed no-AI recipe, never calculation or inference.
    dialog.trace_edit.setPlainText('Attempt 1: checked locally.\n{"literal": "🧪"}')
    dialog.discussion_edit.setPlainText("Unsent literal message.\nKeep this line.")
    _settle_dialog(dialog, qtbot)
    prepared = dialog._prepared
    assert prepared is not None and dialog.apply_button.isEnabled()
    cursor = dialog.json_edit.textCursor()
    cursor.setPosition(3)
    cursor.setPosition(18, QTextCursor.KeepAnchor)
    dialog.json_edit.setTextCursor(cursor)
    before, cursor_before = _state(dialog), _cursor(dialog.json_edit)
    captures = list(assistant.captures)
    assertions = list(assistant.assertions)
    graph = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    changes = []
    dialog.json_edit.textChanged.connect(lambda: changes.append("editor"))
    dialog.json_edit.document().contentsChanged.connect(
        lambda: changes.append("document")
    )
    font = QFont(dialog.font())
    font.setPointSize(16)
    dialog.setFont(font)
    # Existing assistant scoped styles do not propagate a dialog-only font
    # change to every child. Exercise this helper's actual parent contract,
    # without requiring an unrelated whole-window font-cascade redesign.
    dialog.json_edit.parentWidget().setStyleSheet("font-size: 16pt;")
    dialog._theme_probe.setPalette(_palette(theme))
    dialog._refresh_appearance()
    for width, height in ((760, 600), (1040, 820)):
        dialog.resize(width, height)
        dialog.advanced_checkbox.setChecked(True)
        dialog.tabs.setCurrentIndex(1)
        _settle_dialog(dialog, qtbot)
        assert _state(dialog) == before
        assert _cursor(dialog.json_edit) == cursor_before
        assert dialog._prepared is prepared
        assert dialog.json_edit.font().pointSize() == 16
        assert QFontInfo(dialog.json_edit.font()).fixedPitch()
    assert changes == []
    assert assistant.captures == captures and assistant.assertions == assertions
    assert assistant.applies == assistant.runs == []
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == graph
    assert np.array_equal(assistant.data, pixels) and not assistant.data.flags.writeable


def test_history_highlighting_does_not_restore_stale_plan_authority(assistant, qtbot):
    dialog = assistant.dialog
    dialog.prepare()
    first_json, first_prose = (
        dialog.json_edit.toPlainText(),
        dialog.preview_edit.toPlainText(),
    )
    raw = reference_proposal(dialog._context)
    raw["summary"] = "Second literal plan: café 🧪."
    dialog._accept_proposal(raw)
    _settle_dialog(dialog, qtbot)
    latest_json, latest_prose = (
        dialog.json_edit.toPlainText(),
        dialog.preview_edit.toPlainText(),
    )
    captures = list(assistant.captures)
    dialog.draft_combo.setCurrentIndex(0)
    assert dialog.json_edit.isReadOnly() and not dialog.apply_button.isEnabled()
    before = _state(dialog)
    dialog.json_edit.refresh_highlighting()
    _settle_dialog(dialog, qtbot)
    assert _state(dialog) == before
    assert dialog.json_edit.toPlainText() == first_json
    assert dialog.preview_edit.toPlainText() == first_prose
    dialog.draft_combo.setCurrentIndex(dialog._current_draft_index)
    assert not dialog.json_edit.isReadOnly() and dialog.apply_button.isEnabled()
    assert dialog.json_edit.toPlainText() == latest_json
    assert dialog.preview_edit.toPlainText() == latest_prose
    dialog.initiative_slider.setValue(1)
    assert not dialog.apply_button.isEnabled()
    assert dialog._prepared is None and not dialog._draft_authorized
    before = _state(dialog)
    dialog.json_edit.refresh_highlighting()
    _settle_dialog(dialog, qtbot)
    assert _state(dialog) == before
    assert not dialog.run_button.isEnabled()
    assert assistant.captures == captures and assistant.applies == assistant.runs == []


def test_highlighting_does_not_reset_pending_validation_or_authorize_invalid_json(
    assistant, qtbot
):
    dialog = assistant.dialog
    dialog.prepare()
    _settle_dialog(dialog, qtbot)
    captures = list(assistant.captures)
    previous_drafts = tuple(dialog._drafts)
    cursor = dialog.json_edit.textCursor()
    cursor.select(QTextCursor.Document)
    dialog.json_edit.setTextCursor(cursor)
    malformed = '{"unfinished": "Literal café 🧪'
    dialog.json_edit.insertPlainText(malformed)
    assert dialog._validation_timer.isActive()
    assert dialog._prepared is None
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    before = dialog._validation_timer.remainingTime()
    cursor_before = _cursor(dialog.json_edit)
    changes = []
    dialog.json_edit.textChanged.connect(lambda: changes.append("editor"))
    dialog.json_edit.refresh_highlighting()
    assert 0 <= dialog._validation_timer.remainingTime() <= before
    assert changes == []
    assert dialog.json_edit.toPlainText() == malformed
    assert _cursor(dialog.json_edit) == cursor_before
    qtbot.waitUntil(lambda: not dialog._validation_timer.isActive())
    assert dialog._prepared is None
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert dialog.json_edit.toPlainText() == malformed
    assert tuple(dialog._drafts) == previous_drafts
    assert assistant.captures == captures and assistant.applies == assistant.runs == []


def test_copy_and_sharing_preview_keep_exact_literal_technical_text(assistant, qtbot):
    dialog = assistant.dialog
    dialog.prepare()
    request = dialog._context.public_request(dialog._request_objective)
    dialog.request_edit.setPlainText(_JSON)
    dialog.trace_edit.setPlainText('Attempt prose stays plain.\n{"not": "parsed"}\n  ')
    _settle_dialog(dialog, qtbot)
    before = _state(dialog)
    captures = list(assistant.captures)
    clipboard = QApplication.clipboard()
    # Preserve every pre-existing MIME flavor even though this offscreen test
    # process has no interaction with a live assistant or provider.
    previous = QMimeData()
    old_mime = clipboard.mimeData()
    if old_mime is not None:
        for format_ in old_mime.formats():
            previous.setData(format_, old_mime.data(format_))
    try:
        for index in range(dialog.tabs.count()):
            dialog.tabs.setCurrentIndex(index)
            dialog.copy_button.click()
            assert clipboard.text() == dialog.tabs.widget(index).toPlainText()
    finally:
        clipboard.setMimeData(previous)
    assert _state(dialog) == before
    preview = dialog._sharing_preview_dialog(request)
    qtbot.addWidget(preview)
    details = preview.findChild(AssistantDetailsEditor, "SharedRequestJSON")
    assert details is not None and details.isReadOnly()
    assert details._json_highlighter is not None
    _settle(details, qtbot)
    assert details.toPlainText() == _JSON
    assert QFontInfo(details.font()).fixedPitch()
    assert _state(dialog) == before
    assert assistant.captures == captures and assistant.applies == assistant.runs == []
