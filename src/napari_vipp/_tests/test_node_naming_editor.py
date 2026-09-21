"""Inspector name drafts stay with their owning node and preserve plain text."""

from __future__ import annotations

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QTextDocument

from napari_vipp.core.node_names import MAX_NODE_NAME_LENGTH, normalize_node_name
from napari_vipp.ui.node_labels import NodePresentation
from napari_vipp.ui.node_naming import NodeNameEditor


def _presentation(automatic="area by Well", summary="Mean, SD · Grouped by Well"):
    return NodePresentation(automatic, "Statistics", summary, automatic)


@pytest.fixture
def editor(qtbot):
    widget = NodeNameEditor()
    qtbot.addWidget(widget)
    widget.resize(520, 180)
    widget.set_node("first", "", _presentation())
    widget.show()
    widget.activateWindow()
    widget.name_edit.setFocus()
    qtbot.waitUntil(widget.name_edit.hasFocus)
    return widget


def _record(editor):
    values = []
    editor.name_committed.connect(
        lambda node_id, value: values.append((node_id, value))
    )
    return values


def test_loaded_name_and_background_refresh_do_not_commit(editor):
    values = _record(editor)
    editor.set_node("first", "Well means", _presentation())
    editor.commit_pending()
    editor.set_node("first", "Well means", _presentation(summary="Median"))
    assert values == []
    assert editor.name_edit.text() == "Well means"
    assert editor.name_edit.placeholderText() == "area by Well"
    assert editor.operation_label.text() == "Operation: Statistics"
    assert editor.summary_label.text() == "Median"


def test_enter_commits_typed_name_once(editor, qtbot):
    values = _record(editor)
    qtbot.keyClicks(editor.name_edit, "Well means")
    assert values == []
    qtbot.keyClick(editor.name_edit, Qt.Key_Return)
    editor.commit_pending()
    editor.name_edit.clearFocus()
    assert values == [("first", "Well means")]
    assert not editor.name_edit.isModified()


def test_focus_loss_commits_to_original_node(editor, qtbot):
    values = _record(editor)
    qtbot.keyClicks(editor.name_edit, "Cell area")
    editor.name_edit.clearFocus()
    assert values == [("first", "Cell area")]


def test_reset_discards_uncommitted_draft_and_emits_automatic_name(editor, qtbot):
    values = _record(editor)
    assert not editor.reset_button.isEnabled()
    qtbot.keyClicks(editor.name_edit, "Draft")
    assert editor.reset_button.isEnabled()
    editor.reset_button.click()
    editor.commit_pending()
    assert values == [("first", "")]
    editor.set_node("first", "", _presentation())
    assert editor.name_edit.text() == ""
    assert not editor.reset_button.isEnabled()


def test_clearing_a_custom_name_commits_reset(editor, qtbot):
    values = _record(editor)
    editor.set_node("first", "Custom", _presentation())
    editor.name_edit.selectAll()
    qtbot.keyClick(editor.name_edit, Qt.Key_Backspace)
    editor.commit_pending()
    assert values == [("first", "")]


def test_same_node_refresh_preserves_focused_draft_cursor_and_selection(editor, qtbot):
    values = _record(editor)
    qtbot.keyClicks(editor.name_edit, "Draft for first")
    editor.name_edit.setSelection(6, 3)
    editor.set_node("first", "", _presentation(summary="New calculated context"))
    assert editor.name_edit.text() == "Draft for first"
    assert editor.name_edit.selectedText() == "for"
    assert editor.name_edit.selectionStart() == 6
    assert editor.name_edit.hasFocus()
    assert editor.name_edit.isModified()
    assert editor.summary_label.text() == "New calculated context"
    assert values == []


def test_owner_commits_before_switching_selection(editor, qtbot):
    values = _record(editor)
    qtbot.keyClicks(editor.name_edit, "First branch")
    editor.commit_pending()
    editor.clear()
    editor.set_node("second", "Second branch", _presentation("area distribution"))
    editor.commit_pending()
    assert values == [("first", "First branch")]
    assert editor.name_edit.text() == "Second branch"
    assert editor.name_edit.placeholderText() == "area distribution"


def test_owner_commit_before_same_id_in_another_workflow_prevents_leak(editor, qtbot):
    values = _record(editor)
    qtbot.keyClicks(editor.name_edit, "Source workflow name")
    editor.commit_pending()
    editor.set_node("first", "Destination workflow name", _presentation())
    editor.commit_pending()
    assert values == [("first", "Source workflow name")]
    assert editor.name_edit.text() == "Destination workflow name"


@pytest.mark.parametrize("clear_first", [False, True])
def test_passive_replacement_does_not_retarget_an_uncommitted_draft(
    editor, qtbot, clear_first
):
    values = _record(editor)
    qtbot.keyClicks(editor.name_edit, "Old draft")
    if clear_first:
        editor.clear()
    editor.set_node("second", "Other node", _presentation())
    editor.commit_pending()
    assert values == []
    assert editor.name_edit.text() == "Other node"


def test_clear_hides_editor_and_cannot_emit_a_late_commit(editor, qtbot):
    values = _record(editor)
    qtbot.keyClicks(editor.name_edit, "Draft")
    editor.clear()
    editor.commit_pending()
    editor.reset_button.click()
    assert values == []
    assert editor.isHidden()
    assert editor.name_edit.text() == ""


def test_markup_in_names_and_settings_remains_literal(editor):
    summary = "<b>area</b> < 100 & intensity > 2"
    presentation = NodePresentation(
        "<b>Cells</b>", "<i>Statistics</i>", summary, "Auto"
    )
    editor.set_node("first", "<b>Cells</b>", presentation)
    assert editor.name_edit.text() == "<b>Cells</b>"
    assert editor.operation_label.textFormat() == Qt.PlainText
    assert editor.operation_label.text() == "Operation: <i>Statistics</i>"
    assert editor.summary_label.textFormat() == Qt.PlainText
    assert editor.summary_label.text() == summary
    assert "&lt;b&gt;area&lt;/b&gt;" in editor.summary_label.toolTip()
    assert "&amp; intensity &gt; 2" in editor.summary_label.toolTip()


def test_long_summary_is_shortened_visibly_but_retained_in_tooltip(editor):
    summary = "Grouped measurements and exact settings " * 20
    editor.set_node("first", "", _presentation(summary=summary))
    assert len(editor.summary_label.text()) < len(summary)
    assert editor.summary_label.text().endswith("…")
    assert summary in editor.summary_label.toolTip()


@pytest.mark.parametrize(
    ("width", "operation"),
    [(230, "Statistics"), (210, "Grow Regions from Seeds — CellProfiler Propagation")],
)
def test_narrow_inspector_gives_name_full_width_and_shows_its_beginning(
    editor, qtbot, width, operation
):
    name = "Nuclear signal by treatment and independently sampled experimental wells"
    presentation = NodePresentation(name, operation, "Mean, SD", "area by Well")
    editor.set_node("first", name, presentation)
    editor.resize(width, 180)
    qtbot.waitUntil(lambda: editor.name_edit.width() == editor.width())

    assert editor.width() == width
    assert editor.name_edit.geometry().left() == 0
    assert (
        editor.name_edit.geometry().bottom() > editor.reset_button.geometry().bottom()
    )
    assert editor.reset_button.geometry().bottom() < editor.name_edit.geometry().top()
    assert (
        editor.name_edit.fontMetrics().horizontalAdvance(name)
        > editor.name_edit.width()
    )
    assert editor.name_edit.cursorPosition() == 0
    # Qt includes repaint padding around the caret, which can extend left of
    # the field. Its center must remain visible at the start of the text.
    assert 0 <= editor.name_edit.cursorRect().center().x() < 10
    assert editor.name_edit.text() == name
    assert editor.operation_label.width() <= width
    assert editor.operation_label.text() == f"Operation: {operation}"
    assert editor.rect().contains(editor.operation_label.geometry())
    tooltip = QTextDocument()
    tooltip.setHtml(editor.name_edit.toolTip())
    assert tooltip.toPlainText().startswith(name + "\n\n")


def test_narrow_inspector_summary_uses_three_lines_and_retains_full_context(
    editor, qtbot
):
    summary = (
        "Mean, SD · Measurements: nuclear_intensity, cell_intensity, "
        "cytoplasmic_intensity · Grouped by Well · Sample averages · "
        "Image identity: image_id · Sample identity: Well · "
        "Within sample: Equal objects · Equal weight per sample"
    )
    editor.set_node("first", "", _presentation(summary=summary))
    editor.resize(230, 180)
    qtbot.waitUntil(lambda: editor.summary_label.width() == 230)

    label = editor.summary_label
    assert label.height() <= 3 * label.fontMetrics().lineSpacing()
    assert label.text().endswith("…")
    assert len(label.text()) < len(summary)
    assert editor.rect().contains(label.geometry())
    assert label.accessibleDescription() == summary
    tooltip = QTextDocument()
    tooltip.setHtml(label.toolTip())
    assert tooltip.toPlainText() == summary
    narrow_text = label.text()

    editor.resize(460, 180)
    qtbot.waitUntil(lambda: editor.summary_label.width() == 460)
    assert len(label.text()) > len(narrow_text)
    assert label.height() <= 3 * label.fontMetrics().lineSpacing()
    editor.resize(230, 180)
    qtbot.waitUntil(lambda: editor.summary_label.width() == 230)
    assert label.text() == narrow_text
    assert label.accessibleDescription() == summary


def test_typing_enforces_documented_name_length(editor, qtbot):
    values = _record(editor)
    qtbot.keyClicks(editor.name_edit, "x" * (MAX_NODE_NAME_LENGTH + 1))
    editor.commit_pending()
    assert values == [("first", "x" * MAX_NODE_NAME_LENGTH)]
    assert editor.name_edit.maxLength() == MAX_NODE_NAME_LENGTH


def test_owner_can_reject_invalid_text_and_restore_prior_name(editor):
    rejected = []

    def accept(node_id, value):
        try:
            normalize_node_name(value)
        except ValueError as error:
            rejected.append(str(error))
            editor.set_node(node_id, "Previously saved", _presentation())

    editor.name_committed.connect(accept)
    editor.name_edit.setText("bad\u200bname")
    editor.name_edit.setModified(True)
    editor.name_edit.textEdited.emit(editor.name_edit.text())
    editor.commit_pending()
    assert rejected == ["Node name must be a single line without control characters."]
    assert editor.name_edit.text() == "Previously saved"
    assert not editor.name_edit.isModified()
