"""Readable review lists preserve literal assumptions and scientific authority."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import (
    QColor,
    QFont,
    QPalette,
    QTextCursor,
    QTextDocument,
    QTextListFormat,
)

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import (
    ProposalError,
    reference_proposal,
    validate_proposal,
)
from napari_vipp.ui.workflow_assistant import _PLAN_REVIEW_CHECKS

_RAW_ASSUMPTIONS = (
    "  "
    + (
        "Do NOT fill missing boundaries unless observed evidence supports it. "
        "Keep incomplete acquisition edges open; no complete-object claim. "
    )
    * 4
    + "  ",
    "Unknown zz_option=±0.14 is a draft convention, not a calibrated fact.\n"
    "Only if source metadata confirms units may those units be used; "
    "never invent them.",
    'Literal <b>RAW_NOT_BOLD</b> & <a href="https://example.invalid">not a link</a>.\n'
    '<img src="file:///not-a-real-private-image.png"> must remain literal text.',
)


def _document(label, width=None):
    # Keep this owner alive while reading QTextBlocks/cursors returned from it.
    assert label.textFormat() == Qt.RichText and label.wordWrap()
    document = QTextDocument()
    document.setDefaultFont(label.font())
    document.setHtml(label.text())
    document.setTextWidth(width or max(1, label.width()))
    document.size()  # Force native line layout before inspecting wrapped bullets.
    return document


def _bullets(document):
    blocks = []
    block = document.begin()
    while block.isValid():
        if block.textList() is not None:
            blocks.append(block)
        block = block.next()
    return blocks


def _assert_disclosure_heading(button, document, heading):
    assert button.text() == heading
    assert button.font().bold()
    assert document.find(heading).isNull()  # The header is not duplicated in its body.


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


def _state(dialog):
    return (
        id(dialog._context),
        id(dialog._owner),
        id(dialog._prepared),
        dialog._generation,
        dialog._draft_authorized,
        dialog._added_ids,
        dialog._current_draft_index,
        dialog.draft_combo.currentIndex(),
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
        dialog._request_objective,
        dialog._context.public_request(dialog._request_objective),
        dialog.request_edit.toPlainText(),
        dialog.json_edit.toPlainText(),
        dialog.preview_edit.toPlainText(),
        tuple(dialog._chat),
        tuple(dialog._drafts),
        dialog.goal_edit.toPlainText(),
        dialog.discussion_edit.toPlainText(),
        dialog.goal_edit.textCursor().position(),
        dialog.goal_edit.textCursor().anchor(),
        dialog.images_checkbox.isChecked(),
        dialog.consent_checkbox.isChecked(),
        dialog._planning_preferences(),
    )


@pytest.fixture
def assumption_plan(assistant):
    dialog = assistant.dialog
    dialog.prepare()  # Fixed offline reference; no provider or calculation.
    raw = reference_proposal(dialog._context)
    raw["assumptions"] = list(_RAW_ASSUMPTIONS)
    dialog._accept_proposal(raw)
    assert dialog._prepared is not None and dialog._prepared.can_apply
    assert dialog._prepared.assumptions == tuple(
        item.strip() for item in _RAW_ASSUMPTIONS
    )
    return assistant, raw


@pytest.mark.parametrize("width", [760, 1040])
@pytest.mark.parametrize("points", [12, 16])
def test_true_hanging_lists_bold_headings_and_literal_assumptions_are_passive(
    assumption_plan, qtbot, monkeypatch, width, points
):
    assistant, raw = assumption_plan
    dialog = assistant.dialog
    dialog.assumptions_toggle.setChecked(True)
    dialog.review_checks_toggle.setChecked(True)
    cursor = dialog.goal_edit.textCursor()
    cursor.setPosition(3)
    cursor.setPosition(24, QTextCursor.KeepAnchor)
    dialog.goal_edit.setTextCursor(cursor)
    actions = []
    for method in ("prepare", "_send", "apply_proposal", "_run_added_plan"):
        monkeypatch.setattr(dialog, method, lambda: actions.append("action"))
    before = _state(dialog)
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
    font = QFont(dialog.font())
    font.setPointSize(points)
    dialog.setFont(font)
    dialog.resize(width, 800)
    for dark in (True, False):
        palette = QPalette(dialog._theme_probe.palette())
        background, foreground = (
            ("#202632", "#e9edf4") if dark else ("#f4f1ee", "#202126")
        )
        for role in (QPalette.Window, QPalette.Base, QPalette.Button):
            palette.setColor(role, QColor(background))
        for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
            palette.setColor(role, QColor(foreground))
        dialog._theme_probe.setPalette(palette)
        dialog._refresh_appearance()
        qtbot.waitUntil(lambda: not dialog._appearance_timer.isActive())
        qtbot.wait(10)
        assumptions = _document(dialog.assumptions_label, width=220)
        _assert_disclosure_heading(
            dialog.assumptions_toggle, assumptions, "Assumptions to check"
        )
        rows = _bullets(assumptions)
        assert len(rows) == len(_RAW_ASSUMPTIONS)
        wrapped = False
        for block, original in zip(rows, dialog._prepared.assumptions, strict=True):
            assert block.text().replace("\u2028", "\n") == original
            assert block.textList().format().style() == QTextListFormat.ListDisc
            assert block.textList().format().indent() >= 1
            layout = block.layout()
            if layout.lineCount() > 1:
                wrapped = True
                left = layout.lineAt(0).x()
                assert all(
                    abs(layout.lineAt(index).x() - left) <= 0.5
                    for index in range(1, layout.lineCount())
                )
        assert wrapped  # Continuation lines align with body text, not a typed bullet.
        literal = assumptions.find("RAW_NOT_BOLD")
        assert not literal.isNull() and literal.charFormat().fontWeight() < QFont.Bold
        _assert_literal_nonlinks(assumptions)
        assert not dialog.assumptions_label.openExternalLinks()
        assert dialog.assumptions_label.accessibleDescription() == "\n".join(
            dialog._prepared.assumptions
        )
        assert (
            dialog.assumptions_label.height()
            >= dialog.assumptions_label.heightForWidth(dialog.assumptions_label.width())
        )

        checks = _document(dialog.review_checks_label)
        _assert_disclosure_heading(
            dialog.review_checks_toggle, checks, "What to check after running"
        )
        assert [block.text() for block in _bullets(checks)] == list(_PLAN_REVIEW_CHECKS)
        assert len(_PLAN_REVIEW_CHECKS) == 3
        assert all(
            any(word in text.lower() for word in ("compare", "check"))
            for text in _PLAN_REVIEW_CHECKS
        )
        note = dialog.review_checks_note.text()
        assert dialog.review_checks_note.textFormat() == Qt.PlainText
        assert "general checks" in note and "not a complete assessment" in note
        assert "has not automatically assessed" in note
        assert "not confirmed facts" in note
        assert "AI's" not in note and "AI’s" not in note
        assert (
            dialog.review_checks_label.isVisible()
            and dialog.review_checks_note.isVisible()
        )
        assert (
            dialog.review_checks_label.accessibleDescription()
            == note + "\n" + "\n".join(_PLAN_REVIEW_CHECKS)
        )
        assert dialog.review_scroll.widget().isAncestorOf(dialog.review_checks_label)
        _assert_literal_nonlinks(checks)
    assert _state(dialog) == before
    assert json.loads(dialog.json_edit.toPlainText()) == raw
    assert "What to check after running" not in dialog.preview_edit.toPlainText()
    assert all(
        check not in dialog.preview_edit.toPlainText() for check in _PLAN_REVIEW_CHECKS
    )
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert (
        not assistant.data.flags.writeable
        and actions == assistant.applies == assistant.runs == []
    )


def test_historical_assumptions_and_advanced_raw_details_remain_exact_and_read_only(
    assumption_plan,
):
    assistant, first_raw = assumption_plan
    dialog = assistant.dialog
    first_index = dialog.draft_combo.currentIndex()
    first_draft = dialog._drafts[first_index]
    second_raw = reference_proposal(dialog._context)
    second_raw["summary"] = "A distinct later draft for review history."
    second_raw["assumptions"] = [
        "Do not infer units; unknown new_key=0.25 remains an explicit draft choice."
    ]
    dialog._accept_proposal(second_raw)
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
    dialog.draft_combo.setCurrentIndex(first_index)
    assert dialog.json_edit.isReadOnly()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert json.loads(dialog.json_edit.toPlainText()) == first_raw
    assert dialog.preview_edit.toPlainText() == first_draft.plan_text
    historical_document = _document(dialog.assumptions_label)
    assert [
        block.text().replace("\u2028", "\n") for block in _bullets(historical_document)
    ] == list(first_draft.assumptions)
    assert (
        "Assumptions to review\n" in first_draft.plan_text
    )  # Existing raw heading unchanged.
    for original in first_draft.assumptions:
        assert original in first_draft.plan_text
    assert all(check not in first_draft.plan_text for check in _PLAN_REVIEW_CHECKS)
    assert all(check not in first_draft.proposal_text for check in _PLAN_REVIEW_CHECKS)
    assert json.loads(first_draft.proposal_text)["assumptions"] == list(
        _RAW_ASSUMPTIONS
    )
    dialog.goal_edit.setPlainText("An altered goal must not reauthorize old history.")
    dialog.draft_combo.setCurrentIndex(len(dialog._drafts) - 1)
    assert dialog._prepared is None and not dialog._draft_authorized
    assert dialog.json_edit.isReadOnly()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert json.loads(dialog.json_edit.toPlainText()) == second_raw
    assert (
        dialog.assumptions_label.accessibleDescription() == second_raw["assumptions"][0]
    )
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert assistant.applies == assistant.runs == []


def test_empty_assumptions_clear_old_list_and_failed_review_clears_all_check_widgets(
    assumption_plan,
):
    assistant, _raw = assumption_plan
    dialog = assistant.dialog
    empty = reference_proposal(dialog._context)
    empty["assumptions"] = []
    dialog._accept_proposal(empty)
    assert not dialog.assumptions_label.isVisible()
    assert (
        dialog.assumptions_label.text()
        == dialog.assumptions_label.accessibleDescription()
        == ""
    )
    assert not dialog.assumptions_toggle.isVisible()
    assert not dialog.assumptions_content.isVisible()
    assert dialog.review_checks_toggle.isVisible()
    assert not dialog.review_checks_toggle.isChecked()
    assert not dialog.review_checks_content.isVisible()
    dialog.review_checks_toggle.setChecked(True)
    assert (
        dialog.review_checks_label.isVisible() and dialog.review_checks_note.isVisible()
    )
    checks_document = _document(dialog.review_checks_label)
    assert len(_bullets(checks_document)) == 3
    assert "Assumptions to review" not in dialog.preview_edit.toPlainText()
    assert json.loads(dialog.json_edit.toPlainText()) == empty
    dialog._accept_proposal({})
    assert dialog._prepared is None
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    for label in (dialog.review_checks_label, dialog.assumptions_label):
        assert not label.isVisible() and label.text() == ""
    assert not dialog.review_checks_note.isVisible()
    for toggle in (dialog.assumptions_toggle, dialog.review_checks_toggle):
        assert not toggle.isVisible() and not toggle.isChecked()
    assert assistant.applies == assistant.runs == []


def test_shared_future_plain_language_guidance_keeps_machine_contract_and_guards(
    assistant,
):
    dialog = assistant.dialog
    request = dialog._ensure_context()
    guidance = next(
        rule
        for rule in request["rules"]
        if rule.startswith("Write the summary, node reasons and assumptions")
    )
    for phrase in (
        "everyday language",
        "non-expert",
        "starting choice",
        "why it matters",
        "what the user can check",
        "before or after running",
        "internal parameter names",
        "exact machine settings in node params",
        "essential technical term",
        "uncertainty, negations, numerical qualifications",
        "scientific limitations",
        "do not invent observations",
        "output has already been assessed",
    ):
        assert phrase in guidance
    assert guidance in json.loads(json.dumps(request))["rules"]
    assert "supplied contract and catalogue" in local._SYSTEM
    assert "rules all apply" in codex._developer_instructions(request)
    raw = reference_proposal(dialog._context)
    assert set(raw) == {
        "kind",
        "version",
        "context_id",
        "summary",
        "assumptions",
        "questions",
        "nodes",
        "connections",
    }
    assert validate_proposal(raw, dialog._context).can_apply
    raw["questions"] = ["A genuinely unanswered question?"]
    assert not validate_proposal(raw, dialog._context).can_apply
    raw["questions"] = [f"Question {index}?" for index in range(4)]
    with pytest.raises(ProposalError, match="at most 3"):
        validate_proposal(raw, dialog._context)
    assert assistant.captures == [("input", False)]
    assert assistant.applies == assistant.runs == []
