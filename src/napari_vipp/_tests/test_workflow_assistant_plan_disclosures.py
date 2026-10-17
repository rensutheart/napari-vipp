"""Plan review disclosures save space without hiding scientific authority."""

from __future__ import annotations

import json

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPalette, QTextDocument

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core import workflow_proposal_codex as codex
from napari_vipp.core import workflow_proposal_model as local
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui.workflow_assistant import _PLAN_REVIEW_CHECKS
from napari_vipp.ui.workflow_assistant_style import AssistantDisclosureButton

_ASSUMPTIONS = (
    "Keep observed gaps open; do NOT infer missing structure. " * 4,
    'Literal <b>not bold</b> & <a href="https://example.invalid">not a link</a>.\n'
    '<img src="file:///not-a-real-image.png"> remains literal text.',
)


def _settle(dialog, qtbot):
    qtbot.waitUntil(lambda: not dialog._appearance_timer.isActive())
    qtbot.wait(10)


def _sections(dialog):
    return (
        (
            dialog.assumptions_frame,
            dialog.assumptions_toggle,
            dialog.assumptions_content,
            "Assumptions to check",
        ),
        (
            dialog.review_checks_frame,
            dialog.review_checks_toggle,
            dialog.review_checks_content,
            "What to check after running",
        ),
    )


def _authority(dialog):
    return (
        id(dialog._context),
        id(dialog._owner),
        id(dialog._prepared),
        dialog._generation,
        dialog._draft_authorized,
        dialog._added_ids,
        dialog._current_draft_index,
        dialog.draft_combo.currentIndex(),
        dialog._busy,
        dialog._request_objective,
        dialog.request_edit.toPlainText(),
        dialog.json_edit.toPlainText(),
        dialog.preview_edit.toPlainText(),
        tuple(dialog._chat),
        tuple(dialog._drafts),
        tuple(dialog._question_fields),
        dialog.goal_edit.toPlainText(),
        dialog.discussion_edit.toPlainText(),
        dialog.images_checkbox.isChecked(),
        dialog.consent_checkbox.isChecked(),
        dialog._planning_preferences(),
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
    )


@pytest.fixture
def review_plan(assistant, qtbot):
    dialog = assistant.dialog
    dialog.prepare()  # Fixed offline reference: no provider, Add or Run.
    raw = reference_proposal(dialog._context)
    raw["assumptions"] = list(_ASSUMPTIONS)
    dialog._accept_proposal(raw)
    assert dialog._prepared is not None and dialog._prepared.can_apply
    dialog.resize(760, 800)
    dialog.stage_tabs.setCurrentIndex(2)
    _settle(dialog, qtbot)
    return assistant, raw


def test_unpopulated_plan_hides_both_disclosures(assistant):
    dialog = assistant.dialog
    dialog.stage_tabs.setCurrentIndex(2)
    for frame, toggle, content, _caption in _sections(dialog):
        assert not frame.isVisible()
        assert not toggle.isVisible() and not toggle.isChecked()
        assert not content.isVisible()
    assert dialog.assumptions_label.text() == dialog.review_checks_label.text() == ""
    assert assistant.captures == assistant.applies == assistant.runs == []


def test_checked_plan_starts_with_compact_collapsed_headers(review_plan):
    assistant, _raw = review_plan
    dialog = assistant.dialog
    for frame, toggle, content, caption in _sections(dialog):
        assert frame.isVisible() and toggle.isVisible()
        assert isinstance(toggle, AssistantDisclosureButton)
        assert toggle.isCheckable() and not toggle.isChecked()
        assert toggle.text() == caption
        assert not toggle.isDefault() and not toggle.autoDefault()
        assert not content.isVisible()
        margins = frame.layout().contentsMargins()
        assert (margins.left(), margins.top(), margins.right(), margins.bottom()) == (
            14,
            0,
            14,
            0,
        )
        assert frame.layout().spacing() == 0
        assert toggle.height() >= max(34, toggle.fontMetrics().height() + 16)
        assert frame.height() <= toggle.height() + 4
    assert not dialog.assumptions_label.isVisible()
    assert not dialog.review_checks_label.isVisible()
    assert not dialog.review_checks_note.isVisible()
    assert dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()


def test_disclosures_expand_independently_with_chevrons_and_accessible_keyboard(
    review_plan, qtbot
):
    assistant, _raw = review_plan
    dialog = assistant.dialog
    sections = _sections(dialog)
    for index, (_frame, toggle, content, caption) in enumerate(sections):
        other = sections[1 - index][1]
        collapsed_icon = toggle.icon().pixmap(24, 24).toImage()
        assert not collapsed_icon.isNull()
        assert toggle.accessibleName() == f"Expand {caption}"
        assert toggle.toolTip() == f"Expand {caption.lower()}"
        dialog.review_scroll.ensureWidgetVisible(toggle)
        toggle.setFocus()
        for key in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            before = toggle.isChecked()
            qtbot.keyClick(toggle, key)
            _settle(dialog, qtbot)
            expanded = not before
            assert toggle.isChecked() is expanded
            assert content.isVisible() is expanded
            assert not other.isChecked()
            assert toggle.text() == caption
            assert toggle.accessibleName() == (
                f"{'Collapse' if expanded else 'Expand'} {caption}"
            )
            assert (
                toggle.icon().pixmap(24, 24).toImage() != collapsed_icon
            ) is expanded
            assert not dialog._closed
        assert not toggle.isChecked()
    assert assistant.applies == assistant.runs == []


def test_collapsed_content_retains_complete_escaped_lists_and_raw_draft(review_plan):
    assistant, raw = review_plan
    dialog = assistant.dialog
    for label, originals, heading in (
        (
            dialog.assumptions_label,
            dialog._prepared.assumptions,
            "Assumptions to check",
        ),
        (
            dialog.review_checks_label,
            _PLAN_REVIEW_CHECKS,
            "What to check after running",
        ),
    ):
        assert label.textFormat() == Qt.RichText and label.wordWrap()
        document = QTextDocument()
        document.setHtml(label.text())
        assert document.find(heading).isNull()
        blocks = []
        block = document.begin()
        while block.isValid():
            if block.textList() is not None:
                blocks.append(block.text().replace("\u2028", "\n"))
            fragments = block.begin()
            while not fragments.atEnd():
                fragment = fragments.fragment()
                if fragment.isValid():
                    assert not fragment.charFormat().isAnchor()
                    assert not fragment.charFormat().isImageFormat()
                fragments += 1
            block = block.next()
        assert blocks == list(originals)
        assert not label.openExternalLinks()
    assert json.loads(dialog.json_edit.toPlainText()) == raw
    assert all(
        item in dialog.preview_edit.toPlainText()
        for item in dialog._prepared.assumptions
    )
    assert all(
        item not in dialog.preview_edit.toPlainText() for item in _PLAN_REVIEW_CHECKS
    )


def test_same_draft_redisplay_retains_expansion_but_other_draft_resets(
    review_plan, qtbot
):
    assistant, first_raw = review_plan
    dialog = assistant.dialog
    first = dialog.draft_combo.currentIndex()
    dialog.assumptions_toggle.setChecked(True)
    dialog._draft_selected(first)
    _settle(dialog, qtbot)
    assert dialog.assumptions_toggle.isChecked()
    assert dialog.assumptions_content.isVisible()
    assert not dialog.review_checks_toggle.isChecked()
    later = reference_proposal(dialog._context)
    later["summary"] = "A distinct draft should start compact."
    later["assumptions"] = ["Check the second draft's observed boundaries."]
    dialog._accept_proposal(later)
    second = dialog.draft_combo.currentIndex()
    assert second != first
    for _frame, toggle, content, _caption in _sections(dialog):
        assert not toggle.isChecked() and not content.isVisible()
    dialog.review_checks_toggle.setChecked(True)
    dialog.draft_combo.setCurrentIndex(first)
    _settle(dialog, qtbot)
    assert json.loads(dialog.json_edit.toPlainText()) == first_raw
    for _frame, toggle, content, _caption in _sections(dialog):
        assert not toggle.isChecked() and not content.isVisible()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    dialog.assumptions_toggle.setChecked(True)
    dialog.review_checks_toggle.setChecked(True)
    dialog._draft_selected(first)
    assert (
        dialog.assumptions_toggle.isChecked()
        and dialog.review_checks_toggle.isChecked()
    )
    dialog.draft_combo.setCurrentIndex(second)
    assert not dialog.assumptions_toggle.isChecked()
    assert not dialog.review_checks_toggle.isChecked()
    assert assistant.applies == assistant.runs == []


def test_missing_assumptions_hide_only_their_section_and_questions_still_block(
    review_plan, qtbot
):
    assistant, _raw = review_plan
    dialog = assistant.dialog
    raw = reference_proposal(dialog._context)
    raw["assumptions"] = []
    raw["questions"] = ["Which observed structures must stay separate?"]
    dialog._accept_proposal(raw)
    assert dialog.stage_tabs.currentIndex() == 1
    assert dialog.questions_group.isVisible()
    assert [question for question, _field in dialog._question_fields] == raw[
        "questions"
    ]
    assert dialog._prepared is not None and not dialog._prepared.can_apply
    dialog.stage_tabs.setCurrentIndex(2)
    _settle(dialog, qtbot)
    assert not dialog.assumptions_frame.isVisible()
    assert not dialog.assumptions_toggle.isVisible()
    assert dialog.assumptions_label.text() == ""
    assert dialog.review_checks_toggle.isVisible()
    assert not dialog.review_checks_toggle.isChecked()
    dialog.review_checks_toggle.setChecked(True)
    assert dialog.review_checks_content.isVisible()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert [question for question, _field in dialog._question_fields] == raw[
        "questions"
    ]
    assert assistant.applies == assistant.runs == []


def test_distinct_history_drafts_with_identical_json_still_reset_expansion(
    review_plan, qtbot
):
    assistant, raw = review_plan
    dialog = assistant.dialog
    first = dialog.draft_combo.currentIndex()
    first_text = dialog.json_edit.toPlainText()
    dialog.assumptions_toggle.setChecked(True)
    dialog.review_checks_toggle.setChecked(True)
    # A fresh request revokes the earlier draft's authority even when the model
    # returns byte-for-byte identical proposal JSON. It is a new history entry.
    dialog._draft_authorized = False
    dialog._accept_proposal(raw)
    second = dialog.draft_combo.currentIndex()
    assert second != first
    assert dialog._drafts[first].proposal_text == first_text
    assert dialog._drafts[second].proposal_text == first_text
    for _frame, toggle, content, _caption in _sections(dialog):
        assert not toggle.isChecked() and not content.isVisible()
    dialog.assumptions_toggle.setChecked(True)
    dialog.review_checks_toggle.setChecked(True)
    dialog.draft_combo.setCurrentIndex(first)
    _settle(dialog, qtbot)
    assert dialog.json_edit.toPlainText() == first_text
    for _frame, toggle, content, _caption in _sections(dialog):
        assert not toggle.isChecked() and not content.isVisible()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    dialog.assumptions_toggle.setChecked(True)
    dialog.draft_combo.setCurrentIndex(second)
    _settle(dialog, qtbot)
    assert dialog.json_edit.toPlainText() == first_text
    assert not dialog.assumptions_toggle.isChecked()
    assert not dialog.review_checks_toggle.isChecked()
    assert assistant.applies == assistant.runs == []


def test_same_current_draft_local_validation_preserves_expansion(review_plan, qtbot):
    assistant, raw = review_plan
    dialog = assistant.dialog
    current = dialog.draft_combo.currentIndex()
    history = tuple(dialog._drafts)
    captures = list(assistant.captures)
    dialog.assumptions_toggle.setChecked(True)
    dialog.review_checks_toggle.setChecked(True)
    dialog.validate()  # Local schema/ownership recheck, never a fresh request.
    _settle(dialog, qtbot)
    assert dialog.draft_combo.currentIndex() == current
    assert tuple(dialog._drafts) == history
    assert json.loads(dialog.json_edit.toPlainText()) == raw
    assert dialog._prepared is not None and dialog._prepared.can_apply
    assert dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    for _frame, toggle, content, _caption in _sections(dialog):
        assert toggle.isChecked() and content.isVisible()
    assert assistant.captures == captures
    assert assistant.applies == assistant.runs == []


def test_first_proposal_keeps_expansion_during_revalidation_and_redisplay(
    assistant, qtbot
):
    dialog = assistant.dialog
    assert dialog.draft_combo.currentIndex() == -1 and not dialog._drafts
    dialog.prepare()  # Exercise the first addItem's implicit selection of index zero.
    assert dialog.draft_combo.currentIndex() == 0 and len(dialog._drafts) == 1
    assert dialog._prepared is not None and dialog._prepared.assumptions
    history = tuple(dialog._drafts)
    captures = list(assistant.captures)
    dialog.assumptions_toggle.setChecked(True)
    dialog.review_checks_toggle.setChecked(True)
    dialog.validate()
    _settle(dialog, qtbot)
    for _frame, toggle, content, _caption in _sections(dialog):
        assert toggle.isChecked() and content.isVisible()
    dialog._draft_selected(0)
    _settle(dialog, qtbot)
    for _frame, toggle, content, _caption in _sections(dialog):
        assert toggle.isChecked() and content.isVisible()
    assert tuple(dialog._drafts) == history
    assert dialog.draft_combo.currentIndex() == dialog._current_draft_index == 0
    assert dialog._prepared is not None and dialog._prepared.can_apply
    assert dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert assistant.captures == captures
    assert assistant.applies == assistant.runs == []


def test_invalid_proposal_clears_expanded_review_disclosures(review_plan):
    assistant, _raw = review_plan
    dialog = assistant.dialog
    dialog.assumptions_toggle.setChecked(True)
    dialog.review_checks_toggle.setChecked(True)
    dialog._accept_proposal({})
    assert dialog._prepared is None
    for frame, toggle, content, _caption in _sections(dialog):
        assert not frame.isVisible() and not toggle.isChecked()
        assert not content.isVisible()
    for label in (dialog.assumptions_label, dialog.review_checks_label):
        assert label.text() == label.accessibleDescription() == ""
    assert not dialog.review_checks_note.isVisible()
    assert not dialog.apply_button.isEnabled() and not dialog.run_button.isEnabled()
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("width,points,dark", [(760, 16, True), (1040, 12, False)])
def test_plan_disclosures_and_appearance_never_send_apply_run_or_mutate_authority(
    review_plan, qtbot, monkeypatch, width, points, dark
):
    assistant, _raw = review_plan
    dialog = assistant.dialog
    calls = []

    def unexpected(*_args, **_kwargs):
        calls.append("unexpected action")
        raise AssertionError("A passive disclosure must not act on a plan or provider")

    for method in (
        "prepare",
        "_send",
        "revise_plan",
        "apply_proposal",
        "_run_added_plan",
    ):
        monkeypatch.setattr(dialog, method, unexpected)
    monkeypatch.setattr(codex, "generate_codex_proposal", unexpected)
    monkeypatch.setattr(codex, "repair_codex_profile", unexpected)
    monkeypatch.setattr(codex, "start_codex_login", unexpected)
    monkeypatch.setattr(local, "generate_proposal", unexpected)
    before = _authority(dialog)
    captures, assertions = list(assistant.captures), list(assistant.assertions)
    snapshot, pixels = (
        GraphSnapshot.from_pipeline(assistant.pipeline),
        assistant.data.copy(),
    )
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
    for _frame, toggle, content, _caption in _sections(dialog):
        for expanded in (True, False):
            toggle.setChecked(expanded)
            _settle(dialog, qtbot)
            assert content.isVisible() is expanded
    assert _authority(dialog) == before
    assert assistant.captures == captures and assistant.assertions == assertions
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    np.testing.assert_array_equal(assistant.data, pixels)
    assert not assistant.data.flags.writeable
    assert calls == assistant.applies == assistant.runs == []
