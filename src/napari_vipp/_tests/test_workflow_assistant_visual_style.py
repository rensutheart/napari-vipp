"""Native presentation checks; no provider or calculation is invoked."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from napari.qt import get_stylesheet
from qtpy.QtCore import QBuffer, QByteArray, QIODevice, QPoint
from qtpy.QtGui import QColor, QFont, QIcon, QPalette
from qtpy.QtWidgets import QLabel, QPushButton, QStyleFactory

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow_proposals import reference_proposal
from napari_vipp.ui import workflow_assistant
from napari_vipp.ui.toolbar_controls import toolbar_icon


@pytest.fixture(scope="session")
def qapp(qapp):
    from napari_vipp._tests._qt_fonts import load_offscreen_windows_fonts

    load_offscreen_windows_fonts(qapp)
    return qapp


def _icon_bytes(icon, mode=QIcon.Normal):
    buffer_data = QByteArray()
    buffer = QBuffer(buffer_data)
    assert buffer.open(QIODevice.WriteOnly)
    try:
        assert icon.pixmap(24, 24, mode).save(buffer, "PNG")
    finally:
        buffer.close()
    return bytes(buffer_data)


def _rect_in(widget, ancestor):
    rect = widget.rect()
    rect.moveTopLeft(widget.mapTo(ancestor, QPoint(0, 0)))
    return rect


def _luminance(colour):
    def linear(value):
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    channels = [linear(value) for value in colour.getRgbF()[:3]]
    return sum(
        weight * value
        for weight, value in zip((0.2126, 0.7152, 0.0722), channels, strict=True)
    )


def _contrast(first, second):
    bright, dark = sorted((_luminance(first), _luminance(second)), reverse=True)
    return (bright + 0.05) / (dark + 0.05)


def _assert_equal_tabs(dialog):
    tabs = dialog.stage_tabs
    bar = tabs.tabBar()
    rectangles = [bar.tabRect(index) for index in range(3)]
    widths = [rect.width() for rect in rectangles]
    assert min(widths) > 0
    assert max(widths) - min(widths) <= 1
    assert abs(rectangles[0].left()) <= 1
    assert abs(rectangles[-1].right() + 1 - bar.width()) <= 1
    assert abs(bar.width() - tabs.contentsRect().width()) <= 4
    for index, rect in enumerate(rectangles):
        assert rect.width() >= bar.fontMetrics().horizontalAdvance(tabs.tabText(index))


def test_assistant_uses_expanded_stage_tab_widget(assistant):
    assert type(assistant.dialog.stage_tabs).__name__ == "AssistantStageTabs"


@pytest.mark.parametrize("width", [760, 1040, 1280])
def test_three_stages_have_equal_expanded_widths_at_narrow_and_wide_sizes(
    assistant, qtbot, width
):
    dialog = assistant.dialog
    dialog.resize(width, 820)
    qtbot.waitUntil(lambda: dialog.stage_tabs.tabBar().width() > 0)
    qtbot.wait(10)
    assert dialog.width() == width
    _assert_equal_tabs(dialog)
    assert assistant.captures == assistant.applies == assistant.runs == []


def test_resize_round_trip_keeps_presentation_passive(assistant, qtbot, monkeypatch):
    dialog = assistant.dialog
    goal = dialog.goal_edit.toPlainText()
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    sends = []
    monkeypatch.setattr(dialog, "_send", lambda: sends.append(True))
    for width in (1040, 760, 1280, 760, 1040):
        dialog.resize(width, 820)
        qtbot.wait(10)
        _assert_equal_tabs(dialog)
        for stage in range(3):
            dialog.stage_tabs.setCurrentIndex(stage)
    assert dialog.goal_edit.toPlainText() == goal
    assert dialog._context is dialog._owner is dialog._prepared is None
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    assert np.array_equal(assistant.data, pixels)
    assert not assistant.data.flags.writeable
    assert sends == assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize(
    ("supported", "authenticated", "expected"),
    [
        (False, True, "disconnected"),
        (True, False, "disconnected"),
        (True, True, "connected"),
    ],
)
def test_badge_uses_confirmed_codex_status_not_selected_provider_alone(
    assistant, supported, authenticated, expected
):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
    dialog._connection_kind = "status"
    dialog._connection_ready(
        dialog._connection_generation,
        SimpleNamespace(
            supported=supported,
            authenticated=authenticated,
            detail="Synthetic connection result; no real check was run.",
        ),
    )
    assert dialog.provider_status_icon.property("status_state") == expected
    assert dialog._codex_authenticated is (supported and authenticated)
    assert dialog.provider_status_icon.pixmap() is not None
    assert not dialog.provider_status_icon.pixmap().isNull()
    assert not isinstance(dialog.provider_status_label, QPushButton)
    assert ("Not connected" in dialog.provider_status_label.text()) is (
        expected == "disconnected"
    )
    if expected == "connected":
        assert "Connected" in dialog.provider_status_label.text()
    assert assistant.captures == assistant.applies == assistant.runs == []


def test_checking_badge_does_not_announce_a_confirmed_connection(assistant):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData("codex"))
    dialog._codex_authenticated = False
    dialog._connection_busy = True
    dialog._connection_kind = "status"
    dialog._sync_buttons()
    assert dialog.provider_status_icon.property("status_state") == "checking"
    assert "Checking" in dialog.provider_status_label.text()
    assert "Connected" not in dialog.provider_status_label.text()
    assert not dialog.prepare_button.isEnabled()
    assert assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize("provider", ["local", "example"])
def test_non_cloud_badges_do_not_invent_connection_evidence(assistant, provider):
    dialog = assistant.dialog
    dialog.provider_combo.setCurrentIndex(dialog.provider_combo.findData(provider))
    dialog._sync_buttons()
    assert dialog.provider_status_icon.property("status_state") == provider
    assert "connected" not in dialog.provider_status_label.text().lower()
    if provider == "example":
        assert "No AI" in dialog.provider_status_label.text()
    assert assistant.captures == assistant.applies == assistant.runs == []


def test_settings_gear_is_isolated_from_existing_toolbar_icons(assistant):
    dialog = assistant.dialog
    # The fixture changes provider after construction; synchronize appearance
    # before comparing glyphs in the resulting (possibly disabled) palettes.
    dialog._refresh_appearance()
    palette = dialog._theme_probe.palette()
    expected_gear = toolbar_icon("settings", palette)
    assert _icon_bytes(dialog.settings_button.icon()) == _icon_bytes(expected_gear)
    assert _icon_bytes(dialog.settings_button.icon()) != _icon_bytes(
        toolbar_icon("setup", dialog.palette())
    )
    unchanged_icons = {
        "prepare_button": "send",
        "revise_button": "send",
        "send_button": "send",
        "apply_button": "checklist",
        "run_button": "next",
        "preview_button": "image",
        "discussion_preview_button": "image",
        "back_button": "previous",
        "next_button": "next",
    }
    for attribute, kind in unchanged_icons.items():
        button = getattr(dialog, attribute)
        assert _icon_bytes(button.icon()) == _icon_bytes(
            workflow_assistant.assistant_action_icon(
                kind,
                palette,
                primary=button.property("assistantAction") == "primary",
            )
        ), attribute
        assert _icon_bytes(button.icon(), QIcon.Disabled) == _icon_bytes(
            toolbar_icon(kind, palette), QIcon.Disabled
        ), attribute
    assert assistant.captures == assistant.applies == assistant.runs == []


@pytest.mark.parametrize("width", [760, 1040, 1280])
def test_footer_navigation_and_close_in_bottom_note(assistant, qtbot, width):
    dialog = assistant.dialog
    dialog.prepare()
    dialog.resize(width, 820)
    dialog.stage_tabs.setCurrentIndex(1)
    qtbot.wait(10)
    footer = _rect_in(dialog.footer_frame, dialog)
    back = _rect_in(dialog.back_button, dialog)
    next_button = _rect_in(dialog.next_button, dialog)
    stage_label = _rect_in(dialog.stage_label, dialog)
    assert footer.contains(back)
    assert footer.contains(next_button)
    assert back.left() - footer.left() <= 20
    assert back.right() < stage_label.left()
    assert stage_label.right() < next_button.left()
    assert abs(stage_label.center().x() - footer.center().x()) <= 12
    assert dialog.stage_label.text().split(" · ", 1)[0] == "Step 2 of 3"
    close_buttons = [
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Close"
    ]
    assert len(close_buttons) == 1
    close = _rect_in(close_buttons[0], dialog)
    bottom = _rect_in(dialog.bottom_frame, dialog)
    assert bottom.contains(close)
    assert close.top() > footer.bottom()
    assert bottom.right() - close.right() <= 20
    actions = _rect_in(dialog.footer_actions, dialog)
    assert footer.right() - actions.right() <= 20
    assert not close.intersects(next_button)
    assert not close_buttons[0].autoDefault()
    dialog.stage_tabs.setCurrentIndex(2)
    assert dialog.next_button.isHidden()
    assert not dialog.next_button.isEnabled()
    assert dialog.stage_label.text().split(" · ", 1)[0] == "Step 3 of 3"
    assert assistant.captures == [("input", False)]
    assert assistant.applies == assistant.runs == []


def test_style_and_stage_switching_cannot_run_or_apply_a_historical_draft(
    assistant, qtbot
):
    dialog = assistant.dialog
    dialog.prepare()
    dialog.apply_button.click()
    assert len(assistant.applies) == 1
    assert dialog.run_button.isEnabled()
    original = tuple(dialog._drafts)
    dialog.goal_edit.setPlainText("Review a fresh observed-object starting point.")
    dialog.prepare()
    assert dialog.draft_combo.count() == 2
    captures = list(assistant.captures)
    dialog.draft_combo.setCurrentIndex(0)
    for width in (760, 1280, 1040):
        dialog.resize(width, 820)
        qtbot.wait(10)
        _assert_equal_tabs(dialog)
        for stage in (0, 1, 2):
            dialog.stage_tabs.setCurrentIndex(stage)
    assert tuple(dialog._drafts[:1]) == original
    assert dialog.json_edit.isReadOnly()
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    dialog.apply_proposal()
    dialog._run_added_plan()
    assert len(assistant.applies) == 1
    assert assistant.runs == []
    assert assistant.captures == captures


def test_presentation_changes_do_not_authorize_a_pending_or_late_response(
    assistant, qtbot
):
    dialog = assistant.dialog
    dialog._ensure_context()
    response = SimpleNamespace(
        proposal=reference_proposal(dialog._context), attempts=()
    )
    dialog._busy = True
    dialog._job_kind = "proposal"
    dialog._draft_authorized = False
    generation = dialog._generation
    captures = list(assistant.captures)
    dialog._sync_buttons()
    for width in (760, 1280):
        dialog.resize(width, 820)
        qtbot.wait(10)
        dialog.stage_tabs.setCurrentIndex(2)
        dialog._update_provider_status()
        assert dialog._busy
        assert dialog._generation == generation
        assert not dialog.apply_button.isEnabled()
        assert not dialog.run_button.isEnabled()
    dialog.cancel_request()
    dialog._ready(generation, response)
    assert dialog._prepared is None
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert assistant.captures == captures
    assert assistant.applies == assistant.runs == []


@pytest.mark.parametrize("state", ["checked", "added", "pending"])
def test_palette_style_and_font_updates_neither_grant_nor_revoke_authority(
    assistant, qtbot, monkeypatch, state
):
    dialog = assistant.dialog
    dialog.prepare()
    if state == "added":
        dialog.apply_button.click()
    elif state == "pending":
        dialog._busy = True
        dialog._job_kind = "proposal"
        dialog._draft_authorized = False
        dialog._invalidate_proposal()
        dialog._sync_buttons()
    dialog._append_chat("You", "Keep the observed source and existing calibration.")
    authority = (
        dialog._generation,
        dialog._context,
        dialog._owner,
        dialog._prepared,
        dialog._draft_authorized,
        dialog._added_ids,
        dialog._busy,
        dialog._current_draft_index,
    )
    permissions = (
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
        dialog.consent_checkbox.isChecked(),
    )
    captures = list(assistant.captures)
    applies = list(assistant.applies)
    transcript = dialog.transcript_edit.toPlainText()
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    pixels = assistant.data.copy()
    sends = []
    monkeypatch.setattr(dialog, "_send", lambda: sends.append(True))
    palette = QPalette(dialog.palette())
    for role, colour in (
        (QPalette.Window, "#202632"),
        (QPalette.WindowText, "#e9edf4"),
        (QPalette.Base, "#151a22"),
        (QPalette.Text, "#e9edf4"),
        (QPalette.Button, "#303642"),
        (QPalette.ButtonText, "#e9edf4"),
    ):
        palette.setColor(role, QColor(colour))
    dialog.setPalette(palette)
    style = QStyleFactory.create("Fusion")
    assert style is not None
    # QWidget does not own a style supplied to setStyle. Retain it through the
    # deferred Qt teardown and any subsequent application-wide theme change.
    style.setParent(dialog)
    dialog._test_style = style
    dialog.setStyle(style)
    font = QFont(dialog.font())
    font.setPointSize(max(11, font.pointSize() + 1))
    dialog.setFont(font)
    dialog._refresh_appearance()
    dialog._refresh_appearance()
    qtbot.wait(10)

    assert (
        dialog._generation,
        dialog._context,
        dialog._owner,
        dialog._prepared,
        dialog._draft_authorized,
        dialog._added_ids,
        dialog._busy,
        dialog._current_draft_index,
    ) == authority
    assert (
        dialog.apply_button.isEnabled(),
        dialog.run_button.isEnabled(),
        dialog.consent_checkbox.isChecked(),
    ) == permissions
    assert not dialog._appearance_updating
    assert dialog.transcript_edit.toPlainText() == transcript
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    assert np.array_equal(assistant.data, pixels)
    assert assistant.captures == captures
    assert assistant.applies == applies
    assert sends == assistant.runs == []


def test_refine_button_navigates_to_discussion_without_sending_or_mutating(
    assistant, monkeypatch
):
    dialog = assistant.dialog
    dialog.prepare()
    assert dialog.stage_tabs.currentIndex() == 2
    assert dialog.refine_button.isVisible()
    assert not dialog.refine_button.autoDefault()
    prepared = dialog._prepared
    context, owner = dialog._context, dialog._owner
    history = tuple(dialog._drafts)
    captures = list(assistant.captures)
    generation = dialog._generation
    sends = []
    monkeypatch.setattr(dialog, "_send", lambda: sends.append(True))
    dialog.refine_button.click()
    assert dialog.stage_tabs.currentIndex() == 1
    assert dialog._prepared is prepared
    assert dialog._context is context and dialog._owner is owner
    assert tuple(dialog._drafts) == history
    assert dialog._generation == generation
    assert dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert assistant.captures == captures
    assert sends == assistant.applies == assistant.runs == []


def test_actual_napari_light_dark_light_surfaces_and_text_keep_plan_unchanged(
    assistant, qtbot, qapp, monkeypatch
):
    dialog = assistant.dialog
    dialog.prepare()
    dialog._append_chat("You", "Preserve the observed source and physical units.")
    prepared, context, owner = dialog._prepared, dialog._context, dialog._owner
    generation = dialog._generation
    drafts = tuple(dialog._drafts)
    captures = list(assistant.captures)
    transcript = dialog.transcript_edit.toPlainText()
    pixels = assistant.data.copy()
    snapshot = GraphSnapshot.from_pipeline(assistant.pipeline)
    sends = []
    monkeypatch.setattr(dialog, "_send", lambda: sends.append(True))
    original_stylesheet = qapp.styleSheet()
    palettes = []
    try:
        for theme in ("light", "dark", "light"):
            qapp.setStyleSheet(get_stylesheet(theme))
            qapp.processEvents()
            qtbot.waitUntil(lambda: not dialog._appearance_timer.isActive())
            dialog._theme_probe.ensurePolished()
            dialog.ensurePolished()
            panel = dialog._theme_probe.palette().color(QPalette.Window)
            foreground = dialog._theme_probe.palette().color(QPalette.WindowText)
            assert (_luminance(panel) > 0.4) is (theme == "light")
            assert _contrast(foreground, panel) >= 4.5
            assert f"background: {panel.name()}" in dialog.styleSheet()
            for label in (dialog.provider_status_label, dialog.summary_label):
                label.ensurePolished()
                assert (
                    _contrast(label.palette().color(QPalette.WindowText), panel) >= 4.5
                ), (theme, label.text())
            dialog.goal_edit.ensurePolished()
            assert (
                _contrast(
                    dialog.goal_edit.palette().color(QPalette.Text),
                    dialog.goal_edit.palette().color(QPalette.Base),
                )
                >= 4.5
            )
            palettes.append((panel.name(), foreground.name()))
            assert dialog._prepared is prepared
            assert dialog._context is context and dialog._owner is owner
            assert dialog._generation == generation
            assert dialog._draft_authorized
            assert dialog.apply_button.isEnabled()
            assert not dialog.run_button.isEnabled()
            assert tuple(dialog._drafts) == drafts
            assert dialog.transcript_edit.toPlainText() == transcript
        assert palettes[0] == palettes[2]
        assert palettes[0] != palettes[1]
    finally:
        qapp.setStyleSheet(original_stylesheet)
        qapp.processEvents()
    assert not dialog._appearance_updating
    assert GraphSnapshot.from_pipeline(assistant.pipeline) == snapshot
    assert np.array_equal(assistant.data, pixels)
    assert assistant.captures == captures
    assert sends == assistant.applies == assistant.runs == []


def test_default_geometry_keeps_goal_bounded_and_questions_readable_in_main_scroll(
    assistant, qtbot
):
    dialog = assistant.dialog
    dialog.resize(1040, 820)
    dialog.stage_tabs.setCurrentIndex(0)
    qtbot.wait(10)
    minimum, maximum = dialog.goal_edit.height_bounds()
    assert minimum <= dialog.goal_edit.height() <= maximum
    assert dialog.goal_edit.minimumHeight() == dialog.goal_edit.maximumHeight()
    dialog.prepare()
    raw = reference_proposal(dialog._context)
    raw["questions"] = [
        "Which observed structure should have a separate label?",
        "Should gaps remain unfilled rather than reconstructed?",
    ]
    dialog._accept_proposal(raw)
    qtbot.wait(10)
    assert dialog.stage_tabs.currentIndex() == 1
    assert dialog.questions_group.isVisible()
    assert not dialog.conversation_toggle.isChecked()
    assert dialog.transcript_edit.isHidden()
    dialog.conversation_toggle.setChecked(True)
    qtbot.waitUntil(lambda: not dialog.transcript_edit._fit_timer.isActive())
    qtbot.waitUntil(lambda: dialog.transcript_edit.verticalScrollBar().maximum() == 0)
    assert dialog.transcript_edit.verticalScrollBar().maximum() == 0
    viewport = dialog.discussion_scroll.viewport()
    for question, field in dialog._question_fields:
        label = next(
            label
            for label in dialog.questions_group.findChildren(QLabel)
            if label.text() == question
        )
        for widget in (label, field):
            dialog.discussion_scroll.ensureWidgetVisible(widget, 0, 12)
            qtbot.wait(10)
            assert viewport.rect().contains(_rect_in(widget, viewport)), question
    assert not dialog.apply_button.isEnabled()
    assert not dialog.run_button.isEnabled()
    assert assistant.captures == [("input", False)]
    assert assistant.applies == assistant.runs == []
