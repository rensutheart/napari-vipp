"""Assistant-local icon gaps preserve literal labels and passive actions."""

from __future__ import annotations

import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QPushButton, QStyle

from napari_vipp._tests.test_workflow_assistant_stages import assistant as assistant
from napari_vipp.ui.toolbar_controls import ToolbarCommandButton, toolbar_icon
from napari_vipp.ui.workflow_assistant import _AssistantSettingsDialog
from napari_vipp.ui.workflow_assistant_style import (
    AssistantButton,
    AssistantSendButton,
    SharingPreviewDialog,
)


def _painted_label(label, direction):
    return label + "\u2002" if direction == Qt.RightToLeft else "\u2002" + label


@pytest.mark.parametrize("button_type", [AssistantButton, AssistantSendButton])
@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
@pytest.mark.parametrize(
    ("label", "has_icon"),
    [("&Send goal", True), ("Close", False), ("", True)],
)
def test_icon_gap_is_directional_and_paint_only(
    qtbot, button_type, direction, label, has_icon
):
    button = button_type(label)
    qtbot.addWidget(button)
    button.setLayoutDirection(direction)
    button.setAccessibleName("Unchanged accessible action")
    if has_icon:
        button.setIcon(toolbar_icon("send"))

    option = button._toolbar_style_option()
    expected = _painted_label(label, direction) if label and has_icon else label
    assert option.text == expected
    assert option.direction == direction
    assert button.text() == button.property("text") == label
    assert button.accessibleName() == "Unchanged accessible action"
    assert button.styleSheet() == ""


@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
def test_text_only_buttons_have_native_size_and_no_synthetic_gap(qtbot, direction):
    button = AssistantButton("Close preview")
    reference = QPushButton("Close preview")
    for widget in (button, reference):
        qtbot.addWidget(widget)
        widget.setLayoutDirection(direction)
        widget.ensurePolished()
    assert button._toolbar_style_option().text == "Close preview"
    assert button.sizeHint() == reference.sizeHint()
    assert button.minimumSizeHint() == reference.minimumSizeHint()


@pytest.mark.parametrize("button_type", [AssistantButton, AssistantSendButton])
@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
@pytest.mark.parametrize("bold", [False, True])
def test_size_hints_measure_the_current_font_and_directional_gap(
    qtbot, button_type, direction, bold
):
    label = "Preview what is shared"
    button = button_type(label)
    reference = QPushButton(_painted_label(label, direction))
    for widget in (button, reference):
        qtbot.addWidget(widget)
        widget.setLayoutDirection(direction)
        widget.setIcon(toolbar_icon("image"))
        widget.setStyleSheet("QPushButton { padding: 7px 11px; }")

    widths = []
    for point_size in (10, 16):
        font = QFont(button.font())
        font.setPointSize(point_size)
        font.setBold(bold)
        for widget in (button, reference):
            widget.setFont(font)
            widget.ensurePolished()
        option = button._toolbar_style_option()
        assert option.text == _painted_label(label, direction)
        assert option.fontMetrics.horizontalAdvance("\u2002") > (
            option.fontMetrics.horizontalAdvance("\u2009")
        )
        assert button.sizeHint().width() >= reference.sizeHint().width()
        assert button.minimumSizeHint().width() >= reference.minimumSizeHint().width()
        widths.append(button.sizeHint().width())
    assert widths[1] > widths[0]
    assert button.text() == label


@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
def test_disabled_send_spinner_retains_gap_and_measured_width(qtbot, direction):
    button = AssistantSendButton("Sending goal…")
    qtbot.addWidget(button)
    button.setLayoutDirection(direction)
    button.set_idle_icon(toolbar_icon("send"))
    button.ensurePolished()
    idle_width = button.sizeHint().width()
    idle_option = button._toolbar_style_option()
    assert idle_option.text == _painted_label(button.text(), direction)

    button.setEnabled(False)
    button.set_sending(True)
    try:
        option = button._toolbar_style_option()
        assert option.text == idle_option.text
        assert not option.icon.isNull()
        assert not option.state & QStyle.State_Enabled
        assert button.sizeHint().width() == idle_width
        assert button.text() == "Sending goal…"
    finally:
        button.set_sending(False)
    assert button._toolbar_style_option().text == idle_option.text


def test_all_assistant_action_buttons_use_the_shared_local_helper(assistant, qtbot):
    dialog = assistant.dialog
    for name in (
        "prepare_button",
        "send_button",
        "revise_button",
        "settings_button",
        "apply_button",
        "run_button",
        "preview_button",
        "discussion_preview_button",
        "back_button",
        "next_button",
        "cancel_button",
        "close_button",
    ):
        button = getattr(dialog, name)
        assert isinstance(button, AssistantButton), name
        assert button.text() == button.property("text")
    assert all(
        isinstance(button, AssistantButton)
        for button in dialog.findChildren(QPushButton)
    )

    # Construct auxiliary dialogs passively: no modal loop, capture, or sign-in.
    settings = _AssistantSettingsDialog(dialog)
    preview = SharingPreviewDialog(dialog, dialog._theme_probe.palette())
    qtbot.addWidget(settings)
    qtbot.addWidget(preview)
    for window in (settings, preview):
        buttons = window.findChildren(QPushButton)
        assert buttons
        assert all(isinstance(button, AssistantButton) for button in buttons)
    assert assistant.captures == assistant.assertions == []
    assert assistant.applies == assistant.runs == []


def test_assistant_gap_does_not_change_unrelated_toolbar_buttons(qtbot):
    assistant_button = AssistantButton("Open")
    toolbar_button = ToolbarCommandButton("Open")
    for button in (assistant_button, toolbar_button):
        qtbot.addWidget(button)
        button.setIcon(toolbar_icon("open"))
    assert assistant_button._toolbar_style_option().text == "\u2002Open"
    assert toolbar_button._toolbar_style_option().text == "\u2009Open"
    assert assistant_button.text() == toolbar_button.text() == "Open"
