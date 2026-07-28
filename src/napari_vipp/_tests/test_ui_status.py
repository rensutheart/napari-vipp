from __future__ import annotations

import pytest
from qtpy.QtCore import QObject, Signal
from qtpy.QtWidgets import QLabel

from napari_vipp.ui.status import MessageSeverity, StatusMessageStrip


class _StatusEmitter(QObject):
    message = Signal(str)


def test_status_message_strip_is_a_backward_compatible_label(qtbot):
    strip = StatusMessageStrip("Ready")
    qtbot.addWidget(strip)

    assert isinstance(strip, QLabel)
    assert strip.text() == "Ready"
    assert strip.wordWrap()
    assert strip.severity is MessageSeverity.NEUTRAL
    assert not strip.actionable
    assert strip.property("messageSeverity") == "neutral"
    assert strip.property("messageActionable") is False
    assert strip.property("fullWidthAlert") is False
    assert strip.accessibleName() == "Ready"


@pytest.mark.parametrize(
    ("severity", "actionable", "is_full_width_alert"),
    [
        (MessageSeverity.NEUTRAL, False, False),
        (MessageSeverity.NEUTRAL, True, False),
        (MessageSeverity.INFO, False, False),
        (MessageSeverity.INFO, True, False),
        (MessageSeverity.SUCCESS, False, False),
        (MessageSeverity.SUCCESS, True, False),
        (MessageSeverity.WARNING, False, False),
        (MessageSeverity.WARNING, True, False),
        (MessageSeverity.ERROR, False, False),
        (MessageSeverity.ERROR, True, True),
    ],
)
def test_only_actionable_errors_use_full_width_alert_presentation(
    qtbot,
    severity,
    actionable,
    is_full_width_alert,
):
    strip = StatusMessageStrip()
    qtbot.addWidget(strip)

    strip.show_message("Status detail", severity=severity, actionable=actionable)

    assert strip.text() == "Status detail"
    assert strip.severity is severity
    assert strip.actionable is actionable
    assert strip.property("fullWidthAlert") is is_full_width_alert
    assert ("background-color: #450a0a" in strip.styleSheet()) is (is_full_width_alert)


def test_show_message_accepts_a_string_severity(qtbot):
    strip = StatusMessageStrip()
    qtbot.addWidget(strip)

    strip.show_message("Saved", severity="success")

    assert strip.severity is MessageSeverity.SUCCESS
    assert strip.property("messageSeverity") == "success"
    assert "#22c55e" in strip.styleSheet()


def test_legacy_set_text_clears_an_actionable_error(qtbot):
    strip = StatusMessageStrip()
    qtbot.addWidget(strip)
    strip.show_message(
        "Pipeline failed",
        severity=MessageSeverity.ERROR,
        actionable=True,
    )

    strip.setText("Centered workflow graph.")

    assert strip.text() == "Centered workflow graph."
    assert strip.severity is MessageSeverity.NEUTRAL
    assert not strip.actionable
    assert strip.property("fullWidthAlert") is False
    assert "background-color: #450a0a" not in strip.styleSheet()
    assert strip.accessibleName() == "Centered workflow graph."


def test_legacy_signal_connection_resets_to_neutral(qtbot):
    strip = StatusMessageStrip()
    emitter = _StatusEmitter()
    qtbot.addWidget(strip)
    strip.show_message(
        "Image source failed",
        severity=MessageSeverity.ERROR,
        actionable=True,
    )
    emitter.message.connect(strip.setText)

    emitter.message.emit("Focused 'Input'.")

    assert strip.text() == "Focused 'Input'."
    assert strip.severity is MessageSeverity.NEUTRAL
    assert not strip.actionable
    assert strip.property("fullWidthAlert") is False
