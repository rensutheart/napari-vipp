from __future__ import annotations

import pytest
from qtpy.QtCore import QEvent, QObject, Signal
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication, QLabel, QWidget

from napari_vipp.ui.palette_roles import theme_colors
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
    assert ("background-color:" in strip.styleSheet()) is is_full_width_alert


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
    assert "background-color:" not in strip.styleSheet()
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


def _palette(*, base: str, text: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.Window, QColor(base))
    palette.setColor(QPalette.Text, QColor(text))
    palette.setColor(QPalette.WindowText, QColor(text))
    palette.setColor(QPalette.AlternateBase, QColor(base))
    return palette


@pytest.mark.parametrize(
    "severity",
    [
        MessageSeverity.NEUTRAL,
        MessageSeverity.INFO,
        MessageSeverity.SUCCESS,
        MessageSeverity.WARNING,
        MessageSeverity.ERROR,
    ],
)
def test_status_strip_restyles_live_for_dark_and_light_palettes(qtbot, severity):
    host = QWidget()
    host.setPalette(_palette(base="#111827", text="#f8fafc"))
    strip = StatusMessageStrip(parent=host)
    qtbot.addWidget(host)
    strip.show_message("Status", severity=severity)

    dark_style = strip.styleSheet()
    dark_colors = theme_colors(host.palette())
    dark_foreground = (
        dark_colors.muted_text
        if severity is MessageSeverity.NEUTRAL
        else getattr(dark_colors, severity.value).foreground
    )
    assert dark_foreground.name() in dark_style

    host.setPalette(_palette(base="#ffffff", text="#111827"))
    qtbot.waitUntil(lambda: strip.styleSheet() != dark_style)

    light_style = strip.styleSheet()
    light_colors = theme_colors(host.palette())
    light_foreground = (
        light_colors.muted_text
        if severity is MessageSeverity.NEUTRAL
        else getattr(light_colors, severity.value).foreground
    )
    assert light_foreground.name() in light_style
    assert light_style != dark_style
    assert strip.severity is severity


def test_status_strip_style_change_refresh_is_guarded_against_recursion(
    qtbot,
    monkeypatch,
):
    host = QWidget()
    strip = StatusMessageStrip(parent=host)
    qtbot.addWidget(host)
    calls = 0
    original = strip._apply_theme_style

    def tracked_apply_theme_style():
        nonlocal calls
        calls += 1
        original()

    monkeypatch.setattr(strip, "_apply_theme_style", tracked_apply_theme_style)

    QApplication.sendEvent(strip, QEvent(QEvent.StyleChange))

    assert calls == 1


def test_actionable_error_restyles_without_losing_alert_semantics(qtbot):
    host = QWidget()
    host.setPalette(_palette(base="#111827", text="#f8fafc"))
    strip = StatusMessageStrip(parent=host)
    qtbot.addWidget(host)
    strip.show_message("Fix the source", severity="error", actionable=True)
    dark_style = strip.styleSheet()

    host.setPalette(_palette(base="#ffffff", text="#111827"))
    strip.changeEvent(QEvent(QEvent.StyleChange))

    colors = theme_colors(host.palette()).error
    assert f"background-color: {colors.surface.name()}" in strip.styleSheet()
    assert f"color: {colors.foreground.name()}" in strip.styleSheet()
    assert strip.styleSheet() != dark_style
    assert strip.property("fullWidthAlert") is True
    assert strip.severity is MessageSeverity.ERROR
    assert strip.actionable
