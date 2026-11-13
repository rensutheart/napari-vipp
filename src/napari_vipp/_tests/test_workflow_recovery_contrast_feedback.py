"""Recovery checkbox contrast under inherited napari styles, without a store."""

import pytest
from napari._qt.qt_resources import get_stylesheet
from napari.utils.theme import get_theme
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QFont, QPalette
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QListWidgetItem,
    QVBoxLayout,
)

from napari_vipp.ui import workflow_recovery as recovery


def _palette(theme):
    selected = get_theme(theme)
    palette = QPalette()
    for role, name in (
        (QPalette.Window, "background"),
        (QPalette.Base, "background"),
        (QPalette.AlternateBase, "foreground"),
        (QPalette.WindowText, "text"),
        (QPalette.Text, "text"),
        (QPalette.Button, "primary"),
        (QPalette.ButtonText, "text"),
        (QPalette.Highlight, "current"),
        (QPalette.HighlightedText, "text"),
    ):
        color = QColor(getattr(selected, name).as_hex())
        assert color.isValid()
        palette.setColor(role, color)
    return palette


def _set_theme(host, theme, points):
    host.setPalette(_palette(theme))
    host.setFont(QFont("Arial", points))
    host.setStyleSheet(
        get_stylesheet(theme, extra_variables={"font_size": f"{points}pt"})
    )


def _sessions(qtbot, theme, points, direction=Qt.LeftToRight):
    host = QDialog()
    qtbot.addWidget(host)
    _set_theme(host, theme, points)
    layout = QVBoxLayout(host)
    choices = recovery._RecoverySessions()
    choices.setLayoutDirection(direction)
    layout.addWidget(choices)
    for number in range(3):
        item = QListWidgetItem(f"Synthetic session {number}\n2 workflow tabs")
        item.setData(Qt.UserRole, f"checkpoint-{number}")
        item.setData(recovery._SESSION_TITLE, f"Synthetic session {number}")
        item.setData(recovery._SESSION_DETAILS, "25 Oct 2026, 12:00 · 2 workflow tabs")
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Unchecked)
        choices.addItem(item)
    choices.setCurrentRow(0)
    host.resize(640, 380)
    host.show()
    qtbot.waitUntil(lambda: choices.visualItemRect(choices.item(2)).isValid())
    QApplication.processEvents()
    return host, choices


def _luminance(color):
    channels = [color.redF(), color.greenF(), color.blueF()]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return sum(
        value * weight
        for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
    )


def _contrast(left, right):
    values = sorted((_luminance(left), _luminance(right)))
    return (values[1] + 0.05) / (values[0] + 0.05)


def _pixel(image, x, y):
    ratio = image.devicePixelRatio()
    return image.pixelColor(round(x * ratio), round(y * ratio))


def _indicator_pixels(choices, row):
    rect = choices.check_indicator_rect(choices.model().index(row, 0))
    assert rect.isValid()
    image = choices.viewport().grab().toImage()
    return rect, [
        _pixel(image, x, y)
        for y in range(rect.top() + 2, rect.bottom() - 1)
        for x in range(rect.left() + 2, rect.right() - 1)
    ]


@pytest.mark.parametrize("theme", ["dark", "light"])
@pytest.mark.parametrize("points", [9, 16])
@pytest.mark.parametrize("direction", [Qt.LeftToRight, Qt.RightToLeft])
def test_recovery_checkboxes_are_high_contrast_with_real_napari_styles(
    qtbot, theme, points, direction
):
    host, choices = _sessions(qtbot, theme, points, direction)
    assert choices.selectionMode() == QAbstractItemView.NoSelection
    surface = host.palette().color(QPalette.Window)
    foreground = host.palette().color(QPalette.WindowText)
    assert _contrast(surface, foreground) >= 4.5
    image = choices.viewport().grab().toImage()
    background = _pixel(
        image, choices.viewport().width() // 2, choices.viewport().height() - 12
    )
    assert background == surface  # Do not inherit napari's lighter input Base.

    rect, unchecked = _indicator_pixels(choices, 1)
    whole = choices.viewport().grab().toImage()
    assert _pixel(whole, rect.center().x(), rect.center().y()) == surface
    border = [
        _pixel(whole, x, rect.top() + 1)
        for x in range(rect.left() + 3, rect.right() - 2)
    ]
    assert sum(_contrast(pixel, surface) >= 3 for pixel in border) >= 3
    choices.item(1).setCheckState(Qt.Checked)
    QApplication.processEvents()
    _rect, checked = _indicator_pixels(choices, 1)
    assert sum(_contrast(pixel, surface) >= 3 for pixel in checked) > sum(
        _contrast(pixel, surface) >= 3 for pixel in unchecked
    )
    assert sum(_contrast(pixel, foreground) >= 3 for pixel in checked) >= 3
    assert choices.item(1).checkState() == Qt.Checked
    assert choices.item(1).flags() & Qt.ItemIsUserCheckable
    assert [choices.item(row).data(Qt.UserRole) for row in range(3)] == [
        "checkpoint-0",
        "checkpoint-1",
        "checkpoint-2",
    ]


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_recovery_contrast_refresh_keeps_keyboard_checks_and_identity(qtbot, theme):
    host, choices = _sessions(qtbot, theme, 9)
    choices.setFocus()
    qtbot.keyClick(choices, Qt.Key_Space)
    qtbot.keyClick(choices, Qt.Key_Down)
    qtbot.keyClick(choices, Qt.Key_Space)
    assert [choices.item(row).checkState() for row in range(3)] == [
        Qt.Checked,
        Qt.Checked,
        Qt.Unchecked,
    ]
    assert choices.currentRow() == 1
    before = [choices.item(row).data(Qt.UserRole) for row in range(3)]
    replacement = "light" if theme == "dark" else "dark"
    _set_theme(host, replacement, 16)
    QApplication.processEvents()
    image = choices.viewport().grab().toImage()
    assert _pixel(
        image, choices.viewport().width() // 2, choices.viewport().height() - 12
    ) == host.palette().color(QPalette.Window)
    assert [choices.item(row).data(Qt.UserRole) for row in range(3)] == before
    assert [choices.item(row).checkState() for row in range(3)] == [
        Qt.Checked,
        Qt.Checked,
        Qt.Unchecked,
    ]
    qtbot.keyClick(choices, Qt.Key_Space)
    assert [choices.item(row).checkState() for row in range(3)] == [
        Qt.Checked,
        Qt.Unchecked,
        Qt.Unchecked,
    ]
    assert choices.currentRow() == 1
