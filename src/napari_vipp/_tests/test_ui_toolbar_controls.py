from __future__ import annotations

import pytest
from qtpy.QtCore import QSize
from qtpy.QtGui import QColor, QIcon, QPalette
from qtpy.QtWidgets import QPushButton

from napari_vipp.ui.toolbar_controls import (
    ToolbarCommandButton,
    _toolbar_icon_pixmap,
    toolbar_icon,
)


@pytest.mark.parametrize(
    "kind",
    (
        "redo",
        "reset",
        "refresh",
        "undo",
        "new",
        "open",
        "save",
        "batch",
        "columns",
        "edit",
        "select_all",
        "deselect",
        "previous",
        "image",
        "images",
        "destination",
        "workflow",
        "archive",
        "setup",
        "checklist",
        "recheck_all",
        "next",
        "compute",
        "search",
        "preview",
        "calculate",
        "optimize",
        "settings",
        "focus",
        "arrange",
        "tunnels",
        "activity",
        "stop",
        "more",
    ),
)
def test_toolbar_icons_use_normal_and_disabled_palette_colors(qapp, kind):
    palette = QPalette(qapp.palette())
    palette.setColor(QPalette.ButtonText, QColor("#173f91"))
    palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor("#9a647e"))

    icon = toolbar_icon(kind, palette)

    for mode, expected in (
        (QIcon.Normal, "#173f91"),
        (QIcon.Disabled, "#9a647e"),
    ):
        image = icon.pixmap(QSize(24, 24), mode).toImage()
        assert image.size() == QSize(24, 24)
        assert image.pixelColor(0, 0).alpha() == 0
        opaque_colors = {
            image.pixelColor(x, y).name()
            for y in range(image.height())
            for x in range(image.width())
            if image.pixelColor(x, y).alpha() == 255
        }
        assert opaque_colors == {expected}


@pytest.mark.parametrize("foreground", ("#e5e7eb", "#1f2937"))
@pytest.mark.parametrize("size", (18, 24))
def test_batch_toolbar_glyphs_are_distinct_in_light_and_dark_themes(
    qapp, foreground, size
):
    palette = QPalette(qapp.palette())
    palette.setColor(QPalette.ButtonText, QColor(foreground))
    fallback = toolbar_icon("unknown", palette).pixmap(size, size).toImage()
    images = []
    for kind in (
        "setup",
        "checklist",
        "recheck_all",
        "next",
        "compute",
        "search",
        "refresh",
    ):
        image = toolbar_icon(kind, palette).pixmap(size, size).toImage()
        assert image != fallback
        assert all(image != previous for previous in images)
        assert any(
            image.pixelColor(x, y).alpha() > 200
            for y in range(size)
            for x in range(size)
        )
        images.append(image)


def test_toolbar_icon_defaults_to_application_palette(qapp):
    assert (
        toolbar_icon("open").pixmap(24, 24).toImage()
        == toolbar_icon("open", qapp.palette()).pixmap(24, 24).toImage()
    )


def test_more_icon_has_three_separate_dots_in_one_horizontal_band(qapp):
    image = _toolbar_icon_pixmap("more", "#173f91").toImage()
    occupied_columns = {
        x
        for y in range(image.height())
        for x in range(image.width())
        if image.pixelColor(x, y).alpha() > 0
    }
    assert sum(x - 1 not in occupied_columns for x in occupied_columns) == 3
    for y in (0, 5, 18, 23):
        assert all(image.pixelColor(x, y).alpha() == 0 for x in range(24))


@pytest.mark.parametrize(
    ("text", "kind", "painted_text"),
    (
        ("Open", "open", "\u2009Open"),
        ("Open", None, "Open"),
        ("", "open", ""),
    ),
)
def test_command_button_adds_only_visual_icon_label_spacing(
    qtbot, text, kind, painted_text
):
    button = ToolbarCommandButton(text)
    qtbot.addWidget(button)
    if kind is not None:
        button.setIcon(toolbar_icon(kind))

    assert button._toolbar_style_option().text == painted_text
    assert button.text() == text
    assert button.property("text") == text


def test_main_toolbar_preserves_shared_helper_aliases(qapp):
    from napari_vipp._widget import (
        _toolbar_icon,
        _ToolbarChevronButton,
        _ToolbarCommandButton,
    )

    assert _ToolbarCommandButton is ToolbarCommandButton
    assert _toolbar_icon is toolbar_icon
    assert issubclass(_ToolbarChevronButton, ToolbarCommandButton)


@pytest.mark.parametrize("text", ["Review items", "View run report", "Open"])
@pytest.mark.parametrize("bold", [False, True])
def test_command_button_sizes_the_same_text_it_paints(qtbot, text, bold):
    button = ToolbarCommandButton(text)
    reference = QPushButton(ToolbarCommandButton._ICON_TEXT_SPACER + text)
    for widget in (button, reference):
        qtbot.addWidget(widget)
        widget.setIcon(toolbar_icon("activity"))
        widget.setStyleSheet(
            "QPushButton {padding: 6px 12px; font-size: 12pt; "
            f"font-weight: {'bold' if bold else 'normal'};}}"
        )
        widget.ensurePolished()
    assert button.sizeHint().width() >= reference.sizeHint().width()
    assert button.minimumSizeHint().width() >= reference.minimumSizeHint().width()
