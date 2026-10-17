from __future__ import annotations

import pytest
from qtpy.QtCore import QByteArray, QSize, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from qtpy.QtSvg import QSvgRenderer
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
        "assistant",
        "archive",
        "setup",
        "checklist",
        "recheck_all",
        "next",
        "send",
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


@pytest.mark.parametrize("foreground", ("#e5e7eb", "#1f2937"))
def test_send_icon_preserves_official_codicons_svg_geometry(qapp, foreground):
    # Original Microsoft Codicons send.svg path, CC BY 4.0; see NOTICE.
    svg = (
        '<svg width="16" height="16" viewBox="0 0 16 16" '
        'xmlns="http://www.w3.org/2000/svg" '
        f'fill="{foreground}"><path d="'
        'M1.17683 1.11898C1.32953 0.989634 1.54464 0.963786 1.72363 1.05328'
        'L14.7236 7.55328C14.893 7.63797 15 7.8111 15 8.00049'
        'C15 8.18987 14.893 8.36301 14.7236 8.4477L1.72363 14.9477'
        'C1.54464 15.0372 1.32953 15.0113 1.17683 14.882'
        'C1.02414 14.7526 0.96328 14.5447 1.02213 14.3534'
        'L2.97688 8.00049L1.02213 1.64754'
        'C0.96328 1.45627 1.02414 1.24833 1.17683 1.11898Z'
        'M3.8693 8.50049L2.32155 13.5307L13.382 8.00049L2.32155 2.47027'
        'L3.8693 7.50049H9.50001C9.77615 7.50049 10 7.72435 10 8.00049'
        'C10 8.27663 9.77615 8.50049 9.50001 8.50049H3.8693Z"/></svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    assert renderer.isValid()
    expected = QPixmap(24, 24)
    expected.fill(Qt.transparent)
    painter = QPainter(expected)
    try:
        renderer.render(painter)
    finally:
        painter.end()
    assert _toolbar_icon_pixmap("send", foreground).toImage() == expected.toImage()


@pytest.mark.parametrize("foreground", ("#e5e7eb", "#1f2937"))
@pytest.mark.parametrize("size", (18, 24))
def test_send_icon_is_distinct_from_navigation_and_assistant(
    qapp, foreground, size
):
    palette = QPalette(qapp.palette())
    palette.setColor(QPalette.ButtonText, QColor(foreground))
    image = toolbar_icon("send", palette).pixmap(size, size).toImage()
    for kind in ("unknown", "next", "calculate", "assistant"):
        assert image != toolbar_icon(kind, palette).pixmap(size, size).toImage()
    assert any(
        image.pixelColor(x, y).alpha() > 200
        for y in range(size)
        for x in range(size)
    )


@pytest.mark.parametrize("foreground", ("#e5e7eb", "#1f2937"))
@pytest.mark.parametrize("size", (18, 24))
def test_assistant_sparkles_are_distinct_at_header_and_toolbar_sizes(
    qapp, foreground, size
):
    palette = QPalette(qapp.palette())
    palette.setColor(QPalette.ButtonText, QColor(foreground))
    image = toolbar_icon("assistant", palette).pixmap(size, size).toImage()
    for kind in ("unknown", "workflow", "arrange", "settings"):
        assert image != toolbar_icon(kind, palette).pixmap(size, size).toImage()

    # Separate sparkle silhouettes remain recognizable after downscaling.
    occupied = {
        (x, y)
        for y in range(size)
        for x in range(size)
        if image.pixelColor(x, y).alpha() > 40
    }
    component_sizes = []
    while occupied:
        pending = [occupied.pop()]
        count = 0
        while pending:
            x, y = pending.pop()
            count += 1
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    neighbor = (x + dx, y + dy)
                    if neighbor in occupied:
                        occupied.remove(neighbor)
                        pending.append(neighbor)
        component_sizes.append(count)
    assert len(component_sizes) == 3
    assert max(component_sizes) > 2 * min(component_sizes)


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
