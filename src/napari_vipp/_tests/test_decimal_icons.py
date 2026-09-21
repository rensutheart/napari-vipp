"""Decimal-place toolbar glyphs stay distinct and legible in both themes."""

import pytest
from qtpy.QtGui import QColor, QFont, QIcon, QPalette

from napari_vipp.ui.iconography import _draw_icon_pixmap, interface_icon


def _palette(dark):
    palette = QPalette()
    palette.setColor(QPalette.Button, QColor("#414851" if dark else "#e7ebef"))
    palette.setColor(QPalette.ButtonText, QColor("#eef2f6" if dark else "#202530"))
    return palette


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("size", [24, 48])
def test_decimal_place_icons_are_distinct_palette_aware_and_unclipped(
    qapp, dark, size
):
    palette = _palette(dark)
    images = []
    for kind in ("increase-decimals", "decrease-decimals"):
        icon = interface_icon(kind, palette, 24)
        normal = icon.pixmap(size, size, QIcon.Normal).toImage()
        disabled = icon.pixmap(size, size, QIcon.Disabled).toImage()
        assert not normal.isNull()
        assert normal.size().width() == size
        assert normal != disabled
        # Fully covered glyph pixels use the host button foreground, and no
        # stroke reaches the bounding edge at standard or high-DPI resolution.
        foreground = palette.color(QPalette.ButtonText).rgba()
        pixels = [normal.pixel(x, y) for y in range(size) for x in range(size)]
        assert foreground in pixels
        for edge in range(size):
            assert normal.pixelColor(edge, 0).alpha() == 0
            assert normal.pixelColor(edge, size - 1).alpha() == 0
            assert normal.pixelColor(0, edge).alpha() == 0
            assert normal.pixelColor(size - 1, edge).alpha() == 0
        images.append(normal)
    assert images[0] != images[1]
    fallback = interface_icon("nodes", palette, 24).pixmap(size, size).toImage()
    assert all(image != fallback for image in images)


@pytest.mark.parametrize("kind", ["increase-decimals", "decrease-decimals"])
def test_decimal_digits_do_not_depend_on_application_font(qapp, kind):
    color = QColor("#eef2f6")
    before = _draw_icon_pixmap(kind, color, 24).toImage()
    original = QFont(qapp.font())
    try:
        qapp.setFont(QFont("An intentionally unavailable font", 32, QFont.Bold))
        after = _draw_icon_pixmap(kind, color, 24).toImage()
    finally:
        qapp.setFont(original)
    assert before == after
