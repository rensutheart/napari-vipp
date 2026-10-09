"""Review controls use recognizable, labelled, palette-aware vector glyphs."""

import pytest
from qtpy.QtGui import QColor, QIcon, QPalette

from napari_vipp.ui.iconography import interface_icon

_KINDS = (
    "side-by-side", "layers", "link", "axes", "fit-view", "reset", "mesh",
    "plane-xy", "plane-xz", "plane-yz",
)


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("size", [24, 48])
def test_review_glyphs_are_legible_distinct_and_palette_aware(qapp, dark, size):
    palette = QPalette()
    palette.setColor(QPalette.Button, QColor("#414851" if dark else "#e7ebef"))
    palette.setColor(QPalette.ButtonText, QColor("#eef2f6" if dark else "#202530"))
    fallback = interface_icon("nodes", palette, 24).pixmap(size, size).toImage()
    images = []
    for kind in _KINDS:
        icon = interface_icon(kind, palette, 24)
        normal = icon.pixmap(size, size, QIcon.Normal).toImage()
        disabled = icon.pixmap(size, size, QIcon.Disabled).toImage()
        assert not normal.isNull() and normal.width() == size
        assert normal != disabled and normal != fallback
        foreground = palette.color(QPalette.ButtonText).rgba()
        assert foreground in [
            normal.pixel(x, y) for y in range(size) for x in range(size)
        ]
        for edge in range(size):
            assert normal.pixelColor(edge, 0).alpha() == 0
            assert normal.pixelColor(edge, size - 1).alpha() == 0
            assert normal.pixelColor(0, edge).alpha() == 0
            assert normal.pixelColor(size - 1, edge).alpha() == 0
        assert all(normal != previous for previous in images)
        images.append(normal)


def test_review_glyphs_refresh_for_changed_palette(qapp):
    palette = QPalette()
    for kind in _KINDS:
        palette.setColor(QPalette.ButtonText, QColor("#202530"))
        light = interface_icon(kind, palette).pixmap(24, 24).toImage()
        palette.setColor(QPalette.ButtonText, QColor("#eef2f6"))
        dark = interface_icon(kind, palette).pixmap(24, 24).toImage()
        assert light != dark
