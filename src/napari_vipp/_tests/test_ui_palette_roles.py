from __future__ import annotations

import pytest
from qtpy.QtGui import QColor, QPalette

from napari_vipp.ui.palette_roles import (
    blend_colors,
    custom_paint_colors,
    palette_is_dark,
    theme_colors,
)


def _palette(*, base: str, text: str, alternate: str | None = None) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.Base, QColor(base))
    palette.setColor(QPalette.Text, QColor(text))
    if alternate is not None:
        palette.setColor(QPalette.AlternateBase, QColor(alternate))
    return palette


@pytest.mark.parametrize(
    ("base", "text", "dark"),
    [
        ("#111827", "#f8fafc", True),
        ("#ffffff", "#111827", False),
    ],
)
def test_custom_paint_colors_follow_dark_and_light_palettes(base, text, dark):
    palette = _palette(base=base, text=text, alternate=base)

    colors = custom_paint_colors(palette)

    assert colors.surface == QColor(base)
    assert colors.text == QColor(text)
    assert colors.alternate_surface != colors.surface
    assert colors.border != colors.surface
    assert colors.axis != colors.surface
    assert colors.muted_text != colors.surface
    assert palette_is_dark(palette) is dark


def test_blend_colors_clamps_amount_and_preserves_endpoints():
    background = QColor("#102030")
    foreground = QColor("#f0e0d0")

    assert blend_colors(background, foreground, -1) == background
    assert blend_colors(background, foreground, 2) == foreground
    assert blend_colors(background, foreground, 0.5) == QColor("#808080")


def test_custom_paint_colors_reject_stale_light_alternate_for_dark_palette():
    palette = _palette(
        base="#23242b",
        text="#f0f1f2",
        alternate="#f7f7f7",
    )

    colors = custom_paint_colors(palette)

    assert colors.alternate_surface == blend_colors(
        QColor("#23242b"),
        QColor("#f0f1f2"),
        0.05,
    )
    assert colors.alternate_surface.lightnessF() < colors.text.lightnessF()


@pytest.mark.parametrize(
    ("base", "text"),
    [("#111827", "#f8fafc"), ("#ffffff", "#111827")],
)
def test_theme_colors_keep_semantic_surfaces_distinct(base, text):
    palette = _palette(base=base, text=text, alternate=base)

    colors = theme_colors(palette)

    assert colors.surface == QColor(base)
    assert colors.text == QColor(text)
    assert colors.raised_surface != colors.surface
    for tone in (
        colors.info,
        colors.active_mode,
        colors.success,
        colors.warning,
        colors.error,
    ):
        assert tone.surface != colors.surface
        assert tone.foreground != tone.surface
        assert tone.border != tone.surface
    assert colors.active_mode.accent != colors.warning.accent
    assert colors.active_mode.surface != colors.warning.surface
