"""Palette-derived colors shared by custom-painted VIPP controls."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtGui import QColor, QPalette


@dataclass(frozen=True)
class CustomPaintColors:
    """Theme colors for custom-painted surfaces such as plots and diagrams."""

    surface: QColor
    alternate_surface: QColor
    border: QColor
    axis: QColor
    muted_text: QColor
    text: QColor


@dataclass(frozen=True)
class SemanticToneColors:
    """Readable treatment for one semantic status on the active palette."""

    surface: QColor
    foreground: QColor
    border: QColor
    accent: QColor


@dataclass(frozen=True)
class ThemeColors:
    """Palette-derived neutral and semantic colors for styled Qt widgets."""

    surface: QColor
    alternate_surface: QColor
    raised_surface: QColor
    border: QColor
    text: QColor
    muted_text: QColor
    info: SemanticToneColors
    active_mode: SemanticToneColors
    success: SemanticToneColors
    warning: SemanticToneColors
    error: SemanticToneColors


def blend_colors(background: QColor, foreground: QColor, amount: float) -> QColor:
    """Blend ``foreground`` over ``background`` by ``amount`` in RGB space."""

    ratio = min(max(float(amount), 0.0), 1.0)
    inverse = 1.0 - ratio
    return QColor(
        round(background.red() * inverse + foreground.red() * ratio),
        round(background.green() * inverse + foreground.green() * ratio),
        round(background.blue() * inverse + foreground.blue() * ratio),
        round(background.alpha() * inverse + foreground.alpha() * ratio),
    )


def custom_paint_colors(palette: QPalette) -> CustomPaintColors:
    """Resolve readable custom-paint colors from the active Qt palette."""

    surface = QColor(palette.color(QPalette.Base))
    text = QColor(palette.color(QPalette.Text))
    alternate = QColor(palette.color(QPalette.AlternateBase))
    alternate_surface_distance = abs(
        alternate.lightnessF() - surface.lightnessF()
    )
    alternate_text_distance = abs(
        alternate.lightnessF() - text.lightnessF()
    )
    if (
        not alternate.isValid()
        or alternate == surface
        or alternate_surface_distance >= alternate_text_distance
    ):
        alternate = blend_colors(surface, text, 0.05)
    return CustomPaintColors(
        surface=surface,
        alternate_surface=alternate,
        border=blend_colors(surface, text, 0.26),
        axis=blend_colors(surface, text, 0.46),
        muted_text=blend_colors(surface, text, 0.62),
        text=text,
    )


def palette_is_dark(palette: QPalette) -> bool:
    """Return whether the palette uses lighter text over a darker base."""

    colors = custom_paint_colors(palette)
    return colors.surface.lightnessF() < colors.text.lightnessF()


def _semantic_tone(
    palette: QPalette,
    *,
    accent: str,
    dark_foreground: str,
    light_foreground: str,
) -> SemanticToneColors:
    colors = custom_paint_colors(palette)
    dark = palette_is_dark(palette)
    accent_color = QColor(accent)
    return SemanticToneColors(
        surface=blend_colors(colors.surface, accent_color, 0.18 if dark else 0.10),
        foreground=QColor(dark_foreground if dark else light_foreground),
        border=blend_colors(colors.surface, accent_color, 0.78 if dark else 0.72),
        accent=accent_color,
    )


def theme_colors(palette: QPalette) -> ThemeColors:
    """Resolve cohesive neutral and semantic colors from ``palette``."""

    colors = custom_paint_colors(palette)
    raised = blend_colors(colors.surface, colors.text, 0.08)
    return ThemeColors(
        surface=colors.surface,
        alternate_surface=colors.alternate_surface,
        raised_surface=raised,
        border=colors.border,
        text=colors.text,
        muted_text=colors.muted_text,
        info=_semantic_tone(
            palette,
            accent="#0ea5e9",
            dark_foreground="#bae6fd",
            light_foreground="#0c4a6e",
        ),
        active_mode=_semantic_tone(
            palette,
            accent="#8b5cf6",
            dark_foreground="#ede9fe",
            light_foreground="#4c1d95",
        ),
        success=_semantic_tone(
            palette,
            accent="#22c55e",
            dark_foreground="#bbf7d0",
            light_foreground="#14532d",
        ),
        warning=_semantic_tone(
            palette,
            accent="#f59e0b",
            dark_foreground="#fde68a",
            light_foreground="#713f12",
        ),
        error=_semantic_tone(
            palette,
            accent="#ef4444",
            dark_foreground="#fecaca",
            light_foreground="#7f1d1d",
        ),
    )


__all__ = [
    "CustomPaintColors",
    "SemanticToneColors",
    "ThemeColors",
    "blend_colors",
    "custom_paint_colors",
    "palette_is_dark",
    "theme_colors",
]
