"""Theme-aware slider styling shared by all VIPP controls."""

from __future__ import annotations

from dataclasses import dataclass

from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QSlider

from napari_vipp.ui.palette_roles import (
    blend_colors,
    palette_is_dark,
    theme_colors,
)


@dataclass(frozen=True)
class SliderColors:
    """Palette-derived colors for native and custom VIPP sliders."""

    groove: QColor
    groove_border: QColor
    fill: QColor
    handle: QColor
    handle_border: QColor
    hover: QColor
    pressed: QColor
    disabled: QColor


def slider_colors(palette: QPalette) -> SliderColors:
    """Return a readable blue slider treatment for the active theme."""

    colors = theme_colors(palette)
    dark = palette_is_dark(palette)
    # Match napari 0.9's built-in ``current`` colors so VIPP sliders belong to
    # the same visual system as the viewer's dimension controls.
    accent = QColor("#4560c4" if dark else "#a0b8ff")
    return SliderColors(
        groove=blend_colors(
            colors.surface,
            colors.text,
            0.22 if dark else 0.14,
        ),
        groove_border=blend_colors(
            colors.surface,
            colors.text,
            0.34 if dark else 0.28,
        ),
        fill=accent,
        handle=accent,
        handle_border=accent,
        hover=blend_colors(accent, colors.text, 0.28),
        pressed=blend_colors(accent, colors.surface, 0.18),
        disabled=blend_colors(colors.surface, colors.text, 0.28),
    )


def slider_style_sheet(palette: QPalette) -> str:
    """Build the shared stylesheet used by every native VIPP slider."""

    colors = slider_colors(palette)
    groove = colors.groove.name()
    groove_border = colors.groove_border.name()
    fill = colors.fill.name()
    handle = colors.handle.name()
    handle_border = colors.handle_border.name()
    hover = colors.hover.name()
    pressed = colors.pressed.name()
    disabled = colors.disabled.name()
    return (
        "QSlider::groove:horizontal {"
        f" background: {groove}; border: 0px solid {groove_border};"
        " height: 6px; border-radius: 3px;"
        " }"
        "QSlider::sub-page:horizontal {"
        f" background: {fill}; border-radius: 3px;"
        " }"
        "QSlider::add-page:horizontal {"
        f" background: {groove}; border-radius: 3px;"
        " }"
        "QSlider::handle:horizontal {"
        f" background: {handle}; border: 0px solid {handle_border};"
        " width: 12px; margin: -3px 0; border-radius: 6px;"
        " }"
        "QSlider::handle:horizontal:hover {"
        f" background: {hover};"
        " }"
        "QSlider::handle:horizontal:pressed {"
        f" background: {pressed};"
        " }"
        "QSlider::sub-page:horizontal:disabled,"
        " QSlider::handle:horizontal:disabled {"
        f" background: {disabled}; border-color: {groove_border};"
        " }"
        "QSlider::groove:vertical {"
        f" background: {groove}; border: 0px solid {groove_border};"
        " width: 6px; border-radius: 3px;"
        " }"
        "QSlider::sub-page:vertical {"
        f" background: {groove}; border-radius: 3px;"
        " }"
        "QSlider::add-page:vertical {"
        f" background: {fill}; border-radius: 3px;"
        " }"
        "QSlider::handle:vertical {"
        f" background: {handle}; border: 0px solid {handle_border};"
        " height: 12px; margin: 0 -3px; border-radius: 6px;"
        " }"
        "QSlider::handle:vertical:hover {"
        f" background: {hover};"
        " }"
        "QSlider::handle:vertical:pressed {"
        f" background: {pressed};"
        " }"
        "QSlider::add-page:vertical:disabled,"
        " QSlider::handle:vertical:disabled {"
        f" background: {disabled}; border-color: {groove_border};"
        " }"
    )


class VippSlider(QSlider):
    """A native slider that automatically follows VIPP's active palette."""

    def __init__(self, orientation=Qt.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setMinimumHeight(14)
        self._applying_vipp_slider_style = False
        self._theme_owner = None
        self._watch_theme_owner()
        self.refresh_theme()

    def event(self, event) -> bool:
        handled = super().event(event)
        if (
            hasattr(self, "_theme_owner")
            and event.type() == QEvent.ParentChange
        ):
            self._watch_theme_owner()
            self.refresh_theme()
        return handled

    def eventFilter(self, watched, event) -> bool:  # noqa: N802
        if (
            watched is self._theme_owner
            and event.type() in (QEvent.PaletteChange, QEvent.StyleChange)
        ):
            self.refresh_theme(QPalette(watched.palette()))
        return super().eventFilter(watched, event)

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        if (
            not self._applying_vipp_slider_style
            and event.type() in (QEvent.PaletteChange, QEvent.StyleChange)
        ):
            self.refresh_theme()

    def showEvent(self, event) -> None:  # noqa: N802
        self._watch_theme_owner()
        self.refresh_theme()
        super().showEvent(event)

    def refresh_theme(self, palette: QPalette | None = None) -> None:
        """Refresh the slider from ``palette`` or its inherited palette."""

        if self._applying_vipp_slider_style:
            return
        self._applying_vipp_slider_style = True
        try:
            style = slider_style_sheet(palette or self._owner_palette())
            if self.styleSheet() != style:
                self.setStyleSheet(style)
        finally:
            self._applying_vipp_slider_style = False

    def _owner_palette(self) -> QPalette:
        """Resolve theme roles before this slider's own QSS can mask them."""

        parent = self.parentWidget()
        if parent is None:
            return QPalette(self.palette())
        return QPalette(parent.palette())

    def _watch_theme_owner(self) -> None:
        owner = self.parentWidget()
        if owner is self._theme_owner:
            return
        if self._theme_owner is not None:
            try:
                self._theme_owner.removeEventFilter(self)
            except RuntimeError:
                pass
        self._theme_owner = owner
        if owner is not None:
            owner.installEventFilter(self)


__all__ = [
    "SliderColors",
    "VippSlider",
    "slider_colors",
    "slider_style_sheet",
]
