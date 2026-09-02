from __future__ import annotations

from qtpy.QtCore import QEvent
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication

from napari_vipp.ui.axis_controls import (
    AxisSelectionRow,
    AxisSliceOption,
    ReorderAxesControl,
    SelectTableColumnsControl,
)
from napari_vipp.ui.axis_interpretation import AxisInterpretationControl
from napari_vipp.ui.controls import ImageSourceControl
from napari_vipp.ui.palette_roles import theme_colors


def _palette(*, base: str, text: str) -> QPalette:
    palette = QPalette()
    surface = QColor(base)
    foreground = QColor(text)
    alternate = (
        surface.lighter(106)
        if surface.lightness() < 128
        else surface.darker(104)
    )
    for role in (QPalette.Base, QPalette.Window, QPalette.Button):
        palette.setColor(role, surface)
    palette.setColor(QPalette.AlternateBase, alternate)
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(role, foreground)
    return palette


def _restyle(widget, palette: QPalette) -> None:
    widget.setPalette(palette)
    QApplication.sendEvent(widget, QEvent(QEvent.StyleChange))


def test_axis_interpretation_notice_follows_light_and_dark_palettes(qtbot):
    control = AxisInterpretationControl()
    qtbot.addWidget(control)

    for palette in (
        _palette(base="#ffffff", text="#111827"),
        _palette(base="#111827", text="#f8fafc"),
    ):
        _restyle(control, palette)
        warning = theme_colors(palette).warning
        style = control.notice_label.styleSheet()
        assert warning.surface.name() in style
        assert warning.foreground.name() in style
        assert warning.border.name() in style


def test_axis_selection_row_neutrals_retheme_without_changing_markers(qtbot):
    row = AxisSelectionRow(AxisSliceOption(0, "z", "space", 12))
    qtbot.addWidget(row)

    light = _palette(base="#ffffff", text="#111827")
    _restyle(row, light)
    colors = theme_colors(light)
    assert colors.raised_surface.name() in row.keep_button.styleSheet()
    assert colors.text.name() in row.title_label.text()
    assert colors.muted_text.name() in row.title_label.text()
    assert colors.border.name() in row.divider.styleSheet()
    assert "#93c5fd" in row.title_label.text()

    dark = _palette(base="#111827", text="#f8fafc")
    _restyle(row, dark)
    colors = theme_colors(dark)
    assert colors.raised_surface.name() in row.keep_button.styleSheet()
    assert colors.text.name() in row.title_label.text()
    assert "#93c5fd" in row.title_label.text()


def test_axis_list_guidance_and_summaries_follow_palette(qtbot):
    reorder = ReorderAxesControl(
        [
            AxisSliceOption(0, "z", "space", 3),
            AxisSliceOption(1, "y", "space", 8),
        ]
    )
    columns = SelectTableColumnsControl(["label", "area"])
    qtbot.addWidget(reorder)
    qtbot.addWidget(columns)

    for palette in (
        _palette(base="#ffffff", text="#111827"),
        _palette(base="#111827", text="#f8fafc"),
    ):
        colors = theme_colors(palette)
        _restyle(reorder, palette)
        _restyle(columns, palette)
        assert colors.muted_text.name() in reorder.hint_label.styleSheet()
        assert colors.muted_text.name() in reorder.serialized_label.styleSheet()
        assert colors.warning.foreground.name() in (
            reorder.warning_label.styleSheet()
        )
        assert colors.muted_text.name() in columns.hint_label.styleSheet()
        assert colors.muted_text.name() in columns.summary_label.styleSheet()


def test_image_source_inspector_surfaces_follow_palette(qtbot):
    control = ImageSourceControl(None, layer_names=[], sample_names=[])
    qtbot.addWidget(control)

    for palette in (
        _palette(base="#ffffff", text="#111827"),
        _palette(base="#111827", text="#f8fafc"),
    ):
        colors = theme_colors(palette)
        _restyle(control, palette)
        assert colors.muted_text.name() in control.source_summary.styleSheet()
        assert colors.muted_text.name() in (
            control.pyramid_levels_label.styleSheet()
        )
        warning_style = control.memory_repair_panel.styleSheet()
        assert colors.warning.surface.name() in warning_style
        assert colors.warning.foreground.name() in warning_style
        assert colors.warning.border.name() in warning_style
