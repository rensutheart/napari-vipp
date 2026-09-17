"""Warning contrast and wrapped text stay stable across narrow inspector updates."""

from types import SimpleNamespace

import pytest
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QFormLayout, QScrollArea, QVBoxLayout, QWidget

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.statistics import summarize_statistics
from napari_vipp.core.tables import TableData
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.result_plots import (
    PlotResultsPanel,
    _PlotWarningLabel,
    _set_plot_warnings,
)

MESSAGE = (
    "Each point represents one summary row, not an original object or independent "
    "sample. Summary counts do not recreate the original observations, and SD "
    "columns are not automatically used as error bars."
)


def _contrast(first, second):
    def luminance(color):
        rgb = [
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
            for value in (color.redF(), color.greenF(), color.blueF())
        ]
        return sum(
            channel * weight
            for channel, weight in zip(rgb, (0.2126, 0.7152, 0.0722), strict=True)
        )

    low, high = sorted((luminance(first), luminance(second)))
    return (high + 0.05) / (low + 0.05)


def _palette(base, text):
    palette = QPalette()
    for role in (QPalette.Base, QPalette.Window, QPalette.Button):
        palette.setColor(role, QColor(base))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(role, QColor(text))
    return palette


@pytest.mark.parametrize("base,text", [("#23242b", "#f0f1f2"), ("#fafafa", "#20252d")])
def test_refresh_does_not_accumulate_tint_and_preserves_strong_contrast(
    qtbot, base, text
):
    host = QWidget()
    qtbot.addWidget(host)
    host.setPalette(_palette(base, text))
    layout = QVBoxLayout(host)
    label = _PlotWarningLabel(host)
    layout.addWidget(label)
    host.show()
    result = SimpleNamespace(warnings=(MESSAGE,))
    _set_plot_warnings(label, result)
    qtbot.wait(10)
    expected = label.styleSheet()
    for _ in range(30):
        _set_plot_warnings(label, result)
        qtbot.wait(1)
        assert label.styleSheet() == expected
    colors = theme_colors(host.palette())
    assert _contrast(colors.text, colors.warning.surface) >= 7
    assert colors.warning.surface.name() in expected
    assert (
        label.palette().color(QPalette.Window).name() == colors.warning.surface.name()
    )


def test_live_theme_change_uses_parent_palette_not_previous_warning(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.setPalette(_palette("#23242b", "#f0f1f2"))
    label = _PlotWarningLabel(host)
    label.setText(MESSAGE)
    host.show()
    dark = label.styleSheet()
    host.setPalette(_palette("#fafafa", "#20252d"))
    qtbot.wait(10)
    assert label.styleSheet() != dark
    expected = theme_colors(host.palette())
    assert expected.warning.surface.name() in label.styleSheet()
    assert expected.text.name() in label.styleSheet()


def test_wrapping_expands_and_shrinks_with_width_and_font(qtbot):
    label = _PlotWarningLabel()
    qtbot.addWidget(label)
    label.setText(MESSAGE)
    label.resize(200, 10)
    label.show()
    qtbot.wait(10)
    narrow = label.minimumHeight()
    assert label.height() >= label.heightForWidth(label.width())
    label.resize(440, 10)
    qtbot.wait(10)
    assert label.minimumHeight() < narrow
    wide = label.minimumHeight()
    font = label.font()
    font.setPointSizeF(font.pointSizeF() + 5)
    label.setFont(font)
    qtbot.wait(10)
    assert label.minimumHeight() > wide
    assert label.height() >= label.heightForWidth(label.width())
    assert label.accessibleDescription() == MESSAGE


@pytest.mark.parametrize("width", [200, 280, 440])
def test_summary_warning_fits_fixed_height_inspector_form(qtbot, width):
    source = TableData(("condition", "area"), (("A", 10.0), ("B", 20.0)))
    table = summarize_statistics(source, value_columns="area", group_by="condition")
    result = build_plot_result(table, y_column="area_mean", group_column="condition")
    scroll = QScrollArea()
    qtbot.addWidget(scroll)
    scroll.setWidgetResizable(True)
    content = QWidget()
    form = QFormLayout(content)
    form.setContentsMargins(0, 0, 0, 0)
    panel = PlotResultsPanel(table, result.recipe.to_params(), result)
    form.addRow(panel)
    scroll.setWidget(content)
    scroll.resize(width + 20, 600)
    scroll.show()

    def fit_form():
        form.invalidate()
        content.setFixedHeight(form.totalHeightForWidth(scroll.viewport().width()))

    panel.layout_changed.connect(fit_form)
    for _ in range(3):
        fit_form()
        qtbot.wait(10)
    warning = panel.warning
    assert warning.text() == MESSAGE
    assert warning.height() >= warning.heightForWidth(warning.width())
    assert warning.geometry().bottom() < panel.open_button.y()
    before = warning.minimumHeight()
    _set_plot_warnings(warning, result, stale=True)
    assert warning.isHidden()
    assert warning.minimumHeight() == before
    assert warning.sizePolicy().retainSizeWhenHidden()
    _set_plot_warnings(warning, result)
    assert not warning.isHidden()
    assert warning.height() >= warning.heightForWidth(warning.width())
