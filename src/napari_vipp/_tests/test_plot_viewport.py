"""Plots fit the visible viewport without modifying results or export sizes."""

import pytest
from qtpy.QtCore import QPoint
from qtpy.QtWidgets import QScrollArea

from napari_vipp._tests.test_result_plots_ui import _panel
from napari_vipp._tests.test_results_workspace_dialog import _dialog, _table
from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.statistics import summarize_statistics
from napari_vipp.ui.result_plots import PlotExportDialog


def _assert_inside(parent, child):
    start = child.mapTo(parent, QPoint(0, 0))
    assert parent.rect().contains(start)
    assert parent.rect().contains(start + child.rect().bottomRight())


@pytest.mark.parametrize("width,height", [(760, 520), (1200, 800)])
@pytest.mark.parametrize("summary_input", [False, True])
@pytest.mark.parametrize("fit_delay", [0, 150])
def test_full_plot_and_axes_stay_visible_when_workspace_resizes(
    qtbot, monkeypatch, width, height, summary_input, fit_delay
):
    dialog = _dialog(qtbot)
    if fit_delay:
        # Model a busy event loop deterministically. The old fixed 100 ms sleep
        # captured the unfitted title and reproduced the exact Windows CI clip.
        timer = dialog.plot_canvas._fit_timer
        start = timer.start
        monkeypatch.setattr(timer, "start", lambda _interval=0: start(fit_delay))
    table = _table()
    y_column = "area"
    if summary_input:
        table = summarize_statistics(table, value_columns="area", group_by="condition")
        y_column = "area_mean"
    result = build_plot_result(
        table,
        y_column=y_column,
        group_column="condition",
        title="One descriptive group mean per condition · synthetic data",
    )
    params = result.recipe.to_params()
    changes = []
    dialog.plot_params_changed.connect(changes.append)
    dialog.set_plot(table=table, params=params, result=result, stale=False)
    dialog.show_tab("plots")
    dialog.resize(width, height)
    canvas = dialog.plot_canvas.canvas
    assert not isinstance(dialog.plot_area.parentWidget(), QScrollArea)

    def assert_fitted_plot():
        # Qt sizes the viewport, then its deferred fit adjusts typography.
        # Assert the eventual rendered bounds, not a machine-speed deadline.
        assert not dialog.plot_canvas._fit_timer.isActive()
        _assert_inside(dialog.plot_area, canvas)
        _assert_inside(dialog, dialog.plot_area)
        assert canvas.height() >= 100
        canvas.draw()
        figure = canvas.figure
        bounds = figure.axes[0].get_tightbbox(canvas.get_renderer())
        assert bounds.x0 >= -2 and bounds.y0 >= -2
        assert bounds.x1 <= figure.bbox.width + 2
        assert bounds.y1 <= figure.bbox.height + 2

    qtbot.waitUntil(assert_fitted_plot, timeout=3000)
    before = canvas.size()
    dialog.resize(width + 200, height + 150)
    qtbot.waitUntil(
        lambda: canvas.width() > before.width() and canvas.height() > before.height()
    )
    qtbot.waitUntil(assert_fitted_plot, timeout=3000)
    assert result.recipe.to_params() == params
    assert dialog._plot_result is result
    assert changes == []


def test_notices_and_plotted_data_do_not_scroll_the_figure_out_of_view(qtbot):
    dialog = _dialog(qtbot)
    result = build_plot_result(_table(), y_column="area")
    dialog.set_plot(
        table=_table(),
        params=result.recipe.to_params(),
        result=result,
        stale=False,
        message="An informative long notice. " * 80,
    )
    dialog.resize(760, 520)
    dialog.show_tab("plots")
    dialog.plotted_data_button.setChecked(True)
    qtbot.wait(100)
    assert dialog.plot_info.verticalScrollBar().maximum() > 0
    _assert_inside(dialog.plot_area, dialog.plot_canvas.canvas)
    assert dialog.plot_canvas.canvas.height() > 40
    _assert_inside(dialog.plot_area, dialog.plotted_panel)
    bounds = dialog.plot_canvas.geometry()
    dialog.plot_info.verticalScrollBar().setValue(9999)
    qtbot.wait(20)
    assert dialog.plot_canvas.geometry() == bounds
    assert dialog.plotted_panel.table is result.plotted_table


def test_standalone_preview_resize_keeps_recipe_export_and_fixed_intervals(qtbot):
    panel = _panel(qtbot)
    result = build_plot_result(panel.table, y_column="area", y_tick_interval="5")
    panel.set_state(table=panel.table, params=result.recipe.to_params(), result=result)
    window = panel.open_plot()
    window.resize(900, 600)
    qtbot.wait(100)
    canvas = window.plot.canvas
    locator = canvas.figure.axes[0].yaxis.get_major_locator()
    ticks = locator.tick_values(10, 25)
    window.resize(1200, 800)
    qtbot.wait(100)
    _assert_inside(window, canvas)
    assert canvas.figure.axes[0].yaxis.get_major_locator() is locator
    assert list(locator.tick_values(10, 25)) == list(ticks)
    assert panel.result is result
    export = PlotExportDialog(result)
    qtbot.addWidget(export)
    assert export.width_spin.value() == 160
    assert export.height_spin.value() == 110
    assert export.dpi.value() == 300
