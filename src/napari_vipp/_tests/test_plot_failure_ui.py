"""Failed recipes explain the correction without masquerading as missing data."""

import pytest
from qtpy.QtCore import Qt

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData
from napari_vipp.ui.result_plots import PlotResultsPanel


def _failure(qtbot, *, previous_result=False):
    table = TableData(
        (
            "label_id",
            "image_id",
            "major_axis_length_pixels",
            "minor_axis_length_pixels",
        ),
        ((1, "synthetic_field_01", 30.0, 10.0), (2, "synthetic_field_01", 42.0, 20.0)),
        column_units=(
            ("major_axis_length_pixels", "pixels"),
            ("minor_axis_length_pixels", "pixels"),
        ),
    )
    params = dict(
        y_column="major_axis_length_pixels",
        group_column="minor_axis_length_pixels",
        point_unit="Mean per image",
        image_column="image_id",
    )
    previous = build_plot_result(table, **{**params, "point_unit": "Objects"})
    panel = PlotResultsPanel(table, previous.recipe.to_params(), previous)
    qtbot.addWidget(panel)
    panel.show()
    window = panel.open_plot()
    window.data_button.setChecked(True)
    with pytest.raises(ValueError) as failure:
        build_plot_result(table, **params)
    panel.set_state(
        table=table,
        params=params,
        result=previous if previous_result else None,
        failed=True,
        message=str(failure.value),
    )
    return panel, window, table, params


@pytest.mark.parametrize("previous_result", [False, True])
def test_failure_explanation_replaces_placeholder_and_stale_plot(
    qtbot, previous_result
):
    panel, window, table, _params = _failure(qtbot, previous_result=previous_result)
    assert panel.failed and panel.stale
    assert panel.open_button.isEnabled()
    assert panel.open_button.text() == "Review plot…"
    for plot in (panel.plot, window.plot):
        assert plot.error_view.isVisible()
        assert not plot.placeholder.isVisible()
        assert plot.canvas is None
        assert "one mean per image" in plot.error_title.text()
        assert "synthetic_field_01" in plot.error_detail.text()
        assert "Minor axis length (pixels)" in plot.error_detail.text()
        assert "Each point represents to Objects" in plot.error_detail.text()
        assert "Group by to None" in plot.error_detail.text()
    assert not window.export_button.isEnabled()
    assert not window.data_button.isEnabled()
    assert not window.data_view.isVisible()
    assert window.data_view.model() is None
    assert not window.point_label.isVisible()
    assert panel.table is table


def test_failed_plot_can_be_reopened_then_recovers_after_explicit_edit(qtbot):
    panel, window, table, params = _failure(qtbot)
    panel.close_plot()
    assert panel.open_plot() is window
    assert window.plot.error_view.isVisible()
    with qtbot.waitSignal(panel.params_changed) as signal:
        combo = window.controls.controls["group_column"]
        combo.setCurrentIndex(combo.findData(""))
    assert signal.args[0]["point_unit"] == "Mean per image"
    assert signal.args[0]["group_column"] == ""
    assert not panel.failed
    assert not window.plot.error_view.isVisible()
    result = build_plot_result(table, **signal.args[0])
    panel.set_result(result)
    assert window.export_button.isEnabled()
    assert window.data_button.isEnabled()
    assert window.plot.canvas is not None
    assert result.series[0].y == (36.0,)
    assert len(result.source_table.rows) == 2
    assert params["group_column"] == "minor_axis_length_pixels"


def test_generic_failures_are_plain_text_and_remain_visible_without_a_table(qtbot):
    panel = PlotResultsPanel()
    qtbot.addWidget(panel)
    panel.set_state(failed=True, message="Cannot read <source> & measurement data.")
    window = panel.open_plot()
    assert window is not None
    assert window.plot.error_detail.textFormat() == Qt.PlainText
    assert "<source> &" in window.plot.error_detail.text()
    assert not window.export_button.isEnabled()


def test_mean_per_image_controls_explain_grouping_before_failure(qtbot):
    panel, window, _table, _params = _failure(qtbot)
    assert "shared by all objects" in window.controls.note.text()
    assert "treatment" in window.controls.note.text()
    assert "every object in an image" in (
        panel.controls.controls["group_column"].toolTip()
    )
