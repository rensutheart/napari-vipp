"""Shared recipe ownership and nonmodal measurement-plot controls."""

from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData, save_table_output
from napari_vipp.ui.result_plots import (
    PlotExportDialog,
    PlotResultsPanel,
    _PreparedTableModel,
)


def _panel(qtbot):
    table = TableData(
        ("label_id", "image", "condition", "area", "intensity"),
        ((9, "one", "A", 10.0, 100.0), (17, "two", "B", 20.0, 150.0)),
        column_units=(("area", "µm²"),),
    )
    result = build_plot_result(table, y_column="area", x_column="intensity")
    panel = PlotResultsPanel(table, result.recipe.to_params(), result)
    qtbot.addWidget(panel)
    panel.resize(280, 950)
    panel.show()
    return panel


def test_inspector_has_narrow_controls_and_measured_units(qtbot):
    panel = _panel(qtbot)
    combo = panel.controls.controls["y_column"]
    assert combo.currentData() == "area"
    assert "µm²" in combo.currentText()
    assert combo.findData("label_id") == -1
    assert panel.open_button.isEnabled()
    assert panel.controls.width() <= 280


def test_window_edits_same_recipe_and_stale_export_is_disabled(qtbot):
    panel = _panel(qtbot)
    window = panel.open_plot()
    assert not window.isModal()
    assert window.export_button.isEnabled()
    with qtbot.waitSignal(panel.params_changed) as signal:
        window.controls.controls["plot_type"].setCurrentText("Scatter")
    assert signal.args[0]["plot_type"] == "Scatter"
    assert panel.controls.controls["plot_type"].currentData() == "Scatter"
    assert panel.stale
    assert not window.export_button.isEnabled()
    result = build_plot_result(panel.table, **signal.args[0])
    panel.set_result(result)
    assert window.export_button.isEnabled()
    assert window.plot.result.recipe.plot_type == "Scatter"
    panel.close_plot()
    assert not window.isVisible()


def test_programmatic_sync_does_not_emit_edits(qtbot):
    panel = _panel(qtbot)
    emitted = []
    panel.params_changed.connect(emitted.append)
    panel.set_state(table=panel.table, params=panel.params, result=panel.result)
    assert emitted == []


def test_data_inspection_is_complete_and_points_identify_labels(qtbot):
    panel = _panel(qtbot)
    window = panel.open_plot()
    qtbot.mouseClick(window.data_button, Qt.LeftButton)
    assert window.data_view.isVisible()
    assert window.data_view.model().rowCount() == 2
    artist = window.plot.canvas.figure.axes[0].collections[0]
    window.plot._picked(SimpleNamespace(artist=artist, ind=[0]))
    assert "label_id: 9" in window.point_label.text()


@pytest.mark.parametrize(
    ("value", "display"),
    [
        (1.23456789, "1.235"),
        (-2.1234567, "-2.123"),
        (10.0, "10.000"),
        (17, "17"),
        (2**80, str(2**80)),
        (True, "True"),
        (False, "False"),
        ("1.23456789", "1.23456789"),
        (None, ""),
        (float("nan"), "nan"),
        (float("inf"), "inf"),
        (float("-inf"), "-inf"),
    ],
)
def test_prepared_table_decimal_display_keeps_exact_values_and_tooltips(value, display):
    table = TableData(("measurement",), ((value,),))
    rows = table.rows
    model = _PreparedTableModel(table)
    assert model.data(model.index(0, 0)) == display
    assert model.data(model.index(0, 0), Qt.ToolTipRole) == (
        "" if value is None else str(value)
    )
    assert model.table is table
    assert table.rows is rows
    assert table.rows[0][0] is value


def test_inline_table_decimal_controls_are_display_only_and_toggle_with_data(
    qtbot, tmp_path
):
    table = TableData(("area",), ((1.23456789,), (8.7654321,)))
    result = build_plot_result(table, y_column="area")
    original_source = result.source_table
    panel = PlotResultsPanel(result=result)
    qtbot.addWidget(panel)
    panel.show()
    window = panel.open_plot()
    controls = window.data_decimal_controls
    emitted = []
    panel.params_changed.connect(emitted.append)
    original_params = dict(panel.params)
    save_table_output(result.plotted_table, tmp_path / "before.csv")

    assert controls.decimal_places == 3
    assert not controls.isVisible()
    window.data_button.setChecked(True)
    assert controls.isVisible()
    y_column = result.plotted_table.columns.index("y")
    model = window.data_view.model()
    assert model.data(model.index(0, y_column)) == "1.235"
    qtbot.mouseClick(controls.increase_button, Qt.LeftButton)
    assert controls.decimal_places == 4
    assert model.data(model.index(0, y_column)) == "1.2346"
    assert model.data(model.index(0, y_column), Qt.ToolTipRole) == "1.23456789"

    window.data_button.setChecked(False)
    assert not window.data_view.isVisible() and not controls.isVisible()
    window.data_button.setChecked(True)
    assert controls.isVisible() and controls.decimal_places == 4
    controls.set_decimal_places(0)
    assert not controls.decrease_button.isEnabled()
    assert model.data(model.index(0, y_column)) == "1"
    controls.set_decimal_places(15)
    assert not controls.increase_button.isEnabled()
    assert model.data(model.index(0, y_column)) == "1.234567890000000"

    save_table_output(result.plotted_table, tmp_path / "after.csv")
    assert (tmp_path / "before.csv").read_bytes() == (
        tmp_path / "after.csv"
    ).read_bytes()
    assert panel.result is result and model.table is result.plotted_table
    assert result.source_table is original_source and original_source == table
    assert panel.params == original_params and emitted == []
    assert not panel.stale and window.export_button.isEnabled()


def test_inline_table_decimal_precision_survives_refresh_busy_and_failure(qtbot):
    panel = _panel(qtbot)
    window = panel.open_plot()
    controls = window.data_decimal_controls
    window.data_button.setChecked(True)
    controls.set_decimal_places(7)
    result = panel.result
    old_model = window.data_view.model()
    panel.set_result(result)
    assert window.data_view.model() is not old_model
    assert window.data_view.model().decimal_places == controls.decimal_places == 7
    for _ in range(3):
        QApplication.processEvents()
    geometry = controls.geometry()

    panel.set_state(result=result, busy=True)
    for _ in range(3):
        QApplication.processEvents()
    assert not controls.isVisible() and not controls.isEnabled()
    assert controls.sizePolicy().retainSizeWhenHidden()
    assert controls.geometry() == geometry
    assert window.data_view.model() is None
    panel.set_result(result)
    for _ in range(3):
        QApplication.processEvents()
    assert controls.isVisible() and controls.isEnabled()
    assert not controls.sizePolicy().retainSizeWhenHidden()
    assert controls.geometry() == geometry
    assert window.data_view.model().decimal_places == controls.decimal_places == 7

    panel.set_state(result=result, stale=True)
    assert not window.export_button.isEnabled()
    assert controls.decimal_places == 7
    panel.set_state(result=result, failed=True, message="Review the plot settings.")
    assert not controls.isVisible() and not controls.isEnabled()
    assert window.data_view.model() is None
    panel.set_result(result)
    assert controls.decimal_places == 7 and not controls.isVisible()
    window.data_button.setChecked(True)
    assert controls.isVisible() and window.data_view.model().decimal_places == 7


def test_prepared_table_precision_updates_display_role_without_replacing_data():
    table = TableData(("area",), ((1.23456789,),))
    model = _PreparedTableModel(table)
    emitted = []
    model.dataChanged.connect(lambda *args: emitted.append(args))
    model.set_decimal_places(6)
    assert model.data(model.index(0, 0)) == "1.234568"
    assert model.table is table
    assert len(emitted) == 1 and emitted[0][2] == [Qt.DisplayRole]
    model.set_decimal_places(6)
    assert len(emitted) == 1


def test_export_dimensions_do_not_follow_window_size(qtbot):
    panel = _panel(qtbot)
    export = PlotExportDialog(panel.result)
    qtbot.addWidget(export)
    export.resize(900, 800)
    assert export.width_spin.value() == 160
    assert export.height_spin.value() == 110
    assert export.dpi.value() == 300
    export.format_combo.setCurrentText("SVG")
    assert not export.dpi.isEnabled()
    assert "Vector" in export.size_hint.text()


def test_unavailable_column_is_not_silently_replaced(qtbot):
    panel = _panel(qtbot)
    params = {**panel.params, "y_column": "removed_measurement"}
    panel.set_state(table=panel.table, params=params, result=panel.result, stale=True)
    assert panel.controls.controls["y_column"].currentData() == "removed_measurement"
    assert "Unavailable" in panel.controls.controls["y_column"].currentText()
    assert not panel.open_plot().export_button.isEnabled()


def test_mean_per_image_waits_for_explicit_identity_column(qtbot):
    panel = _panel(qtbot)
    emitted = []
    panel.params_changed.connect(emitted.append)
    controls = panel.controls.controls
    controls["point_unit"].setCurrentText("Mean per image")
    assert emitted == []
    assert controls["image_column"].isVisible()
    controls["image_column"].setCurrentIndex(controls["image_column"].findData("image"))
    assert len(emitted) == 1
    assert emitted[0]["point_unit"] == "Mean per image"
    assert emitted[0]["image_column"] == "image"


def test_appearance_collapsed_and_same_result_does_not_rebuild_canvas(qtbot):
    panel = _panel(qtbot)
    assert not panel.controls.controls["point_size"].isVisible()
    original_canvas = panel.plot.canvas
    panel.set_state(table=panel.table, params=panel.params, result=panel.result)
    assert panel.plot.canvas is original_canvas
    panel.controls.appearance_button.click()
    assert panel.controls.controls["point_size"].isVisible()


def test_export_worker_finishes_without_blocking_dialog(qtbot, tmp_path):
    panel = _panel(qtbot)
    export = PlotExportDialog(panel.result)
    qtbot.addWidget(export)
    export.show()
    target = tmp_path / "worker.svg"
    export._start_export(target, {"cancellation": export._cancel})
    assert not export.save_button.isEnabled()
    assert export.progress.isVisible()
    qtbot.waitUntil(lambda: export._thread is None, timeout=10000)
    assert export.exported.paths == (target,)
    assert target.is_file()


def test_export_worker_cancellation_closes_only_after_safe_stop(qtbot, tmp_path):
    panel = _panel(qtbot)
    export = PlotExportDialog(panel.result)
    qtbot.addWidget(export)
    export.show()
    target = tmp_path / "cancelled.svg"
    export._start_export(target, {"cancellation": export._cancel})
    export.reject()
    assert export._cancel.is_set()
    qtbot.waitUntil(lambda: export._thread is None, timeout=10000)
    assert export.exported is None
    assert not target.exists()


def test_numeric_groups_warning_visible_in_inspector_and_above_popup_plot(qtbot):
    table = TableData(
        ("major_axis_length_pixels", "minor_axis_length_pixels"),
        tuple((30 + i, 19.123456789012345 + i / 100) for i in range(60)),
        column_units=(
            ("major_axis_length_pixels", "pixels"),
            ("minor_axis_length_pixels", "pixels"),
        ),
    )
    result = build_plot_result(
        table,
        y_column="major_axis_length_pixels",
        group_column="minor_axis_length_pixels",
    )
    panel = PlotResultsPanel(table, result.recipe.to_params(), result)
    qtbot.addWidget(panel)
    panel.show()
    window = panel.open_plot()
    qtbot.waitUntil(window.isVisible)
    assert panel.warning.isVisible()
    assert window.warning.isVisible()
    assert "60 numeric groups" in panel.warning.text()
    assert window.warning.text() == panel.warning.text()
    assert (
        window.warning.geometry().bottom()
        < window.plot.mapTo(window, window.plot.rect().topLeft()).y()
    )
    group = panel.controls.controls["group_column"]
    assert group.currentText() == "Minor axis length (pixels)"
    assert group.currentData() == "minor_axis_length_pixels"
    assert (
        group.itemData(group.currentIndex(), Qt.ToolTipRole)
        == "minor_axis_length_pixels"
    )
    measurement = window.controls.controls["y_column"]
    assert measurement.currentText() == "Major axis length (pixels)"
    artist = window.plot.canvas.figure.axes[0].collections[0]
    window.plot._picked(SimpleNamespace(artist=artist, ind=[0]))
    assert "minor_axis_length_pixels: 19.123456789012344" in window.point_label.text()
    assert "Major axis length (pixels): 30" in window.point_label.text()
    window.controls.controls["plot_type"].setCurrentText("Scatter")
    assert panel.stale
    assert not panel.warning.isVisible()
    assert not window.warning.isVisible()
    panel.close_plot()


def test_picked_image_mean_reports_prepared_value_not_first_contributor(qtbot):
    table = TableData(
        ("area", "image_id", "dose"),
        ((2.0, "one", 1.23456789), (8.0, "one", 1.23456789)),
    )
    result = build_plot_result(
        table,
        y_column="area",
        group_column="dose",
        image_column="image_id",
        point_unit="Mean per image",
    )
    panel = PlotResultsPanel(table, result.recipe.to_params(), result)
    qtbot.addWidget(panel)
    window = panel.open_plot()
    artist = window.plot.canvas.figure.axes[0].collections[0]
    window.plot._picked(SimpleNamespace(artist=artist, ind=[0]))
    assert "dose: 1.23456789" in window.point_label.text()
    assert "Mean Area: 5.0" in window.point_label.text()
    assert "Image mean from 2 rows" in window.point_label.text()
    assert "Area: 2.0" not in window.point_label.text()
    panel.close_plot()
