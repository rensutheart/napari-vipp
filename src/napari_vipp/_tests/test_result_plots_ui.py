"""Shared recipe ownership and nonmodal measurement-plot controls."""

from types import SimpleNamespace

from qtpy.QtCore import Qt

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData
from napari_vipp.ui.result_plots import PlotExportDialog, PlotResultsPanel


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
