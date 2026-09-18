"""All result viewers and figure export use the shared platform convention."""

import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtWidgets import QDialogButtonBox, QPushButton

import napari_vipp.ui.dialog_buttons as dialog_buttons
from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData
from napari_vipp.ui.colocalization_scatter_dialog import ColocalizationScatterDialog
from napari_vipp.ui.histogram_dialog import HistogramDialog
from napari_vipp.ui.result_plots import PlotExportDialog, PlotResultsPanel
from napari_vipp.ui.result_table_dialog import ResultTableDialog


def _set_platform(monkeypatch, *, macos):
    chosen = QDialogButtonBox.MacLayout if macos else QDialogButtonBox.WinLayout
    monkeypatch.setattr(
        dialog_buttons,
        "dialog_button_layout",
        lambda platform=None: getattr(chosen, "value", chosen),
    )


def _assert_action_order(dialog, action, dismiss, *, macos):
    left, right = (dismiss, action) if macos else (action, dismiss)
    left_pos = left.mapTo(dialog, QPoint())
    right_pos = right.mapTo(dialog, QPoint())
    assert left_pos.x() + left.width() <= right_pos.x()
    assert left_pos.y() == right_pos.y()
    assert not dismiss.autoDefault()
    assert not dismiss.isDefault()


@pytest.mark.parametrize("macos", [False, True], ids=["windows", "macos"])
@pytest.mark.parametrize(
    "dialog_type", [HistogramDialog, ColocalizationScatterDialog, ResultTableDialog]
)
def test_result_viewer_footer_order_and_close(qtbot, monkeypatch, macos, dialog_type):
    _set_platform(monkeypatch, macos=macos)
    dialog = dialog_type()
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.waitUntil(dialog.isVisible)
    _assert_action_order(dialog, dialog.export_button, dialog.close_button, macos=macos)
    qtbot.mouseClick(dialog.close_button, Qt.LeftButton)
    assert not dialog.isVisible()


@pytest.mark.parametrize("macos", [False, True], ids=["windows", "macos"])
def test_plot_footer_keeps_data_utility_apart(qtbot, monkeypatch, macos):
    _set_platform(monkeypatch, macos=macos)
    table = TableData(("label", "area"), ((1, 12.0), (2, 23.0)))
    result = build_plot_result(table, y_column="area")
    panel = PlotResultsPanel(table, result.recipe.to_params(), result)
    qtbot.addWidget(panel)
    window = panel.open_plot()
    qtbot.addWidget(window)
    close = next(b for b in window.findChildren(QPushButton) if b.text() == "Close")
    _assert_action_order(window, window.export_button, close, macos=macos)
    utility_right = window.data_button.mapTo(window, QPoint()).x() + (
        window.data_button.width()
    )
    assert utility_right < min(
        close.mapTo(window, QPoint()).x(),
        window.export_button.mapTo(window, QPoint()).x(),
    )
    qtbot.mouseClick(close, Qt.LeftButton)
    assert not window.isVisible()


@pytest.mark.parametrize("macos", [False, True], ids=["windows", "macos"])
def test_figure_export_footer_order_and_cancel(qtbot, monkeypatch, macos):
    _set_platform(monkeypatch, macos=macos)
    table = TableData(("area",), ((12.0,), (23.0,)))
    export = PlotExportDialog(build_plot_result(table, y_column="area"))
    qtbot.addWidget(export)
    export.show()
    _assert_action_order(export, export.save_button, export.cancel_button, macos=macos)
    assert export.save_button.isDefault()
    qtbot.mouseClick(export.cancel_button, Qt.LeftButton)
    assert not export.isVisible()
    assert export.exported is None
