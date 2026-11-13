"""Shared dialog presentation leaves scientific and deliberate action state alone."""

import pytest
from qtpy.QtCore import QCoreApplication, QEvent
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QLabel, QWidget

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData
from napari_vipp.ui.colocalization_scatter_dialog import ColocalizationScatterDialog
from napari_vipp.ui.compute_benchmark_dialog import NodeBenchmarkDialog
from napari_vipp.ui.compute_pipeline_optimizer_dialog import PipelineOptimizerDialog
from napari_vipp.ui.detection_results import DetectionResultsDialog
from napari_vipp.ui.dialogs import (
    ConnectionInsertDialog,
    ConnectionInsertMappingDialog,
    ConnectionInsertPortMapping,
    TunnelManagerDialog,
)
from napari_vipp.ui.example_chooser import ExampleWorkflowDialog
from napari_vipp.ui.histogram_dialog import HistogramDialog
from napari_vipp.ui.measurement_collection import _ExportOptionsDialog
from napari_vipp.ui.reproduction import ReproductionChoiceDialog
from napari_vipp.ui.result_plots import PlotExportDialog
from napari_vipp.ui.result_table_dialog import ResultTableDialog, ResultTablePanel
from napari_vipp.ui.results_workspace import ResultsWorkspaceDialog
from napari_vipp.ui.tracking_results import TrackingResultsDialog


@pytest.mark.parametrize(
    "factory, profile",
    [
        (lambda: ConnectionInsertDialog([]), "chooser"),
        (
            lambda: ConnectionInsertMappingDialog([], "Node", "Source", "Target"),
            "chooser",
        ),
        (TunnelManagerDialog, "chooser"),
        (ExampleWorkflowDialog, "chooser"),
        (ResultTableDialog, "workspace"),
        (HistogramDialog, "workspace"),
        (ColocalizationScatterDialog, "workspace"),
        (DetectionResultsDialog, "workspace"),
        (TrackingResultsDialog, "workspace"),
        (ResultsWorkspaceDialog, "workspace"),
        (lambda: NodeBenchmarkDialog("Synthetic node"), "workspace"),
        (PipelineOptimizerDialog, "workspace"),
        (_ExportOptionsDialog, "compact"),
        (lambda: ReproductionChoiceDialog("0.16.0a7", "0.16.0a7"), "compact"),
    ],
)
def test_dialog_families_share_an_initial_only_window_profile(qtbot, factory, profile):
    window = factory()
    qtbot.addWidget(window)
    assert window._vipp_window_policy.window is window
    assert window._vipp_window_policy.profile == profile
    assert not window._vipp_window_policy.fitted


@pytest.mark.parametrize("dialog_type", [ResultTableDialog, ResultTablePanel])
def test_result_table_rows_refresh_from_effective_font_without_changing_data(
    qtbot, dialog_type
):
    dialog = dialog_type()
    qtbot.addWidget(dialog)
    table = TableData(("area", "label"), ((12.00000001, 1), (23.0, 2)))
    dialog.set_table(table, title="Synthetic table", default_export_name="unused.csv")
    dialog.show()
    for points in (9, 16, 9, 16):
        dialog.setFont(QFont("Arial", points))
        QCoreApplication.sendEvent(dialog, QEvent(QEvent.FontChange))
        qtbot.waitUntil(
            lambda: (
                dialog.table_view.verticalHeader().defaultSectionSize()
                >= dialog.table_view.fontMetrics().height() + 12
            )
        )
        assert dialog.table is table
        assert table.rows == ((12.00000001, 1), (23.0, 2))
    if dialog_type is ResultTablePanel:
        assert not hasattr(dialog, "_vipp_window_policy")


def test_older_choosers_label_search_and_wrap_long_mapping_context(qtbot):
    insert = ConnectionInsertDialog([])
    tunnels = TunnelManagerDialog()
    mapping = ConnectionInsertMappingDialog(
        [ConnectionInsertPortMapping(0, 0, "Input", "Output", "Image")],
        "Long node name " * 12,
        "Long source " * 12,
        "Long target " * 12,
        axis_choices=[("z", "Z axis")],
    )
    for widget in (insert, tunnels, mapping):
        qtbot.addWidget(widget)
    assert insert.search.accessibleName() == "Search compatible nodes"
    assert insert.tree.accessibleName() == "Compatible nodes"
    assert tunnels.filter_edit.accessibleName()
    context = next(
        label
        for label in mapping.findChildren(QLabel)
        if label.text().startswith("Choose how")
    )
    assert context.wordWrap()
    assert mapping.axis_combo.accessibleName() == "Insertion axis"
    assert (
        next(
            label for label in mapping.findChildren(QLabel) if label.text() == "Axis"
        ).buddy()
        is mapping.axis_combo
    )
    assert insert.ok_button.text() == "Insert node"
    assert mapping.ok_button.text() == "Use mapping"


def test_figure_export_keeps_physical_dimensions_and_explicit_save_default(qtbot):
    table = TableData(("area",), ((12.0,), (23.0,)))
    dialog = PlotExportDialog(build_plot_result(table, y_column="area"))
    qtbot.addWidget(dialog)
    original = (
        dialog.width_spin.value(),
        dialog.height_spin.value(),
        dialog.dpi.value(),
    )
    dialog.show()
    dialog.resize(620, 520)
    QCoreApplication.processEvents()
    assert (
        dialog.width_spin.value(),
        dialog.height_spin.value(),
        dialog.dpi.value(),
    ) == original
    assert dialog.save_button.isDefault()
    assert not dialog.save_button.autoDefault()
    assert not dialog.cancel_button.isDefault()
    assert dialog.width_spin.accessibleName() and dialog.dpi.accessibleName()
    assert dialog.exported is None


def test_example_title_role_refresh_does_not_accumulate_size(qtbot):
    host = QWidget()
    qtbot.addWidget(host)
    host.setFont(QFont("Arial", 12))
    dialog = ExampleWorkflowDialog(host)
    qtbot.addWidget(dialog)
    dialog.select_example(dialog._examples[0].id)
    dialog.show()
    for _ in range(4):
        dialog._apply_theme()
        QCoreApplication.processEvents()
    first = dialog.title_label.styleSheet()
    dialog._apply_theme()
    assert dialog.title_label.styleSheet() == first
    assert "16pt" in first
    assert dialog.open_button.isDefault()
    assert dialog.result() == dialog.Rejected


@pytest.mark.parametrize("points", [9, 16])
def test_scatter_settings_reflow_without_clipping_or_changing_values(qtbot, points):
    dialog = ColocalizationScatterDialog()
    qtbot.addWidget(dialog)
    dialog.setFont(QFont("Arial", points))
    before = (
        dialog.density_bins_spin.value(),
        dialog.populated_range_spin.value(),
        dialog.export_size_spin.value(),
        dialog.log_density_checkbox.isChecked(),
        dialog.zoom_to_data_checkbox.isChecked(),
        dialog.equal_axes_checkbox.isChecked(),
    )
    edits = []
    dialog.densitySettingsChanged.connect(lambda *values: edits.append(values))
    dialog.show()
    dialog.resize(900, 650)
    QCoreApplication.processEvents()
    QCoreApplication.processEvents()
    if points == 16:
        assert dialog._settings_compact
    for label, control in dialog._settings_pairs:
        assert label.width() >= label.fontMetrics().horizontalAdvance(label.text())
        assert control.width() >= control.minimumSizeHint().width()
        assert label.buddy() is control
        assert (
            dialog.body_scroll.widget()
            .rect()
            .contains(label.mapTo(dialog.body_scroll.widget(), label.rect().topLeft()))
        )
    assert (
        dialog.density_bins_spin.value(),
        dialog.populated_range_spin.value(),
        dialog.export_size_spin.value(),
        dialog.log_density_checkbox.isChecked(),
        dialog.zoom_to_data_checkbox.isChecked(),
        dialog.equal_axes_checkbox.isChecked(),
    ) == before
    assert edits == []
    assert dialog.export_button.isVisibleTo(dialog)
    assert dialog.close_button.isVisibleTo(dialog)
    assert not dialog.export_button.isDefault()


@pytest.mark.parametrize("points", [9, 16])
def test_reproduction_decision_scrolls_body_without_compressing_version_or_gate(
    qtbot, points
):
    dialog = ReproductionChoiceDialog("0.16.0a7", "0.16.0a8.dev15")
    qtbot.addWidget(dialog)
    dialog.setFont(QFont("Arial", points))
    dialog.show()
    dialog.resize(800, 520)
    QCoreApplication.processEvents()
    QCoreApplication.processEvents()
    assert dialog.height() <= 520
    assert dialog.version_label.height() >= dialog.version_label.heightForWidth(
        dialog.version_label.width()
    )
    assert "Installed here: VIPP 0.16.0a8.dev15" in dialog.version_label.text()
    dialog.body_scroll.ensureWidgetVisible(dialog.override_checkbox)
    QCoreApplication.processEvents()
    assert dialog.open_button.isVisibleTo(dialog)
    assert dialog.footer.isVisibleTo(dialog)
    assert not dialog.open_button.isEnabled()
    assert not dialog.version_override_accepted
    assert dialog.result() == dialog.Rejected
