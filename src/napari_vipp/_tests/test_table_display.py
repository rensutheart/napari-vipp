"""Display precision never enters scientific data, sorting or exports."""

from decimal import Decimal

import numpy as np
import pytest
from qtpy.QtCore import QItemSelectionModel, Qt

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.tables import TableData
from napari_vipp.ui.result_table_dialog import ResultTableDialog, ResultTableModel
from napari_vipp.ui.results_workspace import ResultsWorkspaceDialog
from napari_vipp.ui.table_display import (
    MAX_DECIMAL_PLACES,
    TableDecimalControls,
    format_table_value,
)


@pytest.mark.parametrize(
    ("value", "places", "expected"),
    [
        (1.23456789, 3, "1.235"),
        (1.23456789, 6, "1.234568"),
        (1.75, 0, "2"),
        (2.0, 3, "2.000"),
        (-0.0001, 3, "0.000"),
        (-0.0, 0, "0"),
        (np.float32(1.234567), 3, "1.235"),
        (np.float16(1.5), 2, "1.50"),
        (np.float64(-1.234567), 4, "-1.2346"),
        (np.longdouble("1.75"), 0, "2"),
        (12, 4, "12"),
        (np.int64(2**62 + 1), 4, str(2**62 + 1)),
        (np.uint64(2**64 - 1), 4, str(2**64 - 1)),
        (True, 3, "True"),
        (np.bool_(False), 3, "False"),
        ("1.23456789", 2, "1.23456789"),
        (Decimal("1.23456789"), 2, "1.23456789"),
        (None, 3, ""),
        (float("nan"), 3, "nan"),
        (float("inf"), 3, "inf"),
        (float("-inf"), 3, "-inf"),
    ],
)
def test_format_only_floating_point_cells(value, places, expected):
    assert format_table_value(value, places) == expected


@pytest.mark.skipif(
    np.finfo(np.longdouble).nmant <= np.finfo(float).nmant,
    reason="This platform has no wider-than-Python floating scalar.",
)
def test_extended_precision_float_is_not_narrowed_for_display():
    value = np.longdouble("9007199254740993.125")
    assert format_table_value(value, 3) == "9007199254740993.125"


@pytest.mark.parametrize("places", [-1, MAX_DECIMAL_PLACES + 1, True, 2.5, "3"])
def test_precision_rejects_invalid_values(qtbot, places):
    controls = TableDecimalControls()
    qtbot.addWidget(controls)
    model = ResultTableModel()
    with pytest.raises(ValueError, match="integer from 0 to 15"):
        controls.set_decimal_places(places)
    with pytest.raises(ValueError, match="integer from 0 to 15"):
        model.set_decimal_places(places)


def test_precision_controls_have_accessible_icons_and_bounded_steps(qtbot):
    controls = TableDecimalControls()
    qtbot.addWidget(controls)
    controls.show()
    changed = []
    controls.decimals_changed.connect(changed.append)
    assert controls.decimal_places == 3
    assert controls.label.text() == "Decimals: 3"
    for button in (controls.increase_button, controls.decrease_button):
        assert not button.icon().isNull()
        assert "Decimal" in button.accessibleName()
        assert "Display only" in button.toolTip()
    controls.increase_button.click()
    assert controls.decimal_places == 4
    controls.decrease_button.click()
    assert controls.decimal_places == 3
    controls.set_decimal_places(0)
    assert not controls.decrease_button.isEnabled()
    controls.decrease_button.click()
    assert controls.decimal_places == 0
    controls.set_decimal_places(MAX_DECIMAL_PLACES)
    assert not controls.increase_button.isEnabled()
    controls.increase_button.click()
    assert changed == [4, 3, 0, MAX_DECIMAL_PLACES]


def test_result_table_precision_preserves_sort_selection_exports_and_refresh(
    qtbot, tmp_path
):
    table = TableData(
        ("id", "value", "note"),
        ((2**63 + 1, 1.23459, "first"), (2, 1.23451, "second")),
    )
    dialog = ResultTableDialog()
    qtbot.addWidget(dialog)
    dialog.set_table(table, title="Results", default_export_name="results.csv")
    dialog.resize(700, 400)
    dialog.show()
    original = dialog.export_table(tmp_path / "original.csv").read_bytes()
    dialog.table_view.horizontalHeader().sectionClicked.emit(1)
    assert dialog.model.raw_value(0, 2) == "second"
    index = dialog.model.index(0, 1)
    selection = dialog.table_view.selectionModel()
    selection.select(index, QItemSelectionModel.Select)
    resets, changes = [], []
    dialog.model.modelReset.connect(lambda: resets.append(True))
    dialog.model.dataChanged.connect(lambda *args: changes.append(args))
    dialog.decimal_controls.set_decimal_places(0)
    assert dialog.model.data(index) == "1"
    assert dialog.model.data(index, Qt.ToolTipRole) == "1.23451"
    assert dialog.model.data(index, Qt.UserRole) == "1.23451"
    assert dialog.model.data(dialog.model.index(1, 0)) == str(2**63 + 1)
    assert selection.isSelected(index)
    assert not resets
    assert len(changes) == 1
    assert dialog.model.raw_value(0, 2) == "second"
    assert dialog.export_table(tmp_path / "rounded.csv").read_bytes() == original
    assert dialog.table is table
    replacement = TableData(table.columns, ((7, 9.876543, "new"),))
    dialog.set_table(replacement, title="Results", default_export_name="results.csv")
    assert dialog.decimal_controls.decimal_places == dialog.model.decimal_places == 0
    assert dialog.model.data(dialog.model.index(0, 1)) == "10"
    assert replacement.rows[0][1] == 9.876543


def test_workspace_precision_all_three_tables_and_full_precision_search(qtbot):
    dialog = ResultsWorkspaceDialog()
    qtbot.addWidget(dialog)
    table = TableData(("id", "value"), ((1, 1.23456789), (2, 1.2)))
    dialog.set_data(table, title="Measurements", node_id="source")
    dialog.search.setText("1.23456789")
    dialog._search_changed()
    assert dialog.search_proxy.rowCount() == 1
    for panel in (dialog.data_panel, dialog.summary_panel, dialog.plotted_panel):
        panel.set_table(table, title="Measurements", default_export_name="table.csv")
        panel.decimal_controls.set_decimal_places(1)
        assert panel.model.data(panel.model.index(0, 1)) == "1.2"
        assert panel.model.data(panel.model.index(0, 0)) == "1"
        assert panel.model.data(panel.model.index(0, 1), Qt.ToolTipRole) == "1.23456789"
        assert panel.model.table is table
        if panel is not dialog.plotted_panel:
            assert not panel.decimal_controls.isHidden()
    # Both displayed values now look identical, but search still finds the
    # original exact value and is not changed by formatting or column hiding.
    assert dialog.search_proxy.rowCount() == 1
    assert dialog.search_proxy.data(dialog.search_proxy.index(0, 0)) == "1"
    dialog.data_panel.table_view.setColumnHidden(1, True)
    dialog.data_panel.decimal_controls.set_decimal_places(0)
    assert dialog.search_proxy.rowCount() == 1
    dialog.summary_panel.set_table(
        TableData(("mean", "count"), ((None, 2),)),
        title="Undefined SD",
        default_export_name="summary.csv",
    )
    undefined = dialog.summary_panel.model.index(0, 0)
    assert dialog.summary_panel.model.data(undefined) == "—"


@pytest.mark.parametrize("header_height", [24, 60])
def test_small_workspace_keeps_a_plotted_data_row_below_precision_controls(
    qtbot, header_height
):
    dialog = ResultsWorkspaceDialog()
    qtbot.addWidget(dialog)
    table = TableData(("area", "condition"), ((1.2345, "A"), (9.8765, "B")))
    result = build_plot_result(table, y_column="area", group_column="condition")
    dialog.set_data(table, node_id="measure", title="Measurements")
    dialog.set_choices(
        plots=[("plot", "Area")],
        plot_id="plot",
        plot_sources=[("measure", "Original measurements")],
        plot_source_id="measure",
    )
    dialog.set_plot(table=table, params=result.recipe.to_params(), result=result)
    dialog.resize(760, 520)
    dialog.show()
    dialog.show_tab("plots")
    controls = dialog.plotted_panel.decimal_controls
    assert not controls.isVisible()
    view = dialog.plotted_panel.table_view
    view.horizontalHeader().setFixedHeight(header_height)
    view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
    dialog.plotted_data_button.setChecked(True)
    qtbot.waitUntil(lambda: view.viewport().height() >= view.rowHeight(0))
    assert controls.isVisible()
    assert controls.mapTo(dialog, controls.rect().bottomLeft()).y() < (
        view.mapTo(dialog, view.rect().topLeft()).y()
    )
    controls.increase_button.click()
    assert dialog.plotted_panel.model.decimal_places == 4
    dialog.plotted_data_button.setChecked(False)
    assert not controls.isVisible()
    dialog.plotted_data_button.setChecked(True)
    assert controls.decimal_places == 4
    dialog.set_plot(table=table, params=result.recipe.to_params(), result=None)
    assert not controls.isVisible()
