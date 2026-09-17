"""Workspace presentation shares existing controls and exact table outputs."""

import sys
import threading

import pytest
from qtpy.QtCore import QEvent, Qt
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QDialog

from napari_vipp.core.result_plots import build_plot_result
from napari_vipp.core.statistics import StatisticsRecipe, summarize_statistics
from napari_vipp.core.tables import TableData
from napari_vipp.ui.result_table_dialog import ResultTablePanel
from napari_vipp.ui.results_workspace import ResultsWorkspaceDialog


def _table():
    return TableData(
        ("label_id", "image_id", "condition", "area"),
        (
            (1, "a", "Control", 10.0),
            (2, "a", "Control", 20.0),
            (1, "b", "Treated", 30.0),
        ),
        column_units=(("area", "µm²"),),
    )


def _dialog(qtbot):
    dialog = ResultsWorkspaceDialog()
    qtbot.addWidget(dialog)
    dialog.set_data(_table(), node_id="measure", title="Measure Objects")
    dialog.set_choices(
        summaries=[("summary", "Statistics")],
        summary_id="summary",
        plots=[("plot", "Plot Results")],
        plot_id="plot",
        plot_sources=[
            ("measure", "Original measurements"),
            ("summary", "Summary table — Statistics"),
        ],
        plot_source_id="measure",
    )
    dialog.show()
    return dialog


def test_open_empty_workspace_does_not_invent_analysis(qtbot):
    dialog = ResultsWorkspaceDialog()
    qtbot.addWidget(dialog)
    assert dialog.tabs.count() == 3
    assert not dialog.add_summary_button.isEnabled()
    assert not dialog.add_plot_button.isEnabled()
    assert not dialog.export_button.isEnabled()
    assert dialog.summary_selector.currentData() == ""
    assert dialog.plot_selector.currentData() == ""
    assert not dialog.isModal()


def test_tables_are_embedded_widgets_not_child_dialogs(qtbot):
    dialog = _dialog(qtbot)
    assert isinstance(dialog.data_panel, ResultTablePanel)
    assert not isinstance(dialog.data_panel, QDialog)
    assert not dialog.data_panel.isWindow()
    assert dialog.findChildren(QDialog) == []


def test_sync_choices_and_parameters_does_not_emit_user_edits(qtbot):
    dialog = _dialog(qtbot)
    events = []
    for signal in (
        dialog.summary_selected,
        dialog.plot_selected,
        dialog.plot_source_changed,
        dialog.summary_params_changed,
        dialog.plot_params_changed,
    ):
        signal.connect(events.append)
    dialog.set_choices(
        summaries=[("summary", "Statistics")],
        summary_id="summary",
        plots=[("plot", "Plot Results")],
        plot_id="plot",
        plot_sources=[("measure", "Original measurements")],
        plot_source_id="measure",
    )
    dialog.set_summary(table=_table(), params=StatisticsRecipe().to_params())
    dialog.set_plot(table=_table(), params={"y_column": "area"})
    assert events == []


def test_search_and_columns_change_view_not_analysis_or_export(qtbot, tmp_path):
    dialog = _dialog(qtbot)
    table = dialog.data_panel.table
    dialog.search.setText("Treated")
    dialog._search_changed()
    assert dialog.data_panel.table_view.model().rowCount() == 1
    assert "Showing 1 of 3" in dialog.data_view_note.text()
    dialog.column_list.item(0).setCheckState(Qt.Unchecked)
    assert dialog.data_panel.table_view.isColumnHidden(0)
    assert dialog.data_panel.table is table
    target = dialog.data_panel.export_table(tmp_path / "all.csv")
    assert len(target.read_text(encoding="utf-8-sig").splitlines()) == 4


def test_unchanged_refresh_preserves_sort_search_and_hidden_columns(qtbot):
    dialog = _dialog(qtbot)
    table = dialog.data_panel.table
    dialog.data_panel._sort_by_header(3)
    dialog.data_panel._sort_by_header(3)
    dialog.search.setText("Control")
    dialog._search_changed()
    dialog.column_list.item(0).setCheckState(Qt.Unchecked)
    dialog.set_data(table, node_id="measure", title="Measure Objects")
    assert dialog.data_panel.model.raw_value(0, 3) == 30
    assert dialog.search.text() == "Control"
    assert dialog.search_proxy.rowCount() == 2
    assert dialog.data_panel.table_view.isColumnHidden(0)


def test_select_all_none_columns_are_view_only_and_keep_search_and_export(
    qtbot, tmp_path
):
    dialog = _dialog(qtbot)
    table = dialog.data_panel.table
    edits = []
    dialog.summary_params_changed.connect(edits.append)
    dialog.plot_params_changed.connect(edits.append)
    dialog.search.setText("Treated")
    dialog._search_changed()
    assert not dialog.select_all_columns_button.isEnabled()
    assert dialog.select_no_columns_button.isEnabled()
    dialog.select_no_columns_button.click()
    for index in range(table.column_count):
        assert dialog.column_list.item(index).checkState() == Qt.Unchecked
        assert dialog.data_panel.table_view.isColumnHidden(index)
    assert "No columns visible" in dialog.data_view_note.text()
    assert dialog.select_all_columns_button.isEnabled()
    assert not dialog.select_no_columns_button.isEnabled()
    dialog.set_data(table, node_id="measure", title="Measure Objects")
    assert not dialog.select_no_columns_button.isEnabled()
    assert dialog.data_panel.table is table
    assert dialog.search.text() == "Treated"
    assert dialog.search_proxy.rowCount() == 1
    exported = dialog.data_panel.export_table(tmp_path / "all-columns.csv")
    lines = exported.read_text(encoding="utf-8-sig").splitlines()
    assert len(lines) == 4
    assert all(column in lines[0] for column in table.columns)
    dialog.select_all_columns_button.click()
    for index in range(table.column_count):
        assert dialog.column_list.item(index).checkState() == Qt.Checked
        assert not dialog.data_panel.table_view.isColumnHidden(index)
    assert "No columns visible" not in dialog.data_view_note.text()
    assert not dialog.select_all_columns_button.isEnabled()
    assert dialog.select_no_columns_button.isEnabled()
    assert dialog.search.text() == "Treated"
    assert edits == []


def test_bulk_column_actions_follow_individual_checks_and_empty_inputs(qtbot):
    dialog = _dialog(qtbot)
    dialog.column_list.item(0).setCheckState(Qt.Unchecked)
    assert dialog.select_all_columns_button.isEnabled()
    assert dialog.select_no_columns_button.isEnabled()
    dialog.select_no_columns_button.click()
    dialog.column_list.item(1).setCheckState(Qt.Checked)
    assert dialog.select_all_columns_button.isEnabled()
    assert dialog.select_no_columns_button.isEnabled()
    dialog.set_data(None)
    assert not dialog.select_all_columns_button.isEnabled()
    assert not dialog.select_no_columns_button.isEnabled()
    dialog.set_data(TableData((), ()), node_id="measure")
    assert not dialog.select_all_columns_button.isEnabled()
    assert not dialog.select_no_columns_button.isEnabled()


def test_cleared_column_selection_survives_new_rows(qtbot):
    dialog = _dialog(qtbot)
    dialog.select_no_columns_button.click()
    original = _table()
    dialog.set_data(TableData(original.columns, original.rows[:1]), node_id="measure")
    assert all(
        dialog.column_list.item(i).checkState() == Qt.Unchecked
        and dialog.data_panel.table_view.isColumnHidden(i)
        for i in range(original.column_count)
    )
    assert dialog.select_all_columns_button.isEnabled()
    assert not dialog.select_no_columns_button.isEnabled()


def test_summary_compact_view_exports_the_complete_result(qtbot, tmp_path):
    dialog = _dialog(qtbot)
    table = _table()
    params = {
        **StatisticsRecipe().to_params(),
        "value_columns": "area",
        "group_by": "condition",
    }
    summary = summarize_statistics(table, **params)
    dialog.set_summary(table=table, params=params, result=summary, stale=False)
    dialog.show_tab("summary")
    assert dialog.export_button.isEnabled()
    index = summary.columns.index("statistics_recipe")
    assert dialog.summary_panel.table_view.isColumnHidden(index)
    assert dialog.summary_panel.table is summary
    visible = sum(
        not dialog.summary_panel.table_view.isColumnHidden(i)
        for i in range(summary.column_count)
    )
    assert (
        f"{visible:,} of {summary.column_count:,} fields shown"
        in dialog.summary_panel.summary_label.text()
    )
    target = dialog.summary_panel.export_table(tmp_path / "summary.csv")
    assert "statistics_recipe" in target.read_text(encoding="utf-8-sig")
    dialog.summary_evidence.setChecked(True)
    assert not dialog.summary_panel.table_view.isColumnHidden(index)
    assert (
        f"{summary.column_count:,} of {summary.column_count:,} fields shown"
        in dialog.summary_panel.summary_label.text()
    )


def test_summary_edits_emit_same_recipe_and_disable_stale_export(qtbot):
    dialog = _dialog(qtbot)
    table = _table()
    params = {**StatisticsRecipe().to_params(), "value_columns": "area"}
    dialog.set_summary(
        table=table,
        params=params,
        result=summarize_statistics(table, **params),
        stale=False,
    )
    dialog.show_tab("summary")
    edits = []
    dialog.summary_params_changed.connect(edits.append)
    dialog.statistics_panel.controls["group_by"].setCurrentIndex(
        dialog.statistics_panel.controls["group_by"].findData("condition")
    )
    assert edits[-1]["group_by"] == "condition"
    assert not dialog.export_button.isEnabled()


def test_plot_source_is_explicit_connection_not_averaging_setting(qtbot):
    dialog = _dialog(qtbot)
    changes = []
    dialog.plot_source_changed.connect(changes.append)
    dialog.plot_source.setCurrentIndex(dialog.plot_source.findData("summary"))
    assert changes == ["summary"]
    table = _table()
    summary = summarize_statistics(table, group_by="condition", value_columns="area")
    plot = build_plot_result(summary, y_column="area_mean", group_column="condition")
    dialog.set_plot(
        table=summary,
        result=plot,
        params=plot.recipe.to_params(),
        source_kind="summary",
        stale=False,
    )
    dialog.show_tab("plots")
    assert "2 summary rows" in dialog.plot_source_note.text()
    assert "not an original object" in dialog.plot_source_note.text()
    assert "2 plotted values" in dialog.plot_source_note.text()
    assert dialog.export_button.isEnabled()


def test_plot_from_original_data_explains_image_averaging(qtbot):
    dialog = _dialog(qtbot)
    table = _table()
    plot = build_plot_result(
        table, y_column="area", point_unit="Mean per image", image_column="image_id"
    )
    dialog.set_plot(
        table=table, result=plot, params=plot.recipe.to_params(), stale=False
    )
    assert "3 original measurement rows" in dialog.plot_source_note.text()
    assert "2 plotted values" in dialog.plot_source_note.text()
    assert "image means" in dialog.plot_status.text()


def test_plot_controls_keep_unavailable_column_instead_of_substitution(qtbot):
    dialog = _dialog(qtbot)
    dialog.set_plot(table=_table(), params={"y_column": "missing"}, stale=True)
    combo = dialog.plot_controls.controls["y_column"]
    assert combo.currentData() == "missing"
    assert "Unavailable" in combo.currentText()


def test_unavailable_workflow_disables_all_mutating_actions(qtbot):
    dialog = _dialog(qtbot)
    dialog.set_available(False, "Switch to this workflow to edit it.")
    assert not dialog.export_button.isEnabled()
    assert not dialog.add_plot_button.isEnabled()
    assert not dialog.add_summary_button.isEnabled()
    assert not dialog.plot_controls.isEnabled()
    assert not dialog.statistics_panel.isEnabled()
    assert not dialog.plot_source.isEnabled()
    assert dialog.close_button.isEnabled()


def test_busy_slot_retains_geometry_when_hidden(qtbot):
    dialog = _dialog(qtbot)
    qtbot.wait(10)
    bounds = dialog.tabs.geometry()
    footer = dialog.close_button.geometry()
    dialog.set_busy(True, "Preparing plot…")
    qtbot.wait(10)
    assert dialog.tabs.geometry() == bounds
    assert dialog.close_button.geometry() == footer
    assert dialog.progress.isVisible()
    dialog.set_busy(False)
    qtbot.wait(10)
    assert dialog.tabs.geometry() == bounds
    assert not dialog.progress.isVisible()


def test_close_is_rightmost_on_windows(qtbot):
    if sys.platform != "win32":
        return
    dialog = _dialog(qtbot)
    assert dialog.close_button.x() > dialog.export_button.x()


def test_add_and_show_actions_emit_bound_node_ids(qtbot):
    dialog = _dialog(qtbot)
    added = []
    shown = []
    dialog.add_plot_requested.connect(added.append)
    dialog.show_node_requested.connect(shown.append)
    dialog.add_plot_button.click()
    assert added == ["summary"]
    dialog.show_tab("summary")
    dialog.show_node_button.click()
    assert shown == ["summary"]


@pytest.mark.parametrize("width", [760, 1200])
def test_navigation_has_full_width_icon_tabs_and_prominent_show_node(qtbot, width):
    dialog = _dialog(qtbot)
    dialog.resize(width, 800)
    qtbot.wait(10)
    bar = dialog.tabs.tabBar()
    assert [dialog.tabs.tabText(i) for i in range(3)] == ["Data", "Summary", "Plots"]
    assert bar.width() >= dialog.tabs.width() - 4
    assert (
        max(bar.tabRect(i).width() for i in range(3))
        - min(bar.tabRect(i).width() for i in range(3))
        <= 2
    )
    shown = []
    dialog.show_node_requested.connect(shown.append)
    for index, node in enumerate(("measure", "summary", "plot")):
        dialog.tabs.setCurrentIndex(index)
        qtbot.wait(5)
        assert not dialog.tabs.tabIcon(index).isNull()
        button = dialog.show_node_button
        assert button.isVisible()
        assert not button.icon().isNull()
        assert button.mapTo(dialog, button.rect().topLeft()).y() < 210
        button.click()
        assert shown[-1] == node
    assert dialog.add_plot_button.accessibleName() == "Add plot"
    assert dialog.add_plot_button.text() == "+"


def test_unconfigured_summary_plot_is_a_setup_step_not_an_error(qtbot):
    dialog = _dialog(qtbot)
    summary = summarize_statistics(_table(), group_by="condition", value_columns="area")
    prompt = "Select a measurement on the left, such as Area — mean."
    dialog.set_plot(table=summary, setup_message=prompt)
    dialog.show_tab("plots")
    assert dialog.plot_setup_card.isVisible()
    assert dialog.plot_setup_note.text() == prompt
    assert not dialog.plot_canvas.isVisible()
    assert not dialog.plot_canvas.error_view.isVisible()
    assert not dialog.plot_status.isVisible()
    assert not dialog.export_button.isEnabled()
    assert not dialog.recalculate_button.isEnabled()
    plot = build_plot_result(summary, y_column="area_mean")
    dialog.set_plot(
        table=summary, result=plot, params=plot.recipe.to_params(), stale=False
    )
    assert not dialog.plot_setup_card.isVisible()
    assert dialog.plot_canvas.isVisible()
    assert dialog.export_button.isEnabled()
    assert dialog.recalculate_button.isEnabled()


def test_calculation_error_is_shown_once_and_input_description_stays_separate(qtbot):
    dialog = _dialog(qtbot)
    message = "Choose another field.\n\nThe saved field is missing from this table."
    dialog.set_plot(table=_table(), failed=True, message=message)
    dialog.show_tab("plots")
    assert not dialog.plot_status.isVisible()
    assert dialog.plot_canvas.error_view.isVisible()
    assert dialog.plot_canvas.error_title.text() == "Choose another field."
    assert dialog.plot_canvas.error_detail.text() == message.split("\n\n")[1]
    assert dialog.plot_input_toggle.text() == "Input: Original measurements · 3 rows"
    assert "No Statistics node is used" in dialog.plot_source_note.text()


def test_setup_action_opens_the_remaining_scatter_axis(qtbot, monkeypatch):
    dialog = _dialog(qtbot)
    summary = summarize_statistics(_table(), group_by="condition", value_columns="area")
    dialog.set_plot(
        table=summary,
        params={"plot_type": "Scatter", "y_column": "area_mean"},
        setup_message="Choose the X measurement.",
    )
    dialog.show_tab("plots")
    opened = []
    for name in ("x_column", "y_column"):
        monkeypatch.setattr(
            dialog.plot_controls.controls[name],
            "showPopup",
            lambda n=name: opened.append(n),
        )
    assert dialog.choose_measurement_button.text() == "Choose X measurement"
    dialog.choose_measurement_button.click()
    assert opened == ["x_column"]
    dialog.set_available(False)
    assert not dialog.choose_measurement_button.isEnabled()


def test_deleted_node_never_reenables_recipe_edits(qtbot):
    dialog = _dialog(qtbot)
    dialog.set_choices(summary_id="deleted-summary", plot_id="deleted-plot")
    dialog.set_available(True)
    dialog.set_summary(table=_table())
    dialog.set_plot(table=_table())
    assert not dialog.statistics_panel.isEnabled()
    assert not dialog.plot_controls.isEnabled()


def test_summary_root_is_not_mislabelled_original_objects(qtbot):
    dialog = _dialog(qtbot)
    summary = summarize_statistics(_table(), group_by="condition", value_columns="area")
    dialog.set_plot(table=summary, source_kind="original")
    assert "2 summary rows" in dialog.plot_source_note.text()
    assert "original measurement rows" not in dialog.plot_source_note.text()


def test_stale_table_cannot_export_through_hidden_panel(qtbot, tmp_path):
    dialog = _dialog(qtbot)
    dialog.set_data(dialog.data_panel.table, node_id="measure", stale=True)
    with pytest.raises(ValueError, match="not current"):
        dialog.data_panel.export_table(tmp_path / "stale.csv")
    assert not (tmp_path / "stale.csv").exists()


def test_file_dialog_rechecks_currentness_before_writing(qtbot, monkeypatch, tmp_path):
    from napari_vipp.ui import result_table_dialog

    dialog = _dialog(qtbot)
    target = tmp_path / "stale.csv"

    def choose(*_args, **_kwargs):
        dialog.set_data(dialog.data_panel.table, node_id="measure", stale=True)
        return target, "csv"

    monkeypatch.setattr(result_table_dialog, "choose_table_export_target", choose)
    dialog.data_panel.request_export()
    assert dialog.data_panel._active_export_worker is None
    assert not target.exists()


def test_bypassed_node_stays_read_only_across_availability_updates(qtbot):
    dialog = _dialog(qtbot)
    dialog.set_summary(table=_table(), editable=False)
    dialog.set_plot(table=_table(), editable=False)
    dialog.set_available(False)
    dialog.set_available(True)
    assert not dialog.statistics_panel.isEnabled()
    assert not dialog.plot_controls.isEnabled()
    assert not dialog.plot_source.isEnabled()


def test_close_during_background_export_keeps_snapshot_write_safe(
    qtbot, monkeypatch, tmp_path
):
    from napari_vipp.ui import result_table_dialog

    dialog = ResultsWorkspaceDialog()
    dialog.setAttribute(Qt.WA_DeleteOnClose, True)
    table = _table()
    dialog.set_data(table, node_id="measure")
    dialog.show()
    started = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    saved = []

    def save(snapshot, path, **_kwargs):
        started.set()
        if not release.wait(5):
            raise RuntimeError("Timed out waiting for test release")
        saved.append(snapshot)
        finished.set()
        return path

    monkeypatch.setattr(result_table_dialog, "save_table_output", save)
    dialog.data_panel._start_background_export(
        table, tmp_path / "result.csv", format="csv"
    )
    qtbot.waitUntil(started.is_set)
    dialog.close()
    qtbot.wait(10)
    release.set()
    qtbot.waitUntil(finished.is_set)
    qtbot.wait(20)
    assert saved == [table]


@pytest.mark.parametrize("width,height", [(760, 520), (1200, 800)])
def test_busy_geometry_is_stable_across_tabs_and_narrow_layout(qtbot, width, height):
    dialog = _dialog(qtbot)
    table = _table()
    summary = summarize_statistics(table, group_by="condition", value_columns="area")
    plot = build_plot_result(table, y_column="area")
    dialog.set_summary(table=table, result=summary, stale=False)
    dialog.set_plot(
        table=table, result=plot, params=plot.recipe.to_params(), stale=False
    )
    dialog.resize(width, height)
    for tab in ("data", "summary", "plots"):
        dialog.show_tab(tab)
        qtbot.wait(10)
        bounds = dialog.tabs.geometry()
        footer = dialog.close_button.geometry()
        for busy in (True, False, True, False):
            message = "Preparing the connected measurements and calculations. " * 8
            dialog.set_busy(busy, message)
            qtbot.wait(10)
            assert dialog.tabs.geometry() == bounds
            assert dialog.close_button.geometry() == footer
            assert dialog.rect().contains(footer)
            assert dialog.close_button.y() > dialog.tabs.geometry().bottom()
            assert dialog.busy_label.toolTip() == message
    assert dialog.width() == width
    assert dialog.height() == height
    for scroll in dialog._right_scrolls:
        assert scroll.horizontalScrollBar().maximum() == 0


def test_palette_and_style_changes_refresh_sidebar_table_and_canvas(qtbot):
    from napari_vipp.ui.palette_roles import theme_colors

    dialog = _dialog(qtbot)
    table = _table()
    plot = build_plot_result(table, y_column="area")
    dialog.set_plot(
        table=table, result=plot, params=plot.recipe.to_params(), stale=False
    )
    palette = QPalette(dialog.palette())
    for role, value in (
        (QPalette.Base, "#21252c"),
        (QPalette.AlternateBase, "#313845"),
        (QPalette.Window, "#21252c"),
        (QPalette.Text, "#ededed"),
        (QPalette.WindowText, "#ededed"),
    ):
        palette.setColor(role, QColor(value))
    dialog.setPalette(palette)
    expected = theme_colors(palette)
    qtbot.waitUntil(
        lambda: (
            expected.alternate_surface.name()
            in dialog._control_surfaces[0].styleSheet()
        )
    )
    assert all(
        expected.alternate_surface.name() in surface.styleSheet()
        for surface in dialog._control_surfaces
    )
    assert expected.surface.name() in dialog.data_panel.styleSheet()
    assert dialog.plot_canvas.result is plot
    face = dialog.plot_canvas.canvas.figure.get_facecolor()
    assert face[:3] == pytest.approx(expected.surface.getRgbF()[:3], abs=1e-6)
    dialog.changeEvent(QEvent(QEvent.StyleChange))
    qtbot.waitUntil(lambda: not dialog.theme_timer.isActive())
    assert dialog.plot_canvas.result is plot
