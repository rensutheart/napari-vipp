"""Results navigation changes workflow ownership without changing its analysis."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QMessageBox

from napari_vipp._tests.test_results_workspace_widget import _plot, _settle
from napari_vipp._tests.test_statistics_widget import _statistics_widget
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
)
from napari_vipp.core.result_plots import build_plot_result, plot_state_from_data
from napari_vipp.core.statistics import summarize_statistics
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import save_workflow


def _two_workflows(qtbot):
    widget, source, summary, table, result, _panel = _statistics_widget(qtbot)
    plot, plotted = _plot(widget, source, table)
    first = SimpleNamespace(
        session=widget._workflow_tabs.current,
        source=source,
        summary=summary,
        table=table,
        result=result,
        plot=plot,
        plotted=plotted,
    )
    widget._new_workflow()
    source = widget.add_node_from_palette("table_source")
    summary = widget.add_node_from_palette("summarize_measurements")
    table = TableData(
        ("label_id", "image_id", "sample_id", "condition", "area", "intensity"),
        ((1, "c", "s3", "treated", 70.0, 150.0),),
        name="Independent experiment",
    )
    widget._connect_nodes(source.id, summary.id)
    summary.params.update(
        summary_version=2,
        group_by="condition",
        value_columns="area",
        statistics="count,mean,std",
    )
    result = summarize_statistics(table, **summary.params)
    _publish_table_output(widget, source.id, table)
    _publish_table_output(widget, summary.id, result)
    plot, plotted = _plot(widget, source, table)
    second = SimpleNamespace(
        session=widget._workflow_tabs.current,
        source=source,
        summary=summary,
        table=table,
        result=result,
        plot=plot,
        plotted=plotted,
    )
    # These collisions are typical when different workflows use the same nodes.
    assert first.source.id == second.source.id
    assert first.summary.id == second.summary.id
    assert first.plot.id == second.plot.id
    assert widget._activate_workflow_tab(0)
    _settle(widget)
    return widget, first, second


def _entry(widget, dialog):
    return next(
        entry
        for entry in widget._results_workspace.windows.values()
        if entry.dialog is dialog
    )


def _browse(dialog, session):
    index = dialog.workflow_selector.findData(session.session_id)
    assert index >= 0
    dialog.workflow_selector.setCurrentIndex(index)


def _analysis_state(session):
    pipeline = session.pipeline
    return (
        deepcopy([(n.id, n.operation_id, n.params) for n in pipeline.nodes.values()]),
        tuple(pipeline.connections),
        {key: id(value) for key, value in pipeline.outputs.items()},
        len(session.history.undo_stack),
        len(session.history.redo_stack),
    )


def test_browse_activates_owner_without_changing_graph_history_or_values(
    qtbot, monkeypatch
):
    widget, first, second = _two_workflows(qtbot)
    dialog = widget._results_workspace.open_node(first.source.id)
    original_key = next(iter(widget._results_workspace.windows))
    before = [_analysis_state(case.session) for case in (first, second)]
    calculations = []
    for case in (first, second):
        monkeypatch.setattr(
            case.session.pipeline, "run", lambda *a, **k: calculations.append(True)
        )
    for case in (second, first, second):
        _browse(dialog, case.session)
        assert widget._workflow_tabs.current is case.session
        assert _entry(widget, dialog).session_id == case.session.session_id
        assert dialog._data_table is case.table
        assert dialog.statistics_panel.result is case.result
        assert dialog.workflow_selector.currentData() == case.session.session_id
    assert list(widget._results_workspace.windows) == [original_key]
    assert widget._results_workspace.open_node(second.source.id) is dialog
    assert [_analysis_state(case.session) for case in (first, second)] == before
    assert calculations == []


@pytest.mark.parametrize("kind", ["summary", "plot"])
def test_edit_routes_to_selected_workflow_even_when_node_ids_collide(qtbot, kind):
    widget, first, second = _two_workflows(qtbot)
    dialog = widget._results_workspace.open_node(first.source.id)
    _browse(dialog, second.session)
    node = getattr(second, kind)
    original = deepcopy(getattr(first, kind).params)
    before_first = _analysis_state(first.session)
    before_history = len(second.session.history.undo_stack)
    if kind == "summary":
        dialog.summary_params_changed.emit({**node.params, "statistics": "median"})
        assert node.params["statistics"] == "median"
    else:
        dialog.summary_selector.setCurrentIndex(dialog.summary_selector.findData(""))
        assert dialog.plot_selector.currentData() == node.id
        dialog.plot_params_changed.emit({**node.params, "summary": "Median"})
        assert node.params["summary"] == "Median"
    assert getattr(first, kind).params == original
    assert _analysis_state(first.session) == before_first
    assert len(second.session.history.undo_stack) == before_history + 1
    assert second.session.pipeline.outputs[second.source.id] is second.table
    widget.undo()
    assert widget.pipeline.nodes[node.id].params == original
    assert _analysis_state(first.session) == before_first


@pytest.mark.parametrize("kind", ["summary", "plot"])
def test_add_node_is_owned_by_selected_workflow_and_one_undo_step(qtbot, kind):
    widget, first, second = _two_workflows(qtbot)
    dialog = widget._results_workspace.open_node(first.source.id)
    _browse(dialog, second.session)
    before_first = _analysis_state(first.session)
    nodes = set(second.session.pipeline.nodes)
    undo_count = len(second.session.history.undo_stack)
    button = dialog.add_summary_button if kind == "summary" else dialog.add_plot_button
    button.click()
    node_id = getattr(_entry(widget, dialog), f"{kind}_id")
    assert set(widget.pipeline.nodes) == nodes | {node_id}
    assert len(second.session.history.undo_stack) == undo_count + 1
    assert _analysis_state(first.session) == before_first
    connection = next(c for c in widget.pipeline.connections if c.target_id == node_id)
    expected = second.source.id if kind == "summary" else second.summary.id
    assert connection.source_id == expected
    widget.undo()
    assert set(widget.pipeline.nodes) == nodes
    assert _analysis_state(first.session) == before_first


def test_returning_to_workflow_restores_chosen_data_summary_and_plot(qtbot):
    widget, first, second = _two_workflows(qtbot)
    alternate_source = widget.add_node_from_palette("table_source")
    _publish_table_output(widget, alternate_source.id, first.table)
    alternate_summary = widget.add_node_from_palette("summarize_measurements")
    widget._connect_nodes(alternate_source.id, alternate_summary.id)
    alternate_summary.params.update(first.summary.params)
    _publish_table_output(widget, alternate_summary.id, first.result)
    alternate_plot, _ = _plot(widget, alternate_summary, first.table)
    alternate_plot.params["y_column"] = "area_mean"
    plotted = build_plot_result(first.result, **alternate_plot.params)
    state = plot_state_from_data(plotted)
    widget.pipeline.outputs[alternate_plot.id] = plotted
    widget.pipeline.node_outputs[alternate_plot.id] = [plotted]
    widget.pipeline.output_states[alternate_plot.id] = state
    widget.pipeline.node_output_states[alternate_plot.id] = [state]
    _settle(widget)
    dialog = widget._results_workspace.open_node(alternate_plot.id)
    assert tuple(dialog.data_selector.currentData()) == (alternate_source.id, 0)
    assert dialog.summary_selector.currentData() == alternate_summary.id
    assert dialog.plot_selector.currentData() == alternate_plot.id
    _browse(dialog, second.session)
    dialog.summary_selector.setCurrentIndex(dialog.summary_selector.findData(""))
    assert dialog.plot_selector.currentData() == second.plot.id
    _browse(dialog, first.session)
    assert tuple(dialog.data_selector.currentData()) == (alternate_source.id, 0)
    assert dialog.summary_selector.currentData() == alternate_summary.id
    assert dialog.plot_selector.currentData() == alternate_plot.id
    _browse(dialog, second.session)
    assert dialog.summary_selector.currentData() == ""
    assert dialog.plot_selector.currentData() == second.plot.id
    assert dialog._plot_result is second.plotted


def test_workflow_without_tables_clears_display_but_can_browse_back(qtbot):
    widget, first, _second = _two_workflows(qtbot)
    dialog = widget._results_workspace.open_node(first.plot.id)
    widget._new_workflow()
    empty = widget._workflow_tabs.current
    _browse(dialog, empty)
    assert _entry(widget, dialog).session_id == empty.session_id
    assert dialog._data_table is None
    assert dialog.statistics_panel.table is None
    assert dialog._plot_result is None
    assert dialog.current_node_id() == ""
    assert not dialog.add_summary_button.isEnabled()
    assert not dialog.add_plot_button.isEnabled()
    assert not dialog.export_button.isEnabled()
    assert dialog.workflow_selector.isEnabled()
    _browse(dialog, first.session)
    assert widget._workflow_tabs.current is first.session
    assert dialog._data_table is first.table
    assert dialog._plot_result is first.plotted
    assert dialog.export_button.isEnabled()


def test_external_tab_change_disables_edits_but_workflow_choice_can_reactivate(qtbot):
    widget, first, second = _two_workflows(qtbot)
    dialog = widget._results_workspace.open_node(first.source.id)
    assert widget._activate_workflow_tab(1)
    assert _entry(widget, dialog).session_id == first.session.session_id
    assert not dialog.statistics_panel.isEnabled()
    assert dialog.workflow_selector.isEnabled()
    before = [_analysis_state(case.session) for case in (first, second)]
    dialog.summary_params_changed.emit({**first.summary.params, "statistics": "median"})
    dialog.add_summary_requested.emit()
    assert [_analysis_state(case.session) for case in (first, second)] == before
    # QComboBox emits activated even when the same visible choice is picked.
    dialog.workflow_selector.activated.emit(dialog.workflow_selector.currentIndex())
    assert widget._workflow_tabs.current is first.session
    assert dialog.statistics_panel.isEnabled()
    assert widget._activate_workflow_tab(1)
    _browse(dialog, second.session)
    assert _entry(widget, dialog).session_id == second.session.session_id
    assert dialog.statistics_panel.isEnabled()
    assert dialog._data_table is second.table


def test_blocked_tab_switch_keeps_owner_and_does_not_route_edits_to_target(
    qtbot, monkeypatch
):
    widget, first, second = _two_workflows(qtbot)
    dialog = widget._results_workspace.open_node(first.source.id)
    before = [_analysis_state(case.session) for case in (first, second)]
    monkeypatch.setattr(
        widget, "_workflow_tab_switch_block_reason", lambda: "calculation finishes"
    )
    _browse(dialog, second.session)
    assert widget._workflow_tabs.current is first.session
    assert _entry(widget, dialog).session_id == first.session.session_id
    assert dialog.workflow_selector.currentData() == first.session.session_id
    assert dialog._data_table is first.table
    assert [_analysis_state(case.session) for case in (first, second)] == before
    assert "calculation finishes" in widget.status_label.text()


@pytest.mark.parametrize("failure", ["rejected", "partially_installed"])
def test_failed_activation_restores_owner_pipeline_and_visible_workflow_choice(
    qtbot, monkeypatch, failure
):
    widget, first, second = _two_workflows(qtbot)
    dialog = widget._results_workspace.open_node(first.source.id)
    before = [_analysis_state(case.session) for case in (first, second)]
    installed = []
    if failure == "partially_installed":
        install = widget._install_workflow_tab_session

        def fail_after_target_install(session):
            result = install(session)
            installed.append(session.session_id)
            if session is second.session:
                assert widget._workflow_tabs.current is second.session
                assert widget.pipeline is second.session.pipeline
                raise RuntimeError("Target presentation failed")
            return result

        monkeypatch.setattr(
            widget, "_install_workflow_tab_session", fail_after_target_install
        )
    else:
        monkeypatch.setattr(widget, "_activate_workflow_tab", lambda *a, **k: False)
    _browse(dialog, second.session)
    assert widget._workflow_tabs.current is first.session
    assert widget.pipeline is first.session.pipeline
    assert _entry(widget, dialog).session_id == first.session.session_id
    # The choice list and selected owner did not change, so cached refreshes
    # must still reset the user's newly selected, unsuccessful target.
    assert dialog.workflow_selector.currentData() == first.session.session_id
    assert dialog._data_table is first.table
    assert dialog.statistics_panel.result is first.result
    assert [_analysis_state(case.session) for case in (first, second)] == before
    if failure == "partially_installed":
        assert installed == [second.session.session_id, first.session.session_id]
        assert "Target presentation failed" in widget.status_label.text()


def test_closing_original_tab_preserves_window_but_closing_selected_owner_closes_it(
    qtbot, monkeypatch
):
    widget, first, second = _two_workflows(qtbot)
    dialog = widget._results_workspace.open_node(first.source.id)
    _browse(dialog, second.session)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Discard)
    widget._close_workflow_tab(widget._workflow_tabs.index_of(first.session.session_id))
    assert _entry(widget, dialog).session_id == second.session.session_id
    assert dialog.isVisible()
    assert dialog.workflow_selector.count() == 1
    assert dialog._data_table is second.table
    widget._close_workflow_tab(
        widget._workflow_tabs.index_of(second.session.session_id)
    )
    assert not widget._results_workspace.windows


def test_duplicate_workflow_names_rename_and_reorder_keep_identity(qtbot):
    widget, first, second = _two_workflows(qtbot)
    widget._rename_workflow_tab(0, "Experiment")
    widget._rename_workflow_tab(1, "Experiment")
    dialog = widget._results_workspace.open_node(first.source.id)
    labels = [dialog.workflow_selector.itemText(i) for i in range(2)]
    assert labels == ["Experiment · tab 1", "Experiment · tab 2"]
    widget._rename_workflow_tab(1, "Validation")
    assert dialog.workflow_selector.itemText(1) == "Validation"
    widget._reorder_workflow_tabs(1, 0)
    assert dialog.workflow_selector.itemData(0) == second.session.session_id
    assert dialog.workflow_selector.itemText(0) == "Validation"
    assert dialog.workflow_selector.itemData(1) == first.session.session_id
    assert dialog.workflow_selector.currentData() == first.session.session_id
    _browse(dialog, second.session)
    assert widget._workflow_tabs.current is second.session
    assert dialog._data_table is second.table


def test_save_as_refreshes_automatic_workflow_title_and_path_tooltip(
    qtbot, monkeypatch, tmp_path
):
    widget, source, _summary, _table, _result, _panel = _statistics_widget(qtbot)
    session = widget._workflow_tabs.current
    assert not session.title_is_custom
    dialog = widget._results_workspace.open_node(source.id)
    before = _analysis_state(session)
    for name in ("original-results", "renamed-results"):
        target = tmp_path / f"{name}.json"
        monkeypatch.setattr(
            QFileDialog, "getSaveFileName", lambda *a, path=target, **k: (str(path), "")
        )
        assert widget._save_workflow_dialog(force_choose_path=True)
        assert target.is_file()
        assert session.path == target.resolve()
        assert session.title == name
        assert dialog.workflow_selector.currentText() == name
        assert dialog.workflow_selector.currentData() == session.session_id
        tooltip = dialog.workflow_selector.itemData(
            dialog.workflow_selector.currentIndex(), Qt.ToolTipRole
        )
        assert str(target.resolve()) in tooltip
        assert str(target.resolve()) in dialog.workflow_selector.toolTip()
        assert _analysis_state(session) == before


@pytest.mark.parametrize("preserve_batch_workspace", [False, True])
def test_completed_load_refreshes_workflow_names_and_paths(
    qtbot, tmp_path, preserve_batch_workspace
):
    widget, source, _summary, _table, _result, _panel = _statistics_widget(qtbot)
    original = widget._workflow_tabs.current
    dialog = widget._results_workspace.open_node(source.id)
    target = tmp_path / "loaded-results.json"
    save_workflow(target, widget.pipeline)
    loaded = widget.load_workflow_file(
        target, preserve_batch_workspace=preserve_batch_workspace
    )
    assert loaded == target
    session = widget._workflow_tabs.current
    assert session.path == target.resolve()
    assert session.title == "loaded-results"
    index = dialog.workflow_selector.findData(session.session_id)
    assert index >= 0
    assert dialog.workflow_selector.itemText(index) == "loaded-results"
    assert str(target.resolve()) in dialog.workflow_selector.itemData(
        index, Qt.ToolTipRole
    )
    assert dialog.workflow_selector.count() == (1 if preserve_batch_workspace else 2)
    assert _entry(widget, dialog).session_id == original.session_id
    _browse(dialog, session)
    assert dialog.workflow_selector.currentData() == session.session_id
    assert str(target.resolve()) in dialog.workflow_selector.toolTip()
