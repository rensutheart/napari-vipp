"""Results Workspace owns ordinary nodes, not a detached analysis recipe."""

from __future__ import annotations

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, Qt

from napari_vipp._tests.test_statistics_widget import _statistics_widget
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
    _select,
    _widget,
)
from napari_vipp.core.pipeline import EXECUTION_ERROR, EXECUTION_READY, SourcePayload
from napari_vipp.core.result_plots import build_plot_result, plot_state_from_data
from napari_vipp.core.statistics import summarize_statistics
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.workflow import load_workflow, save_workflow


def _settle(widget):
    widget._pending_dirty_node_ids.clear()
    widget._debounce_timer.stop()
    widget._results_workspace.refresh()


def _edges(widget):
    return {
        (c.source_id, c.source_port, c.target_id, c.target_port)
        for c in widget.pipeline.connections
    }


def _plot(widget, source, table, **params):
    """Publish a small known result without running image analysis."""
    node = widget.pipeline.add_node("plot_results")
    widget.graph_view.add_node(
        node, widget.graph_view.suggest_append_position(source.id)
    )
    widget._sync_node_input_ports(node.id)
    widget._sync_node_output_ports(node.id)
    widget._connect_nodes(source.id, node.id)
    node.params.update({"y_column": "area", "group_column": "condition", **params})
    result = build_plot_result(table, **node.params)
    state = plot_state_from_data(result)
    widget.pipeline.outputs[node.id] = result
    widget.pipeline.node_outputs[node.id] = [result]
    widget.pipeline.output_states[node.id] = state
    widget.pipeline.node_output_states[node.id] = [state]
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    widget.pipeline.completed_node_ids.add(node.id)
    _settle(widget)
    return node, result


def _open(widget, source):
    dialog = widget._results_workspace.open_node(source.id)
    key = (widget._workflow_tabs.current.session_id, source.id, 0)
    return dialog, key


def test_open_resolves_all_entries_without_mutating_or_calculating(qtbot, monkeypatch):
    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    plot, _ = _plot(widget, source, table)
    controller = widget._results_workspace
    before = widget._current_history_snapshot()
    undo_count = len(widget._history.undo_stack)
    calls = []
    monkeypatch.setattr(widget.pipeline, "run", lambda *a, **k: calls.append(True))
    dialog, key = _open(widget, source)
    assert dialog.tabs.currentIndex() == 0
    assert controller.open_node(summary.id) is dialog
    assert dialog.tabs.currentIndex() == 1
    assert dialog.summary_selector.currentData() == summary.id
    assert controller.open_node(plot.id) is dialog
    assert dialog.tabs.currentIndex() == 2
    assert dialog.plot_selector.currentData() == plot.id
    assert list(controller.windows) == [key]
    assert widget._current_history_snapshot() == before
    assert len(widget._history.undo_stack) == undo_count
    assert calls == []
    assert widget.pipeline.outputs[source.id] is table


def test_inspector_buttons_open_the_bound_workspace(qtbot):
    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    _select(widget, summary.id)
    widget._update_metadata_panel()
    assert not widget.results_workspace_button.isHidden()
    widget.results_workspace_button.click()
    dialog, key = _open(widget, source)
    plot, _ = _plot(widget, source, table)
    panel = widget._result_plots.panel_for(plot.id)
    panel.workspace_button.click()
    assert widget._results_workspace.windows[key].dialog is dialog
    assert dialog.tabs.currentIndex() == 2
    assert dialog.plot_selector.currentData() == plot.id


@pytest.mark.parametrize("kind", ["summary", "plot"])
def test_explicit_add_is_one_undoable_branch_and_preserves_other_edges(qtbot, kind):
    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    existing, _ = _plot(widget, source, table)
    dialog, key = _open(widget, source)
    before_nodes = set(widget.pipeline.nodes)
    before_edges = _edges(widget)
    before_history = len(widget._history.undo_stack)
    button = dialog.add_summary_button if kind == "summary" else dialog.add_plot_button
    button.click()
    node_id = getattr(widget._results_workspace.windows[key], f"{kind}_id")
    source_id = source.id if kind == "summary" else summary.id
    assert set(widget.pipeline.nodes) == before_nodes | {node_id}
    assert _edges(widget) == before_edges | {(source_id, 0, node_id, 0)}
    assert len(widget._history.undo_stack) == before_history + 1
    assert widget.pipeline.outputs[source.id] is table
    assert source.id not in widget._pending_dirty_node_ids
    assert summary.id in widget.pipeline.nodes and existing.id in widget.pipeline.nodes
    widget.undo()
    assert set(widget.pipeline.nodes) == before_nodes
    assert _edges(widget) == before_edges
    widget.redo()
    assert set(widget.pipeline.nodes) == before_nodes | {node_id}
    assert _edges(widget) == before_edges | {(source_id, 0, node_id, 0)}


def test_summary_edits_sync_both_directions_without_touching_measurements(qtbot):
    widget, source, summary, table, _result, inspector = _statistics_widget(qtbot)
    dialog, _key = _open(widget, source)
    before_history = len(widget._history.undo_stack)
    dialog.statistics_panel.params_changed.emit(
        {**summary.params, "statistics": "mean,median"}
    )
    assert summary.params["statistics"] == "mean,median"
    assert inspector.params["statistics"] == "mean,median"
    assert len(widget._history.undo_stack) == before_history + 1
    inspector.params_changed.emit({**summary.params, "statistics": "mean,std"})
    assert dialog.statistics_panel.params["statistics"] == "mean,std"
    assert widget.pipeline.outputs[source.id] is table
    assert source.id not in widget._pending_dirty_node_ids
    assert dialog.statistics_panel.stale


def test_plot_workspace_inspector_and_popout_share_only_selected_node(qtbot):
    widget, source, _summary, table, _result, _panel = _statistics_widget(qtbot)
    first, _ = _plot(widget, source, table)
    second, _ = _plot(widget, source, table)
    second_before = dict(second.params)
    dialog = widget._results_workspace.open_node(first.id)
    panel = widget._result_plots.panel_for(first.id)
    window = panel.open_plot()
    dialog.plot_controls.controls["summary"].setCurrentText("Median")
    assert first.params["summary"] == "Median"
    assert panel.params["summary"] == "Median"
    assert window.controls.controls["summary"].currentText() == "Median"
    window.controls.controls["summary"].setCurrentText("None")
    assert first.params["summary"] == "None"
    assert dialog.plot_controls.controls["summary"].currentText() == "None"
    assert second.params == second_before
    assert widget.pipeline.outputs[source.id] is table
    assert source.id not in widget._pending_dirty_node_ids
    assert not window.export_button.isEnabled()
    assert not dialog.export_button.isEnabled()


def test_summary_source_rewire_is_explicit_atomic_and_never_guesses_fields(qtbot):
    widget, source, summary, table, result, _panel = _statistics_widget(qtbot)
    plot, _ = _plot(
        widget, source, table, point_unit="Mean per image", image_column="image_id"
    )
    dialog = widget._results_workspace.open_node(plot.id)
    original = dict(plot.params)
    count = len(widget._history.undo_stack)
    dialog.plot_source.setCurrentIndex(dialog.plot_source.findData(summary.id))
    assert widget._results_workspace._input(plot.id) == (summary.id, 0)
    assert plot.params["point_unit"] == "Objects"
    assert plot.params["y_column"] == "area"
    assert len(widget._history.undo_stack) == count + 1
    assert "summary rows" in dialog.plot_source_note.text()
    with pytest.raises(ValueError, match="area"):
        build_plot_result(result, **plot.params)
    widget.undo()
    assert widget._results_workspace._input(plot.id) == (source.id, 0)
    assert widget.pipeline.nodes[plot.id].params == original
    widget.redo()
    assert widget._results_workspace._input(plot.id) == (summary.id, 0)
    assert widget.pipeline.nodes[plot.id].params["point_unit"] == "Objects"
    assert widget.pipeline.nodes[plot.id].params["y_column"] == "area"


def test_upstream_stale_invalidates_all_results_and_export(qtbot):
    widget, source, _summary, table, _result, _panel = _statistics_widget(qtbot)
    plot, _ = _plot(widget, source, table)
    dialog = widget._results_workspace.open_node(plot.id)
    assert dialog.export_button.isEnabled()
    widget._mark_pipeline_dirty(source.id)
    assert dialog.statistics_panel.stale
    assert not dialog._data_current
    assert not dialog._summary_current
    assert not dialog._plot_current
    for tab in range(3):
        dialog.tabs.setCurrentIndex(tab)
        assert not dialog.export_button.isEnabled()
    assert widget.pipeline.outputs[source.id] is table


def test_switch_workflow_rejects_edits_then_restores_window_owner(qtbot):
    widget, source, summary, _table, _result, _panel = _statistics_widget(qtbot)
    dialog, key = _open(widget, source)
    before = dict(summary.params)
    widget._new_workflow()
    assert not dialog.statistics_panel.isEnabled()
    assert not dialog.export_button.isEnabled()
    dialog.summary_params_changed.emit({**before, "statistics": "median"})
    assert summary.params == before
    assert widget._activate_workflow_tab(0, check_safety=False)
    assert widget._results_workspace.windows[key].dialog is dialog
    assert dialog.statistics_panel.isEnabled()
    assert dialog.statistics_panel.params == before


def test_deleted_source_disables_workspace_and_undo_restores_it(qtbot):
    widget, source, summary, _table, _result, _panel = _statistics_widget(qtbot)
    dialog, key = _open(widget, source)
    before = dict(summary.params)
    widget._delete_node(source.id)
    assert not dialog.add_summary_button.isEnabled()
    assert not dialog.statistics_panel.isEnabled()
    dialog.summary_params_changed.emit({**before, "statistics": "median"})
    assert summary.params == before
    widget.undo()
    assert widget._results_workspace.windows[key].dialog is dialog
    assert dialog.add_summary_button.isEnabled()
    assert dialog.statistics_panel.isEnabled()
    assert widget.pipeline.nodes[summary.id].params == before


@pytest.mark.parametrize("kind", ["summary", "plot"])
def test_stale_editor_revision_is_rejected_without_overwriting_new_recipe(qtbot, kind):
    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    plot, _ = _plot(widget, source, table)
    node = summary if kind == "summary" else plot
    dialog = widget._results_workspace.open_node(node.id)
    old = dict(node.params)
    changed = "statistics" if kind == "summary" else "summary"
    node.params[changed] = "median" if kind == "summary" else "Median"
    before = dict(node.params)
    getattr(dialog, f"{kind}_params_changed").emit({**old, "group_by": ""})
    assert node.params == before
    controls = dialog.statistics_panel if kind == "summary" else dialog.plot_controls
    params = controls.params if kind == "summary" else controls._params
    assert params[changed] == before[changed]


def test_new_upstream_table_revision_rejects_old_editor_values(qtbot):
    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    dialog, _key = _open(widget, source)
    old = dict(summary.params)
    replacement = type(table)(table.columns, table.rows + (table.rows[0],))
    _publish_table_output(widget, source.id, replacement)
    dialog.summary_params_changed.emit({**old, "statistics": "median"})
    assert summary.params == old
    assert dialog.statistics_panel.table is replacement
    # Republishing a current result makes this new revision editable normally.
    _publish_table_output(
        widget, summary.id, summarize_statistics(replacement, **summary.params)
    )
    _settle(widget)
    dialog.summary_params_changed.emit({**old, "statistics": "median"})
    assert summary.params["statistics"] == "median"


@pytest.mark.parametrize("kind", ["summary", "plot"])
def test_deleted_selection_does_not_retarget_edits_to_another_node(qtbot, kind):
    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    controller = widget._results_workspace
    dialog, key = _open(widget, source)
    if kind == "summary":
        selected = summary
        other = controller._add(key, "summarize_measurements")
    else:
        selected, _ = _plot(widget, source, table)
        other, _ = _plot(widget, source, table)
    controller.open_node(selected.id)
    other_params = dict(other.params)
    old = dict(selected.params)
    widget._delete_node(selected.id)
    assert getattr(controller.windows[key], f"{kind}_id") == selected.id
    controls = dialog.statistics_panel if kind == "summary" else dialog.plot_controls
    assert not controls.isEnabled()
    getattr(dialog, f"{kind}_params_changed").emit(
        {**old, "statistics": "median", "summary": "Median"}
    )
    assert other.params == other_params
    widget.undo()
    assert selected.id in widget.pipeline.nodes
    assert controls.isEnabled()
    assert getattr(controller.windows[key], f"{kind}_id") == selected.id


def test_workspace_edits_and_source_connections_survive_workflow_save(qtbot, tmp_path):
    widget, source, summary, table, result, _panel = _statistics_widget(qtbot)
    plot, _ = _plot(widget, source, table)
    dialog = widget._results_workspace.open_node(plot.id)
    dialog.plot_source.setCurrentIndex(dialog.plot_source.findData(summary.id))
    dialog.plot_params_changed.emit(
        {**plot.params, "y_column": "area_mean", "y_tick_interval": "5"}
    )
    assert plot.params["y_column"] == "area_mean"
    prepared = build_plot_result(result, **plot.params)
    assert prepared.source_table == result
    assert prepared.counts.plotted_points == result.row_count
    source.params.update(
        dataset_path="measurements.vipp-results.json", dataset_sha256="a" * 64
    )
    saved = tmp_path / "results-workspace.json"
    save_workflow(saved, widget.pipeline)
    restored = load_workflow(saved)
    nodes = {node.id: node for node in restored["nodes"]}
    assert nodes[summary.id].params == summary.params
    assert nodes[plot.id].params == plot.params
    restored_edges = {
        (c.source_id, c.source_port, c.target_id, c.target_port)
        for c in restored["connections"]
    }
    assert restored_edges == _edges(widget)
    assert (summary.id, 0, plot.id, 0) in restored_edges
    assert (source.id, 0, plot.id, 0) not in restored_edges
    assert '"rows"' not in saved.read_text(encoding="utf-8")


def test_added_branches_are_spaced_and_duplicate_titles_are_distinguishable(qtbot):
    widget, source, _summary, _table, _result, _panel = _statistics_widget(qtbot)
    dialog, key = _open(widget, source)
    controller = widget._results_workspace
    original_positions = {
        node_id: widget.graph_view.node_scene_rect(node_id).topLeft()
        for node_id in widget.pipeline.nodes
    }
    added = []
    for operation in (
        "summarize_measurements",
        "summarize_measurements",
        "plot_results",
        "plot_results",
    ):
        node = controller._add(key, operation)
        assert node is not None
        added.append(node.id)
    assert {
        node_id: widget.graph_view.node_scene_rect(node_id).topLeft()
        for node_id in original_positions
    } == original_positions
    for node_id in added:
        rect = widget.graph_view.node_scene_rect(node_id)
        for other_id in widget.pipeline.nodes:
            if node_id != other_id:
                assert not rect.intersects(widget.graph_view.node_scene_rect(other_id))
    for combo in (dialog.summary_selector, dialog.plot_selector):
        labels = [combo.itemText(index) for index in range(combo.count())]
        assert len(labels) == len(set(labels))
        assert all(
            combo.itemData(index) not in label
            for index, label in enumerate(labels)
            if combo.itemData(index)
        )
        assert all(
            label == widget._node_title(combo.itemData(index))
            for index, label in enumerate(labels)
            if combo.itemData(index)
        )


def test_workspace_uses_shared_node_names_and_live_settings_descriptions(
    qtbot, monkeypatch
):
    from napari_vipp.ui.node_labels import build_node_presentations

    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    plot, _ = _plot(widget, source, table)
    aliases = {source.id: "Cell measurements", summary.id: "Well comparison"}

    def presentation(node_id):
        return build_node_presentations(
            widget.pipeline, aliases, tables={source.id: table}
        ).get(node_id)

    monkeypatch.setattr(widget, "_node_presentation", presentation, raising=False)
    monkeypatch.setattr(
        widget, "_node_title", lambda node_id: presentation(node_id).name
    )
    dialog, _key = _open(widget, source)
    widget._results_workspace.open_node(summary.id)
    assert dialog.data_selector.currentText() == "Cell measurements"
    assert dialog.summary_selector.currentText() == "Well comparison"
    tooltip = dialog.summary_selector.currentData(Qt.ToolTipRole)
    assert "Operation: Statistics" in tooltip
    assert f"Node ID: {summary.id}" in tooltip
    assert "Mean, SD" in tooltip
    assert "Grouped by condition" in tooltip
    assert "Output:" in dialog.data_selector.currentData(Qt.ToolTipRole)

    summary.params["statistics"] = "median"
    widget._results_workspace.refresh()
    assert dialog.summary_selector.currentText() == "Well comparison"
    assert "Median" in dialog.summary_selector.currentData(Qt.ToolTipRole)
    assert "Mean, SD" not in dialog.summary_selector.toolTip()

    widget._results_workspace.open_node(plot.id)
    assert dialog.plot_selector.currentText() == "area by condition"
    assert "Operation: Plot Results" in dialog.plot_selector.toolTip()
    assert f"Node ID: {plot.id}" in dialog.plot_selector.toolTip()


@pytest.mark.parametrize("use_summary", [False, True])
def test_renaming_plot_input_refreshes_caption_without_recomputing(qtbot, use_summary):
    widget, source, summary, table, summary_result, _panel = _statistics_widget(qtbot)
    input_node = summary if use_summary else source
    plot, result = _plot(
        widget, input_node, summary_result if use_summary else table,
        y_column="area_mean" if use_summary else "area",
    )
    dialog = widget._results_workspace.open_node(plot.id)
    automatic_name = widget._node_title(input_node.id)
    assert automatic_name in dialog.plot_input_toggle.text()
    outputs = dict(widget.pipeline.outputs)
    params = {key: dict(node.params) for key, node in widget.pipeline.nodes.items()}

    assert widget._set_node_name(input_node.id, "All reviewed cells")
    assert "All reviewed cells" in dialog.plot_input_toggle.text()
    assert "All reviewed cells" in dialog.plot_input_toggle.toolTip()
    assert "All reviewed cells" in dialog.plot_source.currentText()
    assert automatic_name not in dialog.plot_input_toggle.text()
    assert dialog._plot_result is result
    assert all(widget.pipeline.outputs[key] is value for key, value in outputs.items())
    assert {
        key: dict(node.params) for key, node in widget.pipeline.nodes.items()
    } == params
    assert not widget._pending_dirty_node_ids
    assert not widget._debounce_timer.isActive()

    widget.undo()
    assert automatic_name in dialog.plot_input_toggle.text()
    assert "All reviewed cells" not in dialog.plot_input_toggle.text()
    assert dialog._plot_result is result
    assert all(widget.pipeline.outputs[key] is value for key, value in outputs.items())


def test_added_summary_plot_waits_for_selection_without_pipeline_error(qtbot):
    widget, source, summary, _table, result, _panel = _statistics_widget(qtbot)
    dialog, key = _open(widget, source)
    controller = widget._results_workspace
    node = controller._add(key, "plot_results", summary.id)
    assert node is not None
    assert node.params["y_column"] == "auto"
    assert not widget._debounce_timer.isActive()
    assert controller.pending_plot_setup_node_ids() == {node.id}
    assert "Choose a measurement" in dialog._plot_setup_message
    assert not dialog.recalculate_button.isEnabled()
    assert not dialog.export_button.isEnabled()
    assert widget.pipeline.node_execution_states.get(node.id) != EXECUTION_ERROR
    assert "Pipeline error" not in widget.status_label.text()
    assert widget.pipeline.outputs[summary.id] is result
    # Other settings can be authored without treating an incomplete recipe as
    # a failed computation. Scatter requires both explicit summary fields.
    dialog.plot_params_changed.emit({**node.params, "plot_type": "Scatter"})
    assert not widget._debounce_timer.isActive()
    dialog.plot_params_changed.emit({**node.params, "y_column": "area_mean"})
    assert "X measurement" in dialog._plot_setup_message
    assert not widget._debounce_timer.isActive()
    # Inspector edits share exactly the same readiness gate.
    panel = widget._result_plots.panel_for(node.id)
    panel.params_changed.emit({**node.params, "x_column": "area_count"})
    assert not controller.pending_plot_setup_node_ids()
    assert widget._debounce_timer.isActive()
    assert not dialog._plot_setup_message
    assert node.params["y_column"] == "area_mean"
    assert node.params["x_column"] == "area_count"


def test_unconfigured_plot_is_excluded_from_unrelated_automatic_runs(
    qtbot, monkeypatch
):
    widget, source, summary, table, result, _panel = _statistics_widget(qtbot)
    existing, _ = _plot(widget, source, table)
    dialog, key = _open(widget, source)
    controller = widget._results_workspace
    node = controller._add(key, "plot_results", summary.id)
    # Closing the editor must not remove the safety gate. The saved graph, not
    # the dialog's lifetime, determines whether a field remains unselected.
    dialog.close()
    calls = []
    monkeypatch.setattr(
        widget,
        "_source_payloads_for_pipeline",
        lambda: ({source.id: SourcePayload(table, name="Measurements")}, []),
    )
    monkeypatch.setattr(widget, "_uncached_async_file_source_specs", lambda: ())
    monkeypatch.setattr(widget._table_sources, "pending", lambda: False)
    monkeypatch.setattr(
        widget, "_start_background_pipeline_run", lambda *a, **k: calls.append(a[9])
    )
    monkeypatch.setattr(
        widget, "_run_pipeline_synchronously", lambda *a, **k: calls.append(a[9])
    )
    widget._mark_pipeline_dirty(source.id)
    type(widget).run_pipeline(widget, force_sync=True)
    assert len(calls) == 1
    assert node.id not in calls[0]
    assert {source.id, summary.id, existing.id} <= calls[0]
    assert node.params["y_column"] == "auto"
    # Interactive deferral must not weaken executable scientific contracts.
    with pytest.raises(ValueError, match="Choose a summary measurement"):
        build_plot_result(result, **node.params)


def test_authored_missing_summary_field_is_not_hidden_as_setup(qtbot):
    widget, source, summary, _table, result, _panel = _statistics_widget(qtbot)
    dialog, key = _open(widget, source)
    controller = widget._results_workspace
    node = controller._add(key, "plot_results", summary.id)
    dialog.plot_params_changed.emit({**node.params, "y_column": "old_measurement"})
    assert not controller.plot_setup_message(node.id)
    assert node.id not in controller.pending_plot_setup_node_ids()
    assert widget._debounce_timer.isActive()
    with pytest.raises(ValueError, match="old_measurement"):
        build_plot_result(result, **node.params)


def test_summary_setup_gate_uses_effective_input_through_bypass(qtbot):
    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    dialog, key = _open(widget, source)
    controller = widget._results_workspace
    node = controller._add(key, "plot_results", summary.id)
    assert controller.plot_setup_message(node.id)
    summary.execution_mode = "bypass"
    assert not controller.plot_setup_message(node.id)
    _publish_table_output(widget, summary.id, table)
    controller.refresh()
    assert "Original measurements" in dialog.plot_source.currentText()
    assert "bypassed" in dialog.plot_source.currentText()
    assert "original measurement rows" in dialog.plot_source_note.text()
    summary.execution_mode = "run"
    _publish_table_output(
        widget, summary.id, summarize_statistics(table, **summary.params)
    )
    assert controller.plot_setup_message(node.id)


def test_summary_setup_gate_uses_cold_table_source_payload(qtbot):
    widget, source, _summary, _table, result, _panel = _statistics_widget(qtbot)
    _dialog, key = _open(widget, source)
    controller = widget._results_workspace
    node = controller._add(key, "plot_results", source.id)
    assert not controller.plot_setup_message(node.id)
    assert controller.pending_plot_setup_node_ids(
        source_payloads={source.id: SourcePayload(result)}
    ) == {node.id}


def test_unselected_measurement_cannot_export_cached_plot_after_other_run(
    qtbot, monkeypatch
):
    widget, source, summary, table, summary_result, _panel = _statistics_widget(qtbot)
    node, _ = _plot(widget, source, table)
    widget.pipeline.connect(summary.id, node.id)
    node.params["y_column"] = "area_mean"
    result = build_plot_result(summary_result, **node.params)
    widget.pipeline.outputs[node.id] = result
    widget.pipeline.node_outputs[node.id] = [result]
    widget.pipeline.output_states[node.id] = plot_state_from_data(result)
    widget.pipeline.node_output_states[node.id] = [plot_state_from_data(result)]
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    _settle(widget)
    controller = widget._results_workspace
    dialog = controller.open_node(node.id)
    panel = widget._result_plots.panel_for(node.id)
    window = panel.open_plot()
    assert window.export_button.isEnabled()
    dialog.plot_params_changed.emit({**node.params, "y_column": "auto"})
    assert not window.export_button.isEnabled()
    monkeypatch.setattr(
        widget,
        "_source_payloads_for_pipeline",
        lambda: ({source.id: SourcePayload(table, name="Measurements")}, []),
    )
    monkeypatch.setattr(widget, "_uncached_async_file_source_specs", lambda: ())
    monkeypatch.setattr(widget._table_sources, "pending", lambda: False)

    def dispatch(*args, **kwargs):
        assert node.id not in args[9]
        widget._begin_pipeline_dispatch(args[7])

    monkeypatch.setattr(widget, "_start_background_pipeline_run", dispatch)
    monkeypatch.setattr(widget, "_run_pipeline_synchronously", dispatch)
    type(widget).run_pipeline(widget, force_sync=True)
    widget._result_plots.refresh()
    controller.refresh()
    assert node.id not in widget._pending_dirty_node_ids
    assert widget.pipeline.node_execution_states[node.id] != EXECUTION_READY
    assert widget.pipeline.outputs[node.id] is result
    assert not window.export_button.isEnabled()
    assert not dialog.export_button.isEnabled()
    widget._show_workflow_ready_status("Measurements", snapshots_pinned=False)
    assert "1 plot needs a measurement" in widget.status_label.text()
    assert "calculations complete" not in widget.status_label.text()


def test_workspace_calculate_updates_automatic_plot(qtbot):
    widget, source, _summary, table, _result, _panel = _statistics_widget(qtbot)
    node, _ = _plot(widget, source, table)
    _dialog, key = _open(widget, source)
    widget._results_workspace._calculate(key, node.id)
    assert widget._debounce_timer.isActive()
    assert node.id in widget._pending_dirty_node_ids


def test_show_node_focuses_graph_and_reveals_main_window(qtbot, monkeypatch):
    widget, source, summary, _table, _result, _panel = _statistics_widget(qtbot)
    dialog, key = _open(widget, source)
    controller = widget._results_workspace
    calls = []
    monkeypatch.setattr(
        widget.graph_view, "focus_node", lambda value: calls.append(value)
    )
    monkeypatch.setattr(widget, "raise_", lambda: calls.append("raise"))
    monkeypatch.setattr(widget, "activateWindow", lambda: calls.append("activate"))
    controller._show_node(key, summary.id)
    assert widget._selected_node_id == summary.id
    assert calls == [summary.id, "raise", "activate"]
    assert controller.windows[key].dialog is dialog
    assert not dialog.isVisible()
    assert controller.open_node(summary.id) is dialog
    assert dialog.isVisible()


def test_adding_from_removed_source_choice_is_a_nonmutating_noop(qtbot):
    widget, source, summary, _table, _result, _panel = _statistics_widget(qtbot)
    _dialog, key = _open(widget, source)
    widget._delete_node(summary.id)
    before = widget._current_history_snapshot()
    history = len(widget._history.undo_stack)
    assert widget._results_workspace._add(key, "plot_results", summary.id) is None
    assert widget._current_history_snapshot() == before
    assert len(widget._history.undo_stack) == history
    widget._delete_node(source.id)
    before = widget._current_history_snapshot()
    history = len(widget._history.undo_stack)
    assert widget._results_workspace._add(key, "summarize_measurements") is None
    assert widget._current_history_snapshot() == before
    assert len(widget._history.undo_stack) == history


def test_bypassed_summary_cannot_export_passthrough_as_a_current_summary(qtbot):
    widget, source, summary, table, _result, _panel = _statistics_widget(qtbot)
    plot, _ = _plot(widget, source, table)
    dialog = widget._results_workspace.open_node(plot.id)
    before = dict(summary.params)
    assert widget.pipeline.node_supports_bypass(summary.id)
    widget.pipeline.add_output_tunnel("Summary output", summary.id)
    assert widget._set_node_bypassed(summary.id, True), widget.status_label.text()
    # A successful bypass forwards raw rows but does not calculate Statistics.
    _publish_table_output(widget, summary.id, table)
    _settle(widget)
    dialog.summary_selector.setCurrentIndex(
        dialog.summary_selector.findData(summary.id)
    )
    dialog.tabs.setCurrentIndex(1)
    assert not dialog.statistics_panel.isEnabled()
    assert dialog.statistics_panel.stale
    assert not dialog._summary_current
    assert not dialog.export_button.isEnabled()
    dialog.summary_params_changed.emit({**before, "statistics": "median"})
    assert summary.params == before
    dialog.tabs.setCurrentIndex(2)
    # Tabs keep the selected chain. Select the direct-input path explicitly
    # to return to the original-data plot.
    assert not dialog._plot_bound
    dialog.summary_selector.setCurrentIndex(dialog.summary_selector.findData(""))
    assert dialog._plot_current
    assert dialog.export_button.isEnabled()
    assert widget._results_workspace._input(plot.id) == (source.id, 0)


def test_tunnel_inputs_resolve_to_one_workspace_and_remain_tunnels_on_save(
    qtbot, tmp_path
):
    widget, source, summary, table, result, _panel = _statistics_widget(qtbot)
    plot, _ = _plot(widget, source, table)
    widget.pipeline.add_output_tunnel("Measured objects", source.id)
    widget._connect_input_to_tunnel("Measured objects", summary.id, 0)
    widget._connect_input_to_tunnel("Measured objects", plot.id, 0)
    _publish_table_output(widget, summary.id, result)
    _settle(widget)
    dialog, key = _open(widget, source)
    assert widget._results_workspace.open_node(summary.id) is dialog
    assert widget._results_workspace.open_node(plot.id) is dialog
    assert list(widget._results_workspace.windows) == [key]
    dialog.plot_params_changed.emit({**plot.params, "summary": "Median"})
    assert plot.params["summary"] == "Median"
    for node in (summary, plot):
        connection = widget.pipeline.tunnel_connection_for_input(node.id, 0)
        assert connection is not None and connection.tunnel_name == "Measured objects"
    saved = tmp_path / "tunnel-results.json"
    save_workflow(saved, widget.pipeline)
    restored = load_workflow(saved)
    consumers = {summary.id, plot.id}
    saved_connections = [c for c in restored["connections"] if c.target_id in consumers]
    assert len(saved_connections) == 2
    assert all(c.tunnel_name == "Measured objects" for c in saved_connections)
    assert all(
        c.source_id == source.id and c.source_port == 0 for c in saved_connections
    )


def test_multi_output_tables_keep_distinct_workspace_and_created_branch_ports(qtbot):
    widget = _widget(qtbot)
    source = widget.add_node_from_palette("skeleton_graph_tables")
    nodes = TableData(("node_id", "degree"), ((1, 2), (2, 1)), name="Graph nodes")
    edges = TableData(("edge_id", "length"), ((1, 4.5),), name="Graph edges")
    _publish_table_output(widget, source.id, nodes)
    widget.pipeline.node_outputs[source.id].append(edges)
    widget.pipeline.node_output_states[source.id].append(
        TableState(edges.row_count, edges.column_count, edges.columns)
    )
    _settle(widget)
    controller = widget._results_workspace
    session = widget._workflow_tabs.current.session_id
    first = controller.open_node(source.id, source_port=0)
    second = controller.open_node(source.id, source_port=1)
    assert first is not second
    assert first._data_table is nodes
    assert second._data_table is edges
    assert set(controller.windows) == {(session, source.id, 0), (session, source.id, 1)}
    summary = controller._add((session, source.id, 1), "summarize_measurements")
    assert summary is not None
    assert controller._input(summary.id) == (source.id, 1)
    assert controller.open_node(summary.id) is second
    assert second.summary_selector.currentData() == summary.id
    assert first.summary_selector.findData(summary.id) == -1


def test_closed_workspace_late_callback_cannot_edit_the_graph(qtbot):
    widget, source, summary, _table, _result, _panel = _statistics_widget(qtbot)
    dialog, key = _open(widget, source)
    controller = widget._results_workspace
    before = dict(summary.params)
    dialog.close_button.click()
    qtbot.waitUntil(lambda: key not in controller.windows)
    controller._commit(key, "summary", {**before, "statistics": "median"})
    controller.refresh()
    assert summary.params == before


def test_immediate_close_reopen_ignores_old_signals_and_old_deferred_deletion(qtbot):
    widget, source, summary, _table, _result, _panel = _statistics_widget(qtbot)
    old, key = _open(widget, source)
    before = dict(summary.params)
    old.close()
    new = widget._results_workspace.open_node(summary.id)
    assert new is not old
    old.summary_params_changed.emit({**before, "statistics": "median"})
    assert summary.params == before
    QCoreApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    assert widget._results_workspace.windows[key].dialog is new
    assert new.isVisible()
    new.summary_params_changed.emit({**before, "statistics": "mean,median"})
    assert summary.params["statistics"] == "mean,median"
