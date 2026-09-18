"""Results navigation follows exact table connections, without editing them."""

from __future__ import annotations

import pytest

from napari_vipp._tests.test_results_workspace_widget import (
    _edges,
    _open,
    _plot,
    _settle,
)
from napari_vipp._tests.test_statistics_widget import _statistics_widget
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
    _widget,
)
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.result_plots import build_plot_result, plot_state_from_data
from napari_vipp.core.statistics import summarize_statistics
from napari_vipp.core.tables import TableData, TableState


def _branch(widget, operation, source, *, source_port=0):
    node = widget.pipeline.add_node(operation)
    widget.graph_view.add_node(
        node, widget.graph_view.suggest_append_position(source.id)
    )
    widget._sync_node_input_ports(node.id)
    widget._sync_node_output_ports(node.id)
    result = widget.pipeline.connect(
        source.id, node.id, source_port=source_port, target_port=0
    )
    assert result.success, result.message
    widget._apply_connection_result_to_graph(result)
    return node


def _summary(widget, source, table, *, source_port=0):
    node = _branch(widget, "summarize_measurements", source, source_port=source_port)
    node.params.update(
        group_by="condition", value_columns="area", statistics="count,mean"
    )
    result = summarize_statistics(table, **node.params)
    _publish_table_output(widget, node.id, result)
    _settle(widget)
    return node, result


def _summary_plot(widget, summary, table):
    node = _branch(widget, "plot_results", summary)
    node.params.update(y_column="area_mean", group_column="condition")
    result = build_plot_result(table, **node.params)
    state = plot_state_from_data(result)
    widget.pipeline.outputs[node.id] = result
    widget.pipeline.node_outputs[node.id] = [result]
    widget.pipeline.output_states[node.id] = state
    widget.pipeline.node_output_states[node.id] = [state]
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    widget.pipeline.completed_node_ids.add(node.id)
    _settle(widget)
    return node


def _values(combo):
    return {combo.itemData(index) for index in range(combo.count())}


def _choose(combo, value):
    index = next(
        (index for index in range(combo.count()) if combo.itemData(index) == value), -1
    )
    assert index >= 0, f"Missing choice {value!r}: {_values(combo)!r}"
    combo.setCurrentIndex(index)


def _chain(dialog):
    return (
        dialog.data_selector.currentData(),
        dialog.summary_selector.currentData(),
        dialog.plot_selector.currentData(),
    )


def test_opening_unplotted_summary_cannot_display_sibling_or_original_plot(qtbot):
    widget, source, first, table, result, _panel = _statistics_widget(qtbot)
    original_plot, _ = _plot(widget, source, table)
    sibling_plot = _summary_plot(widget, first, result)
    unplotted, _ = _summary(widget, source, table)
    controller = widget._results_workspace
    dialog = controller.open_node(sibling_plot.id)
    before = widget._current_history_snapshot()
    assert controller.open_node(unplotted.id) is dialog
    dialog.show_tab("plots")

    assert dialog.summary_selector.currentData() == unplotted.id
    assert not dialog.plot_selector.currentData()
    assert sibling_plot.id not in _values(dialog.plot_selector)
    assert original_plot.id not in _values(dialog.plot_selector)
    assert not dialog._plot_bound
    assert not dialog.show_node_button.isEnabled()
    assert not dialog.export_button.isEnabled()
    assert not dialog.recalculate_button.isEnabled()
    assert widget._current_history_snapshot() == before


def test_statistics_middle_choice_shows_only_its_connected_plots(qtbot):
    widget, source, first, table, result, _panel = _statistics_widget(qtbot)
    original, _ = _plot(widget, source, table)
    first_plot = _summary_plot(widget, first, result)
    second, second_result = _summary(widget, source, table)
    second_plot = _summary_plot(widget, second, second_result)
    other_second_plot = _summary_plot(widget, second, second_result)
    dialog = widget._results_workspace.open_node(first.id)
    before = widget._current_history_snapshot()
    dialog.show_tab("plots")
    assert _values(dialog.plot_selector) == {first_plot.id}

    _choose(dialog.summary_selector, second.id)
    assert _values(dialog.plot_selector) == {second_plot.id, other_second_plot.id}
    assert dialog.current_node_id() in {second_plot.id, other_second_plot.id}

    _choose(dialog.summary_selector, "")
    assert _values(dialog.plot_selector) == {original.id}
    assert dialog.current_node_id() == original.id
    assert dialog.plot_source.currentData() == source.id

    _choose(dialog.summary_selector, first.id)
    assert dialog.summary_selector.currentData() == first.id
    assert _values(dialog.plot_selector) == {first_plot.id}
    assert dialog.current_node_id() == first_plot.id
    assert widget._current_history_snapshot() == before


@pytest.mark.parametrize("initial_tab", ["data", "summary", "plots"])
def test_data_navigation_keeps_independent_sources_separate_and_does_no_work(
    qtbot, monkeypatch, initial_tab
):
    widget, source, first, table, result, _panel = _statistics_widget(qtbot)
    first_plot = _summary_plot(widget, first, result)
    other_source = widget.add_node_from_palette("table_source")
    other_table = TableData(table.columns, table.rows[:1], name="Other measurements")
    _publish_table_output(widget, other_source.id, other_table)
    other_summary, other_result = _summary(widget, other_source, other_table)
    other_plot = _summary_plot(widget, other_summary, other_result)
    dialog, key = _open(widget, source)
    dialog.show_tab(initial_tab)
    before = widget._current_history_snapshot()
    history = len(widget._history.undo_stack)
    calls = []
    monkeypatch.setattr(widget.pipeline, "run", lambda *a, **k: calls.append(True))

    _choose(dialog.data_selector, (other_source.id, 0))
    assert widget._results_workspace.windows[key].root == (other_source.id, 0)
    assert dialog._data_table is other_table
    assert _values(dialog.summary_selector) == {"", other_summary.id}
    assert first_plot.id not in _values(dialog.plot_selector)
    # All three choices form one coherent chain immediately, whichever tab was
    # visible when the input changed. Tabs must not choose a different branch.
    assert _chain(dialog) == ((other_source.id, 0), other_summary.id, other_plot.id)
    dialog.show_tab("summary")
    dialog.show_tab("plots")
    assert _values(dialog.plot_selector) == {other_plot.id}

    _choose(dialog.data_selector, (source.id, 0))
    assert widget._results_workspace.windows[key].root == (source.id, 0)
    assert dialog._data_table is table
    assert _values(dialog.summary_selector) == {"", first.id}
    assert _chain(dialog) == ((source.id, 0), first.id, first_plot.id)
    assert other_plot.id not in _values(dialog.plot_selector)
    assert widget._current_history_snapshot() == before
    assert len(widget._history.undo_stack) == history
    assert calls == []
    assert not widget._debounce_timer.isActive()


def test_transformed_table_output_is_a_distinct_input_not_its_ancestor(qtbot):
    widget, source, original_summary, table, _result, _panel = _statistics_widget(qtbot)
    selected = _branch(widget, "select_table_columns", source)
    selected.params["columns"] = "condition,area"
    transformed = TableData(
        ("condition", "area"), tuple((row[3], row[4]) for row in table.rows)
    )
    _publish_table_output(widget, selected.id, transformed)
    summary, result = _summary(widget, selected, transformed)
    plot = _summary_plot(widget, summary, result)
    dialog = widget._results_workspace.open_node(plot.id)
    before = _edges(widget)

    assert dialog.data_selector.currentData() == (selected.id, 0)
    assert dialog._data_table is transformed
    assert _values(dialog.summary_selector) == {"", summary.id}
    assert original_summary.id not in _values(dialog.summary_selector)
    assert dialog.summary_selector.currentData() == summary.id
    assert dialog.plot_selector.currentData() == plot.id
    _choose(dialog.data_selector, (source.id, 0))
    assert _values(dialog.summary_selector) == {"", original_summary.id}
    assert plot.id not in _values(dialog.plot_selector)
    assert _edges(widget) == before


def test_data_selector_distinguishes_ports_and_keeps_exact_table_payload(qtbot):
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
    dialog = controller.open_node(source.id, source_port=0)
    labels = []
    for port in (0, 1):
        index = next(
            (
                index
                for index in range(dialog.data_selector.count())
                if dialog.data_selector.itemData(index) == (source.id, port)
            ),
            -1,
        )
        assert index >= 0
        label = dialog.data_selector.itemText(index)
        assert (
            widget.pipeline.output_ports(source.id)[port].label.lower() in label.lower()
        )
        labels.append(label)
    assert labels[0] != labels[1]
    before = widget._current_history_snapshot()
    _choose(dialog.data_selector, (source.id, 1))
    assert dialog._data_table is edges
    _choose(dialog.data_selector, (source.id, 0))
    assert dialog._data_table is nodes
    assert widget._current_history_snapshot() == before
    # The window key was created for port 0. Adding after navigating to port 1
    # must use the active table output, not that original key's output port.
    _choose(dialog.data_selector, (source.id, 1))
    dialog.show_tab("plots")
    before_edges = _edges(widget)
    dialog.add_plot_button.click()
    plot_id = dialog.plot_selector.currentData()
    assert plot_id
    assert _edges(widget) == before_edges | {(source.id, 1, plot_id, 0)}
    assert dialog.summary_selector.currentData() == ""


def test_opening_plot_synchronizes_its_summary_after_other_context_was_viewed(qtbot):
    widget, source, first, table, first_result, _panel = _statistics_widget(qtbot)
    first_plot = _summary_plot(widget, first, first_result)
    second, second_result = _summary(widget, source, table)
    second_plot = _summary_plot(widget, second, second_result)
    original_plot, _ = _plot(widget, source, table)
    controller = widget._results_workspace
    dialog = controller.open_node(first_plot.id)
    assert controller.open_node(second.id) is dialog
    assert dialog.summary_selector.currentData() == second.id
    assert dialog.plot_selector.currentData() == second_plot.id

    assert controller.open_node(first_plot.id) is dialog
    assert dialog.summary_selector.currentData() == first.id
    assert dialog.plot_selector.currentData() == first_plot.id
    assert dialog.current_node_id() == first_plot.id
    assert controller.open_node(original_plot.id) is dialog
    assert dialog.summary_selector.currentData() == ""
    assert dialog.plot_selector.currentData() == original_plot.id
    assert dialog.current_node_id() == original_plot.id


def test_add_plot_from_unplotted_summary_connects_that_summary_in_one_undo(qtbot):
    widget, source, first, table, result, _panel = _statistics_widget(qtbot)
    _summary_plot(widget, first, result)
    selected, _ = _summary(widget, source, table)
    controller = widget._results_workspace
    dialog = controller.open_node(selected.id)
    dialog.show_tab("plots")
    before_nodes = set(widget.pipeline.nodes)
    before_edges = _edges(widget)
    history = len(widget._history.undo_stack)

    dialog.add_plot_button.click()

    added = set(widget.pipeline.nodes) - before_nodes
    assert len(added) == 1
    plot_id = added.pop()
    assert widget.pipeline.nodes[plot_id].operation_id == "plot_results"
    assert _edges(widget) == before_edges | {(selected.id, 0, plot_id, 0)}
    assert len(widget._history.undo_stack) == history + 1
    assert dialog.summary_selector.currentData() == selected.id
    assert dialog.plot_selector.currentData() == plot_id
    assert controller.plot_setup_message(plot_id)
    widget.undo()
    assert set(widget.pipeline.nodes) == before_nodes
    assert _edges(widget) == before_edges


def test_show_node_follows_the_scoped_plot_and_is_disabled_for_empty_scope(
    qtbot, monkeypatch
):
    widget, source, first, table, result, _panel = _statistics_widget(qtbot)
    unrelated = _summary_plot(widget, first, result)
    second, second_result = _summary(widget, source, table)
    selected_plot = _summary_plot(widget, second, second_result)
    empty, _ = _summary(widget, source, table)
    controller = widget._results_workspace
    dialog = controller.open_node(first.id)
    dialog.show_tab("plots")
    _choose(dialog.summary_selector, second.id)
    focused = []
    monkeypatch.setattr(widget.graph_view, "focus_node", focused.append)

    dialog.show_node_button.click()

    assert focused == [selected_plot.id]
    assert widget._selected_node_id == selected_plot.id
    assert unrelated.id not in focused
    assert not dialog.isVisible()
    assert controller.open_node(empty.id) is dialog
    dialog.show_tab("plots")
    assert not dialog.show_node_button.isEnabled()
    dialog.show_node_button.click()
    assert focused == [selected_plot.id]


def test_explicit_direct_input_choice_survives_refresh_and_every_tab(qtbot):
    widget, source, summary, table, result, _panel = _statistics_widget(qtbot)
    summary_plot = _summary_plot(widget, summary, result)
    original_plot, _ = _plot(widget, source, table)
    dialog, _key = _open(widget, source)
    controller = widget._results_workspace
    assert _chain(dialog) == ((source.id, 0), summary.id, summary_plot.id)
    before = widget._current_history_snapshot()
    history = len(widget._history.undo_stack)

    _choose(dialog.summary_selector, "")
    assert "None" in dialog.summary_selector.currentText()
    expected = ((source.id, 0), "", original_plot.id)
    assert _chain(dialog) == expected
    for tab in ("summary", "plots", "data", "plots", "summary"):
        dialog.show_tab(tab)
        controller.refresh()
        assert _chain(dialog) == expected
        assert not dialog._summary_bound
        assert not dialog.statistics_panel.isEnabled()
        assert _values(dialog.plot_selector) == {original_plot.id}
        if tab == "summary":
            assert not dialog.show_node_button.isEnabled()
            assert not dialog.recalculate_button.isEnabled()
            assert not dialog.export_button.isEnabled()

    assert widget._current_history_snapshot() == before
    assert len(widget._history.undo_stack) == history


def test_tabs_never_change_global_chain_or_selected_plot(qtbot):
    widget, source, first, table, result, _panel = _statistics_widget(qtbot)
    _summary_plot(widget, first, result)
    second, second_result = _summary(widget, source, table)
    _summary_plot(widget, second, second_result)
    second_plot = _summary_plot(widget, second, second_result)
    dialog = widget._results_workspace.open_node(first.id)
    before = widget._current_history_snapshot()
    history = len(widget._history.undo_stack)
    _choose(dialog.summary_selector, second.id)
    _choose(dialog.plot_selector, second_plot.id)
    expected = ((source.id, 0), second.id, second_plot.id)

    for tab in ("data", "summary", "plots", "summary", "data", "plots"):
        dialog.show_tab(tab)
        widget._results_workspace.refresh()
        assert _chain(dialog) == expected
        assert dialog.data_selector.isVisible()
        assert dialog.summary_selector.isVisible()
        assert dialog.plot_selector.isVisible()
    assert widget._current_history_snapshot() == before
    assert len(widget._history.undo_stack) == history


def test_add_plot_with_no_statistics_connects_direct_input_in_one_undo(qtbot):
    widget, source, summary, _table, result, _panel = _statistics_widget(qtbot)
    _summary_plot(widget, summary, result)
    dialog = widget._results_workspace.open_node(summary.id)
    _choose(dialog.summary_selector, "")
    dialog.show_tab("plots")
    assert not dialog.plot_selector.currentData()
    before_nodes = set(widget.pipeline.nodes)
    before_edges = _edges(widget)
    history = len(widget._history.undo_stack)

    dialog.add_plot_button.click()

    added = set(widget.pipeline.nodes) - before_nodes
    assert len(added) == 1
    plot_id = added.pop()
    assert _edges(widget) == before_edges | {(source.id, 0, plot_id, 0)}
    assert len(widget._history.undo_stack) == history + 1
    assert _chain(dialog) == ((source.id, 0), "", plot_id)
    widget.undo()
    assert set(widget.pipeline.nodes) == before_nodes
    assert _edges(widget) == before_edges


@pytest.mark.parametrize("kind", ["summary", "plot"])
def test_deleted_global_choice_stays_unavailable_and_does_not_edit_another_node(
    qtbot, kind
):
    widget, source, first, table, result, _panel = _statistics_widget(qtbot)
    first_plot = _summary_plot(widget, first, result)
    second, second_result = _summary(widget, source, table)
    second_plot = _summary_plot(widget, second, second_result)
    selected = first if kind == "summary" else first_plot
    other = second if kind == "summary" else second_plot
    dialog = widget._results_workspace.open_node(first_plot.id)
    before = dict(other.params)
    widget._delete_node(selected.id)

    dialog.show_tab(kind if kind == "summary" else "plots")
    combo = dialog.summary_selector if kind == "summary" else dialog.plot_selector
    controls = dialog.statistics_panel if kind == "summary" else dialog.plot_controls
    assert combo.currentData() == selected.id
    assert not controls.isEnabled()
    assert not dialog.show_node_button.isEnabled()
    assert not dialog.recalculate_button.isEnabled()
    assert not dialog.export_button.isEnabled()
    getattr(dialog, f"{kind}_params_changed").emit(
        {**selected.params, "statistics": "median", "summary": "Median"}
    )
    assert other.params == before

    widget.undo()
    assert combo.currentData() == selected.id
    assert controls.isEnabled()
    assert dialog.show_node_button.isEnabled()
