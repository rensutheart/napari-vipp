"""Acknowledgement follows the displayed scientific input through real edits."""

from napari_vipp._tests.test_results_workspace_lineage import _summary_plot
from napari_vipp._tests.test_results_workspace_widget import _settle
from napari_vipp._tests.test_statistics_widget import _statistics_widget
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
)
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.result_plots import build_plot_result, plot_state_from_data
from napari_vipp.core.statistics import summarize_statistics


def _publish_plot(widget, node, table):
    result = build_plot_result(table, **node.params)
    state = plot_state_from_data(result)
    widget.pipeline.outputs[node.id] = result
    widget.pipeline.node_outputs[node.id] = [result]
    widget.pipeline.output_states[node.id] = state
    widget.pipeline.node_output_states[node.id] = [state]
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    widget.pipeline.completed_node_ids.add(node.id)
    _settle(widget)
    return result


def test_dismissed_notes_follow_cached_input_not_each_plot_redraw(qtbot):
    widget, _source, summary, table, summary_result, _panel = _statistics_widget(qtbot)
    plot = _summary_plot(widget, summary, summary_result)
    dialog = widget._results_workspace.open_node(plot.id)
    assert dialog.plot_notes_frame.isVisible()
    before = widget._current_history_snapshot()
    dialog.plot_notes_dismiss_button.click()
    assert not dialog.plot_notes_frame.isVisible()
    assert dialog.plot_notes_reopen_button.isVisible()
    assert widget._current_history_snapshot() == before

    # A normal appearance edit creates another PlotData, not another input table.
    dialog.plot_params_changed.emit({**plot.params, "title": "New title"})
    assert widget.pipeline.input_data_for_node(plot.id) is summary_result
    redrawn = _publish_plot(widget, plot, summary_result)
    # PlotData freezes its own input snapshot; this copy must not be mistaken
    # for a fresh upstream calculation when deciding to restore notes.
    assert redrawn.source_table is not summary_result
    assert redrawn.source_table == summary_result
    assert widget.pipeline.input_data_for_node(plot.id) is summary_result
    assert not dialog.plot_notes_frame.isVisible()
    assert dialog.plot_notes_reopen_button.isVisible()

    # A new summary calculation is a new reviewed input revision, even if its
    # warning wording and resulting numeric values happen to be unchanged.
    replacement = summarize_statistics(table, **summary.params)
    assert replacement is not summary_result
    _publish_table_output(widget, summary.id, replacement)
    _publish_plot(widget, plot, replacement)
    assert dialog.plot_notes_frame.isVisible()
    assert not dialog.plot_notes_reopen_button.isVisible()
