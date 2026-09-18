"""Plot activity follows dispatch ownership rather than stale cached results."""

from dataclasses import replace
from threading import Event

import pytest

from napari_vipp._tests.test_plot_results_widget import _plot_widget
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
    _select,
)
from napari_vipp.core.pipeline import (
    EXECUTION_BLOCKED,
    EXECUTION_ERROR,
    EXECUTION_READY,
    EXECUTION_STALE,
)


def _panel(widget, node):
    return widget._result_plots.panels[widget._result_plots._context(node.id)]


def _start_run(widget, node_ids, *, active_node=None):
    widget._debounce_timer.stop()
    widget._pending_dirty_node_ids.clear()
    widget._active_pipeline_run_id = 987
    widget._pipeline_cancel_events[987] = Event()
    widget._pipeline_run_context[987] = (
        None,
        "test",
        active_node,
        None,
        None,
        None,
        frozenset(node_ids),
    )
    widget._set_pipeline_busy(True, active_node)


def _finish_run(widget):
    widget._active_pipeline_run_id = None
    widget._pipeline_run_context.pop(987, None)
    widget._pipeline_cancel_events.pop(987, None)
    widget._set_pipeline_busy(False)


def test_plot_edit_is_busy_during_debounce_and_clears_without_dispatch(qtbot):
    widget, source, node, _table, result = _plot_widget(qtbot)
    panel = _panel(widget, node)
    panel.open_plot()
    _select(widget, source.id)
    panel.params_changed.emit(replace(result.recipe, title="Updated").to_params())
    assert widget._debounce_timer.isActive()
    assert panel.busy
    assert panel.busy_message == "Plot update queued…"
    assert not panel.dialog.export_button.isEnabled()

    # The fixture's dispatcher is a no-op, as is an early no-input return. The
    # timeout must clear the queued indication without needing another edit.
    widget._debounce_timer.stop()
    widget._debounce_timer.timeout.emit()
    assert not panel.busy
    assert panel.stale


@pytest.mark.parametrize("state", [EXECUTION_STALE, EXECUTION_BLOCKED, EXECUTION_ERROR])
def test_idle_stale_blocked_or_failed_plot_does_not_spin(qtbot, state):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    widget._debounce_timer.stop()
    widget._mark_pipeline_dirty(source.id)
    widget.pipeline.node_execution_states[node.id] = state
    widget._result_plots.refresh()
    assert not _panel(widget, node).busy


def test_upstream_queued_edit_is_observed_after_parameter_timer_starts(qtbot):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    widget._debounce_timer.stop()
    widget._mark_pipeline_dirty(source.id)
    assert not _panel(widget, node).busy
    widget._debounce_timer.start(2000)
    qtbot.waitUntil(lambda: _panel(widget, node).busy)
    widget._debounce_timer.stop()
    widget._result_plots.refresh()
    assert not _panel(widget, node).busy


@pytest.mark.parametrize("active", ["source", "plot", "graph"])
def test_plot_busy_tracks_its_active_run_including_generic_graph_dispatch(
    qtbot, active
):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    panel = _panel(widget, node)
    panel.open_plot()
    active_id = {"source": source.id, "plot": node.id, "graph": None}[active]
    _start_run(widget, {source.id, node.id}, active_node=active_id)
    assert panel.busy
    assert panel.busy_message == (
        "Updating plot…" if active == "plot" else "Preparing plot measurements…"
    )
    assert not panel.dialog.export_button.isEnabled()
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    _finish_run(widget)
    assert not panel.busy
    assert not panel.stale
    assert panel.dialog.export_button.isEnabled()


def test_completed_plot_does_not_spin_while_other_branch_continues(qtbot):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    _start_run(widget, {source.id, node.id}, active_node=source.id)
    assert _panel(widget, node).busy
    widget._background_execution_state_overrides[node.id] = (987, EXECUTION_READY, "")
    widget._result_plots.refresh()
    assert not _panel(widget, node).busy
    _finish_run(widget)


def test_unrelated_background_run_never_marks_plot_busy(qtbot):
    widget, _source, node, _table, _result = _plot_widget(qtbot)
    _start_run(widget, {"input"}, active_node="input")
    assert not _panel(widget, node).busy
    _finish_run(widget)


@pytest.mark.parametrize("stop", ["cancel", "failure", "quarantine"])
def test_busy_clears_for_cancel_failure_or_runtime_quarantine(qtbot, stop):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    _start_run(widget, {source.id, node.id}, active_node=node.id)
    assert _panel(widget, node).busy
    if stop == "cancel":
        widget._pipeline_user_cancel_requested_run_id = 987
        widget._pipeline_cancel_events[987].set()
    elif stop == "quarantine":
        widget._compute_runtime_quarantined_reason = "Restart required"
    else:
        widget.pipeline.set_node_execution_error(node.id, "Invalid plot settings")
    widget._result_plots.refresh()
    assert not _panel(widget, node).busy
    _finish_run(widget)


def test_cancelled_dirty_intent_is_not_progress(qtbot):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    widget._debounce_timer.stop()
    widget._mark_pipeline_dirty(source.id)
    # Cancellation deliberately preserves dirty/manual intent for a later
    # explicit retry, but it does not schedule that retry.
    widget._pending_manual_node_ids.add(source.id)
    widget._pipeline_run_pending = False
    widget._result_plots.refresh()
    assert not _panel(widget, node).busy


def test_isolated_upstream_tuning_is_not_a_pending_plot(qtbot):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    widget._mark_pipeline_dirty(source.id)
    widget._isolated_tuning_node_id = source.id
    widget._debounce_timer.start(2000)
    widget._result_plots.refresh()
    assert not _panel(widget, node).busy
    widget._isolated_tuning_node_id = None
    widget._debounce_timer.stop()


def test_removed_node_closes_busy_detached_plot(qtbot):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    context = widget._result_plots._context(node.id)
    panel = _panel(widget, node)
    panel.open_plot()
    _start_run(widget, {source.id, node.id}, active_node=node.id)
    assert panel.busy
    widget._delete_node(node.id)
    widget._result_plots.refresh()
    assert context not in widget._result_plots.panels
    assert not panel.dialog.isVisible()
    _finish_run(widget)


def test_pending_plot_behind_unrequested_manual_measurement_does_not_spin(qtbot):
    widget, _source, node, table, _result = _plot_widget(qtbot)
    measurement = widget.add_node_from_palette("measure_objects")
    widget._connect_nodes("input", measurement.id)
    widget._connect_nodes(measurement.id, node.id)
    _publish_table_output(widget, measurement.id, table)
    widget.pipeline.set_node_auto_recalculate(measurement.id, False)
    widget._pending_dirty_node_ids.clear()
    widget._mark_pipeline_dirty(measurement.id)
    widget._debounce_timer.start(2000)
    widget._result_plots.refresh()
    assert not _panel(widget, node).busy
    widget._debounce_timer.stop()


def test_inactive_workflow_clears_busy_presentation(qtbot):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    panel = _panel(widget, node)
    original_index = widget._workflow_tabs.current_index
    _start_run(widget, {source.id, node.id}, active_node=node.id)
    assert panel.busy
    # Exercise the controller's session guard without altering the current
    # pipeline cache. Normal widget switching also has its own worker barrier.
    widget._workflow_tabs.create_blank()
    try:
        widget._result_plots.refresh()
        assert not panel.busy
        assert panel.stale
        assert "Return to" in panel.summary.text()
    finally:
        widget._workflow_tabs.activate(original_index)
        _finish_run(widget)


def test_superseded_calculation_keeps_only_real_replacement_queued(qtbot):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    _start_run(widget, {source.id, node.id}, active_node=node.id)
    widget._mark_pipeline_dirty(node.id)
    widget._pipeline_run_pending = True
    widget._pipeline_cancel_events[987].set()
    widget._result_plots.refresh()
    assert _panel(widget, node).busy
    assert _panel(widget, node).busy_message == "Plot update queued…"
    widget._pipeline_run_pending = False
    widget._result_plots.refresh()
    assert not _panel(widget, node).busy
    _finish_run(widget)
