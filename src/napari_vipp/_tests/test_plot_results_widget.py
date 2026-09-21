"""Plot Results crosses the inspector boundary without becoming image data."""

from __future__ import annotations

from dataclasses import replace

import pytest

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
    _select,
    _widget,
)
from napari_vipp.core.pipeline import EXECUTION_ERROR, EXECUTION_READY
from napari_vipp.core.result_plots import build_plot_result, plot_state_from_data
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import load_workflow, save_workflow
from napari_vipp.ui.status import MessageSeverity


def _plot_widget(qtbot):
    widget = _widget(qtbot)
    source = widget.add_node_from_palette("table_source")
    node = widget.add_node_from_palette("plot_results")
    table = TableData(
        ("label_id", "area", "intensity", "image_id", "condition"),
        (
            (1, 12.0, 45.0, "A", "control"),
            (2, 18.0, 64.0, "A", "control"),
            (1, 25.0, 86.0, "B", "treated"),
        ),
        name="Labelled objects",
        table_kind="object measurements",
        column_units=(("area", "µm²"),),
    )
    _publish_table_output(widget, source.id, table)
    widget._connect_nodes(source.id, node.id)
    node.params.update(y_column="area", group_column="condition")
    result = build_plot_result(table, **node.params)
    state = plot_state_from_data(result)
    widget.pipeline.outputs[node.id] = result
    widget.pipeline.node_outputs[node.id] = [result]
    widget.pipeline.output_states[node.id] = state
    widget.pipeline.node_output_states[node.id] = [state]
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    widget.pipeline.completed_node_ids.add(node.id)
    widget._pending_dirty_node_ids.clear()
    _select(widget, node.id)
    return widget, source, node, table, result


def test_plot_inspector_and_graph_are_not_image_surfaces(qtbot):
    widget, _source, node, _table, result = _plot_widget(qtbot)
    widget._update_metadata_panel()
    widget._update_histogram()
    profile = widget._inspector_profile_for_node(node.id)
    assert profile.output_action_kind == "plot"
    assert not profile.supports_pin
    assert not widget._node_can_pin(node.id)
    assert widget.histograms_section.isHidden()
    assert widget.table_group.isHidden()
    key = widget._result_plots._context(node.id)
    panel = widget._result_plots.panels[key]
    assert panel.result is result
    assert not panel.stale
    thumbnail = widget._result_plots.thumbnail(node.id, result)
    assert thumbnail.shape == (240, 384, 3)
    # Exercise the actual card seam, not just the renderer in isolation.
    widget._update_node_thumbnail(
        node.id, result, plot_state_from_data(result), 0, queue_stack_contrast=False
    )
    widget._update_thumbnails()
    assert widget._data_kind(result) == "plot"


def test_detached_plot_edits_remain_bound_to_owner_and_preserve_upstream(qtbot):
    widget, source, node, table, result = _plot_widget(qtbot)
    key = widget._result_plots._context(node.id)
    panel = widget._result_plots.panels[key]
    panel.open_plot()
    _select(widget, source.id)
    assert panel.dialog.isVisible()
    values = replace(result.recipe, title="My morphology", summary="Median").to_params()
    panel.params_changed.emit(values)
    assert node.params["title"] == "My morphology"
    assert node.params["summary"] == "Median"
    assert "title" not in source.params
    assert widget.pipeline.outputs[source.id] is table
    assert source.id not in widget._pending_dirty_node_ids
    assert node.id in widget._pending_dirty_node_ids
    assert panel.stale
    assert not panel.dialog.export_button.isEnabled()


def test_plot_recipe_round_trip_does_not_embed_measurements(qtbot, tmp_path):
    widget, source, node, _table, result = _plot_widget(qtbot)
    # A configured source reference is stored separately from cached rows.
    source.params.update(
        dataset_path="measurements.vipp-results.json", dataset_sha256="a" * 64
    )
    key = widget._result_plots._context(node.id)
    widget._result_plots.panels[key].params_changed.emit(
        replace(result.recipe, plot_type="Scatter", x_column="intensity").to_params()
    )
    path = tmp_path / "plots.json"
    save_workflow(path, widget.pipeline)
    loaded = load_workflow(path)
    restored = next(item for item in loaded["nodes"] if item.id == node.id)
    assert restored.params["plot_type"] == "Scatter"
    assert restored.params["x_column"] == "intensity"
    assert '"rows"' not in path.read_text(encoding="utf-8")


def test_upstream_edit_marks_open_plot_stale(qtbot):
    widget, source, node, _table, _result = _plot_widget(qtbot)
    panel = widget._result_plots.panels[widget._result_plots._context(node.id)]
    panel.open_plot()
    widget._mark_pipeline_dirty(source.id)
    widget._result_plots.refresh()
    assert panel.stale
    assert not panel.dialog.export_button.isEnabled()


@pytest.mark.parametrize("actionable", [True, False])
@pytest.mark.parametrize("plot_count", [1, 2])
def test_plot_failure_status_is_compact_but_keeps_full_error(
    qtbot, actionable, plot_count
):
    widget = _widget(qtbot)
    plots = [widget.add_node_from_palette("plot_results") for _ in range(plot_count)]
    error = (
        "Cannot average one image into several groups.\n\n"
        "Choose Objects, or a grouping column that is constant within each image."
    )
    for node in plots:
        widget.pipeline.set_node_execution_error(node.id, error)
    formatted_error = f"Pipeline error: {error}"

    widget._set_pipeline_error_status(
        formatted_error,
        failed_node_ids=(node.id for node in plots),
        actionable=actionable,
    )

    assert widget.status_label.text() == (
        "Pipeline error: Cannot average one image into several groups. "
        "See the plot window or inspector for how to fix it."
    )
    assert widget.status_label.toolTip() == formatted_error
    assert formatted_error in widget.status_label.accessibleDescription()
    assert widget.status_label.severity is MessageSeverity.ERROR
    assert widget.status_label.actionable is actionable
    for node in plots:
        assert widget.pipeline.node_execution_states[node.id] == EXECUTION_ERROR
        assert widget.pipeline.node_execution_messages[node.id] == error


@pytest.mark.parametrize("failures", ["other", "mixed", "unknown", "unattributed"])
def test_other_pipeline_failures_do_not_use_plot_specific_status(qtbot, failures):
    widget = _widget(qtbot)
    plot = widget.add_node_from_palette("plot_results")
    other = widget.add_node_from_palette("binary_threshold")
    failed_ids = {
        "other": (other.id,),
        "mixed": (plot.id, other.id),
        "unknown": (plot.id, "removed-node"),
        "unattributed": (),
    }[failures]
    error = (
        "Pipeline error: Scientific calculation failed.\n"
        "Do not obscure this unrelated explanation or partial-result warning."
    )
    widget._set_pipeline_error_status(error, failed_node_ids=failed_ids)

    assert widget.status_label.text() == error
    assert widget.status_label.toolTip() == ""
    assert widget.status_label.severity is MessageSeverity.ERROR
    assert widget.status_label.actionable


@pytest.mark.parametrize("select_source", [False, True])
def test_plot_execution_error_refreshes_inspector_and_detached_plot(
    qtbot, select_source
):
    widget, source, node, _table, result = _plot_widget(qtbot)
    panel = widget._result_plots.panels[widget._result_plots._context(node.id)]
    panel.open_plot()
    if select_source:
        _select(widget, source.id)
    error = (
        "Cannot average one image into several groups.\n\n"
        "Choose Objects, or a grouping column that is constant within each image."
    )
    widget.pipeline.set_node_execution_error(node.id, error)
    widget._result_plots.refresh()

    assert panel.failed
    assert panel.error_message == error
    assert panel.result is result  # Retain the scientific cache, not its old drawing.
    assert panel.plot.canvas is None
    assert panel.dialog.plot.canvas is None
    assert not panel.plot.error_view.isHidden()
    assert not panel.dialog.plot.error_view.isHidden()
    assert panel.plot.error_title.text() == error.split("\n\n")[0]
    assert panel.dialog.plot.error_detail.text() == error.split("\n\n")[1]
    assert not panel.dialog.export_button.isEnabled()
    assert not panel.dialog.data_button.isEnabled()
    assert panel.open_button.text() == "Review plot…"
    assert panel.open_button.isEnabled()

    # Execution state belongs in the refresh stamp even if cached data survives.
    widget.pipeline.node_execution_states[node.id] = EXECUTION_READY
    widget.pipeline.node_execution_messages.pop(node.id, None)
    widget._result_plots.refresh()
    assert not panel.failed
    assert panel.plot.error_view.isHidden()
    assert panel.dialog.plot.error_view.isHidden()
    assert panel.dialog.plot.canvas is not None
    assert panel.dialog.export_button.isEnabled()
