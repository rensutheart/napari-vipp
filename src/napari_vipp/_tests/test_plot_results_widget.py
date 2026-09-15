"""Plot Results crosses the inspector boundary without becoming image data."""

from __future__ import annotations

from dataclasses import replace

from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
    _select,
    _widget,
)
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.result_plots import build_plot_result, plot_state_from_data
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import load_workflow, save_workflow


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
