"""Measurement bulk actions remain one workflow edit and retain upstream data."""

import json

from napari_vipp._tests.test_statistics_widget import _statistics_widget


def test_measurement_bulk_actions_are_single_undo_steps(qtbot):
    widget, source, node, table, _result, panel = _statistics_widget(qtbot)
    original = dict(node.params)
    history_count = len(widget._history.undo_stack)
    panel.select_all_measurements_button.click()
    widget._debounce_timer.stop()
    assert json.loads(node.params["value_columns"]) == ["area", "intensity"]
    assert not panel.auto_measurements.isChecked()
    assert len(widget._history.undo_stack) == history_count + 1
    assert widget.pipeline.outputs[source.id] is table
    assert source.id not in widget._pending_dirty_node_ids
    assert panel.stale

    panel.select_no_measurements_button.click()
    widget._debounce_timer.stop()
    assert node.params["value_columns"] == ""
    assert len(widget._history.undo_stack) == history_count + 2
    assert panel._checked(panel.measurements) == ()
    assert "Choose at least one" in panel.measurement_selection_note.text()
    assert widget.pipeline.outputs[source.id] is table

    widget.undo()
    widget._debounce_timer.stop()
    assert json.loads(widget.pipeline.nodes[node.id].params["value_columns"]) == [
        "area",
        "intensity",
    ]
    widget.undo()
    widget._debounce_timer.stop()
    assert widget.pipeline.nodes[node.id].params == original
    assert widget.pipeline.outputs[source.id] is table
