"""Graph shortcuts open cached owner results, never another node or a run."""

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_tracking_ui import _result
from napari_vipp._tests.test_ui_inspector_widget_integration import (
    _publish_table_output,
    _select,
    _widget,
)
from napari_vipp.core.operations import intensity_histogram
from napari_vipp.core.pipeline import EXECUTION_READY, EXECUTION_STALE
from napari_vipp.core.tables import TableData, TableState
from napari_vipp.core.tracking import build_tracks


def _record_execution_requests(widget, monkeypatch):
    requests = []
    for name in ("run_pipeline", "_calculate_node", "_commit_crop_draft"):
        monkeypatch.setattr(
            widget, name,
            lambda *args, name=name, **kwargs: requests.append((name, args, kwargs)),
        )
    return requests


def _trajectory_widget(qtbot):
    """Publish a resident fixture route without requesting application execution."""
    widget = _widget(qtbot)
    detector = widget.add_node_from_palette("detect_spots_per_frame")
    linker = widget.add_node_from_palette("build_tracks")
    for source_id, target_id in (("input", detector.id), (detector.id, linker.id)):
        connection = widget.pipeline.connect(source_id, target_id)
        assert connection.success
        widget._apply_connection_result_to_graph(connection)
    observations, source, state = _result(linked=False)
    tracked, summary = build_tracks(
        observations, maximum_gap=1, maximum_displacement=4,
    )
    widget.pipeline.outputs["input"] = source
    widget.pipeline.node_outputs["input"] = [source]
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.completed_node_ids.add("input")
    widget.pipeline.node_execution_states["input"] = EXECUTION_READY
    _publish_table_output(widget, detector.id, observations)
    _publish_table_output(widget, linker.id, tracked)
    widget.pipeline.node_outputs[linker.id].append(summary)
    widget.pipeline.node_output_states[linker.id].append(
        TableState(summary.row_count, summary.column_count, summary.columns),
    )
    widget._pending_dirty_node_ids.clear()
    _select(widget, "gaussian")
    widget._debounce_timer.stop()
    return widget, detector, linker, observations, tracked, summary, source


def test_trajectory_cards_open_exact_owner_off_selection_and_reuse_window(
    qtbot, monkeypatch,
):
    widget, detector, linker, observations, tracked, summary, source = (
        _trajectory_widget(qtbot)
    )
    widget._inspector_output_port_by_node[linker.id] = 1
    requests = _record_execution_requests(widget, monkeypatch)
    source_before = source.copy()
    rows_before = observations.rows, tracked.rows, summary.rows
    controller = widget._tracking_results
    dialog = None

    for node, table in (
        (detector, observations),
        (linker, tracked),
        (detector, observations),
    ):
        card = widget.graph_view._cards[node.id]
        assert not card.trajectory_review_button.isHidden()
        assert not card.result_button.isHidden()
        card.trajectory_review_button.click()
        if dialog is None:
            dialog = controller.dialog
        assert controller.dialog is dialog
        assert dialog is not None and dialog.isVisible()
        assert dialog.node_id == node.id
        assert dialog.table_panel.table is table
        assert dialog._source is source
        assert controller._dialog_is_current()
        assert widget._selected_node_id == "gaussian"
        assert widget._inspector_output_port_by_node[linker.id] == 1

    assert observations.rows == rows_before[0]
    assert tracked.rows == rows_before[1]
    assert summary.rows == rows_before[2]
    np.testing.assert_array_equal(source, source_before)
    assert not source.flags.writeable
    assert requests == []


@pytest.mark.parametrize("operation_id", ("detect_spots_per_frame", "build_tracks"))
@pytest.mark.parametrize(
    "unavailable,reason",
    (
        ("missing", "requires observations"),
        ("stale", "current settings"),
        ("pending", "pending calculation"),
        ("nonresident", "current and resident"),
    ),
)
def test_trajectory_card_unavailable_results_do_not_open_or_calculate(
    qtbot, monkeypatch, operation_id, unavailable, reason,
):
    widget, detector, linker, *_outputs = _trajectory_widget(qtbot)
    node = detector if operation_id == "detect_spots_per_frame" else linker
    if unavailable == "missing":
        widget.pipeline.outputs.pop(node.id)
        widget.pipeline.node_outputs.pop(node.id)
    elif unavailable == "stale":
        widget.pipeline.node_execution_states[node.id] = EXECUTION_STALE
    elif unavailable == "pending":
        widget._pipeline_run_pending = True
    else:
        widget.pipeline.outputs.pop("input")
        widget.pipeline.node_outputs.pop("input")
    requests = _record_execution_requests(widget, monkeypatch)
    card = widget.graph_view._cards[node.id]

    card.trajectory_review_button.click()

    assert widget._tracking_results.dialog is None
    assert reason in widget.status_label.text()
    assert widget._selected_node_id == "gaussian"
    assert not card.trajectory_review_button.isHidden()
    assert requests == []


@pytest.mark.parametrize(
    "operation_id", ("measure_objects", "measure_objects_intensity"),
)
def test_temporal_measurement_card_review_visibility_uses_exact_primary_table(
    qtbot, monkeypatch, operation_id,
):
    widget = _widget(qtbot)
    node = widget.add_node_from_palette(operation_id)
    _select(widget, "gaussian")
    widget._debounce_timer.stop()
    requests = _record_execution_requests(widget, monkeypatch)
    card = widget.graph_view._cards[node.id]
    observations, *_source = _result(linked=False)
    static = replace(observations, observation_metadata=None)

    assert card.trajectory_review_button.isHidden()
    for table in (static, observations, static):
        _publish_table_output(widget, node.id, table)
        widget._tracking_results.refresh()
        assert card.trajectory_review_button.isHidden() == (
            table.observation_metadata is None
        )
        assert not card.result_button.isHidden()
        assert widget._selected_node_id == "gaussian"
    assert requests == []


@pytest.mark.parametrize("state", (EXECUTION_READY, EXECUTION_STALE))
def test_measurements_card_opens_its_owner_off_selection_and_reuses_window(
    qtbot, monkeypatch, state,
):
    widget = _widget(qtbot)
    measurements = widget.add_node_from_palette("measure_objects")
    _publish_table_output(widget, measurements.id)
    widget.pipeline.node_execution_states[measurements.id] = state
    _select(widget, "gaussian")
    requests = _record_execution_requests(widget, monkeypatch)
    table = widget.pipeline.outputs[measurements.id]
    card = widget.graph_view._cards[measurements.id]

    card.result_button.click()
    dialog = widget._result_table_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.table is table
    assert dialog.context_key == (measurements.id, 0)
    assert widget._selected_node_id == "gaussian"
    assert widget.pipeline.node_execution_states[measurements.id] == state
    if state == EXECUTION_STALE:
        assert "stale cached result" in dialog.result_status_label.text().lower()

    card.result_button.click()
    assert widget._result_table_dialog is dialog
    refreshed = TableData(("label_id", "area"), ((8, 3.5),))
    _publish_table_output(widget, measurements.id, refreshed)
    widget._refresh_node_presentation_surfaces({measurements.id})
    assert dialog.table is refreshed
    assert widget._selected_node_id == "gaussian"
    assert requests == []


@pytest.mark.parametrize(
    "operation_id",
    ("skeleton_graph_tables", "analyze_skeleton_per_label", "build_tracks"),
)
def test_measurements_card_respects_remembered_multi_table_output(
    qtbot, monkeypatch, operation_id,
):
    widget = _widget(qtbot)
    node = widget.add_node_from_palette(operation_id)
    tables = [
        TableData(("identity", "value"), ((index, index + 0.5),), name=f"Table {index}")
        for index in range(len(widget.pipeline.output_ports(node.id)))
    ]
    states = [TableState(1, 2, table.columns) for table in tables]
    _publish_table_output(widget, node.id, tables[0])
    widget.pipeline.node_outputs[node.id] = tables
    widget.pipeline.node_output_states[node.id] = states
    widget._inspector_output_port_by_node[node.id] = 1
    _select(widget, "input")
    requests = _record_execution_requests(widget, monkeypatch)

    widget.graph_view._cards[node.id].result_button.click()

    dialog = widget._result_table_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.table is tables[1]
    assert dialog.context_key == (node.id, 1)
    assert widget._inspector_output_port_by_node[node.id] == 1
    assert widget._selected_node_id == "input"
    assert requests == []


def test_measurements_card_accepts_empty_table_and_missing_results_do_not_run(
    qtbot, monkeypatch,
):
    widget = _widget(qtbot)
    node = widget.add_node_from_palette("measure_objects")
    _select(widget, "input")
    requests = _record_execution_requests(widget, monkeypatch)

    widget.graph_view._cards[node.id].result_button.click()
    assert widget._result_table_dialog is None
    assert "before opening" in widget.status_label.text()
    assert widget._selected_node_id == "input"

    empty = TableData(("label_id", "area"), ())
    _publish_table_output(widget, node.id, empty)
    widget.graph_view._cards[node.id].result_button.click()
    assert widget._result_table_dialog.table is empty
    assert widget._result_table_dialog.model.rowCount() == 0
    assert widget._selected_node_id == "input"
    widget._open_node_result_window("already-deleted-node")
    assert widget._result_table_dialog.table is empty
    assert requests == []


@pytest.mark.parametrize("state", (EXECUTION_READY, EXECUTION_STALE))
def test_histogram_card_opens_exact_owner_and_rebinds_reusable_window(
    qtbot, monkeypatch, state,
):
    widget = _widget(qtbot)
    nodes = [widget.add_node_from_palette("intensity_histogram") for _ in range(2)]
    tables = [
        intensity_histogram(
            np.arange(16, dtype=np.uint16).reshape(4, 4), bin_count=count,
        )
        for count in (4, 8)
    ]
    for node, table, count in zip(nodes, tables, (4, 8), strict=True):
        widget.pipeline.set_param(node.id, "bin_count", count)
        _publish_table_output(widget, node.id, table)
        widget.pipeline.node_execution_states[node.id] = state
    _select(widget, "gaussian")
    requests = _record_execution_requests(widget, monkeypatch)

    dialog = None
    for node, table in zip(nodes, tables, strict=True):
        widget.graph_view._cards[node.id].result_button.click()
        if dialog is None:
            dialog = widget._histogram_dialog
        assert widget._histogram_dialog is dialog
        assert dialog.isVisible()
        assert dialog.table is table
        assert widget._histogram_dialog_node_id == node.id
        assert dialog.calculation_parameters["bin_count"] == node.params["bin_count"]
        assert widget._selected_node_id == "gaussian"
        assert widget.pipeline.node_execution_states[node.id] == state
        if state == EXECUTION_STALE:
            assert "stale" in dialog.summary_label.text().lower()
    assert requests == []


@pytest.mark.parametrize("payload", ("missing", "not_histogram", "no_binned_values"))
def test_histogram_card_missing_or_unplottable_cache_never_calculates(
    qtbot, monkeypatch, payload,
):
    widget = _widget(qtbot)
    node = widget.add_node_from_palette("intensity_histogram")
    if payload != "missing":
        table = (
            intensity_histogram(np.full((4, 4), np.nan), bin_count=4)
            if payload == "no_binned_values"
            else TableData(("value",), ((1.0,),))
        )
        _publish_table_output(widget, node.id, table)
    _select(widget, "input")
    requests = _record_execution_requests(widget, monkeypatch)

    widget.graph_view._cards[node.id].result_button.click()

    assert widget._histogram_dialog is None
    assert "at least one binned value" in widget.status_label.text()
    assert widget._selected_node_id == "input"
    assert requests == []
