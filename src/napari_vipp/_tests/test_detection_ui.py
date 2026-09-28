"""Detection graph suggestions and linked read-only image/table inspection."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPalette

from napari_vipp.core.detection import find_peaks, template_match
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import EXECUTION_READY, ConnectionResult
from napari_vipp.core.tables import table_state_from_data
from napari_vipp.ui.detection_results import DetectionCanvas, DetectionResultsDialog
from napari_vipp.ui.inspector import (
    DETECTION_NEXT_STEP_SECTION,
    DETECTION_RESULTS_SECTION,
    TABLE_RESULTS_SECTION,
    inspector_profile,
)


def _state(array):
    names = ("y", "x") if array.ndim == 2 else ("z", "y", "x")
    return image_state_from_array(
        array, axes=tuple(AxisMetadata(name, "space") for name in names)
    )


def _table_and_source(shape=(12, 14)):
    source = np.zeros(shape, dtype=np.float32)
    if source.ndim == 2:
        source[3, 4], source[7, 10] = 9, 5
    else:
        source[1, 3, 4], source[3, 7, 10] = 9, 5
    source.setflags(write=False)
    state = _state(source)
    table = find_peaks(source, image_state=state, minimum_value=1)
    return table, source, state


@pytest.fixture
def detection_widget(qtbot, monkeypatch):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((12, 14))), defer_initial_run=True)
    qtbot.addWidget(widget)
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda *a, **k: calls.append(k))
    widget._debounce_timer.stop()
    widget._history.clear()
    return widget, calls


def _connect(widget, source_id, target_id, *, source_port=0, target_port=0):
    result = widget.pipeline.connect(
        source_id, target_id, source_port=source_port, target_port=target_port
    )
    assert result.success, result.message
    widget._apply_connection_result_to_graph(result)
    return result


def _matcher(widget):
    node = widget.add_node_from_palette("template_match")
    _connect(widget, "input", node.id)
    _connect(widget, "input", node.id, target_port=1)
    widget.graph_view.select_node(node.id)
    widget._debounce_timer.stop()
    widget._history.clear()
    return node


def test_detection_profiles_have_manual_review_and_standard_table_routes():
    from napari_vipp.core.pipeline import PrototypePipeline

    pipeline = PrototypePipeline()
    match_profile = inspector_profile(pipeline.operation_spec("template_match"))
    peaks_profile = inspector_profile(pipeline.operation_spec("find_peaks"))
    assert DETECTION_NEXT_STEP_SECTION in match_profile.primary_sections
    assert match_profile.execution_is_manual
    assert peaks_profile.primary_sections[-2:] == (
        DETECTION_RESULTS_SECTION,
        TABLE_RESULTS_SECTION,
    )
    assert peaks_profile.execution_is_manual
    assert peaks_profile.output_action_kind == "table"
    assert not peaks_profile.supports_pin


def test_next_step_connects_both_outputs_one_undo_without_calculating(detection_widget):
    widget, calls = detection_widget
    matcher = _matcher(widget)
    controller = widget._detection_next_step
    before = widget._current_history_snapshot()
    old_positions = widget.graph_view.node_positions()
    controller.refresh()
    assert controller.section.title() == "Next step"
    assert "warning" not in controller.section.styleSheet().lower()
    assert controller.main_button.text() == "Add Find Peaks"
    added = controller.add_find_peaks()
    assert added is not None
    assert added.params["use_mask"] is True
    assert widget._selected_node_id == added.id
    wires = [c for c in widget.pipeline.connections if c.target_id == added.id]
    assert [(c.source_id, c.source_port, c.target_port) for c in wires] == [
        (matcher.id, 0, 0),
        (matcher.id, 1, 1),
    ]
    assert len(widget._history.undo_stack) == 1
    assert widget.pipeline.outputs.get(added.id) is None
    assert not calls
    for node_id, position in old_positions.items():
        assert widget.graph_view.node_positions()[node_id] == position
        assert not widget.graph_view.node_scene_rect(added.id).intersects(
            widget.graph_view.node_scene_rect(node_id)
        )
    after = widget._current_history_snapshot()
    widget.undo()
    assert widget._current_history_snapshot().workflow == before.workflow
    widget.redo()
    assert widget._current_history_snapshot().workflow == after.workflow
    assert widget.pipeline.outputs.get(added.id) is None
    assert all(not request.get("manual_node_ids") for request in calls)


def test_next_step_finds_only_complete_matching_consumers(detection_widget):
    widget, calls = detection_widget
    matcher = _matcher(widget)
    controller = widget._detection_next_step
    first = controller.add_find_peaks()
    widget.graph_view.select_node(matcher.id)
    controller.refresh()
    assert controller.matching_peak_nodes() == (first.id,)
    history_length = len(widget._history.undo_stack)
    controller.main_button.click()
    assert widget._selected_node_id == first.id
    assert len(widget._history.undo_stack) == history_length
    widget.graph_view.select_node(matcher.id)
    second = controller.add_find_peaks()
    widget.graph_view.select_node(matcher.id)
    controller.refresh()
    assert len(controller.main_button.menu().actions()) == 2
    controller.main_button.menu().actions()[1].trigger()
    assert widget._selected_node_id == second.id
    widget.pipeline.set_param(first.id, "use_mask", False)
    widget.graph_view.select_node(matcher.id)
    assert controller.matching_peak_nodes() == (second.id,)
    assert not calls


def test_next_step_rolls_back_on_second_connection_failure(
    detection_widget, monkeypatch
):
    widget, calls = detection_widget
    _matcher(widget)
    before = widget._current_history_snapshot()
    original = widget.pipeline.connect

    def connect(*args, **kwargs):
        if kwargs.get("source_port") == 1:
            return ConnectionResult(False, "Test validity-mask connection failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(widget.pipeline, "connect", connect)
    assert widget._detection_next_step.add_find_peaks() is None
    assert widget._current_history_snapshot().workflow == before.workflow
    assert not widget._history.undo_stack
    assert "workflow was restored" in widget.status_label.text()
    assert not calls


@pytest.mark.parametrize("blocked", ["pending", "busy", "disconnected", "selection"])
def test_next_step_rechecks_edit_guards(detection_widget, monkeypatch, blocked):
    widget, calls = detection_widget
    matcher = _matcher(widget)
    if blocked == "pending":
        widget._debounce_timer.start(60_000)
    elif blocked == "busy":
        monkeypatch.setattr(widget, "_active_pipeline_run_id", 7)
    elif blocked == "disconnected":
        widget.pipeline.connections = [
            c
            for c in widget.pipeline.connections
            if not (c.target_id == matcher.id and c.target_port == 1)
        ]
    else:
        widget.graph_view.select_node("input")
    before = widget._current_history_snapshot()
    try:
        widget._detection_next_step.refresh()
        assert not widget._detection_next_step.main_button.isEnabled()
        assert widget._detection_next_step.add_find_peaks() is None
        assert widget._current_history_snapshot().workflow == before.workflow
        assert not calls
    finally:
        widget._debounce_timer.stop()


def test_detection_dialog_links_sorted_rows_and_markers_without_mutation(qtbot):
    table, source, _state_value = _table_and_source()
    before = source.copy()
    dialog = DetectionResultsDialog()
    qtbot.addWidget(dialog)
    dialog.set_result(
        table, source, node_id="peaks", token=(1,), title="Peaks", source_title="Raw"
    )
    dialog.show()
    dialog.table_panel._sort_by_header(table.columns.index("score"))
    assert dialog.table_panel.model.source_row(0) == 1
    dialog.select_source_row(0)
    assert dialog.table_panel.table_view.currentIndex().row() == 1
    assert dialog.canvas.selected_row == 0
    assert "Detection 1" in dialog.selection_label.text()
    dialog.table_panel.table_view.selectRow(0)
    assert dialog.canvas.selected_row == 1
    assert "Detection 2" in dialog.selection_label.text()
    dialog.table_panel._sort_by_header(table.columns.index("score"))
    assert (
        dialog.table_panel.model.source_row(
            dialog.table_panel.table_view.currentIndex().row()
        )
        == 1
    )
    marker = dialog.canvas.marker_position(3, 4).toPoint()
    qtbot.mouseClick(dialog.canvas, Qt.LeftButton, pos=marker)
    assert dialog.canvas.selected_row == 0
    dialog.canvas._image.fill(QColor("red"))
    np.testing.assert_array_equal(source, before)
    assert table.rows[0][0] == 1


def test_volume_inspection_has_explicit_plane_and_fractional_center_policy(qtbot):
    table, source, _state_value = _table_and_source((5, 12, 14))
    rows = list(table.rows)
    rows[1] = (rows[1][0], 2.5, *rows[1][2:])
    table = replace(table, rows=tuple(rows))
    dialog = DetectionResultsDialog()
    qtbot.addWidget(dialog)
    dialog.set_result(
        table, source, node_id="peaks", token=(1,), title="Peaks", source_title="Raw"
    )
    assert not dialog.plane_controls.isHidden()
    assert dialog.plane_spin.value() == 0
    assert not dialog.canvas._markers
    dialog.select_source_row(1)
    assert dialog.plane_spin.value() == 3
    assert [marker[1] for marker in dialog.canvas._markers] == [2]
    assert "Z = 2.5" in dialog.selection_label.text()
    assert "half-planes round up" in dialog.plane_note.text()
    assert "No projection" in dialog.plane_note.text()
    dialog.plane_spin.setValue(1)
    assert [marker[1] for marker in dialog.canvas._markers] == [1]


@pytest.mark.parametrize("dark", [False, True])
def test_overlay_keeps_accessible_palette_and_standard_exact_export(
    qtbot, tmp_path, dark
):
    table, source, _state_value = _table_and_source()
    dialog = DetectionResultsDialog()
    qtbot.addWidget(dialog)
    dialog.set_result(
        table, source, node_id="peaks", token=(1,), title="Peaks", source_title="Raw"
    )
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#202020" if dark else "#ffffff"))
    palette.setColor(QPalette.WindowText, QColor("#eeeeee" if dark else "#101010"))
    dialog.refresh_theme(palette)
    assert "Read-only" in dialog.canvas.accessibleName()
    assert "zero-based" in dialog.plane_spin.accessibleName()
    assert dialog.table_panel.model.table is table
    path = dialog.table_panel.export_table(tmp_path / "peaks.csv")
    assert "detection_id,y_index,x_index,score" in path.read_text()
    assert "1,3.0,4.0,9.0" in path.read_text()


def _seed_detection(widget):
    node = widget.add_node_from_palette("find_peaks")
    _connect(widget, "input", node.id)
    table, source, state = _table_and_source()
    widget.pipeline.outputs.update({"input": source, node.id: table})
    widget.pipeline.node_outputs[node.id] = [table]
    widget.pipeline.output_states[node.id] = table_state_from_data(table)
    widget.pipeline.node_output_states[node.id] = [
        widget.pipeline.output_states[node.id]
    ]
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_outputs["input"] = [source]
    widget.pipeline.node_output_states["input"] = [state]
    widget.pipeline.node_execution_states.update(
        {"input": EXECUTION_READY, node.id: EXECUTION_READY}
    )
    widget._debounce_timer.stop()
    return node, table, source, state


@pytest.mark.parametrize(
    "change", ["stale", "source", "table", "calibration", "pending"]
)
def test_currentness_guard_hides_old_markers_and_blocks_export(
    detection_widget,
    tmp_path,
    change,
):
    widget, calls = detection_widget
    node, table, source, state = _seed_detection(widget)
    controller = widget._detection_results
    controller.open_node(node.id)
    assert controller.dialog is not None
    assert controller._dialog_is_current()
    if change == "stale":
        widget.pipeline.node_execution_states[node.id] = "stale"
    elif change == "source":
        widget.pipeline.node_outputs["input"] = [source.copy()]
    elif change == "table":
        widget.pipeline.outputs[node.id] = replace(table, name="New result")
    elif change == "calibration":
        widget.pipeline.node_output_states["input"] = [
            replace(state, axes=(replace(state.axes[0], scale=2), state.axes[1]))
        ]
    else:
        widget._debounce_timer.start(60_000)
    try:
        controller.refresh()
        assert not controller.dialog.canvas.isEnabled()
        assert not controller.dialog.table_panel.export_button.isEnabled()
        with pytest.raises(ValueError, match="not current"):
            controller.dialog.table_panel.export_table(tmp_path / "stale.csv")
        assert not (tmp_path / "stale.csv").exists()
        assert not calls
    finally:
        widget._debounce_timer.stop()


def test_refresh_updates_already_calculated_result_without_computing(detection_widget):
    widget, calls = detection_widget
    node, table, _source, _state_value = _seed_detection(widget)
    controller = widget._detection_results
    controller.open_node(node.id)
    replacement = replace(table, name="New current detections")
    widget.pipeline.outputs[node.id] = replacement
    controller.refresh()
    assert not controller.dialog.canvas.isEnabled()
    controller.dialog.refresh_button.click()
    assert controller.dialog.canvas.isEnabled()
    assert controller.dialog.table_panel.table is replacement
    assert not calls


def test_template_overlay_uses_original_search_nonzero_tunnel_port(detection_widget):
    widget, calls = detection_widget
    source_node = widget.add_node_from_palette("split_channels")
    matcher = widget.add_node_from_palette("template_match")
    widget.pipeline.add_output_tunnel("Search", source_node.id, 1)
    connection = widget.pipeline.connect(
        source_node.id, matcher.id, source_port=1, target_port=0, tunnel_name="Search"
    )
    assert connection.success
    widget._apply_connection_result_to_graph(connection)
    _connect(widget, "input", matcher.id, target_port=1)
    widget.graph_view.select_node(matcher.id)
    widget._debounce_timer.stop()
    peaks = widget._detection_next_step.add_find_peaks()
    source = np.arange(12 * 14, dtype=float).reshape(12, 14)
    template = source[2:6, 4:8].copy()
    state = _state(source)
    scores, valid, score_state, mask_state = template_match(
        source, template, search_state=state, template_state=_state(template)
    )
    table = find_peaks(
        scores,
        valid,
        image_state=score_state,
        mask_state=mask_state,
        minimum_value=0.8,
    )
    widget.pipeline.outputs[peaks.id] = table
    widget.pipeline.node_outputs[source_node.id] = [np.zeros_like(source), source]
    widget.pipeline.node_output_states[source_node.id] = [state, state]
    widget.pipeline.node_execution_states.update(
        {
            source_node.id: EXECUTION_READY,
            matcher.id: EXECUTION_READY,
            peaks.id: EXECUTION_READY,
        }
    )
    controller = widget._detection_results
    controller.open_node(peaks.id)
    assert controller.dialog is not None
    assert controller.dialog._source is source
    assert controller.dialog.canvas._shape == source.shape
    assert np.all(controller.dialog._coordinates % 1 == 0.5)
    assert not calls


def test_switching_workflows_never_retargets_retained_overlay(detection_widget):
    from napari_vipp.core.pipeline import PrototypePipeline

    widget, calls = detection_widget
    node, _table, _source, _state_value = _seed_detection(widget)
    controller = widget._detection_results
    controller.open_node(node.id)
    token = controller.dialog.token
    original = widget.pipeline
    widget.pipeline = PrototypePipeline()
    try:
        controller.refresh()
        controller.dialog.refresh_button.click()
        assert controller.dialog.token == token
        assert not controller.dialog.canvas.isEnabled()
        assert "another workflow" in controller.dialog.status_label.text()
        assert not calls
    finally:
        widget.pipeline = original


def test_valid_mask_control_updates_actual_graph_ports(detection_widget):
    widget, _calls = detection_widget
    node = widget.add_node_from_palette("find_peaks")
    widget.graph_view.select_node(node.id)
    assert len(widget.pipeline.input_ports(node.id)) == 1
    widget._on_param_changed("use_mask", True)
    widget._debounce_timer.stop()
    assert len(widget.pipeline.input_ports(node.id)) == 2
    widget._on_param_changed("use_mask", False)
    widget._debounce_timer.stop()
    assert len(widget.pipeline.input_ports(node.id)) == 1


@pytest.mark.parametrize(
    "values",
    [
        np.array([[2**52, 2**52 + 4]], dtype=np.uint64),
        np.array([[-1e308, 1e308]], dtype=float),
    ],
)
def test_preview_preserves_wide_value_contrast_without_changing_source(qtbot, values):
    canvas = DetectionCanvas()
    qtbot.addWidget(canvas)
    before = values.copy()
    canvas.set_plane(values, [])
    assert canvas._image.pixelColor(0, 0).red() == 0
    assert canvas._image.pixelColor(canvas._image.width() - 1, 0).red() == 255
    np.testing.assert_array_equal(values, before)


def test_preview_memory_refusal_preserves_table_and_scientific_source(
    qtbot,
    monkeypatch,
):
    from napari_vipp.ui import detection_results

    table, source, _state_value = _table_and_source()
    before = source.copy()
    requests = []

    def refuse(_snapshot, **kwargs):
        requests.append(kwargs)
        return SimpleNamespace(allowed=False, reason="Test available-memory limit")

    monkeypatch.setattr(detection_results, "preflight_host_allocation", refuse)
    dialog = DetectionResultsDialog()
    qtbot.addWidget(dialog)
    dialog.set_result(
        table, source, node_id="peaks", token=(1,), title="Peaks", source_title="Raw"
    )
    assert requests[0]["required_bytes"] >= source.size * 64
    assert not dialog.canvas.isEnabled()
    assert "more free memory" in dialog.status_label.text()
    assert dialog.table_panel.table is table
    assert dialog.table_panel.export_button.isEnabled()
    dialog.select_source_row(0)
    assert "Detection 1" in dialog.selection_label.text()
    np.testing.assert_array_equal(source, before)


def test_detection_table_uses_standard_popout_and_results_workspace(detection_widget):
    widget, calls = detection_widget
    node, table, _source, _state_value = _seed_detection(widget)
    widget.graph_view.select_node(node.id)
    widget._pending_dirty_node_ids.clear()
    widget._debounce_timer.stop()
    widget._update_table_preview()
    assert not widget.results_workspace_button.isHidden()
    assert widget.table_popout_button.isEnabled()
    widget.table_popout_button.click()
    assert widget._result_table_dialog.table is table
    dialog = widget._results_workspace.open_node(node.id)
    assert dialog is not None
    assert dialog.data_panel.table is table
    assert not calls


@pytest.mark.parametrize("shape", [(12, 14), (5, 12, 14)])
def test_minimum_dialog_layout_does_not_overlap_image_and_explanations(qtbot, shape):
    table, source, _state_value = _table_and_source(shape)
    dialog = DetectionResultsDialog()
    qtbot.addWidget(dialog)
    dialog.set_result(
        table, source, node_id="peaks", token=(1,), title="Peaks", source_title="Raw"
    )
    dialog.resize(600, 400)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.canvas.isVisible())
    assert dialog.size().width() == 600
    assert not dialog.canvas.geometry().intersects(dialog.plane_note.geometry())
    assert not dialog.canvas.geometry().intersects(dialog.selection_label.geometry())
    assert dialog.refresh_button.isVisible()
    assert dialog.close_button.isVisible()
