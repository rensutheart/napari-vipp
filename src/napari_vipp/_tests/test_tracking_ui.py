"""Read-only trajectory review: time/plane selection, trails and stale guards."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from qtpy.QtCore import Qt, QTimer
from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QWidget

from napari_vipp._tests.test_tracking import series
from napari_vipp.core.detection import _frame, _revision
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.tables import table_state_from_data
from napari_vipp.core.tracking import build_tracks
from napari_vipp.core.tracking_metadata import FramePopulation
from napari_vipp.ui.tracking_results import (
    TrackingResultsController,
    TrackingResultsDialog,
)


def _result(*, volume=False, gap=False, ambiguous=False, linked=True):
    shape = (4, 5, 12, 14) if volume else (4, 12, 14)
    source = np.arange(np.prod(shape), dtype=float).reshape(shape)
    source.setflags(write=False)
    axes = ("z", "y", "x") if volume else ("y", "x")
    state = image_state_from_array(
        source,
        axes=(
            AxisMetadata("t", "time", scale=0.5, unit="s"),
            *(AxisMetadata(axis, "space") for axis in axes),
        ),
    )
    end_frame = 2 if gap else 1
    rows = [(0, 9, 3, 4), (end_frame, 7, 4, 5), (3, 2, 5, 6)]
    if ambiguous:
        rows = [(0, 9, 3, 3), (0, 2, 3, 5), (1, 7, 3, 4), (1, 3, 3, 4)]
    if volume:
        rows = [(t, identity, 2.5 if t == 3 else 1, y, x) for t, identity, y, x in rows]
    table = series(
        rows,
        rank=len(axes),
        frame_count=4,
        source_shape=shape[1:],
        time_scale=0.5,
        time_unit="s",
        source_revision=_revision(source),
        source_frame=_frame(state, _revision(source)),
    )
    if linked:
        table = build_tracks(table, maximum_displacement=4, maximum_gap=1)[0]
    return table, source, state


def _dialog(qtbot, **kwargs):
    table, source, state = _result(**kwargs)
    dialog = TrackingResultsDialog()
    qtbot.addWidget(dialog)
    dialog.set_result(
        table,
        source,
        node_id="tracks",
        token=(1,),
        title="Tracks",
        source_title="Raw series",
    )
    return dialog, table, source, state


def test_time_selection_and_sorted_table_preserve_observation_identity(qtbot):
    dialog, table, source, _state = _dialog(qtbot)
    before = source.copy()
    dialog.show()
    assert dialog.time_spin.maximum() == 3
    assert not dialog.plane_spin.isVisible()
    assert len(dialog.canvas._markers) == 1
    dialog.table_panel._sort_by_header(table.columns.index("detection_id"))
    assert dialog.table_panel.model.source_row(0) == 2
    dialog.table_panel.table_view.selectRow(0)
    assert dialog.canvas.selected_row == 2
    assert dialog.time_spin.value() == 3
    assert "detection_id = 2" in dialog.selection_label.text()
    assert "Track 1" in dialog.selection_label.text()
    assert len(dialog.canvas.segments) == 2
    assert all(segment[-1] for segment in dialog.canvas.segments)
    dialog.table_panel._sort_by_header(table.columns.index("detection_id"))
    assert (
        dialog.table_panel.model.source_row(
            dialog.table_panel.table_view.currentIndex().row()
        )
        == 2
    )
    dialog.time_spin.setValue(0)
    qtbot.mouseClick(
        dialog.canvas, Qt.LeftButton, pos=dialog.canvas.marker_position(3, 4).toPoint()
    )
    assert dialog.canvas.selected_row == 0
    assert "detection_id = 9" in dialog.selection_label.text()
    dialog.canvas._image.fill(QColor("red"))
    np.testing.assert_array_equal(source, before)
    assert not source.flags.writeable


def test_volume_selection_changes_t_and_z_without_projected_links(qtbot):
    dialog, _table, _source, _state = _dialog(qtbot, volume=True)
    assert not dialog.canvas._markers
    dialog.select_source_row(2)
    assert dialog.time_spin.value() == 3
    assert dialog.plane_spin.value() == 3  # half-plane rounds up
    assert len(dialog.canvas._markers) == 1
    assert "Z = 2.5" in dialog.selection_label.text()
    assert not dialog.canvas.segments  # previous endpoint lies on Z=1
    assert "both endpoints on this plane" in dialog.plane_note.text()
    assert "no projection" in dialog.plane_note.text()
    dialog.time_spin.setValue(1)
    dialog.plane_spin.setValue(1)
    assert len(dialog.canvas._markers) == 1
    assert len(dialog.canvas.segments) == 1


def test_gap_trails_are_explicit_and_trail_window_is_presentation_only(qtbot):
    dialog, table, _source, _state = _dialog(qtbot, gap=True)
    before = table.rows
    dialog.time_spin.setValue(2)
    assert len(dialog.canvas.segments) == 1
    assert dialog.canvas.segments[0][2] is True
    assert "missing frames" in dialog.review_note.text()
    dialog.trail_spin.setValue(1)
    assert not dialog.canvas.segments
    dialog.trail_spin.setValue(0)
    assert not dialog.canvas.segments
    assert len(dialog.canvas._markers) == 1
    assert table.rows == before


def test_competing_candidates_have_review_explanations(qtbot):
    dialog, _table, _source, _state = _dialog(qtbot, ambiguous=True)
    dialog.time_spin.setValue(1)
    assert dialog.canvas.segments
    assert all(segment[3] for segment in dialog.canvas.segments)
    assert "flagged" in dialog.status_label.text()
    dialog.select_source_row(2)
    assert "competing feasible links" in dialog.selection_label.text()
    assert "not proof of identity" in dialog.review_note.text()


def test_capped_unlinked_frames_are_reviewable_but_have_no_trails(qtbot):
    table, source, _state = _result(linked=False)
    metadata = table.observation_metadata
    populations = list(metadata.frame_populations)
    populations[0] = FramePopulation(0, 2, 1, True)
    table = replace(
        table,
        observation_metadata=replace(metadata, frame_populations=tuple(populations)),
    )
    dialog = TrackingResultsDialog()
    qtbot.addWidget(dialog)
    dialog.set_result(
        table,
        source,
        node_id="detect",
        token=(1,),
        title="Detections",
        source_title="Raw",
    )
    assert not dialog.trail_spin.isEnabled()
    assert not dialog.canvas.segments
    assert "cap" in dialog.status_label.text()
    assert "Unlinked observations" in dialog.context_label.text()
    dialog.time_spin.setValue(2)
    assert not dialog.canvas._markers


@pytest.mark.parametrize("dark", [False, True])
@pytest.mark.parametrize("volume", [False, True])
def test_theme_minimum_layout_and_exact_export(qtbot, tmp_path, dark, volume):
    dialog, table, _source, _state = _dialog(qtbot, volume=volume)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#202020" if dark else "#ffffff"))
    palette.setColor(QPalette.WindowText, QColor("#eeeeee" if dark else "#101010"))
    dialog.refresh_theme(palette)
    dialog.resize(600, 400)
    dialog.show()
    qtbot.waitUntil(lambda: dialog.canvas.isVisible())
    assert dialog.width() == 600
    assert not dialog.canvas.geometry().intersects(dialog.plane_note.geometry())
    assert not dialog.canvas.geometry().intersects(dialog.selection_label.geometry())
    assert dialog.refresh_button.isVisible() and dialog.close_button.isVisible()
    assert "Read-only" in dialog.canvas.accessibleName()
    assert "zero-based" in dialog.time_spin.accessibleName()
    path = dialog.table_panel.export_table(tmp_path / "tracks.csv")
    lines = path.read_text().splitlines()
    assert lines[0].split(",") == list(table.columns)
    assert len(lines) == table.row_count + 1


class _Host(QWidget):
    def __init__(self, table, source, state):
        super().__init__()
        self.pipeline = SimpleNamespace(
            nodes={
                name: SimpleNamespace(id=name, operation_id=operation)
                for name, operation in (
                    ("raw", "input"),
                    ("detect", "detect_spots_per_frame"),
                    ("tracks", "build_tracks"),
                )
            },
            connections=[
                SimpleNamespace(
                    source_id="raw", source_port=0, target_id="detect", target_port=0
                ),
                SimpleNamespace(
                    source_id="detect", source_port=0, target_id="tracks", target_port=0
                ),
            ],
        )
        self.payloads = {
            "raw": (source, state),
            "detect": (table, table_state_from_data(table)),
            "tracks": (table, table_state_from_data(table)),
        }
        self.states = {name: EXECUTION_READY for name in self.pipeline.nodes}
        self._selected_node_id = "tracks"
        self._debounce_timer = QTimer(self)
        self._active_pipeline_run_id = self._active_source_load_id = None
        self._pipeline_run_pending = self._source_load_pending = False
        self.statuses = []

    def _node_execution_ui_state(self, node_id):
        return self.states.get(node_id), ""

    def _node_output_payload_for_port(self, node_id, port):
        return self.payloads.get(node_id, (None, None)) if port == 0 else (None, None)

    def _node_title(self, node_id):
        return node_id

    def _set_status(self, message, **kwargs):
        self.statuses.append(message)


@pytest.fixture
def controller(qtbot):
    table, source, state = _result()
    host = _Host(table, source, state)
    qtbot.addWidget(host)
    result = TrackingResultsController(host)
    yield result, host, table, source, state
    result._timer.stop()
    if result.dialog is not None:
        result.dialog.close()


@pytest.mark.parametrize(
    "change",
    ["disconnected", "source", "table", "calibration", "time", "stale", "pending"],
)
def test_controller_stale_route_disables_overlay_navigation_and_export(
    controller, tmp_path, change
):
    control, host, table, source, state = controller
    control.open_node("tracks")
    assert control._dialog_is_current()
    before = source.copy()
    if change == "disconnected":
        host.pipeline.connections.clear()
    elif change == "source":
        host.payloads["raw"] = (source.copy(), state)
    elif change == "table":
        replacement = replace(table, name="Replacement")
        host.payloads["tracks"] = (replacement, table_state_from_data(replacement))
    elif change in ("calibration", "time"):
        axes = list(state.axes)
        index = 1 if change == "calibration" else 0
        axes[index] = replace(axes[index], scale=2)
        host.payloads["raw"] = (source, replace(state, axes=tuple(axes)))
    elif change == "stale":
        host.states["detect"] = "stale"
    else:
        host._pipeline_run_pending = True
    control.refresh()
    assert not control.dialog.canvas.isEnabled()
    assert not control.dialog.time_spin.isEnabled()
    assert not control.dialog.trail_spin.isEnabled()
    assert not control.dialog.table_panel.export_button.isEnabled()
    with pytest.raises(ValueError, match="not current"):
        control.dialog.table_panel.export_table(tmp_path / "stale.csv")
    assert not (tmp_path / "stale.csv").exists()
    np.testing.assert_array_equal(source, before)


def test_controller_refresh_only_binds_already_calculated_current_result(controller):
    control, host, table, _source, _state = controller
    control.open_node("tracks")
    token = control.dialog.token
    replacement = replace(table, name="Replacement")
    host.payloads["tracks"] = (replacement, table_state_from_data(replacement))
    control.refresh()
    assert not control.dialog.canvas.isEnabled()
    control.dialog.refresh_button.click()
    assert control.dialog.canvas.isEnabled()
    assert control.dialog.token != token
    assert control.dialog.table_panel.table is replacement


def test_controller_rejects_unavailable_original_series(controller):
    control, host, _table, _source, _state = controller
    host.payloads["raw"] = (None, None)
    control.open_node("tracks")
    assert control.dialog is None
    assert "resident" in host.statuses[-1]


@pytest.mark.parametrize(
    "invalid",
    [
        "missing_node",
        "inferred_axes",
        "wrong_axis_type",
        "state_shape",
        "state_dtype",
        "source_identity",
        "missing_raw_node",
    ],
)
def test_controller_rejects_broken_route_or_source_evidence(controller, invalid):
    control, host, _table, source, state = controller
    if invalid == "missing_node":
        del host.pipeline.nodes["detect"]
    elif invalid == "missing_raw_node":
        del host.pipeline.nodes["raw"]
    elif invalid in ("inferred_axes", "wrong_axis_type"):
        axes = list(state.axes)
        axes[0] = replace(
            axes[0],
            **(
                {"confidence": "inferred"}
                if invalid == "inferred_axes"
                else {"type": "space"}
            ),
        )
        host.payloads["raw"] = (source, replace(state, axes=tuple(axes)))
    elif invalid == "state_shape":
        host.payloads["raw"] = (source, replace(state, shape=(9, *state.shape[1:])))
    elif invalid == "state_dtype":
        host.payloads["raw"] = (source, replace(state, dtype="uint8"))
    else:
        host.payloads["raw"] = (
            source,
            replace(
                state, source=replace(state.source, source_uuid="different-acquisition")
            ),
        )
    payload, reason = control._payload("tracks")
    assert payload is None
    assert reason


def test_preview_memory_refusal_keeps_exact_table_without_mutating_source(
    qtbot, monkeypatch
):
    from napari_vipp.ui import detection_results

    monkeypatch.setattr(
        detection_results,
        "preflight_host_allocation",
        lambda *a, **k: SimpleNamespace(allowed=False, reason="test resource limit"),
    )
    dialog, table, source, _state = _dialog(qtbot)
    before = source.copy()
    assert not dialog.canvas.isEnabled()
    assert "memory" in dialog.status_label.text()
    assert dialog.table_panel.table is table
    assert dialog.table_panel.export_button.isEnabled()
    dialog.select_source_row(1)
    assert "detection_id = 7" in dialog.selection_label.text()
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize("failure", ["admission", "coordinate_allocation"])
def test_full_overlay_allocation_failure_preserves_table_sort_and_export(
    qtbot, monkeypatch, tmp_path, failure
):
    from napari_vipp.ui import tracking_results

    dialog, table, source, _state = _dialog(qtbot, volume=True)
    before = source.copy()
    baseline = dialog.table_panel.export_table(tmp_path / "before.csv").read_bytes()
    dialog.table_panel._sort_by_header(table.columns.index("detection_id"))
    order = tuple(
        dialog.table_panel.model.source_row(i) for i in range(table.row_count)
    )
    allocations = []

    def deny(*_args, **kwargs):
        allocations.append(kwargs)
        return SimpleNamespace(allowed=False, reason="test resource limit")

    def fail_coordinates(*_args, **_kwargs):
        if failure == "admission":
            pytest.fail("Coordinate conversion ran before admission")
        raise MemoryError("test coordinate allocation failure")

    with monkeypatch.context() as limited:
        if failure == "admission":
            limited.setattr(tracking_results, "preflight_host_allocation", deny)
        limited.setattr(tracking_results.np, "asarray", fail_coordinates)
        dialog.set_result(
            table,
            source,
            node_id="tracks",
            token=(2,),
            title="Tracks",
            source_title="Raw series",
        )
    if failure == "admission":
        assert len(allocations) == 1
        assert allocations[0]["required_bytes"] > table.row_count * 3 * 8
        assert "coordinates" in allocations[0]["purpose"]
    assert dialog._current
    assert not dialog._overlay_available
    assert not dialog.canvas.isEnabled()
    assert not dialog.canvas._markers and not dialog.canvas.segments
    assert not dialog._coordinates and not dialog._frame_rows
    assert not dialog.time_spin.isEnabled()
    assert not dialog.plane_spin.isEnabled()
    assert not dialog.trail_spin.isEnabled()
    assert "full table and export remain available" in dialog.status_label.text()
    assert "Refresh view" in dialog.status_label.text()
    assert dialog.table_panel.table is table
    assert dialog.table_panel.table_view.isEnabled()
    assert dialog.table_panel.export_button.isEnabled()
    assert (
        tuple(dialog.table_panel.model.source_row(i) for i in range(table.row_count))
        == order
    )
    dialog.table_panel.table_view.selectRow(0)
    assert "detection_id = 2" in dialog.selection_label.text()
    dialog.table_panel._sort_by_header(table.columns.index("detection_id"))
    assert dialog.canvas.selected_row == 2
    assert (
        dialog.table_panel.model.source_row(
            dialog.table_panel.table_view.currentIndex().row()
        )
        == 2
    )
    assert (
        dialog.table_panel.export_table(tmp_path / "after.csv").read_bytes() == baseline
    )
    np.testing.assert_array_equal(source, before)
    assert not source.flags.writeable
    dialog.set_result(
        table,
        source,
        node_id="tracks",
        token=(3,),
        title="Tracks",
        source_title="Raw series",
    )
    assert dialog._overlay_available and dialog.canvas.isEnabled()
    assert dialog.time_spin.isEnabled() and dialog.plane_spin.isEnabled()
    assert dialog.trail_spin.isEnabled()


@pytest.mark.parametrize("failure", ["admission", "plane_allocation"])
def test_navigation_memory_failure_keeps_table_selection_safe(
    qtbot, monkeypatch, failure
):
    from napari_vipp.ui import tracking_results

    dialog, table, _source, _state = _dialog(qtbot, volume=True)
    if failure == "admission":
        monkeypatch.setattr(
            tracking_results,
            "preflight_host_allocation",
            lambda *a, **k: SimpleNamespace(
                allowed=False, reason="test resource limit"
            ),
        )
    else:

        def fail_plane(*_args):
            raise MemoryError("test plane allocation failure")

        monkeypatch.setattr(dialog.canvas, "set_plane", fail_plane)
    # Identification itself changes T and Z, which can fail in a nested signal.
    dialog.select_source_row(2)
    assert "detection_id = 2" in dialog.selection_label.text()
    assert "Overlay unavailable" in dialog.selection_label.text()
    assert not dialog._overlay_available
    assert not dialog.canvas._markers and not dialog.canvas.segments
    assert dialog.table_panel.table is table
    assert dialog.table_panel.export_button.isEnabled()
    dialog.select_source_row(1)
    assert "detection_id = 7" in dialog.selection_label.text()


def test_time_navigation_reuses_coordinate_and_frame_index_buffers(qtbot):
    dialog, _table, _source, _state = _dialog(qtbot)
    coordinates, frame_rows = dialog._coordinates, dialog._frame_rows
    for frame in (1, 3, 0, 2):
        dialog.time_spin.setValue(frame)
        assert dialog._coordinates is coordinates
        assert dialog._frame_rows is frame_rows


def test_overlay_denial_does_not_bypass_controller_stale_export_guard(
    controller, monkeypatch, tmp_path
):
    from napari_vipp.ui import tracking_results

    control, host, table, source, state = controller
    monkeypatch.setattr(
        tracking_results,
        "preflight_host_allocation",
        lambda *a, **k: SimpleNamespace(allowed=False, reason="test resource limit"),
    )
    control.open_node("tracks")
    assert control._dialog_is_current()
    assert control.dialog.table_panel.table is table
    assert control.dialog.table_panel.export_button.isEnabled()
    host.payloads["raw"] = (source.copy(), state)
    control.refresh()
    assert not control.dialog._current
    assert not control.dialog.table_panel.table_view.isEnabled()
    assert not control.dialog.table_panel.export_button.isEnabled()
    with pytest.raises(ValueError, match="not current"):
        control.dialog.table_panel.export_table(tmp_path / "stale.csv")
    assert not (tmp_path / "stale.csv").exists()


def test_real_widget_composes_trajectory_controller_without_recalculation(
    qtbot, monkeypatch
):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget
    from napari_vipp.ui.inspector import TRACKING_RESULTS_SECTION

    widget = VippWidget(_Viewer(np.zeros((12, 14))), defer_initial_run=True)
    qtbot.addWidget(widget)
    calls = []
    monkeypatch.setattr(widget, "run_pipeline", lambda *a, **k: calls.append(k))
    detector = widget.add_node_from_palette("detect_spots_per_frame")
    linker = widget.add_node_from_palette("build_tracks")
    for start, end in (("input", detector.id), (detector.id, linker.id)):
        connected = widget.pipeline.connect(start, end)
        assert connected.success
        widget._apply_connection_result_to_graph(connected)
    observations, source, state = _result(linked=False)
    tracked, summary = build_tracks(observations, maximum_gap=1, maximum_displacement=4)
    for node_id, values, states in (
        ("input", [source], [state]),
        (detector.id, [observations], [table_state_from_data(observations)]),
        (
            linker.id,
            [tracked, summary],
            [table_state_from_data(tracked), table_state_from_data(summary)],
        ),
    ):
        widget.pipeline.outputs[node_id] = values[0]
        widget.pipeline.node_outputs[node_id] = values
        widget.pipeline.output_states[node_id] = states[0]
        widget.pipeline.node_output_states[node_id] = states
        widget.pipeline.node_execution_states[node_id] = EXECUTION_READY
    widget._pending_dirty_node_ids.clear()
    widget._debounce_timer.stop()
    widget.graph_view.select_node(linker.id)
    widget._debounce_timer.stop()
    controller = widget._tracking_results
    controller.refresh()
    widget.graph_view.select_node("input")
    widget._debounce_timer.stop()
    widget.graph_view._cards[linker.id].trajectory_review_button.click()
    assert controller.dialog is not None
    assert controller.dialog.table_panel.table is tracked
    assert widget._selected_node_id == "input"
    assert controller._dialog_is_current()
    controller.dialog.select_source_row(2)
    assert controller.dialog.time_spin.value() == 3
    assert not calls
    dialog = controller.dialog
    widget.graph_view._cards[detector.id].trajectory_review_button.click()
    assert controller.dialog is dialog
    assert dialog.table_panel.table is observations
    assert widget._selected_node_id == "input"
    assert controller._dialog_is_current()
    controller.close()

    measurement = widget.add_node_from_palette("measure_objects")
    static = replace(observations, observation_metadata=None)
    for table in (static, observations, static):
        table_state = table_state_from_data(table)
        widget.pipeline.outputs[measurement.id] = table
        widget.pipeline.node_outputs[measurement.id] = [table]
        widget.pipeline.output_states[measurement.id] = table_state
        widget.pipeline.node_output_states[measurement.id] = [table_state]
        widget.pipeline.node_execution_states[measurement.id] = EXECUTION_READY
        widget.graph_view.select_node(measurement.id)
        widget._debounce_timer.stop()
        widget._sync_inspector_presentation()
        assert widget._inspector_sections[TRACKING_RESULTS_SECTION].isHidden() == (
            table.observation_metadata is None
        )
        assert widget.graph_view._cards[
            measurement.id
        ].trajectory_review_button.isHidden() == (table.observation_metadata is None)
    assert not calls
