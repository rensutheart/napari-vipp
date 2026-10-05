"""Read-only time/plane navigation and trajectory review, detached from analysis."""

from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPainter, QPen
from qtpy.QtWidgets import QLabel, QSpinBox, QWidget

from napari_vipp.core.detection import _frame
from napari_vipp.core.host_memory import capture_host_memory, preflight_host_allocation
from napari_vipp.core.metadata import ImageState
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.tables import is_table_data
from napari_vipp.ui.detection_results import (
    DetectionCanvas,
    DetectionResultsController,
    DetectionResultsDialog,
)
from napari_vipp.ui.status import MessageSeverity

SERIES_REVIEW_OPERATIONS = frozenset(
    {
        "detect_spots_per_frame",
        "build_tracks",
        "measure_objects",
        "measure_objects_intensity",
    }
)


class TrackingCanvas(DetectionCanvas):
    """Draw observed links only; dashed segments explicitly span missing frames."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.segments = ()
        self.setAccessibleName("Read-only time-series trajectory overlay")

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if not self._current or self._image.isNull():
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        for start, end, gap, review, selected in self.segments:
            color = QColor(
                "#ffdc48" if selected else "#ff9e5b" if review else "#52e3ff"
            )
            pen = QPen(color, 3 if selected else 1.7)
            if gap:
                pen.setStyle(Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(self.marker_position(*start), self.marker_position(*end))


class TrackingResultsDialog(DetectionResultsDialog):
    """Current-frame markers with observed trails and a linked sortable table."""

    canvas_class = TrackingCanvas

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames = ()
        self._track_ids = ()
        self._reviews = ()
        self._frame_rows = ()
        self._review_count = 0
        self._truncated = False
        self._overlay_available = False
        self._metadata = None
        controls = self.plane_controls.layout()
        self.plane_label.setText("Z plane")
        self.plane_label.setToolTip("Source Z plane, zero-based; no projection.")
        self.time_spin = QSpinBox()
        self.time_spin.setAccessibleName("Time frame, zero-based")
        self.time_spin.setToolTip(
            "Display one existing time point; never changes linking."
        )
        controls.insertWidget(0, QLabel("T frame"))
        controls.insertWidget(1, self.time_spin)
        self.trail_spin = QSpinBox()
        self.trail_spin.setRange(0, 1000000)
        self.trail_spin.setValue(10)
        self.trail_spin.setAccessibleName("Trail length in frames")
        self.trail_spin.setToolTip(
            "Show this many preceding frames; zero hides trails."
        )
        controls.addWidget(QLabel("Trail frames"))
        controls.addWidget(self.trail_spin)
        self.time_spin.valueChanged.connect(self._update_plane)
        self.trail_spin.valueChanged.connect(self._update_plane)
        self.review_note.setText(
            "Read-only review. Cyan links join observed positions; dashed links span "
            "missing frames, not measured intermediate positions. Orange marks links "
            "requiring review; yellow marks the selected track. A plausible path is "
            "not proof of identity. No division/fusion inference or drift correction."
        )

    def set_result(self, table, source, *, node_id, token, title, source_title):
        metadata = table.observation_metadata
        self.node_id, self.token = node_id, token
        self._table, self._source, self._metadata = table, source, metadata
        self._current = False
        self._overlay_available = False
        self._coordinates = ()
        self._frames = self._ids = self._track_ids = self._reviews = ()
        self._frame_rows = ()
        self.canvas.segments = ()
        self.canvas._markers = ()
        self.canvas.selected_row = None
        self.canvas.set_current(False)
        # Install the ordinary table first: an optional overlay allocation must
        # never make an otherwise current scientific result or export unusable.
        self.table_panel.set_table(
            table,
            title=title,
            default_export_name="tracked-observations.csv",
            context_key=(node_id, 0),
        )
        self.table_panel.export_button.setEnabled(True)
        self.table_panel.table_view.setEnabled(True)
        self.table_panel.set_result_status()
        self._current = True
        self.plane_controls.show()
        is_volume = len(metadata.spatial_axes) == 3
        self.plane_label.setVisible(is_volume)
        self.plane_spin.setVisible(is_volume)
        self.plane_spin.setEnabled(True)
        self.time_spin.setEnabled(True)
        self.trail_spin.setEnabled(table.tracking_metadata is not None)
        self.time_spin.blockSignals(True)
        self.time_spin.setRange(0, metadata.frame_count - 1)
        self.time_spin.blockSignals(False)
        self.plane_spin.blockSignals(True)
        self.plane_spin.setRange(0, metadata.source_shape[0] - 1 if is_volume else 0)
        self.plane_spin.blockSignals(False)
        self.setWindowTitle(f"{title} — Review trajectories")
        context = (
            f"{table.row_count:,} observations · {metadata.frame_count} frames · "
            f"{source_title}"
        )
        self.context_label.setText(context)
        self.selection_label.setText(
            "Select a row or current-frame marker to identify it."
        )
        try:
            # Include temporary nested coordinate lists, NumPy storage, IDs,
            # frame-index lists/tuples, track-count sets, and trail structures.
            self._admit_overlay(
                table.row_count * 1024 + metadata.frame_count * 128 + 65536,
                "Trajectory overlay coordinates and frame indices",
            )
            columns = [
                table.columns.index(name) for name in metadata.coordinate_columns
            ]
            self._coordinates = np.asarray(
                [[row[column] for column in columns] for row in table.rows], dtype=float
            ).reshape((-1, len(columns)))
            frame_column = table.columns.index("t_index")
            id_column = table.columns.index(metadata.id_column)
            self._frames = tuple(row[frame_column] for row in table.rows)
            self._ids = tuple(row[id_column] for row in table.rows)
            self._track_ids = (
                tuple(row[table.columns.index("track_id")] for row in table.rows)
                if table.tracking_metadata is not None
                else ()
            )
            self._reviews = (
                tuple(row[table.columns.index("review_flag")] for row in table.rows)
                if self._track_ids
                else ()
            )
            frame_rows = [[] for _ in range(metadata.frame_count)]
            for row, frame in enumerate(self._frames):
                frame_rows[frame].append(row)
            self._frame_rows = tuple(tuple(rows) for rows in frame_rows)
            self._review_count = sum(self._reviews)
            self._truncated = metadata.truncated
            track_count = len(set(self._track_ids))
            prefix = (
                f"{track_count:,} {'track' if track_count == 1 else 'tracks'} · "
                if table.tracking_metadata is not None
                else "Unlinked observations · "
            )
            self.context_label.setText(prefix + context)
        except MemoryError as exc:
            self._disable_overlay(exc)
            return
        self._overlay_available = True
        self._update_plane()

    @staticmethod
    def _admit_overlay(required_bytes, purpose):
        decision = preflight_host_allocation(
            capture_host_memory(), required_bytes=int(required_bytes), purpose=purpose
        )
        if not decision.allowed:
            raise MemoryError(decision.reason)

    def _disable_overlay(self, reason):
        self._overlay_available = False
        self._coordinates = ()
        self._frames = self._ids = self._track_ids = self._reviews = ()
        self._frame_rows = ()
        self.canvas.segments = ()
        self.canvas._markers = ()
        self.canvas.set_current(False)
        self.time_spin.setEnabled(False)
        self.plane_spin.setEnabled(False)
        self.trail_spin.setEnabled(False)
        self.plane_note.setText(
            "Image/trajectory overlay unavailable; no observations were removed."
        )
        self.status_label.setText(
            "The trajectory overlay needs more free memory. The full table and "
            "export remain available. Close other images, then Refresh view. "
            f"{reason}"
        )
        self.status_label.show()

    def set_stale(self, message):
        super().set_stale(message)
        self.time_spin.setEnabled(False)
        self.trail_spin.setEnabled(False)

    def _update_plane(self, *_args):
        if not self._current or self._source is None or not self._overlay_available:
            return
        frame, plane = self.time_spin.value(), self.plane_spin.value()
        is_volume = len(self._metadata.spatial_axes) == 3

        def on_plane(row):
            return not is_volume or math.floor(self._coordinates[row][0] + 0.5) == plane

        try:
            first = max(0, frame - self.trail_spin.value())
            trail_rows = (
                sum(len(self._frame_rows[t]) for t in range(first, frame + 1))
                if self._track_ids and self.trail_spin.value()
                else 0
            )
            self._admit_overlay(
                len(self._frame_rows[frame]) * 160 + trail_rows * 512 + 65536,
                "Trajectory current-frame markers and observed trails",
            )
            markers = [
                (
                    row,
                    self._track_ids[row] if self._track_ids else self._ids[row],
                    *self._coordinates[row][-2:],
                )
                for row in self._frame_rows[frame]
                if on_plane(row)
            ]
            selected = self.canvas.selected_row
            selected_track = (
                self._track_ids[selected]
                if self._track_ids and selected is not None
                else None
            )
            segments, previous = [], {}
            if trail_rows:
                for t in range(first, frame + 1):
                    for row in self._frame_rows[t]:
                        track = self._track_ids[row]
                        old = previous.get(track)
                        if old is not None and on_plane(old) and on_plane(row):
                            segments.append(
                                (
                                    self._coordinates[old][-2:],
                                    self._coordinates[row][-2:],
                                    t - self._frames[old] > 1,
                                    self._reviews[row] or self._reviews[old],
                                    track == selected_track,
                                )
                            )
                        previous[track] = row
            self.canvas.segments = tuple(segments)
            self.canvas.set_plane(
                self._source[frame, plane] if is_volume else self._source[frame],
                markers,
            )
        except MemoryError as exc:
            self._disable_overlay(exc)
            return
        self.canvas.set_current(True)
        population = self._metadata.frame_populations[frame]
        self.status_label.setText(
            f"{self._review_count} observations flagged for competing feasible links."
            if self._review_count
            else ""
        )
        if self._truncated:
            self.status_label.setText(
                "Some frames reached the detection cap. "
                "Increase it before linking tracks."
            )
        self.status_label.setVisible(bool(self.status_label.text()))
        time = self._metadata.time_origin + frame * self._metadata.time_scale
        self.plane_note.setText(
            f"T = {frame} ({time:g} {self._metadata.time_unit or 'frame units'}) · "
            f"{population.retained_count} observations in frame · "
            f"{len(markers)} on this plane. "
            + (
                f"Z = {plane}. Only links with both endpoints on this plane "
                "are shown; no projection."
                if is_volume
                else "YX view; no projection."
            )
        )

    def _identify(self, source_row):
        if not self._current or not 0 <= source_row < self._table.row_count:
            return
        if not self._overlay_available:
            # Ordinary sorted-table selection stays useful after preview denial;
            # inspect only the selected row, never rebuild full overlay buffers.
            row = self._table.rows[source_row]
            self.canvas.selected_row = source_row
            self.selection_label.setText(
                f"T = {row[self._table.columns.index('t_index')]} · "
                f"{self._metadata.id_column} = "
                f"{row[self._table.columns.index(self._metadata.id_column)]} · "
                "Overlay unavailable; the full observation remains in the table."
            )
            return
        coordinate = self._coordinates[source_row]
        self.canvas.selected_row = source_row
        self.time_spin.setValue(self._frames[source_row])
        if len(coordinate) == 3:
            self.plane_spin.setValue(math.floor(coordinate[0] + 0.5))
        self._update_plane()
        if not self._overlay_available:
            self._identify(source_row)
            return
        track = f"Track {self._track_ids[source_row]} · " if self._track_ids else ""
        note = (
            " · Review: competing feasible links"
            if self._reviews and self._reviews[source_row]
            else ""
        )
        self.selection_label.setText(
            f"{track}T = {self._frames[source_row]} · "
            f"{self._metadata.id_column} = {self._ids[source_row]}"
            + " · "
            + ", ".join(
                f"{axis.upper()} = {value:g}"
                for axis, value in zip(
                    self._metadata.spatial_axes, coordinate, strict=True
                )
            )
            + note
        )


class TrackingResultsController(DetectionResultsController):
    """Resolve only the exact resident source along a supported observation route."""

    def __init__(self, host):
        super().__init__(host)
        self.section.setTitle("Time-series review")
        self.inspect_button.setText("Review trajectories on source…")
        self.inspect_button.setObjectName("ReviewTrajectoriesOnSource")

    def _payload(self, node_id):
        host, pipeline = self.host, self.host.pipeline
        node = pipeline.nodes.get(node_id)
        if node is None or node.operation_id not in SERIES_REVIEW_OPERATIONS:
            return (
                None,
                "Select time-series detections, object measurements or Build Tracks.",
            )
        if (
            host._debounce_timer.isActive()
            or host._active_pipeline_run_id is not None
            or host._active_source_load_id is not None
            or host._pipeline_run_pending
            or host._source_load_pending
        ):
            return None, "Wait for the pending calculation, then refresh the view."
        if host._node_execution_ui_state(node_id)[0] != EXECUTION_READY:
            return None, "Calculate current settings before reviewing trajectories."
        table, _state = host._node_output_payload_for_port(node_id, 0)
        metadata = getattr(table, "observation_metadata", None)
        if not is_table_data(table) or metadata is None:
            return (
                None,
                "Review requires observations from a scalar, explicit TYX/TZYX series.",
            )
        route, seen, current = [], set(), node
        while current.id not in seen:
            seen.add(current.id)
            connection = next(
                (
                    c
                    for c in pipeline.connections
                    if c.target_id == current.id and c.target_port == 0
                ),
                None,
            )
            if connection is None:
                return None, "The original time series is not connected or resident."
            route.append(connection)
            if current.operation_id in {
                "detect_spots_per_frame",
                "measure_objects",
                "measure_objects_intensity",
            }:
                break
            if current.operation_id not in {
                "build_tracks",
                "add_metadata_columns",
                "select_table_columns",
            }:
                return (
                    None,
                    "Source review requires an unmerged observation route "
                    "to its original series.",
                )
            current = pipeline.nodes.get(connection.source_id)
            if current is None:
                return (
                    None,
                    "The observation source was removed. Reconnect and calculate.",
                )
            if host._node_execution_ui_state(current.id)[0] != EXECUTION_READY:
                return None, "Recalculate the changed upstream observations."
        else:
            return None, "Invalid cyclic source route."
        source_id, port = connection.source_id, connection.source_port
        if source_id not in pipeline.nodes:
            return None, "The original source was removed. Reconnect and calculate."
        source, state = host._node_output_payload_for_port(source_id, port)
        if host._node_execution_ui_state(source_id)[
            0
        ] != EXECUTION_READY or not isinstance(source, np.ndarray):
            return (
                None,
                "The original series is not current and resident. Recalculate first.",
            )
        if (
            source.shape != (metadata.frame_count, *metadata.source_shape)
            or not isinstance(state, ImageState)
            or not state.axes_explicit
            or state.shape != source.shape
            or state.dtype != source.dtype.name
            or state.axes[0].type != "time"
            or any(a.type != "space" for a in state.axes[1:])
            or tuple(a.name.lower() for a in state.axes)
            != ("t", *metadata.spatial_axes)
            or tuple(a.scale for a in state.axes[1:]) != metadata.source_scale
            or tuple(a.translation for a in state.axes[1:]) != metadata.source_origin
            or tuple(a.unit for a in state.axes[1:]) != metadata.source_units
            or (state.axes[0].scale, state.axes[0].translation, state.axes[0].unit)
            != (metadata.time_scale, metadata.time_origin, metadata.time_unit)
        ):
            return None, "Source time/space geometry changed. Recalculate observations."
        if _frame(state, metadata.source_revision) != metadata.source_frame:
            return (
                None,
                "The connected source identity changed. Recalculate observations.",
            )
        return (
            table,
            source,
            source_id,
            (id(pipeline), id(table), id(source), id(state), tuple(route)),
        ), ""

    def refresh(self):
        graph = getattr(self.host, "graph_view", None)
        if graph is not None:
            for node in self.host.pipeline.nodes.values():
                if node.operation_id not in SERIES_REVIEW_OPERATIONS:
                    continue
                table, _state = self.host._node_output_payload_for_port(node.id, 0)
                visible = node.operation_id in {
                    "detect_spots_per_frame", "build_tracks"
                } or (
                    is_table_data(table) and table.observation_metadata is not None
                )
                graph.set_node_trajectory_review_visible(node.id, visible)
        node = self.host.pipeline.nodes.get(self.host._selected_node_id)
        if node is None or node.operation_id not in SERIES_REVIEW_OPERATIONS:
            self.section.hide()
        else:
            payload, reason = self._payload(node.id)
            self.inspect_button.setEnabled(payload is not None)
            self.inspect_button.setToolTip(
                reason
                or "Read-only T/Z navigation, observed trails "
                "and linked table selection."
            )
            self.summary.setText(
                f"{payload[0].row_count:,} observations. Review ambiguous links "
                "and gaps; no lineage inference."
                if payload
                else reason
            )
            self.section.setSummary(
                "Current observations" if payload else "Not available"
            )
        self._check_dialog()

    def open_node(self, node_id):
        payload, reason = self._payload(node_id)
        if payload is None:
            self.host._set_status(reason, severity=MessageSeverity.INFO)
            self._check_dialog()
            return
        table, source, source_id, token = payload
        if self.dialog is None:
            self.dialog = TrackingResultsDialog(self.host)
            self.dialog.refreshRequested.connect(self._refresh_dialog)
            self.dialog.finished.connect(lambda _result: self._timer.stop())
            self.dialog.table_panel.export_guard = self._dialog_is_current
        self.dialog.set_result(
            table,
            source,
            node_id=node_id,
            token=token,
            title=self.host._node_title(node_id),
            source_title=self.host._node_title(source_id),
        )
        self.dialog.refresh_theme(QWidget.palette(self.host))
        self.dialog.show()
        self.dialog.raise_()
        self.dialog.activateWindow()
        self._timer.start()
