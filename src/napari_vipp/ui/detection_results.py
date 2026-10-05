"""Read-only source-image inspection of typed detection tables.

The dialog is presentation only: its plane, selection and contrast never enter
the workflow. It shares the ordinary result-table sorting and export surface.
"""

from __future__ import annotations

import math

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen
from qtpy.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from napari_vipp.core.host_memory import capture_host_memory, preflight_host_allocation
from napari_vipp.core.metadata import ImageState
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.preview import normalize_thumbnail_with_colormap
from napari_vipp.core.tables import is_table_data
from napari_vipp.ui.dialog_buttons import add_dialog_buttons
from napari_vipp.ui.inspector import InspectorSection
from napari_vipp.ui.palette_roles import theme_colors
from napari_vipp.ui.result_table_dialog import ResultTablePanel
from napari_vipp.ui.status import MessageSeverity


def _label(text="", parent=None):
    label = QLabel(text, parent)
    label.setTextFormat(Qt.PlainText)
    label.setWordWrap(True)
    label.setMinimumWidth(0)
    label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    return label


class DetectionCanvas(QWidget):
    """Fit-to-window image and non-editing ID markers on one explicit YX plane."""

    markerActivated = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(160, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setAccessibleName("Read-only detection overlay")
        self.setAccessibleDescription(
            "Click a marker to identify its table row. Markers cannot be edited. "
            "The detection table offers keyboard selection."
        )
        self._image = QImage()
        self._shape = (1, 1)
        self._markers = ()
        self.selected_row = None
        self._current = True

    def set_plane(self, plane, markers):
        decision = preflight_host_allocation(
            capture_host_memory(),
            required_bytes=int(plane.size) * 64 + 1400 * 1400 * 12,
            purpose="Detection source-plane preview",
        )
        if not decision.allowed:
            raise MemoryError(decision.reason)
        # Both the pixels and QImage storage are detached from scientific caches.
        plane = np.array(plane, dtype=np.float64, copy=True)
        finite = plane[np.isfinite(plane)]
        limits = (float(finite.min()), float(finite.max())) if finite.size else (0, 1)
        # Remove a wide absolute offset before the shared thumbnail renderer's
        # float32 presentation conversion, so small contrast in wide integers
        # and large finite floats remains visible. This never touches the source.
        center = limits[0] / 2 + limits[1] / 2
        span = max(abs(limits[0] - center), abs(limits[1] - center))
        if span > 0:
            plane -= center
            plane /= span
            limits = ((limits[0] - center) / span, (limits[1] - center) / span)
        rgb = normalize_thumbnail_with_colormap(
            plane,
            size=(1400, 1400),
            contrast_mode="Min-max",
            contrast_limits=limits,
        )
        self._shape = plane.shape
        self._image = QImage(
            rgb.tobytes(order="C"),
            rgb.shape[1],
            rgb.shape[0],
            rgb.shape[1] * 3,
            QImage.Format_RGB888,
        ).copy()
        self._markers = tuple(markers)
        self.update()

    def set_current(self, current):
        self._current = bool(current)
        self.setEnabled(self._current)
        self.update()

    def image_rect(self):
        height, width = self._shape
        available = QRectF(self.rect()).adjusted(8, 8, -8, -8)
        scale = min(available.width() / width, available.height() / height)
        result = QRectF(0, 0, width * scale, height * scale)
        result.moveCenter(available.center())
        return result

    def marker_position(self, y, x):
        rect = self.image_rect()
        return QPointF(
            rect.left() + (float(x) + 0.5) * rect.width() / self._shape[1],
            rect.top() + (float(y) + 0.5) * rect.height() / self._shape[0],
        )

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme_colors(self.palette()).surface)
        if not self._current:
            painter.setPen(theme_colors(self.palette()).text)
            painter.drawText(
                self.rect().adjusted(14, 14, -14, -14),
                Qt.AlignCenter | Qt.TextWordWrap,
                "Overlay unavailable — refresh the current result.",
            )
            return
        if self._image.isNull():
            return
        painter.drawImage(self.image_rect(), self._image)
        painter.setRenderHint(QPainter.Antialiasing)
        for source_row, detection_id, y, x in self._markers:
            point = self.marker_position(y, x)
            selected = source_row == self.selected_row
            radius = 8.0 if selected else 5.0
            # Dark outer and bright inner strokes remain visible over either
            # bright or dark scientific pixels in both application themes.
            painter.setPen(QPen(QColor("#101010"), 4.0))
            painter.drawEllipse(point, radius, radius)
            painter.setPen(QPen(QColor("#ffdc48" if selected else "#52e3ff"), 2.0))
            painter.drawEllipse(point, radius, radius)
            if selected or len(self._markers) <= 40:
                label_point = point + QPointF(radius + 3, -radius - 2)
                text = str(detection_id)
                bounds = painter.fontMetrics().boundingRect(text)
                background = QRectF(bounds)
                background.translate(label_point)
                painter.fillRect(background.adjusted(-2, -1, 2, 1), QColor("#101010"))
                painter.drawText(label_point, text)

    def mousePressEvent(self, event):  # noqa: N802
        if not self._current or event.button() != Qt.LeftButton:
            return
        point = event.position() if hasattr(event, "position") else event.localPos()
        hits = []
        for row, _identity, y, x in self._markers:
            marker = self.marker_position(y, x)
            distance = (marker.x() - point.x()) ** 2 + (marker.y() - point.y()) ** 2
            if distance <= 12**2:
                hits.append((distance, row))
        if hits:
            self.markerActivated.emit(min(hits)[1])


class DetectionResultsDialog(QDialog):
    """Linked full table and original-source overlay, with no editable geometry."""

    refreshRequested = Signal()
    canvas_class = DetectionCanvas

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Inspect detections")
        self.setWindowModality(Qt.NonModal)
        self.setAttribute(Qt.WA_WindowPropagation, True)
        self.resize(1120, 700)
        self.setMinimumSize(600, 400)
        self.node_id = ""
        self.token = None
        self._source = None
        self._coordinates = np.empty((0, 2))
        self._ids = ()
        self._table = None
        self._current = False
        self._selecting = False
        self._preview_memory_error = False
        layout = QVBoxLayout(self)
        self.context_label = _label()
        self.status_label = _label()
        self.selection_label = _label("Select a row or marker to identify a detection.")
        layout.addWidget(self.context_label)
        layout.addWidget(self.status_label)
        splitter = QSplitter(Qt.Horizontal, self)
        image_panel = QWidget()
        image_layout = QVBoxLayout(image_panel)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.plane_controls = QWidget()
        plane_layout = QHBoxLayout(self.plane_controls)
        plane_layout.setContentsMargins(0, 0, 0, 0)
        self.plane_label = QLabel("Z plane (zero-based)")
        self.plane_spin = QSpinBox()
        self.plane_spin.setAccessibleName("Detection inspection Z plane, zero-based")
        self.plane_label.setBuddy(self.plane_spin)
        plane_layout.addWidget(self.plane_label)
        plane_layout.addWidget(self.plane_spin)
        plane_layout.addStretch()
        image_layout.addWidget(self.plane_controls)
        self.canvas = self.canvas_class()
        image_layout.addWidget(self.canvas, 1)
        self.plane_note = _label()
        image_layout.addWidget(self.plane_note)
        image_layout.addWidget(self.selection_label)
        splitter.addWidget(image_panel)
        self.table_panel = ResultTablePanel()
        self.table_panel.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_panel.table_view.setSelectionMode(QAbstractItemView.SingleSelection)
        splitter.addWidget(self.table_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        note = _label(
            "Review candidates on the original image: a high score is not proof "
            "of an object. This fit-to-window preview uses per-plane min–max contrast; "
            "selection, display contrast and plane changes do not alter detections."
        )
        self.review_note = note
        layout.addWidget(note)
        buttons = QHBoxLayout()
        buttons.addStretch()
        self.refresh_button = QPushButton("Refresh view")
        self.refresh_button.setToolTip(
            "Display an already calculated, current result. Does not calculate."
        )
        self.close_button = QPushButton("Close")
        add_dialog_buttons(
            buttons, actions=(self.refresh_button,), dismiss=self.close_button
        )
        layout.addLayout(buttons)
        self.close_button.clicked.connect(self.close)
        self.refresh_button.clicked.connect(self.refreshRequested.emit)
        self.plane_spin.valueChanged.connect(self._update_plane)
        self.canvas.markerActivated.connect(self.select_source_row)
        self.table_panel.table_view.selectionModel().currentRowChanged.connect(
            self._on_table_row
        )
        self.table_panel.sortCompleted.connect(self._after_sort)
        layout.activate()
        splitter.setSizes([540, 540])

    def set_result(self, table, source, *, node_id, token, title, source_title):
        metadata = table.detection_metadata
        self.node_id, self.token = node_id, token
        self._table = table
        # The dialog never exposes or modifies the resident scientific volume.
        # Only the currently displayed plane is copied by DetectionCanvas.
        self._source = source
        columns = [table.columns.index(name) for name in metadata.coordinate_columns]
        self._coordinates = np.asarray(
            [[row[column] for column in columns] for row in table.rows],
            dtype=float,
        ).reshape((-1, len(columns)))
        self._ids = tuple(
            row[table.columns.index("detection_id")] for row in table.rows
        )
        self._current = True
        self._preview_memory_error = False
        self.table_panel.set_table(
            table,
            title=title,
            default_export_name="detections.csv",
            context_key=(node_id, 0),
        )
        self.table_panel.export_button.setEnabled(True)
        self.table_panel.table_view.setEnabled(True)
        self.table_panel.set_result_status()
        self.canvas.selected_row = None
        self.canvas.set_current(True)
        self.plane_controls.setVisible(source.ndim == 3)
        self.plane_spin.setEnabled(True)
        self.plane_spin.setRange(0, source.shape[0] - 1 if source.ndim == 3 else 0)
        self.setWindowTitle(f"{title} — Inspect detections")
        self.context_label.setText(
            f"{table.row_count:,} detections · source: {source_title} · "
            f"{''.join(metadata.axes).upper()} · read-only markers "
            "labelled by detection_id"
        )
        truncated = bool(metadata.truncated)
        self.status_label.setText(
            f"Result limit reached: showing {metadata.returned_count:,} of "
            f"{metadata.accepted_count:,} accepted detections."
            if truncated
            else ""
        )
        self.status_label.setVisible(truncated)
        self.selection_label.setText("Select a row or marker to identify a detection.")
        self._update_plane()

    def set_stale(self, message):
        self._current = False
        self.canvas.set_current(False)
        self.plane_spin.setEnabled(False)
        self.table_panel.table_view.setEnabled(False)
        self.table_panel.export_button.setEnabled(False)
        self.table_panel.set_result_status(message)
        self.status_label.setText(message)
        self.status_label.show()
        self.selection_label.setText("Previous result — identification is disabled.")

    def _update_plane(self, *_args):
        if not self._current or self._source is None:
            return
        plane = self.plane_spin.value()
        is_volume = self._source.ndim == 3
        markers = []
        for row, coordinate in enumerate(self._coordinates):
            if not is_volume or math.floor(coordinate[0] + 0.5) == plane:
                markers.append((row, self._ids[row], *coordinate[-2:]))
        try:
            self.canvas.set_plane(
                self._source[plane] if is_volume else self._source, markers
            )
        except MemoryError as exc:
            self._preview_memory_error = True
            self.canvas.set_current(False)
            self.status_label.setText(
                "The image preview needs more free memory. The detection table "
                f"remains available; close other images and refresh the view. {exc}"
            )
            self.status_label.show()
            return
        self.canvas.set_current(True)
        if self._preview_memory_error:
            metadata = self._table.detection_metadata
            self.status_label.setText(
                f"Result limit reached: showing {metadata.returned_count:,} of "
                f"{metadata.accepted_count:,} accepted detections."
                if metadata.truncated else ""
            )
            self.status_label.setVisible(metadata.truncated)
            self._preview_memory_error = False
        self.plane_note.setText(
            f"Z = {plane} · {len(markers):,} markers on this plane. "
            "Fractional Z centers appear on the nearest plane; half-planes round up. "
            "Selecting a table row moves to its plane. No projection."
            if is_volume
            else f"YX image · {len(markers):,} markers. No projection."
        )

    def _on_table_row(self, current, _previous):
        if self._current and current.isValid() and not self._selecting:
            self._identify(self.table_panel.model.source_row(current.row()))

    def _identify(self, source_row):
        if not self._current or not 0 <= source_row < len(self._coordinates):
            return
        coordinate = self._coordinates[source_row]
        self.canvas.selected_row = source_row
        if self._source.ndim == 3:
            self.plane_spin.setValue(math.floor(coordinate[0] + 0.5))
        self.canvas.update()
        metadata = self._table.detection_metadata
        fields = ", ".join(
            f"{axis.upper()} = {value:g}"
            for axis, value in zip(metadata.axes, coordinate, strict=True)
        )
        value = self._table.rows[source_row][
            self._table.columns.index(metadata.score_column)
        ]
        self.selection_label.setText(
            f"Detection {self._ids[source_row]} · source indices: {fields} "
            f"· value: {value}"
        )

    def select_source_row(self, source_row):
        if not self._current:
            return
        model = self.table_panel.model
        displayed = next(
            (
                row
                for row in range(model.rowCount())
                if model.source_row(row) == source_row
            ),
            None,
        )
        if displayed is None:
            return
        self._selecting = True
        try:
            self.table_panel.table_view.selectRow(displayed)
            self.table_panel.table_view.scrollTo(model.index(displayed, 0))
        finally:
            self._selecting = False
        self._identify(source_row)

    def _after_sort(self, *_args):
        if self.canvas.selected_row is not None:
            self.select_source_row(self.canvas.selected_row)

    def refresh_theme(self, palette):
        self.setPalette(palette)
        self.table_panel.refresh_theme(palette)
        self.canvas.update()

    def closeEvent(self, event):  # noqa: N802
        self.table_panel._cancel_active_sort()
        self._source = None
        self._current = False
        super().closeEvent(event)


class DetectionResultsController:
    """Keep retained tables and source overlays tied to one current cache result."""

    def __init__(self, host):
        self.host = host
        self.section = InspectorSection("Detection review", expanded=True, parent=host)
        layout = QVBoxLayout(self.section.content_widget)
        layout.setContentsMargins(7, 5, 7, 7)
        self.summary = _label()
        self.inspect_button = QPushButton("Inspect detections on source…")
        self.inspect_button.setObjectName("InspectDetectionsOnSource")
        layout.addWidget(self.summary)
        layout.addWidget(self.inspect_button)
        self.inspect_button.clicked.connect(self.open_selected)
        self.dialog = None
        self._timer = QTimer(host)
        self._timer.setInterval(250)
        self._timer.timeout.connect(self._check_dialog)

    def _payload(self, node_id):
        host = self.host
        pipeline = host.pipeline
        node = pipeline.nodes.get(node_id)
        if node is None or node.operation_id != "find_peaks":
            return None, "Select a Find Peaks node to inspect detections."
        if (
            host._debounce_timer.isActive()
            or host._active_pipeline_run_id is not None
            or host._active_source_load_id is not None
            or host._pipeline_run_pending
            or host._source_load_pending
        ):
            return (
                None,
                "Wait for the pending edit or calculation, then refresh the view.",
            )
        if host._node_execution_ui_state(node_id)[0] != EXECUTION_READY:
            return None, "Calculate the current Find Peaks settings before inspection."
        table = pipeline.outputs.get(node_id)
        metadata = getattr(table, "detection_metadata", None)
        if not is_table_data(table) or metadata is None:
            return None, "Calculate Find Peaks to create a detection table."
        connection = next(
            (
                item
                for item in pipeline.connections
                if item.target_id == node_id and item.target_port == 0
            ),
            None,
        )
        if connection is None:
            return None, "The detection image is disconnected."
        route = [connection]
        upstream = pipeline.nodes.get(connection.source_id)
        if upstream is None:
            return None, "The detection source no longer exists."
        if metadata.template_shape is not None:
            if upstream.operation_id != "template_match" or connection.source_port != 0:
                return None, (
                    "Source overlay requires Find Peaks connected directly to the "
                    "Template Match scores that produced these coordinates."
                )
            if host._node_execution_ui_state(upstream.id)[0] != EXECUTION_READY:
                return None, "Recalculate the changed Template Match result first."
            connection = next(
                (
                    item
                    for item in pipeline.connections
                    if item.target_id == upstream.id and item.target_port == 0
                ),
                None,
            )
            if connection is None:
                return None, "The original search image is disconnected."
            route.append(connection)
        source_id, port = connection.source_id, connection.source_port
        if host._node_execution_ui_state(source_id)[0] != EXECUTION_READY:
            return None, "The original source is no longer current. Recalculate first."
        source, state = host._node_output_payload_for_port(source_id, port)
        if not isinstance(source, np.ndarray):
            return None, "The original source image is not resident. Recalculate first."
        if tuple(source.shape) != tuple(metadata.source_shape):
            return None, "The source shape has changed. Recalculate detections."
        if source.ndim not in (2, 3):
            return (
                None,
                "Detection inspection requires a scalar YX image or ZYX volume.",
            )
        if not isinstance(state, ImageState) or (
            tuple(axis.name.lower() for axis in state.axes) != metadata.axes
            or tuple(axis.scale for axis in state.axes) != metadata.source_scale
            or tuple(axis.translation for axis in state.axes) != metadata.source_origin
            or tuple(axis.unit for axis in state.axes) != metadata.source_units
        ):
            return None, "The source calibration has changed. Recalculate detections."
        token = (id(pipeline), id(table), id(source), id(state), tuple(route))
        return (table, source, source_id, token), ""

    def refresh(self):
        node = self.host.pipeline.nodes.get(self.host._selected_node_id)
        if node is None or node.operation_id != "find_peaks":
            self.section.hide()
        else:
            payload, reason = self._payload(node.id)
            self.inspect_button.setEnabled(payload is not None)
            self.inspect_button.setToolTip(
                reason
                or (
                    "Read-only source markers and sortable table "
                    "linked by detection ID."
                )
            )
            table = self.host.pipeline.outputs.get(node.id)
            metadata = getattr(table, "detection_metadata", None)
            if metadata is None:
                text = (
                    "Calculate to create candidate detections, "
                    "then review their locations."
                )
            else:
                text = f"{metadata.returned_count:,} detections"
                if metadata.truncated:
                    text += f" · limit reached ({metadata.accepted_count:,} accepted)"
                text += (
                    ". A peak is a candidate, not an object label "
                    "or confidence probability."
                )
            self.summary.setText(text)
            self.section.setSummary(
                f"{metadata.returned_count:,} candidates"
                if metadata
                else "Not calculated"
            )
        self._check_dialog()

    def open_selected(self):
        self.open_node(self.host._selected_node_id)

    def open_node(self, node_id):
        payload, reason = self._payload(node_id)
        if payload is None:
            self.host._set_status(reason, severity=MessageSeverity.INFO)
            self._check_dialog()
            return
        table, source, source_id, token = payload
        if self.dialog is None:
            self.dialog = DetectionResultsDialog(self.host)
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

    def _refresh_dialog(self):
        if self.dialog is None:
            return
        if self.dialog.token is None or self.dialog.token[0] != id(self.host.pipeline):
            self.dialog.set_stale(
                "This window belongs to another workflow. Return to that workflow "
                "or open Inspect detections from the selected workflow."
            )
            return
        self.open_node(self.dialog.node_id)

    def _dialog_is_current(self):
        if self.dialog is None:
            return False
        payload, _reason = self._payload(self.dialog.node_id)
        return payload is not None and payload[3] == self.dialog.token

    def _check_dialog(self):
        if self.dialog is None or not self.dialog.isVisible():
            return
        payload, reason = self._payload(self.dialog.node_id)
        if payload is None or payload[3] != self.dialog.token:
            self.dialog.set_stale(
                reason
                or (
                    "The result or source changed. Refresh view to inspect the current "
                    "calculation; no calculation starts automatically."
                )
            )

    def refresh_theme(self, palette):
        if self.dialog is not None:
            self.dialog.refresh_theme(palette)

    def close(self):
        self._timer.stop()
        if self.dialog is not None:
            self.dialog.close()
            self.dialog._source = None
