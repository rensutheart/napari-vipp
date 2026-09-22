"""Present registration's transform and diagnostics together, without routing UI."""

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QFileDialog, QLabel, QPushButton, QSizePolicy, QVBoxLayout

from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.core.tables import is_table_data
from napari_vipp.core.transforms import is_transform_data
from napari_vipp.ui.inspector import InspectorSection
from napari_vipp.ui.result_table_dialog import choose_table_export_target


class RegistrationResultsController:
    """Explicit result actions; no display selection or graph changes."""

    def __init__(self, host):
        self.host = host
        self.section = InspectorSection("Transform and exports", expanded=True)
        layout = QVBoxLayout(self.section.content_widget)
        layout.setContentsMargins(7, 5, 7, 7)
        layout.setSpacing(5)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.PlainText)
        self.summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.summary.setMinimumWidth(0)
        self.summary.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        layout.addWidget(self.summary)
        self.export_diagnostics = QPushButton("Export diagnostics…")
        self.export_transform = QPushButton("Export transform…")
        self.export_both = QPushButton("Export both…")
        self.export_diagnostics.setToolTip(
            "Export the diagnostics table as CSV or TSV. Displacement units "
            "are shown in the table and preserved in the transform JSON."
        )
        self.export_transform.setToolTip(
            "Save exact movement matrices, reference grid, calibration and settings "
            "as JSON for record-keeping or programmatic reuse. This is not an "
            "image; importing transform JSON through the VIPP UI is not supported."
        )
        self.export_both.setToolTip(
            "Save the transform JSON and diagnostics CSV together in one folder."
        )
        for button in (
            self.export_diagnostics, self.export_transform, self.export_both,
        ):
            layout.addWidget(button)
        self.export_transform.clicked.connect(lambda: self.export_output(0))
        self.export_diagnostics.clicked.connect(lambda: self.export_output(1))
        self.export_both.clicked.connect(host._save_all_selected_node_outputs_dialog)

    def refresh(self):
        host = self.host
        node = host.pipeline.nodes.get(host._selected_node_id)
        if node is None or node.operation_id != "estimate_registration":
            self.section.hide()
            return
        transform, _state = host._node_output_payload_for_port(node.id, 0)
        diagnostics, _state = host._node_output_payload_for_port(node.id, 1)
        state, _message = host._node_execution_ui_state(node.id)
        ready = state == EXECUTION_READY
        if is_transform_data(transform):
            count = transform.state.transform_count
            reference = (
                f"Reference time: {transform.reference_time} (zero-based)"
                if transform.is_time_series
                else "Reference: " + (
                    transform.state.reference_name or "connected reference image"
                )
            )
            unit = "µm" if transform.state.coordinate_unit == "micrometer" else "pixels"
            axes = ''.join(transform.state.spatial_axes).upper()
            noun = "transforms" if count != 1 else "transform"
            text = (
                f"{transform.state.model} · {count} {noun}\n"
                f"Spatial axes: {axes} · {unit}\n"
                f"{reference}\n"
                "Movement instructions only; Apply Transform creates the aligned image."
            )
            if not ready:
                text = "Previous result — recalculate before exporting.\n" + text
            self.summary.setText(text)
            self.section.setSummary(f"{count} transform{'s' if count != 1 else ''}")
        else:
            self.summary.setText(
                "Calculate registration to create movement instructions and the "
                "diagnostics table. No aligned image is produced by this node."
            )
            self.section.setSummary("Not calculated")
        self.export_transform.setEnabled(ready and is_transform_data(transform))
        self.export_diagnostics.setEnabled(ready and is_table_data(diagnostics))
        self.export_both.setEnabled(
            ready and is_transform_data(transform) and is_table_data(diagnostics)
        )

    def export_output(self, port):
        host = self.host
        node = host.pipeline.nodes.get(host._selected_node_id)
        if node is None or node.operation_id != "estimate_registration":
            return
        if host.pipeline.node_execution_states.get(node.id) != EXECUTION_READY:
            host.status_label.setText(
                "Recalculate registration before exporting its results."
            )
            return
        if port == 0:
            path, _filter = QFileDialog.getSaveFileName(
                host, "Export registration transform", "registration_transform.json",
                "VIPP registration transform (*.json)",
            )
            request = (path, "json") if path else None
        else:
            request = choose_table_export_target(
                host, default_name="registration_diagnostics.csv",
                caption="Export registration diagnostics",
            )
        if request is not None:
            path, file_format = request
            host._save_node_output(
                node.id, str(path), format=file_format, output_port=port,
            )
