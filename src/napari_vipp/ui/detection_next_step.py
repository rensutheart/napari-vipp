"""Manual, undoable Template Match to Find Peaks graph authoring."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtWidgets import QLabel, QMenu, QPushButton, QSizePolicy, QVBoxLayout

from napari_vipp.ui.inspector import InspectorSection
from napari_vipp.ui.registration_next_step import RegistrationNextStepController
from napari_vipp.ui.status import MessageSeverity


class DetectionNextStepController:
    """Connect both score and validity outputs, never calculate implicitly."""

    def __init__(self, host):
        self._host = host
        self.section = InspectorSection("Next step", expanded=True, parent=host)
        layout = QVBoxLayout(self.section.content_widget)
        layout.setContentsMargins(8, 7, 8, 8)
        self.description = QLabel(
            "Template Match creates a score map. Find Peaks turns its local "
            "maxima into a detection table for review on the source image."
        )
        self.explanation = QLabel()
        for label in (self.description, self.explanation):
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.main_button = QPushButton("Add Find Peaks")
        self.main_button.setObjectName("DetectionNextStepMain")
        self.main_button.clicked.connect(self._activate)
        self.add_another_button = QPushButton("Add another Find Peaks")
        self.add_another_button.setObjectName("DetectionNextStepAddAnother")
        self.add_another_button.clicked.connect(self.add_find_peaks)
        self._show_menu = None
        for child in (
            self.description,
            self.main_button,
            self.add_another_button,
            self.explanation,
        ):
            layout.addWidget(child)

    def _selected_matcher(self):
        node = self._host.pipeline.nodes.get(self._host._selected_node_id)
        return node if node and node.operation_id == "template_match" else None

    def matching_peak_nodes(self):
        matcher = self._selected_matcher()
        if matcher is None:
            return ()
        inputs = {
            (item.target_id, item.target_port): (item.source_id, item.source_port)
            for item in self._host.pipeline.connections
        }
        return tuple(
            node.id
            for node in self._host.pipeline.nodes.values()
            if node.operation_id == "find_peaks"
            and node.params.get("use_mask", False) is True
            and inputs.get((node.id, 0)) == (matcher.id, 0)
            and inputs.get((node.id, 1)) == (matcher.id, 1)
        )

    def _add_block_reason(self):
        host = self._host
        matcher = self._selected_matcher()
        if matcher is None:
            return "Select a Template Match node first."
        connected = {
            item.target_port
            for item in host.pipeline.connections
            if item.target_id == matcher.id
        }
        if not {0, 1}.issubset(connected):
            return "Connect the search image and template before adding Find Peaks."
        if host._isolated_tuning_node_id is not None:
            return "Apply or cancel isolated tuning before adding a node."
        if host._debounce_timer.isActive():
            return "Wait for the pending parameter edit before adding a node."
        return (
            host._compute_policy_edit_block_reason()
            .replace("changing this workflow's compute policy", "editing this workflow")
            .replace("changing compute policy", "editing this workflow")
            .replace("changing policy", "editing this workflow")
        )

    def refresh(self):
        matches = self.matching_peak_nodes()
        reason = self._add_block_reason()
        self.main_button.setMenu(None)
        if self._show_menu is not None:
            self._show_menu.deleteLater()
            self._show_menu = None
        self.main_button.setText("Show Find Peaks" if matches else "Add Find Peaks")
        self.main_button.setEnabled(bool(matches) or not reason)
        self.main_button.setToolTip(
            "Select and center the connected Find Peaks node."
            if matches
            else reason or "Connect the scores and their valid-scores mask."
        )
        if len(matches) > 1:
            self._show_menu = QMenu(self.main_button)
            for node_id in matches:
                action = self._show_menu.addAction(self._host._node_title(node_id))
                action.triggered.connect(
                    lambda _checked=False, selected=node_id: self.show_find_peaks(
                        selected
                    )
                )
            self.main_button.setMenu(self._show_menu)
        self.add_another_button.setVisible(bool(matches))
        self.add_another_button.setEnabled(not reason)
        self.add_another_button.setToolTip(
            reason or "Add another detection branch without replacing connections."
        )
        self.explanation.setText(
            reason
            or (
                "Already connected. Add another only for a separate detection branch."
                if matches
                else "Adds both connections and enables the mask. "
                "Review the minimum score "
                "and separation, then calculate."
            )
        )

    def _activate(self):
        matches = self.matching_peak_nodes()
        if len(matches) == 1:
            self.show_find_peaks(matches[0])
        elif not matches:
            self.add_find_peaks()

    def show_find_peaks(self, node_id=None):
        matches = self.matching_peak_nodes()
        if node_id is None and len(matches) == 1:
            node_id = matches[0]
        if node_id not in matches:
            self.refresh()
            return
        self._host.graph_view.focus_node(node_id)

    def add_find_peaks(self):
        host = self._host
        reason = self._add_block_reason()
        if reason:
            host._set_status(reason, severity=MessageSeverity.INFO)
            self.refresh()
            return None
        matcher = self._selected_matcher()
        host._finish_parameter_history_group()
        before = host._current_history_snapshot()
        try:
            node = host.pipeline.add_node("find_peaks")
            host.pipeline.set_param(node.id, "use_mask", True)
            host.graph_view.add_node(
                node, host.graph_view.suggest_append_position(matcher.id)
            )
            host._sync_node_input_ports(node.id)
            host._sync_node_output_ports(node.id)
            for port in (0, 1):
                result = host.pipeline.connect(
                    matcher.id, node.id, source_port=port, target_port=port
                )
                if not result.success or result.removed:
                    raise RuntimeError(
                        result.message or "An existing connection would be replaced."
                    )
                host._apply_connection_result_to_graph(result)
            host._sync_input_node_subtitle(node.id)
            self._place_without_overlap(node.id)
            host._mark_pipeline_dirty(node.id)
            host._sync_pin_ui()
            host._refresh_graph_search_matches(reset_index=True)
            host.graph_view.focus_node(node.id)
            self._place_without_overlap(node.id)
            host._push_undo_if_changed(before)
        except Exception as exc:
            host._restore_history_snapshot(before, schedule_run=False)
            host._set_status(
                f"Could not add Find Peaks; the workflow was restored. {exc}",
                severity=MessageSeverity.ERROR,
                actionable=True,
            )
            self.refresh()
            return None
        host._set_status(
            "Added Find Peaks with scores and valid-scores mask connected. "
            "Review the settings, then calculate. Undo removes this addition "
            "in one step.",
            severity=MessageSeverity.SUCCESS,
        )
        return node

    def _place_without_overlap(self, node_id):
        # Share the existing presentation-only placement rule, not science or
        # authoring behavior, with the registration suggestion.
        RegistrationNextStepController._place_without_overlap(self, node_id)
