"""A contextual, undoable next step for estimated registration transforms.

The composition root supplies the existing graph/history authoring services.
This controller never calculates or changes scientific registration settings.
"""

from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtWidgets import QLabel, QMenu, QPushButton, QSizePolicy, QVBoxLayout

from napari_vipp.ui.inspector import InspectorSection
from napari_vipp.ui.status import MessageSeverity


class RegistrationNextStepController:
    """Offer Add/Show Apply Transform for the selected estimator.

    ``host`` supplies the widget's existing graph, history and presentation
    adapters. No import of the application composition root is required.
    """

    def __init__(self, host) -> None:
        self._host = host
        self.section = InspectorSection("Next step", expanded=True, parent=host)
        layout = QVBoxLayout(self.section.content_widget)
        layout.setContentsMargins(8, 7, 8, 8)
        layout.setSpacing(6)

        self.heading = QLabel("Create the aligned image", self.section)
        heading_font = self.heading.font()
        heading_font.setBold(True)
        self.heading.setFont(heading_font)
        self.description = QLabel(
            "This node calculates the movement. Apply Transform uses it to align "
            "the original image.",
            self.section,
        )
        self.explanation = QLabel(self.section)
        for label in (self.heading, self.description, self.explanation):
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setMinimumWidth(0)
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.main_button = QPushButton("Add Apply Transform", self.section)
        self.main_button.setObjectName("RegistrationNextStepMain")
        self.main_button.clicked.connect(self._activate)
        self.add_another_button = QPushButton(
            "Add another Apply Transform", self.section
        )
        self.add_another_button.setObjectName("RegistrationNextStepAddAnother")
        self.add_another_button.clicked.connect(self.add_apply_transform)
        self.add_another_button.hide()
        self._show_menu: QMenu | None = None
        for child in (
            self.heading,
            self.description,
            self.main_button,
            self.add_another_button,
            self.explanation,
        ):
            layout.addWidget(child)

    def _selected_estimator(self):
        node = self._host.pipeline.nodes.get(self._host._selected_node_id)
        return (
            node
            if node is not None and node.operation_id == "estimate_registration"
            else None
        )

    def _moving_connection(self, estimator_id: str):
        return next(
            (
                connection
                for connection in self._host.pipeline.connections
                if connection.target_id == estimator_id and connection.target_port == 0
            ),
            None,
        )

    def matching_apply_nodes(self) -> tuple[str, ...]:
        """Match both exact input ports, not merely any transform consumer."""
        estimator = self._selected_estimator()
        if estimator is None:
            return ()
        moving = self._moving_connection(estimator.id)
        if moving is None:
            return ()
        inputs = {
            (connection.target_id, connection.target_port): (
                connection.source_id,
                connection.source_port,
            )
            for connection in self._host.pipeline.connections
        }
        return tuple(
            node.id
            for node in self._host.pipeline.nodes.values()
            if node.operation_id == "apply_transform"
            and inputs.get((node.id, 0)) == (moving.source_id, moving.source_port)
            and inputs.get((node.id, 1)) == (estimator.id, 0)
        )

    def _add_block_reason(self) -> str:
        host = self._host
        estimator = self._selected_estimator()
        if estimator is None:
            return "Select an Estimate Registration node first."
        if self._moving_connection(estimator.id) is None:
            return (
                "Connect the moving image or time series before adding Apply Transform."
            )
        if host._isolated_tuning_node_id is not None:
            return "Apply or cancel isolated tuning before adding a node."
        if host._debounce_timer.isActive():
            return "Wait for the pending parameter edit before adding a node."
        reason = host._compute_policy_edit_block_reason()
        return (
            reason.replace(
                "changing this workflow's compute policy", "editing this workflow"
            )
            .replace("changing compute policy", "editing this workflow")
            .replace("changing policy", "editing this workflow")
        )

    def refresh(self) -> None:
        """Synchronize the card; the inspector owns section placement/visibility."""
        matches = self.matching_apply_nodes()
        reason = self._add_block_reason()
        self.main_button.setMenu(None)
        if self._show_menu is not None:
            self._show_menu.deleteLater()
            self._show_menu = None
        if matches:
            self.main_button.setText("Show Apply Transform")
            self.main_button.setEnabled(True)
            self.main_button.setToolTip(
                "Select and center the connected Apply Transform node."
                if len(matches) == 1
                else "Choose which connected Apply Transform node to show."
            )
            if len(matches) > 1:
                self._show_menu = QMenu(self.main_button)
                for node_id in matches:
                    title = self._host._node_title(node_id)
                    action = self._show_menu.addAction(title)
                    action.triggered.connect(
                        lambda _checked=False, selected=node_id: (
                            self.show_apply_transform(selected)
                        )
                    )
                self.main_button.setMenu(self._show_menu)
        else:
            self.main_button.setText("Add Apply Transform")
            self.main_button.setEnabled(not reason)
            self.main_button.setToolTip(
                reason
                or "Connect a new node to this transform and its original moving image."
            )
        self.add_another_button.setVisible(bool(matches))
        self.add_another_button.setEnabled(not reason)
        self.add_another_button.setToolTip(
            reason
            or "Add a separate branch without replacing existing connections. "
            "Its image input initially uses the same moving image; you can reconnect "
            "compatible labels or another image."
        )
        note = reason or (
            "Already connected. Add another only if you need a separate aligned output."
            if matches
            else "Adds both connections. Review its settings, then calculate."
        )
        self.explanation.setText(note)
        self.explanation.setVisible(bool(note))

    def _activate(self) -> None:
        matches = self.matching_apply_nodes()
        if len(matches) == 1:
            self.show_apply_transform(matches[0])
        elif not matches:
            self.add_apply_transform()

    def show_apply_transform(self, node_id: str | None = None) -> None:
        """Navigate without editing; stale menu targets never select another branch."""
        matches = self.matching_apply_nodes()
        if node_id is None and len(matches) == 1:
            node_id = matches[0]
        if node_id not in matches:
            self.refresh()
            return
        self._host.graph_view.focus_node(node_id)

    def _place_without_overlap(self, node_id: str) -> None:
        graph = self._host.graph_view
        rect = graph.node_scene_rect(node_id)
        if rect is None:
            return
        obstacles = [
            other
            for other_id in self._host.pipeline.nodes
            if other_id != node_id
            and (other := graph.node_scene_rect(other_id)) is not None
        ]
        # Keep the suggested rightward column and step below occupied cards.
        # Each move passes at least one obstacle, so the loop is bounded by N.
        candidate = QRectF(rect)
        for _ in range(len(obstacles) + 1):
            collisions = [
                other
                for other in obstacles
                if candidate.adjusted(-24.0, -24.0, 24.0, 24.0).intersects(other)
            ]
            if not collisions:
                break
            candidate.moveTop(max(other.bottom() for other in collisions) + 40.0)
        x, y = graph.node_positions()[node_id]
        graph.apply_node_positions(
            {
                node_id: QPointF(
                    x + candidate.left() - rect.left(), y + candidate.top() - rect.top()
                )
            }
        )

    def add_apply_transform(self) -> object | None:
        """Connect both inputs as one undoable edit, leaving calculation manual."""
        host = self._host
        reason = self._add_block_reason()
        if reason:
            host._set_status(reason, severity=MessageSeverity.INFO)
            self.refresh()
            return None
        estimator = self._selected_estimator()
        moving = self._moving_connection(estimator.id)
        host._finish_parameter_history_group()
        before = host._current_history_snapshot()
        try:
            node = host.pipeline.add_node("apply_transform")
            position = host.graph_view.suggest_append_position(estimator.id)
            host.graph_view.add_node(node, position)
            host._sync_node_input_ports(node.id)
            host._sync_node_output_ports(node.id)
            for source_id, source_port, target_port, tunnel_name in (
                (moving.source_id, moving.source_port, 0, moving.tunnel_name),
                (estimator.id, 0, 1, ""),
            ):
                result = host.pipeline.connect(
                    source_id,
                    node.id,
                    source_port=source_port,
                    target_port=target_port,
                    tunnel_name=tunnel_name,
                )
                if not result.success or result.removed:
                    raise RuntimeError(
                        result.message or "An existing connection would be replaced."
                    )
                host._apply_connection_result_to_graph(result)
            host._sync_node_output_ports(node.id)
            host._sync_input_node_subtitle(node.id)
            self._place_without_overlap(node.id)
            host._mark_pipeline_dirty(node.id)
            host._sync_pin_ui()
            host._refresh_graph_search_matches(reset_index=True)
            host.graph_view.focus_node(node.id)
            # Selection/presentation may change card height. Check the final
            # geometry too, without moving any pre-existing node or branch.
            self._place_without_overlap(node.id)
            host._push_undo_if_changed(before)
        except Exception as exc:
            host._restore_history_snapshot(before)
            host._set_status(
                f"Could not add Apply Transform; the workflow was restored. {exc}",
                severity=MessageSeverity.ERROR,
                actionable=True,
            )
            self.refresh()
            return None
        host._set_status(
            "Added Apply Transform with the original moving image and its "
            "transform connected. Review the settings, then calculate. "
            "Undo removes this addition in one step.",
            severity=MessageSeverity.SUCCESS,
        )
        return node


__all__ = ["RegistrationNextStepController"]
