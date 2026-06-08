"""Qt graph canvas and node card widgets for the VIPP prototype."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPoint, QPointF, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
)

from napari_vipp._theme import category_color, category_tint

OPERATION_MIME = "application/x-napari-vipp-operation"


class ClickablePreview(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class NodeCard(QFrame):
    """Small embedded node UI with a thumbnail and graph actions."""

    selected = Signal(str)
    pin_requested = Signal(str)

    def __init__(
        self,
        node_id: str,
        title: str,
        category: str,
        can_pin: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.node_id = node_id
        self.category = category
        self._category_color = category_color(category)
        self._category_tint = category_tint(category)
        self._can_pin = can_pin
        self._selected = False
        self._pinned = False
        self._preview_enabled = True
        self.setObjectName("NodeCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(220)
        self.setCursor(Qt.OpenHandCursor)

        self.accent_bar = QFrame()
        self.accent_bar.setObjectName("NodeAccent")
        self.accent_bar.setFixedHeight(4)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 650;")
        self.category_label = QLabel(category)
        self.category_label.setObjectName("NodeCategory")

        self.preview = ClickablePreview()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(180, 110)
        self.preview.setText("No preview")
        self.preview.setStyleSheet(
            "background: #111827; color: #9ca3af; border-radius: 4px;"
        )
        self.metadata_label = QLabel("No output")
        self.metadata_label.setWordWrap(True)
        self.metadata_label.setStyleSheet(
            "color: #cbd5e1; font-size: 10px; padding-top: 2px;"
        )
        self.pin_button = QPushButton("Pin")
        self.pin_button.clicked.connect(lambda: self.pin_requested.emit(self.node_id))
        self.pin_button.setVisible(can_pin)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self.pin_button)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(self.accent_bar)
        layout.addWidget(self.category_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.preview)
        layout.addWidget(self.metadata_label)
        layout.addLayout(actions)
        self._refresh_style()

    def mousePressEvent(self, event):  # noqa: N802
        self.selected.emit(self.node_id)
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self._refresh_style()

    def set_pinned(self, pinned: bool) -> None:
        self._pinned = pinned
        self.pin_button.setText("Unpin" if pinned else "Pin")
        self._refresh_style()

    def set_can_pin(self, can_pin: bool) -> None:
        self._can_pin = can_pin
        if not can_pin:
            self._pinned = False
            self.pin_button.setText("Pin")
        self._refresh_style()

    def set_preview_enabled(self, enabled: bool) -> None:
        self._preview_enabled = enabled
        self.preview.setVisible(enabled)
        if not enabled:
            self.preview.setText("")
            self.preview.setPixmap(QPixmap())
        elif self.preview.pixmap() is None:
            self.preview.setText("No preview")

    def set_thumbnail(self, thumbnail: np.ndarray | None) -> None:
        if not self._preview_enabled:
            return
        if thumbnail is None:
            self.preview.setText("Preview off")
            self.preview.setPixmap(QPixmap())
            return

        thumb = np.ascontiguousarray(thumbnail[..., :3].astype(np.uint8, copy=False))
        h, w = thumb.shape[:2]
        qimage = QImage(thumb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.preview.setText("")
        self.preview.setPixmap(
            pixmap.scaled(
                self.preview.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def set_metadata_summary(self, text: str) -> None:
        self.metadata_label.setText(text)

    def _refresh_style(self) -> None:
        border = "#4b5563"
        if self._selected:
            border = "#60a5fa"
        if self._pinned:
            border = "#facc15"
        self.setStyleSheet(
            f"""
            QFrame#NodeCard {{
                background: #20242b;
                border: 2px solid {border};
                border-radius: 6px;
            }}
            QLabel {{
                color: #f3f4f6;
            }}
            QFrame#NodeAccent {{
                background: {self._category_color};
                border: none;
                border-radius: 2px;
            }}
            QPushButton {{
                padding: 3px 7px;
            }}
            """
        )
        self.category_label.setStyleSheet(
            f"""
            QLabel#NodeCategory {{
                background: {self._category_tint};
                color: {self._category_color};
                border-radius: 4px;
                font-size: 10px;
                font-weight: 650;
                padding: 2px 5px;
            }}
            """
        )
        self.pin_button.setVisible(self._can_pin)


class PortItem(QGraphicsEllipseItem):
    """Clickable node port used for graph connections."""

    radius = 6.0
    hover_radius = 8.0
    target_radius = 10.0

    def __init__(self, node_id: str, kind: str, data_type: str, parent):
        super().__init__(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius)
        self.node_id = node_id
        self.kind = kind
        self.data_type = data_type
        self.setParentItem(parent)
        self.setZValue(30)
        self.setCursor(Qt.CrossCursor)
        self.setAcceptHoverEvents(True)
        self.setToolTip(f"{kind}: {data_type}")
        self._hovered = False
        self._active = False
        self._drop_state: str | None = None
        self._refresh_style()

    def set_data_type(self, data_type: str) -> None:
        self.data_type = data_type
        self.setToolTip(f"{self.kind}: {data_type}")
        self._refresh_style()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._refresh_style()

    def set_drop_state(self, state: str | None) -> None:
        self._drop_state = state
        self._refresh_style()

    def hoverEnterEvent(self, event):  # noqa: N802
        self._hovered = True
        self._refresh_style()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):  # noqa: N802
        self._hovered = False
        self._refresh_style()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        view = _view_for_scene(self.scene())
        if view is not None and event.button() == Qt.LeftButton:
            if self.kind == "output":
                view.begin_connection(self, event.scenePos())
            else:
                view.complete_connection(self)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        view = _view_for_scene(self.scene())
        if (
            view is not None
            and self.kind == "output"
            and event.buttons() & Qt.LeftButton
        ):
            view.update_pending_connection(event.scenePos(), dragging=True)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        view = _view_for_scene(self.scene())
        if (
            view is not None
            and self.kind == "output"
            and event.button() == Qt.LeftButton
        ):
            view.release_connection(event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _refresh_style(self) -> None:
        color = "#22c55e"
        if self.data_type == "mask":
            color = "#c084fc"
        elif self.data_type == "array":
            color = "#38bdf8"
        elif self.data_type == "any":
            color = "#f59e0b"
        radius = self.radius
        pen_color = "#111827"
        pen_width = 1.5
        if self._hovered:
            radius = self.hover_radius
            pen_color = "#f9fafb"
            pen_width = 2.0
        if self._active:
            radius = self.target_radius
            pen_color = "#bfdbfe"
            pen_width = 2.4
        if self._drop_state == "compatible":
            radius = self.target_radius
            pen_color = "#f9fafb"
            pen_width = 3.0
        elif self._drop_state == "incompatible":
            radius = self.hover_radius
            pen_color = "#fb7185"
            pen_width = 2.6
        self.setRect(-radius, -radius, radius * 2, radius * 2)
        self.setBrush(QColor(color))
        self.setPen(QPen(QColor(pen_color), pen_width))


class NodeProxy(QGraphicsProxyWidget):
    """Movable graphics item that keeps connected wires attached."""

    def __init__(
        self,
        node_id: str,
        input_type: str | None,
        output_type: str,
        has_input: bool,
        has_output: bool = True,
    ):
        super().__init__()
        self.node_id = node_id
        self.input_type = input_type
        self.output_type = output_type
        self.connections: list[ConnectionItem] = []
        self.input_port: PortItem | None = None
        self.output_port: PortItem | None = None
        self._has_input = has_input
        self._has_output = has_output
        self._drag_start_scene: QPointF | None = None
        self._drag_start_pos: QPointF | None = None
        self._dragging = False
        self._press_was_preview = False
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def refresh_ports(self) -> None:
        rect = self.boundingRect()
        if self._has_input and self.input_type is not None:
            if self.input_port is None:
                self.input_port = PortItem(self.node_id, "input", self.input_type, self)
            self.input_port.setPos(rect.left(), rect.center().y())
        if self._has_output:
            if self.output_port is None:
                self.output_port = PortItem(
                    self.node_id, "output", self.output_type, self
                )
            self.output_port.setPos(rect.right(), rect.center().y())

    def set_output_type(self, output_type: str) -> None:
        self.output_type = output_type
        if self.output_port is not None:
            self.output_port.set_data_type(output_type)

    def port_scene_pos(self, kind: str) -> QPointF:
        port = self.output_port if kind == "output" else self.input_port
        if port is not None:
            return port.mapToScene(QPointF(0, 0))
        rect = self.sceneBoundingRect()
        if kind == "output":
            return QPointF(rect.right(), rect.center().y())
        return QPointF(rect.left(), rect.center().y())

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton and not self._press_on_button(event):
            card = self._card()
            if card is not None:
                card.selected.emit(card.node_id)
                card.setCursor(Qt.ClosedHandCursor)
            self.setSelected(True)
            self._drag_start_scene = QPointF(event.scenePos())
            self._drag_start_pos = QPointF(self.pos())
            self._dragging = False
            self._press_was_preview = self._press_on_preview(event)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._drag_start_scene is not None and event.buttons() & Qt.LeftButton:
            delta = event.scenePos() - self._drag_start_scene
            if delta.manhattanLength() >= 3:
                self._dragging = True
                self.setPos(self._drag_start_pos + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag_start_scene is not None and event.button() == Qt.LeftButton:
            card = self._card()
            if card is not None:
                card.setCursor(Qt.OpenHandCursor)
            self._drag_start_scene = None
            self._drag_start_pos = None
            self._dragging = False
            self._press_was_preview = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):  # noqa: N802
        result = super().itemChange(change, value)
        if change in (
            QGraphicsItem.ItemPositionHasChanged,
            QGraphicsItem.ItemTransformHasChanged,
        ):
            for connection in self.connections:
                connection.update_path()
        return result

    def _card(self) -> NodeCard | None:
        widget = self.widget()
        return widget if isinstance(widget, NodeCard) else None

    def _press_on_button(self, event) -> bool:
        return self._has_parent_widget_type(event, QPushButton)

    def _press_on_preview(self, event) -> bool:
        return self._has_parent_widget_type(event, ClickablePreview)

    def _has_parent_widget_type(self, event, widget_type: type) -> bool:
        card = self._card()
        if card is None:
            return False
        child = card.childAt(_point_from_event(event))
        while child is not None and child is not card:
            if isinstance(child, widget_type):
                return True
            child = child.parentWidget()
        return False


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, source: NodeProxy, target: NodeProxy):
        super().__init__()
        self.source = source
        self.target = target
        self.source_id = source.node_id
        self.target_id = target.node_id
        self.setZValue(-10)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self._refresh_pen()
        self.update_path()

    def update_path(self) -> None:
        start = self.source.port_scene_pos("output")
        end = self.target.port_scene_pos("input")
        self.setPath(_wire_path(start, end))

    def itemChange(self, change, value):  # noqa: N802
        result = super().itemChange(change, value)
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._refresh_pen()
        return result

    def contextMenuEvent(self, event):  # noqa: N802
        view = _view_for_scene(self.scene())
        menu = QMenu()
        info_action = menu.addAction("Info")
        delete_action = menu.addAction("Delete")
        action = _exec_menu(menu, event.screenPos())
        if view is not None and action == delete_action:
            view.delete_connection_item(self, notify=True)
        elif view is not None and action == info_action:
            view.status_message.emit(
                f"Connection {self.source_id} -> {self.target_id}: "
                f"{self.source.output_type} to {self.target.input_type}."
            )

    def _refresh_pen(self) -> None:
        color = "#facc15" if self.isSelected() else "#8aa0c8"
        width = 3.0 if self.isSelected() else 2.0
        self.setPen(QPen(QColor(color), width))


class PendingConnectionItem(QGraphicsPathItem):
    def __init__(self, source_port: PortItem, end: QPointF):
        super().__init__()
        self.source_port = source_port
        pen = QPen(QColor("#d1d5db"), 2.0, Qt.DashLine)
        self.setPen(pen)
        self.setZValue(-5)
        self.update_end(end)

    def update_end(self, end: QPointF) -> None:
        self.setPath(_wire_path(self.source_port.mapToScene(QPointF(0, 0)), end))


class PipelineGraphView(QGraphicsView):
    """Large pan/zoom graph canvas hosted inside napari."""

    node_selected = Signal(str)
    node_delete_requested = Signal(str)
    pin_requested = Signal(str)
    node_create_requested = Signal(str, QPointF)
    connection_requested = Signal(str, str)
    connection_removed = Signal(str, str)
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#151922"))
        self.setAcceptDrops(True)
        self._proxies: dict[str, NodeProxy] = {}
        self._cards: dict[str, NodeCard] = {}
        self._connections: list[ConnectionItem] = []
        self._pending_source: PortItem | None = None
        self._pending_wire: PendingConnectionItem | None = None
        self._highlighted_input_port: PortItem | None = None
        self._connection_dragging = False
        self._panning = False
        self._pan_start = QPoint()
        self._pan_h_value = 0
        self._pan_v_value = 0

    def build_graph(self, nodes, connections, positions=None) -> None:
        self.scene.clear()
        self._proxies.clear()
        self._cards.clear()
        self._connections.clear()
        self._pending_source = None
        self._pending_wire = None
        self._highlighted_input_port = None

        default_positions = {
            "input": QPointF(0, 20),
            "gaussian": QPointF(330, 20),
            "threshold": QPointF(660, 20),
        }
        positions = positions or default_positions
        for index, node in enumerate(nodes):
            fallback = QPointF(330 * index, 20)
            self.add_node(node, positions.get(node.id, fallback))

        for connection in connections:
            self.add_connection(connection.source_id, connection.target_id)

        graph_rect = self.scene.itemsBoundingRect()
        self.scene.setSceneRect(graph_rect.adjusted(-1600, -1200, 1800, 1200))
        self.resetTransform()
        self.fitInView(graph_rect.adjusted(-80, -80, 120, 80), Qt.KeepAspectRatio)

    def build_demo_graph(self, nodes) -> None:
        self.build_graph(nodes, [])

    def add_node(self, node, position: QPointF) -> None:
        card = NodeCard(
            node.id,
            node.title,
            node.category,
            can_pin=node.output_type == "mask",
        )
        card.selected.connect(self._select_node)
        card.pin_requested.connect(self.pin_requested)
        proxy = NodeProxy(
            node.id,
            node.input_type,
            node.output_type,
            node.has_input,
            True,
        )
        proxy.setWidget(card)
        self.scene.addItem(proxy)
        proxy.setPos(position)
        proxy.refresh_ports()
        self._cards[node.id] = card
        self._proxies[node.id] = proxy

    def add_connection(self, source_id: str, target_id: str) -> None:
        if self._connection_exists(source_id, target_id):
            return
        source = self._proxies[source_id]
        target = self._proxies[target_id]
        item = ConnectionItem(source, target)
        self.scene.addItem(item)
        source.connections.append(item)
        target.connections.append(item)
        self._connections.append(item)

    def remove_node(self, node_id: str) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        for connection in list(proxy.connections):
            self.delete_connection_item(connection, notify=False)
        if self._pending_source is not None and self._pending_source.node_id == node_id:
            self._cancel_pending_connection()
        elif (
            self._highlighted_input_port is not None
            and self._highlighted_input_port.node_id == node_id
        ):
            self._highlighted_input_port.set_drop_state(None)
            self._highlighted_input_port = None
        self._cards.pop(node_id, None)
        self._proxies.pop(node_id, None)
        self.scene.removeItem(proxy)

    def remove_connection(
        self,
        source_id: str,
        target_id: str,
        notify: bool = False,
    ) -> None:
        for item in list(self._connections):
            if item.source_id == source_id and item.target_id == target_id:
                self.delete_connection_item(item, notify=notify)

    def delete_connection_item(
        self,
        item: ConnectionItem,
        notify: bool = False,
    ) -> None:
        if item not in self._connections:
            return
        self._connections.remove(item)
        if item in item.source.connections:
            item.source.connections.remove(item)
        if item in item.target.connections:
            item.target.connections.remove(item)
        self.scene.removeItem(item)
        if notify:
            self.connection_removed.emit(item.source_id, item.target_id)

    def set_thumbnail(self, node_id: str, thumbnail: np.ndarray | None) -> None:
        card = self._cards.get(node_id)
        if card is not None:
            card.set_thumbnail(thumbnail)

    def set_node_metadata(self, node_id: str, text: str) -> None:
        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None or proxy is None:
            return
        card.set_metadata_summary(text)
        card.adjustSize()
        proxy.refresh_ports()
        for connection in proxy.connections:
            connection.update_path()

    def set_pinned_node(self, node_id: str | None) -> None:
        for card_id, card in self._cards.items():
            card.set_pinned(card_id == node_id)

    def set_node_can_pin(self, node_id: str, can_pin: bool) -> None:
        card = self._cards.get(node_id)
        if card is not None:
            card.set_can_pin(can_pin)

    def set_node_output_type(self, node_id: str, output_type: str) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is not None:
            proxy.set_output_type(output_type)

    def set_node_preview_enabled(self, node_id: str, enabled: bool) -> None:
        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None or proxy is None:
            return
        card.set_preview_enabled(enabled)
        card.adjustSize()
        proxy.refresh_ports()
        for connection in proxy.connections:
            connection.update_path()

    def select_node(self, node_id: str) -> None:
        if node_id in self._cards:
            self._select_node(node_id)

    def begin_connection(self, source_port: PortItem, scene_pos: QPointF) -> None:
        if source_port.kind != "output":
            return
        self._cancel_pending_connection()
        self._pending_source = source_port
        self._pending_source.set_active(True)
        self._connection_dragging = False
        self._pending_wire = PendingConnectionItem(source_port, scene_pos)
        self.scene.addItem(self._pending_wire)
        self._update_drop_target_feedback(scene_pos)

    def update_pending_connection(self, scene_pos: QPointF, dragging: bool) -> None:
        if self._pending_wire is None:
            return
        self._connection_dragging = self._connection_dragging or dragging
        self._pending_wire.update_end(scene_pos)
        self._update_drop_target_feedback(scene_pos)

    def release_connection(self, scene_pos: QPointF) -> None:
        target = self._input_port_at(scene_pos)
        if target is not None:
            self.complete_connection(target)
            return
        if self._connection_dragging:
            self._cancel_pending_connection()
        elif self._pending_wire is not None:
            self.scene.removeItem(self._pending_wire)
            self._pending_wire = None

    def complete_connection(self, target_port: PortItem) -> None:
        if self._pending_source is None:
            return
        source_port = self._pending_source
        self._cancel_pending_connection()
        if target_port.kind != "input":
            return
        self.connection_requested.emit(source_port.node_id, target_port.node_id)

    def suggest_node_position(self) -> QPointF:
        center = self.mapToScene(self.viewport().rect().center())
        return center + QPointF(40 + len(self._proxies) * 18, 40)

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat(OPERATION_MIME):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat(OPERATION_MIME):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat(OPERATION_MIME):
            operation_id = bytes(event.mimeData().data(OPERATION_MIME)).decode()
            self.node_create_requested.emit(
                operation_id,
                self.mapToScene(_point_from_event(event)),
            )
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected_nodes = [
                item
                for item in self.scene.selectedItems()
                if isinstance(item, NodeProxy)
            ]
            if not selected_nodes:
                selected_nodes = [
                    proxy
                    for node_id, proxy in self._proxies.items()
                    if self._cards[node_id]._selected
                ]
            for item in selected_nodes:
                self.node_delete_requested.emit(item.node_id)
            if selected_nodes:
                event.accept()
                return

            selected = [
                item
                for item in self.scene.selectedItems()
                if isinstance(item, ConnectionItem)
            ]
            for item in selected:
                self.delete_connection_item(item, notify=True)
            if selected:
                event.accept()
                return
        super().keyPressEvent(event)

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        pos = _point_from_event(event)
        background_click = self.itemAt(pos) is None
        if (
            self._pending_source is not None
            and event.button() == Qt.LeftButton
            and background_click
        ):
            self._cancel_pending_connection()
            event.accept()
            return
        if event.button() in (Qt.MiddleButton, Qt.RightButton) or (
            event.button() == Qt.LeftButton and background_click
        ):
            self._start_panning(pos)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._panning:
            pos = _point_from_event(event)
            delta = pos - self._pan_start
            self.horizontalScrollBar().setValue(self._pan_h_value - delta.x())
            self.verticalScrollBar().setValue(self._pan_v_value - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._panning and event.button() in (
            Qt.LeftButton,
            Qt.MiddleButton,
            Qt.RightButton,
        ):
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _select_node(self, node_id: str) -> None:
        for card_id, card in self._cards.items():
            card.set_selected(card_id == node_id)
            proxy = self._proxies.get(card_id)
            if proxy is not None:
                proxy.setSelected(card_id == node_id)
        self.node_selected.emit(node_id)

    def _start_panning(self, pos: QPoint) -> None:
        self._panning = True
        self._pan_start = QPoint(pos)
        self._pan_h_value = self.horizontalScrollBar().value()
        self._pan_v_value = self.verticalScrollBar().value()
        self.setCursor(Qt.ClosedHandCursor)

    def _input_port_at(self, scene_pos: QPointF) -> PortItem | None:
        for item in self.scene.items(scene_pos):
            if isinstance(item, PortItem) and item.kind == "input":
                return item
        return None

    def _update_drop_target_feedback(self, scene_pos: QPointF) -> None:
        target = self._input_port_at(scene_pos)
        if target is self._highlighted_input_port:
            return
        if self._highlighted_input_port is not None:
            self._highlighted_input_port.set_drop_state(None)
        self._highlighted_input_port = target
        if target is None:
            return
        state = "compatible" if self._can_pending_connect_to(target) else "incompatible"
        target.set_drop_state(state)

    def _can_pending_connect_to(self, target_port: PortItem) -> bool:
        if self._pending_source is None or target_port.kind != "input":
            return False
        if self._pending_source.node_id == target_port.node_id:
            return False
        source_proxy = self._proxies.get(self._pending_source.node_id)
        target_proxy = self._proxies.get(target_port.node_id)
        if source_proxy is None or target_proxy is None:
            return False
        return _types_compatible(source_proxy.output_type, target_proxy.input_type)

    def _cancel_pending_connection(self) -> None:
        if self._pending_source is not None:
            self._pending_source.set_active(False)
        if self._highlighted_input_port is not None:
            self._highlighted_input_port.set_drop_state(None)
        if self._pending_wire is not None:
            self.scene.removeItem(self._pending_wire)
        self._pending_source = None
        self._pending_wire = None
        self._highlighted_input_port = None
        self._connection_dragging = False

    def _connection_exists(self, source_id: str, target_id: str) -> bool:
        return any(
            item.source_id == source_id and item.target_id == target_id
            for item in self._connections
        )


def _wire_path(start: QPointF, end: QPointF) -> QPainterPath:
    dx = max(80.0, abs(end.x() - start.x()) * 0.5)
    path = QPainterPath(start)
    path.cubicTo(
        QPointF(start.x() + dx, start.y()),
        QPointF(end.x() - dx, end.y()),
        end,
    )
    return path


def _types_compatible(output_type: str, input_type: str | None) -> bool:
    if input_type is None or input_type == "any" or output_type == "any":
        return True
    if input_type == "array":
        return output_type in {"array", "image", "mask"}
    return output_type == input_type


def _point_from_event(event) -> QPoint:
    pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
    return pos.toPoint() if hasattr(pos, "toPoint") else pos


def _view_for_scene(scene) -> PipelineGraphView | None:
    if scene is None or not scene.views():
        return None
    view = scene.views()[0]
    return view if isinstance(view, PipelineGraphView) else None


def _exec_menu(menu: QMenu, pos):
    if hasattr(menu, "exec"):
        return menu.exec(pos)
    return menu.exec_(pos)
