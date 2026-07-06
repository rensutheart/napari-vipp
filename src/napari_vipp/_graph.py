"""Qt graph canvas and node card widgets for the VIPP prototype."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from qtpy.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QTransform,
)
from qtpy.QtWidgets import (
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QMenu,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from napari_vipp._theme import category_color, category_tint

OPERATION_MIME = "application/x-napari-vipp-operation"
PINNABLE_OUTPUT_TYPES = {"array", "image", "mask", "labels"}


class ClickablePreview(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class ProcessingBadge(QWidget):
    """Raised busy indicator that stays visible above card child widgets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._angle = 0
        self._queued = False
        self.setFixedSize(30, 30)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.hide()

    def set_queued(self, queued: bool) -> None:
        self._queued = queued
        self.update()

    def set_angle(self, angle: int) -> None:
        self._angle = int(angle) % 360
        self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(15, 23, 42, 225))
        painter.setPen(QPen(QColor("#475569"), 1.1))
        painter.drawRoundedRect(QRectF(1.0, 1.0, 28.0, 28.0), 8.0, 8.0)

        color = QColor("#f59e0b" if self._queued else "#93c5fd")
        painter.setPen(QPen(color, 2.4))
        painter.drawArc(QRectF(7.0, 7.0, 16.0, 16.0), self._angle * 16, 285 * 16)
        if self._queued:
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(15.0, 15.0), 2.4, 2.4)


class NodeCard(QFrame):
    """Small embedded node UI with a thumbnail and graph actions."""

    selected = Signal(str)
    pin_requested = Signal(str)
    calculate_requested = Signal(str)

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
        self._processing = False
        self._processing_queued = False
        self._processing_angle = 0
        self._manual_execution = False
        self._execution_state = "not_calculated"
        self._execution_message = ""
        self._auto_recalculate = False
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
        self.execution_label = QLabel("")
        self.execution_label.setWordWrap(True)
        self.execution_label.setStyleSheet(
            "color: #fbbf24; font-size: 10px; padding-top: 1px;"
        )
        self.calculate_button = QPushButton("Calculate", self)
        self.calculate_button.clicked.connect(
            lambda: self.calculate_requested.emit(self.node_id)
        )
        self.calculate_button.setVisible(False)
        self.pin_button = QPushButton("Pin", self)
        self.pin_button.clicked.connect(lambda: self.pin_requested.emit(self.node_id))
        self.pin_button.setVisible(False)
        self.processing_badge = ProcessingBadge(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(self.accent_bar)
        layout.addWidget(self.category_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.preview)
        layout.addWidget(self.metadata_label)
        layout.addWidget(self.execution_label)
        layout.addWidget(self.calculate_button)
        self._refresh_style()

    def mousePressEvent(self, event):  # noqa: N802
        self.selected.emit(self.node_id)
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._position_processing_badge()

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

    def set_processing(self, processing: bool, *, queued: bool = False) -> None:
        self._processing = processing
        self._processing_queued = queued if processing else False
        self.processing_badge.set_queued(self._processing_queued)
        self.processing_badge.setVisible(processing)
        if processing:
            self.setToolTip(
                "Processing in background; latest edit is queued."
                if queued
                else "Processing in background."
            )
        else:
            self.setToolTip("")
        self._position_processing_badge()
        self._refresh_style()
        self.update()

    def set_execution_state(
        self,
        state: str,
        *,
        manual: bool,
        message: str = "",
        auto_recalculate: bool = False,
    ) -> None:
        self._manual_execution = bool(manual)
        self._execution_state = str(state)
        self._execution_message = str(message or "")
        self._auto_recalculate = bool(auto_recalculate)
        self.calculate_button.setVisible(
            self._manual_execution and not self._auto_recalculate
        )
        self.calculate_button.setText(
            "Calculate"
            if self._execution_state == "not_calculated"
            else "Recalculate"
        )
        if not self._manual_execution:
            self.execution_label.setVisible(False)
            self.execution_label.setText("")
        else:
            self.execution_label.setVisible(True)
            self.execution_label.setText(self._execution_summary())
        self._refresh_style()
        self.update()

    def is_processing(self) -> bool:
        return self._processing

    def advance_processing_spinner(self) -> None:
        if not self._processing:
            return
        self._processing_angle = (self._processing_angle - 32) % 360
        self.processing_badge.set_angle(self._processing_angle)
        self.update()

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
        border_width = 2
        background = "#20242b"
        if self._selected:
            border = "#60a5fa"
        if self._pinned:
            border = "#facc15"
            border_width = 4
            background = "#2a271b"
        if self._manual_execution:
            if self._execution_state == "ready":
                border = "#22c55e"
                background = "#182a20"
            elif self._execution_state == "stale":
                border = "#f59e0b"
                background = "#2a2416"
            elif self._execution_state == "not_calculated":
                border = "#64748b"
                background = "#242932"
            elif self._execution_state == "error":
                border = "#ef4444"
                background = "#2f1d1d"
            if self._selected:
                border_width = 3
            if self._pinned:
                border = "#facc15"
                border_width = 4
        if self._processing:
            background = "#303640"
            if not self._pinned and not self._selected:
                border = "#94a3b8"
            if self._processing_queued and not self._pinned:
                border = "#f59e0b"
        accent_color = self._category_color
        category_background = self._category_tint
        category_color = self._category_color
        if self._manual_execution:
            if self._execution_state == "ready":
                accent_color = "#22c55e"
                category_background = "#064e3b"
                category_color = "#bbf7d0"
            elif self._execution_state == "stale":
                accent_color = "#f59e0b"
                category_background = "#78350f"
                category_color = "#fde68a"
            elif self._execution_state == "not_calculated":
                accent_color = "#94a3b8"
                category_background = "#334155"
                category_color = "#cbd5e1"
            elif self._execution_state == "error":
                accent_color = "#ef4444"
                category_background = "#7f1d1d"
                category_color = "#fecaca"
        if self._processing:
            accent_color = "#94a3b8"
            category_background = "#3a414c"
            category_color = "#d1d5db"
        self.setStyleSheet(
            f"""
            QFrame#NodeCard {{
                background: {background};
                border: {border_width}px solid {border};
                border-radius: 6px;
            }}
            QLabel {{
                color: #f3f4f6;
            }}
            QFrame#NodeAccent {{
                background: {accent_color};
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
                background: {category_background};
                color: {category_color};
                border-radius: 4px;
                font-size: 10px;
                font-weight: 650;
                padding: 2px 5px;
            }}
            """
        )
        self.pin_button.setVisible(False)
        self.processing_badge.raise_()

    def _execution_summary(self) -> str:
        if self._execution_state == "ready":
            return (
                "Auto result ready"
                if self._auto_recalculate
                else "Cached result ready"
            )
        if self._execution_state == "running":
            return "Calculating..."
        if self._execution_state == "stale":
            return (
                "Auto recalculation pending"
                if self._auto_recalculate
                else "Stale cached result"
            )
        if self._execution_state == "error":
            return self._execution_message or "Calculation failed"
        return "Not calculated"

    def _position_processing_badge(self) -> None:
        badge_size = self.processing_badge.size()
        preview_rect = self.preview.geometry()
        if self.preview.isVisible() and preview_rect.isValid():
            x = preview_rect.right() - badge_size.width() - 8
            y = preview_rect.top() + 8
        else:
            x = self.width() - badge_size.width() - 10
            y = 10
        self.processing_badge.move(max(0, x), max(0, y))
        self.processing_badge.raise_()


class TunnelBadgeItem(QGraphicsItem):
    """Compact schematic-style badge for named graph tunnels."""

    _connector_length = 14.0
    _tip_width = 12.0
    _minimum_body_width = 42.0
    _padding_x = 9.0
    _height = 22.0
    _margin = 1.0

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self._label = ""
        self._highlight_role = ""
        self._font = QFont()
        self._font.setPointSizeF(8.5)
        self.setZValue(36)
        self.setAcceptedMouseButtons(Qt.NoButton)
        self.hide()

    def set_label(self, label: str) -> None:
        cleaned = str(label or "").strip()
        if cleaned == self._label:
            return
        self.prepareGeometryChange()
        self._label = cleaned
        self.update()

    def set_highlight_role(self, role: str) -> None:
        role = role if role in {"source", "subscriber", "dimmed"} else ""
        if role == self._highlight_role:
            return
        self._highlight_role = role
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802
        if not self._label:
            return QRectF()
        width = self._connector_length + self._tag_width() + self._margin * 2.0
        height = self._tag_height() + self._margin * 2.0
        return QRectF(0.0, 0.0, width, height)

    def paint(self, painter, option, widget=None):  # noqa: N802
        if not self._label:
            return
        rect = self.boundingRect()
        tag_width = self._tag_width()
        tag_height = self._tag_height()
        tag_y = rect.center().y() - tag_height / 2.0
        wire_y = rect.center().y()

        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(self._font)
        pen_color = QColor("#93c5fd")
        fill_color = QColor(15, 23, 42, 210)
        if self._highlight_role == "source":
            pen_color = QColor("#fbbf24")
            fill_color = QColor(69, 42, 8, 230)
        elif self._highlight_role == "subscriber":
            pen_color = QColor("#60a5fa")
            fill_color = QColor(18, 43, 84, 230)
        elif self._highlight_role == "dimmed":
            pen_color = QColor("#64748b")
            fill_color = QColor(15, 23, 42, 150)
        pen_width = 1.3
        if self._highlight_role in {"source", "subscriber"}:
            pen_width = 1.7
        elif self._highlight_role == "dimmed":
            pen_width = 1.1
        painter.setPen(QPen(pen_color, pen_width))
        painter.setBrush(fill_color)

        if self.kind == "output":
            wire_start = self._margin
            tag_x = self._margin + self._connector_length
            painter.drawLine(QPointF(wire_start, wire_y), QPointF(tag_x, wire_y))
            self._draw_net_port(painter, tag_x, tag_y, tag_width, tag_height)
            self._draw_label(painter, tag_x, tag_y, tag_width, tag_height)
            return

        tag_x = self._margin
        wire_start = tag_x + tag_width
        wire_end = wire_start + self._connector_length
        self._draw_net_port(painter, tag_x, tag_y, tag_width, tag_height)
        painter.drawLine(QPointF(wire_start, wire_y), QPointF(wire_end, wire_y))
        self._draw_label(painter, tag_x, tag_y, tag_width, tag_height)

    def _tag_width(self) -> float:
        metrics = QFontMetricsF(self._font)
        body_width = max(
            self._minimum_body_width,
            metrics.horizontalAdvance(self._label) + self._padding_x * 2.0,
        )
        return body_width + self._tip_width

    def _tag_height(self) -> float:
        metrics = QFontMetricsF(self._font)
        return max(self._height, metrics.height() + 5.0)

    def _draw_net_port(
        self,
        painter: QPainter,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        body_width = width - self._tip_width
        mid_y = y + height / 2.0
        path = QPainterPath()
        path.moveTo(x, y)
        path.lineTo(x + body_width, y)
        path.lineTo(x + width, mid_y)
        path.lineTo(x + body_width, y + height)
        path.lineTo(x, y + height)
        path.closeSubpath()
        painter.drawPath(path)

    def _draw_label(
        self,
        painter: QPainter,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> None:
        body_width = width - self._tip_width
        text_rect = QRectF(x + 1.0, y, body_width - 2.0, height)
        text_color = QColor("#dbeafe")
        if self._highlight_role == "source":
            text_color = QColor("#fef3c7")
        elif self._highlight_role == "subscriber":
            text_color = QColor("#eff6ff")
        elif self._highlight_role == "dimmed":
            text_color = QColor("#94a3b8")
        painter.setPen(text_color)
        painter.drawText(text_rect, Qt.AlignCenter, self._label)
        if self._highlight_role == "source":
            painter.setPen(QPen(QColor("#fbbf24"), 1.7))
        elif self._highlight_role == "subscriber":
            painter.setPen(QPen(QColor("#60a5fa"), 1.7))
        elif self._highlight_role == "dimmed":
            painter.setPen(QPen(QColor("#64748b"), 1.2))
        else:
            painter.setPen(QPen(QColor("#93c5fd"), 1.3))


class PortItem(QGraphicsEllipseItem):
    """Clickable node port used for graph connections."""

    radius = 6.0
    hover_radius = 8.0
    target_radius = 10.0

    def __init__(
        self,
        node_id: str,
        kind: str,
        data_type: str,
        parent,
        *,
        port_index: int = 0,
        label: str = "",
        accent_color: str | None = None,
    ):
        super().__init__(-self.radius, -self.radius, 2 * self.radius, 2 * self.radius)
        self.node_id = node_id
        self.kind = kind
        self.data_type = data_type
        self.port_index = int(port_index)
        self.label = label
        self.accent_color = accent_color
        self._tunnel_label = ""
        self.setParentItem(parent)
        self.setZValue(30)
        self.setCursor(Qt.CrossCursor)
        self.setAcceptHoverEvents(True)
        self._update_tooltip()
        self._hovered = False
        self._active = False
        self._drop_state: str | None = None
        self._tunnel_highlight_role = ""
        self._tunnel_badge = TunnelBadgeItem(self.kind, self)
        self._refresh_style()

    def set_data_type(self, data_type: str) -> None:
        self.data_type = data_type
        self._update_tooltip()
        self._refresh_style()

    def set_label(self, label: str, accent_color: str | None = None) -> None:
        self.label = label
        self.accent_color = accent_color
        self._update_tooltip()
        self._refresh_style()

    def set_active(self, active: bool) -> None:
        self._active = active
        self._refresh_style()

    def set_drop_state(self, state: str | None) -> None:
        self._drop_state = state
        self._refresh_style()

    def set_tunnel_label(self, label: str) -> None:
        self._tunnel_label = str(label or "").strip()
        if not self._tunnel_label:
            self.set_tunnel_highlight_role("")
            self._tunnel_badge.set_label("")
            self._tunnel_badge.hide()
            self._update_tooltip()
            return
        self._tunnel_badge.set_label(self._tunnel_label)
        self._position_tunnel_badge()
        self._tunnel_badge.show()
        self._update_tooltip()

    def set_tunnel_highlight_role(self, role: str) -> None:
        role = role if role in {"source", "subscriber", "dimmed"} else ""
        if role == self._tunnel_highlight_role:
            return
        self._tunnel_highlight_role = role
        self._tunnel_badge.set_highlight_role(role)
        self.setOpacity(0.34 if role == "dimmed" else 1.0)
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
        elif self.data_type == "labels":
            color = "#f472b6"
        elif self.data_type == "mask_or_labels":
            color = "#f472b6"
        elif self.data_type == "table":
            color = "#facc15"
        elif self.data_type == "array":
            color = "#38bdf8"
        elif self.data_type == "any":
            color = "#f59e0b"
        if self.accent_color:
            color = self.accent_color
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
        if self._tunnel_highlight_role == "dimmed":
            pen_color = "#475569"
            pen_width = 1.4
        elif self._tunnel_highlight_role:
            radius = max(radius, self.hover_radius)
            pen_color = (
                "#fbbf24"
                if self._tunnel_highlight_role == "source"
                else "#93c5fd"
            )
            pen_width = 3.0
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
        if self._tunnel_label:
            self._position_tunnel_badge(radius)

    def _update_tooltip(self) -> None:
        name = self.kind
        if self.kind == "input":
            name = f"input {self.port_index + 1}"
        elif self.kind == "output" and self.label:
            name = "output"
        if self.label:
            name = f"{name}: {self.label}"
        tunnel = f"\nTunnel: {self._tunnel_label}" if self._tunnel_label else ""
        self.setToolTip(f"{name} ({self.data_type}){tunnel}")

    def _position_tunnel_badge(self, port_radius: float | None = None) -> None:
        rect = self._tunnel_badge.boundingRect()
        if port_radius is None:
            port_radius = max(self.rect().width(), self.rect().height()) / 2.0
        y = -rect.height() / 2.0
        if self.kind == "output":
            x = port_radius + 1.0
        else:
            x = -rect.width() - port_radius - 1.0
        self._tunnel_badge.setPos(x, y)


class NodeProxy(QGraphicsProxyWidget):
    """Movable graphics item that keeps connected wires attached."""

    DRAG_OPACITY = 0.62

    def __init__(
        self,
        node_id: str,
        operation_id: str,
        input_type: str | None,
        output_type: str,
        has_input: bool,
        has_output: bool = True,
    ):
        super().__init__()
        self.node_id = node_id
        self.operation_id = operation_id
        self.input_type = input_type
        self.output_type = output_type
        self.connections: list[ConnectionItem] = []
        self.input_ports: list[PortItem] = []
        self._has_input = has_input
        self._has_output = has_output
        self._input_port_count = 1
        self._input_port_labels: list[str] = []
        self._input_port_colors: list[str | None] = []
        self._input_port_types: list[str] = []
        self._output_port_count = 1
        self._output_port_labels: list[str] = []
        self._output_port_colors: list[str | None] = []
        self._output_port_types: list[str] = []
        self.output_ports: list[PortItem] = []
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
            self._ensure_input_ports()
            top = rect.top() + 42
            bottom = rect.bottom() - 42
            if bottom <= top:
                top = rect.top()
                bottom = rect.bottom()
            for index, port in enumerate(self.input_ports):
                if len(self.input_ports) == 1:
                    y = rect.center().y()
                else:
                    step = (bottom - top) / max(len(self.input_ports) - 1, 1)
                    y = top + step * index
                port.setPos(rect.left(), y)
        if self._has_output:
            self._ensure_output_ports()
            top = rect.top() + 42
            bottom = rect.bottom() - 42
            if bottom <= top:
                top = rect.top()
                bottom = rect.bottom()
            for index, port in enumerate(self.output_ports):
                if len(self.output_ports) == 1:
                    y = rect.center().y()
                else:
                    step = (bottom - top) / max(len(self.output_ports) - 1, 1)
                    y = top + step * index
                port.setPos(rect.right(), y)

    @property
    def input_port(self) -> PortItem | None:
        return self.input_ports[0] if self.input_ports else None

    @property
    def output_port(self) -> PortItem | None:
        return self.output_ports[0] if self.output_ports else None

    def set_input_ports(
        self,
        count: int,
        labels: list[str] | None = None,
        colors: list[str | None] | None = None,
        data_types: list[str] | None = None,
    ) -> None:
        self._input_port_count = max(int(count), 1)
        self._input_port_labels = labels or []
        self._input_port_colors = colors or []
        self._input_port_types = data_types or []
        self._ensure_input_ports()
        self.refresh_ports()
        for connection in self.connections:
            connection.update_path()

    def set_output_ports(
        self,
        count: int,
        labels: list[str] | None = None,
        colors: list[str | None] | None = None,
        data_types: list[str] | None = None,
    ) -> None:
        self._output_port_count = max(int(count), 1)
        self._output_port_labels = labels or []
        self._output_port_colors = colors or []
        self._output_port_types = data_types or []
        self._ensure_output_ports()
        self.refresh_ports()
        for connection in self.connections:
            connection.update_path()

    def set_output_type(self, output_type: str) -> None:
        self.output_type = output_type
        if self.output_ports and self._output_port_count == 1:
            self.output_ports[0].set_data_type(output_type)

    def port_scene_pos(self, kind: str, port_index: int = 0) -> QPointF:
        if kind == "output":
            port = self.output_port_at(port_index)
        else:
            port = self.input_port_at(port_index)
        if port is not None:
            return port.mapToScene(QPointF(0, 0))
        rect = self.sceneBoundingRect()
        if kind == "output":
            return QPointF(rect.right(), rect.center().y())
        return QPointF(rect.left(), rect.center().y())

    def input_port_at(self, port_index: int) -> PortItem | None:
        if not self.input_ports:
            return None
        port_index = int(np.clip(port_index, 0, len(self.input_ports) - 1))
        return self.input_ports[port_index]

    def output_port_at(self, port_index: int) -> PortItem | None:
        if not self.output_ports:
            return None
        port_index = int(np.clip(port_index, 0, len(self.output_ports) - 1))
        return self.output_ports[port_index]

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
                self.setOpacity(self.DRAG_OPACITY)
                self.setPos(self._drag_start_pos + delta)
                view = _view_for_scene(self.scene())
                if view is not None:
                    view.update_existing_node_insert_preview(
                        self.node_id,
                        event.scenePos(),
                    )
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag_start_scene is not None and event.button() == Qt.LeftButton:
            start_pos = self._drag_start_pos
            end_pos = QPointF(self.pos())
            moved = (
                self._dragging
                and start_pos is not None
                and (
                    abs(end_pos.x() - start_pos.x())
                    + abs(end_pos.y() - start_pos.y())
                )
                > 0.001
            )
            card = self._card()
            if card is not None:
                card.setCursor(Qt.OpenHandCursor)
            self.setOpacity(1.0)
            self._drag_start_scene = None
            self._drag_start_pos = None
            self._dragging = False
            self._press_was_preview = False
            if moved:
                view = _view_for_scene(self.scene())
                if view is not None:
                    was_loose = not self.connections
                    inserted = view.release_existing_node_insert(
                        self.node_id,
                        start_pos,
                        end_pos,
                        event.scenePos(),
                    )
                    if not inserted:
                        if was_loose:
                            view.finish_loose_node_drag(
                                self.node_id,
                                start_pos,
                                end_pos,
                            )
                        view.node_moved.emit(self.node_id, start_pos, end_pos)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def itemChange(self, change, value):  # noqa: N802
        result = super().itemChange(change, value)
        if change in (
            QGraphicsItem.ItemPositionHasChanged,
            QGraphicsItem.ItemTransformHasChanged,
        ):
            view = _view_for_scene(self.scene())
            if view is not None:
                view._mark_graph_geometry_changed()
                for connection in self.connections:
                    connection.update_path()
                if not (self._dragging and not self.connections):
                    view.reroute_connections(affected_rect=self.sceneBoundingRect())
                view._ensure_scene_space_for_rect(self.sceneBoundingRect())
            else:
                for connection in self.connections:
                    connection.update_path()
        return result

    def _card(self) -> NodeCard | None:
        widget = self.widget()
        return widget if isinstance(widget, NodeCard) else None

    def _ensure_input_ports(self) -> None:
        if not self._has_input or self.input_type is None:
            return
        while len(self.input_ports) < self._input_port_count:
            index = len(self.input_ports)
            self.input_ports.append(
                PortItem(
                    self.node_id,
                    "input",
                    self.input_type,
                    self,
                    port_index=index,
                )
            )
        while len(self.input_ports) > self._input_port_count:
            port = self.input_ports.pop()
            if port.scene() is not None:
                port.scene().removeItem(port)
            port.setParentItem(None)
        for index, port in enumerate(self.input_ports):
            label = (
                self._input_port_labels[index]
                if index < len(self._input_port_labels)
                else f"Input {index + 1}"
            )
            color = (
                self._input_port_colors[index]
                if index < len(self._input_port_colors)
                else None
            )
            data_type = (
                self._input_port_types[index]
                if index < len(self._input_port_types)
                else self.input_type
            )
            port.port_index = index
            port.set_data_type(data_type)
            port.set_label(label, color)

    def _ensure_output_ports(self) -> None:
        if not self._has_output:
            return
        while len(self.output_ports) < self._output_port_count:
            index = len(self.output_ports)
            self.output_ports.append(
                PortItem(
                    self.node_id,
                    "output",
                    self.output_type,
                    self,
                    port_index=index,
                )
            )
        while len(self.output_ports) > self._output_port_count:
            port = self.output_ports.pop()
            if port.scene() is not None:
                port.scene().removeItem(port)
            port.setParentItem(None)
        for index, port in enumerate(self.output_ports):
            label = (
                self._output_port_labels[index]
                if index < len(self._output_port_labels)
                else ""
            )
            color = (
                self._output_port_colors[index]
                if index < len(self._output_port_colors)
                else None
            )
            data_type = (
                self._output_port_types[index]
                if index < len(self._output_port_types)
                else self.output_type
            )
            port.port_index = index
            port.set_data_type(data_type)
            port.set_label(label, color)

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
    HIT_WIDTH = 18.0
    PREVIEW_STATES = {"full", "partial", "place", "incompatible"}

    def __init__(
        self,
        source: NodeProxy,
        target: NodeProxy,
        target_port: int = 0,
        source_port: int = 0,
    ):
        super().__init__()
        self.source = source
        self.target = target
        self.source_id = source.node_id
        self.target_id = target.node_id
        self.target_port = int(target_port)
        self.source_port = int(source_port)
        self._insert_preview_state: str | None = None
        self._pulse_phase = 0
        self._last_route_key: tuple[float, float, float, float, int] | None = None
        self.setZValue(-10)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self._refresh_pen()
        self.update_path()

    def update_path(self) -> None:
        start = self.source.port_scene_pos("output", self.source_port)
        end = self.target.port_scene_pos("input", self.target_port)
        view = _view_for_scene(self.scene())
        revision = int(view.route_revision) if view is not None else -1
        route_key = (
            round(start.x(), 3),
            round(start.y(), 3),
            round(end.x(), 3),
            round(end.y(), 3),
            revision,
        )
        if route_key == self._last_route_key:
            return
        obstacles = (
            view.connection_obstacle_rects(self)
            if view is not None
            else ()
        )
        self.setPath(_wire_path(start, end, obstacles=obstacles))
        self._last_route_key = route_key

    def shape(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(self.HIT_WIDTH)
        stroker.setCapStyle(Qt.RoundCap)
        stroker.setJoinStyle(Qt.RoundJoin)
        return stroker.createStroke(self.path())

    def boundingRect(self) -> QRectF:  # noqa: N802
        pad = self.HIT_WIDTH / 2.0 + 8.0
        return super().boundingRect().adjusted(-pad, -pad, pad, pad)

    def set_insert_preview_state(self, state: str | None) -> None:
        if state not in self.PREVIEW_STATES:
            state = None
        if state == self._insert_preview_state:
            return
        self.prepareGeometryChange()
        self._insert_preview_state = state
        self._pulse_phase = 0
        self._refresh_pen()
        self.update()

    def advance_insert_preview_pulse(self) -> None:
        if self._insert_preview_state is None:
            return
        self._pulse_phase = (self._pulse_phase + 1) % 24
        self._refresh_pen()
        self.update()

    def itemChange(self, change, value):  # noqa: N802
        result = super().itemChange(change, value)
        if change == QGraphicsItem.ItemSelectedHasChanged:
            self._refresh_pen()
        return result

    def contextMenuEvent(self, event):  # noqa: N802
        view = _view_for_scene(self.scene())
        menu = QMenu()
        info_action = menu.addAction("Info")
        insert_action = menu.addAction("Insert node here...")
        delete_action = menu.addAction("Delete")
        action = _exec_menu(menu, event.screenPos())
        if view is not None and action == delete_action:
            view.delete_connection_item(self, notify=True)
        elif view is not None and action == insert_action:
            view.connection_insert_requested.emit(
                (
                    self.source_id,
                    self.target_id,
                    self.target_port,
                    self.source_port,
                ),
                event.scenePos(),
            )
        elif view is not None and action == info_action:
            source_port = self.source.output_port_at(self.source_port)
            target_port = self.target.input_port_at(self.target_port)
            source_type = (
                source_port.data_type
                if source_port is not None
                else self.source.output_type
            )
            target_type = (
                target_port.data_type
                if target_port is not None
                else self.target.input_type
            )
            view.status_message.emit(
                f"Connection {self.source_id} -> {self.target_id}: "
                f"{source_type} to {target_type} "
                f"input {self.target_port + 1}."
            )

    def _refresh_pen(self) -> None:
        color = "#facc15" if self.isSelected() else "#8aa0c8"
        width = 3.0 if self.isSelected() else 2.0
        style = Qt.SolidLine
        if self._insert_preview_state == "incompatible":
            color = "#fb7185"
            width = 4.0
            style = Qt.DashLine
        elif self._insert_preview_state == "full":
            color = "#22c55e"
            width = 4.0 + self._pulse_amount()
        elif self._insert_preview_state in {"partial", "place"}:
            color = "#38bdf8"
            width = 4.0 + self._pulse_amount()
            style = Qt.DashLine
        pen = QPen(QColor(color), width)
        pen.setStyle(style)
        pen.setCapStyle(Qt.RoundCap)
        pen.setJoinStyle(Qt.RoundJoin)
        self.setPen(pen)

    def _pulse_amount(self) -> float:
        distance = abs(12 - self._pulse_phase) / 12.0
        return 1.5 * (1.0 - distance)

    def paint(self, painter, option, widget=None):  # noqa: N802
        if self._insert_preview_state is not None:
            glow_color = QColor("#fb7185")
            if self._insert_preview_state == "full":
                glow_color = QColor("#22c55e")
            elif self._insert_preview_state in {"partial", "place"}:
                glow_color = QColor("#38bdf8")
            alpha = 55 + int(45 * self._pulse_amount())
            glow_color.setAlpha(alpha)
            glow = QPen(glow_color, 11.0 + self._pulse_amount() * 3.0)
            glow.setCapStyle(Qt.RoundCap)
            glow.setJoinStyle(Qt.RoundJoin)
            painter.setPen(glow)
            painter.drawPath(self.path())
        super().paint(painter, option, widget)


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

    SLIDER_MIN_ZOOM = 40
    SLIDER_MAX_ZOOM = 250
    DEFAULT_ZOOM = 100
    WHEEL_MIN_ZOOM = 20
    WHEEL_MAX_ZOOM = 400
    SCENE_EDGE_MARGIN = 260.0
    SCENE_EXPAND_STEP_X = 1400.0
    SCENE_EXPAND_STEP_Y = 1100.0
    WIRE_OBSTACLE_MARGIN = 24.0

    node_selected = Signal(str)
    node_delete_requested = Signal(str)
    node_duplicate_requested = Signal(str)
    node_code_requested = Signal(str)
    node_moved = Signal(str, object, object)
    node_splice_requested = Signal(str, object, object, object)
    pin_requested = Signal(str)
    node_calculate_requested = Signal(str)
    node_create_requested = Signal(str, QPointF)
    node_insert_requested = Signal(str, object, QPointF)
    connection_insert_requested = Signal(object, QPointF)
    connection_requested = Signal(str, str, int, int)
    connection_removed = Signal(str, str, int)
    port_context_requested = Signal(str, str, int, object)
    tunnel_selected = Signal(str)
    status_message = Signal(str)
    zoom_changed = Signal(float)

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
        self._highlighted_connection: ConnectionItem | None = None
        self._highlighted_connection_state: str | None = None
        self._highlighted_connection_operation: str | None = None
        self._connection_insert_validator: Callable[
            [str, tuple[str, str, int, int]],
            tuple[str, str],
        ] | None = None
        self._connection_dragging = False
        self._panning = False
        self._pan_start = QPoint()
        self._pan_h_value = 0
        self._pan_v_value = 0
        self._base_transform = QTransform()
        self._zoom_percent = float(self.DEFAULT_ZOOM)
        self._rerouting_connections = False
        self._route_revision = 0
        self._processing_timer = QTimer(self)
        self._processing_timer.setInterval(80)
        self._processing_timer.timeout.connect(self._advance_processing_spinners)
        self._connection_pulse_timer = QTimer(self)
        self._connection_pulse_timer.setInterval(80)
        self._connection_pulse_timer.timeout.connect(
            self._advance_connection_insert_pulse
        )
        self._tunnel_source_ports: dict[str, PortItem] = {}
        self._tunnel_subscriber_ports: dict[str, list[PortItem]] = {}
        self._active_tunnel_name = ""
        self._hover_tunnel_name = ""

    def build_graph(
        self,
        nodes,
        connections,
        positions=None,
        *,
        output_tunnels=(),
        preserve_view: bool = False,
    ) -> None:
        preserved_center = self.mapToScene(self.viewport().rect().center())
        preserved_transform = QTransform(self.transform())
        preserved_base_transform = QTransform(self._base_transform)
        preserved_zoom = float(self._zoom_percent)
        self.clear_node_processing()
        self.scene.clear()
        self._proxies.clear()
        self._cards.clear()
        self._connections.clear()
        self._pending_source = None
        self._pending_wire = None
        self._highlighted_input_port = None
        self._clear_connection_insert_preview()
        self._tunnel_source_ports.clear()
        self._tunnel_subscriber_ports.clear()
        self._hover_tunnel_name = ""

        default_positions = {
            "input": QPointF(0, 20),
            "gaussian": QPointF(330, 20),
            "threshold": QPointF(660, 20),
        }
        if positions is None:
            positions = default_positions
        for index, node in enumerate(nodes):
            fallback = QPointF(330 * index, 20)
            point = _to_pointf(positions.get(node.id)) or fallback
            self.add_node(node, point)

        for connection in connections:
            if getattr(connection, "tunnel_name", ""):
                continue
            self.add_connection(
                connection.source_id,
                connection.target_id,
                connection.target_port,
                getattr(connection, "source_port", 0),
            )
        self.set_port_tunnels(output_tunnels, connections)

        graph_rect = self.scene.itemsBoundingRect()
        scene_rect = graph_rect.adjusted(-1600, -1200, 1800, 1200)
        if preserve_view:
            center_rect = QRectF(preserved_center, preserved_center).adjusted(
                -120,
                -120,
                120,
                120,
            )
            scene_rect = scene_rect.united(center_rect)
        self.scene.setSceneRect(scene_rect)
        self._mark_graph_geometry_changed()
        if preserve_view:
            self.setTransform(preserved_transform)
            self._base_transform = preserved_base_transform
            self._zoom_percent = preserved_zoom
            self.centerOn(preserved_center)
            self.reroute_connections()
            return
        self.resetTransform()
        self._base_transform = QTransform()
        self._zoom_percent = float(self.DEFAULT_ZOOM)
        self._apply_zoom_from_base(graph_rect.center())
        self.reroute_connections()
        self.zoom_changed.emit(self._zoom_percent)

    @property
    def zoom_percent(self) -> float:
        return float(self._zoom_percent)

    @property
    def route_revision(self) -> int:
        return int(self._route_revision)

    def _mark_graph_geometry_changed(self) -> None:
        self._route_revision += 1

    def set_zoom_percent(self, value: float) -> None:
        zoom = float(np.clip(value, self.WHEEL_MIN_ZOOM, self.WHEEL_MAX_ZOOM))
        if abs(zoom - self._zoom_percent) < 0.001:
            return
        self._zoom_percent = zoom
        self._apply_zoom_from_base()
        self.zoom_changed.emit(self._zoom_percent)

    def reset_zoom(self) -> None:
        self.set_zoom_percent(float(self.DEFAULT_ZOOM))

    def _apply_zoom_from_base(self, center: QPointF | None = None) -> None:
        if center is None:
            center = self.mapToScene(self.viewport().rect().center())
        self.setTransform(QTransform(self._base_transform))
        factor = self._zoom_percent / float(self.DEFAULT_ZOOM)
        self.scale(factor, factor)
        self.centerOn(center)

    def build_demo_graph(self, nodes) -> None:
        self.build_graph(nodes, [])

    def node_positions(self) -> dict[str, tuple[float, float]]:
        """Return the current scene position of each node proxy by id."""
        positions: dict[str, tuple[float, float]] = {}
        for node_id, proxy in self._proxies.items():
            pos = proxy.pos()
            positions[node_id] = (float(pos.x()), float(pos.y()))
        return positions

    def node_position(self, node_id: str) -> QPointF | None:
        proxy = self._proxies.get(node_id)
        return QPointF(proxy.pos()) if proxy is not None else None

    def node_card_sizes(self) -> dict[str, tuple[float, float]]:
        """Return current node-card sizes in scene units."""
        sizes: dict[str, tuple[float, float]] = {}
        for node_id, proxy in self._proxies.items():
            rect = proxy.sceneBoundingRect()
            sizes[node_id] = (float(rect.width()), float(rect.height()))
        return sizes

    def node_scene_rect(self, node_id: str) -> QRectF | None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return None
        return proxy.sceneBoundingRect()

    def apply_node_positions(
        self,
        positions: Mapping[str, tuple[float, float] | QPointF],
        *,
        animate: bool = False,
    ) -> bool:
        """Move existing nodes to absolute scene positions.

        ``animate`` is accepted for the future live-layout path; phase 1 applies
        positions immediately.
        """
        del animate
        moved_rect: QRectF | None = None
        changed = False
        for node_id, value in positions.items():
            proxy = self._proxies.get(node_id)
            point = _to_pointf(value)
            if proxy is None or point is None:
                continue
            if _points_close(proxy.pos(), point):
                continue
            before = proxy.sceneBoundingRect()
            proxy.setPos(point)
            after = proxy.sceneBoundingRect()
            combined = before.united(after)
            moved_rect = combined if moved_rect is None else moved_rect.united(combined)
            changed = True

        if not changed:
            return False
        if moved_rect is not None and moved_rect.isValid():
            self._ensure_scene_space_for_rect(moved_rect)
        self._mark_graph_geometry_changed()
        self.reroute_connections()
        return True

    def center_node_on(self, node_id: str, scene_pos: QPointF) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        rect = proxy.sceneBoundingRect()
        proxy.setPos(proxy.pos() + (scene_pos - rect.center()))
        moved_rect = rect.united(proxy.sceneBoundingRect())
        self._ensure_scene_space_for_rect(proxy.sceneBoundingRect())
        self._mark_graph_geometry_changed()
        self.reroute_connections(affected_rect=moved_rect)

    def move_nodes_by(self, node_ids: set[str], delta: QPointF) -> None:
        if not node_ids or (abs(delta.x()) < 0.001 and abs(delta.y()) < 0.001):
            return
        moved_rect = QRectF()
        for node_id in node_ids:
            proxy = self._proxies.get(node_id)
            if proxy is None:
                continue
            proxy.setPos(proxy.pos() + delta)
            moved_rect = moved_rect.united(proxy.sceneBoundingRect())
        if moved_rect.isValid():
            self._ensure_scene_space_for_rect(moved_rect)
            self._mark_graph_geometry_changed()
            self.reroute_connections(affected_rect=moved_rect)

    def set_connection_insert_validator(
        self,
        validator: Callable[[str, tuple[str, str, int, int]], tuple[str, str]]
        | None,
    ) -> None:
        self._connection_insert_validator = validator

    def finish_loose_node_drag(
        self,
        node_id: str,
        old_pos: QPointF,
        new_pos: QPointF,
    ) -> None:
        """Reroute wires once a loose node has been dropped."""
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        local_rect = proxy.boundingRect()
        old_rect = QRectF(local_rect).translated(old_pos)
        new_rect = QRectF(local_rect).translated(new_pos)
        affected = old_rect.united(new_rect)
        self._mark_graph_geometry_changed()
        self.reroute_connections(affected_rect=affected)

    def update_existing_node_insert_preview(
        self,
        node_id: str,
        scene_pos: QPointF,
    ) -> None:
        """Preview wire insertion while a loose existing node is dragged."""
        proxy = self._proxies.get(node_id)
        if proxy is None or proxy.connections:
            self._clear_connection_insert_preview()
            return
        self._update_connection_insert_preview(proxy.operation_id, scene_pos)

    def release_existing_node_insert(
        self,
        node_id: str,
        old_pos: QPointF,
        new_pos: QPointF,
        scene_pos: QPointF,
    ) -> bool:
        """Emit a splice request if a loose node is dropped on a valid wire."""
        proxy = self._proxies.get(node_id)
        if proxy is None or proxy.connections:
            self._clear_connection_insert_preview()
            return False
        self._update_connection_insert_preview(proxy.operation_id, scene_pos)
        connection_key = self._connection_key(self._highlighted_connection)
        state = self._highlighted_connection_state
        if connection_key is None or state == "incompatible":
            self._clear_connection_insert_preview()
            return False

        self._clear_connection_insert_preview()
        self.node_splice_requested.emit(
            node_id,
            connection_key,
            QPointF(old_pos),
            QPointF(new_pos),
        )
        return True

    def connection_obstacle_rects(
        self,
        connection: ConnectionItem | None = None,
        *,
        exclude_node_ids: set[str] | None = None,
    ) -> tuple[QRectF, ...]:
        excluded = set(exclude_node_ids or set())
        if connection is not None:
            excluded.update({connection.source_id, connection.target_id})
        margin = float(self.WIRE_OBSTACLE_MARGIN)
        rects: list[QRectF] = []
        for node_id, proxy in self._proxies.items():
            if node_id in excluded:
                continue
            rect = proxy.sceneBoundingRect()
            if rect.isNull() or not rect.isValid():
                continue
            rects.append(rect.adjusted(-margin, -margin, margin, margin))
        return tuple(rects)

    def reroute_connections(self, affected_rect: QRectF | None = None) -> None:
        if self._rerouting_connections:
            return
        if affected_rect is not None:
            affected = QRectF(affected_rect)
            if not affected.isValid() or affected.isNull():
                return
            margin = float(self.WIRE_OBSTACLE_MARGIN) * 2.0
            affected = affected.adjusted(-margin, -margin, margin, margin)
            connections = [
                connection
                for connection in self._connections
                if self._connection_route_rect(connection).intersects(affected)
            ]
        else:
            connections = list(self._connections)
        if not connections:
            return
        self._rerouting_connections = True
        try:
            for connection in connections:
                connection.update_path()
        finally:
            self._rerouting_connections = False

    def _connection_route_rect(self, connection: ConnectionItem) -> QRectF:
        start = connection.source.port_scene_pos("output", connection.source_port)
        end = connection.target.port_scene_pos("input", connection.target_port)
        corridor = QRectF(start, end).normalized().adjusted(
            -180.0,
            -240.0,
            180.0,
            240.0,
        )
        return corridor.united(connection.sceneBoundingRect())

    def add_node(self, node, position: QPointF) -> None:
        card = NodeCard(
            node.id,
            node.title,
            node.category,
            can_pin=node.output_type in PINNABLE_OUTPUT_TYPES,
        )
        card.selected.connect(self._select_node)
        card.pin_requested.connect(self.pin_requested)
        card.calculate_requested.connect(self.node_calculate_requested)
        proxy = NodeProxy(
            node.id,
            node.operation_id,
            node.input_type,
            node.output_type,
            node.has_input,
            True,
        )
        proxy.setWidget(card)
        self.scene.addItem(proxy)
        proxy.setPos(position)
        proxy.set_input_ports(
            _node_input_port_count(node),
            _node_input_port_labels(node),
            _node_input_port_colors(node),
            _node_input_port_types(node),
        )
        proxy.refresh_ports()
        self._cards[node.id] = card
        self._proxies[node.id] = proxy
        self._ensure_scene_space_for_rect(proxy.sceneBoundingRect())
        self._mark_graph_geometry_changed()
        self.reroute_connections(affected_rect=proxy.sceneBoundingRect())

    def add_connection(
        self,
        source_id: str,
        target_id: str,
        target_port: int = 0,
        source_port: int = 0,
    ) -> None:
        if self._connection_exists(source_id, target_id, target_port, source_port):
            return
        source = self._proxies[source_id]
        target = self._proxies[target_id]
        item = ConnectionItem(source, target, target_port, source_port)
        self.scene.addItem(item)
        source.connections.append(item)
        target.connections.append(item)
        self._connections.append(item)
        item.update_path()

    def set_port_tunnels(self, output_tunnels=(), connections=()) -> None:
        self._tunnel_source_ports.clear()
        self._tunnel_subscriber_ports.clear()
        for proxy in self._proxies.values():
            for port in proxy.input_ports:
                port.set_tunnel_label("")
            for port in proxy.output_ports:
                port.set_tunnel_label("")

        for tunnel in output_tunnels or ():
            tunnel_name = str(getattr(tunnel, "name", "") or "").strip()
            if not tunnel_name:
                continue
            source = self._proxies.get(getattr(tunnel, "source_id", ""))
            if source is None:
                continue
            port = source.output_port_at(getattr(tunnel, "source_port", 0))
            if port is not None:
                port.set_tunnel_label(tunnel_name)
                self._tunnel_source_ports[tunnel_name] = port
                self._tunnel_subscriber_ports.setdefault(tunnel_name, [])

        for connection in connections or ():
            tunnel_name = str(getattr(connection, "tunnel_name", "") or "").strip()
            if not tunnel_name:
                continue
            target = self._proxies.get(getattr(connection, "target_id", ""))
            if target is None:
                continue
            port = target.input_port_at(getattr(connection, "target_port", 0))
            if port is not None:
                port.set_tunnel_label(tunnel_name)
                self._tunnel_subscriber_ports.setdefault(tunnel_name, []).append(port)
        self._apply_tunnel_highlight()

    def highlight_tunnel(self, name: str, *, sticky: bool = True) -> None:
        tunnel_name = str(name or "").strip()
        if tunnel_name and tunnel_name not in self._tunnel_source_ports:
            tunnel_name = ""
        if sticky:
            self._active_tunnel_name = tunnel_name
        else:
            self._hover_tunnel_name = tunnel_name
        self._apply_tunnel_highlight()
        if tunnel_name and sticky:
            self.tunnel_selected.emit(tunnel_name)

    def clear_tunnel_highlight(self, *, sticky: bool = True) -> None:
        if sticky:
            self._active_tunnel_name = ""
        else:
            self._hover_tunnel_name = ""
        self._apply_tunnel_highlight()

    def reveal_tunnel(self, name: str) -> None:
        tunnel_name = str(name or "").strip()
        self.highlight_tunnel(tunnel_name, sticky=True)
        ports = self._ports_for_tunnel(tunnel_name)
        if not ports:
            return
        rect = ports[0].sceneBoundingRect()
        for port in ports[1:]:
            rect = rect.united(port.sceneBoundingRect())
        for port in ports:
            parent = port.parentItem()
            if parent is not None:
                rect = rect.united(parent.sceneBoundingRect())
        self._ensure_scene_space_for_rect(rect.adjusted(-160, -120, 160, 120))
        self.centerOn(rect.center())

    def focus_node(self, node_id: str) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        self._select_node(node_id)
        self._ensure_scene_space_for_rect(proxy.sceneBoundingRect())
        self.centerOn(proxy.sceneBoundingRect().center())

    def _effective_tunnel_highlight(self) -> str:
        return self._active_tunnel_name or self._hover_tunnel_name

    def _ports_for_tunnel(self, name: str) -> list[PortItem]:
        ports: list[PortItem] = []
        source = self._tunnel_source_ports.get(name)
        if source is not None:
            ports.append(source)
        ports.extend(self._tunnel_subscriber_ports.get(name, ()))
        return ports

    def _apply_tunnel_highlight(self) -> None:
        active = self._effective_tunnel_highlight()
        if active and active not in self._tunnel_source_ports:
            self._active_tunnel_name = ""
            self._hover_tunnel_name = ""
            active = ""
        active_node_ids: set[str] = set()
        for name, source in self._tunnel_source_ports.items():
            role = ""
            if active:
                role = "source" if name == active else "dimmed"
            source.set_tunnel_highlight_role(role)
            if name == active:
                active_node_ids.add(source.node_id)
        for name, ports in self._tunnel_subscriber_ports.items():
            for port in ports:
                role = ""
                if active:
                    role = "subscriber" if name == active else "dimmed"
                port.set_tunnel_highlight_role(role)
                if name == active:
                    active_node_ids.add(port.node_id)
        for node_id, proxy in self._proxies.items():
            proxy.setOpacity(1.0 if not active or node_id in active_node_ids else 0.38)
        for connection in self._connections:
            connection.setOpacity(0.18 if active else 1.0)
        self.scene.update()

    def remove_node(self, node_id: str) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        affected_rect = proxy.sceneBoundingRect()
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
        self._mark_graph_geometry_changed()
        self.reroute_connections(affected_rect=affected_rect)

    def remove_connection(
        self,
        source_id: str,
        target_id: str,
        target_port: int | None = None,
        notify: bool = False,
    ) -> None:
        for item in list(self._connections):
            if (
                item.source_id == source_id
                and item.target_id == target_id
                and (
                    target_port is None
                    or item.target_port == int(target_port)
                )
            ):
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
            self.connection_removed.emit(
                item.source_id,
                item.target_id,
                item.target_port,
            )

    def set_thumbnail(self, node_id: str, thumbnail: np.ndarray | None) -> None:
        card = self._cards.get(node_id)
        if card is not None:
            card.set_thumbnail(thumbnail)
            card.update()
        proxy = self._proxies.get(node_id)
        if proxy is not None:
            proxy.update()
        if self.scene is not None:
            self.scene.update()

    def set_node_metadata(self, node_id: str, text: str) -> None:
        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None or proxy is None:
            return
        before = proxy.sceneBoundingRect()
        card.set_metadata_summary(text)
        card.adjustSize()
        proxy.refresh_ports()
        after = proxy.sceneBoundingRect()
        if _rect_changed(before, after):
            self._mark_graph_geometry_changed()
            self.reroute_connections(affected_rect=before.united(after))
            return
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
        before = proxy.sceneBoundingRect()
        card.set_preview_enabled(enabled)
        card.adjustSize()
        proxy.refresh_ports()
        after = proxy.sceneBoundingRect()
        if _rect_changed(before, after):
            self._mark_graph_geometry_changed()
            self.reroute_connections(affected_rect=before.united(after))
            return
        for connection in proxy.connections:
            connection.update_path()

    def set_node_execution_state(
        self,
        node_id: str,
        state: str,
        *,
        manual: bool,
        message: str = "",
        auto_recalculate: bool = False,
    ) -> None:
        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None or proxy is None:
            return
        if (
            card._manual_execution == bool(manual)
            and card._execution_state == str(state)
            and card._execution_message == str(message or "")
            and card._auto_recalculate == bool(auto_recalculate)
        ):
            return
        before = proxy.sceneBoundingRect()
        card.set_execution_state(
            state,
            manual=manual,
            message=message,
            auto_recalculate=auto_recalculate,
        )
        card.adjustSize()
        proxy.refresh_ports()
        after = proxy.sceneBoundingRect()
        if _rect_changed(before, after):
            self._mark_graph_geometry_changed()
            self.reroute_connections(affected_rect=before.united(after))
            return
        proxy.update()
        for connection in proxy.connections:
            connection.update_path()

    def set_node_processing(
        self,
        node_id: str,
        processing: bool,
        *,
        queued: bool = False,
    ) -> None:
        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None:
            return
        card.set_processing(processing, queued=queued)
        if proxy is not None:
            proxy.update()
        self._sync_processing_timer()

    def clear_node_processing(self) -> None:
        for node_id, card in self._cards.items():
            card.set_processing(False)
            proxy = self._proxies.get(node_id)
            if proxy is not None:
                proxy.update()
        self._processing_timer.stop()

    def set_node_input_ports(
        self,
        node_id: str,
        count: int,
        labels: list[str] | None = None,
        colors: list[str | None] | None = None,
        data_types: list[str] | None = None,
    ) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        proxy.set_input_ports(count, labels, colors, data_types)

    def set_node_output_ports(
        self,
        node_id: str,
        count: int,
        labels: list[str] | None = None,
        colors: list[str | None] | None = None,
        data_types: list[str] | None = None,
    ) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        proxy.set_output_ports(count, labels, colors, data_types)

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
        self.connection_requested.emit(
            source_port.node_id,
            target_port.node_id,
            target_port.port_index,
            source_port.port_index,
        )

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
            operation_id = bytes(event.mimeData().data(OPERATION_MIME)).decode()
            self._update_connection_insert_preview(
                operation_id,
                self.mapToScene(_point_from_event(event)),
            )
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):  # noqa: N802
        self._clear_connection_insert_preview()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat(OPERATION_MIME):
            operation_id = bytes(event.mimeData().data(OPERATION_MIME)).decode()
            scene_pos = self.mapToScene(_point_from_event(event))
            self._update_connection_insert_preview(operation_id, scene_pos)
            connection = self._highlighted_connection
            state = self._highlighted_connection_state
            connection_key = self._connection_key(connection)
            if connection_key is not None and state != "incompatible":
                self.node_insert_requested.emit(operation_id, connection_key, scene_pos)
            else:
                self.node_create_requested.emit(operation_id, scene_pos)
            self._clear_connection_insert_preview()
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected_connections = [
                item
                for item in self.scene.selectedItems()
                if isinstance(item, ConnectionItem)
            ]
            for item in selected_connections:
                self.delete_connection_item(item, notify=True)
            if selected_connections:
                event.accept()
                return

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
        super().keyPressEvent(event)

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            requested = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            old_zoom = self._zoom_percent
            new_zoom = float(
                np.clip(
                    old_zoom * requested,
                    self.WHEEL_MIN_ZOOM,
                    self.WHEEL_MAX_ZOOM,
                )
            )
            if abs(new_zoom - old_zoom) > 0.001:
                self._zoom_percent = new_zoom
                self.scale(new_zoom / old_zoom, new_zoom / old_zoom)
                self.zoom_changed.emit(self._zoom_percent)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        pos = _point_from_event(event)
        background_click = self.itemAt(pos) is None
        if event.button() == Qt.LeftButton:
            tunnel_name = self._tunnel_badge_name_at_view_pos(pos)
            if tunnel_name:
                self.highlight_tunnel(tunnel_name, sticky=True)
                event.accept()
                return
            if background_click and self._active_tunnel_name:
                self.clear_tunnel_highlight(sticky=True)
        if event.button() == Qt.RightButton:
            port = self._port_at_view_pos(pos)
            if port is not None:
                self.port_context_requested.emit(
                    port.kind,
                    port.node_id,
                    port.port_index,
                    _global_pos_from_event(event),
                )
                event.accept()
                return
            node_id = self._node_id_at_view_pos(pos)
            if node_id is not None:
                self._select_node(node_id)
                self._show_node_context_menu(node_id, _global_pos_from_event(event))
                event.accept()
                return
            if not background_click:
                super().mousePressEvent(event)
                return
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
            self._ensure_scene_space_for_rect(self.mapToScene(self.viewport().rect()).boundingRect())
            event.accept()
            return
        if self._pending_source is None and not self._active_tunnel_name:
            tunnel_name = self._tunnel_name_at_view_pos(_point_from_event(event))
            if tunnel_name != self._hover_tunnel_name:
                self.highlight_tunnel(tunnel_name, sticky=False)
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

    def _node_id_at_view_pos(self, pos: QPoint) -> str | None:
        scene_pos = self.mapToScene(pos)
        for item in self.scene.items(scene_pos):
            current = item
            while current is not None:
                if isinstance(current, NodeProxy):
                    return current.node_id
                current = current.parentItem()
        return None

    def _show_node_context_menu(self, node_id: str, global_pos: QPoint) -> None:
        card = self._cards.get(node_id)
        if card is None:
            return
        menu = QMenu(self)
        delete_action = menu.addAction("Delete")
        code_action = menu.addAction("Inspect Code")
        duplicate_action = menu.addAction("Duplicate Node")
        pin_action = None
        if card._can_pin:
            menu.addSeparator()
            pin_action = menu.addAction("Unpin" if card._pinned else "Pin")
        action = _exec_menu(menu, global_pos)
        if action == delete_action:
            self.node_delete_requested.emit(node_id)
        elif action == code_action:
            self.node_code_requested.emit(node_id)
        elif action == duplicate_action:
            self.node_duplicate_requested.emit(node_id)
        elif pin_action is not None and action == pin_action:
            self.pin_requested.emit(node_id)

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

    def _port_at_view_pos(self, pos: QPoint) -> PortItem | None:
        scene_pos = self.mapToScene(pos)
        for item in self.scene.items(scene_pos):
            current = item
            while current is not None:
                if isinstance(current, PortItem):
                    return current
                current = current.parentItem()
        return None

    def _tunnel_name_at_view_pos(self, pos: QPoint) -> str:
        port = self._port_at_view_pos(pos)
        if port is None:
            return ""
        return str(getattr(port, "_tunnel_label", "") or "").strip()

    def _tunnel_badge_name_at_view_pos(self, pos: QPoint) -> str:
        scene_pos = self.mapToScene(pos)
        for item in self.scene.items(scene_pos):
            if not isinstance(item, TunnelBadgeItem):
                continue
            current = item.parentItem()
            while current is not None:
                if isinstance(current, PortItem):
                    return str(getattr(current, "_tunnel_label", "") or "").strip()
                current = current.parentItem()
        return ""

    def _connection_at(self, scene_pos: QPointF) -> ConnectionItem | None:
        candidates: list[tuple[float, ConnectionItem]] = []
        for item in self.scene.items(scene_pos):
            if not isinstance(item, ConnectionItem):
                continue
            distance = self._distance_to_connection(item, scene_pos)
            candidates.append((distance, item))
        if not candidates:
            return None
        candidates.sort(key=lambda candidate: candidate[0])
        return candidates[0][1]

    @staticmethod
    def _distance_to_connection(item: ConnectionItem, scene_pos: QPointF) -> float:
        path = item.path()
        if path.isEmpty():
            return float("inf")
        best = float("inf")
        samples = 48
        for index in range(samples + 1):
            point = path.pointAtPercent(index / samples)
            dx = point.x() - scene_pos.x()
            dy = point.y() - scene_pos.y()
            best = min(best, dx * dx + dy * dy)
        return best

    @staticmethod
    def _connection_key(
        connection: ConnectionItem | None,
    ) -> tuple[str, str, int, int] | None:
        if connection is None:
            return None
        return (
            connection.source_id,
            connection.target_id,
            connection.target_port,
            connection.source_port,
        )

    def _update_connection_insert_preview(
        self,
        operation_id: str,
        scene_pos: QPointF,
    ) -> None:
        connection = self._connection_at(scene_pos)
        key = self._connection_key(connection)
        state = None
        message = ""
        if key is not None and self._connection_insert_validator is not None:
            state, message = self._connection_insert_validator(operation_id, key)
        elif key is not None:
            state = "full"
            message = "Drop to insert node on this connection."
        if state not in ConnectionItem.PREVIEW_STATES:
            state = None
        if (
            connection is self._highlighted_connection
            and operation_id == self._highlighted_connection_operation
            and state == self._highlighted_connection_state
        ):
            return
        self._clear_connection_insert_preview()
        if connection is None or state is None:
            return
        self._highlighted_connection = connection
        self._highlighted_connection_state = state
        self._highlighted_connection_operation = operation_id
        connection.set_insert_preview_state(state)
        self._connection_pulse_timer.start()
        if message:
            self.status_message.emit(message)

    def _clear_connection_insert_preview(self) -> None:
        if self._highlighted_connection is not None:
            self._highlighted_connection.set_insert_preview_state(None)
        self._highlighted_connection = None
        self._highlighted_connection_state = None
        self._highlighted_connection_operation = None
        self._connection_pulse_timer.stop()

    def _advance_connection_insert_pulse(self) -> None:
        if self._highlighted_connection is None:
            self._connection_pulse_timer.stop()
            return
        self._highlighted_connection.advance_insert_preview_pulse()

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
        target_proxy = self._proxies.get(target_port.node_id)
        if target_proxy is None:
            return False
        return _types_compatible(
            self._pending_source.data_type, target_port.data_type
        )

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

    def _connection_exists(
        self,
        source_id: str,
        target_id: str,
        target_port: int = 0,
        source_port: int = 0,
    ) -> bool:
        return any(
            item.source_id == source_id
            and item.target_id == target_id
            and item.target_port == int(target_port)
            and item.source_port == int(source_port)
            for item in self._connections
        )

    def _advance_processing_spinners(self) -> None:
        active = False
        for node_id, card in self._cards.items():
            if not card.is_processing():
                continue
            active = True
            card.advance_processing_spinner()
            proxy = self._proxies.get(node_id)
            if proxy is not None:
                proxy.update()
        if not active:
            self._processing_timer.stop()

    def _sync_processing_timer(self) -> None:
        active = any(card.is_processing() for card in self._cards.values())
        if active and not self._processing_timer.isActive():
            self._processing_timer.start()
        elif not active and self._processing_timer.isActive():
            self._processing_timer.stop()

    def _ensure_scene_space_for_rect(self, rect: QRectF) -> None:
        if rect.isNull() or not rect.isValid():
            return
        scene_rect = QRectF(self.scene.sceneRect())
        if scene_rect.isNull() or not scene_rect.isValid():
            scene_rect = rect.adjusted(-400.0, -300.0, 400.0, 300.0)

        margin = float(self.SCENE_EDGE_MARGIN)
        expand_x = float(self.SCENE_EXPAND_STEP_X)
        expand_y = float(self.SCENE_EXPAND_STEP_Y)

        changed = False
        while rect.left() < scene_rect.left() + margin:
            scene_rect.setLeft(scene_rect.left() - expand_x)
            changed = True
        while rect.right() > scene_rect.right() - margin:
            scene_rect.setRight(scene_rect.right() + expand_x)
            changed = True
        while rect.top() < scene_rect.top() + margin:
            scene_rect.setTop(scene_rect.top() - expand_y)
            changed = True
        while rect.bottom() > scene_rect.bottom() - margin:
            scene_rect.setBottom(scene_rect.bottom() + expand_y)
            changed = True

        if changed:
            self.scene.setSceneRect(scene_rect)


def _to_pointf(value) -> QPointF | None:
    """Coerce a stored (x, y) pair or QPointF into a QPointF (or None)."""
    if value is None:
        return None
    if isinstance(value, QPointF):
        return value
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return QPointF(float(value[0]), float(value[1]))
    return None


def _wire_path(
    start: QPointF,
    end: QPointF,
    *,
    obstacles: tuple[QRectF, ...] | list[QRectF] = (),
) -> QPainterPath:
    clean_obstacles = tuple(
        rect for rect in obstacles if rect.isValid() and not rect.isNull()
    )
    if not clean_obstacles:
        return _bezier_wire_path(start, end)
    relevant_obstacles = _route_corridor_obstacles(start, end, clean_obstacles)
    if not relevant_obstacles:
        return _bezier_wire_path(start, end)
    if _should_use_close_port_curve(start, end):
        return _bezier_wire_path(start, end)

    bezier = _bezier_wire_path(start, end)
    bezier_points = _sample_path_points(bezier, samples=24)
    if not _route_collision_penalty(bezier_points, relevant_obstacles):
        return bezier

    candidates = _wire_route_candidates(start, end, relevant_obstacles)
    best_path, _points, _score = min(
        candidates,
        key=lambda candidate: candidate[2],
    )
    return best_path


def _bezier_wire_path(start: QPointF, end: QPointF) -> QPainterPath:
    horizontal_gap = end.x() - start.x()
    if horizontal_gap > 0:
        dx = min(80.0, max(1.0, horizontal_gap * 0.45))
    else:
        dx = max(80.0, abs(horizontal_gap) * 0.5)
    path = QPainterPath(start)
    path.cubicTo(
        QPointF(start.x() + dx, start.y()),
        QPointF(end.x() - dx, end.y()),
        end,
    )
    return path


def _should_use_close_port_curve(start: QPointF, end: QPointF) -> bool:
    horizontal_gap = end.x() - start.x()
    return 0 < horizontal_gap <= 220.0


def _wire_route_candidates(
    start: QPointF,
    end: QPointF,
    obstacles: tuple[QRectF, ...],
) -> list[tuple[QPainterPath, tuple[QPointF, ...], float]]:
    candidates: list[tuple[QPainterPath, tuple[QPointF, ...], float]] = []
    bezier = _bezier_wire_path(start, end)
    bezier_points = _sample_path_points(bezier, samples=32)
    candidates.append(
        (
            bezier,
            bezier_points,
            _route_score(bezier_points, obstacles, bends=0),
        )
    )

    for points in _orthogonal_route_candidates(start, end, obstacles):
        clean = _clean_route_points(points)
        if len(clean) < 2:
            continue
        path = _rounded_polyline_path(clean)
        candidates.append(
            (
                path,
                tuple(clean),
                _route_score(tuple(clean), obstacles, bends=max(len(clean) - 2, 0)),
            )
        )
    return candidates


def _orthogonal_route_candidates(
    start: QPointF,
    end: QPointF,
    obstacles: tuple[QRectF, ...],
) -> list[list[QPointF]]:
    port_stub = _port_stub_length(start, end)
    route_start = QPointF(start.x() + port_stub, start.y())
    route_end = QPointF(end.x() - port_stub, end.y())
    sign = 1.0 if route_end.x() >= route_start.x() else -1.0
    horizontal_gap = abs(route_end.x() - route_start.x())
    lead = min(max(horizontal_gap * 0.22, 56.0), 130.0)
    x1 = route_start.x() + sign * lead
    x2 = route_end.x() - sign * lead
    mid_x = (route_start.x() + route_end.x()) / 2.0
    mid_y = (route_start.y() + route_end.y()) / 2.0
    candidates = [
        [
            start,
            route_start,
            QPointF(mid_x, route_start.y()),
            QPointF(mid_x, route_end.y()),
            route_end,
            end,
        ],
        [
            start,
            route_start,
            QPointF(route_start.x(), route_end.y()),
            route_end,
            end,
        ],
        [
            start,
            route_start,
            QPointF(route_start.x(), mid_y),
            QPointF(route_end.x(), mid_y),
            route_end,
            end,
        ],
    ]

    blockers = _route_relevant_obstacles(route_start, route_end, obstacles)
    if blockers:
        top = min(rect.top() for rect in blockers)
        bottom = max(rect.bottom() for rect in blockers)
        left = min(rect.left() for rect in blockers)
        right = max(rect.right() for rect in blockers)
    else:
        top = min(route_start.y(), route_end.y())
        bottom = max(route_start.y(), route_end.y())
        left = min(route_start.x(), route_end.x())
        right = max(route_start.x(), route_end.x())
    pad = 44.0
    above_y = top - pad
    below_y = bottom + pad
    left_x = left - pad
    right_x = right + pad
    lo_x = min(route_start.x(), route_end.x())
    hi_x = max(route_start.x(), route_end.x())
    if sign >= 0:
        detour_start_x = min(max(min(x1, left_x), lo_x), hi_x)
        detour_end_x = min(max(max(x2, right_x), lo_x), hi_x)
    else:
        detour_start_x = min(max(max(x1, right_x), lo_x), hi_x)
        detour_end_x = min(max(min(x2, left_x), lo_x), hi_x)
    for y in (above_y, below_y):
        candidates.append(
            [
                start,
                route_start,
                QPointF(detour_start_x, route_start.y()),
                QPointF(detour_start_x, y),
                QPointF(detour_end_x, y),
                QPointF(detour_end_x, route_end.y()),
                route_end,
                end,
            ]
        )
    for x in (left_x, right_x):
        if x < lo_x or x > hi_x:
            continue
        candidates.append(
            [
                start,
                route_start,
                QPointF(x, route_start.y()),
                QPointF(x, route_end.y()),
                route_end,
                end,
            ]
        )
    return candidates


def _port_stub_length(start: QPointF, end: QPointF) -> float:
    preferred = 36.0
    horizontal_gap = end.x() - start.x()
    if horizontal_gap > 0:
        return min(preferred, max(1.0, horizontal_gap / 3.0))
    return min(preferred, max(10.0, abs(horizontal_gap) * 0.18))


def _route_relevant_obstacles(
    start: QPointF,
    end: QPointF,
    obstacles: tuple[QRectF, ...],
) -> tuple[QRectF, ...]:
    relevant = _route_corridor_obstacles(start, end, obstacles)
    return relevant or obstacles


def _route_corridor_obstacles(
    start: QPointF,
    end: QPointF,
    obstacles: tuple[QRectF, ...],
) -> tuple[QRectF, ...]:
    corridor = QRectF(start, end).normalized().adjusted(-80.0, -110.0, 80.0, 110.0)
    relevant = [rect for rect in obstacles if rect.intersects(corridor)]
    if len(relevant) <= 12:
        return tuple(relevant)
    center = QPointF((start.x() + end.x()) / 2.0, (start.y() + end.y()) / 2.0)
    relevant.sort(key=lambda rect: _point_distance(rect.center(), center))
    return tuple(relevant[:12])


def _route_score(
    points: tuple[QPointF, ...],
    obstacles: tuple[QRectF, ...],
    *,
    bends: int,
) -> float:
    collision = _route_collision_penalty(points, obstacles)
    return collision * 1000.0 + _polyline_length(points) + bends * 42.0


def _route_collision_penalty(
    points: tuple[QPointF, ...],
    obstacles: tuple[QRectF, ...],
) -> float:
    return sum(_polyline_rect_penalty(points, rect) for rect in obstacles)


def _polyline_rect_penalty(points: tuple[QPointF, ...], rect: QRectF) -> float:
    penalty = 0.0
    for start, end in zip(points, points[1:], strict=False):
        penalty += _segment_rect_penalty(start, end, rect)
    return penalty


def _segment_rect_penalty(start: QPointF, end: QPointF, rect: QRectF) -> float:
    segment_rect = QRectF(start, end).normalized().adjusted(-1.0, -1.0, 1.0, 1.0)
    if not segment_rect.intersects(rect):
        return 0.0
    dx = end.x() - start.x()
    dy = end.y() - start.y()
    if abs(dy) < 0.001:
        if rect.top() <= start.y() <= rect.bottom():
            overlap = _range_overlap(start.x(), end.x(), rect.left(), rect.right())
            return max(overlap, 0.0) + 25.0
        return 0.0
    if abs(dx) < 0.001:
        if rect.left() <= start.x() <= rect.right():
            overlap = _range_overlap(start.y(), end.y(), rect.top(), rect.bottom())
            return max(overlap, 0.0) + 25.0
        return 0.0

    samples = max(int(np.hypot(dx, dy) / 18.0), 8)
    inside = 0
    for index in range(samples + 1):
        t = index / max(samples, 1)
        point = QPointF(start.x() + dx * t, start.y() + dy * t)
        if rect.contains(point):
            inside += 1
    if inside:
        return inside * 18.0 + 25.0
    return 0.0


def _range_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    lo_a, hi_a = sorted((float(a0), float(a1)))
    lo_b, hi_b = sorted((float(b0), float(b1)))
    return max(0.0, min(hi_a, hi_b) - max(lo_a, lo_b))


def _polyline_length(points: tuple[QPointF, ...]) -> float:
    return sum(
        float(np.hypot(end.x() - start.x(), end.y() - start.y()))
        for start, end in zip(points, points[1:], strict=False)
    )


def _sample_path_points(path: QPainterPath, *, samples: int) -> tuple[QPointF, ...]:
    count = max(int(samples), 2)
    return tuple(path.pointAtPercent(index / count) for index in range(count + 1))


def _rounded_polyline_path(points: list[QPointF], radius: float = 18.0) -> QPainterPath:
    clean = _clean_route_points(points)
    path = QPainterPath(clean[0])
    if len(clean) == 2:
        path.lineTo(clean[-1])
        return path
    for index in range(1, len(clean) - 1):
        previous = clean[index - 1]
        corner = clean[index]
        next_point = clean[index + 1]
        distance_in = _point_distance(previous, corner)
        distance_out = _point_distance(corner, next_point)
        bend_radius = min(float(radius), distance_in / 2.0, distance_out / 2.0)
        if bend_radius < 1.0:
            path.lineTo(corner)
            continue
        before = _point_towards(corner, previous, bend_radius)
        after = _point_towards(corner, next_point, bend_radius)
        path.lineTo(before)
        path.quadTo(corner, after)
    path.lineTo(clean[-1])
    return path


def _clean_route_points(points: list[QPointF]) -> list[QPointF]:
    clean: list[QPointF] = []
    for point in points:
        if clean and _point_distance(clean[-1], point) < 0.5:
            continue
        clean.append(QPointF(point))
    return clean


def _point_distance(first: QPointF, second: QPointF) -> float:
    return float(np.hypot(second.x() - first.x(), second.y() - first.y()))


def _point_towards(origin: QPointF, target: QPointF, distance: float) -> QPointF:
    total = _point_distance(origin, target)
    if total <= 0:
        return QPointF(origin)
    ratio = float(distance) / total
    return QPointF(
        origin.x() + (target.x() - origin.x()) * ratio,
        origin.y() + (target.y() - origin.y()) * ratio,
    )


def _rect_changed(first: QRectF, second: QRectF, tolerance: float = 0.5) -> bool:
    return any(
        abs(a - b) > tolerance
        for a, b in (
            (first.left(), second.left()),
            (first.top(), second.top()),
            (first.width(), second.width()),
            (first.height(), second.height()),
        )
    )


def _points_close(first: QPointF, second: QPointF, tolerance: float = 0.5) -> bool:
    return (
        abs(float(first.x()) - float(second.x())) <= tolerance
        and abs(float(first.y()) - float(second.y())) <= tolerance
    )


def _types_compatible(output_type: str, input_type: str | None) -> bool:
    if input_type is None or input_type == "any" or output_type == "any":
        return True
    if input_type == "array":
        return output_type in {"array", "image", "mask", "labels"}
    if input_type == "mask_or_labels":
        return output_type in {"mask", "labels"}
    if input_type == "table":
        return output_type == "table"
    return output_type == input_type


def _node_input_port_count(node) -> int:
    if not getattr(node, "has_input", False):
        return 0
    max_inputs = getattr(node, "max_inputs", 1)
    if max_inputs is None or max_inputs != 1:
        try:
            requested = max(int(node.params.get("input_count", 1)), 1)
        except Exception:
            return 1
        if max_inputs is not None:
            return min(requested, max(int(max_inputs), 1))
        return requested
    return 1


def _node_input_port_labels(node) -> list[str]:
    count = _node_input_port_count(node)
    if getattr(node, "operation_id", "") == "combine_channels":
        colors = _channel_color_names(node)
        return [
            f"Channel {index + 1}: {colors[index]}"
            for index in range(count)
        ]
    return [f"Input {index + 1}" for index in range(count)]


def _node_input_port_colors(node) -> list[str | None]:
    if getattr(node, "operation_id", "") != "combine_channels":
        return []
    return [_CHANNEL_COLOR_HEX.get(name.lower()) for name in _channel_color_names(node)]


def _node_input_port_types(node) -> list[str]:
    input_type = getattr(node, "input_type", None) or "any"
    return [input_type for _index in range(_node_input_port_count(node))]


def _channel_color_names(node) -> list[str]:
    count = _node_input_port_count(node)
    defaults = ["Red", "Green", "Blue", "Magenta", "Cyan", "Yellow"]
    raw = str(node.params.get("channel_colors", "")).strip()
    values = [part.strip().title() for part in raw.split(",") if part.strip()]
    while len(values) < count:
        values.append(defaults[len(values) % len(defaults)])
    return values[:count]


_CHANNEL_COLOR_HEX = {
    "red": "#ef4444",
    "green": "#22c55e",
    "blue": "#60a5fa",
    "magenta": "#d946ef",
    "cyan": "#06b6d4",
    "yellow": "#eab308",
}


def _point_from_event(event) -> QPoint:
    pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
    return pos.toPoint() if hasattr(pos, "toPoint") else pos


def _global_pos_from_event(event) -> QPoint:
    if hasattr(event, "globalPosition"):
        return event.globalPosition().toPoint()
    return event.globalPos()


def _view_for_scene(scene) -> PipelineGraphView | None:
    if scene is None or not scene.views():
        return None
    view = scene.views()[0]
    return view if isinstance(view, PipelineGraphView) else None


def _exec_menu(menu: QMenu, pos):
    if hasattr(menu, "exec"):
        return menu.exec(pos)
    return menu.exec_(pos)
