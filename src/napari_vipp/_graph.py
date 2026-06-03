"""Qt graph canvas and node card widgets for the VIPP prototype."""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from qtpy.QtWidgets import (
    QFrame,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


class ClickablePreview(QLabel):
    clicked = Signal()

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class NodeCard(QFrame):
    """Small embedded node UI with a thumbnail and graph actions."""

    selected = Signal(str)
    inspect_requested = Signal(str)
    pin_requested = Signal(str)

    def __init__(self, node_id: str, title: str, category: str, parent=None):
        super().__init__(parent)
        self.node_id = node_id
        self.setObjectName("NodeCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(220)
        self.setStyleSheet(
            """
            QFrame#NodeCard {
                background: #20242b;
                border: 1px solid #4b5563;
                border-radius: 6px;
            }
            QLabel {
                color: #f3f4f6;
            }
            QPushButton {
                padding: 3px 7px;
            }
            """
        )

        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 650;")
        self.category_label = QLabel(category)
        self.category_label.setStyleSheet("color: #a5b4fc; font-size: 10px;")

        self.preview = ClickablePreview()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(180, 110)
        self.preview.setText("No preview")
        self.preview.setStyleSheet(
            "background: #111827; color: #9ca3af; border-radius: 4px;"
        )
        self.preview.clicked.connect(lambda: self.inspect_requested.emit(self.node_id))

        self.inspect_button = QPushButton("Inspect")
        self.inspect_button.clicked.connect(
            lambda: self.inspect_requested.emit(self.node_id)
        )
        self.pin_button = QPushButton("Pin")
        self.pin_button.clicked.connect(lambda: self.pin_requested.emit(self.node_id))

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.addWidget(self.inspect_button)
        actions.addWidget(self.pin_button)
        actions.addStretch(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(self.category_label)
        layout.addWidget(self.title_label)
        layout.addWidget(self.preview)
        layout.addLayout(actions)

    def mousePressEvent(self, event):  # noqa: N802
        self.selected.emit(self.node_id)
        super().mousePressEvent(event)

    def set_thumbnail(self, thumbnail: np.ndarray | None) -> None:
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


class ConnectionItem(QGraphicsPathItem):
    def __init__(self, source: QGraphicsProxyWidget, target: QGraphicsProxyWidget):
        super().__init__()
        self.source = source
        self.target = target
        self.setPen(QPen(QColor("#8aa0c8"), 2.0))
        self.setZValue(-10)
        self.update_path()

    def update_path(self) -> None:
        src = self.source.sceneBoundingRect()
        dst = self.target.sceneBoundingRect()
        start = QPointF(src.right(), src.center().y())
        end = QPointF(dst.left(), dst.center().y())
        dx = max(80.0, (end.x() - start.x()) * 0.5)
        path = QPainterPath(start)
        path.cubicTo(
            QPointF(start.x() + dx, start.y()),
            QPointF(end.x() - dx, end.y()),
            end,
        )
        self.setPath(path)


class PipelineGraphView(QGraphicsView):
    """Large pan/zoom graph canvas hosted inside napari."""

    node_selected = Signal(str)
    inspect_requested = Signal(str)
    pin_requested = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(QColor("#151922"))
        self._proxies: dict[str, QGraphicsProxyWidget] = {}
        self._cards: dict[str, NodeCard] = {}
        self._connections: list[ConnectionItem] = []

    def build_demo_graph(self, nodes) -> None:
        self.scene.clear()
        self._proxies.clear()
        self._cards.clear()
        self._connections.clear()

        positions = {
            "input": QPointF(0, 20),
            "gaussian": QPointF(330, 20),
            "threshold": QPointF(660, 20),
        }
        for node in nodes:
            card = NodeCard(node.id, node.title, node.category)
            card.selected.connect(self.node_selected)
            card.inspect_requested.connect(self.inspect_requested)
            card.pin_requested.connect(self.pin_requested)
            proxy = self.scene.addWidget(card)
            proxy.setPos(positions.get(node.id, QPointF(0, 0)))
            self._cards[node.id] = card
            self._proxies[node.id] = proxy

        self._add_connection("input", "gaussian")
        self._add_connection("gaussian", "threshold")
        scene_rect = self.scene.itemsBoundingRect().adjusted(-80, -80, 120, 80)
        self.scene.setSceneRect(scene_rect)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def set_thumbnail(self, node_id: str, thumbnail: np.ndarray | None) -> None:
        card = self._cards.get(node_id)
        if card is not None:
            card.set_thumbnail(thumbnail)

    def _add_connection(self, source_id: str, target_id: str) -> None:
        source = self._proxies[source_id]
        target = self._proxies[target_id]
        item = ConnectionItem(source, target)
        self.scene.addItem(item)
        self._connections.append(item)

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)
