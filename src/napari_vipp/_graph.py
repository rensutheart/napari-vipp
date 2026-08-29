"""Qt graph canvas and node card widgets for the VIPP prototype."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from math import ceil

import numpy as np
from qtpy.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from qtpy.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QTransform,
)
from qtpy.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsProxyWidget,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_vipp._theme import category_color, category_tint
from napari_vipp.core.pipeline import EXECUTION_BLOCKED, NODE_LIBRARY_BY_ID

OPERATION_MIME = "application/x-napari-vipp-operation"
PINNABLE_OUTPUT_TYPES = {"array", "image", "mask", "labels"}
STALE_EXECUTION_ACCENT = "#f59e0b"
BLOCKED_EXECUTION_ACCENT = "#b45309"
BYPASSED_NODE_OUTLINE = "#22d3ee"
BYPASSED_NODE_PASS_THROUGH = "#67e8f9"
BYPASSED_NODE_OPACITY = 0.72


class PortLabelMode(StrEnum):
    """Control which graph ports have persistent labels on the canvas."""

    HIDE_ALL = "Hide all"
    AMBIGUOUS_ONLY = "Ambiguous only"
    SHOW_ALL = "Show all"


class ComputeBadgeKind(StrEnum):
    """Presentation-only identities for a node's accepted compute result."""

    CPU = "cpu"
    CUPY = "cupy"
    CUCIM = "cucim"
    CPU_FALLBACK = "cpu_fallback"
    BYPASSED = "bypassed"


class ThumbnailStatsBadgeKind(StrEnum):
    """Presentation identities for non-scientific thumbnail statistics work."""

    PENDING = "pending"
    CPU = "cpu"
    GPU = "gpu"
    CPU_FALLBACK = "cpu_fallback"
    ERROR = "error"


_COMPUTE_BADGE_LABELS = {
    ComputeBadgeKind.CPU: "CPU",
    ComputeBadgeKind.CUPY: "GPU · CuPy",
    ComputeBadgeKind.CUCIM: "GPU · cuCIM",
    ComputeBadgeKind.CPU_FALLBACK: "CPU fallback",
    ComputeBadgeKind.BYPASSED: "Bypassed",
}

_COMPUTE_BADGE_COLORS = {
    ComputeBadgeKind.CPU: ("#334155", "#e2e8f0", "#64748b"),
    ComputeBadgeKind.CUPY: ("#1e3a5f", "#bfdbfe", "#3b82f6"),
    ComputeBadgeKind.CUCIM: ("#064e3b", "#bbf7d0", "#10b981"),
    ComputeBadgeKind.CPU_FALLBACK: ("#78350f", "#fde68a", "#f59e0b"),
    ComputeBadgeKind.BYPASSED: ("#164e63", "#cffafe", "#0891b2"),
}


def _coerce_compute_badge_kind(
    value: ComputeBadgeKind | str,
) -> ComputeBadgeKind:
    if isinstance(value, ComputeBadgeKind):
        return value
    normalized = str(value).strip().casefold().replace("-", "_").replace(" ", "_")
    aliases = {
        "gpu_·_cupy": ComputeBadgeKind.CUPY,
        "gpu_cupy": ComputeBadgeKind.CUPY,
        "gpu_·_cucim": ComputeBadgeKind.CUCIM,
        "gpu_cucim": ComputeBadgeKind.CUCIM,
        "fallback_cpu": ComputeBadgeKind.CPU_FALLBACK,
    }
    if normalized in aliases:
        return aliases[normalized]
    try:
        return ComputeBadgeKind(normalized)
    except ValueError as exc:
        choices = ", ".join(kind.value for kind in ComputeBadgeKind)
        raise ValueError(
            f"Unknown compute badge kind {value!r}; expected one of: {choices}."
        ) from exc


def _coerce_port_label_mode(value: PortLabelMode | str) -> PortLabelMode:
    if isinstance(value, PortLabelMode):
        return value
    normalized = str(value).strip().casefold().replace("_", " ")
    for mode in PortLabelMode:
        if normalized in {
            mode.value.casefold(),
            mode.name.casefold().replace("_", " "),
        }:
            return mode
    choices = ", ".join(mode.value for mode in PortLabelMode)
    raise ValueError(f"Unknown port-label mode {value!r}; expected one of: {choices}.")


class ClickablePreview(QLabel):
    clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap = QPixmap()

    def set_source_pixmap(self, pixmap: QPixmap) -> None:
        """Retain full render detail for zoom-aware custom painting."""
        self._source_pixmap = QPixmap(pixmap)
        # QLabel normalizes a high-DPR pixmap to the widget's current physical
        # size, losing detail that QGraphicsView could use when the graph zooms.
        # Keep its ordinary content empty and paint the backing directly instead.
        super().setPixmap(QPixmap())
        self.update()

    def source_pixmap(self) -> QPixmap:
        """Return the retained full-detail thumbnail backing pixmap."""
        return QPixmap(self._source_pixmap)

    def clear_source_pixmap(self) -> None:
        self._source_pixmap = QPixmap()
        super().setPixmap(QPixmap())
        self.update()

    def has_source_pixmap(self) -> bool:
        """Return whether a custom-painted thumbnail backing is available."""
        return not self._source_pixmap.isNull()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if not self._source_pixmap.isNull():
            self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        if self._source_pixmap.isNull() or self.width() < 1 or self.height() < 1:
            return
        bounds = QRectF(self.contentsRect())
        source_width = float(self._source_pixmap.width())
        source_height = float(self._source_pixmap.height())
        scale = min(
            bounds.width() / source_width,
            bounds.height() / source_height,
        )
        target = QRectF(
            0.0,
            0.0,
            source_width * scale,
            source_height * scale,
        )
        target.moveCenter(bounds.center())
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform, True)
        painter.drawPixmap(
            target,
            self._source_pixmap,
            QRectF(self._source_pixmap.rect()),
        )
        painter.end()

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


class ElidedSubtitleLabel(QLabel):
    """One-line subtitle that keeps its complete value in a tooltip."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._full_text = ""

    def set_full_text(self, text: str, tooltip: str | None = None) -> None:
        self._full_text = str(text or "").strip()
        detail = self._full_text if tooltip is None else str(tooltip or "").strip()
        self.setToolTip(detail)
        self.setAccessibleName(
            f"Input binding: {self._full_text}" if self._full_text else ""
        )
        self.setVisible(bool(self._full_text))
        self._refresh_elision()

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        self._refresh_elision()

    def _refresh_elision(self) -> None:
        available = max(self.contentsRect().width(), 0)
        elided = self.fontMetrics().elidedText(
            self._full_text,
            Qt.ElideMiddle,
            available,
        )
        super().setText(elided)


def _bypass_outline_pen() -> QPen:
    """Return the prominent theme-safe dotted bypass outline."""

    pen = QPen(QColor(BYPASSED_NODE_OUTLINE), 3.0, Qt.DotLine)
    pen.setCapStyle(Qt.RoundCap)
    pen.setCosmetic(True)
    return pen


def _bypass_pass_through_pen() -> QPen:
    """Return the subtle input-to-output cue drawn through a bypassed card."""

    color = QColor(BYPASSED_NODE_PASS_THROUGH)
    color.setAlpha(178)
    pen = QPen(color, 2.0, Qt.SolidLine)
    pen.setCapStyle(Qt.RoundCap)
    pen.setCosmetic(True)
    return pen


class BypassCardOverlay(QWidget):
    """Purely visual, non-interactive bypass treatment above card content."""

    def __init__(self, parent: QWidget):
        super().__init__(parent)
        self._input_y: float | None = None
        self._output_y: float | None = None
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAccessibleName("")

    def set_pass_through_positions(
        self,
        input_y: float | None,
        output_y: float | None,
    ) -> None:
        """Align the visual alias cue with the primary input and sole output."""

        resolved_input = None if input_y is None else float(input_y)
        resolved_output = None if output_y is None else float(output_y)
        if (
            self._input_y == resolved_input
            and self._output_y == resolved_output
        ):
            return
        self._input_y = resolved_input
        self._output_y = resolved_output
        self.update()

    def paintEvent(self, event):  # noqa: N802
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # A line through the port axis makes the alias relationship legible at
        # a glance. This is paint only: it cannot receive hover or mouse input.
        painter.setPen(_bypass_pass_through_pen())
        center_y = float(self.rect().center().y())
        input_y = center_y if self._input_y is None else self._input_y
        output_y = center_y if self._output_y is None else self._output_y
        painter.drawLine(
            QPointF(3.0, input_y),
            QPointF(float(self.width()) - 3.0, output_y),
        )

        painter.setPen(_bypass_outline_pen())
        painter.setBrush(Qt.NoBrush)
        outline = QRectF(self.rect()).adjusted(2.0, 2.0, -2.0, -2.0)
        painter.drawRoundedRect(outline, 5.0, 5.0)


class NodeCard(QFrame):
    """Small embedded node UI with a thumbnail and graph actions."""

    BASE_MINIMUM_WIDTH = 220
    BASE_CONTENT_MARGINS = (10, 8, 10, 10)

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
        self._search_highlight = False
        self._preview_enabled = True
        self._thumbnail_stats_tooltip = ""
        self._processing = False
        self._processing_queued = False
        self._processing_angle = 0
        self._manual_execution = False
        self._execution_state = "not_calculated"
        self._execution_message = ""
        self._auto_recalculate = False
        self._isolated_tuning = False
        self._bypassed = False
        self.setObjectName("NodeCard")
        self.setFrameShape(QFrame.StyledPanel)
        self.setMinimumWidth(self.BASE_MINIMUM_WIDTH)
        self.setCursor(Qt.OpenHandCursor)

        self.accent_bar = QFrame()
        self.accent_bar.setObjectName("NodeAccent")
        self.accent_bar.setFixedHeight(4)
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet("font-weight: 650;")
        self.title_label.setSizePolicy(
            QSizePolicy.Expanding,
            QSizePolicy.Preferred,
        )
        self.compute_badge = QLabel("")
        self.compute_badge.setObjectName("NodeComputeBadge")
        self.compute_badge.setAlignment(Qt.AlignCenter)
        self.compute_badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.compute_badge.hide()
        self._compute_badge_kind: ComputeBadgeKind | None = None
        self._compute_badge_stale = False
        self._compute_badge_tooltip = ""
        self.optimization_badge = QLabel("GPU tip")
        self.optimization_badge.setObjectName("NodeOptimizationBadge")
        self.optimization_badge.setAlignment(Qt.AlignCenter)
        self.optimization_badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.optimization_badge.setStyleSheet(
            "QLabel#NodeOptimizationBadge {"
            " background: #422006; color: #fde68a;"
            " border: 1px solid #d97706; border-radius: 7px;"
            " font-size: 9px; font-weight: 650; padding: 1px 5px;"
            "}"
        )
        self.optimization_badge.hide()
        self._optimization_hint = ""
        self.title_row = QWidget(self)
        title_layout = QHBoxLayout(self.title_row)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(6)
        title_layout.addWidget(self.title_label, 1)
        title_layout.addWidget(
            self.optimization_badge,
            0,
            Qt.AlignRight | Qt.AlignVCenter,
        )
        title_layout.addWidget(self.compute_badge, 0, Qt.AlignRight | Qt.AlignVCenter)
        self.category_label = QLabel(category)
        self.category_label.setObjectName("NodeCategory")
        self.subtitle_label = ElidedSubtitleLabel()
        self.subtitle_label.setObjectName("NodeSubtitle")
        self.subtitle_label.setStyleSheet(
            "color: #93c5fd; font-size: 10px; padding-bottom: 1px;"
        )
        self.subtitle_label.hide()

        self.preview = ClickablePreview()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(180, 110)
        self.preview.setText("No preview")
        self.preview.setAccessibleName(f"{title} thumbnail preview")
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

        self.card_layout = QVBoxLayout(self)
        self.card_layout.setContentsMargins(*self.BASE_CONTENT_MARGINS)
        self.card_layout.addWidget(self.accent_bar)
        self.card_layout.addWidget(self.category_label)
        self.card_layout.addWidget(self.title_row)
        self.card_layout.addWidget(self.subtitle_label)
        self.card_layout.addWidget(self.preview)
        self.card_layout.addWidget(self.metadata_label)
        self.card_layout.addWidget(self.execution_label)
        self.card_layout.addWidget(self.calculate_button)
        self._bypass_overlay = BypassCardOverlay(self)
        self._bypass_overlay.setGeometry(self.rect())
        self._bypass_overlay.hide()
        self._refresh_style()

    def set_port_label_gutters(self, left: float, right: float) -> None:
        """Reserve clear card space for persistent labels on either edge."""
        left_gutter = max(int(ceil(left)), 0)
        right_gutter = max(int(ceil(right)), 0)
        base_left, top, base_right, bottom = self.BASE_CONTENT_MARGINS
        margins = (
            base_left + left_gutter,
            top,
            base_right + right_gutter,
            bottom,
        )
        current = self.card_layout.contentsMargins()
        if (
            current.left(),
            current.top(),
            current.right(),
            current.bottom(),
        ) == margins:
            return
        self.card_layout.setContentsMargins(*margins)
        self.setMinimumWidth(self.BASE_MINIMUM_WIDTH + left_gutter + right_gutter)
        self.card_layout.invalidate()
        self.updateGeometry()
        self.adjustSize()

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
        self._bypass_overlay.setGeometry(self.rect())

    @staticmethod
    def _bypass_outline_pen() -> QPen:
        """Return the theme-safe dotted pen used for bypassed node cards."""

        return _bypass_outline_pen()

    @staticmethod
    def _bypass_pass_through_pen() -> QPen:
        """Return the non-interactive line connecting input and output sides."""

        return _bypass_pass_through_pen()

    def set_selected(self, selected: bool) -> None:
        selected = bool(selected)
        if self._selected == selected:
            return
        self._selected = selected
        self._refresh_style()

    def set_pinned(self, pinned: bool) -> None:
        pinned = bool(pinned)
        if self._pinned == pinned:
            return
        self._pinned = pinned
        self.pin_button.setText("Unpin" if pinned else "Pin")
        self._refresh_style()

    def set_search_highlight(self, highlighted: bool) -> None:
        highlighted = bool(highlighted)
        if self._search_highlight == highlighted:
            return
        self._search_highlight = highlighted
        self._refresh_style()

    def set_can_pin(self, can_pin: bool) -> None:
        can_pin = bool(can_pin)
        pinned = self._pinned if can_pin else False
        if self._can_pin == can_pin and self._pinned == pinned:
            return
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
            self.preview.clear_source_pixmap()
            self.set_thumbnail_stats_tooltip("")
        elif not self.preview.has_source_pixmap():
            self.preview.setText("No preview")

    def set_processing(self, processing: bool, *, queued: bool = False) -> None:
        self._processing = processing
        self._processing_queued = queued if processing else False
        self.processing_badge.set_queued(self._processing_queued)
        self.processing_badge.setVisible(processing)
        self._refresh_tooltip()
        self._position_processing_badge()
        self._refresh_style()
        self.update()

    def set_isolated_tuning(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._isolated_tuning == enabled:
            return
        self._isolated_tuning = enabled
        self._refresh_execution_label()
        self._refresh_tooltip()
        self._refresh_style()
        self.update()

    def set_bypassed(self, bypassed: bool) -> bool:
        """Show authored bypass as a neutral, non-backend badge."""

        bypassed = bool(bypassed)
        state_changed = self._bypassed != bypassed
        self._bypassed = bypassed
        self._bypass_overlay.setVisible(bypassed)
        if bypassed:
            self._bypass_overlay.raise_()
        presentation_changed = False
        if bypassed:
            presentation_changed = self.set_compute_badge(
                ComputeBadgeKind.BYPASSED,
                tooltip=(
                    "Workflow output forwards this node's exact primary input without "
                    "calling the operation. Its thumbnail remains a "
                    "presentation-only preview of what the node would produce "
                    "if run."
                ),
            )
        elif self._compute_badge_kind is ComputeBadgeKind.BYPASSED:
            presentation_changed = self.set_compute_badge(None)
        if not state_changed and not presentation_changed:
            return False
        self._refresh_tooltip()
        self._refresh_style()
        self.update()
        return True

    def set_bypass_pass_through_positions(
        self,
        input_y: float | None,
        output_y: float | None,
    ) -> None:
        """Place the bypass cue on the graph's primary input/output port axis."""

        self._bypass_overlay.set_pass_through_positions(input_y, output_y)

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
        self.calculate_button.setEnabled(
            self._execution_state not in {"running", EXECUTION_BLOCKED}
        )
        if self._execution_state == EXECUTION_BLOCKED:
            self.calculate_button.setText("Waiting")
        else:
            self.calculate_button.setText(
                "Calculate"
                if self._execution_state == "not_calculated"
                else "Recalculate"
            )
        self._refresh_execution_label()
        self._refresh_tooltip()
        self._refresh_style()
        self.update()

    def _refresh_execution_label(self) -> None:
        visible = (
            self._isolated_tuning
            or self._manual_execution
            or self._execution_state == "stale"
        )
        self.execution_label.setVisible(visible)
        self.execution_label.setText(self._execution_summary() if visible else "")

    def _refresh_tooltip(self) -> None:
        if self._processing:
            self.setToolTip(
                "Processing in background; latest edit is queued."
                if self._processing_queued
                else "Processing in background."
            )
            return
        if self._isolated_tuning:
            self.setToolTip(
                "Tuning this node in isolation. Its downstream branch remains "
                "paused until Apply and continue."
            )
            return
        if self._bypassed:
            self.setToolTip(
                "Workflow output: Bypass. VIPP forwards this node's exact primary "
                "input. "
                "The thumbnail is a presentation-only preview of what this node "
                "would produce if run."
            )
            return
        if self._execution_state == "stale":
            self.setToolTip(
                self._execution_message or "This node's cached result is stale."
            )
            return
        if self._execution_state == EXECUTION_BLOCKED:
            self.setToolTip(
                self._execution_message
                or "Downstream result is stale; waiting for an upstream manual "
                "node to be recalculated."
            )
            return
        if self._manual_execution and self._execution_state == "not_calculated":
            if self._auto_recalculate:
                self.setToolTip(
                    "This manual node has not been calculated; auto "
                    "recalculation is pending."
                )
                return
            self.setToolTip(
                self._execution_message
                or "This manual node has not been calculated. Click Calculate "
                "or use Calculate all."
            )
            return
        self.setToolTip("")

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
            self.preview.clear_source_pixmap()
            self.preview.setAccessibleDescription("")
            return

        thumb = np.ascontiguousarray(thumbnail[..., :3].astype(np.uint8, copy=False))
        h, w = thumb.shape[:2]
        qimage = QImage(thumb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage)
        self.preview.setText("")
        self.preview.set_source_pixmap(pixmap)
        self.preview.setAccessibleDescription("")

    def set_thumbnail_pending(
        self,
        text: str = "Calculating preview…",
        *,
        accessible_description: str = "",
    ) -> None:
        """Show an intentional first-load state without replacing valid pixels."""

        if not self._preview_enabled or self.preview.has_source_pixmap():
            return
        self.preview.setText(str(text or "Calculating preview…"))
        self.preview.setAccessibleDescription(
            str(accessible_description or "").strip()
            or (
                "A complete thumbnail is not available yet. "
                "VIPP is waiting for final thumbnail contrast statistics."
            )
        )

    def set_thumbnail_stats_tooltip(self, tooltip: str = "") -> bool:
        """Keep detailed presentation provenance discoverable without card chrome."""
        detail = str(tooltip or "").strip()
        if detail == self._thumbnail_stats_tooltip:
            return False
        self._thumbnail_stats_tooltip = detail
        self.preview.setToolTip(detail)
        # Final pixmaps clear this description in set_thumbnail(). Pending or
        # unavailable first-load states own it because their inspector is hidden.
        return True

    def set_metadata_summary(self, text: str) -> None:
        self.metadata_label.setText(text)

    def set_subtitle(self, text: str, tooltip: str | None = None) -> None:
        self.subtitle_label.set_full_text(text, tooltip)

    def set_compute_badge(
        self,
        kind: ComputeBadgeKind | str | None,
        *,
        tooltip: str = "",
        stale: bool = False,
    ) -> bool:
        """Show one compact, presentation-only accepted-compute badge.

        The graph deliberately does not interpret execution reports. Its owner
        supplies an already resolved identity and may mark it stale while a
        replacement result is pending. Passing ``None`` clears the badge. The
        return value reports whether the card presentation changed.
        """
        if kind is None:
            if (
                self._compute_badge_kind is None
                and not self._compute_badge_stale
                and not self._compute_badge_tooltip
            ):
                return False
            self._compute_badge_kind = None
            self._compute_badge_stale = False
            self._compute_badge_tooltip = ""
            self.compute_badge.clear()
            self.compute_badge.setToolTip("")
            self.title_row.setToolTip("")
            self.compute_badge.setAccessibleName("")
            self.compute_badge.hide()
            return True

        resolved = _coerce_compute_badge_kind(kind)
        is_stale = bool(stale)
        detail = str(tooltip or "").strip()
        if (
            self._compute_badge_kind is resolved
            and self._compute_badge_stale == is_stale
            and self._compute_badge_tooltip == detail
        ):
            return False
        label = _COMPUTE_BADGE_LABELS[resolved]
        if is_stale:
            background, foreground, border = ("#292524", "#a8a29e", "#78716c")
            visible_tooltip = "Previous result (stale)."
            if detail:
                visible_tooltip = f"{visible_tooltip} {detail}"
        else:
            background, foreground, border = _COMPUTE_BADGE_COLORS[resolved]
            visible_tooltip = detail

        self._compute_badge_kind = resolved
        self._compute_badge_stale = is_stale
        self._compute_badge_tooltip = detail
        self.compute_badge.setText(label)
        self.compute_badge.setToolTip(visible_tooltip)
        # Mirror the explanation on the row as well, so it remains available
        # while moving between the compact badge and its title.
        self.title_row.setToolTip(visible_tooltip)
        accessible_state = "stale previous result" if is_stale else "used"
        self.compute_badge.setAccessibleName(
            "Execution mode: Bypassed"
            if resolved is ComputeBadgeKind.BYPASSED
            else f"Compute {accessible_state}: {label}"
        )
        self.compute_badge.setStyleSheet(
            "QLabel#NodeComputeBadge {"
            f" background: {background}; color: {foreground};"
            f" border: 1px solid {border}; border-radius: 7px;"
            " font-size: 9px; font-weight: 650; padding: 1px 5px;"
            "}"
        )
        self.compute_badge.show()
        return True

    def set_optimization_hint(self, tooltip: str = "") -> bool:
        """Show one derived, presentation-only GPU optimization hint."""
        detail = str(tooltip or "").strip()
        if detail == self._optimization_hint:
            return False
        self._optimization_hint = detail
        if not detail:
            self.optimization_badge.setToolTip("")
            self.optimization_badge.setAccessibleName("")
            self.optimization_badge.hide()
            return True
        self.optimization_badge.setToolTip(detail)
        self.optimization_badge.setAccessibleName(
            "GPU eligibility tip; select this node to review the suggested change"
        )
        self.optimization_badge.show()
        return True

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
        if self._execution_state == EXECUTION_BLOCKED:
            border = BLOCKED_EXECUTION_ACCENT
            background = "#21170f"
            if self._selected:
                border_width = 3
            if self._pinned:
                border = "#facc15"
                border_width = 4
        elif self._execution_state == "stale":
            border = STALE_EXECUTION_ACCENT
            background = "#2a2416"
            if self._selected:
                border_width = 3
            if self._pinned:
                border = "#facc15"
                border_width = 4
        elif self._manual_execution:
            if self._execution_state == "ready":
                border = "#22c55e"
                background = "#182a20"
            elif self._execution_state == "not_calculated":
                border = STALE_EXECUTION_ACCENT
                background = "#2a2416"
            elif self._execution_state == "error":
                border = "#ef4444"
                background = "#2f1d1d"
            if self._selected:
                border_width = 3
            if self._pinned:
                border = "#facc15"
                border_width = 4
        if (
            self._isolated_tuning
            and self._execution_state != "error"
            and not self._pinned
        ):
            border = STALE_EXECUTION_ACCENT
            border_width = max(border_width, 3)
            background = "#2a2416"
        if self._processing:
            background = "#303640"
            if not self._pinned and not self._selected:
                border = "#94a3b8"
            if self._processing_queued and not self._pinned:
                border = "#f59e0b"
        if self._search_highlight and not self._selected and not self._pinned:
            border = "#38bdf8"
            border_width = max(border_width, 3)
            background = "#1e2e38"
        accent_color = self._category_color
        category_background = self._category_tint
        category_color = self._category_color
        if self._execution_state == EXECUTION_BLOCKED:
            accent_color = BLOCKED_EXECUTION_ACCENT
            category_background = "#431407"
            category_color = "#fdba74"
        elif self._execution_state == "stale":
            accent_color = STALE_EXECUTION_ACCENT
            category_background = "#78350f"
            category_color = "#fde68a"
        elif self._manual_execution:
            if self._execution_state == "ready":
                accent_color = "#22c55e"
                category_background = "#064e3b"
                category_color = "#bbf7d0"
            elif self._execution_state == "not_calculated":
                accent_color = STALE_EXECUTION_ACCENT
                category_background = "#78350f"
                category_color = "#fde68a"
            elif self._execution_state == "error":
                accent_color = "#ef4444"
                category_background = "#7f1d1d"
                category_color = "#fecaca"
        if self._isolated_tuning and self._execution_state != "error":
            accent_color = STALE_EXECUTION_ACCENT
            category_background = "#78350f"
            category_color = "#fde68a"
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
        if self._isolated_tuning:
            return "Tuning in isolation"
        if self._execution_state == "ready":
            return (
                "Auto result ready" if self._auto_recalculate else "Cached result ready"
            )
        if self._execution_state == "running":
            return "Calculating..."
        if self._execution_state == "stale":
            if "paused" in self._execution_message.casefold():
                return "Stale; downstream paused"
            return (
                "Auto recalculation pending"
                if self._auto_recalculate
                else "Stale cached result"
            )
        if self._execution_state == EXECUTION_BLOCKED:
            return "Stale; waiting upstream"
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


class GraphNoteItem(QGraphicsTextItem):
    """Movable canvas annotation that does not participate in pipeline execution."""

    DEFAULT_WIDTH = 240.0

    def __init__(
        self,
        note_id: str,
        text: str,
        width: float = DEFAULT_WIDTH,
        *,
        attached_node: str = "",
    ):
        super().__init__()
        self.note_id = str(note_id)
        self.attached_node = str(attached_node or "")
        self._drag_start_pos: QPointF | None = None
        self.setPlainText(str(text or ""))
        self.setTextWidth(max(float(width), 140.0))
        self.document().setDocumentMargin(8.0)
        font = QFont()
        font.setPointSizeF(9.0)
        self.setFont(font)
        self.setDefaultTextColor(QColor("#f8fafc"))
        self.setZValue(80)
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCursor(Qt.OpenHandCursor)

    def set_text(self, text: str) -> None:
        if text == self.toPlainText():
            return
        self.setPlainText(str(text))

    def paint(self, painter, option, widget=None):  # noqa: N802
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.boundingRect()
        fill = QColor(51, 65, 85, 230)
        border = QColor("#fbbf24") if self.isSelected() else QColor("#64748b")
        painter.setBrush(fill)
        painter.setPen(QPen(border, 1.4 if self.isSelected() else 1.0))
        painter.drawRoundedRect(rect.adjusted(0.5, 0.5, -0.5, -0.5), 5.0, 5.0)
        super().paint(painter, option, widget)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            view = _view_for_scene(self.scene())
            if view is not None:
                view.scene.clearSelection()
                view._clear_node_selection()
            self.setSelected(True)
            self._drag_start_pos = QPointF(self.pos())
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        start_pos = self._drag_start_pos
        super().mouseReleaseEvent(event)
        if event.button() == Qt.LeftButton:
            self.setCursor(Qt.OpenHandCursor)
            self._drag_start_pos = None
            if start_pos is not None and not _points_close(start_pos, self.pos()):
                view = _view_for_scene(self.scene())
                if view is not None:
                    view.note_moved.emit(self.note_id, start_pos, QPointF(self.pos()))

    def mouseDoubleClickEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            view = _view_for_scene(self.scene())
            if view is not None:
                view.note_edit_requested.emit(self.note_id)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):  # noqa: N802
        view = _view_for_scene(self.scene())
        if view is None:
            return
        menu = QMenu()
        edit_action = menu.addAction("Edit note...")
        delete_action = menu.addAction("Delete note")
        action = _exec_menu(menu, event.screenPos())
        if action == edit_action:
            view.note_edit_requested.emit(self.note_id)
        elif action == delete_action:
            view.note_delete_requested.emit(self.note_id)


class PortItem(QGraphicsEllipseItem):
    """Clickable node port used for graph connections."""

    radius = 6.0
    hover_radius = 8.0
    target_radius = 10.0
    label_gap = 7.0

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
        self._persistent_label_visible = False
        self._persistent_label_max_width = 86.0
        self.label_item = QGraphicsTextItem("", self)
        label_font = QFont()
        label_font.setPointSizeF(8.5)
        self.label_item.setFont(label_font)
        self.label_item.setDefaultTextColor(QColor("#dbe4f0"))
        self.label_item.setAcceptedMouseButtons(Qt.NoButton)
        self.label_item.setAcceptHoverEvents(False)
        self.label_item.setZValue(2)
        self.label_item.hide()
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
        self._refresh_persistent_label()
        self._refresh_style()

    def set_persistent_label_visible(
        self,
        visible: bool,
        *,
        maximum_width: float,
    ) -> None:
        self._persistent_label_visible = bool(visible)
        self._persistent_label_max_width = max(float(maximum_width), 24.0)
        self._refresh_persistent_label()

    def preferred_persistent_label_width(self, maximum_width: float) -> float:
        """Return the rendered label width, capped for practical node sizes."""
        full_label = str(self.label or "").strip()
        if not full_label:
            return 0.0
        metrics = QFontMetricsF(self.label_item.font())
        # QGraphicsTextItem's document contributes a small horizontal margin.
        natural_width = metrics.horizontalAdvance(full_label) + 8.0
        return min(natural_width, max(float(maximum_width), 24.0))

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
                "#fbbf24" if self._tunnel_highlight_role == "source" else "#93c5fd"
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
        self._position_persistent_label()
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

    def _refresh_persistent_label(self) -> None:
        full_label = str(self.label or "").strip()
        visible = self._persistent_label_visible and bool(full_label)
        if not visible:
            self.label_item.hide()
            self.label_item.setPlainText("")
            self.label_item.setToolTip("")
            return
        metrics = QFontMetricsF(self.label_item.font())
        elided = metrics.elidedText(
            full_label,
            Qt.ElideRight,
            int(self._persistent_label_max_width),
        )
        self.label_item.setPlainText(elided)
        self.label_item.setToolTip(full_label)
        self._position_persistent_label()
        self.label_item.show()

    def _position_persistent_label(self) -> None:
        if not self.label_item.isVisible() and not self.label_item.toPlainText():
            return
        rect = self.label_item.boundingRect()
        y = -rect.height() / 2.0
        if self.kind == "input":
            x = self.radius + self.label_gap
        else:
            x = -self.radius - self.label_gap - rect.width()
        self.label_item.setPos(x, y)

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
    PORT_EDGE_INSET = 42.0
    PORT_ROW_SPACING = 24.0
    PORT_LABEL_CENTER_GAP = 24.0
    PORT_LABEL_MAX_WIDTH = 140.0
    PORT_LABEL_CONTENT_GAP = 8.0

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
        self._port_label_mode = PortLabelMode.AMBIGUOUS_ONLY
        self.output_ports: list[PortItem] = []
        self._drag_start_scene: QPointF | None = None
        self._drag_start_pos: QPointF | None = None
        self._drag_group_start_positions: dict[str, QPointF] = {}
        self._dragging = False
        self._press_was_preview = False
        self._press_preserved_group = False
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setCacheMode(QGraphicsItem.DeviceCoordinateCache)

    def refresh_ports(self) -> None:
        if self._has_input and self.input_type is not None:
            self._ensure_input_ports()
        if self._has_output:
            self._ensure_output_ports()
        self._update_card_minimum_height()
        self._update_card_port_label_gutters()
        rect = self.boundingRect()
        if self._has_input and self.input_type is not None:
            top = rect.top() + self.PORT_EDGE_INSET
            bottom = rect.bottom() - self.PORT_EDGE_INSET
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
            top = rect.top() + self.PORT_EDGE_INSET
            bottom = rect.bottom() - self.PORT_EDGE_INSET
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
        card = self._card()
        if card is not None:
            primary_input_y = (
                float(self.input_ports[0].pos().y())
                if self.input_ports
                else None
            )
            sole_output_y = (
                float(self.output_ports[0].pos().y())
                if len(self.output_ports) == 1
                else None
            )
            card.set_bypass_pass_through_positions(
                primary_input_y,
                sole_output_y,
            )
        self._refresh_persistent_port_labels(rect)

    @property
    def port_label_mode(self) -> PortLabelMode:
        return self._port_label_mode

    def set_port_label_mode(self, mode: PortLabelMode | str) -> None:
        resolved = _coerce_port_label_mode(mode)
        if resolved == self._port_label_mode:
            return
        self._port_label_mode = resolved
        self.refresh_ports()
        for connection in self.connections:
            connection.update_path()

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

    def _persistent_port_labels_are_visible(self) -> bool:
        if self._port_label_mode == PortLabelMode.HIDE_ALL:
            return False
        if self._port_label_mode == PortLabelMode.SHOW_ALL:
            return True
        return len(self.input_ports) > 1 or len(self.output_ports) > 1

    def _refresh_persistent_port_labels(self, rect: QRectF) -> None:
        visible = self._persistent_port_labels_are_visible()
        maximum_width = min(
            max(
                (
                    rect.width()
                    - 2.0 * (PortItem.radius + PortItem.label_gap)
                    - self.PORT_LABEL_CENTER_GAP
                )
                / 2.0,
                24.0,
            ),
            self.PORT_LABEL_MAX_WIDTH,
        )
        for port in (*self.input_ports, *self.output_ports):
            port.set_persistent_label_visible(
                visible,
                maximum_width=maximum_width,
            )

    def _update_card_port_label_gutters(self) -> None:
        card = self._card()
        if card is None:
            return
        if not self._persistent_port_labels_are_visible():
            card.set_port_label_gutters(0.0, 0.0)
            return

        input_width = max(
            (
                port.preferred_persistent_label_width(self.PORT_LABEL_MAX_WIDTH)
                for port in self.input_ports
            ),
            default=0.0,
        )
        output_width = max(
            (
                port.preferred_persistent_label_width(self.PORT_LABEL_MAX_WIDTH)
                for port in self.output_ports
            ),
            default=0.0,
        )
        base_left, _top, base_right, _bottom = card.BASE_CONTENT_MARGINS

        def gutter(label_width: float, base_margin: int) -> float:
            if label_width <= 0.0:
                return 0.0
            label_end = (
                PortItem.radius
                + PortItem.label_gap
                + label_width
                + self.PORT_LABEL_CONTENT_GAP
            )
            return max(label_end - float(base_margin), 0.0)

        card.set_port_label_gutters(
            gutter(input_width, base_left),
            gutter(output_width, base_right),
        )

    def _update_card_minimum_height(self) -> None:
        card = self._card()
        if card is None:
            return
        port_rows = max(len(self.input_ports), len(self.output_ports), 1)
        required = int(
            2.0 * self.PORT_EDGE_INSET + max(port_rows - 1, 0) * self.PORT_ROW_SPACING
        )
        card.setMinimumHeight(required if port_rows > 1 else 0)
        card.updateGeometry()
        card.adjustSize()

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
            view = _view_for_scene(self.scene())
            if view is not None:
                self._press_preserved_group = view._handle_node_press(
                    self.node_id,
                    event.modifiers(),
                    preserve_group_for_drag=True,
                )
                self._drag_group_start_positions = view._selected_node_start_positions(
                    self.node_id
                )
            else:
                self.setSelected(True)
                self._drag_group_start_positions = {self.node_id: QPointF(self.pos())}
            if card is not None:
                card.setCursor(Qt.ClosedHandCursor)
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
                view = _view_for_scene(self.scene())
                if view is not None:
                    view._move_selected_nodes_during_drag(
                        self._drag_group_start_positions,
                        delta,
                    )
                    if len(self._drag_group_start_positions) == 1:
                        view.update_existing_node_insert_preview(
                            self.node_id,
                            event.scenePos(),
                        )
                    else:
                        view._clear_all_insert_previews()
                else:
                    self.setOpacity(self.DRAG_OPACITY)
                    self.setPos(self._drag_start_pos + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._drag_start_scene is not None and event.button() == Qt.LeftButton:
            start_pos = self._drag_start_pos
            end_pos = QPointF(self.pos())
            group_start_positions = {
                node_id: QPointF(position)
                for node_id, position in self._drag_group_start_positions.items()
            }
            moved = (
                self._dragging
                and start_pos is not None
                and (
                    abs(end_pos.x() - start_pos.x()) + abs(end_pos.y() - start_pos.y())
                )
                > 0.001
            )
            card = self._card()
            if card is not None:
                card.setCursor(Qt.OpenHandCursor)
            self._drag_start_scene = None
            self._drag_start_pos = None
            self._drag_group_start_positions = {}
            self._dragging = False
            self._press_was_preview = False
            if moved:
                view = _view_for_scene(self.scene())
                if view is not None:
                    if len(group_start_positions) > 1:
                        view._finish_selected_node_drag(group_start_positions)
                        end_positions = {
                            node_id: QPointF(view._proxies[node_id].pos())
                            for node_id in group_start_positions
                            if node_id in view._proxies
                        }
                        view.nodes_moved.emit(
                            group_start_positions,
                            end_positions,
                        )
                        self._press_preserved_group = False
                        event.accept()
                        return
                    was_loose = not self.connections
                    # Resolve a highlighted insertion before finishing the drag.
                    # Finishing reroutes ordinary wires around the newly placed
                    # node; doing that first moves the green drop target away
                    # from the release point and turns a valid splice into a
                    # plain layout move.
                    inserted = view.release_existing_node_insert(
                        self.node_id,
                        start_pos,
                        end_pos,
                        event.scenePos(),
                    )
                    view._finish_selected_node_drag(group_start_positions)
                    if not inserted:
                        if was_loose:
                            view.finish_loose_node_drag(
                                self.node_id,
                                start_pos,
                                end_pos,
                            )
                        view.node_moved.emit(self.node_id, start_pos, end_pos)
            else:
                view = _view_for_scene(self.scene())
                if view is not None:
                    view._finish_selected_node_drag(group_start_positions)
                    if self._press_preserved_group:
                        view._select_node(self.node_id)
                else:
                    self.setOpacity(1.0)
            self._press_preserved_group = False
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
        obstacles = view.connection_obstacle_rects(self) if view is not None else ()
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
    node_selection_changed = Signal(object, str)
    node_delete_requested = Signal(str)
    nodes_delete_requested = Signal(object)
    nodes_copy_requested = Signal(object)
    paste_requested = Signal(QPointF)
    node_paste_values_requested = Signal(str)
    node_duplicate_requested = Signal(str)
    node_code_requested = Signal(str)
    node_note_requested = Signal(str)
    node_isolation_requested = Signal(str)
    node_bypass_requested = Signal(str, bool)
    node_moved = Signal(str, object, object)
    nodes_moved = Signal(object, object)
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
    tunnel_reroute_requested = Signal(str, str, int)
    tunnel_insert_requested = Signal(str, QPointF)
    tunnel_node_insert_requested = Signal(str, str, QPointF)
    tunnel_node_splice_requested = Signal(str, str, object, object)
    note_moved = Signal(str, object, object)
    note_edit_requested = Signal(str)
    note_delete_requested = Signal(str)
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
        self._primary_node_id: str | None = None
        self._node_press_dispatch_depth = 0
        self._clipboard_can_paste = False
        self._clipboard_single_operation_id = ""
        self._clipboard_single_title = ""
        self._isolated_tuning_node_id: str | None = None
        self._connections: list[ConnectionItem] = []
        self._pending_source: PortItem | None = None
        self._pending_wire: PendingConnectionItem | None = None
        self._highlighted_input_port: PortItem | None = None
        self._highlighted_connection: ConnectionItem | None = None
        self._highlighted_connection_state: str | None = None
        self._highlighted_connection_operation: str | None = None
        self._highlighted_tunnel_insert_port: PortItem | None = None
        self._highlighted_tunnel_insert_name = ""
        self._highlighted_tunnel_insert_state: str | None = None
        self._connection_insert_validator: (
            Callable[
                [str, tuple[str, str, int, int]],
                tuple[str, str],
            ]
            | None
        ) = None
        self._node_bypass_state_resolver: (
            Callable[[str], tuple[bool, bool, str]] | None
        ) = None
        self._tunnel_reroute_validator: (
            Callable[
                [str, str, int],
                tuple[str, str],
            ]
            | None
        ) = None
        self._tunnel_insert_validator: (
            Callable[
                [str, str, str],
                tuple[str, str],
            ]
            | None
        ) = None
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
        self._pending_tunnel_name = ""
        self._pending_tunnel_source: PortItem | None = None
        self._pending_tunnel_wire: PendingConnectionItem | None = None
        self._highlighted_tunnel_output_port: PortItem | None = None
        self._tunnel_reroute_dragging = False
        self._search_match_node_ids: set[str] = set()
        self._notes: dict[str, GraphNoteItem] = {}
        self._port_label_mode = PortLabelMode.AMBIGUOUS_ONLY

    def build_graph(
        self,
        nodes,
        connections,
        positions=None,
        *,
        output_tunnels=(),
        notes=(),
        preserve_view: bool = False,
    ) -> None:
        preserved_center = self.mapToScene(self.viewport().rect().center())
        preserved_transform = QTransform(self.transform())
        preserved_base_transform = QTransform(self._base_transform)
        preserved_zoom = float(self._zoom_percent)
        self.clear_node_processing()
        self._clear_tunnel_insert_preview()
        self.scene.clear()
        self._proxies.clear()
        self._cards.clear()
        self._primary_node_id = None
        self._connections.clear()
        self._notes.clear()
        self._pending_source = None
        self._pending_wire = None
        self._highlighted_input_port = None
        self._pending_tunnel_name = ""
        self._pending_tunnel_source = None
        self._pending_tunnel_wire = None
        self._highlighted_tunnel_output_port = None
        self._tunnel_reroute_dragging = False
        self._clear_connection_insert_preview()
        self._tunnel_source_ports.clear()
        self._tunnel_subscriber_ports.clear()
        self._hover_tunnel_name = ""
        self._search_match_node_ids.clear()

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
        self.set_notes(notes)

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

    @property
    def port_label_mode(self) -> PortLabelMode:
        return self._port_label_mode

    def set_port_label_mode(self, mode: PortLabelMode | str) -> None:
        """Show persistent port labels without repositioning graph nodes."""
        resolved = _coerce_port_label_mode(mode)
        if resolved == self._port_label_mode:
            return
        self._port_label_mode = resolved
        affected_rect = QRectF()
        for proxy in self._proxies.values():
            before = proxy.sceneBoundingRect()
            proxy.set_port_label_mode(resolved)
            after = proxy.sceneBoundingRect()
            affected_rect = affected_rect.united(before).united(after)
            self._ensure_scene_space_for_rect(after)
        self._mark_graph_geometry_changed()
        self.reroute_connections(
            affected_rect=affected_rect if affected_rect.isValid() else None
        )
        self.scene.update()

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

    def center_graph(self) -> bool:
        """Center on stable workflow content without changing the zoom."""
        graph_rect = self._graph_content_rect()
        target = graph_rect.center() if graph_rect is not None else QPointF()
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        target_viewport_rect = QRectF(
            target.x() - visible_rect.width() / 2.0,
            target.y() - visible_rect.height() / 2.0,
            visible_rect.width(),
            visible_rect.height(),
        )
        self._ensure_scene_space_for_rect(target_viewport_rect)
        self.centerOn(target)
        return graph_rect is not None

    def _graph_content_rect(self) -> QRectF | None:
        """Return node bounds, falling back to notes on a node-free canvas."""
        items = list(self._proxies.values())
        if not items:
            items = list(self._notes.values())
        if not items:
            return None
        rect = QRectF(items[0].sceneBoundingRect())
        for item in items[1:]:
            rect = rect.united(item.sceneBoundingRect())
        return rect

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

    def note_positions(self) -> dict[str, tuple[float, float]]:
        """Return the current scene position of each graph note by id."""
        positions: dict[str, tuple[float, float]] = {}
        for note_id, item in self._notes.items():
            pos = item.pos()
            positions[note_id] = (float(pos.x()), float(pos.y()))
        return positions

    def add_note(
        self,
        note_id: str,
        text: str,
        position: QPointF,
        *,
        width: float = GraphNoteItem.DEFAULT_WIDTH,
        attached_node: str = "",
    ) -> None:
        if not note_id:
            return
        existing = self._notes.get(note_id)
        if existing is not None:
            existing.set_text(text)
            existing.attached_node = str(attached_node or "")
            existing.setTextWidth(max(float(width), 140.0))
            existing.setPos(position)
            self._ensure_scene_space_for_rect(existing.sceneBoundingRect())
            return
        item = GraphNoteItem(
            note_id,
            text,
            width,
            attached_node=attached_node,
        )
        self.scene.addItem(item)
        item.setPos(position)
        self._notes[note_id] = item
        self._ensure_scene_space_for_rect(item.sceneBoundingRect())

    def set_notes(self, notes) -> None:
        for item in list(self._notes.values()):
            if item.scene() is not None:
                self.scene.removeItem(item)
        self._notes.clear()
        for note in notes or ():
            if isinstance(note, Mapping):
                note_id = str(note.get("id", "")).strip()
                text = str(note.get("text", ""))
                position = _to_pointf(note.get("position"))
                width = float(note.get("width", GraphNoteItem.DEFAULT_WIDTH))
                attached_node = str(note.get("attached_node", "") or "")
            else:
                note_id = str(getattr(note, "id", "")).strip()
                text = str(getattr(note, "text", ""))
                position = _to_pointf(getattr(note, "position", None))
                width = float(getattr(note, "width", GraphNoteItem.DEFAULT_WIDTH))
                attached_node = str(getattr(note, "attached_node", "") or "")
            if position is None:
                continue
            self.add_note(
                note_id,
                text,
                position,
                width=width,
                attached_node=attached_node,
            )

    def set_note_text(self, note_id: str, text: str) -> None:
        item = self._notes.get(note_id)
        if item is not None:
            item.set_text(text)

    def remove_note(self, note_id: str) -> None:
        item = self._notes.pop(note_id, None)
        if item is not None and item.scene() is not None:
            self.scene.removeItem(item)

    def select_note(self, note_id: str) -> None:
        item = self._notes.get(note_id)
        if item is None:
            return
        self._clear_node_selection()
        for other in self._notes.values():
            other.setSelected(other is item)
        self._ensure_scene_space_for_rect(item.sceneBoundingRect())
        self.centerOn(item.sceneBoundingRect().center())

    def suggest_note_position(self) -> QPointF:
        center = self.mapToScene(self.viewport().rect().center())
        return QPointF(center.x() - 120.0, center.y() - 45.0)

    def suggest_note_position_for_node(self, node_id: str) -> QPointF:
        rect = self.node_scene_rect(node_id)
        if rect is None:
            return self.suggest_note_position()
        return QPointF(rect.right() + 24.0, rect.top() + 18.0)

    def _move_attached_notes(self, node_id: str, delta: QPointF) -> None:
        if abs(delta.x()) < 0.001 and abs(delta.y()) < 0.001:
            return
        moved_rect = QRectF()
        for item in self._notes.values():
            if item.attached_node != node_id:
                continue
            before = item.sceneBoundingRect()
            item.setPos(item.pos() + delta)
            moved_rect = moved_rect.united(before.united(item.sceneBoundingRect()))
        if moved_rect.isValid():
            self._ensure_scene_space_for_rect(moved_rect)

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

    def overlapping_node_pairs(self) -> list[tuple[str, str]]:
        """Return card pairs that currently overlap in scene coordinates."""
        node_ids = list(self._proxies)
        overlaps: list[tuple[str, str]] = []
        for index, first_id in enumerate(node_ids):
            first_rect = self._proxies[first_id].sceneBoundingRect()
            for second_id in node_ids[index + 1 :]:
                second_rect = self._proxies[second_id].sceneBoundingRect()
                intersection = first_rect.intersected(second_rect)
                if intersection.width() > 0.5 and intersection.height() > 0.5:
                    overlaps.append((first_id, second_id))
        return overlaps

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
            old_pos = QPointF(proxy.pos())
            proxy.setPos(point)
            self._move_attached_notes(node_id, point - old_pos)
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
        delta = scene_pos - rect.center()
        proxy.setPos(proxy.pos() + delta)
        self._move_attached_notes(node_id, delta)
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
            self._move_attached_notes(node_id, delta)
            moved_rect = moved_rect.united(proxy.sceneBoundingRect())
        if moved_rect.isValid():
            self._ensure_scene_space_for_rect(moved_rect)
            self._mark_graph_geometry_changed()
            self.reroute_connections(affected_rect=moved_rect)

    def set_clipboard_state(
        self,
        can_paste: bool,
        *,
        copied_single_operation_id: str = "",
        copied_single_title: str = "",
    ) -> None:
        """Update menu availability from clipboard data inspected by the owner."""
        self._clipboard_can_paste = bool(can_paste)
        self._clipboard_single_operation_id = str(
            copied_single_operation_id or ""
        ).strip()
        self._clipboard_single_title = str(copied_single_title or "").strip()

    def viewport_center_scene_position(self) -> QPointF:
        """Return the scene point used for keyboard paste."""
        return self.mapToScene(self.viewport().rect().center())

    def _selected_node_start_positions(
        self,
        dragged_node_id: str,
    ) -> dict[str, QPointF]:
        selected = self.selected_node_ids()
        if dragged_node_id not in selected:
            selected = (dragged_node_id,)
        return {
            node_id: QPointF(self._proxies[node_id].pos())
            for node_id in selected
            if node_id in self._proxies
        }

    def _move_selected_nodes_during_drag(
        self,
        start_positions: Mapping[str, QPointF],
        delta: QPointF,
    ) -> None:
        for node_id, start_position in start_positions.items():
            proxy = self._proxies.get(node_id)
            if proxy is None:
                continue
            proxy.setOpacity(NodeProxy.DRAG_OPACITY)
            old_position = QPointF(proxy.pos())
            new_position = QPointF(start_position + delta)
            if _points_close(old_position, new_position):
                continue
            proxy.setPos(new_position)
            self._move_attached_notes(node_id, new_position - old_position)

    def _finish_selected_node_drag(
        self,
        start_positions: Mapping[str, QPointF],
    ) -> None:
        if not start_positions:
            return
        moved_rect = QRectF()
        for node_id, start_position in start_positions.items():
            proxy = self._proxies.get(node_id)
            if proxy is None:
                continue
            local_rect = proxy.boundingRect()
            old_rect = QRectF(local_rect).translated(start_position)
            moved_rect = moved_rect.united(old_rect.united(proxy.sceneBoundingRect()))
        self._apply_graph_focus_opacity()
        if moved_rect.isValid():
            self._ensure_scene_space_for_rect(moved_rect)
            self._mark_graph_geometry_changed()
            self.reroute_connections(affected_rect=moved_rect)

    def set_connection_insert_validator(
        self,
        validator: Callable[[str, tuple[str, str, int, int]], tuple[str, str]] | None,
    ) -> None:
        self._connection_insert_validator = validator

    def set_node_bypass_state_resolver(
        self,
        resolver: Callable[[str], tuple[bool, bool, str]] | None,
    ) -> None:
        """Set the owner callback for live topology-aware bypass availability."""

        self._node_bypass_state_resolver = resolver

    def set_tunnel_reroute_validator(
        self,
        validator: Callable[[str, str, int], tuple[str, str]] | None,
    ) -> None:
        """Set the topology-aware validator used during tunnel source drags."""
        self._tunnel_reroute_validator = validator

    def set_tunnel_insert_validator(
        self,
        validator: Callable[[str, str, str], tuple[str, str]] | None,
    ) -> None:
        """Set the validator used for palette drops on source tunnel badges."""
        self._tunnel_insert_validator = validator

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
        if proxy is None or not self._is_loose_node(node_id):
            self._clear_all_insert_previews()
            return
        tunnel_name = self._update_tunnel_insert_preview(
            scene_pos,
            proxy.operation_id,
            node_id,
        )
        if tunnel_name:
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
        if proxy is None or not self._is_loose_node(node_id):
            self._clear_all_insert_previews()
            return False
        tunnel_name = self._update_tunnel_insert_preview(
            scene_pos,
            proxy.operation_id,
            node_id,
        )
        if tunnel_name:
            if self._highlighted_tunnel_insert_state != "compatible":
                self._clear_all_insert_previews()
                return False
            self._clear_all_insert_previews()
            self.tunnel_node_splice_requested.emit(
                node_id,
                tunnel_name,
                QPointF(old_pos),
                QPointF(new_pos),
            )
            return True
        self._update_connection_insert_preview(proxy.operation_id, scene_pos)
        connection_key = self._connection_key(self._highlighted_connection)
        state = self._highlighted_connection_state
        if connection_key is None or state == "incompatible":
            self._clear_all_insert_previews()
            return False

        self._clear_all_insert_previews()
        self.node_splice_requested.emit(
            node_id,
            connection_key,
            QPointF(old_pos),
            QPointF(new_pos),
        )
        return True

    def _is_loose_node(self, node_id: str) -> bool:
        proxy = self._proxies.get(node_id)
        if proxy is None or proxy.connections:
            return False
        tunnel_ports = tuple(self._tunnel_source_ports.values()) + tuple(
            port for ports in self._tunnel_subscriber_ports.values() for port in ports
        )
        return all(port.node_id != node_id for port in tunnel_ports)

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
        corridor = (
            QRectF(start, end)
            .normalized()
            .adjusted(
                -180.0,
                -240.0,
                180.0,
                240.0,
            )
        )
        return corridor.united(connection.sceneBoundingRect())

    def add_node(self, node, position: QPointF) -> None:
        card = NodeCard(
            node.id,
            node.title,
            node.category,
            can_pin=node.output_type in PINNABLE_OUTPUT_TYPES,
        )
        card.set_bypassed(getattr(node, "execution_mode", "run") == "bypass")
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
        proxy.set_port_label_mode(self._port_label_mode)
        proxy.set_input_ports(
            _node_input_port_count(node),
            _node_input_port_labels(node),
            _node_input_port_colors(node),
            _node_input_port_types(node),
        )
        proxy.set_output_ports(
            _node_output_port_count(node),
            _node_output_port_labels(node),
            _node_output_port_colors(node),
            _node_output_port_types(node),
        )
        proxy.refresh_ports()
        self._cards[node.id] = card
        self._proxies[node.id] = proxy
        if node.id in self._search_match_node_ids:
            card.set_search_highlight(True)
        self._apply_graph_focus_opacity()
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
        self._cancel_pending_tunnel_reroute()
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

    def begin_tunnel_reroute(self, name: str, scene_pos: QPointF) -> None:
        """Arm a source-badge drag for one named tunnel."""
        tunnel_name = str(name or "").strip()
        source = self._tunnel_source_ports.get(tunnel_name)
        if source is None:
            return
        self._cancel_pending_connection()
        self._cancel_pending_tunnel_reroute()
        self.highlight_tunnel(tunnel_name, sticky=True)
        self._pending_tunnel_name = tunnel_name
        self._pending_tunnel_source = source
        self._tunnel_reroute_dragging = False
        self.status_message.emit(
            f"Drag tunnel '{tunnel_name}' to another compatible output."
        )

    def update_pending_tunnel_reroute(
        self,
        scene_pos: QPointF,
        *,
        dragging: bool,
    ) -> None:
        if self._pending_tunnel_source is None:
            return
        self._tunnel_reroute_dragging = self._tunnel_reroute_dragging or dragging
        if self._pending_tunnel_wire is None:
            self._pending_tunnel_wire = PendingConnectionItem(
                self._pending_tunnel_source,
                scene_pos,
            )
            self.scene.addItem(self._pending_tunnel_wire)
        else:
            self._pending_tunnel_wire.update_end(scene_pos)
        self._update_tunnel_reroute_feedback(scene_pos)

    def release_tunnel_reroute(self, scene_pos: QPointF) -> None:
        tunnel_name = self._pending_tunnel_name
        target = self._output_port_at(scene_pos)
        state, _message = self._tunnel_reroute_target_feedback(target)
        dragged = self._tunnel_reroute_dragging
        self._cancel_pending_tunnel_reroute()
        if not dragged or target is None or state != "compatible":
            return
        self.tunnel_reroute_requested.emit(
            tunnel_name,
            target.node_id,
            target.port_index,
        )

    def focus_node(self, node_id: str) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        self._select_node(node_id)
        self._ensure_scene_space_for_rect(proxy.sceneBoundingRect())
        self.centerOn(proxy.sceneBoundingRect().center())

    def set_search_matches(self, node_ids) -> None:
        self._search_match_node_ids = {
            str(node_id) for node_id in node_ids or () if str(node_id) in self._proxies
        }
        for node_id, card in self._cards.items():
            card.set_search_highlight(node_id in self._search_match_node_ids)
        self._apply_graph_focus_opacity()

    def clear_search_matches(self) -> None:
        if not self._search_match_node_ids:
            return
        self._search_match_node_ids.clear()
        for card in self._cards.values():
            card.set_search_highlight(False)
        self._apply_graph_focus_opacity()

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
        for name, source in self._tunnel_source_ports.items():
            role = ""
            if active:
                role = "source" if name == active else "dimmed"
            source.set_tunnel_highlight_role(role)
        for name, ports in self._tunnel_subscriber_ports.items():
            for port in ports:
                role = ""
                if active:
                    role = "subscriber" if name == active else "dimmed"
                port.set_tunnel_highlight_role(role)
        self._apply_graph_focus_opacity()

    def _active_tunnel_node_ids(self) -> set[str]:
        active = self._effective_tunnel_highlight()
        if not active:
            return set()
        active_node_ids: set[str] = set()
        source = self._tunnel_source_ports.get(active)
        if source is not None:
            active_node_ids.add(source.node_id)
        for port in self._tunnel_subscriber_ports.get(active, ()):
            active_node_ids.add(port.node_id)
        return active_node_ids

    def _apply_graph_focus_opacity(self) -> None:
        active = self._effective_tunnel_highlight()
        active_tunnel_node_ids = self._active_tunnel_node_ids()
        search_node_ids = self._search_match_node_ids
        for node_id, proxy in self._proxies.items():
            opacity = 1.0
            card = self._cards.get(node_id)
            if card is not None and card._bypassed:
                opacity = min(opacity, BYPASSED_NODE_OPACITY)
            if active and node_id not in active_tunnel_node_ids:
                opacity = min(opacity, 0.38)
            if search_node_ids and node_id not in search_node_ids:
                opacity = min(opacity, 0.34)
            proxy.setOpacity(opacity)
        for connection in self._connections:
            opacity = 1.0
            if active:
                opacity = min(opacity, 0.18)
            if search_node_ids and (
                connection.source_id not in search_node_ids
                and connection.target_id not in search_node_ids
            ):
                opacity = min(opacity, 0.28)
            connection.setOpacity(opacity)
        self.scene.update()

    def remove_node(self, node_id: str) -> None:
        proxy = self._proxies.get(node_id)
        if proxy is None:
            return
        was_selected = node_id in self.selected_node_ids()
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
        if self._primary_node_id == node_id:
            self._primary_node_id = None
        self._search_match_node_ids.discard(node_id)
        self.scene.removeItem(proxy)
        self._apply_graph_focus_opacity()
        self._mark_graph_geometry_changed()
        self.reroute_connections(affected_rect=affected_rect)
        if was_selected:
            remaining = set(self.selected_node_ids())
            self._set_node_selection(remaining, self._primary_node_id)

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
                and (target_port is None or item.target_port == int(target_port))
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

    def set_thumbnail_pending(
        self,
        node_id: str,
        text: str = "Calculating preview…",
        *,
        accessible_description: str = "",
    ) -> None:
        """Retain an existing thumbnail or show a deliberate loading message."""

        card = self._cards.get(node_id)
        if card is not None:
            card.set_thumbnail_pending(
                text,
                accessible_description=accessible_description,
            )
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

    def set_node_subtitle(
        self,
        node_id: str,
        text: str,
        tooltip: str | None = None,
    ) -> None:
        """Set compact secondary text without changing the node title."""
        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None or proxy is None:
            return
        before = proxy.sceneBoundingRect()
        card.set_subtitle(text, tooltip)
        card.adjustSize()
        proxy.refresh_ports()
        after = proxy.sceneBoundingRect()
        if _rect_changed(before, after):
            self._mark_graph_geometry_changed()
            self.reroute_connections(affected_rect=before.united(after))
            return
        proxy.update()

    def set_node_compute_badge(
        self,
        node_id: str,
        kind: ComputeBadgeKind | str | None,
        *,
        tooltip: str = "",
        stale: bool = False,
    ) -> None:
        """Set one node's accepted-compute presentation without policy logic."""
        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None or proxy is None:
            return
        before = proxy.sceneBoundingRect()
        if not card.set_compute_badge(kind, tooltip=tooltip, stale=stale):
            return
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

    def set_node_bypassed(self, node_id: str, bypassed: bool) -> None:
        """Set one node's authored bypass badge without interpreting policy."""

        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None or proxy is None:
            return
        before = proxy.sceneBoundingRect()
        if not card.set_bypassed(bypassed):
            return
        card.adjustSize()
        proxy.refresh_ports()
        self._apply_graph_focus_opacity()
        after = proxy.sceneBoundingRect()
        if _rect_changed(before, after):
            self._mark_graph_geometry_changed()
            self.reroute_connections(affected_rect=before.united(after))
            return
        proxy.update()
        for connection in proxy.connections:
            connection.update_path()

    def clear_node_compute_badge(self, node_id: str) -> None:
        """Hide one node's accepted-compute presentation."""
        card = self._cards.get(node_id)
        if card is not None and card._bypassed:
            self.set_node_compute_badge(
                node_id,
                ComputeBadgeKind.BYPASSED,
                tooltip=(
                    "Workflow output forwards this node's exact primary input without "
                    "calling the operation. Its thumbnail remains a "
                    "presentation-only preview of what the node would produce "
                    "if run."
                ),
            )
            return
        self.set_node_compute_badge(node_id, None)

    def clear_node_compute_badges(self, node_ids=None) -> None:
        """Hide accepted-compute presentation for selected or all graph nodes."""
        selected = tuple(self._cards) if node_ids is None else tuple(node_ids)
        for node_id in selected:
            self.clear_node_compute_badge(str(node_id))

    def set_node_optimization_hint(self, node_id: str, tooltip: str = "") -> None:
        """Set one node's derived GPU optimization hint without policy logic."""
        card = self._cards.get(node_id)
        proxy = self._proxies.get(node_id)
        if card is None or proxy is None:
            return
        before = proxy.sceneBoundingRect()
        if not card.set_optimization_hint(tooltip):
            return
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

    def clear_node_optimization_hints(self, node_ids=None) -> None:
        """Hide derived GPU optimization hints for selected or all nodes."""
        selected = tuple(self._cards) if node_ids is None else tuple(node_ids)
        for node_id in selected:
            self.set_node_optimization_hint(str(node_id), "")

    def set_node_thumbnail_stats_tooltip(
        self,
        node_id: str,
        tooltip: str = "",
    ) -> None:
        """Expose thumbnail-statistics detail on a card without visible chrome."""
        card = self._cards.get(node_id)
        if card is None:
            return
        if not card.set_thumbnail_stats_tooltip(tooltip):
            return
        card.update()
        proxy = self._proxies.get(node_id)
        if proxy is not None:
            proxy.update()
        if self.scene is not None:
            self.scene.update()

    def node_has_thumbnail(self, node_id: str) -> bool:
        """Return whether a card currently presents a rendered thumbnail."""
        card = self._cards.get(str(node_id))
        return bool(card is not None and card.preview.has_source_pixmap())

    def clear_node_thumbnail_stats_tooltips(self, node_ids=None) -> None:
        """Clear thumbnail-statistics details for selected or all cards."""
        selected = tuple(self._cards) if node_ids is None else tuple(node_ids)
        for node_id in selected:
            self.set_node_thumbnail_stats_tooltip(str(node_id), "")

    def set_pinned_node(self, node_id: str | None) -> None:
        for card_id, card in self._cards.items():
            card.set_pinned(card_id == node_id)

    def set_isolated_tuning_node(self, node_id: str | None) -> None:
        active_node_id = node_id if node_id in self._cards else None
        self._isolated_tuning_node_id = active_node_id
        for card_id, card in self._cards.items():
            enabled = card_id == active_node_id
            if card._isolated_tuning == enabled:
                continue
            proxy = self._proxies.get(card_id)
            before = proxy.sceneBoundingRect() if proxy is not None else QRectF()
            card.set_isolated_tuning(enabled)
            if proxy is None:
                continue
            card.adjustSize()
            proxy.refresh_ports()
            after = proxy.sceneBoundingRect()
            if _rect_changed(before, after):
                self._mark_graph_geometry_changed()
                self.reroute_connections(affected_rect=before.united(after))
            else:
                proxy.update()

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
        before = proxy.sceneBoundingRect()
        proxy.set_input_ports(count, labels, colors, data_types)
        self._finish_port_geometry_update(proxy, before)

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
        before = proxy.sceneBoundingRect()
        proxy.set_output_ports(count, labels, colors, data_types)
        self._finish_port_geometry_update(proxy, before)

    def _finish_port_geometry_update(
        self,
        proxy: NodeProxy,
        before: QRectF,
    ) -> None:
        after = proxy.sceneBoundingRect()
        self._ensure_scene_space_for_rect(after)
        self._mark_graph_geometry_changed()
        self.reroute_connections(affected_rect=before.united(after))
        self.scene.update()

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
            scene_pos = self.mapToScene(_point_from_event(event))
            tunnel_name = self._update_tunnel_insert_preview(
                scene_pos,
                operation_id,
            )
            if tunnel_name:
                self._clear_connection_insert_preview()
            else:
                self._update_connection_insert_preview(operation_id, scene_pos)
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event):  # noqa: N802
        self._clear_all_insert_previews()
        super().dragLeaveEvent(event)

    def dropEvent(self, event):  # noqa: N802
        if event.mimeData().hasFormat(OPERATION_MIME):
            operation_id = bytes(event.mimeData().data(OPERATION_MIME)).decode()
            scene_pos = self.mapToScene(_point_from_event(event))
            tunnel_name = self._update_tunnel_insert_preview(
                scene_pos,
                operation_id,
            )
            if tunnel_name:
                if self._highlighted_tunnel_insert_state == "compatible":
                    self.tunnel_node_insert_requested.emit(
                        operation_id,
                        tunnel_name,
                        scene_pos,
                    )
                self._clear_all_insert_previews()
                event.acceptProposedAction()
                return
            self._update_connection_insert_preview(operation_id, scene_pos)
            connection = self._highlighted_connection
            state = self._highlighted_connection_state
            connection_key = self._connection_key(connection)
            if connection_key is not None and state != "incompatible":
                self.node_insert_requested.emit(operation_id, connection_key, scene_pos)
            else:
                self.node_create_requested.emit(operation_id, scene_pos)
            self._clear_all_insert_previews()
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key_Escape and self._pending_tunnel_source is not None:
            self._cancel_pending_tunnel_reroute()
            event.accept()
            return
        if not self._shortcut_belongs_to_text_editor():
            if event.matches(QKeySequence.Copy):
                selected = self.selected_node_ids()
                if selected:
                    self.nodes_copy_requested.emit(selected)
                    event.accept()
                    return
            if event.matches(QKeySequence.Paste) and self._clipboard_can_paste:
                self.paste_requested.emit(self.viewport_center_scene_position())
                event.accept()
                return
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

            selected_notes = [
                item
                for item in self.scene.selectedItems()
                if isinstance(item, GraphNoteItem)
            ]
            for item in selected_notes:
                self.note_delete_requested.emit(item.note_id)
            if selected_notes:
                event.accept()
                return

            selected_node_ids = self.selected_node_ids()
            if not selected_node_ids:
                selected_node_ids = tuple(
                    node_id
                    for node_id, proxy in self._proxies.items()
                    if proxy.isSelected()
                )
            if len(selected_node_ids) == 1:
                self.node_delete_requested.emit(selected_node_ids[0])
            elif selected_node_ids:
                self.nodes_delete_requested.emit(selected_node_ids)
            if selected_node_ids:
                event.accept()
                return
        super().keyPressEvent(event)

    def _shortcut_belongs_to_text_editor(self) -> bool:
        focus = QApplication.focusWidget()
        if focus is None or focus in {self, self.viewport()}:
            return False
        text_editor_types = (
            QLineEdit,
            QTextEdit,
            QPlainTextEdit,
            QAbstractSpinBox,
        )
        if isinstance(focus, text_editor_types):
            return True
        return isinstance(focus, QComboBox) and focus.isEditable()

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
            tunnel_badge = self._tunnel_badge_port_at_view_pos(pos)
            if tunnel_badge is not None:
                tunnel_name, port = tunnel_badge
                if port.kind == "output":
                    self.begin_tunnel_reroute(tunnel_name, self.mapToScene(pos))
                else:
                    self.highlight_tunnel(tunnel_name, sticky=True)
                event.accept()
                return
            if background_click and self._active_tunnel_name:
                self.clear_tunnel_highlight(sticky=True)
        if event.button() == Qt.RightButton:
            tunnel_badge = self._tunnel_badge_port_at_view_pos(pos)
            if tunnel_badge is not None and tunnel_badge[1].kind == "output":
                tunnel_name, source_port = tunnel_badge
                self._show_tunnel_source_context_menu(
                    tunnel_name,
                    source_port,
                    self.mapToScene(pos),
                    _global_pos_from_event(event),
                )
                event.accept()
                return
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
                selected = set(self.selected_node_ids())
                if node_id in selected:
                    self._set_node_selection(selected, node_id)
                else:
                    self._select_node(node_id)
                self._show_node_context_menu(node_id, _global_pos_from_event(event))
                event.accept()
                return
            if background_click:
                self._clear_node_selection()
                self._show_canvas_context_menu(
                    self.mapToScene(pos),
                    _global_pos_from_event(event),
                )
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
        if event.button() == Qt.LeftButton and background_click:
            self._clear_node_selection()
        if event.button() in (Qt.MiddleButton, Qt.RightButton) or (
            event.button() == Qt.LeftButton and background_click
        ):
            self._start_panning(pos)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._pending_tunnel_source is not None:
            if event.buttons() & Qt.LeftButton:
                self.update_pending_tunnel_reroute(
                    self.mapToScene(_point_from_event(event)),
                    dragging=True,
                )
            event.accept()
            return
        if self._panning:
            pos = _point_from_event(event)
            delta = pos - self._pan_start
            self.horizontalScrollBar().setValue(self._pan_h_value - delta.x())
            self.verticalScrollBar().setValue(self._pan_v_value - delta.y())
            self._ensure_scene_space_for_rect(
                self.mapToScene(self.viewport().rect()).boundingRect()
            )
            event.accept()
            return
        if self._pending_source is None and not self._active_tunnel_name:
            tunnel_name = self._tunnel_name_at_view_pos(_point_from_event(event))
            if tunnel_name != self._hover_tunnel_name:
                self.highlight_tunnel(tunnel_name, sticky=False)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._pending_tunnel_source is not None and event.button() == Qt.LeftButton:
            self.release_tunnel_reroute(
                self.mapToScene(_point_from_event(event)),
            )
            event.accept()
            return
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

    def selected_node_ids(self) -> tuple[str, ...]:
        """Return selected node ids in stable graph order."""
        return tuple(node_id for node_id, card in self._cards.items() if card._selected)

    def primary_node_id(self) -> str | None:
        """Return the most recently selected node used by the inspector."""
        if self._primary_node_id in self._cards:
            return self._primary_node_id
        return None

    def set_selected_nodes(
        self,
        node_ids,
        *,
        primary_node_id: str | None = None,
    ) -> None:
        """Replace the node selection while retaining a single inspector target."""
        requested = {
            str(node_id) for node_id in node_ids or () if str(node_id) in self._cards
        }
        primary = str(primary_node_id or "") or None
        if primary not in requested:
            primary = next(
                (
                    node_id
                    for node_id in reversed(tuple(self._cards))
                    if node_id in requested
                ),
                None,
            )
        self._set_node_selection(requested, primary)

    def _select_node(self, node_id: str) -> None:
        if node_id not in self._cards:
            return
        self._set_node_selection({node_id}, node_id)

    def _handle_node_press(
        self,
        node_id: str,
        modifiers,
        *,
        preserve_group_for_drag: bool = False,
    ) -> bool:
        """Dispatch one node press while exposing its nested Qt event boundary."""

        self._node_press_dispatch_depth += 1
        try:
            return self._dispatch_node_press(
                node_id,
                modifiers,
                preserve_group_for_drag=preserve_group_for_drag,
            )
        finally:
            self._node_press_dispatch_depth -= 1

    def node_press_dispatch_active(self) -> bool:
        """Return whether node-selection signals are inside a mouse press."""

        return self._node_press_dispatch_depth > 0

    def _dispatch_node_press(
        self,
        node_id: str,
        modifiers,
        *,
        preserve_group_for_drag: bool = False,
    ) -> bool:
        """Apply Windows-style click selection before a node drag starts."""
        if node_id not in self._cards:
            return False
        selected = set(self.selected_node_ids())
        control = bool(modifiers & (Qt.ControlModifier | Qt.MetaModifier))
        shift = bool(modifiers & Qt.ShiftModifier)
        if control:
            if node_id in selected:
                selected.remove(node_id)
                primary = self._primary_node_id
                if primary == node_id:
                    primary = next(
                        (
                            candidate
                            for candidate in reversed(tuple(self._cards))
                            if candidate in selected
                        ),
                        None,
                    )
            else:
                selected.add(node_id)
                primary = node_id
        elif shift:
            selected.add(node_id)
            primary = node_id
        else:
            if preserve_group_for_drag and node_id in selected and len(selected) > 1:
                self._set_node_selection(selected, node_id)
                return True
            selected = {node_id}
            primary = node_id
        self._set_node_selection(selected, primary)
        return False

    def _set_node_selection(
        self,
        selected_node_ids: set[str],
        primary_node_id: str | None,
    ) -> None:
        selected = {node_id for node_id in selected_node_ids if node_id in self._cards}
        if primary_node_id not in selected:
            primary_node_id = next(
                (
                    node_id
                    for node_id in reversed(tuple(self._cards))
                    if node_id in selected
                ),
                None,
            )
        for note in self._notes.values():
            note.setSelected(False)
        for connection in self._connections:
            connection.setSelected(False)
        for card_id, card in self._cards.items():
            is_selected = card_id in selected
            card.set_selected(is_selected)
            proxy = self._proxies.get(card_id)
            if proxy is not None and proxy.isSelected() != is_selected:
                proxy.setSelected(is_selected)
        self._primary_node_id = primary_node_id
        ordered = self.selected_node_ids()
        primary = primary_node_id or ""
        self.node_selection_changed.emit(ordered, primary)
        if primary:
            self.node_selected.emit(primary)

    def _clear_node_selection(self) -> None:
        if not self.selected_node_ids() and self._primary_node_id is None:
            return
        self._set_node_selection(set(), None)

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
        selected_node_ids = self.selected_node_ids()
        if node_id not in selected_node_ids:
            selected_node_ids = (node_id,)
        selected_count = len(selected_node_ids)
        menu = QMenu(self)
        copy_label = (
            "Copy node" if selected_count == 1 else f"Copy {selected_count} nodes"
        )
        copy_action = menu.addAction(copy_label)
        paste_values_action = menu.addAction("Paste values")
        proxy = self._proxies.get(node_id)
        paste_values_enabled = bool(
            selected_count == 1
            and proxy is not None
            and self._clipboard_single_operation_id
            and proxy.operation_id == self._clipboard_single_operation_id
        )
        paste_values_action.setEnabled(paste_values_enabled)
        if paste_values_enabled and self._clipboard_single_title:
            paste_values_action.setText(
                f"Paste values from {self._clipboard_single_title}"
            )
        menu.addSeparator()
        delete_action = menu.addAction(
            "Delete" if selected_count == 1 else f"Delete {selected_count} nodes"
        )
        code_action = menu.addAction("Inspect Code") if selected_count == 1 else None
        duplicate_action = (
            menu.addAction("Duplicate Node") if selected_count == 1 else None
        )
        add_note_action = menu.addAction("Add note") if selected_count == 1 else None
        menu.addSeparator()
        isolation_action = None
        bypass_action = None
        if selected_count == 1:
            operation = (
                NODE_LIBRARY_BY_ID.get(proxy.operation_id)
                if proxy is not None
                else None
            )
            bypass_visible, bypass_enabled, bypass_tooltip = (
                self._node_bypass_action_state(
                    node_id,
                    operation,
                    bypassed=card._bypassed,
                )
            )
            if bypass_visible:
                bypass_action = menu.addAction("Bypass node")
                bypass_action.setCheckable(True)
                bypass_action.setChecked(card._bypassed)
                bypass_action.setEnabled(bypass_enabled)
                bypass_action.setToolTip(bypass_tooltip)
                bypass_action.setStatusTip(bypass_tooltip)
            isolation_action = menu.addAction("Tune node in isolation")
            isolation_action.setCheckable(True)
            isolation_action.setChecked(node_id == self._isolated_tuning_node_id)
            isolation_action.setEnabled(
                self._isolated_tuning_node_id in {None, node_id}
            )
        pin_action = None
        if selected_count == 1 and card._can_pin:
            menu.addSeparator()
            pin_action = menu.addAction("Unpin" if card._pinned else "Pin")
        action = _exec_menu(menu, global_pos)
        if action == copy_action:
            self.nodes_copy_requested.emit(selected_node_ids)
        elif action == paste_values_action:
            self.node_paste_values_requested.emit(node_id)
        elif action == delete_action:
            if selected_count == 1:
                self.node_delete_requested.emit(node_id)
            else:
                self.nodes_delete_requested.emit(selected_node_ids)
        elif code_action is not None and action == code_action:
            self.node_code_requested.emit(node_id)
        elif duplicate_action is not None and action == duplicate_action:
            self.node_duplicate_requested.emit(node_id)
        elif add_note_action is not None and action == add_note_action:
            self.node_note_requested.emit(node_id)
        elif bypass_action is not None and action == bypass_action:
            self.node_bypass_requested.emit(node_id, not card._bypassed)
        elif isolation_action is not None and action == isolation_action:
            self.node_isolation_requested.emit(node_id)
        elif pin_action is not None and action == pin_action:
            self.pin_requested.emit(node_id)

    def _node_bypass_action_state(
        self,
        node_id: str,
        operation,
        *,
        bypassed: bool,
    ) -> tuple[bool, bool, str]:
        """Resolve one context-menu action without owning scientific policy."""

        if self._node_bypass_state_resolver is not None:
            try:
                visible, enabled, tooltip = self._node_bypass_state_resolver(node_id)
            except Exception as exc:  # pragma: no cover - defensive UI boundary
                return (
                    bool(operation is not None and operation.supports_bypass),
                    False,
                    f"Bypass availability could not be checked: {exc}",
                )
            return bool(visible), bool(enabled), str(tooltip or "")

        supported = bool(operation is not None and operation.supports_bypass)
        if not supported:
            return False, False, ""
        primary = getattr(operation, "bypass_primary_input", None)
        primary_label = str(getattr(primary, "label", "") or "Input")
        if bypassed:
            enabled = self._isolated_tuning_node_id is None
            tooltip = (
                f"Clear bypass to run this node again. Its {primary_label} input "
                "is currently forwarded unchanged."
            )
        elif self._isolated_tuning_node_id is not None:
            enabled = False
            tooltip = (
                "Apply or cancel isolated tuning before changing Bypass node."
            )
        elif not self._node_has_primary_bypass_input(node_id):
            enabled = False
            tooltip = (
                f"Connect the {primary_label} input before bypassing this node."
            )
        elif not self._node_has_scientific_output_use(node_id):
            enabled = False
            tooltip = (
                "This node has no downstream connection or output tunnel, so "
                "bypassing it would have no effect. Connect its output first."
            )
        else:
            enabled = True
            tooltip = (
                f"Skip this operation and forward its exact {primary_label} "
                "input to the output unchanged."
            )
        return True, enabled, tooltip

    def _node_has_primary_bypass_input(self, node_id: str) -> bool:
        if any(
            connection.target_id == node_id and connection.target_port == 0
            for connection in self._connections
        ):
            return True
        proxy = self._proxies.get(node_id)
        primary = proxy.input_port_at(0) if proxy is not None else None
        return bool(
            primary is not None
            and any(
                primary in ports
                for ports in self._tunnel_subscriber_ports.values()
            )
        )

    def _node_has_scientific_output_use(self, node_id: str) -> bool:
        if any(connection.source_id == node_id for connection in self._connections):
            return True
        return any(
            port.node_id == node_id
            for port in self._tunnel_source_ports.values()
        )

    def _show_canvas_context_menu(
        self,
        scene_pos: QPointF,
        global_pos: QPoint,
    ) -> None:
        menu = QMenu(self)
        paste_action = menu.addAction("Paste nodes here")
        paste_action.setEnabled(self._clipboard_can_paste)
        action = _exec_menu(menu, global_pos)
        if action == paste_action:
            self.paste_requested.emit(QPointF(scene_pos))

    def _show_tunnel_source_context_menu(
        self,
        tunnel_name: str,
        source_port: PortItem,
        scene_pos: QPointF,
        global_pos: QPoint,
    ) -> None:
        if source_port.kind != "output":
            return
        if self._tunnel_source_ports.get(tunnel_name) is not source_port:
            return
        menu = QMenu(self)
        insert_action = menu.addAction(f"Insert node before '{tunnel_name}'...")
        tunnel_options_action = menu.addAction("Tunnel options...")
        action = _exec_menu(menu, global_pos)
        if action == insert_action:
            self.tunnel_insert_requested.emit(tunnel_name, QPointF(scene_pos))
        elif action == tunnel_options_action:
            self.port_context_requested.emit(
                source_port.kind,
                source_port.node_id,
                source_port.port_index,
                global_pos,
            )

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

    def _output_port_at(self, scene_pos: QPointF) -> PortItem | None:
        for item in self.scene.items(scene_pos):
            if isinstance(item, PortItem) and item.kind == "output":
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
        match = self._tunnel_badge_port_at_view_pos(pos)
        return match[0] if match is not None else ""

    def _tunnel_badge_port_at_view_pos(
        self,
        pos: QPoint,
    ) -> tuple[str, PortItem] | None:
        return self._tunnel_badge_port_at_scene_pos(self.mapToScene(pos))

    def _tunnel_badge_port_at_scene_pos(
        self,
        scene_pos: QPointF,
    ) -> tuple[str, PortItem] | None:
        matches: list[tuple[float, str, PortItem]] = []
        for item in self.scene.items(scene_pos):
            if not isinstance(item, TunnelBadgeItem):
                continue
            current = item.parentItem()
            while current is not None:
                if isinstance(current, PortItem):
                    name = str(getattr(current, "_tunnel_label", "") or "").strip()
                    if not name:
                        break
                    center = item.mapToScene(item.boundingRect().center())
                    dx = center.x() - scene_pos.x()
                    dy = center.y() - scene_pos.y()
                    matches.append((dx * dx + dy * dy, name, current))
                    break
                current = current.parentItem()
        if not matches:
            return None
        _distance, name, port = min(matches, key=lambda match: match[0])
        return name, port

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

    def _update_tunnel_insert_preview(
        self,
        scene_pos: QPointF,
        operation_id: str = "",
        inserted_node_id: str = "",
    ) -> str:
        """Highlight only a source tunnel badge as a node-insertion target."""
        match = self._tunnel_badge_port_at_scene_pos(scene_pos)
        tunnel_name = ""
        port = None
        if match is not None:
            candidate_name, candidate_port = match
            if (
                candidate_port.kind == "output"
                and self._tunnel_source_ports.get(candidate_name) is candidate_port
            ):
                tunnel_name = candidate_name
                port = candidate_port
        state = "compatible"
        message = ""
        if port is not None and operation_id and self._tunnel_insert_validator:
            try:
                state, message = self._tunnel_insert_validator(
                    operation_id,
                    tunnel_name,
                    inserted_node_id,
                )
            except Exception as exc:
                state, message = "incompatible", str(exc)
            if state != "compatible":
                state = "incompatible"
        if (
            port is self._highlighted_tunnel_insert_port
            and tunnel_name == self._highlighted_tunnel_insert_name
            and state == self._highlighted_tunnel_insert_state
        ):
            return tunnel_name
        self._clear_tunnel_insert_preview()
        if port is None:
            return ""
        self._highlighted_tunnel_insert_port = port
        self._highlighted_tunnel_insert_name = tunnel_name
        self._highlighted_tunnel_insert_state = state
        port.set_drop_state(state)
        self.status_message.emit(
            message
            or (
                f"Drop to insert the node before tunnel '{tunnel_name}'."
                if state == "compatible"
                else f"That node cannot be inserted before tunnel '{tunnel_name}'."
            )
        )
        return tunnel_name

    def _clear_tunnel_insert_preview(self) -> None:
        port = getattr(self, "_highlighted_tunnel_insert_port", None)
        if port is not None:
            port.set_drop_state(None)
        self._highlighted_tunnel_insert_port = None
        self._highlighted_tunnel_insert_name = ""
        self._highlighted_tunnel_insert_state = None

    def _clear_all_insert_previews(self) -> None:
        self._clear_connection_insert_preview()
        self._clear_tunnel_insert_preview()

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
        return _types_compatible(self._pending_source.data_type, target_port.data_type)

    def _update_tunnel_reroute_feedback(self, scene_pos: QPointF) -> None:
        target = self._output_port_at(scene_pos)
        if target is self._highlighted_tunnel_output_port:
            return
        if self._highlighted_tunnel_output_port is not None:
            self._highlighted_tunnel_output_port.set_drop_state(None)
        self._highlighted_tunnel_output_port = target
        if target is None:
            return
        state, message = self._tunnel_reroute_target_feedback(target)
        target.set_drop_state(state)
        self.status_message.emit(message)

    def _tunnel_reroute_target_state(
        self,
        target: PortItem | None,
    ) -> str | None:
        state, _message = self._tunnel_reroute_target_feedback(target)
        return state

    def _tunnel_reroute_target_feedback(
        self,
        target: PortItem | None,
    ) -> tuple[str | None, str]:
        source = self._pending_tunnel_source
        if source is None or target is None or target.kind != "output":
            return None, ""
        generic_error = (
            "That output cannot source this tunnel or all of its subscribers."
        )
        if target is source:
            return "incompatible", "That output already sources this tunnel."
        if any(
            port is target and name != self._pending_tunnel_name
            for name, port in self._tunnel_source_ports.items()
        ):
            return "incompatible", "That output already has another tunnel."
        subscribers = self._tunnel_subscriber_ports.get(
            self._pending_tunnel_name,
            (),
        )
        if any(target.node_id == subscriber.node_id for subscriber in subscribers):
            return "incompatible", generic_error
        if any(
            not _types_compatible(target.data_type, subscriber.data_type)
            for subscriber in subscribers
        ):
            return "incompatible", generic_error
        if self._tunnel_reroute_validator is not None:
            try:
                state, message = self._tunnel_reroute_validator(
                    self._pending_tunnel_name,
                    target.node_id,
                    target.port_index,
                )
            except Exception as exc:
                return "incompatible", str(exc) or generic_error
            if state != "compatible":
                return "incompatible", str(message or generic_error)
        return (
            "compatible",
            f"Release to reroute tunnel '{self._pending_tunnel_name}'.",
        )

    def _cancel_pending_tunnel_reroute(self) -> None:
        if self._highlighted_tunnel_output_port is not None:
            self._highlighted_tunnel_output_port.set_drop_state(None)
        if self._pending_tunnel_wire is not None:
            self.scene.removeItem(self._pending_tunnel_wire)
        self._pending_tunnel_name = ""
        self._pending_tunnel_source = None
        self._pending_tunnel_wire = None
        self._highlighted_tunnel_output_port = None
        self._tunnel_reroute_dragging = False

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
    spec = _operation_spec_for_node(node)
    if spec is not None and spec.inputs:
        return len(spec.input_ports)
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
        return [f"Channel {index + 1}: {colors[index]}" for index in range(count)]
    spec = _operation_spec_for_node(node)
    if spec is not None:
        labels = [port.label for port in spec.input_ports]
        if len(labels) == count:
            return labels
    return [f"Input {index + 1}" for index in range(count)]


def _node_input_port_colors(node) -> list[str | None]:
    if getattr(node, "operation_id", "") != "combine_channels":
        return []
    return [_CHANNEL_COLOR_HEX.get(name.lower()) for name in _channel_color_names(node)]


def _node_input_port_types(node) -> list[str]:
    spec = _operation_spec_for_node(node)
    if spec is not None and spec.inputs:
        return [port.input_type for port in spec.input_ports]
    input_type = getattr(node, "input_type", None) or "any"
    return [input_type for _index in range(_node_input_port_count(node))]


def _node_output_port_count(node) -> int:
    spec = _operation_spec_for_node(node)
    if spec is None or spec.output_factory is not None:
        return 1
    return max(len(spec.output_ports), 1)


def _node_output_port_labels(node) -> list[str]:
    spec = _operation_spec_for_node(node)
    if spec is None or spec.output_factory is not None:
        return ["Output"]
    return [port.label for port in spec.output_ports]


def _node_output_port_colors(node) -> list[str | None]:
    spec = _operation_spec_for_node(node)
    if spec is None or spec.output_factory is not None:
        return []
    return [_CHANNEL_COLOR_HEX.get(port.name.lower()) for port in spec.output_ports]


def _node_output_port_types(node) -> list[str]:
    spec = _operation_spec_for_node(node)
    if spec is None or spec.output_factory is not None:
        return [getattr(node, "output_type", "any") or "any"]
    return [port.output_type for port in spec.output_ports]


def _operation_spec_for_node(node):
    return NODE_LIBRARY_BY_ID.get(str(getattr(node, "operation_id", "")))


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
