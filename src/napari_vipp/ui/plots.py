"""Reusable Qt plotting and scientific-density presentation widgets."""

from __future__ import annotations

from numbers import Rational

import numpy as np
from qtpy.QtCore import QPointF, QRect, Qt, Signal
from qtpy.QtGui import QColor, QImage, QPainter, QPen
from qtpy.QtWidgets import QWidget

from napari_vipp.core.channel_colors import (
    CHANNEL_COLOR_HEX,
    FLUORESCENCE_COLORS,
    color_value_to_rgb,
)
from napari_vipp.core.preview import _apply_monochrome_colormap

COLOCALIZATION_SCATTER_BINS = 255
COLOCALIZATION_SCATTER_COLORMAPS = (
    "Viridis",
    "Magma",
    "Inferno",
    "Plasma",
    "Cividis",
    "Gray",
)
SCATTER_DENSITY_CHUNK_ELEMENTS = 1_048_576


class HistogramPlot(QWidget):
    """Compact histogram display for the selected node output."""

    markerChanged = Signal(str, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = np.array([], dtype=np.float32)
        self._series_counts = np.empty((0, 0), dtype=np.float32)
        self._series_colors: list[QColor] = []
        self._log_scale = False
        self._x_min_label = ""
        self._x_max_label = ""
        self._x_range: tuple[float, float] | None = None
        self._x_scale = "linear"
        self._markers: list[tuple[str, float, QColor]] = []
        self._draggable_markers: set[str] = set()
        self._drag_marker: str | None = None
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

    def set_histogram(
        self,
        counts: np.ndarray | None,
        log_scale: bool,
        x_range: tuple[float, float] | None = None,
        colors: list[QColor] | None = None,
        markers: list[tuple[str, float, QColor]] | None = None,
        x_scale: str = "linear",
        draggable_markers: set[str] | None = None,
    ) -> None:
        self._counts = (
            np.asarray(counts, dtype=np.float32)
            if counts is not None
            else np.array([], dtype=np.float32)
        )
        if self._counts.ndim == 1:
            self._series_counts = self._counts.reshape(1, -1)
        elif self._counts.ndim == 2:
            self._series_counts = self._counts
            self._counts = self._series_counts.sum(axis=0)
        else:
            self._series_counts = np.empty((0, 0), dtype=np.float32)
            self._counts = np.array([], dtype=np.float32)
        self._series_colors = colors or _histogram_series_colors(
            self._series_counts.shape[0]
        )
        self._log_scale = log_scale
        self._x_range = x_range
        self._x_scale = x_scale
        self._markers = markers or []
        marker_labels = {label for label, _value, _color in self._markers}
        self._draggable_markers = set(draggable_markers or set()) & marker_labels
        if self._drag_marker not in self._draggable_markers:
            self._drag_marker = None
        if x_range is None or self._series_counts.size == 0:
            self._x_min_label = ""
            self._x_max_label = ""
        else:
            self._x_min_label = _format_histogram_label(x_range[0])
            self._x_max_label = _format_histogram_label(x_range[1])
        self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#111827"))
        painter.setPen(QPen(QColor("#374151"), 1))
        painter.drawRect(rect)

        label_height = painter.fontMetrics().height() + 5
        plot_frame = rect.adjusted(0, 0, 0, -label_height)
        plot_rect = plot_frame.adjusted(10, 8, -8, -10)
        self._draw_axes(painter, plot_rect)

        if self._series_counts.size == 0:
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(plot_frame, Qt.AlignCenter, "No data")
            painter.end()
            return

        values = self._series_counts
        if self._log_scale:
            values = np.log10(values + 1.0)
        maximum = float(values.max())
        if maximum <= 0:
            painter.end()
            return

        self._draw_histogram_series(painter, plot_rect, values, maximum)
        self._draw_markers(painter, plot_rect)
        if self._x_min_label or self._x_max_label:
            painter.setPen(QColor("#9ca3af"))
            metrics = painter.fontMetrics()
            baseline = min(rect.bottom() - 2, plot_rect.bottom() + metrics.ascent() + 3)
            painter.drawText(plot_rect.left(), baseline, self._x_min_label)
            right_width = metrics.horizontalAdvance(self._x_max_label)
            painter.drawText(
                plot_rect.right() - right_width,
                baseline,
                self._x_max_label,
            )
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        marker = self._marker_at_point(_event_position(event))
        if marker is None:
            return
        self._drag_marker = marker
        self._emit_marker_from_point(marker, _event_position(event))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        point = _event_position(event)
        if self._drag_marker is not None:
            self._emit_marker_from_point(self._drag_marker, point)
            return
        if self._marker_at_point(point) is None:
            self.unsetCursor()
        else:
            self.setCursor(Qt.SizeHorCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_marker is not None:
            self._emit_marker_from_point(self._drag_marker, _event_position(event))
        self._drag_marker = None

    def _draw_axes(self, painter: QPainter, plot_rect: QRect) -> None:
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.drawLine(
            plot_rect.left(),
            plot_rect.bottom(),
            plot_rect.right(),
            plot_rect.bottom(),
        )
        painter.drawLine(
            plot_rect.left(),
            plot_rect.top(),
            plot_rect.left(),
            plot_rect.bottom(),
        )

    def _draw_histogram_series(
        self,
        painter: QPainter,
        plot_rect: QRect,
        values: np.ndarray,
        maximum: float,
    ) -> None:
        width = max(plot_rect.width(), 1)
        height = max(plot_rect.height(), 1)
        step = max(int(np.ceil(values.shape[1] / width)), 1)
        for series_index, series_values in enumerate(values):
            reduced = np.array(
                [
                    series_values[i : i + step].max()
                    for i in range(0, series_values.size, step)
                ],
                dtype=np.float32,
            )
            if reduced.size == 0:
                continue
            color = self._series_colors[series_index % len(self._series_colors)]
            if values.shape[0] > 1:
                color = QColor(color)
                color.setAlpha(175)
            painter.setPen(QPen(color, 1.2))
            for index, value in enumerate(reduced):
                x = plot_rect.left() + int(index * width / max(reduced.size - 1, 1))
                y = plot_rect.bottom() - int((float(value) / maximum) * height)
                painter.drawLine(x, plot_rect.bottom(), x, y)

    def _draw_markers(self, painter: QPainter, plot_rect: QRect) -> None:
        if not self._markers or self._x_range is None:
            return
        metrics = painter.fontMetrics()
        label_y = plot_rect.top() + metrics.ascent() + 2
        for index, (label, value, color) in enumerate(self._markers):
            fraction = self._x_fraction(value)
            x = plot_rect.left() + int(fraction * max(plot_rect.width(), 1))
            painter.setPen(QPen(color, 2.0, Qt.DashLine))
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
            text = f"{label} {_format_histogram_label(value)}"
            text_width = metrics.horizontalAdvance(text)
            rightmost_text_x = max(
                plot_rect.left(),
                plot_rect.right() - text_width,
            )
            text_x = int(np.clip(x + 3, plot_rect.left(), rightmost_text_x))
            painter.setPen(color)
            painter.drawText(
                text_x,
                label_y + index * (metrics.height() + 1),
                text,
            )

    def _x_fraction(self, value: int | float | Rational) -> float:
        if self._x_range is None:
            return 0.0
        minimum, maximum = self._x_range
        if maximum <= minimum:
            return 0.0
        integer_range = all(
            isinstance(item, (int, np.integer)) for item in (minimum, maximum)
        )
        if integer_range and isinstance(value, Rational):
            shifted_maximum = int(maximum) - int(minimum)
            shifted_value = min(max(value - int(minimum), 0), shifted_maximum)
            if self._x_scale == "log":
                return float(
                    np.log1p(float(shifted_value))
                    / np.log1p(max(shifted_maximum, 1))
                )
            return float(shifted_value / shifted_maximum)
        value = float(np.clip(value, minimum, maximum))
        if self._x_scale == "log":
            shifted_value = max(value - minimum, 0.0)
            shifted_maximum = maximum - minimum
            return float(np.log1p(shifted_value) / np.log1p(max(shifted_maximum, 1.0)))
        return float((value - minimum) / (maximum - minimum))

    def _plot_rect(self) -> QRect:
        rect = self.rect().adjusted(8, 8, -8, -8)
        label_height = self.fontMetrics().height() + 5
        plot_frame = rect.adjusted(0, 0, 0, -label_height)
        return plot_frame.adjusted(10, 8, -8, -10)

    def _marker_at_point(self, point) -> str | None:
        if not self._draggable_markers or self._x_range is None:
            return None
        plot_rect = self._plot_rect()
        if not plot_rect.adjusted(-8, 0, 8, 0).contains(point):
            return None
        candidates: list[tuple[float, str]] = []
        for label, value, _color in self._markers:
            if label not in self._draggable_markers:
                continue
            x = plot_rect.left() + self._x_fraction(value) * max(plot_rect.width(), 1)
            distance = abs(float(point.x()) - float(x))
            if distance <= 8.0:
                candidates.append((distance, label))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _emit_marker_from_point(self, label: str, point) -> None:
        if self._x_range is None:
            return
        value = self._value_from_x(float(point.x()), self._plot_rect())
        self._replace_marker_value(label, value)
        self.markerChanged.emit(label, value)
        self.update()

    def _replace_marker_value(self, label: str, value: float) -> None:
        self._markers = [
            (
                marker_label,
                float(value) if marker_label == label else marker_value,
                color,
            )
            for marker_label, marker_value, color in self._markers
        ]

    def _value_from_x(self, x: float, plot_rect: QRect) -> float:
        if self._x_range is None:
            return 0.0
        minimum, maximum = self._x_range
        if maximum <= minimum:
            return float(minimum)
        width = max(float(plot_rect.width()), 1.0)
        fraction = float(np.clip((float(x) - plot_rect.left()) / width, 0.0, 1.0))
        if self._x_scale == "log":
            shifted_maximum = maximum - minimum
            shifted = np.expm1(fraction * np.log1p(max(shifted_maximum, 1.0)))
            return float(np.clip(minimum + shifted, minimum, maximum))
        return float(minimum + fraction * (maximum - minimum))


def _prepare_colocalization_scatter_density(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    threshold_1: float,
    threshold_2: float,
    roi_mask: np.ndarray | None,
    intensity_max: float,
    bins: int,
    progress=None,
    chunk_elements: int = SCATTER_DENSITY_CHUNK_ELEMENTS,
) -> tuple[np.ndarray, int, int]:
    """Return an exact bounded-memory density and exact summary counts."""
    ch1 = np.asarray(channel_1)
    ch2 = np.asarray(channel_2)
    if ch1.shape != ch2.shape or ch1.size == 0:
        raise ValueError("Scatter channels must be non-empty and have matching shapes.")
    flat_1 = ch1.reshape(-1)
    flat_2 = ch2.reshape(-1)
    flat_roi: np.ndarray | None = None
    if roi_mask is not None:
        roi = np.asarray(roi_mask, dtype=bool)
        if roi.shape != ch1.shape:
            raise ValueError(
                f"ROI mask shape {roi.shape} does not match channels {ch1.shape}."
            )
        flat_roi = roi.reshape(-1)

    bins = int(np.clip(int(bins), 32, 512))
    chunk_elements = int(chunk_elements)
    intensity_max = max(float(intensity_max), 1.0)
    edges = np.linspace(0.0, intensity_max, bins + 1)
    density_counts = np.zeros((bins, bins), dtype=np.float64)
    roi_voxels = 0
    colocalized_voxels = 0
    threshold_1 = float(threshold_1)
    threshold_2 = float(threshold_2)
    for start in range(0, int(flat_1.size), chunk_elements):
        if progress is not None:
            progress.check_cancelled()
        stop = min(start + chunk_elements, int(flat_1.size))
        values_1 = flat_1[start:stop]
        values_2 = flat_2[start:stop]
        positive = np.greater_equal(values_1, threshold_1)
        np.logical_and(positive, values_2 >= threshold_2, out=positive)
        if flat_roi is None:
            roi_voxels += stop - start
            density_values_1 = values_1
            density_values_2 = values_2
        else:
            chunk_roi = flat_roi[start:stop]
            roi_voxels += int(np.count_nonzero(chunk_roi))
            np.logical_and(positive, chunk_roi, out=positive)
            density_values_1 = values_1[chunk_roi]
            density_values_2 = values_2[chunk_roi]
        colocalized_voxels += int(np.count_nonzero(positive))
        if density_values_1.size:
            chunk_density, _x_edges, _y_edges = np.histogram2d(
                density_values_1,
                density_values_2,
                bins=(edges, edges),
            )
            density_counts += chunk_density

    return density_counts, roi_voxels, colocalized_voxels


class ColocalizationScatterPlot(QWidget):
    """Interactive two-channel scatter-density plot with threshold guides."""

    thresholdChanged = Signal(int, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image: QImage | None = None
        self._threshold_1 = 25.0
        self._threshold_2 = 25.0
        self._intensity_max = 255.0
        self._channel_1_color = QColor("#ef4444")
        self._channel_2_color = QColor("#22c55e")
        self._colormap = "Viridis"
        self._summary = ""
        self._drag_axis: int | None = None
        self.setMinimumHeight(300)
        self.setMouseTracking(True)

    def set_density(
        self,
        density_counts: np.ndarray | None,
        *,
        threshold_1: float,
        threshold_2: float,
        intensity_max: float = 255.0,
        channel_1_color: object = "Red",
        channel_2_color: object = "Green",
        colormap: str = "Viridis",
        log_counts: bool = True,
        summary: str = "",
    ) -> None:
        """Render worker-prepared density counts without touching source images."""
        self._threshold_1 = float(threshold_1)
        self._threshold_2 = float(threshold_2)
        self._intensity_max = max(float(intensity_max), 1.0)
        self._channel_1_color = _qcolor_from_channel_color(
            channel_1_color,
            fallback="#ef4444",
        )
        self._channel_2_color = _qcolor_from_channel_color(
            channel_2_color,
            fallback="#22c55e",
        )
        self._colormap = str(colormap or "Viridis")
        self._summary = str(summary)
        self._image = self._density_image(density_counts, log_counts=log_counts)
        self.update()

    def clear(self, message: str = "Connect two channel inputs.") -> None:
        self._image = None
        self._summary = message
        self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(rect, QColor("#111827"))
        painter.setPen(QPen(QColor("#374151"), 1))
        painter.drawRect(rect)

        plot_rect = self._plot_rect()
        if self._image is None:
            painter.setPen(QColor("#9ca3af"))
            painter.drawText(rect, Qt.AlignCenter, self._summary or "No data")
            painter.end()
            return

        painter.drawImage(plot_rect, self._image)
        painter.setPen(QPen(QColor("#64748b"), 1.2))
        painter.drawRect(plot_rect)
        self._draw_thresholds(painter, plot_rect)
        self._draw_labels(painter, rect, plot_rect)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        plot_rect = self._plot_rect()
        point = _event_position(event)
        if not plot_rect.contains(point):
            return
        vertical_x = self._x_from_value(self._threshold_1, plot_rect)
        horizontal_y = self._y_from_value(self._threshold_2, plot_rect)
        dx = abs(point.x() - vertical_x)
        dy = abs(point.y() - horizontal_y)
        self._drag_axis = 1 if dx <= dy else 2
        self._emit_threshold_from_point(point, plot_rect)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_axis is None:
            return
        point = _event_position(event)
        self._emit_threshold_from_point(point, self._plot_rect())

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_axis is not None:
            point = _event_position(event)
            self._emit_threshold_from_point(point, self._plot_rect())
        self._drag_axis = None

    def _emit_threshold_from_point(self, point, plot_rect: QRect) -> None:
        if self._drag_axis == 1:
            value = self._value_from_x(point.x(), plot_rect)
            self._threshold_1 = value
            self.thresholdChanged.emit(1, value)
        elif self._drag_axis == 2:
            value = self._value_from_y(point.y(), plot_rect)
            self._threshold_2 = value
            self.thresholdChanged.emit(2, value)
        self.update()

    def _density_image(
        self,
        density_counts: np.ndarray | None,
        *,
        log_counts: bool,
    ) -> QImage | None:
        if density_counts is None:
            return None
        hist = np.asarray(density_counts)
        if hist.ndim != 2 or hist.size == 0:
            return None
        values = hist.T
        if bool(log_counts):
            values = np.log1p(values)
        maximum = float(np.max(values))
        if maximum > 0:
            values = values / maximum
        values = np.flipud(values)
        gray = np.clip(np.rint(np.sqrt(values) * 255.0), 0, 255).astype(np.uint8)
        rgb = _apply_monochrome_colormap(gray, self._colormap)
        rgb[values <= 0] = (4, 7, 15)
        return QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            int(rgb.strides[0]),
            QImage.Format_RGB888,
        ).copy()

    def _draw_thresholds(self, painter: QPainter, plot_rect: QRect) -> None:
        x = self._x_from_value(self._threshold_1, plot_rect)
        y = self._y_from_value(self._threshold_2, plot_rect)
        painter.setPen(QPen(self._channel_1_color, 2.0, Qt.DashLine))
        painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
        painter.setPen(QPen(self._channel_2_color, 2.0, Qt.DashLine))
        painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)
        painter.setPen(QPen(QColor("#f8fafc"), 1.5))
        painter.drawEllipse(QPointF(float(x), float(y)), 3.5, 3.5)

    def _draw_labels(self, painter: QPainter, rect: QRect, plot_rect: QRect) -> None:
        metrics = painter.fontMetrics()
        axis_color = QColor("#9ca3af")
        zero_label = _format_histogram_label(0.0)
        max_label = _format_histogram_label(self._intensity_max)

        painter.setPen(axis_color)
        axis_value_y = plot_rect.bottom() + metrics.ascent() + 6
        painter.drawText(plot_rect.left(), axis_value_y, zero_label)
        painter.drawText(
            plot_rect.right() - metrics.horizontalAdvance(max_label),
            axis_value_y,
            max_label,
        )

        x_label = "Ch 1 intensity"
        painter.drawText(
            plot_rect.center().x() - metrics.horizontalAdvance(x_label) // 2,
            axis_value_y + metrics.height(),
            x_label,
        )

        y_value_x = plot_rect.left() - metrics.horizontalAdvance(max_label) - 8
        painter.drawText(y_value_x, plot_rect.top() + metrics.ascent(), max_label)
        painter.drawText(
            plot_rect.left() - metrics.horizontalAdvance(zero_label) - 8,
            plot_rect.bottom(),
            zero_label,
        )

        y_label = "Ch 2 intensity"
        painter.save()
        painter.translate(
            plot_rect.left() - 42,
            plot_rect.center().y() + metrics.horizontalAdvance(y_label) // 2,
        )
        painter.rotate(-90)
        painter.setPen(axis_color)
        painter.drawText(0, 0, y_label)
        painter.restore()

        t1_text = f"T1 {_format_histogram_label(self._threshold_1)}"
        t1_width = metrics.horizontalAdvance(t1_text)
        t1_x = int(
            np.clip(
                self._x_from_value(self._threshold_1, plot_rect) + 4,
                plot_rect.left() + 2,
                plot_rect.right() - t1_width - 2,
            )
        )
        painter.setPen(self._channel_1_color)
        painter.drawText(t1_x, plot_rect.bottom() - 4, t1_text)

        t2_text = f"T2 {_format_histogram_label(self._threshold_2)}"
        t2_y = int(
            np.clip(
                self._y_from_value(self._threshold_2, plot_rect) - 4,
                plot_rect.top() + metrics.ascent() + 2,
                plot_rect.bottom() - 4,
            )
        )
        painter.setPen(self._channel_2_color)
        painter.drawText(
            plot_rect.left() + 3,
            t2_y,
            t2_text,
        )

        if self._summary:
            summary_width = metrics.horizontalAdvance(self._summary)
            painter.setPen(axis_color)
            painter.drawText(
                max(plot_rect.left(), plot_rect.right() - summary_width),
                plot_rect.top() + metrics.ascent() + 2,
                self._summary,
            )

    def _plot_rect(self) -> QRect:
        rect = self.rect().adjusted(8, 8, -8, -8)
        metrics = self.fontMetrics()
        max_label = _format_histogram_label(self._intensity_max)
        left_margin = max(56, metrics.horizontalAdvance(max_label) + 22)
        right_margin = 10
        top_margin = 8
        bottom_margin = metrics.height() * 2 + 12
        available_width = max(1, rect.width() - left_margin - right_margin)
        available_height = max(1, rect.height() - top_margin - bottom_margin)
        side = max(1, min(available_width, available_height))
        x = rect.left() + left_margin + (available_width - side) // 2
        y = rect.top() + top_margin + (available_height - side) // 2
        return QRect(x, y, side, side)

    def _x_from_value(self, value: float, plot_rect: QRect) -> int:
        fraction = float(np.clip(value / self._intensity_max, 0.0, 1.0))
        return plot_rect.left() + int(round(fraction * max(plot_rect.width(), 1)))

    def _y_from_value(self, value: float, plot_rect: QRect) -> int:
        fraction = float(np.clip(value / self._intensity_max, 0.0, 1.0))
        return plot_rect.bottom() - int(round(fraction * max(plot_rect.height(), 1)))

    def _value_from_x(self, x: int, plot_rect: QRect) -> float:
        fraction = (float(x) - plot_rect.left()) / max(plot_rect.width(), 1)
        return float(np.clip(fraction, 0.0, 1.0) * self._intensity_max)

    def _value_from_y(self, y: int, plot_rect: QRect) -> float:
        fraction = (plot_rect.bottom() - float(y)) / max(plot_rect.height(), 1)
        return float(np.clip(fraction, 0.0, 1.0) * self._intensity_max)


def _event_position(event):
    if hasattr(event, "position"):
        return event.position().toPoint()
    return event.pos()


def _qcolor_from_channel_color(value, *, fallback: str) -> QColor:
    if isinstance(value, QColor):
        return QColor(value)
    rgb = color_value_to_rgb(value)
    if rgb is None:
        rgb = color_value_to_rgb(CHANNEL_COLOR_HEX.get(str(value).strip().lower()))
    if rgb is None:
        return QColor(fallback)
    values = np.clip(np.rint(np.asarray(rgb, dtype=np.float32) * 255.0), 0, 255)
    return QColor(int(values[0]), int(values[1]), int(values[2]))


def _histogram_series_colors(
    count: int,
    channel_axis_name: str = "",
) -> list[QColor]:
    if count <= 0:
        return []
    if channel_axis_name == "rgb":
        base = [QColor("#ef4444"), QColor("#22c55e"), QColor("#60a5fa")]
    elif count > 1:
        base = [_qcolor_from_unit_rgb(color) for color in FLUORESCENCE_COLORS]
    else:
        base = [QColor("#60a5fa")]
    return [QColor(base[index % len(base)]) for index in range(count)]


def _qcolor_from_unit_rgb(color: np.ndarray) -> QColor:
    return QColor.fromRgbF(
        float(np.clip(color[0], 0, 1)),
        float(np.clip(color[1], 0, 1)),
        float(np.clip(color[2], 0, 1)),
    )


def _format_histogram_label(value: int | float | Rational) -> str:
    if isinstance(value, Rational):
        if value.denominator == 1:
            return str(int(value))
        whole = value.numerator // value.denominator
        remainder = value - whole
        return f"{whole} + {remainder.numerator}/{remainder.denominator}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if not np.isfinite(value):
        return ""
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.4g}"
