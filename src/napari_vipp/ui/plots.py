"""Reusable Qt plotting and scientific-density presentation widgets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Rational

import numpy as np
from qtpy.QtCore import QEvent, QPointF, QRect, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QFont, QFontMetrics, QImage, QPainter, QPen
from qtpy.QtWidgets import QToolTip, QWidget

from napari_vipp.core.channel_colors import (
    CHANNEL_COLOR_HEX,
    FLUORESCENCE_COLORS,
    color_value_to_rgb,
)
from napari_vipp.core.operations import colocalization_populated_ranges
from napari_vipp.core.preview import _apply_monochrome_colormap
from napari_vipp.ui.palette_roles import custom_paint_colors

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
SCATTER_DENSITY_BACKGROUND_COST_BYTES = 16 * 1024 * 1024
COLOCALIZATION_SCATTER_CACHE_BUDGET_BYTES = 192 * 1024 * 1024
# Interactive density calculation supports the same upper resolution as the
# legacy graph operation. Compact inspector plots retain a bounded derivative;
# the detached plot opts into the complete texture explicitly.
COLOCALIZATION_SCATTER_INSPECTOR_MAX_BINS = 4_096
COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS = 1_024
COLOCALIZATION_SCATTER_THRESHOLD_HIT_TOLERANCE = 8
DETAILED_HISTOGRAM_DEFAULT_GRID_DIVISIONS = 4
DETAILED_HISTOGRAM_MIN_GRID_DIVISIONS = 2
DETAILED_HISTOGRAM_MAX_GRID_DIVISIONS = 12
HISTOGRAM_SINGLE_SERIES_FILL_ALPHA = 185
HISTOGRAM_MULTI_SERIES_FILL_ALPHA = 72
HISTOGRAM_MULTI_SERIES_STROKE_ALPHA = 128


def colocalization_scatter_density_bytes(bins: int) -> int:
    """Return the retained float64 density footprint for a square histogram."""
    bins = max(int(bins), 0)
    return bins * bins * np.dtype(np.float64).itemsize


def colocalization_scatter_peak_bytes(bins: int) -> int:
    """Estimate peak density accumulation memory, excluding source arrays."""
    # The retained accumulator coexists with np.histogram2d's flattened count,
    # float conversion, and indexing scratch. Four grids are a conservative
    # upper bound for the bin-dependent part of that worker allocation.
    return 4 * colocalization_scatter_density_bytes(bins)


def colocalization_scatter_requires_background(bins: int) -> bool:
    """Whether histogram allocation alone warrants worker execution."""
    return (
        colocalization_scatter_peak_bytes(bins) >= SCATTER_DENSITY_BACKGROUND_COST_BYTES
    )


def colocalization_scatter_inspector_bins(bins: int) -> int:
    """Clamp requested graph bins to the GUI inspector's safe density limit."""
    return int(
        np.clip(
            int(bins),
            32,
            COLOCALIZATION_SCATTER_INSPECTOR_MAX_BINS,
        )
    )


def cap_colocalization_scatter_density_for_display(
    density_counts: np.ndarray,
    *,
    max_bins: int | None = None,
) -> np.ndarray:
    """Sum adjacent cells so GUI rendering never exceeds its bin limit."""
    density = np.asarray(density_counts)
    if density.ndim != 2:
        return density
    max_bins = max(
        int(
            COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS
            if max_bins is None
            else max_bins
        ),
        1,
    )
    if density.shape[0] <= max_bins and density.shape[1] <= max_bins:
        return density
    row_step = max(int(np.ceil(density.shape[0] / max_bins)), 1)
    col_step = max(int(np.ceil(density.shape[1] / max_bins)), 1)
    row_starts = np.arange(0, density.shape[0], row_step)
    col_starts = np.arange(0, density.shape[1], col_step)
    reduced = np.add.reduceat(density, row_starts, axis=0)
    return np.add.reduceat(reduced, col_starts, axis=1)


class HistogramPlot(QWidget):
    """Compact histogram display for the selected node output."""

    markerChanged = Signal(str, float)
    gestureStarted = Signal()
    gestureFinished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._counts = np.array([], dtype=np.float32)
        self._series_counts = np.empty((0, 0), dtype=np.float32)
        self._series_colors: list[QColor] = []
        self._log_scale = False
        self._title = ""
        self._x_axis_label = ""
        self._y_axis_label = ""
        self._x_min_label = ""
        self._x_max_label = ""
        self._x_range: tuple[float, float] | None = None
        self._x_scale = "linear"
        self._markers: list[tuple[str, float, QColor]] = []
        self._draggable_markers: set[str] = set()
        self._drag_marker: str | None = None
        self._drag_start_x: float | None = None
        self._drag_moved = False
        self._gesture_active = False
        self.setMinimumHeight(120)
        self.setMouseTracking(True)

    def set_plot_labels(
        self,
        *,
        title: str = "",
        x_axis_label: str = "",
        y_axis_label: str = "",
    ) -> None:
        """Set compact, in-frame scientific labels for this histogram.

        Labels are presentation-only and do not affect histogram values or
        marker coordinates.  Keeping them on the plot makes paired input and
        output distributions unambiguous without spending inspector height on
        separate headings.
        """
        self._title = str(title).strip()
        self._x_axis_label = str(x_axis_label).strip()
        self._y_axis_label = str(y_axis_label).strip()
        self.update()

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
            self._cancel_marker_drag()
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
        colors = custom_paint_colors(self.palette())
        painter.fillRect(rect, colors.surface)
        painter.setPen(QPen(colors.border, 1))
        painter.drawRect(rect)

        plot_rect = self._plot_rect()
        self._draw_plot_text(painter, rect, plot_rect)
        self._draw_axes(painter, plot_rect)

        if self._series_counts.size == 0:
            painter.setPen(colors.muted_text)
            painter.drawText(plot_rect, Qt.AlignCenter, "No data")
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
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        point = _event_position(event)
        marker = self._marker_at_point(point)
        if marker is None:
            super().mousePressEvent(event)
            return
        self._drag_marker = marker
        self._drag_start_x = float(point.x())
        self._drag_moved = False
        self._begin_gesture()
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        point = _event_position(event)
        if self._drag_marker is not None:
            if self._drag_start_x is None or not np.isclose(
                float(point.x()),
                self._drag_start_x,
            ):
                self._drag_moved = True
                self._emit_marker_from_point(self._drag_marker, point)
            event.accept()
            return
        if self._marker_at_point(point) is None:
            self.unsetCursor()
        else:
            self.setCursor(Qt.SizeHorCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_marker is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        if self._drag_moved:
            self._emit_marker_from_point(self._drag_marker, _event_position(event))
        self._cancel_marker_drag()
        event.accept()

    def event(self, event) -> bool:
        if event.type() in {QEvent.Hide, QEvent.UngrabMouse}:
            self._cancel_marker_drag()
        return super().event(event)

    def _begin_gesture(self) -> None:
        if self._gesture_active:
            return
        self._gesture_active = True
        self.gestureStarted.emit()

    def _cancel_marker_drag(self) -> None:
        self._drag_marker = None
        self._drag_start_x = None
        self._drag_moved = False
        if not self._gesture_active:
            return
        self._gesture_active = False
        self.gestureFinished.emit()

    def marker_values(self) -> dict[str, Rational | float]:
        """Return the currently displayed marker values by label."""
        return {str(label): value for label, value, _color in self._markers}

    def _draw_axes(self, painter: QPainter, plot_rect: QRect) -> None:
        painter.setPen(QPen(custom_paint_colors(self.palette()).axis, 1.2))
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

    def _draw_plot_text(
        self,
        painter: QPainter,
        rect: QRect,
        plot_rect: QRect,
    ) -> None:
        colors = custom_paint_colors(self.palette())
        base_font = painter.font()
        metrics = painter.fontMetrics()

        if self._title:
            title_font = painter.font()
            title_font.setBold(True)
            painter.setFont(title_font)
            title_metrics = painter.fontMetrics()
            title_rect = QRect(
                plot_rect.left(),
                rect.top() + 3,
                max(plot_rect.right() - plot_rect.left(), 1),
                title_metrics.height() + 2,
            )
            title = title_metrics.elidedText(
                self._title,
                Qt.ElideRight,
                title_rect.width(),
            )
            painter.setPen(colors.text)
            painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter, title)
            painter.setFont(base_font)
            metrics = painter.fontMetrics()

        painter.setPen(colors.muted_text)
        # Range ticks and the scientific x-axis label share one compact band.
        # This keeps the data region useful in a 120 px-high docked inspector.
        if self._x_min_label or self._x_max_label:
            baseline = plot_rect.bottom() + metrics.ascent() + 3
            painter.drawText(plot_rect.left(), baseline, self._x_min_label)
            right_width = metrics.horizontalAdvance(self._x_max_label)
            painter.drawText(
                plot_rect.right() - right_width,
                baseline,
                self._x_max_label,
            )

        if self._x_axis_label:
            left_tick_width = metrics.horizontalAdvance(self._x_min_label)
            right_tick_width = metrics.horizontalAdvance(self._x_max_label)
            label_width = max(
                plot_rect.width() - left_tick_width - right_tick_width - 12,
                1,
            )
            label = metrics.elidedText(
                self._x_axis_label,
                Qt.ElideRight,
                label_width,
            )
            label_y = plot_rect.bottom() + metrics.ascent() + 3
            painter.drawText(
                plot_rect.center().x() - metrics.horizontalAdvance(label) // 2,
                label_y,
                label,
            )

        if self._y_axis_label:
            label = metrics.elidedText(
                self._y_axis_label,
                Qt.ElideRight,
                max(plot_rect.height() - 4, 1),
            )
            painter.save()
            painter.translate(
                rect.left() + metrics.ascent() + 2,
                plot_rect.center().y() + metrics.horizontalAdvance(label) // 2,
            )
            painter.rotate(-90)
            painter.setPen(colors.muted_text)
            painter.drawText(0, 0, label)
            painter.restore()

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
        multiple_series = values.shape[0] > 1
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
            source_color = self._series_colors[
                series_index % len(self._series_colors)
            ]
            color = QColor(source_color)
            if multiple_series:
                color.setAlpha(
                    min(color.alpha(), HISTOGRAM_MULTI_SERIES_STROKE_ALPHA)
                )
            if reduced.size <= 8:
                # Discrete distributions (especially boolean masks) need
                # visible inset bars.  One-pixel strokes at x=0 and x=1 sit
                # directly on the axes and otherwise look like an empty plot.
                slot_width = width / max(int(reduced.size), 1)
                bar_width = max(min(int(slot_width * 0.55), int(slot_width) - 2), 2)
                painter.setPen(QPen(color, 1.0))
                fill = QColor(source_color)
                fill.setAlpha(
                    min(fill.alpha(), HISTOGRAM_MULTI_SERIES_FILL_ALPHA)
                    if multiple_series
                    else max(fill.alpha(), 150)
                )
                painter.setBrush(fill)
                for index, value in enumerate(reduced):
                    center_x = plot_rect.left() + int((index + 0.5) * slot_width)
                    x = center_x - bar_width // 2
                    y = plot_rect.bottom() - int((float(value) / maximum) * height)
                    bar_height = max(plot_rect.bottom() - y, 1)
                    painter.drawRect(QRect(x, y, bar_width, bar_height))
                painter.setBrush(Qt.NoBrush)
                continue
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
                    np.log1p(float(shifted_value)) / np.log1p(max(shifted_maximum, 1))
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
        metrics = self.fontMetrics()
        # Reserve a small deliberate gap below the title so the first bars do
        # not visually collide with its glyphs.  The plot remains compact at
        # the inspector's 120 px minimum height.
        title_height = metrics.height() + 8 if self._title else 0
        bottom_band_height = (
            metrics.height() + 3
            if self._x_min_label or self._x_max_label or self._x_axis_label
            else 0
        )
        y_axis_width = metrics.height() + 5 if self._y_axis_label else 0
        return rect.adjusted(
            10 + y_axis_width,
            title_height,
            -8,
            -bottom_band_height,
        )

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


class DetailedHistogramPlot(QWidget):
    """Edge-aware, palette-native histogram for detailed inspection.

    ``HistogramPlot`` above intentionally remains a small diagnostic that can
    reduce hundreds of display bins into a narrow inspector.  This widget is
    the exact presentation counterpart: it retains explicit bin edges, maps
    every bar through the selected axis transforms, and exposes the original
    bin values on hover.  Logarithmic axes are genuine base-10 coordinate
    transforms; no ``log1p`` or shifted pseudo-logarithmic presentation is
    used.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bin_edges = np.array([], dtype=np.float64)
        self._series_values = np.empty((0, 0), dtype=np.float64)
        self._series_labels: tuple[str, ...] = ()
        self._series_colors: list[QColor] = []
        self._hover_details: dict[str, np.ndarray] = {}
        self._x_scale = "linear"
        self._y_scale = "linear"
        self._x_grid_divisions = DETAILED_HISTOGRAM_DEFAULT_GRID_DIVISIONS
        self._y_grid_divisions = DETAILED_HISTOGRAM_DEFAULT_GRID_DIVISIONS
        self._title = "Histogram"
        self._x_axis_label = "Value"
        self._y_axis_label = "Count"
        self._hovered_bin: int | None = None
        self.setMinimumHeight(220)
        self.setMouseTracking(True)
        self.setAccessibleName("Detailed histogram")

    @property
    def supports_log_x(self) -> bool:
        """Whether every retained edge is valid on a logarithmic x axis."""

        return bool(self._bin_edges.size and np.all(self._bin_edges > 0.0))

    @property
    def bin_edges(self) -> np.ndarray:
        """Return a read-only view of the retained exact bin edges."""

        view = self._bin_edges.view()
        view.flags.writeable = False
        return view

    @property
    def values(self) -> np.ndarray:
        """Return a read-only view of the currently displayed y values."""

        view = self._series_values.view()
        view.flags.writeable = False
        return view

    @property
    def y_values(self) -> np.ndarray:
        """Alias for :attr:`values` with an explicit axis-oriented name."""

        return self.values

    @property
    def x_logarithmic(self) -> bool:
        return self._x_scale == "log10"

    @property
    def y_logarithmic(self) -> bool:
        return self._y_scale == "log10"

    @property
    def x_grid_divisions(self) -> int:
        """Number of equal intervals between major x-axis grid lines."""

        return self._x_grid_divisions

    @property
    def y_grid_divisions(self) -> int:
        """Number of equal intervals between major y-axis grid lines."""

        return self._y_grid_divisions

    def clear(self, message: str = "No histogram data") -> None:
        """Clear retained histogram data and expose an accessible reason."""

        self._bin_edges = np.array([], dtype=np.float64)
        self._series_values = np.empty((0, 0), dtype=np.float64)
        self._series_labels = ()
        self._series_colors = []
        self._hover_details = {}
        self._hovered_bin = None
        self.setToolTip("")
        self.setAccessibleDescription(str(message))
        self.update()

    def set_histogram(
        self,
        bin_edges,
        y_values,
        *,
        title: str = "Histogram",
        x_axis_label: str = "Value",
        y_axis_label: str = "Count",
        x_scale: str = "linear",
        y_scale: str = "linear",
        series_labels: Sequence[str] | None = None,
        colors: Sequence[QColor | str] | None = None,
        hover_details: Mapping[str, object] | None = None,
    ) -> None:
        """Replace the exact bin geometry and values displayed by the plot.

        Parameters
        ----------
        bin_edges:
            One finite, strictly increasing edge vector of length ``N + 1``.
        y_values:
            Either an ``N`` vector or a ``series x N`` matrix of finite,
            non-negative values.
        x_scale, y_scale:
            ``"linear"`` or ``"log10"``.  Logarithmic x presentation requires
            every edge to be strictly positive.  Zero-valued bars are omitted
            on a logarithmic y axis because zero has no logarithm.
        hover_details:
            Optional named vectors/matrices with the same bin geometry.  Their
            exact values are included in each bin's hover tooltip even when a
            different y representation is displayed.
        """

        edges = np.asarray(bin_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("Histogram bin edges must be a one-dimensional vector.")
        if not np.all(np.isfinite(edges)):
            raise ValueError("Histogram bin edges must all be finite.")
        if not np.all(np.diff(edges) > 0.0):
            raise ValueError("Histogram bin edges must be strictly increasing.")

        values = np.asarray(y_values)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or values.shape[0] < 1:
            raise ValueError(
                "Histogram values must contain one value per interval between edges."
            )
        if values.shape[1] != edges.size - 1:
            raise ValueError(
                "Histogram values must contain one value per interval between edges."
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError("Histogram values must be numeric.")
        values = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError("Histogram values must be finite and non-negative.")

        resolved_x_scale = _detailed_histogram_scale(x_scale, axis="x")
        resolved_y_scale = _detailed_histogram_scale(y_scale, axis="y")
        if resolved_x_scale == "log10" and np.any(edges <= 0.0):
            raise ValueError("Logarithmic x axes require strictly positive bin edges.")

        series_count = int(values.shape[0])
        if series_labels is None:
            labels = tuple(
                "Histogram" if series_count == 1 else f"Series {index + 1}"
                for index in range(series_count)
            )
        else:
            labels = tuple(str(label).strip() for label in series_labels)
            if len(labels) != series_count:
                raise ValueError("Provide exactly one label for each histogram series.")
            labels = tuple(
                label or f"Series {index + 1}" for index, label in enumerate(labels)
            )

        if colors is None:
            series_colors = _histogram_series_colors(series_count)
        else:
            if len(colors) != series_count:
                raise ValueError("Provide exactly one color for each histogram series.")
            series_colors = [
                QColor(color) if not isinstance(color, QColor) else QColor(color)
                for color in colors
            ]
            if not all(color.isValid() for color in series_colors):
                raise ValueError("Every histogram series color must be valid.")

        details: dict[str, np.ndarray] = {}
        for label, detail_values in (hover_details or {}).items():
            detail = np.asarray(detail_values)
            if detail.ndim == 1:
                detail = detail.reshape(1, -1)
            if detail.ndim != 2 or detail.shape[1] != values.shape[1]:
                raise ValueError(
                    f"Hover detail {label!r} must contain one value per histogram bin."
                )
            if detail.shape[0] not in {1, series_count}:
                raise ValueError(
                    f"Hover detail {label!r} must have one row or one row per series."
                )
            if not np.issubdtype(detail.dtype, np.number):
                raise TypeError(f"Hover detail {label!r} must be numeric.")
            detail = np.asarray(detail, dtype=np.float64)
            if not np.all(np.isfinite(detail)):
                raise ValueError(f"Hover detail {label!r} must be finite.")
            details[str(label)] = detail.copy()

        self._bin_edges = edges.copy()
        self._series_values = values.copy()
        self._series_labels = labels
        self._series_colors = series_colors
        self._hover_details = details
        self._x_scale = resolved_x_scale
        self._y_scale = resolved_y_scale
        self._title = str(title).strip()
        self._x_axis_label = str(x_axis_label).strip()
        self._y_axis_label = str(y_axis_label).strip()
        self._hovered_bin = None
        self.setToolTip("")
        self.setAccessibleName(self._title or "Detailed histogram")
        self.setAccessibleDescription(
            f"{values.shape[1]:,} bins; {series_count:,} "
            "series; "
            f"{self._x_scale} x axis; {self._y_scale} y axis."
        )
        self.update()

    def set_scales(
        self,
        *,
        x_scale: str | None = None,
        y_scale: str | None = None,
    ) -> None:
        """Change axis presentation without replacing retained histogram data."""

        resolved_x = (
            self._x_scale
            if x_scale is None
            else _detailed_histogram_scale(x_scale, axis="x")
        )
        resolved_y = (
            self._y_scale
            if y_scale is None
            else _detailed_histogram_scale(y_scale, axis="y")
        )
        if resolved_x == "log10" and not self.supports_log_x:
            raise ValueError("Logarithmic x axes require strictly positive bin edges.")
        self._x_scale = resolved_x
        self._y_scale = resolved_y
        self._hovered_bin = None
        self.setToolTip("")
        self.update()

    def set_grid_divisions(
        self,
        *,
        x: int | None = None,
        y: int | None = None,
    ) -> None:
        """Set major-grid intervals without changing retained histogram data.

        Endpoints are included as ticks, so four divisions produce five major
        grid lines. Logarithmic axes divide their transformed base-10 range.
        """

        if x is not None:
            self._x_grid_divisions = _detailed_histogram_grid_divisions(x, axis="x")
        if y is not None:
            self._y_grid_divisions = _detailed_histogram_grid_divisions(y, axis="y")
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        colors = custom_paint_colors(self.palette())
        outer = self.rect().adjusted(8, 8, -8, -8)
        painter.fillRect(outer, colors.surface)
        painter.setPen(QPen(colors.border, 1.0))
        painter.drawRect(outer)

        plot_rect = self._plot_rect()
        self._draw_title_and_labels(painter, outer, plot_rect)
        if self._series_values.size == 0:
            painter.setPen(colors.muted_text)
            painter.drawText(plot_rect, Qt.AlignCenter, "No histogram data")
            painter.end()
            return

        x_ticks = self._x_ticks()
        y_ticks = self._y_ticks()
        self._draw_grid_and_ticks(painter, plot_rect, x_ticks, y_ticks)
        self._draw_bars(painter, plot_rect)
        self._draw_axes(painter, plot_rect)
        self._draw_legend(painter, outer, plot_rect)
        painter.end()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        point = _event_position(event)
        bin_index = self._bin_at_point(point)
        if bin_index == self._hovered_bin:
            event.accept()
            return
        self._hovered_bin = bin_index
        if bin_index is None:
            self.setToolTip("")
            QToolTip.hideText()
        else:
            detail = self._bin_tooltip(bin_index)
            self.setToolTip(detail)
            QToolTip.showText(self.mapToGlobal(point), detail, self)
        self.update()
        event.accept()

    def leaveEvent(self, event) -> None:  # noqa: N802
        self._hovered_bin = None
        self.setToolTip("")
        QToolTip.hideText()
        self.update()
        super().leaveEvent(event)

    def event(self, event) -> bool:
        if event.type() == QEvent.Hide:
            self._hovered_bin = None
            self.setToolTip("")
            QToolTip.hideText()
        return super().event(event)

    def _plot_rect(self) -> QRect:
        outer = self.rect().adjusted(8, 8, -8, -8)
        tick_metrics = QFontMetrics(self._tick_label_font())
        axis_metrics = QFontMetrics(self._axis_label_font())
        title_height = tick_metrics.height() + 11 if self._title else 6
        y_tick_width = max(
            (
                tick_metrics.horizontalAdvance(
                    _format_detailed_histogram_tick(float(value))
                )
                for value in self._y_ticks()
            ),
            default=0,
        )
        y_title_band = axis_metrics.height() + 8 if self._y_axis_label else 0
        left = max(24, y_title_band + y_tick_width + 14)
        legend_width = max(outer.width() - left - 15, 1)
        legend_rows = self._legend_row_count(tick_metrics, legend_width)
        legend_height = (
            legend_rows * (tick_metrics.height() + 4) + 4
            if legend_rows
            else 0
        )
        x_tick_band = tick_metrics.height() + 7
        x_title_band = axis_metrics.height() + 8 if self._x_axis_label else 0
        bottom = x_tick_band + x_title_band + 6
        return outer.adjusted(left, title_height + legend_height, -15, -bottom)

    def _tick_label_font(self) -> QFont:
        return QFont(self.font())

    def _axis_label_font(self) -> QFont:
        font = QFont(self.font())
        point_size = font.pointSizeF()
        if point_size > 0:
            font.setPointSizeF(point_size + 1.0)
        else:
            font.setPixelSize(max(font.pixelSize() + 1, 1))
        font.setWeight(QFont.DemiBold)
        return font

    def _draw_title_and_labels(
        self,
        painter: QPainter,
        outer: QRect,
        plot_rect: QRect,
    ) -> None:
        colors = custom_paint_colors(self.palette())
        base_font = painter.font()
        painter.setFont(self._tick_label_font())
        if self._title:
            title_font = painter.font()
            title_font.setBold(True)
            painter.setFont(title_font)
            title_metrics = painter.fontMetrics()
            title = title_metrics.elidedText(
                self._title,
                Qt.ElideRight,
                max(plot_rect.width(), 1),
            )
            painter.setPen(colors.text)
            painter.drawText(
                plot_rect.left(),
                outer.top() + title_metrics.ascent() + 4,
                title,
            )
            painter.setFont(base_font)

        axis_font = self._axis_label_font()
        painter.setFont(axis_font)
        axis_metrics = painter.fontMetrics()
        painter.setPen(colors.text)
        if self._x_axis_label:
            label = axis_metrics.elidedText(
                self._x_axis_label,
                Qt.ElideRight,
                max(plot_rect.width(), 1),
            )
            painter.drawText(
                plot_rect.center().x() - axis_metrics.horizontalAdvance(label) // 2,
                outer.bottom() - axis_metrics.descent() - 8,
                label,
            )
        if self._y_axis_label:
            label = axis_metrics.elidedText(
                self._y_axis_label,
                Qt.ElideRight,
                max(plot_rect.height(), 1),
            )
            painter.save()
            painter.translate(
                outer.left() + axis_metrics.ascent() + 2,
                plot_rect.center().y()
                + axis_metrics.horizontalAdvance(label) // 2,
            )
            painter.rotate(-90)
            painter.drawText(0, 0, label)
            painter.restore()
        painter.setFont(base_font)

    def _draw_grid_and_ticks(
        self,
        painter: QPainter,
        plot_rect: QRect,
        x_ticks: Sequence[float],
        y_ticks: Sequence[float],
    ) -> None:
        colors = custom_paint_colors(self.palette())
        grid = QColor(colors.border)
        grid.setAlpha(120)
        painter.setFont(self._tick_label_font())
        metrics = painter.fontMetrics()
        painter.setPen(QPen(grid, 1.0, Qt.DotLine))
        for value in x_ticks:
            x = self._x_pixel(float(value), plot_rect)
            painter.drawLine(x, plot_rect.top(), x, plot_rect.bottom())
        for value in y_ticks:
            y = self._y_pixel(float(value), plot_rect)
            if y is not None:
                painter.drawLine(plot_rect.left(), y, plot_rect.right(), y)

        painter.setPen(colors.muted_text)
        x_label_y = plot_rect.bottom() + metrics.ascent() + 5
        x_labels: list[tuple[str, int, tuple[int, int]]] = []
        for value in x_ticks:
            label = _format_detailed_histogram_tick(float(value))
            x = self._x_pixel(float(value), plot_rect)
            label_width = metrics.horizontalAdvance(label)
            text_x = int(
                np.clip(
                    x - label_width // 2,
                    plot_rect.left(),
                    max(
                        plot_rect.left(),
                        plot_rect.right() - label_width,
                    ),
                )
            )
            x_labels.append(
                (
                    label,
                    text_x,
                    (text_x, text_x + label_width),
                )
            )
        visible_x_labels = _non_overlapping_tick_label_indices(
            [bounds for _label, _text_x, bounds in x_labels],
            gap=6,
        )
        for index, (label, text_x, _bounds) in enumerate(x_labels):
            if index not in visible_x_labels:
                continue
            painter.drawText(text_x, x_label_y, label)

        y_labels: list[tuple[str, int, int, tuple[int, int]]] = []
        for value in y_ticks:
            y = self._y_pixel(float(value), plot_rect)
            if y is None:
                continue
            label = _format_detailed_histogram_tick(float(value))
            baseline = int(
                np.clip(
                    y + metrics.ascent() // 2,
                    plot_rect.top() + metrics.ascent(),
                    plot_rect.bottom() - metrics.descent(),
                )
            )
            y_labels.append(
                (
                    label,
                    plot_rect.left() - metrics.horizontalAdvance(label) - 6,
                    baseline,
                    (baseline - metrics.ascent(), baseline + metrics.descent()),
                )
            )
        ordered_y_labels = sorted(
            enumerate(y_labels),
            key=lambda item: item[1][3][0],
        )
        visible_ordered_y_labels = _non_overlapping_tick_label_indices(
            [entry[3] for _index, entry in ordered_y_labels],
            gap=3,
        )
        visible_y_labels = {
            ordered_y_labels[index][0] for index in visible_ordered_y_labels
        }
        for index, (label, text_x, baseline, _bounds) in enumerate(y_labels):
            if index not in visible_y_labels:
                continue
            painter.drawText(
                text_x,
                baseline,
                label,
            )

    def _draw_axes(self, painter: QPainter, plot_rect: QRect) -> None:
        colors = custom_paint_colors(self.palette())
        painter.setPen(QPen(colors.axis, 1.2))
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

    def _draw_bars(self, painter: QPainter, plot_rect: QRect) -> None:
        if self._series_values.size == 0:
            return
        series_count = int(self._series_values.shape[0])
        bin_count = int(self._series_values.shape[1])
        aggregate_for_pixels = bin_count > max(int(plot_rect.width()) * 2, 512)
        for series_index, values in enumerate(self._series_values):
            color = QColor(self._series_colors[series_index])
            fill = QColor(color)
            fill.setAlpha(
                HISTOGRAM_SINGLE_SERIES_FILL_ALPHA
                if series_count == 1
                else HISTOGRAM_MULTI_SERIES_FILL_ALPHA
            )
            painter.setPen(QPen(color, 1.0))
            painter.setBrush(fill)
            if aggregate_for_pixels:
                self._draw_pixel_aggregated_bars(
                    painter,
                    plot_rect,
                    values,
                )
            else:
                left_pixels = self._x_pixels(self._bin_edges[:-1], plot_rect)
                right_pixels = self._x_pixels(self._bin_edges[1:], plot_rect)
                y_pixels = self._y_pixels(values, plot_rect)
                for left, right, y in zip(
                    left_pixels,
                    right_pixels,
                    y_pixels,
                    strict=True,
                ):
                    if y < 0:
                        continue
                    if right <= left:
                        right = left + 1
                    top = min(int(y), plot_rect.bottom() - 1)
                    rect = QRectF(
                        float(left) + 0.5,
                        float(top),
                        max(float(right - left) - 1.0, 1.0),
                        max(float(plot_rect.bottom() - top), 1.0),
                    )
                    painter.drawRect(rect)
            painter.setBrush(Qt.NoBrush)

        if self._hovered_bin is None:
            return
        index = int(self._hovered_bin)
        left = self._x_pixel(float(self._bin_edges[index]), plot_rect)
        right = self._x_pixel(float(self._bin_edges[index + 1]), plot_rect)
        highlight = QColor(custom_paint_colors(self.palette()).text)
        highlight.setAlpha(48)
        painter.fillRect(
            QRectF(
                float(left),
                float(plot_rect.top()),
                max(float(right - left), 1.0),
                float(plot_rect.height()),
            ),
            highlight,
        )

    def _draw_pixel_aggregated_bars(
        self,
        painter: QPainter,
        plot_rect: QRect,
        values: np.ndarray,
    ) -> None:
        """Draw at most one peak-preserving bar per horizontal plot pixel."""

        width = max(int(plot_rect.width()), 1)
        left_pixels = self._x_pixels(self._bin_edges[:-1], plot_rect)
        right_pixels = self._x_pixels(self._bin_edges[1:], plot_rect)
        center_pixels = (left_pixels + right_pixels) // 2 - plot_rect.left()
        center_pixels = np.clip(center_pixels, 0, width - 1)
        reduced = np.zeros(width, dtype=np.float64)
        occupied = np.zeros(width, dtype=bool)
        np.maximum.at(reduced, center_pixels, values)
        occupied[center_pixels] = True
        y_pixels = self._y_pixels(reduced, plot_rect)
        for offset in np.flatnonzero(occupied & (y_pixels >= 0)):
            top = min(int(y_pixels[offset]), plot_rect.bottom() - 1)
            painter.drawRect(
                QRectF(
                    float(plot_rect.left() + int(offset)),
                    float(top),
                    1.0,
                    max(float(plot_rect.bottom() - top), 1.0),
                )
            )

    def _draw_legend(
        self,
        painter: QPainter,
        outer: QRect,
        plot_rect: QRect,
    ) -> None:
        if len(self._series_labels) <= 1:
            return
        colors = custom_paint_colors(self.palette())
        base_font = painter.font()
        painter.setFont(self._tick_label_font())
        metrics = painter.fontMetrics()
        x = plot_rect.left()
        row_count = self._legend_row_count(metrics, max(plot_rect.width(), 1))
        row_height = metrics.height() + 4
        # `_plot_rect()` reserves a dedicated legend band below the title.
        # Anchor the legend to the bottom of that band so neither compact nor
        # expanded plots can let it drift back over the title glyphs.
        y = (
            plot_rect.top()
            - metrics.descent()
            - 4
            - max(row_count - 1, 0) * row_height
        )
        available_right = plot_rect.right() + 1
        for label, series_color in zip(
            self._series_labels,
            self._series_colors,
            strict=True,
        ):
            label_width = metrics.horizontalAdvance(label)
            required = 12 + 4 + label_width + 12
            if x > plot_rect.left() and x + required > available_right:
                x = plot_rect.left()
                y += row_height
            painter.fillRect(QRect(x, y - 9, 10, 8), series_color)
            painter.setPen(colors.text)
            available_label_width = max(available_right - x - 14, 1)
            visible_label = metrics.elidedText(
                label,
                Qt.ElideRight,
                available_label_width,
            )
            painter.drawText(x + 14, y, visible_label)
            x += 12 + 4 + metrics.horizontalAdvance(visible_label) + 12
        painter.setFont(base_font)

    def _legend_row_count(self, metrics: QFontMetrics, width: int) -> int:
        if len(self._series_labels) <= 1:
            return 0
        available_width = max(int(width), 1)
        rows = 1
        used = 0
        for label in self._series_labels:
            required = 12 + 4 + metrics.horizontalAdvance(label) + 12
            if used and used + required > available_width:
                rows += 1
                used = 0
            used += min(required, available_width)
        return rows

    def _x_ticks(self) -> tuple[float, ...]:
        if self._bin_edges.size == 0:
            return ()
        minimum = float(self._bin_edges[0])
        maximum = float(self._bin_edges[-1])
        tick_count = self._x_grid_divisions + 1
        if self._x_scale == "log10":
            transformed = np.linspace(
                np.log10(minimum),
                np.log10(maximum),
                tick_count,
            )
            return tuple(float(value) for value in np.power(10.0, transformed))
        return tuple(
            float(value) for value in np.linspace(minimum, maximum, tick_count)
        )

    def _y_ticks(self) -> tuple[float, ...]:
        positive = self._series_values[self._series_values > 0.0]
        if self._series_values.size == 0:
            return ()
        maximum = float(np.max(self._series_values))
        tick_count = self._y_grid_divisions + 1
        if self._y_scale == "linear":
            if maximum <= 0.0:
                return (0.0,)
            return tuple(
                float(value) for value in np.linspace(0.0, maximum, tick_count)
            )
        if positive.size == 0:
            return ()
        lower_exponent, upper_exponent = self._log_y_exponent_range()
        exponents = np.linspace(lower_exponent, upper_exponent, tick_count)
        return tuple(float(value) for value in np.power(10.0, exponents))

    def _log_y_exponent_range(self) -> tuple[float, float]:
        positive = self._series_values[self._series_values > 0.0]
        if positive.size == 0:
            return (0.0, 1.0)
        lower = float(np.floor(np.log10(float(np.min(positive)))))
        upper = float(np.ceil(np.log10(float(np.max(positive)))))
        if upper <= lower:
            lower -= 1.0
        return lower, upper

    def _x_pixel(self, value: float, plot_rect: QRect) -> int:
        return int(self._x_pixels(np.asarray([value]), plot_rect)[0])

    def _x_pixels(self, values, plot_rect: QRect) -> np.ndarray:
        coordinates = np.asarray(values, dtype=np.float64)
        minimum = float(self._bin_edges[0])
        maximum = float(self._bin_edges[-1])
        if self._x_scale == "log10":
            minimum = float(np.log10(minimum))
            maximum = float(np.log10(maximum))
            coordinates = np.log10(coordinates)
        fractions = (coordinates - minimum) / max(
            maximum - minimum,
            np.finfo(float).eps,
        )
        horizontal_span = max(int(plot_rect.width()) - 1, 1)
        return plot_rect.left() + np.rint(
            np.clip(fractions, 0.0, 1.0) * horizontal_span
        ).astype(np.int32)

    def _y_pixel(self, value: float, plot_rect: QRect) -> int | None:
        pixel = int(self._y_pixels(np.asarray([value]), plot_rect)[0])
        return None if pixel < 0 else pixel

    def _y_pixels(self, values, plot_rect: QRect) -> np.ndarray:
        coordinates = np.asarray(values, dtype=np.float64)
        invalid = np.zeros(coordinates.shape, dtype=bool)
        if self._y_scale == "log10":
            invalid = coordinates <= 0.0
            lower, upper = self._log_y_exponent_range()
            with np.errstate(divide="ignore", invalid="ignore"):
                fractions = (np.log10(coordinates) - lower) / max(
                    upper - lower,
                    np.finfo(float).eps,
                )
        else:
            maximum = float(np.max(self._series_values))
            if maximum <= 0.0:
                fractions = np.zeros(coordinates.shape, dtype=np.float64)
            else:
                fractions = coordinates / maximum
        vertical_span = max(int(plot_rect.height()) - 1, 1)
        pixels = plot_rect.bottom() - np.rint(
            np.clip(fractions, 0.0, 1.0) * vertical_span
        ).astype(np.int32)
        pixels[invalid] = -1
        return pixels

    def _bin_at_point(self, point) -> int | None:
        if self._bin_edges.size < 2:
            return None
        plot_rect = self._plot_rect()
        if not plot_rect.contains(point):
            return None
        fraction = float(
            np.clip(
                (float(point.x()) - plot_rect.left()) / max(plot_rect.width(), 1),
                0.0,
                1.0,
            )
        )
        minimum = float(self._bin_edges[0])
        maximum = float(self._bin_edges[-1])
        if self._x_scale == "log10":
            transformed = np.log10(minimum) + fraction * (
                np.log10(maximum) - np.log10(minimum)
            )
            value = float(np.power(10.0, transformed))
        else:
            value = minimum + fraction * (maximum - minimum)
        index = int(np.searchsorted(self._bin_edges, value, side="right") - 1)
        return int(np.clip(index, 0, self._bin_edges.size - 2))

    def _bin_tooltip(self, bin_index: int) -> str:
        left = float(self._bin_edges[bin_index])
        right = float(self._bin_edges[bin_index + 1])
        interval_close = "]" if bin_index == self._bin_edges.size - 2 else ")"
        lines = [
            f"Bin {bin_index + 1:,}",
            "Range: "
            f"[{_format_detailed_histogram_value(left)}, "
            f"{_format_detailed_histogram_value(right)}{interval_close}",
        ]
        for series_index, label in enumerate(self._series_labels):
            value = float(self._series_values[series_index, bin_index])
            prefix = f"{label}: " if len(self._series_labels) > 1 else ""
            lines.append(
                f"{prefix}{self._y_axis_label}: "
                f"{_format_detailed_histogram_value(value)}"
            )
        for label, detail in self._hover_details.items():
            if detail.shape[0] == 1:
                lines.append(
                    f"{label}: {_format_detailed_histogram_value(detail[0, bin_index])}"
                )
                continue
            values = ", ".join(
                f"{series_label} "
                f"{_format_detailed_histogram_value(detail[index, bin_index])}"
                for index, series_label in enumerate(self._series_labels)
            )
            lines.append(f"{label}: {values}")
        return "\n".join(lines)


def _detailed_histogram_scale(value: object, *, axis: str) -> str:
    normalized = str(value or "linear").strip().casefold()
    aliases = {
        "linear": "linear",
        "log": "log10",
        "log10": "log10",
        "logarithmic": "log10",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Histogram {axis} scale must be 'linear' or 'log10'."
        ) from exc


def _detailed_histogram_grid_divisions(value: object, *, axis: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Histogram {axis} grid divisions must be an integer between "
            f"{DETAILED_HISTOGRAM_MIN_GRID_DIVISIONS} and "
            f"{DETAILED_HISTOGRAM_MAX_GRID_DIVISIONS}."
        ) from exc
    if isinstance(value, (bool, np.bool_)) or not numeric.is_integer():
        raise ValueError(
            f"Histogram {axis} grid divisions must be an integer between "
            f"{DETAILED_HISTOGRAM_MIN_GRID_DIVISIONS} and "
            f"{DETAILED_HISTOGRAM_MAX_GRID_DIVISIONS}."
        )
    divisions = int(numeric)
    if not (
        DETAILED_HISTOGRAM_MIN_GRID_DIVISIONS
        <= divisions
        <= DETAILED_HISTOGRAM_MAX_GRID_DIVISIONS
    ):
        raise ValueError(
            f"Histogram {axis} grid divisions must be between "
            f"{DETAILED_HISTOGRAM_MIN_GRID_DIVISIONS} and "
            f"{DETAILED_HISTOGRAM_MAX_GRID_DIVISIONS}."
        )
    return divisions


def _non_overlapping_tick_label_indices(
    bounds: Sequence[tuple[int, int]],
    *,
    gap: int,
) -> set[int]:
    """Keep readable tick labels while leaving every requested grid line visible."""

    if not bounds:
        return set()
    if len(bounds) == 1:
        return {0}
    selected = [0]
    final_index = len(bounds) - 1
    final_start = int(bounds[final_index][0])
    for index in range(1, final_index):
        start, end = (int(value) for value in bounds[index])
        prior_end = int(bounds[selected[-1]][1])
        if start >= prior_end + gap and end + gap <= final_start:
            selected.append(index)
    prior_end = int(bounds[selected[-1]][1])
    if final_start >= prior_end + gap:
        selected.append(final_index)
    return set(selected)


def _format_detailed_histogram_tick(value: float) -> str:
    if not np.isfinite(value):
        return ""
    absolute = abs(float(value))
    if absolute == 0.0:
        return "0"
    if absolute >= 1_000_000.0 or absolute < 0.001:
        return f"{value:.2e}"
    if absolute >= 1_000.0:
        return f"{value:,.0f}"
    return f"{value:.4g}"


def _format_detailed_histogram_value(value: float) -> str:
    if not np.isfinite(value):
        return ""
    absolute = abs(float(value))
    if absolute != 0.0 and (absolute >= 1_000_000.0 or absolute < 0.0001):
        return f"{value:.6e}"
    if abs(value - round(value)) < 1e-12:
        return f"{int(round(value)):,}"
    return f"{value:.8g}"


def _prepare_colocalization_scatter_density(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    threshold_1: float,
    threshold_2: float,
    roi_mask: np.ndarray | None,
    intensity_max: float,
    bins: int,
    range_percentile: float = 100.0,
    progress=None,
    chunk_elements: int = SCATTER_DENSITY_CHUNK_ELEMENTS,
    include_full_ranges: bool = False,
) -> tuple:
    """Return exact density/counts and independent populated axis ranges."""
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

    bins = int(np.clip(int(bins), 32, 4_096))
    chunk_elements = int(chunk_elements)
    range_1, range_2 = colocalization_populated_ranges(
        ch1,
        ch2,
        roi_mask=roi_mask,
        percentile=range_percentile,
    )
    if include_full_ranges and float(range_percentile) < 100.0:
        full_range_1, full_range_2 = colocalization_populated_ranges(
            ch1,
            ch2,
            roi_mask=roi_mask,
            percentile=100.0,
        )
    else:
        full_range_1, full_range_2 = range_1, range_2
    del intensity_max
    edges_1 = np.linspace(range_1[0], range_1[1], bins + 1)
    edges_2 = np.linspace(range_2[0], range_2[1], bins + 1)
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
                bins=(edges_1, edges_2),
            )
            density_counts += chunk_density
            del chunk_density

    result = (
        density_counts,
        roi_voxels,
        colocalized_voxels,
        range_1[0],
        range_1[1],
        range_2[0],
        range_2[1],
    )
    if not include_full_ranges:
        return result
    return result + (
        full_range_1[0],
        full_range_1[1],
        full_range_2[0],
        full_range_2[1],
    )


def _count_colocalization_thresholds(
    channel_1: np.ndarray,
    channel_2: np.ndarray,
    *,
    threshold_1: float,
    threshold_2: float,
    roi_mask: np.ndarray | None,
    progress=None,
    chunk_elements: int = SCATTER_DENSITY_CHUNK_ELEMENTS,
) -> tuple[int, int]:
    """Count the complete ROI and threshold intersection without a histogram."""
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
    chunk_elements = max(int(chunk_elements), 1)
    threshold_1 = float(threshold_1)
    threshold_2 = float(threshold_2)
    roi_voxels = 0
    colocalized_voxels = 0
    for start in range(0, int(flat_1.size), chunk_elements):
        if progress is not None:
            progress.check_cancelled()
        stop = min(start + chunk_elements, int(flat_1.size))
        positive = np.greater_equal(flat_1[start:stop], threshold_1)
        np.logical_and(
            positive,
            flat_2[start:stop] >= threshold_2,
            out=positive,
        )
        if flat_roi is None:
            roi_voxels += stop - start
        else:
            chunk_roi = flat_roi[start:stop]
            roi_voxels += int(np.count_nonzero(chunk_roi))
            np.logical_and(positive, chunk_roi, out=positive)
        colocalized_voxels += int(np.count_nonzero(positive))
    return roi_voxels, colocalized_voxels


class ColocalizationScatterPlot(QWidget):
    """Interactive two-channel scatter-density plot with threshold guides."""

    # ``thresholdChanged`` is a lightweight live-preview signal.  Consumers
    # must not use it to invalidate or recalculate a workflow while the mouse
    # is still moving.  ``thresholdCommitted`` is emitted once on release and
    # is the authoring boundary for expensive work.
    thresholdChanged = Signal(int, float)
    thresholdCommitted = Signal(int, float)
    gestureStarted = Signal()
    gestureFinished = Signal()

    def __init__(self, parent=None, *, max_density_bins: int | None = None):
        super().__init__(parent)
        self._max_density_bins = max(
            int(
                COLOCALIZATION_SCATTER_DISPLAY_MAX_BINS
                if max_density_bins is None
                else max_density_bins
            ),
            1,
        )
        self._image: QImage | None = None
        self._density_counts: np.ndarray | None = None
        self._log_counts = True
        self._threshold_1 = 25.0
        self._threshold_2 = 25.0
        self._intensity_min = 0.0
        self._intensity_max = 255.0
        self._channel_1_min = 0.0
        self._channel_1_max = 255.0
        self._channel_2_min = 0.0
        self._channel_2_max = 255.0
        self._full_channel_1_min = 0.0
        self._full_channel_1_max = 255.0
        self._full_channel_2_min = 0.0
        self._full_channel_2_max = 255.0
        self._display_channel_1_min = 0.0
        self._display_channel_1_max = 255.0
        self._display_channel_2_min = 0.0
        self._display_channel_2_max = 255.0
        self._zoom_to_data = False
        self._equal_axes = True
        self._channel_1_color = QColor("#ef4444")
        self._channel_2_color = QColor("#22c55e")
        self._colormap = "Viridis"
        self._summary = ""
        self._drag_axis: int | None = None
        self._gesture_active = False
        self.setMinimumHeight(300)
        self.setMouseTracking(True)

    def set_density(
        self,
        density_counts: np.ndarray | None,
        *,
        threshold_1: float,
        threshold_2: float,
        intensity_min: float = 0.0,
        intensity_max: float = 255.0,
        channel_1_range: tuple[float, float] | None = None,
        channel_2_range: tuple[float, float] | None = None,
        full_channel_1_range: tuple[float, float] | None = None,
        full_channel_2_range: tuple[float, float] | None = None,
        channel_1_color: object = "Red",
        channel_2_color: object = "Green",
        colormap: str = "Viridis",
        log_counts: bool = True,
        summary: str = "",
    ) -> None:
        """Render worker-prepared density counts without touching source images."""
        self._threshold_1 = float(threshold_1)
        self._threshold_2 = float(threshold_2)
        shared_range = (float(intensity_min), float(intensity_max))
        self._channel_1_min, self._channel_1_max = _valid_scatter_axis_range(
            channel_1_range or shared_range
        )
        self._channel_2_min, self._channel_2_max = _valid_scatter_axis_range(
            channel_2_range or shared_range
        )
        self._full_channel_1_min, self._full_channel_1_max = (
            _valid_scatter_axis_range(
                full_channel_1_range
                or (self._channel_1_min, self._channel_1_max)
            )
        )
        self._full_channel_2_min, self._full_channel_2_max = (
            _valid_scatter_axis_range(
                full_channel_2_range
                or (self._channel_2_min, self._channel_2_max)
            )
        )
        self._density_counts = (
            None
            if density_counts is None
            else cap_colocalization_scatter_density_for_display(
                np.asarray(density_counts),
                max_bins=self._max_density_bins,
            )
        )
        # Backward-compatible union used by older callers/tests that inspect
        # these presentation fields directly.
        self._intensity_min = min(self._channel_1_min, self._channel_2_min)
        self._intensity_max = max(self._channel_1_max, self._channel_2_max)
        self._update_display_ranges()
        self._channel_1_color = _qcolor_from_channel_color(
            channel_1_color,
            fallback="#ef4444",
        )
        self._channel_2_color = _qcolor_from_channel_color(
            channel_2_color,
            fallback="#22c55e",
        )
        self._colormap = str(colormap or "Viridis")
        self._log_counts = bool(log_counts)
        self._summary = str(summary)
        self._image = self._density_image(
            self._density_counts,
            log_counts=self._log_counts,
        )
        self.update()

    def set_colormap(self, colormap: str) -> None:
        """Recolor the retained presentation density without recomputing it."""
        self._colormap = str(colormap or "Viridis")
        if self._image is not None and (
            self._image.format() == QImage.Format_Indexed8
        ):
            recolored = QImage(self._image)
            recolored.setColorTable(
                _colocalization_scatter_color_table(self._colormap)
            )
            self._image = recolored
        else:
            self._image = self._density_image(
                self._density_counts,
                log_counts=self._log_counts,
            )
        self.update()

    def set_log_counts(self, enabled: bool) -> None:
        """Switch density transfer functions without recomputing the histogram."""

        enabled = bool(enabled)
        if self._log_counts == enabled:
            return
        self._log_counts = enabled
        self._image = self._density_image(
            self._density_counts,
            log_counts=self._log_counts,
        )
        self.update()

    def set_zoom_to_data(self, enabled: bool) -> None:
        """Crop the presentation axes to populated data without changing bins."""

        enabled = bool(enabled)
        if self._zoom_to_data == enabled:
            return
        self._zoom_to_data = enabled
        self._update_display_ranges()
        self.update()

    def set_equal_axes(self, enabled: bool) -> None:
        """Use equal numeric spans for both axes so slopes are not distorted."""

        enabled = bool(enabled)
        if self._equal_axes == enabled:
            return
        self._equal_axes = enabled
        self._update_display_ranges()
        self.update()

    def clear(self, message: str = "Connect two channel inputs.") -> None:
        self._cancel_threshold_drag()
        self.unsetCursor()
        self._image = None
        self._density_counts = None
        self._summary = message
        self.update()

    def set_pending_thresholds(
        self,
        *,
        threshold_1: float,
        threshold_2: float,
        intensity_max: float = 255.0,
        preserve_density: bool,
        summary: str = "Calculating exact count...",
    ) -> None:
        """Move guides and mark exact counts pending without stale values."""
        self._threshold_1 = float(threshold_1)
        self._threshold_2 = float(threshold_2)
        self._summary = str(summary)
        if not preserve_density:
            self._cancel_threshold_drag()
            self.unsetCursor()
            self._density_counts = None
            self._intensity_min = 0.0
            self._intensity_max = max(float(intensity_max), 1.0)
            self._channel_1_min = self._intensity_min
            self._channel_1_max = self._intensity_max
            self._channel_2_min = self._intensity_min
            self._channel_2_max = self._intensity_max
            self._full_channel_1_min = self._intensity_min
            self._full_channel_1_max = self._intensity_max
            self._full_channel_2_min = self._intensity_min
            self._full_channel_2_max = self._intensity_max
            self._update_display_ranges()
            self._image = None
        self.update()

    def paintEvent(self, event):  # noqa: N802
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        rect = self.rect().adjusted(8, 8, -8, -8)
        colors = custom_paint_colors(self.palette())
        painter.fillRect(rect, colors.surface)
        painter.setPen(QPen(colors.border, 1))
        painter.drawRect(rect)

        plot_rect = self._plot_rect()
        if self._image is None:
            painter.setPen(colors.muted_text)
            painter.drawText(rect, Qt.AlignCenter, self._summary or "No data")
            painter.end()
            return

        # The retained density is binned over the populated source ranges.  The
        # presentation can include zero or share one range across both axes,
        # so draw that density only into the corresponding data rectangle
        # instead of relabelling/stretching it across the wider view.
        painter.fillRect(plot_rect, QColor(4, 7, 15))
        painter.save()
        painter.setClipRect(plot_rect)
        painter.drawImage(self._density_target_rect(plot_rect), self._image)
        painter.restore()
        painter.setPen(QPen(colors.axis, 1.2))
        painter.drawRect(plot_rect)
        self._draw_thresholds(painter, plot_rect)
        self._draw_labels(painter, rect, plot_rect)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        point = _event_position(event)
        drag_axis = self._threshold_axis_at_point(point)
        if drag_axis is None:
            super().mousePressEvent(event)
            return
        self._drag_axis = drag_axis
        self._set_threshold_cursor(drag_axis)
        self._begin_gesture()
        self._emit_threshold_from_point(point, self._plot_rect())
        event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        point = _event_position(event)
        if self._drag_axis is not None:
            self._set_threshold_cursor(self._drag_axis)
            self._emit_threshold_from_point(point, self._plot_rect())
            event.accept()
            return
        self._set_threshold_cursor(self._threshold_axis_at_point(point))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._drag_axis is None or event.button() != Qt.LeftButton:
            super().mouseReleaseEvent(event)
            return
        point = _event_position(event)
        drag_axis = self._drag_axis
        self._emit_threshold_from_point(point, self._plot_rect())
        value = self._threshold_1 if drag_axis == 1 else self._threshold_2
        self.thresholdCommitted.emit(drag_axis, value)
        self._cancel_threshold_drag()
        self._set_threshold_cursor(self._threshold_axis_at_point(point))
        event.accept()

    def event(self, event) -> bool:
        if event.type() in {QEvent.Hide, QEvent.UngrabMouse}:
            self._cancel_threshold_drag()
            self.unsetCursor()
        elif event.type() == QEvent.Leave and self._drag_axis is None:
            self.unsetCursor()
        return super().event(event)

    def _threshold_axis_at_point(self, point) -> int | None:
        """Return the draggable guide under ``point``, if any."""

        if self._image is None:
            return None
        plot_rect = self._plot_rect()
        tolerance = COLOCALIZATION_SCATTER_THRESHOLD_HIT_TOLERANCE
        if not plot_rect.adjusted(
            -tolerance,
            -tolerance,
            tolerance,
            tolerance,
        ).contains(point):
            return None
        vertical_x = self._x_from_value(self._threshold_1, plot_rect)
        horizontal_y = self._y_from_value(self._threshold_2, plot_rect)
        candidates = []
        dx = abs(point.x() - vertical_x)
        dy = abs(point.y() - horizontal_y)
        if dx <= tolerance:
            candidates.append((dx, 1))
        if dy <= tolerance:
            candidates.append((dy, 2))
        if not candidates:
            return None
        return min(candidates)[1]

    def _set_threshold_cursor(self, axis: int | None) -> None:
        if axis == 1:
            self.setCursor(Qt.SizeHorCursor)
        elif axis == 2:
            self.setCursor(Qt.SizeVerCursor)
        else:
            self.unsetCursor()

    def _begin_gesture(self) -> None:
        if self._gesture_active:
            return
        self._gesture_active = True
        self.gestureStarted.emit()

    def _cancel_threshold_drag(self) -> None:
        self._drag_axis = None
        if not self._gesture_active:
            return
        self._gesture_active = False
        self.gestureFinished.emit()

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
        hist = cap_colocalization_scatter_density_for_display(
            hist,
            max_bins=self._max_density_bins,
        )
        # Convert in short float32 row bands into an indexed 8-bit texture.
        # This keeps a genuine 4096-bin pop-out near a 16 MiB raster plus about
        # 1 MiB scratch instead of allocating full float and RGB work planes.
        height, width = int(hist.shape[1]), int(hist.shape[0])
        bytes_per_line = (width + 3) & ~3
        indices = np.zeros((height, bytes_per_line), dtype=np.uint8)
        maximum = float(np.max(hist))
        transformed_maximum = (
            float(np.log1p(maximum)) if bool(log_counts) else maximum
        )
        source_rows = hist.T[::-1]
        if transformed_maximum > 0.0:
            for start in range(0, height, 64):
                stop = min(start + 64, height)
                values = np.array(
                    source_rows[start:stop],
                    dtype=np.float32,
                    order="C",
                    copy=True,
                )
                positive = values > 0.0
                if bool(log_counts):
                    np.log1p(values, out=values)
                values /= transformed_maximum
                np.sqrt(values, out=values)
                values *= 254.0
                np.rint(values, out=values)
                np.clip(values, 0.0, 254.0, out=values)
                target = indices[start:stop, :width]
                target[:] = values.astype(np.uint8)
                target[positive] += 1
        image = QImage(
            indices.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_Indexed8,
        ).copy()
        image.setColorTable(_colocalization_scatter_color_table(self._colormap))
        return image

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
        base_font = painter.font()
        painter.setFont(self._tick_label_font())
        metrics = painter.fontMetrics()
        colors = custom_paint_colors(self.palette())
        axis_value_color = colors.muted_text
        axis_title_color = colors.text
        x_min_label = _format_histogram_label(self._display_channel_1_min)
        x_max_label = _format_histogram_label(self._display_channel_1_max)
        y_min_label = _format_histogram_label(self._display_channel_2_min)
        y_max_label = _format_histogram_label(self._display_channel_2_max)

        painter.setPen(axis_value_color)
        axis_value_y = plot_rect.bottom() + metrics.ascent() + 6
        painter.drawText(plot_rect.left(), axis_value_y, x_min_label)
        painter.drawText(
            plot_rect.right() - metrics.horizontalAdvance(x_max_label),
            axis_value_y,
            x_max_label,
        )

        x_label = "Ch 1 intensity"
        axis_font = self._axis_label_font()
        painter.setFont(axis_font)
        painter.setPen(axis_title_color)
        axis_metrics = painter.fontMetrics()
        painter.drawText(
            plot_rect.center().x() - axis_metrics.horizontalAdvance(x_label) // 2,
            axis_value_y + axis_metrics.height(),
            x_label,
        )

        painter.setFont(self._tick_label_font())
        painter.setPen(axis_value_color)
        metrics = painter.fontMetrics()
        y_value_x = plot_rect.left() - metrics.horizontalAdvance(y_max_label) - 8
        painter.drawText(y_value_x, plot_rect.top() + metrics.ascent(), y_max_label)
        painter.drawText(
            plot_rect.left() - metrics.horizontalAdvance(y_min_label) - 8,
            plot_rect.bottom(),
            y_min_label,
        )

        y_label = "Ch 2 intensity"
        painter.setFont(axis_font)
        axis_metrics = painter.fontMetrics()
        painter.save()
        painter.translate(
            plot_rect.left()
            - max(
                metrics.horizontalAdvance(y_min_label),
                metrics.horizontalAdvance(y_max_label),
            )
            - axis_metrics.ascent()
            - 12,
            plot_rect.center().y() + axis_metrics.horizontalAdvance(y_label) // 2,
        )
        painter.rotate(-90)
        painter.setPen(axis_title_color)
        painter.drawText(0, 0, y_label)
        painter.restore()

        painter.setFont(self._tick_label_font())
        metrics = painter.fontMetrics()
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
            painter.setPen(axis_value_color)
            painter.drawText(
                max(plot_rect.left(), plot_rect.right() - summary_width),
                plot_rect.top() + metrics.ascent() + 2,
                self._summary,
            )
        painter.setFont(base_font)

    def _plot_rect(self) -> QRect:
        rect = self.rect().adjusted(8, 8, -8, -8)
        tick_metrics = QFontMetrics(self._tick_label_font())
        axis_metrics = QFontMetrics(self._axis_label_font())
        range_labels = (
            _format_histogram_label(self._display_channel_1_min),
            _format_histogram_label(self._display_channel_1_max),
            _format_histogram_label(self._display_channel_2_min),
            _format_histogram_label(self._display_channel_2_max),
        )
        value_width = max(
            *(tick_metrics.horizontalAdvance(label) for label in range_labels),
        )
        left_margin = max(56, value_width + axis_metrics.height() + 22)
        right_margin = 10
        top_margin = 8
        bottom_margin = tick_metrics.height() + axis_metrics.height() + 14
        available_width = max(1, rect.width() - left_margin - right_margin)
        available_height = max(1, rect.height() - top_margin - bottom_margin)
        side = max(1, min(available_width, available_height))
        x = rect.left() + left_margin + (available_width - side) // 2
        y = rect.top() + top_margin + (available_height - side) // 2
        return QRect(x, y, side, side)

    def _x_from_value(self, value: float, plot_rect: QRect) -> int:
        span = self._display_channel_1_max - self._display_channel_1_min
        fraction = float(
            np.clip((value - self._display_channel_1_min) / span, 0.0, 1.0)
        )
        drawable_width = max(plot_rect.width() - 1, 0)
        return plot_rect.left() + int(round(fraction * drawable_width))

    def _y_from_value(self, value: float, plot_rect: QRect) -> int:
        span = self._display_channel_2_max - self._display_channel_2_min
        fraction = float(
            np.clip((value - self._display_channel_2_min) / span, 0.0, 1.0)
        )
        drawable_height = max(plot_rect.height() - 1, 0)
        return plot_rect.bottom() - int(round(fraction * drawable_height))

    def _value_from_x(self, x: int, plot_rect: QRect) -> float:
        fraction = (float(x) - plot_rect.left()) / max(plot_rect.width() - 1, 1)
        span = self._display_channel_1_max - self._display_channel_1_min
        return float(
            self._display_channel_1_min + np.clip(fraction, 0.0, 1.0) * span
        )

    def _value_from_y(self, y: int, plot_rect: QRect) -> float:
        fraction = (plot_rect.bottom() - float(y)) / max(plot_rect.height() - 1, 1)
        span = self._display_channel_2_max - self._display_channel_2_min
        return float(
            self._display_channel_2_min + np.clip(fraction, 0.0, 1.0) * span
        )

    def _tick_label_font(self) -> QFont:
        return QFont(self.font())

    def _axis_label_font(self) -> QFont:
        font = QFont(self.font())
        point_size = font.pointSizeF()
        if point_size > 0:
            font.setPointSizeF(point_size + 1.0)
        else:
            font.setPixelSize(max(font.pixelSize() + 1, 1))
        font.setWeight(QFont.DemiBold)
        return font

    def _update_display_ranges(self) -> None:
        data_1 = (self._channel_1_min, self._channel_1_max)
        data_2 = (self._channel_2_min, self._channel_2_max)
        if self._zoom_to_data:
            # The histogram is binned over the explicitly selected populated
            # percentile.  Those bounds, rather than its inevitably occupied
            # edge bins, are the authoritative zoom target.
            display_1, display_2 = data_1, data_2
        else:
            full_1 = (self._full_channel_1_min, self._full_channel_1_max)
            full_2 = (self._full_channel_2_min, self._full_channel_2_max)
            display_1 = (min(0.0, full_1[0]), max(0.0, full_1[1]))
            display_2 = (min(0.0, full_2[0]), max(0.0, full_2[1]))
        if self._equal_axes:
            if self._zoom_to_data:
                shared_span = max(
                    display_1[1] - display_1[0],
                    display_2[1] - display_2[0],
                )
                available = (
                    min(data_1[0], data_2[0]),
                    max(data_1[1], data_2[1]),
                )
                display_1 = _expand_scatter_range(
                    display_1,
                    span=shared_span,
                    available=available,
                )
                display_2 = _expand_scatter_range(
                    display_2,
                    span=shared_span,
                    available=available,
                )
            else:
                shared = _valid_scatter_axis_range(
                    (
                        min(display_1[0], display_2[0]),
                        max(display_1[1], display_2[1]),
                    )
                )
                display_1 = shared
                display_2 = shared
        self._display_channel_1_min, self._display_channel_1_max = (
            _valid_scatter_axis_range(display_1)
        )
        self._display_channel_2_min, self._display_channel_2_max = (
            _valid_scatter_axis_range(display_2)
        )

    def _density_target_rect(self, plot_rect: QRect) -> QRectF:
        x_span = self._display_channel_1_max - self._display_channel_1_min
        y_span = self._display_channel_2_max - self._display_channel_2_min
        plot_left = float(plot_rect.left())
        plot_bottom = float(plot_rect.bottom() + 1)
        left = plot_left + (
            (self._channel_1_min - self._display_channel_1_min) / x_span
        ) * plot_rect.width()
        right = plot_left + (
            (self._channel_1_max - self._display_channel_1_min) / x_span
        ) * plot_rect.width()
        top = plot_bottom - (
            (self._channel_2_max - self._display_channel_2_min) / y_span
        ) * plot_rect.height()
        bottom = plot_bottom - (
            (self._channel_2_min - self._display_channel_2_min) / y_span
        ) * plot_rect.height()
        return QRectF(left, top, max(right - left, 1.0), max(bottom - top, 1.0))


def _colocalization_scatter_color_table(colormap: str) -> list[int]:
    """Build the indexed density texture's compact 256-colour table."""

    colors = np.asarray(
        _apply_monochrome_colormap(
            np.arange(256, dtype=np.uint8),
            colormap,
        ),
        dtype=np.uint8,
    ).reshape(256, 3)
    colors[0] = (4, 7, 15)
    return [
        QColor(int(red), int(green), int(blue)).rgb()
        for red, green, blue in colors
    ]


def _valid_scatter_axis_range(
    values: tuple[float, float],
) -> tuple[float, float]:
    minimum, maximum = (float(value) for value in values)
    if not np.isfinite(minimum) or not np.isfinite(maximum):
        raise ValueError("Scatter axis ranges must be finite.")
    if maximum <= minimum:
        maximum = minimum + 1.0
    return minimum, maximum


def _expand_scatter_range(
    values: tuple[float, float],
    *,
    span: float,
    available: tuple[float, float],
) -> tuple[float, float]:
    """Expand around a populated range while remaining in shared data bounds."""

    minimum, maximum = _valid_scatter_axis_range(values)
    available_min, available_max = _valid_scatter_axis_range(available)
    target_span = min(
        max(float(span), maximum - minimum),
        available_max - available_min,
    )
    center = (minimum + maximum) / 2.0
    expanded_min = center - target_span / 2.0
    expanded_max = center + target_span / 2.0
    if expanded_min < available_min:
        expanded_max += available_min - expanded_min
        expanded_min = available_min
    if expanded_max > available_max:
        expanded_min -= expanded_max - available_max
        expanded_max = available_max
    return _valid_scatter_axis_range((expanded_min, expanded_max))


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
    channel_colors: Sequence[object] | str | None = None,
) -> list[QColor]:
    if count <= 0:
        return []
    normalized_axis_name = str(channel_axis_name).strip().casefold()
    if normalized_axis_name == "rgb":
        base = [QColor("#ef4444"), QColor("#22c55e"), QColor("#60a5fa")]
    elif normalized_axis_name == "rgba":
        base = [
            QColor("#ef4444"),
            QColor("#22c55e"),
            QColor("#60a5fa"),
            QColor("#d1d5db"),
        ]
    elif count > 1:
        base = [_qcolor_from_unit_rgb(color) for color in FLUORESCENCE_COLORS]
    else:
        base = [QColor("#60a5fa")]
    colors = [QColor(base[index % len(base)]) for index in range(count)]
    if isinstance(channel_colors, str):
        authored_colors: Sequence[object] = tuple(
            part.strip() for part in channel_colors.split(",") if part.strip()
        )
    else:
        authored_colors = () if channel_colors is None else channel_colors
    for index, value in enumerate(authored_colors):
        if index >= count:
            break
        colors[index] = _qcolor_from_channel_color(
            value,
            fallback=colors[index].name(),
        )
    return colors


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
