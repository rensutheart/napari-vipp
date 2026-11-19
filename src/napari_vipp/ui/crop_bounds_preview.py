"""A raster-painted schematic of crop margins, independent of napari's canvas."""

from __future__ import annotations

from operator import index

from qtpy.QtCore import QRectF, QSize, Qt
from qtpy.QtGui import QColor, QPainter, QPalette, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget

_Axis = tuple[int, int, int]  # sample count, margin at start, margin at end


def _valid_axis(axis: _Axis | None) -> _Axis | None:
    if axis is None:
        return None
    try:
        n, start, end = (index(value) for value in axis)
    except (TypeError, ValueError):
        return None
    return (n, start, end) if min(start, end) >= 0 and start + end < n else None


def _inner_interval(axis: _Axis, origin: float, extent: float) -> tuple[float, float]:
    n, start, end = axis
    return origin + extent * (start / n), extent * ((n - start - end) / n)


class CropBoundsPreview(QWidget):
    """Show the retained XY rectangle and optional Z interval, never pixel data."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bounds: tuple[_Axis, _Axis, _Axis | None] | None = None
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setFocusPolicy(Qt.NoFocus)
        self.setAccessibleName("Crop bounds schematic")
        self.setAccessibleDescription("Crop bounds unavailable.")
        self.setToolTip(
            "Shaded regions indicate retained samples. This is a schematic of "
            "array bounds, not a microscopy image or a physical-scale drawing."
        )

    def sizeHint(self) -> QSize:  # noqa: N802
        line = self.fontMetrics().height()
        return QSize(230, 68 + 2 * line)

    def minimumSizeHint(self) -> QSize:  # noqa: N802
        return QSize(0, self.sizeHint().height())

    def set_bounds(
        self, *, x: _Axis | None = None, y: _Axis | None = None, z: _Axis | None = None
    ) -> None:
        """Set integer sample bounds; discard stale drawings on incomplete input."""
        xx, yy, zz = _valid_axis(x), _valid_axis(y), _valid_axis(z)
        bounds = (xx, yy, zz) if xx and yy and (z is None or zz) else None
        if bounds == self._bounds:
            return
        self._bounds = bounds
        text = "Crop bounds unavailable."
        if bounds:
            spans = []
            for label, axis in zip(("X", "Y", "Z"), bounds, strict=True):
                if axis:
                    n, first, last = axis
                    spans.append(f"{label} {first + 1}–{n - last} of {n}")
            text = "Retained samples: " + ", ".join(spans) + "."
        self.setAccessibleDescription(text)
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QPainter(self)
        try:
            palette = self.palette()
            fg = palette.color(QPalette.WindowText)
            painter.setPen(fg)
            line = self.fontMetrics().height()
            w = self.width()
            title = (
                "Crop bounds (schematic)" if self._bounds else "Crop bounds unavailable"
            )
            painter.drawText(
                QRectF(0, 0, w, line),
                Qt.AlignLeft | Qt.AlignVCenter,
                self.fontMetrics().elidedText(title, Qt.ElideRight, max(0, w)),
            )
            if not self._bounds:
                return
            x, y, z = self._bounds
            label_w = self.fontMetrics().horizontalAdvance("XY") + 6
            left, top = label_w + 2, line + 5
            width = w - left - 3
            height = self.height() - top - (line + 7 if z else 4)
            if min(width, height) < 2:
                return
            # Preserve XY sample aspect (not physical calibration). One-pixel
            # landmarks remain visible at display size without allocating arrays.
            ratio = min(width / x[0], height / y[0])
            fw, fh = max(1.0, x[0] * ratio), max(1.0, y[0] * ratio)
            frame = QRectF(left + (width - fw) / 2, top + (height - fh) / 2, fw, fh)
            px, pw = _inner_interval(x, frame.x(), fw)
            py, ph = _inner_interval(y, frame.y(), fh)
            shade = QColor(palette.color(QPalette.Highlight))
            shade.setAlpha(80)
            edge = palette.color(QPalette.Highlight)
            outline = palette.color(QPalette.Mid)

            def draw_interval(outer, inner):
                painter.fillRect(outer, palette.color(QPalette.Base))
                painter.setPen(QPen(outline, 1))
                painter.drawRect(outer)
                painter.fillRect(inner, shade)
                painter.setPen(QPen(edge, 1))
                painter.drawRect(inner)

            painter.drawText(QRectF(0, top, label_w, height), Qt.AlignVCenter, "XY")
            draw_interval(frame, QRectF(px, py, pw, ph))
            if z:
                by = top + height + 5
                painter.setPen(fg)
                painter.drawText(QRectF(0, by, label_w, line), Qt.AlignVCenter, "Z")
                bar = QRectF(left, by + (line - 7) / 2, width, 7)
                pz, dz = _inner_interval(z, bar.x(), bar.width())
                draw_interval(bar, QRectF(pz, bar.y(), dz, bar.height()))
        finally:
            painter.end()
