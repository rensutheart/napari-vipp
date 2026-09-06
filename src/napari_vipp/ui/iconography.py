"""Small, palette-aware scientific icons used by VIPP's Qt interface."""

from __future__ import annotations

from functools import lru_cache

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import (
    QColor,
    QIcon,
    QPainter,
    QPainterPath,
    QPalette,
    QPen,
    QPixmap,
    QPolygonF,
)

from napari_vipp._theme import category_color
from napari_vipp.core.pipeline import OperationSpec
from napari_vipp.ui.palette_roles import theme_colors

_CATEGORY_ICON_KINDS = {
    "Image Data": "image",
    "Intensity & Contrast": "sun",
    "Filtering": "waves",
    "Projection": "layers",
    "Segmentation": "regions",
    "Morphology": "contours",
    "Label Operations": "tag",
    "Measurements": "ruler",
    "Colocalization & Spatial Analysis": "overlap",
}

_OPERATION_ICON_KINDS = {
    "input": "image-source",
    "save_output": "save",
    "batch_output": "save",
    "crop_stack": "crop",
    "select_axis_slice": "axes",
    "split_axis": "axes",
    "reorder_axes": "axes",
    "rescale_axes": "axes",
    "set_pixel_size": "ruler",
    "extract_channel": "channels",
    "combine_channels": "channels",
    "split_channels": "channels",
    "composite_to_rgb": "channels",
    "assign_channel_colors": "channels",
    "calculate_weighted_image": "math",
    "add_images": "math",
    "subtract_images": "math",
    "ratio_image": "math",
    "mask_image": "math",
    "logical_and": "math",
    "logical_or": "math",
    "logical_xor": "math",
    "invert": "math",
    "otsu_threshold": "threshold",
    "triangle_threshold": "threshold",
    "imagej_auto_threshold": "threshold",
    "li_threshold": "threshold",
    "yen_threshold": "threshold",
    "isodata_threshold": "threshold",
    "minimum_threshold": "threshold",
    "binary_threshold": "threshold",
    "hysteresis_threshold": "threshold",
    "adaptive_mean_threshold": "threshold",
    "adaptive_gaussian_threshold": "threshold",
    "sauvola_threshold": "threshold",
    "niblack_threshold": "threshold",
    "mip": "layers",
    "project_image": "layers",
    "orthogonal_projection": "layers",
    "label_connected_components": "tag",
    "relabel_sequential": "tag",
    "label_skeleton_components": "tag",
    "label_skeleton_branches": "tag",
    "merge_tables": "table",
    "add_metadata_columns": "table",
    "select_table_columns": "table",
    "summarize_measurements": "table",
}


def category_icon_kind(category: str) -> str:
    """Return the stable scientific glyph family for ``category``."""

    return _CATEGORY_ICON_KINDS.get(str(category), "nodes")


def operation_icon_kind(spec: OperationSpec) -> str:
    """Return a stable glyph selected by operation ID, then category."""

    return _OPERATION_ICON_KINDS.get(spec.id, category_icon_kind(spec.category))


def palette_category_colors(
    category: str,
    palette: QPalette,
) -> tuple[QColor, QColor, QColor]:
    """Return readable text, subtle row fill, and category accent colors."""

    text = QColor(palette.color(QPalette.Text))
    base = QColor(palette.color(QPalette.Base))
    accent = QColor(category_color(category))
    tint = _blend(base, accent, 0.12)
    accent = _ensure_contrast(accent, tint, text)
    return text, tint, accent


def palette_branch_color(category: str, palette: QPalette) -> QColor:
    """Return a category-colored disclosure glyph readable in either theme."""

    text, tint, accent = palette_category_colors(category, palette)
    base = QColor(palette.color(QPalette.Base))
    backgrounds = (base, tint)
    if all(_contrast_ratio(accent, background) >= 4.5 for background in backgrounds):
        return accent
    for step in range(1, 9):
        candidate = _blend(accent, text, step / 8.0)
        if all(
            _contrast_ratio(candidate, background) >= 4.5
            for background in backgrounds
        ):
            return candidate
    return text


def category_icon(category: str, palette: QPalette, size: int = 18) -> QIcon:
    """Return a theme-aware category icon with normal and disabled states."""

    _text, tint, accent = palette_category_colors(category, palette)
    disabled = _blend(accent, tint, 0.60)
    return _icon(category_icon_kind(category), accent, disabled, int(size))


def operation_icon(
    spec: OperationSpec,
    palette: QPalette,
    size: int = 16,
) -> QIcon:
    """Return a supplemental operation icon using its category accent."""

    _text, tint, accent = palette_category_colors(spec.category, palette)
    disabled = _blend(accent, tint, 0.60)
    return _icon(operation_icon_kind(spec), accent, disabled, int(size))


def interface_icon(
    kind: str,
    palette: QPalette,
    size: int = 18,
) -> QIcon:
    """Return a neutral palette-aware icon for node-library controls."""

    foreground = QColor(palette.color(QPalette.ButtonText))
    background = QColor(palette.color(QPalette.Button))
    disabled = _blend(foreground, background, 0.62)
    return _icon(str(kind), foreground, disabled, int(size))


def data_type_icon(
    data_type: str,
    palette: QPalette,
    size: int = 16,
) -> QIcon:
    """Return a palette-aware glyph for a graph input or output type."""

    normalized = str(data_type).strip().lower()
    if normalized in {"labels", "mask", "mask_or_labels"}:
        kind = "tag"
    elif normalized == "table":
        kind = "table"
    elif normalized in {"image", "array"}:
        kind = "image"
    else:
        kind = "nodes"
    colors = theme_colors(palette)
    foreground = colors.info.foreground
    disabled = _blend(foreground, colors.info.surface, 0.62)
    return _icon(kind, foreground, disabled, int(size))


def _blend(first: QColor, second: QColor, amount: float) -> QColor:
    amount = min(max(float(amount), 0.0), 1.0)
    return QColor(
        round(first.red() * (1.0 - amount) + second.red() * amount),
        round(first.green() * (1.0 - amount) + second.green() * amount),
        round(first.blue() * (1.0 - amount) + second.blue() * amount),
        round(first.alpha() * (1.0 - amount) + second.alpha() * amount),
    )


def _relative_luminance(color: QColor) -> float:
    channels = []
    for value in (color.redF(), color.greenF(), color.blueF()):
        channels.append(
            value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        )
    return 0.2126 * channels[0] + 0.7152 * channels[1] + 0.0722 * channels[2]


def _contrast_ratio(first: QColor, second: QColor) -> float:
    bright, dark = sorted(
        (_relative_luminance(first), _relative_luminance(second)),
        reverse=True,
    )
    return (bright + 0.05) / (dark + 0.05)


def _ensure_contrast(accent: QColor, background: QColor, text: QColor) -> QColor:
    if _contrast_ratio(accent, background) >= 3.0:
        return accent
    target = (
        QColor("#ffffff")
        if _relative_luminance(background) < 0.45
        else QColor("#111827")
    )
    candidate = QColor(accent)
    for step in range(1, 8):
        candidate = _blend(accent, target, step / 8.0)
        if _contrast_ratio(candidate, background) >= 3.0:
            return candidate
    return QColor(text)


def _icon(kind: str, normal: QColor, disabled: QColor, size: int) -> QIcon:
    # QColor is hashable inconsistently across Qt bindings, so cached calls are
    # normalized to RGBA integers here.
    return _cached_icon(
        str(kind),
        int(normal.rgba()),
        int(disabled.rgba()),
        max(int(size), 12),
    )


@lru_cache(maxsize=256)
def _cached_icon(
    kind: str,
    normal_rgba: int,
    disabled_rgba: int,
    size: int,
) -> QIcon:
    icon = QIcon()
    normal = QColor.fromRgba(normal_rgba)
    disabled = QColor.fromRgba(disabled_rgba)
    for render_size in sorted({size, size * 2}):
        icon.addPixmap(
            _draw_icon_pixmap(kind, normal, render_size),
            QIcon.Normal,
            QIcon.Off,
        )
        icon.addPixmap(
            _draw_icon_pixmap(kind, disabled, render_size),
            QIcon.Disabled,
            QIcon.Off,
        )
    return icon


def _draw_icon_pixmap(kind: str, color: QColor, size: int) -> QPixmap:
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)
    painter.scale(size / 20.0, size / 20.0)
    pen = QPen(color, 1.65)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    _draw_glyph(painter, str(kind), color)
    painter.end()
    return pixmap


def _draw_glyph(painter: QPainter, kind: str, color: QColor) -> None:
    path = QPainterPath()
    if kind in {"image", "image-source"}:
        painter.drawRoundedRect(QRectF(2.5, 3.5, 15, 13), 1.5, 1.5)
        painter.drawEllipse(QPointF(6.3, 7.2), 1.3, 1.3)
        path.moveTo(4.2, 14.2)
        path.lineTo(8.2, 10.2)
        path.lineTo(10.8, 12.4)
        path.lineTo(13.3, 9.5)
        path.lineTo(16.2, 13.5)
        painter.drawPath(path)
        if kind == "image-source":
            painter.drawLine(QPointF(14.5, 2.0), QPointF(18.5, 2.0))
            painter.drawLine(QPointF(16.5, 0.0), QPointF(16.5, 4.0))
        return
    if kind == "sun":
        painter.drawEllipse(QPointF(10, 10), 3.0, 3.0)
        for start, end in (
            ((10, 2), (10, 5)),
            ((10, 15), (10, 18)),
            ((2, 10), (5, 10)),
            ((15, 10), (18, 10)),
            ((4.2, 4.2), (6.2, 6.2)),
            ((13.8, 13.8), (15.8, 15.8)),
            ((15.8, 4.2), (13.8, 6.2)),
            ((6.2, 13.8), (4.2, 15.8)),
        ):
            painter.drawLine(QPointF(*start), QPointF(*end))
        return
    if kind == "waves":
        for offset in (5.0, 10.0, 15.0):
            wave = QPainterPath(QPointF(2.5, offset))
            wave.cubicTo(5.2, offset - 3.0, 7.6, offset + 3.0, 10.0, offset)
            wave.cubicTo(12.5, offset - 3.0, 15.0, offset + 3.0, 17.5, offset)
            painter.drawPath(wave)
        return
    if kind == "layers":
        for offset in (3.0, 6.5, 10.0):
            painter.drawPolyline(
                QPolygonF(
                    [
                        QPointF(3, offset + 2.0),
                        QPointF(10, offset - 1.0),
                        QPointF(17, offset + 2.0),
                        QPointF(10, offset + 5.0),
                        QPointF(3, offset + 2.0),
                    ]
                )
            )
        return
    if kind == "regions":
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        painter.drawEllipse(QRectF(2.5, 3.0, 6.0, 5.0))
        painter.drawEllipse(QRectF(11.5, 2.5, 5.5, 6.0))
        painter.drawEllipse(QRectF(6.5, 11.0, 7.0, 6.0))
        return
    if kind == "contours":
        painter.drawEllipse(QRectF(2.5, 3.0, 15.0, 14.0))
        painter.drawEllipse(QRectF(6.0, 6.2, 8.0, 7.6))
        painter.drawLine(QPointF(1.5, 10), QPointF(4.5, 10))
        painter.drawLine(QPointF(15.5, 10), QPointF(18.5, 10))
        return
    if kind == "tag":
        path.moveTo(3, 5)
        path.lineTo(10.5, 3)
        path.lineTo(17, 9.5)
        path.lineTo(10, 16.5)
        path.lineTo(3.5, 10)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawEllipse(QPointF(7.0, 7.0), 1.2, 1.2)
        return
    if kind == "ruler":
        path.moveTo(4, 15)
        path.lineTo(15, 4)
        path.lineTo(18, 7)
        path.lineTo(7, 18)
        path.closeSubpath()
        painter.drawPath(path)
        for value in (7.2, 10.0, 12.8):
            painter.drawLine(
                QPointF(value, 13.0 - (value - 7.2)),
                QPointF(value + 1.6, 14.6 - (value - 7.2)),
            )
        return
    if kind == "overlap":
        painter.drawEllipse(QRectF(2.5, 4.0, 10.0, 12.0))
        painter.drawEllipse(QRectF(7.5, 4.0, 10.0, 12.0))
        return
    if kind == "crop":
        for start, corner, end in (
            ((3, 7), (3, 3), (7, 3)),
            ((13, 3), (17, 3), (17, 7)),
            ((17, 13), (17, 17), (13, 17)),
            ((7, 17), (3, 17), (3, 13)),
        ):
            painter.drawLine(QPointF(*start), QPointF(*corner))
            painter.drawLine(QPointF(*corner), QPointF(*end))
        return
    if kind == "save":
        painter.drawLine(QPointF(10, 2.5), QPointF(10, 12.5))
        painter.drawLine(QPointF(6.5, 9.0), QPointF(10, 12.5))
        painter.drawLine(QPointF(13.5, 9.0), QPointF(10, 12.5))
        painter.drawLine(QPointF(3, 16.5), QPointF(17, 16.5))
        return
    if kind == "axes":
        painter.drawLine(QPointF(4, 16), QPointF(4, 4))
        painter.drawLine(QPointF(4, 16), QPointF(16, 16))
        painter.drawLine(QPointF(4, 16), QPointF(13, 7))
        return
    if kind == "channels":
        painter.drawEllipse(QRectF(2.5, 6.5, 7.0, 7.0))
        painter.drawEllipse(QRectF(7.0, 3.0, 7.0, 7.0))
        painter.drawEllipse(QRectF(10.5, 9.0, 7.0, 7.0))
        return
    if kind == "threshold":
        painter.drawLine(QPointF(3, 15.5), QPointF(17, 15.5))
        painter.drawRect(QRectF(4, 10, 2.5, 5.5))
        painter.drawRect(QRectF(8.7, 7, 2.5, 8.5))
        painter.drawRect(QRectF(13.4, 3.5, 2.5, 12))
        return
    if kind == "histogram":
        painter.drawLine(QPointF(3, 16), QPointF(17, 16))
        painter.drawLine(QPointF(3, 16), QPointF(3, 4))
        painter.drawRect(QRectF(5.0, 11.0, 2.3, 5.0))
        painter.drawRect(QRectF(8.8, 7.5, 2.3, 8.5))
        painter.drawRect(QRectF(12.6, 4.0, 2.3, 12.0))
        return
    if kind == "sliders":
        for y, knob_x in ((5.0, 7.0), (10.0, 13.0), (15.0, 9.5)):
            painter.drawLine(QPointF(3, y), QPointF(17, y))
            painter.setBrush(color)
            painter.drawEllipse(QPointF(knob_x, y), 1.55, 1.55)
            painter.setBrush(Qt.NoBrush)
        return
    if kind == "eye":
        path.moveTo(2.5, 10)
        path.cubicTo(6.0, 4.5, 14.0, 4.5, 17.5, 10)
        path.cubicTo(14.0, 15.5, 6.0, 15.5, 2.5, 10)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawEllipse(QPointF(10, 10), 2.3, 2.3)
        return
    if kind == "chip":
        painter.drawRoundedRect(QRectF(5, 5, 10, 10), 1.2, 1.2)
        painter.drawRoundedRect(QRectF(8, 8, 4, 4), 0.7, 0.7)
        for value in (7.0, 10.0, 13.0):
            painter.drawLine(QPointF(value, 2.5), QPointF(value, 5.0))
            painter.drawLine(QPointF(value, 15.0), QPointF(value, 17.5))
            painter.drawLine(QPointF(2.5, value), QPointF(5.0, value))
            painter.drawLine(QPointF(15.0, value), QPointF(17.5, value))
        return
    if kind == "database":
        painter.drawEllipse(QRectF(3, 3, 14, 5))
        painter.drawLine(QPointF(3, 5.5), QPointF(3, 14.5))
        painter.drawLine(QPointF(17, 5.5), QPointF(17, 14.5))
        path.moveTo(3, 9.5)
        path.cubicTo(6.5, 13.0, 13.5, 13.0, 17, 9.5)
        painter.drawPath(path)
        path = QPainterPath(QPointF(3, 14.0))
        path.cubicTo(6.5, 17.5, 13.5, 17.5, 17, 14.0)
        painter.drawPath(path)
        return
    if kind == "history":
        path.moveTo(5.3, 6.0)
        path.cubicTo(8.6, 2.5, 14.2, 3.2, 16.5, 7.2)
        path.cubicTo(19.0, 11.6, 16.0, 17.0, 11.0, 17.0)
        path.cubicTo(7.0, 17.0, 3.8, 14.2, 3.5, 10.5)
        painter.drawPath(path)
        painter.drawLine(QPointF(5.3, 6.0), QPointF(5.0, 2.8))
        painter.drawLine(QPointF(5.3, 6.0), QPointF(2.2, 5.8))
        painter.drawLine(QPointF(10.5, 6.3), QPointF(10.5, 10.5))
        painter.drawLine(QPointF(10.5, 10.5), QPointF(13.5, 12.0))
        return
    if kind == "table":
        painter.drawRect(QRectF(3, 4, 14, 12))
        painter.drawLine(QPointF(3, 8), QPointF(17, 8))
        painter.drawLine(QPointF(3, 12), QPointF(17, 12))
        painter.drawLine(QPointF(8, 4), QPointF(8, 16))
        painter.drawLine(QPointF(13, 4), QPointF(13, 16))
        return
    if kind == "math":
        painter.drawLine(QPointF(3, 7), QPointF(10, 7))
        painter.drawLine(QPointF(6.5, 3.5), QPointF(6.5, 10.5))
        painter.drawLine(QPointF(11.5, 13.5), QPointF(17.5, 13.5))
        return
    if kind == "search":
        painter.drawEllipse(QRectF(3.0, 3.0, 10.5, 10.5))
        painter.drawLine(QPointF(12.0, 12.0), QPointF(17.0, 17.0))
        return
    if kind == "chevron-down":
        path.moveTo(5.0, 7.5)
        path.lineTo(10.0, 12.5)
        path.lineTo(15.0, 7.5)
        painter.drawPath(path)
        return
    if kind == "chevron-right":
        path.moveTo(7.5, 5.0)
        path.lineTo(12.5, 10.0)
        path.lineTo(7.5, 15.0)
        painter.drawPath(path)
        return
    if kind in {
        "panel-left-left",
        "panel-left-right",
        "panel-right-left",
        "panel-right-right",
    }:
        _prefix, panel_side, arrow_direction = kind.split("-")
        painter.drawRoundedRect(QRectF(2.5, 3.0, 15.0, 14.0), 1.0, 1.0)
        divider_x = 7.0 if panel_side == "left" else 13.0
        painter.drawLine(QPointF(divider_x, 3.0), QPointF(divider_x, 17.0))
        center_x = 12.0 if panel_side == "left" else 8.0
        tip_x = center_x - 1.8 if arrow_direction == "left" else center_x + 1.8
        tail_x = center_x + 1.5 if arrow_direction == "left" else center_x - 1.5
        painter.drawLine(QPointF(tail_x, 7.0), QPointF(tip_x, 10.0))
        painter.drawLine(QPointF(tip_x, 10.0), QPointF(tail_x, 13.0))
        return
    if kind == "close":
        painter.drawLine(QPointF(5, 5), QPointF(15, 15))
        painter.drawLine(QPointF(15, 5), QPointF(5, 15))
        return
    painter.drawEllipse(QRectF(3, 3, 5, 5))
    painter.drawEllipse(QRectF(12, 3, 5, 5))
    painter.drawEllipse(QRectF(7.5, 12, 5, 5))
    painter.drawLine(QPointF(7, 7), QPointF(9, 12))
    painter.drawLine(QPointF(13, 7), QPointF(11, 12))


__all__ = [
    "category_icon",
    "category_icon_kind",
    "data_type_icon",
    "interface_icon",
    "operation_icon",
    "operation_icon_kind",
    "palette_branch_color",
    "palette_category_colors",
]
