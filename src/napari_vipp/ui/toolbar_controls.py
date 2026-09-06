"""Shared outline icons and command-button painting for VIPP toolbars."""

from __future__ import annotations

import math

from qtpy.QtCore import QPointF, QRect, Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPainterPath, QPalette, QPen, QPixmap
from qtpy.QtWidgets import (
    QApplication,
    QPushButton,
    QStyle,
    QStyleOptionButton,
    QStylePainter,
)


def toolbar_icon(kind: str, palette: QPalette | None = None) -> QIcon:
    if palette is None:
        application = QApplication.instance()
        palette = application.palette() if application is not None else QPalette()
    icon = QIcon()
    icon.addPixmap(
        _toolbar_icon_pixmap(kind, palette.color(QPalette.ButtonText).name()),
        QIcon.Normal,
        QIcon.Off,
    )
    icon.addPixmap(
        _toolbar_icon_pixmap(
            kind,
            palette.color(QPalette.Disabled, QPalette.ButtonText).name(),
        ),
        QIcon.Disabled,
        QIcon.Off,
    )
    return icon


def _toolbar_icon_pixmap(kind: str, foreground: str) -> QPixmap:
    size = 24
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing, True)

    pen = QPen(QColor(foreground), 2.2)
    pen.setCapStyle(Qt.RoundCap)
    pen.setJoinStyle(Qt.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.NoBrush)
    path = QPainterPath()
    arrow = QPainterPath()
    if kind == "redo":
        path.moveTo(16, 8)
        path.cubicTo(13, 5.5, 5, 6.5, 5, 13)
        path.cubicTo(5, 18, 9.5, 20, 16, 20)
        arrow.moveTo(20.5, 8)
        arrow.lineTo(15, 4)
        arrow.lineTo(15, 12)
        arrow.closeSubpath()
    elif kind in {"reset", "refresh"}:
        path.moveTo(18.5, 8)
        path.cubicTo(15.8, 4.5, 9.8, 3.8, 6.5, 8)
        path.cubicTo(3.2, 12.2, 5.6, 19.5, 12.5, 19.5)
        path.cubicTo(16, 19.5, 18.8, 17.4, 19.8, 14.4)
        arrow.moveTo(20.4, 7.3)
        arrow.lineTo(15, 7.1)
        arrow.lineTo(18.4, 11.7)
        arrow.closeSubpath()
    elif kind == "undo":
        path.moveTo(8, 8)
        path.cubicTo(11, 5.5, 19, 6.5, 19, 13)
        path.cubicTo(19, 18, 14.5, 20, 8, 20)
        arrow.moveTo(3.5, 8)
        arrow.lineTo(9, 4)
        arrow.lineTo(9, 12)
        arrow.closeSubpath()
    elif kind == "new":
        path.moveTo(6, 3)
        path.lineTo(14, 3)
        path.lineTo(19, 8)
        path.lineTo(19, 21)
        path.lineTo(6, 21)
        path.closeSubpath()
        path.moveTo(14, 3)
        path.lineTo(14, 8)
        path.lineTo(19, 8)
        painter.drawPath(path)
        painter.drawLine(9, 14, 16, 14)
        painter.drawLine(13, 11, 13, 18)
    elif kind == "open":
        path.moveTo(3, 7)
        path.lineTo(10, 7)
        path.lineTo(12, 9)
        path.lineTo(21, 9)
        path.lineTo(18, 20)
        path.lineTo(4, 20)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(4, 7, 4, 5)
        painter.drawLine(4, 5, 11, 5)
        painter.drawLine(11, 5, 13, 7)
    elif kind == "save":
        painter.drawRect(4, 3, 16, 18)
        painter.drawRect(8, 4, 8, 6)
        painter.drawRect(8, 14, 8, 7)
    elif kind in {"image", "images"}:
        if kind == "images":
            painter.drawLine(3, 7, 3, 3)
            painter.drawLine(3, 3, 17, 3)
        painter.drawRect(
            6 if kind == "images" else 3,
            6 if kind == "images" else 4,
            15 if kind == "images" else 18,
            14 if kind == "images" else 16,
        )
        painter.drawEllipse(9, 8, 3, 3)
        path.moveTo(7, 18)
        path.lineTo(12, 13)
        path.lineTo(15, 16)
        path.lineTo(18, 12)
        path.lineTo(21, 15)
        painter.drawPath(path)
    elif kind == "destination":
        path.moveTo(3, 9)
        path.lineTo(3, 5)
        path.lineTo(10, 5)
        path.lineTo(12, 8)
        path.lineTo(21, 8)
        path.lineTo(21, 20)
        path.lineTo(3, 20)
        path.lineTo(3, 17)
        path.moveTo(2, 13)
        path.lineTo(13, 13)
        path.moveTo(10, 10)
        path.lineTo(13, 13)
        path.lineTo(10, 16)
        painter.drawPath(path)
    elif kind == "workflow":
        painter.drawRoundedRect(3, 3, 7, 6, 1, 1)
        painter.drawRoundedRect(14, 15, 7, 6, 1, 1)
        path.moveTo(6, 9)
        path.lineTo(6, 18)
        path.lineTo(14, 18)
        painter.drawPath(path)
    elif kind == "archive":
        path.moveTo(3, 7)
        path.lineTo(12, 3)
        path.lineTo(21, 7)
        path.lineTo(21, 17)
        path.lineTo(12, 21)
        path.lineTo(3, 17)
        path.closeSubpath()
        path.moveTo(3, 7)
        path.lineTo(12, 11)
        path.lineTo(21, 7)
        path.moveTo(12, 11)
        path.lineTo(12, 21)
        path.moveTo(7, 5)
        path.lineTo(16, 9)
        path.lineTo(16, 13)
        painter.drawPath(path)
    elif kind == "batch":
        painter.drawRect(3, 4, 18, 16)
        painter.drawLine(3, 10, 21, 10)
        painter.drawLine(3, 15, 21, 15)
        painter.drawLine(9, 4, 9, 20)
        painter.drawLine(15, 4, 15, 20)
    elif kind == "columns":
        for x in (3, 10, 17):
            painter.drawRoundedRect(x, 4, 4, 16, 1, 1)
    elif kind == "edit":
        path.moveTo(4, 16)
        path.lineTo(15, 5)
        path.lineTo(19, 9)
        path.lineTo(8, 20)
        path.lineTo(3, 21)
        path.closeSubpath()
        path.moveTo(13, 7)
        path.lineTo(17, 11)
        painter.drawPath(path)
    elif kind in {"select_all", "deselect"}:
        painter.drawRoundedRect(4, 4, 16, 16, 2, 2)
        if kind == "select_all":
            path.moveTo(8, 12)
            path.lineTo(11, 15)
            path.lineTo(16, 9)
            painter.drawPath(path)
        else:
            painter.drawLine(8, 12, 16, 12)
    elif kind == "setup":
        for y, knob_x in ((6, 8), (12, 16), (18, 10)):
            painter.drawLine(3, y, knob_x - 3, y)
            painter.drawLine(knob_x + 3, y, 21, y)
            painter.drawEllipse(QPointF(knob_x, y), 2.5, 2.5)
    elif kind == "checklist":
        for y in (5, 12, 19):
            path.moveTo(3, y)
            path.lineTo(5, y + 2)
            path.lineTo(8, y - 2)
            painter.drawLine(12, y, 21, y)
        painter.drawPath(path)
    elif kind == "recheck_all":
        painter.drawLine(3, 4, 20, 4)
        painter.drawLine(3, 9, 7, 9)
        painter.drawLine(3, 14, 5, 14)
        painter.drawLine(3, 19, 5, 19)
        path.moveTo(20, 12)
        path.cubicTo(17, 8, 10, 10, 10, 15.5)
        path.cubicTo(10, 21, 18, 23, 21, 17.5)
        path.moveTo(20, 8.5)
        path.lineTo(20, 12.5)
        path.lineTo(16, 12.5)
        painter.drawPath(path)
    elif kind in {"next", "previous"}:
        if kind == "previous":
            painter.translate(24, 0)
            painter.scale(-1, 1)
        painter.drawLine(3, 12, 20, 12)
        path.moveTo(13, 5)
        path.lineTo(20, 12)
        path.lineTo(13, 19)
        painter.drawPath(path)
    elif kind == "compute":
        painter.drawRect(6, 6, 12, 12)
        painter.drawRect(9, 9, 6, 6)
        for pin in (9, 15):
            painter.drawLine(pin, 3, pin, 6)
            painter.drawLine(pin, 18, pin, 21)
            painter.drawLine(3, pin, 6, pin)
            painter.drawLine(18, pin, 21, pin)
    elif kind == "search":
        painter.drawEllipse(3, 3, 13, 13)
        painter.drawLine(14, 14, 21, 21)
    elif kind == "preview":
        path.moveTo(2.5, 12)
        path.cubicTo(6, 6.5, 9, 5, 12, 5)
        path.cubicTo(15, 5, 18, 6.5, 21.5, 12)
        path.cubicTo(18, 17.5, 15, 19, 12, 19)
        path.cubicTo(9, 19, 6, 17.5, 2.5, 12)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawEllipse(9, 9, 6, 6)
    elif kind == "calculate":
        arrow.moveTo(7, 4)
        arrow.lineTo(20, 12)
        arrow.lineTo(7, 20)
        arrow.closeSubpath()
    elif kind == "optimize":
        # A compact speedometer remains recognizable when the responsive
        # toolbar removes the Find-fastest label.
        painter.drawArc(QRect(4, 4, 16, 16), 0, 180 * 16)
        painter.drawLine(12, 13, 17, 9)
        painter.drawEllipse(10, 11, 4, 4)
    elif kind == "settings":
        # Use one connected eight-tooth outline rather than radial spokes.  At
        # toolbar size, detached spokes read as a star or brightness control.
        gear = QPainterPath()
        gear_points: list[QPointF] = []
        tooth_step = math.tau / 8.0
        for tooth in range(8):
            center_angle = tooth * tooth_step - math.pi / 2.0
            for offset, radius in (
                (-0.50, 7.8),
                (-0.34, 7.8),
                (-0.27, 10.0),
                (0.27, 10.0),
                (0.34, 7.8),
                (0.50, 7.8),
            ):
                angle = center_angle + offset * tooth_step
                gear_points.append(
                    QPointF(
                        12.0 + math.cos(angle) * radius,
                        12.0 + math.sin(angle) * radius,
                    )
                )
        gear.moveTo(gear_points[0])
        for point in gear_points[1:]:
            gear.lineTo(point)
        gear.closeSubpath()
        painter.drawPath(gear)
        painter.drawEllipse(QPointF(12.0, 12.0), 3.0, 3.0)
    elif kind == "focus":
        for points in (
            ((4, 9), (4, 4), (9, 4)),
            ((15, 4), (20, 4), (20, 9)),
            ((20, 15), (20, 20), (15, 20)),
            ((9, 20), (4, 20), (4, 15)),
        ):
            path.moveTo(*points[0])
            path.lineTo(*points[1])
            path.lineTo(*points[2])
        painter.drawPath(path)
        painter.drawEllipse(9, 9, 6, 6)
    elif kind == "arrange":
        painter.drawLine(7, 7, 17, 12)
        painter.drawLine(7, 17, 17, 12)
        painter.drawEllipse(3, 3, 7, 7)
        painter.drawEllipse(3, 14, 7, 7)
        painter.drawEllipse(14, 9, 7, 7)
    elif kind == "tunnels":
        painter.drawEllipse(3, 5, 6, 6)
        painter.drawEllipse(15, 13, 6, 6)
        painter.drawLine(9, 8, 14, 8)
        painter.drawLine(14, 8, 14, 16)
        painter.drawLine(14, 16, 15, 16)
        painter.drawLine(10, 16, 14, 16)
    elif kind == "activity":
        path.moveTo(2, 13)
        path.lineTo(7, 13)
        path.lineTo(10, 5)
        path.lineTo(14, 19)
        path.lineTo(17, 10)
        path.lineTo(22, 10)
        painter.drawPath(path)
    elif kind == "more":
        for x in (5.0, 12.0, 19.0):
            painter.drawEllipse(QPointF(x, 12.0), 1.7, 1.7)
    elif kind == "stop":
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(foreground))
        painter.drawRect(6, 6, 12, 12)
    else:
        painter.drawEllipse(5, 5, 14, 14)

    if not path.isEmpty() and kind in {"undo", "redo", "reset", "refresh"}:
        painter.drawPath(path)
    if not arrow.isEmpty():
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(foreground))
        painter.drawPath(arrow)
    painter.end()
    return pixmap


class ToolbarCommandButton(QPushButton):
    """Toolbar push button with a little extra icon-to-label breathing room."""

    _ICON_TEXT_SPACER = "\u2009"

    def _toolbar_style_option(self) -> QStyleOptionButton:
        """Return the native button option with VIPP's icon/text spacing."""
        option = QStyleOptionButton()
        self.initStyleOption(option)
        if option.text and not option.icon.isNull():
            option.text = f"{self._ICON_TEXT_SPACER}{option.text}"
        return option

    def paintEvent(self, event) -> None:  # noqa: N802
        del event
        painter = QStylePainter(self)
        painter.drawControl(QStyle.CE_PushButton, self._toolbar_style_option())
