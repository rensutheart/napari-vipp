"""Scalable prepared lettering for the splash, without runtime font discovery."""

from __future__ import annotations

import json
from importlib import resources

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QPainter, QPainterPath, QPalette
from qtpy.QtWidgets import QLabel, QWidget


class PreparedTypeface:
    """Load packaged, licensed glyph outlines; never query the system fonts."""

    def __init__(self, payload: dict) -> None:
        self.styles = {}
        for name, source in payload["styles"].items():
            glyphs = {}
            for character, glyph in source["glyphs"].items():
                path = QPainterPath()
                path.setFillRule(Qt.WindingFill)
                for command in glyph["path"]:
                    kind, *points = command
                    if kind == "M":
                        path.moveTo(*points)
                    elif kind == "L":
                        path.lineTo(*points)
                    elif kind == "C":
                        path.cubicTo(*points)
                    else:
                        raise ValueError("Unsupported prepared glyph command")
                glyphs[character] = (float(glyph["advance"]), path)
            self.styles[name] = (
                float(source["ascent"]),
                float(source["descent"]),
                glyphs,
            )
        if not {"regular", "status"}.issubset(self.styles):
            raise ValueError("Prepared splash lettering is incomplete")

    @classmethod
    def load(cls) -> PreparedTypeface | None:
        try:
            assets = resources.files("napari_vipp").joinpath("assets", "branding")
            payload = {"styles": {}}
            for name in (
                "vipp-startup-lettering.json",
                "vipp-startup-status-lettering.json",
            ):
                source = json.loads(assets.joinpath(name).read_text(encoding="utf-8"))
                payload["styles"].update(source["styles"])
            return cls(payload)
        except (OSError, TypeError, ValueError, KeyError):
            return None

    def paint(self, painter, rect, text, *, pixels, style, color, alignment):
        """Draw the same outlines for the early frame and ordinary status label."""
        ascent, descent, glyphs = self.styles[style]
        scale = pixels / 100

        def glyph(character):
            return glyphs.get(character, glyphs["?"])

        def width(value):
            return sum(glyph(character)[0] for character in value) * scale

        lines = []
        for paragraph in text.split("\n"):
            line = ""
            for word in paragraph.split(" "):
                candidate = f"{line} {word}" if line else word
                if line and width(candidate) > rect.width():
                    lines.append(line)
                    line = word
                else:
                    line = candidate
            lines.append(line)
        line_height = (ascent + descent) * scale
        baseline = rect.top() + (rect.height() - line_height * len(lines)) / 2
        baseline += ascent * scale
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(color)
        for line in lines:
            if alignment & Qt.AlignRight:
                cursor = rect.right() + 1 - width(line)
            elif alignment & Qt.AlignHCenter:
                cursor = rect.left() + (rect.width() - width(line)) / 2
            else:
                cursor = rect.left()
            painter.save()
            painter.translate(cursor, baseline)
            painter.scale(scale, scale)
            for character in line:
                advance, path = glyph(character)
                painter.drawPath(path)
                painter.translate(advance, 0)
            painter.restore()
            baseline += line_height


class PreparedText(QWidget):
    """Paint ordinary text as scalable outlines, including live version/status."""

    def __init__(
        self,
        face: PreparedTypeface,
        parent: QWidget,
        *,
        pixels: int,
        color: str = "#94A3B8",
        alignment=Qt.AlignCenter,
        style: str = "regular",
    ) -> None:
        super().__init__(parent)
        self._face = face
        self._pixels = pixels
        self._color = QColor(color)
        self._alignment = alignment
        self._style = style
        self._text = ""
        self.setAttribute(Qt.WA_TransparentForMouseEvents)

    def setText(self, text: str) -> None:  # noqa: N802
        self._text = text
        self.setAccessibleDescription(text)
        self.update()

    def text(self) -> str:
        return self._text

    def set_color(self, color: str) -> None:
        self._color = QColor(color)
        self.update()

    def display(self, elapsed: str) -> None:
        self.setText(f"Elapsed {elapsed}")

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        self._face.paint(
            painter,
            self.rect(),
            self._text,
            pixels=self._pixels,
            style=self._style,
            color=self._color,
            alignment=self._alignment,
        )
        painter.end()


class PreparedStatusLabel(QLabel):
    """Keep normal label semantics and phase colours with portable Inter outlines."""

    def __init__(self, face: PreparedTypeface, text: str) -> None:
        super().__init__(text)
        self._face = face

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        self._face.paint(
            painter,
            self.rect(),
            self.text(),
            pixels=14,
            style="status",
            color=self.palette().color(QPalette.WindowText),
            alignment=self.alignment(),
        )
        painter.end()
