"""Prepare sharp splash artwork and licensed scalable text for early startup."""

import argparse
import json
from importlib.util import find_spec
from pathlib import Path

from qtpy.QtCore import QRectF
from qtpy.QtGui import (
    QColor,
    QFont,
    QFontDatabase,
    QFontMetricsF,
    QGuiApplication,
    QImage,
    QPainter,
    QPainterPath,
    QTransform,
)
from qtpy.QtSvg import QSvgRenderer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "src/napari_vipp/assets/branding"


def glyph_commands(path: QPainterPath) -> list[list]:
    commands = []
    index = 0
    while index < path.elementCount():
        element = path.elementAt(index)
        if element.isMoveTo():
            commands.append(["M", round(element.x, 4), round(element.y, 4)])
        elif element.isLineTo():
            commands.append(["L", round(element.x, 4), round(element.y, 4)])
        elif element.isCurveTo():
            points = [path.elementAt(index + offset) for offset in range(3)]
            commands.append(
                [
                    "C",
                    *[
                        round(value, 4)
                        for point in points
                        for value in (point.x, point.y)
                    ],
                ]
            )
            index += 2
        else:
            raise ValueError("Unexpected font outline element")
        index += 1
    return commands


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--font-directory", type=Path)
    parser.add_argument("--inter-font", type=Path, required=True)
    parser.add_argument("--inter-license", type=Path, required=True)
    args = parser.parse_args()
    font_directory = args.font_directory
    if font_directory is None:
        spec = find_spec("matplotlib")
        if spec is None or spec.origin is None:
            parser.error("Supply --font-directory containing the DejaVu Sans fonts")
        font_directory = Path(spec.origin).parent / "mpl-data/fonts/ttf"

    app = QGuiApplication.instance() or QGuiApplication([])
    image = QImage(2560, 1560, QImage.Format_ARGB32_Premultiplied)
    image.setDevicePixelRatio(4)
    image.fill(QColor("#08151F"))
    painter = QPainter(image)
    painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
    painter.setPen(QColor("#334155"))
    painter.drawRect(QRectF(0.5, 0.5, 639, 389))
    logo = QSvgRenderer(str(ASSETS / "vipp-logo-dark.svg"))
    logo.render(painter, QRectF(70, 22, 500, 146.34))
    painter.setPen(QColor("#334155"))
    painter.setBrush(QColor("#172432"))
    painter.drawRoundedRect(QRectF(42, 265, 556, 12), 6, 6)
    painter.setPen(QColor("#A8B7C8"))
    painter.drawLine(609, 24, 623, 24)
    painter.end()
    if not image.save(str(ASSETS / "vipp-startup-first-frame.png")):
        raise RuntimeError("Could not save startup first frame")
    logo_image = image.copy(280, 88, 2000, 586)
    if not logo_image.save(str(ASSETS / "vipp-startup-logo.png")):
        raise RuntimeError("Could not save high-resolution splash logo")

    characters = "".join(chr(code) for code in range(32, 127)) + "–—…µΩ"
    for (
        style,
        font_path,
        weight,
        design_pixels,
        asset_name,
        family_name,
        license_name,
    ) in (
        (
            "regular",
            font_directory / "DejaVuSans.ttf",
            QFont.Normal,
            100,
            "vipp-startup-lettering.json",
            "DejaVu Sans",
            "LICENSE-DejaVu.txt",
        ),
        (
            "status",
            args.inter_font,
            QFont.Medium,
            14,
            "vipp-startup-status-lettering.json",
            "Inter Medium",
            "LICENSE-Inter.txt",
        ),
    ):
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        if not families:
            raise RuntimeError(f"Could not load {font_path}")
        font = QFont(families[0])
        font.setWeight(weight)
        # Shape the variable Inter face at its actual 14 px optical size, then
        # normalize outlines to the same 100-unit coordinate system as the
        # other lettering. It can still be painted at any display resolution.
        font.setPixelSize(design_pixels)
        factor = 100 / design_pixels
        transform = QTransform.fromScale(factor, factor)
        metrics = QFontMetricsF(font)
        glyphs = {}
        for character in characters:
            path = QPainterPath()
            path.addText(0, 0, font, character)
            glyphs[character] = {
                "advance": round(metrics.horizontalAdvance(character) * factor, 4),
                "path": glyph_commands(transform.map(path)),
            }
        payload = {
            "name": "VIPP Splash Lettering",
            "source": family_name,
            "license": license_name,
            "styles": {
                style: {
                    "ascent": round(metrics.ascent() * factor, 4),
                    "descent": round(metrics.descent() * factor, 4),
                    "glyphs": glyphs,
                }
            },
        }
        (ASSETS / asset_name).write_text(
            json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )
    (ASSETS / "LICENSE-DejaVu.txt").write_text(
        (font_directory / "LICENSE_DEJAVU").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (ASSETS / "LICENSE-Inter.txt").write_text(
        args.inter_license.read_text(encoding="utf-8"), encoding="utf-8"
    )
    del app


if __name__ == "__main__":
    main()
