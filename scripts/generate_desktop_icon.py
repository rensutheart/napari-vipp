"""Regenerate the Windows shortcut icon from VIPP's reviewed SVG mark."""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image
from qtpy.QtCore import QBuffer, QIODevice
from qtpy.QtGui import QImage, QPainter
from qtpy.QtSvg import QSvgRenderer

ICON_SIZES = (16, 24, 32, 48, 64, 128, 256)


def render_icon(source: Path, output: Path) -> None:
    """Render each Windows size directly from vector artwork, with alpha."""
    renderer = QSvgRenderer(str(source))
    if not renderer.isValid():
        raise ValueError(f"Invalid VIPP mark: {source}")
    images = []
    for size in ICON_SIZES:
        image = QImage(size, size, QImage.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        renderer.render(painter)
        painter.end()
        buffer = QBuffer()
        buffer.open(QIODevice.WriteOnly)
        if not image.save(buffer, "PNG"):
            raise RuntimeError("Qt could not render the VIPP icon.")
        images.append(Image.open(io.BytesIO(bytes(buffer.data()))).convert("RGBA"))
    images[-1].save(
        output,
        format="ICO",
        sizes=[(size, size) for size in ICON_SIZES],
        append_images=images[:-1],
    )


if __name__ == "__main__":
    branding = Path(__file__).resolve().parents[1] / "src/napari_vipp/assets/branding"
    render_icon(branding / "vipp-mark.svg", branding / "vipp-mark.ico")
