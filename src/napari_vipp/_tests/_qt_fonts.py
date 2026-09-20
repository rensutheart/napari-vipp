"""Provide native glyph metrics when Windows' offscreen Qt cannot find fonts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from qtpy.QtGui import QFontDatabase


def load_offscreen_windows_fonts(qapp) -> tuple[int, ...]:
    """Repair an empty test font database, without changing healthy hosts."""
    if (
        sys.platform != "win32"
        or qapp.platformName().lower() != "offscreen"
        or QFontDatabase.families()
    ):
        return ()

    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    loaded = []
    for filename in ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "seguisb.ttf"):
        path = fonts / filename
        if path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id >= 0:
                loaded.append(font_id)
    if not QFontDatabase.families():
        raise RuntimeError(
            "Windows offscreen Qt found no fonts, and could not load installed "
            f"Segoe UI fonts from {fonts}. UI geometry tests require real glyph "
            "metrics; configure a working Qt font directory or native platform."
        )
    return tuple(loaded)
