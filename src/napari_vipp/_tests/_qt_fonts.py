"""Provide native glyph metrics when Windows' offscreen Qt cannot find fonts."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from qtpy.QtGui import QFontDatabase


def load_offscreen_windows_fonts(qapp) -> tuple[int, ...]:
    """Provide deterministic native UI metrics on Windows' offscreen platform.

    Another test can register a fixed font before this helper runs, so a
    nonempty database does not necessarily contain a proportional UI font.
    An unresolved generic application family can also select that fixed font
    after registration. Bind the fallback we load instead of depending on
    first-resolution order. Native platforms and existing UI families stay
    unchanged.
    """
    if sys.platform != "win32" or qapp.platformName().lower() != "offscreen":
        return ()
    families = {family.casefold() for family in QFontDatabase.families()}
    if "segoe ui" in families:
        return ()

    fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    loaded = []
    for filename in ("segoeui.ttf", "segoeuib.ttf", "segoeuii.ttf", "seguisb.ttf"):
        path = fonts / filename
        if path.is_file():
            font_id = QFontDatabase.addApplicationFont(str(path))
            if font_id >= 0:
                loaded.append(font_id)
    families = {family.casefold() for family in QFontDatabase.families()}
    if "segoe ui" not in families:
        raise RuntimeError(
            "Windows offscreen Qt found no Segoe UI font, and could not load installed "
            f"Segoe UI fonts from {fonts}. UI geometry tests require real glyph "
            "metrics; configure a working Qt font directory or native platform."
        )
    font = qapp.font()
    if font.family().casefold() not in families:
        font.setFamily("Segoe UI")
        qapp.setFont(font)
    return tuple(loaded)
