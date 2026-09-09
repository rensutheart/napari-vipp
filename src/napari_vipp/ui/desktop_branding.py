"""Window branding for installer-launched VIPP, never a napari plugin hook."""

from __future__ import annotations

import logging
import sys
from importlib import resources

from qtpy.QtCore import Qt
from qtpy.QtGui import QIcon, QPixmap

DESKTOP_APP_ID = "io.github.rensutheart.VIPP"


def application_icon() -> QIcon:
    """Load the multi-resolution desktop mark from the installed package."""
    icon = QIcon()
    asset = resources.files("napari_vipp").joinpath(
        "assets", "branding", "vipp-mark.svg"
    )
    pixmap = QPixmap()
    if pixmap.loadFromData(asset.read_bytes()):
        for size in (16, 24, 32, 48, 64, 128, 256):
            icon.addPixmap(
                pixmap.scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            )
    return icon


def set_desktop_process_identity() -> None:
    """Give Windows its own taskbar group without changing Python or napari."""
    if sys.platform != "win32":
        return
    import ctypes

    try:
        set_identity = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_identity.argtypes = [ctypes.c_wchar_p]
        set_identity.restype = ctypes.c_long
        if set_identity(DESKTOP_APP_ID) < 0:
            raise OSError("Windows could not set the VIPP application identity.")
    except (AttributeError, OSError):
        # Branding must never prevent the scientific application from starting.
        logging.getLogger(__name__).warning(
            "Could not set the VIPP taskbar identity", exc_info=True
        )


def apply_desktop_branding(application, window) -> None:
    """Set public Qt icons after napari has initialized its window defaults."""
    icon = application_icon()
    application.setWindowIcon(icon)
    window.setWindowIcon(icon)
