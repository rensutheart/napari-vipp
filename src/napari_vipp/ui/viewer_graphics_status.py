"""Read the OpenGL renderer of this viewer, not a requested launch mode."""

from __future__ import annotations

import ctypes
import sys
import warnings
from dataclasses import dataclass

from qtpy.QtCore import QThread
from qtpy.QtGui import QGuiApplication, QOpenGLContext

_GL_VENDOR = 0x1F00
_GL_RENDERER = 0x1F01
_GL_VERSION = 0x1F02


@dataclass(frozen=True)
class ViewerGraphicsStatus:
    renderer: str
    vendor: str = ""
    version: str = ""

    @property
    def _renderer_type(self) -> str:
        description = f"{self.renderer} {self.vendor}".casefold()
        if any(
            marker in description
            for marker in (
                "llvmpipe",
                "softpipe",
                "swrast",
                "swiftshader",
                "software",
                "gdi generic",
                "basic render driver",
                "software rasterizer",
            )
        ):
            return "Software OpenGL"
        if any(
            marker in description
            for marker in (
                "nvidia",
                "radeon",
                "intel",
                "ati technologies",
                "adreno",
                "apple m",
                "mali-",
                "powervr",
            )
        ):
            return "Hardware OpenGL"
        return ""

    @property
    def summary(self) -> str:
        kind = self._renderer_type
        return f"{kind} — {self.renderer}" if kind else self.renderer

    @property
    def display_name(self) -> str:
        """A menu label, with the unabridged driver string in its tooltip."""
        if self._renderer_type == "Software OpenGL":
            return "CPU (software)"
        return self.renderer.partition("/PCIe/")[0].strip() or self.renderer


def _owned_canvas_widget(viewer):
    window = getattr(viewer, "window", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        qt_viewer = getattr(window, "qt_viewer", None)
    if qt_viewer is None:
        qt_viewer = getattr(window, "_qt_viewer", None)
    if qt_viewer is None or getattr(qt_viewer, "viewer", None) is not viewer:
        return None
    return getattr(getattr(qt_viewer, "canvas", None), "native", None)


def _read_gl_strings(context) -> ViewerGraphicsStatus | None:
    # Resolve against this Qt context, including Qt's Windows software DLL.
    # A process-wide PyOpenGL entry point may point to a different GL library.
    address = context.getProcAddress(b"glGetString")
    if not address or int(address) in {0, 1, 2, 3, -1, ctypes.c_void_p(-1).value}:
        return None
    factory = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE
    read_string = factory(ctypes.c_char_p, ctypes.c_uint)(int(address))

    def text(key):
        raw = (read_string(key) or b"")[:384].decode("utf-8", errors="replace")
        return "".join(char if char.isprintable() else " " for char in raw).strip()

    renderer = text(_GL_RENDERER)
    if not renderer:
        return None
    return ViewerGraphicsStatus(renderer, text(_GL_VENDOR), text(_GL_VERSION))


def query_viewer_graphics(viewer) -> ViewerGraphicsStatus | None:
    """Read only GL strings from this canvas; never create a context/window."""
    app = QGuiApplication.instance()
    if not isinstance(app, QGuiApplication) or QThread.currentThread() != app.thread():
        return None
    try:
        canvas = _owned_canvas_widget(viewer)
        if canvas is None or not canvas.isValid():
            return None
        target = canvas.context()
        if not isinstance(target, QOpenGLContext) or not target.isValid():
            return None
        previous = QOpenGLContext.currentContext()
        previous_surface = previous.surface() if previous is not None else None
        if previous is not None and previous is not target and previous_surface is None:
            return None  # Do not displace a context we cannot restore.
    except (AttributeError, RuntimeError, TypeError):
        return None

    if previous is target:
        try:
            return _read_gl_strings(target)
        except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
            return None

    result = None
    restored = False
    try:
        canvas.makeCurrent()
        if QOpenGLContext.currentContext() is target:
            result = _read_gl_strings(target)
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError):
        pass
    finally:
        try:
            current = QOpenGLContext.currentContext()
            if previous is None:
                if current is target:
                    canvas.doneCurrent()
                restored = QOpenGLContext.currentContext() is None
            elif current is previous:
                restored = True
            else:
                restored = (
                    previous.makeCurrent(previous_surface)
                    and QOpenGLContext.currentContext() is previous
                )
                if not restored and QOpenGLContext.currentContext() is target:
                    canvas.doneCurrent()
        except (AttributeError, RuntimeError, TypeError):
            pass
    return result if restored else None
