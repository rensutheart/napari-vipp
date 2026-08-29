"""Feature-detected access to napari presentation objects.

Keep the small amount of Qt integration needed outside VIPP's main widget in
one place.  These helpers deliberately detect available objects instead of
branching on napari or Qt-binding version numbers.
"""

from __future__ import annotations

import warnings
from typing import Any

from qtpy.QtWidgets import QWidget


def viewer_camera(viewer: Any) -> Any:
    """Return the active napari camera across old and new viewer layouts.

    Napari 0.9 exposes the camera through ``viewer.scene.camera``.  Older
    supported releases expose it directly on ``viewer.camera``.  ``None`` is
    treated as unavailable so partially constructed test or viewer objects can
    still use the legacy location.
    """

    scene = getattr(viewer, "scene", None)
    camera = getattr(scene, "camera", None) if scene is not None else None
    if camera is not None:
        return camera

    camera = getattr(viewer, "camera", None)
    if camera is not None:
        return camera

    raise RuntimeError("The napari viewer does not expose a camera.")


def _top_level_qt_widget(widget: object) -> QWidget | None:
    """Resolve a QWidget's owner using binding-neutral Qt parent traversal."""

    if not isinstance(widget, QWidget):
        return None

    current = widget
    visited: set[int] = set()
    while True:
        identity = id(current)
        if identity in visited:
            return None
        visited.add(identity)

        parent = current.parentWidget()
        if parent is None:
            return current
        current = parent


def viewer_qt_window(viewer: Any, *, anchor: QWidget | None = None) -> QWidget:
    """Return the owning Qt window without depending on a particular binding.

    An existing dock or child widget is the strongest seam because ordinary
    Qt parent traversal is stable across PyQt6 and PySide6.  When no anchor is
    available, use napari's exposed ``Window.qt_viewer`` object and traverse
    from there.  Some old napari releases expose neither route, so the exact
    historical ``Window._qt_window`` attribute remains a final, bounded
    fallback.  No other private napari object graph is inspected.
    """

    resolved = _top_level_qt_widget(anchor) if anchor is not None else None
    if resolved is not None:
        return resolved

    window = getattr(viewer, "window", None)
    if window is None:
        raise RuntimeError("The napari viewer does not expose a window.")

    # napari 0.9 emits a FutureWarning for this still-exposed compatibility
    # seam.  The capture tool needs the underlying QWidget, and centralizing
    # this access lets future napari layouts be accommodated in one place.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        qt_viewer = getattr(window, "qt_viewer", None)
    resolved = _top_level_qt_widget(qt_viewer)
    if resolved is not None:
        return resolved

    legacy_window = getattr(window, "_qt_window", None)
    resolved = _top_level_qt_widget(legacy_window)
    if resolved is not None:
        return resolved

    raise RuntimeError("Could not resolve napari's native Qt window.")
