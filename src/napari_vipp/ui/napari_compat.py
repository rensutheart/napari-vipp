"""Feature-detected access to napari presentation objects.

Keep the small amount of Qt integration needed outside VIPP's main widget in
one place.  These helpers deliberately detect available objects instead of
branching on napari or Qt-binding version numbers.
"""

from __future__ import annotations

import warnings
from functools import wraps
from types import MethodType
from typing import Any
from weakref import ref

from qtpy.QtCore import QSignalBlocker, QSize, Qt
from qtpy.QtGui import QPixmap
from qtpy.QtWidgets import QLineEdit, QWidget


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


def install_layer_cursor_safety(layer: Any) -> bool:
    """Skip the known 3D cursor read against a stale 2D slice, on this layer."""
    current = getattr(layer, "_get_value_ray", None)
    if not callable(current):
        return False
    if getattr(current, "_vipp_safe_cursor_ray", False):
        return True
    try:
        from napari.layers._scalar_field.scalar_field import ScalarFieldBase
    except ImportError:
        return False

    original = getattr(ScalarFieldBase, "_get_value_ray", None)
    code = getattr(original, "__code__", None)
    if (
        not isinstance(layer, ScalarFieldBase)
        or original is not getattr(current, "__func__", None)
        or code is None
        or not {"slice_shape", "level0_shape", "dims_displayed", "ds"}.issubset(
            code.co_varnames
        )
    ):
        return False
    layer_ref = ref(layer)

    @wraps(original)
    def read_ray(*args, **kwargs):
        instance = layer_ref()
        if instance is None:
            return None
        try:
            return original(instance, *args, **kwargs)
        except ValueError as error:
            # The status thread can race a 2D/3D change even after a readiness
            # check. Inspect only the stock ray method's *failed snapshot*,
            # not the live slice (which may already have caught up).
            trace = error.__traceback__
            while trace is not None and trace.tb_next is not None:
                trace = trace.tb_next
            if trace is not None and trace.tb_frame.f_code is code:
                state = trace.tb_frame.f_locals
                if (
                    len(state.get("dims_displayed", ())) == 3
                    and getattr(state.get("slice_shape"), "shape", ()) == (2,)
                    and "level0_shape" in state
                    and "ds" not in state
                ):
                    return None
            raise

    read_ray._vipp_safe_cursor_ray = True
    layer._get_value_ray = read_ray
    return True


def install_layer_delegate_safety(viewer: Any) -> bool:
    """Guard this viewer's napari layer decorations during model replacement.

    A queued layer-list repaint can hold a nominally valid proxy index whose
    roles have disappeared. Napari's loading painter treats the absent loading
    role as False, then calls ``height()`` on the absent size hint. Missing roles
    describe no drawable decoration; they do not mean a layer is loading.

    Keep the existing delegate, signals, editing and ordinary painting intact.
    Only its two Python decoration helpers are wrapped, only for the stock
    napari delegate. A real loading/loaded item without an optional size hint
    uses the actual row rectangle. No layer data or model roles are modified.
    Return False for model-only viewers or unfamiliar delegate layouts.
    """
    window = getattr(viewer, "window", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        qt_viewer = getattr(window, "qt_viewer", None)
    layers = getattr(qt_viewer, "layers", None)
    get_delegate = getattr(layers, "itemDelegate", None)
    if not callable(get_delegate):
        return False

    # This is an intentionally bounded napari-private compatibility seam.
    try:
        from napari._qt.containers._layer_delegate import LayerDelegate
        from napari._qt.containers.qt_layer_model import LoadedRole, ThumbnailRole
    except ImportError:
        return False

    delegate = get_delegate()
    if type(delegate) is not LayerDelegate:
        return False
    if getattr(delegate, "_vipp_safe_layer_decorations", False):
        return True
    original_loading = getattr(type(delegate), "_paint_loading", None)
    original_thumbnail = getattr(type(delegate), "_paint_thumbnail", None)
    if not (
        callable(original_loading)
        and callable(original_thumbnail)
        and getattr(delegate, "_load_movie", None) is not None
    ):
        return False

    @wraps(original_loading)
    def paint_loading(instance, painter, option, index):
        if not index.isValid():
            return
        loaded = index.data(LoadedRole)
        if loaded is None:
            return
        size = index.data(Qt.ItemDataRole.SizeHintRole)
        if isinstance(size, QSize) and size.isValid():
            return original_loading(instance, painter, option, index)
        if loaded:
            return
        height = option.rect.height() - 16
        if height <= 0:
            return
        instance._load_movie.start()
        rect = option.rect.translated(4, 8)
        rect.setSize(QSize(height, height))
        painter.drawPixmap(rect, instance._load_movie.currentPixmap())

    @wraps(original_thumbnail)
    def paint_thumbnail(instance, painter, option, index):
        if not index.isValid() or not index.data(LoadedRole):
            return
        image = index.data(ThumbnailRole)
        if image is None:
            return
        size = index.data(Qt.ItemDataRole.SizeHintRole)
        if isinstance(size, QSize) and size.isValid():
            return original_thumbnail(instance, painter, option, index)
        height = option.rect.height() - 4
        if height <= 0:
            return
        # Preserve napari's shared animation lifetime when its source model
        # exposes the usual all_loaded contract; a generic Qt model need not.
        model = index.model()
        source_model = getattr(model, "sourceModel", lambda: model)()
        all_loaded = getattr(source_model, "all_loaded", None)
        if callable(all_loaded) and all_loaded():
            instance._load_movie.setPaused(True)
        rect = option.rect.translated(-2, 2)
        rect.setSize(QSize(height, height))
        painter.drawPixmap(rect, QPixmap.fromImage(image))

    delegate._paint_loading = MethodType(paint_loading, delegate)
    delegate._paint_thumbnail = MethodType(paint_thumbnail, delegate)
    delegate._vipp_safe_layer_decorations = True
    return True


def install_dim_slider_retirement_safety(viewer: Any) -> bool:
    """Hide retired native slice controls before their deferred deletion."""

    window = getattr(viewer, "window", None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        qt_viewer = getattr(window, "qt_viewer", None)
    panel = getattr(qt_viewer, "dims", None)
    if panel is None:
        return False
    try:
        from napari._qt.widgets.qt_dims import QtDims
    except ImportError:
        return False
    if type(panel) is not QtDims or panel.dims is not getattr(viewer, "dims", None):
        return False
    current = getattr(panel, "_remove_slider_widget", None)
    if getattr(current, "_vipp_hide_retired_slider", False):
        return True
    original = getattr(QtDims, "_remove_slider_widget", None)
    code = getattr(original, "__code__", None)
    if (
        original is not getattr(current, "__func__", None)
        or code is None
        or not {"slider_widgets", "removeWidget", "deleteLater"}.issubset(
            code.co_names
        )
    ):
        return False

    @wraps(original)
    def hide_retired_slider(instance, index):
        row = instance.slider_widgets[index]
        original(instance, index)
        if not row.isHidden():
            # removeWidget() does not hide the child. Keep stock disposal,
            # but don't show the obsolete row in a nested Qt event loop.
            # Hiding a focused editor must not commit its obsolete value.
            blockers = [QSignalBlocker(edit) for edit in row.findChildren(QLineEdit)]
            try:
                row.hide()
            finally:
                for blocker in blockers:
                    blocker.unblock()

    hide_retired_slider._vipp_hide_retired_slider = True
    panel._remove_slider_widget = MethodType(hide_retired_slider, panel)
    return True
