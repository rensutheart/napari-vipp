"""Recognize a deferred OpenGL volume failure without intercepting the draw."""

from __future__ import annotations

from collections.abc import Iterable
from itertools import islice

_GL_OUT_OF_MEMORY = 0x0505
_GL_TEXTURE_3D = 0x806F
_MAX_CANVASES = 8
_MAX_TRACEBACK_FRAMES = 64


def renderer_error_notifications():
    """Use napari's existing notifications when available; install no hook."""
    try:
        from napari.utils.notifications import notification_manager

        emitter = notification_manager.notification_ready
    except (ImportError, AttributeError):
        return None
    if callable(getattr(emitter, "connect", None)) and callable(
        getattr(emitter, "disconnect", None)
    ):
        return emitter
    return None


def scene_canvas(qt_viewer):
    """Return the recognized SceneCanvas of this exact Qt viewer."""
    try:
        from napari._vispy.canvas import NapariSceneCanvas, VispyCanvas

        canvas = qt_viewer.canvas
        model = qt_viewer.viewer
        if (
            model is not None
            and isinstance(canvas, VispyCanvas)
            and canvas.viewer is model
            and isinstance(canvas._scene_canvas, NapariSceneCanvas)
        ):
            return canvas._scene_canvas
    except (ImportError, AttributeError, RuntimeError):
        pass
    return None


def is_3d_texture_allocation_failure(
    notification, scene_canvases: Iterable[object]
) -> bool:
    """Identify a real 3D texture allocation/upload OOM in an owned canvas.

    A draw flushes queued GLIR commands for the scene, so the traceback can
    establish the canvas but not the currently selected layer. In particular
    a proxy rejection, a black frame, and a driver crash without a Python
    exception are not evidence of this failure. Do not retain the traceback,
    issue GL calls, or consume the host notification.
    """
    try:
        from napari.utils.notifications import ErrorNotification
        from OpenGL.error import GLError
    except ImportError:
        return False
    if not isinstance(notification, ErrorNotification):
        return False
    error = notification.exception
    if not isinstance(error, GLError) or error.err != _GL_OUT_OF_MEMORY:
        return False

    try:
        from napari._vispy.canvas import NapariSceneCanvas
        from vispy.gloo.glir import glTexImage3D, glTexSubImage3D
        from vispy.scene import SceneCanvas

        owned = tuple(
            canvas
            for canvas in islice(scene_canvases, _MAX_CANVASES)
            if isinstance(canvas, NapariSceneCanvas)
        )
        draw_codes = {
            NapariSceneCanvas.draw_visual.__code__,
            SceneCanvas.draw_visual.__code__,
        }
        upload_codes = {glTexImage3D.__code__, glTexSubImage3D.__code__}
    except (ImportError, AttributeError, TypeError):
        return False
    if not owned:
        return False

    draw_canvas = None
    traceback = error.__traceback__
    for _ in range(_MAX_TRACEBACK_FRAMES):
        if traceback is None:
            return False
        frame = traceback.tb_frame
        if frame.f_code in draw_codes:
            # Use the closest actual draw; an unrelated canvas can draw
            # inside another canvas callback in the same GUI process.
            draw_canvas = frame.f_locals.get("self")
        elif (
            frame.f_code in upload_codes
            and frame.f_locals.get("target") == _GL_TEXTURE_3D
        ):
            return any(draw_canvas is canvas for canvas in owned)
        traceback = traceback.tb_next
    return False
