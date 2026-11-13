"""Plan native 3D display without modifying a scientific result."""

from __future__ import annotations

import ctypes
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod

import numpy as np
from qtpy.QtCore import QThread
from qtpy.QtGui import (
    QGuiApplication,
    QOffscreenSurface,
    QOpenGLContext,
    QSurfaceFormat,
)

_MIB = 1024**2
_UNKNOWN_BYTES = 512 * _MIB
_DEFAULT_BYTES = 1024 * _MIB
_SOFTWARE_BYTES = 896 * _MIB
_GL_MAX_3D_TEXTURE_SIZE = 0x8073
_GL_PROXY_TEXTURE_3D = 0x8070
_GL_TEXTURE_DIMENSIONS = (0x1000, 0x1001, 0x8071)
_GL_FORMAT_BY_BYTES = {1: 0x8229, 2: 0x822A, 4: 0x822E}  # R8, R16, R32F

Admission = Callable[[tuple[int, int, int], int], bool | None]


@dataclass(frozen=True)
class RendererVolumeLimits:
    max_texture_axis: int | None = 2048
    max_texture_bytes: int = _UNKNOWN_BYTES
    admit: Admission | None = None


@dataclass(frozen=True)
class VolumePreviewPlan:
    strides: tuple[int, ...]
    shape: tuple[int, ...]
    can_display_3d: bool = True
    reason: str = ""

    @property
    def is_reduced(self) -> bool:
        return any(step != 1 for step in self.strides)


def _texture_bytes_per_voxel(dtype: np.dtype) -> int:
    # napari's scalar display keeps uint8/uint16 and converts other numeric
    # types to float32. Boolean occupancy uses the native byte path.
    if dtype.kind == "b" or dtype == np.dtype(np.uint8):
        return 1
    if dtype == np.dtype(np.uint16):
        return 2
    return 4


def plan_volume_preview(
    data: np.ndarray,
    spatial_axes: Sequence[int],
    *,
    is_binary_mask: bool = False,
    allow_reduction: bool = True,
    texture_bytes_per_voxel: int | None = None,
    max_texture_axis: int | None = None,
    max_texture_bytes: int | None = None,
    admit: Admission | None = None,
) -> VolumePreviewPlan:
    """Choose a calibrated sampling grid for a scalar 3D texture.

    Supply spatial axes in Z, Y, X order. Time, channel and other leading
    dimensions are not sampled. The input dtype and individual voxel values
    are unchanged. Use the native display texture size, when known, rather than
    the scientific dtype's size. Pass allow_reduction=False for labels.
    """
    shape = tuple(data.shape)
    axes = tuple(spatial_axes)
    full = (1,) * len(shape)
    if (
        len(axes) != 3
        or len(set(axes)) != 3
        or any(not 0 <= axis < len(shape) or shape[axis] == 0 for axis in axes)
    ):
        return VolumePreviewPlan(full, shape, False, "No valid three-axis volume.")
    byte_limit = max_texture_bytes if max_texture_bytes is not None else _UNKNOWN_BYTES
    bytes_per_voxel = texture_bytes_per_voxel
    if bytes_per_voxel is None:
        bytes_per_voxel = 1 if is_binary_mask else _texture_bytes_per_voxel(data.dtype)
    elif bytes_per_voxel not in _GL_FORMAT_BY_BYTES:
        raise ValueError("Unsupported scalar texture size.")

    def dimensions(steps: list[int]) -> tuple[int, ...]:
        return tuple(
            (length + step - 1) // step
            for length, step in zip(shape, steps, strict=True)
        )

    def fits(view_shape: tuple[int, ...]) -> bool:
        x, y, z = (view_shape[axis] for axis in reversed(axes))
        if max_texture_axis is not None and max(x, y, z) > max_texture_axis:
            return False
        if x * y * z * bytes_per_voxel > byte_limit:
            return False
        return admit is None or admit((x, y, z), bytes_per_voxel) is not False

    if fits(shape):
        return VolumePreviewPlan(full, shape)
    reason = "The full-resolution volume exceeds this renderer's 3D texture budget."
    if is_binary_mask or not allow_reduction:
        return VolumePreviewPlan(full, shape, False, reason)

    steps = list(full)
    if max_texture_axis and max_texture_axis > 0:
        for axis in axes:
            steps[axis] = (shape[axis] + max_texture_axis - 1) // max_texture_axis
    while True:
        view_shape = dimensions(steps)
        if view_shape != shape and fits(view_shape):
            return VolumePreviewPlan(tuple(steps), view_shape, reason=reason)
        # Prefer reducing the two longest dimensions (normally X/Y), keeping
        # the finer sampling along the already shorter stack axis as long as possible.
        largest = sorted(axes, key=view_shape.__getitem__, reverse=True)[:2]
        if prod(view_shape[axis] for axis in axes) == 1:
            return VolumePreviewPlan(full, shape, False, reason)
        for axis in largest:
            if view_shape[axis] > 1:
                steps[axis] = min(shape[axis], steps[axis] * 2)


def preview_data(data: np.ndarray, plan: VolumePreviewPlan) -> np.ndarray:
    """Return a read-only, no-copy view with the original voxel dtype."""
    if not plan.can_display_3d:
        raise ValueError(plan.reason)
    view = data[tuple(slice(None, None, step) for step in plan.strides)].view()
    view.setflags(write=False)
    return view


def _proc(context: QOpenGLContext, name: str, result, *args):
    pointer = context.getProcAddress(name.encode("ascii"))
    address = int(pointer) if pointer else 0
    invalid = {0, 1, 2, 3, (1 << (8 * ctypes.sizeof(ctypes.c_void_p))) - 1}
    if address in invalid:
        raise RuntimeError(f"OpenGL function unavailable: {name}")
    factory = ctypes.WINFUNCTYPE if sys.platform == "win32" else ctypes.CFUNCTYPE
    return factory(result, *args)(address)


class _VolumeProbe:
    """One private context; never probes or allocates on a viewer's canvas."""

    def __init__(self, context: QOpenGLContext, surface: QOffscreenSurface):
        self.context = context
        self.surface = surface
        self._admission: dict[tuple[tuple[int, int, int], int], bool] = {}
        enum, integer, pointer = ctypes.c_uint, ctypes.c_int, ctypes.c_void_p
        self.error = _proc(context, "glGetError", enum)
        self.get_int = _proc(
            context, "glGetIntegerv", None, enum, ctypes.POINTER(integer)
        )
        self.get_string = _proc(context, "glGetString", ctypes.c_char_p, enum)
        self.level = _proc(
            context,
            "glGetTexLevelParameteriv",
            None,
            enum,
            integer,
            enum,
            ctypes.POINTER(integer),
        )
        self.image = _proc(
            context,
            "glTexImage3D",
            None,
            enum,
            integer,
            integer,
            integer,
            integer,
            integer,
            integer,
            enum,
            enum,
            pointer,
        )

    def integer(self, key: int) -> int:
        value = ctypes.c_int()
        self.get_int(key, ctypes.byref(value))
        return value.value if self.error() == 0 else 0

    def _proxy(self, xyz: tuple[int, int, int], size: int) -> bool | None:
        internal = _GL_FORMAT_BY_BYTES.get(size)
        if internal is None:
            return None
        # GL_RED/GL_UNSIGNED_BYTE describes only the (NULL) transfer buffer.
        # GL_PROXY_TEXTURE_3D performs the size check without allocating pixels.
        self.image(_GL_PROXY_TEXTURE_3D, 0, internal, *xyz, 0, 0x1903, 0x1401, None)
        if self.error():
            return None
        result = []
        for key in _GL_TEXTURE_DIMENSIONS:
            value = ctypes.c_int()
            self.level(_GL_PROXY_TEXTURE_3D, 0, key, ctypes.byref(value))
            if self.error():
                return None
            result.append(value.value)
        return tuple(result) == xyz

    def admit(self, xyz: tuple[int, int, int], size: int) -> bool | None:
        key = xyz, size
        if key in self._admission:
            return self._admission[key]
        app = QGuiApplication.instance()
        if app is None or QThread.currentThread() != app.thread():
            return None
        # Do not interrupt a paint callback or steal another context.
        if QOpenGLContext.currentContext() is not None:
            return None
        if not self.context.makeCurrent(self.surface):
            return None
        try:
            # Some platforms expose the entry points but not proxy textures
            # for this format. Only trust a rejection after a small control.
            if self._proxy((8, 8, 8), size) is not True:
                return None
            result = self._proxy(xyz, size)
            if result is not None:
                if len(self._admission) >= 128:
                    self._admission.pop(next(iter(self._admission)))
                self._admission[key] = result
            return result
        finally:
            self.context.doneCurrent()


def renderer_volume_limits(
    context: QOpenGLContext | None = None,
) -> RendererVolumeLimits:
    """Query the process's chosen desktop renderer, with a conservative fallback."""
    fallback = RendererVolumeLimits()
    app = QGuiApplication.instance()
    if (
        not isinstance(app, QGuiApplication)
        or QThread.currentThread() != app.thread()
        or app.platformName() in {"offscreen", "minimal"}
    ):
        return fallback
    existing = getattr(app, "_vipp_renderer_volume_limits", None)
    if existing is not None:
        return existing
    if QOpenGLContext.currentContext() is not None:
        return fallback
    surface = QOffscreenSurface()
    probe_context = QOpenGLContext()
    probe_context.setFormat(
        context.format() if context is not None else QSurfaceFormat.defaultFormat()
    )
    result = fallback
    keep_surface = False
    try:
        if probe_context.create() and not probe_context.isOpenGLES():
            surface.setFormat(probe_context.format())
            surface.create()
            if surface.isValid() and probe_context.makeCurrent(surface):
                try:
                    probe = _VolumeProbe(probe_context, surface)
                    maximum = probe.integer(_GL_MAX_3D_TEXTURE_SIZE)
                    if maximum > 0:
                        renderer = (
                            probe.get_string(0x1F01) or b""
                        ).lower()  # GL_RENDERER
                        software = any(
                            name in renderer
                            for name in (b"llvmpipe", b"softpipe", b"software")
                        )
                        budget = _SOFTWARE_BYTES if software else _DEFAULT_BYTES
                        result = RendererVolumeLimits(maximum, budget, probe.admit)
                        keep_surface = True
                finally:
                    probe_context.doneCurrent()
    except (RuntimeError, TypeError, ValueError):
        # Unsupported proxy facilities are not a reason to break the viewer.
        result = fallback
    finally:
        if not keep_surface:
            surface.destroy()
    app._vipp_renderer_volume_limits = result
    return result
