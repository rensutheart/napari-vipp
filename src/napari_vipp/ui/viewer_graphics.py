"""Local, next-launch graphics preference for the standalone VIPP viewer."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Protocol

from qtpy.QtCore import QCoreApplication, QLibraryInfo, QSettings, Qt

VIEWER_GRAPHICS_SETTING = "viewer/graphics-mode-v1"
_SOFTWARE_GL_NAME = "opengl32sw.dll"
_LOG = logging.getLogger(__name__)


class ViewerGraphicsMode(StrEnum):
    """Viewer rendering preference, independent of workflow compute."""

    AUTO = "auto"
    GPU = "gpu"
    CPU = "cpu"

    @property
    def label(self) -> str:
        return {self.AUTO: "Auto", self.GPU: "GPU", self.CPU: "CPU"}[self]


class _Settings(Protocol):
    def value(self, key: str, default=None): ...
    def setValue(self, key: str, value) -> None: ...  # noqa: N802
    def sync(self) -> None: ...


@dataclass(frozen=True)
class ViewerGraphicsAvailability:
    can_configure: bool
    reason: str = ""
    cpu_available: bool = False
    cpu_reason: str = ""


@dataclass(frozen=True)
class ViewerGraphicsStartup:
    requested: ViewerGraphicsMode
    selected: ViewerGraphicsMode
    reason: str = ""


_standalone_startup: ViewerGraphicsStartup | None = None


class _GraphicsRollbackError(RuntimeError):
    """Startup must stop if it cannot restore the original backend selection."""


def _settings() -> QSettings:
    # The two-string overload always uses NativeFormat, ignoring a caller's
    # default format (for example an isolated, file-backed standalone profile).
    store = QSettings(
        QSettings.defaultFormat(),
        QSettings.Scope.UserScope,
        "napari-vipp",
        "napari-vipp",
    )
    store.setFallbacksEnabled(False)
    return store


def load_viewer_graphics_mode(settings: _Settings | None = None) -> ViewerGraphicsMode:
    """Read the local setting without changing the graphics environment."""
    store = _settings() if settings is None else settings
    value = store.value(VIEWER_GRAPHICS_SETTING, ViewerGraphicsMode.AUTO.value)
    try:
        return ViewerGraphicsMode(str(value).strip().casefold())
    except ValueError:
        return ViewerGraphicsMode.AUTO


def save_viewer_graphics_mode(
    mode: ViewerGraphicsMode | str, settings: _Settings | None = None
) -> ViewerGraphicsMode:
    """Persist a canonical mode; never reconfigure a running Qt application."""
    choice = ViewerGraphicsMode(str(mode).strip().casefold())
    store = _settings() if settings is None else settings
    store.setValue(VIEWER_GRAPHICS_SETTING, choice.value)
    store.sync()
    if isinstance(store, QSettings) and store.status() != QSettings.Status.NoError:
        raise OSError("The Viewer graphics preference could not be saved.")
    return choice


def _software_gl_library() -> Path | None:
    """Find the selected Qt binding's own software DLL, never one on PATH."""
    try:
        binary_dir = Path(QLibraryInfo.path(QLibraryInfo.LibraryPath.BinariesPath))
        if not binary_dir.is_absolute():
            return None
        library = binary_dir / _SOFTWARE_GL_NAME
        if library.is_file():
            return library
    except (AttributeError, OSError, RuntimeError, ValueError):
        pass
    return None


def viewer_graphics_availability() -> ViewerGraphicsAvailability:
    """Report whether standalone next-launch choices are supported here."""
    if QCoreApplication.instance() is not None and _standalone_startup is None:
        return ViewerGraphicsAvailability(
            False, "This napari session owns the viewer renderer."
        )
    if sys.platform != "win32":
        return ViewerGraphicsAvailability(
            False,
            "Viewer graphics choices are available in standalone VIPP on Windows.",
        )
    if _software_gl_library() is None:
        return ViewerGraphicsAvailability(
            True, cpu_reason="This Qt installation does not include software OpenGL."
        )
    if find_spec("OpenGL") is None:
        return ViewerGraphicsAvailability(
            True, cpu_reason="Software OpenGL needs the installed PyOpenGL bindings."
        )
    return ViewerGraphicsAvailability(True, cpu_available=True)


def viewer_graphics_can_configure() -> bool:
    """Return whether this standalone process can save a next-launch choice."""
    return viewer_graphics_availability().can_configure


def viewer_graphics_startup() -> ViewerGraphicsStartup | None:
    """Return the standalone startup choice, not a claim about the active driver."""
    return _standalone_startup


def _render_bindings_loaded() -> bool:
    prefixes = (
        "OpenGL.platform",
        "OpenGL.GL",
        "OpenGL.raw",
        "OpenGL.error",
        "vispy.gloo",
    )
    return any(name.startswith(prefixes) for name in sys.modules)


def _prepare_software_bindings(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stage matching WGL exports before touching PyOpenGL's live namespace."""
    dll = ctypes.WinDLL(str(path))
    # Import under the existing environment. If a custom platform is selected,
    # do not switch or leave its cached bindings in an inconsistent state.
    from OpenGL import platform
    from OpenGL.platform.win32 import Win32Platform

    if type(platform.PLATFORM) is not Win32Platform:
        raise RuntimeError("PyOpenGL is configured for a non-Windows graphics backend.")
    candidate = Win32Platform()
    candidate.GL = dll
    exports: dict[str, Any] = {}
    candidate.install(exports)
    if (
        candidate.OpenGL is not dll
        or candidate.WGL is not dll
        or exports.get("GetCurrentContext") is not dll.wglGetCurrentContext
        or candidate.getExtensionProcedure is not dll.wglGetProcAddress
    ):
        raise RuntimeError(
            "Qt and PyOpenGL software context bindings could not be matched."
        )
    # The application needs both ordinary GL and WGL extension dispatch.
    _ = dll.glGetString
    return vars(platform), exports


def _apply_windows_mode(mode: ViewerGraphicsMode) -> None:
    """Select a consistent process-only backend before Qt creates a window."""
    software = mode is ViewerGraphicsMode.CPU
    namespace = exports = {}
    path = None
    if software:
        path = _software_gl_library()
        if path is None:
            raise RuntimeError("This Qt installation does not include software OpenGL.")
        namespace, exports = _prepare_software_bindings(path)
    values = {
        "QT_OPENGL": "software" if software else "desktop",
        "QT_OPENGL_DLL": str(path) if software else None,
        # A desktop request uses VisPy's native Windows loader, not an
        # inherited software/custom DLL override.
        "VISPY_GL_LIB": str(path) if software else None,
        "PYOPENGL_PLATFORM": "win32",
    }
    attributes = {
        Qt.ApplicationAttribute.AA_UseSoftwareOpenGL: software,
        Qt.ApplicationAttribute.AA_UseDesktopOpenGL: not software,
    }
    old_env = {key: os.environ.get(key) for key in values}
    old_attrs = {key: QCoreApplication.testAttribute(key) for key in attributes}

    def set_environment(entries: dict[str, str | None]) -> None:
        for key, value in entries.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    try:
        set_environment(values)
        for attr, enabled in attributes.items():
            QCoreApplication.setAttribute(attr, enabled)
    except (OSError, RuntimeError, TypeError, ValueError):
        try:
            set_environment(old_env)
            for attr, enabled in old_attrs.items():
                QCoreApplication.setAttribute(attr, enabled)
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise _GraphicsRollbackError(
                "Viewer graphics startup could not restore the original configuration."
            ) from exc
        raise
    # The staged export mapping contains no calls that can activate a context.
    namespace.update(exports)


def configure_viewer_graphics_startup(
    settings: _Settings | None = None,
) -> ViewerGraphicsStartup | None:
    """Apply the local preference once, before importing standalone napari."""
    global _standalone_startup
    if _standalone_startup is not None:
        return _standalone_startup
    if QCoreApplication.instance() is not None:
        return None
    mode = load_viewer_graphics_mode(settings)
    selected = ViewerGraphicsMode.AUTO
    reason = ""
    if mode is not ViewerGraphicsMode.AUTO:
        availability = viewer_graphics_availability()
        if not availability.can_configure:
            reason = availability.reason
        elif mode is ViewerGraphicsMode.CPU and not availability.cpu_available:
            reason = availability.cpu_reason
        elif _render_bindings_loaded():
            reason = "The process has already loaded its OpenGL bindings."
        else:
            try:
                _apply_windows_mode(mode)
            except _GraphicsRollbackError:
                # Starting Qt with an indeterminate backend is unsafe.
                raise
            except (
                AttributeError,
                ImportError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ):
                # A missing or incompatible DLL never selects a mixed backend.
                reason = f"The {mode.label} viewer renderer could not be configured."
                _LOG.debug("Viewer graphics startup detail", exc_info=True)
            else:
                selected = mode
    _standalone_startup = ViewerGraphicsStartup(mode, selected, reason)
    if reason:
        _LOG.warning("Viewer graphics: %s Keeping Auto.", reason)
    return _standalone_startup
