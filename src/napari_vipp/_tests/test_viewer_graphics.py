"""Startup-only graphics policy; no native viewer or host settings required."""

from __future__ import annotations

import builtins
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QSettings, Qt

from napari_vipp import app
from napari_vipp.startup import LaunchProfile
from napari_vipp.ui import viewer_graphics as graphics
from napari_vipp.ui.viewer_graphics import ViewerGraphicsMode as Mode


class _Store:
    def __init__(self, value=None):
        self.values = {} if value is None else {graphics.VIEWER_GRAPHICS_SETTING: value}
        self.syncs = 0

    def value(self, key, default=None):
        return self.values.get(key, default)

    def setValue(self, key, value):  # noqa: N802
        self.values[key] = value

    def sync(self):
        self.syncs += 1


@pytest.fixture
def isolated_graphics(monkeypatch):
    attributes = {}
    env = {}
    modules = {}
    core = SimpleNamespace(
        instance=lambda: None,
        testAttribute=lambda attr: attributes.get(attr, False),
        setAttribute=lambda attr, enabled: attributes.__setitem__(attr, enabled),
    )
    monkeypatch.setattr(graphics, "QCoreApplication", core)
    monkeypatch.setattr(graphics, "os", SimpleNamespace(environ=env))
    monkeypatch.setattr(
        graphics, "sys", SimpleNamespace(platform="win32", modules=modules)
    )
    monkeypatch.setattr(graphics, "_standalone_startup", None)
    return SimpleNamespace(core=core, env=env, attributes=attributes, modules=modules)


def test_preferences_use_local_canonical_values_and_default_to_auto(tmp_path):
    filename = str(tmp_path / "viewer.ini")
    settings = QSettings(filename, QSettings.Format.IniFormat)
    assert graphics.load_viewer_graphics_mode(settings) is Mode.AUTO
    assert [mode.label for mode in Mode] == ["Auto", "GPU", "CPU"]
    assert graphics.save_viewer_graphics_mode(" GPU ", settings) is Mode.GPU
    again = QSettings(filename, QSettings.Format.IniFormat)
    assert graphics.load_viewer_graphics_mode(again) is Mode.GPU
    with pytest.raises(ValueError):
        graphics.save_viewer_graphics_mode("vulkan", again)
    assert graphics.load_viewer_graphics_mode(again) is Mode.GPU
    again.setValue(graphics.VIEWER_GRAPHICS_SETTING, "not-a-mode")
    assert graphics.load_viewer_graphics_mode(again) is Mode.AUTO


def test_default_store_honors_explicit_isolated_ini_format_and_path(tmp_path):
    script = """
import sys
from pathlib import Path
from qtpy.QtCore import QSettings

sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[2]).resolve()
QSettings.setDefaultFormat(QSettings.Format.IniFormat)
QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(root))
from napari_vipp.ui.viewer_graphics import (
    _settings, load_viewer_graphics_mode, save_viewer_graphics_mode,
)
store = _settings()
assert store.format() == QSettings.Format.IniFormat
assert not store.fallbacksEnabled()
filename = Path(store.fileName()).resolve()
assert filename == root / "napari-vipp" / "napari-vipp.ini"
assert load_viewer_graphics_mode() == "auto"
assert save_viewer_graphics_mode("cpu") == "cpu"
assert filename.is_file()
assert load_viewer_graphics_mode() == "cpu"
"""
    source = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(source), str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_auto_leaves_all_inherited_renderer_variables_and_qt_attributes(
    isolated_graphics, monkeypatch
):
    isolated_graphics.env.update(
        QT_OPENGL="software", QT_OPENGL_DLL="inherited", VISPY_GL_LIB="inherited"
    )
    isolated_graphics.attributes[Qt.ApplicationAttribute.AA_UseSoftwareOpenGL] = True
    old_env = dict(isolated_graphics.env)
    old_attributes = dict(isolated_graphics.attributes)
    monkeypatch.setattr(
        graphics, "_software_gl_library", lambda: pytest.fail("Auto probed")
    )
    startup = graphics.configure_viewer_graphics_startup(_Store())
    assert startup.requested is startup.selected is Mode.AUTO
    assert startup.reason == ""
    assert isolated_graphics.env == old_env
    assert isolated_graphics.attributes == old_attributes


def test_embedded_qt_never_reads_or_applies_the_standalone_setting(
    isolated_graphics, monkeypatch
):
    monkeypatch.setattr(isolated_graphics.core, "instance", object)
    monkeypatch.setattr(
        graphics,
        "load_viewer_graphics_mode",
        lambda *_: pytest.fail("Host read settings"),
    )
    assert graphics.configure_viewer_graphics_startup() is None
    availability = graphics.viewer_graphics_availability()
    assert not availability.can_configure
    assert "napari session" in availability.reason
    assert not isolated_graphics.env
    assert not isolated_graphics.attributes


def test_unsupported_platform_falls_back_without_saving_or_mutating(
    isolated_graphics, monkeypatch
):
    monkeypatch.setattr(graphics.sys, "platform", "darwin")
    store = _Store("gpu")
    startup = graphics.configure_viewer_graphics_startup(store)
    assert startup.requested is Mode.GPU
    assert startup.selected is Mode.AUTO
    assert "Windows" in startup.reason
    assert graphics.load_viewer_graphics_mode(store) is Mode.GPU
    assert store.syncs == 0
    assert not isolated_graphics.env
    assert not isolated_graphics.attributes


def test_gpu_requests_desktop_gl_and_clears_only_conflicting_process_overrides(
    isolated_graphics, monkeypatch
):
    monkeypatch.setattr(graphics, "_software_gl_library", lambda: None)
    isolated_graphics.env.update(
        PATH="unchanged",
        QT_OPENGL="software",
        QT_OPENGL_DLL="old.dll",
        VISPY_GL_LIB="old.dll",
        PYOPENGL_PLATFORM="egl",
        CUDA_VISIBLE_DEVICES="2",
    )
    startup = graphics.configure_viewer_graphics_startup(_Store("gpu"))
    assert startup.selected is Mode.GPU
    assert isolated_graphics.env == {
        "PATH": "unchanged",
        "QT_OPENGL": "desktop",
        "PYOPENGL_PLATFORM": "win32",
        "CUDA_VISIBLE_DEVICES": "2",
    }
    assert isolated_graphics.attributes == {
        Qt.ApplicationAttribute.AA_UseSoftwareOpenGL: False,
        Qt.ApplicationAttribute.AA_UseDesktopOpenGL: True,
    }
    monkeypatch.setattr(isolated_graphics.core, "instance", object)
    assert graphics.viewer_graphics_can_configure()
    assert not graphics.viewer_graphics_availability().cpu_available
    assert graphics.configure_viewer_graphics_startup(_Store("cpu")) is startup


def test_cpu_binds_the_same_library_for_qt_vispy_and_pyopengl(
    tmp_path, isolated_graphics, monkeypatch
):
    dll = tmp_path / "opengl32sw.dll"
    old_dispatch, new_dispatch = object(), object()
    namespace = {"GetCurrentContext": old_dispatch}
    monkeypatch.setattr(graphics, "_software_gl_library", lambda: dll)
    monkeypatch.setattr(graphics, "find_spec", lambda _: object())

    def stage(path):
        assert path == dll
        assert not isolated_graphics.env
        assert namespace["GetCurrentContext"] is old_dispatch
        return namespace, {"GetCurrentContext": new_dispatch}

    monkeypatch.setattr(graphics, "_prepare_software_bindings", stage)
    startup = graphics.configure_viewer_graphics_startup(_Store("cpu"))
    assert startup.selected is Mode.CPU
    assert namespace["GetCurrentContext"] is new_dispatch
    assert isolated_graphics.env == {
        "QT_OPENGL": "software",
        "QT_OPENGL_DLL": str(dll),
        "VISPY_GL_LIB": str(dll),
        "PYOPENGL_PLATFORM": "win32",
    }
    assert isolated_graphics.attributes == {
        Qt.ApplicationAttribute.AA_UseSoftwareOpenGL: True,
        Qt.ApplicationAttribute.AA_UseDesktopOpenGL: False,
    }


@pytest.mark.parametrize("problem", ["missing", "already-loaded", "bad-binary"])
def test_cpu_unavailable_or_too_late_leaves_auto_without_partial_changes(
    problem, tmp_path, isolated_graphics, monkeypatch
):
    dll = None if problem == "missing" else tmp_path / "opengl32sw.dll"
    monkeypatch.setattr(graphics, "_software_gl_library", lambda: dll)
    monkeypatch.setattr(graphics, "find_spec", lambda _: object())
    if problem == "already-loaded":
        isolated_graphics.modules["OpenGL.raw.GL"] = object()

    def fail(_path):
        if problem != "bad-binary":
            pytest.fail("Tried to bind an unavailable or already-loaded backend")
        raise OSError("incompatible DLL")

    monkeypatch.setattr(graphics, "_prepare_software_bindings", fail)
    startup = graphics.configure_viewer_graphics_startup(_Store("cpu"))
    assert startup.requested is Mode.CPU
    assert startup.selected is Mode.AUTO
    assert startup.reason
    assert not isolated_graphics.env
    assert not isolated_graphics.attributes


def test_cpu_configuration_restores_environment_and_does_not_install_on_failure(
    tmp_path, isolated_graphics, monkeypatch
):
    old_dispatch = object()
    namespace = {"GetCurrentContext": old_dispatch}
    monkeypatch.setattr(
        graphics, "_software_gl_library", lambda: tmp_path / "software.dll"
    )
    monkeypatch.setattr(graphics, "find_spec", lambda _: object())
    monkeypatch.setattr(
        graphics,
        "_prepare_software_bindings",
        lambda _: (namespace, {"GetCurrentContext": object()}),
    )
    isolated_graphics.env.update(PATH="unchanged", QT_OPENGL="original")
    software = Qt.ApplicationAttribute.AA_UseSoftwareOpenGL
    desktop = Qt.ApplicationAttribute.AA_UseDesktopOpenGL
    isolated_graphics.attributes.update({software: False, desktop: True})

    def set_attribute(attr, enabled):
        if attr == desktop and not enabled:
            raise RuntimeError("attribute failed")
        isolated_graphics.attributes[attr] = enabled

    monkeypatch.setattr(isolated_graphics.core, "setAttribute", set_attribute)
    startup = graphics.configure_viewer_graphics_startup(_Store("cpu"))
    assert startup.selected is Mode.AUTO
    assert namespace["GetCurrentContext"] is old_dispatch
    assert isolated_graphics.env == {"PATH": "unchanged", "QT_OPENGL": "original"}
    assert isolated_graphics.attributes == {software: False, desktop: True}


def test_standalone_configures_graphics_before_importing_napari(monkeypatch):
    calls = []
    original_import = builtins.__import__
    monkeypatch.setattr(
        graphics, "configure_viewer_graphics_startup", lambda: calls.append("graphics")
    )

    def import_before_gui(name, *args, **kwargs):
        if name == "napari":
            assert calls == ["loading_napari", "graphics"]
            raise RuntimeError("stopped before viewer construction")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_before_gui)
    reporter = SimpleNamespace(progress=calls.append)
    with pytest.raises(RuntimeError, match="stopped before viewer construction"):
        app.run_application(LaunchProfile.CPU, reporter)
