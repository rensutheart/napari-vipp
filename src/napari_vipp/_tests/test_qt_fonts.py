"""Offscreen font setup is conditional and cannot silently test tofu metrics."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp._tests import _qt_fonts


def test_packaging_conftest_loads_without_installed_test_package():
    """Installer checks use a wheel without _tests and never request qapp."""
    conftest = Path(__file__).with_name("conftest.py")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.abc, runpy, sys\n"
            "class NoInstalledTests(importlib.abc.MetaPathFinder):\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.startswith('napari_vipp._tests'):\n"
            "            raise ModuleNotFoundError(fullname)\n"
            "sys.meta_path.insert(0, NoInstalledTests())\n"
            "runpy.run_path(sys.argv[1])\n",
            str(conftest),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


class _FontDatabase:
    def __init__(self, families=(), *, valid=True):
        self.names = list(families)
        self.loaded = []
        self.valid = valid

    def families(self):
        return self.names

    def addApplicationFont(self, path):  # noqa: N802
        self.loaded.append(Path(path).name)
        if not self.valid:
            return -1
        self.names = ["Segoe UI"]
        return len(self.loaded) - 1


def _environment(monkeypatch, tmp_path, *, platform="win32", names=(), valid=True):
    monkeypatch.setattr(_qt_fonts, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setenv("WINDIR", str(tmp_path))
    database = _FontDatabase(names, valid=valid)
    monkeypatch.setattr(_qt_fonts, "QFontDatabase", database)
    return database


def test_empty_windows_offscreen_font_database_loads_available_native_faces(
    monkeypatch, tmp_path
):
    database = _environment(monkeypatch, tmp_path)
    fonts = tmp_path / "Fonts"
    fonts.mkdir()
    for name in ("segoeui.ttf", "segoeuib.ttf", "seguisb.ttf"):
        (fonts / name).write_bytes(b"test font handled by fake database")
    qapp = SimpleNamespace(platformName=lambda: "offscreen")

    assert _qt_fonts.load_offscreen_windows_fonts(qapp) == (0, 1, 2)
    assert database.loaded == ["segoeui.ttf", "segoeuib.ttf", "seguisb.ttf"]
    assert _qt_fonts.load_offscreen_windows_fonts(qapp) == ()
    assert len(database.loaded) == 3


@pytest.mark.parametrize(
    ("platform", "plugin", "families"),
    [
        ("linux", "offscreen", ()),
        ("darwin", "offscreen", ()),
        ("win32", "windows", ()),
        ("win32", "offscreen", ("Already available",)),
    ],
)
def test_healthy_or_non_windows_font_environment_is_unchanged(
    monkeypatch, tmp_path, platform, plugin, families
):
    database = _environment(monkeypatch, tmp_path, platform=platform, names=families)
    assert _qt_fonts.load_offscreen_windows_fonts(
        SimpleNamespace(platformName=lambda: plugin)
    ) == ()
    assert database.loaded == []


@pytest.mark.parametrize("file_exists", [False, True])
def test_missing_or_unreadable_native_fonts_report_actionable_setup_failure(
    monkeypatch, tmp_path, file_exists
):
    _environment(monkeypatch, tmp_path, valid=False)
    if file_exists:
        fonts = tmp_path / "Fonts"
        fonts.mkdir()
        (fonts / "segoeui.ttf").write_bytes(b"unreadable font")
    with pytest.raises(RuntimeError, match="UI geometry tests require real glyph"):
        _qt_fonts.load_offscreen_windows_fonts(
            SimpleNamespace(platformName=lambda: "offscreen")
        )
