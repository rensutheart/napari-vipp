from __future__ import annotations

import os
from threading import Event
from types import SimpleNamespace

import pytest

from napari_vipp.core.font_prefetch import prefetch_windows_fonts

pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows font registry")


def _registry(monkeypatch, entries):
    import winreg

    class Key:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    def enumerate_value(_key, index):
        if index >= len(entries):
            raise OSError("end of registry values")
        return str(index), entries[index], winreg.REG_SZ

    monkeypatch.setattr(winreg, "OpenKey", lambda *_args: Key())
    monkeypatch.setattr(winreg, "EnumValue", enumerate_value)


def test_font_prefetch_reads_only_unique_fixed_drive_font_files(monkeypatch, tmp_path):
    import ctypes
    from pathlib import Path

    monkeypatch.setenv("WINDIR", str(tmp_path))
    fonts = tmp_path / "Fonts"
    fonts.mkdir()
    local_font = fonts / "local.ttf"
    local_font.write_bytes(b"font header")
    _registry(
        monkeypatch,
        [
            "local.ttf",
            str(local_font).upper(),
            str(fonts / "missing.otf"),
            "ignored.txt",
            r"\\server\fonts\network.ttf",
            r"Z:\mapped\font.ttf",
            123,
        ],
    )
    drives = []

    def drive_type(root):
        drives.append(root)
        return 4 if root == "Z:\\" else 3

    monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", drive_type)
    opened = []
    original_open = Path.open

    def record_open(path, *args, **kwargs):
        opened.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", record_open)
    prefetch_windows_fonts(Event())
    assert opened == [local_font, fonts / "missing.otf"]
    assert drives.count(local_font.drive + "\\") == 1
    assert drives.count("Z:\\") == 1
    assert local_font.read_bytes() == b"font header"


def test_font_prefetch_stops_between_files_and_skips_qt_work(monkeypatch, tmp_path):
    import ctypes
    from pathlib import Path

    from napari_vipp.ui import startup_splash

    monkeypatch.setenv("WINDIR", str(tmp_path))
    _registry(monkeypatch, ["first.ttf", "second.ttf"])
    monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", lambda _root: 3)
    cancelled = Event()
    opened = []

    def stop_after_first(path, *_args, **_kwargs):
        opened.append(path)
        cancelled.set()
        raise OSError("unavailable font")

    monkeypatch.setattr(Path, "open", stop_after_first)
    monkeypatch.setattr(
        startup_splash,
        "QFontDatabase",
        SimpleNamespace(families=lambda: pytest.fail("font work after cancel")),
    )
    done = Event()
    startup_splash._prepare_font_database(done, cancelled)
    assert done.is_set()
    assert opened == [tmp_path / "Fonts" / "first.ttf"]


def test_font_prefetch_tolerates_inaccessible_registry(monkeypatch):
    import winreg

    def denied(*_args):
        raise PermissionError("unavailable font registry")

    monkeypatch.setattr(winreg, "OpenKey", denied)
    prefetch_windows_fonts(Event())
