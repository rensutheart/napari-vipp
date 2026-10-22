"""Read-only Windows font prefetching for the standalone startup process."""

from __future__ import annotations

import os
from pathlib import Path
from threading import Event


def prefetch_windows_fonts(cancelled: Event) -> None:
    """Warm installed local font files using I/O that releases Python's GIL.

    Qt's Python font-database call can hold the GIL during file scanning. Read
    the configured font files first so those waits happen outside the GUI and
    outside that binding call. No font is registered, removed or changed.
    """
    if os.name != "nt":
        return
    import ctypes
    import winreg

    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    seen: set[str] = set()
    drive_types: dict[str, int] = {}
    get_drive_type = ctypes.windll.kernel32.GetDriveTypeW
    get_drive_type.argtypes = [ctypes.c_wchar_p]
    get_drive_type.restype = ctypes.c_uint
    for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        if cancelled.is_set():
            return
        try:
            key = winreg.OpenKey(
                hive, r"SOFTWARE\Microsoft\Windows NT\CurrentVersion\Fonts"
            )
        except OSError:
            continue
        with key:
            index = 0
            while not cancelled.is_set():
                try:
                    _name, value, _kind = winreg.EnumValue(key, index)
                except OSError:
                    break
                index += 1
                if not isinstance(value, str):
                    continue
                path = Path(os.path.expandvars(value))
                if not path.is_absolute():
                    path = windows_fonts / path
                if path.suffix.lower() not in {".ttf", ".otf", ".ttc", ".fon"}:
                    continue
                # Avoid turning a startup optimization into network access.
                if len(path.drive) != 2 or path.drive[1] != ":":
                    continue
                drive = path.drive + "\\"
                if drive not in drive_types:
                    drive_types[drive] = get_drive_type(drive)
                if drive_types[drive] != 3:  # DRIVE_FIXED
                    continue
                identity = os.path.normcase(str(path))
                if identity in seen:
                    continue
                seen.add(identity)
                try:
                    with path.open("rb") as stream:
                        stream.read(4096)
                except OSError:
                    # Font availability/interpretation remains Qt's decision.
                    continue
