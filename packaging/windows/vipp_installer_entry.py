"""Frozen entry point kept separate from the importable setup GUI."""

from __future__ import annotations

import ctypes
import sys
from ctypes import wintypes


def _single_instance_mutex():
    if sys.platform != "win32" or "--version" in sys.argv[1:]:
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateMutexW.argtypes = (
        ctypes.c_void_p,
        wintypes.BOOL,
        wintypes.LPCWSTR,
    )
    kernel32.CreateMutexW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    # The first setup process owns this mutex until it exits. The detached
    # uninstall cleanup helper acquires the same mutex before deleting a cached
    # setup executable, so a new same-version setup cannot race that deletion.
    handle = kernel32.CreateMutexW(None, True, "Local\\VIPP.Setup.SingleInstance")
    if not handle:
        return None
    if ctypes.get_last_error() == 183:  # ERROR_ALREADY_EXISTS
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        user32.MessageBoxW.argtypes = (
            wintypes.HWND,
            wintypes.LPCWSTR,
            wintypes.LPCWSTR,
            wintypes.UINT,
        )
        user32.MessageBoxW.restype = ctypes.c_int
        user32.MessageBoxW(
            None,
            "VIPP Setup is already open.",
            "VIPP Setup",
            0x00000040,
        )
        kernel32.CloseHandle(handle)
        raise SystemExit(4)
    return handle


def _release_single_instance_mutex(handle) -> None:
    if not handle or sys.platform != "win32":
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.ReleaseMutex.argtypes = (wintypes.HANDLE,)
    kernel32.ReleaseMutex.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.ReleaseMutex(handle)
    kernel32.CloseHandle(handle)


_MUTEX = _single_instance_mutex()

try:
    import pyi_splash

    pyi_splash.update_text("Opening the setup window…")
except ImportError:
    pass

if "--uninstall" in sys.argv[1:]:
    from napari_vipp.installer.uninstall_gui import main  # noqa: E402
else:
    from napari_vipp.installer.gui import main  # noqa: E402

try:
    raise SystemExit(main())
finally:
    _release_single_instance_mutex(_MUTEX)
