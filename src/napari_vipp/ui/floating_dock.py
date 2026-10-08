"""Native owner repair for a floating dock, without changing its Qt lifetime."""

from __future__ import annotations

import ctypes
import sys

from qtpy.QtWidgets import QApplication, QDockWidget

_GW_OWNER = 4
_GWLP_HWNDPARENT = -8
_GWL_STYLE = -16
_WS_CHILD = 0x40000000


class _WindowsOwnerApi:
    """Use pointer-sized HWND/LONG_PTR declarations on 64-bit Windows."""

    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.IsWindow.argtypes = [ctypes.c_void_p]
        self._user32.IsWindow.restype = ctypes.c_int
        self._user32.GetWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self._user32.GetWindow.restype = ctypes.c_void_p
        if ctypes.sizeof(ctypes.c_void_p) == 8:
            self._get_long = self._user32.GetWindowLongPtrW
            self._set_long = self._user32.SetWindowLongPtrW
            long_type = ctypes.c_ssize_t
        else:
            # SetWindowLongPtr is a C macro for SetWindowLong on 32-bit Windows.
            self._get_long = self._user32.GetWindowLongW
            self._set_long = self._user32.SetWindowLongW
            long_type = ctypes.c_long
        self._get_long.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._get_long.restype = long_type
        self._set_long.argtypes = [ctypes.c_void_p, ctypes.c_int, long_type]
        self._set_long.restype = long_type

    def is_window(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindow(hwnd))

    def is_child_window(self, hwnd: int) -> bool:
        ctypes.set_last_error(0)
        style = self._get_long(hwnd, _GWL_STYLE)
        error = ctypes.get_last_error()
        if not style and error:
            raise ctypes.WinError(error)
        return bool(style & _WS_CHILD)

    def owner(self, hwnd: int) -> int:
        return int(self._user32.GetWindow(hwnd, _GW_OWNER) or 0)

    def clear_owner(self, hwnd: int) -> None:
        # A zero return can mean either success or failure. Clear last error
        # first, as required by the Win32 SetWindowLongPtr contract.
        ctypes.set_last_error(0)
        previous_owner = self._set_long(hwnd, _GWLP_HWNDPARENT, 0)
        error = ctypes.get_last_error()
        if not previous_owner and error:
            raise ctypes.WinError(error)


def make_floating_dock_independent(dock: QDockWidget) -> bool:
    """Remove only a floating Windows dock's native ownership by its Qt host.

    The QMainWindow/QObject parent stays intact for docking and destruction.
    Windows owned windows are hidden when their owner is minimized, regardless
    of Qt.Window flags. Clearing GWLP_HWNDPARENT avoids that coupling without
    moving VIPP into another widget or scientific session.

    Return True only when native Windows independence has been verified. Return
    False for docked widgets, other operating systems and non-native Qt plugins
    (including offscreen tests). Native API errors and unexpected ownership are
    raised so the caller can report the failure, not silently claim success.
    """
    if (
        sys.platform != "win32"
        or QApplication.platformName() != "windows"
        or not dock.isFloating()
    ):
        return False

    hwnd = int(dock.winId())
    api = _WindowsOwnerApi()
    if not api.is_window(hwnd):
        raise OSError("The floating VIPP window has no valid native window handle.")
    if api.is_child_window(hwnd):
        raise RuntimeError("Refusing to change the parent of a native child window.")
    owner = api.owner(hwnd)
    if not owner:
        return True

    parent = dock.parentWidget()
    expected_owner = int(parent.window().winId()) if parent is not None else 0
    if not expected_owner or owner != expected_owner:
        raise RuntimeError("The floating VIPP window has an unexpected native owner.")

    api.clear_owner(hwnd)
    if api.owner(hwnd):
        raise OSError("Windows did not remove the floating VIPP window's owner.")
    return True
