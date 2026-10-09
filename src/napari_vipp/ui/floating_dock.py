"""Native taskbar/owner repair without changing a floating dock's Qt lifetime."""

from __future__ import annotations

import ctypes
import sys

from qtpy.QtWidgets import QApplication, QDockWidget

_GW_OWNER = 4
_GWLP_HWNDPARENT = -8
_GWL_STYLE = -16
_GWL_EXSTYLE = -20
_WS_CHILD = 0x40000000
_WS_EX_TOOLWINDOW = 0x00000080
_WS_EX_APPWINDOW = 0x00040000
_SW_HIDE = 0
_SW_SHOWMINNOACTIVE = 7
_SW_SHOWNA = 8
_SWP_NOSIZE = 0x0001
_SWP_NOMOVE = 0x0002
_SWP_NOZORDER = 0x0004
_SWP_NOACTIVATE = 0x0010
_SWP_FRAMECHANGED = 0x0020


class _WindowsOwnerApi:
    """Use pointer-sized HWND/LONG_PTR declarations on 64-bit Windows."""

    def __init__(self) -> None:
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.IsWindow.argtypes = [ctypes.c_void_p]
        self._user32.IsWindow.restype = ctypes.c_int
        self._user32.GetWindow.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        self._user32.GetWindow.restype = ctypes.c_void_p
        for name in ("IsWindowVisible", "IsIconic", "IsZoomed"):
            function = getattr(self._user32, name)
            function.argtypes = [ctypes.c_void_p]
            function.restype = ctypes.c_int
        self._user32.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._user32.ShowWindow.restype = ctypes.c_int
        self._user32.SetWindowPos.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
        ]
        self._user32.SetWindowPos.restype = ctypes.c_int
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
        return bool(self._window_long(hwnd, _GWL_STYLE) & _WS_CHILD)

    def _window_long(self, hwnd: int, index: int) -> int:
        ctypes.set_last_error(0)
        value = self._get_long(hwnd, index)
        error = ctypes.get_last_error()
        if not value and error:
            raise ctypes.WinError(error)
        return int(value)

    def extended_style(self, hwnd: int) -> int:
        return self._window_long(hwnd, _GWL_EXSTYLE)

    def is_visible(self, hwnd: int) -> bool:
        return bool(self._user32.IsWindowVisible(hwnd))

    def is_minimized(self, hwnd: int) -> bool:
        return bool(self._user32.IsIconic(hwnd))

    def is_maximized(self, hwnd: int) -> bool:
        return bool(self._user32.IsZoomed(hwnd))

    def owner(self, hwnd: int) -> int:
        ctypes.set_last_error(0)
        owner = self._user32.GetWindow(hwnd, _GW_OWNER)
        error = ctypes.get_last_error()
        if not owner and error:
            raise ctypes.WinError(error)
        return int(owner or 0)

    def clear_owner(self, hwnd: int) -> None:
        self._set_window_long(hwnd, _GWLP_HWNDPARENT, 0)

    def _set_window_long(self, hwnd: int, index: int, value: int) -> None:
        # A zero return can mean either success or failure. Clear last error
        # first, as required by the Win32 SetWindowLongPtr contract.
        ctypes.set_last_error(0)
        previous = self._set_long(hwnd, index, value)
        error = ctypes.get_last_error()
        if not previous and error:
            raise ctypes.WinError(error)

    def set_extended_style(self, hwnd: int, style: int) -> None:
        self._set_window_long(hwnd, _GWL_EXSTYLE, style)

    def refresh_frame(self, hwnd: int) -> None:
        flags = (
            _SWP_NOSIZE
            | _SWP_NOMOVE
            | _SWP_NOZORDER
            | _SWP_NOACTIVATE
            | _SWP_FRAMECHANGED
        )
        ctypes.set_last_error(0)
        if not self._user32.SetWindowPos(hwnd, None, 0, 0, 0, 0, flags):
            raise ctypes.WinError(ctypes.get_last_error())

    def hide(self, hwnd: int) -> None:
        # ShowWindow returns the old visibility, not success/failure.
        self._user32.ShowWindow(hwnd, _SW_HIDE)
        if self.is_visible(hwnd):
            raise OSError("Windows did not hide the floating VIPP window.")

    def show_without_activation(self, hwnd: int, *, minimized: bool) -> None:
        command = _SW_SHOWMINNOACTIVE if minimized else _SW_SHOWNA
        self._user32.ShowWindow(hwnd, command)
        if not self.is_visible(hwnd):
            raise OSError("Windows did not show the floating VIPP window again.")


def make_floating_dock_independent(dock: QDockWidget) -> bool:
    """Give a floating Windows dock independent owner and taskbar behavior.

    The QMainWindow/QObject parent stays intact for docking and destruction.
    Windows owned windows are hidden when their owner is minimized, regardless
    of Qt.Window flags. Clearing GWLP_HWNDPARENT avoids that coupling; forcing
    APPWINDOW and removing TOOLWINDOW makes a minimized editor discoverable.
    A visible dock is briefly hidden for the Shell to refresh its registration,
    then shown in its original state without activation. Qt parentage, docking
    membership and scientific state remain intact.

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
    if owner:
        parent = dock.parentWidget()
        expected_owner = int(parent.window().winId()) if parent is not None else 0
        if not expected_owner or owner != expected_owner:
            raise RuntimeError(
                "The floating VIPP window has an unexpected native owner."
            )
    style = api.extended_style(hwnd)
    desired_style = (style | _WS_EX_APPWINDOW) & ~_WS_EX_TOOLWINDOW
    if not owner and style == desired_style:
        return True

    visible = api.is_visible(hwnd)
    minimized = api.is_minimized(hwnd)
    maximized = api.is_maximized(hwnd)
    try:
        if visible:
            # The Shell must see hide/change/show to reconsider taskbar
            # registration, including a dock first shown with a native owner.
            api.hide(hwnd)
        if owner:
            api.clear_owner(hwnd)
        if style != desired_style:
            api.set_extended_style(hwnd, desired_style)
            api.refresh_frame(hwnd)
    finally:
        # An API failure must not strand a previously visible editor hidden.
        if visible:
            api.show_without_activation(hwnd, minimized=minimized)

    if api.owner(hwnd):
        raise OSError("Windows did not remove the floating VIPP window's owner.")
    repaired_style = api.extended_style(hwnd)
    if not repaired_style & _WS_EX_APPWINDOW or repaired_style & _WS_EX_TOOLWINDOW:
        raise OSError("Windows did not register the floating VIPP taskbar window.")
    if (api.is_visible(hwnd), api.is_minimized(hwnd), api.is_maximized(hwnd)) != (
        visible,
        minimized,
        maximized,
    ):
        raise OSError("Windows changed the floating VIPP window's display state.")
    return True
