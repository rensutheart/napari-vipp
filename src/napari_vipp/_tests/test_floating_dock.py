from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

from napari_vipp.ui import floating_dock


class _FakeOwnerApi:
    def __init__(self, owner, *, valid=True, child=False, retain_owner=False):
        self.current_owner = owner
        self.valid = valid
        self.child = child
        self.retain_owner = retain_owner
        self.cleared = []

    def is_window(self, _hwnd):
        return self.valid

    def is_child_window(self, _hwnd):
        return self.child

    def owner(self, _hwnd):
        return self.current_owner

    def clear_owner(self, hwnd):
        self.cleared.append(hwnd)
        if not self.retain_owner:
            self.current_owner = 0


@pytest.fixture
def floating_window(qtbot):
    window = QMainWindow()
    qtbot.addWidget(window)
    dock = QDockWidget("VIPP", window)
    dock.setWidget(QWidget())
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)
    dock.setFloating(True)
    return window, dock


def _mock_native_windows(monkeypatch, api):
    monkeypatch.setattr(floating_dock.sys, "platform", "win32")
    monkeypatch.setattr(QApplication, "platformName", lambda: "windows")
    monkeypatch.setattr(floating_dock, "_WindowsOwnerApi", lambda: api)


def test_independence_preserves_qt_parent_and_dock_membership(
    floating_window, monkeypatch
):
    window, dock = floating_window
    api = _FakeOwnerApi(int(window.winId()))
    _mock_native_windows(monkeypatch, api)
    original_widget = dock.widget()

    assert floating_dock.make_floating_dock_independent(dock)
    assert api.cleared == [int(dock.winId())]
    assert dock.parentWidget() is window
    assert dock.widget() is original_widget
    assert window.dockWidgetArea(dock) == Qt.BottomDockWidgetArea
    dock.setFloating(False)
    assert dock.parentWidget() is window
    assert dock.widget() is original_widget
    assert window.dockWidgetArea(dock) == Qt.BottomDockWidgetArea


def test_independence_is_idempotent(floating_window, monkeypatch):
    window, dock = floating_window
    api = _FakeOwnerApi(int(window.winId()))
    _mock_native_windows(monkeypatch, api)

    assert floating_dock.make_floating_dock_independent(dock)
    assert floating_dock.make_floating_dock_independent(dock)
    assert len(api.cleared) == 1


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_other_operating_systems_do_not_load_win32(
    floating_window, monkeypatch, platform
):
    _window, dock = floating_window
    monkeypatch.setattr(floating_dock.sys, "platform", platform)
    monkeypatch.setattr(
        floating_dock, "_WindowsOwnerApi", lambda: pytest.fail("Win32 API loaded")
    )
    assert not floating_dock.make_floating_dock_independent(dock)


@pytest.mark.parametrize("plugin", ["offscreen", "minimal"])
def test_non_native_qt_plugins_do_not_treat_win_id_as_hwnd(
    floating_window, monkeypatch, plugin
):
    _window, dock = floating_window
    monkeypatch.setattr(floating_dock.sys, "platform", "win32")
    monkeypatch.setattr(QApplication, "platformName", lambda: plugin)
    monkeypatch.setattr(
        floating_dock, "_WindowsOwnerApi", lambda: pytest.fail("Win32 API loaded")
    )
    assert not floating_dock.make_floating_dock_independent(dock)


def test_docked_widget_does_not_load_win32(floating_window, monkeypatch):
    _window, dock = floating_window
    dock.setFloating(False)
    monkeypatch.setattr(floating_dock.sys, "platform", "win32")
    monkeypatch.setattr(QApplication, "platformName", lambda: "windows")
    monkeypatch.setattr(
        floating_dock, "_WindowsOwnerApi", lambda: pytest.fail("Win32 API loaded")
    )
    assert not floating_dock.make_floating_dock_independent(dock)


@pytest.mark.parametrize(
    "api_args,message",
    [
        ({"valid": False}, "no valid native window handle"),
        ({"child": True}, "native child window"),
    ],
)
def test_invalid_or_child_native_window_is_not_mutated(
    floating_window, monkeypatch, api_args, message
):
    window, dock = floating_window
    api = _FakeOwnerApi(int(window.winId()), **api_args)
    _mock_native_windows(monkeypatch, api)
    with pytest.raises((OSError, RuntimeError), match=message):
        floating_dock.make_floating_dock_independent(dock)
    assert not api.cleared


def test_unexpected_native_owner_is_not_mutated(floating_window, monkeypatch):
    window, dock = floating_window
    api = _FakeOwnerApi(int(window.winId()) + 1)
    _mock_native_windows(monkeypatch, api)
    with pytest.raises(RuntimeError, match="unexpected native owner"):
        floating_dock.make_floating_dock_independent(dock)
    assert not api.cleared


def test_failed_owner_removal_is_not_reported_as_success(
    floating_window, monkeypatch
):
    window, dock = floating_window
    api = _FakeOwnerApi(int(window.winId()), retain_owner=True)
    _mock_native_windows(monkeypatch, api)
    with pytest.raises(OSError, match="did not remove"):
        floating_dock.make_floating_dock_independent(dock)
    assert api.cleared == [int(dock.winId())]


class _FakeFunction:
    def __init__(self, result=0):
        self.result = result
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


@pytest.mark.parametrize("pointer_size", [4, 8])
def test_win32_bindings_are_pointer_safe(monkeypatch, pointer_size):
    user32 = SimpleNamespace(
        IsWindow=_FakeFunction(1),
        GetWindow=_FakeFunction(0x123456789AB if pointer_size == 8 else 0x12345),
        GetWindowLongPtrW=_FakeFunction(),
        SetWindowLongPtrW=_FakeFunction(),
        GetWindowLongW=_FakeFunction(),
        SetWindowLongW=_FakeFunction(),
    )
    monkeypatch.setattr(
        floating_dock.ctypes, "WinDLL", lambda *_args, **_kwargs: user32, raising=False
    )
    native_sizeof = ctypes.sizeof
    monkeypatch.setattr(
        floating_dock.ctypes,
        "sizeof",
        lambda kind: pointer_size if kind is ctypes.c_void_p else native_sizeof(kind),
    )
    api = floating_dock._WindowsOwnerApi()
    get_long = (
        user32.GetWindowLongPtrW if pointer_size == 8 else user32.GetWindowLongW
    )
    set_long = (
        user32.SetWindowLongPtrW if pointer_size == 8 else user32.SetWindowLongW
    )
    expected_long = ctypes.c_ssize_t if pointer_size == 8 else ctypes.c_long
    assert get_long.argtypes == [ctypes.c_void_p, ctypes.c_int]
    assert get_long.restype is expected_long
    assert set_long.argtypes == [ctypes.c_void_p, ctypes.c_int, expected_long]
    assert set_long.restype is expected_long
    assert user32.GetWindow.restype is ctypes.c_void_p
    assert api.owner(0x123456789AB) == user32.GetWindow.result
    assert user32.GetWindow.calls == [(0x123456789AB, floating_dock._GW_OWNER)]


@pytest.mark.parametrize("previous_owner,error", [(0, 0), (123, 0), (0, 5)])
def test_clear_owner_distinguishes_zero_success_from_win32_failure(
    monkeypatch, previous_owner, error
):
    api = object.__new__(floating_dock._WindowsOwnerApi)
    api._set_long = _FakeFunction(previous_owner)
    last_error_resets = []
    monkeypatch.setattr(
        floating_dock.ctypes,
        "set_last_error",
        last_error_resets.append,
        raising=False,
    )
    monkeypatch.setattr(
        floating_dock.ctypes, "get_last_error", lambda: error, raising=False
    )
    monkeypatch.setattr(
        floating_dock.ctypes,
        "WinError",
        lambda code: OSError(code, "native owner failure"),
        raising=False,
    )
    if error:
        with pytest.raises(OSError, match="native owner failure"):
            api.clear_owner(0x123456789AB)
    else:
        api.clear_owner(0x123456789AB)
    assert last_error_resets == [0]
    assert api._set_long.calls == [(0x123456789AB, floating_dock._GWLP_HWNDPARENT, 0)]
