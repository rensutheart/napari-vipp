from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

from napari_vipp.ui import floating_dock


class _FakeOwnerApi:
    def __init__(
        self,
        owner,
        *,
        valid=True,
        child=False,
        retain_owner=False,
        style=0,
        visible=False,
        minimized=False,
        maximized=False,
        retain_style=False,
        fail_at=None,
        change_state_on_show=False,
    ):
        self.current_owner = owner
        self.valid = valid
        self.child = child
        self.retain_owner = retain_owner
        self.style = style
        self.visible = visible
        self.minimized = minimized
        self.maximized = maximized
        self.retain_style = retain_style
        self.fail_at = fail_at
        self.change_state_on_show = change_state_on_show
        self.cleared = []
        self.mutations = []

    def is_window(self, _hwnd):
        return self.valid

    def is_child_window(self, _hwnd):
        return self.child

    def owner(self, _hwnd):
        return self.current_owner

    def clear_owner(self, hwnd):
        self._mutate("clear_owner", hwnd)
        self.cleared.append(hwnd)
        if not self.retain_owner:
            self.current_owner = 0

    def extended_style(self, _hwnd):
        return self.style

    def is_visible(self, _hwnd):
        return self.visible

    def is_minimized(self, _hwnd):
        return self.minimized

    def is_maximized(self, _hwnd):
        return self.maximized

    def _mutate(self, name, *args):
        self.mutations.append((name, *args))
        if self.fail_at == name:
            raise OSError(f"native {name} failure")

    def set_extended_style(self, hwnd, style):
        self._mutate("set_extended_style", hwnd, style)
        if not self.retain_style:
            self.style = style

    def refresh_frame(self, hwnd):
        self._mutate("refresh_frame", hwnd)

    def hide(self, hwnd):
        self._mutate("hide", hwnd)
        self.visible = False

    def show_without_activation(self, hwnd, *, minimized):
        self._mutate("show_without_activation", hwnd, minimized)
        self.visible = True
        if self.change_state_on_show:
            self.minimized = False
            self.maximized = False


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
    first_mutations = api.mutations.copy()
    assert floating_dock.make_floating_dock_independent(dock)
    assert len(api.cleared) == 1
    assert api.mutations == first_mutations


@pytest.mark.parametrize("owner_present", [False, True])
@pytest.mark.parametrize("tool_window", [False, True])
def test_taskbar_style_is_repaired_even_without_an_owner(
    floating_window, monkeypatch, owner_present, tool_window
):
    window, dock = floating_window
    unrelated_bits = 0x00000100 | 0x00080000
    style = unrelated_bits | (floating_dock._WS_EX_TOOLWINDOW if tool_window else 0)
    api = _FakeOwnerApi(
        int(window.winId()) if owner_present else 0,
        style=style,
        visible=True,
    )
    _mock_native_windows(monkeypatch, api)

    assert floating_dock.make_floating_dock_independent(dock)

    hwnd = int(dock.winId())
    desired_style = unrelated_bits | floating_dock._WS_EX_APPWINDOW
    assert api.style == desired_style
    names = [entry[0] for entry in api.mutations]
    expected = ["hide"]
    if owner_present:
        expected.append("clear_owner")
    expected += ["set_extended_style", "refresh_frame", "show_without_activation"]
    assert names == expected
    assert ("set_extended_style", hwnd, desired_style) in api.mutations


def test_unowned_taskbar_window_is_a_complete_no_op(floating_window, monkeypatch):
    _window, dock = floating_window
    api = _FakeOwnerApi(
        0,
        style=floating_dock._WS_EX_APPWINDOW | 0x100,
        visible=True,
        minimized=True,
    )
    _mock_native_windows(monkeypatch, api)

    assert floating_dock.make_floating_dock_independent(dock)

    assert api.mutations == []
    assert api.visible and api.minimized


def test_owned_taskbar_window_refreshes_registration_without_rewriting_style(
    floating_window, monkeypatch
):
    window, dock = floating_window
    api = _FakeOwnerApi(
        int(window.winId()),
        style=floating_dock._WS_EX_APPWINDOW,
        visible=True,
    )
    _mock_native_windows(monkeypatch, api)

    assert floating_dock.make_floating_dock_independent(dock)

    assert [entry[0] for entry in api.mutations] == [
        "hide",
        "clear_owner",
        "show_without_activation",
    ]


@pytest.mark.parametrize("visible", [False, True])
@pytest.mark.parametrize(
    "minimized,maximized",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_taskbar_repair_preserves_visibility_and_window_state(
    floating_window, monkeypatch, visible, minimized, maximized
):
    window, dock = floating_window
    state = Qt.WindowNoState
    if minimized:
        state |= Qt.WindowMinimized
    if maximized:
        state |= Qt.WindowMaximized
    dock.setWindowState(state)
    dock.setVisible(visible)
    qt_snapshot = (dock.windowState(), dock.isVisible(), dock.windowFlags())
    api = _FakeOwnerApi(
        int(window.winId()),
        style=floating_dock._WS_EX_TOOLWINDOW,
        visible=visible,
        minimized=minimized,
        maximized=maximized,
    )
    _mock_native_windows(monkeypatch, api)

    assert floating_dock.make_floating_dock_independent(dock)

    assert (api.visible, api.minimized, api.maximized) == (
        visible,
        minimized,
        maximized,
    )
    assert (dock.windowState(), dock.isVisible(), dock.windowFlags()) == qt_snapshot
    assert dock.parentWidget() is window
    assert window.dockWidgetArea(dock) == Qt.BottomDockWidgetArea
    show_calls = [
        entry for entry in api.mutations if entry[0] == "show_without_activation"
    ]
    assert show_calls == (
        [
            ("show_without_activation", int(dock.winId()), minimized),
        ]
        if visible
        else []
    )


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
    assert not api.mutations


def test_unexpected_native_owner_is_not_mutated(floating_window, monkeypatch):
    window, dock = floating_window
    api = _FakeOwnerApi(int(window.winId()) + 1, visible=True)
    _mock_native_windows(monkeypatch, api)
    with pytest.raises(RuntimeError, match="unexpected native owner"):
        floating_dock.make_floating_dock_independent(dock)
    assert not api.mutations


def test_failed_owner_removal_is_not_reported_as_success(floating_window, monkeypatch):
    window, dock = floating_window
    api = _FakeOwnerApi(int(window.winId()), retain_owner=True)
    _mock_native_windows(monkeypatch, api)
    with pytest.raises(OSError, match="did not remove"):
        floating_dock.make_floating_dock_independent(dock)
    assert api.cleared == [int(dock.winId())]


def test_failed_style_repair_is_not_reported_as_success(floating_window, monkeypatch):
    _window, dock = floating_window
    api = _FakeOwnerApi(0, visible=True, retain_style=True)
    _mock_native_windows(monkeypatch, api)
    with pytest.raises(OSError, match="did not register"):
        floating_dock.make_floating_dock_independent(dock)
    assert api.visible


@pytest.mark.parametrize(
    "failure", ["clear_owner", "set_extended_style", "refresh_frame"]
)
def test_native_failure_restores_a_previously_visible_window(
    floating_window, monkeypatch, failure
):
    window, dock = floating_window
    api = _FakeOwnerApi(
        int(window.winId()),
        visible=True,
        minimized=True,
        fail_at=failure,
    )
    _mock_native_windows(monkeypatch, api)
    with pytest.raises(OSError, match=f"native {failure} failure"):
        floating_dock.make_floating_dock_independent(dock)
    assert api.visible and api.minimized
    assert api.mutations[-1] == (
        "show_without_activation",
        int(dock.winId()),
        True,
    )


def test_native_display_state_change_is_not_reported_as_success(
    floating_window, monkeypatch
):
    _window, dock = floating_window
    api = _FakeOwnerApi(0, visible=True, maximized=True, change_state_on_show=True)
    _mock_native_windows(monkeypatch, api)
    with pytest.raises(OSError, match="display state"):
        floating_dock.make_floating_dock_independent(dock)


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
        IsWindowVisible=_FakeFunction(1),
        IsIconic=_FakeFunction(),
        IsZoomed=_FakeFunction(),
        ShowWindow=_FakeFunction(),
        SetWindowPos=_FakeFunction(1),
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
    get_long = user32.GetWindowLongPtrW if pointer_size == 8 else user32.GetWindowLongW
    set_long = user32.SetWindowLongPtrW if pointer_size == 8 else user32.SetWindowLongW
    expected_long = ctypes.c_ssize_t if pointer_size == 8 else ctypes.c_long
    assert get_long.argtypes == [ctypes.c_void_p, ctypes.c_int]
    assert get_long.restype is expected_long
    assert set_long.argtypes == [ctypes.c_void_p, ctypes.c_int, expected_long]
    assert set_long.restype is expected_long
    assert user32.GetWindow.restype is ctypes.c_void_p
    assert api.owner(0x123456789AB) == user32.GetWindow.result
    assert user32.GetWindow.calls == [(0x123456789AB, floating_dock._GW_OWNER)]
    for name in ("IsWindow", "IsWindowVisible", "IsIconic", "IsZoomed"):
        function = getattr(user32, name)
        assert function.argtypes == [ctypes.c_void_p]
        assert function.restype is ctypes.c_int
    assert user32.ShowWindow.argtypes == [ctypes.c_void_p, ctypes.c_int]
    assert user32.SetWindowPos.argtypes == [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint,
    ]
    assert user32.ShowWindow.restype is ctypes.c_int
    assert user32.SetWindowPos.restype is ctypes.c_int


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


@pytest.mark.parametrize("value,error", [(0, 0), (0x100, 0), (0, 5)])
def test_style_read_distinguishes_zero_style_from_native_failure(
    monkeypatch, value, error
):
    api = object.__new__(floating_dock._WindowsOwnerApi)
    api._get_long = _FakeFunction(value)
    resets = []
    monkeypatch.setattr(
        floating_dock.ctypes, "set_last_error", resets.append, raising=False
    )
    monkeypatch.setattr(
        floating_dock.ctypes, "get_last_error", lambda: error, raising=False
    )
    monkeypatch.setattr(
        floating_dock.ctypes,
        "WinError",
        lambda code: OSError(code, "style failure"),
        raising=False,
    )
    if error:
        with pytest.raises(OSError, match="style failure"):
            api.extended_style(0x123456789AB)
    else:
        assert api.extended_style(0x123456789AB) == value
    assert resets == [0]
    assert api._get_long.calls == [(0x123456789AB, floating_dock._GWL_EXSTYLE)]


@pytest.mark.parametrize("owner,error", [(0, 0), (0x123456789AB, 0), (0, 5)])
def test_owner_read_distinguishes_unowned_window_from_native_failure(
    monkeypatch, owner, error
):
    api = object.__new__(floating_dock._WindowsOwnerApi)
    api._user32 = SimpleNamespace(GetWindow=_FakeFunction(owner))
    resets = []
    monkeypatch.setattr(
        floating_dock.ctypes, "set_last_error", resets.append, raising=False
    )
    monkeypatch.setattr(
        floating_dock.ctypes, "get_last_error", lambda: error, raising=False
    )
    monkeypatch.setattr(
        floating_dock.ctypes,
        "WinError",
        lambda code: OSError(code, "owner failure"),
        raising=False,
    )
    if error:
        with pytest.raises(OSError, match="owner failure"):
            api.owner(0x123456789AB)
    else:
        assert api.owner(0x123456789AB) == owner
    assert resets == [0]


@pytest.mark.parametrize("result,error", [(1, 0), (0, 5)])
def test_frame_refresh_reports_native_failure_without_activation(
    monkeypatch, result, error
):
    api = object.__new__(floating_dock._WindowsOwnerApi)
    api._user32 = SimpleNamespace(SetWindowPos=_FakeFunction(result))
    monkeypatch.setattr(
        floating_dock.ctypes, "set_last_error", lambda _value: None, raising=False
    )
    monkeypatch.setattr(
        floating_dock.ctypes, "get_last_error", lambda: error, raising=False
    )
    monkeypatch.setattr(
        floating_dock.ctypes,
        "WinError",
        lambda code: OSError(code, "frame failure"),
        raising=False,
    )
    if not result:
        with pytest.raises(OSError, match="frame failure"):
            api.refresh_frame(0x123456789AB)
    else:
        api.refresh_frame(0x123456789AB)
    assert api._user32.SetWindowPos.calls == [
        (
            0x123456789AB,
            None,
            0,
            0,
            0,
            0,
            floating_dock._SWP_NOSIZE
            | floating_dock._SWP_NOMOVE
            | floating_dock._SWP_NOZORDER
            | floating_dock._SWP_NOACTIVATE
            | floating_dock._SWP_FRAMECHANGED,
        )
    ]


@pytest.mark.parametrize(
    "minimized,command",
    [
        (False, floating_dock._SW_SHOWNA),
        (True, floating_dock._SW_SHOWMINNOACTIVE),
    ],
)
@pytest.mark.parametrize("old_visibility", [0, 1])
def test_show_preserves_current_state_without_activation(
    minimized, command, old_visibility
):
    api = object.__new__(floating_dock._WindowsOwnerApi)
    api._user32 = SimpleNamespace(
        ShowWindow=_FakeFunction(old_visibility),
        IsWindowVisible=_FakeFunction(1),
    )
    api.show_without_activation(0x123456789AB, minimized=minimized)
    assert api._user32.ShowWindow.calls == [(0x123456789AB, command)]


@pytest.mark.parametrize("operation", ["hide", "show_without_activation"])
def test_show_window_checks_resulting_visibility_instead_of_previous_visibility(
    operation,
):
    api = object.__new__(floating_dock._WindowsOwnerApi)
    api._user32 = SimpleNamespace(
        ShowWindow=_FakeFunction(1),
        IsWindowVisible=_FakeFunction(1 if operation == "hide" else 0),
    )
    with pytest.raises(OSError, match="Windows did not"):
        if operation == "hide":
            api.hide(0x123456789AB)
        else:
            api.show_without_activation(0x123456789AB, minimized=False)
