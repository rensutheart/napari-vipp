"""Qualify native Windows floating docks using only this process's Qt windows.

Run in a fresh process with VIPP installed and QT_QPA_PLATFORM=windows. The
probe creates temporary windows and controls only handles obtained from those
widgets; it never enumerates, finds or changes an existing application window.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import sys
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path

import qtpy
from qtpy.QtCore import Qt, qVersion
from qtpy.QtTest import QTest
from qtpy.QtWidgets import QApplication, QDockWidget, QMainWindow, QWidget

from napari_vipp.ui import floating_dock

_CASES = (
    "hidden",
    "normal",
    "minimized",
    "maximized",
    "minimized_from_maximized",
    "unowned_normal_toolwindow",
    "unowned_minimized_from_maximized_toolwindow",
)
_WINDOW_FLAGS = (
    Qt.Window
    | Qt.WindowTitleHint
    | Qt.WindowSystemMenuHint
    | Qt.WindowMinimizeButtonHint
    | Qt.WindowMaximizeButtonHint
    | Qt.WindowCloseButtonHint
)


class _WindowPlacement(ctypes.Structure):
    _fields_ = [
        ("length", wintypes.UINT),
        ("flags", wintypes.UINT),
        ("showCmd", wintypes.UINT),
        ("ptMinPosition", wintypes.POINT),
        ("ptMaxPosition", wintypes.POINT),
        ("rcNormalPosition", wintypes.RECT),
    ]


def _flush(app: QApplication) -> None:
    app.processEvents()
    QTest.qWait(40)
    app.processEvents()


def _snapshot(api: floating_dock._WindowsOwnerApi, dock: QDockWidget) -> dict:
    hwnd = int(dock.winId())
    placement = _WindowPlacement(length=ctypes.sizeof(_WindowPlacement))
    ctypes.set_last_error(0)
    if not api._user32.GetWindowPlacement(hwnd, ctypes.byref(placement)):
        raise ctypes.WinError(ctypes.get_last_error())
    normal = placement.rcNormalPosition
    owner = api.owner(hwnd)
    parent = dock.parentWidget()
    expected_owner = int(parent.window().winId()) if parent is not None else 0
    # Raw HWND values are ephemeral and unnecessary in the durable report.
    return {
        "owner_present": bool(owner),
        "owner_matches_qt_host": bool(owner and owner == expected_owner),
        "exstyle": api.extended_style(hwnd),
        "child": api.is_child_window(hwnd),
        "visible": api.is_visible(hwnd),
        "minimized": api.is_minimized(hwnd),
        "maximized": api.is_maximized(hwnd),
        "qt_visible": dock.isVisible(),
        "qt_state": dock.windowState().value,
        "qt_flags": dock.windowFlags().value,
        "placement_flags": placement.flags,
        "placement_show_command": placement.showCmd,
        "normal_rectangle": [normal.left, normal.top, normal.right, normal.bottom],
    }


def _assert_repaired(
    window: QMainWindow,
    dock: QDockWidget,
    before: dict,
    after: dict,
) -> None:
    assert not after["owner_present"]
    assert after["exstyle"] & floating_dock._WS_EX_APPWINDOW
    assert not after["exstyle"] & floating_dock._WS_EX_TOOLWINDOW
    assert not after["child"]
    assert dock.parentWidget() is window
    assert window.dockWidgetArea(dock) == Qt.BottomDockWidgetArea
    for key in (
        "visible",
        "minimized",
        "maximized",
        "qt_visible",
        "qt_state",
        "qt_flags",
        "normal_rectangle",
    ):
        assert before[key] == after[key], (key, before[key], after[key])
    # WPF_RESTORETOMAXIMIZED must survive minimizing a maximized window.
    assert before["placement_flags"] & 2 == after["placement_flags"] & 2


def _qualify_case(
    app: QApplication,
    api: floating_dock._WindowsOwnerApi,
    state: str,
    record: dict,
) -> None:
    display_state = state.removeprefix("unowned_").removesuffix("_toolwindow")
    window = QMainWindow()
    window.setWindowTitle("VIPP task-owned taskbar qualification")
    window.setGeometry(70, 70, 400, 280)
    dock = QDockWidget("VIPP task-owned floating qualification", window)
    content = QWidget()
    dock.setWidget(content)
    window.addDockWidget(Qt.BottomDockWidgetArea, dock)
    try:
        dock.setFloating(True)
        dock.setWindowFlags(_WINDOW_FLAGS)
        dock.setGeometry(500, 180, 560, 360)
        window.show()
        dock.show()
        _flush(app)
        if display_state == "hidden":
            dock.hide()
        elif display_state == "minimized":
            dock.showMinimized()
        elif display_state == "maximized":
            dock.showMaximized()
        elif display_state == "minimized_from_maximized":
            dock.showMaximized()
            _flush(app)
            dock.showMinimized()
        _flush(app)
        if state.startswith("unowned_"):
            # Reproduce missing taskbar eligibility using this probe's own HWND.
            hwnd = int(dock.winId())
            api.clear_owner(hwnd)
            api.set_extended_style(
                hwnd,
                (api.extended_style(hwnd) | floating_dock._WS_EX_TOOLWINDOW)
                & ~floating_dock._WS_EX_APPWINDOW,
            )
            api.refresh_frame(hwnd)
        before = _snapshot(api, dock)
        record["before"] = before
        if state.startswith("unowned_"):
            assert not before["owner_present"]
            assert before["exstyle"] & floating_dock._WS_EX_TOOLWINDOW
        else:
            assert before["owner_matches_qt_host"]
        assert not before["exstyle"] & floating_dock._WS_EX_APPWINDOW
        assert floating_dock.make_floating_dock_independent(dock)
        _flush(app)
        after = _snapshot(api, dock)
        record["after"] = after
        _assert_repaired(window, dock, before, after)
        assert dock.widget() is content
        assert floating_dock.make_floating_dock_independent(dock)
        _flush(app)
        assert _snapshot(api, dock) == after, "idempotent repair changed native state"
        record["idempotent"] = True
        if display_state != "hidden":
            window.showMinimized()
            _flush(app)
            parent_minimized = _snapshot(api, dock)
            assert parent_minimized["visible"]
            assert parent_minimized["minimized"] == after["minimized"]
            record["parent_minimized"] = parent_minimized
            window.showNormal()
            _flush(app)

        dock.showNormal()
        _flush(app)
        dock.setFloating(False)
        _flush(app)
        redocked = _snapshot(api, dock)
        record["redocked"] = redocked
        assert redocked["child"]
        assert not floating_dock.make_floating_dock_independent(dock)
        assert not redocked["exstyle"] & floating_dock._WS_EX_APPWINDOW
        dock.setFloating(True)
        dock.setWindowFlags(_WINDOW_FLAGS)
        dock.show()
        _flush(app)
        assert floating_dock.make_floating_dock_independent(dock)
        _flush(app)
        record["refloated"] = _snapshot(api, dock)
        for _ in range(3):
            dock.showMaximized()
            _flush(app)
            dock.showMinimized()
            _flush(app)
            cycle_before = _snapshot(api, dock)
            assert floating_dock.make_floating_dock_independent(dock)
            _flush(app)
            _assert_repaired(window, dock, cycle_before, _snapshot(api, dock))
            dock.showNormal()
            _flush(app)
        record["minimize_maximize_cycles"] = 3
        record["passed"] = True
    finally:
        window.close()
        window.deleteLater()
        _flush(app)


def _metadata() -> dict:
    return {
        "utc_date": datetime.now(UTC).isoformat(),
        "environment": {
            "platform": sys.platform,
            "windows_version": platform.version(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "pointer_bits": ctypes.sizeof(ctypes.c_void_p) * 8,
            "qt_version": qVersion(),
            "qtpy_version": qtpy.__version__,
            "qt_binding": qtpy.API_NAME,
            "qt_binding_version": (
                getattr(qtpy, "PYQT_VERSION", None)
                or getattr(qtpy, "PYSIDE_VERSION", None)
            ),
            "qt_platform": QApplication.platformName(),
        },
        "floating_dock_source_sha256": hashlib.sha256(
            Path(floating_dock.__file__).read_bytes(),
        ).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".cache"
        / "floating-dock-windows-qualification.json",
        help="Destination JSON report (defaults to the repository's .cache directory).",
    )
    args = parser.parse_args(argv)
    if sys.platform != "win32":
        parser.error("This qualification requires native Windows.")
    configured_platform = os.environ.get("QT_QPA_PLATFORM", "windows")
    if configured_platform != "windows":
        parser.error(
            "Set QT_QPA_PLATFORM=windows; offscreen/minimal cannot qualify Win32."
        )
    if QApplication.instance() is not None:
        parser.error("Run in a fresh process, outside an existing Qt application.")
    app = QApplication([])
    if QApplication.platformName() != "windows":
        parser.error("Qt must use its native Windows platform plugin.")
    api = floating_dock._WindowsOwnerApi()
    api._user32.GetWindowPlacement.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_WindowPlacement),
    ]
    api._user32.GetWindowPlacement.restype = ctypes.c_int
    records = {"schema_version": 1, **_metadata(), "cases": [], "passed": False}
    try:
        for state in _CASES:
            record = {"state": state, "passed": False}
            records["cases"].append(record)
            _qualify_case(app, api, state, record)
        records["passed"] = True
    except Exception as error:
        records["error"] = repr(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(records, separators=(",", ":")) + "\n", encoding="utf-8"
        )
        print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
