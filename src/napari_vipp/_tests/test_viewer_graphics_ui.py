"""Viewer graphics settings: no native GL window or workflow execution."""

from __future__ import annotations

import ctypes
from types import SimpleNamespace

import pytest
from qtpy.QtCore import QEvent, QPoint, Qt
from qtpy.QtGui import QHelpEvent, QPalette
from qtpy.QtWidgets import QApplication, QLabel, QToolTip, QWidgetAction

from napari_vipp import _widget as widget_module
from napari_vipp._tests.test_widget import _Viewer
from napari_vipp.ui import viewer_graphics_status as graphics_status
from napari_vipp.ui.viewer_graphics import ViewerGraphicsMode
from napari_vipp.ui.viewer_graphics_status import ViewerGraphicsStatus


def _action(menu, text):
    return next(action for action in menu.actions() if action.text() == text)


def _graphics_rows(menu):
    actions = menu.actions()
    start = actions.index(_action(menu, "Viewer graphics"))
    rows = []
    for action in actions[start:]:
        if action.isSeparator():
            break
        rows.append(action.text())
    return rows


def _active_renderer_action(menu, text):
    action = _action(menu, text)
    assert isinstance(action, QWidgetAction)
    assert not action.isEnabled() and not action.isCheckable()
    assert action.menu() is None and "\n" not in action.text()
    label = action.defaultWidget()
    assert isinstance(label, QLabel)
    assert label.objectName() == "VippViewerGraphicsStatus"
    assert label.text() == action.text() and label.textFormat() == Qt.PlainText
    assert label.isEnabled() and not label.wordWrap()
    assert label.focusPolicy() == Qt.NoFocus
    assert label.palette().currentColorGroup() != QPalette.Disabled
    assert menu.toolTipsVisible()
    return action


@pytest.fixture
def widget(qtbot):
    editor = widget_module.VippWidget(_Viewer(), defer_initial_run=True)
    qtbot.addWidget(editor)
    return editor


def test_gear_separates_active_renderer_from_saved_next_launch_mode(
    widget, monkeypatch
):
    current = ViewerGraphicsMode.AUTO
    writes = []

    def save(mode):
        nonlocal current
        writes.append(mode)
        current = mode

    def forbidden(*_args, **_kwargs):
        pytest.fail("Viewer graphics must not request workflow computation")

    monkeypatch.setattr(widget, "run_pipeline", forbidden)
    monkeypatch.setattr(widget_module, "load_viewer_graphics_mode", lambda: current)
    monkeypatch.setattr(widget_module, "save_viewer_graphics_mode", save)
    monkeypatch.setattr(
        widget_module,
        "query_viewer_graphics",
        lambda viewer: ViewerGraphicsStatus("llvmpipe (LLVM 20)", "Mesa", "4.5"),
    )
    monkeypatch.setattr(
        widget_module,
        "viewer_graphics_availability",
        lambda: SimpleNamespace(can_configure=True, cpu_available=True),
    )
    monkeypatch.setattr(
        widget_module,
        "viewer_graphics_startup",
        lambda: SimpleNamespace(requested=ViewerGraphicsMode.AUTO, reason=""),
    )
    before = (
        widget._compute_mode,
        widget._thumbnail_statistics_policy,
        widget._workflow_tabs.current.session_id,
    )
    widget._populate_settings_toolbar_menu()
    menu = widget.settings_menu
    headings = [
        action.text()
        for action in menu.actions()
        if action.text()
        in {
            "Compute",
            "Viewer graphics",
            "Workflow settings",
        }
    ]
    assert headings == ["Compute", "Viewer graphics", "Workflow settings"]
    assert _graphics_rows(menu) == [
        "Viewer graphics",
        "Active: CPU (software)",
        "Next launch: Auto",
    ]
    active = _active_renderer_action(menu, "Active: CPU (software)")
    for text in (
        "llvmpipe (LLVM 20)",
        "Mesa",
        "4.5",
        "workflow compute",
        "thumbnail statistics",
    ):
        assert text in active.toolTip()

    submenu = _action(menu, "Next launch: Auto").menu()
    actions = {action.data(): action for action in submenu.actions()}
    assert set(actions) == {mode.value for mode in ViewerGraphicsMode}
    assert [action.text() for action in submenu.actions()] == ["Auto", "GPU", "CPU"]
    assert submenu.toolTipsVisible()
    for mode in ViewerGraphicsMode:
        action = actions[mode.value]
        assert action.isCheckable() and action.toolTip() != action.text()
    assert [action.data() for action in submenu.actions() if action.isChecked()] == [
        ViewerGraphicsMode.AUTO.value
    ]
    actions[ViewerGraphicsMode.GPU.value].trigger()
    assert writes == [ViewerGraphicsMode.GPU]
    assert (
        widget.status_label.text()
        == "Viewer graphics: GPU on next launch. Close and reopen VIPP to apply."
    )
    assert before == (
        widget._compute_mode,
        widget._thumbnail_statistics_policy,
        widget._workflow_tabs.current.session_id,
    )

    widget._populate_settings_toolbar_menu()
    assert _graphics_rows(menu) == [
        "Viewer graphics",
        "Active: CPU (software)",
        "Next launch: GPU · Restart needed",
    ]
    assert (
        "llvmpipe" in _active_renderer_action(menu, "Active: CPU (software)").toolTip()
    )
    next_launch = _action(menu, "Next launch: GPU · Restart needed")
    for text in ("Close and reopen VIPP", "workflow compute", "thumbnail statistics"):
        assert text in next_launch.toolTip()
    submenu = next_launch.menu()
    assert [action.text() for action in submenu.actions() if action.isChecked()] == [
        "GPU"
    ]


def test_embedded_menu_reports_detected_renderer_but_does_not_offer_settings(
    widget,
    monkeypatch,
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Do not read or write settings for an embedded viewer")

    monkeypatch.setattr(widget_module, "load_viewer_graphics_mode", forbidden)
    monkeypatch.setattr(widget_module, "save_viewer_graphics_mode", forbidden)
    monkeypatch.setattr(
        widget_module,
        "query_viewer_graphics",
        lambda viewer: ViewerGraphicsStatus(
            "NVIDIA GeForce RTX 5090/PCIe/SSE2",
            "NVIDIA Corporation",
            "4.6.0 NVIDIA 617.42",
        ),
    )
    monkeypatch.setattr(
        widget_module,
        "viewer_graphics_availability",
        lambda: SimpleNamespace(can_configure=False, reason="Managed by host napari."),
    )
    widget._populate_settings_toolbar_menu()
    menu = widget.settings_menu
    assert _graphics_rows(menu) == [
        "Viewer graphics",
        "Active: NVIDIA GeForce RTX 5090",
    ]
    active = _active_renderer_action(menu, "Active: NVIDIA GeForce RTX 5090")
    for text in (
        "NVIDIA GeForce RTX 5090/PCIe/SSE2",
        "NVIDIA Corporation",
        "4.6.0 NVIDIA 617.42",
        "Managed by host napari",
    ):
        assert text in active.toolTip()


def test_unavailable_software_mode_explains_limit_and_unknown_renderer(
    qtbot,
    widget,
    monkeypatch,
):
    monkeypatch.setattr(widget_module, "query_viewer_graphics", lambda _viewer: None)
    monkeypatch.setattr(
        widget_module, "load_viewer_graphics_mode", lambda: ViewerGraphicsMode.AUTO
    )
    monkeypatch.setattr(widget_module, "viewer_graphics_startup", lambda: None)
    monkeypatch.setattr(
        widget_module,
        "viewer_graphics_availability",
        lambda: SimpleNamespace(
            can_configure=True,
            cpu_available=False,
            cpu_reason="Software library missing.",
        ),
    )
    menu = widget.settings_menu
    font = menu.font()
    font.setPointSize(14)
    menu.setFont(font)
    menu.popup(QPoint(20, 20))
    try:
        qtbot.waitUntil(menu.isVisible)
        assert _graphics_rows(menu) == [
            "Viewer graphics",
            "Active: Not detected",
            "Next launch: Auto",
        ]
        active = _active_renderer_action(menu, "Active: Not detected")
        label = active.defaultWidget()
        assert active.isVisible() and menu.actionGeometry(active).isValid()
        assert label.isVisible() and label.testAttribute(
            Qt.WA_TransparentForMouseEvents
        )
        point = menu.actionGeometry(active).center()
        assert menu.actionAt(point) is active
        # The status label keeps enabled colors, while the real menu's hover
        # route still delivers full details from its non-command action.
        QApplication.sendEvent(
            menu, QHelpEvent(QEvent.ToolTip, point, menu.mapToGlobal(point))
        )
        assert QToolTip.text() == active.toolTip()
        submenu = _action(menu, "Next launch: Auto").menu()
        assert [action.text() for action in submenu.actions()] == ["Auto", "GPU", "CPU"]
        assert _action(submenu, "Auto").isEnabled()
        assert _action(submenu, "GPU").isEnabled()
        cpu = _action(submenu, "CPU")
        assert not cpu.isEnabled() and not isinstance(cpu, QWidgetAction)
        assert "Software library missing" in cpu.toolTip()
    finally:
        QToolTip.hideText()
        menu.hide()


def test_query_uses_exact_owned_canvas_and_restores_previous_context(
    qtbot, monkeypatch
):
    class Context:
        current = None

        def __init__(self):
            self.surface_id = object()

        @classmethod
        def currentContext(cls):  # noqa: N802
            return cls.current

        def surface(self):
            return self.surface_id

        def isValid(self):  # noqa: N802
            return True

        def makeCurrent(self, surface):  # noqa: N802
            assert surface is self.surface_id
            Context.current = self
            return True

    prior, canvas_context = Context(), Context()
    Context.current = prior
    transitions = []

    def enter_canvas():
        transitions.append("enter")
        canvas_context.makeCurrent(canvas_context.surface_id)

    def read(context):
        assert context is canvas_context and Context.current is canvas_context
        transitions.append("read")
        return ViewerGraphicsStatus("Active canvas")

    native = SimpleNamespace(
        isValid=lambda: True,
        context=lambda: canvas_context,
        makeCurrent=enter_canvas,
        doneCurrent=lambda: setattr(Context, "current", None),
    )
    viewer = SimpleNamespace()
    viewer.window = SimpleNamespace(
        qt_viewer=SimpleNamespace(
            viewer=viewer,
            canvas=SimpleNamespace(native=native),
        )
    )
    monkeypatch.setattr(graphics_status, "QOpenGLContext", Context)
    monkeypatch.setattr(graphics_status, "_read_gl_strings", read)

    assert graphics_status.query_viewer_graphics(viewer) == ViewerGraphicsStatus(
        "Active canvas"
    )
    assert Context.current is prior
    assert transitions == ["enter", "read"]
    other = SimpleNamespace(window=viewer.window)
    assert graphics_status.query_viewer_graphics(other) is None
    assert Context.current is prior
    assert transitions == ["enter", "read"]

    Context.current = None
    assert graphics_status.query_viewer_graphics(viewer) is not None
    assert Context.current is None


@pytest.mark.parametrize("address", (None, 0, 1, 2, 3, -1, ctypes.c_void_p(-1).value))
def test_gl_string_query_ignores_invalid_wgl_addresses(address):
    assert (
        graphics_status._read_gl_strings(
            SimpleNamespace(getProcAddress=lambda _name: address)
        )
        is None
    )
