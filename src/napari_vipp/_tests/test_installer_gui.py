from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp.installer import gui as gui_module
from napari_vipp.installer.frontend import (
    InstallerScreen,
    InstallerSelection,
    InstallerViewState,
    TargetKind,
)
from napari_vipp.installer.gui import (
    DEVELOPMENT_BUILD_LABEL,
    VIPP_TAGLINE,
    InstallerWindow,
    _activity_text,
    _reviewed_message,
    _window_title,
    build_parser,
)
from napari_vipp.installer.models import ComputeTrack


class _Controller:
    def __init__(self):
        self.calls = []
        self.busy = False

    def confirm(self):
        self.calls.append("confirm")

    def open_vipp(self):
        self.calls.append("open")

    def retry(self):
        self.calls.append("retry")

    def request_repair(self):
        self.calls.append("repair")

    def invalidate_selection(self, selection):
        self.calls.append(("invalidate", selection))
        return True

    def start(self, selection):
        self.calls.append(("start", selection))


class _Value:
    def __init__(self, value=None):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class _Widget:
    def __init__(self):
        self.configured = []

    def configure(self, **kwargs):
        self.configured.append(kwargs)


class _Canvas:
    def __init__(self, bounds=(0, 0, 640, 1200), *, height=600):
        self.bounds = bounds
        self.height = height
        self.configured = []
        self.items = []
        self.scrolls = []
        self.moves = []

    def bbox(self, tag):
        assert tag == "all"
        return self.bounds

    def configure(self, **kwargs):
        self.configured.append(kwargs)

    def itemconfigure(self, item, **kwargs):
        self.items.append((item, kwargs))

    def yview_scroll(self, steps, units):
        self.scrolls.append((steps, units))

    def yview_moveto(self, fraction):
        self.moves.append(fraction)

    def winfo_height(self):
        return self.height


def _state(screen, *, kind=None, help_url=""):
    return InstallerViewState(
        screen=screen,
        headline="headline",
        message="message",
        primary_label="primary",
        primary_enabled=True,
        target_kind=kind,
        help_url=help_url,
    )


def _window(state):
    window = object.__new__(InstallerWindow)
    window._state = state
    window._controller = _Controller()
    window._browse_calls = 0
    window._browse_location = lambda: setattr(
        window,
        "_browse_calls",
        window._browse_calls + 1,
    )
    return window


def test_gui_module_imports_without_loading_tkinter():
    code = (
        "import sys; import napari_vipp.installer.gui; "
        "raise SystemExit('tkinter' in sys.modules)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
        cwd=Path(__file__).resolve().parents[3],
    )

    assert completed.returncode == 0, completed.stderr


def test_parser_keeps_recommended_automatic_route_by_default():
    args = build_parser().parse_args([])

    assert args.track == "automatic"
    assert args.install_root is None
    assert args.base_python is None


@pytest.mark.parametrize("development", [False, True])
def test_development_identity_is_visible_in_title_and_version_output(
    development,
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(gui_module, "_installed_version", lambda: "0.13.0a4")
    monkeypatch.setattr(
        gui_module,
        "_is_development_build",
        lambda: development,
    )

    title = _window_title("0.13.0a4", development=development)
    with pytest.raises(SystemExit, match="0"):
        build_parser().parse_args(["--version"])
    version_output = capsys.readouterr().out.strip()

    if development:
        assert title.endswith(DEVELOPMENT_BUILD_LABEL)
        assert version_output.endswith(DEVELOPMENT_BUILD_LABEL)
    else:
        assert DEVELOPMENT_BUILD_LABEL not in title
        assert DEVELOPMENT_BUILD_LABEL not in version_output


def test_ready_and_success_primary_actions_are_unambiguous():
    ready = _window(_state(InstallerScreen.READY, kind=TargetKind.NEW))
    success = _window(_state(InstallerScreen.SUCCESS, kind=TargetKind.NEW))

    ready._primary_requested()
    success._primary_requested()

    assert ready._controller.calls == ["confirm"]
    assert success._controller.calls == ["open"]


def test_foreign_target_routes_to_separate_location_not_confirmation():
    window = _window(_state(InstallerScreen.BLOCKED, kind=TargetKind.FOREIGN))

    window._primary_requested()

    assert window._browse_calls == 1
    assert window._controller.calls == []


def test_current_secondary_action_is_explicit_repair():
    window = _window(_state(InstallerScreen.CURRENT, kind=TargetKind.CURRENT))

    window._secondary_requested()

    assert window._controller.calls == ["repair"]


def test_plain_activity_history_marks_prior_and_current_steps():
    rendered = _activity_text(
        [
            "Starting VIPP Setup…",
            "Checking Windows and Python…",
            "Checking available hardware…",
        ]
    )

    assert rendered.splitlines() == [
        "✓ Starting VIPP Setup…",
        "✓ Checking Windows and Python…",
        "● Checking available hardware…",
    ]


def test_checking_activity_explains_long_package_review_and_elapsed_time():
    rendered = _activity_text(
        ["Starting VIPP Setup…", "Checking Windows and Python…"],
        screen=InstallerScreen.CHECKING,
        elapsed_seconds=134,
    )

    assert "✓ Starting VIPP Setup…" in rendered
    assert "● Checking this computer and reviewing exact packages…" in rendered
    assert "can take several minutes" in rendered
    assert "moving bar means setup is still working" in rendered
    assert "Elapsed in this stage: 2m 14s" in rendered


def test_ready_message_shows_exact_checked_settings_without_technical_details(
    tmp_path,
):
    target = tmp_path / "managed CUDA installation"
    state = InstallerViewState(
        screen=InstallerScreen.READY,
        headline="Ready to install VIPP",
        message="VIPP will be installed in its own safe location.",
        primary_label="Install VIPP",
        primary_enabled=True,
        target=target,
        track=ComputeTrack.CUDA13,
    )

    rendered = _reviewed_message(
        state,
        InstallerSelection(
            track="cuda13",
            install_root=target,
            create_desktop_shortcut=False,
        ),
    )

    assert rendered.startswith(state.message)
    assert "Reviewed settings:" in rendered
    assert f"Installation location: {target}" in rendered
    assert "Computer use: NVIDIA GPU acceleration (CUDA 13)" in rendered
    assert "Shortcuts: Start Menu only" in rendered
    assert "technical" not in rendered.casefold()

    desktop_rendered = _reviewed_message(
        state,
        InstallerSelection(
            track="cuda13",
            install_root=target,
            create_desktop_shortcut=True,
        ),
    )
    assert "Shortcuts: Start Menu and Desktop" in desktop_rendered


def test_non_ready_message_is_not_decorated_with_reviewed_settings():
    state = _state(InstallerScreen.CHECKING)

    assert _reviewed_message(state, InstallerSelection()) == state.message


def test_advanced_details_are_live_and_never_use_placeholder_text(tmp_path):
    window = object.__new__(InstallerWindow)
    window._selection = InstallerSelection(install_root=tmp_path)
    window._status_history = [
        "Starting VIPP Setup…",
        "Checking Windows, Python, and available hardware…",
    ]
    window._development_build = True
    state = InstallerViewState(
        screen=InstallerScreen.CHECKING,
        headline="Checking this computer…",
        message="VIPP is finding the safest setup.",
        primary_label="Please wait",
        primary_enabled=False,
        status_message="Checking available hardware…",
        technical_details="Python candidate: CPython 3.12",
    )

    details = window._rendered_details(state)

    assert "Stage: Checking this computer" in details
    assert DEVELOPMENT_BUILD_LABEL in details.splitlines()
    assert "Current activity: Checking available hardware…" in details
    assert f"Requested location: {tmp_path}" in details
    assert "Python candidate: CPython 3.12" in details
    assert "Recent activity:" in details
    assert "Detailed checks will appear here." not in details

    window._development_build = False
    assert DEVELOPMENT_BUILD_LABEL not in window._rendered_details(state)


def test_installer_header_uses_official_logo_without_reconstructed_circle():
    gui_source = (
        Path(__file__).resolve().parents[1] / "installer" / "gui.py"
    ).read_text(encoding="utf-8")

    assert VIPP_TAGLINE == "Visual image processing made approachable"
    assert "bundled_logo_path" in gui_source
    assert "text=DEVELOPMENT_BUILD_LABEL" in gui_source
    assert "wraplength=0" in gui_source
    assert "wraplength=210" not in gui_source
    assert "create_oval" not in gui_source


def test_installer_has_page_and_console_scrollbars_with_fixed_footer():
    gui_source = (
        Path(__file__).resolve().parents[1] / "installer" / "gui.py"
    ).read_text(encoding="utf-8")

    assert "self._content_scrollbar = ttk.Scrollbar(" in gui_source
    assert "yscrollcommand=self._content_scrollbar.set" in gui_source
    assert "details_scrollbar = ttk.Scrollbar(" in gui_source
    assert "yscrollcommand=details_scrollbar.set" in gui_source
    assert "self._button_bar = ttk.Frame(main" in gui_source
    assert "self._button_bar = ttk.Frame(outer" not in gui_source


def test_scroll_region_and_viewport_width_are_synchronised():
    window = object.__new__(InstallerWindow)
    canvas = _Canvas()
    window._content_canvas = canvas
    window._content_window = 17

    window._refresh_scroll_region()
    window._viewport_resized(SimpleNamespace(width=511))

    assert {"scrollregion": (0, 0, 640, 1200)} in canvas.configured
    assert canvas.items == [(17, {"width": 511})]


def test_mousewheel_scrolls_only_an_overflowing_page():
    window = object.__new__(InstallerWindow)
    canvas = _Canvas(height=600)
    window._content_canvas = canvas
    window._details = object()

    for delta, steps in (
        (120, -1),
        (-120, 1),
        (240, -2),
        (-240, 2),
        (1, -1),
        (-1, 1),
    ):
        assert (
            window._content_mousewheel(
                SimpleNamespace(delta=delta, widget=object())
            )
            == "break"
        )
        assert canvas.scrolls[-1] == (steps, "units")

    assert (
        window._content_mousewheel(
            SimpleNamespace(delta=-120, widget=window._details)
        )
        is None
    )
    assert len(canvas.scrolls) == 6

    canvas.bounds = (0, 0, 640, 500)
    assert (
        window._content_mousewheel(
            SimpleNamespace(delta=-120, widget=object())
        )
        is None
    )
    assert len(canvas.scrolls) == 6


def test_advanced_toggle_reveals_console_after_idle_layout():
    class _Flag:
        value = True

        def get(self):
            return self.value

    class _Frame:
        def __init__(self):
            self.packed = []
            self.forgotten = 0

        def pack(self, **kwargs):
            self.packed.append(kwargs)

        def pack_forget(self):
            self.forgotten += 1

    class _Root:
        def __init__(self):
            self.idle = []
            self.timed = []

        def after_idle(self, callback):
            self.idle.append(callback)

        def after(self, delay, callback):
            self.timed.append((delay, callback))

        @staticmethod
        def update_idletasks():
            return None

    class _Details:
        def __init__(self):
            self.seen = []

        def see(self, index):
            self.seen.append(index)

    window = object.__new__(InstallerWindow)
    window._show_advanced = _Flag()
    window._advanced = _Frame()
    window.root = _Root()
    window._details = _Details()
    window._content_canvas = _Canvas()

    window._toggle_advanced()

    assert window._advanced.packed
    assert window._details.seen == []
    assert window._content_canvas.moves == []
    delay, callback = window.root.timed.pop()
    assert delay == 75
    callback()
    assert window._details.seen == ["end"]
    assert window._content_canvas.moves == [1.0]

    window._show_advanced.value = False
    window._toggle_advanced()
    assert window._advanced.forgotten == 1
    window.root.idle.pop()()
    assert window._content_canvas.moves[-1] == 0.0


def test_responsive_guidance_wrap_leaves_room_for_frame_padding():
    configured: list[int] = []

    class _Label:
        def configure(self, *, wraplength):
            configured.append(wraplength)

    class _Event:
        width = 720

    window = object.__new__(InstallerWindow)
    window._headline_label = _Label()
    window._message_label = _Label()
    window._status_label = _Label()
    window._activity_label = _Label()
    window._content_canvas = _Canvas(bounds=None)

    window._content_resized(_Event())

    assert configured == [648, 648, 648, 648]


def test_short_screen_geometry_remains_inside_scrollable_minimum():
    class _Root:
        @staticmethod
        def winfo_screenwidth():
            return 620

        @staticmethod
        def winfo_screenheight():
            return 540

    window = object.__new__(InstallerWindow)
    window.root = _Root()

    assert window._bounded_geometry(760, 660) == "540x420"


def test_form_edit_invalidates_plan_and_check_uses_exact_visible_values(tmp_path):
    window = object.__new__(InstallerWindow)
    window._controller = _Controller()
    window._selection = InstallerSelection()
    window._track = _Value("NVIDIA GPU acceleration (CUDA 13)")
    window._install_root = _Value(str(tmp_path / "custom GPU root"))
    window._existing_environment = _Value(False)
    window._existing_python = None
    window._desktop_shortcut = _Value(False)
    window._suppress_setting_events = False
    window._primary = _Widget()
    window._secondary = _Widget()
    window._status = _Value()
    window._activity = _Value()
    window._status_history = []
    window._state = _state(InstallerScreen.READY, kind=TargetKind.NEW)
    window._record_status = lambda message: window._status_history.append(message)
    window._elapsed_seconds = lambda: 0

    window._mark_settings_dirty()

    action, invalidated = window._controller.calls[-1]
    assert action == "invalidate"
    assert invalidated.track.value == "cuda13"
    assert invalidated.install_root == tmp_path / "custom GPU root"
    assert invalidated.create_desktop_shortcut is False
    assert window._primary.configured[-1] == {"state": "disabled"}
    assert window._secondary.configured[-1] == {"state": "disabled"}
    assert not any(call == "confirm" for call in window._controller.calls)

    window._check_settings_requested()

    action, checked = window._controller.calls[-1]
    assert action == "start"
    assert checked == invalidated


def test_install_location_uses_variable_trace_not_key_release_only():
    gui_source = (
        Path(__file__).resolve().parents[1] / "installer" / "gui.py"
    ).read_text(encoding="utf-8")

    assert 'self._install_root.trace_add("write"' in gui_source
    assert 'location.bind("<KeyRelease>"' not in gui_source
