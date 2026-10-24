from __future__ import annotations

import hashlib
import io
import subprocess
import sys
import uuid
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_vipp.installer import gui as gui_module
from napari_vipp.installer.frontend import (
    BlockedAction,
    InstallerScreen,
    InstallerSelection,
    InstallerViewState,
    InstallSizeEstimate,
    RegistrationRepairReview,
    TargetKind,
)
from napari_vipp.installer.gui import (
    DEVELOPMENT_BUILD_LABEL,
    VIPP_TAGLINE,
    InstallerWindow,
    _activity_text,
    _format_size,
    _header_inline_width,
    _header_minimum_width,
    _header_should_stack,
    _reviewed_message,
    _size_review_text,
    _window_title,
    build_parser,
)
from napari_vipp.installer.models import ComputeTrack, UpdateStrategy
from napari_vipp.installer.ownership import (
    OwnedPackage,
    OwnershipRecord,
    managed_environments_root,
    write_ownership_record,
)


def _existing_open_target(tmp_path, track):
    target = tmp_path / "private test location" / "environments" / track.value
    environment = managed_environments_root(target) / "installed-version"
    scripts = environment / "Scripts"
    scripts.mkdir(parents=True)
    marker = environment / ".vipp-install-candidate.json"
    marker.write_text("{}", encoding="utf-8")
    (environment / "pyvenv.cfg").write_text("home = fixture-base\n", encoding="utf-8")
    (scripts / "python.exe").touch()
    (scripts / "vipp-app.exe").write_bytes(b"fixture launcher")
    record = OwnershipRecord(
        installation_id=str(uuid.uuid4()),
        managed_root=target,
        environment_root=environment,
        distribution="napari-vipp",
        version="0.16.0a7",
        track=track,
        base_python=tmp_path / "separate base" / "python.exe",
        resolved_plan_id="c" * 64,
        packages=(OwnedPackage("napari-vipp", "0.16.0a7", "d" * 64),),
        created_at="2026-10-24T00:00:00Z",
        updated_at="2026-10-24T00:00:00Z",
        environment_marker_sha256=hashlib.sha256(marker.read_bytes()).hexdigest(),
    )
    write_ownership_record(target, record)
    return target, record


class _OpenOnlyBackend:
    def __init__(self):
        self.opened = []

    def open_vipp(self, launcher):
        self.opened.append(launcher)

    def __getattr__(self, name):
        raise AssertionError(f"Launch-only must not enter setup: {name}")


@pytest.mark.parametrize("track", tuple(ComputeTrack))
def test_open_installed_is_read_only_and_independent_of_bundled_version(
    tmp_path, monkeypatch, track
):
    from napari_vipp.installer import discovery

    target, record = _existing_open_target(tmp_path, track)
    monkeypatch.setattr(
        discovery, "_windows_known_folder", lambda _: tmp_path / "local"
    )
    monkeypatch.setitem(sys.modules, "tkinter", None)
    before = {str(p): p.read_bytes() for p in target.rglob("*") if p.is_file()}
    backend = _OpenOnlyBackend()
    assert (
        gui_module.main(
            ["--open-installed", "--track", track.value, "--install-root", str(target)],
            backend=backend,
        )
        == 0
    )
    assert backend.opened == [record.environment_root / "Scripts" / "vipp-app.exe"]
    assert before == {str(p): p.read_bytes() for p in target.rglob("*") if p.is_file()}


@pytest.mark.parametrize(
    "damage", ["absent", "wrong-track", "launcher", "journal", "marker"]
)
def test_open_installed_refuses_unready_roots_without_setup(
    tmp_path, monkeypatch, damage
):
    from napari_vipp.installer import discovery, engine

    target, record = _existing_open_target(tmp_path, ComputeTrack.CPU)
    local = tmp_path / "local"
    monkeypatch.setattr(discovery, "_windows_known_folder", lambda _: local)
    track = "cpu"
    if damage == "absent":
        target = tmp_path / "unowned"
    elif damage == "wrong-track":
        track = "cuda13"
    elif damage == "launcher":
        (record.environment_root / "Scripts" / "vipp-app.exe").unlink()
    elif damage == "marker":
        (record.environment_root / ".vipp-install-candidate.json").write_text(
            "changed", encoding="utf-8"
        )
    else:
        journal = engine._transaction_journal_path(local / "VIPP" / "installer", target)
        journal.parent.mkdir(parents=True)
        journal.write_text("preserve pending transaction", encoding="utf-8")
    backend = _OpenOnlyBackend()
    assert (
        gui_module.main(
            ["--open-installed", "--track", track, "--install-root", str(target)],
            backend=backend,
        )
        == 1
    )
    assert not backend.opened
    if damage == "journal":
        assert journal.read_text(encoding="utf-8") == "preserve pending transaction"


def test_open_installed_rechecks_manifest_before_spawn(tmp_path, monkeypatch):
    from napari_vipp.installer import discovery, ownership

    target, record = _existing_open_target(tmp_path, ComputeTrack.CPU)
    monkeypatch.setattr(discovery, "_windows_known_folder", lambda _: None)
    original = ownership.inspect_ownership
    calls = []

    def changed(root):
        calls.append(root)
        if len(calls) == 2:
            write_ownership_record(
                target, replace(record, updated_at="2026-10-24T01:00Z")
            )
        return original(root)

    monkeypatch.setattr(ownership, "inspect_ownership", changed)
    backend = _OpenOnlyBackend()
    assert (
        gui_module.main(
            ["--open-installed", "--track", "cpu", "--install-root", str(target)],
            backend=backend,
        )
        == 1
    )
    assert len(calls) == 2
    assert not backend.opened


def test_open_installed_bounds_marker_read_even_if_file_grows(tmp_path, monkeypatch):
    from napari_vipp.installer import discovery

    target, record = _existing_open_target(tmp_path, ComputeTrack.CPU)
    monkeypatch.setattr(discovery, "_windows_known_folder", lambda _: None)
    marker = record.environment_root / ".vipp-install-candidate.json"
    original_open = Path.open
    reads = []

    class GrowingMarker(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    def open_file(path, *args, **kwargs):
        if path == marker:
            return GrowingMarker(b"x" * (1024 * 1024 + 10))
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open_file)
    backend = _OpenOnlyBackend()
    assert (
        gui_module.main(
            ["--open-installed", "--track", "cpu", "--install-root", str(target)],
            backend=backend,
        )
        == 1
    )
    assert reads == [1024 * 1024 + 1]
    assert not backend.opened


@pytest.mark.parametrize("damage", ["redirected-launcher", "custom-root-journal"])
def test_open_installed_checks_direct_paths_and_custom_journal(
    tmp_path, monkeypatch, damage
):
    from napari_vipp.installer import discovery, engine

    target, record = _existing_open_target(tmp_path, ComputeTrack.CPU)
    monkeypatch.setattr(discovery, "_windows_known_folder", lambda _: None)
    if damage == "redirected-launcher":
        original = engine._path_has_reparse_component
        launcher = record.environment_root / "Scripts" / "vipp-app.exe"
        monkeypatch.setattr(
            engine,
            "_path_has_reparse_component",
            lambda path: path == launcher or original(path),
        )
    else:
        journal = engine._transaction_journal_path(
            target.parent.parent / "installer", target
        )
        journal.parent.mkdir(parents=True)
        journal.write_text("pending", encoding="utf-8")
    backend = _OpenOnlyBackend()
    assert (
        gui_module.main(
            ["--open-installed", "--track", "cpu", "--install-root", str(target)],
            backend=backend,
        )
        == 1
    )
    assert not backend.opened


@pytest.mark.parametrize(
    "extra",
    [
        [],
        ["--track", "cpu", "--base-python", "other"],
        ["--track", "cpu", "--update-strategy", "fresh"],
    ],
)
def test_open_installed_requires_explicit_track_and_no_install_options(extra):
    backend = _OpenOnlyBackend()
    assert gui_module.main(["--open-installed", *extra], backend=backend) == 1
    assert not backend.opened


class _Controller:
    def __init__(self):
        self.calls = []
        self.busy = False

    def confirm(self):
        self.calls.append("confirm")

    def open_vipp(self):
        self.calls.append("open")

    def open_owned_uninstaller(self):
        self.calls.append("open-uninstaller")

    def repair_registration(self, *, confirmed, fingerprint):
        self.calls.append(("repair-registration", confirmed, fingerprint))

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


class _StagePanel(_Widget):
    def __init__(self, *, visible):
        super().__init__()
        self.visible = visible
        self.packs = []
        self.grid_calls = 0

    def pack(self, **kwargs):
        self.visible = True
        self.packs.append(kwargs)

    def pack_forget(self):
        self.visible = False

    def grid(self):
        self.visible = True
        self.grid_calls += 1

    def grid_remove(self):
        self.visible = False


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


def _state(
    screen,
    *,
    kind=None,
    help_url="",
    blocked_action=BlockedAction.RETRY,
):
    return InstallerViewState(
        screen=screen,
        headline="headline",
        message="message",
        primary_label="primary",
        primary_enabled=True,
        target_kind=kind,
        help_url=help_url,
        blocked_action=blocked_action,
    )


def _window(state):
    window = object.__new__(InstallerWindow)
    window._state = state
    window._controller = _Controller()
    window._installed_apps_calls = 0
    window._open_installed_apps = lambda: setattr(
        window,
        "_installed_apps_calls",
        window._installed_apps_calls + 1,
    )
    window._default_calls = 0
    window._use_default_location = lambda: setattr(
        window,
        "_default_calls",
        window._default_calls + 1,
    )
    window._cpu_calls = 0
    window._use_cpu = lambda: setattr(
        window,
        "_cpu_calls",
        window._cpu_calls + 1,
    )
    return window


def _staged_window():
    """Pure renderer fixture: no Tk, worker, filesystem or install actions."""

    window = object.__new__(InstallerWindow)
    window._controller = _Controller()
    window._installation_started = False
    window._review_controls_visible = True
    window._method_panel = _StagePanel(visible=True)
    window._advanced_review = _StagePanel(visible=True)
    window._selected_method_label = _StagePanel(visible=False)
    window._selected_method = _Value()
    window._review_stage_label = _Widget()
    window._install_stage_label = _Widget()
    window._content_canvas = _Canvas()
    window._progress = object()
    window.root = SimpleNamespace(after_idle=lambda callback: callback())
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
    assert args.update_strategy == "auto"


@pytest.mark.parametrize("strategy", list(UpdateStrategy))
def test_parser_accepts_each_explicit_update_method(strategy):
    assert (
        build_parser().parse_args(["--update-strategy", strategy.value]).update_strategy
        == strategy.value
    )


def test_review_displays_requested_method_selected_method_and_fallback_reason():
    reason = "A dependency changed; a fresh environment is required."
    state = InstallerViewState(
        screen=InstallerScreen.READY,
        headline="An update is ready",
        message="The old installation stays available until checks pass.",
        primary_label="Update VIPP",
        primary_enabled=True,
        target_kind=TargetKind.UPDATE,
        selected_strategy=UpdateStrategy.FRESH,
        strategy_reason=reason,
    )

    message = _reviewed_message(state, InstallerSelection())

    assert "Update method requested: Automatic" in message
    assert "Selected method: Fresh installation" in message
    assert reason in message


def test_method_choice_invalidates_then_rechecks_without_applying():
    window = object.__new__(InstallerWindow)
    calls = []
    window._mark_settings_dirty = lambda: calls.append("invalidate")
    window._check_settings_requested = lambda: calls.append("check")

    window._update_method_changed()

    assert calls == ["invalidate", "check"]


def test_review_then_installing_hides_choices_without_changing_approval():
    window = _staged_window()
    ready = replace(
        _state(InstallerScreen.READY, kind=TargetKind.UPDATE),
        selected_strategy=UpdateStrategy.INCREMENTAL,
    )

    window._render_setup_stage(ready)

    assert window._review_stage_label.configured[-1]["text"] == "1  Review (current)"
    assert window._install_stage_label.configured[-1]["text"] == "2  Installing"
    assert window._method_panel.visible
    assert window._advanced_review.visible
    assert not window._selected_method_label.visible
    assert window._controller.calls == []

    working = replace(ready, screen=InstallerScreen.WORKING, progress_stage="preparing")
    window._render_setup_stage(working)

    assert not window._method_panel.visible
    assert not window._advanced_review.visible
    assert window._selected_method_label.visible
    assert window._selected_method.get() == "Selected method: Incremental update"
    assert window._review_stage_label.configured[-1]["text"] == "1  Review (complete)"
    assert (
        window._install_stage_label.configured[-1]["text"] == "2  Installing (current)"
    )
    assert window._selected_method_label.packs == [
        {"fill": "x", "pady": (0, 12), "before": window._progress}
    ]
    assert window._content_canvas.moves == [0.0]
    assert window._controller.calls == []

    # Milestones must not repeatedly repack controls or reset reading position.
    window._render_setup_stage(replace(working, progress_stage="installing"))
    assert len(window._selected_method_label.packs) == 1
    assert window._content_canvas.moves == [0.0]
    assert window._controller.calls == []


@pytest.mark.parametrize(
    "screen,suffix",
    [
        (InstallerScreen.SUCCESS, "complete"),
        (InstallerScreen.FAILED, "stopped"),
        (InstallerScreen.CANCELLED, "stopped"),
    ],
)
def test_terminal_install_results_keep_stage_and_diagnostics_not_review_clutter(
    screen, suffix
):
    window = _staged_window()
    # The UI queue may coalesce a short WORKING state directly into its result.
    state = replace(
        _state(screen, kind=TargetKind.UPDATE),
        selected_strategy=UpdateStrategy.FRESH,
    )

    window._render_setup_stage(state)

    assert window._install_stage_label.configured[-1]["text"] == (
        f"2  Installing ({suffix})"
    )
    assert not window._method_panel.visible
    assert not window._advanced_review.visible
    assert window._selected_method_label.visible
    assert window._selected_method.get() == "Selected method: Fresh installation"
    assert window._controller.calls == []

    # Retry is still a read-only controller check; the next checking render
    # restores review controls before a separate, explicit install approval.
    window._render_setup_stage(_state(InstallerScreen.CHECKING))
    assert window._method_panel.visible
    assert window._advanced_review.visible
    assert not window._selected_method_label.visible
    assert window._method_panel.packs[-1]["before"] is window._progress
    assert window._controller.calls == []


def test_registration_only_work_and_cancelled_checks_do_not_claim_installing():
    window = _staged_window()
    registration = replace(
        _state(InstallerScreen.WORKING, kind=TargetKind.BLOCKED),
        progress_stage="repairing_registration",
    )

    window._render_setup_stage(registration)

    assert not window._installation_started
    assert not window._selected_method_label.visible
    assert not window._method_panel.visible
    assert window._review_stage_label.configured[-1]["text"] == "1  Review (current)"
    assert window._install_stage_label.configured[-1]["text"] == "2  Installing"

    window._render_setup_stage(_state(InstallerScreen.CHECKING))
    window._render_setup_stage(_state(InstallerScreen.CANCELLING))
    window._render_setup_stage(_state(InstallerScreen.CANCELLED))
    assert not window._installation_started
    assert not window._selected_method_label.visible
    assert window._controller.calls == []


@pytest.mark.parametrize("screen", [InstallerScreen.CURRENT, InstallerScreen.BLOCKED])
def test_preapproval_current_and_blocked_states_keep_method_choices(screen):
    window = _staged_window()

    window._render_setup_stage(_state(screen))

    assert window._method_panel.visible
    assert window._advanced_review.visible
    assert not window._selected_method_label.visible
    assert window._controller.calls == []


def test_method_help_explains_compatible_versions_and_verified_cache():
    assert (
        "keeps compatible installed dependency versions"
        in gui_module._UPDATE_METHOD_HELP
    )
    assert (
        "requires different versions, it uses Fresh" in gui_module._UPDATE_METHOD_HELP
    )
    assert "reuses verified cached downloads" in gui_module._UPDATE_METHOD_HELP
    assert "downloads missing packages" in gui_module._UPDATE_METHOD_HELP
    assert "newest" not in gui_module._UPDATE_METHOD_HELP
    assert gui_module._PHASE_LABELS["installing"] == "Installing required components"


def test_update_method_control_is_built_outside_collapsed_advanced_panel(monkeypatch):
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    controller = _Controller()
    controller.state = _state(InstallerScreen.CHECKING)
    controller.cancel = lambda: pytest.fail("Construction attempted cancellation")
    monkeypatch.setattr(
        gui_module, "InstallerController", lambda *_args, **_kwargs: controller
    )
    try:
        window = InstallerWindow(root, object())
        root.update_idletasks()

        assert window._strategy_control.winfo_manager() == "pack"
        assert window._strategy_help_label.winfo_manager() == "pack"
        assert window._strategy_control.master is not window._advanced
        assert window._advanced.winfo_manager() == ""
        assert window._show_advanced.get() is False
        assert window._update_strategy.get() == "Automatic — reuse compatible packages"
        assert controller.calls == [("start", InstallerSelection())]
    finally:
        root.destroy()


@pytest.mark.parametrize(
    ("screen", "expected"),
    [
        (InstallerScreen.CHECKING, "readonly"),
        (InstallerScreen.READY, "readonly"),
        (InstallerScreen.WORKING, "disabled"),
        (InstallerScreen.CANCELLING, "disabled"),
        (InstallerScreen.SUCCESS, "disabled"),
        (InstallerScreen.FAILED, "disabled"),
        (InstallerScreen.CANCELLED, "disabled"),
    ],
)
def test_update_method_is_available_before_resolution_but_immutable_during_apply(
    screen, expected
):
    window = object.__new__(InstallerWindow)
    window._track_control = _Widget()
    window._strategy_control = _Widget()
    window._desktop_shortcut_control = _Widget()
    window._check_settings = _Widget()

    window._set_settings_enabled(_state(screen))

    assert window._strategy_control.configured == [{"state": expected}]


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


def test_setup_help_opens_troubleshooting_and_close_does_not_retry(monkeypatch):
    opened = []
    monkeypatch.setattr("napari_vipp.installer.gui.webbrowser.open", opened.append)
    url = "https://example.invalid/installation/#troubleshooting"
    window = _window(
        _state(
            InstallerScreen.BLOCKED,
            kind=TargetKind.BLOCKED,
            blocked_action=BlockedAction.OPEN_SETUP_HELP,
            help_url=url,
        )
    )
    closed = []
    window._close_requested = lambda: closed.append(True)

    window._primary_requested()
    window._secondary_requested()

    assert opened == [url]
    assert closed == [True]
    assert window._controller.calls == []


def test_foreign_canonical_target_retries_without_exposing_custom_location():
    window = _window(_state(InstallerScreen.BLOCKED, kind=TargetKind.FOREIGN))

    window._primary_requested()

    assert window._controller.calls == ["retry"]


def test_owned_cuda_blocker_routes_to_installed_apps_not_location_chooser():
    window = _window(
        _state(
            InstallerScreen.BLOCKED,
            kind=TargetKind.BLOCKED,
            blocked_action=BlockedAction.OPEN_INSTALLED_APPS,
        )
    )

    window._primary_requested()

    assert window._installed_apps_calls == 1
    assert window._controller.calls == []


@pytest.mark.parametrize(
    ("action", "attribute"),
    [
        (BlockedAction.USE_DEFAULT_LOCATION, "_default_calls"),
        (BlockedAction.USE_CPU, "_cpu_calls"),
    ],
)
def test_exact_default_boundary_actions_are_typed(action, attribute):
    window = _window(
        _state(
            InstallerScreen.BLOCKED,
            kind=TargetKind.BLOCKED,
            blocked_action=action,
        )
    )

    window._primary_requested()

    assert getattr(window, attribute) == 1
    assert window._controller.calls == []


def test_owned_uninstaller_action_stays_inside_controller_boundary():
    window = _window(
        _state(
            InstallerScreen.BLOCKED,
            kind=TargetKind.BLOCKED,
            blocked_action=BlockedAction.RUN_OWNED_UNINSTALLER,
        )
    )

    window._primary_requested()

    assert window._controller.calls == ["open-uninstaller"]


@pytest.mark.parametrize("accepted", [False, True])
def test_registration_repair_confirmation_displays_changes_backup_and_scope(
    tmp_path, accepted
):
    review = RegistrationRepairReview(
        "a" * 64,
        "DisplayVersion: '0.16.0a3' → '0.16.0a7'",
        tmp_path / "entry-backup.json",
        object(),
    )
    window = _window(
        replace(
            _state(
                InstallerScreen.BLOCKED,
                blocked_action=BlockedAction.REPAIR_REGISTRATION,
            ),
            registration_repair=review,
        )
    )
    window.root = object()
    prompts = []

    def askyesno(title, message, *, parent, default, icon):
        prompts.append((title, message, parent, default, icon))
        return accepted

    window._messagebox = SimpleNamespace(askyesno=askyesno)
    window._primary_requested()

    assert len(prompts) == 1
    assert review.preview in prompts[0][1]
    unwrapped = prompts[0][1].replace("\n", "")
    normalized = " ".join(prompts[0][1].split())
    assert str(review.backup_path) in unwrapped
    assert "No packages, workflows or shortcuts" in normalized
    assert "updating still needs approval" in normalized
    assert prompts[0][3:] == ("no", "warning")
    assert max(map(len, prompts[0][1].splitlines())) <= 90
    assert window._controller.calls == (
        [("repair-registration", True, "a" * 64)] if accepted else []
    )


def test_close_waits_for_registration_commit_instead_of_interrupting_it():
    window = _window(
        replace(
            _state(InstallerScreen.WORKING), progress_stage="repairing_registration"
        )
    )
    window._controller.busy = True
    window.root = object()
    messages = []
    window._messagebox = SimpleNamespace(
        showinfo=lambda *args, **kwargs: messages.append((args, kwargs)),
        askyesno=lambda *_args, **_kwargs: pytest.fail(
            "Asked to interrupt registry write"
        ),
    )
    window._close_requested()
    assert len(messages) == 1
    assert window._controller.calls == []
    assert window._installed_apps_calls == 0


def test_open_installed_apps_uses_windows_settings_uri(monkeypatch):
    opened = []
    monkeypatch.setattr(gui_module.os, "startfile", opened.append, raising=False)
    window = object.__new__(InstallerWindow)

    window._open_installed_apps()

    assert opened == ["ms-settings:appsfeatures"]


def test_open_installed_apps_failure_keeps_manual_remediation_visible(monkeypatch):
    def fail_to_open(_uri):
        raise OSError("synthetic settings failure")

    errors = []
    monkeypatch.setattr(gui_module.os, "startfile", fail_to_open, raising=False)
    window = object.__new__(InstallerWindow)
    window.root = object()
    window._messagebox = SimpleNamespace(
        showerror=lambda title, message, **kwargs: errors.append(
            (title, message, kwargs)
        )
    )

    window._open_installed_apps()

    assert len(errors) == 1
    title, message, kwargs = errors[0]
    assert title == "Open Installed apps"
    assert "Settings > Apps > Installed apps" in message
    assert "remove VIPP (GPU)" in message
    assert kwargs == {"parent": window.root}


def test_use_default_switches_existing_environment_to_fixed_managed_route(
    tmp_path,
):
    existing_python = tmp_path / "Existing Å" / "Scripts" / "python.exe"
    window = object.__new__(InstallerWindow)
    window.root = object()
    window._suppress_setting_events = False
    window._existing_environment = _Value(True)
    window._existing_python = existing_python
    window._existing_label = _Widget()
    window._install_root = _Value("")
    window._track = _Value("Recommended automatically")
    window._update_strategy = _Value("Automatic — reuse compatible packages")
    window._desktop_shortcut = _Value(True)
    window._selection = InstallerSelection(existing_python=existing_python)
    window._controller = _Controller()
    window._primary = _Widget()
    window._secondary = _Widget()
    window._status = _Value()
    window._activity = _Value()
    window._status_history = []
    window._state = _state(InstallerScreen.BLOCKED, kind=TargetKind.BLOCKED)
    window._elapsed_seconds = lambda: 0
    rendered_details = []
    window._replace_details = rendered_details.append

    window._use_default_location()

    assert window._existing_environment.get() is False
    assert window._existing_python is None
    assert window._existing_label.configured[-1] == {
        "text": "No existing environment selected."
    }
    assert window._install_root.get() == ""
    assert rendered_details
    assert "Requested location: recommended managed location" in rendered_details[-1]
    assert "Existing environment Python:" not in rendered_details[-1]
    assert window._controller.calls[-1][0] == "start"


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


def test_checking_activity_explains_light_preflight_and_elapsed_time():
    rendered = _activity_text(
        ["Starting VIPP Setup…", "Checking Windows and Python…"],
        screen=InstallerScreen.CHECKING,
        elapsed_seconds=134,
    )

    assert "✓ Starting VIPP Setup…" in rendered
    assert "● Checking the installation and available update methods…" in rendered
    assert "Packages will not be downloaded or installed until you approve" in rendered
    assert "reviewing exact packages" not in rendered
    assert "Elapsed in this phase: 2m 14s" in rendered


@pytest.mark.parametrize(
    ("size_bytes", "rendered"),
    [
        (0, "0 B"),
        (1536, "1.5 KiB"),
        (250 * 1024**2, "250 MiB"),
        (1536 * 1024**2, "1.5 GiB"),
    ],
)
def test_installer_sizes_use_compact_binary_units(size_bytes, rendered):
    assert _format_size(size_bytes) == rendered


def test_size_review_separates_approximate_estimates_from_enforced_minimums():
    state = InstallerViewState(
        screen=InstallerScreen.READY,
        headline="Ready",
        message="Ready",
        primary_label="Install",
        primary_enabled=True,
        size_estimate=InstallSizeEstimate(
            download_bytes=1536 * 1024**2,
            installed_bytes=5 * 1024**3,
            peak_temporary_bytes=7 * 1024**3,
        ),
        required_free_bytes=15 * 1024**3,
        temporary_free_bytes=5 * 1024**3,
    )

    rendered = _size_review_text(state)

    assert "Approximate sizes (estimates, not enforced limits)" in rendered
    assert "Download (typical Fresh, before cache reuse" in rendered
    assert "not a measured Incremental transfer): 1.5 GiB" in rendered
    assert "Installed: 5 GiB" in rendered
    assert "Peak temporary working space: 7 GiB" in rendered
    assert "Required minimum free space (enforced): 15 GiB" in rendered
    assert "5 GiB on every drive used for Windows temporary files" in rendered
    reviewed = _reviewed_message(
        state,
        InstallerSelection(track="cuda13"),
    )
    assert rendered in reviewed


def test_quiet_active_phase_reports_heartbeat_without_claiming_failure():
    latest = "Installing VIPP and its required components…"
    rendered = _activity_text(
        ["Preparing files…", latest],
        screen=InstallerScreen.WORKING,
        elapsed_seconds=305,
        quiet_seconds=125,
    )

    assert latest in rendered
    assert "Still working — no new milestone for 2m 05s" in rendered
    assert "taking longer than usual" in rendered
    assert "setup has not reported a failure" in rendered

    stopped = _activity_text(
        ["Setup stopped before completion."],
        screen=InstallerScreen.FAILED,
        elapsed_seconds=305,
        quiet_seconds=305,
    )
    assert "Still working" not in stopped
    assert "taking longer than usual" not in stopped


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


def test_advanced_details_keeps_live_setup_log_location_visible(tmp_path):
    log_path = tmp_path / "installer-state" / "setup.log"
    window = object.__new__(InstallerWindow)
    window._selection = InstallerSelection(install_root=tmp_path / "managed")
    window._status_history = ["Downloading VIPP components…"]
    window._development_build = False
    window._elapsed_seconds = lambda: 75
    state = InstallerViewState(
        screen=InstallerScreen.WORKING,
        headline="Installing",
        message="Installing",
        primary_label="Working",
        primary_enabled=False,
        status_message="Downloading VIPP components…",
        progress_stage="download",
        log_path=log_path,
    )

    details = window._rendered_details(state)

    assert f"Setup log: {log_path}" in details
    assert "Current activity: Downloading VIPP components…" in details
    assert "Elapsed in this phase: 1m 15s" in details


def test_installing_keeps_log_and_diagnostics_but_not_review_size_estimates(tmp_path):
    window = object.__new__(InstallerWindow)
    window._selection = InstallerSelection()
    window._status_history = ["Installing required components"]
    state = replace(
        _state(InstallerScreen.READY, kind=TargetKind.UPDATE),
        size_estimate=InstallSizeEstimate(1024**3, 2 * 1024**3, 3 * 1024**3),
        log_path=tmp_path / "setup.jsonl",
        technical_details="Retained exact diagnostic evidence",
    )

    assert "Approximate sizes" in window._rendered_details(state)
    installing = window._rendered_details(
        replace(state, screen=InstallerScreen.WORKING)
    )
    assert "Approximate sizes" not in installing
    assert f"Setup log: {state.log_path}" in installing
    assert "Retained exact diagnostic evidence" in installing


def test_open_setup_log_uses_exact_live_log_path(tmp_path, monkeypatch):
    log_path = tmp_path / "installer-state" / "setup.log"
    log_path.parent.mkdir()
    log_path.write_text("installer event", encoding="utf-8")
    opened = []
    errors = []
    monkeypatch.setattr(
        gui_module.os,
        "startfile",
        lambda path: opened.append(Path(path)),
        raising=False,
    )
    window = object.__new__(InstallerWindow)
    window.root = object()
    window._messagebox = SimpleNamespace(
        showerror=lambda *args, **kwargs: errors.append((args, kwargs))
    )
    window._state = InstallerViewState(
        screen=InstallerScreen.WORKING,
        headline="Installing",
        message="Installing",
        primary_label="Working",
        primary_enabled=False,
        log_path=log_path,
    )

    window._open_setup_log()

    assert opened == [log_path]
    assert errors == []


def test_open_setup_log_falls_back_to_notepad_without_jsonl_association(
    tmp_path,
    monkeypatch,
):
    log_path = tmp_path / "installer-state" / "setup-events.jsonl"
    log_path.parent.mkdir()
    log_path.write_text('{"event":"started"}\n', encoding="utf-8")
    fallbacks = []
    errors = []
    monkeypatch.setattr(
        gui_module.os,
        "startfile",
        lambda _path: (_ for _ in ()).throw(OSError("no file association")),
        raising=False,
    )
    monkeypatch.setattr(
        gui_module,
        "subprocess",
        SimpleNamespace(
            Popen=lambda argv, **kwargs: fallbacks.append((tuple(argv), kwargs))
        ),
        raising=False,
    )
    window = object.__new__(InstallerWindow)
    window.root = object()
    window._messagebox = SimpleNamespace(
        showerror=lambda *args, **kwargs: errors.append((args, kwargs))
    )
    window._state = InstallerViewState(
        screen=InstallerScreen.WORKING,
        headline="Installing",
        message="Installing",
        primary_label="Working",
        primary_enabled=False,
        log_path=log_path,
    )

    window._open_setup_log()

    assert fallbacks
    assert fallbacks[0][0] == ("notepad.exe", str(log_path))
    assert errors == []


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


def test_installer_header_stacks_only_below_measured_inline_width():
    required = _header_inline_width(287, 263)

    assert _header_should_stack(520, 287, 263)
    assert not _header_should_stack(760, 287, 263)
    assert not _header_should_stack(required, 287, 263)
    assert _header_should_stack(required - 1, 287, 263)


def test_installer_header_minimum_width_preserves_each_unwrapped_row():
    assert _header_minimum_width(287, 263) == 520
    assert _header_minimum_width(287, 600) == 640


def test_installer_header_layout_moves_whole_unwrapped_tagline():
    class _Header:
        def __init__(self):
            self.columns = []

        def columnconfigure(self, column, *, weight):
            self.columns.append((column, weight))

    class _Label:
        def __init__(self):
            self.layouts = []

        def grid(self, **kwargs):
            self.layouts.append(kwargs)

    window = object.__new__(InstallerWindow)
    window._header = _Header()
    window._brand_label = _Label()
    window._tagline_label = _Label()
    window._header_stacked = None

    window._set_header_layout(stacked=True)
    window._set_header_layout(stacked=False)

    assert window._brand_label.layouts[0]["row"] == 0
    assert window._tagline_label.layouts[0]["row"] == 1
    assert window._brand_label.layouts[1]["row"] == 0
    assert window._tagline_label.layouts[1]["row"] == 0
    assert "wraplength=0" in (
        Path(__file__).resolve().parents[1] / "installer" / "gui.py"
    ).read_text(encoding="utf-8")


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
            window._content_mousewheel(SimpleNamespace(delta=delta, widget=object()))
            == "break"
        )
        assert canvas.scrolls[-1] == (steps, "units")

    assert (
        window._content_mousewheel(SimpleNamespace(delta=-120, widget=window._details))
        is None
    )
    assert len(canvas.scrolls) == 6

    canvas.bounds = (0, 0, 640, 500)
    assert (
        window._content_mousewheel(SimpleNamespace(delta=-120, widget=object())) is None
    )
    assert len(canvas.scrolls) == 6


def test_advanced_toggle_reveals_panel_without_moving_diagnostic_text():
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
    assert window._details.seen == []
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
    window._strategy_help_label = _Label()
    window._content_canvas = _Canvas(bounds=None)

    window._content_resized(_Event())

    assert configured == [648, 648, 648, 648, 648]


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
    window._update_strategy = _Value("Fresh installation")
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
    rendered_details = []
    window._replace_details = rendered_details.append

    window._mark_settings_dirty()

    action, invalidated = window._controller.calls[-1]
    assert action == "invalidate"
    assert invalidated.track.value == "cuda13"
    assert invalidated.install_root == tmp_path / "custom GPU root"
    assert invalidated.create_desktop_shortcut is False
    assert invalidated.update_strategy is UpdateStrategy.FRESH
    assert window._primary.configured[-1] == {"state": "disabled"}
    assert window._secondary.configured[-1] == {"state": "disabled"}
    assert not any(call == "confirm" for call in window._controller.calls)
    assert rendered_details
    assert (
        f"Requested location: {tmp_path / 'custom GPU root'}" in (rendered_details[-1])
    )

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


def test_console_buffer_is_bounded_and_drains_in_order():
    buffer = gui_module._ConsoleBuffer(limit=20)
    buffer.push("first\n")
    buffer.push("second\n")
    assert buffer.drain(3) == "fir"
    assert buffer.drain(30) == "st\nsecond\n"
    assert buffer.drain() == ""
    buffer.push("discarded\n")
    buffer.push("newest output\n")
    assert buffer.drain() == (
        "[Some older live output omitted from this view.]\nnewest output\n"
    )
    buffer.push("x" * 100)
    assert buffer.drain().endswith("x" * 20)
    assert buffer.drain() == ""


def test_elapsed_refresh_does_not_rewrite_diagnostics_or_console():
    window = object.__new__(InstallerWindow)
    window._state = _state(InstallerScreen.WORKING)
    window._last_elapsed_second = 0
    window._elapsed_seconds = lambda: 2
    window._quiet_seconds = lambda: 1
    window._status_history = ["Installing components…"]
    window._activity = _Value()
    window._replace_details = lambda *_: pytest.fail("Timer rewrote diagnostics")
    window._append_console = lambda *_: pytest.fail("Timer rewrote console")

    window._refresh_elapsed()

    assert "Elapsed in this phase: 2s" in window._activity.get()


@pytest.fixture
def tk_log_window():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"Tk display unavailable: {exc}")
    root.withdraw()
    root.geometry("500x220")
    # Exercise real Text widget scrolling, without showing a test window on
    # Windows. Linux CI provides a virtual display for Tk/Qt tests.
    if sys.platform == "win32":
        root.attributes("-alpha", 0.0)
    widget = tk.Text(root, height=8, width=50, wrap="word", state="disabled")
    widget.pack(fill="both", expand=True)
    root.deiconify()
    root.update()
    window = object.__new__(InstallerWindow)
    window._console = widget
    window._details = widget
    window._details_text = ""
    try:
        yield window, root
    finally:
        root.destroy()


def test_live_console_follows_bottom_but_keeps_scrolled_text(tk_log_window):
    window, root = tk_log_window
    window._append_console("".join(f"package {i:03d}\n" for i in range(100)))
    root.update()
    assert window._console.yview()[1] == pytest.approx(1.0)

    window._console.yview("25.0")
    root.update()
    before = window._console.get("@0,0 linestart", "@0,0 lineend")
    window._append_console("Downloading next-package.whl\n")
    root.update()
    assert window._console.get("@0,0 linestart", "@0,0 lineend") == before
    assert window._console.yview()[1] < 1.0
    assert "Downloading next-package.whl" in window._console.get("1.0", "end")

    window._follow_console()
    root.update()
    window._append_console("Successfully installed next-package\n")
    root.update()
    assert window._console.yview()[1] == pytest.approx(1.0)
    assert window._console.cget("state") == "disabled"


def test_live_console_trim_keeps_reading_anchor_and_bounds_memory(
    tk_log_window, monkeypatch
):
    window, root = tk_log_window
    monkeypatch.setattr(gui_module, "_CONSOLE_LINE_LIMIT", 100)
    monkeypatch.setattr(gui_module, "_CONSOLE_VISIBLE_LIMIT", 10000)
    window._append_console("".join(f"package {i:03d}\n" for i in range(90)))
    root.update()
    window._console.yview("35.0")
    root.update()
    before = window._console.get("@0,0 linestart", "@0,0 lineend")
    window._append_console("".join(f"package {i:03d}\n" for i in range(90, 110)))
    root.update()
    assert window._console.get("@0,0 linestart", "@0,0 lineend") == before
    assert int(window._console.index("end-1c").split(".")[0]) <= 100
    window._append_console("z" * 20000)
    assert len(window._console.get("1.0", "end-1c")) <= 10000


def test_diagnostic_changes_preserve_scroll_and_skip_identical_text(tk_log_window):
    window, root = tk_log_window
    details = "\n".join(f"diagnostic {i}" for i in range(100))
    window._replace_details(details)
    root.update()
    window._details.yview_moveto(0.4)
    root.update()
    position = window._details.yview()[0]
    window._details.tag_add("sel", "40.0", "40.end")
    selected = window._details.tag_ranges("sel")
    window._replace_details(details)
    assert window._details.tag_ranges("sel") == selected
    assert window._details.yview()[0] == pytest.approx(position)
    window._replace_details(details.replace("diagnostic", "information"))
    root.update()
    assert window._details.yview()[0] == pytest.approx(position, abs=0.002)


def test_complete_setup_window_streams_without_advanced_details(
    tk_log_window, monkeypatch
):
    _, root = tk_log_window
    for child in root.winfo_children():
        child.destroy()
    controller = _Controller()
    controller.state = _state(InstallerScreen.CHECKING)
    controller.cancel = lambda: None
    listener = {}

    def make_controller(backend, state_listener, *, console_listener):
        listener["state"] = state_listener
        listener["console"] = console_listener
        return controller

    monkeypatch.setattr(gui_module, "InstallerController", make_controller)
    window = InstallerWindow(root, object())
    root.update()
    listener["state"](_state(InstallerScreen.WORKING))
    listener["console"]("Downloading component.whl (25 MB)\n")
    listener["console"]("Installing collected packages: component\n")
    window._poll_states()
    root.update()
    assert not window._show_advanced.get()
    assert window._console.winfo_ismapped()
    assert "Downloading component.whl" in window._console.get("1.0", "end")
    assert "Installing collected packages" in window._console.get("1.0", "end")
    assert window._button_bar.winfo_y() + window._button_bar.winfo_height() <= (
        window._button_bar.master.winfo_height()
    )
