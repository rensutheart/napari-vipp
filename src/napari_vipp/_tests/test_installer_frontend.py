from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp.installer import frontend as frontend_module
from napari_vipp.installer.frontend import (
    IN_PLACE_CLOSE_WARNING,
    BlockedAction,
    InstallationCancelled,
    InstallerController,
    InstallerScreen,
    InstallerSelection,
    InstallOutcome,
    InstallSizeEstimate,
    PreparedInstall,
    ProgressUnit,
    ProgressUpdate,
    RegistrationRepairReview,
    TargetKind,
    VippMustBeClosed,
    default_install_size_estimate,
)
from napari_vipp.installer.models import ComputeTrack, UpdateStrategy


class _QueuedWorker:
    def __init__(self, target, queue):
        self._target = target
        self._queue = queue

    def start(self):
        self._queue.append(self._target)


class _Backend:
    def __init__(self, prepared: PreparedInstall):
        self.prepared = prepared
        self.inspect_calls = []
        self.apply_calls = []
        self.opened = []
        self.opened_uninstallers = []
        self.cancel_apply = False
        self.cancel_after_success = False
        self.inspect_error = None
        self.apply_error = None
        self.registration_repairs = []
        self.registration_repair_error = None

    def inspect(self, selection, *, progress, cancellation, repair=False):
        self.inspect_calls.append((selection, repair))
        progress(ProgressUpdate("discovery", "Checking the installation…", 1, 2))
        if self.inspect_error is not None:
            raise self.inspect_error
        if repair:
            return PreparedInstall(
                kind=TargetKind.REPAIR,
                target=self.prepared.target,
                release_version=self.prepared.release_version,
                track=self.prepared.track,
                plain_summary="Setup will restore VIPP's program files.",
                technical_details="repair transaction",
                payload="repair-payload",
            )
        return self.prepared

    def apply(self, prepared, *, confirmed, progress, cancellation):
        self.apply_calls.append((prepared, confirmed))
        progress(ProgressUpdate("packages", "Installing VIPP…", 1, 3))
        if self.cancel_apply or cancellation.is_set():
            raise InstallationCancelled(details="rollback completed")
        if self.apply_error is not None:
            raise self.apply_error
        outcome = InstallOutcome(
            launcher=prepared.target / "Scripts" / "vipp-app.exe",
            technical_details="acceptance passed",
        )
        if self.cancel_after_success:
            cancellation.set()
        return outcome

    def open_vipp(self, launcher):
        self.opened.append(launcher)

    def open_owned_uninstaller(self, prepared):
        self.opened_uninstallers.append(prepared)

    def repair_registration(self, prepared, *, confirmed, progress):
        self.registration_repairs.append((prepared, confirmed))
        if self.registration_repair_error is not None:
            raise self.registration_repair_error
        progress(
            ProgressUpdate("repairing_registration", "Fake backup and repair", 1, 1)
        )
        self.prepared = replace(
            prepared,
            kind=TargetKind.UPDATE,
            blocked_action=BlockedAction.RETRY,
            registration_repair=None,
            payload="freshly checked update",
        )


def _prepared(kind: TargetKind, tmp_path: Path) -> PreparedInstall:
    mutable = kind in {TargetKind.NEW, TargetKind.UPDATE, TargetKind.REPAIR}
    launcher = (
        tmp_path / "managed" / "Scripts" / "vipp-app.exe"
        if kind in {TargetKind.CURRENT, TargetKind.NEWER}
        else None
    )
    return PreparedInstall(
        kind=kind,
        target=tmp_path / "managed",
        release_version="0.13.0a4",
        installed_version=("0.13.0a3" if kind is TargetKind.UPDATE else None),
        track=ComputeTrack.CPU,
        plain_summary="VIPP will be installed in its own safe location.",
        technical_details="reviewed package transaction",
        payload="transaction" if mutable else None,
        launcher=launcher,
        required_free_bytes=5 * 1024**3,
    )


@pytest.mark.parametrize(
    "strategy",
    [UpdateStrategy.IN_PLACE, UpdateStrategy.INCREMENTAL, UpdateStrategy.FRESH],
)
def test_working_copy_warns_only_for_in_place_and_explains_package_reuse(
    tmp_path, strategy
):
    workers = []
    backend = _Backend(
        replace(_prepared(TargetKind.UPDATE, tmp_path), selected_strategy=strategy)
    )
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    assert controller.state.screen is InstallerScreen.WORKING
    assert (IN_PLACE_CLOSE_WARNING in controller.state.message) is (
        strategy is UpdateStrategy.IN_PLACE
    )
    if strategy is UpdateStrategy.IN_PLACE:
        assert "Updating changed packages" in controller.state.message
        assert "Compatible installed versions are retained" in controller.state.message
        assert (
            "cached downloads are reused for required changes"
            in controller.state.message
        )
        assert "dependencies stay unchanged" not in controller.state.message
        assert "previous installation stays available" not in controller.state.message
    assert backend.apply_calls == []


def test_typed_running_app_blocker_requires_manual_close_then_new_review(tmp_path):
    workers = []
    backend = _Backend(
        replace(
            _prepared(TargetKind.UPDATE, tmp_path),
            selected_strategy=UpdateStrategy.IN_PLACE,
        )
    )
    backend.apply_error = VippMustBeClosed(details="Exact runtime user was detected.")
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    started = controller.state.installation_started_at
    workers.pop(0)()
    state = controller.state
    assert state.screen is InstallerScreen.FAILED
    assert state.close_vipp_required is True
    assert state.headline == "Close VIPP to continue"
    assert state.primary_label == "Check again"
    assert "Save your work and close VIPP yourself" in state.message
    assert "not close VIPP for you" in state.message
    assert "automatically" in state.message
    assert "Exact runtime user was detected" in state.technical_details
    assert state.installation_started_at == started and started is not None
    assert state.installation_finished_at is not None
    assert controller._prepared is None
    assert controller._prepared_selection is None
    controller.confirm()
    controller.open_vipp()
    assert len(backend.apply_calls) == 1
    assert backend.opened == []
    controller.retry()
    assert controller.state.screen is InstallerScreen.CHECKING
    assert controller.state.close_vipp_required is False
    workers.pop(0)()
    assert controller.state.screen is InstallerScreen.READY
    assert len(backend.inspect_calls) == 2
    assert len(backend.apply_calls) == 1  # Recheck is not a fresh approval.


def test_typed_inspection_blocker_does_not_invent_started_installation(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.UPDATE, tmp_path))
    backend.inspect_error = VippMustBeClosed(details="An exact running user was found.")
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    assert controller.state.close_vipp_required is True
    assert controller.state.primary_label == "Check again"
    assert controller.state.installation_started_at is None
    assert controller.state.installation_finished_at is None
    assert backend.apply_calls == []
    assert backend.opened == []


def test_unknown_failure_text_is_not_a_positive_running_app_signal(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.UPDATE, tmp_path))
    backend.inspect_error = RuntimeError(
        "Close VIPP if it is running; identity unknown."
    )
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    assert controller.state.screen is InstallerScreen.FAILED
    assert controller.state.close_vipp_required is False
    assert controller.state.headline != "Close VIPP to continue"
    assert backend.apply_calls == []


@pytest.mark.parametrize(
    ("kind", "headline", "button"),
    [
        (TargetKind.NEW, "Ready to install VIPP", "Install VIPP"),
        (TargetKind.UPDATE, "A VIPP update is ready", "Update VIPP"),
        (TargetKind.REPAIR, "VIPP can be repaired", "Repair VIPP"),
    ],
)
def test_new_update_and_repair_require_one_explicit_confirmation(
    tmp_path,
    kind,
    headline,
    button,
):
    workers = []
    states = []
    backend = _Backend(_prepared(kind, tmp_path))
    controller = InstallerController(
        backend,
        states.append,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    assert backend.apply_calls == []
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.READY
    assert controller.state.headline == headline
    assert controller.state.primary_label == button
    assert backend.apply_calls == []

    controller.confirm()
    assert backend.apply_calls == []
    workers.pop(0)()

    assert backend.apply_calls[0][1] is True
    assert controller.state.screen is InstallerScreen.SUCCESS
    assert controller.state.primary_label == "Open VIPP"
    assert "acceptance passed" in controller.state.technical_details


def test_gpu_install_explains_that_the_large_download_can_pause(tmp_path):
    workers = []
    prepared = replace(
        _prepared(TargetKind.NEW, tmp_path),
        track=ComputeTrack.CUDA13,
        required_free_bytes=15 * 1024**3,
        temporary_free_bytes=5 * 1024**3,
    )
    backend = _Backend(prepared)
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()
    status = controller.state.status_message
    assert "at least 15 GiB of free disk space on the installation drive" in status
    assert "during setup" in status
    assert "Windows temporary files and VIPP installer records" in status
    assert "at least 5 GiB" in status
    assert "every drive they use" in status
    assert "VRAM" not in status
    assert "GPU memory" not in status
    controller.confirm()

    assert controller.state.screen is InstallerScreen.WORKING
    assert "large download" in controller.state.message
    assert "several minutes" in controller.state.message
    assert "can be" in controller.state.message
    assert "cached" in controller.state.message


def test_ready_state_keeps_estimates_separate_from_enforced_space_limits(tmp_path):
    workers = []
    estimate = InstallSizeEstimate(
        download_bytes=250 * 1024**2,
        installed_bytes=1536 * 1024**2,
        peak_temporary_bytes=2560 * 1024**2,
    )
    prepared = replace(
        _prepared(TargetKind.NEW, tmp_path),
        size_estimate=estimate,
        required_free_bytes=5 * 1024**3,
        temporary_free_bytes=1 * 1024**3,
    )
    controller = InstallerController(
        _Backend(prepared),
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()

    assert controller.state.size_estimate == estimate
    assert controller.state.temporary_free_bytes == 1 * 1024**3
    assert controller.state.progress_fraction is None
    assert "at least 5 GiB of free disk space" in controller.state.status_message
    assert "at least 1 GiB" in controller.state.status_message
    assert "every drive" in controller.state.status_message


def test_release_declared_size_estimates_are_route_specific_and_validated():
    cpu = default_install_size_estimate(ComputeTrack.CPU)
    cuda = default_install_size_estimate(ComputeTrack.CUDA13)

    assert cpu.download_bytes == 250 * 1024**2
    assert cpu.installed_bytes == 1536 * 1024**2
    assert cpu.peak_temporary_bytes == 2560 * 1024**2
    assert cuda.download_bytes == 1536 * 1024**2
    assert cuda.installed_bytes == 5 * 1024**3
    assert cuda.peak_temporary_bytes == 7 * 1024**3
    with pytest.raises(ValueError, match="download_bytes"):
        InstallSizeEstimate(-1, 1, 1)


def test_progress_units_distinguish_byte_progress_from_unknown_activity():
    byte_progress = ProgressUpdate(
        "download",
        "Downloading dependencies…",
        completed=128 * 1024**2,
        total=512 * 1024**2,
        unit=ProgressUnit.BYTES,
    )
    unknown_activity = ProgressUpdate(
        "installing",
        "Installing dependencies…",
        completed=1,
        total=4,
        unit=ProgressUnit.ACTIVITY,
    )

    assert byte_progress.fraction == pytest.approx(0.25)
    assert unknown_activity.fraction is None


@pytest.mark.parametrize(
    "unit", [ProgressUnit.FILES, ProgressUnit.PACKAGES, ProgressUnit.CHECKS]
)
def test_controller_preserves_exact_measured_counts_through_console_and_rollback(
    tmp_path, unit
):
    workers = []
    controller = InstallerController(
        _Backend(_prepared(TargetKind.UPDATE, tmp_path)),
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    generation = controller._generation
    controller._on_progress(
        generation, ProgressUpdate("installing", "Measured task", 23, 100, unit)
    )
    measured = controller.state
    assert measured.progress_completed == 23
    assert measured.progress_total == 100
    assert measured.progress_unit is unit
    assert measured.progress_fraction == pytest.approx(0.23)
    assert measured.installation_stage == "installing"
    controller._on_progress(
        generation, ProgressUpdate("installing", "", console_text="pip output\n")
    )
    assert controller.state == measured
    controller._on_progress(
        generation,
        ProgressUpdate("rolling_back", "Cleaning up", unit=ProgressUnit.ACTIVITY),
    )
    assert controller.state.installation_stage == "installing"
    assert controller.state.progress_completed is None
    assert controller.state.progress_total is None
    controller._fail(
        "Stopped", "Safe failure", RuntimeError("test only"), generation=generation
    )
    assert controller.state.installation_stage == "installing"
    assert controller.state.progress_fraction is None


def test_cancel_clears_task_counts_but_preserves_last_installation_stage(tmp_path):
    workers = []
    controller = InstallerController(
        _Backend(_prepared(TargetKind.NEW, tmp_path)),
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    controller._on_progress(
        controller._generation,
        ProgressUpdate("verifying", "Checking", 2, 3, ProgressUnit.CHECKS),
    )
    controller.cancel()
    assert controller.state.screen is InstallerScreen.CANCELLING
    assert controller.state.installation_stage == "verifying"
    assert controller.state.progress_completed is None
    assert controller.state.progress_total is None
    assert controller.state.progress_fraction is None
    controller._show_cancelled(controller._generation)
    assert controller.state.installation_stage == "verifying"
    controller.retry()
    assert controller.state.installation_stage == ""
    assert controller.state.progress_completed is None
    assert controller.state.progress_total is None


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_installation_clock_begins_at_accepted_approval_and_freezes_at_terminal(
    tmp_path, monkeypatch, outcome
):
    now = [10.0]
    monkeypatch.setattr(frontend_module.time, "monotonic", lambda: now[0])
    workers = []
    backend = _Backend(_prepared(TargetKind.UPDATE, tmp_path))
    backend.apply_error = RuntimeError("test failure") if outcome == "failure" else None
    backend.cancel_apply = outcome == "cancelled"
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    assert controller.state.installation_started_at is None
    controller.confirm()  # Checking is not permission to start installation.
    assert controller.state.installation_started_at is None
    workers.pop(0)()
    assert controller.state.screen is InstallerScreen.READY
    now[0] = 25.0
    controller.confirm()
    assert controller.state.installation_started_at == 25.0
    assert controller.state.installation_finished_at is None
    controller._on_progress(
        controller._generation,
        ProgressUpdate("installing", "Copying", 1, 2, ProgressUnit.FILES),
    )
    assert controller.state.installation_started_at == 25.0
    now[0] = 65.0
    workers.pop(0)()
    assert (
        controller.state.screen
        is {
            "success": InstallerScreen.SUCCESS,
            "failure": InstallerScreen.FAILED,
            "cancelled": InstallerScreen.CANCELLED,
        }[outcome]
    )
    assert controller.state.installation_started_at == 25.0
    assert controller.state.installation_finished_at == 65.0
    now[0] = 1000.0
    controller._on_progress(
        controller._generation, ProgressUpdate("completed", "Late output")
    )
    assert controller.state.installation_finished_at == 65.0
    if outcome != "success":
        controller.retry()
        assert controller.state.installation_started_at is None
        assert controller.state.installation_finished_at is None


def test_post_install_open_error_cannot_extend_completed_installation_clock(
    tmp_path, monkeypatch
):
    now = [10.0]
    monkeypatch.setattr(frontend_module.time, "monotonic", lambda: now[0])
    workers = []
    controller = InstallerController(
        _Backend(_prepared(TargetKind.NEW, tmp_path)),
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    now[0] = 40.0
    workers.pop(0)()
    now[0] = 1000.0
    controller._fail("Launch failed", "Test only", RuntimeError("launch only"))
    assert controller.state.installation_started_at == 10.0
    assert controller.state.installation_finished_at == 40.0


def test_controller_preserves_phase_progress_unit_and_live_log_path(tmp_path):
    class _PhasedBackend(_Backend):
        def apply(self, prepared, *, confirmed, progress, cancellation):
            self.apply_calls.append((prepared, confirmed))
            progress(
                ProgressUpdate(
                    "download",
                    "Downloading VIPP components…",
                    completed=3,
                    total=12,
                    unit=ProgressUnit.BYTES,
                    log_path=tmp_path / "setup.log",
                )
            )
            progress(
                ProgressUpdate(
                    "installing",
                    "Installing dependencies…",
                    unit=ProgressUnit.ACTIVITY,
                    log_path=tmp_path / "setup.log",
                )
            )
            return InstallOutcome(launcher=prepared.target / "Scripts" / "vipp-app.exe")

    workers = []
    states = []
    backend = _PhasedBackend(_prepared(TargetKind.NEW, tmp_path))
    controller = InstallerController(
        backend,
        states.append,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    workers.pop(0)()

    download = next(state for state in states if state.progress_stage == "download")
    installing = next(state for state in states if state.progress_stage == "installing")
    assert download.progress_unit is ProgressUnit.BYTES
    assert download.progress_fraction == pytest.approx(0.25)
    assert download.log_path == tmp_path / "setup.log"
    assert installing.progress_unit is ProgressUnit.ACTIVITY
    assert installing.progress_fraction is None
    assert installing.log_path == tmp_path / "setup.log"
    assert controller.state.screen is InstallerScreen.SUCCESS
    assert controller.state.log_path == tmp_path / "setup.log"


@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_console_chunks_do_not_publish_state_and_survive_terminal_outcomes(
    tmp_path,
    outcome,
):
    workers = []
    states = []
    console = []
    chunks = ["Collecting dependencies\n", "Downloading ", "50%\r", "100%\r"]
    chunks += ["same output\n"] * 3
    chunks += [f"package {index}\n" for index in range(100)]
    chunks += [f"subprocess {outcome}\n"]

    class _LoggingBackend(_Backend):
        def apply(self, prepared, *, confirmed, progress, cancellation):
            self.apply_calls.append((prepared, confirmed))
            self.progress_callback = progress
            progress(
                ProgressUpdate(
                    "download",
                    "Downloading packages…",
                    1,
                    4,
                    unit=ProgressUnit.BYTES,
                    log_path=tmp_path / "setup.log",
                )
            )
            before = controller.state
            published_count = len(states)
            for chunk in chunks:
                progress(ProgressUpdate("download", "", console_text=chunk))
                assert controller.state is before
                assert len(states) == published_count
            assert before.progress_fraction == pytest.approx(0.25)
            assert before.progress_stage == "download"
            assert before.progress_unit is ProgressUnit.BYTES
            assert before.status_message == "Downloading packages…"
            assert before.log_path == tmp_path / "setup.log"
            if outcome == "failure":
                raise RuntimeError("dependency command failed")
            if outcome == "cancelled":
                raise InstallationCancelled(details="rollback completed")
            return InstallOutcome(launcher=prepared.target / "Scripts/vipp-app.exe")

    backend = _LoggingBackend(_prepared(TargetKind.NEW, tmp_path))
    controller = InstallerController(
        backend,
        states.append,
        console_listener=console.append,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    workers.pop(0)()

    assert console == [
        "Checking the installation…\n",
        "Downloading packages…\n",
        *chunks,
    ]
    assert (
        controller.state.screen
        is {
            "success": InstallerScreen.SUCCESS,
            "failure": InstallerScreen.FAILED,
            "cancelled": InstallerScreen.CANCELLED,
        }[outcome]
    )
    final = controller.state
    count = len(console)
    backend.progress_callback(
        ProgressUpdate("download", "", console_text="late output\n")
    )
    backend.progress_callback(ProgressUpdate("completed", "Late milestone"))
    assert len(console) == count
    assert controller.state is final


def test_console_milestones_emit_once_without_coalescing_raw_output(tmp_path):
    workers = []
    states = []
    console = []
    controller = InstallerController(
        _Backend(_prepared(TargetKind.NEW, tmp_path)),
        states.append,
        console_listener=console.append,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    generation = controller._generation
    for completed in (1, 2):
        controller._on_progress(
            generation,
            ProgressUpdate(
                "download",
                "Downloading packages…\n",
                completed,
                4,
                ProgressUnit.BYTES,
            ),
        )
        controller._on_progress(
            generation,
            ProgressUpdate(
                "download",
                "",
                console_text="same chunk\n",
            ),
        )
    controller._on_progress(generation, ProgressUpdate("installing", "Installing…"))
    assert console == [
        "Downloading packages…\n",
        "same chunk\n",
        "same chunk\n",
        "Installing…\n",
    ]
    download_states = [state for state in states if state.progress_stage == "download"]
    assert [state.progress_fraction for state in download_states] == [0.25, 0.5]


def test_console_and_milestones_from_stale_generation_are_dropped(tmp_path):
    workers = []
    states = []
    console = []
    controller = InstallerController(
        _Backend(_prepared(TargetKind.NEW, tmp_path)),
        states.append,
        console_listener=console.append,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    stale_generation = controller._generation
    changed = InstallerSelection(create_desktop_shortcut=False)
    assert controller.invalidate_selection(changed)
    controller.start(changed)
    before = controller.state
    assert controller.busy
    published_count = len(states)
    for update in (
        ProgressUpdate("checking", "", console_text="old check output\n"),
        ProgressUpdate("checking", "Old check milestone"),
    ):
        controller._on_progress(stale_generation, update)
    assert console == []
    assert len(states) == published_count
    assert controller.state is before
    controller._on_progress(
        controller._generation,
        ProgressUpdate(
            "checking",
            "",
            console_text="current check output\n",
        ),
    )
    assert console == ["current check output\n"]
    assert controller.state is before


def test_console_listener_is_optional_and_positional_progress_fields_still_work(
    tmp_path,
):
    log_path = tmp_path / "setup.log"
    milestone = ProgressUpdate(
        "download", "Downloading…", 2, 4, ProgressUnit.BYTES, log_path
    )
    assert milestone.console_text == ""
    workers = []
    states = []
    controller = InstallerController(
        _Backend(_prepared(TargetKind.NEW, tmp_path)),
        states.append,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    controller._on_progress(controller._generation, milestone)
    before = controller.state
    published_count = len(states)
    controller._on_progress(
        controller._generation,
        ProgressUpdate(
            "download",
            "",
            console_text="Output with no listener\n",
        ),
    )
    assert controller.state is before
    assert len(states) == published_count
    assert before.progress_fraction == pytest.approx(0.5)


def test_current_install_opens_or_prepares_explicit_repair(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.CURRENT, tmp_path))
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.CURRENT
    assert controller.state.primary_label == "Open VIPP"
    assert controller.state.secondary_label == "Repair VIPP"
    controller.open_vipp()
    assert backend.opened == [backend.prepared.launcher]
    assert backend.apply_calls == []

    controller.request_repair()
    workers.pop(0)()
    assert backend.inspect_calls[-1][1] is True
    assert controller.state.screen is InstallerScreen.READY
    assert controller.state.primary_label == "Repair VIPP"
    assert backend.apply_calls == []


@pytest.mark.parametrize("kind", [TargetKind.FOREIGN, TargetKind.BLOCKED])
def test_unowned_or_blocked_target_can_never_be_confirmed(tmp_path, kind):
    workers = []
    backend = _Backend(_prepared(kind, tmp_path))
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()
    controller.confirm()

    assert controller.state.screen is InstallerScreen.BLOCKED
    assert backend.apply_calls == []
    assert workers == []
    if kind is TargetKind.FOREIGN:
        assert controller.state.primary_label == "Check again"
        assert "Nothing has been changed" in controller.state.status_message


def test_cuda_path_blocker_exposes_typed_cpu_action(tmp_path):
    workers = []
    prepared = replace(
        _prepared(TargetKind.BLOCKED, tmp_path),
        track=ComputeTrack.CUDA13,
        blocked_action=BlockedAction.USE_CPU,
        reason="Use CPU one-click setup on this Windows account.",
    )
    controller = InstallerController(
        _Backend(prepared),
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.BLOCKED
    assert controller.state.blocked_action is BlockedAction.USE_CPU
    assert controller.state.primary_label == "Use CPU"
    assert "CPU one-click" in controller.state.status_message


def test_custom_managed_root_exposes_typed_default_location_action(tmp_path):
    workers = []
    prepared = replace(
        _prepared(TargetKind.BLOCKED, tmp_path),
        blocked_action=BlockedAction.USE_DEFAULT_LOCATION,
        reason="Use VIPP's exact per-account Windows default folder.",
    )
    controller = InstallerController(
        _Backend(prepared),
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()

    assert controller.state.blocked_action is BlockedAction.USE_DEFAULT_LOCATION
    assert controller.state.primary_label == "Use default location"


def test_hash_bound_uninstaller_action_is_typed_and_controller_owned(tmp_path):
    workers = []
    prepared = replace(
        _prepared(TargetKind.BLOCKED, tmp_path),
        blocked_action=BlockedAction.RUN_OWNED_UNINSTALLER,
        ownership_manifest_sha256="a" * 64,
        owned_uninstaller_path=tmp_path / "cache" / "VIPP-Setup.exe",
        owned_uninstaller_sha256="b" * 64,
    )
    backend = _Backend(prepared)
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()
    controller.open_owned_uninstaller()

    assert controller.state.primary_label == "Open VIPP uninstaller"
    assert backend.opened_uninstallers == [prepared]


def test_setup_recovery_help_has_no_generic_retry_or_python_action(tmp_path):
    workers = []
    prepared = replace(
        _prepared(TargetKind.BLOCKED, tmp_path),
        blocked_action=BlockedAction.OPEN_SETUP_HELP,
        help_url="https://example.invalid/installation/#troubleshooting",
        reason="Try again cannot resolve a conflicting installation record.",
    )
    backend = _Backend(prepared)
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.BLOCKED
    assert controller.state.primary_label == "Get setup help"
    assert controller.state.secondary_label == "Close"
    assert controller.state.blocked_action is BlockedAction.OPEN_SETUP_HELP
    assert "Try again cannot resolve" in controller.state.status_message
    controller.confirm()
    assert backend.apply_calls == []


def test_owned_cuda_path_blocker_exposes_installed_apps_action(tmp_path):
    workers = []
    prepared = replace(
        _prepared(TargetKind.BLOCKED, tmp_path),
        track=ComputeTrack.CUDA13,
        blocked_action=BlockedAction.OPEN_INSTALLED_APPS,
        reason="Uninstall VIPP (GPU) first, then run setup again.",
    )
    controller = InstallerController(
        _Backend(prepared),
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.BLOCKED
    assert controller.state.blocked_action is BlockedAction.OPEN_INSTALLED_APPS
    assert controller.state.primary_label == "Open Installed apps"
    assert "Uninstall VIPP (GPU) first" in controller.state.status_message


def test_newer_version_is_openable_but_never_downgraded_or_repaired(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.NEWER, tmp_path))
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()
    controller.confirm()
    controller.request_repair()

    assert controller.state.screen is InstallerScreen.CURRENT
    assert "newer VIPP" in controller.state.headline
    assert backend.apply_calls == []
    assert workers == []


def test_cancel_waits_for_backend_cleanup_before_reporting_cancelled(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.NEW, tmp_path))
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()

    controller.cancel()
    assert controller.state.screen is InstallerScreen.CANCELLING
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.CANCELLED
    assert "confirms whether every temporary file was removed" in (
        controller.state.message
    )
    assert "rollback completed" in controller.state.technical_details


def test_late_cancel_click_cannot_replace_committed_success(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.NEW, tmp_path))
    backend.cancel_after_success = True
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()

    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.SUCCESS
    assert controller.state.primary_label == "Open VIPP"


def test_errors_are_plain_in_main_view_and_trace_is_advanced(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.NEW, tmp_path))
    backend.apply_error = RuntimeError("resolver emitted technical failure 731")
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.FAILED
    assert "technical failure 731" not in controller.state.message
    assert "technical failure 731" in controller.state.technical_details
    assert controller.state.primary_label == "Try again"


@pytest.mark.parametrize(
    ("technical_error", "headline", "message_fragment"),
    [
        (
            "TLS connection failed while downloading a package",
            "could not download",
            "internet connection",
        ),
        (
            "Read timed out while fetching botocore from files.pythonhosted.org",
            "could not download",
            "choose Try again",
        ),
        (
            "Setup needs at least 5 GiB of free disk space on the volume containing "
            "C:\\Windows\\Temp for temporary downloads and installer records; "
            "2.00 GiB is available.",
            "More free space",
            "C:\\Windows\\Temp",
        ),
    ],
)
def test_common_preparation_failures_have_plain_next_steps(
    tmp_path,
    technical_error,
    headline,
    message_fragment,
):
    workers = []
    backend = _Backend(_prepared(TargetKind.NEW, tmp_path))
    backend.inspect_error = RuntimeError(technical_error)
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.FAILED
    assert headline in controller.state.headline
    assert message_fragment in controller.state.message
    assert technical_error in controller.state.technical_details


@pytest.mark.parametrize("winerror", [32, 33])
def test_shortcut_lock_has_actionable_message_and_retry_requires_review(
    tmp_path,
    winerror,
):
    workers = []
    backend = _Backend(_prepared(TargetKind.UPDATE, tmp_path))
    technical = (
        f"PermissionError: [WinError {winerror}] file in use: "
        "'.VIPP.staged.tmp' -> 'VIPP.lnk'; rollback incomplete"
    )
    backend.apply_error = RuntimeError(technical)
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.confirm()
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.FAILED
    assert controller.state.headline == "Windows kept the VIPP shortcut open"
    assert "Close VIPP" in controller.state.message
    assert "incomplete cleanup" in controller.state.message
    assert "WinError" not in controller.state.message
    assert technical in controller.state.technical_details
    backend.apply_error = None
    controller.retry()
    workers.pop(0)()
    assert controller.state.screen is InstallerScreen.READY
    assert controller.state.primary_label == "Update VIPP"
    assert len(backend.apply_calls) == 1


def test_transient_package_timeout_during_apply_has_plain_retry_guidance(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.NEW, tmp_path))
    backend.apply_error = RuntimeError(
        "ReadTimeoutError: HTTPS request to files.pythonhosted.org timed out"
    )
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()
    controller.confirm()
    workers.pop(0)()

    assert controller.state.screen is InstallerScreen.FAILED
    assert controller.state.headline == ("VIPP setup could not download its components")
    assert "Check the internet connection" in controller.state.message
    assert "choose Try again" in controller.state.message
    assert "ReadTimeoutError" not in controller.state.message
    assert "ReadTimeoutError" in controller.state.technical_details


def test_selection_rejects_ambiguous_managed_and_existing_routes(tmp_path):
    with pytest.raises(ValueError, match="cannot be selected together"):
        InstallerSelection(
            install_root=tmp_path / "managed",
            existing_python=tmp_path / "venv" / "Scripts" / "python.exe",
        )


def test_controller_does_not_require_tkinter_to_import():
    assert threading.Event is not None


class _SelectionBackend(_Backend):
    """Test backend whose reviewed target is derived from the selection."""

    def inspect(self, selection, *, progress, cancellation, repair=False):
        self.inspect_calls.append((selection, repair))
        progress(ProgressUpdate("discovery", "Checking selection…", 1, 1))
        target = selection.install_root or self.prepared.target
        return replace(self.prepared, target=target)


def test_changed_location_invalidates_ready_plan_until_exact_selection_checked(
    tmp_path,
):
    workers = []
    default_prepared = _prepared(TargetKind.NEW, tmp_path)
    backend = _SelectionBackend(default_prepared)
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()
    assert controller.state.screen is InstallerScreen.READY
    assert controller.state.target == default_prepared.target

    custom_root = tmp_path / "custom-managed-root"
    custom = InstallerSelection(install_root=custom_root)
    assert controller.invalidate_selection(custom) is True

    assert controller.state.screen is InstallerScreen.BLOCKED
    assert controller.state.primary_enabled is False
    assert "exact settings" in controller.state.status_message
    controller.confirm()
    assert backend.apply_calls == []
    assert workers == []

    controller.start(custom)
    workers.pop(0)()
    assert backend.inspect_calls[-1][0] == custom
    assert controller.state.screen is InstallerScreen.READY
    assert controller.state.target == custom_root

    controller.confirm()
    workers.pop(0)()
    assert backend.apply_calls[0][0].target == custom_root


def test_edit_during_initial_check_cannot_publish_or_apply_old_selection(tmp_path):
    workers = []
    backend = _SelectionBackend(_prepared(TargetKind.NEW, tmp_path))
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    original = InstallerSelection()
    changed = InstallerSelection(
        track="cuda13",
        install_root=tmp_path / "gpu-root",
        create_desktop_shortcut=False,
    )

    controller.start(original)
    stale_worker = workers.pop(0)
    assert controller.invalidate_selection(changed) is True
    stale_worker()

    assert backend.inspect_calls == [(original, False)]
    assert controller.state.screen is InstallerScreen.BLOCKED
    assert controller.state.primary_enabled is False
    controller.confirm()
    assert backend.apply_calls == []

    controller.start(changed)
    workers.pop(0)()
    assert backend.inspect_calls[-1] == (changed, False)
    assert controller.state.target == changed.install_root


@pytest.mark.parametrize(
    "changed",
    [
        InstallerSelection(track="cpu"),
        InstallerSelection(create_desktop_shortcut=False),
        InstallerSelection(update_strategy=UpdateStrategy.INCREMENTAL),
        InstallerSelection(update_strategy=UpdateStrategy.FRESH),
        InstallerSelection(existing_python=Path("C:/selected-venv/Scripts/python.exe")),
    ],
)
def test_every_install_relevant_selection_change_revokes_confirmation(
    tmp_path,
    changed,
):
    workers = []
    backend = _Backend(_prepared(TargetKind.NEW, tmp_path))
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()

    assert controller.invalidate_selection(changed) is True
    controller.confirm()

    assert controller.state.primary_enabled is False
    assert backend.apply_calls == []
    assert workers == []


def test_incremental_method_is_reviewed_and_preserved_through_working_and_success(
    tmp_path,
):
    workers = []
    states = []
    prepared = replace(
        _prepared(TargetKind.UPDATE, tmp_path),
        selected_strategy=UpdateStrategy.INCREMENTAL,
        strategy_reason="Reuse verified packages in a separate staged environment.",
    )
    backend = _Backend(prepared)
    controller = InstallerController(
        backend,
        states.append,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    selection = InstallerSelection(update_strategy=UpdateStrategy.INCREMENTAL)

    controller.start(selection)
    workers.pop(0)()

    assert backend.inspect_calls == [(selection, False)]
    assert controller.state.selected_strategy is UpdateStrategy.INCREMENTAL
    assert controller.state.strategy_reason == prepared.strategy_reason
    controller.confirm()
    working = controller.state
    assert working.selected_strategy is UpdateStrategy.INCREMENTAL
    assert "Staging a new VIPP copy" in working.message
    assert "temporary disk space" in working.message
    assert "current compatible dependency versions" in working.message
    workers.pop(0)()
    assert backend.apply_calls == [(prepared, True)]
    assert controller.state.screen is InstallerScreen.SUCCESS
    assert controller.state.selected_strategy is UpdateStrategy.INCREMENTAL


def test_editing_method_during_check_cannot_restore_old_confirmation(tmp_path):
    workers = []
    backend = _Backend(_prepared(TargetKind.UPDATE, tmp_path))
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    old_worker = workers.pop(0)
    selection = InstallerSelection(update_strategy=UpdateStrategy.FRESH)

    assert controller.invalidate_selection(selection)
    old_worker()
    controller.confirm()

    assert controller.state.screen is InstallerScreen.BLOCKED
    assert backend.apply_calls == []
    controller.start(selection)
    workers.pop(0)()
    assert backend.inspect_calls[-1][0].update_strategy is UpdateStrategy.FRESH
    assert controller.state.screen is InstallerScreen.READY


def _registration_blocked(tmp_path):
    return replace(
        _prepared(TargetKind.BLOCKED, tmp_path),
        blocked_action=BlockedAction.REPAIR_REGISTRATION,
        registration_repair=RegistrationRepairReview(
            "a" * 64,
            "DisplayVersion: 'old' → 'verified installed version'",
            tmp_path / "registration-backup.json",
            object(),
        ),
    )


def test_registration_review_has_structured_optional_display_facts(tmp_path):
    review = RegistrationRepairReview(
        "a" * 64,
        "Exact original technical review",
        tmp_path / "backup.json",
        object(),
        managed_root=tmp_path / "owned-installation",
        registered_version="0.16.0a8.dev4",
        installed_version="0.16.0a8.dev5",
        old_uninstaller_state="absent",
    )
    assert review.managed_root == tmp_path / "owned-installation"
    assert review.registered_version == "0.16.0a8.dev4"
    assert review.installed_version == "0.16.0a8.dev5"
    assert review.preview == "Exact original technical review"


@pytest.mark.parametrize(
    "changes",
    [
        {"managed_root": Path("relative-installation")},
        {"managed_root": None},
        {"registered_version": ""},
        {"installed_version": ""},
        {"old_uninstaller_state": "unknown"},
    ],
)
def test_registration_review_rejects_incomplete_display_facts(tmp_path, changes):
    facts = {
        "managed_root": tmp_path / "owned-installation",
        "registered_version": "old",
        "installed_version": "verified",
        "old_uninstaller_state": "absent",
    }
    facts.update(changes)
    with pytest.raises(ValueError):
        RegistrationRepairReview(
            "a" * 64, "Exact review", tmp_path / "backup.json", object(), **facts
        )


def test_registration_repair_is_separately_confirmed_then_rechecks_without_install(
    tmp_path,
):
    workers, states = [], []
    prepared = _registration_blocked(tmp_path)
    backend = _Backend(prepared)
    controller = InstallerController(
        backend,
        states.append,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    assert controller.state.primary_label == "Repair registration…"
    controller.confirm()
    controller.repair_registration(confirmed=False, fingerprint="a" * 64)
    controller.repair_registration(confirmed=True, fingerprint="b" * 64)
    assert workers == [] and backend.registration_repairs == []

    controller.repair_registration(confirmed=True, fingerprint="a" * 64)
    assert controller.state.screen is InstallerScreen.WORKING
    assert controller.state.cancel_enabled is False
    controller.cancel()
    assert not controller._cancellation.is_set()
    workers.pop(0)()

    assert backend.registration_repairs == [(prepared, True)]
    assert len(backend.inspect_calls) == 2
    assert controller.state.screen is InstallerScreen.READY
    assert any("checking again" in state.headline for state in states)
    assert "Registration-only repair completed" in controller.state.technical_details
    assert str(prepared.registration_repair.backup_path) in (
        controller.state.technical_details
    )
    assert backend.apply_calls == []
    assert backend.opened == []


def test_old_registration_confirmation_cannot_apply_after_selection_changes(tmp_path):
    workers = []
    backend = _Backend(_registration_blocked(tmp_path))
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    assert controller.invalidate_selection(InstallerSelection(track="cpu"))
    controller.repair_registration(confirmed=True, fingerprint="a" * 64)
    assert workers == []
    assert backend.registration_repairs == []


def test_registration_repair_failure_does_not_recheck_or_start_install(tmp_path):
    workers = []
    backend = _Backend(_registration_blocked(tmp_path))
    backend.registration_repair_error = RuntimeError(
        "Registration changed after preview"
    )
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )
    controller.start()
    workers.pop(0)()
    controller.repair_registration(confirmed=True, fingerprint="a" * 64)
    workers.pop(0)()
    assert controller.state.screen is InstallerScreen.FAILED
    assert "Registration changed after preview" in controller.state.technical_details
    assert str(backend.prepared.registration_repair.backup_path) in (
        controller.state.technical_details
    )
    assert len(backend.inspect_calls) == 1
    assert backend.apply_calls == [] and backend.opened == []
