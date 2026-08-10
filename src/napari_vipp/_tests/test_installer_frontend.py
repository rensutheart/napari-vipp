from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path

import pytest

from napari_vipp.installer.frontend import (
    InstallationCancelled,
    InstallerController,
    InstallerScreen,
    InstallerSelection,
    InstallOutcome,
    PreparedInstall,
    ProgressUpdate,
    TargetKind,
)
from napari_vipp.installer.models import ComputeTrack


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
        self.cancel_apply = False
        self.cancel_after_success = False
        self.inspect_error = None
        self.apply_error = None

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
    )
    backend = _Backend(prepared)
    controller = InstallerController(
        backend,
        lambda _state: None,
        worker_factory=lambda target: _QueuedWorker(target, workers),
    )

    controller.start()
    workers.pop(0)()
    assert controller.state.status_message == (
        "Setup will use GPU acceleration. It needs at least 15 GiB free on the "
        "installation drive during setup. This is disk storage, not GPU "
        "memory (VRAM)."
    )
    controller.confirm()

    assert controller.state.screen is InstallerScreen.WORKING
    assert "large download" in controller.state.message
    assert "several minutes" in controller.state.message


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
        assert controller.state.primary_label == "Choose another location"
        assert "Nothing has been changed" in controller.state.status_message


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
            "Setup needs more free space for temporary downloads",
            "More free space",
            "Free some space",
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
    assert controller.state.headline == (
        "VIPP setup could not download its components"
    )
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
        InstallerSelection(
            existing_python=Path("C:/selected-venv/Scripts/python.exe")
        ),
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
