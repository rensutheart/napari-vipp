"""Display-independent state machine for the VIPP Windows installer.

The ordinary installer is intentionally small: it checks the computer, shows
one plain-language decision, and then reports progress.  This module contains
no Tk imports so every safety transition can be tested on headless builders.
"""

from __future__ import annotations

import threading
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from napari_vipp.installer.models import ComputeTrack


class TargetKind(StrEnum):
    """How the requested target relates to a previous VIPP installation."""

    NEW = "new"
    UPDATE = "update"
    CURRENT = "current"
    REPAIR = "repair"
    NEWER = "newer"
    FOREIGN = "foreign"
    BLOCKED = "blocked"


class InstallerScreen(StrEnum):
    """Stable screens rendered by the graphical front end."""

    CHECKING = "checking"
    READY = "ready"
    WORKING = "working"
    CANCELLING = "cancelling"
    CURRENT = "current"
    BLOCKED = "blocked"
    SUCCESS = "success"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ProgressUnit(StrEnum):
    """How a progress total should be interpreted by the front end."""

    ACTIVITY = "activity"
    STEPS = "steps"
    BYTES = "bytes"


class TrackChoice(StrEnum):
    """Simple and advanced compute choices offered by the setup program."""

    AUTOMATIC = "automatic"
    CPU = "cpu"
    CUDA13 = "cuda13"


class BlockedAction(StrEnum):
    """Typed primary action for a blocked installer decision."""

    RETRY = "retry"
    OPEN_HELP = "open_help"
    OPEN_INSTALLED_APPS = "open_installed_apps"
    RUN_OWNED_UNINSTALLER = "run_owned_uninstaller"
    USE_DEFAULT_LOCATION = "use_default_location"
    USE_CPU = "use_cpu"


@dataclass(frozen=True, slots=True)
class InstallerSelection:
    """User-adjustable settings; defaults require no technical decisions."""

    track: TrackChoice = TrackChoice.AUTOMATIC
    install_root: Path | None = None
    existing_python: Path | None = None
    create_desktop_shortcut: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "track", TrackChoice(self.track))
        if self.install_root is not None:
            object.__setattr__(self, "install_root", Path(self.install_root))
        if self.existing_python is not None:
            object.__setattr__(self, "existing_python", Path(self.existing_python))
        if self.install_root is not None and self.existing_python is not None:
            raise ValueError(
                "A managed install location and an existing environment cannot "
                "be selected together."
            )


@dataclass(frozen=True, slots=True)
class InstallSizeEstimate:
    """Rounded route-level storage estimates shown before mutation begins."""

    download_bytes: int
    installed_bytes: int
    peak_temporary_bytes: int

    def __post_init__(self) -> None:
        for name in ("download_bytes", "installed_bytes", "peak_temporary_bytes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer byte count.")


_CPU_SIZE_ESTIMATE = InstallSizeEstimate(
    download_bytes=250 * 1024**2,
    installed_bytes=3 * 1024**3 // 2,
    peak_temporary_bytes=5 * 1024**3 // 2,
)
_CUDA13_SIZE_ESTIMATE = InstallSizeEstimate(
    download_bytes=3 * 1024**3 // 2,
    installed_bytes=5 * 1024**3,
    peak_temporary_bytes=7 * 1024**3,
)


def default_install_size_estimate(track: ComputeTrack) -> InstallSizeEstimate:
    """Return conservative rounded estimates from retained qualified installs."""

    track = ComputeTrack(track)
    return _CUDA13_SIZE_ESTIMATE if track is ComputeTrack.CUDA13 else _CPU_SIZE_ESTIMATE


def default_temporary_free_bytes(track: ComputeTrack) -> int:
    """Return the unchanged per-volume temporary/state free-space threshold."""

    return 5 * 1024**3 if ComputeTrack(track) is ComputeTrack.CUDA13 else 1024**3


@dataclass(frozen=True, slots=True)
class ProgressUpdate:
    """One backend milestone normalized for any future platform front end."""

    stage: str
    message: str
    completed: int | None = None
    total: int | None = None
    unit: ProgressUnit = ProgressUnit.STEPS
    log_path: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "unit", ProgressUnit(self.unit))
        if self.log_path is not None:
            object.__setattr__(self, "log_path", Path(self.log_path))

    @property
    def fraction(self) -> float | None:
        if self.unit is ProgressUnit.ACTIVITY:
            return None
        if self.completed is None or self.total is None or self.total <= 0:
            return None
        return min(1.0, max(0.0, self.completed / self.total))


@dataclass(frozen=True, slots=True)
class PreparedInstall:
    """A reviewed executor transaction translated for a nontechnical user."""

    kind: TargetKind
    target: Path
    release_version: str
    track: ComputeTrack | None
    plain_summary: str
    technical_details: str
    payload: object | None = field(default=None, repr=False, compare=False)
    installed_version: str | None = None
    required_free_bytes: int | None = None
    temporary_free_bytes: int | None = None
    size_estimate: InstallSizeEstimate | None = None
    launcher: Path | None = None
    reason: str = ""
    help_url: str = ""
    blocked_action: BlockedAction = BlockedAction.RETRY
    ownership_manifest_sha256: str = ""
    owned_uninstaller_path: Path | None = None
    owned_uninstaller_sha256: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", TargetKind(self.kind))
        object.__setattr__(self, "target", Path(self.target))
        if self.track is not None:
            object.__setattr__(self, "track", ComputeTrack(self.track))
        object.__setattr__(self, "blocked_action", BlockedAction(self.blocked_action))
        if self.launcher is not None:
            object.__setattr__(self, "launcher", Path(self.launcher))
        for name in ("required_free_bytes", "temporary_free_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer byte count.")
        if self.owned_uninstaller_path is not None:
            object.__setattr__(
                self,
                "owned_uninstaller_path",
                Path(self.owned_uninstaller_path),
            )
        fallback_values = (
            self.ownership_manifest_sha256,
            self.owned_uninstaller_path,
            self.owned_uninstaller_sha256,
        )
        if any(fallback_values) != all(fallback_values):
            raise ValueError(
                "An owned uninstaller fallback requires its path and both hashes."
            )
        if self.blocked_action is BlockedAction.RUN_OWNED_UNINSTALLER and not all(
            fallback_values
        ):
            raise ValueError(
                "The owned-uninstaller action requires a hash-bound fallback."
            )
        if self.kind in {
            TargetKind.NEW,
            TargetKind.UPDATE,
            TargetKind.REPAIR,
        } and (self.payload is None or self.track is None):
            raise ValueError(
                f"{self.kind.value} requires a track and executable transaction."
            )


@dataclass(frozen=True, slots=True)
class InstallOutcome:
    """Successful transactional installation result used by the front end."""

    launcher: Path
    message: str = "VIPP is ready to use."
    technical_details: str = ""
    log_path: Path | None = None
    payload: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "launcher", Path(self.launcher))
        if self.log_path is not None:
            object.__setattr__(self, "log_path", Path(self.log_path))


class InstallationCancelled(RuntimeError):
    """Raised when the backend completed a safe cancellation and rollback."""

    def __init__(self, message: str = "Installation cancelled.", *, details: str = ""):
        super().__init__(message)
        self.details = details


class InstallerBackend(Protocol):
    """Mutation boundary consumed by :class:`InstallerController`."""

    def inspect(
        self,
        selection: InstallerSelection,
        *,
        progress: Callable[[ProgressUpdate], None],
        cancellation: threading.Event,
        repair: bool = False,
    ) -> PreparedInstall: ...

    def apply(
        self,
        prepared: PreparedInstall,
        *,
        confirmed: bool,
        progress: Callable[[ProgressUpdate], None],
        cancellation: threading.Event,
    ) -> InstallOutcome: ...

    def open_vipp(self, launcher: Path) -> None: ...

    def open_owned_uninstaller(self, prepared: PreparedInstall) -> None: ...


@dataclass(frozen=True, slots=True)
class InstallerViewState:
    """Complete, immutable view model sent to the graphical window."""

    screen: InstallerScreen
    headline: str
    message: str
    primary_label: str
    primary_enabled: bool
    secondary_label: str = ""
    secondary_enabled: bool = False
    cancel_enabled: bool = False
    status_message: str = ""
    progress_fraction: float | None = None
    progress_stage: str = ""
    progress_unit: ProgressUnit = ProgressUnit.ACTIVITY
    log_path: Path | None = None
    required_free_bytes: int | None = None
    temporary_free_bytes: int | None = None
    size_estimate: InstallSizeEstimate | None = None
    technical_details: str = ""
    target_kind: TargetKind | None = None
    target: Path | None = None
    track: ComputeTrack | None = None
    help_url: str = ""
    blocked_action: BlockedAction = BlockedAction.RETRY

    def __post_init__(self) -> None:
        object.__setattr__(self, "screen", InstallerScreen(self.screen))
        object.__setattr__(self, "blocked_action", BlockedAction(self.blocked_action))
        object.__setattr__(self, "progress_unit", ProgressUnit(self.progress_unit))
        if self.log_path is not None:
            object.__setattr__(self, "log_path", Path(self.log_path))


StateListener = Callable[[InstallerViewState], None]
WorkerFactory = Callable[[Callable[[], None]], object]


def _default_worker_factory(target: Callable[[], None]) -> threading.Thread:
    return threading.Thread(target=target, daemon=True, name="vipp-installer-worker")


class InstallerController:
    """Threaded installer coordinator with one explicit mutation boundary."""

    def __init__(
        self,
        backend: InstallerBackend,
        listener: StateListener,
        *,
        worker_factory: WorkerFactory = _default_worker_factory,
    ) -> None:
        self._backend = backend
        self._listener = listener
        self._worker_factory = worker_factory
        self._lock = threading.RLock()
        self._cancellation = threading.Event()
        self._selection = InstallerSelection()
        self._prepared: PreparedInstall | None = None
        self._prepared_selection: InstallerSelection | None = None
        self._outcome: InstallOutcome | None = None
        self._generation = 0
        self._worker: object | None = None
        self._state = InstallerViewState(
            screen=InstallerScreen.CHECKING,
            headline="Checking this computer…",
            message="VIPP is finding the safest setup for this computer.",
            primary_label="Please wait",
            primary_enabled=False,
            cancel_enabled=True,
            status_message="Starting checks…",
            progress_stage="checking",
        )

    @property
    def state(self) -> InstallerViewState:
        with self._lock:
            return self._state

    @property
    def busy(self) -> bool:
        return self.state.screen in {
            InstallerScreen.CHECKING,
            InstallerScreen.WORKING,
            InstallerScreen.CANCELLING,
        }

    def start(self, selection: InstallerSelection | None = None) -> None:
        """Begin non-mutating checks in a worker and return immediately."""

        with self._lock:
            if self.busy and self._worker is not None:
                return
            selected = selection or self._selection
            self._selection = selected
            self._prepared = None
            self._prepared_selection = None
            self._outcome = None
            generation = self._begin_operation()
            cancellation = self._cancellation
            self._publish(
                InstallerViewState(
                    screen=InstallerScreen.CHECKING,
                    headline="Checking this computer…",
                    message="VIPP is finding the safest setup for this computer.",
                    primary_label="Please wait",
                    primary_enabled=False,
                    cancel_enabled=True,
                    status_message="Checking Windows, Python, and available hardware…",
                    progress_stage="checking",
                )
            )
        self._start_worker(
            lambda: self._inspect_worker(
                generation,
                selected,
                cancellation,
                repair=False,
            )
        )

    def invalidate_selection(self, selection: InstallerSelection) -> bool:
        """Discard a reviewed plan when an install-relevant setting changes.

        A prepared transaction is valid only for the immutable selection that
        produced it.  Invalidation is controller-owned so a late worker result,
        a stale Tk render, or a keyboard activation cannot apply the old plan.
        The caller must run :meth:`start` to review the new selection.

        Returns ``True`` when a different selection was accepted.  Settings are
        intentionally immutable while an installation or rollback is active.
        """

        selection = InstallerSelection(
            track=selection.track,
            install_root=selection.install_root,
            existing_python=selection.existing_python,
            create_desktop_shortcut=selection.create_desktop_shortcut,
        )
        with self._lock:
            if self._state.screen in {
                InstallerScreen.WORKING,
                InstallerScreen.CANCELLING,
            }:
                return False
            if selection == self._selection:
                return False

            # Signal any read-only inspection before replacing its token, then
            # advance the generation so even a backend that notices cancellation
            # late cannot publish a READY state for the previous settings.
            self._cancellation.set()
            self._generation += 1
            self._selection = selection
            self._prepared = None
            self._prepared_selection = None
            self._outcome = None
            self._worker = None
            self._publish(
                InstallerViewState(
                    screen=InstallerScreen.BLOCKED,
                    headline="Settings changed",
                    message=(
                        "Select 'Check these settings' to review the new location "
                        "and computer-use choices before continuing."
                    ),
                    primary_label="Check settings above",
                    primary_enabled=False,
                    secondary_label="Close",
                    secondary_enabled=True,
                    status_message=(
                        "Nothing will be installed until these exact settings "
                        "have been checked."
                    ),
                )
            )
            return True

    def confirm(self) -> None:
        """Apply the prepared plan after the visible primary confirmation."""

        with self._lock:
            if self._state.screen is not InstallerScreen.READY:
                return
            prepared = self._prepared
            if (
                prepared is None
                or self._prepared_selection != self._selection
                or prepared.kind
                not in {
                    TargetKind.NEW,
                    TargetKind.UPDATE,
                    TargetKind.REPAIR,
                }
            ):
                return
            generation = self._begin_operation()
            cancellation = self._cancellation
            verb = {
                TargetKind.NEW: "Installing",
                TargetKind.UPDATE: "Updating",
                TargetKind.REPAIR: "Repairing",
            }[prepared.kind]
            working_message = (
                "The GPU components are a large download and can take several "
                "minutes. The moving bar means setup is still working."
                if prepared.track is ComputeTrack.CUDA13
                else "You can keep using this computer while setup finishes."
            )
            self._publish(
                InstallerViewState(
                    screen=InstallerScreen.WORKING,
                    headline=f"{verb} VIPP…",
                    message=working_message,
                    primary_label="Working…",
                    primary_enabled=False,
                    cancel_enabled=True,
                    status_message="Preparing the installation…",
                    progress_stage="preparing",
                    technical_details=prepared.technical_details,
                    target_kind=prepared.kind,
                    target=prepared.target,
                    track=prepared.track,
                    required_free_bytes=prepared.required_free_bytes,
                    temporary_free_bytes=prepared.temporary_free_bytes,
                    size_estimate=prepared.size_estimate,
                )
            )
        self._start_worker(
            lambda: self._apply_worker(generation, prepared, cancellation)
        )

    def request_repair(self) -> None:
        """Prepare, but do not yet apply, an explicit repair of current VIPP."""

        with self._lock:
            if (
                self._state.screen is not InstallerScreen.CURRENT
                or self._prepared is None
                or self._prepared_selection != self._selection
                or self._prepared.kind is not TargetKind.CURRENT
            ):
                return
            generation = self._begin_operation()
            selected = self._selection
            cancellation = self._cancellation
            self._publish(
                InstallerViewState(
                    screen=InstallerScreen.CHECKING,
                    headline="Checking the VIPP installation…",
                    message="VIPP is preparing a safe repair plan.",
                    primary_label="Please wait",
                    primary_enabled=False,
                    cancel_enabled=True,
                    status_message="Checking installed files and packages…",
                    progress_stage="checking",
                    technical_details=self._prepared.technical_details,
                    target_kind=TargetKind.CURRENT,
                    target=self._prepared.target,
                    track=self._prepared.track,
                )
            )
        self._start_worker(
            lambda: self._inspect_worker(
                generation,
                selected,
                cancellation,
                repair=True,
            )
        )

    def cancel(self) -> None:
        """Request cooperative cancellation; the backend owns safe rollback."""

        with self._lock:
            if not self.busy or self._state.screen is InstallerScreen.CANCELLING:
                return
            self._cancellation.set()
            self._publish(
                replace(
                    self._state,
                    screen=InstallerScreen.CANCELLING,
                    headline="Stopping safely…",
                    message=(
                        "VIPP is finishing the current step and cleaning up files "
                        "created by this setup."
                    ),
                    primary_enabled=False,
                    cancel_enabled=False,
                    status_message="Cancelling and cleaning up…",
                    progress_fraction=None,
                    progress_stage="rolling_back",
                    progress_unit=ProgressUnit.ACTIVITY,
                )
            )

    def retry(self) -> None:
        """Repeat read-only checks with the last selected settings."""

        if self.busy:
            return
        self.start(self._selection)

    def open_vipp(self) -> None:
        """Launch only a verified current/successful VIPP installation."""

        with self._lock:
            if self.busy:
                return
            launcher = self._outcome.launcher if self._outcome else None
            if launcher is None and self._prepared is not None:
                launcher = self._prepared.launcher
            if launcher is None:
                return
        try:
            self._backend.open_vipp(launcher)
        except Exception as exc:  # pragma: no cover - defensive platform boundary
            self._fail(
                "VIPP could not be opened",
                (
                    "The installation is still present. Try its desktop shortcut, "
                    "or retry."
                ),
                exc,
            )

    def open_owned_uninstaller(self) -> None:
        """Launch only the hash-bound uninstaller from the reviewed blocker."""

        with self._lock:
            if self.busy or self._prepared is None:
                return
            prepared = self._prepared
            if prepared.blocked_action is not BlockedAction.RUN_OWNED_UNINSTALLER:
                return
        try:
            self._backend.open_owned_uninstaller(prepared)
        except Exception as exc:  # pragma: no cover - defensive platform boundary
            self._fail(
                "The VIPP uninstaller could not be opened",
                (
                    "Nothing was removed. Check the technical details or use "
                    "Windows Installed apps if VIPP is listed there."
                ),
                exc,
            )

    def _begin_operation(self) -> int:
        self._generation += 1
        self._cancellation = threading.Event()
        return self._generation

    def _start_worker(self, target: Callable[[], None]) -> None:
        worker = self._worker_factory(target)
        self._worker = worker
        start = getattr(worker, "start", None)
        if not callable(start):
            raise TypeError("worker_factory must return an object with start().")
        start()

    def _inspect_worker(
        self,
        generation: int,
        selection: InstallerSelection,
        cancellation: threading.Event,
        *,
        repair: bool,
    ) -> None:
        try:
            prepared = self._backend.inspect(
                selection,
                progress=lambda update: self._on_progress(generation, update),
                cancellation=cancellation,
                repair=repair,
            )
            if cancellation.is_set():
                raise InstallationCancelled()
            with self._lock:
                if generation != self._generation or selection != self._selection:
                    return
                self._prepared = prepared
                self._prepared_selection = selection
                self._publish(_view_for_prepared(prepared))
        except InstallationCancelled as exc:
            self._show_cancelled(generation, details=exc.details)
        except Exception as exc:
            headline, message = _plain_check_failure(exc)
            self._fail(
                headline,
                message,
                exc,
                generation=generation,
            )

    def _apply_worker(
        self,
        generation: int,
        prepared: PreparedInstall,
        cancellation: threading.Event,
    ) -> None:
        try:
            outcome = self._backend.apply(
                prepared,
                confirmed=True,
                progress=lambda update: self._on_progress(generation, update),
                cancellation=cancellation,
            )
            with self._lock:
                if generation != self._generation:
                    return
                self._outcome = outcome
                details = _join_details(
                    prepared.technical_details,
                    outcome.technical_details,
                )
                self._publish(
                    InstallerViewState(
                        screen=InstallerScreen.SUCCESS,
                        headline="VIPP is ready",
                        message=outcome.message,
                        primary_label="Open VIPP",
                        primary_enabled=True,
                        secondary_label="Close",
                        secondary_enabled=True,
                        status_message="Installation completed successfully.",
                        progress_fraction=1.0,
                        progress_stage="completed",
                        progress_unit=ProgressUnit.STEPS,
                        log_path=outcome.log_path or self._state.log_path,
                        required_free_bytes=prepared.required_free_bytes,
                        temporary_free_bytes=prepared.temporary_free_bytes,
                        size_estimate=prepared.size_estimate,
                        technical_details=details,
                        target_kind=prepared.kind,
                        target=prepared.target,
                        track=prepared.track,
                    )
                )
        except InstallationCancelled as exc:
            self._show_cancelled(generation, details=exc.details)
        except Exception as exc:
            headline, message = _plain_apply_failure(exc)
            self._fail(
                headline,
                message,
                exc,
                generation=generation,
                prepared=prepared,
            )

    def _on_progress(self, generation: int, update: ProgressUpdate) -> None:
        with self._lock:
            if generation != self._generation or not self.busy:
                return
            self._publish(
                replace(
                    self._state,
                    status_message=update.message,
                    progress_fraction=update.fraction,
                    progress_stage=update.stage,
                    progress_unit=update.unit,
                    log_path=update.log_path or self._state.log_path,
                )
            )

    def _show_cancelled(self, generation: int, *, details: str = "") -> None:
        with self._lock:
            if generation != self._generation:
                return
            prepared = self._prepared
            self._publish(
                InstallerViewState(
                    screen=InstallerScreen.CANCELLED,
                    headline="Setup was cancelled",
                    message=(
                        "No existing installation was activated or overwritten. "
                        "Advanced details confirms whether every temporary file "
                        "was removed and lists anything that still needs attention."
                    ),
                    primary_label="Try again",
                    primary_enabled=True,
                    secondary_label="Close",
                    secondary_enabled=True,
                    status_message="Cancellation completed.",
                    progress_stage="cancelled",
                    log_path=self._state.log_path,
                    required_free_bytes=(
                        prepared.required_free_bytes if prepared else None
                    ),
                    temporary_free_bytes=(
                        prepared.temporary_free_bytes if prepared else None
                    ),
                    size_estimate=prepared.size_estimate if prepared else None,
                    technical_details=_join_details(
                        prepared.technical_details if prepared else "",
                        details,
                    ),
                    target_kind=prepared.kind if prepared else None,
                    target=prepared.target if prepared else None,
                    track=prepared.track if prepared else None,
                )
            )

    def _fail(
        self,
        headline: str,
        message: str,
        exc: Exception,
        *,
        generation: int | None = None,
        prepared: PreparedInstall | None = None,
    ) -> None:
        details = "".join(traceback.format_exception(exc)).strip()
        with self._lock:
            if generation is not None and generation != self._generation:
                return
            selected = prepared or self._prepared
            self._publish(
                InstallerViewState(
                    screen=InstallerScreen.FAILED,
                    headline=headline,
                    message=message,
                    primary_label="Try again",
                    primary_enabled=True,
                    secondary_label="Close",
                    secondary_enabled=True,
                    status_message="Setup stopped before completion.",
                    progress_stage="failed",
                    log_path=self._state.log_path,
                    required_free_bytes=(
                        selected.required_free_bytes if selected else None
                    ),
                    temporary_free_bytes=(
                        selected.temporary_free_bytes if selected else None
                    ),
                    size_estimate=selected.size_estimate if selected else None,
                    technical_details=_join_details(
                        selected.technical_details if selected else "",
                        details,
                    ),
                    target_kind=selected.kind if selected else None,
                    target=selected.target if selected else None,
                    track=selected.track if selected else None,
                )
            )

    def _publish(self, state: InstallerViewState) -> None:
        self._state = state
        self._listener(state)


def _view_for_prepared(prepared: PreparedInstall) -> InstallerViewState:
    common = {
        "technical_details": prepared.technical_details,
        "target_kind": prepared.kind,
        "target": prepared.target,
        "track": prepared.track,
        "help_url": prepared.help_url,
        "required_free_bytes": prepared.required_free_bytes,
        "temporary_free_bytes": prepared.temporary_free_bytes,
        "size_estimate": prepared.size_estimate,
    }
    if prepared.kind in {TargetKind.NEW, TargetKind.UPDATE, TargetKind.REPAIR}:
        verb = {
            TargetKind.NEW: "Install VIPP",
            TargetKind.UPDATE: "Update VIPP",
            TargetKind.REPAIR: "Repair VIPP",
        }[prepared.kind]
        headline = {
            TargetKind.NEW: "Ready to install VIPP",
            TargetKind.UPDATE: "A VIPP update is ready",
            TargetKind.REPAIR: "VIPP can be repaired",
        }[prepared.kind]
        return InstallerViewState(
            screen=InstallerScreen.READY,
            headline=headline,
            message=prepared.plain_summary,
            primary_label=verb,
            primary_enabled=True,
            secondary_label="Close",
            secondary_enabled=True,
            status_message=_ready_status(prepared),
            progress_stage="ready",
            progress_fraction=None,
            progress_unit=ProgressUnit.ACTIVITY,
            **common,
        )
    if prepared.kind is TargetKind.CURRENT:
        return InstallerViewState(
            screen=InstallerScreen.CURRENT,
            headline="VIPP is already ready",
            message=prepared.plain_summary,
            primary_label="Open VIPP",
            primary_enabled=prepared.launcher is not None,
            secondary_label="Repair VIPP",
            secondary_enabled=True,
            status_message=f"VIPP {prepared.release_version} is installed.",
            progress_stage="current",
            **common,
        )
    if prepared.kind is TargetKind.NEWER:
        return InstallerViewState(
            screen=InstallerScreen.CURRENT,
            headline="A newer VIPP is already installed",
            message=(
                prepared.plain_summary
                or "Setup will not replace a newer version with an older one."
            ),
            primary_label="Open installed VIPP",
            primary_enabled=prepared.launcher is not None,
            secondary_label="Close",
            secondary_enabled=True,
            status_message="Nothing has been changed.",
            progress_stage="current",
            **common,
        )
    if prepared.kind is TargetKind.FOREIGN:
        return InstallerViewState(
            screen=InstallerScreen.BLOCKED,
            headline="This location is already in use",
            message=(
                prepared.plain_summary
                or "Setup found files it did not create and will not overwrite them."
            ),
            primary_label="Check again",
            primary_enabled=True,
            secondary_label="Close",
            secondary_enabled=True,
            status_message="Nothing has been changed.",
            progress_stage="blocked",
            blocked_action=BlockedAction.RETRY,
            **common,
        )
    blocked_action = prepared.blocked_action
    if prepared.help_url and blocked_action is BlockedAction.RETRY:
        blocked_action = BlockedAction.OPEN_HELP
    primary_label = {
        BlockedAction.OPEN_HELP: "Get Python",
        BlockedAction.OPEN_INSTALLED_APPS: "Open Installed apps",
        BlockedAction.RUN_OWNED_UNINSTALLER: "Open VIPP uninstaller",
        BlockedAction.USE_DEFAULT_LOCATION: "Use default location",
        BlockedAction.USE_CPU: "Use CPU",
        BlockedAction.RETRY: "Check again",
    }[blocked_action]
    return InstallerViewState(
        screen=InstallerScreen.BLOCKED,
        headline="VIPP cannot be installed yet",
        message=prepared.plain_summary,
        primary_label=primary_label,
        primary_enabled=True,
        secondary_label=(
            "Check again" if blocked_action is BlockedAction.OPEN_HELP else "Close"
        ),
        secondary_enabled=True,
        status_message=prepared.reason or "Nothing has been changed.",
        progress_stage="blocked",
        blocked_action=blocked_action,
        **common,
    )


def _ready_status(prepared: PreparedInstall) -> str:
    compute = (
        "GPU acceleration"
        if prepared.track is ComputeTrack.CUDA13
        else "reliable CPU processing"
    )
    if prepared.required_free_bytes is None:
        return f"Setup will use {compute}."
    gib = prepared.required_free_bytes / 1024**3
    temporary = prepared.temporary_free_bytes
    temporary_requirement = (
        " Windows temporary files and VIPP installer records also need at "
        f"least {temporary / 1024**3:g} GiB of free disk space on every drive "
        "they use."
        if temporary is not None
        else ""
    )
    if prepared.track is ComputeTrack.CUDA13:
        return (
            f"Setup will use GPU acceleration and needs at least {gib:.0f} GiB "
            "of free disk space on the installation drive."
            f"{temporary_requirement}"
        )
    return (
        f"Setup will use {compute} and needs at least {gib:.0f} GiB of free "
        "disk space on the installation drive during setup."
        f"{temporary_requirement}"
    )


def _join_details(*parts: str) -> str:
    return "\n\n".join(part.strip() for part in parts if part.strip())


def _plain_check_failure(exc: Exception) -> tuple[str, str]:
    text = str(exc).casefold()
    if "free space" in text or "disk space" in text:
        if "temporary downloads and installer records" in text:
            detail = str(exc).strip().rstrip(".")
            return (
                "More free space is needed",
                f"{detail}. Free some space at that location, then choose Try "
                "again. Nothing has been installed.",
            )
        return (
            "More free space is needed",
            "Free some space on the drive named in Advanced details, then choose "
            "Try again. Nothing has been installed.",
        )
    if any(
        phrase in text
        for phrase in (
            "connection",
            "download",
            "network",
            "proxy",
            "ssl",
            "tls",
            "temporary failure",
            "timed out",
            "timeout",
            "max retries exceeded",
            "name resolution",
            "temporarily unavailable",
        )
    ):
        return (
            "VIPP setup could not download its components",
            "Check the internet connection, then choose Try again. If this is a "
            "work or university computer, its network may require help from IT. "
            "Nothing has been installed.",
        )
    if "another vipp installation" in text:
        return (
            "Another VIPP setup is already running",
            "Close the other setup window, wait for it to finish, and choose Try "
            "again.",
        )
    return (
        "VIPP setup could not finish checking",
        "Nothing was changed. Choose Try again, or open Advanced details for help.",
    )


def _plain_apply_failure(exc: Exception) -> tuple[str, str]:
    headline, message = _plain_check_failure(exc)
    if headline != "VIPP setup could not finish checking":
        return headline, message
    return (
        "VIPP could not be installed",
        "The new copy was not activated. Any previous active VIPP remains in "
        "place. Advanced details lists any temporary files that still need "
        "attention, and you can choose Try again.",
    )


__all__ = [
    "BlockedAction",
    "InstallSizeEstimate",
    "InstallOutcome",
    "InstallationCancelled",
    "InstallerBackend",
    "InstallerController",
    "InstallerScreen",
    "InstallerSelection",
    "InstallerViewState",
    "PreparedInstall",
    "ProgressUnit",
    "ProgressUpdate",
    "TargetKind",
    "TrackChoice",
    "default_install_size_estimate",
    "default_temporary_free_bytes",
]
