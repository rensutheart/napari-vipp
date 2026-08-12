"""Lightweight startup protocol shared by the VIPP launcher and application.

This module deliberately uses only the Python standard library.  Importing it
must never initialize napari, Qt, NumPy, or an optional accelerator runtime.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

PROTOCOL_VERSION = 1
MAX_RECORD_BYTES = 64 * 1024
MAX_CHANNEL_BYTES = 2 * 1024 * 1024
DEFAULT_STARTUP_TIMEOUT_SECONDS = 300.0


class LaunchProfile(StrEnum):
    """Compute policy selected by a launcher shortcut for this session."""

    AUTO = "auto"
    CPU = "cpu"
    PREFER_GPU = "prefer_gpu"

    @classmethod
    def parse(cls, value: LaunchProfile | str) -> LaunchProfile:
        """Normalize one command-line or programmatic profile value."""
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower().replace("-", "_")
        try:
            return cls(normalized)
        except ValueError as exc:
            choices = ", ".join(profile.value for profile in cls)
            raise ValueError(
                f"Unsupported VIPP launch profile {value!r}; expected {choices}."
            ) from exc


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """Presentation and compute-mode details for one launch profile."""

    profile: LaunchProfile
    label: str
    short_label: str
    description: str
    accent: str


PROFILE_SPECS: dict[LaunchProfile, ProfileSpec] = {
    LaunchProfile.AUTO: ProfileSpec(
        profile=LaunchProfile.AUTO,
        label="Automatic",
        short_label="AUTO",
        description="VIPP selects eligible CPU or GPU implementations.",
        accent="#38BDF8",
    ),
    LaunchProfile.CPU: ProfileSpec(
        profile=LaunchProfile.CPU,
        label="CPU safe mode",
        short_label="CPU",
        description="This session uses CPU implementations only.",
        accent="#22C55E",
    ),
    LaunchProfile.PREFER_GPU: ProfileSpec(
        profile=LaunchProfile.PREFER_GPU,
        label="Prefer GPU",
        short_label="GPU",
        description="VIPP prefers scientifically eligible GPU implementations.",
        accent="#A78BFA",
    ),
}


@dataclass(frozen=True, slots=True)
class StartupStage:
    """One observable application-startup milestone."""

    key: str
    message: str


STARTUP_STAGES: tuple[StartupStage, ...] = (
    StartupStage("starting_python", "Starting the VIPP application"),
    StartupStage("loading_napari", "Loading napari and its plugins"),
    StartupStage("creating_viewer", "Creating the napari viewer"),
    StartupStage("loading_vipp", "Loading VIPP scientific modules"),
    StartupStage("building_interface", "Building the VIPP workflow interface"),
    StartupStage("preparing_workflow", "Preparing the initial workflow"),
)
STAGE_INDEX = {stage.key: index + 1 for index, stage in enumerate(STARTUP_STAGES)}
STAGE_MESSAGES = {stage.key: stage.message for stage in STARTUP_STAGES}


class StartupPhase(StrEnum):
    """Launcher-visible lifecycle state."""

    STARTING = "starting"
    READY = "ready"
    FAILED = "failed"
    TIMED_OUT = "timed_out"
    HIDDEN = "hidden"


@dataclass(frozen=True, slots=True)
class StartupEvent:
    """One authenticated JSONL record emitted by the child process."""

    token: str
    sequence: int
    kind: str
    stage: str
    message: str
    timestamp: float
    error: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any], *, token: str) -> StartupEvent:
        """Validate an untrusted channel record."""
        if record.get("protocol") != PROTOCOL_VERSION:
            raise StartupProtocolError("Unsupported startup protocol version.")
        if not secrets.compare_digest(str(record.get("token", "")), token):
            raise StartupAuthenticationError("Startup record token did not match.")

        kind = str(record.get("kind", ""))
        if kind not in {"progress", "ready", "failure"}:
            raise StartupProtocolError(f"Unsupported startup event kind {kind!r}.")
        sequence = record.get("sequence")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise StartupProtocolError("Startup event sequence must be positive.")
        timestamp = record.get("timestamp")
        if not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool):
            raise StartupProtocolError("Startup event timestamp must be numeric.")

        stage = str(record.get("stage", ""))
        if kind == "progress" and stage not in STAGE_INDEX:
            raise StartupProtocolError(f"Unknown startup stage {stage!r}.")
        if kind == "ready":
            stage = "ready"
        if kind == "failure":
            stage = "failure"

        message = str(record.get("message", ""))[:2000]
        error = str(record.get("error", ""))[:8000]
        return cls(
            token=token,
            sequence=sequence,
            kind=kind,
            stage=stage,
            message=message,
            timestamp=float(timestamp),
            error=error,
        )


class StartupProtocolError(RuntimeError):
    """Raised when a complete channel record is malformed."""


class StartupAuthenticationError(StartupProtocolError):
    """Raised for a record not authenticated by the channel token."""


@dataclass(slots=True)
class StartupChannel:
    """A private temporary directory containing one authenticated JSONL file."""

    directory: Path
    path: Path
    token: str

    @classmethod
    def create(cls, *, parent: Path | None = None) -> StartupChannel:
        """Create a unique owner-private startup channel."""
        token = secrets.token_urlsafe(32)
        directory = Path(
            tempfile.mkdtemp(
                prefix="vipp-startup-",
                dir=None if parent is None else os.fspath(parent),
            )
        )
        try:
            directory.chmod(0o700)
            path = directory / f"status-{token}.jsonl"
            descriptor = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
            )
            os.close(descriptor)
            try:
                path.chmod(0o600)
            except OSError:
                pass
        except BaseException:
            shutil.rmtree(directory, ignore_errors=True)
            raise
        return cls(directory=directory, path=path, token=token)

    def cleanup(self) -> None:
        """Remove the private channel after both processes have finished with it."""
        shutil.rmtree(self.directory, ignore_errors=True)


class StatusEmitter:
    """Append authenticated startup events from the application process."""

    def __init__(self, path: str | os.PathLike[str], token: str) -> None:
        self._path = Path(path)
        self._token = str(token)
        self._sequence = 0
        self._stream = self._path.open("a", encoding="utf-8", buffering=1)

    def progress(self, stage: str, message: str | None = None) -> None:
        """Report the beginning of a real startup stage."""
        if stage not in STAGE_INDEX:
            raise ValueError(f"Unknown startup stage {stage!r}.")
        self._emit(
            kind="progress",
            stage=stage,
            message=message or STAGE_MESSAGES[stage],
        )

    def ready(self, message: str = "VIPP is ready") -> None:
        """Report that the viewer and VIPP interface are visible."""
        self._emit(kind="ready", stage="ready", message=message)

    def failure(self, message: str, *, error: str = "") -> None:
        """Report a startup failure suitable for presenting to the user."""
        self._emit(kind="failure", stage="failure", message=message, error=error)

    def _emit(self, *, kind: str, stage: str, message: str, error: str = "") -> None:
        if self._stream.closed:
            return
        self._sequence += 1
        record = {
            "protocol": PROTOCOL_VERSION,
            "token": self._token,
            "sequence": self._sequence,
            "kind": kind,
            "stage": stage,
            "message": str(message),
            "timestamp": time.time(),
        }
        if error:
            record["error"] = str(error)
        payload = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._stream.write(f"{payload}\n")
        self._stream.flush()
        try:
            os.fsync(self._stream.fileno())
        except OSError:
            pass

    def close(self) -> None:
        """Close the child side of the channel."""
        if not self._stream.closed:
            self._stream.close()

    def __enter__(self) -> StatusEmitter:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.close()


class StatusReader:
    """Incrementally read complete, bounded records from a startup channel."""

    def __init__(self, path: str | os.PathLike[str], token: str) -> None:
        self._path = Path(path)
        self._token = str(token)
        self._offset = 0
        self._buffer = b""

    def read_new(self) -> list[StartupEvent]:
        """Read and validate records appended since the previous call."""
        size = self._path.stat().st_size
        if size > MAX_CHANNEL_BYTES:
            raise StartupProtocolError("Startup channel exceeded its size limit.")
        if size < self._offset:
            raise StartupProtocolError("Startup channel was unexpectedly truncated.")
        with self._path.open("rb") as stream:
            stream.seek(self._offset)
            chunk = stream.read()
        self._offset += len(chunk)
        self._buffer += chunk
        lines = self._buffer.split(b"\n")
        self._buffer = lines.pop()
        if len(self._buffer) > MAX_RECORD_BYTES:
            raise StartupProtocolError(
                "Startup channel record exceeded its size limit."
            )
        events: list[StartupEvent] = []
        for raw_line in lines:
            if not raw_line:
                continue
            if len(raw_line) > MAX_RECORD_BYTES:
                raise StartupProtocolError(
                    "Startup channel record exceeded its size limit."
                )
            try:
                decoded = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise StartupProtocolError("Malformed startup channel record.") from exc
            if not isinstance(decoded, dict):
                raise StartupProtocolError("Startup channel record must be an object.")
            try:
                event = StartupEvent.from_record(decoded, token=self._token)
            except StartupAuthenticationError:
                # The unpredictable path and per-launch token are both required.
                # Ignore unauthenticated data rather than surfacing it in the UI.
                continue
            events.append(event)
        return events


@dataclass(frozen=True, slots=True)
class StartupSnapshot:
    """Immutable state consumed by a graphical or textual launcher view."""

    phase: StartupPhase
    stage: str
    message: str
    step: int
    total_steps: int
    error: str = ""

    @property
    def progress_percent(self) -> int:
        """Return progress based only on completed/entered real milestones."""
        if self.phase is StartupPhase.READY:
            return 100
        if not self.total_steps:
            return 0
        return max(0, min(99, round(self.step * 100 / self.total_steps)))


class StartupStateMachine:
    """Deterministic launcher state independent of any UI toolkit."""

    def __init__(
        self,
        *,
        timeout_seconds: float = DEFAULT_STARTUP_TIMEOUT_SECONDS,
        now: float | None = None,
    ) -> None:
        timeout_seconds = float(timeout_seconds)
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive.")
        self.timeout_seconds = timeout_seconds
        self.started_at = time.monotonic() if now is None else float(now)
        self.deadline = self.started_at + timeout_seconds
        self.last_sequence = 0
        self.snapshot = StartupSnapshot(
            phase=StartupPhase.STARTING,
            stage="starting_python",
            message=STAGE_MESSAGES["starting_python"],
            step=0,
            total_steps=len(STARTUP_STAGES),
        )

    def accept(self, event: StartupEvent) -> StartupSnapshot:
        """Apply one validated event, ignoring duplicates and regressions."""
        if self.snapshot.phase in {StartupPhase.READY, StartupPhase.FAILED}:
            return self.snapshot
        if event.sequence <= self.last_sequence:
            return self.snapshot
        self.last_sequence = event.sequence

        if event.kind == "progress":
            step = STAGE_INDEX[event.stage]
            if step < self.snapshot.step:
                return self.snapshot
            phase = (
                StartupPhase.HIDDEN
                if self.snapshot.phase is StartupPhase.HIDDEN
                else StartupPhase.STARTING
            )
            self.snapshot = StartupSnapshot(
                phase=phase,
                stage=event.stage,
                message=event.message or STAGE_MESSAGES[event.stage],
                step=step,
                total_steps=len(STARTUP_STAGES),
            )
        elif event.kind == "ready":
            self.snapshot = StartupSnapshot(
                phase=StartupPhase.READY,
                stage="ready",
                message=event.message or "VIPP is ready",
                step=len(STARTUP_STAGES),
                total_steps=len(STARTUP_STAGES),
            )
        else:
            self.snapshot = StartupSnapshot(
                phase=StartupPhase.FAILED,
                stage="failure",
                message=event.message or "VIPP could not start.",
                step=self.snapshot.step,
                total_steps=len(STARTUP_STAGES),
                error=event.error,
            )
        return self.snapshot

    def process_exited(self, returncode: int) -> StartupSnapshot:
        """Fail if the application exits before reporting readiness."""
        if self.snapshot.phase in {StartupPhase.READY, StartupPhase.FAILED}:
            return self.snapshot
        message = (
            "VIPP closed before its window was ready."
            if returncode == 0
            else f"VIPP could not start (exit code {returncode})."
        )
        self.snapshot = StartupSnapshot(
            phase=StartupPhase.FAILED,
            stage="failure",
            message=message,
            step=self.snapshot.step,
            total_steps=len(STARTUP_STAGES),
        )
        return self.snapshot

    def protocol_failed(self, error: BaseException) -> StartupSnapshot:
        """Turn a local channel failure into a user-facing startup failure."""
        if self.snapshot.phase in {StartupPhase.READY, StartupPhase.FAILED}:
            return self.snapshot
        self.snapshot = StartupSnapshot(
            phase=StartupPhase.FAILED,
            stage="failure",
            message="VIPP startup reporting failed.",
            step=self.snapshot.step,
            total_steps=len(STARTUP_STAGES),
            error=f"{type(error).__name__}: {error}",
        )
        return self.snapshot

    def check_timeout(self, *, now: float | None = None) -> StartupSnapshot:
        """Expose a timeout choice without terminating the application."""
        current = time.monotonic() if now is None else float(now)
        if (
            self.snapshot.phase is StartupPhase.STARTING
            and current >= self.deadline
        ):
            self.snapshot = StartupSnapshot(
                phase=StartupPhase.TIMED_OUT,
                stage=self.snapshot.stage,
                message=(
                    "VIPP is still loading. GPU libraries can take several "
                    "minutes on their first run."
                ),
                step=self.snapshot.step,
                total_steps=len(STARTUP_STAGES),
            )
        return self.snapshot

    def keep_waiting(self, *, now: float | None = None) -> StartupSnapshot:
        """Resume visible monitoring after the user accepts a long startup."""
        if self.snapshot.phase is not StartupPhase.TIMED_OUT:
            return self.snapshot
        current = time.monotonic() if now is None else float(now)
        self.deadline = current + self.timeout_seconds
        self.snapshot = StartupSnapshot(
            phase=StartupPhase.STARTING,
            stage=self.snapshot.stage,
            message=STAGE_MESSAGES.get(self.snapshot.stage, self.snapshot.message),
            step=self.snapshot.step,
            total_steps=len(STARTUP_STAGES),
        )
        return self.snapshot

    def hide(self) -> StartupSnapshot:
        """Hide the splash while continuing to monitor and never killing VIPP."""
        if self.snapshot.phase in {StartupPhase.READY, StartupPhase.FAILED}:
            return self.snapshot
        self.snapshot = StartupSnapshot(
            phase=StartupPhase.HIDDEN,
            stage=self.snapshot.stage,
            message=self.snapshot.message,
            step=self.snapshot.step,
            total_steps=len(STARTUP_STAGES),
        )
        return self.snapshot

    def elapsed(self, *, now: float | None = None) -> float:
        """Return non-negative launcher elapsed time."""
        current = time.monotonic() if now is None else float(now)
        return max(0.0, current - self.started_at)


def user_state_directory(
    *,
    platform: str | None = None,
    environ: dict[str, str] | None = None,
    home: Path | None = None,
) -> Path:
    """Return the conventional per-user state directory without dependencies."""
    platform = sys.platform if platform is None else platform
    environment = os.environ if environ is None else environ
    home = Path.home() if home is None else Path(home)
    if platform == "win32":
        base = environment.get("LOCALAPPDATA")
        return (Path(base) if base else home / "AppData" / "Local") / "VIPP"
    if platform == "darwin":
        return home / "Library" / "Logs" / "VIPP"
    base = environment.get("XDG_STATE_HOME")
    state_root = Path(base) if base else home / ".local" / "state"
    return state_root / "vipp"


def create_startup_log_path(
    profile: LaunchProfile | str,
    *,
    state_directory: Path | None = None,
    now: time.struct_time | None = None,
) -> Path:
    """Create a collision-resistant per-launch diagnostic log path."""
    parsed_profile = LaunchProfile.parse(profile)
    root = user_state_directory() if state_directory is None else Path(state_directory)
    log_directory = root / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    try:
        log_directory.chmod(0o700)
    except OSError:
        pass
    timestamp = time.strftime("%Y%m%d-%H%M%S", now or time.localtime())
    suffix = secrets.token_hex(4)
    return log_directory / (
        f"startup-{timestamp}-{parsed_profile.value}-{os.getpid()}-{suffix}.log"
    )


__all__ = [
    "DEFAULT_STARTUP_TIMEOUT_SECONDS",
    "LaunchProfile",
    "PROFILE_SPECS",
    "PROTOCOL_VERSION",
    "STARTUP_STAGES",
    "STAGE_INDEX",
    "STAGE_MESSAGES",
    "ProfileSpec",
    "StartupAuthenticationError",
    "StartupChannel",
    "StartupEvent",
    "StartupPhase",
    "StartupProtocolError",
    "StartupSnapshot",
    "StartupStage",
    "StartupStateMachine",
    "StatusEmitter",
    "StatusReader",
    "create_startup_log_path",
    "user_state_directory",
]
