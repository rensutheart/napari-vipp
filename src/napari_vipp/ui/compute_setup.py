"""Qt-free presentation models for optional compute setup and memory.

The compute doctor may import and probe an optional runtime, so a GUI should run
it on a worker.  This module performs no discovery and executes no command.  It
only turns a pending state or a completed :class:`ComputeDoctorReport` into
immutable, UI-ready text, action metadata, and memory rows.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import StrEnum

from napari_vipp.core.compute import MemoryTopology
from napari_vipp.core.compute_diagnostics import ComputeDoctorReport, DoctorStatus
from napari_vipp.core.compute_registry import RuntimeMemorySnapshot


class ComputeSetupState(StrEnum):
    """Lifecycle and terminal states shown by a compute-setup surface."""

    NOT_CHECKED = "not_checked"
    CHECKING = "checking"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    UNSUPPORTED = "unsupported"
    MISCONFIGURED = "misconfigured"


class ComputeSetupTone(StrEnum):
    """Provider-neutral visual priority for a setup result."""

    NEUTRAL = "neutral"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


class ComputeSetupActionKind(StrEnum):
    """Actions a controller may implement without granting install authority."""

    VERIFY = "verify"
    COPY_COMMAND = "copy_command"


@dataclass(frozen=True, slots=True)
class HostMemorySnapshot:
    """System-memory values collected by the application shell."""

    total_bytes: int | None = None
    available_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in ("total_bytes", "available_bytes"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")
        if (
            self.total_bytes is not None
            and self.available_bytes is not None
            and self.available_bytes > self.total_bytes
        ):
            raise ValueError("available_bytes must not exceed total_bytes.")


@dataclass(frozen=True, slots=True)
class ComputeMemoryRow:
    """One stable label/value pair for a compact memory presentation."""

    key: str
    label: str
    value: str
    detail: str = ""

    def __post_init__(self) -> None:
        for name in ("key", "label", "value"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "detail", str(self.detail).strip())


@dataclass(frozen=True, slots=True)
class ComputeSetupAction:
    """Declarative action metadata; this object never executes an action."""

    action_id: str
    kind: ComputeSetupActionKind | str
    label: str
    enabled: bool = True
    command: str = ""
    track: str = "auto"
    refresh_runtime: bool = False
    automatic: bool = False

    def __post_init__(self) -> None:
        kind = (
            self.kind
            if isinstance(self.kind, ComputeSetupActionKind)
            else ComputeSetupActionKind(str(self.kind).strip().lower())
        )
        object.__setattr__(self, "kind", kind)
        for name in ("action_id", "label", "track"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be a boolean.")
        if not isinstance(self.refresh_runtime, bool):
            raise TypeError("refresh_runtime must be a boolean.")
        if not isinstance(self.automatic, bool):
            raise TypeError("automatic must be a boolean.")
        if self.automatic:
            raise ValueError(
                "Compute setup actions must require an explicit user action."
            )

        command = str(self.command).strip()
        if kind is ComputeSetupActionKind.COPY_COMMAND:
            if not _is_safe_single_line_command(command):
                raise ValueError(
                    "A copy-command action requires a safe, non-empty "
                    "single-line command."
                )
        elif command:
            raise ValueError("Only copy-command actions may carry a shell command.")
        object.__setattr__(self, "command", command)


@dataclass(frozen=True, slots=True)
class ComputeSetupPresentation:
    """Complete immutable state for a nonblocking compute-setup panel."""

    state: ComputeSetupState | str
    tone: ComputeSetupTone | str
    title: str
    summary: str
    reason_code: str
    track: str
    details: tuple[str, ...] = ()
    memory_rows: tuple[ComputeMemoryRow, ...] = ()
    actions: tuple[ComputeSetupAction, ...] = ()
    busy: bool = False
    actionable: bool = False

    def __post_init__(self) -> None:
        state = (
            self.state
            if isinstance(self.state, ComputeSetupState)
            else ComputeSetupState(str(self.state).strip().lower())
        )
        tone = (
            self.tone
            if isinstance(self.tone, ComputeSetupTone)
            else ComputeSetupTone(str(self.tone).strip().lower())
        )
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "tone", tone)
        for name in ("title", "summary", "reason_code", "track"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty.")
            object.__setattr__(self, name, value)
        details = tuple(text for value in self.details if (text := str(value).strip()))
        rows = tuple(self.memory_rows)
        actions = tuple(self.actions)
        if any(not isinstance(row, ComputeMemoryRow) for row in rows):
            raise TypeError("memory_rows must contain ComputeMemoryRow values.")
        if len({row.key for row in rows}) != len(rows):
            raise ValueError("memory row keys must be unique.")
        if any(not isinstance(action, ComputeSetupAction) for action in actions):
            raise TypeError("actions must contain ComputeSetupAction values.")
        if len({action.action_id for action in actions}) != len(actions):
            raise ValueError("compute setup action IDs must be unique.")
        if not isinstance(self.busy, bool) or not isinstance(self.actionable, bool):
            raise TypeError("busy and actionable must be booleans.")
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "memory_rows", rows)
        object.__setattr__(self, "actions", actions)


def compute_setup_not_checked(
    *,
    platform_name: str = sys.platform,
    execution_mode: str = "native",
    track: str = "auto",
    host_memory: HostMemorySnapshot | None = None,
) -> ComputeSetupPresentation:
    """Return the initial state without probing a package or device."""

    return ComputeSetupPresentation(
        state=ComputeSetupState.NOT_CHECKED,
        tone=ComputeSetupTone.NEUTRAL,
        title=_platform_title(platform_name, execution_mode),
        summary="GPU setup has not been checked in this session.",
        reason_code="not_checked",
        track=str(track).strip() or "auto",
        memory_rows=_memory_rows(host_memory, None, device_name=""),
        actions=(_verify_action(track, label="Verify GPU setup"),),
    )


def compute_setup_checking(
    *,
    platform_name: str = sys.platform,
    execution_mode: str = "native",
    track: str = "auto",
    host_memory: HostMemorySnapshot | None = None,
) -> ComputeSetupPresentation:
    """Return a busy state for use while a worker runs the compute doctor."""

    return ComputeSetupPresentation(
        state=ComputeSetupState.CHECKING,
        tone=ComputeSetupTone.INFO,
        title=_platform_title(platform_name, execution_mode),
        summary="Checking optional GPU packages and hardware…",
        reason_code="diagnostic_running",
        track=str(track).strip() or "auto",
        details=("VIPP remains responsive while verification runs.",),
        memory_rows=_memory_rows(host_memory, None, device_name=""),
        actions=(
            _verify_action(
                track,
                label="Verifying GPU setup…",
                enabled=False,
            ),
        ),
        busy=True,
    )


def present_compute_setup(
    report: ComputeDoctorReport,
    *,
    host_memory: HostMemorySnapshot | None = None,
) -> ComputeSetupPresentation:
    """Convert one completed doctor report into UI-ready presentation data."""

    if not isinstance(report, ComputeDoctorReport):
        raise TypeError("report must be a ComputeDoctorReport.")

    state, tone = _result_state(report.status)
    apple_cpu_only = (
        _is_macos(report.platform)
        and report.status is DoctorStatus.UNSUPPORTED
        and report.reason_code == "platform_unsupported"
    )
    if apple_cpu_only:
        summary = (
            "Apple GPU acceleration is not enabled in this VIPP build. "
            "VIPP will use CPU processing on macOS."
        )
        details = ["NVIDIA CUDA packages are neither required nor offered on macOS."]
    else:
        summary = report.summary
        details = []
    details.extend(report.details)

    actions: list[ComputeSetupAction] = []
    repair_command = str(report.repair_command).strip()
    if repair_command:
        if _is_safe_single_line_command(repair_command):
            actions.append(
                ComputeSetupAction(
                    action_id="copy_compute_setup_command",
                    kind=ComputeSetupActionKind.COPY_COMMAND,
                    label="Copy GPU setup command",
                    command=repair_command,
                    track=report.track,
                )
            )
        else:
            details.append(
                "The suggested setup command was hidden because it was not a safe "
                "single-line command."
            )
    actions.append(
        _verify_action(
            report.track,
            label="Verify again" if report.available else "Verify GPU setup",
        )
    )

    device_name = _selected_device_name(report)
    return ComputeSetupPresentation(
        state=state,
        tone=tone,
        title=_platform_title(report.platform, report.execution_mode),
        summary=summary,
        reason_code=report.reason_code,
        track=report.track,
        details=tuple(details),
        memory_rows=_memory_rows(
            host_memory,
            report.memory_snapshot,
            device_name=device_name,
        ),
        actions=tuple(actions),
        actionable=(
            report.status is DoctorStatus.MISCONFIGURED
            or (
                report.status is DoctorStatus.UNAVAILABLE
                and any(
                    action.kind is ComputeSetupActionKind.COPY_COMMAND
                    for action in actions
                )
            )
        ),
    )


def _verify_action(
    track: str,
    *,
    label: str,
    enabled: bool = True,
) -> ComputeSetupAction:
    return ComputeSetupAction(
        action_id="verify_compute_setup",
        kind=ComputeSetupActionKind.VERIFY,
        label=label,
        enabled=enabled,
        track=str(track).strip() or "auto",
        refresh_runtime=True,
    )


def _result_state(
    status: DoctorStatus,
) -> tuple[ComputeSetupState, ComputeSetupTone]:
    return {
        DoctorStatus.AVAILABLE: (
            ComputeSetupState.AVAILABLE,
            ComputeSetupTone.SUCCESS,
        ),
        DoctorStatus.UNAVAILABLE: (
            ComputeSetupState.UNAVAILABLE,
            ComputeSetupTone.WARNING,
        ),
        DoctorStatus.UNSUPPORTED: (
            ComputeSetupState.UNSUPPORTED,
            ComputeSetupTone.INFO,
        ),
        DoctorStatus.MISCONFIGURED: (
            ComputeSetupState.MISCONFIGURED,
            ComputeSetupTone.ERROR,
        ),
    }[status]


def _memory_rows(
    host_memory: HostMemorySnapshot | None,
    runtime_memory: RuntimeMemorySnapshot | None,
    *,
    device_name: str,
) -> tuple[ComputeMemoryRow, ...]:
    if host_memory is not None and not isinstance(host_memory, HostMemorySnapshot):
        raise TypeError("host_memory must be a HostMemorySnapshot or None.")
    if runtime_memory is not None and not isinstance(
        runtime_memory, RuntimeMemorySnapshot
    ):
        raise TypeError("runtime_memory must be a RuntimeMemorySnapshot or None.")

    if runtime_memory is not None and runtime_memory.topology is MemoryTopology.UNIFIED:
        available = (
            host_memory.available_bytes
            if host_memory is not None and host_memory.available_bytes is not None
            else runtime_memory.device_free_bytes
        )
        total = (
            host_memory.total_bytes
            if host_memory is not None and host_memory.total_bytes is not None
            else runtime_memory.device_total_bytes
        )
        return (
            ComputeMemoryRow(
                "shared_memory",
                "Shared CPU/GPU memory",
                _memory_value(available, total),
                "CPU and accelerator use this single budget; RAM and VRAM must "
                "not be added together.",
            ),
        )

    rows: list[ComputeMemoryRow] = []
    if host_memory is not None:
        rows.append(
            ComputeMemoryRow(
                "system_ram",
                "System RAM",
                _memory_value(
                    host_memory.available_bytes,
                    host_memory.total_bytes,
                ),
                "Memory available to VIPP and other host applications.",
            )
        )
    if (
        runtime_memory is not None
        and runtime_memory.topology is MemoryTopology.DISCRETE
    ):
        detail_parts = []
        if device_name:
            detail_parts.append(device_name)
        if runtime_memory.runtime_reserved_bytes:
            detail_parts.append(
                "VIPP runtime reserved "
                + _format_bytes(runtime_memory.runtime_reserved_bytes)
            )
        if runtime_memory.runtime_live_bytes:
            detail_parts.append(
                f"{_format_bytes(runtime_memory.runtime_live_bytes)} live"
            )
        if runtime_memory.out_of_pool_bytes:
            detail_parts.append(
                f"{_format_bytes(runtime_memory.out_of_pool_bytes)} used outside "
                "the VIPP pool"
            )
        rows.append(
            ComputeMemoryRow(
                "gpu_vram",
                "GPU VRAM",
                _memory_value(
                    runtime_memory.device_free_bytes,
                    runtime_memory.device_total_bytes,
                ),
                "; ".join(detail_parts)
                or "Dedicated accelerator memory; separate from system RAM.",
            )
        )
    return tuple(rows)


def _selected_device_name(report: ComputeDoctorReport) -> str:
    probe = report.runtime_probe
    if probe is None or not probe.devices:
        return ""
    selected = probe.selected_device_id
    for device in probe.devices:
        if device.device_id == selected:
            return device.display_name
    return probe.devices[0].display_name


def _memory_value(available: int | None, total: int | None) -> str:
    if available is not None and total is not None:
        return f"{_format_bytes(available)} available of {_format_bytes(total)}"
    if available is not None:
        return f"{_format_bytes(available)} available"
    if total is not None:
        return f"{_format_bytes(total)} total"
    return "Not reported"


def _format_bytes(value: int) -> str:
    amount = float(value)
    units = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")
    unit = units[0]
    for unit in units:
        if amount < 1024.0 or unit == units[-1]:
            break
        amount /= 1024.0
    if unit == "B":
        return f"{int(amount)} B"
    decimals = 0 if amount >= 100 else 1
    return f"{amount:.{decimals}f} {unit}"


def _platform_title(platform_name: str, execution_mode: str) -> str:
    platform_id = str(platform_name).strip().lower()
    mode = str(execution_mode).strip().lower()
    if _is_macos(platform_id):
        return "Compute setup · macOS"
    if platform_id.startswith("win"):
        return "NVIDIA GPU setup · Windows"
    if platform_id.startswith("linux"):
        suffix = "WSL 2" if mode == "wsl2" else "Linux"
        return f"NVIDIA GPU setup · {suffix}"
    return "Compute setup"


def _is_macos(platform_name: str) -> bool:
    return str(platform_name).strip().lower() in {"darwin", "macos"}


def _is_safe_single_line_command(command: str) -> bool:
    text = str(command).strip()
    return bool(
        text
        and len(text) <= 8192
        and "\n" not in text
        and "\r" not in text
        and "\x00" not in text
        and all(character.isprintable() for character in text)
    )


__all__ = [
    "ComputeMemoryRow",
    "ComputeSetupAction",
    "ComputeSetupActionKind",
    "ComputeSetupPresentation",
    "ComputeSetupState",
    "ComputeSetupTone",
    "HostMemorySnapshot",
    "compute_setup_checking",
    "compute_setup_not_checked",
    "present_compute_setup",
]
