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
    DEGRADED = "degraded"
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
    EXPORT_SUPPORT = "export_support"


@dataclass(frozen=True, slots=True)
class ComputeDeviceOption:
    """One provider-neutral device choice shown by a compute surface."""

    runtime_id: str
    device_id: str
    display_name: str
    total_memory_bytes: int | None = None
    available: bool = True

    def __post_init__(self) -> None:
        runtime_id = str(self.runtime_id).strip()
        device_id = str(self.device_id).strip()
        display_name = str(self.display_name).strip()
        if bool(runtime_id) != bool(device_id):
            raise ValueError(
                "runtime_id and device_id must either both be set or both be blank."
            )
        if not display_name:
            raise ValueError("display_name must not be empty.")
        if self.total_memory_bytes is not None and (
            isinstance(self.total_memory_bytes, bool)
            or not isinstance(self.total_memory_bytes, int)
            or self.total_memory_bytes < 0
        ):
            raise ValueError(
                "total_memory_bytes must be a non-negative integer or None."
            )
        if not isinstance(self.available, bool):
            raise TypeError("available must be a boolean.")
        object.__setattr__(self, "runtime_id", runtime_id)
        object.__setattr__(self, "device_id", device_id)
        object.__setattr__(self, "display_name", display_name)


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
class ComputeSetupCheckRow:
    """One plain-language answer in the three-layer setup check."""

    key: str
    label: str
    value: str
    tone: ComputeSetupTone | str = ComputeSetupTone.NEUTRAL
    detail: str = ""

    def __post_init__(self) -> None:
        tone = (
            self.tone
            if isinstance(self.tone, ComputeSetupTone)
            else ComputeSetupTone(str(self.tone).strip().lower())
        )
        object.__setattr__(self, "tone", tone)
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
    next_step: str = ""
    check_rows: tuple[ComputeSetupCheckRow, ...] = ()
    details: tuple[str, ...] = ()
    memory_rows: tuple[ComputeMemoryRow, ...] = ()
    actions: tuple[ComputeSetupAction, ...] = ()
    device_options: tuple[ComputeDeviceOption, ...] = ()
    default_runtime_id: str = ""
    default_device_id: str = ""
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
        object.__setattr__(self, "next_step", str(self.next_step).strip())
        check_rows = tuple(self.check_rows)
        if any(not isinstance(row, ComputeSetupCheckRow) for row in check_rows):
            raise TypeError("check_rows must contain ComputeSetupCheckRow values.")
        if len({row.key for row in check_rows}) != len(check_rows):
            raise ValueError("compute setup check row keys must be unique.")
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
        device_options = tuple(self.device_options)
        if any(
            not isinstance(option, ComputeDeviceOption) for option in device_options
        ):
            raise TypeError("device_options must contain ComputeDeviceOption values.")
        device_keys = tuple(
            (option.runtime_id, option.device_id) for option in device_options
        )
        if len(set(device_keys)) != len(device_keys):
            raise ValueError("compute device option IDs must be unique.")
        default_runtime_id = str(self.default_runtime_id).strip()
        default_device_id = str(self.default_device_id).strip()
        if bool(default_runtime_id) != bool(default_device_id):
            raise ValueError(
                "default_runtime_id and default_device_id must either both be set "
                "or both be blank."
            )
        if (
            default_runtime_id
            and (
                default_runtime_id,
                default_device_id,
            )
            not in device_keys
        ):
            raise ValueError("The default compute device must be in device_options.")
        if not isinstance(self.busy, bool) or not isinstance(self.actionable, bool):
            raise TypeError("busy and actionable must be booleans.")
        object.__setattr__(self, "details", details)
        object.__setattr__(self, "check_rows", check_rows)
        object.__setattr__(self, "memory_rows", rows)
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "device_options", device_options)
        object.__setattr__(self, "default_runtime_id", default_runtime_id)
        object.__setattr__(self, "default_device_id", default_device_id)


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
        check_rows=_pending_check_rows("Not checked", ComputeSetupTone.NEUTRAL),
        memory_rows=_memory_rows(host_memory, None, device_name=""),
        actions=(_verify_action(track, label="Verify GPU setup"),),
        device_options=(_automatic_device_option(),),
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
        check_rows=_pending_check_rows("Checking…", ComputeSetupTone.INFO),
        details=("VIPP remains responsive while verification runs.",),
        memory_rows=_memory_rows(host_memory, None, device_name=""),
        actions=(
            _verify_action(
                track,
                label="Verifying GPU setup…",
                enabled=False,
            ),
        ),
        device_options=(_automatic_device_option(),),
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
    actions.append(
        ComputeSetupAction(
            action_id="export_compute_support",
            kind=ComputeSetupActionKind.EXPORT_SUPPORT,
            label="Save privacy-redacted support report…",
            track=report.track,
        )
    )

    device_name = _selected_device_name(report)
    device_options, default_runtime_id, default_device_id = _device_options(report)
    return ComputeSetupPresentation(
        state=state,
        tone=tone,
        title=_platform_title(report.platform, report.execution_mode),
        summary=summary,
        reason_code=report.reason_code,
        track=report.track,
        next_step=(
            f"Next step: {report.guidance.title}. {report.guidance.summary}"
            if report.guidance is not None
            else ""
        ),
        check_rows=_completed_check_rows(report),
        details=tuple(details),
        memory_rows=_memory_rows(
            host_memory,
            report.memory_snapshot,
            device_name=device_name,
        ),
        actions=tuple(actions),
        device_options=device_options,
        default_runtime_id=default_runtime_id,
        default_device_id=default_device_id,
        actionable=(
            report.status in {DoctorStatus.DEGRADED, DoctorStatus.MISCONFIGURED}
            or (
                report.status is DoctorStatus.UNAVAILABLE
                and any(
                    action.kind is ComputeSetupActionKind.COPY_COMMAND
                    for action in actions
                )
            )
        ),
    )


def _automatic_device_option() -> ComputeDeviceOption:
    return ComputeDeviceOption("", "", "Automatic (runtime default)")


def _device_options(
    report: ComputeDoctorReport,
) -> tuple[tuple[ComputeDeviceOption, ...], str, str]:
    options = [_automatic_device_option()]
    probe = report.runtime_probe
    if probe is None:
        return tuple(options), "", ""
    options.extend(
        ComputeDeviceOption(
            runtime_id=probe.runtime_id,
            device_id=device.device_id,
            display_name=device.display_name,
            total_memory_bytes=device.total_memory_bytes,
            available=probe.available,
        )
        for device in probe.devices
    )
    if not probe.selected_device_id:
        return tuple(options), "", ""
    return tuple(options), probe.runtime_id, probe.selected_device_id


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
        DoctorStatus.DEGRADED: (
            ComputeSetupState.DEGRADED,
            ComputeSetupTone.WARNING,
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


def _pending_check_rows(
    value: str,
    tone: ComputeSetupTone,
) -> tuple[ComputeSetupCheckRow, ...]:
    return (
        ComputeSetupCheckRow("cuda", "CUDA and GPU", value, tone),
        ComputeSetupCheckRow("vipp", "VIPP GPU coverage", value, tone),
    )


def _completed_check_rows(
    report: ComputeDoctorReport,
) -> tuple[ComputeSetupCheckRow, ...]:
    cuda_ready = report.cuda_ready
    cuda_row = ComputeSetupCheckRow(
        "cuda",
        "CUDA and GPU",
        "Ready" if cuda_ready else "Could not start",
        ComputeSetupTone.SUCCESS if cuda_ready else ComputeSetupTone.WARNING,
        (
            report.runtime_probe.message
            if report.runtime_probe is not None
            else "No CUDA runtime result was recorded."
        ),
    )
    admitted = len(report.admitted_regions)
    total = len(report.admission_regions)
    if not total:
        coverage_value = "No reviewed regions available"
        coverage_tone = ComputeSetupTone.NEUTRAL
    elif admitted == total:
        coverage_value = f"{admitted} of {total} reviewed regions ready"
        coverage_tone = ComputeSetupTone.SUCCESS
    else:
        coverage_value = f"{admitted} of {total} reviewed regions ready"
        coverage_tone = ComputeSetupTone.WARNING
    coverage_row = ComputeSetupCheckRow(
        "vipp",
        "VIPP GPU coverage",
        coverage_value,
        coverage_tone,
        "Only reviewed combinations are offered automatically; CPU remains safe.",
    )
    return (cuda_row, coverage_row)


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
    "ComputeDeviceOption",
    "ComputeMemoryRow",
    "ComputeSetupAction",
    "ComputeSetupActionKind",
    "ComputeSetupCheckRow",
    "ComputeSetupPresentation",
    "ComputeSetupState",
    "ComputeSetupTone",
    "HostMemorySnapshot",
    "compute_setup_checking",
    "compute_setup_not_checked",
    "present_compute_setup",
]
