"""Provider-neutral, volatile timing observations for device execution.

These contracts describe diagnostic measurements only.  They are deliberately
separate from workflow serialization, scientific provenance, compute history,
and policy inputs.  Callers opt in for one execution and receive an immutable
observation on the corresponding result.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter

from napari_vipp.core.compute import OutputPortKey

type MonotonicClock = Callable[[], float]


class DeviceExecutionPhase(StrEnum):
    """One directly observed phase of a device execution."""

    ACCELERATOR_LEASE_WAIT = "accelerator_lease_wait"
    PREFLIGHT = "preflight"
    HOST_TO_DEVICE = "host_to_device"
    IMPLEMENTATION_RESOLUTION = "implementation_resolution"
    DEVICE_OPERATION = "device_operation"
    DEVICE_SYNCHRONIZE = "device_synchronize"
    DEVICE_TO_HOST = "device_to_host"
    HOST_FINALIZER = "host_finalizer"


class DeviceTransferDirection(StrEnum):
    """Direction of a host/device payload transfer."""

    HOST_TO_DEVICE = "host_to_device"
    DEVICE_TO_HOST = "device_to_host"


class DeviceSynchronizationPoint(StrEnum):
    """Reason an observed runtime synchronization was requested."""

    AFTER_HOST_TO_DEVICE = "after_host_to_device"
    AFTER_DEVICE_OPERATION = "after_device_operation"
    AFTER_DEVICE_TO_HOST = "after_device_to_host"
    SEGMENT_COMPLETE = "segment_complete"


@dataclass(frozen=True, slots=True)
class DeviceExecutionTelemetryConfig:
    """Opt-in controls for one device-execution observation.

    ``synchronize_device_phases`` inserts a barrier after every transfer and
    operation so their elapsed times include asynchronous device work.  It is
    intentionally disabled by default because those barriers perturb normal
    scheduling.  Even when disabled, the executor records host-call durations
    and its existing segment-complete synchronization.
    """

    clock: MonotonicClock = field(
        default=perf_counter,
        repr=False,
        compare=False,
    )
    synchronize_device_phases: bool = False

    def __post_init__(self) -> None:
        if not callable(self.clock):
            raise TypeError("clock must be callable.")
        if not isinstance(self.synchronize_device_phases, bool):
            raise TypeError("synchronize_device_phases must be a boolean.")


@dataclass(frozen=True, slots=True)
class DeviceExecutionSpan:
    """One immutable timing span with provider-neutral execution identity.

    Detailed transfer and operation spans contain their nested synchronization
    span.  Consumers must therefore not sum unlike phases as if every span were
    disjoint.  ``synchronized=True`` means the instrumentation inserted and
    observed an explicit barrier; ``None`` leaves provider-internal behavior
    unknown.
    """

    phase: DeviceExecutionPhase | str
    start_offset_seconds: float
    elapsed_seconds: float
    runtime_id: str
    device_id: str
    segment_id: str = ""
    node_id: str = ""
    operation_id: str = ""
    implementation_id: str = ""
    port: OutputPortKey | None = None
    byte_count: int | None = None
    synchronized: bool | None = None
    synchronization_point: DeviceSynchronizationPoint | str | None = None
    succeeded: bool = True

    def __post_init__(self) -> None:
        phase = (
            self.phase
            if isinstance(self.phase, DeviceExecutionPhase)
            else DeviceExecutionPhase(str(self.phase).strip())
        )
        object.__setattr__(self, "phase", phase)
        for name in (
            "start_offset_seconds",
            "elapsed_seconds",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
            object.__setattr__(self, name, float(value))
        for name in (
            "runtime_id",
            "device_id",
            "segment_id",
            "node_id",
            "operation_id",
            "implementation_id",
        ):
            object.__setattr__(self, name, str(getattr(self, name)).strip())
        if not self.runtime_id:
            raise ValueError("runtime_id must not be empty.")
        if not self.device_id:
            raise ValueError("device_id must not be empty.")
        if self.port is not None and not isinstance(self.port, OutputPortKey):
            raise TypeError("port must be an OutputPortKey or None.")
        if self.byte_count is not None and (
            isinstance(self.byte_count, bool)
            or not isinstance(self.byte_count, int)
            or self.byte_count < 0
        ):
            raise ValueError("byte_count must be a non-negative integer or None.")
        if self.synchronized is not None and not isinstance(self.synchronized, bool):
            raise TypeError("synchronized must be a boolean or None.")
        point = self.synchronization_point
        if point is not None and not isinstance(point, DeviceSynchronizationPoint):
            point = DeviceSynchronizationPoint(str(point).strip())
        object.__setattr__(self, "synchronization_point", point)
        if not isinstance(self.succeeded, bool):
            raise TypeError("succeeded must be a boolean.")

        segment_phases = {
            DeviceExecutionPhase.PREFLIGHT,
            DeviceExecutionPhase.HOST_TO_DEVICE,
            DeviceExecutionPhase.IMPLEMENTATION_RESOLUTION,
            DeviceExecutionPhase.DEVICE_OPERATION,
            DeviceExecutionPhase.DEVICE_SYNCHRONIZE,
            DeviceExecutionPhase.DEVICE_TO_HOST,
            DeviceExecutionPhase.HOST_FINALIZER,
        }
        if phase in segment_phases and not self.segment_id:
            raise ValueError(f"{phase.value} spans require segment_id.")
        transfer_phases = {
            DeviceExecutionPhase.HOST_TO_DEVICE,
            DeviceExecutionPhase.DEVICE_TO_HOST,
        }
        if phase in transfer_phases:
            if self.port is None or not self.node_id:
                raise ValueError(f"{phase.value} spans require port and node_id.")
            if self.node_id != self.port.node_id:
                raise ValueError("transfer node_id must identify its output port.")
        operation_phases = {
            DeviceExecutionPhase.IMPLEMENTATION_RESOLUTION,
            DeviceExecutionPhase.DEVICE_OPERATION,
            DeviceExecutionPhase.HOST_FINALIZER,
        }
        if phase in operation_phases and not all(
            (self.node_id, self.operation_id, self.implementation_id)
        ):
            raise ValueError(
                f"{phase.value} spans require node, operation, and implementation."
            )
        if phase is DeviceExecutionPhase.DEVICE_SYNCHRONIZE:
            if point is None:
                raise ValueError(
                    "device_synchronize spans require synchronization_point."
                )
            if self.synchronized is not self.succeeded:
                raise ValueError(
                    "device_synchronize spans must report whether the barrier "
                    "succeeded."
                )
        elif point is not None:
            raise ValueError(
                "synchronization_point is valid only for device_synchronize spans."
            )


@dataclass(frozen=True, slots=True)
class DeviceTransferSummary:
    """Directional aggregate for observed transfer attempts."""

    direction: DeviceTransferDirection | str
    count: int
    succeeded_count: int
    byte_count: int
    unknown_byte_count: int
    elapsed_seconds: float
    all_synchronized: bool

    def __post_init__(self) -> None:
        direction = (
            self.direction
            if isinstance(self.direction, DeviceTransferDirection)
            else DeviceTransferDirection(str(self.direction).strip())
        )
        object.__setattr__(self, "direction", direction)
        for name in ("count", "succeeded_count", "byte_count", "unknown_byte_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.succeeded_count > self.count:
            raise ValueError("succeeded_count must not exceed count.")
        if self.unknown_byte_count > self.count:
            raise ValueError("unknown_byte_count must not exceed count.")
        if (
            isinstance(self.elapsed_seconds, bool)
            or not isinstance(self.elapsed_seconds, (int, float))
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0
        ):
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        if not isinstance(self.all_synchronized, bool):
            raise TypeError("all_synchronized must be a boolean.")


@dataclass(frozen=True, slots=True)
class DeviceExecutionObservation:
    """Completed volatile observation attached to a device execution result."""

    started_monotonic_seconds: float
    elapsed_seconds: float
    spans: tuple[DeviceExecutionSpan, ...] = ()
    synchronized_device_phases: bool = False

    def __post_init__(self) -> None:
        started = self.started_monotonic_seconds
        if (
            isinstance(started, bool)
            or not isinstance(started, (int, float))
            or not math.isfinite(float(started))
        ):
            raise ValueError("started_monotonic_seconds must be finite.")
        object.__setattr__(self, "started_monotonic_seconds", float(started))
        elapsed = self.elapsed_seconds
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        object.__setattr__(self, "elapsed_seconds", float(elapsed))
        spans = tuple(self.spans)
        if any(not isinstance(span, DeviceExecutionSpan) for span in spans):
            raise TypeError("spans must contain DeviceExecutionSpan values.")
        if any(
            span.start_offset_seconds + span.elapsed_seconds
            > self.elapsed_seconds + 1e-12
            for span in spans
        ):
            raise ValueError("spans must fit inside the observation duration.")
        object.__setattr__(self, "spans", spans)
        if not isinstance(self.synchronized_device_phases, bool):
            raise TypeError("synchronized_device_phases must be a boolean.")

    @property
    def host_to_device(self) -> DeviceTransferSummary:
        """Aggregate observed host-to-device transfer attempts."""

        return self._transfer_summary(DeviceTransferDirection.HOST_TO_DEVICE)

    @property
    def device_to_host(self) -> DeviceTransferSummary:
        """Aggregate observed device-to-host transfer attempts."""

        return self._transfer_summary(DeviceTransferDirection.DEVICE_TO_HOST)

    def spans_for(
        self,
        phase: DeviceExecutionPhase | str,
    ) -> tuple[DeviceExecutionSpan, ...]:
        """Return spans for one normalized phase in observation order."""

        normalized = (
            phase
            if isinstance(phase, DeviceExecutionPhase)
            else DeviceExecutionPhase(str(phase).strip())
        )
        return tuple(span for span in self.spans if span.phase is normalized)

    def _transfer_summary(
        self,
        direction: DeviceTransferDirection,
    ) -> DeviceTransferSummary:
        phase = DeviceExecutionPhase(direction.value)
        spans = self.spans_for(phase)
        known_bytes = tuple(
            span.byte_count for span in spans if span.byte_count is not None
        )
        return DeviceTransferSummary(
            direction=direction,
            count=len(spans),
            succeeded_count=sum(span.succeeded for span in spans),
            byte_count=sum(known_bytes),
            unknown_byte_count=len(spans) - len(known_bytes),
            elapsed_seconds=sum(span.elapsed_seconds for span in spans),
            all_synchronized=bool(spans)
            and all(span.synchronized is True for span in spans),
        )


class _DeviceExecutionTelemetryRecorder:
    """Best-effort mutable builder kept private to one execution call."""

    __slots__ = (
        "_clock",
        "_finished",
        "_origin",
        "_spans",
        "synchronize_device_phases",
    )

    def __init__(self, config: DeviceExecutionTelemetryConfig) -> None:
        self._clock = config.clock
        self.synchronize_device_phases = config.synchronize_device_phases
        self._spans: list[DeviceExecutionSpan] = []
        self._finished = False
        self._origin = self._sample()

    @property
    def enabled(self) -> bool:
        return self._origin is not None and not self._finished

    def start(self) -> float | None:
        """Sample the configured clock, or return ``None`` when unavailable."""

        if not self.enabled:
            return None
        return self._sample()

    def record(
        self,
        started: float | None,
        phase: DeviceExecutionPhase,
        *,
        runtime_id: str,
        device_id: str,
        segment_id: str = "",
        node_id: str = "",
        operation_id: str = "",
        implementation_id: str = "",
        port: OutputPortKey | None = None,
        byte_count: int | None = None,
        synchronized: bool | None = None,
        synchronization_point: DeviceSynchronizationPoint | None = None,
        succeeded: bool,
    ) -> None:
        """Close one span without allowing a clock failure to affect execution."""

        try:
            if started is None or self._origin is None or self._finished:
                return
            ended = self._sample()
            if ended is None or ended < started or started < self._origin:
                return
            self._spans.append(
                DeviceExecutionSpan(
                    phase=phase,
                    start_offset_seconds=started - self._origin,
                    elapsed_seconds=ended - started,
                    runtime_id=runtime_id,
                    device_id=device_id,
                    segment_id=segment_id,
                    node_id=node_id,
                    operation_id=operation_id,
                    implementation_id=implementation_id,
                    port=port,
                    byte_count=byte_count,
                    synchronized=synchronized,
                    synchronization_point=synchronization_point,
                    succeeded=succeeded,
                )
            )
        except Exception:
            # Telemetry is diagnostic only. Invalid provider metadata, a hostile
            # clock, or span validation must never change scientific execution.
            return

    def finish(self) -> DeviceExecutionObservation | None:
        """Freeze the accumulated spans; repeated calls return no observation."""

        if self._finished or self._origin is None:
            return None
        self._finished = True
        try:
            ended = self._sample()
            observed_end = self._origin + _spans_end(self._spans)
            if ended is None:
                ended = observed_end
            else:
                ended = max(ended, observed_end)
            return DeviceExecutionObservation(
                started_monotonic_seconds=self._origin,
                elapsed_seconds=ended - self._origin,
                spans=tuple(self._spans),
                synchronized_device_phases=self.synchronize_device_phases,
            )
        except Exception:
            return None

    def _sample(self) -> float | None:
        try:
            value = self._clock()
        except Exception:
            return None
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
        ):
            return None
        return float(value)


def _begin_device_execution_telemetry(
    config: DeviceExecutionTelemetryConfig | None,
) -> _DeviceExecutionTelemetryRecorder | None:
    if config is None:
        return None
    if not isinstance(config, DeviceExecutionTelemetryConfig):
        raise TypeError("telemetry must be a DeviceExecutionTelemetryConfig or None.")
    recorder = _DeviceExecutionTelemetryRecorder(config)
    return recorder if recorder.enabled else None


def _finish_device_execution_telemetry(
    recorder: _DeviceExecutionTelemetryRecorder | None,
) -> DeviceExecutionObservation | None:
    if recorder is None:
        return None
    try:
        return recorder.finish()
    except Exception:
        return None


def _observed_host_nbytes(value: object) -> int | None:
    """Read a host payload's byte size without coercing or retaining it."""

    try:
        candidate = getattr(value, "nbytes", None)
    except Exception:
        candidate = None
    if (
        not isinstance(candidate, bool)
        and isinstance(candidate, int)
        and candidate >= 0
    ):
        return candidate
    try:
        candidate = memoryview(value).nbytes
    except Exception:
        return None
    return candidate if candidate >= 0 else None


def _spans_end(spans: Iterable[DeviceExecutionSpan]) -> float:
    """Return the greatest relative span end (test/support helper)."""

    return max(
        (span.start_offset_seconds + span.elapsed_seconds for span in spans),
        default=0.0,
    )


__all__ = [
    "DeviceExecutionObservation",
    "DeviceExecutionPhase",
    "DeviceExecutionSpan",
    "DeviceExecutionTelemetryConfig",
    "DeviceSynchronizationPoint",
    "DeviceTransferDirection",
    "DeviceTransferSummary",
    "MonotonicClock",
]
