"""Volatile end-to-end latency observations for interactive parameter edits.

The contracts in this module are diagnostic only.  They are intentionally
absent from workflow serialization, scientific provenance, compute history,
and policy decisions.  A caller must explicitly construct a recorder, and the
recorder retains only a bounded tuple of immutable completed reports.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from time import perf_counter

from napari_vipp.core.execution_telemetry import (
    DeviceExecutionObservation,
    PipelinePreparationObservation,
)

type MonotonicClock = Callable[[], float]


class InteractionLatencyPhase(StrEnum):
    """One UI-observed milestone in a parameter-edit generation."""

    PARAMETER_COMMITTED = "parameter_committed"
    PARAMETER_INVALIDATION_FINISHED = "parameter_invalidation_finished"
    DEBOUNCE_STARTED = "debounce_started"
    DEBOUNCE_FINISHED = "debounce_finished"
    WAITING_FOR_PREVIOUS_RUN = "waiting_for_previous_run"
    WORKER_QUEUED = "worker_queued"
    WORKER_STARTED = "worker_started"
    PIPELINE_STARTED = "pipeline_started"
    PIPELINE_TERMINAL = "pipeline_terminal"
    PIPELINE_RESULT_DELIVERED = "pipeline_result_delivered"
    PIPELINE_ACCEPTED = "pipeline_accepted"
    THUMBNAIL_STATISTICS_QUEUED = "thumbnail_statistics_queued"
    THUMBNAIL_STATISTICS_STARTED = "thumbnail_statistics_started"
    THUMBNAIL_STATISTICS_FINISHED = "thumbnail_statistics_finished"
    THUMBNAIL_STATISTICS_RESULT_DELIVERED = "thumbnail_statistics_result_delivered"
    THUMBNAIL_RENDER_STARTED = "thumbnail_render_started"
    THUMBNAIL_RENDER_FINISHED = "thumbnail_render_finished"
    PUBLICATION_ACCEPTED = "publication_accepted"


class InteractionLatencyOutcome(StrEnum):
    """Terminal disposition of one parameter-edit generation."""

    PUBLISHED = "published"
    SUPERSEDED_BEFORE_DISPATCH = "superseded_before_dispatch"
    SUPERSEDED_BEFORE_ACCEPTANCE = "superseded_before_acceptance"
    SUPERSEDED_IN_FLIGHT = "superseded_in_flight"
    SUPERSEDED_DURING_PRESENTATION = "superseded_during_presentation"
    CANCELLED = "cancelled"
    FAILED = "failed"
    COMPLETED_WITHOUT_PREVIEW = "completed_without_preview"


@dataclass(frozen=True, slots=True)
class InteractionResidentThumbnailStatistics:
    """Small run-correlated summary of in-pipeline resident contrast work."""

    pipeline_run_id: int
    node_id: str
    output_port: int
    contrast_mode: str
    elapsed_seconds: float
    intended_backend: str
    actual_backend: str
    algorithm_id: str
    runtime_id: str = ""
    device_id: str = ""
    input_path: str = ""
    logical_input_host_to_device_bytes: int = 0
    auxiliary_host_to_device_bytes: int = 0
    device_to_host_bytes: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.pipeline_run_id, bool)
            or not isinstance(self.pipeline_run_id, int)
            or self.pipeline_run_id < 1
        ):
            raise ValueError("pipeline_run_id must be a positive integer.")
        if (
            isinstance(self.output_port, bool)
            or not isinstance(self.output_port, int)
            or self.output_port < 0
        ):
            raise ValueError("output_port must be a non-negative integer.")
        elapsed = self.elapsed_seconds
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ValueError("elapsed_seconds must be finite and non-negative.")
        object.__setattr__(self, "elapsed_seconds", float(elapsed))
        for name in (
            "node_id",
            "contrast_mode",
            "intended_backend",
            "actual_backend",
            "algorithm_id",
            "runtime_id",
            "device_id",
            "input_path",
        ):
            object.__setattr__(self, name, str(getattr(self, name) or "").strip())
        if not all(
            (
                self.node_id,
                self.contrast_mode,
                self.intended_backend,
                self.actual_backend,
                self.algorithm_id,
            )
        ):
            raise ValueError("resident thumbnail identity fields must not be empty.")
        for name in (
            "logical_input_host_to_device_bytes",
            "auxiliary_host_to_device_bytes",
            "device_to_host_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")


@dataclass(frozen=True, slots=True)
class InteractionLatencyTelemetryConfig:
    """Explicit opt-in controls for session-local interaction tracing."""

    clock: MonotonicClock = field(
        default=perf_counter,
        repr=False,
        compare=False,
    )
    history_limit: int = 64
    synchronize_device_phases: bool = False

    def __post_init__(self) -> None:
        if not callable(self.clock):
            raise TypeError("clock must be callable.")
        if (
            isinstance(self.history_limit, bool)
            or not isinstance(self.history_limit, int)
            or not 1 <= self.history_limit <= 4096
        ):
            raise ValueError("history_limit must be an integer from 1 to 4096.")
        if not isinstance(self.synchronize_device_phases, bool):
            raise TypeError("synchronize_device_phases must be a boolean.")


@dataclass(frozen=True, slots=True)
class InteractionLatencyEvent:
    """One immutable phase timestamp relative to its committed edit."""

    phase: InteractionLatencyPhase | str
    offset_seconds: float
    detail: str = ""

    def __post_init__(self) -> None:
        phase = (
            self.phase
            if isinstance(self.phase, InteractionLatencyPhase)
            else InteractionLatencyPhase(str(self.phase).strip())
        )
        object.__setattr__(self, "phase", phase)
        offset = self.offset_seconds
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (int, float))
            or not math.isfinite(float(offset))
            or offset < 0
        ):
            raise ValueError("offset_seconds must be finite and non-negative.")
        object.__setattr__(self, "offset_seconds", float(offset))
        object.__setattr__(self, "detail", str(self.detail or "").strip())


@dataclass(frozen=True, slots=True)
class InteractionLatencyReport:
    """Completed, immutable observation for one edit generation."""

    generation_id: int
    node_id: str
    parameter_names: tuple[str, ...]
    started_monotonic_seconds: float
    elapsed_seconds: float
    outcome: InteractionLatencyOutcome | str
    events: tuple[InteractionLatencyEvent, ...]
    pipeline_run_ids: tuple[int, ...] = ()
    detail: str = ""
    device_execution_telemetry: tuple[
        tuple[int, DeviceExecutionObservation],
        ...,
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    pre_device_execution_telemetry: tuple[
        tuple[int, PipelinePreparationObservation],
        ...,
    ] = field(
        default=(),
        repr=False,
        compare=False,
    )
    resident_thumbnail_statistics: tuple[
        InteractionResidentThumbnailStatistics,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.generation_id, bool)
            or not isinstance(self.generation_id, int)
            or self.generation_id < 1
        ):
            raise ValueError("generation_id must be a positive integer.")
        node_id = str(self.node_id).strip()
        if not node_id:
            raise ValueError("node_id must not be empty.")
        object.__setattr__(self, "node_id", node_id)
        parameter_names = tuple(
            dict.fromkeys(
                name
                for raw_name in self.parameter_names
                if (name := str(raw_name).strip())
            )
        )
        if not parameter_names:
            raise ValueError("parameter_names must contain at least one name.")
        object.__setattr__(self, "parameter_names", parameter_names)
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
        outcome = (
            self.outcome
            if isinstance(self.outcome, InteractionLatencyOutcome)
            else InteractionLatencyOutcome(str(self.outcome).strip())
        )
        object.__setattr__(self, "outcome", outcome)
        events = tuple(self.events)
        if not events or any(
            not isinstance(event, InteractionLatencyEvent) for event in events
        ):
            raise ValueError("events must contain InteractionLatencyEvent values.")
        if events[0].phase is not InteractionLatencyPhase.PARAMETER_COMMITTED:
            raise ValueError("the first event must be parameter_committed.")
        if any(
            current.offset_seconds > following.offset_seconds
            for current, following in zip(events, events[1:], strict=False)
        ):
            raise ValueError("events must be ordered by offset_seconds.")
        if events[-1].offset_seconds > float(elapsed) + 1e-12:
            raise ValueError("events must fit inside elapsed_seconds.")
        object.__setattr__(self, "events", events)
        run_ids = tuple(dict.fromkeys(int(run_id) for run_id in self.pipeline_run_ids))
        if any(run_id < 1 for run_id in run_ids):
            raise ValueError("pipeline_run_ids must contain positive integers.")
        object.__setattr__(self, "pipeline_run_ids", run_ids)
        object.__setattr__(self, "detail", str(self.detail or "").strip())
        device_telemetry = tuple(self.device_execution_telemetry)
        normalized_device_telemetry: list[tuple[int, DeviceExecutionObservation]] = []
        for item in device_telemetry:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "device_execution_telemetry must contain "
                    "(run_id, DeviceExecutionObservation) tuples."
                )
            run_id, observation = item
            if (
                isinstance(run_id, bool)
                or not isinstance(run_id, int)
                or run_id < 1
                or not isinstance(observation, DeviceExecutionObservation)
            ):
                raise TypeError(
                    "device_execution_telemetry must contain positive run IDs "
                    "and DeviceExecutionObservation values."
                )
            normalized_device_telemetry.append((run_id, observation))
        if len({run_id for run_id, _item in normalized_device_telemetry}) != len(
            normalized_device_telemetry
        ):
            raise ValueError(
                "device_execution_telemetry must contain at most one observation "
                "per pipeline run."
            )
        object.__setattr__(
            self,
            "device_execution_telemetry",
            tuple(normalized_device_telemetry),
        )
        preparation_telemetry = tuple(self.pre_device_execution_telemetry)
        normalized_preparation_telemetry: list[
            tuple[int, PipelinePreparationObservation]
        ] = []
        for item in preparation_telemetry:
            if not isinstance(item, tuple) or len(item) != 2:
                raise TypeError(
                    "pre_device_execution_telemetry must contain "
                    "(run_id, PipelinePreparationObservation) tuples."
                )
            run_id, observation = item
            if (
                isinstance(run_id, bool)
                or not isinstance(run_id, int)
                or run_id < 1
                or not isinstance(observation, PipelinePreparationObservation)
            ):
                raise TypeError(
                    "pre_device_execution_telemetry must contain positive run "
                    "IDs and PipelinePreparationObservation values."
                )
            normalized_preparation_telemetry.append((run_id, observation))
        if len({run_id for run_id, _item in normalized_preparation_telemetry}) != len(
            normalized_preparation_telemetry
        ):
            raise ValueError(
                "pre_device_execution_telemetry must contain at most one "
                "observation per pipeline run."
            )
        object.__setattr__(
            self,
            "pre_device_execution_telemetry",
            tuple(normalized_preparation_telemetry),
        )
        resident_statistics = tuple(self.resident_thumbnail_statistics)
        if any(
            not isinstance(item, InteractionResidentThumbnailStatistics)
            for item in resident_statistics
        ):
            raise TypeError(
                "resident_thumbnail_statistics must contain "
                "InteractionResidentThumbnailStatistics values."
            )
        object.__setattr__(
            self,
            "resident_thumbnail_statistics",
            resident_statistics,
        )

    def phase_offsets(
        self,
        phase: InteractionLatencyPhase | str,
    ) -> tuple[float, ...]:
        """Return every observed offset for one normalized phase."""

        normalized = (
            phase
            if isinstance(phase, InteractionLatencyPhase)
            else InteractionLatencyPhase(str(phase).strip())
        )
        return tuple(
            event.offset_seconds for event in self.events if event.phase is normalized
        )

    def latency_between(
        self,
        start: InteractionLatencyPhase | str,
        end: InteractionLatencyPhase | str,
    ) -> float | None:
        """Return the first-start to last-end latency when both were observed."""

        starts = self.phase_offsets(start)
        ends = self.phase_offsets(end)
        if not starts or not ends or ends[-1] < starts[0]:
            return None
        return ends[-1] - starts[0]


@dataclass(slots=True)
class _ActiveInteraction:
    generation_id: int
    node_id: str
    parameter_names: list[str]
    origin: float
    events: list[InteractionLatencyEvent]
    pipeline_run_ids: list[int]
    terminal_pipeline_run_ids: set[int] = field(default_factory=set)
    delivered_pipeline_run_ids: set[int] = field(default_factory=set)
    device_execution_telemetry: list[tuple[int, DeviceExecutionObservation]] = field(
        default_factory=list
    )
    pre_device_execution_telemetry: list[tuple[int, PipelinePreparationObservation]] = (
        field(default_factory=list)
    )
    resident_thumbnail_statistics: list[InteractionResidentThumbnailStatistics] = field(
        default_factory=list
    )


class InteractionLatencyRecorder:
    """Best-effort bounded recorder for one widget/session."""

    __slots__ = (
        "_clock",
        "_current_generation_id",
        "_generation_serial",
        "_generations",
        "_pipeline_generations",
        "_reports",
        "_superseded_in_flight",
        "synchronize_device_phases",
    )

    def __init__(self, config: InteractionLatencyTelemetryConfig) -> None:
        if not isinstance(config, InteractionLatencyTelemetryConfig):
            raise TypeError("config must be an InteractionLatencyTelemetryConfig.")
        self._clock = config.clock
        self.synchronize_device_phases = config.synchronize_device_phases
        self._generation_serial = 0
        self._current_generation_id: int | None = None
        self._generations: dict[int, _ActiveInteraction] = {}
        self._pipeline_generations: dict[int, int] = {}
        self._superseded_in_flight: set[int] = set()
        self._reports: deque[InteractionLatencyReport] = deque(
            maxlen=config.history_limit
        )

    @property
    def clock(self) -> MonotonicClock:
        """Expose the configured clock for correlated device observations."""

        return self._clock

    @property
    def active_generation_id(self) -> int | None:
        return self._current_generation_id

    @property
    def active_node_id(self) -> str:
        active = self._active_for(self._current_generation_id)
        return "" if active is None else active.node_id

    def node_id_for_generation(self, generation_id: int | None) -> str:
        active = self._active_for(generation_id)
        return "" if active is None else active.node_id

    def begin_generation(self, node_id: str, parameter_name: str) -> int | None:
        """Start one edit generation and supersede the previous unfinished one."""

        node_id = str(node_id).strip()
        parameter_name = str(parameter_name).strip()
        if not node_id or not parameter_name:
            return None
        self.supersede_current()
        sampled = self._sample()
        if sampled is None:
            return None
        self._generation_serial += 1
        events = [
            InteractionLatencyEvent(
                InteractionLatencyPhase.PARAMETER_COMMITTED,
                0.0,
            )
        ]
        active = _ActiveInteraction(
            self._generation_serial,
            node_id,
            [parameter_name],
            sampled,
            events,
            [],
        )
        self._generations[active.generation_id] = active
        self._current_generation_id = active.generation_id
        return self._generation_serial

    def supersede_current(self, *, detail: str = "") -> bool:
        """Detach the current trace before a newer traced or excluded UI edit.

        A dispatched generation keeps its run mapping until the worker reaches a
        terminal callback. Generations that have not dispatched, or whose
        pipeline work already ended, can be frozen immediately. No replacement
        generation is created, so excluded specialized handlers stay untraced.
        """

        previous = self._active_for(self._current_generation_id)
        if previous is not None:
            awaiting_result_run_ids = set(previous.pipeline_run_ids).difference(
                previous.delivered_pipeline_run_ids
            )
            pipeline_accepted = any(
                event.phase is InteractionLatencyPhase.PIPELINE_ACCEPTED
                for event in previous.events
            )
            pipeline_result_observed = bool(previous.delivered_pipeline_run_ids) or any(
                event.phase
                in {
                    InteractionLatencyPhase.PIPELINE_TERMINAL,
                    InteractionLatencyPhase.PIPELINE_RESULT_DELIVERED,
                }
                for event in previous.events
            )
            if awaiting_result_run_ids:
                # Preserve the old run correlation until its worker reaches a
                # result-delivery callback. A worker-terminal signal can arrive
                # earlier, while the result still carries the only exact device
                # spans and cleanup evidence for the discarded generation.
                self._superseded_in_flight.add(previous.generation_id)
                self._current_generation_id = None
            elif pipeline_accepted:
                self.finish_generation(
                    previous.generation_id,
                    InteractionLatencyOutcome.SUPERSEDED_DURING_PRESENTATION,
                    detail=(
                        detail
                        or "A newer parameter edit replaced the accepted pipeline "
                        "generation before its final thumbnail was published."
                    ),
                )
            elif pipeline_result_observed:
                self.finish_generation(
                    previous.generation_id,
                    InteractionLatencyOutcome.SUPERSEDED_BEFORE_ACCEPTANCE,
                    detail=(
                        detail
                        or "A newer parameter edit replaced a delivered pipeline "
                        "result before that result was accepted for presentation."
                    ),
                )
            else:
                self.finish_generation(
                    previous.generation_id,
                    InteractionLatencyOutcome.SUPERSEDED_BEFORE_DISPATCH,
                    detail=(
                        detail
                        or "A newer parameter edit was committed before dispatch."
                    ),
                )
            return True
        return False

    def record_phase(
        self,
        generation_id: int | None,
        phase: InteractionLatencyPhase | str,
        *,
        detail: str = "",
        once: bool = False,
    ) -> bool:
        """Append one phase to the active generation without affecting execution."""

        active = self._active_for(generation_id)
        if active is None:
            return False
        try:
            normalized = (
                phase
                if isinstance(phase, InteractionLatencyPhase)
                else InteractionLatencyPhase(str(phase).strip())
            )
            if once and any(event.phase is normalized for event in active.events):
                return False
            offset = self._offset(active)
            if offset is None:
                return False
            active.events.append(
                InteractionLatencyEvent(normalized, offset, detail=detail)
            )
            return True
        except Exception:
            return False

    def record_phase_at(
        self,
        generation_id: int | None,
        phase: InteractionLatencyPhase | str,
        monotonic_seconds: float,
        *,
        detail: str = "",
        once: bool = False,
    ) -> bool:
        """Record a phase at a timestamp sampled on the observing thread."""

        active = self._active_for(generation_id)
        if active is None:
            return False
        try:
            normalized = (
                phase
                if isinstance(phase, InteractionLatencyPhase)
                else InteractionLatencyPhase(str(phase).strip())
            )
            if once and any(event.phase is normalized for event in active.events):
                return False
            if (
                isinstance(monotonic_seconds, bool)
                or not isinstance(monotonic_seconds, (int, float))
                or not math.isfinite(float(monotonic_seconds))
                or float(monotonic_seconds) < active.origin
            ):
                return False
            offset = max(
                float(monotonic_seconds) - active.origin,
                active.events[-1].offset_seconds,
            )
            active.events.append(
                InteractionLatencyEvent(normalized, offset, detail=detail)
            )
            return True
        except Exception:
            return False

    def has_phase(
        self,
        generation_id: int | None,
        phase: InteractionLatencyPhase | str,
    ) -> bool:
        active = self._active_for(generation_id)
        if active is None:
            return False
        try:
            normalized = (
                phase
                if isinstance(phase, InteractionLatencyPhase)
                else InteractionLatencyPhase(str(phase).strip())
            )
        except Exception:
            return False
        return any(event.phase is normalized for event in active.events)

    def bind_pipeline_run(
        self,
        generation_id: int | None,
        run_id: int,
    ) -> bool:
        """Associate a submitted worker run with the active edit generation."""

        active = self._active_for(generation_id)
        if active is None or isinstance(run_id, bool) or int(run_id) < 1:
            return False
        run_id = int(run_id)
        if run_id not in active.pipeline_run_ids:
            active.pipeline_run_ids.append(run_id)
        self._pipeline_generations[run_id] = active.generation_id
        return True

    def mark_pipeline_terminal(
        self,
        generation_id: int | None,
        run_id: int,
        *,
        detail: str = "",
    ) -> bool:
        """Record the terminal callback for one specifically bound run."""

        active = self._active_for(generation_id)
        if (
            active is None
            or isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or run_id not in active.pipeline_run_ids
        ):
            return False
        if run_id in active.terminal_pipeline_run_ids:
            return False
        if not self.record_phase(
            generation_id,
            InteractionLatencyPhase.PIPELINE_TERMINAL,
            detail=detail or f"pipeline run {run_id}",
        ):
            return False
        active.terminal_pipeline_run_ids.add(run_id)
        return True

    def mark_pipeline_terminal_at(
        self,
        generation_id: int | None,
        run_id: int,
        monotonic_seconds: float,
        *,
        detail: str = "",
    ) -> bool:
        """Record one bound run's worker-thread terminal timestamp."""

        active = self._active_for(generation_id)
        if (
            active is None
            or isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or run_id not in active.pipeline_run_ids
            or run_id in active.terminal_pipeline_run_ids
        ):
            return False
        if not self.record_phase_at(
            generation_id,
            InteractionLatencyPhase.PIPELINE_TERMINAL,
            monotonic_seconds,
            detail=detail or f"pipeline run {run_id}",
        ):
            return False
        active.terminal_pipeline_run_ids.add(run_id)
        return True

    def pipeline_run_is_terminal(
        self,
        generation_id: int | None,
        run_id: int,
    ) -> bool:
        active = self._active_for(generation_id)
        return bool(
            active is not None
            and not isinstance(run_id, bool)
            and isinstance(run_id, int)
            and run_id in active.terminal_pipeline_run_ids
        )

    def mark_pipeline_result_delivered(
        self,
        generation_id: int | None,
        run_id: int,
        *,
        detail: str = "",
    ) -> bool:
        """Record UI delivery while retaining terminal-before-delivery evidence."""

        active = self._active_for(generation_id)
        if (
            active is None
            or isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or run_id not in active.pipeline_run_ids
            or run_id in active.delivered_pipeline_run_ids
        ):
            return False
        recorded = self.record_phase(
            generation_id,
            InteractionLatencyPhase.PIPELINE_RESULT_DELIVERED,
            detail=detail or f"pipeline run {run_id}",
        )
        active.delivered_pipeline_run_ids.add(run_id)
        return recorded

    def generation_for_pipeline_run(self, run_id: int) -> int | None:
        try:
            generation_id = self._pipeline_generations.get(int(run_id))
        except Exception:
            return None
        return generation_id if self._active_for(generation_id) is not None else None

    def is_superseded_in_flight(self, generation_id: int | None) -> bool:
        """Return whether a newer edit displaced this already-dispatched run."""

        try:
            return int(generation_id) in self._superseded_in_flight
        except Exception:
            return False

    def attach_device_execution_telemetry(
        self,
        generation_id: int | None,
        run_id: int,
        observation: DeviceExecutionObservation | None,
    ) -> bool:
        """Attach the correlated provider-neutral device observation, if any."""

        active = self._active_for(generation_id)
        if (
            active is None
            or observation is None
            or isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or run_id < 1
        ):
            return False
        if not isinstance(observation, DeviceExecutionObservation):
            return False
        active.device_execution_telemetry[:] = [
            item for item in active.device_execution_telemetry if item[0] != run_id
        ]
        active.device_execution_telemetry.append((run_id, observation))
        return True

    def attach_pre_device_execution_telemetry(
        self,
        generation_id: int | None,
        run_id: int,
        observation: PipelinePreparationObservation | None,
    ) -> bool:
        """Attach one run-correlated detached preparation observation."""

        active = self._active_for(generation_id)
        if (
            active is None
            or observation is None
            or isinstance(run_id, bool)
            or not isinstance(run_id, int)
            or run_id < 1
        ):
            return False
        if not isinstance(observation, PipelinePreparationObservation):
            return False
        active.pre_device_execution_telemetry[:] = [
            item for item in active.pre_device_execution_telemetry if item[0] != run_id
        ]
        active.pre_device_execution_telemetry.append((run_id, observation))
        return True

    def attach_resident_thumbnail_statistics(
        self,
        generation_id: int | None,
        observation: InteractionResidentThumbnailStatistics,
    ) -> bool:
        active = self._active_for(generation_id)
        if active is None or not isinstance(
            observation,
            InteractionResidentThumbnailStatistics,
        ):
            return False
        active.resident_thumbnail_statistics.append(observation)
        return True

    def finish_generation(
        self,
        generation_id: int | None,
        outcome: InteractionLatencyOutcome | str,
        *,
        terminal_phase: InteractionLatencyPhase | str | None = None,
        detail: str = "",
    ) -> InteractionLatencyReport | None:
        """Freeze the active generation and append it to bounded history."""

        active = self._active_for(generation_id)
        if active is None:
            return None
        try:
            normalized_outcome = (
                outcome
                if isinstance(outcome, InteractionLatencyOutcome)
                else InteractionLatencyOutcome(str(outcome).strip())
            )
            if terminal_phase is not None:
                self.record_phase(active.generation_id, terminal_phase, detail=detail)
            elapsed = self._offset(active)
            if elapsed is None:
                elapsed = active.events[-1].offset_seconds
            else:
                elapsed = max(elapsed, active.events[-1].offset_seconds)
            report = InteractionLatencyReport(
                generation_id=active.generation_id,
                node_id=active.node_id,
                parameter_names=tuple(active.parameter_names),
                started_monotonic_seconds=active.origin,
                elapsed_seconds=elapsed,
                outcome=normalized_outcome,
                events=tuple(active.events),
                pipeline_run_ids=tuple(active.pipeline_run_ids),
                detail=detail,
                device_execution_telemetry=tuple(active.device_execution_telemetry),
                pre_device_execution_telemetry=tuple(
                    active.pre_device_execution_telemetry
                ),
                resident_thumbnail_statistics=tuple(
                    active.resident_thumbnail_statistics
                ),
            )
        except Exception:
            # Diagnostic bookkeeping must never change UI or scientific behavior.
            self._clear_active(active.generation_id)
            return None
        self._reports.append(report)
        self._clear_active(active.generation_id)
        return report

    def recent_reports(self) -> tuple[InteractionLatencyReport, ...]:
        """Return an immutable snapshot ordered from oldest to newest."""

        return tuple(self._reports)

    def _active_for(self, generation_id: int | None) -> _ActiveInteraction | None:
        if generation_id is None:
            return None
        try:
            normalized = int(generation_id)
        except Exception:
            return None
        return self._generations.get(normalized)

    def _clear_active(self, generation_id: int) -> None:
        active = self._generations.pop(generation_id, None)
        if active is None:
            return
        for run_id in active.pipeline_run_ids:
            self._pipeline_generations.pop(run_id, None)
        self._superseded_in_flight.discard(generation_id)
        if self._current_generation_id == generation_id:
            self._current_generation_id = None

    def _offset(self, active: _ActiveInteraction) -> float | None:
        sampled = self._sample()
        if sampled is None or sampled < active.origin:
            return None
        observed = sampled - active.origin
        return max(observed, active.events[-1].offset_seconds)

    def _sample(self) -> float | None:
        try:
            sampled = self._clock()
        except Exception:
            return None
        if (
            isinstance(sampled, bool)
            or not isinstance(sampled, (int, float))
            or not math.isfinite(float(sampled))
        ):
            return None
        return float(sampled)


__all__ = [
    "InteractionLatencyEvent",
    "InteractionLatencyOutcome",
    "InteractionLatencyPhase",
    "InteractionLatencyRecorder",
    "InteractionLatencyReport",
    "InteractionLatencyTelemetryConfig",
    "InteractionResidentThumbnailStatistics",
]
