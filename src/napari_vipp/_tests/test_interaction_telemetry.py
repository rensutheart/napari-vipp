from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from napari_vipp.core.execution_telemetry import (
    DeviceExecutionObservation,
    PipelinePreparationObservation,
)
from napari_vipp.core.interaction_telemetry import (
    InteractionLatencyOutcome,
    InteractionLatencyPhase,
    InteractionLatencyRecorder,
    InteractionLatencyTelemetryConfig,
)


class _FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


def test_interaction_trace_reports_exact_fake_clock_phase_latencies() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=4)
    )

    generation = recorder.begin_generation("gaussian", "sigma")
    assert generation == 1
    assert recorder.record_phase(
        generation,
        InteractionLatencyPhase.PARAMETER_INVALIDATION_FINISHED,
    )
    assert recorder.record_phase(
        generation,
        InteractionLatencyPhase.DEBOUNCE_STARTED,
    )
    clock.advance(0.150)
    assert recorder.record_phase(
        generation,
        InteractionLatencyPhase.DEBOUNCE_FINISHED,
    )
    clock.advance(0.025)
    assert recorder.bind_pipeline_run(generation, 17)
    assert recorder.record_phase(generation, InteractionLatencyPhase.WORKER_QUEUED)
    clock.advance(0.010)
    assert recorder.record_phase(generation, InteractionLatencyPhase.WORKER_STARTED)
    clock.advance(0.200)
    assert recorder.record_phase(generation, InteractionLatencyPhase.PIPELINE_TERMINAL)
    clock.advance(0.005)
    assert recorder.record_phase(generation, InteractionLatencyPhase.PIPELINE_ACCEPTED)
    clock.advance(0.030)
    assert recorder.record_phase(
        generation,
        InteractionLatencyPhase.THUMBNAIL_RENDER_STARTED,
    )
    clock.advance(0.004)
    assert recorder.record_phase(
        generation,
        InteractionLatencyPhase.THUMBNAIL_RENDER_FINISHED,
    )
    report = recorder.finish_generation(
        generation,
        InteractionLatencyOutcome.PUBLISHED,
        terminal_phase=InteractionLatencyPhase.PUBLICATION_ACCEPTED,
    )

    assert report is not None
    assert report.outcome is InteractionLatencyOutcome.PUBLISHED
    assert report.pipeline_run_ids == (17,)
    assert report.elapsed_seconds == pytest.approx(0.424)
    assert report.latency_between(
        InteractionLatencyPhase.DEBOUNCE_STARTED,
        InteractionLatencyPhase.DEBOUNCE_FINISHED,
    ) == pytest.approx(0.150)
    assert report.latency_between(
        InteractionLatencyPhase.WORKER_QUEUED,
        InteractionLatencyPhase.WORKER_STARTED,
    ) == pytest.approx(0.010)
    assert report.latency_between(
        InteractionLatencyPhase.THUMBNAIL_RENDER_STARTED,
        InteractionLatencyPhase.THUMBNAIL_RENDER_FINISHED,
    ) == pytest.approx(0.004)


def test_new_parameter_generation_records_superseded_before_dispatch() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=4)
    )

    first = recorder.begin_generation("gaussian", "sigma")
    clock.advance(0.020)
    second = recorder.begin_generation("gaussian", "sigma")

    assert first == 1
    assert second == 2
    reports = recorder.recent_reports()
    assert len(reports) == 1
    assert reports[0].generation_id == first
    assert reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_BEFORE_DISPATCH
    assert reports[0].elapsed_seconds == pytest.approx(0.020)
    assert recorder.active_generation_id == second


def test_superseded_in_flight_generation_keeps_old_run_correlation_until_terminal() -> (
    None
):
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=4)
    )
    first = recorder.begin_generation("gaussian", "sigma")
    assert recorder.bind_pipeline_run(first, 41)
    clock.advance(0.020)

    second = recorder.begin_generation("gaussian", "sigma")

    assert recorder.active_generation_id == second
    assert recorder.generation_for_pipeline_run(41) == first
    assert recorder.is_superseded_in_flight(first)
    assert recorder.recent_reports() == ()

    clock.advance(0.300)
    assert recorder.mark_pipeline_terminal(first, 41)
    report = recorder.finish_generation(
        first,
        InteractionLatencyOutcome.SUPERSEDED_IN_FLIGHT,
        detail="Discarded after the newer parameter edit.",
    )

    assert report is not None
    assert report.outcome is InteractionLatencyOutcome.SUPERSEDED_IN_FLIGHT
    assert report.pipeline_run_ids == (41,)
    assert recorder.generation_for_pipeline_run(41) is None
    assert recorder.active_generation_id == second


def test_worker_terminal_before_result_delivery_keeps_run_evidence_correlated() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=4)
    )
    first = recorder.begin_generation("gaussian", "sigma")
    assert recorder.bind_pipeline_run(first, 43)
    clock.advance(0.1)
    assert recorder.mark_pipeline_terminal(first, 43)

    second = recorder.begin_generation("gaussian", "sigma")

    assert recorder.active_generation_id == second
    assert recorder.recent_reports() == ()
    assert recorder.generation_for_pipeline_run(43) == first
    assert recorder.is_superseded_in_flight(first)
    observation = DeviceExecutionObservation(100.0, 0.2)
    clock.advance(3.0)
    assert recorder.mark_pipeline_result_delivered(first, 43)
    assert recorder.attach_device_execution_telemetry(first, 43, observation)
    report = recorder.finish_generation(
        first,
        InteractionLatencyOutcome.SUPERSEDED_IN_FLIGHT,
    )

    assert report is not None
    assert report.device_execution_telemetry == ((43, observation),)
    assert report.phase_offsets(InteractionLatencyPhase.PIPELINE_RESULT_DELIVERED)


def test_delivered_unaccepted_result_is_superseded_before_presentation() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=4)
    )
    first = recorder.begin_generation("gaussian", "sigma")
    assert recorder.bind_pipeline_run(first, 44)
    clock.advance(0.1)
    assert recorder.mark_pipeline_terminal(first, 44)
    assert recorder.mark_pipeline_result_delivered(first, 44)
    observation = DeviceExecutionObservation(100.0, 0.2)
    assert recorder.attach_device_execution_telemetry(first, 44, observation)

    second = recorder.begin_generation("gaussian", "sigma")

    reports = recorder.recent_reports()
    assert len(reports) == 1
    assert reports[0].generation_id == first
    assert reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_BEFORE_ACCEPTANCE
    assert reports[0].device_execution_telemetry == ((44, observation),)
    assert recorder.generation_for_pipeline_run(44) is None
    assert recorder.active_generation_id == second


def test_terminal_pipeline_waiting_for_preview_is_superseded_without_leaking() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=4)
    )
    first = recorder.begin_generation("gaussian", "sigma")
    recorder.bind_pipeline_run(first, 51)
    recorder.mark_pipeline_terminal(first, 51)
    recorder.mark_pipeline_result_delivered(first, 51)
    recorder.record_phase(first, InteractionLatencyPhase.PIPELINE_ACCEPTED)

    second = recorder.begin_generation("gaussian", "sigma")

    reports = recorder.recent_reports()
    assert len(reports) == 1
    assert (
        reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_DURING_PRESENTATION
    )
    assert recorder.generation_for_pipeline_run(51) is None
    assert recorder.active_generation_id == second


def test_synchronous_pipeline_without_run_id_is_superseded_during_presentation() -> (
    None
):
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=4)
    )
    first = recorder.begin_generation("gaussian", "sigma")
    recorder.record_phase(first, InteractionLatencyPhase.PIPELINE_TERMINAL)
    recorder.record_phase(first, InteractionLatencyPhase.PIPELINE_ACCEPTED)

    second = recorder.begin_generation("gaussian", "sigma")

    reports = recorder.recent_reports()
    assert len(reports) == 1
    assert (
        reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_DURING_PRESENTATION
    )
    assert recorder.active_generation_id == second


def test_excluded_edit_detaches_current_trace_without_starting_a_replacement() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=4)
    )
    generation = recorder.begin_generation("gaussian", "sigma")
    clock.advance(0.025)

    assert recorder.supersede_current(detail="Excluded axis editor committed.")

    assert recorder.active_generation_id is None
    reports = recorder.recent_reports()
    assert len(reports) == 1
    assert reports[0].generation_id == generation
    assert reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_BEFORE_DISPATCH
    assert reports[0].detail == "Excluded axis editor committed."


def test_new_edit_keeps_second_followup_run_mapped_when_first_run_is_terminal() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock)
    )
    first = recorder.begin_generation("gaussian", "sigma")
    recorder.bind_pipeline_run(first, 81)
    recorder.mark_pipeline_terminal(first, 81)
    recorder.bind_pipeline_run(first, 82)

    second = recorder.begin_generation("gaussian", "sigma")

    assert recorder.active_generation_id == second
    assert recorder.is_superseded_in_flight(first)
    assert recorder.generation_for_pipeline_run(82) == first
    assert recorder.mark_pipeline_terminal(first, 82)
    report = recorder.finish_generation(
        first,
        InteractionLatencyOutcome.SUPERSEDED_IN_FLIGHT,
    )
    assert report is not None
    assert report.pipeline_run_ids == (81, 82)


def test_multiple_pipeline_runs_retain_ordered_device_observations() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock)
    )
    generation = recorder.begin_generation("gaussian", "sigma")
    first = DeviceExecutionObservation(100.0, 0.1)
    second = DeviceExecutionObservation(101.0, 0.2)
    for run_id, observation in ((61, first), (62, second)):
        recorder.bind_pipeline_run(generation, run_id)
        assert recorder.attach_device_execution_telemetry(
            generation,
            run_id,
            observation,
        )

    report = recorder.finish_generation(
        generation,
        InteractionLatencyOutcome.COMPLETED_WITHOUT_PREVIEW,
    )

    assert report is not None
    assert report.pipeline_run_ids == (61, 62)
    assert report.device_execution_telemetry == ((61, first), (62, second))


def test_multiple_pipeline_runs_retain_ordered_preparation_observations() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock)
    )
    generation = recorder.begin_generation("subtract_background", "radius_pixels")
    first = PipelinePreparationObservation(100.0, 0.1)
    second = PipelinePreparationObservation(101.0, 0.2, completed=False)
    for run_id, observation in ((71, first), (72, second)):
        recorder.bind_pipeline_run(generation, run_id)
        assert recorder.attach_pre_device_execution_telemetry(
            generation,
            run_id,
            observation,
        )

    report = recorder.finish_generation(
        generation,
        InteractionLatencyOutcome.COMPLETED_WITHOUT_PREVIEW,
    )

    assert report is not None
    assert report.pipeline_run_ids == (71, 72)
    assert report.pre_device_execution_telemetry == (
        (71, first),
        (72, second),
    )


def test_worker_thread_timestamp_is_not_replaced_by_later_clock_sample() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock)
    )
    generation = recorder.begin_generation("gaussian", "sigma")
    clock.advance(0.1)
    recorder.record_phase(generation, InteractionLatencyPhase.WORKER_QUEUED)
    worker_timestamp = clock.value
    clock.advance(3.0)  # queued GUI delivery happens substantially later

    assert recorder.record_phase_at(
        generation,
        InteractionLatencyPhase.WORKER_STARTED,
        worker_timestamp,
    )
    report = recorder.finish_generation(
        generation,
        InteractionLatencyOutcome.FAILED,
    )

    assert report is not None
    assert report.phase_offsets(InteractionLatencyPhase.WORKER_STARTED) == (
        pytest.approx(0.1),
    )


@pytest.mark.parametrize(
    "outcome",
    [InteractionLatencyOutcome.CANCELLED, InteractionLatencyOutcome.FAILED],
)
def test_terminal_nonpublication_outcomes_are_retained(outcome) -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock)
    )
    generation = recorder.begin_generation("median", "radius")
    clock.advance(0.5)

    report = recorder.finish_generation(
        generation,
        outcome,
        terminal_phase=InteractionLatencyPhase.PIPELINE_TERMINAL,
        detail="synthetic terminal result",
    )

    assert report is not None
    assert report.outcome is outcome
    assert report.detail == "synthetic terminal result"
    assert report.phase_offsets(InteractionLatencyPhase.PIPELINE_TERMINAL) == (
        pytest.approx(0.5),
    )


def test_report_history_is_bounded_and_returned_as_immutable_snapshots() -> None:
    clock = _FakeClock()
    recorder = InteractionLatencyRecorder(
        InteractionLatencyTelemetryConfig(clock=clock, history_limit=2)
    )
    for index in range(3):
        generation = recorder.begin_generation("gaussian", f"sigma_{index}")
        clock.advance(0.1)
        recorder.finish_generation(generation, InteractionLatencyOutcome.FAILED)

    reports = recorder.recent_reports()
    assert isinstance(reports, tuple)
    assert [report.generation_id for report in reports] == [2, 3]
    with pytest.raises(FrozenInstanceError):
        reports[0].detail = "mutated"


def test_configuration_rejects_unbounded_or_invalid_diagnostic_settings() -> None:
    with pytest.raises(ValueError, match="history_limit"):
        InteractionLatencyTelemetryConfig(history_limit=0)
    with pytest.raises(TypeError, match="clock"):
        InteractionLatencyTelemetryConfig(clock=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="synchronize"):
        InteractionLatencyTelemetryConfig(
            synchronize_device_phases=1  # type: ignore[arg-type]
        )
