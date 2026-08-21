from __future__ import annotations

import pytest

from napari_vipp.core.compute import OutputPortKey
from napari_vipp.core.execution_telemetry import (
    DeviceExecutionObservation,
    DeviceExecutionPhase,
    DeviceExecutionSpan,
    DeviceExecutionTelemetryConfig,
    DeviceTerminalMemorySnapshot,
    DeviceTransferDirection,
    PipelinePreparationObservation,
    PipelinePreparationPhase,
    PipelinePreparationSpan,
)


def _transfer_span(
    phase: DeviceExecutionPhase,
    *,
    port: OutputPortKey,
    start: float,
    elapsed: float,
    byte_count: int | None,
    synchronized: bool,
    succeeded: bool = True,
) -> DeviceExecutionSpan:
    return DeviceExecutionSpan(
        phase=phase,
        start_offset_seconds=start,
        elapsed_seconds=elapsed,
        runtime_id="fake-device",
        device_id="fake:0",
        segment_id="segment-1",
        node_id=port.node_id,
        port=port,
        byte_count=byte_count,
        synchronized=synchronized,
        succeeded=succeeded,
    )


def test_device_execution_observation_summarizes_directional_transfers():
    input_port = OutputPortKey("input", 0)
    output_port = OutputPortKey("result", 0)
    observation = DeviceExecutionObservation(
        started_monotonic_seconds=10.0,
        elapsed_seconds=2.0,
        spans=(
            _transfer_span(
                DeviceExecutionPhase.HOST_TO_DEVICE,
                port=input_port,
                start=0.1,
                elapsed=0.2,
                byte_count=128,
                synchronized=True,
            ),
            _transfer_span(
                DeviceExecutionPhase.HOST_TO_DEVICE,
                port=input_port,
                start=0.4,
                elapsed=0.1,
                byte_count=None,
                synchronized=False,
                succeeded=False,
            ),
            _transfer_span(
                DeviceExecutionPhase.DEVICE_TO_HOST,
                port=output_port,
                start=0.7,
                elapsed=0.3,
                byte_count=64,
                synchronized=True,
            ),
        ),
        synchronized_device_phases=True,
    )

    assert observation.host_to_device.direction is (
        DeviceTransferDirection.HOST_TO_DEVICE
    )
    assert observation.host_to_device.count == 2
    assert observation.host_to_device.succeeded_count == 1
    assert observation.host_to_device.byte_count == 128
    assert observation.host_to_device.unknown_byte_count == 1
    assert observation.host_to_device.elapsed_seconds == pytest.approx(0.3)
    assert observation.host_to_device.all_synchronized is False
    assert observation.device_to_host.direction is (
        DeviceTransferDirection.DEVICE_TO_HOST
    )
    assert observation.device_to_host.count == 1
    assert observation.device_to_host.byte_count == 64
    assert observation.device_to_host.all_synchronized is True


def test_terminal_memory_snapshot_reports_only_private_pool_cleanup():
    snapshot = DeviceTerminalMemorySnapshot(
        runtime_id="fake-device",
        device_id="fake:0",
        topology="discrete",
        device_total_bytes=4096,
        device_free_bytes=2048,
        runtime_live_bytes=0,
        runtime_reserved_bytes=0,
        out_of_pool_bytes=512,
    )
    observation = DeviceExecutionObservation(
        started_monotonic_seconds=1.0,
        elapsed_seconds=0.5,
        terminal_memory_snapshots=(snapshot,),
    )

    assert snapshot.private_allocations_released is True
    assert snapshot.out_of_pool_bytes == 512
    assert observation.terminal_memory_snapshots == (snapshot,)


def test_observation_rejects_duplicate_terminal_memory_identity():
    snapshot = DeviceTerminalMemorySnapshot(
        runtime_id="fake-device",
        device_id="fake:0",
        topology="discrete",
    )

    with pytest.raises(ValueError, match="at most one snapshot"):
        DeviceExecutionObservation(
            started_monotonic_seconds=1.0,
            elapsed_seconds=0.5,
            terminal_memory_snapshots=(snapshot, snapshot),
        )


@pytest.mark.parametrize(
    ("changes", "match"),
    (
        ({"port": None}, "require port"),
        ({"node_id": "wrong"}, "identify its output port"),
        ({"byte_count": -1}, "non-negative integer"),
    ),
)
def test_transfer_span_rejects_incomplete_or_invalid_identity(changes, match):
    values = {
        "phase": DeviceExecutionPhase.HOST_TO_DEVICE,
        "start_offset_seconds": 0.0,
        "elapsed_seconds": 0.1,
        "runtime_id": "fake-device",
        "device_id": "fake:0",
        "segment_id": "segment-1",
        "node_id": "input",
        "port": OutputPortKey("input", 0),
        "byte_count": 16,
        "synchronized": False,
    }
    values.update(changes)

    with pytest.raises(ValueError, match=match):
        DeviceExecutionSpan(**values)


def test_observation_rejects_span_outside_completed_duration():
    span = _transfer_span(
        DeviceExecutionPhase.DEVICE_TO_HOST,
        port=OutputPortKey("result", 0),
        start=0.9,
        elapsed=0.2,
        byte_count=32,
        synchronized=True,
    )

    with pytest.raises(ValueError, match="fit inside"):
        DeviceExecutionObservation(
            started_monotonic_seconds=3.0,
            elapsed_seconds=1.0,
            spans=(span,),
        )


def test_telemetry_configuration_validates_clock_and_barrier_flag():
    with pytest.raises(TypeError, match="clock"):
        DeviceExecutionTelemetryConfig(clock=None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="boolean"):
        DeviceExecutionTelemetryConfig(  # type: ignore[arg-type]
            synchronize_device_phases=1
        )


def test_pipeline_preparation_observation_allows_repeated_phase_categories():
    observation = PipelinePreparationObservation(
        started_monotonic_seconds=2.0,
        elapsed_seconds=1.0,
        spans=(
            PipelinePreparationSpan(
                PipelinePreparationPhase.CACHE_PREPARATION,
                0.1,
                0.2,
            ),
            PipelinePreparationSpan(
                PipelinePreparationPhase.WORKLOAD_PREPARATION,
                0.4,
                0.1,
            ),
            PipelinePreparationSpan(
                PipelinePreparationPhase.CACHE_PREPARATION,
                0.7,
                0.1,
            ),
        ),
    )

    assert observation.completed is True
    assert len(observation.spans_for(PipelinePreparationPhase.CACHE_PREPARATION)) == 2


def test_pipeline_preparation_observation_rejects_overlapping_spans():
    with pytest.raises(ValueError, match="ordered and non-overlapping"):
        PipelinePreparationObservation(
            started_monotonic_seconds=2.0,
            elapsed_seconds=1.0,
            spans=(
                PipelinePreparationSpan(
                    PipelinePreparationPhase.GRAPH_RESTORATION,
                    0.1,
                    0.5,
                ),
                PipelinePreparationSpan(
                    PipelinePreparationPhase.CACHE_PREPARATION,
                    0.4,
                    0.1,
                ),
            ),
        )
