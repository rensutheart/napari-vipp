from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.compute import ComputeMode
from napari_vipp.core.execution import PipelineExecutionFailure, PipelineRunResult
from napari_vipp.core.execution_telemetry import PipelinePreparationObservation
from napari_vipp.core.interaction_telemetry import (
    InteractionLatencyOutcome,
    InteractionLatencyPhase,
    InteractionLatencyTelemetryConfig,
)
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.pipeline import EXECUTION_READY, EXECUTION_STALE
from napari_vipp.ui.diagnostic_workers import ThumbnailContrastLimitResult
from napari_vipp.ui.file_sources import SourceFileLoadResult


class _FakeClock:
    def __init__(self, value: float = 100.0) -> None:
        self.value = float(value)

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float = 0.01) -> None:
        self.value += float(seconds)


def _traced_widget(qtbot, clock: _FakeClock) -> tuple[VippWidget, object]:
    widget = VippWidget(
        _Viewer(np.arange(64, dtype=np.float32).reshape(8, 8)),
        defer_initial_run=True,
        initial_compute_mode=ComputeMode.CPU,
        interaction_latency_telemetry=InteractionLatencyTelemetryConfig(clock=clock),
    )
    qtbot.addWidget(widget)
    widget.run_pipeline = lambda *_args, **_kwargs: None
    gaussian = widget.add_node_from_palette("gaussian_blur")
    widget._debounce_timer.stop()
    widget.graph_view.select_node(gaussian.id)
    return widget, gaussian


def _begin_gaussian_edit(
    widget: VippWidget,
    gaussian,
    *,
    increment: float = 0.25,
) -> int:
    widget.graph_view.select_node(gaussian.id)
    widget._on_param_changed("sigma", float(gaussian.params["sigma"]) + increment)
    widget._debounce_timer.stop()
    generation = widget._interaction_current_generation()
    assert generation is not None
    return generation


def _accept_pipeline_generation(
    widget: VippWidget,
    clock: _FakeClock,
    generation: int,
    *,
    run_id: int = 41,
) -> None:
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    assert recorder.bind_pipeline_run(generation, run_id)
    clock.advance()
    assert recorder.mark_pipeline_terminal_at(generation, run_id, clock.value)
    clock.advance()
    assert widget._interaction_pipeline_terminal(PipelineRunResult(run_id, {})) == (
        generation
    )
    clock.advance()
    assert widget._record_interaction_phase(
        generation,
        InteractionLatencyPhase.PIPELINE_ACCEPTED,
    )
    node_id = recorder.node_id_for_generation(generation)
    widget.pipeline.node_execution_states[node_id] = EXECUTION_READY
    widget.pipeline.completed_node_ids.add(node_id)


def test_widget_rapid_parameter_edits_report_superseded_generation(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)

    first = _begin_gaussian_edit(widget, gaussian)
    clock.advance(0.02)
    second = _begin_gaussian_edit(widget, gaussian)

    assert second != first
    reports = widget.recent_interaction_latency_reports()
    assert len(reports) == 1
    assert reports[0].generation_id == first
    assert reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_BEFORE_DISPATCH
    assert widget._interaction_current_generation() == second


def test_excluded_source_edit_detaches_standard_trace_without_replacement(
    qtbot,
) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    generation = _begin_gaussian_edit(widget, gaussian)
    widget.graph_view.select_node("input")
    source = widget.pipeline.nodes["input"]
    value = widget._image_source_value(source)
    value["layer_name"] = "different layer"

    widget._on_image_source_changed(value)
    widget._debounce_timer.stop()

    assert widget._interaction_current_generation() is None
    reports = widget.recent_interaction_latency_reports()
    assert reports[-1].generation_id == generation
    assert reports[-1].outcome is InteractionLatencyOutcome.SUPERSEDED_BEFORE_DISPATCH
    assert "Excluded UI edit" in reports[-1].detail


def test_live_source_invalidation_detaches_standard_trace(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    generation = _begin_gaussian_edit(widget, gaussian)
    layer = widget.viewer.layers[0]
    widget._live_source_node_layers["input"] = layer

    widget._on_live_source_invalidated(layer)

    assert widget._interaction_current_generation() is None
    report = widget.recent_interaction_latency_reports()[-1]
    assert report.generation_id == generation
    assert report.outcome is InteractionLatencyOutcome.SUPERSEDED_BEFORE_DISPATCH


def test_deleting_edited_leaf_closes_trace_without_surviving_dirty_target(
    qtbot,
) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    generation = _begin_gaussian_edit(widget, gaussian)
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    recorder.record_phase(generation, InteractionLatencyPhase.PIPELINE_TERMINAL)
    recorder.record_phase(generation, InteractionLatencyPhase.PIPELINE_ACCEPTED)
    assert not widget.pipeline.descendants_inclusive({gaussian.id}) - {gaussian.id}

    widget._delete_nodes({gaussian.id})

    assert gaussian.id not in widget.pipeline.nodes
    assert widget._interaction_current_generation() is None
    report = widget.recent_interaction_latency_reports()[-1]
    assert report.generation_id == generation
    assert report.outcome is InteractionLatencyOutcome.SUPERSEDED_DURING_PRESENTATION


def test_workflow_session_install_closes_waiting_presentation_trace(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    generation = _begin_gaussian_edit(widget, gaussian)
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    recorder.record_phase(generation, InteractionLatencyPhase.PIPELINE_TERMINAL)
    recorder.record_phase(generation, InteractionLatencyPhase.PIPELINE_ACCEPTED)
    widget._interaction_thumbnail_generations[17] = (generation, frozenset())
    replacement = widget._workflow_tabs.create_blank(make_current=False)

    widget._install_workflow_tab_session(replacement)

    assert widget._interaction_current_generation() is None
    assert widget._interaction_thumbnail_generations == {}
    report = widget.recent_interaction_latency_reports()[-1]
    assert report.generation_id == generation
    assert report.outcome is InteractionLatencyOutcome.SUPERSEDED_DURING_PRESENTATION


def test_history_restore_closes_waiting_presentation_trace(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    snapshot = widget._current_history_snapshot()
    generation = _begin_gaussian_edit(widget, gaussian)
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    recorder.record_phase(generation, InteractionLatencyPhase.PIPELINE_TERMINAL)
    recorder.record_phase(generation, InteractionLatencyPhase.PIPELINE_ACCEPTED)
    widget._interaction_thumbnail_generations[18] = (generation, frozenset())

    widget._restore_history_snapshot(snapshot)

    assert widget._interaction_current_generation() is None
    assert widget._interaction_thumbnail_generations == {}
    report = widget.recent_interaction_latency_reports()[-1]
    assert report.generation_id == generation
    assert report.outcome is InteractionLatencyOutcome.SUPERSEDED_DURING_PRESENTATION


def test_immediate_thumbnail_publication_finishes_only_after_card_commit(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    widget.preview_mode_combo.setCurrentText("Slice")
    widget.thumbnail_scope_combo.setCurrentText("Slice")
    generation = _begin_gaussian_edit(widget, gaussian)
    _accept_pipeline_generation(widget, clock, generation)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    state = image_state_from_array(data)

    clock.advance()
    widget._update_node_thumbnail(
        gaussian.id,
        data,
        state,
        0,
        queue_stack_contrast=False,
    )

    assert widget.graph_view.node_has_thumbnail(gaussian.id)
    report = widget.recent_interaction_latency_reports()[-1]
    assert report.outcome is InteractionLatencyOutcome.PUBLISHED
    assert report.events[-1].phase is InteractionLatencyPhase.PUBLICATION_ACCEPTED
    assert report.phase_offsets(InteractionLatencyPhase.THUMBNAIL_RENDER_STARTED)
    assert report.phase_offsets(InteractionLatencyPhase.THUMBNAIL_RENDER_FINISHED)


def test_cached_manual_node_edit_does_not_claim_stale_thumbnail_publication(
    qtbot,
    monkeypatch,
) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    widget.run_pipeline = VippWidget.run_pipeline.__get__(widget, VippWidget)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    original_is_manual_node = widget.pipeline.is_manual_node
    monkeypatch.setattr(
        widget.pipeline,
        "is_manual_node",
        lambda node_id: node_id == gaussian.id or original_is_manual_node(node_id),
    )
    widget._connect_nodes("input", gaussian.id)
    widget.run_pipeline(force_sync=True, manual_node_ids={gaussian.id})
    widget.preview_mode_combo.setCurrentText("Slice")
    widget.thumbnail_scope_combo.setCurrentText("Slice")
    widget._update_thumbnails()
    cached_output = widget.pipeline.outputs[gaussian.id]
    assert cached_output is not None
    assert widget.pipeline.node_execution_states[gaussian.id] == EXECUTION_READY
    assert widget.graph_view.node_has_thumbnail(gaussian.id)

    generation = _begin_gaussian_edit(widget, gaussian)
    assert widget.pipeline.node_execution_states[gaussian.id] == EXECUTION_STALE
    assert widget.pipeline.outputs[gaussian.id] is cached_output
    widget.run_pipeline(force_sync=True)

    report = widget.recent_interaction_latency_reports()[-1]
    assert report.generation_id == generation
    assert report.outcome is InteractionLatencyOutcome.COMPLETED_WITHOUT_PREVIEW
    assert not report.phase_offsets(InteractionLatencyPhase.PUBLICATION_ACCEPTED)
    assert widget.pipeline.node_execution_states[gaussian.id] == EXECUTION_STALE
    assert widget.pipeline.outputs[gaussian.id] is cached_output


def test_inline_thumbnail_statistics_trace_queue_work_delivery_and_publication(
    qtbot,
) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    widget.preview_mode_combo.setCurrentText("Slice")
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget.thumbnail_contrast_combo.setCurrentText("Min-max")
    widget._preview_disabled_node_ids.add("input")
    generation = _begin_gaussian_edit(widget, gaussian)
    _accept_pipeline_generation(widget, clock, generation)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    state = image_state_from_array(data)
    widget.pipeline.outputs[gaussian.id] = data
    widget.pipeline.output_states[gaussian.id] = state
    widget.pipeline.node_outputs[gaussian.id] = [data]
    widget.pipeline.node_output_states[gaussian.id] = [state]

    widget._update_node_thumbnail(
        gaussian.id,
        data,
        state,
        0,
        queue_stack_contrast=True,
    )
    assert widget._queued_thumbnail_contrast_limit_requests
    clock.advance()
    widget._start_thumbnail_contrast_limit_run()

    report = widget.recent_interaction_latency_reports()[-1]
    phases = tuple(event.phase for event in report.events)
    expected = (
        InteractionLatencyPhase.THUMBNAIL_STATISTICS_QUEUED,
        InteractionLatencyPhase.THUMBNAIL_STATISTICS_STARTED,
        InteractionLatencyPhase.THUMBNAIL_STATISTICS_FINISHED,
        InteractionLatencyPhase.THUMBNAIL_STATISTICS_RESULT_DELIVERED,
        InteractionLatencyPhase.THUMBNAIL_RENDER_STARTED,
        InteractionLatencyPhase.THUMBNAIL_RENDER_FINISHED,
        InteractionLatencyPhase.PUBLICATION_ACCEPTED,
    )
    positions = tuple(phases.index(phase) for phase in expected)
    assert positions == tuple(sorted(positions))
    assert report.outcome is InteractionLatencyOutcome.PUBLISHED


def test_unrelated_thumbnail_statistics_failure_does_not_fail_target(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    widget.preview_mode_combo.setCurrentText("Slice")
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    widget.thumbnail_contrast_combo.setCurrentText("Min-max")
    widget._preview_disabled_node_ids.add("input")
    generation = _begin_gaussian_edit(widget, gaussian)
    _accept_pipeline_generation(widget, clock, generation)
    target_data = np.arange(64, dtype=np.float32).reshape(8, 8)
    target_state = image_state_from_array(target_data)
    other_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    target = widget._thumbnail_contrast_limit_request(
        gaussian.id,
        target_data,
        target_state,
        "Min-max",
        "Stack",
        "image",
    )
    other = widget._thumbnail_contrast_limit_request(
        "input",
        other_data,
        image_state_from_array(other_data),
        "Min-max",
        "Stack",
        "image",
    )
    assert target is not None and other is not None
    widget.pipeline.outputs[gaussian.id] = target_data
    widget.pipeline.output_states[gaussian.id] = target_state
    widget.pipeline.node_outputs[gaussian.id] = [target_data]
    widget.pipeline.node_output_states[gaussian.id] = [target_state]
    run_id = 73
    widget._active_thumbnail_contrast_run_id = run_id
    widget._pending_thumbnail_contrast_limit_keys.update({target.key, other.key})
    widget._interaction_thumbnail_generations[run_id] = (
        generation,
        frozenset({target.key}),
    )
    widget._record_interaction_phase(
        generation,
        InteractionLatencyPhase.THUMBNAIL_STATISTICS_QUEUED,
    )

    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            run_id,
            frozenset({target.key, other.key}),
            {target.key: (0.0, 63.0)},
            error="unrelated card failed",
            errors={other.key: "unrelated card failed"},
        )
    )

    report = widget.recent_interaction_latency_reports()[-1]
    assert report.generation_id == generation
    assert report.outcome is InteractionLatencyOutcome.PUBLISHED


def test_cancelled_thumbnail_cleanup_failure_is_reported_as_failed(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    generation = _begin_gaussian_edit(widget, gaussian)
    _accept_pipeline_generation(widget, clock, generation)
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    request = widget._thumbnail_contrast_limit_request(
        gaussian.id,
        data,
        image_state_from_array(data),
        "Min-max",
        "Stack",
        "image",
    )
    assert request is not None
    run_id = 79
    widget._active_thumbnail_contrast_run_id = run_id
    widget._pending_thumbnail_contrast_limit_keys.add(request.key)
    widget._interaction_thumbnail_generations[run_id] = (
        generation,
        frozenset({request.key}),
    )

    widget._on_thumbnail_contrast_limit_finished(
        ThumbnailContrastLimitResult(
            run_id,
            frozenset({request.key}),
            {},
            error="synthetic thumbnail cleanup failure",
            cancelled=True,
            cleanup_failed=True,
        )
    )

    report = widget.recent_interaction_latency_reports()[-1]
    assert report.generation_id == generation
    assert report.outcome is InteractionLatencyOutcome.FAILED
    assert "cleanup failure" in report.detail


def test_source_load_error_terminalizes_predispatch_interaction(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    generation = _begin_gaussian_edit(widget, gaussian)
    widget._active_source_load_id = 91
    widget._source_load_pending = False

    widget._on_source_file_load_finished(
        SourceFileLoadResult(
            91,
            {},
            error="synthetic source failure",
            node_id="input",
        )
    )

    report = widget.recent_interaction_latency_reports()[-1]
    assert report.generation_id == generation
    assert report.outcome is InteractionLatencyOutcome.FAILED
    assert "synthetic source failure" in report.detail


def test_hostile_resident_metadata_cannot_interrupt_pipeline_delivery(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    generation = _begin_gaussian_edit(widget, gaussian)
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    assert recorder.bind_pipeline_run(generation, 101)
    hostile_statistics = SimpleNamespace(
        elapsed_seconds=float("nan"),
        decision=SimpleNamespace(backend=""),
        actual_backend="",
        algorithm_id="",
        runtime_id="",
        device_id="",
        input_path="resident_borrow",
        logical_input_host_to_device_bytes=0,
        auxiliary_host_to_device_bytes=0,
        device_to_host_bytes=0,
    )
    result = SimpleNamespace(
        run_id=101,
        device_execution_telemetry=None,
        resident_thumbnail_statistics=(
            SimpleNamespace(
                node_id=gaussian.id,
                output_port=0,
                contrast_mode="Percentile",
                result=hostile_statistics,
            ),
        ),
    )

    assert widget._interaction_pipeline_terminal(result) == generation
    assert widget.recent_interaction_latency_reports() == ()
    assert recorder.node_id_for_generation(generation) == gaussian.id


def test_pipeline_delivery_attaches_run_correlated_preparation_telemetry(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    generation = _begin_gaussian_edit(widget, gaussian)
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    assert recorder.bind_pipeline_run(generation, 102)
    preparation = PipelinePreparationObservation(100.0, 0.25)

    assert (
        widget._interaction_pipeline_terminal(
            PipelineRunResult(
                102,
                {},
                pre_device_execution_telemetry=preparation,
            )
        )
        == generation
    )
    report = widget._finish_interaction_generation(
        generation,
        InteractionLatencyOutcome.COMPLETED_WITHOUT_PREVIEW,
    )

    assert report is not None
    assert report.pre_device_execution_telemetry == ((102, preparation),)


def test_cancelled_active_run_also_terminalizes_queued_rapid_edit(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    first = _begin_gaussian_edit(widget, gaussian)
    assert recorder.bind_pipeline_run(first, 111)
    second = _begin_gaussian_edit(widget, gaussian)
    widget._active_pipeline_run_id = 111
    widget._pipeline_user_cancel_requested_run_id = 111
    widget._pipeline_run_pending = True

    widget._on_background_pipeline_finished(PipelineRunResult(111, {}, cancelled=True))

    reports = widget.recent_interaction_latency_reports()
    assert tuple(report.generation_id for report in reports) == (first, second)
    assert reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_IN_FLIGHT
    assert reports[1].outcome is InteractionLatencyOutcome.CANCELLED
    assert widget._interaction_current_generation() is None


def test_cleanup_quarantine_terminalizes_queued_rapid_edit(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    first = _begin_gaussian_edit(widget, gaussian)
    assert recorder.bind_pipeline_run(first, 112)
    second = _begin_gaussian_edit(widget, gaussian)
    widget._active_pipeline_run_id = 112
    widget._pipeline_run_pending = True

    widget._on_background_pipeline_finished(
        PipelineRunResult(
            112,
            {},
            error="synthetic cleanup failure",
            failure=PipelineExecutionFailure(
                kind="failed",
                error_type="RuntimeCleanupError",
                message="synthetic cleanup failure",
                cleanup_succeeded=False,
            ),
        )
    )

    reports = widget.recent_interaction_latency_reports()
    assert tuple(report.generation_id for report in reports) == (first, second)
    assert reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_IN_FLIGHT
    assert reports[1].outcome is InteractionLatencyOutcome.FAILED
    assert widget._interaction_current_generation() is None


def test_cancel_cleanup_failure_marks_queued_edit_failed(qtbot) -> None:
    clock = _FakeClock()
    widget, gaussian = _traced_widget(qtbot, clock)
    recorder = widget._interaction_latency_recorder
    assert recorder is not None
    first = _begin_gaussian_edit(widget, gaussian)
    assert recorder.bind_pipeline_run(first, 113)
    second = _begin_gaussian_edit(widget, gaussian)
    widget._active_pipeline_run_id = 113
    widget._pipeline_user_cancel_requested_run_id = 113
    widget._pipeline_run_pending = True

    widget._on_background_pipeline_finished(
        PipelineRunResult(
            113,
            {},
            cancelled=True,
            failure=PipelineExecutionFailure(
                kind="cancelled",
                error_type="RuntimeCleanupError",
                message="synthetic cancel cleanup failure",
                cleanup_succeeded=False,
            ),
        )
    )

    reports = widget.recent_interaction_latency_reports()
    assert tuple(report.generation_id for report in reports) == (first, second)
    assert reports[0].outcome is InteractionLatencyOutcome.SUPERSEDED_IN_FLIGHT
    assert reports[1].outcome is InteractionLatencyOutcome.FAILED
    assert "cleanup failed" in reports[1].detail
