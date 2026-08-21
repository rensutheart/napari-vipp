from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import threading
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.compute import OutputPortKey
from napari_vipp.core.execution_telemetry import (
    DeviceExecutionObservation,
    DeviceExecutionPhase,
    DeviceExecutionSpan,
    DeviceSynchronizationPoint,
    DeviceTerminalMemorySnapshot,
    PipelinePreparationObservation,
    PipelinePreparationPhase,
    PipelinePreparationSpan,
)
from napari_vipp.core.interaction_telemetry import (
    InteractionLatencyEvent,
    InteractionLatencyPhase,
    InteractionLatencyReport,
    InteractionResidentThumbnailStatistics,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "benchmark_ui_interaction.py"


@pytest.fixture(scope="module")
def benchmark_script():
    module_name = "_vipp_test_benchmark_ui_interaction"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def test_help_is_provider_and_ui_lazy_in_a_fresh_process() -> None:
    code = "\n".join(
        (
            "import builtins, runpy, sys",
            "real_import = builtins.__import__",
            "blocked = ('cupy', 'cupyx', 'cucim', 'napari', 'napari_vipp', 'qtpy')",
            "def guarded_import(name, *args, **kwargs):",
            "    blocked_import = any(",
            "        name == item or name.startswith(item + '.') for item in blocked",
            "    )",
            "    if blocked_import:",
            "        raise RuntimeError('help imported optional runtime: ' + name)",
            "    return real_import(name, *args, **kwargs)",
            "builtins.__import__ = guarded_import",
            "sys.argv = [sys.argv[1], '--help']",
            "runpy.run_path(sys.argv[0], run_name='__main__')",
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT_PATH)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--input-profile" in completed.stdout
    assert "--in-flight-radii" in completed.stdout
    assert "--synchronize-device-phases" in completed.stdout
    assert "--device-id" in completed.stdout


def test_source_provenance_covers_affinity_and_terminal_memory_producers(
    benchmark_script,
) -> None:
    paths = set(benchmark_script._source_provenance_paths())

    assert {
        "src/napari_vipp/compute_policies/phase1-gpu-public-v9.json",
        "src/napari_vipp/core/compute_cache.py",
        "src/napari_vipp/core/compute_policy.py",
        "src/napari_vipp/core/compute_planning.py",
        "src/napari_vipp/core/compute_registry.py",
        "src/napari_vipp/core/compute_specs.py",
        "src/napari_vipp/core/device_execution.py",
        "src/napari_vipp/core/execution.py",
        "src/napari_vipp/core/execution_telemetry.py",
        "src/napari_vipp/core/gpu/cucim_background.py",
        "src/napari_vipp/core/gpu/cupy_runtime.py",
        "src/napari_vipp/core/gpu/cupy_thumbnail_statistics.py",
        "src/napari_vipp/core/gpu/cupy_median.py",
        "src/napari_vipp/core/operations.py",
        "src/napari_vipp/core/pipeline.py",
        "src/napari_vipp/core/source_identity.py",
        "src/napari_vipp/core/thumbnail_statistics.py",
        "src/napari_vipp/ui/compute_setup.py",
        "src/napari_vipp/ui/compute_setup_dialog.py",
        "src/napari_vipp/ui/diagnostic_workers.py",
        "src/napari_vipp/ui/source_adapter.py",
    }.issubset(paths)
    assert all("/_tests/" not in f"/{path}/" for path in paths)

    provenance = benchmark_script._source_provenance()

    assert len(provenance["tree_sha256"]) == 64
    assert [record["path"] for record in provenance["files"]] == sorted(paths)


def test_normalization_enforces_truthful_scenario_names(benchmark_script) -> None:
    defaults = _normalization_kwargs(benchmark_script)
    normalized = benchmark_script._normalized_inputs(**defaults)

    assert normalized["cold_radius"] == 3.0
    assert normalized["warm_radii"] == (4.0, 5.0)
    assert normalized["revisit_radii"] == (3.0, 4.0)
    assert normalized["device_id"] == ""
    assert (
        benchmark_script._normalized_inputs(**(defaults | {"device_id": " cuda:7 "}))[
            "device_id"
        ]
        == "cuda:7"
    )

    with pytest.raises(ValueError, match="authored radius"):
        benchmark_script._normalized_inputs(**(defaults | {"cold_radius": 2.0}))
    with pytest.raises(ValueError, match="Warm radii"):
        benchmark_script._normalized_inputs(**(defaults | {"warm_radii": (4, 4)}))
    with pytest.raises(ValueError, match="Warm radii"):
        benchmark_script._normalized_inputs(**(defaults | {"warm_radii": (2, 5)}))
    with pytest.raises(ValueError, match="Revisit radii"):
        benchmark_script._normalized_inputs(**(defaults | {"revisit_radii": (3, 6)}))
    with pytest.raises(ValueError, match="requires Stack"):
        benchmark_script._normalized_inputs(
            **(
                defaults
                | {
                    "input_profile": "resident_large_float32",
                    "thumbnail_scope": "slice",
                }
            )
        )
    with pytest.raises(ValueError, match="control characters"):
        benchmark_script._normalized_inputs(
            **(defaults | {"device_id": "cuda:0\npoisoned"})
        )


def test_derived_large_profile_is_an_owned_read_only_revision(
    benchmark_script,
    monkeypatch,
) -> None:
    from napari_vipp.core import host_memory
    from napari_vipp.core.pipeline import SourcePayload
    from napari_vipp.core.source_identity import BundledSampleRevisionToken

    monkeypatch.setattr(host_memory, "capture_host_memory", lambda: object())
    monkeypatch.setattr(
        host_memory,
        "preflight_host_allocation",
        lambda *_args, **_kwargs: SimpleNamespace(
            allowed=True,
            reason="",
            as_dict=lambda: {"allowed": True},
        ),
    )

    data = np.arange(3 * 2 * 4 * 4, dtype=np.uint16).reshape(3, 2, 4, 4)
    data.setflags(write=False)
    payloads = {
        "sample": SourcePayload(
            data,
            {"vipp_shape": data.shape},
            "sample",
            revision_token=BundledSampleRevisionToken("sample"),
        )
    }
    widget = SimpleNamespace(_sample_payloads=lambda: payloads)

    identity = benchmark_script._configure_input_profile(
        widget,
        sample_name="sample",
        profile="bounded_large",
    )

    derived = payloads["sample"]
    assert isinstance(derived.revision_token, BundledSampleRevisionToken)
    assert derived.revision_token.catalog_schema == ("vipp-issue27-derived-samples-v1")
    assert isinstance(derived.data, np.ndarray)
    assert not derived.data.flags.writeable
    assert identity["owned_read_only_revision"] is True


def test_explicit_device_is_installed_before_widget_execution(
    benchmark_script,
    monkeypatch,
) -> None:
    normalized = benchmark_script._normalized_inputs(
        **(_normalization_kwargs(benchmark_script) | {"device_id": "cuda:7"})
    )
    spec = benchmark_script.WorkerSpec(mode="prefer_gpu", **normalized)
    captured = {}

    class StopAfterConstruction(RuntimeError):
        pass

    class FakeApplication:
        @staticmethod
        def instance():
            return None

        def __init__(self, _args):
            pass

    class FakeWidget:
        def __init__(self, _viewer, **kwargs):
            captured.update(kwargs)
            raise StopAfterConstruction

    monkeypatch.setitem(
        sys.modules,
        "qtpy.QtWidgets",
        SimpleNamespace(QApplication=FakeApplication),
    )
    monkeypatch.setitem(
        sys.modules,
        "napari_vipp._widget",
        SimpleNamespace(VippWidget=FakeWidget),
    )

    with pytest.raises(StopAfterConstruction):
        benchmark_script._drive_widget_session(
            spec=spec,
            workflow_document={},
            workflow_facts={},
        )

    assert captured["defer_initial_run"] is True
    assert captured["initial_compute_runtime_id"] == "cuda-cupy"
    assert captured["initial_compute_device_id"] == "cuda:7"
    assert captured["initial_compute_device_display_name"] == "cuda:7"


def test_fresh_worker_command_forwards_explicit_device_without_provider_probe(
    benchmark_script,
    monkeypatch,
) -> None:
    normalized = benchmark_script._normalized_inputs(
        **(_normalization_kwargs(benchmark_script) | {"device_id": "cuda:7"})
    )
    spec = benchmark_script.WorkerSpec(mode="prefer_gpu", **normalized)
    captured_command = []

    def fake_run(command, **_kwargs):
        captured_command.extend(command)
        output = Path(command[command.index("--output") + 1])
        output.write_text('{"worker": "synthetic"}', encoding="utf-8")
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(benchmark_script.subprocess, "run", fake_run)

    assert benchmark_script._launch_mode_worker(spec) == {"worker": "synthetic"}
    index = captured_command.index("--device-id")
    assert captured_command[index + 1] == "cuda:7"


def test_pipeline_observers_snapshot_small_facts_and_restore(benchmark_script) -> None:
    class Widget:
        def __init__(self) -> None:
            self.terminals = []
            self.node_starts = []

        def _interaction_pipeline_terminal(self, result):
            self.terminals.append(result.run_id)

        def _on_background_pipeline_node_started(self, payload):
            self.node_starts.append(tuple(payload))

    widget = Widget()
    original_terminal = widget._interaction_pipeline_terminal
    original_node_start = widget._on_background_pipeline_node_started
    observed_results, restore_results = (
        benchmark_script._install_pipeline_result_observer(widget)
    )
    observed_starts, restore_starts = (
        benchmark_script._install_pipeline_node_start_observer(widget)
    )
    enormous_pipeline = object()
    failure = SimpleNamespace(
        kind="cancelled",
        error_type="OperationCancelled",
        message="superseded",
        cleanup_succeeded=True,
    )
    result = SimpleNamespace(
        run_id=7,
        pipeline=enormous_pipeline,
        cancelled=True,
        error="superseded",
        failure=failure,
        execution_report=None,
        device_execution_telemetry=None,
    )

    widget._interaction_pipeline_terminal(result)
    payload = [7, "subtract_background_1"]
    widget._on_background_pipeline_node_started(payload)
    payload[1] = "mutated"

    snapshot = observed_results[7]
    assert snapshot is not result
    assert not hasattr(snapshot, "pipeline")
    assert snapshot.cancelled is True
    assert snapshot.cleanup_succeeded is True
    assert snapshot.device_execution_returned is False
    assert observed_starts == {(7, "subtract_background_1")}
    assert widget.terminals == [7]
    assert widget.node_starts == [(7, "subtract_background_1")]

    restore_starts()
    restore_results()
    assert widget._interaction_pipeline_terminal == original_terminal
    assert widget._on_background_pipeline_node_started == original_node_start


def test_breakdown_unions_nested_device_spans_and_keeps_resident_unpositioned(
    benchmark_script,
) -> None:
    report = _published_report(
        include_device=True,
        include_resident=True,
        synchronized_device=True,
    )

    breakdown = benchmark_script._derived_latency_breakdown(report)

    assert breakdown["total_edit_to_outcome_seconds"] == pytest.approx(1.0)
    assert breakdown["edit_to_publication_seconds"] == pytest.approx(1.0)
    assert breakdown["category_seconds"]["device_execution"] == pytest.approx(0.36)
    device = breakdown["pipeline_runs"][0]["device"]
    assert device["phase_union_seconds"]["device_operation"] == pytest.approx(0.25)
    assert device["phase_union_seconds"]["device_synchronize"] == pytest.approx(0.12)
    assert device["observed_span_union_seconds"] == pytest.approx(0.36)
    assert device["cross_phase_overlap_seconds"] == pytest.approx(0.1)
    resident = breakdown["resident_thumbnail_statistics_diagnostic"]
    assert resident["observation_count"] == 1
    assert resident["elapsed_seconds_sum"] == pytest.approx(0.05)
    assert resident["positioned_in_interval_union"] is False
    assert "pipeline_other" in resident["accounting_note"]


def test_published_evidence_fails_closed_on_missing_sidecars_or_fallback(
    benchmark_script,
) -> None:
    cpu_record = benchmark_script._interaction_report_record(
        _published_report(),
        execution_summary=_execution_summary("cpu"),
    )
    benchmark_script._validate_report_evidence(
        cpu_record,
        requested_mode="cpu",
        thumbnail_scope="slice",
        scenario_name="cold_fresh_process_first_edit",
    )

    missing_preparation = deepcopy(cpu_record)
    missing_preparation["pre_device_execution_telemetry"] = []
    with pytest.raises(benchmark_script.EvidenceError, match="preparation"):
        benchmark_script._validate_report_evidence(
            missing_preparation,
            requested_mode="cpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )

    fallback = deepcopy(cpu_record)
    fallback["execution_summary"]["accepted_execution"]["fallback_records"] = [
        {"reason_code": "test-fallback"}
    ]
    with pytest.raises(benchmark_script.EvidenceError, match="fallback"):
        benchmark_script._validate_report_evidence(
            fallback,
            requested_mode="cpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )

    gpu_record = benchmark_script._interaction_report_record(
        _published_report(include_device=True, gpu_preparation=True),
        execution_summary=_execution_summary("prefer_gpu"),
    )
    benchmark_script._validate_report_evidence(
        gpu_record,
        requested_mode="prefer_gpu",
        thumbnail_scope="slice",
        scenario_name="cold_fresh_process_first_edit",
    )
    missing_device = deepcopy(gpu_record)
    missing_device["device_execution_telemetry"] = []
    with pytest.raises(benchmark_script.EvidenceError, match="device telemetry"):
        benchmark_script._validate_report_evidence(
            missing_device,
            requested_mode="prefer_gpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )
    missing_setup = deepcopy(gpu_record)
    preparation = missing_setup["pre_device_execution_telemetry"][0]["observation"]
    preparation["spans"] = [
        span for span in preparation["spans"] if span["phase"] != "accelerator_setup"
    ]
    with pytest.raises(benchmark_script.EvidenceError, match="accelerator_setup"):
        benchmark_script._validate_report_evidence(
            missing_setup,
            requested_mode="prefer_gpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )
    missing_operation = deepcopy(gpu_record)
    observation = missing_operation["device_execution_telemetry"][0]["observation"]
    observation["spans"] = [
        span for span in observation["spans"] if span["phase"] != "device_operation"
    ]
    with pytest.raises(benchmark_script.EvidenceError, match="target device operation"):
        benchmark_script._validate_report_evidence(
            missing_operation,
            requested_mode="prefer_gpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )
    missing_transfer = deepcopy(gpu_record)
    transfer = missing_transfer["device_execution_telemetry"][0]["observation"][
        "transfer_summary"
    ]["device_to_host"]
    transfer["byte_count"] = 0
    with pytest.raises(benchmark_script.EvidenceError, match="device_to_host"):
        benchmark_script._validate_report_evidence(
            missing_transfer,
            requested_mode="prefer_gpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )

    missing_terminal_memory = deepcopy(gpu_record)
    missing_terminal_memory["device_execution_telemetry"][0]["observation"][
        "terminal_memory_snapshots"
    ] = []
    with pytest.raises(benchmark_script.EvidenceError, match="private-memory"):
        benchmark_script._validate_report_evidence(
            missing_terminal_memory,
            requested_mode="prefer_gpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )

    retained_private_memory = deepcopy(gpu_record)
    retained_private_memory["device_execution_telemetry"][0]["observation"][
        "terminal_memory_snapshots"
    ][0]["runtime_reserved_bytes"] = 512
    with pytest.raises(benchmark_script.EvidenceError, match="retained private"):
        benchmark_script._validate_report_evidence(
            retained_private_memory,
            requested_mode="prefer_gpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )

    invalid_out_of_pool = deepcopy(gpu_record)
    del invalid_out_of_pool["device_execution_telemetry"][0]["observation"][
        "terminal_memory_snapshots"
    ][0]["out_of_pool_bytes"]
    with pytest.raises(benchmark_script.EvidenceError, match="out_of_pool_bytes"):
        benchmark_script._validate_report_evidence(
            invalid_out_of_pool,
            requested_mode="prefer_gpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )

    synchronized_record = benchmark_script._interaction_report_record(
        _published_report(
            include_device=True,
            gpu_preparation=True,
            synchronized_device=True,
        ),
        execution_summary=_execution_summary("prefer_gpu"),
    )
    benchmark_script._validate_report_evidence(
        synchronized_record,
        requested_mode="prefer_gpu",
        thumbnail_scope="slice",
        scenario_name="cold_fresh_process_first_edit",
        synchronize_device_phases=True,
    )
    missing_synchronization = deepcopy(synchronized_record)
    sync_observation = missing_synchronization["device_execution_telemetry"][0][
        "observation"
    ]
    sync_observation["spans"] = [
        span
        for span in sync_observation["spans"]
        if span.get("synchronization_point") != "after_device_operation"
    ]
    with pytest.raises(benchmark_script.EvidenceError, match="synchronization proof"):
        benchmark_script._validate_report_evidence(
            missing_synchronization,
            requested_mode="prefer_gpu",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
            synchronize_device_phases=True,
        )


def test_explicit_device_affinity_covers_environment_spans_and_resident_records(
    benchmark_script,
) -> None:
    gpu_record = benchmark_script._interaction_report_record(
        _published_report(
            include_device=True,
            gpu_preparation=True,
            include_resident=True,
        ),
        execution_summary=_execution_summary("prefer_gpu"),
    )
    benchmark_script._validate_report_evidence(
        gpu_record,
        requested_mode="prefer_gpu",
        requested_device_id="cuda:0",
        thumbnail_scope="stack",
        scenario_name="warm_same_process_edits",
    )

    wrong_environment = deepcopy(gpu_record)
    wrong_environment["execution_summary"]["accepted_execution"]["environment"][
        "device_id"
    ] = "cuda:1"
    with pytest.raises(benchmark_script.EvidenceError, match="accepted environment"):
        benchmark_script._validate_report_evidence(
            wrong_environment,
            requested_mode="prefer_gpu",
            requested_device_id="cuda:0",
            thumbnail_scope="stack",
            scenario_name="warm_same_process_edits",
        )

    wrong_span = deepcopy(gpu_record)
    wrong_span["device_execution_telemetry"][0]["observation"]["spans"][0][
        "device_id"
    ] = "cuda:1"
    with pytest.raises(benchmark_script.EvidenceError, match="Device span used"):
        benchmark_script._validate_report_evidence(
            wrong_span,
            requested_mode="prefer_gpu",
            requested_device_id="cuda:0",
            thumbnail_scope="stack",
            scenario_name="warm_same_process_edits",
        )

    wrong_resident = deepcopy(gpu_record)
    wrong_resident["resident_thumbnail_statistics"][0]["device_id"] = "cuda:1"
    with pytest.raises(benchmark_script.EvidenceError, match="Resident thumbnail"):
        benchmark_script._validate_report_evidence(
            wrong_resident,
            requested_mode="prefer_gpu",
            requested_device_id="cuda:0",
            thumbnail_scope="stack",
            scenario_name="warm_same_process_edits",
        )

    cpu_record = benchmark_script._interaction_report_record(
        _published_report(),
        execution_summary=_execution_summary("cpu"),
    )
    cpu_record["device_execution_telemetry"] = deepcopy(
        gpu_record["device_execution_telemetry"]
    )
    with pytest.raises(benchmark_script.EvidenceError, match="CPU interaction"):
        benchmark_script._validate_report_evidence(
            cpu_record,
            requested_mode="cpu",
            requested_device_id="cuda:0",
            thumbnail_scope="slice",
            scenario_name="cold_fresh_process_first_edit",
        )


def test_true_node_start_cancellation_requires_complete_prep_and_marks_contract(
    benchmark_script,
) -> None:
    preparation = _published_report(
        gpu_preparation=True
    ).pre_device_execution_telemetry[0][1]
    report = InteractionLatencyReport(
        generation_id=2,
        node_id="subtract_background_1",
        parameter_names=("radius",),
        started_monotonic_seconds=100.0,
        elapsed_seconds=0.8,
        outcome="superseded_in_flight",
        events=(
            InteractionLatencyEvent(InteractionLatencyPhase.PARAMETER_COMMITTED, 0.0),
            InteractionLatencyEvent(InteractionLatencyPhase.WORKER_STARTED, 0.1),
            InteractionLatencyEvent(InteractionLatencyPhase.PIPELINE_STARTED, 0.2),
            InteractionLatencyEvent(InteractionLatencyPhase.PIPELINE_TERMINAL, 0.7),
            InteractionLatencyEvent(
                InteractionLatencyPhase.PIPELINE_RESULT_DELIVERED,
                0.8,
            ),
        ),
        pipeline_run_ids=(2,),
        pre_device_execution_telemetry=((2, preparation),),
    )
    summary = {
        "pipeline_results": [
            {
                "run_id": 2,
                "cancelled": True,
                "error": "superseded",
                "failure": {
                    "kind": "cancelled",
                    "error_type": "OperationCancelled",
                    "message": "superseded",
                    "cleanup_succeeded": True,
                },
                "cleanup_succeeded": True,
                "device_execution_returned": False,
            }
        ],
        "accepted_execution": None,
        "target_node_started_before_supersession": True,
        "target_node_started_run_id": 2,
        "cancellation_requested_before_worker_gate_release": True,
    }
    record = benchmark_script._interaction_report_record(
        report,
        execution_summary=summary,
    )

    assert record["partial_device_telemetry_unavailable_by_contract"] is True
    benchmark_script._validate_report_evidence(
        record,
        requested_mode="prefer_gpu",
        thumbnail_scope="stack",
        scenario_name="started_in_flight_supersession",
    )
    missing_marker = deepcopy(record)
    missing_marker["partial_device_telemetry_unavailable_by_contract"] = False
    with pytest.raises(benchmark_script.EvidenceError, match="partial-telemetry"):
        benchmark_script._validate_report_evidence(
            missing_marker,
            requested_mode="prefer_gpu",
            thumbnail_scope="stack",
            scenario_name="started_in_flight_supersession",
        )
    incomplete = deepcopy(record)
    incomplete["pre_device_execution_telemetry"][0]["observation"]["completed"] = False
    with pytest.raises(benchmark_script.EvidenceError, match="incomplete"):
        benchmark_script._validate_report_evidence(
            incomplete,
            requested_mode="prefer_gpu",
            thumbnail_scope="stack",
            scenario_name="started_in_flight_supersession",
        )


def test_started_supersession_dispatches_second_commit_without_debounce(
    benchmark_script,
    monkeypatch,
) -> None:
    calls = []

    class Recorder:
        active_generation_id = None
        _generation_serial = 0

        @staticmethod
        def has_phase(_generation, phase):
            return phase is InteractionLatencyPhase.WORKER_STARTED

        @staticmethod
        def generation_for_pipeline_run(run_id):
            return 1 if run_id == 1 else None

    class Widget:
        def __init__(self) -> None:
            self._interaction_latency_recorder = Recorder()
            self._pipeline_cancel_events = {1: threading.Event()}
            self.reports = ()

        def recent_interaction_latency_reports(self):
            return self.reports

        def _on_param_changed(self, _name, value):
            calls.append(("edit", value))
            self._interaction_latency_recorder._generation_serial += 1
            self._interaction_latency_recorder.active_generation_id = (
                self._interaction_latency_recorder._generation_serial
            )

        def run_pipeline(self):
            calls.append(("run_pipeline",))
            self._pipeline_cancel_events[1].set()
            self.reports = (
                SimpleNamespace(
                    generation_id=1,
                    outcome="superseded_in_flight",
                    pipeline_run_ids=(1,),
                ),
                SimpleNamespace(
                    generation_id=2,
                    outcome="published",
                    pipeline_run_ids=(2,),
                ),
            )

    monkeypatch.setattr(
        benchmark_script,
        "_interaction_execution_summary",
        lambda *_args, **kwargs: dict(kwargs),
    )
    widget = Widget()
    gate = benchmark_script.WorkerNodeStartGate(
        threading.Event(),
        threading.Event(),
        run_id=1,
    )
    gate.started.set()

    capture = benchmark_script._drive_in_flight_supersession(
        widget,
        values=(11.0, 13.0),
        timeout_seconds=1.0,
        process_events=lambda: calls.append(("process_events",)),
        observed_results={},
        observed_node_starts={(1, "subtract_background_1")},
        worker_node_start_gate=gate,
    )

    second_edit = calls.index(("edit", 13.0))
    assert calls[second_edit + 1] == ("run_pipeline",)
    assert (
        capture.execution_summaries[0]["target_node_started_before_supersession"]
        is True
    )
    assert capture.execution_summaries[0]["target_node_started_run_id"] == 1
    assert (
        capture.execution_summaries[0][
            "cancellation_requested_before_worker_gate_release"
        ]
        is True
    )
    assert gate.release.is_set()


def test_worker_node_start_gate_blocks_until_release_and_restores(
    benchmark_script,
) -> None:
    from napari_vipp.ui.workers import PipelineRunWorker

    original = PipelineRunWorker._emit_node_started
    emitted = []
    gate, restore = benchmark_script._install_pipeline_worker_node_start_gate(
        target_node_id="subtract_background_1",
        timeout_seconds=1.0,
    )
    worker = SimpleNamespace(
        request=SimpleNamespace(run_id=17),
        signals=SimpleNamespace(
            node_started=SimpleNamespace(emit=lambda payload: emitted.append(payload))
        ),
    )
    thread = threading.Thread(
        target=PipelineRunWorker._emit_node_started,
        args=(worker, "subtract_background_1"),
    )
    try:
        thread.start()
        assert gate.started.wait(0.5)
        assert gate.run_id == 17
        assert emitted == [(17, "subtract_background_1")]
        assert thread.is_alive()
        gate.release.set()
        thread.join(timeout=1.0)
        assert not thread.is_alive()
        assert gate.timed_out is False
    finally:
        restore()
        gate.release.set()
        thread.join(timeout=1.0)
    assert PipelineRunWorker._emit_node_started is original


def test_resident_stack_path_is_correlated_gpu_and_zero_upload(
    benchmark_script,
) -> None:
    record = benchmark_script._interaction_report_record(
        _published_report(
            include_device=True,
            gpu_preparation=True,
            include_resident=True,
        ),
        execution_summary=_execution_summary("prefer_gpu"),
    )
    benchmark_script._validate_report_evidence(
        record,
        requested_mode="prefer_gpu",
        thumbnail_scope="stack",
        scenario_name="warm_same_process_edits",
    )

    uploaded = deepcopy(record)
    uploaded["resident_thumbnail_statistics"][0][
        "logical_input_host_to_device_bytes"
    ] = 4096
    with pytest.raises(benchmark_script.EvidenceError, match="host upload"):
        benchmark_script._validate_report_evidence(
            uploaded,
            requested_mode="prefer_gpu",
            thumbnail_scope="stack",
            scenario_name="warm_same_process_edits",
        )


def test_fresh_process_orchestrator_compares_cpu_and_gpu_scientific_identity(
    benchmark_script,
) -> None:
    normalized = benchmark_script._normalized_inputs(
        **_normalization_kwargs(benchmark_script)
    )
    specs = tuple(
        benchmark_script.WorkerSpec(mode=mode, **normalized)
        for mode in ("cpu", "prefer_gpu")
    )
    launched = []

    def launch(spec):
        launched.append(spec.mode)
        return {
            "schema": benchmark_script.WORKER_SCHEMA,
            "schema_version": benchmark_script.SCHEMA_VERSION,
            "session": _minimal_session(
                spec.mode,
                device_selection=benchmark_script._compute_device_selection(spec),
            ),
        }

    document = benchmark_script.collect_fresh_process_evidence(
        specs,
        launch_worker=launch,
    )

    assert launched == ["cpu", "prefer_gpu"]
    assert document["comparison"]["covered_modes"] == ["cpu", "prefer_gpu"]
    assert document["comparison"]["scientific_output_parity"].startswith("exact")
    assert document["comparison"]["cold_seconds"] == {
        "cpu": 1.0,
        "prefer_gpu": 1.0,
    }


def test_worker_payload_serializes_and_validates_controlled_device_selection(
    benchmark_script,
) -> None:
    normalized = benchmark_script._normalized_inputs(
        **(_normalization_kwargs(benchmark_script) | {"device_id": "cuda:7"})
    )
    spec = benchmark_script.WorkerSpec(mode="prefer_gpu", **normalized)
    selection = benchmark_script._compute_device_selection(spec)
    payload = {
        "schema": benchmark_script.WORKER_SCHEMA,
        "schema_version": benchmark_script.SCHEMA_VERSION,
        "session": _minimal_session(
            "prefer_gpu",
            device_selection=selection,
        ),
    }

    session = benchmark_script._validated_worker_payload(payload, spec)

    assert session["telemetry"]["compute_device_selection"] == {
        "controlled": True,
        "runtime_id": "cuda-cupy",
        "device_id": "cuda:7",
        "display_name": "cuda:7",
        "active_for_mode": True,
        "session_scoped": True,
    }
    mismatched = deepcopy(payload)
    mismatched["session"]["telemetry"]["compute_device_selection"]["device_id"] = (
        "cuda:0"
    )
    with pytest.raises(benchmark_script.EvidenceError, match="wrong session-scoped"):
        benchmark_script._validated_worker_payload(mismatched, spec)


def test_exact_cpu_worker_runs_headless_without_gpu_execution(tmp_path) -> None:
    output = tmp_path / "cpu-worker.json"
    environment = os.environ.copy()
    environment.setdefault("QT_QPA_PLATFORM", "offscreen")
    environment["CUDA_VISIBLE_DEVICES"] = "-1"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--_worker",
            "--output",
            str(output),
            "--mode",
            "cpu",
            "--device-id",
            "cuda:7",
            "--input-profile",
            "exact_sample",
            "--thumbnail-scope",
            "slice",
            "--timeout-seconds",
            "60",
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert completed.returncode == 0, completed.stderr
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schema"] == "napari-vipp-ui-interaction-latency-worker"
    session = document["session"]
    assert session["requested_mode"] == "cpu"
    assert session["summary"]["published_count"] == 6
    assert session["summary"]["pre_device_observation_count"] == 6
    assert session["summary"]["device_observation_count"] == 0
    assert session["telemetry"]["compute_device_selection"] == {
        "active_for_mode": False,
        "controlled": True,
        "device_id": "cuda:7",
        "display_name": "cuda:7",
        "runtime_id": "cuda-cupy",
        "session_scoped": True,
    }
    assert session["telemetry"]["explicit_device_affinity_claimed"] is False
    assert session["telemetry"]["real_multi_device_validation_performed"] is False
    assert session["telemetry"]["thumbnail_worker_backend_identity_attached"] is False
    assert session["telemetry"]["thumbnail_device_affinity"].startswith("CPU mode")
    assert session["summary"]["rapid_superseded_count"] == 2
    assert session["summary"]["started_in_flight_supersession"] == "not_exercised"


def _normalization_kwargs(benchmark_script) -> dict[str, object]:
    return {
        "workflow_path": benchmark_script.DEFAULT_WORKFLOW,
        "cold_radius": 3,
        "warm_radii": (4, 5),
        "revisit_radii": (3, 4),
        "rapid_radii": (5, 7, 9),
        "in_flight_radii": (11, 13),
        "history_limit": 64,
        "timeout_seconds": 60,
        "synchronize_device_phases": False,
        "thumbnail_scope": "stack",
        "input_profile": "exact_sample",
    }


def _minimal_session(
    mode: str,
    *,
    device_selection: dict[str, object] | None = None,
) -> dict[str, object]:
    identity = {
        "dtype": "<u2",
        "shape": [4, 16, 18],
        "nbytes": 2304,
        "sha256_c_order_bytes": "0" * 64,
    }
    report = {
        "outcome": "published",
        "execution_summary": {
            "accepted_execution": {"scientific_output_identity": identity}
        },
    }
    return {
        "requested_mode": mode,
        "fresh_process": True,
        "process_id": 101 if mode == "cpu" else 202,
        "workflow": {"sha256": "workflow-digest"},
        "source_provenance": {"files": [{"sha256": "source-digest"}]},
        "telemetry": {
            "synchronize_device_phases": False,
            "compute_device_selection": (
                device_selection
                if device_selection is not None
                else {
                    "controlled": False,
                    "runtime_id": "",
                    "device_id": "",
                    "display_name": "",
                    "active_for_mode": False,
                    "session_scoped": True,
                }
            ),
        },
        "sample": {"input_identity": identity},
        "scenarios": [
            {
                "name": "cold_fresh_process_first_edit",
                "edited_values": [3.0],
                "reports": [report],
            }
        ],
        "summary": {
            "cold_published_seconds": 1.0,
            "warm_unseen_published_median_seconds": 0.5,
        },
    }


def _published_report(
    *,
    include_device: bool = False,
    gpu_preparation: bool = False,
    include_resident: bool = False,
    synchronized_device: bool = False,
) -> InteractionLatencyReport:
    phases = (
        (InteractionLatencyPhase.PARAMETER_COMMITTED, 0.0),
        (InteractionLatencyPhase.PARAMETER_INVALIDATION_FINISHED, 0.05),
        (InteractionLatencyPhase.DEBOUNCE_STARTED, 0.05),
        (InteractionLatencyPhase.DEBOUNCE_FINISHED, 0.15),
        (InteractionLatencyPhase.WORKER_QUEUED, 0.16),
        (InteractionLatencyPhase.WORKER_STARTED, 0.18),
        (InteractionLatencyPhase.PIPELINE_STARTED, 0.2),
        (InteractionLatencyPhase.PIPELINE_TERMINAL, 0.8),
        (InteractionLatencyPhase.PIPELINE_RESULT_DELIVERED, 0.82),
        (InteractionLatencyPhase.PIPELINE_ACCEPTED, 0.84),
        (InteractionLatencyPhase.THUMBNAIL_RENDER_STARTED, 0.85),
        (InteractionLatencyPhase.THUMBNAIL_RENDER_FINISHED, 0.95),
        (InteractionLatencyPhase.PUBLICATION_ACCEPTED, 1.0),
    )
    preparation_phases = [
        PipelinePreparationPhase.GRAPH_RESTORATION,
        PipelinePreparationPhase.CACHE_PREPARATION,
        PipelinePreparationPhase.WORKLOAD_PREPARATION,
    ]
    if gpu_preparation:
        preparation_phases.extend(
            (
                PipelinePreparationPhase.ACCELERATOR_SETUP,
                PipelinePreparationPhase.RUNTIME_LIBRARY_PROBE,
                PipelinePreparationPhase.COMPUTE_PLANNING,
                PipelinePreparationPhase.DEVICE_PLAN_BUILD,
            )
        )
    spans = tuple(
        PipelinePreparationSpan(phase, index * 0.04, 0.03)
        for index, phase in enumerate(preparation_phases)
    )
    preparation = PipelinePreparationObservation(
        started_monotonic_seconds=100.2,
        elapsed_seconds=max(0.12, len(spans) * 0.04),
        spans=spans,
    )
    device = (
        _device_observation(synchronized=synchronized_device)
        if include_device
        else None
    )
    resident = (
        (
            InteractionResidentThumbnailStatistics(
                pipeline_run_id=1,
                node_id="subtract_background_1",
                output_port=0,
                contrast_mode="percentile_1_99",
                elapsed_seconds=0.05,
                intended_backend="gpu-cupy",
                actual_backend="gpu-cupy",
                algorithm_id="exact-gpu-histogram",
                runtime_id="cuda-cupy",
                device_id="cuda:0",
                input_path="resident_borrow",
                logical_input_host_to_device_bytes=0,
                auxiliary_host_to_device_bytes=1024,
                device_to_host_bytes=16,
            ),
        )
        if include_resident
        else ()
    )
    return InteractionLatencyReport(
        generation_id=1,
        node_id="subtract_background_1",
        parameter_names=("radius",),
        started_monotonic_seconds=100.0,
        elapsed_seconds=1.0,
        outcome="published",
        events=tuple(
            InteractionLatencyEvent(phase, offset) for phase, offset in phases
        ),
        pipeline_run_ids=(1,),
        pre_device_execution_telemetry=((1, preparation),),
        device_execution_telemetry=(() if device is None else ((1, device),)),
        resident_thumbnail_statistics=resident,
    )


def _device_observation(*, synchronized: bool = False) -> DeviceExecutionObservation:
    host_to_device = DeviceExecutionSpan(
        phase=DeviceExecutionPhase.HOST_TO_DEVICE,
        start_offset_seconds=0.0,
        elapsed_seconds=0.04,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        segment_id="segment-1",
        node_id="input_1",
        port=OutputPortKey("input_1", 0),
        byte_count=4096,
        synchronized=(True if synchronized else None),
    )
    operation = DeviceExecutionSpan(
        phase=DeviceExecutionPhase.DEVICE_OPERATION,
        start_offset_seconds=0.05,
        elapsed_seconds=0.25,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        segment_id="segment-1",
        node_id="subtract_background_1",
        operation_id="subtract_background",
        implementation_id="cupy-subtract-background",
        synchronized=(True if synchronized else None),
    )
    device_to_host = DeviceExecutionSpan(
        phase=DeviceExecutionPhase.DEVICE_TO_HOST,
        start_offset_seconds=0.3,
        elapsed_seconds=0.04,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        segment_id="segment-1",
        node_id="subtract_background_1",
        port=OutputPortKey("subtract_background_1", 0),
        byte_count=2304,
        synchronized=(True if synchronized else None),
    )
    segment_synchronization = DeviceExecutionSpan(
        phase=DeviceExecutionPhase.DEVICE_SYNCHRONIZE,
        start_offset_seconds=0.35,
        elapsed_seconds=0.02,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        segment_id="segment-1",
        synchronized=True,
        synchronization_point=DeviceSynchronizationPoint.SEGMENT_COMPLETE,
    )
    terminal_memory = DeviceExecutionSpan(
        phase=DeviceExecutionPhase.TERMINAL_MEMORY_SNAPSHOT,
        start_offset_seconds=0.38,
        elapsed_seconds=0.01,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
    )
    spans = (
        host_to_device,
        operation,
        device_to_host,
        segment_synchronization,
        terminal_memory,
    )
    if synchronized:
        operation_synchronization = DeviceExecutionSpan(
            phase=DeviceExecutionPhase.DEVICE_SYNCHRONIZE,
            start_offset_seconds=0.1,
            elapsed_seconds=0.1,
            runtime_id="cuda-cupy",
            device_id="cuda:0",
            segment_id="segment-1",
            node_id="subtract_background_1",
            operation_id="subtract_background",
            implementation_id="cupy-subtract-background",
            synchronized=True,
            synchronization_point=(DeviceSynchronizationPoint.AFTER_DEVICE_OPERATION),
        )
        spans = (*spans, operation_synchronization)
    return DeviceExecutionObservation(
        started_monotonic_seconds=100.4,
        elapsed_seconds=0.4,
        spans=spans,
        synchronized_device_phases=synchronized,
        terminal_memory_snapshots=(
            DeviceTerminalMemorySnapshot(
                runtime_id="cuda-cupy",
                device_id="cuda:0",
                topology="discrete",
                device_total_bytes=24 * 1024**3,
                device_free_bytes=20 * 1024**3,
                runtime_live_bytes=0,
                runtime_reserved_bytes=0,
                out_of_pool_bytes=1024,
            ),
        ),
    )


def _execution_summary(mode: str) -> dict[str, object]:
    runtime_id = "cpu-numpy" if mode == "cpu" else "cuda-cupy"
    return {
        "pipeline_results": [
            {
                "run_id": 1,
                "cancelled": False,
                "error": "",
                "failure": None,
                "cleanup_succeeded": True,
                "device_execution_returned": mode != "cpu",
            }
        ],
        "accepted_execution": {
            "cleanup_succeeded": True,
            "environment": {"device_id": "" if mode == "cpu" else "cuda:0"},
            "actual_decisions": [
                {
                    "node_id": "subtract_background_1",
                    "runtime_id": runtime_id,
                    "implementation_id": (
                        "cpu-subtract-background"
                        if mode == "cpu"
                        else "cupy-subtract-background"
                    ),
                }
            ],
            "fallback_records": [],
            "warnings": [],
            "scientific_output_identity": {
                "dtype": "<u2",
                "shape": [4, 16, 18],
                "nbytes": 2304,
                "sha256_c_order_bytes": "0" * 64,
            },
        },
        "target_node_started_before_supersession": False,
        "target_node_started_run_id": None,
        "cancellation_requested_before_worker_gate_release": False,
    }
