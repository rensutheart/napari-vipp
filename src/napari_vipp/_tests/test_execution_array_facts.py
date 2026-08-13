from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from unittest.mock import patch

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp._sample_data import make_sample_data
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionPlan,
    FallbackReason,
    NodeExecutionDecision,
    OutputPortKey,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import (
    plan_compute_decisions,
    probe_compute_environment,
)
from napari_vipp.core.compute_policy import (
    PHASE1_CUCIM_BUILD_RECIPE_ID,
    PHASE1_CUCIM_SOURCE_COMMIT,
    PHASE1_CUCIM_SOURCE_TAG,
    PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
    ArrayFactsCache,
    ArrayFactsKey,
    PerformanceEvidence,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryProbeResult,
    RuntimeDevice,
    RuntimeProbeResult,
)
from napari_vipp.core.execution import (
    PipelineRunRequest,
    execute_pipeline_request,
)
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.operations import canny_edges as cpu_canny_edges
from napari_vipp.core.operations import otsu_threshold as cpu_otsu_threshold
from napari_vipp.core.pipeline import (
    MANUAL_RUN_SKIP,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.workflow import load_workflow, serialize_workflow


@dataclass(frozen=True)
class _PlanningResult:
    request: ComputeRequest
    environment: ComputeEnvironment = ComputeEnvironment()
    decisions: tuple = ()
    warnings: tuple[str, ...] = ()

    @property
    def decisions_by_node(self):
        return MappingProxyType({})

    def as_execution_plan(self, *, segments=()):
        return ExecutionPlan(
            self.request.fingerprint,
            self.environment.fingerprint,
            tuple(segments),
            (),
            self.warnings,
        )


class _CapturingPlanner:
    def __init__(self) -> None:
        self.calls = 0
        self.workloads = ()
        self.array_facts = MappingProxyType({})
        self.performance_evidence = MappingProxyType({})
        self.environment = None

    def __call__(self, request, workloads, **kwargs):
        self.calls += 1
        self.workloads = tuple(workloads)
        self.array_facts = MappingProxyType(dict(kwargs.get("array_facts", {})))
        self.performance_evidence = MappingProxyType(
            dict(kwargs.get("performance_evidence", {}))
        )
        self.environment = kwargs.get("environment")
        return _PlanningResult(request)


class _CancelAfterChecks:
    def __init__(self, checks: int) -> None:
        self.remaining = checks

    def is_set(self) -> bool:
        self.remaining -= 1
        return self.remaining <= 0


class _ProbeRegistry(ComputeRegistry):
    """Provider-free registry whose probes are explicit test data."""

    def __init__(
        self,
        *,
        runtime_available: bool = True,
        library_available: bool = True,
        device_name: str = "NVIDIA GeForce RTX 5090",
        compute_capability: str = "12.0",
    ) -> None:
        super().__init__()
        self.runtime_available = runtime_available
        self.library_available = library_available
        self.device_name = device_name
        self.compute_capability = compute_capability
        self.runtime_probe_count = 0
        self.library_probe_count = 0
        self.library_probe_ids: list[str] = []

    def probe_runtime(self, runtime_id, *, refresh=False):
        del refresh
        self.runtime_probe_count += 1
        devices = (
            (
                RuntimeDevice(
                    "cuda:0",
                    self.device_name,
                    8 * 1024**3,
                    metadata=(("compute_capability", self.compute_capability),),
                ),
            )
            if self.runtime_available
            else ()
        )
        return RuntimeProbeResult(
            runtime_id,
            self.runtime_available,
            version="14.1.1" if self.runtime_available else "",
            devices=devices,
            selected_device_id="cuda:0" if self.runtime_available else "",
            reason_code="" if self.runtime_available else "runtime_unavailable",
            message="" if self.runtime_available else "CUDA is unavailable.",
            environment_fingerprint=(
                "test-cuda-environment" if self.runtime_available else ""
            ),
            metadata=(
                (
                    ("cuda_runtime_version", "13020"),
                    ("driver_version", "13030"),
                )
                if self.runtime_available
                else ()
            ),
        )

    def probe_library(self, library_id, *, refresh=False):
        del refresh
        self.library_probe_count += 1
        self.library_probe_ids.append(library_id)
        return ImplementationLibraryProbeResult(
            library_id,
            self.library_available,
            version=(
                "26.6.0"
                if self.library_available and library_id == "cucim"
                else "14.1.1"
                if self.library_available
                else ""
            ),
            reason_code="" if self.library_available else "library_unavailable",
            message=("" if self.library_available else f"{library_id} is unavailable."),
            metadata=(
                (
                    ("environment_record_schema", "napari-vipp-gpu-environment"),
                    ("environment_record_schema_version", "2"),
                    ("environment_track", "cuda13"),
                    ("cupy_distribution", "cupy-cuda13x"),
                    ("cucim_distribution", "cucim-cu13"),
                    ("cucim_distribution_version", "26.6.0"),
                    ("cucim_artifact_sha256", "a" * 64),
                    (
                        "cucim_wheel_payload_sha256",
                        PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
                    ),
                    ("cucim_source_tag", PHASE1_CUCIM_SOURCE_TAG),
                    ("cucim_source_commit", PHASE1_CUCIM_SOURCE_COMMIT),
                    ("cucim_build_recipe_id", PHASE1_CUCIM_BUILD_RECIPE_ID),
                )
                if self.library_available and library_id == "cucim"
                else ()
            ),
        )


_DEFAULT_EVIDENCE = object()


def _viable_evidence_for_pipeline(
    pipeline: PrototypePipeline,
) -> dict[tuple[str, str], PerformanceEvidence]:
    evidence: dict[tuple[str, str], PerformanceEvidence] = {}
    with ComputeRegistry() as registry:
        for node_id, node in pipeline.nodes.items():
            for spec in registry.implementations_for_operation(
                node.operation_id,
                allow_experimental=True,
            ):
                evidence[(node_id, spec.implementation_id)] = PerformanceEvidence(
                    cpu_seconds=1.0,
                    candidate_seconds=0.20,
                    lower_confidence_speedup=2.0,
                )
    return evidence


def _accelerated_request(
    pipeline: PrototypePipeline,
    data: np.ndarray,
    *,
    run_id: int = 1,
    revision: object | None = "revision-1",
    cache: ArrayFactsCache | None = None,
    cancel_event=None,
    source_payloads: dict[str, SourcePayload] | None = None,
    compute_request: ComputeRequest | None = None,
    performance_evidence=_DEFAULT_EVIDENCE,
) -> PipelineRunRequest:
    payloads = (
        {
            "input": SourcePayload(
                data,
                {"axes": "YX"},
                "source",
                revision_token=revision,
            )
        }
        if source_payloads is None
        else dict(source_payloads)
    )
    revisions = tuple(
        payload.revision_token
        for payload in payloads.values()
        if payload.revision_token is not None
    )
    selected_evidence = (
        _viable_evidence_for_pipeline(pipeline)
        if performance_evidence is _DEFAULT_EVIDENCE
        else performance_evidence
    )
    return PipelineRunRequest(
        run_id=run_id,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata={"axes": "YX"},
        input_name="source",
        source_payloads=payloads,
        compute_request=(
            ComputeRequest(
                mode=ComputeMode.AUTO,
                runtime_id="cuda-cupy",
                allow_experimental=True,
            )
            if compute_request is None
            else compute_request
        ),
        cancel_event=cancel_event,
        source_revisions=revisions,
        array_facts_cache=cache,
        performance_evidence=selected_evidence,
    )


def _execute_accelerated(request, planner, *, registry=None):
    selected_registry = registry or _ProbeRegistry()
    validated_host = ComputeEnvironment(
        os_name="Windows",
        os_release="test",
        execution_mode="native",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
    )
    with (
        selected_registry,
        patch(
            "napari_vipp.core.compute_planning.ComputeEnvironment",
            return_value=validated_host,
        ),
    ):
        return execute_pipeline_request(
            request,
            compute_registry=selected_registry,
            compute_planner=planner,
        )


def _cached_cpu_provenance(
    pipeline: PrototypePipeline,
    run_request: PipelineRunRequest,
):
    decisions = []
    for node_id in pipeline.completed_node_ids:
        node = pipeline.nodes[node_id]
        if not node.has_input or not pipeline._has_cached_output(node_id):
            continue
        decision = NodeExecutionDecision(
            node_id=node_id,
            operation_id=node.operation_id,
            requested_preference=run_request.compute_request.preference_for(node_id),
            runtime_id="cpu-numpy",
            implementation_library_id="cpu",
            implementation_id=f"cpu-{node.operation_id}-v1",
            decision_kind=DecisionKind.POLICY_CPU,
            reason=DecisionReason.AUTO_CPU,
            reason_text="Test fixture produced this cached result on CPU.",
        )
        decisions.append(decision)
    source_contexts = execution_module._capture_source_scientific_contexts(
        pipeline,
        run_request,
        cancel_callback=None,
    )
    execution_module._publish_actual_compute_provenance(
        pipeline,
        run_request.compute_request,
        decisions,
        source_scientific_contexts=source_contexts,
        cancel_callback=None,
    )
    return dict(pipeline.node_compute_provenance)


def _scan_spy(monkeypatch):
    calls = []
    original = execution_module._complete_array_facts

    def scan(value, **kwargs):
        calls.append(np.asarray(value))
        return original(value, **kwargs)

    monkeypatch.setattr(execution_module, "_complete_array_facts", scan)
    return calls


def test_cpu_mode_never_scans_array_facts(monkeypatch):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(25, dtype=np.float32).reshape(5, 5)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("CPU execution must not scan compute-policy facts.")

    monkeypatch.setattr(execution_module, "_complete_array_facts", forbidden)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata={"axes": "YX"},
            input_name="source",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.CPU),
            array_facts_cache=ArrayFactsCache(),
        )
    )

    assert result.error == ""
    assert result.pipeline is not None


def _float32_gaussian_pipeline() -> tuple[PrototypePipeline, str, np.ndarray]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 1.0)
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    return pipeline, gaussian.id, data


def test_auto_reviewed_default_scans_required_scientific_facts(monkeypatch):
    calls = _scan_spy(monkeypatch)
    pipeline, gaussian_id, data = _float32_gaussian_pipeline()
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            performance_evidence={},
        ),
        planner,
    )

    assert result.error == ""
    assert len(calls) == 1
    assert calls[0] is data
    assert gaussian_id in planner.array_facts
    assert planner.performance_evidence == {}


def test_auto_below_threshold_performance_evidence_does_not_scan(monkeypatch):
    calls = _scan_spy(monkeypatch)
    pipeline, gaussian_id, data = _float32_gaussian_pipeline()
    implementation_key = next(iter(_viable_evidence_for_pipeline(pipeline)))
    evidence = {
        implementation_key: PerformanceEvidence(
            cpu_seconds=1.0,
            candidate_seconds=0.99,
            lower_confidence_speedup=1.01,
        )
    }
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            performance_evidence=evidence,
        ),
        planner,
    )

    assert result.error == ""
    assert calls == []
    assert gaussian_id not in planner.array_facts
    assert planner.performance_evidence == evidence


def test_auto_viable_performance_evidence_scans_and_reaches_planner(monkeypatch):
    calls = _scan_spy(monkeypatch)
    pipeline, gaussian_id, data = _float32_gaussian_pipeline()
    evidence = _viable_evidence_for_pipeline(pipeline)
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            performance_evidence=evidence,
        ),
        planner,
    )

    assert result.error == ""
    assert len(calls) == 1
    assert planner.array_facts[gaussian_id][0].all_finite is True
    assert planner.performance_evidence == evidence


def test_forced_custom_candidate_scans_without_performance_evidence(monkeypatch):
    calls = _scan_spy(monkeypatch)
    pipeline, gaussian_id, data = _float32_gaussian_pipeline()
    implementation_key = next(iter(_viable_evidence_for_pipeline(pipeline)))
    compute_request = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={
            gaussian_id: f"implementation:{implementation_key[1]}",
        },
        runtime_id="cuda-cupy",
        allow_experimental=True,
    )
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            compute_request=compute_request,
            performance_evidence={},
        ),
        planner,
    )

    assert result.error == ""
    assert len(calls) == 1
    assert planner.array_facts[gaussian_id][0].all_finite is True
    assert planner.performance_evidence == {}


def test_prefer_gpu_scans_without_evidence_and_ignores_dormant_cpu_preference(
    monkeypatch,
):
    calls = _scan_spy(monkeypatch)
    pipeline, gaussian_id, data = _float32_gaussian_pipeline()
    planner = _CapturingPlanner()
    compute_request = ComputeRequest(
        mode=ComputeMode.PREFER_GPU,
        node_preferences={gaussian_id: "cpu"},
        runtime_id="cuda-cupy",
    )

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            compute_request=compute_request,
            performance_evidence={},
        ),
        planner,
    )

    assert result.error == ""
    assert len(calls) == 1
    assert planner.array_facts[gaussian_id][0].all_finite is True
    assert planner.performance_evidence == {}
    assert planner.calls == 1


def test_pipeline_request_copies_and_freezes_performance_evidence():
    pipeline, _gaussian_id, data = _float32_gaussian_pipeline()
    evidence = _viable_evidence_for_pipeline(pipeline)
    expected = dict(evidence)

    request = _accelerated_request(
        pipeline,
        data,
        performance_evidence=evidence,
    )
    evidence.clear()

    assert request.performance_evidence == expected
    with pytest.raises(TypeError):
        request.performance_evidence[next(iter(expected))] = next(
            iter(expected.values())
        )


@pytest.mark.parametrize(
    ("operation_id", "dtype", "parameter", "value"),
    (
        ("gaussian_blur", np.uint8, "sigma", 1.0),
        ("gaussian_blur", np.float64, "sigma", 1.0),
        ("median_filter", np.uint16, "size", 3),
        ("rolling_ball_background", np.float32, "radius", 2.0),
        ("canny_edges", np.uint16, "sigma", 1.0),
        ("canny_edges", np.float32, "sigma", 1.0),
        ("otsu_threshold", np.bool_, "histogram_bins", 256),
        ("otsu_threshold", np.int8, "histogram_bins", 256),
        ("otsu_threshold", np.uint8, "histogram_bins", 256),
        ("otsu_threshold", np.int16, "histogram_bins", 256),
        ("otsu_threshold", np.uint16, "histogram_bins", 256),
        ("otsu_threshold", np.float32, "histogram_bins", 256),
    ),
)
def test_unsupported_or_non_fact_gated_candidates_do_not_scan(
    monkeypatch,
    operation_id,
    dtype,
    parameter,
    value,
):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    pipeline.set_param(node.id, parameter, value)
    assert pipeline.connect("input", node.id).success
    data = np.arange(81).reshape(9, 9).astype(dtype)
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(pipeline, data),
        planner,
    )

    assert result.error == ""
    assert calls == []
    assert planner.array_facts == {}


@pytest.mark.parametrize(
    ("runtime_available", "library_available"),
    ((False, True), (True, False)),
)
def test_unavailable_runtime_or_library_preflight_skips_fact_scan(
    monkeypatch,
    runtime_available,
    library_available,
):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 1.0)
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(pipeline, data),
        planner,
        registry=_ProbeRegistry(
            runtime_available=runtime_available,
            library_available=library_available,
        ),
    )

    assert result.error == ""
    assert calls == []
    assert planner.array_facts == {}
    assert planner.environment is not None
    if not runtime_available:
        assert "cuda-cupy" not in planner.environment.runtime_ids
    if not library_available:
        assert "cupyx" not in planner.environment.implementation_libraries


@pytest.mark.parametrize(
    ("dtype", "sigma", "expected_probes"),
    ((np.uint8, 1.0, 1), (np.float32, 99.0, 0)),
)
def test_static_preflight_probes_repairable_dtype_but_not_invalid_parameters(
    monkeypatch,
    dtype,
    sigma,
    expected_probes,
):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", sigma)
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(81).reshape(9, 9).astype(dtype)
    registry = _ProbeRegistry()
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(pipeline, data),
        planner,
        registry=registry,
    )

    assert result.error == ""
    assert calls == []
    assert registry.runtime_probe_count == expected_probes
    assert registry.library_probe_count == expected_probes


@pytest.mark.parametrize(
    ("operation_id", "parameter", "value"),
    (
        ("gaussian_blur", "sigma", 1.0),
        ("median_filter", "size", 3),
    ),
)
def test_float32_finite_only_candidates_scan_once(
    monkeypatch,
    operation_id,
    parameter,
    value,
):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    pipeline.set_param(node.id, parameter, value)
    assert pipeline.connect("input", node.id).success
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(pipeline, data),
        planner,
    )

    assert result.error == ""
    assert len(calls) == 1
    facts = planner.array_facts[node.id][0]
    assert facts.all_finite is True
    assert facts.minimum == 0.0
    assert facts.maximum == 80.0


@pytest.mark.parametrize(
    ("mode", "preference"),
    (
        (ComputeMode.AUTO, None),
        (ComputeMode.PREFER_GPU, None),
        (ComputeMode.CUSTOM, "auto"),
        (ComputeMode.CUSTOM, "library:cupyx"),
    ),
)
def test_compatible_device_candidate_scans_required_facts_once(
    monkeypatch,
    mode,
    preference,
):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 1.0)
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    planner = _CapturingPlanner()
    registry = _ProbeRegistry(
        device_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        compute_capability="8.9",
    )
    preferences = {} if preference is None else {gaussian.id: preference}

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            compute_request=ComputeRequest(
                mode=mode,
                node_preferences=preferences,
            ),
        ),
        planner,
        registry=registry,
    )

    assert result.error == ""
    assert len(calls) == 1
    assert planner.array_facts[gaussian.id][0].all_finite is True


@pytest.mark.parametrize("dtype", (np.int32, np.uint32, np.int64, np.uint64))
def test_wide_integer_otsu_scans_once_for_exact_native_span(monkeypatch, dtype):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    otsu = pipeline.add_node("otsu_threshold")
    assert pipeline.connect("input", otsu.id).success
    data = np.arange(81).reshape(9, 9).astype(dtype)
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(pipeline, data),
        planner,
    )

    assert result.error == ""
    assert len(calls) == 1
    facts = planner.array_facts[otsu.id][0]
    assert facts.minimum == 0
    assert facts.maximum == 80


def test_float32_canny_executes_visible_cpu_fallback_without_fact_scan(monkeypatch):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    canny = pipeline.add_node("canny_edges")
    assert pipeline.connect("input", canny.id).success
    data = np.arange(81, dtype=np.float32).reshape(9, 9)

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            compute_request=ComputeRequest(
                mode=ComputeMode.CUSTOM,
                node_preferences={
                    canny.id: "implementation:cupyx-canny-edges-exact-v1"
                },
                runtime_id="cuda-cupy",
            ),
        ),
        plan_compute_decisions,
    )

    assert result.error == ""
    assert calls == []
    assert result.pipeline is not None
    np.testing.assert_array_equal(
        result.pipeline.outputs[canny.id],
        cpu_canny_edges(data),
    )
    assert result.execution_report is not None
    decision = next(
        item
        for item in result.execution_report.actual_decisions
        if item.node_id == canny.id
    )
    assert decision.decision_kind is DecisionKind.FALLBACK_CPU
    assert decision.reason is DecisionReason.VISIBLE_FALLBACK
    assert decision.fallback_reason is FallbackReason.WORKLOAD_UNSUPPORTED
    assert "subnormal" in decision.reason_text
    assert result.execution_report.warnings == (decision.reason_text,)


def test_integer_otsu_wide_stack_slice_scope_executes_visible_cpu_fallback():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    otsu = pipeline.add_node("otsu_threshold")
    pipeline.set_param(otsu.id, "threshold_scope", "Slice histogram")
    assert pipeline.connect("input", otsu.id).success
    data = np.zeros((2, 9, 11), dtype=np.uint32)
    data[1] = 1_000_000

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            source_payloads={
                "input": SourcePayload(
                    data,
                    {"axes": "ZYX"},
                    "source",
                    revision_token="wide-slice-revision",
                )
            },
            compute_request=ComputeRequest(
                mode=ComputeMode.CUSTOM,
                node_preferences={
                    otsu.id: (
                        "implementation:cupy-otsu-threshold-exact-v1"
                    )
                },
                runtime_id="cuda-cupy",
            ),
        ),
        plan_compute_decisions,
    )

    assert result.error == ""
    assert result.pipeline is not None
    np.testing.assert_array_equal(
        result.pipeline.outputs[otsu.id],
        cpu_otsu_threshold(data, threshold_scope="Slice histogram"),
    )
    assert result.execution_report is not None
    decision = next(
        item
        for item in result.execution_report.actual_decisions
        if item.node_id == otsu.id
    )
    assert decision.decision_kind is DecisionKind.FALLBACK_CPU
    assert decision.reason is DecisionReason.VISIBLE_FALLBACK
    assert decision.fallback_reason is FallbackReason.WORKLOAD_UNSUPPORTED
    assert "cannot be proved per plane" in decision.reason_text
    assert result.execution_report.warnings == (decision.reason_text,)


def test_background_to_gaussian_scans_source_once_and_propagates(monkeypatch):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    background = pipeline.add_node("subtract_background")
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(background.id, "radius", 2.0)
    pipeline.set_param(gaussian.id, "sigma", 1.0)
    assert pipeline.connect("input", background.id).success
    assert pipeline.connect(background.id, gaussian.id).success
    data = np.linspace(0, 100, 121, dtype=np.float32).reshape(11, 11)
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(pipeline, data),
        planner,
    )

    assert result.error == ""
    assert len(calls) == 1
    source_facts = planner.array_facts[background.id][0]
    propagated = planner.array_facts[gaussian.id][0]
    assert source_facts.all_finite is True
    assert propagated.all_finite is True
    assert ">subtract_background:" in propagated.revision_fingerprint
    assert propagated.scan_seconds == source_facts.scan_seconds


def test_accelerated_workload_preparation_forwards_costes_cancellation(
    monkeypatch,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second_source = pipeline.add_node("input")
    overlay = pipeline.add_node("colocalized_voxels")
    pipeline.set_param(overlay.id, "threshold_mode", "Costes auto")
    assert pipeline.connect("input", overlay.id, target_port=0).success
    assert pipeline.connect(second_source.id, overlay.id, target_port=1).success
    channel_1 = np.arange(64, dtype=np.uint16).reshape(8, 8)
    channel_2 = np.roll(channel_1, 2, axis=0)
    cancelled = threading.Event()
    entered_threshold_search = []
    continued_after_cancel = []

    def cancelling_costes_thresholds(channel_1, channel_2, *, progress=None):
        del channel_1, channel_2
        entered_threshold_search.append(True)
        assert progress is not None
        cancelled.set()
        progress.check_cancelled()
        continued_after_cancel.append(True)
        raise AssertionError("Costes preparation continued after cancellation.")

    monkeypatch.setattr(
        "napari_vipp.core.operations._costes_thresholds",
        cancelling_costes_thresholds,
    )
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            channel_1,
            cancel_event=cancelled,
            source_payloads={
                "input": SourcePayload(
                    channel_1,
                    {"axes": "YX"},
                    "channel 1",
                    revision_token="channel-1-revision",
                ),
                second_source.id: SourcePayload(
                    channel_2,
                    {"axes": "YX"},
                    "channel 2",
                    revision_token="channel-2-revision",
                ),
            },
        ),
        planner,
    )

    assert result.cancelled
    assert entered_threshold_search == [True]
    assert continued_after_cancel == []
    assert planner.calls == 0


def test_accelerated_workload_omits_runtime_costes_diagnostics(monkeypatch):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second_source = pipeline.add_node("input")
    overlay = pipeline.add_node("colocalized_voxels")
    pipeline.set_param(overlay.id, "threshold_mode", "Costes auto")
    assert pipeline.connect("input", overlay.id, target_port=0).success
    assert pipeline.connect(second_source.id, overlay.id, target_port=1).success
    channel_1 = np.arange(64, dtype=np.uint16).reshape(8, 8)
    channel_2 = np.roll(channel_1, 2, axis=0)

    monkeypatch.setattr(
        "napari_vipp.core.operations._costes_thresholds",
        lambda *_args, **_kwargs: {
            "threshold_1": 12.0,
            "threshold_2": 13.0,
            "pearson_below": float("nan"),
        },
    )
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            channel_1,
            source_payloads={
                "input": SourcePayload(channel_1, {"axes": "YX"}, "channel 1"),
                second_source.id: SourcePayload(
                    channel_2,
                    {"axes": "YX"},
                    "channel 2",
                ),
            },
        ),
        planner,
    )

    assert result.error == ""
    assert planner.calls == 1
    workload = next(item for item in planner.workloads if item.node_id == overlay.id)
    assert "_vipp_resolved_costes" not in dict(workload.parameters)


def test_cancellation_interrupts_chunked_scan_without_cache_publication(
    monkeypatch,
):
    calls = _scan_spy(monkeypatch)
    monkeypatch.setattr(execution_module, "_FACT_SCAN_CHUNK_VALUES", 4)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(400, dtype=np.float32).reshape(20, 20)
    cache = ArrayFactsCache()
    cancelled_planner = _CapturingPlanner()

    cancelled = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            cache=cache,
            # Allow exact source-context hashing to finish, then cancel during
            # the deliberately tiny fact-scan chunks below.
            cancel_event=_CancelAfterChecks(90),
        ),
        cancelled_planner,
    )

    assert cancelled.cancelled
    assert cancelled_planner.calls == 0
    assert len(calls) == 1
    coordinator = execution_module._array_facts_cache_coordinator(cache)
    with coordinator.lock:
        assert coordinator.in_flight == {}

    completed_planner = _CapturingPlanner()
    completed = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data,
            run_id=2,
            cache=cache,
        ),
        completed_planner,
    )

    assert completed.error == ""
    assert len(calls) == 2
    assert completed_planner.array_facts[gaussian.id][0].all_finite is True


def test_revision_cache_hit_is_zero_cost_and_changed_revision_rescans(
    monkeypatch,
):
    calls = _scan_spy(monkeypatch)
    clock = iter((10.0, 12.5, 20.0, 21.25))
    monkeypatch.setattr(execution_module, "perf_counter", lambda: next(clock))
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    cache = ArrayFactsCache()

    first_planner = _CapturingPlanner()
    first = _execute_accelerated(
        _accelerated_request(pipeline, data, cache=cache),
        first_planner,
    )
    first_facts = first_planner.array_facts[gaussian.id][0]

    second_planner = _CapturingPlanner()
    second = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data.copy(),
            run_id=2,
            cache=cache,
        ),
        second_planner,
    )
    second_facts = second_planner.array_facts[gaussian.id][0]

    third_planner = _CapturingPlanner()
    third = _execute_accelerated(
        _accelerated_request(
            pipeline,
            data.copy(),
            run_id=3,
            revision="revision-2",
            cache=cache,
        ),
        third_planner,
    )
    third_facts = third_planner.array_facts[gaussian.id][0]

    assert first.error == second.error == third.error == ""
    assert len(calls) == 2
    assert first_facts.scan_seconds == 2.5
    assert second_facts.scan_seconds == 0.0
    assert third_facts.scan_seconds == 1.25
    assert second_facts.revision_fingerprint == first_facts.revision_fingerprint
    assert third_facts.revision_fingerprint != first_facts.revision_fingerprint
    old_key = ArrayFactsKey(
        OutputPortKey("input", 0),
        first_facts.revision_fingerprint,
    )
    assert cache.get(old_key) is None


def test_unrelated_branch_does_not_trigger_an_extra_scan(monkeypatch):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second_source = pipeline.add_node("input")
    gaussian = pipeline.add_node("gaussian_blur")
    background = pipeline.add_node("rolling_ball_background")
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(second_source.id, background.id).success
    first = np.arange(81, dtype=np.float32).reshape(9, 9)
    second = np.arange(121, dtype=np.float32).reshape(11, 11)
    payloads = {
        "input": SourcePayload(first, {"axes": "YX"}, revision_token="first-1"),
        second_source.id: SourcePayload(
            second,
            {"axes": "YX"},
            revision_token="second-1",
        ),
    }
    planner = _CapturingPlanner()

    result = _execute_accelerated(
        _accelerated_request(
            pipeline,
            first,
            source_payloads=payloads,
        ),
        planner,
    )

    assert result.error == ""
    assert len(calls) == 1
    assert calls[0] is first
    assert gaussian.id in planner.array_facts
    assert background.id not in planner.array_facts


def test_untokened_direct_source_never_reuses_an_unrelated_revision(monkeypatch):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    unrelated_source = pipeline.add_node("input")
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 1.0)
    assert pipeline.connect("input", gaussian.id).success
    first = np.arange(81, dtype=np.float32).reshape(9, 9)
    changed = np.full((9, 9), np.nan, dtype=np.float32)
    unrelated = np.ones((3, 3), dtype=np.float32)
    cache = ArrayFactsCache()

    def run(data, run_id):
        payloads = {
            "input": SourcePayload(data, revision_token=None),
            unrelated_source.id: SourcePayload(
                unrelated,
                revision_token="unrelated-revision-1",
            ),
        }
        planner = _CapturingPlanner()
        result = _execute_accelerated(
            _accelerated_request(
                pipeline,
                data,
                run_id=run_id,
                cache=cache,
                source_payloads=payloads,
            ),
            planner,
        )
        return result, planner

    first_result, first_planner = run(first, 1)
    changed_result, changed_planner = run(changed, 2)

    assert first_result.error == changed_result.error == ""
    assert len(calls) == 2
    first_facts = first_planner.array_facts[gaussian.id][0]
    changed_facts = changed_planner.array_facts[gaussian.id][0]
    assert first_facts.all_finite is True
    assert changed_facts.all_finite is False
    assert first_facts.revision_fingerprint != changed_facts.revision_fingerprint


def test_untokened_cached_boundary_never_reuses_an_unrelated_revision(
    monkeypatch,
):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    unrelated_source = pipeline.add_node("input")
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    pipeline.set_param(median.id, "size", 3)
    assert pipeline.connect("input", gaussian.id).success
    assert pipeline.connect(gaussian.id, median.id).success
    first = np.arange(81, dtype=np.float32).reshape(9, 9)
    changed = np.full((9, 9), np.nan, dtype=np.float32)
    unrelated = np.ones((3, 3), dtype=np.float32)
    cache = ArrayFactsCache()

    def payloads_for(data):
        return {
            "input": SourcePayload(data, revision_token=None),
            unrelated_source.id: SourcePayload(
                unrelated,
                revision_token="unrelated-revision-1",
            ),
        }

    def cached_request(data, run_id, payloads):
        base = _accelerated_request(
            pipeline,
            data,
            run_id=run_id,
            cache=cache,
            source_payloads=payloads,
        )
        return replace(
            base,
            dirty_node_ids=frozenset({median.id}),
            cached_outputs=dict(pipeline.outputs),
            cached_output_states=dict(pipeline.output_states),
            cached_node_outputs={
                node_id: list(outputs)
                for node_id, outputs in pipeline.node_outputs.items()
            },
            cached_node_output_states={
                node_id: list(states)
                for node_id, states in pipeline.node_output_states.items()
            },
            cached_execution_states=dict(pipeline.node_execution_states),
                cached_execution_messages=dict(pipeline.node_execution_messages),
                cached_compute_provenance=_cached_cpu_provenance(
                    pipeline,
                    base,
                ),
                completed_node_ids=frozenset(pipeline.completed_node_ids),
            )

    first_payloads = payloads_for(first)
    pipeline.run(
        first,
        input_metadata={"axes": "YX"},
        source_payloads=first_payloads,
    )
    first_request = cached_request(first, 1, first_payloads)

    changed_payloads = payloads_for(changed)
    pipeline.run(
        changed,
        input_metadata={"axes": "YX"},
        source_payloads=changed_payloads,
        dirty_node_ids=frozenset(pipeline.nodes),
    )
    changed_request = cached_request(changed, 2, changed_payloads)

    first_planner = _CapturingPlanner()
    first_result = _execute_accelerated(first_request, first_planner)
    changed_planner = _CapturingPlanner()
    changed_result = _execute_accelerated(changed_request, changed_planner)

    assert first_result.error == changed_result.error == ""
    assert len(calls) == 2
    first_facts = first_planner.array_facts[median.id][0]
    changed_facts = changed_planner.array_facts[median.id][0]
    assert first_facts.all_finite is True
    assert changed_facts.all_finite is False
    assert first_facts.revision_fingerprint != changed_facts.revision_fingerprint


def test_dirty_run_scans_cached_boundary_and_keys_it_by_scientific_graph(
    monkeypatch,
):
    calls = _scan_spy(monkeypatch)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    background = pipeline.add_node("subtract_background")
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(background.id, "radius", 2.0)
    pipeline.set_param(gaussian.id, "sigma", 1.0)
    assert pipeline.connect("input", background.id).success
    assert pipeline.connect(background.id, gaussian.id).success
    data = np.linspace(0, 100, 121, dtype=np.float32).reshape(11, 11)
    pipeline.run(data, input_metadata={"axes": "YX"})
    cached_background = pipeline.outputs[background.id]
    cache = ArrayFactsCache()

    def dirty_request(run_id: int, workflow=None):
        request = _accelerated_request(
            pipeline,
            data,
            run_id=run_id,
            cache=cache,
        )
        return replace(
            request,
            workflow=request.workflow if workflow is None else workflow,
            dirty_node_ids=frozenset({gaussian.id}),
            cached_outputs=dict(pipeline.outputs),
            cached_output_states=dict(pipeline.output_states),
            cached_node_outputs={
                node_id: list(outputs)
                for node_id, outputs in pipeline.node_outputs.items()
            },
            cached_node_output_states={
                node_id: list(states)
                for node_id, states in pipeline.node_output_states.items()
            },
            cached_execution_states=dict(pipeline.node_execution_states),
                cached_execution_messages=dict(pipeline.node_execution_messages),
                cached_compute_provenance=_cached_cpu_provenance(
                    pipeline,
                    request,
                ),
                completed_node_ids=frozenset(pipeline.completed_node_ids),
            )

    first_planner = _CapturingPlanner()
    first = _execute_accelerated(dirty_request(1), first_planner)
    second_planner = _CapturingPlanner()
    second = _execute_accelerated(dirty_request(2), second_planner)

    pipeline.set_param(background.id, "radius", 3.0)
    changed_workflow = serialize_workflow(pipeline)
    third_planner = _CapturingPlanner()
    third = _execute_accelerated(
        dirty_request(3, changed_workflow),
        third_planner,
    )

    assert first.error == second.error == third.error == ""
    assert len(calls) == 2
    assert calls[0] is cached_background
    first_facts = first_planner.array_facts[gaussian.id][0]
    second_facts = second_planner.array_facts[gaussian.id][0]
    third_facts = third_planner.array_facts[gaussian.id][0]
    assert second_facts.scan_seconds == 0.0
    assert second_facts.revision_fingerprint == first_facts.revision_fingerprint
    assert third_facts.revision_fingerprint != first_facts.revision_fingerprint


def test_shared_cache_serializes_one_atomic_fill(monkeypatch):
    original = execution_module._complete_array_facts
    scan_started = threading.Event()
    release_scan = threading.Event()
    counter_lock = threading.Lock()
    scan_count = 0

    def blocking_scan(value, **kwargs):
        nonlocal scan_count
        with counter_lock:
            scan_count += 1
        scan_started.set()
        assert release_scan.wait(timeout=10)
        return original(value, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_complete_array_facts",
        blocking_scan,
    )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    assert pipeline.connect("input", gaussian.id).success
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    cache = ArrayFactsCache()
    planners = (_CapturingPlanner(), _CapturingPlanner())
    requests = (
        _accelerated_request(pipeline, data, run_id=1, cache=cache),
        _accelerated_request(pipeline, data.copy(), run_id=2, cache=cache),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(_execute_accelerated, requests[0], planners[0])
        assert scan_started.wait(timeout=10)
        second = executor.submit(_execute_accelerated, requests[1], planners[1])
        release_scan.set()
        results = (first.result(timeout=30), second.result(timeout=30))

    assert all(result.error == "" for result in results)
    assert scan_count == 1
    costs = sorted(
        planner.array_facts[gaussian.id][0].scan_seconds for planner in planners
    )
    assert costs[0] == 0.0
    assert costs[1] >= 0.0
    coordinator = execution_module._array_facts_cache_coordinator(cache)
    with coordinator.lock:
        assert coordinator.in_flight == {}


def test_waiting_same_key_cancels_promptly_without_interrupting_owner(monkeypatch):
    original = execution_module._complete_array_facts
    scan_started = threading.Event()
    release_scan = threading.Event()
    waiter_polled = threading.Event()
    cancel_waiter = threading.Event()
    scan_count = 0
    counter_lock = threading.Lock()

    def blocking_scan(value, **kwargs):
        nonlocal scan_count
        with counter_lock:
            scan_count += 1
        scan_started.set()
        assert release_scan.wait(timeout=10)
        return original(value, **kwargs)

    def waiter_cancelled():
        waiter_polled.set()
        return cancel_waiter.is_set()

    monkeypatch.setattr(
        execution_module,
        "_complete_array_facts",
        blocking_scan,
    )
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    cache = ArrayFactsCache()
    key = ArrayFactsKey(OutputPortKey("input", 0), "shared-revision")

    with ThreadPoolExecutor(max_workers=2) as executor:
        owner = executor.submit(
            execution_module._cached_complete_array_facts,
            cache,
            key,
            data,
        )
        assert scan_started.wait(timeout=10)
        waiter = executor.submit(
            execution_module._cached_complete_array_facts,
            cache,
            key,
            data.copy(),
            cancel_callback=waiter_cancelled,
        )
        assert waiter_polled.wait(timeout=10)
        cancel_waiter.set()
        with pytest.raises(execution_module.OperationCancelled):
            waiter.result(timeout=2)
        assert not owner.done()
        release_scan.set()
        owner_facts = owner.result(timeout=30)

    assert owner_facts.all_finite is True
    assert scan_count == 1
    coordinator = execution_module._array_facts_cache_coordinator(cache)
    with coordinator.lock:
        assert coordinator.in_flight == {}


def test_different_cache_keys_scan_concurrently_without_key_state_leaks(monkeypatch):
    original = execution_module._complete_array_facts
    both_scanning = threading.Barrier(2)

    def synchronized_scan(value, **kwargs):
        both_scanning.wait(timeout=10)
        return original(value, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_complete_array_facts",
        synchronized_scan,
    )
    cache = ArrayFactsCache()
    data = np.arange(81, dtype=np.float32).reshape(9, 9)
    keys = (
        ArrayFactsKey(OutputPortKey("input", 0), "revision-a"),
        ArrayFactsKey(OutputPortKey("input_2", 0), "revision-b"),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(
            executor.submit(
                execution_module._cached_complete_array_facts,
                cache,
                key,
                data,
            )
            for key in keys
        )
        facts = tuple(future.result(timeout=30) for future in futures)

    assert all(item.all_finite is True for item in facts)
    coordinator = execution_module._array_facts_cache_coordinator(cache)
    with coordinator.lock:
        assert coordinator.in_flight == {}


def test_support_fingerprint_includes_propagation_bounds():
    low = execution_module._complete_array_facts(
        np.array([0.0, 1.0], dtype=np.float32),
        revision_fingerprint="low",
    )
    high = execution_module._complete_array_facts(
        np.array([10.0, 11.0], dtype=np.float32),
        revision_fingerprint="high",
    )

    assert execution_module._support_facts_fingerprint(
        (low,)
    ) != execution_module._support_facts_fingerprint((high,))


def test_unclipped_background_drops_unproven_signed_zero_guarantees():
    source = execution_module._complete_array_facts(
        np.array([-1.0, 1.0], dtype=np.float32),
        revision_fingerprint="source",
    )

    propagated = execution_module._propagate_shape_preserving_facts(
        "subtract_background",
        source,
        {"clip_negative": False},
        output_port=OutputPortKey("background", 0),
    )

    assert propagated is not None
    assert propagated.all_finite is True
    assert "nonnegative" not in propagated.guarantees
    assert "no-negative-zero" not in propagated.guarantees
    assert propagated.minimum is None
    assert propagated.maximum is None


def test_prepare_psf_projects_finite_float32_facts_across_odd_padding():
    source = execution_module._complete_array_facts(
        np.ones((8, 10), dtype=np.uint16),
        revision_fingerprint="psf-source",
    )

    propagated = execution_module._propagate_shape_preserving_facts(
        "prepare_validate_psf",
        source,
        {
            "clip_negatives": True,
            "normalize_sum": True,
            "force_odd_shape": True,
            "crop_empty_border": False,
        },
        output_port=OutputPortKey("prepared-psf", 0),
        output_shape=(9, 11),
        output_dtype="float32",
    )
    unsafe_cancellation = execution_module._propagate_shape_preserving_facts(
        "prepare_validate_psf",
        source,
        {"clip_negatives": False, "normalize_sum": True},
        output_port=OutputPortKey("prepared-psf", 0),
        output_shape=(8, 10),
        output_dtype="float32",
    )

    assert propagated is not None
    assert propagated.shape == (9, 11)
    assert propagated.dtype == "float32"
    assert propagated.element_count == 99
    assert propagated.all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(propagated.guarantees)
    assert unsafe_cancellation is None


@pytest.mark.parametrize(
    "operation_id",
    (
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
    ),
)
def test_deconvolution_projects_its_finite_nonnegative_output_theorem(operation_id):
    source = execution_module._complete_array_facts(
        np.array([[np.nan, -1.0], [1.0, np.inf]], dtype=np.float32),
        revision_fingerprint="deconvolution-source",
    )

    propagated = execution_module._propagate_shape_preserving_facts(
        operation_id,
        source,
        {"clip_output_negative": True},
        output_port=OutputPortKey("deconvolution", 0),
        output_dtype="float32",
    )

    assert propagated is not None
    assert propagated.all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(propagated.guarantees)
    assert propagated.minimum is None
    assert propagated.maximum is None


@pytest.mark.parametrize("operation_id", ("canny_edges", "otsu_threshold"))
def test_segmentation_projects_exact_boolean_facts(operation_id):
    source = execution_module._complete_array_facts(
        np.array([[np.nan, -1.0], [1.0, np.inf]], dtype=np.float32),
        revision_fingerprint="segmentation-source",
    )

    propagated = execution_module._propagate_shape_preserving_facts(
        operation_id,
        source,
        {},
        output_port=OutputPortKey("mask", 0),
        output_shape=(3, 7, 9),
        output_dtype="bool",
    )

    assert propagated is not None
    assert propagated.shape == (3, 7, 9)
    assert propagated.dtype == "bool"
    assert propagated.all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(propagated.guarantees)


@pytest.mark.parametrize("operation_id", ("canny_edges", "otsu_threshold"))
def test_segmentation_luma_planning_projects_shape_dtype_and_metadata(operation_id):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    assert pipeline.connect("input", node.id).success
    pipeline.set_param(node.id, "channel_axis", 1)
    data = np.zeros((2, 3, 9, 11), dtype=np.uint16)
    axes = (
        AxisMetadata("t", "time"),
        AxisMetadata("c", "channel"),
        AxisMetadata("y", "space"),
        AxisMetadata("x", "space"),
    )
    state = image_state_from_array(data, axes=axes)
    assert state is not None
    call = pipeline.prepare_node_call(node.id, (data,), (state,))

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        operation_id,
        call,
        (data.shape,),
        (data.dtype.name,),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert description.shape == (2, 9, 11)
    assert description.dtype == np.dtype(bool)
    assert output_state is not None
    assert output_state.shape == (2, 9, 11)
    assert output_state.dtype == "bool"
    assert tuple(axis.name for axis in output_state.axes) == ("t", "y", "x")
    assert output_state.kind == "binary mask"
    assert output_state.channels == ()

    with ComputeRegistry() as registry:
        implementation = registry.implementations_for_operation(
            operation_id,
            allow_experimental=False,
        )[0]
    (resident_state,) = execution_module._predict_device_node_states(
        pipeline,
        call,
        implementation,
        (np.zeros(description.shape, dtype=bool),),
    )
    assert resident_state == output_state

    with pytest.raises(RuntimeError, match="shape contract"):
        execution_module._predict_device_node_states(
            pipeline,
            call,
            implementation,
            (np.zeros(data.shape, dtype=bool),),
        )


_INTENSITY_EXAMPLE_WORKFLOW = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "red-channel-object-intensity-measurements.json"
)


def _restored_intensity_example() -> PrototypePipeline:
    workflow = load_workflow(_INTENSITY_EXAMPLE_WORKFLOW)
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        workflow["nodes"],
        workflow["connections"],
        workflow.get("output_tunnels", ()),
    )
    return pipeline


@pytest.fixture(scope="module")
def intensity_example_sample():
    data, layer_kwargs, _layer_type = make_sample_data()[1]
    return data, layer_kwargs


def _fresh_example_workloads(
    pipeline: PrototypePipeline,
    data: np.ndarray,
    layer_kwargs: dict,
):
    assert not pipeline.completed_node_ids
    assert not any(
        value is not None
        for outputs in pipeline.node_outputs.values()
        for value in outputs
    )
    request = PipelineRunRequest(
        run_id=101,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
        source_payloads={},
        compute_request=ComputeRequest(mode=ComputeMode.PREFER_GPU),
        manual_node_ids=frozenset(pipeline.manual_node_ids()),
    )
    runnable = pipeline.plan_execution(
        request.dirty_node_ids,
        manual_mode=MANUAL_RUN_SKIP,
        manual_node_ids=request.manual_node_ids,
        target_node_ids=request.target_node_ids,
    ).runnable_node_ids
    host_values, states, _source_results = (
        execution_module._initial_transaction_values(
            pipeline,
            request,
            runnable,
        )
    )
    with ComputeRegistry() as registry:
        workloads, _facts, _lineage = execution_module._assemble_workloads(
            pipeline,
            runnable,
            host_values,
            states,
            registry,
            False,
            seed_facts_by_port={},
        )
    return workloads


@pytest.mark.parametrize(
    ("dtype", "is_native"),
    (
        (np.dtype(np.uint16), True),
        (np.dtype(np.uint16).newbyteorder("S"), False),
    ),
)
def test_split_channels_planning_projects_every_exact_output_port(
    dtype,
    is_native,
):
    assert dtype.isnative is is_native
    pipeline = _restored_intensity_example()
    data = np.arange(4 * 2 * 4 * 5, dtype=np.uint16).reshape(4, 2, 4, 5)
    data = data.astype(dtype)
    source_results = pipeline.source_node_results(
        "input",
        data,
        {"vipp_axis_order": "CZYX"},
        "four-channel source",
        {},
    )
    ((source_value, source_state),) = source_results
    execution = pipeline.prepare_execution()
    pipeline.commit_node_results(execution, "input", source_results)
    assert len(pipeline.output_ports("split_channels_1")) == 4
    call = pipeline.prepare_node_call(
        "split_channels_1",
        (source_value,),
        (source_state,),
    )
    assert call is not None
    input_dtype = dtype.name if dtype.isnative else dtype.str

    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("Planning must not run the Split Channels kernel.")

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "split_channels",
        replace(call, cpu_function=forbidden_kernel),
        (data.shape,),
        (input_dtype,),
    )
    actual = call.cpu_function(source_value, **call.kwargs)

    assert projected is not None
    assert len(projected) == 4
    descriptions = tuple(item[0] for item in projected)
    states = tuple(item[1] for item in projected)
    assert len(actual) == 4
    assert all(item.shape == (2, 4, 5) for item in descriptions)
    assert all(item.dtype == dtype for item in descriptions)
    assert tuple((value.shape, value.dtype) for value in actual) == tuple(
        (item.shape, item.dtype) for item in descriptions
    )
    assert all(state is not None for state in states)
    assert all(state.shape == (2, 4, 5) for state in states)
    assert all(state.dtype == "uint16" for state in states)
    assert all(
        tuple(axis.name for axis in state.axes) == ("z", "y", "x")
        for state in states
    )
    assert [state.channels[0].name for state in states] == [
        channel.name for channel in source_state.channels
    ]
    assert any(
        connection.source_id == "split_channels_1"
        and connection.source_port == 2
        and connection.target_id == "gaussian_blur_1"
        for connection in pipeline.connections
    )


def test_split_channels_projection_ignores_none_dtype_attribute():
    class _ArrayLikeWithNoDtype:
        dtype = None

        def __init__(self, data):
            self._data = data

        def __array__(self, dtype=None, copy=None):
            array = np.asarray(self._data, dtype=dtype)
            return array.copy() if copy else array

    pipeline = _restored_intensity_example()
    data = np.arange(3 * 2 * 4 * 5, dtype=np.uint16).reshape(3, 2, 4, 5)
    ((source_value, source_state),) = pipeline.source_node_results(
        "input",
        data,
        {"vipp_axis_order": "CZYX"},
        "three-channel source",
        {},
    )
    call = pipeline.prepare_node_call(
        "split_channels_1",
        (source_value,),
        (source_state,),
    )
    assert call is not None
    array_like = _ArrayLikeWithNoDtype(source_value)

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "split_channels",
        replace(call, inputs=(array_like,)),
        (data.shape,),
        (data.dtype.name,),
    )
    actual = call.cpu_function(array_like, **call.kwargs)

    assert projected is not None
    assert all(description.dtype == np.dtype(np.uint16) for description, _ in projected)
    assert all(value.dtype == np.dtype(np.uint16) for value in actual)


def test_workload_assembly_publishes_every_split_channels_port():
    pipeline = _restored_intensity_example()
    data = np.arange(4 * 2 * 4 * 5, dtype=np.uint16).reshape(4, 2, 4, 5)
    source_results = pipeline.source_node_results(
        "input",
        data,
        {"vipp_axis_order": "CZYX"},
        "four-channel source",
        {},
    )
    ((source_value, source_state),) = source_results
    execution = pipeline.prepare_execution()
    pipeline.commit_node_results(execution, "input", source_results)
    assert len(pipeline.output_ports("split_channels_1")) == 4
    consumers = []
    for source_port in range(4):
        gaussian = pipeline.add_node("gaussian_blur")
        assert pipeline.connect(
            "split_channels_1",
            gaussian.id,
            source_port=source_port,
        ).success
        consumers.append(gaussian.id)

    with ComputeRegistry() as registry:
        workloads, _facts, _lineage = execution_module._assemble_workloads(
            pipeline,
            frozenset(pipeline.nodes),
            {OutputPortKey("input", 0): source_value},
            {OutputPortKey("input", 0): source_state},
            registry,
            False,
            seed_facts_by_port={},
        )

    by_node = {workload.node_id: workload for workload in workloads}
    for node_id in consumers:
        workload = by_node[node_id]
        assert workload.inputs_resolved is True
        assert workload.input_shapes == ((2, 4, 5),)
        assert workload.input_dtypes == ("uint16",)


def test_fresh_intensity_example_builds_exact_numeric_downstream_workloads(
    intensity_example_sample,
):
    pipeline = _restored_intensity_example()
    data, layer_kwargs = intensity_example_sample

    workloads = _fresh_example_workloads(pipeline, data, layer_kwargs)

    by_node = {workload.node_id: workload for workload in workloads}
    gaussian = by_node["gaussian_blur_1"]
    otsu = by_node["otsu_threshold_1"]
    assert gaussian.inputs_resolved is True
    assert gaussian.input_shapes == ((12, 96, 128),)
    assert gaussian.input_dtypes == ("uint16",)
    assert otsu.inputs_resolved is True
    assert otsu.input_shapes == ((12, 96, 128),)
    assert otsu.input_dtypes == ("uint16",)
    assert "object" not in gaussian.input_dtypes + otsu.input_dtypes


def test_unresolved_host_shape_keeps_all_transitive_workloads_unresolved():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    prepare_psf = pipeline.add_node("prepare_validate_psf")
    gaussian = pipeline.add_node("gaussian_blur")
    otsu = pipeline.add_node("otsu_threshold")
    pipeline.set_param(prepare_psf.id, "crop_empty_border", True)
    assert pipeline.connect("input", prepare_psf.id).success
    assert pipeline.connect(prepare_psf.id, gaussian.id).success
    assert pipeline.connect(gaussian.id, otsu.id).success
    data = np.ones((9, 11), dtype=np.float32)
    state = image_state_from_array(data)
    assert state is not None

    with ComputeRegistry() as registry:
        workloads, _facts, _lineage = execution_module._assemble_workloads(
            pipeline,
            frozenset(pipeline.nodes),
            {OutputPortKey("input", 0): data},
            {OutputPortKey("input", 0): state},
            registry,
            False,
            seed_facts_by_port={},
        )
        planning = plan_compute_decisions(
            ComputeRequest(mode=ComputeMode.PREFER_GPU),
            workloads,
            registry=registry,
            environment=ComputeEnvironment(),
        )

    by_node = {workload.node_id: workload for workload in workloads}
    assert by_node[prepare_psf.id].inputs_resolved is True
    for node_id in (gaussian.id, otsu.id):
        assert by_node[node_id].inputs_resolved is False
        assert by_node[node_id].input_shapes == ((),)
        assert by_node[node_id].input_dtypes == ("object",)
        assert planning.decisions_by_node[node_id].runtime_id == "cpu-numpy"
        assert (
            planning.decisions_by_node[node_id].reason
            is DecisionReason.WORKLOAD_UNSUPPORTED
        )


def test_fresh_intensity_example_prefer_gpu_completes_without_cucim(
    intensity_example_sample,
):
    pipeline = _restored_intensity_example()
    data, layer_kwargs = intensity_example_sample
    request = PipelineRunRequest(
        run_id=102,
        workflow=serialize_workflow(pipeline),
        input_data=data,
        input_metadata=layer_kwargs["metadata"],
        input_name=layer_kwargs["name"],
        source_payloads={},
        compute_request=ComputeRequest(mode=ComputeMode.PREFER_GPU),
        manual_node_ids=frozenset(pipeline.manual_node_ids()),
    )
    accelerator_without_libraries = ComputeEnvironment(
        os_name="Windows",
        os_release="test",
        execution_mode="native",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu",),
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
        device_id="cuda:0",
        device_name="Fake CUDA device",
        device_class="nvidia-cuda",
        total_accelerator_memory_bytes=8 * 1024**3,
        probe_status="available",
    )

    with (
        _ProbeRegistry(
            runtime_available=True,
            library_available=False,
        ) as registry,
        patch(
            "napari_vipp.core.compute_planning.ComputeEnvironment",
            return_value=accelerator_without_libraries,
        ),
    ):
        result = execute_pipeline_request(
            request,
            compute_registry=registry,
        )

    assert result.error == ""
    assert result.pipeline is not None
    table = result.pipeline.outputs["measure_objects_intensity_1"]
    assert table.row_count > 0
    assert result.execution_report is not None
    measurement = next(
        decision
        for decision in result.execution_report.actual_decisions
        if decision.node_id == "measure_objects_intensity_1"
    )
    assert measurement.runtime_id == "cpu-numpy"
    assert measurement.implementation_library_id == "cpu"
    assert measurement.reason_text


def test_resolved_measurement_planning_explains_missing_cucim():
    workload = WorkloadDescriptor(
        node_id="measurement",
        operation_id="measure_objects_intensity",
        input_shapes=((9, 11), (9, 11)),
        input_dtypes=("int32", "uint16"),
        parameters=(
            ("spatial_mode", "Auto from axes"),
            ("axis_names", ("y", "x")),
            ("axis_types", ("space", "space")),
            ("axis_scales", (1.0, 1.0)),
            ("axis_units", ("", "")),
            ("include_shape_descriptors", False),
            ("include_axis_descriptors", False),
            ("include_2d_boundary_descriptors", False),
            ("include_derived_shape_ratios", False),
            ("include_2d_shape_moments", False),
        ),
        resolved_spatial_ndim=2,
    )
    request = ComputeRequest(mode=ComputeMode.PREFER_GPU)
    validated_host = ComputeEnvironment(
        os_name="Windows",
        os_release="test",
        execution_mode="native",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
    )

    with (
        _ProbeRegistry(
            runtime_available=True,
            library_available=False,
        ) as registry,
        patch(
            "napari_vipp.core.compute_planning.ComputeEnvironment",
            return_value=validated_host,
        ),
    ):
        specs = registry.implementations_for_operation(
            workload.operation_id,
            allow_experimental=False,
        )
        environment, _warnings = probe_compute_environment(
            registry,
            request,
            specs,
        )
        planning = plan_compute_decisions(
            request,
            (workload,),
            registry=registry,
            environment=environment,
        )

    assert "cucim" in registry.library_probe_ids
    decision = planning.decisions_by_node[workload.node_id]
    assert decision.runtime_id == "cpu-numpy"
    assert decision.decision_kind is DecisionKind.POLICY_CPU
    assert decision.reason is DecisionReason.DEPENDENCY_UNAVAILABLE
    assert "cucim" in decision.reason_text.casefold()
