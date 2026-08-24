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
    FallbackPolicy,
    FallbackReason,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    OutputPortKey,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import (
    plan_compute_decisions,
    probe_compute_environment,
)
from napari_vipp.core.compute_policy import (
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
from napari_vipp.core.metadata import (
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.operations import canny_edges as cpu_canny_edges
from napari_vipp.core.operations import otsu_threshold as cpu_otsu_threshold
from napari_vipp.core.pipeline import (
    _EXACT_AXIS_CONTRACT_OPERATIONS,
    MANUAL_RUN_SKIP,
    NODE_LIBRARY,
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
            version="14.1.1" if self.library_available else "",
            reason_code="" if self.library_available else "library_unavailable",
            message=("" if self.library_available else f"{library_id} is unavailable."),
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
    captured_source_contexts = execution_module._capture_source_scientific_contexts(
        pipeline,
        run_request,
        cancel_callback=None,
    )
    source_contexts = {
        node_id: captured.scientific_context_fingerprint
        for node_id, captured in captured_source_contexts.items()
    }
    source_reuse_envelopes = {
        node_id: captured.source_reuse_envelope_fingerprint
        for node_id, captured in captured_source_contexts.items()
    }
    execution_module._publish_actual_compute_provenance(
        pipeline,
        run_request.compute_request,
        decisions,
        source_scientific_contexts=source_contexts,
        source_reuse_envelope_fingerprints=source_reuse_envelopes,
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
        (ComputeMode.CUSTOM, "library:cupy"),
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
                    otsu.id: ("implementation:cupy-otsu-threshold-exact-v1")
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
    calls = []
    cancel_event = threading.Event()
    original_scan = execution_module._complete_array_facts

    def cancel_during_scan(value, **kwargs):
        calls.append(np.asarray(value))
        cancel_event.set()
        return original_scan(value, **kwargs)

    monkeypatch.setattr(
        execution_module,
        "_complete_array_facts",
        cancel_during_scan,
    )
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
            # The patched scan raises the event only after exact source-context
            # hashing has finished and the fact scan has begun.
            cancel_event=cancel_event,
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


def test_integer_rescale_projects_only_its_dtype_proven_facts():
    integer_source = execution_module._complete_array_facts(
        np.arange(12, dtype=np.uint16).reshape(3, 4),
        revision_fingerprint="integer-rescale-source",
    )
    float_source = execution_module._complete_array_facts(
        np.arange(12, dtype=np.float32).reshape(3, 4),
        revision_fingerprint="float-rescale-source",
    )

    integer_output = execution_module._propagate_shape_preserving_facts(
        "rescale_intensity",
        integer_source,
        {"out_min": 0.0, "out_max": 65535.0},
        output_port=OutputPortKey("rescale", 0),
        output_dtype="uint16",
    )
    float_output = execution_module._propagate_shape_preserving_facts(
        "rescale_intensity",
        float_source,
        {"out_min": 0.0, "out_max": 1.0},
        output_port=OutputPortKey("rescale", 0),
        output_dtype="float32",
    )

    assert integer_output is not None
    assert integer_output.all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(integer_output.guarantees)
    assert integer_output.minimum is None
    assert integer_output.maximum is None
    assert float_output is None


def test_integer_rescale_axes_projects_facts_across_shape_change():
    integer_source = execution_module._complete_array_facts(
        np.arange(12, dtype=np.uint16).reshape(3, 4),
        revision_fingerprint="integer-axis-rescale-source",
    )
    float_source = execution_module._complete_array_facts(
        np.arange(12, dtype=np.float32).reshape(3, 4),
        revision_fingerprint="float-axis-rescale-source",
    )

    integer_output = execution_module._propagate_shape_preserving_facts(
        "rescale_axes",
        integer_source,
        {},
        output_port=OutputPortKey("rescale-axes", 0),
        output_shape=(6, 8),
        output_dtype="uint16",
    )
    float_output = execution_module._propagate_shape_preserving_facts(
        "rescale_axes",
        float_source,
        {},
        output_port=OutputPortKey("rescale-axes", 0),
        output_shape=(6, 8),
        output_dtype="float32",
    )

    assert integer_output is not None
    assert integer_output.shape == (6, 8)
    assert integer_output.element_count == 48
    assert integer_output.all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(integer_output.guarantees)
    assert integer_output.minimum is None
    assert integer_output.maximum is None
    assert float_output is None


def test_crop_projects_finite_float_facts_to_its_exact_output_shape():
    source = execution_module._complete_array_facts(
        np.arange(6 * 8, dtype=np.float32).reshape(6, 8),
        revision_fingerprint="crop-facts-source",
    )

    output = execution_module._propagate_shape_preserving_facts(
        "crop_stack",
        source,
        {},
        output_port=OutputPortKey("crop", 0),
        output_shape=(4, 5),
        output_dtype="float32",
    )

    assert output is not None
    assert output.shape == (4, 5)
    assert output.all_finite is True
    assert output.minimum == source.minimum
    assert output.maximum == source.maximum
    assert "extrema-conservative-enclosure" in output.guarantees


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


@pytest.mark.parametrize(
    "operation_id",
    ("binary_threshold", "canny_edges", "otsu_threshold"),
)
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


def test_extract_channel_projects_only_proven_subset_facts_without_extrema():
    finite_source = execution_module._complete_array_facts(
        np.arange(2 * 3 * 5, dtype=np.float32).reshape(2, 3, 5),
        revision_fingerprint="finite-channel-source",
    )
    partly_nonfinite_source = execution_module._complete_array_facts(
        np.asarray(
            [
                [[0.0, 1.0], [2.0, 3.0]],
                [[np.nan, 4.0], [5.0, 6.0]],
            ],
            dtype=np.float32,
        ),
        revision_fingerprint="nonfinite-channel-source",
    )

    finite = execution_module._propagate_shape_preserving_facts(
        "extract_channel",
        finite_source,
        {"channel": 1},
        output_port=OutputPortKey("channel", 0),
        output_shape=(3, 5),
        output_dtype="float32",
    )
    uncertain = execution_module._propagate_shape_preserving_facts(
        "extract_channel",
        partly_nonfinite_source,
        {"channel": 0},
        output_port=OutputPortKey("channel", 0),
        output_shape=(2, 2),
        output_dtype="float32",
    )

    assert finite is not None
    assert finite.shape == (3, 5)
    assert finite.dtype == "float32"
    assert finite.element_count == 15
    assert finite.all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(finite.guarantees)
    assert finite.minimum is None
    assert finite.maximum is None
    assert uncertain is not None
    assert uncertain.shape == (2, 2)
    assert uncertain.element_count == 4
    assert uncertain.all_finite is None
    assert uncertain.minimum is None
    assert uncertain.maximum is None


def test_extract_channel_planning_and_resident_metadata_match_exactly():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("extract_channel")
    pipeline.set_param(node.id, "channel", -1)
    assert pipeline.connect("input", node.id).success
    data = np.zeros((2, 3, 5, 7), dtype=np.uint16)
    axes = (
        AxisMetadata("t", "time", "s", 2.0, 4.0),
        AxisMetadata("c", "channel"),
        AxisMetadata("y", "space", "µm", 0.4, 1.25),
        AxisMetadata("x", "space", "µm", 0.6, -2.5),
    )
    channels = (
        ChannelMetadata(name="DAPI", emission_wavelength=461.0),
        ChannelMetadata(name="FITC", emission_wavelength=519.0),
        ChannelMetadata(name="TRITC", emission_wavelength=573.0),
    )
    state = image_state_from_array(
        data,
        axes=axes,
        channels=channels,
        history=("Imported calibrated TCYX source",),
    )
    assert state is not None
    call = pipeline.prepare_node_call(node.id, (data,), (state,))

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "extract_channel",
        call,
        (data.shape,),
        (data.dtype.name,),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert description.shape == (2, 5, 7)
    assert description.dtype == np.dtype(np.uint16)
    assert output_state is not None
    assert output_state.shape == (2, 5, 7)
    assert output_state.dtype == "uint16"
    assert output_state.axes == (state.axes[0], state.axes[2], state.axes[3])
    assert output_state.channels == (channels[2],)
    assert output_state.kind == "intensity image"
    assert output_state.history == (
        "Imported calibrated TCYX source",
        "Extract Channel: selected channel -1",
    )

    with ComputeRegistry() as registry:
        implementation = registry.implementations_for_operation(
            "extract_channel",
            allow_experimental=False,
        )[0]
    (resident_state,) = execution_module._predict_device_node_states(
        pipeline,
        call,
        implementation,
        (np.zeros(description.shape, dtype=np.uint16),),
    )
    assert resident_state == output_state

    with pytest.raises(RuntimeError, match="shape or dtype contract"):
        execution_module._predict_device_node_states(
            pipeline,
            call,
            implementation,
            (np.zeros(data.shape, dtype=np.uint16),),
        )
    with pytest.raises(RuntimeError, match="shape or dtype contract"):
        execution_module._predict_device_node_states(
            pipeline,
            call,
            implementation,
            (np.zeros(description.shape, dtype=np.float32),),
        )


def test_binary_threshold_resident_metadata_is_shape_preserving_bool():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("binary_threshold")
    assert pipeline.connect("input", node.id).success
    data = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    axes = (
        AxisMetadata("y", "space", "µm", 0.4, 1.25),
        AxisMetadata("x", "space", "µm", 0.6, -2.5),
    )
    state = image_state_from_array(
        data,
        axes=axes,
        history=("Imported calibrated YX source",),
    )
    assert state is not None
    call = pipeline.prepare_node_call(node.id, (data,), (state,))
    with ComputeRegistry() as registry:
        implementation = registry.implementations_for_operation(
            "binary_threshold",
            allow_experimental=False,
        )[0]

    (resident_state,) = execution_module._predict_device_node_states(
        pipeline,
        call,
        implementation,
        (np.zeros(data.shape, dtype=bool),),
    )

    assert resident_state is not None
    assert resident_state.shape == data.shape
    assert resident_state.dtype == "bool"
    assert resident_state.axes == state.axes
    assert resident_state.kind == "binary mask"
    assert resident_state.history[-1] == "Binary Threshold"
    with pytest.raises(RuntimeError, match="shape-preserving bool-mask contract"):
        execution_module._predict_device_node_states(
            pipeline,
            call,
            implementation,
            (np.zeros(data.shape, dtype=np.float32),),
        )


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
    host_values, states, _source_results = execution_module._initial_transaction_values(
        pipeline,
        request,
        runnable,
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
        tuple(axis.name for axis in state.axes) == ("z", "y", "x") for state in states
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


@pytest.mark.parametrize(
    "dtype",
    (
        np.dtype(np.uint16),
        np.dtype(np.uint16).newbyteorder("S"),
    ),
)
def test_rescale_axes_planning_projects_exact_shape_dtype_and_metadata(dtype):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    rescale = pipeline.add_node("rescale_axes")
    assert pipeline.connect("input", rescale.id).success
    pipeline.set_param(rescale.id, "x_scale", 2.0)
    pipeline.set_param(rescale.id, "y_scale", 0.5)
    pipeline.set_param(rescale.id, "z_scale", 1.5)
    pipeline.set_param(rescale.id, "lock_xy", False)
    pipeline.set_param(rescale.id, "interpolation", "Nearest neighbor")

    data = np.arange(2 * 6 * 5, dtype=np.uint16).reshape(2, 6, 5).astype(dtype)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "µm", 1.0),
            AxisMetadata("y", "space", "µm", 1.0),
            AxisMetadata("x", "space", "µm", 1.0),
        ),
    )
    assert state is not None
    call = pipeline.prepare_node_call(rescale.id, (data,), (state,))
    assert call is not None

    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("Planning must not run the Rescale Axes kernel.")

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "rescale_axes",
        replace(call, cpu_function=forbidden_kernel),
        (data.shape,),
        (data.dtype.str,),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert description.shape == (3, 3, 10)
    assert description.dtype == dtype
    assert output_state is not None
    assert output_state.shape == description.shape
    assert output_state.dtype == "uint16"
    assert tuple(axis.name for axis in output_state.axes) == ("z", "y", "x")
    assert tuple(axis.scale for axis in output_state.axes) == pytest.approx(
        (2.0 / 3.0, 2.0, 0.5)
    )


def test_crop_and_reorder_planning_use_exact_axis_contracts_without_kernels():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    reorder = pipeline.add_node("reorder_axes")
    assert pipeline.connect("input", crop.id).success
    assert pipeline.connect(crop.id, reorder.id).success
    pipeline.set_param(crop.id, "top", 1)
    pipeline.set_param(crop.id, "bottom", 1)
    pipeline.set_param(crop.id, "left", 2)
    pipeline.set_param(crop.id, "right", 0)
    pipeline.set_param(reorder.id, "order", "XZY")
    dtype = np.dtype(np.uint16).newbyteorder("S")
    data = np.arange(3 * 6 * 7, dtype=np.uint16).reshape(3, 6, 7).astype(dtype)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert state is not None

    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("Planning must not run an axis-transform kernel.")

    crop_call = pipeline.prepare_node_call(crop.id, (data,), (state,))
    assert crop_call is not None
    crop_projected = execution_module._project_host_planning_outputs(
        pipeline,
        "crop_stack",
        replace(crop_call, cpu_function=forbidden_kernel),
        (data.shape,),
        (data.dtype.str,),
    )
    assert crop_projected is not None
    ((crop_description, crop_state),) = crop_projected
    assert crop_description.shape == (3, 4, 5)
    assert crop_description.dtype == dtype
    assert crop_state is not None

    crop_proxy = execution_module._ArrayDescription(
        crop_description.shape,
        crop_description.dtype,
    )
    reorder_call = pipeline.prepare_node_call(
        reorder.id,
        (crop_proxy,),
        (crop_state,),
    )
    assert reorder_call is not None
    reorder_projected = execution_module._project_host_planning_outputs(
        pipeline,
        "reorder_axes",
        replace(reorder_call, cpu_function=forbidden_kernel),
        (crop_description.shape,),
        (crop_description.dtype.str,),
    )
    assert reorder_projected is not None
    ((reorder_description, reorder_state),) = reorder_projected
    assert reorder_description.shape == (5, 3, 4)
    assert reorder_description.dtype == dtype
    assert reorder_state is not None
    assert tuple(axis.name for axis in reorder_state.axes) == ("x", "z", "y")


@pytest.mark.parametrize(
    ("operation_id", "parameters"),
    (
        ("mip", {"axis": 0}),
        (
            "orthogonal_projection",
            {"method": "Maximum", "use_physical_scale": False},
        ),
    ),
)
def test_reducing_axis_contract_matches_native_endian_cpu_output(
    operation_id,
    parameters,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    assert pipeline.connect("input", node.id).success
    for name, value in parameters.items():
        pipeline.set_param(node.id, name, value)
    data = (
        np.arange(3 * 5 * 7, dtype=np.uint16).reshape(3, 5, 7).astype(np.dtype(">u2"))
    )
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert state is not None
    call = pipeline.prepare_node_call(node.id, (data,), (state,))
    assert call is not None
    actual = call.cpu_function(*call.inputs, **call.kwargs)

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        operation_id,
        call,
        (data.shape,),
        (data.dtype.str,),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert (description.shape, description.dtype) == (actual.shape, actual.dtype)
    assert description.dtype.isnative
    assert output_state is not None
    assert output_state.shape == actual.shape
    assert output_state.dtype == actual.dtype.name


@pytest.mark.parametrize(
    ("channel_colors", "expected_dtype"),
    (
        ("Red,Red", np.dtype(np.float64)),
        ("Red,Green", np.dtype(np.float32)),
    ),
)
def test_composite_axis_contract_matches_additive_cpu_dtype(
    channel_colors,
    expected_dtype,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    composite = pipeline.add_node("composite_to_rgb")
    assert pipeline.connect("input", composite.id).success
    pipeline.set_param(composite.id, "mapping_mode", "Manual")
    pipeline.set_param(composite.id, "channel_colors", channel_colors)
    pipeline.set_param(composite.id, "intensity_mapping", "Preserve numeric values")
    data = np.arange(2 * 5 * 7, dtype=np.uint16).reshape(2, 5, 7)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
        channels=(ChannelMetadata("first"), ChannelMetadata("second")),
    )
    assert state is not None
    call = pipeline.prepare_node_call(composite.id, (data,), (state,))
    assert call is not None
    actual = call.cpu_function(*call.inputs, **call.kwargs)

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "composite_to_rgb",
        call,
        (data.shape,),
        (data.dtype.name,),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert description.shape == actual.shape == (5, 7, 3)
    assert description.dtype == actual.dtype == expected_dtype
    assert output_state is not None
    assert output_state.dtype == expected_dtype.name


def test_set_pixel_size_planning_metadata_matches_authoritative_finalization():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pixel_size = pipeline.add_node("set_pixel_size")
    assert pipeline.connect("input", pixel_size.id).success
    pipeline.set_param(pixel_size.id, "x_size", 0.5)
    pipeline.set_param(pixel_size.id, "y_size", 0.75)
    pipeline.set_param(pixel_size.id, "z_size", 2.0)
    pipeline.set_param(pixel_size.id, "unit", "micrometer")
    data = np.zeros((3, 5, 7), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert state is not None
    call = pipeline.prepare_node_call(pixel_size.id, (data,), (state,))
    assert call is not None
    actual = call.cpu_function(*call.inputs, **call.kwargs)
    ((_, actual_state),) = pipeline.finalize_node_call(call, actual)

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "set_pixel_size",
        call,
        (data.shape,),
        (data.dtype.name,),
    )

    assert projected is not None
    ((_, projected_state),) = projected
    assert projected_state is not None and actual_state is not None
    assert projected_state.axes == actual_state.axes
    assert projected_state.history == actual_state.history
    assert tuple(axis.scale for axis in projected_state.axes) == (2.0, 0.75, 0.5)


def test_rescale_axes_keeps_downstream_prefer_gpu_workload_resolved():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    rescale = pipeline.add_node("rescale_axes")
    background = pipeline.add_node("subtract_background")
    convert = pipeline.add_node("convert_dtype")
    gaussian = pipeline.add_node("gaussian_blur_3d")
    assert pipeline.connect("input", rescale.id).success
    assert pipeline.connect(rescale.id, background.id).success
    assert pipeline.connect(background.id, convert.id).success
    assert pipeline.connect(convert.id, gaussian.id).success
    pipeline.set_param(rescale.id, "x_scale", 2.0)
    pipeline.set_param(rescale.id, "y_scale", 0.5)
    pipeline.set_param(rescale.id, "z_scale", 1.5)
    pipeline.set_param(rescale.id, "lock_xy", False)
    pipeline.set_param(rescale.id, "interpolation", "Nearest neighbor")
    pipeline.set_param(background.id, "radius", 2.0)
    pipeline.set_param(background.id, "spatial_mode", "2D YX")
    pipeline.set_param(convert.id, "output_dtype", "float32")
    pipeline.set_param(convert.id, "scaling", "preserve")

    data = np.arange(2 * 6 * 5, dtype=np.uint16).reshape(2, 6, 5)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "µm", 1.0),
            AxisMetadata("y", "space", "µm", 1.0),
            AxisMetadata("x", "space", "µm", 1.0),
        ),
    )
    assert state is not None
    source_port = OutputPortKey("input", 0)
    source_facts = execution_module._complete_array_facts(
        data,
        revision_fingerprint="rescale-prefer-gpu-source",
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
        _ProbeRegistry() as registry,
        patch(
            "napari_vipp.core.compute_planning.ComputeEnvironment",
            return_value=validated_host,
        ),
    ):
        workloads, facts_by_node, _lineage = execution_module._assemble_workloads(
            pipeline,
            frozenset(pipeline.nodes),
            {source_port: data},
            {source_port: state},
            registry,
            False,
            seed_facts_by_port={source_port: source_facts},
        )
        accelerated_nodes = (background, convert, gaussian)
        specs = tuple(
            spec
            for node in accelerated_nodes
            for spec in registry.implementations_for_operation(
                node.operation_id,
                allow_experimental=False,
            )
        )
        environment, _warnings = probe_compute_environment(
            registry,
            request,
            specs,
        )
        planning = plan_compute_decisions(
            request,
            workloads,
            registry=registry,
            environment=environment,
            array_facts=facts_by_node,
        )

    by_node = {workload.node_id: workload for workload in workloads}
    downstream = by_node[background.id]
    assert downstream.inputs_resolved is True
    assert downstream.input_shapes == ((3, 3, 10),)
    assert downstream.input_dtypes == ("uint16",)
    assert by_node[gaussian.id].inputs_resolved is True
    assert by_node[gaussian.id].input_shapes == ((3, 3, 10),)
    assert by_node[gaussian.id].input_dtypes == ("float32",)
    expected_implementations = {
        background.id: "cupy-subtract-background-v1",
        convert.id: "cupyx-convert-dtype-preserve-f32-v1",
        gaussian.id: "cupy-gaussian-blur-3d-v1",
    }
    for node_id, implementation_id in expected_implementations.items():
        decision = planning.decisions_by_node[node_id]
        assert decision.runtime_id == "cuda-cupy"
        assert decision.implementation_id == implementation_id
        assert decision.fallback_used is False


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


@pytest.mark.parametrize(
    ("operation_id", "input_dtype", "output_dtype"),
    (
        ("rescale_intensity", np.dtype(np.uint16), np.dtype(np.uint16)),
        ("unsharp_mask", np.dtype(np.float32), np.dtype(np.float32)),
        ("gamma_correction", np.dtype(np.uint16), np.dtype(np.uint16)),
        ("average_blur", np.dtype(bool), np.dtype(np.float32)),
        ("difference_of_gaussians", np.dtype(np.uint16), np.dtype(np.float32)),
        ("triangle_threshold", np.dtype(np.uint16), np.dtype(bool)),
        ("euclidean_distance_transform", np.dtype(bool), np.dtype(np.float32)),
        ("h_maxima_markers", np.dtype(np.float32), np.dtype(np.int32)),
        ("save_output", np.dtype(np.uint16), np.dtype(np.uint16)),
    ),
)
def test_exact_host_shape_dtype_operations_project_without_running_kernel(
    operation_id,
    input_dtype,
    output_dtype,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node(operation_id)
    assert pipeline.connect("input", node.id).success
    data = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7).astype(input_dtype)
    state = image_state_from_array(data)
    assert state is not None
    call = pipeline.prepare_node_call(node.id, (data,), (state,))
    assert call is not None

    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("Planning must not run the authoritative CPU kernel.")

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        operation_id,
        replace(call, cpu_function=forbidden_kernel),
        (data.shape,),
        (data.dtype.name,),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert description.shape == data.shape
    assert description.dtype == output_dtype
    assert output_state is not None
    assert output_state.shape == data.shape
    assert output_state.dtype == output_dtype.name


def test_every_cpu_only_image_transform_has_a_planning_contract():
    with ComputeRegistry() as registry:
        cpu_only = {
            operation.id
            for operation in NODE_LIBRARY
            if operation.has_input
            and operation.output_type != "table"
            and not any(
                implementation.runtime_id != "cpu-numpy"
                for implementation in registry.implementations_for_operation(
                    operation.id,
                    allow_experimental=True,
                )
            )
        }
    handled = (
        set(execution_module._EXACT_HOST_SHAPE_DTYPE_POLICIES)
        | set(execution_module._EXACT_HOST_IDENTITY_OPERATIONS)
        | set(execution_module._EXACT_HOST_MULTI_INPUT_DTYPE_POLICIES)
        | set(_EXACT_AXIS_CONTRACT_OPERATIONS)
        | {"prepare_validate_psf"}
    )

    assert cpu_only - handled == set()


def test_scientific_preflight_uses_the_same_boolean_contract_as_execution():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    triangle = pipeline.add_node("triangle_threshold")
    assert pipeline.connect("input", triangle.id).success
    data = np.arange(7 * 9, dtype=np.uint16).reshape(7, 9)
    state = image_state_from_array(data)
    assert state is not None
    call = pipeline.prepare_node_call(
        triangle.id,
        (data,),
        (state,),
        axis_contract_only=True,
    )
    assert call is not None

    ((projected_value, projected_state),) = pipeline._axis_contract_preflight_results(
        call
    )
    actual_value = call.cpu_function(*call.inputs, **call.kwargs)
    ((_, actual_state),) = pipeline.finalize_node_call(call, actual_value)

    assert projected_value.shape == actual_value.shape == data.shape
    assert projected_value.dtype == actual_value.dtype == np.dtype(bool)
    assert projected_state is not None and actual_state is not None
    assert projected_state.dtype == actual_state.dtype == "bool"
    assert projected_state.kind == actual_state.kind == "binary mask"
    assert projected_state.axes == actual_state.axes
    assert projected_state.history == actual_state.history


def test_batch_output_projection_is_an_exact_state_identity():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    batch = pipeline.add_node("batch_output")
    assert pipeline.connect("input", batch.id).success
    dtype = np.dtype(np.uint16).newbyteorder("S")
    data = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7).astype(dtype)
    state = image_state_from_array(data)
    assert state is not None
    call = pipeline.prepare_node_call(batch.id, (data,), (state,))
    assert call is not None

    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("Planning must not run the Batch Output function.")

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "batch_output",
        replace(call, cpu_function=forbidden_kernel),
        (data.shape,),
        (data.dtype.str,),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert description.shape == data.shape
    assert description.dtype == dtype
    assert output_state is state


def test_multi_input_scatter_projection_has_exact_raster_descriptor_and_axes():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second_source = pipeline.add_node("input")
    scatter = pipeline.add_node("colocalization_scatter_plot")
    assert pipeline.connect("input", scatter.id, target_port=0).success
    assert pipeline.connect(second_source.id, scatter.id, target_port=1).success
    pipeline.set_param(scatter.id, "output_size", 128)
    channel_1 = np.arange(8 * 9, dtype=np.uint16).reshape(8, 9)
    channel_2 = np.flip(channel_1, axis=1).copy()
    state_1 = image_state_from_array(channel_1)
    state_2 = image_state_from_array(channel_2)
    assert state_1 is not None and state_2 is not None
    call = pipeline.prepare_node_call(
        scatter.id,
        (channel_1, channel_2),
        (state_1, state_2),
    )
    assert call is not None

    def forbidden_kernel(*_args, **_kwargs):
        raise AssertionError("Planning must not render a scatter plot.")

    projected = execution_module._project_host_planning_outputs(
        pipeline,
        "colocalization_scatter_plot",
        replace(call, cpu_function=forbidden_kernel),
        (channel_1.shape, channel_2.shape),
        (channel_1.dtype.name, channel_2.dtype.name),
    )

    assert projected is not None
    ((description, output_state),) = projected
    assert description.shape == (128, 128, 3)
    assert description.dtype == np.dtype(np.float32)
    assert output_state is not None
    assert output_state.shape == description.shape
    assert tuple(axis.name for axis in output_state.axes) == ("y", "x", "rgb")
    assert all(axis.source_axis is None for axis in output_state.axes)


def test_cpu_only_boundaries_keep_later_prefer_gpu_nodes_resolved():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    rescale_axes = pipeline.add_node("rescale_axes")
    background = pipeline.add_node("subtract_background")
    rescale_intensity = pipeline.add_node("rescale_intensity")
    gamma = pipeline.add_node("gamma_correction")
    sigma = pipeline.add_node("sigma_filter")
    unsharp = pipeline.add_node("unsharp_mask")
    otsu = pipeline.add_node("otsu_threshold")
    outliers = pipeline.add_node("remove_binary_outliers")
    small = pipeline.add_node("remove_small_objects")
    batch = pipeline.add_node("batch_output")
    labels = pipeline.add_node("label_connected_components")
    for source_id, target_id in (
        ("input", rescale_axes.id),
        (rescale_axes.id, background.id),
        (background.id, rescale_intensity.id),
        (rescale_intensity.id, gamma.id),
        (gamma.id, sigma.id),
        (sigma.id, unsharp.id),
        (unsharp.id, otsu.id),
        (otsu.id, outliers.id),
        (outliers.id, small.id),
        (small.id, batch.id),
        (batch.id, labels.id),
    ):
        assert pipeline.connect(source_id, target_id).success
    pipeline.set_param(rescale_axes.id, "x_scale", 1.0)
    pipeline.set_param(rescale_axes.id, "y_scale", 1.0)
    pipeline.set_param(rescale_axes.id, "z_scale", 1.0)
    pipeline.set_param(rescale_axes.id, "lock_xy", False)
    pipeline.set_param(rescale_axes.id, "interpolation", "Nearest neighbor")
    pipeline.set_param(background.id, "radius", 2.0)
    pipeline.set_param(background.id, "spatial_mode", "2D YX")
    pipeline.set_param(rescale_intensity.id, "cutoff_mode", "Values")
    pipeline.set_param(rescale_intensity.id, "in_low_value", 0.0)
    pipeline.set_param(rescale_intensity.id, "in_high_value", 65535.0)
    pipeline.set_param(rescale_intensity.id, "out_min", 0.0)
    pipeline.set_param(rescale_intensity.id, "out_max", 65535.0)

    data = np.arange(4 * 24 * 24, dtype=np.uint16).reshape(4, 24, 24)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert state is not None
    source_port = OutputPortKey("input", 0)
    source_facts = execution_module._complete_array_facts(
        data,
        revision_fingerprint="cpu-boundary-chain-source",
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
        _ProbeRegistry() as registry,
        patch(
            "napari_vipp.core.compute_planning.ComputeEnvironment",
            return_value=validated_host,
        ),
    ):
        workloads, facts_by_node, _lineage = execution_module._assemble_workloads(
            pipeline,
            frozenset(pipeline.nodes),
            {source_port: data},
            {source_port: state},
            registry,
            False,
            seed_facts_by_port={source_port: source_facts},
        )
        accelerated_nodes = (background, sigma, otsu, outliers, small, labels)
        specs = tuple(
            spec
            for node in accelerated_nodes
            for spec in registry.implementations_for_operation(
                node.operation_id,
                allow_experimental=False,
            )
        )
        environment, _warnings = probe_compute_environment(
            registry,
            request,
            specs,
        )
        planning = plan_compute_decisions(
            request,
            workloads,
            registry=registry,
            environment=environment,
            array_facts=facts_by_node,
        )

    by_node = {workload.node_id: workload for workload in workloads}
    assert all(workload.inputs_resolved for workload in by_node.values())
    assert by_node[sigma.id].input_shapes == (data.shape,)
    assert by_node[sigma.id].input_dtypes == ("uint16",)
    assert by_node[labels.id].input_shapes == (data.shape,)
    assert by_node[labels.id].input_dtypes == ("bool",)
    for node in accelerated_nodes:
        decision = planning.decisions_by_node[node.id]
        assert decision.runtime_id == "cuda-cupy"
        assert decision.fallback_used is False


def test_student_filter_chain_has_exact_fresh_planning_descriptors():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    rescale = pipeline.add_node("rescale_intensity")
    convert = pipeline.add_node("convert_dtype")
    gaussian = pipeline.add_node("gaussian_blur")
    unsharp = pipeline.add_node("unsharp_mask")
    otsu = pipeline.add_node("otsu_threshold")
    remove = pipeline.add_node("remove_small_objects")
    pipeline.set_param(rescale.id, "cutoff_mode", "Values")
    pipeline.set_param(rescale.id, "in_low_value", 0.0)
    pipeline.set_param(rescale.id, "in_high_value", 100.0)
    pipeline.set_param(rescale.id, "out_min", 0.0)
    pipeline.set_param(rescale.id, "out_max", 65535.0)
    pipeline.set_param(convert.id, "output_dtype", "float32")
    pipeline.set_param(convert.id, "scaling", "preserve")
    for source_id, target_id in (
        ("input", rescale.id),
        (rescale.id, convert.id),
        (convert.id, gaussian.id),
        (gaussian.id, unsharp.id),
        (unsharp.id, otsu.id),
        (otsu.id, remove.id),
    ):
        assert pipeline.connect(source_id, target_id).success

    data = np.arange(2 * 7 * 9, dtype=np.uint16).reshape(2, 7, 9)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    assert state is not None
    source_port = OutputPortKey("input", 0)
    source_facts = execution_module._complete_array_facts(
        data,
        revision_fingerprint="student-chain-source",
    )
    with ComputeRegistry() as registry:
        workloads, facts_by_node, _lineage = execution_module._assemble_workloads(
            pipeline,
            frozenset(pipeline.nodes),
            {source_port: data},
            {source_port: state},
            registry,
            False,
            seed_facts_by_port={source_port: source_facts},
        )

    by_node = {workload.node_id: workload for workload in workloads}
    assert all(
        by_node[node_id].inputs_resolved
        for node_id in (
            rescale.id,
            convert.id,
            gaussian.id,
            unsharp.id,
            otsu.id,
            remove.id,
        )
    )
    assert by_node[gaussian.id].input_shapes == (data.shape,)
    assert by_node[gaussian.id].input_dtypes == ("float32",)
    assert by_node[otsu.id].input_shapes == (data.shape,)
    assert by_node[otsu.id].input_dtypes == ("float32",)
    assert by_node[remove.id].input_shapes == (data.shape,)
    assert by_node[remove.id].input_dtypes == ("bool",)
    assert facts_by_node[gaussian.id][0].all_finite is True
    assert {"nonnegative", "no-negative-zero"} <= set(
        facts_by_node[gaussian.id][0].guarantees
    )

    exact_specs = {}
    with ComputeRegistry() as registry:
        for node in (gaussian, otsu, remove):
            exact_specs[node.id] = registry.implementations_for_operation(
                node.operation_id,
                allow_experimental=True,
            )[0]
    request = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={
            node_id: NodeComputePreference(
                NodePreferenceKind.IMPLEMENTATION,
                spec.implementation_id,
            )
            for node_id, spec in exact_specs.items()
        },
        fallback_policy=FallbackPolicy.STRICT,
        allow_experimental=True,
    )
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
        _ProbeRegistry() as registry,
        patch(
            "napari_vipp.core.compute_planning.ComputeEnvironment",
            return_value=validated_host,
        ),
    ):
        environment, _warnings = probe_compute_environment(
            registry,
            request,
            tuple(exact_specs.values()),
        )
        planning = plan_compute_decisions(
            request,
            workloads,
            registry=registry,
            environment=environment,
            array_facts=facts_by_node,
        )

    for node_id, expected in exact_specs.items():
        decision = planning.decisions_by_node[node_id]
        assert decision.implementation_id == expected.implementation_id
        assert decision.runtime_id == expected.runtime_id
        assert not decision.fallback_used


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


def test_fresh_intensity_example_prefer_gpu_completes_without_gpu_libraries(
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


def test_resolved_measurement_planning_explains_missing_cupy():
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

    assert registry.library_probe_ids == ["cupy"]
    decision = planning.decisions_by_node[workload.node_id]
    assert decision.runtime_id == "cpu-numpy"
    assert decision.decision_kind is DecisionKind.POLICY_CPU
    assert decision.reason is DecisionReason.DEPENDENCY_UNAVAILABLE
    assert "cupy" in decision.reason_text.casefold()
