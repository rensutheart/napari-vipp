from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.compute_planning as planning_module
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    FallbackReason,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import (
    ComputePreflightError,
    actual_cpu_fallback_decision,
    plan_compute_decisions,
    probe_compute_environment,
)
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    FactCompleteness,
    PerformanceEvidence,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryProbeResult,
    RuntimeDevice,
    RuntimeProbeResult,
)
from napari_vipp.core.compute_specs import compute_specs_for


def _environment(*, runtime=True, libraries=("cpu", "cupyx", "cucim")):
    return ComputeEnvironment(
        os_name="Windows",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        runtime_ids=("cpu-numpy", "cuda-cupy") if runtime else ("cpu-numpy",),
        implementation_libraries=libraries,
        runtime_versions=(
            (("cuda-cupy", "14.1.1"), ("cupyx", "14.1.1")) if runtime else ()
        ),
        runtime_probe_fingerprints=(
            (("cuda-cupy", "fake-runtime-fingerprint"),) if runtime else ()
        ),
        runtime_metadata=(
            (
                (
                    "cuda-cupy",
                    (
                        ("cuda_runtime_version", "13020"),
                        ("driver_version", "13030"),
                    ),
                ),
            )
            if runtime
            else ()
        ),
        driver_version="13030" if runtime else "",
        device_id="cuda:0" if runtime else "cpu:0",
        device_name="Fake RTX" if runtime else "Host CPU",
        device_class="nvidia-cuda" if runtime else "host",
        device_metadata=((("compute_capability", "12.0"),) if runtime else ()),
        memory_topology="discrete" if runtime else "host",
        total_accelerator_memory_bytes=16 * 1024**3 if runtime else 0,
        probe_status="available",
    )


def _workload(
    operation_id="gaussian_blur",
    *,
    node_id="node",
    shape=(31, 37),
    dtype="float32",
    parameters=(("sigma", 1.2),),
    spatial_ndim=2,
):
    return WorkloadDescriptor(
        node_id,
        operation_id,
        (shape,),
        (dtype,),
        parameters=parameters,
        resolved_spatial_ndim=spatial_ndim,
    )


def _facts(workload, *, guarantees=()):
    shape = workload.input_shapes[0]
    return (
        ArrayFacts(
            shape,
            workload.input_dtypes[0],
            int(np.prod(shape)),
            "revision",
            completeness=FactCompleteness.COMPLETE,
            finite_count=int(np.prod(shape)),
            guarantees=guarantees,
        ),
    )


def _request(preference, *, fallback="visible", allow_experimental=True):
    return ComputeRequest(
        mode="selective",
        node_preferences={"node": preference},
        fallback_policy=fallback,
        allow_experimental=allow_experimental,
    )


def test_cpu_mode_returns_before_registry_construction_or_gpu_probe(monkeypatch):
    def forbidden_registry():
        raise AssertionError("CPU planning must not construct the GPU registry.")

    monkeypatch.setattr(planning_module, "ComputeRegistry", forbidden_registry)
    result = plan_compute_decisions(ComputeRequest(mode="cpu"), (_workload(),))

    assert result.decisions[0].runtime_id == "cpu-numpy"
    assert result.decisions[0].reason is DecisionReason.EXPLICIT_CPU
    assert result.environment.runtime_ids == ("cpu-numpy",)


def test_selective_all_cpu_preserves_a_healthy_host_environment(monkeypatch):
    registry = ComputeRegistry()

    def forbidden_probe(*_args, **_kwargs):
        raise AssertionError("An all-CPU selective plan must not probe a provider.")

    monkeypatch.setattr(registry, "probe_runtime", forbidden_probe)
    monkeypatch.setattr(registry, "probe_library", forbidden_probe)

    result = plan_compute_decisions(
        _request("cpu"),
        (_workload(),),
        registry=registry,
    )

    assert result.environment.probe_status == "available"
    assert result.environment.device_class == "host"
    assert result.environment.runtime_ids == ("cpu-numpy",)
    assert not result.warnings
    registry.close()


@pytest.mark.parametrize(
    ("os_name", "warning_fragment"),
    (("Linux", "Linux"), ("Darwin", "macOS")),
)
def test_unsupported_hosts_reject_exact_policy_without_provider_probe(
    monkeypatch,
    os_name,
    warning_fragment,
):
    registry = ComputeRegistry()

    def forbidden_probe(*_args, **_kwargs):
        raise AssertionError("An unsupported host must not import or probe a provider.")

    monkeypatch.setattr(registry, "probe_runtime", forbidden_probe)
    monkeypatch.setattr(registry, "probe_library", forbidden_probe)
    monkeypatch.setattr(
        planning_module,
        "ComputeEnvironment",
        lambda: ComputeEnvironment(os_name=os_name),
    )
    specs = compute_specs_for(
        "gaussian_blur",
        include_cpu=False,
        allow_experimental=True,
    )

    environment, warnings = probe_compute_environment(
        registry,
        ComputeRequest(mode="auto", allow_experimental=True),
        specs,
    )

    assert environment.probe_status == "available"
    assert environment.device_class == "host"
    assert warning_fragment in environment.probe_reason
    assert warnings == (environment.probe_reason,)
    registry.close()


def test_static_workload_rejection_precedes_environment_but_missing_facts_do_not():
    environment = _environment(runtime=False, libraries=("cpu",))

    invalid_dtype = plan_compute_decisions(
        ComputeRequest(mode="auto", allow_experimental=True),
        (_workload(dtype="uint16"),),
        environment=environment,
    )
    missing_facts = plan_compute_decisions(
        ComputeRequest(mode="auto", allow_experimental=True),
        (_workload(dtype="float32"),),
        environment=environment,
    )

    assert invalid_dtype.decisions[0].reason is DecisionReason.WORKLOAD_UNSUPPORTED
    assert missing_facts.decisions[0].reason is DecisionReason.ENVIRONMENT_UNSUPPORTED


def test_selective_exact_and_library_preferences_bypass_auto_threshold():
    workload = _workload()
    facts = {"node": _facts(workload)}

    exact = plan_compute_decisions(
        _request("implementation:cupyx-gaussian-blur-v1"),
        (workload,),
        environment=_environment(),
        array_facts=facts,
    )
    library = plan_compute_decisions(
        _request("library:cupyx"),
        (workload,),
        environment=_environment(),
        array_facts=facts,
    )

    assert exact.decisions[0].implementation_id == "cupyx-gaussian-blur-v1"
    assert library.decisions[0].implementation_library_id == "cupyx"
    assert exact.decisions[0].decision_kind is DecisionKind.SELECTED
    assert library.decisions[0].decision_kind is DecisionKind.SELECTED


def test_best_gpu_and_library_choose_fastest_of_multiple_valid_candidates():
    primary = compute_specs_for(
        "gaussian_blur", include_cpu=False, allow_experimental=True
    )[0]
    alternate = replace(
        primary,
        implementation_id="cupyx-gaussian-blur-alternate-v1",
        implementation_version="2",
    )
    registry = ComputeRegistry(implementation_specs=(primary, alternate))
    workload = _workload()
    evidence = {
        ("node", primary.implementation_id): PerformanceEvidence(
            0.2, 0.08, local_benchmark=True
        ),
        ("node", alternate.implementation_id): PerformanceEvidence(
            0.2, 0.05, local_benchmark=True
        ),
    }

    for preference in ("best_gpu", "library:cupyx"):
        result = plan_compute_decisions(
            _request(preference),
            (workload,),
            registry=registry,
            environment=_environment(),
            array_facts={"node": _facts(workload)},
            performance_evidence=evidence,
        )
        assert (
            result.decisions[0].implementation_id == "cupyx-gaussian-blur-alternate-v1"
        )
    registry.close()


def test_auto_cpu_is_policy_not_fallback_until_evidence_clears_gate():
    workload = _workload()
    request = ComputeRequest(mode="auto", allow_experimental=True)
    without_evidence = plan_compute_decisions(
        request,
        (workload,),
        environment=_environment(),
        array_facts={"node": _facts(workload)},
    )
    with_evidence = plan_compute_decisions(
        request,
        (workload,),
        environment=_environment(),
        array_facts={"node": _facts(workload)},
        performance_evidence={
            ("node", "cupyx-gaussian-blur-v1"): PerformanceEvidence(
                0.2,
                0.1,
                lower_confidence_speedup=1.5,
            )
        },
    )

    assert without_evidence.decisions[0].decision_kind is DecisionKind.POLICY_CPU
    assert not without_evidence.decisions[0].fallback_used
    assert without_evidence.decisions[0].reason is DecisionReason.PERFORMANCE_GATE
    assert with_evidence.decisions[0].decision_kind is DecisionKind.SELECTED


@pytest.mark.parametrize(
    "evidence",
    (
        None,
        PerformanceEvidence(
            0.2,
            0.19,
            lower_confidence_speedup=1.01,
        ),
    ),
    ids=("missing", "below-threshold"),
)
def test_unforced_auto_performance_gate_precedes_complete_fact_requirement(evidence):
    performance_evidence = (
        {} if evidence is None else {("node", "cupyx-gaussian-blur-v1"): evidence}
    )

    result = plan_compute_decisions(
        ComputeRequest(mode="auto", allow_experimental=True),
        (_workload(),),
        environment=_environment(),
        performance_evidence=performance_evidence,
    )

    assert result.decisions[0].reason is DecisionReason.PERFORMANCE_GATE


def test_viable_auto_evidence_still_requires_complete_scientific_facts():
    result = plan_compute_decisions(
        ComputeRequest(mode="auto", allow_experimental=True),
        (_workload(),),
        environment=_environment(),
        performance_evidence={
            ("node", "cupyx-gaussian-blur-v1"): PerformanceEvidence(
                0.2,
                0.1,
                lower_confidence_speedup=1.5,
            )
        },
    )

    assert result.decisions[0].reason is DecisionReason.WORKLOAD_UNSUPPORTED
    assert "facts" in result.decisions[0].reason_text.lower()


@pytest.mark.parametrize(
    ("environment", "reason", "fallback_reason"),
    [
        (
            _environment(runtime=False),
            DecisionReason.ENVIRONMENT_UNSUPPORTED,
            FallbackReason.ENVIRONMENT_UNSUPPORTED,
        ),
        (
            _environment(libraries=("cpu",)),
            DecisionReason.DEPENDENCY_UNAVAILABLE,
            FallbackReason.DEPENDENCY_UNAVAILABLE,
        ),
    ],
)
def test_visible_forced_preference_has_typed_fallback_and_warning(
    environment,
    reason,
    fallback_reason,
):
    workload = _workload()
    result = plan_compute_decisions(
        _request("library:cupyx"),
        (workload,),
        environment=environment,
        array_facts={"node": _facts(workload)},
    )
    decision = result.decisions[0]

    assert decision.decision_kind is DecisionKind.FALLBACK_CPU
    assert decision.reason is DecisionReason.VISIBLE_FALLBACK
    assert decision.fallback_reason is fallback_reason
    assert reason.value in decision.reason_text or result.warnings
    assert result.warnings


def test_strict_forced_preference_fails_complete_preflight():
    workload = _workload()

    with pytest.raises(ComputePreflightError) as error:
        plan_compute_decisions(
            _request("library:cupyx", fallback="strict"),
            (workload,),
            environment=_environment(libraries=("cpu",)),
            array_facts={"node": _facts(workload)},
        )

    assert len(error.value.failures) == 1
    assert error.value.failures[0].reason is DecisionReason.DEPENDENCY_UNAVAILABLE


@pytest.mark.parametrize("dtype", ("uint8", "uint16", "float64"))
def test_gaussian_nonpromoted_dtypes_are_explicit_typed_cpu_regions(dtype):
    workload = _workload(dtype=dtype)
    result = plan_compute_decisions(
        _request("best_gpu"),
        (workload,),
        environment=_environment(),
    )

    decision = result.decisions[0]
    assert decision.fallback_used
    assert decision.fallback_reason is FallbackReason.WORKLOAD_UNSUPPORTED
    assert "CPU" in decision.reason_text or "proven" in decision.reason_text


def test_float32_median_without_signed_zero_facts_uses_typed_cpu():
    workload = _workload(
        "median_filter",
        shape=(51, 53),
        parameters=(("size", 5),),
    )
    result = plan_compute_decisions(
        _request("best_gpu"),
        (workload,),
        environment=_environment(),
        array_facts={"node": _facts(workload)},
    )

    assert result.decisions[0].fallback_used
    assert "negative zero" in result.decisions[0].reason_text


def test_invalid_axis_is_preflight_error_even_under_visible_fallback():
    workload = _workload(parameters=(("sigma", 1.2), ("channel_axis", True)))

    with pytest.raises(ComputePreflightError, match="channel"):
        plan_compute_decisions(
            _request("best_gpu"),
            (workload,),
            environment=_environment(),
            array_facts={"node": _facts(workload)},
        )


def test_memory_cap_rejects_forced_gpu_before_execution():
    workload = _workload()
    request = ComputeRequest(
        mode="selective",
        node_preferences={"node": "best_gpu"},
        accelerator_memory_cap_bytes=1,
        allow_experimental=True,
    )
    result = plan_compute_decisions(
        request,
        (workload,),
        environment=_environment(),
        array_facts={"node": _facts(workload)},
    )

    assert result.decisions[0].fallback_reason is FallbackReason.MEMORY_LIMIT


def test_exact_hidden_candidate_is_unavailable_without_experimental_flag():
    result = plan_compute_decisions(
        _request(
            "implementation:cupyx-gaussian-blur-v1",
            allow_experimental=False,
        ),
        (_workload(),),
        environment=_environment(),
    )

    assert result.decisions[0].fallback_used
    assert result.decisions[0].fallback_reason is FallbackReason.DEPENDENCY_UNAVAILABLE


def test_execution_plan_shell_and_actual_fallback_preserve_typed_identity():
    workload = _workload()
    planned = plan_compute_decisions(
        _request("best_gpu"),
        (workload,),
        environment=_environment(),
        array_facts={"node": _facts(workload)},
    )
    selected = planned.decisions[0]
    # best_gpu has one scientifically valid candidate and therefore does not
    # need comparative timing evidence.
    assert selected.decision_kind is DecisionKind.SELECTED

    actual = actual_cpu_fallback_decision(
        selected,
        FallbackReason.OUT_OF_MEMORY,
        reason_text="The selected segment exhausted its admitted pool.",
    )
    public_plan = planned.as_execution_plan()

    assert actual.fallback_used
    assert actual.runtime_id == "cpu-numpy"
    assert actual.reason is DecisionReason.OUT_OF_MEMORY_FALLBACK
    assert public_plan.decisions == planned.decisions
    assert public_plan.request_fingerprint == planned.request.fingerprint


def test_public_environment_probe_preserves_exact_provider_provenance(monkeypatch):
    registry = ComputeRegistry()
    runtime_fingerprint = "runtime-fingerprint-a"
    runtime_metadata = (
        ("cuda_runtime_version", "13020"),
        ("driver_version", "13030"),
    )
    device_metadata = (("compute_capability", "12.0"),)
    cucim_metadata = (
        ("environment_record_schema", "napari-vipp-gpu-environment"),
        ("environment_record_schema_version", "1"),
        ("environment_track", "cuda13"),
        ("cupy_distribution", "cupy-cuda13x"),
        ("cucim_distribution", "cucim-cu13"),
        ("cucim_distribution_version", "26.6.0"),
        (
            "cucim_artifact_sha256",
            "586d3443091eea67ce2c697be2c490ca51977a5dbdf894b9318b270977134cf8",
        ),
    )

    def runtime_probe(_runtime_id):
        return RuntimeProbeResult(
            "cuda-cupy",
            True,
            version="14.1.1",
            devices=(
                RuntimeDevice(
                    "cuda:0",
                    "Fake RTX",
                    16 * 1024**3,
                    metadata=device_metadata,
                ),
            ),
            selected_device_id="cuda:0",
            environment_fingerprint=runtime_fingerprint,
            metadata=runtime_metadata,
        )

    def library_probe(library_id):
        if library_id == "cupyx":
            return ImplementationLibraryProbeResult(
                "cupyx",
                True,
                version="14.1.1",
            )
        return ImplementationLibraryProbeResult(
            "cucim",
            True,
            version="26.6.0",
            metadata=cucim_metadata,
        )

    monkeypatch.setattr(registry, "probe_runtime", runtime_probe)
    monkeypatch.setattr(registry, "probe_library", library_probe)
    specs = (
        compute_specs_for("gaussian_blur", include_cpu=False, allow_experimental=True)[
            0
        ],
        compute_specs_for(
            "rolling_ball_background",
            include_cpu=False,
            allow_experimental=True,
        )[0],
    )

    environment, warnings = probe_compute_environment(
        registry,
        ComputeRequest(mode="auto", allow_experimental=True),
        specs,
    )

    assert not warnings
    assert dict(environment.runtime_probe_fingerprints) == {
        "cuda-cupy": runtime_fingerprint
    }
    assert dict(dict(environment.runtime_metadata)["cuda-cupy"]) == dict(
        runtime_metadata
    )
    assert environment.driver_version == "13030"
    assert dict(environment.device_metadata) == dict(device_metadata)
    assert dict(dict(environment.implementation_library_metadata)["cucim"]) == dict(
        cucim_metadata
    )

    baseline_fingerprint = environment.fingerprint
    runtime_fingerprint = "runtime-fingerprint-b"
    changed, _warnings = probe_compute_environment(
        registry,
        ComputeRequest(mode="auto", allow_experimental=True),
        specs,
    )
    assert changed.fingerprint != baseline_fingerprint
    registry.close()


def test_preprobed_environment_reason_remains_a_visible_planning_warning():
    environment = replace(
        _environment(),
        probe_reason="A requested optional provider was unavailable.",
    )
    workload = _workload()

    result = plan_compute_decisions(
        _request("best_gpu"),
        (workload,),
        environment=environment,
        array_facts={"node": _facts(workload)},
    )

    assert result.warnings == (environment.probe_reason,)
