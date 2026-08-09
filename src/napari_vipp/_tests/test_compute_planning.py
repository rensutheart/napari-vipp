from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import napari_vipp.core.compute as compute_module
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
    PHASE1_CUCIM_BUILD_RECIPE_ID,
    PHASE1_CUCIM_SOURCE_COMMIT,
    PHASE1_CUCIM_SOURCE_TAG,
    PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256,
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
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for


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
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
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
        device_name=(
            "NVIDIA GeForce RTX 5090" if runtime else "Host CPU"
        ),
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
        mode="custom",
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


def test_cpu_mode_records_scientific_stack_without_optional_provider_probe(
    monkeypatch,
):
    requested_distributions: list[str] = []

    def version(distribution):
        requested_distributions.append(distribution)
        return {
            "numpy": "2.5.1",
            "scipy": "1.18.0",
            "scikit-image": "0.26.0",
        }[distribution]

    monkeypatch.setattr(compute_module.importlib.metadata, "version", version)

    result = plan_compute_decisions(ComputeRequest(mode="cpu"), (_workload(),))

    assert dict(result.environment.scientific_stack_versions) == {
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "scikit-image": "0.26.0",
    }
    assert requested_distributions == ["numpy", "scipy", "scikit-image"]


def test_custom_all_cpu_preserves_a_healthy_host_environment(monkeypatch):
    registry = ComputeRegistry()

    def forbidden_probe(*_args, **_kwargs):
        raise AssertionError("An all-CPU custom plan must not probe a provider.")

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


def test_custom_exact_and_library_preferences_bypass_auto_threshold():
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


def test_compatible_secondary_nvidia_device_is_admitted_in_every_gpu_mode():
    workload = _workload()
    facts = {"node": _facts(workload)}
    secondary = replace(
        _environment(),
        device_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        device_metadata=(("compute_capability", "8.9"),),
    )

    automatic = plan_compute_decisions(
        ComputeRequest(mode="auto"),
        (workload,),
        environment=secondary,
        array_facts=facts,
    )
    preferred = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (workload,),
        environment=secondary,
        array_facts=facts,
    )
    custom_auto = plan_compute_decisions(
        _request("auto"),
        (workload,),
        environment=secondary,
        array_facts=facts,
    )
    pinned = plan_compute_decisions(
        _request("library:cupyx"),
        (workload,),
        environment=secondary,
        array_facts=facts,
    )

    decisions = (
        automatic.decisions[0],
        preferred.decisions[0],
        custom_auto.decisions[0],
        pinned.decisions[0],
    )
    assert all(
        decision.decision_kind is DecisionKind.SELECTED for decision in decisions
    )
    assert all(decision.runtime_id == "cuda-cupy" for decision in decisions)
    assert all(
        decision.reason is DecisionReason.SELECTED_IMPLEMENTATION
        for decision in decisions
    )


def test_compatible_device_keeps_auto_performance_semantics():
    workload = _workload()
    facts = {"node": _facts(workload)}
    environment = replace(
        _environment(),
        device_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        device_metadata=(("compute_capability", "8.9"),),
    )
    slow = {
        ("node", "cupyx-gaussian-blur-v1"): PerformanceEvidence(
            0.2,
            0.19,
            lower_confidence_speedup=1.01,
        )
    }
    fast = {
        ("node", "cupyx-gaussian-blur-v1"): PerformanceEvidence(
            0.2,
            0.1,
            lower_confidence_speedup=1.5,
        )
    }

    automatic_slow = plan_compute_decisions(
        ComputeRequest(mode="auto"),
        (workload,),
        environment=environment,
        array_facts=facts,
        performance_evidence=slow,
    )
    automatic_fast = plan_compute_decisions(
        ComputeRequest(mode="auto"),
        (workload,),
        environment=environment,
        array_facts=facts,
        performance_evidence=fast,
    )
    preferred = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (workload,),
        environment=environment,
        array_facts=facts,
        performance_evidence=slow,
    )
    pinned = plan_compute_decisions(
        _request("library:cupyx"),
        (workload,),
        environment=environment,
        array_facts=facts,
        performance_evidence=slow,
    )

    assert automatic_slow.decisions[0].decision_kind is DecisionKind.POLICY_CPU
    assert automatic_slow.decisions[0].reason is DecisionReason.PERFORMANCE_GATE
    assert automatic_fast.decisions[0].decision_kind is DecisionKind.SELECTED
    assert preferred.decisions[0].decision_kind is DecisionKind.SELECTED
    assert pinned.decisions[0].decision_kind is DecisionKind.SELECTED


def test_public_custom_candidate_requires_explicit_custom_choice():
    base = compute_specs_for("gaussian_blur", include_cpu=False)[0]
    custom = replace(base, admission_tier=AdmissionTier.PUBLIC_CUSTOM)
    registry = ComputeRegistry(implementation_specs=(custom,))
    workload = _workload()
    facts = {"node": _facts(workload)}
    evidence = {
        ("node", custom.implementation_id): PerformanceEvidence(
            0.2,
            0.05,
            lower_confidence_speedup=2.0,
        )
    }

    automatic = plan_compute_decisions(
        ComputeRequest(mode="auto"),
        (workload,),
        registry=registry,
        environment=_environment(),
        array_facts=facts,
        performance_evidence=evidence,
    )
    explicit = plan_compute_decisions(
        _request(
            f"implementation:{custom.implementation_id}",
            allow_experimental=False,
        ),
        (workload,),
        registry=registry,
        environment=_environment(),
        array_facts=facts,
    )

    assert automatic.decisions[0].decision_kind is DecisionKind.POLICY_CPU
    assert explicit.decisions[0].implementation_id == custom.implementation_id
    assert explicit.decisions[0].decision_kind is DecisionKind.SELECTED
    registry.close()


def test_prefer_gpu_includes_reviewed_public_custom_candidates():
    base = compute_specs_for("gaussian_blur", include_cpu=False)[0]
    custom = replace(base, admission_tier=AdmissionTier.PUBLIC_CUSTOM)
    registry = ComputeRegistry(implementation_specs=(custom,))
    workload = _workload()

    result = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (workload,),
        registry=registry,
        environment=_environment(),
        array_facts={"node": _facts(workload)},
        performance_evidence={},
    )

    decision = result.decisions[0]
    assert decision.decision_kind is DecisionKind.SELECTED
    assert decision.implementation_id == custom.implementation_id
    assert "speed threshold" in decision.reason_text
    registry.close()


def test_prefer_gpu_requires_opt_in_for_developer_hidden_candidates():
    public = compute_specs_for("gaussian_blur", include_cpu=False)[0]
    hidden = replace(public, admission_tier=AdmissionTier.DEVELOPER_HIDDEN)
    registry = ComputeRegistry(implementation_specs=(hidden,))
    workload = _workload()

    default = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (workload,),
        registry=registry,
        environment=_environment(),
        array_facts={"node": _facts(workload)},
    )
    experimental = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu", allow_experimental=True),
        (workload,),
        registry=registry,
        environment=_environment(),
        array_facts={"node": _facts(workload)},
    )

    assert default.decisions[0].decision_kind is DecisionKind.POLICY_CPU
    assert default.decisions[0].reason is DecisionReason.NO_VALIDATED_IMPLEMENTATION
    assert not default.decisions[0].fallback_used
    assert experimental.decisions[0].implementation_id == hidden.implementation_id
    assert experimental.decisions[0].decision_kind is DecisionKind.SELECTED
    registry.close()


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


@pytest.mark.parametrize(
    "evidence",
    (
        {},
        {
            ("node", "cupyx-gaussian-blur-v1"): PerformanceEvidence(
                cpu_seconds=1.0,
                candidate_seconds=2.0,
                lower_confidence_speedup=0.4,
            )
        },
    ),
    ids=("missing", "gpu-measured-slower-than-cpu"),
)
def test_prefer_gpu_bypasses_cpu_performance_gate(evidence):
    workload = _workload()

    result = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (workload,),
        environment=_environment(),
        array_facts={"node": _facts(workload)},
        performance_evidence=evidence,
    )

    decision = result.decisions[0]
    assert decision.decision_kind is DecisionKind.SELECTED
    assert decision.implementation_id == "cupyx-gaussian-blur-v1"
    assert decision.reason is DecisionReason.SELECTED_IMPLEMENTATION


@pytest.mark.parametrize("evidence_kind", ("missing", "partial"))
def test_prefer_gpu_uses_deterministic_id_when_candidate_evidence_is_incomplete(
    evidence_kind,
):
    base = compute_specs_for("gaussian_blur", include_cpu=False)[0]
    alphabetical = replace(base, implementation_id="aaa-gaussian-gpu-v1")
    measured = replace(base, implementation_id="zzz-gaussian-gpu-v1")
    registry = ComputeRegistry(implementation_specs=(measured, alphabetical))
    workload = _workload()
    evidence = (
        {}
        if evidence_kind == "missing"
        else {
            ("node", measured.implementation_id): PerformanceEvidence(
                cpu_seconds=1.0,
                candidate_seconds=0.01,
                local_benchmark=True,
            )
        }
    )

    result = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (workload,),
        registry=registry,
        environment=_environment(),
        array_facts={"node": _facts(workload)},
        performance_evidence=evidence,
    )

    assert result.decisions[0].implementation_id == alphabetical.implementation_id
    registry.close()


def test_prefer_gpu_uses_fastest_gpu_only_with_complete_candidate_evidence():
    base = compute_specs_for("gaussian_blur", include_cpu=False)[0]
    alphabetical = replace(base, implementation_id="aaa-gaussian-gpu-v1")
    fastest = replace(base, implementation_id="zzz-gaussian-gpu-v1")
    registry = ComputeRegistry(implementation_specs=(fastest, alphabetical))
    workload = _workload()
    evidence = {
        ("node", alphabetical.implementation_id): PerformanceEvidence(
            cpu_seconds=1.0,
            candidate_seconds=0.50,
            local_benchmark=True,
        ),
        ("node", fastest.implementation_id): PerformanceEvidence(
            cpu_seconds=1.0,
            candidate_seconds=0.25,
            local_benchmark=True,
        ),
    }

    result = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (workload,),
        registry=registry,
        environment=_environment(),
        array_facts={"node": _facts(workload)},
        performance_evidence=evidence,
    )

    assert result.decisions[0].implementation_id == fastest.implementation_id
    registry.close()


def test_prefer_gpu_ignores_dormant_node_preferences():
    workload = _workload()
    request = ComputeRequest(
        mode="prefer_gpu",
        node_preferences={"node": "cpu"},
    )

    result = plan_compute_decisions(
        request,
        (workload,),
        environment=_environment(),
        array_facts={"node": _facts(workload)},
    )

    decision = result.decisions[0]
    assert decision.decision_kind is DecisionKind.SELECTED
    assert decision.requested_preference.kind.value == "auto"
    assert decision.runtime_id == "cuda-cupy"


def test_auto_uses_reviewed_default_without_evidence_and_measured_candidate_with_it():
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

    assert without_evidence.decisions[0].decision_kind is DecisionKind.SELECTED
    assert not without_evidence.decisions[0].fallback_used
    assert without_evidence.decisions[0].implementation_id == (
        "cupyx-gaussian-blur-v1"
    )
    assert "reviewed Auto default" in without_evidence.decisions[0].reason_text
    assert with_evidence.decisions[0].decision_kind is DecisionKind.SELECTED


@pytest.mark.parametrize(
    ("evidence", "expected_reason"),
    (
        (None, DecisionReason.WORKLOAD_UNSUPPORTED),
        (
            PerformanceEvidence(
                0.2,
                0.19,
                lower_confidence_speedup=1.01,
            ),
            DecisionReason.PERFORMANCE_GATE,
        ),
    ),
    ids=("missing", "below-threshold"),
)
def test_unforced_auto_applies_reviewed_default_or_measured_performance_gate(
    evidence,
    expected_reason,
):
    performance_evidence = (
        {} if evidence is None else {("node", "cupyx-gaussian-blur-v1"): evidence}
    )

    result = plan_compute_decisions(
        ComputeRequest(mode="auto", allow_experimental=True),
        (_workload(),),
        environment=_environment(),
        performance_evidence=performance_evidence,
    )

    assert result.decisions[0].reason is expected_reason


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


@pytest.mark.parametrize(
    "scientific_stack_versions",
    (
        (),
        (
            ("numpy", "2.5.1"),
            ("scipy", "1.17.0"),
            ("scikit-image", "0.26.0"),
        ),
    ),
    ids=("missing", "mismatched"),
)
def test_exact_public_gpu_pin_has_visible_cpu_fallback_for_unvalidated_cpu_stack(
    scientific_stack_versions,
):
    workload = _workload()
    result = plan_compute_decisions(
        _request("library:cupyx"),
        (workload,),
        environment=replace(
            _environment(),
            scientific_stack_versions=scientific_stack_versions,
        ),
        array_facts={"node": _facts(workload)},
    )

    decision = result.decisions[0]
    assert decision.decision_kind is DecisionKind.FALLBACK_CPU
    assert decision.fallback_reason is FallbackReason.ENVIRONMENT_UNSUPPORTED
    assert "validated authoritative CPU scientific stack" in decision.reason_text
    assert "CPU remains authoritative" in decision.reason_text
    assert result.warnings == (decision.reason_text,)


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


def test_unresolved_upstream_workload_defers_gpu_even_for_strict_preference():
    workload = replace(_workload(), inputs_resolved=False)

    result = plan_compute_decisions(
        _request("library:cupyx", fallback="strict"),
        (workload,),
        environment=_environment(),
    )

    decision = result.decisions[0]
    assert decision.decision_kind is DecisionKind.POLICY_CPU
    assert decision.reason is DecisionReason.WORKLOAD_UNSUPPORTED
    assert not decision.fallback_used
    assert "upstream output" in decision.reason_text


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


def test_invalid_axis_is_preflight_error_in_prefer_gpu_mode():
    workload = _workload(parameters=(("sigma", 1.2), ("channel_axis", True)))

    with pytest.raises(ComputePreflightError, match="channel"):
        plan_compute_decisions(
            ComputeRequest(mode="prefer_gpu"),
            (workload,),
            environment=_environment(),
            array_facts={"node": _facts(workload)},
        )


@pytest.mark.parametrize(
    ("workload", "compute_request", "environment", "reason"),
    (
        (
            _workload(dtype="uint16"),
            ComputeRequest(mode="prefer_gpu"),
            _environment(),
            DecisionReason.WORKLOAD_UNSUPPORTED,
        ),
        (
            _workload(),
            ComputeRequest(mode="prefer_gpu"),
            _environment(runtime=False),
            DecisionReason.ENVIRONMENT_UNSUPPORTED,
        ),
        (
            _workload(),
            ComputeRequest(
                mode="prefer_gpu",
                accelerator_memory_cap_bytes=1,
            ),
            _environment(),
            DecisionReason.MEMORY_LIMIT,
        ),
    ),
    ids=("unsupported-workload", "unavailable-environment", "memory-limit"),
)
def test_prefer_gpu_safe_rejections_are_policy_cpu_not_fallback(
    workload,
    compute_request,
    environment,
    reason,
):
    result = plan_compute_decisions(
        compute_request,
        (workload,),
        environment=environment,
        array_facts={"node": _facts(workload)},
    )

    decision = result.decisions[0]
    assert decision.decision_kind is DecisionKind.POLICY_CPU
    assert decision.reason is reason
    assert not decision.fallback_used
    assert decision.fallback_reason is FallbackReason.NONE
    assert result.warnings == ()


def test_memory_cap_rejects_forced_gpu_before_execution():
    workload = _workload()
    request = ComputeRequest(
        mode="custom",
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
    public = compute_specs_for("gaussian_blur", include_cpu=False)[0]
    hidden = replace(public, admission_tier=AdmissionTier.DEVELOPER_HIDDEN)
    registry = ComputeRegistry(implementation_specs=(hidden,))
    result = plan_compute_decisions(
        _request(
            "implementation:cupyx-gaussian-blur-v1",
            allow_experimental=False,
        ),
        (_workload(),),
        registry=registry,
        environment=_environment(),
    )

    assert result.decisions[0].fallback_used
    assert result.decisions[0].fallback_reason is FallbackReason.DEPENDENCY_UNAVAILABLE
    registry.close()


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
    real_environment = ComputeEnvironment
    monkeypatch.setattr(
        planning_module,
        "ComputeEnvironment",
        lambda: real_environment(
            os_name="Windows",
            execution_mode="native",
            python_implementation="CPython",
            python_version="3.12",
            python_abi="cpython-312",
        ),
    )
    registry = ComputeRegistry()
    scientific_stack = {
        "numpy": "2.5.1",
        "scipy": "1.18.0",
        "scikit-image": "0.26.0",
    }
    monkeypatch.setattr(
        compute_module.importlib.metadata,
        "version",
        scientific_stack.__getitem__,
    )
    runtime_fingerprint = "runtime-fingerprint-a"
    runtime_metadata = (
        ("cuda_runtime_version", "13020"),
        ("driver_version", "13030"),
    )
    device_metadata = (("compute_capability", "12.0"),)
    cucim_metadata = (
        ("environment_record_schema", "napari-vipp-gpu-environment"),
        ("environment_record_schema_version", "2"),
        ("environment_track", "cuda13"),
        ("cupy_distribution", "cupy-cuda13x"),
        ("cucim_distribution", "cucim-cu13"),
        ("cucim_distribution_version", "26.6.0"),
        ("cucim_artifact_sha256", "a" * 64),
        ("cucim_wheel_payload_sha256", PHASE1_CUCIM_WHEEL_PAYLOAD_SHA256),
        ("cucim_source_tag", PHASE1_CUCIM_SOURCE_TAG),
        ("cucim_source_commit", PHASE1_CUCIM_SOURCE_COMMIT),
        ("cucim_build_recipe_id", PHASE1_CUCIM_BUILD_RECIPE_ID),
    )

    def runtime_probe(_runtime_id):
        return RuntimeProbeResult(
            "cuda-cupy",
            True,
            version="14.1.1",
            devices=(
                RuntimeDevice(
                    "cuda:0",
                    "NVIDIA GeForce RTX 5090",
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
    assert dict(environment.scientific_stack_versions) == scientific_stack
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
