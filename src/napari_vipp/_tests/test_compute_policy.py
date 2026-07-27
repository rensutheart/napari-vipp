from __future__ import annotations

from dataclasses import replace

import pytest

from napari_vipp.core.compute import (
    ComputeEnvironment,
    DecisionReason,
    OutputPortKey,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    ArrayFactsCache,
    ArrayFactsKey,
    FactCompleteness,
    PerformanceEvidence,
    ValueDescriptor,
    evaluate_auto_performance,
    evaluate_candidate_support,
    propagate_output_descriptors,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for


def test_synthesized_cpu_spec_uses_registered_policies():
    validate_spec_policy_references(compute_specs_for("gaussian_blur")[0])


def test_unknown_policy_reference_fails_declaration_validation():
    cpu_spec = compute_specs_for("median_filter")[0]

    with pytest.raises(ValueError, match="unknown parity policy"):
        validate_spec_policy_references(
            replace(cpu_spec, parity_policy_id="missing-policy-v1")
        )


def _gpu_spec(*, finite_only: bool = False):
    cpu_spec = compute_specs_for("gaussian_blur")[0]
    input_port = replace(
        cpu_spec.input_ports[0],
        public_dtypes=("float32",),
        nonfinite_policy_id=(
            "finite-only-v1" if finite_only else "cpu-reference-v1"
        ),
    )
    output_port = replace(
        cpu_spec.output_ports[0],
        output_dtype_policy_id="dtype-same-v1",
    )
    return replace(
        cpu_spec,
        implementation_id="cupyx-gaussian-v1",
        runtime_id="cuda-cupy",
        array_domain="cuda-cupy",
        implementation_library_id="cupyx",
        admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE,
        validated_environment_policy_id="cuda-cupy-py312-windows-linux-v1",
        input_ports=(input_port,),
        output_ports=(output_port,),
        shape_policy_id="shape-preserving-v1",
        supports_device_residency=True,
    )


def _workload(*, dtype: str = "float32", spatial_ndim: int = 2):
    return WorkloadDescriptor(
        node_id="node-1",
        operation_id="gaussian_blur",
        input_shapes=((64, 64),),
        input_dtypes=(dtype,),
        resolved_spatial_ndim=spatial_ndim,
    )


def _cuda_environment(**updates):
    values = {
        "os_name": "Windows",
        "python_implementation": "CPython",
        "python_version": "3.12",
        "python_abi": "cpython-312",
        "runtime_ids": ("cpu-numpy", "cuda-cupy"),
        "implementation_libraries": ("cpu", "cupyx"),
        "probe_status": "available",
    }
    values.update(updates)
    return ComputeEnvironment(**values)


def _facts(
    revision: str,
    *,
    completeness: FactCompleteness = FactCompleteness.COMPLETE,
    finite_count: int = 4096,
):
    return ArrayFacts(
        shape=(64, 64),
        dtype="float32",
        element_count=4096,
        revision_fingerprint=revision,
        completeness=completeness,
        finite_count=finite_count,
        strides=(256, 4),
        contiguous=True,
    )


def test_array_facts_are_revision_keyed_and_validate_layout():
    port = OutputPortKey("source", 0)
    first_key = ArrayFactsKey(port, "revision-1")
    second_key = ArrayFactsKey(port, "revision-2")
    cache = ArrayFactsCache()

    cache.put(first_key, _facts("revision-1"))
    cache.put(second_key, _facts("revision-2"))

    assert cache.get(first_key) is None
    assert cache.get(second_key) == _facts("revision-2")
    with pytest.raises(ValueError, match="cache key"):
        cache.put(first_key, _facts("other-revision"))
    with pytest.raises(ValueError, match="one integer"):
        replace(_facts("revision-3"), strides=(4,))


def test_sampled_facts_never_prove_a_finite_only_scientific_region():
    spec = _gpu_spec(finite_only=True)

    sampled = evaluate_candidate_support(
        spec,
        _workload(),
        _cuda_environment(),
        allow_experimental=False,
        array_facts=(
            _facts("sampled", completeness=FactCompleteness.SAMPLED),
        ),
    )
    complete = evaluate_candidate_support(
        spec,
        _workload(),
        _cuda_environment(),
        allow_experimental=False,
        array_facts=(_facts("complete"),),
    )

    assert not sampled.supported
    assert sampled.requires_complete_facts
    assert sampled.reason is DecisionReason.WORKLOAD_UNSUPPORTED
    assert complete.supported


@pytest.mark.parametrize(
    ("environment_updates", "dtype", "supported"),
    [
        ({}, "float32", True),
        ({"os_name": "Darwin"}, "float32", False),
        ({"python_version": "3.13"}, "float32", False),
        ({"probe_status": "failed"}, "float32", False),
        ({}, "uint16", False),
    ],
)
def test_support_evaluation_is_conservative_outside_the_validated_matrix(
    environment_updates,
    dtype,
    supported,
):
    decision = evaluate_candidate_support(
        _gpu_spec(),
        _workload(dtype=dtype),
        _cuda_environment(**environment_updates),
        allow_experimental=False,
    )

    assert decision.supported is supported


def test_shape_preserving_policy_propagates_schema_dtype_and_guarantees():
    outputs = propagate_output_descriptors(
        _gpu_spec(),
        (ValueDescriptor((4, 8), "float32", guarantees=("finite",)),),
    )

    assert outputs == (
        ValueDescriptor(
            (4, 8),
            "float32",
            _gpu_spec().output_ports[0].schema_id,
            ("finite",),
        ),
    )


def test_auto_performance_requires_confidence_and_absolute_saving():
    accepted = evaluate_auto_performance(
        PerformanceEvidence(
            cpu_seconds=0.120,
            candidate_seconds=0.075,
            transfer_seconds=0.005,
            lower_confidence_speedup=1.25,
        )
    )
    low_confidence = evaluate_auto_performance(
        PerformanceEvidence(
            cpu_seconds=0.120,
            candidate_seconds=0.075,
            lower_confidence_speedup=1.19,
        )
    )
    small_saving = evaluate_auto_performance(
        PerformanceEvidence(
            cpu_seconds=0.100,
            candidate_seconds=0.081,
            lower_confidence_speedup=1.25,
        )
    )

    assert accepted.select_candidate
    assert not low_confidence.select_candidate
    assert not small_saving.select_candidate


def test_local_performance_uses_the_greater_of_five_percent_or_ten_ms():
    within_noise = evaluate_auto_performance(
        PerformanceEvidence(0.100, 0.091, local_benchmark=True)
    )
    clear_short_win = evaluate_auto_performance(
        PerformanceEvidence(0.100, 0.089, local_benchmark=True)
    )
    clear_long_win = evaluate_auto_performance(
        PerformanceEvidence(1.000, 0.949, local_benchmark=True)
    )

    assert not within_noise.select_candidate
    assert clear_short_win.select_candidate
    assert clear_long_win.select_candidate
