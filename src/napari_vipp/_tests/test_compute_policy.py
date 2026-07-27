from __future__ import annotations

from dataclasses import replace

import numpy as np
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
    estimate_candidate_memory,
    evaluate_auto_performance,
    evaluate_candidate_support,
    propagate_output_descriptors,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_specs import (
    AdmissionTier,
    accelerator_compute_specs,
    compute_specs_for,
)


def test_synthesized_cpu_spec_uses_registered_policies():
    validate_spec_policy_references(compute_specs_for("gaussian_blur")[0])


def test_all_builtin_accelerator_specs_use_versioned_registered_policies():
    for spec in accelerator_compute_specs():
        validate_spec_policy_references(spec)


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


def _builtin_spec(operation_id: str):
    return compute_specs_for(
        operation_id,
        include_cpu=False,
        allow_experimental=True,
    )[0]


def _operation_workload(
    operation_id: str,
    *,
    shape=(64, 64),
    dtype="float32",
    parameters=(),
    spatial_ndim=None,
):
    return WorkloadDescriptor(
        "node",
        operation_id,
        (shape,),
        (dtype,),
        parameters=parameters,
        resolved_spatial_ndim=spatial_ndim,
    )


def _operation_facts(
    *,
    shape=(64, 64),
    dtype="float32",
    guarantees=(),
):
    return (
        ArrayFacts(
            shape,
            dtype,
            int(np.prod(shape)),
            "revision",
            completeness=FactCompleteness.COMPLETE,
            finite_count=int(np.prod(shape)),
            guarantees=guarantees,
        ),
    )


def test_background_policy_uses_public_2d_and_conservative_3d_radius_bounds():
    spec = _builtin_spec("rolling_ball_background")
    environment = _cuda_environment(
        implementation_libraries=("cpu", "cupyx", "cucim")
    )

    two_dimensional = evaluate_candidate_support(
        spec,
        _operation_workload(
            "rolling_ball_background",
            parameters=(("radius", 500.0), ("spatial_mode", "2D YX")),
            spatial_ndim=2,
        ),
        environment,
        allow_experimental=True,
    )
    three_dimensional = evaluate_candidate_support(
        spec,
        _operation_workload(
            "rolling_ball_background",
            shape=(8, 16, 16),
            parameters=(("radius", 50.0), ("spatial_mode", "3D ZYX")),
            spatial_ndim=3,
        ),
        environment,
        allow_experimental=True,
    )
    too_large_3d = evaluate_candidate_support(
        spec,
        _operation_workload(
            "rolling_ball_background",
            shape=(8, 16, 16),
            parameters=(("radius", 51.0), ("spatial_mode", "3D ZYX")),
            spatial_ndim=3,
        ),
        environment,
        allow_experimental=True,
    )

    assert two_dimensional.supported
    assert three_dimensional.supported
    assert not too_large_3d.supported
    assert "1..50" in too_large_3d.reason_text


def test_median_float32_requires_complete_no_negative_zero_proof():
    spec = _builtin_spec("median_filter")
    workload = _operation_workload(
        "median_filter",
        shape=(51, 53),
        parameters=(("size", 51),),
        spatial_ndim=2,
    )

    missing = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=True,
    )
    proven = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_operation_facts(
            shape=(51, 53),
            guarantees=("no-negative-zero",),
        ),
    )

    assert not missing.supported
    assert missing.requires_complete_facts
    assert proven.supported


@pytest.mark.parametrize("dtype", ("uint8", "uint16"))
def test_median_integer_exact_matrix_is_admitted_without_value_scan(dtype):
    decision = evaluate_candidate_support(
        _builtin_spec("median_filter"),
        _operation_workload(
            "median_filter",
            shape=(51, 53),
            dtype=dtype,
            parameters=(("size", 51),),
            spatial_ndim=2,
        ),
        _cuda_environment(),
        allow_experimental=True,
    )

    assert decision.supported


def test_gaussian_advertises_full_public_float32_sigma_but_not_integer_or_float64():
    spec = _builtin_spec("gaussian_blur_3d")
    environment = _cuda_environment()
    float_workload = _operation_workload(
        "gaussian_blur_3d",
        shape=(5, 17, 19),
        parameters=(("sigma_z", 12.0), ("sigma_y", 0.0), ("sigma_x", 12.0)),
        spatial_ndim=3,
    )
    accepted = evaluate_candidate_support(
        spec,
        float_workload,
        environment,
        allow_experimental=True,
        array_facts=_operation_facts(shape=(5, 17, 19)),
    )

    assert accepted.supported
    for dtype in ("uint8", "uint16", "float64"):
        rejected = evaluate_candidate_support(
            spec,
            replace(float_workload, input_dtypes=(dtype,)),
            environment,
            allow_experimental=True,
        )
        assert not rejected.supported
        assert "CPU" in rejected.reason_text or "proven" in rejected.reason_text


def test_background_memory_model_scales_with_radius_and_spatial_rank():
    spec = _builtin_spec("rolling_ball_background")
    small = estimate_candidate_memory(
        spec,
        _operation_workload(
            "rolling_ball_background",
            shape=(16, 16),
            dtype="uint16",
            parameters=(("radius", 2.0), ("spatial_mode", "2D YX")),
            spatial_ndim=2,
        ),
    )
    wide = estimate_candidate_memory(
        spec,
        _operation_workload(
            "rolling_ball_background",
            shape=(16, 16),
            dtype="uint16",
            parameters=(("radius", 500.0), ("spatial_mode", "2D YX")),
            spatial_ndim=2,
        ),
    )
    volumetric = estimate_candidate_memory(
        spec,
        _operation_workload(
            "rolling_ball_background",
            shape=(4, 16, 16),
            dtype="uint16",
            parameters=(("radius", 50.0), ("spatial_mode", "3D ZYX")),
            spatial_ndim=3,
        ),
    )

    assert small.model_id == "cucim-background-memory-v1"
    assert wide.total_device_peak_bytes > small.total_device_peak_bytes
    assert volumetric.total_device_peak_bytes > small.total_device_peak_bytes
