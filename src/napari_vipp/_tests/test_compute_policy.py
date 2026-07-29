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
    CUDA_CUPY_CUCIM_WINDOWS_ENVIRONMENT_POLICY_ID,
    CUDA_CUPY_WINDOWS_ENVIRONMENT_POLICY_ID,
    CUDA_ENVIRONMENT_POLICIES,
    ArrayFacts,
    ArrayFactsCache,
    ArrayFactsKey,
    FactCompleteness,
    PerformanceEvidence,
    ValueDescriptor,
    _evaluate_phase1_cuda_host_environment,
    estimate_candidate_memory,
    evaluate_auto_performance,
    evaluate_candidate_support,
    evaluate_candidate_workload_support,
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
        nonfinite_policy_id=("finite-only-v1" if finite_only else "cpu-reference-v1"),
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
        validated_environment_policy_id=CUDA_CUPY_WINDOWS_ENVIRONMENT_POLICY_ID,
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
        "execution_mode": "native",
        "python_implementation": "CPython",
        "python_version": "3.12",
        "python_abi": "cpython-312",
        "runtime_ids": ("cpu-numpy", "cuda-cupy"),
        "implementation_libraries": ("cpu", "cupyx"),
        "runtime_versions": (
            ("cuda-cupy", "14.1.1"),
            ("cupyx", "14.1.1"),
        ),
        "scientific_stack_versions": (
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
        "runtime_probe_fingerprints": (("cuda-cupy", "probe-fingerprint"),),
        "runtime_metadata": (
            (
                "cuda-cupy",
                (
                    ("cuda_runtime_version", "13020"),
                    ("driver_version", "13030"),
                ),
            ),
        ),
        "driver_version": "13030",
        "device_id": "cuda:0",
        "device_name": "NVIDIA GeForce RTX 5090",
        "device_class": "nvidia-cuda",
        "device_metadata": (("compute_capability", "12.0"),),
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
        array_facts=(_facts("sampled", completeness=FactCompleteness.SAMPLED),),
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
        ({"os_name": "Linux"}, "float32", False),
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


def test_provider_free_workload_gate_rejects_static_region_without_environment():
    decision = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(dtype="uint16"),
    )

    assert not decision.supported
    assert decision.reason is DecisionReason.WORKLOAD_UNSUPPORTED


@pytest.mark.parametrize(
    "environment_updates",
    [
        {"runtime_versions": (("cuda-cupy", "999"), ("cupyx", "14.1.1"))},
        {"runtime_versions": (("cuda-cupy", "14.1.1"), ("cupyx", "999"))},
        {"runtime_probe_fingerprints": ()},
        {
            "runtime_metadata": (
                (
                    "cuda-cupy",
                    (
                        ("cuda_runtime_version", "11080"),
                        ("driver_version", "13030"),
                    ),
                ),
            )
        },
        {
            "runtime_metadata": (
                (
                    "cuda-cupy",
                    (
                        ("cuda_runtime_version", "14000"),
                        ("driver_version", "14000"),
                    ),
                ),
            ),
            "driver_version": "14000",
        },
        {
            "runtime_metadata": (
                (
                    "cuda-cupy",
                    (
                        ("cuda_runtime_version", "CUDA 13.2"),
                        ("driver_version", "13030"),
                    ),
                ),
            )
        },
        {"driver_version": ""},
        {"driver_version": "13040"},
        {"device_metadata": ()},
        {"device_class": "host"},
        {"execution_mode": "wsl2"},
    ],
)
def test_cupyx_environment_policy_fails_closed_on_unproven_provenance(
    environment_updates,
):
    decision = evaluate_candidate_support(
        _gpu_spec(),
        _workload(),
        _cuda_environment(**environment_updates),
        allow_experimental=False,
    )

    assert not decision.supported
    assert decision.reason is DecisionReason.ENVIRONMENT_UNSUPPORTED


@pytest.mark.parametrize(
    ("environment_updates", "reason_fragment"),
    [
        (
            {
                "runtime_metadata": (
                    (
                        "cuda-cupy",
                        (
                            ("cuda_runtime_version", "12090"),
                            ("driver_version", "13030"),
                        ),
                    ),
                )
            },
            "cuda runtime api 13.2",
        ),
        (
            {
                "runtime_metadata": (
                    (
                        "cuda-cupy",
                        (
                            ("cuda_runtime_version", "13030"),
                            ("driver_version", "13030"),
                        ),
                    ),
                )
            },
            "cuda runtime api 13.2",
        ),
        (
            {
                "runtime_metadata": (
                    (
                        "cuda-cupy",
                        (
                            ("cuda_runtime_version", "13020"),
                            ("driver_version", "13040"),
                        ),
                    ),
                ),
                "driver_version": "13040",
            },
            "cuda driver api 13.3",
        ),
        ({"device_name": "NVIDIA GeForce RTX 5080"}, "rtx 5090"),
        (
            {"device_metadata": (("compute_capability", "8.9"),)},
            "compute capability 12.0",
        ),
    ],
    ids=(
        "cuda12",
        "other-cuda13-runtime",
        "other-driver",
        "secondary-device-same-vendor",
        "secondary-compute-capability",
    ),
)
def test_public_gpu_environment_is_exactly_the_recorded_host_region(
    environment_updates,
    reason_fragment,
):
    for allow_experimental in (False, True):
        decision = evaluate_candidate_support(
            _gpu_spec(),
            _workload(),
            _cuda_environment(**environment_updates),
            allow_experimental=allow_experimental,
        )

        assert not decision.supported
        assert decision.reason is DecisionReason.ENVIRONMENT_UNSUPPORTED
        assert reason_fragment in decision.reason_text.lower()
        assert "cpu remains authoritative" in decision.reason_text.lower()


@pytest.mark.parametrize(
    ("scientific_stack_versions", "reason_fragment"),
    [
        ((), "missing"),
        (
            (
                ("numpy", "2.4.0"),
                ("scipy", "1.18.0"),
                ("scikit-image", "0.26.0"),
            ),
            "numpy 2.4.0 (validated 2.5.1)",
        ),
    ],
    ids=("missing-metadata", "version-mismatch"),
)
def test_exact_gpu_environment_requires_validated_cpu_scientific_stack(
    scientific_stack_versions,
    reason_fragment,
):
    decision = evaluate_candidate_support(
        _gpu_spec(),
        _workload(),
        _cuda_environment(
            scientific_stack_versions=scientific_stack_versions,
        ),
        allow_experimental=False,
    )

    assert not decision.supported
    assert decision.reason is DecisionReason.ENVIRONMENT_UNSUPPORTED
    assert reason_fragment in decision.reason_text.lower()
    assert "cpu remains authoritative" in decision.reason_text.lower()


def test_every_public_phase1_gpu_spec_uses_the_shared_cpu_stack_gate():
    environment = _cuda_environment(
        scientific_stack_versions=(
            ("numpy", "2.5.0"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        )
    )
    public_specs = tuple(
        spec
        for spec in accelerator_compute_specs()
        if spec.admission_tier
        in {AdmissionTier.PUBLIC_SELECTIVE, AdmissionTier.PUBLIC_AUTO_CANDIDATE}
        and spec.validated_environment_policy_id in CUDA_ENVIRONMENT_POLICIES
    )

    assert public_specs
    for spec in public_specs:
        decision = _evaluate_phase1_cuda_host_environment(spec, environment)
        assert decision is not None, spec.implementation_id
        assert decision.reason is DecisionReason.ENVIRONMENT_UNSUPPORTED
        assert "numpy 2.5.0 (validated 2.5.1)" in decision.reason_text.lower()


def _cucim_environment(**updates):
    values = {
        "implementation_libraries": ("cpu", "cucim"),
        "runtime_versions": (
            ("cuda-cupy", "14.1.1"),
            ("cucim", "26.6.0"),
        ),
        "implementation_library_metadata": (
            (
                "cucim",
                (
                    (
                        "environment_record_schema",
                        "napari-vipp-gpu-environment",
                    ),
                    ("environment_record_schema_version", "1"),
                    ("environment_track", "cuda13"),
                    ("cupy_distribution", "cupy-cuda13x"),
                    ("cucim_distribution", "cucim-cu13"),
                    ("cucim_distribution_version", "26.6.0"),
                    (
                        "cucim_artifact_sha256",
                        "586d3443091eea67ce2c697be2c490ca51977a5dbdf894b9318b270977134cf8",
                    ),
                ),
            ),
        ),
    }
    values.update(updates)
    return _cuda_environment(**values)


def _background_workload():
    return WorkloadDescriptor(
        node_id="node-1",
        operation_id="rolling_ball_background",
        input_shapes=((31, 37),),
        input_dtypes=("float32",),
        parameters=(("radius", 5),),
        resolved_spatial_ndim=2,
    )


def _cucim_spec():
    spec = compute_specs_for(
        "rolling_ball_background",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    assert (
        spec.validated_environment_policy_id
        == CUDA_CUPY_CUCIM_WINDOWS_ENVIRONMENT_POLICY_ID
    )
    return spec


@pytest.mark.parametrize("version", ("26.6.0", "26.06.00"))
def test_cucim_environment_policy_accepts_only_the_approved_windows_artifact(
    version,
):
    metadata = dict(dict(_cucim_environment().implementation_library_metadata)["cucim"])
    metadata["cucim_distribution_version"] = version
    environment = _cucim_environment(
        runtime_versions=(("cuda-cupy", "14.1.1"), ("cucim", version)),
        implementation_library_metadata=(("cucim", tuple(metadata.items())),),
    )

    decision = evaluate_candidate_support(
        _cucim_spec(),
        _background_workload(),
        environment,
        allow_experimental=True,
    )

    assert decision.supported


@pytest.mark.parametrize(
    "environment",
    [
        _cucim_environment(
            runtime_versions=(("cuda-cupy", "14.1.1"), ("cucim", "0.0.0"))
        ),
        _cucim_environment(implementation_library_metadata=()),
        _cucim_environment(
            implementation_library_metadata=(
                (
                    "cucim",
                    (
                        ("environment_record_schema", "napari-vipp-gpu-environment"),
                        ("environment_record_schema_version", "1"),
                        ("environment_track", "cuda13"),
                        ("cupy_distribution", "cupy-cuda13x"),
                        ("cucim_distribution", "cucim-cu13"),
                        ("cucim_distribution_version", "26.6.0"),
                        ("cucim_artifact_sha256", "0" * 64),
                    ),
                ),
            )
        ),
        _cucim_environment(os_name="Linux"),
        _cucim_environment(os_name="Darwin"),
    ],
    ids=("version", "missing-metadata", "digest", "linux", "darwin"),
)
def test_cucim_environment_policy_rejects_unapproved_provenance(environment):
    decision = evaluate_candidate_support(
        _cucim_spec(),
        _background_workload(),
        environment,
        allow_experimental=True,
    )

    assert not decision.supported
    assert decision.reason is DecisionReason.ENVIRONMENT_UNSUPPORTED


def test_exact_environment_policies_reject_runtime_library_relabeling():
    relabeled_cucim = replace(
        _cucim_spec(),
        validated_environment_policy_id=CUDA_CUPY_WINDOWS_ENVIRONMENT_POLICY_ID,
    )
    relabeled_cupyx = replace(
        _gpu_spec(),
        validated_environment_policy_id=(CUDA_CUPY_CUCIM_WINDOWS_ENVIRONMENT_POLICY_ID),
    )

    for spec, workload, environment in (
        (
            relabeled_cucim,
            _background_workload(),
            _cucim_environment(
                runtime_versions=(("cuda-cupy", "14.1.1"),),
                implementation_library_metadata=(),
            ),
        ),
        (relabeled_cupyx, _workload(), _cuda_environment()),
    ):
        decision = evaluate_candidate_support(
            spec,
            workload,
            environment,
            allow_experimental=True,
        )

        assert not decision.supported
        assert decision.reason is DecisionReason.ENVIRONMENT_UNSUPPORTED
        assert "is bound to runtime/library" in decision.reason_text


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
    environment = _cucim_environment()

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


def _rl_workload(
    *,
    image_shape=(3, 64, 64),
    psf_shape=(9, 9),
    image_dtype="float32",
    psf_dtype="float32",
    spatial_ndim=2,
    iterations=25,
):
    return WorkloadDescriptor(
        "rl-node",
        "richardson_lucy_deconvolution",
        (image_shape, psf_shape),
        (image_dtype, psf_dtype),
        parameters=(
            ("spatial_mode", "2D YX" if spatial_ndim == 2 else "3D ZYX"),
            ("iterations", iterations),
            ("normalize_psf", True),
            ("clip_negative_input", True),
            ("clip_output_negative", True),
            ("preserve_input_scale", True),
            ("filter_epsilon", 1e-8),
        ),
        resolved_spatial_ndim=spatial_ndim,
    )


def _rl_facts(*, image_shape=(3, 64, 64), psf_shape=(9, 9)):
    return tuple(
        ArrayFacts(
            shape,
            "float32",
            int(np.prod(shape)),
            f"revision-{index}",
            completeness=FactCompleteness.COMPLETE,
            finite_count=int(np.prod(shape)),
            maximum=1.0,
        )
        for index, shape in enumerate((image_shape, psf_shape))
    )


def _rl_tv_workload(
    *,
    image_shape=(3, 64, 64),
    psf_shape=(9, 9),
    spatial_ndim=2,
    iterations=25,
):
    ordinary = _rl_workload(
        image_shape=image_shape,
        psf_shape=psf_shape,
        spatial_ndim=spatial_ndim,
        iterations=iterations,
    )
    parameters = dict(ordinary.parameters)
    parameters.update(
        {
            "tv_regularization": 0.002,
            "tv_epsilon": 1e-6,
            "filter_epsilon": 1e-12,
            "denominator_floor": 0.05,
        }
    )
    return replace(
        ordinary,
        node_id="rl-tv-node",
        operation_id="richardson_lucy_tv_deconvolution",
        parameters=tuple(parameters.items()),
    )


def test_richardson_lucy_requires_explicit_finite_float32_image_and_psf():
    spec = _builtin_spec("richardson_lucy_deconvolution")
    workload = _rl_workload()

    missing_facts = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=True,
    )
    accepted = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(),
    )
    integer_image = evaluate_candidate_support(
        spec,
        _rl_workload(image_dtype="uint16"),
        _cuda_environment(),
        allow_experimental=True,
    )

    assert not missing_facts.supported
    assert missing_facts.requires_complete_facts
    assert accepted.supported
    assert not integer_image.supported
    assert "Convert Dtype" in integer_image.reason_text


def test_richardson_lucy_rejects_invalid_psf_geometry_and_empty_mass():
    spec = _builtin_spec("richardson_lucy_deconvolution")
    oversized = evaluate_candidate_support(
        spec,
        _rl_workload(psf_shape=(65, 9)),
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(psf_shape=(65, 9)),
    )
    image_facts, psf_facts = _rl_facts()
    empty_psf = evaluate_candidate_support(
        spec,
        _rl_workload(),
        _cuda_environment(),
        allow_experimental=True,
        array_facts=(image_facts, replace(psf_facts, maximum=0.0)),
    )

    assert not oversized.supported
    assert "PSF extent" in oversized.reason_text
    assert not empty_psf.supported
    assert "positive mass" in empty_psf.reason_text


@pytest.mark.parametrize("epsilon", (1e-10, 1e-7, 1e-6))
def test_richardson_lucy_rejects_epsilon_outside_validated_point(epsilon):
    spec = _builtin_spec("richardson_lucy_deconvolution")

    outside_point = evaluate_candidate_support(
        spec,
        replace(
            _rl_workload(),
            parameters=tuple(
                (name, epsilon if name == "filter_epsilon" else value)
                for name, value in _rl_workload().parameters
            ),
        ),
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(),
    )

    assert not outside_point.supported
    assert outside_point.fallback_allowed
    assert "exactly 1e-08" in outside_point.reason_text
    assert "not monotonic" in outside_point.reason_text


def test_richardson_lucy_rejects_iterations_above_validated_parity_region():
    spec = _builtin_spec("richardson_lucy_deconvolution")

    too_many = evaluate_candidate_support(
        spec,
        _rl_workload(iterations=26),
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(),
    )

    assert not too_many.supported
    assert too_many.fallback_allowed
    assert "1 through 25" in too_many.reason_text
    assert "roundoff" in too_many.reason_text


def test_richardson_lucy_rejects_even_psf_and_nondefault_safety_options():
    spec = _builtin_spec("richardson_lucy_deconvolution")
    even_psf = evaluate_candidate_support(
        spec,
        _rl_workload(psf_shape=(8, 9)),
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(psf_shape=(8, 9)),
    )
    unsafe_options = evaluate_candidate_support(
        spec,
        replace(
            _rl_workload(),
            parameters=tuple(
                (name, False if name == "preserve_input_scale" else value)
                for name, value in _rl_workload().parameters
            ),
        ),
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(),
    )

    assert not even_psf.supported
    assert even_psf.fallback_allowed
    assert "odd PSF extents" in even_psf.reason_text
    assert not unsafe_options.supported
    assert unsafe_options.fallback_allowed
    assert "preserve_input_scale" in unsafe_options.reason_text


def test_richardson_lucy_projects_fixed_float32_output_and_conservative_memory():
    spec = _builtin_spec("richardson_lucy_deconvolution")
    workload = _rl_workload()
    projected = propagate_output_descriptors(
        spec,
        (
            ValueDescriptor((3, 64, 64), "float32"),
            ValueDescriptor((9, 9), "float32"),
        ),
    )
    estimate = estimate_candidate_memory(spec, workload)

    assert projected == (ValueDescriptor((3, 64, 64), "float32"),)
    assert estimate.model_id == "cupyx-richardson-lucy-fft-memory-v2"
    assert (
        estimate.total_device_peak_bytes
        > (np.prod((3, 64, 64)) + np.prod((9, 9))) * np.dtype(np.float32).itemsize
    )


def test_richardson_lucy_fft_memory_model_covers_padded_workspaces_and_first_use():
    spec = _builtin_spec("richardson_lucy_deconvolution")
    workload = _rl_workload(
        image_shape=(512, 512),
        psf_shape=(13, 13),
        iterations=10,
    )

    estimate = estimate_candidate_memory(spec, workload)

    # 524-pixel full convolution extents are conservatively padded to the next
    # 2/3/5-smooth real-FFT length (540). The exact assertion makes accidental
    # removal of complex buffers, plan workspaces, or first-use allowance loud.
    assert estimate.runtime_managed_peak_bytes == 22_419_028
    assert estimate.total_device_peak_bytes == 22_419_028
    assert estimate.uncertainty_bytes == 32 * 1024**2
    assert estimate.total_device_peak_bytes + estimate.uncertainty_bytes == 55_973_460


def test_richardson_lucy_fft_memory_model_scales_for_near_image_sized_3d_psf():
    spec = _builtin_spec("richardson_lucy_deconvolution")
    small = estimate_candidate_memory(
        spec,
        _rl_workload(
            image_shape=(32, 32, 32),
            psf_shape=(3, 3, 3),
            spatial_ndim=3,
        ),
    )
    large = estimate_candidate_memory(
        spec,
        _rl_workload(
            image_shape=(32, 32, 32),
            psf_shape=(31, 31, 31),
            spatial_ndim=3,
        ),
    )
    padded_real_elements = 64**3
    padded_complex_elements = 64 * 64 * (64 // 2 + 1)
    explicit_fft_array_bytes = (
        padded_real_elements * np.dtype(np.float32).itemsize
        + 3 * padded_complex_elements * np.dtype(np.complex64).itemsize
    )
    resident_bytes = (
        2 * np.prod((32, 32, 32)) + np.prod((31, 31, 31))
    ) * np.dtype(np.float32).itemsize

    assert large.total_device_peak_bytes > small.total_device_peak_bytes
    assert large.total_device_peak_bytes >= resident_bytes + explicit_fft_array_bytes


def test_richardson_lucy_tv_admits_only_the_finite_float32_default_profile():
    spec = _builtin_spec("richardson_lucy_tv_deconvolution")
    workload = _rl_tv_workload()

    accepted = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(),
    )
    missing_facts = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=True,
    )
    integer_image = evaluate_candidate_support(
        spec,
        replace(workload, input_dtypes=("uint16", "float32")),
        _cuda_environment(),
        allow_experimental=True,
    )

    assert accepted.supported
    assert not missing_facts.supported
    assert missing_facts.requires_complete_facts
    assert not integer_image.supported
    assert "Convert Dtype" in integer_image.reason_text


def test_richardson_lucy_tv_positive_profile_admits_only_measured_iterations():
    spec = _builtin_spec("richardson_lucy_tv_deconvolution")

    for iterations in (10, 25):
        decision = evaluate_candidate_support(
            spec,
            _rl_tv_workload(iterations=iterations),
            _cuda_environment(),
            allow_experimental=True,
            array_facts=_rl_facts(),
        )
        assert decision.supported

    for iterations in (1, 5, 11, 24):
        decision = evaluate_candidate_support(
            spec,
            _rl_tv_workload(iterations=iterations),
            _cuda_environment(),
            allow_experimental=True,
            array_facts=_rl_facts(),
        )
        assert not decision.supported
        assert decision.fallback_allowed
        assert "10, 25 iterations" in decision.reason_text


@pytest.mark.parametrize(
    ("name", "value", "text"),
    (
        ("tv_regularization", 0.008, "TV regularization"),
        ("tv_epsilon", 1e-3, "TV epsilon"),
        ("filter_epsilon", 1e-8, "filter epsilon"),
        ("denominator_floor", 0.15, "denominator floor"),
    ),
)
def test_richardson_lucy_tv_rejects_parameters_outside_initial_profile(
    name,
    value,
    text,
):
    spec = _builtin_spec("richardson_lucy_tv_deconvolution")
    workload = _rl_tv_workload()
    changed = replace(
        workload,
        parameters=tuple(
            (parameter, value if parameter == name else current)
            for parameter, current in workload.parameters
        ),
    )

    decision = evaluate_candidate_support(
        spec,
        changed,
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(),
    )

    assert not decision.supported
    assert decision.fallback_allowed
    assert text in decision.reason_text


def test_richardson_lucy_tv_rejects_singleton_gradient_axis_and_long_runs():
    spec = _builtin_spec("richardson_lucy_tv_deconvolution")
    singleton = _rl_tv_workload(image_shape=(3, 1, 64), psf_shape=(1, 9))
    singleton_decision = evaluate_candidate_support(
        spec,
        singleton,
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(image_shape=(3, 1, 64), psf_shape=(1, 9)),
    )
    long_run = evaluate_candidate_support(
        spec,
        _rl_tv_workload(iterations=26),
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(),
    )

    assert not singleton_decision.supported
    assert not singleton_decision.fallback_allowed
    assert "at least two samples" in singleton_decision.reason_text
    assert not long_run.supported
    assert long_run.fallback_allowed
    assert "1 through 25" in long_run.reason_text


def test_richardson_lucy_tv_lambda_zero_inherits_the_ordinary_rl_profile():
    spec = _builtin_spec("richardson_lucy_tv_deconvolution")
    workload = _rl_tv_workload()
    lambda_zero = replace(
        workload,
        parameters=tuple(
            (
                name,
                0.0
                if name == "tv_regularization"
                else 1e-8
                if name == "filter_epsilon"
                else 1e-3
                if name == "tv_epsilon"
                else 0.5
                if name == "denominator_floor"
                else value,
            )
            for name, value in workload.parameters
        ),
    )

    accepted = evaluate_candidate_support(
        spec,
        lambda_zero,
        _cuda_environment(),
        allow_experimental=True,
        array_facts=_rl_facts(),
    )
    tv_estimate = estimate_candidate_memory(spec, lambda_zero)
    ordinary_estimate = estimate_candidate_memory(
        _builtin_spec("richardson_lucy_deconvolution"),
        _rl_workload(),
    )

    assert accepted.supported
    assert (
        tv_estimate.runtime_managed_peak_bytes
        == ordinary_estimate.runtime_managed_peak_bytes
    )


def test_richardson_lucy_tv_projects_float32_and_reserves_tv_workspaces():
    tv_spec = _builtin_spec("richardson_lucy_tv_deconvolution")
    rl_spec = _builtin_spec("richardson_lucy_deconvolution")
    projected = propagate_output_descriptors(
        tv_spec,
        (
            ValueDescriptor((16, 64, 64), "float32"),
            ValueDescriptor((5, 9, 9), "float32"),
        ),
    )
    tv_estimate = estimate_candidate_memory(
        tv_spec,
        _rl_tv_workload(
            image_shape=(16, 64, 64),
            psf_shape=(5, 9, 9),
            spatial_ndim=3,
        ),
    )
    rl_estimate = estimate_candidate_memory(
        rl_spec,
        _rl_workload(
            image_shape=(16, 64, 64),
            psf_shape=(5, 9, 9),
            spatial_ndim=3,
        ),
    )

    assert projected == (ValueDescriptor((16, 64, 64), "float32"),)
    assert tv_estimate.model_id == "cupyx-richardson-lucy-tv-fft-memory-v1"
    assert tv_estimate.total_device_peak_bytes > rl_estimate.total_device_peak_bytes


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
