from __future__ import annotations

import pytest

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeRequest,
    DecisionReason,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import (
    ComputePreflightError,
    plan_compute_decisions,
)
from napari_vipp.core.compute_policy import (
    CUDA_CUPY_CORE_WINDOWS_ENVIRONMENT_POLICY_ID,
    estimate_candidate_memory,
    evaluate_candidate_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for


def _cuda_environment() -> ComputeEnvironment:
    return ComputeEnvironment(
        os_name="Windows",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupy", "cupyx"),
        runtime_versions=(
            ("cuda-cupy", "14.1.1"),
            ("cupy", "14.1.1"),
            ("cupyx", "14.1.1"),
        ),
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
        runtime_probe_fingerprints=(("cuda-cupy", "test-fingerprint"),),
        runtime_metadata=(
            (
                "cuda-cupy",
                (
                    ("cuda_runtime_version", "13020"),
                    ("driver_version", "13030"),
                ),
            ),
        ),
        driver_version="13030",
        device_id="cuda:0",
        device_name="NVIDIA GeForce RTX 5090",
        device_class="nvidia-cuda",
        device_metadata=(("compute_capability", "12.0"),),
        memory_topology="discrete",
        total_accelerator_memory_bytes=16 * 1024**3,
        probe_status="available",
    )


def _binary_workload(
    *,
    dtype: str = "float32",
    threshold: object = 0.5,
    channel_axis: object = None,
    shape: tuple[int, ...] = (31, 37),
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        "threshold",
        "binary_threshold",
        (shape,),
        (dtype,),
        parameters=(("channel_axis", channel_axis), ("threshold", threshold)),
        resolved_spatial_ndim=2,
    )


def _extract_workload(
    *,
    node_id: str = "extract",
    dtype: str = "uint16",
    channel: object = 1,
    axis_names: tuple[str, ...] = ("C", "Y", "X"),
    axis_types: tuple[str, ...] = ("channel", "space", "space"),
    predecessors: tuple[str, ...] = (),
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        node_id,
        "extract_channel",
        ((3, 31, 37),),
        (dtype,),
        parameters=(
            ("axis_names", axis_names),
            ("axis_types", axis_types),
            ("channel", channel),
        ),
        resolved_spatial_ndim=2,
        resident_predecessors=predecessors,
    )


def test_bridge_declarations_are_public_custom_resident_and_policy_complete():
    binary = compute_specs_for("binary_threshold", include_cpu=False)[0]
    extract = compute_specs_for("extract_channel", include_cpu=False)[0]

    assert binary.implementation_id == "cupy-binary-threshold-f32-exact-v1"
    assert extract.implementation_id == "cupy-extract-channel-view-v1"
    assert binary.admission_tier is AdmissionTier.PUBLIC_CUSTOM
    assert extract.admission_tier is AdmissionTier.PUBLIC_CUSTOM
    assert binary.validated_environment_policy_id == (
        CUDA_CUPY_CORE_WINDOWS_ENVIRONMENT_POLICY_ID
    )
    assert extract.validated_environment_policy_id == (
        CUDA_CUPY_CORE_WINDOWS_ENVIRONMENT_POLICY_ID
    )
    assert binary.supports_device_residency
    assert extract.supports_device_residency
    validate_spec_policy_references(binary)
    validate_spec_policy_references(extract)


def test_binary_threshold_admits_nonfinite_pixels_without_requiring_a_scan():
    spec = compute_specs_for("binary_threshold", include_cpu=False)[0]
    decision = evaluate_candidate_support(
        spec,
        _binary_workload(),
        _cuda_environment(),
        allow_experimental=False,
    )

    assert decision.supported
    assert not decision.requires_complete_facts


@pytest.mark.parametrize(
    "threshold",
    (5613.0001, 5613.375, 5613.9999, 17906.348),
)
def test_reported_decimal_thresholds_are_admitted_and_prefer_gpu_selects_cupy(
    threshold,
):
    spec = compute_specs_for("binary_threshold", include_cpu=False)[0]
    workload = _binary_workload(threshold=threshold)

    support = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=False,
    )
    result = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (workload,),
        environment=_cuda_environment(),
    )

    assert support.supported
    assert result.decisions[0].runtime_id == "cuda-cupy"
    assert result.decisions[0].implementation_id == (
        "cupy-binary-threshold-f32-exact-v1"
    )


def test_binary_threshold_admits_a_zero_dimensional_float32_scalar():
    spec = compute_specs_for("binary_threshold", include_cpu=False)[0]
    workload = WorkloadDescriptor(
        "threshold",
        "binary_threshold",
        ((),),
        ("float32",),
        parameters=(("channel_axis", None), ("threshold", 0.5)),
        resolved_spatial_ndim=None,
    )

    support = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=False,
    )

    assert support.supported


def test_non_native_float32_threshold_input_is_a_safe_cpu_region():
    spec = compute_specs_for("binary_threshold", include_cpu=False)[0]
    support = evaluate_candidate_support(
        spec,
        _binary_workload(dtype=">f4"),
        _cuda_environment(),
        allow_experimental=False,
    )

    assert not support.supported
    assert support.fallback_allowed
    assert "native-endian" in support.reason_text


@pytest.mark.parametrize("threshold", ("nan", "inf", "-inf"))
def test_nonfinite_authored_threshold_is_a_safe_visible_cpu_fallback(threshold):
    spec = compute_specs_for("binary_threshold", include_cpu=False)[0]
    support = evaluate_candidate_support(
        spec,
        _binary_workload(threshold=threshold),
        _cuda_environment(),
        allow_experimental=False,
    )

    assert not support.supported
    assert support.fallback_allowed
    result = plan_compute_decisions(
        ComputeRequest(
            mode="custom",
            node_preferences={"threshold": f"implementation:{spec.implementation_id}"},
            fallback_policy="visible",
        ),
        (_binary_workload(threshold=threshold),),
        environment=_cuda_environment(),
    )
    assert result.decisions[0].runtime_id == "cpu-numpy"
    assert result.decisions[0].reason is DecisionReason.VISIBLE_FALLBACK


@pytest.mark.parametrize(
    "workload",
    (
        _binary_workload(threshold="not-a-number"),
        _binary_workload(shape=(0, 31)),
    ),
)
def test_invalid_binary_authoring_never_hides_behind_cpu_fallback(workload):
    spec = compute_specs_for("binary_threshold", include_cpu=False)[0]
    support = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=False,
    )
    assert not support.supported
    assert not support.fallback_allowed

    with pytest.raises(ComputePreflightError):
        plan_compute_decisions(
            ComputeRequest(
                mode="custom",
                node_preferences={
                    "threshold": f"implementation:{spec.implementation_id}"
                },
                fallback_policy="visible",
            ),
            (workload,),
            environment=_cuda_environment(),
        )


def test_extract_channel_view_memory_retains_input_but_materializes_one_channel():
    spec = compute_specs_for("extract_channel", include_cpu=False)[0]
    workload = _extract_workload()
    estimate = estimate_candidate_memory(spec, workload)

    full_input_bytes = 3 * 31 * 37 * 2
    rounded_source_allocation_bytes = 7_168
    rounded_selected_staging_bytes = 2_560
    selected_channel_bytes = 31 * 37 * 2
    assert full_input_bytes == 6_882
    assert estimate.runtime_managed_peak_bytes == (
        rounded_source_allocation_bytes + rounded_selected_staging_bytes
    )
    assert estimate.total_device_peak_bytes == (
        rounded_source_allocation_bytes + rounded_selected_staging_bytes
    )
    assert estimate.host_materialization_peak_bytes == selected_channel_bytes
    assert estimate.uncertainty_bytes == 0


@pytest.mark.parametrize(
    "workload",
    (
        _extract_workload(dtype="float64"),
        _extract_workload(dtype=">u2"),
        _extract_workload(channel=-4),
        _extract_workload(axis_names=("Z", "Y", "X"), axis_types=("space",) * 3),
    ),
)
def test_extract_channel_distinguishes_cpu_regions_from_invalid_authoring(workload):
    spec = compute_specs_for("extract_channel", include_cpu=False)[0]
    support = evaluate_candidate_support(
        spec,
        workload,
        _cuda_environment(),
        allow_experimental=False,
    )

    assert not support.supported
    assert support.fallback_allowed is (
        workload.input_dtypes[0] in {"float64", ">u2"}
    )


def test_prefer_gpu_slices_on_host_before_uploading_a_multichannel_source():
    source = WorkloadDescriptor(
        "input",
        "input",
        (),
        (),
        resident_successors=("extract",),
    )
    extract = _extract_workload(predecessors=("input",))

    result = plan_compute_decisions(
        ComputeRequest(mode="prefer_gpu"),
        (source, extract),
        environment=_cuda_environment(),
    )
    decision = result.decisions_by_node["extract"]
    assert decision.runtime_id == "cpu-numpy"
    assert decision.reason is DecisionReason.PERFORMANCE_GATE
    assert "uploads only that selected channel" in decision.reason_text


def test_prefer_gpu_uses_extract_view_after_a_compatible_resident_predecessor():
    convert = WorkloadDescriptor(
        "convert",
        "convert_dtype",
        ((3, 31, 37),),
        ("uint16",),
        parameters=(("output_dtype", "float32"), ("scaling", "preserve")),
        resolved_spatial_ndim=2,
        resident_successors=("extract",),
    )
    extract = _extract_workload(
        dtype="float32",
        predecessors=("convert",),
    )

    with ComputeRegistry() as registry:
        result = plan_compute_decisions(
            ComputeRequest(mode="prefer_gpu"),
            (convert, extract),
            registry=registry,
            environment=_cuda_environment(),
        )
    assert result.decisions_by_node["convert"].runtime_id == "cuda-cupy"
    assert result.decisions_by_node["extract"].implementation_id == (
        "cupy-extract-channel-view-v1"
    )


def test_custom_exact_pin_can_explicitly_upload_the_full_multichannel_input():
    spec = compute_specs_for("extract_channel", include_cpu=False)[0]
    workload = _extract_workload(predecessors=("input",))
    result = plan_compute_decisions(
        ComputeRequest(
            mode="custom",
            node_preferences={"extract": f"implementation:{spec.implementation_id}"},
            fallback_policy="strict",
        ),
        (workload,),
        environment=_cuda_environment(),
    )

    assert result.decisions[0].implementation_id == spec.implementation_id


def test_auto_waits_for_promotion_evidence_for_both_bridge_implementations():
    binary = plan_compute_decisions(
        ComputeRequest(mode="auto"),
        (_binary_workload(),),
        environment=_cuda_environment(),
    )
    extract = plan_compute_decisions(
        ComputeRequest(mode="auto"),
        (_extract_workload(),),
        environment=_cuda_environment(),
    )

    assert binary.decisions[0].runtime_id == "cpu-numpy"
    assert extract.decisions[0].runtime_id == "cpu-numpy"
    assert all(
        decision.reason is DecisionReason.NO_VALIDATED_IMPLEMENTATION
        for decision in (binary.decisions[0], extract.decisions[0])
    )
