from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.compute import (
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    NodeExecutionDecision,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_benchmark_adapter import operation_parity
from napari_vipp.core.compute_cache import (
    build_cached_node_compute_provenance,
    build_scientific_result_key,
    cached_node_provenance_matches,
    implementation_identity,
    required_scientific_dependency_ids,
)
from napari_vipp.core.compute_policy import (
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for

IMPLEMENTATION_ID = "cupyx-connected-components-v1"
OPERATION_ID = "label_connected_components"
MAXIMUM_SPATIAL_BLOCK_ELEMENTS = 2**31 - 2


def _gpu_spec():
    (spec,) = compute_specs_for(OPERATION_ID, include_cpu=False)
    return spec


def _workload(
    *,
    shape: tuple[int, ...] = (5, 64, 80),
    dtype: str = "bool",
    spatial_ndim: int | None = 2,
    spatial_mode: str = "2D YX",
    connectivity: str = "Full connectivity",
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        "labels",
        OPERATION_ID,
        (shape,),
        (dtype,),
        parameters=(
            ("spatial_mode", spatial_mode),
            ("connectivity", connectivity),
        ),
        resolved_spatial_ndim=spatial_ndim,
    )


def test_connected_components_spec_is_public_and_names_every_exact_contract() -> None:
    spec = _gpu_spec()

    assert spec.implementation_id == IMPLEMENTATION_ID
    assert spec.implementation_version == "1"
    assert spec.runtime_id == "cuda-cupy"
    assert spec.array_domain == "cuda-cupy"
    assert spec.implementation_library_id == "cupyx"
    assert spec.callable_ref == (
        "napari_vipp.core.gpu.cupy_connected_components:"
        "label_connected_components"
    )
    assert spec.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE
    assert spec.supported_spatial_ndims == (2, 3)
    assert spec.supports_device_residency
    assert spec.parameter_policy_id == "connected-components-parameters-v1"
    assert spec.workload_policy_id == "connected-components-bool-2d-3d-v1"
    assert spec.parity_policy_id == "labels-bitwise-int32-v1"
    assert spec.memory_model_id == "cupyx-connected-components-memory-v1"
    assert spec.input_ports[0].public_dtypes == ("bool",)
    assert spec.output_ports[0].public_dtypes == ("int32",)
    assert spec.output_ports[0].output_dtype_policy_id == "fixed:int32"
    assert spec.cache_equivalence_group == ""
    validate_spec_policy_references(spec)

    with ComputeRegistry() as registry:
        assert registry.implementations_for_operation(
            OPERATION_ID,
            allow_experimental=False,
        ) == (spec,)


@pytest.mark.parametrize(
    ("shape", "spatial_ndim", "spatial_mode"),
    (
        ((5, 64, 80), 2, "Auto from axes"),
        ((5, 64, 80), 2, "2D YX"),
        ((3, 17, 19, 23), 3, "3D ZYX"),
    ),
)
@pytest.mark.parametrize("connectivity", ("Face connected", "Full connectivity"))
def test_bool_2d_and_3d_policy_regions_need_no_array_fact_scan(
    shape,
    spatial_ndim,
    spatial_mode,
    connectivity,
) -> None:
    decision = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(
            shape=shape,
            spatial_ndim=spatial_ndim,
            spatial_mode=spatial_mode,
            connectivity=connectivity,
        ),
    )

    assert decision.supported
    assert not decision.requires_complete_facts


@pytest.mark.parametrize("dtype", ("uint8", "uint16", "int32", "float32"))
def test_numeric_nonzero_coercion_remains_a_visible_cpu_fallback(dtype) -> None:
    decision = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(dtype=dtype),
    )

    assert not decision.supported
    assert decision.fallback_allowed
    assert not decision.requires_complete_facts
    assert "boolean mask" in decision.reason_text
    assert "CPU" in decision.reason_text


@pytest.mark.parametrize(
    ("shape", "spatial_ndim", "spatial_mode", "fallback_allowed", "message"),
    (
        ((64, 80), 1, "Auto from axes", True, "unsupported"),
        ((64, 80), 2, "3D ZYX", False, "disagree"),
        ((3, 64, 80), 3, "2D YX", False, "disagree"),
        ((64, 80), 2, "not a mode", False, "disagree"),
        ((64, 80), 3, "3D ZYX", False, "exceeds the input rank"),
    ),
)
def test_rank_and_spatial_mode_contract_errors_fail_closed(
    shape,
    spatial_ndim,
    spatial_mode,
    fallback_allowed,
    message,
) -> None:
    decision = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(
            shape=shape,
            spatial_ndim=spatial_ndim,
            spatial_mode=spatial_mode,
        ),
    )

    assert not decision.supported
    assert decision.fallback_allowed is fallback_allowed
    assert message in decision.reason_text


def test_invalid_connectivity_fails_closed() -> None:
    decision = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(connectivity="edge connected"),
    )

    assert not decision.supported
    assert not decision.fallback_allowed
    assert "Face connected" in decision.reason_text
    assert "Full connectivity" in decision.reason_text


def test_spatial_block_int32_limit_is_an_exact_exclusive_boundary() -> None:
    admitted = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(shape=(1, MAXIMUM_SPATIAL_BLOCK_ELEMENTS - 1)),
    )
    rejected = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(shape=(1, MAXIMUM_SPATIAL_BLOCK_ELEMENTS)),
    )

    assert admitted.supported
    assert not rejected.supported
    assert rejected.fallback_allowed
    assert "fewer than 2,147,483,646" in rejected.reason_text


def test_memory_model_counts_full_arrays_and_one_active_spatial_block() -> None:
    spec = _gpu_spec()
    workload = _workload(shape=(5, 64, 80))
    estimate = estimate_candidate_memory(spec, workload)
    elements = 5 * 64 * 80
    block_elements = 64 * 80
    expected_runtime_peak = elements * (1 + 4) + block_elements * 7

    assert estimate.model_id == "cupyx-connected-components-memory-v1"
    assert estimate.runtime_managed_peak_bytes == expected_runtime_peak
    assert estimate.total_device_peak_bytes == expected_runtime_peak
    assert estimate.host_materialization_peak_bytes == elements * 4
    assert estimate.uncertainty_bytes == 8 * 1024**2

    one_block = estimate_candidate_memory(
        spec,
        replace(workload, input_shapes=((1, 64, 80),)),
    )
    additional_leading_blocks = (
        estimate.runtime_managed_peak_bytes
        - one_block.runtime_managed_peak_bytes
    )
    assert additional_leading_blocks == 4 * block_elements * (1 + 4)


def test_exact_label_parity_rejects_swapped_ids_dtype_and_shape() -> None:
    reference = np.array(
        [[0, 1, 1, 0], [2, 0, 0, 3], [2, 0, 3, 3]],
        dtype=np.int32,
    )
    assert operation_parity(OPERATION_ID, reference, reference.copy()).passed

    swapped = reference.copy()
    swapped[reference == 1] = 2
    swapped[reference == 2] = 1
    swapped_result = operation_parity(OPERATION_ID, reference, swapped)
    dtype_result = operation_parity(OPERATION_ID, reference, reference.astype(np.int64))
    shape_result = operation_parity(OPERATION_ID, reference, reference[:, :-1])

    assert not swapped_result.passed
    assert "bitwise mismatch" in swapped_result.detail
    assert not dtype_result.passed
    assert "dtype differs" in dtype_result.detail
    assert not shape_result.passed
    assert "shape differs" in shape_result.detail


def _result_key(spec, dependency_versions):
    return build_scientific_result_key(
        spec,
        output_port_index=0,
        output_contract_id="vipp-label-image-v1",
        public_parameters={
            "spatial_mode": "2D YX",
            "connectivity": "Face connected",
        },
        upstream_results=("source-mask-revision-v1",),
        dependency_versions=dependency_versions,
        result_contract_id="connected-components-exact-int32-v1",
        axis_grid_identity={"axes": ("t", "y", "x")},
    )


def test_cpu_and_cupyx_keep_distinct_scientific_cache_and_provenance_identity() -> None:
    cpu_spec, gpu_spec = compute_specs_for(OPERATION_ID)
    dependency_ids = set(required_scientific_dependency_ids(cpu_spec)) | set(
        required_scientific_dependency_ids(gpu_spec)
    )
    dependencies = {name: f"test:{name}:v1" for name in dependency_ids}
    cpu_key = _result_key(cpu_spec, dependencies)
    gpu_key = _result_key(gpu_spec, dependencies)

    assert cpu_key.digest != gpu_key.digest
    assert cpu_key.implementation_id == f"cpu-{OPERATION_ID}-v1"
    assert gpu_key.implementation_id == IMPLEMENTATION_ID
    assert implementation_identity(cpu_spec) != implementation_identity(gpu_spec)

    request = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={"labels": f"implementation:{IMPLEMENTATION_ID}"},
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        fallback_policy="strict",
    )
    decision = NodeExecutionDecision(
        node_id="labels",
        operation_id=OPERATION_ID,
        requested_preference=request.preference_for("labels"),
        runtime_id=gpu_spec.runtime_id,
        implementation_library_id=gpu_spec.implementation_library_id,
        implementation_id=gpu_spec.implementation_id,
        decision_kind=DecisionKind.SELECTED,
        reason=DecisionReason.SELECTED_IMPLEMENTATION,
        reason_text="Selected exact CuPyX labels.",
    )
    provenance = build_cached_node_compute_provenance(
        decision,
        request,
        scientific_context_fingerprint="connected-components-science-v1",
        implementation_spec=gpu_spec,
    )

    assert provenance.actual_implementation == implementation_identity(gpu_spec)
    assert provenance.actual_implementation != implementation_identity(cpu_spec)
    assert cached_node_provenance_matches(
        provenance,
        request=request,
        node_id="labels",
        operation_id=OPERATION_ID,
        scientific_context_fingerprint="connected-components-science-v1",
        implementation_specs=(cpu_spec, gpu_spec),
    )
