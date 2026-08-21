from __future__ import annotations

import math

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp.core.compute import WorkloadDescriptor
from napari_vipp.core.compute_policy import (
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.remove_outliers import (
    BACKGROUND_OUTLIERS,
    FOREGROUND_OUTLIERS,
)

IMPLEMENTATION_ID = "cupy-remove-binary-outliers-v1"


def _gpu_spec():
    (spec,) = compute_specs_for("remove_binary_outliers", include_cpu=False)
    return spec


def _workload(
    *,
    shape: tuple[int, ...] = (3, 64, 80),
    dtype: str = "bool",
    radius: object = 2.0,
    which_outliers: object = FOREGROUND_OUTLIERS,
    spatial_ndim: int | None = None,
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        "outliers",
        "remove_binary_outliers",
        (shape,),
        (dtype,),
        parameters=(
            ("radius", radius),
            ("which_outliers", which_outliers),
        ),
        resolved_spatial_ndim=spatial_ndim,
    )


def test_spec_is_exact_public_custom_resident_cupy_contract() -> None:
    spec = _gpu_spec()

    assert spec.implementation_id == IMPLEMENTATION_ID
    assert spec.implementation_version == "1"
    assert spec.runtime_id == "cuda-cupy"
    assert spec.array_domain == "cuda-cupy"
    assert spec.implementation_library_id == "cupy"
    assert spec.callable_ref == (
        "napari_vipp.core.gpu.cupy_remove_binary_outliers:remove_binary_outliers"
    )
    assert spec.admission_tier is AdmissionTier.PUBLIC_CUSTOM
    assert spec.validated_environment_policy_id == (
        "cuda-cupy-14.1.1-rawkernel-cpython312-windows-native-v1"
    )
    assert spec.parameter_policy_id == "remove-binary-outliers-parameters-v1"
    assert spec.workload_policy_id == "remove-binary-outliers-bool-yx-v1"
    assert spec.parity_policy_id == "mask-bitwise-v1"
    assert spec.memory_model_id == "cupy-remove-binary-outliers-memory-v1"
    assert spec.boundary_policy_id == "imagej-rankfilters-nearest-yx-v1"
    assert spec.progress_policy_id == (
        "remove-binary-outliers-pixel-tile-sync-progress-v1"
    )
    assert spec.cancellation_policy_id == (
        "remove-binary-outliers-pixel-tile-boundary-cancel-v1"
    )
    assert spec.supported_spatial_ndims == (2,)
    assert spec.supports_device_residency
    assert spec.input_ports[0].public_dtypes == ("bool",)
    assert spec.input_ports[0].accumulation_dtype == "uint32"
    assert spec.output_ports[0].public_dtypes == ("bool",)
    assert spec.output_ports[0].output_dtype_policy_id == "fixed:bool"
    validate_spec_policy_references(spec)

    with ComputeRegistry() as registry:
        assert registry.implementations_for_operation("remove_binary_outliers") == (
            spec,
        )


@pytest.mark.parametrize(
    "shape",
    ((17, 19), (5, 17, 19), (2, 3, 17, 19)),
)
@pytest.mark.parametrize("spatial_ndim", (None, 2))
@pytest.mark.parametrize(
    "which_outliers",
    (FOREGROUND_OUTLIERS, BACKGROUND_OUTLIERS),
)
@pytest.mark.parametrize("radius", (0.5, 1.5, 1.75, 2.5, 2.85, 25.0))
def test_bool_trailing_yx_region_is_admitted_without_fact_scan(
    shape,
    spatial_ndim,
    which_outliers,
    radius,
) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(
            shape=shape,
            spatial_ndim=spatial_ndim,
            which_outliers=which_outliers,
            radius=radius,
        ),
    )

    assert support.supported
    assert not support.requires_complete_facts


@pytest.mark.parametrize("shape", ((5, 17, 19), (2, 3, 17, 19)))
def test_unresolved_zyx_and_tzyx_workloads_statically_keep_gpu_candidate(shape) -> None:
    workload = _workload(shape=shape, spatial_ndim=None)
    spec = _gpu_spec()

    assert execution_module._candidate_statically_matches(spec, workload)
    assert evaluate_candidate_workload_support(spec, workload).supported


@pytest.mark.parametrize("shape", ((5, 17, 19), (2, 3, 17, 19)))
def test_prepared_stack_call_preserves_positional_yx_gpu_eligibility(shape) -> None:
    source = np.zeros(shape, dtype=bool)
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    node = pipeline.add_node("remove_binary_outliers")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, node.id).success

    call = pipeline.prepare_node_call(
        node.id,
        (source,),
        (image_state_from_array(source),),
    )
    assert call is not None
    assert "resolved_spatial_ndim" not in call.kwargs
    workload = _workload(
        shape=shape,
        radius=call.kwargs["radius"],
        which_outliers=call.kwargs["which_outliers"],
        spatial_ndim=None,
    )

    spec = _gpu_spec()
    assert execution_module._candidate_statically_matches(spec, workload)
    assert evaluate_candidate_workload_support(spec, workload).supported


@pytest.mark.parametrize("radius", (25.0001, 50.0, 100.0))
def test_valid_radius_above_public_gpu_cap_falls_back_to_cpu(radius) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(radius=radius),
    )

    assert not support.supported
    assert support.fallback_allowed
    assert "qualified through radius 25" in support.reason_text


@pytest.mark.parametrize("radius", (True, 0.49, 100.01))
def test_invalid_radius_fails_closed(radius) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(radius=radius),
    )

    assert not support.supported
    assert not support.fallback_allowed
    assert "between 0.5 and 100" in support.reason_text


@pytest.mark.parametrize("radius", (math.nan, math.inf, -math.inf))
def test_nonfinite_radius_is_rejected_by_canonical_workload_descriptor(radius) -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        _workload(radius=radius)


def test_canonical_uint8_region_is_safe_cpu_fallback() -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(dtype="uint8"),
    )

    assert not support.supported
    assert support.fallback_allowed
    assert "canonical uint8 validation" in support.reason_text


@pytest.mark.parametrize(
    ("workload", "reason"),
    (
        (_workload(dtype="float32"), "bool or canonical uint8"),
        (_workload(shape=(17,)), "two trailing YX"),
        (_workload(shape=(0, 17)), "empty masks"),
        (_workload(which_outliers="Both"), "outlier type"),
    ),
)
def test_invalid_semantic_regions_fail_closed(workload, reason) -> None:
    support = evaluate_candidate_workload_support(_gpu_spec(), workload)

    assert not support.supported
    assert not support.fallback_allowed
    assert reason in support.reason_text


def test_conflicting_resolved_3d_rank_is_a_safe_cpu_fallback() -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(spatial_ndim=3),
    )

    assert not support.supported
    assert support.fallback_allowed
    assert "spatial dimensionality" in support.reason_text


@pytest.mark.parametrize(
    ("radius", "row_table_bytes"),
    ((0.5, 512), (25.0, 512), (100.0, 1024)),
)
def test_memory_model_counts_input_output_contiguous_staging_and_row_table(
    radius,
    row_table_bytes,
) -> None:
    workload = _workload(shape=(2, 3, 17, 19), radius=radius)
    estimate = estimate_candidate_memory(_gpu_spec(), workload)
    elements = math.prod(workload.input_shapes[0])

    assert estimate.runtime_managed_peak_bytes == 3 * elements + row_table_bytes
    assert estimate.total_device_peak_bytes == estimate.runtime_managed_peak_bytes
    assert estimate.host_materialization_peak_bytes == elements
    assert estimate.model_id == "cupy-remove-binary-outliers-memory-v1"
