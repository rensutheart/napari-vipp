from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp.core.compute import OutputPortKey, WorkloadDescriptor
from napari_vipp.core.compute_benchmark_adapter import (
    EXACT_MASK_PARITY_OPERATION_IDS,
    EXACT_PARITY_OPERATION_IDS,
    operation_parity,
)
from napari_vipp.core.compute_policy import (
    FILL_HOLES_WORKSPACE_BYTES_PER_PADDED_SPATIAL_ELEMENT,
    MASK_CLEANUP_MAXIMUM_SPATIAL_BLOCK_ELEMENTS,
    REMOVE_SMALL_OBJECTS_WORKSPACE_BYTES_PER_SPATIAL_ELEMENT,
    ArrayFacts,
    FactCompleteness,
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline

FILL_IMPLEMENTATION_ID = "cupyx-fill-holes-all-v1"
REMOVE_IMPLEMENTATION_ID = "cupyx-remove-small-objects-bool-v1"


def _gpu_spec(operation_id: str):
    (spec,) = compute_specs_for(operation_id, include_cpu=False)
    return spec


def _workload(
    operation_id: str,
    *,
    shape: tuple[int, ...] = (5, 64, 80),
    dtype: str = "bool",
    spatial_ndim: int | None = 2,
    spatial_mode: str = "2D YX",
    connectivity: str = "Face connected",
    size: object | None = None,
) -> WorkloadDescriptor:
    size_name = "max_hole_size" if operation_id == "fill_holes" else "min_size"
    default_size = 0 if operation_id == "fill_holes" else 10
    return WorkloadDescriptor(
        "cleanup",
        operation_id,
        (shape,),
        (dtype,),
        parameters=(
            (size_name, default_size if size is None else size),
            ("spatial_mode", spatial_mode),
            ("connectivity", connectivity),
        ),
        resolved_spatial_ndim=spatial_ndim,
    )


@pytest.mark.parametrize(
    (
        "operation_id",
        "implementation_id",
        "callable_ref",
        "parameter_policy_id",
        "workload_policy_id",
        "memory_model_id",
        "boundary_policy_id",
    ),
    (
        (
            "fill_holes",
            FILL_IMPLEMENTATION_ID,
            "napari_vipp.core.gpu.cupy_fill_holes:fill_holes",
            "fill-holes-all-parameters-v1",
            "fill-holes-bool-all-2d-3d-v1",
            "cupyx-fill-holes-memory-v1",
            "scipy-binary-fill-holes-connectivity-v1",
        ),
        (
            "remove_small_objects",
            REMOVE_IMPLEMENTATION_ID,
            (
                "napari_vipp.core.gpu.cupy_remove_small_objects:"
                "remove_small_objects"
            ),
            "remove-small-objects-bool-parameters-v1",
            "remove-small-objects-bool-2d-3d-v1",
            "cupyx-remove-small-objects-memory-v1",
            "scipy-connected-component-size-filter-v1",
        ),
    ),
)
def test_mask_cleanup_specs_are_exact_public_custom_resident_contracts(
    operation_id,
    implementation_id,
    callable_ref,
    parameter_policy_id,
    workload_policy_id,
    memory_model_id,
    boundary_policy_id,
) -> None:
    spec = _gpu_spec(operation_id)

    assert spec.implementation_id == implementation_id
    assert spec.implementation_version == "1"
    assert spec.runtime_id == "cuda-cupy"
    assert spec.array_domain == "cuda-cupy"
    assert spec.implementation_library_id == "cupyx"
    assert spec.callable_ref == callable_ref
    assert spec.admission_tier is AdmissionTier.PUBLIC_CUSTOM
    assert spec.parameter_policy_id == parameter_policy_id
    assert spec.workload_policy_id == workload_policy_id
    assert spec.parity_policy_id == "mask-bitwise-v1"
    assert spec.memory_model_id == memory_model_id
    assert spec.boundary_policy_id == boundary_policy_id
    assert spec.input_ports[0].boundary_policy_id == boundary_policy_id
    assert spec.output_ports[0].boundary_policy_id == boundary_policy_id
    assert spec.supported_spatial_ndims == (2, 3)
    assert spec.supports_device_residency
    assert spec.input_ports[0].public_dtypes == ("bool",)
    assert spec.output_ports[0].public_dtypes == ("bool",)
    assert spec.output_ports[0].output_dtype_policy_id == "fixed:bool"
    validate_spec_policy_references(spec)

    with ComputeRegistry() as registry:
        assert registry.implementations_for_operation(operation_id) == (spec,)


@pytest.mark.parametrize("operation_id", ("fill_holes", "remove_small_objects"))
@pytest.mark.parametrize(
    ("shape", "spatial_ndim", "spatial_mode"),
    (
        ((5, 64, 80), 2, "Auto from axes"),
        ((5, 64, 80), 2, "2D YX"),
        ((2, 9, 17, 19), 3, "3D ZYX volume"),
    ),
)
@pytest.mark.parametrize("connectivity", ("Face connected", "Full connectivity"))
def test_mask_cleanup_bool_2d_and_3d_regions_need_no_fact_scan(
    operation_id,
    shape,
    spatial_ndim,
    spatial_mode,
    connectivity,
) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec(operation_id),
        _workload(
            operation_id,
            shape=shape,
            spatial_ndim=spatial_ndim,
            spatial_mode=spatial_mode,
            connectivity=connectivity,
        ),
    )

    assert support.supported
    assert not support.requires_complete_facts


@pytest.mark.parametrize("size", (0, np.int64(0)))
def test_fill_holes_admits_only_canonical_fill_all_values(size) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec("fill_holes"),
        _workload("fill_holes", size=size),
    )

    assert support.supported


@pytest.mark.parametrize("size", (-1, 1, 50, np.int64(1000)))
def test_nonzero_fill_holes_size_is_a_safe_cpu_fallback(size) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec("fill_holes"),
        _workload("fill_holes", size=size),
    )

    assert not support.supported
    assert support.fallback_allowed
    assert "authored maximum hole size of 0" in support.reason_text


@pytest.mark.parametrize(
    "size",
    (0, 1, 37, -5, np.int64(200), 2**64, 2**100),
)
def test_remove_small_objects_admits_canonical_nonnegative_integer_sizes(size) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec("remove_small_objects"),
        _workload("remove_small_objects", size=size),
    )

    assert support.supported


@pytest.mark.parametrize("operation_id", ("fill_holes", "remove_small_objects"))
@pytest.mark.parametrize("size", (True, 1.5, "10", None))
def test_bad_cleanup_size_authoring_fails_closed(operation_id, size) -> None:
    workload = _workload(operation_id, size=size)
    # ``None`` means the helper's normal default, so replace it explicitly.
    if size is None:
        size_name = "max_hole_size" if operation_id == "fill_holes" else "min_size"
        workload = WorkloadDescriptor(
            workload.node_id,
            workload.operation_id,
            workload.input_shapes,
            workload.input_dtypes,
            parameters=(
                (size_name, None),
                ("spatial_mode", "2D YX"),
                ("connectivity", "Face connected"),
            ),
            resolved_spatial_ndim=2,
        )
    support = evaluate_candidate_workload_support(_gpu_spec(operation_id), workload)

    assert not support.supported
    assert not support.fallback_allowed
    assert "must be an integer" in support.reason_text


@pytest.mark.parametrize("dtype", ("uint8", "uint16", "int32"))
def test_numeric_fill_holes_remains_a_safe_cpu_coercion_region(dtype) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec("fill_holes"),
        _workload("fill_holes", dtype=dtype),
    )

    assert not support.supported
    assert support.fallback_allowed
    assert "boolean mask" in support.reason_text


@pytest.mark.parametrize("dtype", ("uint8", "uint16", "int32", "int64"))
def test_integer_label_cleanup_remains_a_safe_type_preserving_cpu_region(dtype) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec("remove_small_objects"),
        _workload("remove_small_objects", dtype=dtype),
    )

    assert not support.supported
    assert support.fallback_allowed
    assert "integer-label preservation" in support.reason_text


def test_non_mask_non_label_remove_small_objects_input_fails_closed() -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec("remove_small_objects"),
        _workload("remove_small_objects", dtype="float32"),
    )

    assert not support.supported
    assert not support.fallback_allowed


@pytest.mark.parametrize("operation_id", ("fill_holes", "remove_small_objects"))
@pytest.mark.parametrize(
    ("spatial_ndim", "spatial_mode", "connectivity"),
    (
        (1, "Auto from axes", "Face connected"),
        (2, "3D ZYX", "Face connected"),
        (3, "2D YX", "Face connected"),
        (2, "not a mode", "Face connected"),
        (2, "2D YX", "edge connected"),
    ),
)
def test_cleanup_spatial_contracts_fail_closed(
    operation_id,
    spatial_ndim,
    spatial_mode,
    connectivity,
) -> None:
    support = evaluate_candidate_workload_support(
        _gpu_spec(operation_id),
        _workload(
            operation_id,
            spatial_ndim=spatial_ndim,
            spatial_mode=spatial_mode,
            connectivity=connectivity,
        ),
    )

    assert not support.supported
    assert not support.fallback_allowed or spatial_ndim == 1


def test_fill_holes_int32_safety_uses_padded_spatial_shape() -> None:
    admitted = evaluate_candidate_workload_support(
        _gpu_spec("fill_holes"),
        _workload("fill_holes", shape=(46_338, 46_338)),
    )
    rejected = evaluate_candidate_workload_support(
        _gpu_spec("fill_holes"),
        _workload("fill_holes", shape=(46_339, 46_339)),
    )

    assert admitted.supported
    assert not rejected.supported
    assert rejected.fallback_allowed
    assert "padded spatial" in rejected.reason_text


def test_remove_small_objects_int32_safety_is_an_exclusive_raw_block_limit() -> None:
    admitted = evaluate_candidate_workload_support(
        _gpu_spec("remove_small_objects"),
        _workload(
            "remove_small_objects",
            shape=(1, MASK_CLEANUP_MAXIMUM_SPATIAL_BLOCK_ELEMENTS - 1),
        ),
    )
    rejected = evaluate_candidate_workload_support(
        _gpu_spec("remove_small_objects"),
        _workload(
            "remove_small_objects",
            shape=(1, MASK_CLEANUP_MAXIMUM_SPATIAL_BLOCK_ELEMENTS),
        ),
    )

    assert admitted.supported
    assert not rejected.supported
    assert rejected.fallback_allowed


@pytest.mark.parametrize(
    ("operation_id", "expected_workspace"),
    (
        (
            "fill_holes",
            66
            * 82
            * FILL_HOLES_WORKSPACE_BYTES_PER_PADDED_SPATIAL_ELEMENT,
        ),
        (
            "remove_small_objects",
            64 * 80 * REMOVE_SMALL_OBJECTS_WORKSPACE_BYTES_PER_SPATIAL_ELEMENT,
        ),
    ),
)
def test_cleanup_memory_models_count_full_arrays_and_largest_active_block(
    operation_id,
    expected_workspace,
) -> None:
    spec = _gpu_spec(operation_id)
    workload = _workload(operation_id)
    estimate = estimate_candidate_memory(spec, workload)
    elements = 5 * 64 * 80

    assert estimate.runtime_managed_peak_bytes == elements * 2 + expected_workspace
    assert estimate.total_device_peak_bytes == estimate.runtime_managed_peak_bytes
    assert estimate.host_materialization_peak_bytes == elements
    assert estimate.uncertainty_bytes == 8 * 1024**2

    one_block = estimate_candidate_memory(
        spec,
        _workload(operation_id, shape=(1, 64, 80)),
    )
    assert (
        estimate.runtime_managed_peak_bytes
        - one_block.runtime_managed_peak_bytes
        == 4 * 64 * 80 * 2
    )


@pytest.mark.parametrize("operation_id", ("fill_holes", "remove_small_objects"))
def test_cleanup_benchmark_parity_is_exact_mask_bitwise(operation_id) -> None:
    reference = np.array([[False, True], [True, False]], dtype=bool)
    mismatch = reference.copy()
    mismatch[0, 0] = True

    assert operation_id in EXACT_PARITY_OPERATION_IDS
    assert operation_id in EXACT_MASK_PARITY_OPERATION_IDS
    assert operation_parity(operation_id, reference, reference.copy()).passed
    assert not operation_parity(operation_id, reference, mismatch).passed
    assert not operation_parity(
        operation_id,
        reference,
        reference.astype(np.uint8),
    ).passed


@pytest.mark.parametrize("operation_id", ("fill_holes", "remove_small_objects"))
def test_cleanup_resident_metadata_and_bool_facts_are_projected_without_data(
    operation_id,
) -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    cleanup = pipeline.add_node(operation_id)
    pipeline.set_param(cleanup.id, "spatial_mode", "2D YX")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, cleanup.id).success

    mask = np.zeros((17, 19), dtype=bool)
    axes = (AxisMetadata("y", "space"), AxisMetadata("x", "space"))
    state = image_state_from_array(mask, axes=axes)
    assert state is not None
    call = pipeline.prepare_node_call(cleanup.id, (mask,), (state,))
    assert call is not None
    spec = _gpu_spec(operation_id)

    (predicted_state,) = execution_module._predict_device_node_states(
        pipeline,
        call,
        spec,
        (SimpleNamespace(shape=mask.shape, dtype=np.dtype(bool)),),
    )

    assert predicted_state is not None
    assert predicted_state.shape == mask.shape
    assert predicted_state.dtype == "bool"
    assert predicted_state.kind == "binary mask"
    assert tuple(axis.name for axis in predicted_state.axes) == ("y", "x")

    source_facts = ArrayFacts(
        mask.shape,
        "bool",
        mask.size,
        "resident-mask-revision-v1",
        completeness=FactCompleteness.UNKNOWN,
    )
    propagated = execution_module._propagate_shape_preserving_facts(
        operation_id,
        source_facts,
        dict(call.kwargs),
        output_port=OutputPortKey(cleanup.id, 0),
        output_shape=mask.shape,
        output_dtype="bool",
    )

    assert propagated is not None
    assert propagated.completeness is FactCompleteness.COMPLETE
    assert propagated.finite_count == mask.size
    assert {"nonnegative", "no-negative-zero"} <= set(propagated.guarantees)


def test_remove_small_objects_preserves_mask_and_label_metadata_kinds() -> None:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    mask_cleanup = pipeline.add_node("remove_small_objects")
    components = pipeline.add_node("label_connected_components")
    label_cleanup = pipeline.add_node("remove_small_objects")
    for node in (mask_cleanup, components, label_cleanup):
        pipeline.set_param(node.id, "spatial_mode", "2D YX")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, mask_cleanup.id).success
    assert pipeline.connect(threshold.id, components.id).success
    assert pipeline.connect(components.id, label_cleanup.id).success

    data = np.zeros((11, 13), dtype=np.uint16)
    data[2:7, 3:9] = 10
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.run(data, input_metadata={"axes": "YX"})

    assert pipeline.output_states[mask_cleanup.id].kind == "binary mask"
    assert pipeline.output_states[label_cleanup.id].kind == "label image"
    assert pipeline.output_states[label_cleanup.id].dtype == "int32"

    label_facts = ArrayFacts(
        data.shape,
        "int32",
        data.size,
        "labels-revision-v1",
        completeness=FactCompleteness.UNKNOWN,
    )
    propagated = execution_module._propagate_shape_preserving_facts(
        "remove_small_objects",
        label_facts,
        {"min_size": 10},
        output_port=OutputPortKey(label_cleanup.id, 0),
        output_shape=data.shape,
        output_dtype="int32",
    )
    assert propagated is not None
    assert {"integer-labels", "nonnegative", "no-negative-zero"} <= set(
        propagated.guarantees
    )
