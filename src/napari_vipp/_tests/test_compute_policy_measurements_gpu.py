from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.compute import WorkloadDescriptor
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    FactCompleteness,
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for


def _spec(operation_id: str):
    (spec,) = compute_specs_for(operation_id, include_cpu=False)
    return spec


def _workload(
    operation_id: str,
    *,
    shape: tuple[int, ...] = (2, 10, 20),
    labels_dtype: str = "int32",
    intensity_dtype: str = "uint16",
    spatial_ndim: int = 2,
    parameters: tuple[tuple[str, object], ...] = (),
) -> WorkloadDescriptor:
    shapes = (shape, shape) if operation_id == "measure_objects_intensity" else (shape,)
    dtypes = (
        (labels_dtype, intensity_dtype)
        if operation_id == "measure_objects_intensity"
        else (labels_dtype,)
    )
    return WorkloadDescriptor(
        "measurements",
        operation_id,
        shapes,
        dtypes,
        parameters=(
            ("spatial_mode", "2D YX"),
            *parameters,
        ),
        resolved_spatial_ndim=spatial_ndim,
    )


def _facts(
    workload: WorkloadDescriptor,
    *,
    labels_minimum: int = 0,
    intensity_complete: bool = True,
    intensity_finite: bool = True,
) -> tuple[ArrayFacts, ...]:
    shape = workload.input_shapes[0]
    elements = int(np.prod(shape))
    values = [
        ArrayFacts(
            shape,
            workload.input_dtypes[0],
            elements,
            "labels-revision",
            completeness=FactCompleteness.COMPLETE,
            finite_count=elements,
            minimum=labels_minimum,
            maximum=7,
            guarantees=(("nonnegative",) if labels_minimum >= 0 else ()),
        )
    ]
    if len(workload.input_shapes) == 2:
        values.append(
            ArrayFacts(
                shape,
                workload.input_dtypes[1],
                elements,
                "intensity-revision",
                completeness=(
                    FactCompleteness.COMPLETE
                    if intensity_complete
                    else FactCompleteness.UNKNOWN
                ),
                finite_count=(
                    elements
                    if intensity_complete and intensity_finite
                    else elements - 1 if intensity_complete else None
                ),
            )
        )
    return tuple(values)


@pytest.mark.parametrize(
    ("operation_id", "implementation_id", "callable_name", "input_count"),
    (
        (
            "measure_objects",
            "cucim-measure-objects-basic-v1",
            "measure_objects",
            1,
        ),
        (
            "measure_objects_intensity",
            "cucim-measure-objects-intensity-basic-v1",
            "measure_objects_with_intensity",
            2,
        ),
    ),
)
def test_measurement_specs_are_public_typed_host_boundaries(
    operation_id,
    implementation_id,
    callable_name,
    input_count,
) -> None:
    spec = _spec(operation_id)

    assert spec.implementation_id == implementation_id
    assert spec.callable_ref.endswith(f":{callable_name}")
    assert spec.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE
    assert spec.implementation_library_id == "cucim"
    assert spec.supports_device_residency
    assert not spec.host_boundary
    assert len(spec.input_ports) == input_count
    assert spec.output_ports[0].value_kind.value == "table"
    assert spec.output_ports[0].internal_dtypes == ("float64",)
    assert spec.parity_policy_id == "basic-measurement-table-v1"
    assert spec.memory_model_id == "cucim-basic-measurements-memory-v1"
    assert spec.host_finalizer_ref == (
        "napari_vipp.core.measurements:finalize_basic_measurement_outputs"
    )
    validate_spec_policy_references(spec)


def test_morphology_requires_complete_nonnegative_label_facts() -> None:
    spec = _spec("measure_objects")
    workload = _workload("measure_objects")

    missing = evaluate_candidate_workload_support(spec, workload)
    supported = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(workload),
    )
    negative = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(workload, labels_minimum=-1),
    )

    assert not missing.supported and missing.requires_complete_facts
    assert supported.supported
    assert not negative.supported and not negative.fallback_allowed


@pytest.mark.parametrize("dtype", ("bool", "uint8", "uint16", "float32"))
def test_intensity_dtype_region_is_explicit(dtype: str) -> None:
    spec = _spec("measure_objects_intensity")
    workload = _workload("measure_objects_intensity", intensity_dtype=dtype)
    decision = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(workload),
    )

    assert decision.supported


def test_float32_intensity_requires_complete_finite_facts() -> None:
    spec = _spec("measure_objects_intensity")
    workload = _workload("measure_objects_intensity", intensity_dtype="float32")

    unknown = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(workload, intensity_complete=False),
    )
    nonfinite = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(workload, intensity_finite=False),
    )

    assert not unknown.supported and unknown.requires_complete_facts
    assert not nonfinite.supported and nonfinite.requires_complete_facts


@pytest.mark.parametrize(
    ("labels_dtype", "intensity_dtype"),
    (("int64", "uint16"), (">i4", "uint16"), ("int32", "float64"), ("int32", ">u2")),
)
def test_unpromoted_or_non_native_dtypes_fall_back_to_cpu(
    labels_dtype: str,
    intensity_dtype: str,
) -> None:
    spec = _spec("measure_objects_intensity")
    workload = _workload(
        "measure_objects_intensity",
        labels_dtype=labels_dtype,
        intensity_dtype=intensity_dtype,
    )
    decision = evaluate_candidate_workload_support(spec, workload)

    assert not decision.supported
    assert decision.fallback_allowed


@pytest.mark.parametrize("labels_dtype", ("bool", "float32", "complex64"))
def test_invalid_label_domains_do_not_claim_a_cpu_fallback(
    labels_dtype: str,
) -> None:
    spec = _spec("measure_objects")
    workload = _workload("measure_objects", labels_dtype=labels_dtype)

    decision = evaluate_candidate_workload_support(spec, workload)

    assert not decision.supported
    assert not decision.fallback_allowed
    assert "invalid for both CPU and GPU" in decision.reason_text


def test_extended_columns_and_mismatched_intensity_shape_are_not_admitted() -> None:
    spec = _spec("measure_objects_intensity")
    extended = _workload(
        "measure_objects_intensity",
        parameters=(("include_shape_descriptors", True),),
    )
    mismatched = WorkloadDescriptor(
        "measurements",
        "measure_objects_intensity",
        ((10, 20), (9, 20)),
        ("int32", "uint16"),
        parameters=(("spatial_mode", "2D YX"),),
        resolved_spatial_ndim=2,
    )

    extended_decision = evaluate_candidate_workload_support(
        spec,
        extended,
        array_facts=_facts(extended),
    )
    mismatch_decision = evaluate_candidate_workload_support(spec, mismatched)

    assert not extended_decision.supported
    assert extended_decision.fallback_allowed
    assert "Extended measurement columns" in extended_decision.reason_text
    assert not mismatch_decision.supported
    assert not mismatch_decision.fallback_allowed


@pytest.mark.parametrize(
    ("operation_id", "intensity_dtype", "packed_width", "block_multiplier"),
    (
        ("measure_objects", "uint16", 10, 128),
        ("measure_objects_intensity", "uint16", 15, 224),
    ),
)
def test_measurement_memory_model_covers_packed_boundary_and_active_block(
    operation_id: str,
    intensity_dtype: str,
    packed_width: int,
    block_multiplier: int,
) -> None:
    spec = _spec(operation_id)
    workload = _workload(operation_id, intensity_dtype=intensity_dtype)
    estimate = estimate_candidate_memory(spec, workload)
    elements = 2 * 10 * 20
    input_bytes = elements * 4
    if operation_id == "measure_objects_intensity":
        input_bytes += elements * 2
    output_bytes = elements * packed_width * 8
    workspace = input_bytes + (10 * 20 * block_multiplier) + output_bytes

    assert estimate.runtime_managed_peak_bytes == (
        input_bytes + output_bytes + workspace
    )
    assert estimate.runtime_managed_peak_bytes >= input_bytes + (2 * output_bytes)
    assert estimate.total_device_peak_bytes == estimate.runtime_managed_peak_bytes
    assert estimate.host_materialization_peak_bytes == output_bytes * 5
    assert estimate.uncertainty_bytes >= 64 * 1024**2
