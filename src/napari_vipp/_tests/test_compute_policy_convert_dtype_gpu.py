from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.compute import WorkloadDescriptor
from napari_vipp.core.compute_benchmark_adapter import operation_parity
from napari_vipp.core.compute_policy import (
    ValueDescriptor,
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    propagate_output_descriptors,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for

OPERATION_ID = "convert_dtype"
IMPLEMENTATION_ID = "cupyx-convert-dtype-preserve-f32-v1"


def _gpu_spec():
    (spec,) = compute_specs_for(OPERATION_ID, include_cpu=False)
    return spec


def _workload(
    *,
    dtype: str = "uint16",
    output_dtype: object = "float32",
    scaling: object = "preserve",
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        "convert",
        OPERATION_ID,
        ((3, 64, 80),),
        (dtype,),
        parameters=(("output_dtype", output_dtype), ("scaling", scaling)),
        resolved_spatial_ndim=2,
    )


def test_gpu_spec_declares_the_exact_lossless_resident_region() -> None:
    spec = _gpu_spec()

    assert spec.implementation_id == IMPLEMENTATION_ID
    assert spec.implementation_version == "1"
    assert spec.runtime_id == "cuda-cupy"
    assert spec.array_domain == "cuda-cupy"
    assert spec.implementation_library_id == "cupyx"
    assert spec.callable_ref == (
        "napari_vipp.core.gpu.cupy_convert_dtype:convert_dtype"
    )
    assert spec.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE
    assert spec.supported_spatial_ndims == (1, 2, 3)
    assert spec.supports_device_residency
    assert spec.parameter_policy_id == "convert-dtype-f32-preserve-parameters-v1"
    assert spec.workload_policy_id == "convert-dtype-u8-u16-to-f32-preserve-v1"
    assert spec.parity_policy_id == "array-bitwise-v1"
    assert spec.memory_model_id == "cupy-convert-dtype-memory-v1"
    assert spec.input_ports[0].public_dtypes == ("uint8", "uint16")
    assert spec.output_ports[0].public_dtypes == ("float32",)
    assert spec.output_ports[0].output_dtype_policy_id == "fixed:float32"
    validate_spec_policy_references(spec)

    with ComputeRegistry() as registry:
        assert registry.implementations_for_operation(
            OPERATION_ID,
            allow_experimental=False,
        ) == (spec,)


@pytest.mark.parametrize("dtype", ("uint8", "uint16"))
def test_lossless_integer_to_float32_preserve_is_admitted_without_fact_scan(
    dtype,
) -> None:
    decision = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(dtype=dtype),
    )

    assert decision.supported
    assert not decision.requires_complete_facts


@pytest.mark.parametrize(
    ("dtype", "output_dtype", "scaling"),
    (
        ("bool", "float32", "preserve"),
        ("float32", "float32", "preserve"),
        ("uint16", "uint8", "preserve"),
        ("uint16", "float32", "clip"),
        ("uint16", "float32", "rescale"),
        ("uint16", "bool", "rescale"),
    ),
)
def test_other_valid_conversions_remain_visible_cpu_fallbacks(
    dtype,
    output_dtype,
    scaling,
) -> None:
    decision = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(
            dtype=dtype,
            output_dtype=output_dtype,
            scaling=scaling,
        ),
    )

    assert not decision.supported
    assert decision.fallback_allowed
    assert not decision.requires_complete_facts
    assert "lossless" in decision.reason_text
    assert "CPU" in decision.reason_text


@pytest.mark.parametrize(
    ("output_dtype", "scaling", "message"),
    (
        ("int8", "preserve", "output_dtype must be"),
        ("float32", "automatic", "scaling must be"),
    ),
)
def test_invalid_authored_parameters_fail_instead_of_falling_back(
    output_dtype,
    scaling,
    message,
) -> None:
    decision = evaluate_candidate_workload_support(
        _gpu_spec(),
        _workload(output_dtype=output_dtype, scaling=scaling),
    )

    assert not decision.supported
    assert not decision.fallback_allowed
    assert message in decision.reason_text


def test_static_descriptor_and_memory_models_name_float32_output_exactly() -> None:
    spec = _gpu_spec()
    workload = _workload()
    descriptor = ValueDescriptor((3, 64, 80), "uint16")

    assert propagate_output_descriptors(spec, (descriptor,)) == (
        ValueDescriptor((3, 64, 80), "float32"),
    )

    estimate = estimate_candidate_memory(spec, workload)
    elements = 3 * 64 * 80
    expected_input = elements * np.dtype(np.uint16).itemsize
    expected_output = elements * np.dtype(np.float32).itemsize
    assert estimate.model_id == "cupy-convert-dtype-memory-v1"
    assert estimate.runtime_managed_peak_bytes == expected_input + expected_output
    assert estimate.total_device_peak_bytes == expected_input + expected_output
    assert estimate.host_materialization_peak_bytes == expected_output
    assert estimate.uncertainty_bytes == 8 * 1024**2


def test_production_benchmark_parity_is_bitwise_including_contract() -> None:
    reference = np.asarray([0.0, 1.0, 65_535.0], dtype=np.float32)

    assert operation_parity(OPERATION_ID, reference, reference.copy()).passed

    changed = reference.copy()
    changed[1] = np.nextafter(changed[1], np.float32(2.0))
    mismatch = operation_parity(OPERATION_ID, reference, changed)
    wrong_dtype = operation_parity(
        OPERATION_ID,
        reference,
        reference.astype(np.float64),
    )

    assert not mismatch.passed
    assert "bitwise mismatch" in mismatch.detail
    assert not wrong_dtype.passed
    assert "dtype differs" in wrong_dtype.detail
