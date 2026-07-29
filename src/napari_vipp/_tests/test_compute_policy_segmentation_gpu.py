from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.compute import DecisionReason, WorkloadDescriptor
from napari_vipp.core.compute_policy import (
    ArrayFacts,
    FactCompleteness,
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    evaluate_memory_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_specs import compute_specs_for


def _spec(operation_id: str):
    return compute_specs_for(
        operation_id,
        include_cpu=False,
        allow_experimental=True,
    )[0]


def _workload(
    operation_id: str,
    *,
    shape=(4, 64, 64),
    dtype="float32",
    parameters=(),
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        "segmentation-node",
        operation_id,
        (shape,),
        (dtype,),
        parameters=parameters,
        resolved_spatial_ndim=2,
    )


def _facts(
    *,
    shape=(4, 64, 64),
    dtype="float32",
    completeness=FactCompleteness.COMPLETE,
    finite_count: int | None = None,
    minimum=None,
    maximum=None,
) -> tuple[ArrayFacts, ...]:
    elements = int(np.prod(shape))
    return (
        ArrayFacts(
            shape,
            dtype,
            elements,
            "segmentation-revision",
            completeness=completeness,
            finite_count=elements if finite_count is None else finite_count,
            minimum=minimum,
            maximum=maximum,
        ),
    )


@pytest.mark.parametrize("operation_id", ("canny_edges", "otsu_threshold"))
def test_segmentation_gpu_specs_reference_registered_policy_ids(operation_id):
    validate_spec_policy_references(_spec(operation_id))


@pytest.mark.parametrize("dtype", ("bool", "uint8", "uint16"))
def test_canny_intrinsically_finite_integer_and_bool_inputs_need_no_value_scan(dtype):
    decision = evaluate_candidate_workload_support(
        _spec("canny_edges"),
        _workload("canny_edges", dtype=dtype),
    )

    assert decision.supported


def test_canny_float32_falls_back_before_scanning_for_subnormal_risk():
    spec = _spec("canny_edges")
    workload = _workload("canny_edges")

    missing = evaluate_candidate_workload_support(spec, workload)
    sampled = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(completeness=FactCompleteness.SAMPLED),
    )
    nonfinite = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(finite_count=int(np.prod((4, 64, 64))) - 1),
    )
    complete = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(),
    )

    for decision in (missing, sampled, nonfinite, complete):
        assert not decision.supported
        assert decision.fallback_allowed
        assert not decision.requires_complete_facts
        assert "subnormal" in decision.reason_text


def test_canny_extreme_finite_float32_facts_keep_visible_cpu_fallback():
    maximum = np.finfo(np.float32).max
    decision = evaluate_candidate_workload_support(
        _spec("canny_edges"),
        _workload("canny_edges", shape=(3, 3)),
        array_facts=_facts(
            shape=(3, 3),
            minimum=float(-maximum / 2),
            maximum=float(maximum / 2),
        ),
    )

    assert not decision.supported
    assert decision.fallback_allowed
    assert "CUDA flush-to-zero" in decision.reason_text
    assert "bool, uint8, and uint16" in decision.reason_text


@pytest.mark.parametrize("dtype", ("int8", "int32", "float16", "float64"))
def test_canny_unpromoted_numeric_dtypes_fall_back_to_cpu(dtype):
    decision = evaluate_candidate_workload_support(
        _spec("canny_edges"),
        _workload("canny_edges", dtype=dtype),
    )

    assert not decision.supported
    assert decision.fallback_allowed
    assert "CPU" in decision.reason_text


@pytest.mark.parametrize("sigma", (-20.0, -0.0, 0.0, 1.5, 12.0))
def test_canny_admits_canonical_sigma_zero_through_twelve(sigma):
    decision = evaluate_candidate_workload_support(
        _spec("canny_edges"),
        _workload(
            "canny_edges",
            dtype="uint16",
            parameters=(("sigma", sigma),),
        ),
    )

    assert decision.supported


def test_canny_sigma_above_validated_region_falls_back_but_invalid_sigma_fails():
    spec = _spec("canny_edges")
    too_large = evaluate_candidate_workload_support(
        spec,
        _workload(
            "canny_edges",
            dtype="uint8",
            parameters=(("sigma", 12.01),),
        ),
    )
    invalid = evaluate_candidate_workload_support(
        spec,
        _workload(
            "canny_edges",
            dtype="uint8",
            parameters=(("sigma", "nan"),),
        ),
    )

    assert not too_large.supported and too_large.fallback_allowed
    assert not invalid.supported and not invalid.fallback_allowed


@pytest.mark.parametrize(
    "parameters",
    (
        (("low_quantile", -0.01), ("high_quantile", 0.2)),
        (("low_quantile", 0.1), ("high_quantile", 1.01)),
        (("low_quantile", 0.8), ("high_quantile", 0.2)),
        (("low_quantile", "inf"), ("high_quantile", 0.2)),
    ),
)
def test_canny_invalid_authored_quantiles_fail_instead_of_falling_back(parameters):
    decision = evaluate_candidate_workload_support(
        _spec("canny_edges"),
        _workload("canny_edges", dtype="uint16", parameters=parameters),
    )

    assert not decision.supported
    assert not decision.fallback_allowed


def test_canny_explicit_luma_is_admitted_and_invalid_channel_contract_fails():
    spec = _spec("canny_edges")
    accepted = evaluate_candidate_workload_support(
        spec,
        _workload(
            "canny_edges",
            shape=(3, 37, 41),
            dtype="uint16",
            parameters=(("channel_axis", 0),),
        ),
    )
    bad_count = evaluate_candidate_workload_support(
        spec,
        _workload(
            "canny_edges",
            shape=(2, 37, 41),
            dtype="uint16",
            parameters=(("channel_axis", 0),),
        ),
    )
    bad_axis = evaluate_candidate_workload_support(
        spec,
        _workload(
            "canny_edges",
            shape=(37, 41),
            dtype="uint16",
            parameters=(("channel_axis", 0),),
        ),
    )

    assert accepted.supported
    assert not bad_count.supported and not bad_count.fallback_allowed
    assert not bad_axis.supported and not bad_axis.fallback_allowed


@pytest.mark.parametrize("operation_id", ("canny_edges", "otsu_threshold"))
def test_segmentation_luma_accepts_numpy_integral_axis(operation_id):
    decision = evaluate_candidate_workload_support(
        _spec(operation_id),
        _workload(
            operation_id,
            shape=(3, 37, 41),
            dtype="uint16",
            parameters=(("channel_axis", np.int64(0)),),
        ),
    )

    assert decision.supported


@pytest.mark.parametrize("operation", ("Canny", "Otsu"))
def test_segmentation_luma_rejects_numpy_boolean_axis(operation):
    from napari_vipp.core.compute_policy import _validated_luma_axis

    axis, error = _validated_luma_axis(
        (3, 37, 41),
        np.bool_(True),
        operation=operation,
    )

    assert axis is None
    assert error == f"{operation} channel_axis must be an integer or None."


@pytest.mark.parametrize(
    "dtype",
    (
        "int8",
        "uint8",
        "int16",
        "uint16",
    ),
)
def test_otsu_narrow_integer_region_is_admitted_without_extrema_facts(dtype):
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload("otsu_threshold", dtype=dtype),
    )

    assert decision.supported
    assert not decision.requires_complete_facts


@pytest.mark.parametrize(
    "dtype",
    (
        "int32",
        "uint32",
        "int64",
        "uint64",
    ),
)
def test_otsu_wide_integer_region_requires_and_accepts_exact_span_facts(dtype):
    spec = _spec("otsu_threshold")
    workload = _workload("otsu_threshold", dtype=dtype)

    missing = evaluate_candidate_workload_support(spec, workload)
    sampled = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(
            dtype=dtype,
            completeness=FactCompleteness.SAMPLED,
            minimum=-100,
            maximum=200,
        ),
    )
    accepted = evaluate_candidate_workload_support(
        spec,
        workload,
        array_facts=_facts(dtype=dtype, minimum=-100, maximum=200),
    )

    assert not missing.supported and missing.requires_complete_facts
    assert not sampled.supported and sampled.requires_complete_facts
    assert accepted.supported


def test_otsu_integer_span_above_exact_limit_fails_without_cpu_retry():
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload("otsu_threshold", dtype="int32"),
        array_facts=_facts(dtype="int32", minimum=0, maximum=65_536),
    )

    assert not decision.supported
    assert not decision.fallback_allowed
    assert "65,537 levels" in decision.reason_text


def test_otsu_slice_scope_wide_stack_span_falls_back_for_per_plane_cpu_check():
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload(
            "otsu_threshold",
            shape=(2, 32, 32),
            dtype="uint32",
            parameters=(("threshold_scope", "Slice histogram"),),
        ),
        array_facts=_facts(
            shape=(2, 32, 32),
            dtype="uint32",
            minimum=0,
            maximum=1_000_000,
        ),
    )

    assert not decision.supported
    assert decision.fallback_allowed
    assert "cannot be proved per plane" in decision.reason_text
    assert "Individual planes may still be valid" in decision.reason_text


def test_otsu_single_plane_wide_integer_span_remains_an_authored_error():
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload(
            "otsu_threshold",
            shape=(1, 32, 32),
            dtype="uint32",
            parameters=(("threshold_scope", "Slice histogram"),),
        ),
        array_facts=_facts(
            shape=(1, 32, 32),
            dtype="uint32",
            minimum=0,
            maximum=1_000_000,
        ),
    )

    assert not decision.supported
    assert not decision.fallback_allowed
    assert "1,000,001 levels" in decision.reason_text


@pytest.mark.parametrize("dtype", ("float16", "float32", "float64"))
@pytest.mark.parametrize("scope", ("Stack histogram", " slice HISTOGRAM "))
@pytest.mark.parametrize("histogram_bins", (2, 17, 65_536))
def test_otsu_float_histogram_region_accepts_both_scopes_and_full_bin_range(
    dtype,
    scope,
    histogram_bins,
):
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload(
            "otsu_threshold",
            dtype=dtype,
            parameters=(
                ("threshold_scope", scope),
                ("histogram_bins", histogram_bins),
            ),
        ),
    )

    assert decision.supported


@pytest.mark.parametrize("histogram_bins", (True, 1, 65_537, 2.5, "nope"))
def test_otsu_invalid_float_histogram_bins_fail_without_cpu_fallback(histogram_bins):
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload(
            "otsu_threshold",
            parameters=(("histogram_bins", histogram_bins),),
        ),
    )

    assert not decision.supported
    assert not decision.fallback_allowed
    assert "2 to 65,536" in decision.reason_text


def test_otsu_boolean_identity_ignores_histogram_bins_like_cpu():
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload(
            "otsu_threshold",
            dtype="bool",
            parameters=(("histogram_bins", "not used"),),
        ),
    )

    assert decision.supported


def test_otsu_luma_uses_float_histogram_without_integer_span_scan():
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload(
            "otsu_threshold",
            shape=(4, 31, 37),
            dtype="uint16",
            parameters=(("channel_axis", 0), ("histogram_bins", 31)),
        ),
    )

    assert decision.supported


@pytest.mark.parametrize(
    "parameters",
    (
        (("threshold_scope", "volume"),),
        (("channel_axis", True),),
        (("channel_axis", 8),),
    ),
)
def test_otsu_invalid_authored_scope_or_channel_fails_without_fallback(parameters):
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload("otsu_threshold", dtype="float32", parameters=parameters),
    )

    assert not decision.supported
    assert not decision.fallback_allowed


def test_otsu_complete_all_nonfinite_facts_fail_as_invalid_input():
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload("otsu_threshold"),
        array_facts=_facts(finite_count=0),
    )

    assert not decision.supported
    assert not decision.fallback_allowed
    assert "at least one finite" in decision.reason_text


def test_canny_memory_model_scales_with_plane_size_and_reserves_workspaces():
    spec = _spec("canny_edges")
    small_workload = _workload(
        "canny_edges",
        shape=(8, 128, 128),
        dtype="uint16",
    )
    large_workload = replace(
        small_workload,
        input_shapes=((8, 512, 512),),
    )

    small = estimate_candidate_memory(spec, small_workload)
    large = estimate_candidate_memory(spec, large_workload)

    assert small.model_id == "cupyx-canny-exact-memory-v1"
    assert large.total_device_peak_bytes > small.total_device_peak_bytes
    assert large.total_device_peak_bytes > (
        np.prod((8, 512, 512)) * (np.dtype(np.uint16).itemsize + 1)
    )


@pytest.mark.parametrize(
    ("dtype", "expected_luma_buffers"),
    (("uint16", 7), ("float32", 4)),
)
def test_canny_luma_memory_counts_cast_product_and_scalar_output(
    dtype,
    expected_luma_buffers,
):
    scalar_elements = 4 * 256 * 256
    workload = _workload(
        "canny_edges",
        shape=(4, 256, 256, 3),
        dtype=dtype,
        parameters=(("channel_axis", -1),),
    )

    estimate = estimate_candidate_memory(_spec("canny_edges"), workload)

    raw_input = scalar_elements * 3 * np.dtype(dtype).itemsize
    luma_intermediates = (
        scalar_elements * np.dtype(np.float32).itemsize * expected_luma_buffers
    )
    assert estimate.total_device_peak_bytes >= raw_input + luma_intermediates


def test_canny_luma_memory_cap_rejects_the_pre_fix_undercount():
    scalar_elements = 4 * 256 * 256
    plane_elements = 256 * 256
    workload = _workload(
        "canny_edges",
        shape=(4, 256, 256, 3),
        dtype="uint16",
        parameters=(("channel_axis", -1),),
    )
    estimate = estimate_candidate_memory(_spec("canny_edges"), workload)
    raw_input = scalar_elements * 3 * np.dtype(np.uint16).itemsize
    output = scalar_elements * np.dtype(bool).itemsize
    old_luma_workspace = scalar_elements * np.dtype(np.float32).itemsize
    plane_workspace = plane_elements * np.dtype(np.float32).itemsize * 24
    old_runtime_peak = (
        raw_input + output + old_luma_workspace + plane_workspace
    )
    old_required_with_uncertainty = old_runtime_peak + max(
        8 * 1024**2,
        old_runtime_peak // 4,
    )

    decision = evaluate_memory_support(
        estimate,
        memory_cap_bytes=old_required_with_uncertainty,
    )

    assert not decision.supported
    assert decision.reason is DecisionReason.MEMORY_LIMIT


def test_otsu_memory_model_accounts_for_scope_bins_and_bounded_host_finalizer():
    spec = _spec("otsu_threshold")
    stack = _workload(
        "otsu_threshold",
        shape=(8, 256, 256),
        parameters=(
            ("threshold_scope", "Stack histogram"),
            ("histogram_bins", 65_536),
        ),
    )
    slices = replace(
        stack,
        parameters=(
            ("threshold_scope", "Slice histogram"),
            ("histogram_bins", 2),
        ),
    )

    stack_estimate = estimate_candidate_memory(spec, stack)
    slice_estimate = estimate_candidate_memory(spec, slices)

    assert stack_estimate.model_id == "cupy-otsu-histogram-memory-v1"
    assert (
        stack_estimate.total_device_peak_bytes
        > slice_estimate.total_device_peak_bytes
    )
    assert stack_estimate.host_materialization_peak_bytes >= 65_536 * np.dtype(
        np.intp
    ).itemsize
    assert slice_estimate.host_materialization_peak_bytes >= 2 * np.dtype(
        np.intp
    ).itemsize


def test_otsu_luma_memory_model_counts_cast_product_and_scalar_output():
    scalar_elements = 4 * 256 * 256
    workload = _workload(
        "otsu_threshold",
        shape=(4, 256, 256, 3),
        dtype="uint16",
        parameters=(("channel_axis", -1),),
    )

    estimate = estimate_candidate_memory(_spec("otsu_threshold"), workload)

    # float32 RGB cast (3N), coefficient product (3N), and luma (N) must all
    # fit concurrently before histogram work and the output mask are counted.
    luma_intermediates = scalar_elements * np.dtype(np.float32).itemsize * 7
    raw_input = scalar_elements * 3 * np.dtype(np.uint16).itemsize
    assert estimate.total_device_peak_bytes >= raw_input + luma_intermediates


def test_otsu_canonical_uint16_stack_uses_bounded_atomic_memory_model():
    elements = 8 * 1024 * 1024
    workload = _workload(
        "otsu_threshold",
        shape=(8, 1024, 1024),
        dtype="uint16",
        parameters=(
            ("threshold_scope", "Stack histogram"),
            ("histogram_bins", 256),
        ),
    )

    estimate = estimate_candidate_memory(_spec("otsu_threshold"), workload)

    raw_input = elements * np.dtype(np.uint16).itemsize
    bool_output = elements * np.dtype(bool).itemsize
    # Relative uint64 levels plus an equally sized conservative allowance for
    # extrema reductions/private-pool granularity.  The atomic implementation
    # owns only one bounded uint64 counts array, never CUB's per-block copies.
    image_workspace = elements * 2 * np.dtype(np.uint64).itemsize
    counts = 65_536 * np.dtype(np.uint64).itemsize
    edges = (65_536 + 1) * np.dtype(np.uint16).itemsize
    expected_runtime = raw_input + bool_output + image_workspace + counts + edges

    assert estimate.model_id == "cupy-otsu-histogram-memory-v1"
    assert estimate.runtime_managed_peak_bytes == expected_runtime
    assert estimate.total_device_peak_bytes == expected_runtime
    assert estimate.uncertainty_bytes == expected_runtime // 4

    rejected = evaluate_memory_support(
        estimate,
        memory_cap_bytes=(
            estimate.total_device_peak_bytes + estimate.uncertainty_bytes - 1
        ),
    )
    admitted = evaluate_memory_support(
        estimate,
        memory_cap_bytes=(
            estimate.total_device_peak_bytes + estimate.uncertainty_bytes
        ),
    )
    assert not rejected.supported
    assert rejected.reason is DecisionReason.MEMORY_LIMIT
    assert admitted.supported


def test_policy_rejections_retain_workload_reason_type():
    decision = evaluate_candidate_workload_support(
        _spec("otsu_threshold"),
        _workload(
            "otsu_threshold",
            parameters=(("threshold_scope", "invalid"),),
        ),
    )

    assert decision.reason is DecisionReason.WORKLOAD_UNSUPPORTED
