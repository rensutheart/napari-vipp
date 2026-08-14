from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionFallbackRecord,
    ExecutionReport,
    FallbackPolicy,
    FallbackReason,
    MemoryEstimate,
    NodeComputePreference,
    NodeExecutionDecision,
    NodePreferenceKind,
    ScientificResultKey,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_specs import (
    AdmissionTier,
    ComputePortContract,
    OperationComputeSpec,
    ValueKind,
    accelerator_compute_specs,
    compute_specs_for,
    validate_compute_specs,
)


def test_compute_request_is_strict_immutable_and_json_safe():
    request = ComputeRequest(
        mode="custom",
        fallback_policy="strict",
        node_preferences={
            "median": "cpu",
            "gaussian": "library:cupyx",
            "background": {
                "kind": "implementation",
                "value": "cucim-background-f32-v1",
            },
        },
        accelerator_memory_cap_bytes=4_000_000_000,
    )

    assert request.mode is ComputeMode.CUSTOM
    assert request.fallback_policy is FallbackPolicy.STRICT
    assert request.preference_for("median").kind is NodePreferenceKind.CPU
    assert request.preference_for("gaussian").value == "cupyx"
    assert request.preference_for("missing").kind is NodePreferenceKind.AUTO
    with pytest.raises(TypeError):
        request.node_preferences["new"] = NodeComputePreference("cpu")

    payload = request.as_dict()
    assert ComputeRequest.from_dict(payload) == request
    json.dumps(payload, allow_nan=False)
    assert len(request.fingerprint) == 64


def test_legacy_selective_mode_loads_as_canonical_custom():
    request = ComputeRequest(mode="selective")

    assert request.mode is ComputeMode.CUSTOM
    assert request.as_dict()["mode"] == "custom"


def test_prefer_gpu_is_a_round_trippable_visible_fallback_policy():
    request = ComputeRequest(
        mode="prefer_gpu",
        node_preferences={"dormant": "cpu"},
    )

    assert ComputeMode.PREFER_GPU.value == "prefer_gpu"
    assert request.mode is ComputeMode.PREFER_GPU
    assert request.fallback_policy is FallbackPolicy.VISIBLE
    assert ComputeRequest.from_dict(request.as_dict()) == request
    json.dumps(request.as_dict(), allow_nan=False)

    with pytest.raises(ValueError, match="Prefer GPU requires visible CPU fallback"):
        ComputeRequest(mode="prefer_gpu", fallback_policy="strict")


def test_contract_import_does_not_import_optional_accelerators():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import napari_vipp.core.compute; "
                "import napari_vipp.core.compute_specs; "
                "import napari_vipp.core.compute_planning; "
                "from napari_vipp.core.compute import ComputeEnvironment; "
                "from napari_vipp.core.compute_registry import ComputeRegistry; "
                "ComputeEnvironment(); "
                "ComputeRegistry(); "
                "assert 'cupy' not in sys.modules; "
                "assert 'cupyx' not in sys.modules; "
                "assert 'cucim' not in sys.modules; "
                "assert 'napari_vipp.core.pipeline' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("value", ("gpu", "fastest", ""))
def test_compute_mode_rejects_old_or_ambiguous_global_modes(value):
    with pytest.raises(ValueError, match="Unsupported compute mode"):
        ComputeMode.parse(value)


def test_node_preference_requires_a_value_only_for_pins():
    with pytest.raises(ValueError, match="requires a stable value"):
        NodeComputePreference("library")
    with pytest.raises(ValueError, match="must not include a value"):
        NodeComputePreference("cpu", "cupyx")
    with pytest.raises(ValueError, match="Unknown node preference"):
        NodeComputePreference.parse({"kind": "cpu", "surprise": True})


def test_environment_fingerprint_includes_python_abi_and_is_json_safe():
    environment = ComputeEnvironment(
        os_name="Windows",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        runtime_ids=("cuda-cupy", "cpu-numpy"),
        implementation_libraries=("cupyx", "cpu"),
        runtime_versions=(("cupy", "14.1.1"),),
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
        runtime_probe_fingerprints=(("cuda-cupy", "probe-a"),),
        runtime_metadata=(
            (
                "cuda-cupy",
                (
                    ("cuda_runtime_version", "13020"),
                    ("driver_version", "13030"),
                ),
            ),
        ),
        implementation_library_metadata=(("cupyx", (("build", "reviewed"),)),),
        driver_version="13030",
        device_id="cuda:0",
        device_name="NVIDIA GeForce RTX 5090",
        device_class="nvidia-cuda",
        device_metadata=(("compute_capability", "12.0"),),
        memory_topology="discrete",
        total_accelerator_memory_bytes=8_000_000_000,
    )
    changed = replace(environment, python_abi="cpython-313")

    assert environment.fingerprint != changed.fingerprint
    assert (
        environment.fingerprint
        != replace(
            environment,
            runtime_probe_fingerprints=(("cuda-cupy", "probe-b"),),
        ).fingerprint
    )
    assert (
        environment.fingerprint
        != replace(
            environment,
            scientific_stack_versions=(
                ("numpy", "2.5.0"),
                ("scipy", "1.18.0"),
                ("scikit-image", "0.26.0"),
            ),
        ).fingerprint
    )
    assert (
        environment.fingerprint
        != replace(
            environment,
            device_metadata=(("compute_capability", "11.0"),),
        ).fingerprint
    )
    assert (
        environment.fingerprint
        != replace(
            environment,
            device_name="NVIDIA GeForce RTX 4050 Laptop GPU",
            device_metadata=(("compute_capability", "8.9"),),
        ).fingerprint
    )
    assert (
        environment.fingerprint
        != replace(
            environment,
            implementation_library_metadata=(("cupyx", (("build", "changed"),)),),
        ).fingerprint
    )
    json.dumps(environment.as_dict(), allow_nan=False)
    assert ComputeEnvironment.from_dict(environment.as_dict()) == environment
    json.dumps(ExecutionReport(ComputeRequest(), environment).as_dict())


def test_execution_report_serializes_durable_fallback_attempt_schema():
    record = ExecutionFallbackRecord(
        segment_id="segment-1",
        runtime_id="cuda-cupy",
        node_ids=("gaussian-1",),
        reason="out_of_memory",
        reason_code="cupy_oom",
        exception_type="OutOfMemoryError",
        message="allocation failed",
        cpu_retry_succeeded=True,
        memory_estimate=MemoryEstimate(
            runtime_managed_peak_bytes=300,
            total_device_peak_bytes=500,
            host_materialization_peak_bytes=100,
            model_id="test-model-v1",
        ),
        memory_topology="discrete",
        device_total_bytes=1_000,
        device_free_bytes=500,
    )
    payload = ExecutionReport(
        ComputeRequest(mode="custom"),
        ComputeEnvironment(),
        fallback_records=(record,),
    ).as_dict()

    fallback = payload["fallback_records"][0]
    assert fallback == record.as_dict()
    assert fallback["reason"] == "out_of_memory"
    assert fallback["cpu_retry_succeeded"] is True
    assert fallback["memory_topology"] == "discrete"
    assert fallback["memory_estimate"]["runtime_managed_peak_bytes"] == 300
    json.dumps(payload, allow_nan=False)


def test_environment_provenance_rejects_duplicate_metadata_keys():
    with pytest.raises(ValueError, match="keys must be unique"):
        ComputeEnvironment(
            device_metadata=(("compute_capability", "12.0"),) * 2,
        )


def test_numeric_contracts_reject_bool_nan_and_noncanonical_values():
    with pytest.raises(ValueError, match="non-negative integer"):
        MemoryEstimate(runtime_managed_peak_bytes=True)
    with pytest.raises(ValueError, match="NaN"):
        WorkloadDescriptor(
            "node",
            "gaussian_blur",
            ((8, 8),),
            ("float32",),
            parameters=(("sigma", float("nan")),),
        )
    with pytest.raises(TypeError, match="inputs_resolved must be a boolean"):
        WorkloadDescriptor(
            "node",
            "gaussian_blur",
            ((8, 8),),
            ("float32",),
            inputs_resolved=1,
        )

    numpy_integer = WorkloadDescriptor(
        "node",
        "canny_edges",
        ((3, 8, 8),),
        ("uint16",),
        parameters=(("channel_axis", np.int64(0)),),
    )
    assert dict(numpy_integer.parameters)["channel_axis"] == 0
    assert type(dict(numpy_integer.parameters)["channel_axis"]) is int


def test_fallback_is_typed_and_distinct_from_a_policy_cpu_choice():
    policy_cpu = NodeExecutionDecision(
        "node",
        "median_filter",
        NodeComputePreference("auto"),
        "cpu-numpy",
        "cpu",
        "cpu-median-filter-v1",
        DecisionKind.POLICY_CPU,
        DecisionReason.PERFORMANCE_GATE,
        "CPU is faster for this workload.",
    )
    assert not policy_cpu.fallback_used

    with pytest.raises(ValueError, match="requires a typed fallback"):
        replace(policy_cpu, decision_kind=DecisionKind.FALLBACK_CPU)
    fallback = replace(
        policy_cpu,
        decision_kind=DecisionKind.FALLBACK_CPU,
        fallback_reason=FallbackReason.DEPENDENCY_UNAVAILABLE,
    )
    assert fallback.fallback_used


def test_scientific_cache_identity_is_output_port_specific():
    first = ScientificResultKey(
        "split_channels",
        0,
        "channel-output-v1",
        "params",
        ("upstream",),
        "cpu-split-channels-v1",
        "1",
        "deps",
        "result-v1",
    )

    assert first.digest != replace(first, output_port_index=1).digest


def _candidate_spec(**updates) -> OperationComputeSpec:
    values = {
        "operation_id": "gaussian_blur",
        "implementation_id": "test-gaussian-v1",
        "implementation_version": "1",
        "runtime_id": "fake-device",
        "array_domain": "fake-array",
        "implementation_library_id": "test-library",
        "callable_ref": "tests.fake:gaussian",
        "host_boundary": False,
        "admission_tier": AdmissionTier.DEVELOPER_HIDDEN,
        "validated_environment_policy_id": "test-env-v1",
        "input_ports": (ComputePortContract(0, ValueKind.ARRAY),),
        "output_ports": (ComputePortContract(0, ValueKind.IMAGE),),
        "parameter_policy_id": "test-params-v1",
        "workload_policy_id": "test-workload-v1",
        "parity_policy_id": "test-parity-v1",
        "memory_model_id": "test-memory-v1",
        "shape_policy_id": "shape-preserving-v1",
        "boundary_policy_id": "reflect-v1",
        "precision_policy_id": "scientific-default-v1",
        "progress_policy_id": "test-progress-v1",
        "cancellation_policy_id": "test-cancel-v1",
        "side_effect_policy_id": "pure-v1",
    }
    values.update(updates)
    return OperationComputeSpec(**values)


def test_compute_spec_registry_declares_only_lazy_accelerator_candidates():
    accelerator_specs = accelerator_compute_specs()
    assert {spec.operation_id for spec in accelerator_specs} == {
        "binary_threshold",
        "canny_edges",
        "rolling_ball_background",
        "subtract_background",
        "median_filter",
        "gaussian_blur",
        "gaussian_blur_3d",
        "convert_dtype",
        "extract_channel",
        "otsu_threshold",
        "richardson_lucy_deconvolution",
        "richardson_lucy_tv_deconvolution",
        "sigma_filter",
        "label_connected_components",
        "measure_objects",
        "measure_objects_intensity",
    }
    custom_ids = {
        spec.implementation_id
        for spec in accelerator_specs
        if spec.admission_tier is AdmissionTier.PUBLIC_CUSTOM
    }
    assert custom_ids == {
        "cupy-binary-threshold-f32-exact-v1",
        "cupy-extract-channel-view-v1",
    }
    assert all(
        spec.admission_tier
        in {AdmissionTier.PUBLIC_AUTO_CANDIDATE, AdmissionTier.PUBLIC_CUSTOM}
        for spec in accelerator_specs
    )
    assert all(spec.runtime_id == "cuda-cupy" for spec in accelerator_specs)
    assert all(spec.array_domain == "cuda-cupy" for spec in accelerator_specs)
    richardson_lucy = next(
        spec
        for spec in accelerator_specs
        if spec.operation_id == "richardson_lucy_deconvolution"
    )
    assert richardson_lucy.input_ports[1].accumulation_dtype == "float64"
    assert richardson_lucy.boundary_policy_id == "scipy-signal-zero-fill-same-v1"
    richardson_lucy_tv = next(
        spec
        for spec in accelerator_specs
        if spec.operation_id == "richardson_lucy_tv_deconvolution"
    )
    expected_tv_boundary = "rl-tv-zero-fill-same-central-gradient-edge1-v1"
    assert richardson_lucy_tv.boundary_policy_id == expected_tv_boundary
    assert all(
        port.boundary_policy_id == expected_tv_boundary
        for port in (*richardson_lucy_tv.input_ports, *richardson_lucy_tv.output_ports)
    )
    assert "iterations-at-most-100-v2" in richardson_lucy.limitations
    assert richardson_lucy.parity_policy_id == "rl-scientific-equivalence-v2"
    assert richardson_lucy_tv.parity_policy_id == (
        "rl-tv-scientific-equivalence-v2"
    )
    assert "lambda-zero-iterations-at-most-100-v2" in (
        richardson_lucy_tv.limitations
    )
    assert "positive-tv-iterations-10-or-25-v1" in richardson_lucy_tv.limitations
    measurements = tuple(
        spec
        for spec in accelerator_specs
        if spec.operation_id in {"measure_objects", "measure_objects_intensity"}
    )
    assert len(measurements) == 2
    assert all(
        spec.host_finalizer_ref
        == "napari_vipp.core.measurements:finalize_basic_measurement_outputs"
        for spec in measurements
    )
    assert all(
        spec.output_ports[0].value_kind is ValueKind.TABLE for spec in measurements
    )
    assert all(
        spec.output_ports[0].internal_dtypes == ("float64",) for spec in measurements
    )
    assert (
        len(
            compute_specs_for(
                "gaussian_blur",
                include_cpu=False,
            )
        )
        == 1
    )


def test_compute_spec_validation_can_cross_check_a_lightweight_catalog():
    cpu_spec = compute_specs_for("gaussian_blur")[0]
    assert cpu_spec.runtime_id == "cpu-numpy"
    assert cpu_spec.implementation_library_id == "cpu"
    assert not cpu_spec.is_gpu

    known = {"gaussian_blur"}
    validate_compute_specs((_candidate_spec(),), known_operation_ids=known)
    with pytest.raises(ValueError, match="Duplicate implementation ID"):
        validate_compute_specs(
            (_candidate_spec(), _candidate_spec()),
            known_operation_ids=known,
        )
    with pytest.raises(ValueError, match="unknown operation"):
        validate_compute_specs(
            (_candidate_spec(operation_id="missing"),),
            known_operation_ids=known,
        )


def test_host_finalizer_contract_is_optional_lazy_and_device_resident_only():
    assert _candidate_spec().host_finalizer_ref == ""
    spec = _candidate_spec(
        supports_device_residency=True,
        host_finalizer_ref="tests.fake:finalize_table",
    )
    assert spec.host_finalizer_ref == "tests.fake:finalize_table"

    with pytest.raises(ValueError, match="module:attribute"):
        _candidate_spec(
            supports_device_residency=True,
            host_finalizer_ref="not-an-import-reference",
        )
    with pytest.raises(ValueError, match="resident, non-boundary"):
        _candidate_spec(host_finalizer_ref="tests.fake:finalize_table")
    with pytest.raises(ValueError, match="resident, non-boundary"):
        _candidate_spec(
            host_boundary=True,
            supports_device_residency=True,
            host_finalizer_ref="tests.fake:finalize_table",
        )


def test_cpu_source_writer_and_dynamic_output_boundaries_are_explicit():
    source = compute_specs_for("input")[0]
    writer = compute_specs_for("save_output")[0]
    dynamic = compute_specs_for("split_channels")[0]

    assert source.host_boundary
    assert source.input_ports == ()
    assert source.callable_ref == ""
    assert writer.host_boundary
    assert writer.side_effect_policy_id == "host-writer-v1"
    assert dynamic.dynamic_output_policy_id == "cpu-dynamic-output-v1"
    assert dynamic.output_ports[0].schema_id == "dynamic-ports-v1"


def test_developer_hidden_declaration_requires_explicit_visibility():
    spec = _candidate_spec()

    assert not spec.visible_for(allow_experimental=False)
    assert spec.visible_for(allow_experimental=True)
    assert not spec.eligible_for_auto(allow_experimental=False)
    assert spec.eligible_for_auto(allow_experimental=True)


def test_public_auto_declaration_is_visible_without_experimental_enablement():
    spec = _candidate_spec(admission_tier=AdmissionTier.PUBLIC_AUTO_CANDIDATE)

    assert spec.visible_for(allow_experimental=False)
    assert spec.eligible_for_auto(allow_experimental=False)
