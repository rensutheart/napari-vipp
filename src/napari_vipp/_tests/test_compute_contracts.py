from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import pytest

from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
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
        mode="selective",
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

    assert request.mode is ComputeMode.SELECTIVE
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


def test_contract_import_does_not_import_optional_accelerators():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import napari_vipp.core.compute; "
                "import napari_vipp.core.compute_specs; "
                "assert 'cupy' not in sys.modules; "
                "assert 'cucim' not in sys.modules"
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
        memory_topology="discrete",
        total_accelerator_memory_bytes=8_000_000_000,
    )
    changed = replace(environment, python_abi="cpython-313")

    assert environment.fingerprint != changed.fingerprint
    json.dumps(environment.as_dict(), allow_nan=False)
    assert ComputeEnvironment.from_dict(environment.as_dict()) == environment
    json.dumps(ExecutionReport(ComputeRequest(), environment).as_dict())


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


def test_compute_spec_registry_starts_cpu_only_and_validates_declarations():
    assert accelerator_compute_specs() == ()
    cpu_spec = compute_specs_for("gaussian_blur")[0]
    assert cpu_spec.runtime_id == "cpu-numpy"
    assert cpu_spec.implementation_library_id == "cpu"
    assert not cpu_spec.is_gpu

    validate_compute_specs((_candidate_spec(),))
    with pytest.raises(ValueError, match="Duplicate implementation ID"):
        validate_compute_specs((_candidate_spec(), _candidate_spec()))
    with pytest.raises(ValueError, match="unknown operation"):
        validate_compute_specs((_candidate_spec(operation_id="missing"),))


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
