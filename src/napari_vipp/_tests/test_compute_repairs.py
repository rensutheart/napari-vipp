from __future__ import annotations

import math

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRepairAction,
    ComputeRepairCandidate,
    ComputeRepairSuggestion,
    ComputeRequest,
    OutputPortKey,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.compute_policy import ArrayFacts, FactCompleteness
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_repairs import (
    potential_compute_repair_specs,
    suggest_compute_repairs,
)
from napari_vipp.core.pipeline import PrototypePipeline


def _cuda_environment(**updates) -> ComputeEnvironment:
    values = {
        "os_name": "Windows",
        "execution_mode": "native",
        "python_implementation": "CPython",
        "python_version": "3.12",
        "python_abi": "cpython-312",
        "runtime_ids": ("cpu-numpy", "cuda-cupy"),
        "implementation_libraries": ("cpu", "cupy", "cupyx"),
        "runtime_versions": (
            ("cuda-cupy", "14.1.1"),
            ("cupy", "14.1.1"),
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
        "memory_topology": "discrete",
        "total_accelerator_memory_bytes": 16 * 1024**3,
        "probe_status": "available",
    }
    values.update(updates)
    return ComputeEnvironment(**values)


def _gaussian_workload(
    dtype: str,
    *,
    sigma: float = 1.2,
    shape: tuple[int, ...] = (31, 37),
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        node_id="gaussian-node",
        operation_id="gaussian_blur",
        input_shapes=(shape,),
        input_dtypes=(dtype,),
        parameters=(("sigma", sigma),),
        resolved_spatial_ndim=2,
    )


@pytest.mark.parametrize("dtype", ("uint8", "uint16"))
def test_safe_integer_gaussian_returns_structured_exact_repair(dtype):
    workload = _gaussian_workload(dtype)
    request = ComputeRequest(mode=ComputeMode.AUTO)
    with ComputeRegistry() as registry:
        suggestions = suggest_compute_repairs(
            request,
            workload,
            registry,
            _cuda_environment(),
        )

    assert len(suggestions) == 1
    suggestion = suggestions[0]
    assert suggestion.action is ComputeRepairAction.INSERT_CONVERT_DTYPE
    assert suggestion.node_id == "gaussian-node"
    assert suggestion.operation_id == "gaussian_blur"
    assert suggestion.input_port_index == 0
    assert suggestion.input_port_name == "image"
    assert suggestion.current_dtype == dtype
    assert suggestion.target_dtype == "float32"
    assert suggestion.scaling == "preserve"
    assert suggestion.exact is True
    assert suggestion.conversion_parameters == {
        "output_dtype": "float32",
        "scaling": "preserve",
    }
    assert suggestion.candidate.runtime_id == "cuda-cupy"
    assert suggestion.candidate.implementation_library_id == "cupy"
    assert "could become eligible for GPU use" in suggestion.message
    assert "preserve every pixel value exactly" in suggestion.message
    expected_factor = 4 if dtype == "uint8" else 2
    assert f"uses {expected_factor}× as much memory" in suggestion.message


def test_repair_contract_is_json_ready_and_validated():
    candidate = ComputeRepairCandidate(
        "cupy-gaussian-blur-v1",
        "1",
        "cuda-cupy",
        "cupy",
    )
    suggestion = ComputeRepairSuggestion(
        "insert_convert_dtype",
        "node",
        "gaussian_blur",
        0,
        "image",
        "uint16",
        "float32",
        "preserve",
        True,
        "Values will be preserved exactly.",
        candidate,
    )

    assert suggestion.as_dict() == {
        "action": "insert_convert_dtype",
        "node_id": "node",
        "operation_id": "gaussian_blur",
        "input_port_index": 0,
        "input_port_name": "image",
        "current_dtype": "uint16",
        "target_dtype": "float32",
        "scaling": "preserve",
        "exact": True,
        "message": "Values will be preserved exactly.",
        "candidate": candidate.as_dict(),
    }


@pytest.mark.parametrize(
    ("current_dtype", "target_dtype", "scaling", "exact"),
    (
        ("float64", "float32", "preserve", True),
        ("uint16", "uint8", "preserve", True),
        ("uint16", "float32", "rescale", True),
        ("uint16", "float32", "preserve", False),
    ),
)
def test_insert_conversion_contract_rejects_unreviewed_or_lossy_payloads(
    current_dtype,
    target_dtype,
    scaling,
    exact,
):
    candidate = ComputeRepairCandidate(
        "cupy-gaussian-blur-v1",
        "1",
        "cuda-cupy",
        "cupy",
    )

    with pytest.raises(ValueError):
        ComputeRepairSuggestion(
            "insert_convert_dtype",
            "node",
            "gaussian_blur",
            0,
            "image",
            current_dtype,
            target_dtype,
            scaling,
            exact,
            "Untrusted payload.",
            candidate,
        )


@pytest.mark.parametrize("dtype", ("float32", "int16", "float64", "bool"))
def test_supported_or_non_exact_source_dtype_is_not_suggested(dtype):
    with ComputeRegistry() as registry:
        suggestions = suggest_compute_repairs(
            ComputeRequest(mode="auto"),
            _gaussian_workload(dtype),
            registry,
            _cuda_environment(),
        )

    assert suggestions == ()


def test_environment_parameter_and_memory_blockers_suppress_repair():
    request = ComputeRequest(mode="auto")
    missing_library = _cuda_environment(implementation_libraries=("cpu",))
    invalid_parameters = _gaussian_workload("uint16", sigma=13.0)
    memory_limited = ComputeRequest(
        mode="auto",
        accelerator_memory_cap_bytes=1,
    )
    with ComputeRegistry() as registry:
        assert (
            suggest_compute_repairs(
                request,
                _gaussian_workload("uint16"),
                registry,
                missing_library,
            )
            == ()
        )
        assert (
            suggest_compute_repairs(
                request,
                invalid_parameters,
                registry,
                _cuda_environment(),
            )
            == ()
        )
        assert (
            suggest_compute_repairs(
                memory_limited,
                _gaussian_workload("uint16"),
                registry,
                _cuda_environment(),
            )
            == ()
        )


def test_preprobe_includes_only_a_valid_one_conversion_counterfactual():
    request = ComputeRequest(mode="auto")
    valid = _gaussian_workload("uint16")
    invalid_parameters = _gaussian_workload("uint16", sigma=13.0)
    with ComputeRegistry() as registry:
        potential = potential_compute_repair_specs(request, valid, registry)
        invalid = potential_compute_repair_specs(
            request,
            invalid_parameters,
            registry,
        )
        execution_potential = execution_module._potential_accelerator_specs(
            registry,
            request,
            (valid,),
        )

    assert [item.implementation_id for item in potential] == ["cupy-gaussian-blur-v1"]
    assert [item.implementation_id for item in execution_potential] == [
        "cupy-gaussian-blur-v1"
    ]
    assert invalid == ()


def test_nonfinite_other_input_remains_a_blocker():
    image_shape = (3, 64, 64)
    psf_shape = (9, 9)
    workload = WorkloadDescriptor(
        "rl-node",
        "richardson_lucy_deconvolution",
        (image_shape, psf_shape),
        ("uint16", "float32"),
        parameters=(
            ("spatial_mode", "2D YX"),
            ("iterations", 25),
            ("normalize_psf", True),
            ("clip_negative_input", True),
            ("clip_output_negative", True),
            ("preserve_input_scale", True),
            ("filter_epsilon", 1e-8),
        ),
        resolved_spatial_ndim=2,
    )
    facts = (
        ArrayFacts(
            image_shape,
            "uint16",
            math.prod(image_shape),
            "image-revision",
            completeness=FactCompleteness.COMPLETE,
            finite_count=math.prod(image_shape),
        ),
        ArrayFacts(
            psf_shape,
            "float32",
            math.prod(psf_shape),
            "psf-revision",
            completeness=FactCompleteness.COMPLETE,
            finite_count=math.prod(psf_shape) - 1,
            maximum=1.0,
        ),
    )
    with ComputeRegistry() as registry:
        suggestions = suggest_compute_repairs(
            ComputeRequest(mode="auto"),
            workload,
            registry,
            _cuda_environment(),
            array_facts=facts,
        )

    assert suggestions == ()


def test_another_dtype_blocked_input_suppresses_single_port_repair():
    workload = WorkloadDescriptor(
        "rl-node",
        "richardson_lucy_deconvolution",
        ((3, 64, 64), (9, 9)),
        ("uint16", "uint8"),
        parameters=(
            ("spatial_mode", "2D YX"),
            ("iterations", 25),
            ("normalize_psf", True),
            ("clip_negative_input", True),
            ("clip_output_negative", True),
            ("preserve_input_scale", True),
            ("filter_epsilon", 1e-8),
        ),
        resolved_spatial_ndim=2,
    )
    with ComputeRegistry() as registry:
        suggestions = suggest_compute_repairs(
            ComputeRequest(mode="auto"),
            workload,
            registry,
            _cuda_environment(),
        )

    assert suggestions == ()


def test_planning_attaches_repair_to_execution_plan_but_cpu_mode_does_not():
    workload = _gaussian_workload("uint16")
    with ComputeRegistry() as registry:
        result = plan_compute_decisions(
            ComputeRequest(mode="auto"),
            (workload,),
            registry=registry,
            environment=_cuda_environment(),
        )
        cpu_result = plan_compute_decisions(
            ComputeRequest(mode="cpu"),
            (workload,),
            registry=registry,
            environment=_cuda_environment(),
        )

    assert result.decisions[0].runtime_id == "cpu-numpy"
    assert len(result.repair_suggestions) == 1
    assert result.as_execution_plan().repair_suggestions == result.repair_suggestions
    assert cpu_result.repair_suggestions == ()


def test_stale_facts_fail_closed():
    workload = _gaussian_workload("uint16")
    stale = (
        ArrayFacts(
            workload.input_shapes[0],
            "uint8",
            math.prod(workload.input_shapes[0]),
            "stale-revision",
        ),
    )
    with ComputeRegistry() as registry:
        suggestions = suggest_compute_repairs(
            ComputeRequest(mode="auto"),
            workload,
            registry,
            _cuda_environment(),
            array_facts=stale,
        )

    assert suggestions == ()


def test_exact_preserve_conversion_propagates_complete_float32_theorem():
    source = execution_module._complete_array_facts(
        np.arange(25, dtype=np.uint16).reshape(5, 5),
        revision_fingerprint="source",
    )

    propagated = execution_module._propagate_shape_preserving_facts(
        "convert_dtype",
        source,
        {"output_dtype": "float32", "scaling": "preserve"},
        output_port=OutputPortKey("convert", 0),
        output_dtype="float32",
    )
    rescaled = execution_module._propagate_shape_preserving_facts(
        "convert_dtype",
        source,
        {"output_dtype": "float32", "scaling": "rescale"},
        output_port=OutputPortKey("convert", 0),
        output_dtype="float32",
    )

    assert propagated is not None
    assert propagated.dtype == "float32"
    assert propagated.all_finite is True
    assert propagated.minimum == source.minimum
    assert propagated.maximum == source.maximum
    assert {"nonnegative", "no-negative-zero"} <= set(propagated.guarantees)
    assert rescaled is None


def test_inserted_conversion_supplies_facts_for_live_gpu_corridor_planning():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    conversion = pipeline.add_node("convert_dtype")
    pipeline.set_param(conversion.id, "output_dtype", "float32")
    pipeline.set_param(conversion.id, "scaling", "preserve")
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 1.2)
    assert pipeline.connect("input", conversion.id).success
    assert pipeline.connect(conversion.id, gaussian.id).success
    data = np.arange(81, dtype=np.uint16).reshape(9, 9)
    source_port = OutputPortKey("input", 0)
    source_facts = execution_module._complete_array_facts(
        data,
        revision_fingerprint="source",
    )

    with ComputeRegistry() as registry:
        workloads, facts_by_node, _lineage = execution_module._assemble_workloads(
            pipeline,
            frozenset({conversion.id, gaussian.id}),
            {source_port: data},
            {},
            registry,
            False,
            seed_facts_by_port={source_port: source_facts},
        )
        planning = plan_compute_decisions(
            ComputeRequest(mode="auto"),
            workloads,
            registry=registry,
            environment=_cuda_environment(),
            array_facts=facts_by_node,
        )

    gaussian_facts = facts_by_node[gaussian.id][0]
    assert gaussian_facts.dtype == "float32"
    assert gaussian_facts.all_finite is True
    assert gaussian_facts.minimum == source_facts.minimum
    assert gaussian_facts.maximum == source_facts.maximum
    assert {decision.runtime_id for decision in planning.decisions} == {"cuda-cupy"}
    assert planning.repair_suggestions == ()
