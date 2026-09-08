from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp._tests.test_compute_policy import (
    _builtin_spec,
    _cuda_environment,
    _rl_facts,
    _rl_tv_workload,
    _rl_workload,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.compute_policy import FactCompleteness, evaluate_candidate_support
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.execution_provenance import serialize_execution_provenance
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.workflow import serialize_workflow


@pytest.mark.parametrize("variant", ("rl", "tv"))
@pytest.mark.parametrize(
    "problem", ("missing", "partial", "nonfinite", "dtype", "psf_mass")
)
def test_advisory_cannot_bypass_hard_input_checks(variant, problem):
    workload = (
        _rl_workload(iterations=500)
        if variant == "rl"
        else _rl_tv_workload(iterations=100)
    )
    facts = _rl_facts()
    if problem == "missing":
        facts = ()
    elif problem == "partial":
        facts = (replace(facts[0], completeness=FactCompleteness.UNKNOWN), facts[1])
    elif problem == "nonfinite":
        facts = (
            replace(facts[0], finite_count=facts[0].element_count - 1),
            facts[1],
        )
    elif problem == "dtype":
        workload = replace(workload, input_dtypes=("uint16", "float32"))
        facts = ()
    else:
        facts = (facts[0], replace(facts[1], maximum=0.0))
    decision = evaluate_candidate_support(
        _builtin_spec(workload.operation_id),
        workload,
        _cuda_environment(),
        allow_experimental=False,
        array_facts=facts,
    )
    assert not decision.supported
    assert not decision.parity_warnings


@pytest.mark.parametrize(
    "variant,parameter,value",
    (
        ("rl", "iterations", 0),
        ("rl", "iterations", 501),
        ("rl", "iterations", True),
        ("rl", "filter_epsilon", -1),
        ("rl", "filter_epsilon", float("nan")),
        ("rl", "normalize_psf", "false"),
        ("tv", "iterations", 101),
        ("tv", "tv_regularization", -0.1),
        ("tv", "tv_regularization", 0.2),
        ("tv", "tv_epsilon", 0),
        ("tv", "denominator_floor", 0),
        ("tv", "filter_epsilon", float("inf")),
    ),
)
def test_invalid_authored_parameters_remain_rejected(variant, parameter, value):
    workload = _rl_workload() if variant == "rl" else _rl_tv_workload()
    parameters = dict(workload.parameters)
    parameters[parameter] = value
    if isinstance(value, float) and not np.isfinite(value):
        with pytest.raises(ValueError, match="NaN or infinity"):
            replace(workload, parameters=tuple(sorted(parameters.items())))
        return
    workload = replace(workload, parameters=tuple(sorted(parameters.items())))
    decision = evaluate_candidate_support(
        _builtin_spec(workload.operation_id),
        workload,
        _cuda_environment(),
        allow_experimental=False,
        array_facts=_rl_facts(),
    )
    assert not decision.supported
    assert not decision.fallback_allowed
    assert not decision.exact_workload_test_allowed


@pytest.fixture(scope="module")
def cuda_registry():
    registry = ComputeRegistry()
    try:
        runtime = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime.available or not runtime.selected_device_id:
            pytest.skip(runtime.message or "CUDA unavailable")
        library = registry.probe_library("cupyx", refresh=True)
        if not library.available:
            pytest.skip(library.message or "CuPyX unavailable")
        yield registry, runtime.selected_device_id
    finally:
        registry.close()


@pytest.mark.parametrize(
    "operation,image_shape,psf_shape,parameters",
    (
        (
            "richardson_lucy_deconvolution",
            (16, 18),
            (3, 3),
            {"iterations": 500, "filter_epsilon": 0.0},
        ),
        ("richardson_lucy_deconvolution", (16, 18), (4, 6), {"iterations": 5}),
        (
            "richardson_lucy_deconvolution",
            (12, 14),
            (15, 17),
            {"iterations": 3, "preserve_input_scale": False},
        ),
        (
            "richardson_lucy_tv_deconvolution",
            (5, 12, 14),
            (2, 3, 4),
            {
                "iterations": 100,
                "tv_regularization": 0.1,
                "tv_epsilon": 0.001,
                "denominator_floor": 0.15,
                "filter_epsilon": 0.0,
            },
        ),
    ),
)
@pytest.mark.parametrize("mode", ("prefer_gpu", "custom"))
def test_real_broad_gpu_execution_records_advisory_without_cpu_qualification(
    cuda_registry,
    operation,
    image_shape,
    psf_shape,
    parameters,
    mode,
):
    registry, device_id = cuda_registry
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    image_source_id = next(iter(pipeline.nodes))
    psf_source = pipeline.add_node("input")
    node = pipeline.add_node(operation)
    axes = "YX" if len(image_shape) == 2 else "ZYX"
    pipeline.set_param(node.id, "spatial_mode", "2D YX" if axes == "YX" else "3D ZYX")
    for name, value in parameters.items():
        pipeline.set_param(node.id, name, value)
    assert pipeline.connect(image_source_id, node.id, target_port=0).success
    assert pipeline.connect(psf_source.id, node.id, target_port=1).success
    image = np.random.default_rng(804).uniform(0.05, 1, image_shape).astype(np.float32)
    psf = np.ones(psf_shape, dtype=np.float32) / np.float32(np.prod(psf_shape))
    image_before, psf_before = image.copy(), psf.copy()
    image.flags.writeable = psf.flags.writeable = False
    implementation = "rl-cupy-f32-v1" if "tv_" not in operation else "rl-tv-cupy-f32-v1"
    request = ComputeRequest(
        mode=mode,
        runtime_id="cuda-cupy",
        device_id=device_id,
        node_preferences={node.id: f"implementation:{implementation}"},
        fallback_policy="strict" if mode == "custom" else "visible",
    )
    workflow = serialize_workflow(pipeline)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=804,
            workflow=workflow,
            input_data=image,
            input_name="Synthetic RL warning test",
            input_metadata={"axes": axes},
            source_payloads={psf_source.id: SourcePayload(psf, {"axes": axes}, "PSF")},
            compute_request=request,
            manual_node_ids=frozenset({node.id}),
            retain_node_ids=frozenset({node.id}),
            prune_unretained=True,
        ),
        compute_registry=registry,
    )
    assert not result.error, result.error
    report = result.execution_report
    assert report is not None and report.cleanup_succeeded
    decision = next(item for item in report.actual_decisions if item.node_id == node.id)
    assert decision.runtime_id == "cuda-cupy"
    assert not decision.fallback_used
    assert decision.parity_warnings
    assert not decision.benchmark_record_digest
    output = result.pipeline.outputs[node.id]
    assert output.shape == image.shape and output.dtype == np.float32
    assert np.isfinite(output).all()
    np.testing.assert_array_equal(image, image_before)
    np.testing.assert_array_equal(psf, psf_before)
    actual_parameters = result.pipeline.nodes[node.id].params
    for name, value in pipeline.nodes[node.id].params.items():
        assert actual_parameters[name] == value
    assert actual_parameters["resolved_spatial_ndim"] == len(image_shape)
    payload = serialize_execution_provenance(request, result.pipeline, report)
    entry = next(item for item in payload["nodes"] if item["node_id"] == node.id)
    assert entry["parity_warnings"] == list(decision.parity_warnings)
    assert entry["actual_implementation"]["implementation_version"] == "2"
