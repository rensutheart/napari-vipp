from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import napari_vipp.core.execution as execution_module
from napari_vipp._tests.test_device_execution import (
    _library_descriptor,
    _runtime_descriptor,
)
from napari_vipp._tests.test_gpu_execution_integration import (
    _accelerated_request,
    _decision,
    _ShapeAwareRuntime,
    _StaticPlanner,
)
from napari_vipp.core import pipeline as pipeline_module
from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeRequest,
    DecisionKind,
    FallbackPolicy,
    OutputPortKey,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_benchmark_adapter import operation_parity
from napari_vipp.core.compute_cache import (
    build_cached_node_compute_provenance,
    cached_node_provenance_matches,
    required_scientific_dependency_ids,
)
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.compute_policy import (
    SIGMA_FILTER_FLOAT32_SQUARE_LIMIT,
    ArrayFacts,
    FactCompleteness,
    estimate_candidate_memory,
    evaluate_candidate_workload_support,
    validate_spec_policy_references,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryProbeResult,
)
from napari_vipp.core.compute_specs import AdmissionTier, compute_specs_for
from napari_vipp.core.execution import execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.operations import sigma_filter
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.snapshots import GraphSnapshot
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

SIGMA_IMPLEMENTATION_ID = "cupy-sigma-filter-v1"
SIGMA_PARAMETERS = {
    "radius": 1.75,
    "sigma_width": 1.25,
    "minimum_pixel_fraction": 0.4,
    "outlier_aware": False,
    "channel_axis": 1,
}


def _sigma_pipeline(*, channel_axis: int = 1) -> tuple[PrototypePipeline, str]:
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("sigma_filter")
    for name, value in {**SIGMA_PARAMETERS, "channel_axis": channel_axis}.items():
        pipeline.set_param(node.id, name, value)
    assert pipeline.connect("input", node.id).success
    return pipeline, node.id


def _sigma_workload(
    *,
    dtype: str = "uint16",
    shape: tuple[int, ...] = (3, 7, 11),
    channel_axis: int | None = None,
) -> WorkloadDescriptor:
    return WorkloadDescriptor(
        "sigma",
        "sigma_filter",
        (shape,),
        (dtype,),
        parameters=(
            ("radius", 2.0),
            ("sigma_width", 2.0),
            ("minimum_pixel_fraction", 0.2),
            ("outlier_aware", True),
            ("channel_axis", channel_axis),
        ),
        resolved_spatial_ndim=2,
    )


def _complete_facts(
    workload: WorkloadDescriptor,
    *,
    minimum: float = 0.0,
    maximum: float = 65_535.0,
    guarantees: tuple[str, ...] = ("nonnegative", "no-negative-zero"),
) -> tuple[ArrayFacts, ...]:
    shape = workload.input_shapes[0]
    dtype = np.dtype(workload.input_dtypes[0])
    element_count = int(np.prod(shape))
    return (
        ArrayFacts(
            shape,
            dtype.name,
            element_count,
            "sigma-source-revision",
            completeness=FactCompleteness.COMPLETE,
            finite_count=element_count,
            minimum=minimum,
            maximum=maximum,
            strides=tuple(
                int(stride) for stride in np.empty(shape, dtype=dtype).strides
            ),
            contiguous=True,
            guarantees=guarantees,
        ),
    )


def _validated_sigma_environment() -> ComputeEnvironment:
    return ComputeEnvironment(
        os_name="Windows",
        execution_mode="native",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupy"),
        runtime_versions=(("cuda-cupy", "14.1.1"), ("cupy", "14.1.1")),
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
        runtime_probe_fingerprints=(("cuda-cupy", "sigma-fake-runtime"),),
        runtime_metadata=(
            (
                "cuda-cupy",
                (
                    ("cuda_runtime_version", "13020"),
                    ("driver_version", "13030"),
                ),
            ),
        ),
        driver_version="13030",
        device_id="cuda:0",
        device_name="NVIDIA GeForce RTX 5090",
        device_class="nvidia-cuda",
        device_metadata=(("compute_capability", "12.0"),),
        memory_topology="discrete",
        total_accelerator_memory_bytes=16 * 1024**3,
        probe_status="available",
    )


def test_sigma_node_is_in_the_pipeline_palette_with_the_authored_contract() -> None:
    spec = pipeline_module.NODE_LIBRARY_BY_ID["sigma_filter"]

    assert spec in pipeline_module.PALETTE_NODE_LIBRARY
    assert (
        spec
        in pipeline_module.grouped_palette_specs()["Filtering"]["Smoothing & Denoising"]
    )
    assert spec.title == "Sigma Filter"
    assert spec.function is sigma_filter
    assert [parameter.name for parameter in spec.parameters] == [
        "radius",
        "sigma_width",
        "minimum_pixel_fraction",
        "outlier_aware",
        "channel_axis",
    ]
    assert {
        parameter.name: (parameter.default, parameter.minimum, parameter.maximum)
        for parameter in spec.parameters
    } == {
        "radius": (2.0, 0.5, 10.0),
        "sigma_width": (2.0, 0.0, 1_000_000.0),
        "minimum_pixel_fraction": (0.2, 0.0, 1.0),
        "outlier_aware": (True, 0, 1),
        "channel_axis": (-1, -1, 64),
    }
    assert "sigma_filter" in pipeline_module._POSITIONAL_YX_OPERATIONS
    assert "sigma_filter" in pipeline_module.SCALAR_DEFAULT_CHANNEL_AXIS_OPERATIONS
    assert (
        pipeline_module.operation_call_parameter_value(
            "sigma_filter",
            "channel_axis",
            -1,
        )
        is None
    )


def test_sigma_pipeline_preserves_tczyx_metadata_and_records_history() -> None:
    pipeline, node_id = _sigma_pipeline()
    source = np.arange(1 * 2 * 2 * 5 * 7, dtype=np.uint16).reshape(1, 2, 2, 5, 7)

    output = pipeline.run(
        source,
        input_metadata={"axes": "TCZYX"},
        input_name="sigma-source",
    )[node_id]
    expected = sigma_filter(source, **SIGMA_PARAMETERS)

    np.testing.assert_array_equal(output, expected)
    state = pipeline.output_states[node_id]
    assert state is not None
    assert state.axis_order == "TCZYX"
    assert state.shape == source.shape
    assert state.dtype == "uint16"
    assert state.source_name == "sigma-source"
    assert state.history[-1] == (
        "Sigma Filter: radius 1.75 px, sigma width 1.25, minimum 40%, "
        "full-mean fallback, independent channel axis 1"
    )


def test_sigma_workflow_snapshot_and_export_round_trip_execute_identically() -> None:
    pipeline, node_id = _sigma_pipeline()
    document = serialize_workflow(pipeline)
    restored_document = deserialize_workflow(document)
    restored = PrototypePipeline()
    restored.restore_graph(
        restored_document["nodes"],
        restored_document["connections"],
        restored_document["output_tunnels"],
    )
    snapshotted = GraphSnapshot.from_pipeline(restored).to_pipeline()
    source = np.arange(1 * 2 * 2 * 4 * 5, dtype=np.uint16).reshape(1, 2, 2, 4, 5)

    native = snapshotted.run(source, input_metadata={"axes": "TCZYX"})[node_id]
    code = export_pipeline_to_python(snapshotted)
    namespace: dict[str, object] = {"__name__": "exported_sigma_pipeline"}
    exec(compile(code, "<exported-sigma-pipeline>", "exec"), namespace)
    exported = namespace["run_pipeline"](
        source,
        input_metadata={"axes": "TCZYX"},
    )[node_id]

    assert restored.nodes[node_id].params == snapshotted.nodes[node_id].params
    assert restored.nodes[node_id].params == SIGMA_PARAMETERS
    assert '"operation_id":"sigma_filter"' in code
    assert '"minimum_pixel_fraction":0.4' in code
    assert namespace["OUTPUT_NODES"] == (node_id,)
    np.testing.assert_array_equal(native, sigma_filter(source, **SIGMA_PARAMETERS))
    np.testing.assert_array_equal(exported, native)


def test_sigma_pipeline_executes_through_the_batch_image_path(tmp_path: Path) -> None:
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    source = np.arange(5 * 7, dtype=np.uint16).reshape(5, 7)
    np.save(input_dir / "field.npy", source)

    pipeline, sigma_id = _sigma_pipeline(channel_axis=-1)
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "sigma")
    pipeline.set_param(output.id, "format", "npy")
    pipeline.set_param(output.id, "filename_template", "{source_stem}__{tag}")
    assert pipeline.connect(sigma_id, output.id).success
    document = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("sigma-workflow.json"),
        workflow_sha256=scientific_workflow_hash(document),
        output_dir=output_dir,
        sources=(
            BatchSourceConfig(
                node_id="input",
                title="Image Source",
                input_dir=input_dir,
                pattern="*.npy",
            ),
        ),
        outputs=(
            BatchOutputConfig(
                node_id=output.id,
                node_title="Batch Output",
                tag="sigma",
                kind="image",
                format="npy",
                subfolder="",
                filename_template="{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_workflow_snapshot=False,
        save_python_script=False,
        base_dir=tmp_path,
    )

    result = run_batch(document, config)

    assert result.summary == {
        "completed": 1,
        "partial": 0,
        "skipped": 0,
        "cancelled": 0,
        "failed": 0,
    }
    assert len(result.saved_paths) == 1
    expected = sigma_filter(
        source,
        **{**SIGMA_PARAMETERS, "channel_axis": None},
    )
    np.testing.assert_array_equal(np.load(result.saved_paths[0]), expected)


def test_sigma_compute_declaration_uses_cupy_rawkernel_without_eager_import() -> None:
    spec = compute_specs_for(
        "sigma_filter",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    registry = ComputeRegistry()
    try:
        library = registry.library_descriptor("cupy")
        assert spec.implementation_id == SIGMA_IMPLEMENTATION_ID
        assert spec.implementation_library_id == "cupy"
        assert spec.runtime_id == "cuda-cupy"
        assert spec.array_domain == "cuda-cupy"
        assert spec.callable_ref == "napari_vipp.core.gpu.cupy_sigma:sigma_filter"
        assert not spec.host_boundary
        assert spec.supports_device_residency
        assert library.display_name == "CuPy custom kernels"
        assert library.probe_ref.endswith(":_probe_cupy_library")
        assert library.interoperability_claims == (
            "cupy-array-stream-device-lifetime-v1",
        )
    finally:
        registry.close()

    source_root = Path(__file__).resolve().parents[2]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source_root), environment.get("PYTHONPATH", "")))
    )
    code = r"""
import builtins
import sys

real_import = builtins.__import__

def guarded_import(name, *args, **kwargs):
    if name == "cupy" or name.startswith("cupyx"):
        raise AssertionError(f"optional CUDA import attempted: {name}")
    return real_import(name, *args, **kwargs)

builtins.__import__ = guarded_import
from napari_vipp.core.compute_registry import ComputeRegistry
from napari_vipp.core.compute_specs import compute_specs_for

registry = ComputeRegistry()
specs = compute_specs_for(
    "sigma_filter", include_cpu=False, allow_experimental=True
)
assert specs[0].implementation_library_id == "cupy"
assert registry.library_descriptor("cupy").library_id == "cupy"
assert "cupy" not in sys.modules
assert not any(name.startswith("cupyx") for name in sys.modules)
assert "napari_vipp.core.gpu.cupy_sigma" not in sys.modules
registry.close()
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr


def test_sigma_policy_admission_facts_and_memory_contracts_are_executable() -> None:
    spec = compute_specs_for(
        "sigma_filter",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    validate_spec_policy_references(spec)
    assert set(required_scientific_dependency_ids(spec)) == {
        "napari-vipp",
        "numpy",
        "scipy",
        "scikit-image",
        "cupy",
        "cuda-runtime",
    }

    integer_workload = _sigma_workload()
    integer_support = evaluate_candidate_workload_support(spec, integer_workload)
    assert integer_support.supported
    assert "native-endian-only-v1" in spec.limitations

    non_native = evaluate_candidate_workload_support(
        spec,
        _sigma_workload(dtype=">u2"),
    )
    assert not non_native.supported
    assert not non_native.requires_complete_facts
    assert "native-endian" in non_native.reason_text
    shape, dtype_identity = execution_module._shape_and_dtype(
        np.ones((3, 7, 11), dtype=">u2"),
        None,
    )
    assert shape == (3, 7, 11)
    assert dtype_identity == ">u2"

    float_workload = _sigma_workload(dtype="float32")
    missing_facts = evaluate_candidate_workload_support(spec, float_workload)
    assert not missing_facts.supported
    assert missing_facts.requires_complete_facts
    admitted_facts = _complete_facts(
        float_workload,
        minimum=-1_000.0,
        maximum=2_000.0,
        guarantees=(),
    )
    assert evaluate_candidate_workload_support(
        spec,
        float_workload,
        array_facts=admitted_facts,
    ).supported
    unsafe_facts = _complete_facts(
        float_workload,
        minimum=0.0,
        maximum=np.nextafter(
            SIGMA_FILTER_FLOAT32_SQUARE_LIMIT,
            np.inf,
        ),
    )
    unsafe = evaluate_candidate_workload_support(
        spec,
        float_workload,
        array_facts=unsafe_facts,
    )
    assert not unsafe.supported
    assert "overflow" in unsafe.reason_text.lower()

    estimate = estimate_candidate_memory(spec, integer_workload)
    elements = int(np.prod(integer_workload.input_shapes[0]))
    expected_workspace = elements * 4 + elements * 2 + (325 * 2 * 4) + 4
    expected_peak = elements * 2 + elements * 2 + expected_workspace
    assert estimate.model_id == "cupy-sigma-filter-memory-v1"
    assert estimate.runtime_managed_peak_bytes == expected_peak
    assert estimate.total_device_peak_bytes == expected_peak
    assert estimate.host_materialization_peak_bytes == elements * 2
    assert estimate.uncertainty_bytes == 8 * 1024**2

    source_facts = admitted_facts[0]
    propagated = execution_module._propagate_shape_preserving_facts(
        "sigma_filter",
        source_facts,
        {},
        output_port=OutputPortKey("sigma", 0),
        output_dtype="float32",
    )
    assert propagated is not None
    assert propagated.completeness is FactCompleteness.COMPLETE
    assert propagated.all_finite is True
    assert propagated.minimum == source_facts.minimum
    assert propagated.maximum == source_facts.maximum
    assert "extrema-conservative-enclosure" in propagated.guarantees


def test_sigma_benchmark_dispatch_is_exact_for_integer_and_tight_for_float32() -> None:
    integer_reference = np.arange(12, dtype=np.uint16).reshape(3, 4)
    integer_candidate = integer_reference.copy()
    assert operation_parity(
        "sigma_filter",
        integer_reference,
        integer_candidate,
    ).passed
    integer_candidate[1, 2] += 1
    assert not operation_parity(
        "sigma_filter",
        integer_reference,
        integer_candidate,
    ).passed

    float_reference = np.asarray([[0.0, 1.0, 2.0], [4.0, 8.0, 16.0]], dtype=np.float32)
    float_candidate = float_reference.copy()
    float_candidate[1, 1] = np.nextafter(float_candidate[1, 1], np.float32(np.inf))
    close = operation_parity(
        "sigma_filter",
        float_reference,
        float_candidate,
        input_peak=16.0,
    )
    assert close.passed, close.detail
    float_candidate[1, 1] += np.float32(0.01)
    assert not operation_parity(
        "sigma_filter",
        float_reference,
        float_candidate,
        input_peak=16.0,
    ).passed


def test_sigma_cache_provenance_names_the_exact_cupy_implementation() -> None:
    spec = compute_specs_for(
        "sigma_filter",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    request = ComputeRequest(
        mode="selective",
        node_preferences={"sigma": f"implementation:{SIGMA_IMPLEMENTATION_ID}"},
        fallback_policy="strict",
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        allow_experimental=True,
    )
    decision = _decision("sigma", spec)
    decision = replace(
        decision,
        requested_preference=request.preference_for("sigma"),
    )
    provenance = build_cached_node_compute_provenance(
        decision,
        request,
        scientific_context_fingerprint="sigma-science-v1",
        implementation_spec=spec,
    )

    identity = provenance.actual_implementation
    assert identity.operation_id == "sigma_filter"
    assert identity.runtime_id == "cuda-cupy"
    assert identity.implementation_library_id == "cupy"
    assert identity.implementation_id == SIGMA_IMPLEMENTATION_ID
    assert identity.implementation_version == "1"
    assert identity.parity_policy_id == "sigma-dtype-parity-v1"
    assert identity.cache_equivalence_group == ""
    assert cached_node_provenance_matches(
        provenance,
        request=request,
        node_id="sigma",
        operation_id="sigma_filter",
        scientific_context_fingerprint="sigma-science-v1",
        implementation_specs=(spec,),
    )


def test_sigma_public_admission_is_visible_and_exact_pin_selects() -> None:
    spec = compute_specs_for(
        "sigma_filter",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    assert spec.admission_tier is AdmissionTier.PUBLIC_AUTO_CANDIDATE
    registry = ComputeRegistry()
    try:
        assert registry.implementations_for_operation(
            "sigma_filter",
            allow_experimental=False,
        ) == (spec,)
        assert registry.implementations_for_operation(
            "sigma_filter",
            allow_experimental=True,
        ) == (spec,)

        workload = _sigma_workload()
        public_request = ComputeRequest(
            mode="selective",
            node_preferences={
                "sigma": f"implementation:{SIGMA_IMPLEMENTATION_ID}",
            },
            allow_experimental=False,
        )
        selected = plan_compute_decisions(
            public_request,
            (workload,),
            registry=registry,
            environment=_validated_sigma_environment(),
        ).decisions[0]
        assert selected.decision_kind is DecisionKind.SELECTED
        assert not selected.fallback_used
        assert selected.implementation_library_id == "cupy"
        assert selected.implementation_id == SIGMA_IMPLEMENTATION_ID
    finally:
        registry.close()


def test_sigma_forced_gpu_path_executes_with_device_residency_and_provenance() -> None:
    pipeline, node_id = _sigma_pipeline(channel_axis=-1)
    actual_spec = compute_specs_for(
        "sigma_filter",
        include_cpu=False,
        allow_experimental=True,
    )[0]
    fake_spec = replace(
        actual_spec,
        callable_ref="napari_vipp._tests.test_device_execution:_device_copy",
    )
    runtime = _ShapeAwareRuntime()
    registry = ComputeRegistry(
        runtime_descriptors=(
            replace(
                _runtime_descriptor(),
                runtime_id="cuda-cupy",
                array_domain="cuda-cupy",
                device_domain="nvidia-cuda",
                interoperability_claims=("cupy-array-stream-device-lifetime-v1",),
            ),
        ),
        library_descriptors=(
            replace(
                _library_descriptor("cupy"),
                runtime_ids=("cuda-cupy",),
                array_domain="cuda-cupy",
                interoperability_claims=("cupy-array-stream-device-lifetime-v1",),
            ),
        ),
        implementation_specs=(fake_spec,),
        runtime_factories={"cuda-cupy": lambda: runtime},
        library_probes={
            "cupy": lambda: ImplementationLibraryProbeResult(
                "cupy",
                True,
                version="14.1.1",
            ),
        },
    )
    request = ComputeRequest(
        mode="selective",
        node_preferences={node_id: f"implementation:{SIGMA_IMPLEMENTATION_ID}"},
        fallback_policy=FallbackPolicy.STRICT,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        allow_experimental=True,
    )
    selected = replace(
        _decision(node_id, fake_spec),
        requested_preference=request.preference_for(node_id),
    )
    planner = _StaticPlanner(request, (selected,))
    source = np.full((5, 7), 42, dtype=np.uint16)
    try:
        result = execute_pipeline_request(
            _accelerated_request(
                pipeline,
                source,
                request,
                retain_node_ids=frozenset({node_id}),
                prune_unretained=True,
            ),
            compute_registry=registry,
            compute_planner=planner,
        )

        assert result.error == ""
        assert result.pipeline is not None
        assert result.execution_report is not None
        assert result.execution_report.cleanup_succeeded
        actual = next(
            decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id == node_id
        )
        assert actual.decision_kind is DecisionKind.SELECTED
        assert actual.implementation_library_id == "cupy"
        assert actual.implementation_id == SIGMA_IMPLEMENTATION_ID
        assert runtime.host_to_device_count == 1
        assert runtime.device_to_host_count == 1
        assert runtime.operation_count == 1
        assert runtime.live == {}
        np.testing.assert_array_equal(result.pipeline.outputs[node_id], source)
        state = result.pipeline.output_states[node_id]
        assert state is not None
        assert state.axis_order == "YX"
        assert state.history[-1].startswith("Sigma Filter: radius 1.75 px")
        identity = result.pipeline.node_compute_provenance[
            node_id
        ].actual_implementation
        assert identity.implementation_library_id == "cupy"
        assert identity.implementation_id == SIGMA_IMPLEMENTATION_ID
    finally:
        registry.close()


def test_real_two_sigma_nodes_form_one_resident_cuda_segment() -> None:
    registry = ComputeRegistry()
    try:
        runtime_probe = registry.probe_runtime("cuda-cupy", refresh=True)
        if not runtime_probe.available or not runtime_probe.selected_device_id:
            pytest.skip(runtime_probe.message or "CUDA runtime unavailable")
        library_probe = registry.probe_library("cupy", refresh=True)
        if not library_probe.available:
            pytest.skip(library_probe.message or "CuPy RawKernel unavailable")

        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        first = pipeline.add_node("sigma_filter")
        second = pipeline.add_node("sigma_filter")
        pipeline.set_param(first.id, "radius", 0.5)
        pipeline.set_param(second.id, "radius", 2.0)
        assert pipeline.connect("input", first.id).success
        assert pipeline.connect(first.id, second.id).success

        source = np.random.default_rng(509).integers(
            0,
            65_536,
            size=(35, 41),
            dtype=np.uint16,
        )
        expected = sigma_filter(
            sigma_filter(source, radius=0.5),
            radius=2.0,
        )
        request = ComputeRequest(
            mode="selective",
            node_preferences={
                first.id: f"implementation:{SIGMA_IMPLEMENTATION_ID}",
                second.id: f"implementation:{SIGMA_IMPLEMENTATION_ID}",
            },
            fallback_policy="strict",
            runtime_id="cuda-cupy",
            device_id=runtime_probe.selected_device_id,
            allow_experimental=True,
        )

        result = execute_pipeline_request(
            _accelerated_request(
                pipeline,
                source,
                request,
                retain_node_ids=frozenset({second.id}),
                prune_unretained=True,
            ),
            compute_registry=registry,
        )

        assert result.error == ""
        assert result.pipeline is not None
        assert result.execution_report is not None
        assert result.execution_report.cleanup_succeeded
        assert len(result.execution_report.plan.segments) == 1
        assert result.execution_report.plan.segments[0].node_ids == (
            first.id,
            second.id,
        )
        selected = {
            decision.node_id: decision
            for decision in result.execution_report.actual_decisions
            if decision.node_id in {first.id, second.id}
        }
        assert set(selected) == {first.id, second.id}
        assert all(
            decision.decision_kind is DecisionKind.SELECTED
            and decision.implementation_id == SIGMA_IMPLEMENTATION_ID
            for decision in selected.values()
        )
        np.testing.assert_array_equal(result.pipeline.outputs[second.id], expected)
        terminal = registry.runtime("cuda-cupy").memory_snapshot(
            device_id=runtime_probe.selected_device_id,
        )
        assert terminal.runtime_live_bytes == 0
        assert terminal.runtime_reserved_bytes == 0
    finally:
        registry.close()
