"""Fresh example parity for CPU and provider-free Prefer GPU planning."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import fields, is_dataclass
from unittest.mock import patch

import numpy as np
import pytest

from napari_vipp._sample_data import make_sample_data
from napari_vipp.core.batch import BatchStatus, load_batch_config, run_batch
from napari_vipp.core.batch_demo import (
    create_synthetic_batch_demo,
    synthetic_batch_demo_workflow,
    validate_synthetic_batch_demo,
)
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    FallbackReason,
)
from napari_vipp.core.compute_registry import (
    ComputeRegistry,
    ImplementationLibraryProbeResult,
    RuntimeDevice,
    RuntimeProbeResult,
)
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.pipeline import (
    EXECUTION_READY,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import load_workflow, serialize_workflow
from napari_vipp.ui.examples import EXAMPLE_WORKFLOWS, _example_workflow_path


class _UnavailableAcceleratorLibrariesRegistry(ComputeRegistry):
    """Provider-free registry with CUDA present and its libraries absent."""

    def probe_runtime(self, runtime_id: str, *, refresh: bool = False):
        del refresh
        if runtime_id == "cpu-numpy":
            return RuntimeProbeResult(
                runtime_id,
                True,
                version=np.__version__,
                devices=(RuntimeDevice("cpu:0", "Test host CPU"),),
                selected_device_id="cpu:0",
                environment_fingerprint="test-cpu-environment",
            )
        assert runtime_id == "cuda-cupy"
        return RuntimeProbeResult(
            runtime_id,
            True,
            version="14.1.1",
            devices=(
                RuntimeDevice(
                    "cuda:0",
                    "Provider-free test CUDA device",
                    8 * 1024**3,
                    metadata=(("compute_capability", "12.0"),),
                ),
            ),
            selected_device_id="cuda:0",
            environment_fingerprint="test-cuda-environment",
            metadata=(
                ("cuda_runtime_version", "13020"),
                ("driver_version", "13030"),
            ),
        )

    def probe_library(self, library_id: str, *, refresh: bool = False):
        del refresh
        if library_id == "cpu":
            return ImplementationLibraryProbeResult(
                library_id,
                True,
                version=np.__version__,
            )
        return ImplementationLibraryProbeResult(
            library_id,
            False,
            reason_code="library_unavailable",
            message=f"{library_id} is unavailable in this provider-free test.",
        )


_VALIDATED_TEST_HOST = ComputeEnvironment(
    os_name="Windows",
    os_release="test",
    execution_mode="native",
    python_implementation="CPython",
    python_version="3.12",
    python_abi="cpython-312",
    runtime_ids=("cpu-numpy", "cuda-cupy"),
    implementation_libraries=("cpu",),
    scientific_stack_versions=(
        ("numpy", "2.5.1"),
        ("scipy", "1.18.0"),
        ("scikit-image", "0.26.0"),
    ),
    device_id="cuda:0",
    device_name="Provider-free test CUDA device",
    device_class="nvidia-cuda",
    total_accelerator_memory_bytes=8 * 1024**3,
    probe_status="available",
)


@pytest.fixture(scope="module")
def sample_catalog():
    return {
        layer_kwargs["name"]: (data, layer_kwargs)
        for data, layer_kwargs, _layer_type in make_sample_data()
    }


def _restore_example(spec) -> PrototypePipeline:
    workflow = load_workflow(_example_workflow_path(spec))
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        workflow["nodes"],
        workflow["connections"],
        workflow.get("output_tunnels", ()),
    )
    assert not pipeline.completed_node_ids
    assert not any(
        output is not None
        for outputs in pipeline.node_outputs.values()
        for output in outputs
    )
    return pipeline


def _batch_source_payloads() -> dict[str, SourcePayload]:
    primary = np.zeros((8, 8), dtype=np.uint16)
    primary[1:4, 1:4] = 100
    reference = np.zeros((8, 8), dtype=np.uint16)
    reference[2:5, 2:5] = 200
    return {
        "input": SourcePayload(
            primary,
            {"vipp_axis_order": "YX"},
            "primary",
        ),
        "input_2": SourcePayload(
            reference,
            {"vipp_axis_order": "YX"},
            "reference",
        ),
    }


def _source_payloads(spec, pipeline, sample_catalog) -> dict[str, SourcePayload]:
    if spec.generated_batch_demo:
        payloads = _batch_source_payloads()
    else:
        source_ids = tuple(
            "input" if index == 0 else f"input_{index + 1}"
            for index in range(len(spec.samples))
        )
        payloads = {}
        for source_id, sample_name in zip(source_ids, spec.samples, strict=True):
            data, layer_kwargs = sample_catalog[sample_name]
            payloads[source_id] = SourcePayload(
                np.array(data, copy=True),
                deepcopy(layer_kwargs["metadata"]),
                layer_kwargs["name"],
            )

    source_node_ids = {
        node_id
        for node_id, node in pipeline.nodes.items()
        if node.operation_id == "input"
    }
    assert set(payloads) == source_node_ids
    return payloads


def _execute_example(spec, sample_catalog, mode: ComputeMode):
    pipeline = _restore_example(spec)
    manual_node_ids = frozenset(pipeline.manual_node_ids())
    request = PipelineRunRequest(
        run_id=1 if mode is ComputeMode.CPU else 2,
        workflow=serialize_workflow(pipeline),
        input_data=None,
        input_metadata={},
        input_name="",
        source_payloads=_source_payloads(spec, pipeline, sample_catalog),
        compute_request=ComputeRequest(mode=mode),
        manual_node_ids=manual_node_ids,
    )

    if mode is ComputeMode.CPU:
        result = execute_pipeline_request(request)
    else:
        with (
            _UnavailableAcceleratorLibrariesRegistry() as registry,
            patch(
                "napari_vipp.core.compute_planning.ComputeEnvironment",
                return_value=_VALIDATED_TEST_HOST,
            ),
        ):
            result = execute_pipeline_request(
                request,
                compute_registry=registry,
            )

    assert result.error == ""
    assert result.cancelled is False
    assert result.failure is None
    assert result.pipeline is not None
    completed = result.pipeline
    assert completed.completed_node_ids == frozenset(completed.nodes)
    assert manual_node_ids <= completed.completed_node_ids
    assert set(completed.node_outputs) == set(completed.nodes)
    for node_id in completed.topological_order():
        assert completed.node_execution_states[node_id] == EXECUTION_READY
        assert len(completed.node_outputs[node_id]) == len(
            completed.output_ports(node_id)
        )
        assert all(output is not None for output in completed.node_outputs[node_id])
        assert len(completed.node_output_states[node_id]) == len(
            completed.node_outputs[node_id]
        )

    assert result.execution_report is not None
    assert result.execution_report.request.mode is mode
    assert result.execution_report.cleanup_succeeded is True
    assert result.execution_report.fallback_records == ()
    assert result.execution_report.actual_decisions
    decision_node_ids = {
        decision.node_id for decision in result.execution_report.actual_decisions
    }
    source_node_ids = {
        node_id
        for node_id, node in completed.nodes.items()
        if node.operation_id == "input"
    }
    expected_decision_node_ids = set(completed.nodes) - source_node_ids
    if mode is not ComputeMode.CPU:
        expected_decision_node_ids |= source_node_ids
    assert expected_decision_node_ids == decision_node_ids
    assert len(decision_node_ids) == len(result.execution_report.actual_decisions)
    safe_prefer_gpu_reasons = {
        DecisionReason.NO_VALIDATED_IMPLEMENTATION,
        DecisionReason.ENVIRONMENT_UNSUPPORTED,
        DecisionReason.DEPENDENCY_UNAVAILABLE,
        DecisionReason.WORKLOAD_UNSUPPORTED,
        DecisionReason.MEMORY_LIMIT,
    }
    for decision in result.execution_report.actual_decisions:
        assert decision.runtime_id == "cpu-numpy"
        assert decision.implementation_library_id == "cpu"
        assert decision.decision_kind is DecisionKind.POLICY_CPU
        assert decision.fallback_used is False
        assert decision.fallback_reason is FallbackReason.NONE
        assert decision.reason_text.strip()
        if mode is ComputeMode.CPU:
            assert decision.reason is DecisionReason.EXPLICIT_CPU
        elif decision.node_id not in source_node_ids:
            assert decision.reason in safe_prefer_gpu_reasons
    for connection in completed.connections:
        assert connection.source_port < len(
            completed.node_outputs[connection.source_id]
        )
    for tunnel in completed.output_tunnels.values():
        assert tunnel.source_port < len(completed.node_outputs[tunnel.source_id])
    return completed


def _assert_equivalent(actual, expected, *, path: str) -> None:
    assert type(actual) is type(expected), path
    if isinstance(actual, np.ndarray):
        assert actual.shape == expected.shape, path
        assert actual.dtype == expected.dtype, path
        np.testing.assert_array_equal(actual, expected, err_msg=path)
        return
    if isinstance(actual, TableData):
        assert actual.columns == expected.columns, path
        assert actual.name == expected.name, path
        assert actual.table_kind == expected.table_kind, path
        assert actual.source_name == expected.source_name, path
        assert actual.column_units == expected.column_units, path
        _assert_equivalent(
            actual.records(),
            expected.records(),
            path=f"{path}.records",
        )
        return
    if isinstance(actual, Mapping):
        assert set(actual) == set(expected), path
        for key in actual:
            _assert_equivalent(
                actual[key],
                expected[key],
                path=f"{path}[{key!r}]",
            )
        return
    if isinstance(actual, (tuple, list)):
        assert len(actual) == len(expected), path
        for index, (actual_item, expected_item) in enumerate(
            zip(actual, expected, strict=True)
        ):
            _assert_equivalent(
                actual_item,
                expected_item,
                path=f"{path}[{index}]",
            )
        return
    if is_dataclass(actual) and not isinstance(actual, type):
        for field in fields(actual):
            _assert_equivalent(
                getattr(actual, field.name),
                getattr(expected, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(actual, (float, complex, np.number)):
        np.testing.assert_equal(actual, expected, err_msg=path)
        return
    assert actual == expected, path


@pytest.mark.parametrize(
    "spec",
    EXAMPLE_WORKFLOWS,
    ids=lambda spec: spec.id,
)
def test_fresh_example_cpu_and_prefer_gpu_outputs_match(spec, sample_catalog):
    cpu_pipeline = _execute_example(spec, sample_catalog, ComputeMode.CPU)
    prefer_gpu_pipeline = _execute_example(
        spec,
        sample_catalog,
        ComputeMode.PREFER_GPU,
    )

    assert cpu_pipeline.topological_order() == prefer_gpu_pipeline.topological_order()
    for node_id in cpu_pipeline.topological_order():
        cpu_outputs = cpu_pipeline.node_outputs[node_id]
        prefer_gpu_outputs = prefer_gpu_pipeline.node_outputs[node_id]
        assert len(cpu_outputs) == len(prefer_gpu_outputs)
        for port_index, (cpu_output, prefer_gpu_output) in enumerate(
            zip(cpu_outputs, prefer_gpu_outputs, strict=True)
        ):
            _assert_equivalent(
                prefer_gpu_output,
                cpu_output,
                path=f"{spec.id}.{node_id}[{port_index}]",
            )
        _assert_equivalent(
            prefer_gpu_pipeline.node_output_states[node_id],
            cpu_pipeline.node_output_states[node_id],
            path=f"{spec.id}.{node_id}.states",
        )


def test_compute_matrix_covers_every_bundled_example():
    assert len(EXAMPLE_WORKFLOWS) == 13
    ids = [spec.id for spec in EXAMPLE_WORKFLOWS]
    filenames = [spec.filename for spec in EXAMPLE_WORKFLOWS]
    assert len(ids) == len(set(ids))
    assert len(filenames) == len(set(filenames))
    example_directory = _example_workflow_path(EXAMPLE_WORKFLOWS[0]).parent
    assert set(filenames) == {
        path.name for path in example_directory.glob("*.json")
    }


@pytest.mark.parametrize(
    "mode",
    (ComputeMode.CPU, ComputeMode.PREFER_GPU),
    ids=("cpu", "prefer-gpu"),
)
def test_full_synthetic_batch_demo_completes_in_each_compute_mode(tmp_path, mode):
    demo = create_synthetic_batch_demo(tmp_path / "bundle")
    config = load_batch_config(demo.config_path)
    request = ComputeRequest(mode=mode)

    if mode is ComputeMode.CPU:
        result = run_batch(
            synthetic_batch_demo_workflow(),
            config,
            workflow_path=demo.workflow_path,
            config_path=demo.config_path,
            compute_request=request,
        )
    else:
        with (
            _UnavailableAcceleratorLibrariesRegistry() as registry,
            patch(
                "napari_vipp.core.compute_planning.ComputeEnvironment",
                return_value=_VALIDATED_TEST_HOST,
            ),
        ):
            result = run_batch(
                synthetic_batch_demo_workflow(),
                config,
                workflow_path=demo.workflow_path,
                config_path=demo.config_path,
                compute_request=request,
                compute_registry=registry,
            )

    validation = validate_synthetic_batch_demo(demo, result=result)
    assert validation.ok
    assert result.has_failures is False
    assert result.summary == {
        "completed": 3,
        "partial": 0,
        "skipped": 0,
        "cancelled": 0,
        "failed": 0,
    }
    assert len(result.saved_paths) == 9
    assert result.manifest.compute["configured_request"]["mode"] == "cpu"
    assert result.manifest.compute["effective_request"]["mode"] == mode.value
    assert result.manifest.compute["override_used"] is True
    assert result.manifest.compute["runtime_cleanup_succeeded"] is True

    expected_nodes = {
        "binary_threshold_1": "binary_threshold",
        "binary_threshold_2": "binary_threshold",
        "logical_and_1": "logical_and",
        "label_connected_components_1": "label_connected_components",
        "measure_objects_1": "measure_objects",
        "add_images_1": "add_images",
        "batch_output_1": "batch_output",
        "batch_output_2": "batch_output",
        "batch_output_3": "batch_output",
    }
    for item in result.manifest.items:
        assert item.status is BatchStatus.COMPLETED
        execution = item.execution
        assert execution["request"]["mode"] == mode.value
        assert execution["outcome"] == execution["status"] == "completed"
        assert execution["failure"] is None
        assert execution["cleanup_succeeded"] is True
        assert execution["fallbacks"] == []
        assert execution["fallback_records"] == []
        nodes = {node["node_id"]: node for node in execution["nodes"]}
        assert {
            node_id: node["operation_id"] for node_id, node in nodes.items()
        } == expected_nodes
        for node in nodes.values():
            assert node["decision_kind"] == DecisionKind.POLICY_CPU.value
            assert node["fallback_used"] is False
            assert node["fallback_reason"] == FallbackReason.NONE.value
            assert node["reason_text"].strip()
            actual = node["actual_implementation"]
            assert actual["runtime_id"] == "cpu-numpy"
            assert actual["implementation_library_id"] == "cpu"
            assert actual["implementation_id"] == (
                f"cpu-{node['operation_id']}-v1"
            )

        if mode is ComputeMode.CPU:
            assert all(
                node["reason"] == DecisionReason.EXPLICIT_CPU.value
                for node in nodes.values()
            )
            continue

        no_implementation_nodes = {
            "binary_threshold_1",
            "binary_threshold_2",
            "add_images_1",
        }
        assert all(
            nodes[node_id]["reason"]
            == DecisionReason.NO_VALIDATED_IMPLEMENTATION.value
            for node_id in no_implementation_nodes
        )
        deferred_nodes = set(nodes) - no_implementation_nodes
        assert all(
            nodes[node_id]["reason"]
            == DecisionReason.WORKLOAD_UNSUPPORTED.value
            for node_id in deferred_nodes
        )
