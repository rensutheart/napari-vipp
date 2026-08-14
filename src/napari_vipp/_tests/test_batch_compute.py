from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

import napari_vipp.core.batch as batch_module
from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_CONFIG_VERSION,
    BatchConfig,
    BatchOutputConfig,
    BatchScientificPreflightError,
    BatchSourceConfig,
    BatchStatus,
    ExistingFilePolicy,
    batch_config_hash,
    effective_batch_config_hash,
    load_batch_config,
    run_batch,
    save_batch_config,
    scientific_workflow_hash,
)
from napari_vipp.core.compute import (
    ComputeEnvironment,
    ComputeMode,
    ComputeRequest,
    DecisionKind,
    DecisionReason,
    ExecutionFallbackRecord,
    ExecutionReport,
    FallbackReason,
    MemoryEstimate,
    MemoryTopology,
    NodeExecutionDecision,
    WorkloadDescriptor,
)
from napari_vipp.core.compute_planning import plan_compute_decisions
from napari_vipp.core.execution import (
    PipelineExecutionFailure,
    PipelineRunResult,
)
from napari_vipp.core.execution_provenance import serialize_execution_provenance
from napari_vipp.core.metadata import AxisDeclaration
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow


def _image_batch(tmp_path, *, item_count: int = 1):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for index in range(item_count):
        np.save(
            inputs / f"{index + 1:02d}.npy",
            np.arange(20, dtype=np.uint16).reshape(4, 5) + index,
        )
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    output = pipeline.add_node("batch_output")
    assert pipeline.connect("input", output.id).success
    pipeline.set_param(output.id, "tag", "result")
    pipeline.set_param(output.id, "format", "npy")
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Input", inputs, "*.npy"),),
        outputs=(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "result",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
    )
    return workflow, config, output.id


def _generic_stack_batch(
    tmp_path,
    *,
    mode: ComputeMode,
    continue_on_error: bool,
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    source_path = inputs / "ordinary-stack.tif"
    tifffile.imwrite(
        source_path,
        np.arange(3 * 8 * 9, dtype=np.uint16).reshape(3, 8, 9),
        photometric="minisblack",
    )
    with tifffile.TiffFile(source_path) as tif:
        assert tif.series[0].axes == "QYX"

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params["binding_mode"] = "collection"
    background = pipeline.add_node("subtract_background")
    pipeline.set_param(background.id, "radius", 1.0)
    pipeline.set_param(background.id, "spatial_mode", "3D ZYX")
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "result")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect("input", background.id).success
    assert pipeline.connect(background.id, output.id).success
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Input", inputs, "*.tif"),),
        outputs=(
            BatchOutputConfig(
                output.id,
                output.title,
                "result",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
        continue_on_error=continue_on_error,
        compute_request=ComputeRequest(mode=mode),
    )
    return workflow, config


def test_batch_config_roundtrip_preserves_segmentation_bridge_compute_intent(
    tmp_path,
):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "mask-source.npy", np.zeros((32, 48), dtype=np.float32))

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    threshold = pipeline.add_node("binary_threshold")
    remove_small = pipeline.add_node("remove_small_objects")
    fill = pipeline.add_node("fill_holes")
    components = pipeline.add_node("label_connected_components")
    output = pipeline.add_node("batch_output")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(remove_small.id, "min_size", 8)
    pipeline.set_param(remove_small.id, "spatial_mode", "2D YX")
    pipeline.set_param(remove_small.id, "connectivity", "Face connected")
    pipeline.set_param(fill.id, "max_hole_size", 0)
    pipeline.set_param(fill.id, "spatial_mode", "2D YX")
    pipeline.set_param(fill.id, "connectivity", "Face connected")
    pipeline.set_param(components.id, "spatial_mode", "2D YX")
    pipeline.set_param(components.id, "connectivity", "Full connectivity")
    pipeline.set_param(output.id, "tag", "labels")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect("input", threshold.id).success
    assert pipeline.connect(threshold.id, remove_small.id).success
    assert pipeline.connect(remove_small.id, fill.id).success
    assert pipeline.connect(fill.id, components.id).success
    assert pipeline.connect(components.id, output.id).success

    workflow = serialize_workflow(pipeline)
    request = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={
            threshold.id: (
                "implementation:cupy-binary-threshold-f32-exact-v1"
            ),
            remove_small.id: (
                "implementation:cupyx-remove-small-objects-bool-v1"
            ),
            fill.id: "implementation:cupyx-fill-holes-all-v1",
            components.id: "implementation:cupyx-connected-components-v1",
        },
        fallback_policy="visible",
    )
    configured = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Input", inputs, "*.npy"),),
        outputs=(
            BatchOutputConfig(
                output.id,
                output.title,
                "labels",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
        compute_request=request,
    )
    config_path = tmp_path / BATCH_CONFIG_FILENAME

    save_batch_config(config_path, configured)
    loaded = load_batch_config(config_path)

    assert loaded.compute_request == request
    assert workflow["version"] == 4
    assert loaded.workflow_sha256 == scientific_workflow_hash(workflow)
    assert {
        node["id"]: node["operation_id"] for node in workflow["nodes"]
    } == {
        "input": "input",
        threshold.id: "binary_threshold",
        remove_small.id: "remove_small_objects",
        fill.id: "fill_holes",
        components.id: "label_connected_components",
        output.id: "batch_output",
    }
    assert [
        (connection["source"], connection["target"])
        for connection in workflow["connections"]
    ] == [
        ("input", threshold.id),
        (threshold.id, remove_small.id),
        (remove_small.id, fill.id),
        (fill.id, components.id),
        (components.id, output.id),
    ]

    shape = (32, 48)
    workloads = (
        WorkloadDescriptor(
            threshold.id,
            "binary_threshold",
            (shape,),
            ("float32",),
            parameters=(("channel_axis", None), ("threshold", 0.5)),
            resolved_spatial_ndim=2,
            resident_successors=(remove_small.id,),
        ),
        WorkloadDescriptor(
            remove_small.id,
            "remove_small_objects",
            (shape,),
            ("bool",),
            parameters=(
                ("min_size", 8),
                ("spatial_mode", "2D YX"),
                ("connectivity", "Face connected"),
            ),
            resolved_spatial_ndim=2,
            resident_predecessors=(threshold.id,),
            resident_successors=(fill.id,),
        ),
        WorkloadDescriptor(
            fill.id,
            "fill_holes",
            (shape,),
            ("bool",),
            parameters=(
                ("max_hole_size", 0),
                ("spatial_mode", "2D YX"),
                ("connectivity", "Face connected"),
            ),
            resolved_spatial_ndim=2,
            resident_predecessors=(remove_small.id,),
            resident_successors=(components.id,),
        ),
        WorkloadDescriptor(
            components.id,
            "label_connected_components",
            (shape,),
            ("bool",),
            parameters=(
                ("spatial_mode", "2D YX"),
                ("connectivity", "Full connectivity"),
            ),
            resolved_spatial_ndim=2,
            resident_predecessors=(fill.id,),
            required_host_boundaries=1,
        ),
    )
    environment = ComputeEnvironment(
        os_name="Windows",
        python_implementation="CPython",
        python_version="3.12",
        python_abi="cpython-312",
        runtime_ids=("cpu-numpy", "cuda-cupy"),
        implementation_libraries=("cpu", "cupy", "cupyx"),
        runtime_versions=(
            ("cuda-cupy", "14.1.1"),
            ("cupy", "14.1.1"),
            ("cupyx", "14.1.1"),
        ),
        scientific_stack_versions=(
            ("numpy", "2.5.1"),
            ("scipy", "1.18.0"),
            ("scikit-image", "0.26.0"),
        ),
        runtime_probe_fingerprints=(("cuda-cupy", "test-fingerprint"),),
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

    planned = plan_compute_decisions(
        loaded.compute_request,
        workloads,
        environment=environment,
    )

    assert {
        decision.node_id: (decision.runtime_id, decision.implementation_id)
        for decision in planned.decisions
    } == {
        threshold.id: (
            "cuda-cupy",
            "cupy-binary-threshold-f32-exact-v1",
        ),
        remove_small.id: (
            "cuda-cupy",
            "cupyx-remove-small-objects-bool-v1",
        ),
        fill.id: ("cuda-cupy", "cupyx-fill-holes-all-v1"),
        components.id: ("cuda-cupy", "cupyx-connected-components-v1"),
    }


def test_batch_config_v3_roundtrip_and_v1_v2_migration_preserve_replay(tmp_path):
    workflow, config, _output_id = _image_batch(tmp_path)
    auto = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        accelerator_memory_cap_bytes=2_000_000_000,
        accelerator_safety_reserve_bytes=250_000_000,
        allow_experimental=True,
    )
    current = replace(
        config,
        sources=(
            replace(
                config.sources[0],
                axis_declaration=AxisDeclaration("QYX", "ZYX"),
            ),
        ),
        compute_request=auto,
    )
    path = tmp_path / BATCH_CONFIG_FILENAME

    save_batch_config(path, current)
    loaded = load_batch_config(path)

    assert loaded.compute_request == auto
    assert loaded.to_dict()["version"] == BATCH_CONFIG_VERSION
    assert loaded.to_dict()["compute"] == auto.as_dict()
    assert loaded.sources[0].axis_declaration == AxisDeclaration("QYX", "ZYX")
    assert loaded.to_dict()["sources"][0]["axis_declaration"] == {
        "source_axes": "QYX",
        "effective_axes": "ZYX",
    }

    version_two_document = current.to_dict()
    version_two_document["version"] = 2
    version_two_document["sources"][0].pop("axis_declaration")
    path.write_text(json.dumps(version_two_document), encoding="utf-8")

    migrated_v2 = load_batch_config(path)

    assert migrated_v2.compute_request == auto
    assert migrated_v2.sources[0].axis_declaration is None
    assert migrated_v2.to_dict()["version"] == BATCH_CONFIG_VERSION

    version_one_document = dict(version_two_document)
    version_one_document["version"] = 1
    version_one_document.pop("compute")
    path.write_text(json.dumps(version_one_document), encoding="utf-8")

    migrated_v1 = load_batch_config(path)

    assert migrated_v1.compute_request.mode is ComputeMode.CPU
    assert migrated_v1.sources[0].axis_declaration is None
    assert migrated_v1.to_dict()["version"] == BATCH_CONFIG_VERSION
    assert migrated_v1.workflow_sha256 == scientific_workflow_hash(workflow)


@pytest.mark.parametrize(
    "mode",
    (ComputeMode.CPU, ComputeMode.AUTO, ComputeMode.PREFER_GPU),
)
@pytest.mark.parametrize("continue_on_error", (False, True))
def test_axis_preflight_precedes_executor_registry_and_item_failure_policy(
    tmp_path,
    monkeypatch,
    mode,
    continue_on_error,
):
    workflow, config = _generic_stack_batch(
        tmp_path,
        mode=mode,
        continue_on_error=continue_on_error,
    )
    progress = []

    def forbidden_execute(*_args, **_kwargs):
        raise AssertionError("scientific preflight reached item execution")

    def forbidden_registry(*_args, **_kwargs):
        raise AssertionError("scientific preflight allocated a GPU registry")

    monkeypatch.setattr(
        batch_module,
        "execute_pipeline_request",
        forbidden_execute,
    )
    import napari_vipp.core.compute_registry as registry_module

    monkeypatch.setattr(registry_module, "ComputeRegistry", forbidden_registry)

    with pytest.raises(
        BatchScientificPreflightError,
        match=r"before item processing, output creation, or CPU/GPU device setup",
    ) as caught:
        run_batch(
            workflow,
            config,
            progress_callback=lambda *update: progress.append(update),
        )

    message = str(caught.value)
    assert "raw QYX, effective QYX" in message
    assert "QYX -> ZYX" in message
    assert progress == []
    assert not config.output_dir.exists()


def test_prefer_gpu_batch_config_and_provenance_preserve_global_policy(tmp_path):
    workflow, config, output_id = _image_batch(tmp_path)
    request = ComputeRequest(mode=ComputeMode.PREFER_GPU)
    current = replace(config, compute_request=request)
    path = tmp_path / BATCH_CONFIG_FILENAME

    save_batch_config(path, current)
    loaded = load_batch_config(path)
    result = run_batch(workflow, loaded)

    assert loaded.compute_request == request
    assert loaded.to_dict()["compute"]["mode"] == "prefer_gpu"
    assert loaded.to_dict()["compute"]["fallback_policy"] == "visible"
    assert result.manifest.compute["configured_request"]["mode"] == "prefer_gpu"
    assert result.manifest.compute["effective_request"]["mode"] == "prefer_gpu"
    item = result.manifest.items[0]
    assert item.execution["request"]["mode"] == "prefer_gpu"
    decision = next(
        node for node in item.execution["nodes"] if node["node_id"] == output_id
    )
    assert decision["actual_implementation"]["runtime_id"] == "cpu-numpy"
    assert decision["fallback_used"] is False


def test_cpu_batch_records_exact_node_provenance_and_links_output(tmp_path):
    workflow, config, output_id = _image_batch(tmp_path)

    result = run_batch(workflow, config)

    item = result.manifest.items[0]
    assert item.execution["request"]["mode"] == "cpu"
    assert item.execution["cleanup_succeeded"] is True
    assert item.execution_provenance_sha256
    assert item.outputs[0].execution_provenance_sha256 == (
        item.execution_provenance_sha256
    )
    assert len(item.execution["nodes"]) == 1
    node = item.execution["nodes"][0]
    assert node["node_id"] == output_id
    assert node["operation_id"] == "batch_output"
    assert node["decision_kind"] == "policy_cpu"
    assert node["reason"] == "explicit_cpu"
    assert node["fallback_used"] is False
    assert node["actual_implementation"] == {
        "identity_complete": True,
        "runtime_id": "cpu-numpy",
        "array_domain": "host-numpy",
        "implementation_library_id": "cpu",
        "implementation_id": "cpu-batch_output-v1",
        "implementation_version": "1",
        "parity_policy_id": "authoritative-cpu-v1",
        "cache_equivalence_group": "",
    }
    assert all(
        node["operation_id"] != "input" for node in item.execution["nodes"]
    )


def test_run_override_changes_effective_hash_and_records_typed_fallback_progress(
    tmp_path,
    monkeypatch,
):
    workflow, config, output_id = _image_batch(tmp_path)
    requested = ComputeRequest(mode=ComputeMode.AUTO)
    original_execute = batch_module.execute_pipeline_request
    progress = []

    def execute_with_visible_fallback(request, **kwargs):
        cpu_result = original_execute(
            replace(request, compute_request=ComputeRequest(mode=ComputeMode.CPU)),
            **kwargs,
        )
        kwargs["node_started_callback"](output_id)
        kwargs["progress_callback"]("batch_output", 1, 2, "writing result")
        decision = NodeExecutionDecision(
            node_id=output_id,
            operation_id="batch_output",
            requested_preference=request.compute_request.preference_for(output_id),
            runtime_id="cpu-numpy",
            implementation_library_id="cpu",
            implementation_id="cpu-batch_output-v1",
            decision_kind=DecisionKind.FALLBACK_CPU,
            reason=DecisionReason.OUT_OF_MEMORY_FALLBACK,
            reason_text="The device segment ran out of memory and retried on CPU.",
            fallback_reason=FallbackReason.OUT_OF_MEMORY,
        )
        return PipelineRunResult(
            request.run_id,
            request.workflow,
            pipeline=cpu_result.pipeline,
            execution_report=ExecutionReport(
                request=request.compute_request,
                environment=ComputeEnvironment(),
                actual_decisions=(decision,),
                fallback_records=(
                    ExecutionFallbackRecord(
                        segment_id="cuda-cupy:001",
                        runtime_id="cuda-cupy",
                        node_ids=(output_id,),
                        reason=FallbackReason.OUT_OF_MEMORY,
                        reason_code="cuda_memory_allocation",
                        exception_type="OutOfMemoryError",
                        message="allocation failed",
                        device_attempt_count=1,
                        cpu_retry_count=1,
                        cleanup_succeeded=True,
                        memory_estimate=MemoryEstimate(
                            runtime_managed_peak_bytes=1024,
                            total_device_peak_bytes=2048,
                            model_id="test-model-v1",
                        ),
                        memory_topology=MemoryTopology.DISCRETE,
                        device_total_bytes=8192,
                        device_free_bytes=512,
                        runtime_live_bytes=0,
                        runtime_reserved_bytes=256,
                        out_of_pool_bytes=64,
                    ),
                ),
                warnings=("Visible CPU fallback used.",),
            ),
        )

    monkeypatch.setattr(
        batch_module,
        "execute_pipeline_request",
        execute_with_visible_fallback,
    )

    result = run_batch(
        workflow,
        config,
        compute_request=requested,
        execution_progress_callback=progress.append,
    )

    assert result.manifest.config_sha256 == batch_config_hash(config)
    assert result.manifest.effective_config_sha256 == effective_batch_config_hash(
        config,
        requested,
    )
    assert result.manifest.effective_config_sha256 != result.manifest.config_sha256
    assert result.manifest.compute["override_used"] is True
    assert result.manifest.compute["effective_request"]["mode"] == "auto"
    fallback = result.manifest.items[0].execution["fallbacks"][0]
    assert fallback["node_id"] == output_id
    assert fallback["fallback_reason"] == "out_of_memory"
    assert fallback["out_of_memory"] is True
    runtime_fallback = result.manifest.items[0].execution["fallback_records"][0]
    assert runtime_fallback["reason_code"] == "cuda_memory_allocation"
    assert runtime_fallback["segment_id"] == "cuda-cupy:001"
    assert runtime_fallback["runtime_id"] == "cuda-cupy"
    assert runtime_fallback["node_ids"] == [output_id]
    assert runtime_fallback["device_attempt_count"] == 1
    assert runtime_fallback["cpu_retry_count"] == 1
    assert runtime_fallback["cleanup_succeeded"] is True
    assert runtime_fallback["memory_estimate"]["model_id"] == "test-model-v1"
    assert runtime_fallback["memory_topology"] == "discrete"
    assert runtime_fallback["device_total_bytes"] == 8192
    assert runtime_fallback["device_free_bytes"] == 512
    assert runtime_fallback["runtime_reserved_bytes"] == 256
    assert runtime_fallback["out_of_pool_bytes"] == 64
    operation_update = next(
        update for update in progress if update.message == "writing result"
    )
    assert operation_update.node_id == output_id
    assert operation_update.current == 1
    assert operation_update.total == 2


def test_cleanup_failure_blocks_staging_and_publication(tmp_path, monkeypatch):
    workflow, config, _output_id = _image_batch(tmp_path, item_count=2)
    original_execute = batch_module.execute_pipeline_request
    execution_calls = 0

    def execute_with_failed_cleanup(request, **kwargs):
        nonlocal execution_calls
        execution_calls += 1
        result = original_execute(request, **kwargs)
        return replace(
            result,
            execution_report=replace(
                result.execution_report
                or ExecutionReport(
                    request=request.compute_request,
                    environment=ComputeEnvironment(),
                ),
                cleanup_succeeded=False,
            ),
        )

    monkeypatch.setattr(
        batch_module,
        "execute_pipeline_request",
        execute_with_failed_cleanup,
    )
    save_calls = 0

    def forbidden_save(*_args, **_kwargs):
        nonlocal save_calls
        save_calls += 1
        raise AssertionError("cleanup failure must block private staging")

    monkeypatch.setattr(batch_module, "_save_planned_output", forbidden_save)

    result = run_batch(workflow, config)

    assert execution_calls == 1
    assert save_calls == 0
    assert result.saved_paths == ()
    assert result.manifest.items[0].status is BatchStatus.FAILED
    assert result.manifest.items[0].error_type == "BatchRuntimeCleanupError"
    assert result.manifest.items[1].status is BatchStatus.SKIPPED
    assert "runtime is no longer trusted" in result.manifest.items[1].error_message
    assert not list(config.output_dir.glob("*.npy"))


def test_compute_preflight_failure_does_not_poison_later_batch_items(
    tmp_path,
    monkeypatch,
):
    workflow, config, _output_id = _image_batch(tmp_path, item_count=2)
    original_execute = batch_module.execute_pipeline_request
    calls = 0

    def fail_first_preflight(request, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            message = "Compute preflight rejected the first workload."
            return PipelineRunResult(
                request.run_id,
                request.workflow,
                error=message,
                failure=PipelineExecutionFailure(
                    kind="compute_preflight",
                    error_type="ComputePreflightError",
                    message=message,
                    reason_code="compute_preflight_rejected",
                    cleanup_succeeded=True,
                ),
            )
        return original_execute(request, **kwargs)

    monkeypatch.setattr(
        batch_module,
        "execute_pipeline_request",
        fail_first_preflight,
    )

    result = run_batch(workflow, config)

    assert calls == 2
    assert result.manifest.items[0].status is BatchStatus.FAILED
    assert result.manifest.items[0].error_type != "BatchRuntimeCleanupError"
    assert result.manifest.items[0].execution["cleanup_succeeded"] is True
    assert result.manifest.items[1].status is BatchStatus.COMPLETED
    assert result.manifest.compute["runtime_cleanup_succeeded"] is True


def test_pre_cancelled_batch_is_first_class_and_never_discovers_accelerator(
    tmp_path,
    monkeypatch,
):
    workflow, config, _output_id = _image_batch(tmp_path, item_count=2)
    config = replace(config, compute_request=ComputeRequest(mode=ComputeMode.AUTO))
    cancelled = threading.Event()
    cancelled.set()

    def forbidden_registry(*_args, **_kwargs):
        raise AssertionError("a pre-cancelled batch must not construct a registry")

    import napari_vipp.core.compute_registry as registry_module

    monkeypatch.setattr(registry_module, "ComputeRegistry", forbidden_registry)

    result = run_batch(workflow, config, cancel_event=cancelled)

    assert result.cancelled
    assert not result.has_failures
    assert result.summary == {
        "completed": 0,
        "partial": 0,
        "skipped": 1,
        "cancelled": 1,
        "failed": 0,
    }
    assert [item.status for item in result.manifest.items] == [
        BatchStatus.CANCELLED,
        BatchStatus.SKIPPED,
    ]
    assert result.manifest_path.is_file()
    assert result.saved_paths == ()


def test_stale_compute_preference_fails_before_batch_artifacts(tmp_path):
    workflow, config, _output_id = _image_batch(tmp_path)
    invalid = replace(
        config,
        compute_request=ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences={"deleted-node": "cpu"},
        ),
    )

    with pytest.raises(ValueError, match="missing node IDs: deleted-node"):
        run_batch(workflow, invalid)

    assert not invalid.output_dir.exists()

    override = ComputeRequest(
        mode=ComputeMode.CUSTOM,
        node_preferences={"deleted-node": "best_gpu"},
    )
    with pytest.raises(ValueError, match="missing node IDs: deleted-node"):
        run_batch(workflow, config, compute_request=override)

    assert not config.output_dir.exists()


def test_existing_skip_never_claims_current_execution_provenance(tmp_path):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    np.save(inputs / "sample.npy", np.arange(12).reshape(3, 4))
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    output_nodes = []
    for tag in ("existing", "new"):
        output = pipeline.add_node("batch_output")
        assert pipeline.connect("input", output.id).success
        pipeline.set_param(output.id, "tag", tag)
        pipeline.set_param(output.id, "format", "npy")
        output_nodes.append(output)
    workflow = serialize_workflow(pipeline)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    existing_path = output_dir / "sample__existing.npy"
    np.save(existing_path, np.full((2, 2), 99))
    existing_bytes = existing_path.read_bytes()
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=output_dir,
        sources=(BatchSourceConfig("input", "Input", inputs, "*.npy"),),
        outputs=tuple(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                tag,
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            )
            for output, tag in zip(output_nodes, ("existing", "new"), strict=True)
        ),
        default_image_format="npy",
        existing_file_policy=ExistingFilePolicy.SKIP,
        save_python_script=False,
    )

    result = run_batch(workflow, config)

    skipped, produced = result.manifest.items[0].outputs
    assert skipped.status is BatchStatus.SKIPPED
    assert skipped.provenance_status == "not_produced"
    assert skipped.execution_provenance_sha256 == ""
    assert existing_path.read_bytes() == existing_bytes
    assert produced.status is BatchStatus.COMPLETED
    assert produced.provenance_status == "produced"
    assert produced.execution_provenance_sha256 == (
        result.manifest.items[0].execution_provenance_sha256
    )


def test_failure_mapping_is_preserved_and_mirrors_fallback_records():
    fallback = {
        "segment_id": "cuda-cupy:001",
        "runtime_id": "cuda-cupy",
        "node_ids": ["gaussian"],
        "reason": "out_of_memory",
        "reason_code": "cuda_memory_allocation",
        "cpu_retry_succeeded": False,
        "cleanup_succeeded": True,
    }
    failure = {
        "kind": "out_of_memory",
        "error_type": "OutOfMemoryError",
        "message": "allocation failed",
        "cleanup_succeeded": True,
        "fallback_records": [fallback],
    }

    payload = serialize_execution_provenance(
        ComputeRequest(mode=ComputeMode.AUTO),
        None,
        None,
        failure=failure,
    )

    assert payload["failure"] == failure
    assert payload["fallback_records"] == [fallback]
    assert payload["outcome"] == "failed"
    assert payload["cleanup_succeeded"] is True


@pytest.mark.parametrize("mode", (ComputeMode.AUTO, ComputeMode.PREFER_GPU))
def test_synthesized_global_policy_provenance_ignores_dormant_preference(mode):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    pipeline.set_param(gaussian.id, "sigma", 0.0)
    assert pipeline.connect("input", gaussian.id).success
    pipeline.run(
        np.ones((5, 5), dtype=np.float32),
        input_metadata={"axes": "YX"},
    )
    request = ComputeRequest(
        mode=mode,
        node_preferences={gaussian.id: "cpu"},
    )

    payload = serialize_execution_provenance(
        request,
        pipeline,
        None,
        completed_node_ids=(gaussian.id,),
    )

    assert request.preference_for(gaussian.id).kind.value == "cpu"
    assert payload["nodes"][0]["requested_preference"] == {"kind": "auto"}
    assert payload["nodes"][0]["reason"] == "auto_cpu"


def test_nested_progress_covers_node_boundaries_and_resets_per_item(tmp_path):
    workflow, config, _output_id = _image_batch(tmp_path, item_count=2)
    updates = []

    result = run_batch(
        workflow,
        config,
        execution_progress_callback=updates.append,
    )

    assert result.summary["completed"] == 2
    assert {update.item_index for update in updates} == {1, 2}
    assert {
        "batch_capture_source_identity",
        "batch_stage_output",
        "batch_verify_source_identity",
        "batch_publish_output",
    }.issubset({update.operation_id for update in updates})
    for item_index in (1, 2):
        item_updates = [
            update for update in updates if update.item_index == item_index
        ]
        node_start = next(
            update for update in item_updates if update.message == "Node started."
        )
        assert node_start.node_id == "input"
        assert node_start.current == 0
        assert node_start.total == 0
        assert any(
            update.current == 1
            and update.total == 1
            and update.message == "Node completed."
            for update in item_updates
        )
