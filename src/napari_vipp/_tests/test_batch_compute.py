from __future__ import annotations

import json
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import napari_vipp.core.batch as batch_module
from napari_vipp.core.batch import (
    BATCH_CONFIG_FILENAME,
    BATCH_CONFIG_VERSION,
    BatchConfig,
    BatchOutputConfig,
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
)
from napari_vipp.core.execution import PipelineRunResult
from napari_vipp.core.execution_provenance import serialize_execution_provenance
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


def test_batch_config_v2_roundtrip_and_v1_migration_preserve_cpu_replay(tmp_path):
    workflow, config, _output_id = _image_batch(tmp_path)
    auto = ComputeRequest(
        mode=ComputeMode.AUTO,
        runtime_id="cuda-cupy",
        device_id="cuda:0",
        accelerator_memory_cap_bytes=2_000_000_000,
        accelerator_safety_reserve_bytes=250_000_000,
        allow_experimental=True,
    )
    current = replace(config, compute_request=auto)
    path = tmp_path / BATCH_CONFIG_FILENAME

    save_batch_config(path, current)
    loaded = load_batch_config(path)

    assert loaded.compute_request == auto
    assert loaded.to_dict()["version"] == BATCH_CONFIG_VERSION
    assert loaded.to_dict()["compute"] == auto.as_dict()

    legacy_document = config.to_dict()
    legacy_document["version"] = 1
    legacy_document.pop("compute")
    path.write_text(json.dumps(legacy_document), encoding="utf-8")

    migrated = load_batch_config(path)

    assert migrated.compute_request.mode is ComputeMode.CPU
    assert migrated.to_dict()["version"] == BATCH_CONFIG_VERSION
    assert migrated.workflow_sha256 == scientific_workflow_hash(workflow)


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
            mode=ComputeMode.SELECTIVE,
            node_preferences={"deleted-node": "cpu"},
        ),
    )

    with pytest.raises(ValueError, match="missing node IDs: deleted-node"):
        run_batch(workflow, invalid)

    assert not invalid.output_dir.exists()

    override = ComputeRequest(
        mode=ComputeMode.SELECTIVE,
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
