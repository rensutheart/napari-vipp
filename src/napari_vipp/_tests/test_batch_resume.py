"""Crash recovery is verified content reuse, never an existing-file shortcut."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import napari_vipp.core.batch as batch
from napari_vipp._tests.test_batch import (
    _batch_config,
    _batch_workflow,
    _write_arrays,
)
from napari_vipp.core.batch_resume import (
    BatchResumeError,
    _destination_lock,
    inspect_batch_resume,
    run_batch_resume_from_manifest,
    seal_document,
)
from napari_vipp.core.progress import OperationCancelled


@pytest.fixture
def request_data(tmp_path):
    workflow, outputs = _batch_workflow()
    _write_arrays(tmp_path / "in", a=np.ones((5, 5)), b=np.full((5, 5), 2.0))
    config = _batch_config(workflow, tmp_path / "in", tmp_path / "out", outputs)
    return workflow, config


def _run_interrupted(monkeypatch, workflow, config):
    original = batch._save_item_record

    def crash_after_checkpoint(directory, item):
        result = original(directory, item)
        if item.index == 1 and item.status is batch.BatchStatus.COMPLETED:
            raise KeyboardInterrupt("simulated process death after durable receipt")
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(batch, "_save_item_record", crash_after_checkpoint)
        with pytest.raises(KeyboardInterrupt):
            batch.run_batch(workflow, config)
    return config.resolve_path(config.output_dir) / batch.BATCH_MANIFEST_FILENAME


def _inventory(directory):
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _mutate_without_stat_change(path):
    stat = path.stat()
    with path.open("r+b") as stream:
        stream.seek(-1, os.SEEK_END)
        byte = stream.read(1)
        stream.seek(-1, os.SEEK_END)
        stream.write(bytes([byte[0] ^ 1]))
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))


def test_resume_reconciles_crash_sidecar_and_preserves_original_run(
    monkeypatch, request_data
):
    workflow, config = request_data
    manifest_path = _run_interrupted(monkeypatch, workflow, config)
    source = inspect_batch_resume(manifest_path)
    old_archive = source.output_dir / f"vipp_batch_manifest_{source.run_id}.json"
    archive_bytes = old_archive.read_bytes()
    sidecars = _inventory(source.output_dir / source.document["item_records_dir"])
    assert source.document["items"][0]["status"] == "pending"
    assert source.items[0]["status"] == "completed"
    first_path = Path(source.items[0]["outputs"][0]["path"])
    first_stat = first_path.stat()
    execution = batch.execute_pipeline_request
    calls = []

    def counted(request, **kwargs):
        calls.append(request.run_id)
        return execution(request, **kwargs)

    monkeypatch.setattr(batch, "execute_pipeline_request", counted)
    statuses = []
    result = run_batch_resume_from_manifest(
        manifest_path, progress_callback=lambda *event: statuses.append(event[-1])
    )
    assert calls == [2]
    assert result.summary["completed"] == 2
    assert len(result.saved_paths) == 1
    assert "reused" in statuses
    assert result.manifest.resumed_from_run_id == source.run_id
    assert result.manifest.items[0].resumed_from_run_id == source.run_id
    assert first_path.stat().st_mtime_ns == first_stat.st_mtime_ns
    assert old_archive.read_bytes() == archive_bytes
    assert (
        _inventory(source.output_dir / source.document["item_records_dir"]) == sidecars
    )


@pytest.mark.parametrize("target", ["input", "output"])
def test_resume_detects_same_size_and_timestamp_byte_changes(
    monkeypatch, request_data, target
):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    source = inspect_batch_resume(path)
    changed = (
        config.sources[0].input_dir / "a.npy"
        if target == "input"
        else Path(source.items[0]["outputs"][0]["path"])
    )
    _mutate_without_stat_change(changed)
    before = _inventory(source.output_dir)
    with pytest.raises(
        (BatchResumeError, batch.SourceChangedError), match="changed|content|revision"
    ):
        run_batch_resume_from_manifest(path)
    assert _inventory(source.output_dir) == before


def test_resume_refuses_missing_completed_output(monkeypatch, request_data):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    source = inspect_batch_resume(path)
    Path(source.items[0]["outputs"][0]["path"]).unlink()
    with pytest.raises(BatchResumeError, match="ordinary file"):
        run_batch_resume_from_manifest(path)


@pytest.mark.parametrize("policy", list(batch.ExistingFilePolicy))
def test_resume_refuses_unverified_existing_outputs_even_skip_or_overwrite(
    monkeypatch, request_data, policy
):
    workflow, config = request_data
    config = replace(config, existing_file_policy=policy)
    path = _run_interrupted(monkeypatch, workflow, config)
    source = inspect_batch_resume(path)
    pending = Path(source.items[1]["outputs"][0]["path"])
    pending.write_bytes(b"not a verified result")
    before = _inventory(source.output_dir)
    with pytest.raises(BatchResumeError, match="unverified existing output"):
        run_batch_resume_from_manifest(path)
    assert _inventory(source.output_dir) == before


@pytest.mark.parametrize(
    "mutation", ["checksum", "provenance", "foreign_run", "directory", "old_version"]
)
def test_resume_refuses_corrupt_or_foreign_evidence(
    monkeypatch, request_data, mutation
):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    source = inspect_batch_resume(path)
    if mutation in {"checksum", "provenance", "foreign_run"}:
        destination = next(
            (source.output_dir / source.document["item_records_dir"]).glob("*.json")
        )
    else:
        destination = path
    document = json.loads(destination.read_text(encoding="utf-8"))
    if mutation == "checksum":
        document["batch_id"] = "other"
    elif mutation == "provenance":
        document["execution"]["cleanup_succeeded"] = False
    elif mutation == "foreign_run":
        document["run_id"] = "a" * 32
    elif mutation == "directory":
        document["item_records_dir"] = "../elsewhere"
    else:
        document["version"] = 5
    if mutation != "checksum":
        document = seal_document(document)
    batch.atomic_write_json(destination, document)
    before = _inventory(source.output_dir)
    with pytest.raises(BatchResumeError):
        run_batch_resume_from_manifest(path)
    assert _inventory(source.output_dir) == before


def test_resume_checks_environment_identity(monkeypatch, request_data):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    import napari_vipp.core.batch_resume as recovery

    monkeypatch.setattr(recovery, "implementation_identity", lambda: {"changed": True})
    with pytest.raises(BatchResumeError, match="environment changed"):
        run_batch_resume_from_manifest(path)


def test_resume_cancellation_during_verification_does_not_modify_run(
    monkeypatch, request_data
):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    before = _inventory(path.parent)
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(OperationCancelled):
        run_batch_resume_from_manifest(path, cancel_event=cancelled)
    assert _inventory(path.parent) == before


def test_resume_uses_embedded_workflow_not_mutable_companions(
    monkeypatch, request_data
):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    # No workflow.json or saved config exists: immutable embedded request suffices.
    result = run_batch_resume_from_manifest(path)
    assert result.summary["completed"] == 2


def test_completed_resumed_run_can_be_resumed_again(monkeypatch, request_data):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    first = run_batch_resume_from_manifest(path)
    second = run_batch_resume_from_manifest(first.manifest_path)
    assert second.summary["completed"] == 2
    assert not second.saved_paths
    assert all(
        item.resumed_from_run_id == first.manifest.run_id
        for item in second.manifest.items
    )


def test_cancelling_resume_preserves_already_verified_work_for_next_resume(
    monkeypatch, request_data
):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    cancelled = threading.Event()
    original = batch._save_run_manifest

    def cancel_after_seed(latest, archive, manifest):
        original(latest, archive, manifest)
        if manifest.resumed_from_run_id and not manifest.finished_at:
            cancelled.set()

    with monkeypatch.context() as scoped:
        scoped.setattr(batch, "_save_run_manifest", cancel_after_seed)
        result = run_batch_resume_from_manifest(path, cancel_event=cancelled)
    assert result.summary["completed"] == 1
    assert result.summary["cancelled"] == 1
    assert not result.saved_paths
    next_attempt = run_batch_resume_from_manifest(result.manifest_path)
    assert next_attempt.summary["completed"] == 2
    assert len(next_attempt.saved_paths) == 1


def test_batch_destination_lock_guards_regular_runs_and_releases(request_data):
    workflow, config = request_data
    with _destination_lock(config.output_dir):
        with pytest.raises(BatchResumeError, match="already using"):
            batch.run_batch(workflow, config)
    assert not config.output_dir.exists()
    assert batch.run_batch(workflow, config).summary["completed"] == 2


def test_resume_ignores_private_staged_file_as_completion(monkeypatch, request_data):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    staged = path.parent / ".abandoned.tmp.npy"
    staged.write_bytes(b"private incomplete bytes")
    result = run_batch_resume_from_manifest(path)
    assert result.summary["completed"] == 2
    assert staged.read_bytes() == b"private incomplete bytes"


def test_resume_rejects_changed_output_declaration(monkeypatch, request_data):
    workflow, config = request_data
    path = _run_interrupted(monkeypatch, workflow, config)
    changed = replace(config, output_dir=config.output_dir.parent / "elsewhere")
    with pytest.raises(BatchResumeError, match="config"):
        batch.run_batch(workflow, changed, resume_manifest_path=path)
    assert not changed.output_dir.exists()


def test_old_unsealed_manifest_has_actionable_diagnostic(tmp_path):
    path = tmp_path / "old.json"
    batch.atomic_write_json(path, {"type": batch.BATCH_MANIFEST_TYPE, "version": 5})
    with pytest.raises(BatchResumeError, match="predates verified resume"):
        inspect_batch_resume(path)


def test_partial_item_publication_is_not_reused(monkeypatch, tmp_path):
    spec = {
        "format": "npy",
        "subfolder": "",
        "filename_template": "{source_stem}__{tag}",
        "overwrite": "batch default",
    }
    workflow, outputs = _batch_workflow((dict(spec, tag="one"), dict(spec, tag="two")))
    _write_arrays(tmp_path / "in", a=np.ones((5, 5)))
    config = _batch_config(workflow, tmp_path / "in", tmp_path / "out", outputs)
    original = batch._save_item_record

    def crash_after_first_output(directory, item):
        result = original(directory, item)
        if item.status is batch.BatchStatus.RUNNING and any(
            output.status is batch.BatchStatus.COMPLETED for output in item.outputs
        ):
            raise KeyboardInterrupt("process death between output promotions")
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(batch, "_save_item_record", crash_after_first_output)
        with pytest.raises(KeyboardInterrupt):
            batch.run_batch(workflow, config)
    manifest_path = config.output_dir / batch.BATCH_MANIFEST_FILENAME
    before = _inventory(config.output_dir)
    with pytest.raises(BatchResumeError, match="unverified existing output"):
        run_batch_resume_from_manifest(manifest_path)
    assert _inventory(config.output_dir) == before


def test_resume_never_overwrites_output_appearing_after_validation(
    monkeypatch, request_data
):
    workflow, config = request_data
    config = replace(config, existing_file_policy=batch.ExistingFilePolicy.OVERWRITE)
    path = _run_interrupted(monkeypatch, workflow, config)
    source = inspect_batch_resume(path)
    pending = Path(source.items[1]["outputs"][0]["path"])

    def appear_after_validation(index, _total, _batch_id, status):
        if index == 2 and status == "running":
            pending.write_bytes(b"another process owns this result")

    result = run_batch_resume_from_manifest(
        path, progress_callback=appear_after_validation
    )
    assert result.has_failures
    assert pending.read_bytes() == b"another process owns this result"
    assert result.summary["completed"] == 1


@pytest.mark.real_cuda
def test_real_cuda_batch_restart_resume_checks_runtime_and_device(tmp_path):
    if os.environ.get("VIPP_RUN_REAL_CUDA_BATCH") != "1":
        pytest.skip("Set VIPP_RUN_REAL_CUDA_BATCH=1 for the real CUDA resume smoke.")
    from napari_vipp.core.compute import ComputeMode, ComputeRequest, FallbackPolicy
    from napari_vipp.core.compute_registry import ComputeRegistry
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import serialize_workflow

    registry = ComputeRegistry()
    try:
        probe = registry.probe_runtime("cuda-cupy", refresh=True)
        assert probe.available and probe.selected_device_id
        selected = next(
            device
            for device in probe.devices
            if device.device_id == probe.selected_device_id
        )
        expected_device = os.environ.get("VIPP_EXPECT_CUDA_DEVICE", "")
        assert expected_device.casefold() in selected.display_name.casefold()
        pipeline = PrototypePipeline()
        pipeline.reset_empty_graph()
        gaussian = pipeline.add_node("gaussian_blur")
        output = pipeline.add_node("batch_output")
        pipeline.set_param(gaussian.id, "sigma", 1.25)
        pipeline.set_param(output.id, "tag", "result")
        pipeline.set_param(output.id, "format", "npy")
        assert pipeline.connect("input", gaussian.id).success
        assert pipeline.connect(gaussian.id, output.id).success
        request = ComputeRequest(
            mode=ComputeMode.CUSTOM,
            node_preferences={
                gaussian.id: "implementation:cupy-gaussian-blur-v1",
                output.id: "cpu",
            },
            runtime_id="cuda-cupy",
            device_id=probe.selected_device_id,
            fallback_policy=FallbackPolicy.STRICT,
        )
        workflow = serialize_workflow(pipeline, compute_request=request)
        _write_arrays(
            tmp_path / "in",
            a=np.random.default_rng(123).random((256, 320), dtype=np.float32),
        )
        config = replace(
            _batch_config(workflow, tmp_path / "in", tmp_path / "out", (output.id,)),
            compute_request=request,
        )
        result = batch.run_batch(workflow, config, compute_registry=registry)
        assert not result.has_failures
        environment = result.manifest.items[0].execution["environment"]
        assert environment["device_id"] == probe.selected_device_id
        assert environment["runtime_probe_fingerprints"]["cuda-cupy"]
        assert any(
            node["actual_implementation"]["runtime_id"] == "cuda-cupy"
            for node in result.manifest.items[0].execution["nodes"]
        )
    finally:
        registry.close()
    resumed = run_batch_resume_from_manifest(result.manifest_path)
    assert resumed.summary["completed"] == 1
    assert not resumed.saved_paths
    assert resumed.manifest.items[0].resumed_from_run_id == result.manifest.run_id
