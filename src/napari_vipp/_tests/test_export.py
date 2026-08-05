from __future__ import annotations

import gc
import hashlib
import json
import threading
import weakref
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.export import (
    export_batch_runner_to_python,
    export_pipeline_to_python,
)
from napari_vipp.core.io import ImageDataset, ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import (
    AmbiguousAxisError,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.operations import (
    COMPOSITE_RGB_PERCENTILE_1_99,
    COMPOSITE_RGB_PRESERVE_VALUES,
)
from napari_vipp.core.pipeline import (
    GraphConnection,
    PrototypePipeline,
    SourcePayload,
)
from napari_vipp.core.source_identity import capture_local_source_identity
from napari_vipp.core.workflow import serialize_workflow


def _starter_pipeline() -> PrototypePipeline:
    pipeline = PrototypePipeline()
    pipeline.reset_starter_graph()
    return pipeline


def _assert_embedded_operation(code: str, operation_id: str) -> None:
    assert f'"operation_id":"{operation_id}"' in code


def _install_generated_batch_inputs(
    monkeypatch,
    tmp_path,
    *,
    workflow_path=None,
    compute_request=None,
):
    workflow_path = workflow_path or (tmp_path / "vipp_pipeline.json")
    workflow_path.write_text(
        json.dumps(
            serialize_workflow(
                _starter_pipeline(),
                compute_request=compute_request,
            )
        ),
        encoding="utf-8",
    )
    config = SimpleNamespace(
        workflow_file=workflow_path.name,
        compute_request=compute_request or ComputeRequest(),
        resolve_path=lambda _path: workflow_path,
    )
    monkeypatch.setattr(
        "napari_vipp.core.batch.load_batch_config",
        lambda _path: config,
    )
    return config, workflow_path


def test_exported_batch_runner_uses_sibling_defaults_and_prints_summary(
    monkeypatch,
    tmp_path,
    capsys,
):
    script_path = tmp_path / "vipp_batch_runner.py"
    manifest_path = tmp_path / "vipp_batch_manifest.json"
    config, workflow_path = _install_generated_batch_inputs(
        monkeypatch,
        tmp_path,
    )
    calls: list[tuple[object, object, dict[str, object]]] = []
    result = SimpleNamespace(
        summary={
            "completed": 3,
            "partial": 0,
            "skipped": 2,
            "cancelled": 0,
            "failed": 0,
        },
        saved_paths=[tmp_path / "first.tif", tmp_path / "second.tif"],
        manifest_path=manifest_path,
        has_failures=False,
        cancelled=False,
    )

    def fake_run_batch(workflow, config, **kwargs):
        calls.append((workflow, config, kwargs))
        return result

    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch",
        fake_run_batch,
    )
    code = export_batch_runner_to_python()
    compiled = compile(code, "<exported-batch-runner>", "exec")
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(script_path),
    }
    exec(compiled, namespace)

    assert namespace["main"]([]) == 0
    assert len(calls) == 1
    assert calls[0][0] == json.loads(workflow_path.read_text(encoding="utf-8"))
    assert calls[0][1] is config
    assert calls[0][2]["workflow_path"] == workflow_path.resolve()
    assert calls[0][2]["config_path"] == (
        tmp_path / "vipp_batch_config.json"
    ).resolve()
    assert calls[0][2]["compute_request"] is None
    assert isinstance(calls[0][2]["cancel_event"], threading.Event)
    assert calls[0][2]["progress_callback"] is None
    assert calls[0][2]["execution_progress_callback"] is None
    assert capsys.readouterr().out == (
        "3 completed, 0 partial, 2 skipped, 0 cancelled, 0 failed; "
        f"2 outputs saved; manifest: {manifest_path}\n"
    )


def test_exported_batch_runner_passes_cli_overrides_and_reports_failure(
    monkeypatch,
    tmp_path,
    capsys,
):
    workflow_path = tmp_path / "custom-workflow.json"
    config_path = tmp_path / "custom-config.json"
    config, _workflow_path = _install_generated_batch_inputs(
        monkeypatch,
        tmp_path,
        workflow_path=workflow_path,
    )
    calls: list[tuple[object, object, dict[str, object]]] = []
    result = SimpleNamespace(
        summary={
            "completed": 1,
            "partial": 0,
            "skipped": 0,
            "cancelled": 0,
            "failed": 1,
        },
        saved_paths=[tmp_path / "successful-output.tif"],
        manifest_path=tmp_path / "manifest.json",
        has_failures=True,
        cancelled=False,
    )

    def fake_run_batch(workflow, config, **kwargs):
        calls.append((workflow, config, kwargs))
        return result

    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch",
        fake_run_batch,
    )
    code = export_batch_runner_to_python()
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(tmp_path / "vipp_batch_runner.py"),
    }
    exec(compile(code, "<exported-batch-runner>", "exec"), namespace)

    assert (
        namespace["main"](
            [
                "--workflow",
                str(workflow_path),
                "--config",
                str(config_path),
            ]
        )
        == 1
    )
    assert len(calls) == 1
    assert calls[0][0] == json.loads(workflow_path.read_text(encoding="utf-8"))
    assert calls[0][1] is config
    assert calls[0][2]["workflow_path"] == workflow_path.resolve()
    assert calls[0][2]["config_path"] == config_path.resolve()
    assert calls[0][2]["compute_request"] is None
    assert isinstance(calls[0][2]["cancel_event"], threading.Event)
    assert calls[0][2]["progress_callback"] is None
    assert calls[0][2]["execution_progress_callback"] is None
    assert capsys.readouterr().out == (
        "1 completed, 0 partial, 0 skipped, 0 cancelled, 1 failed; "
        f"1 outputs saved; manifest: {result.manifest_path}\n"
    )


def test_exported_batch_runner_reports_preflight_exception(
    monkeypatch,
    tmp_path,
    capsys,
):
    _install_generated_batch_inputs(monkeypatch, tmp_path)

    def failing_run(_workflow, _config, **_kwargs):
        raise ValueError("workflow/config mismatch")

    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch",
        failing_run,
    )
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(tmp_path / "vipp_batch_pipeline.py"),
    }
    exec(
        compile(
            export_batch_runner_to_python(),
            "<exported-batch-runner>",
            "exec",
        ),
        namespace,
    )

    assert namespace["main"]([]) == 2
    assert "workflow/config mismatch" in capsys.readouterr().err


def test_exported_batch_runner_reports_cancelled_count_and_exit_code(
    monkeypatch,
    tmp_path,
    capsys,
):
    _install_generated_batch_inputs(monkeypatch, tmp_path)
    result = SimpleNamespace(
        summary={
            "completed": 1,
            "partial": 0,
            "skipped": 0,
            "cancelled": 2,
            "failed": 0,
        },
        saved_paths=(),
        manifest_path=tmp_path / "manifest.json",
        has_failures=False,
        cancelled=True,
    )
    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch",
        lambda *_args, **_kwargs: result,
    )
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(tmp_path / "vipp_batch_runner.py"),
    }
    exec(
        compile(
            export_batch_runner_to_python(),
            "<exported-batch-runner>",
            "exec",
        ),
        namespace,
    )

    assert namespace["main"]([]) == 130
    assert "2 cancelled" in capsys.readouterr().out


def test_exported_batch_runner_overlays_compute_flags_and_nested_progress(
    monkeypatch,
    tmp_path,
):
    workflow_path = tmp_path / "workflow.json"
    workflow_path.write_text(
        json.dumps(
            serialize_workflow(
                _starter_pipeline(),
                compute_request=ComputeRequest(
                    mode="auto",
                    node_preferences={"gaussian": "cpu"},
                ),
            )
        ),
        encoding="utf-8",
    )
    authored_workflow = json.loads(workflow_path.read_text(encoding="utf-8"))

    class Config:
        workflow_file = "workflow.json"
        compute_request = ComputeRequest(
            mode="auto",
            node_preferences={"gaussian": "cpu"},
        )

        def resolve_path(self, _path):
            return workflow_path

    config = Config()
    monkeypatch.setattr(
        "napari_vipp.core.batch.load_batch_config",
        lambda _path: config,
    )
    captured: dict[str, object] = {}
    result = SimpleNamespace(
        summary={
            "completed": 1,
            "partial": 0,
            "skipped": 0,
            "cancelled": 0,
            "failed": 0,
        },
        saved_paths=(),
        manifest_path=tmp_path / "manifest.json",
        has_failures=False,
        cancelled=False,
    )

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        changed = json.loads(json.dumps(authored_workflow))
        gaussian = next(
            node for node in changed["nodes"] if node["id"] == "gaussian"
        )
        gaussian["params"]["sigma"] = 9.0
        workflow_path.write_text(json.dumps(changed), encoding="utf-8")
        return result

    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch",
        fake_run,
    )
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(tmp_path / "vipp_batch_runner.py"),
    }
    exec(
        compile(
            export_batch_runner_to_python(),
            "<exported-batch-runner>",
            "exec",
        ),
        namespace,
    )

    assert (
        namespace["main"](
            [
                "--workflow",
                str(workflow_path),
                "--config",
                str(tmp_path / "config.json"),
                "--compute-mode",
                "custom",
                "--fallback-policy",
                "strict",
                "--node-preference",
                "threshold=cpu",
                "--progress",
            ]
        )
        == 0
    )

    override = captured["kwargs"]["compute_request"]
    assert captured["args"][0] == authored_workflow
    assert override.mode.value == "custom"
    assert override.fallback_policy.value == "strict"
    assert override.preference_for("gaussian").kind.value == "cpu"
    assert override.preference_for("threshold").kind.value == "cpu"
    assert config.compute_request.mode.value == "auto"
    assert "threshold" not in config.compute_request.node_preferences
    assert callable(captured["kwargs"]["progress_callback"])
    assert callable(captured["kwargs"]["execution_progress_callback"])


def test_exported_batch_cli_accepts_prefer_gpu_and_resets_inherited_strict(
    monkeypatch,
    tmp_path,
):
    authored = ComputeRequest(
        mode="custom",
        node_preferences={"gaussian": "cpu"},
        fallback_policy="strict",
    )
    _install_generated_batch_inputs(
        monkeypatch,
        tmp_path,
        compute_request=authored,
    )
    captured: list[ComputeRequest | None] = []
    result = SimpleNamespace(
        summary={
            "completed": 1,
            "partial": 0,
            "skipped": 0,
            "cancelled": 0,
            "failed": 0,
        },
        saved_paths=(),
        manifest_path=tmp_path / "manifest.json",
        has_failures=False,
        cancelled=False,
    )

    def fake_run_batch(_workflow, _config, **kwargs):
        captured.append(kwargs["compute_request"])
        return result

    monkeypatch.setattr(
        "napari_vipp.core.batch.run_batch",
        fake_run_batch,
    )
    namespace: dict[str, object] = {
        "__name__": "exported_batch_runner",
        "__file__": str(tmp_path / "vipp_batch_runner.py"),
    }
    exec(
        compile(
            export_batch_runner_to_python(),
            "<exported-batch-runner>",
            "exec",
        ),
        namespace,
    )

    assert namespace["main"](["--compute-mode", "prefer_gpu"]) == 0
    assert len(captured) == 1
    override = captured[0]
    assert override is not None
    assert override.mode.value == "prefer_gpu"
    assert override.fallback_policy.value == "visible"
    assert override.preference_for("gaussian").kind.value == "cpu"

    with pytest.raises(SystemExit) as caught:
        namespace["main"](
            [
                "--compute-mode",
                "prefer_gpu",
                "--fallback-policy",
                "strict",
            ]
        )
    assert caught.value.code == 2
    assert len(captured) == 1


def test_export_produces_valid_python():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)

    # Must compile as a module without syntax errors.
    compile(code, "<exported>", "exec")

    assert "def run_pipeline(" in code
    assert "def batch_process(" in code
    _assert_embedded_operation(code, "gaussian_blur")
    _assert_embedded_operation(code, "otsu_threshold")
    assert '"sigma":1.2' in code
    assert "execute_pipeline_request(" in code
    assert "pipeline_from_workflow(document)" in code
    assert "ImageDataset, write_image" in code
    assert "load_frozen_file_source_snapshot" in code
    assert "skimage" not in code


def test_exported_run_pipeline_executes():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)

    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    run_pipeline = namespace["run_pipeline"]
    image = np.random.rand(4, 8, 8).astype(np.float32)
    results = run_pipeline(image)

    assert "threshold" in results
    assert results["threshold"].shape == image.shape
    assert results["threshold"].dtype == bool
    assert namespace["OUTPUT_NODES"] == ("threshold",)


def test_exported_run_reports_generic_node_start_and_finish_progress():
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    updates = []

    namespace["run_pipeline"](
        np.ones((8, 9), dtype=np.float32),
        input_metadata={"axes": "YX"},
        progress_callback=lambda *update: updates.append(update),
    )

    starts = {
        node_id
        for node_id, current, total, message in updates
        if current == 0 and total == 0 and message.startswith("Node started")
    }
    finishes = {
        node_id
        for node_id, current, total, message in updates
        if current == 1 and total == 1 and message.startswith("Node completed")
    }
    assert {"gaussian", "threshold"}.issubset(starts)
    assert {"gaussian", "threshold"}.issubset(finishes)
    assert all(node_id in {"input", "gaussian", "threshold"} for node_id in starts)


def test_exported_progress_callback_errors_do_not_invalidate_execution():
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )

    def presentation_failure(*_args):
        raise RuntimeError("presentation callback failed")

    results = namespace["run_pipeline"](
        np.ones((8, 9), dtype=np.float32),
        input_metadata={"axes": "YX"},
        progress_callback=presentation_failure,
    )

    assert results["threshold"].dtype == bool
    assert results.execution_report.cleanup_succeeded


def test_exported_results_report_exact_cpu_provenance_and_stable_hashes():
    pipeline = _starter_pipeline()
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<exported>", "exec"),
        namespace,
    )
    embedded = namespace["_WORKFLOW_JSON"]
    image = np.ones((8, 9), dtype=np.float32)

    first = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )
    second = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
        compute_request=ComputeRequest(
            mode="cpu",
            fallback_policy="strict",
            node_preferences={"gaussian": "cpu"},
        ),
    )

    assert first.execution_report is not None
    assert first.execution_report.cleanup_succeeded
    assert first.effective_compute_request == ComputeRequest(mode="cpu")
    assert set(first.node_compute_provenance) == {
        "input",
        "gaussian",
        "threshold",
    }
    source_identity = first.node_compute_provenance[
        "input"
    ].actual_implementation
    assert source_identity.runtime_id == "source-boundary"
    assert source_identity.implementation_id == "source-input-v1"
    actual = {
        item["node_id"]: item["actual_implementation"]
        for item in first.execution_provenance["nodes"]
    }
    assert actual["gaussian"] == {
        "identity_complete": True,
        "runtime_id": "cpu-numpy",
        "array_domain": "host-numpy",
        "implementation_library_id": "cpu",
        "implementation_id": "cpu-gaussian_blur-v1",
        "implementation_version": "1",
        "parity_policy_id": "authoritative-cpu-v1",
        "cache_equivalence_group": "",
    }
    assert first.workflow_sha256 == second.workflow_sha256
    assert first.generated_template_fingerprint == (
        second.generated_template_fingerprint
    )
    assert first.effective_execution_fingerprint != (
        second.effective_execution_fingerprint
    )
    assert namespace["_WORKFLOW_JSON"] == embedded


def test_exported_results_bind_provenance_to_each_terminal_output():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.connect("input", gaussian.id)
    pipeline.connect("input", median.id)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<exported>", "exec"),
        namespace,
    )

    results = namespace["run_pipeline"](
        np.ones((8, 9), dtype=np.float32),
        input_metadata={"axes": "YX"},
    )

    gaussian_record = results.output_provenance[gaussian.id]
    median_record = results.output_provenance[median.id]
    assert gaussian_record["output"]["node_id"] == gaussian.id
    assert median_record["output"]["node_id"] == median.id
    assert gaussian_record["output"]["output_port_index"] == 0
    assert gaussian_record["output"]["result_context_fingerprint"]
    assert median_record["output"]["result_context_fingerprint"]
    assert gaussian_record["output"]["execution_provenance_sha256"] == (
        median_record["output"]["execution_provenance_sha256"]
    )
    assert gaussian_record["provenance_sha256"] != median_record[
        "provenance_sha256"
    ]


def test_exported_output_provenance_preserves_supplied_exact_source_identity(
    tmp_path,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    identity = {
        "kind": "file",
        "sha256": "a" * 64,
        "regular_file_count": 1,
        "size_bytes": 1234,
    }
    payload = SourcePayload(
        np.ones((8, 9), dtype=np.float32),
        {
            "axes": "YX",
            "vipp_source_path": "C:/data/source.ome.tif",
            "vipp_source_identity": identity,
            "vipp_source_provenance": {"reader": "tifffile"},
        },
        "source.ome.tif",
    )

    results = namespace["run_pipeline"](payload)

    assert results.provenance["sources"] == [
        {
            "node_id": "input",
            "name": "source.ome.tif",
            "path": "C:/data/source.ome.tif",
            "identity_complete": True,
            "identity": identity,
            "reader_provenance": {"reader": "tifffile"},
            "binding_sha256": results.provenance["sources"][0][
                "binding_sha256"
            ],
        }
    ]
    def write_staged(_data, path, **_kwargs):
        Path(path).write_bytes(b"staged-output")
        return path

    namespace["write_image"] = write_staged
    output_path = tmp_path / "output.tif"
    namespace["save_image"](
        results["threshold"],
        output_path,
        image_state=results.image_states["threshold"],
        provenance=results,
        output_node_id="threshold",
    )
    sidecar = json.loads(
        (tmp_path / "output.tif.vipp-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["sources"][0]["identity"]["sha256"] == "a" * 64
    assert sidecar["sources"][0]["binding_sha256"]


def test_generated_file_loader_binds_verified_local_revision_to_output(
    tmp_path,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    source_path = tmp_path / "source.npy"
    np.save(source_path, np.arange(8 * 9, dtype=np.float32).reshape(8, 9))
    expected_identity = capture_local_source_identity(source_path)
    progress = []

    payload = namespace["load_image"](
        source_path,
        progress_callback=lambda *update: progress.append(update),
    )
    results = namespace["run_pipeline"](payload)
    output_provenance = results.provenance_for("threshold")
    source_record = output_provenance["sources"][0]

    assert source_record["node_id"] == "input"
    assert source_record["path"] == str(source_path.resolve())
    assert source_record["identity_complete"] is True
    assert source_record["identity"] == expected_identity.to_dict()
    assert source_record["identity"]["sha256"] == expected_identity.sha256
    assert source_record["binding_sha256"]
    assert output_provenance["output"]["node_id"] == "threshold"
    messages = [message for _node, _current, _total, message in progress]
    assert any("Source validation 1/3" in message for message in messages)
    assert any("Source materialization 2/3" in message for message in messages)
    assert any("Source reverification 3/3" in message for message in messages)


def test_generated_cli_sidecar_binds_exact_local_source_path_and_sha256(
    tmp_path,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    source_path = tmp_path / "source.npy"
    output_path = tmp_path / "output.npy"
    np.save(source_path, np.arange(8 * 9, dtype=np.float32).reshape(8, 9))
    expected_identity = capture_local_source_identity(source_path)

    assert namespace["main"]([str(source_path), str(output_path)]) == 0

    sidecar = json.loads(
        (tmp_path / "output.npy.vipp-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["sources"] == [
        {
            "node_id": "input",
            "name": "source",
            "path": str(source_path.resolve()),
            "identity_complete": True,
            "identity": expected_identity.to_dict(),
            "reader_provenance": {},
            "binding_sha256": sidecar["sources"][0]["binding_sha256"],
        }
    ]
    assert sidecar["sources"][0]["identity"]["sha256"] == (
        expected_identity.sha256
    )
    assert sidecar["output"]["node_id"] == "threshold"


def test_generated_artifact_sha256_hashes_saved_script_bytes(tmp_path):
    script = tmp_path / "pipeline.py"
    script.write_text(
        export_pipeline_to_python(_starter_pipeline()),
        encoding="utf-8",
        newline="\n",
    )
    namespace: dict[str, object] = {
        "__name__": "exported_pipeline",
        "__file__": str(script),
    }
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)

    results = namespace["run_pipeline"](
        np.ones((4, 5), dtype=np.float32),
        input_metadata={"axes": "YX"},
    )

    expected = hashlib.sha256(script.read_bytes()).hexdigest()
    assert results.generated_artifact_sha256 == expected
    assert results.provenance["generated_artifact"]["source_sha256"] == expected


def test_generated_artifact_sha256_is_immutable_for_loaded_module(tmp_path):
    script = tmp_path / "pipeline.py"
    script.write_text(
        export_pipeline_to_python(_starter_pipeline()),
        encoding="utf-8",
        newline="\n",
    )
    loaded_sha256 = hashlib.sha256(script.read_bytes()).hexdigest()
    namespace: dict[str, object] = {
        "__name__": "exported_pipeline",
        "__file__": str(script),
    }
    exec(compile(script.read_text(encoding="utf-8"), str(script), "exec"), namespace)
    script.write_text(
        script.read_text(encoding="utf-8") + "\n# edited after import\n",
        encoding="utf-8",
        newline="\n",
    )

    results = namespace["run_pipeline"](
        np.ones((4, 5), dtype=np.float32),
        input_metadata={"axes": "YX"},
    )

    assert results.generated_artifact_sha256 == loaded_sha256
    assert results.generated_artifact_sha256 != hashlib.sha256(
        script.read_bytes()
    ).hexdigest()


def test_exported_compute_override_validates_nodes_without_mutating_snapshot():
    code = export_pipeline_to_python(_starter_pipeline())
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    embedded = namespace["_WORKFLOW_JSON"]

    with pytest.raises(ValueError, match="unknown exported nodes"):
        namespace["run_pipeline"](
            np.ones((4, 5), dtype=np.float32),
            input_metadata={"axes": "YX"},
            compute_request={
                "mode": "cpu",
                "node_preferences": {"missing": "cpu"},
            },
        )

    assert namespace["_WORKFLOW_JSON"] == embedded


def test_export_preserves_scalar_channel_contract_with_pipeline_parity():
    pipeline = PrototypePipeline()
    filtered = pipeline.add_node("bilateral_filter")
    pipeline.connect("input", filtered.id)
    image = np.random.default_rng(12).random((5, 7, 3), dtype=np.float32)

    native = pipeline.run(image, input_metadata={"axes": "ZYX"})[filtered.id]
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    exported = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )[filtered.id]

    assert '"channel_axis":-1' in code
    np.testing.assert_allclose(exported, native, rtol=0.0, atol=0.0)


def test_exported_rescale_axes_preserves_output_size_mode():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("rescale_axes")
    pipeline.connect("input", node.id)
    pipeline.set_param(node.id, "resize_mode", "Output size")
    pipeline.set_param(node.id, "x_size", 12)
    pipeline.set_param(node.id, "y_size", 8)
    pipeline.set_param(node.id, "z_size", 5)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        np.zeros((3, 4, 6), dtype=np.uint8),
        input_metadata={"axes": "ZYX"},
    )

    assert '"resize_mode":"Output size"' in code
    assert '"x_size":12' in code
    assert results[node.id].shape == (5, 8, 12)


def test_export_handles_multi_input_nodes():
    pipeline = _starter_pipeline()
    add = pipeline.add_node("add_images")
    pipeline.connect("gaussian", add.id, target_port=1)
    pipeline.connect("input", add.id, target_port=0)

    code = export_pipeline_to_python(pipeline)
    compile(code, "<exported>", "exec")
    _assert_embedded_operation(code, "add_images")
    assert '"source":"input"' in code
    assert '"source":"gaussian"' in code


def test_export_keeps_incomplete_multi_input_node_uncomputed():
    pipeline = PrototypePipeline()
    add = pipeline.add_node("add_images")
    pipeline.connect("input", add.id, target_port=0)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    results = namespace["run_pipeline"](np.ones((3, 4), dtype=np.float32))
    assert results[add.id] is None
    _assert_embedded_operation(code, "add_images")


@pytest.mark.parametrize(
    "name",
    [
        "bad-name",
        "class",
        "",
        "OUTPUT_NODES",
        "Path",
        "batch_process",
        "load_image",
        "read_image",
        "save_image",
        "ComputeRequest",
        "Mapping",
        "OperationCancelled",
        "PipelineRunRequest",
        "atomic_write_json",
        "canonical_digest",
        "count",
        "deserialize_workflow",
        "dict",
        "execute_pipeline_request",
        "hashlib",
        "main",
        "next",
        "serialize_execution_provenance",
        "signal",
        "sys",
        "threading",
        "warnings",
    ],
)
def test_export_rejects_invalid_function_name(name):
    with pytest.raises(ValueError, match="function name"):
        export_pipeline_to_python(PrototypePipeline(), function_name=name)


def test_export_rejects_function_name_that_shadows_used_operation():
    with pytest.raises(ValueError, match="function name"):
        export_pipeline_to_python(
            _starter_pipeline(),
            function_name="gaussian_blur",
        )


def test_source_only_export_compiles_without_empty_operation_import():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.ones((3, 4), dtype=np.float32)

    assert "from napari_vipp.core.operations import" not in code
    np.testing.assert_array_equal(namespace["run_pipeline"](image)["input"], image)


def test_custom_export_function_name_is_used_by_generated_harness():
    pipeline = _starter_pipeline()

    code = export_pipeline_to_python(pipeline, function_name="analyze_image")
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.ones((3, 4), dtype=np.float32)

    assert "def analyze_image(" in code
    assert "results = analyze_image(" in code
    assert (
        "load_image(source_path, progress_callback=progress_callback, "
        "cancel_event=cancel_event),"
    ) in code
    assert namespace["analyze_image"](image)["threshold"].dtype == bool


def test_generated_cli_overlays_only_explicit_compute_fields():
    authored = ComputeRequest(
        mode="custom",
        fallback_policy="visible",
        node_preferences={"gaussian": "cpu"},
    )
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(
                _starter_pipeline(),
                compute_request=authored,
            ),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    embedded = namespace["_WORKFLOW_JSON"]

    request = namespace["_cli_compute_request"](
        fallback_policy="strict",
        node_preferences=["threshold=cpu"],
    )

    assert request.mode.value == "custom"
    assert request.fallback_policy.value == "strict"
    assert request.preference_for("gaussian").kind.value == "cpu"
    assert request.preference_for("threshold").kind.value == "cpu"
    assert namespace["_WORKFLOW_JSON"] == embedded
    with pytest.raises(ValueError, match="unknown exported nodes"):
        namespace["_cli_compute_request"](
            node_preferences=["missing=cpu"],
        )
    with pytest.raises(ValueError, match="Duplicate node preference"):
        namespace["_cli_compute_request"](
            node_preferences=["gaussian=cpu", "gaussian=auto"],
        )


def test_generated_pipeline_cli_accepts_prefer_gpu_and_resets_inherited_strict():
    authored = ComputeRequest(
        mode="custom",
        fallback_policy="strict",
        node_preferences={"gaussian": "cpu"},
    )
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(
                _starter_pipeline(),
                compute_request=authored,
            ),
            "<exported>",
            "exec",
        ),
        namespace,
    )

    request = namespace["_cli_compute_request"](mode="prefer_gpu")

    assert request.mode.value == "prefer_gpu"
    assert request.fallback_policy.value == "visible"
    assert request.preference_for("gaussian").kind.value == "cpu"
    with pytest.raises(
        ValueError,
        match="Prefer GPU requires visible CPU fallback",
    ):
        namespace["_cli_compute_request"](
            mode="prefer_gpu",
            fallback_policy="strict",
        )


@pytest.mark.parametrize("write_provenance", [True, False])
def test_generated_cli_passes_run_override_and_provenance_choice(
    tmp_path,
    write_provenance,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    calls: dict[str, object] = {}

    class Results(dict):
        image_states = {}
        provenance = {"type": "test"}

    results = Results(threshold=np.ones((2, 3), dtype=np.float32))

    def fake_run(*_args, **kwargs):
        calls["request"] = kwargs["compute_request"]
        calls["run_cancel_event"] = kwargs["cancel_event"]
        return results

    def fake_load(_path, **kwargs):
        calls["load"] = kwargs
        return np.ones((2, 3), dtype=np.float32)

    def fake_save(data, path, **kwargs):
        calls["saved"] = (data, path, kwargs)
        path = Path(path)
        path.write_bytes(b"staged-output")
        if kwargs["provenance"] is not None:
            namespace["_provenance_sidecar_path"](path).write_text(
                "{}",
                encoding="utf-8",
            )
        return path

    namespace["load_image"] = fake_load
    namespace["run_pipeline"] = fake_run
    namespace["_write_output_uncommitted"] = fake_save
    args = [str(tmp_path / "input.tif"), str(tmp_path / "output.tif")]
    args.append("--progress")
    if not write_provenance:
        args.append("--no-provenance")

    assert namespace["main"](args) == 0

    assert calls["request"].mode.value == "cpu"
    assert callable(calls["load"]["progress_callback"])
    assert calls["load"]["cancel_event"] is calls["run_cancel_event"]
    assert calls["saved"][2]["provenance"] is (
        results if write_provenance else None
    )


def test_generated_cli_distinguishes_cancellation_and_writes_failure_sidecar(
    tmp_path,
    capsys,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    failure = namespace["OperationCancelled"]("cancelled at checkpoint")
    failure.provenance = {
        "type": "napari-vipp-generated-execution-provenance",
        "version": 1,
        "execution": {
            "failure": {"kind": "cancelled"},
            "cleanup_succeeded": True,
        },
    }
    namespace["load_image"] = lambda _path, **_kwargs: np.ones(
        (2, 3), dtype=np.float32
    )

    def cancel(*_args, **_kwargs):
        raise failure

    namespace["run_pipeline"] = cancel
    output_path = tmp_path / "output.tif"

    assert (
        namespace["main"](
            [str(tmp_path / "input.tif"), str(output_path)]
        )
        == 130
    )

    assert "Pipeline cancelled" in capsys.readouterr().err
    sidecar = tmp_path / "output.tif.vipp-provenance.json"
    assert json.loads(sidecar.read_text(encoding="utf-8"))[
        "execution"
    ]["failure"]["kind"] == "cancelled"


def test_generated_cli_reports_publication_failure_with_no_fallback(
    tmp_path,
    capsys,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    namespace["load_image"] = lambda _path, **_kwargs: np.ones(
        (4, 5), dtype=np.float32
    )

    def disk_full(*_args, **_kwargs):
        raise OSError("simulated disk full")

    namespace["write_image"] = disk_full
    output_path = tmp_path / "output.tif"

    assert namespace["main"]([str(tmp_path / "input.tif"), str(output_path)]) == 2

    assert "simulated disk full" in capsys.readouterr().err
    sidecar = tmp_path / "output.tif.vipp-provenance.json"
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["execution"]["outcome"] == "completed"
    assert document["publication"]["outcome"] == "failed"
    assert document["publication"]["node_id"] == "threshold"
    assert document["publication"]["fallback_used"] is False


def test_generated_cli_defers_publication_cancellation_until_writer_returns(
    tmp_path,
    capsys,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    namespace["load_image"] = lambda _path, **_kwargs: np.ones(
        (4, 5), dtype=np.float32
    )
    real_run = namespace["run_pipeline"]
    captured: dict[str, object] = {}

    def capture_event(*args, **kwargs):
        captured["cancel_event"] = kwargs["cancel_event"]
        return real_run(*args, **kwargs)

    def finish_writer_then_cancel(_data, path, **_kwargs):
        Path(path).write_bytes(b"staged-output")
        captured["cancel_event"].set()
        return path

    namespace["run_pipeline"] = capture_event
    namespace["write_image"] = finish_writer_then_cancel
    output_path = tmp_path / "output.tif"

    assert (
        namespace["main"]([str(tmp_path / "input.tif"), str(output_path)])
        == 130
    )

    assert "Pipeline cancelled" in capsys.readouterr().err
    assert not output_path.exists()
    assert not tuple(tmp_path.glob(".vipp-publish-*"))
    document = json.loads(
        (tmp_path / "output.tif.vipp-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert document["execution"]["outcome"] == "completed"
    assert document["publication"]["outcome"] == "cancelled"
    assert "output" not in document


def test_generated_cli_provenance_failure_preserves_preexisting_output_set(
    tmp_path,
    capsys,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    namespace["load_image"] = lambda _path, **_kwargs: np.ones(
        (4, 5), dtype=np.float32
    )
    output_path = tmp_path / "output.tif"
    output_path.write_bytes(b"pre-existing-output")
    output_sidecar = tmp_path / "output.tif.vipp-provenance.json"
    output_sidecar.write_text(
        '{"type":"pre-existing-provenance"}\n',
        encoding="utf-8",
    )
    real_save_provenance = namespace["save_provenance_sidecar"]

    def fail_staged_provenance(path, provenance):
        if ".vipp-publish-" in str(path):
            raise OSError("simulated provenance write failure")
        return real_save_provenance(path, provenance)

    namespace["save_provenance_sidecar"] = fail_staged_provenance

    assert namespace["main"](
        [str(tmp_path / "input.tif"), str(output_path)]
    ) == 2

    assert "simulated provenance write failure" in capsys.readouterr().err
    assert output_path.read_bytes() == b"pre-existing-output"
    assert json.loads(output_sidecar.read_text(encoding="utf-8")) == {
        "type": "pre-existing-provenance"
    }
    failure_sidecar = (
        tmp_path
        / "output.tif.vipp-run-failure.vipp-provenance.json"
    )
    failure = json.loads(failure_sidecar.read_text(encoding="utf-8"))
    assert failure["publication"]["outcome"] == "failed"
    assert failure["publication"]["fallback_used"] is False
    assert not tuple(tmp_path.glob(".vipp-publish-*"))


def test_generated_cli_later_multi_output_failure_publishes_none(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.connect("input", gaussian.id)
    pipeline.connect("input", median.id)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<exported>", "exec"),
        namespace,
    )
    namespace["load_image"] = lambda _path, **_kwargs: np.ones(
        (5, 6), dtype=np.float32
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    gaussian_path = output_dir / f"input__{gaussian.id}.ome.tif"
    gaussian_sidecar = namespace["_provenance_sidecar_path"](gaussian_path)
    gaussian_path.write_bytes(b"pre-existing-gaussian")
    gaussian_sidecar.write_text(
        '{"type":"pre-existing-provenance"}\n',
        encoding="utf-8",
    )
    calls = 0

    def fail_second_writer(_data, path, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated later output failure")
        Path(path).write_bytes(b"new-staged-output")
        return path

    namespace["write_image"] = fail_second_writer

    assert namespace["main"](
        [str(tmp_path / "input.tif"), str(output_dir)]
    ) == 2

    median_path = output_dir / f"input__{median.id}.ome.tif"
    assert gaussian_path.read_bytes() == b"pre-existing-gaussian"
    assert json.loads(gaussian_sidecar.read_text(encoding="utf-8")) == {
        "type": "pre-existing-provenance"
    }
    assert not median_path.exists()
    assert not namespace["_provenance_sidecar_path"](median_path).exists()
    failure = json.loads(
        (
            output_dir
            / "vipp-run-failure.vipp-provenance.json"
        ).read_text(encoding="utf-8")
    )
    assert failure["publication"]["outcome"] == "failed"
    assert failure["publication"]["node_id"] == median.id
    assert not tuple(output_dir.glob(".vipp-publish-*"))


def test_generated_cli_rolls_back_partial_multi_output_commit(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    gaussian = pipeline.add_node("gaussian_blur")
    median = pipeline.add_node("median_filter")
    pipeline.connect("input", gaussian.id)
    pipeline.connect("input", median.id)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<exported>", "exec"),
        namespace,
    )
    namespace["load_image"] = lambda _path, **_kwargs: np.ones(
        (5, 6), dtype=np.float32
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()
    paths = {
        node_id: output_dir / f"input__{node_id}.ome.tif"
        for node_id in (gaussian.id, median.id)
    }
    original_payloads = {}
    for node_id, path in paths.items():
        sidecar = namespace["_provenance_sidecar_path"](path)
        output_bytes = f"old-output-{node_id}".encode()
        sidecar_bytes = f'{{"old":"{node_id}"}}\n'.encode()
        path.write_bytes(output_bytes)
        sidecar.write_bytes(sidecar_bytes)
        original_payloads[path] = output_bytes
        original_payloads[sidecar] = sidecar_bytes

    real_atomic_replace = namespace["atomic_replace"]
    failed = False

    def fail_last_output_promotion(source, target):
        nonlocal failed
        source = Path(source)
        target = Path(target)
        is_stage_promotion = (
            ".vipp-publish-" in str(source)
            and ".vipp-publish-" not in str(target)
        )
        if (
            not failed
            and is_stage_promotion
            and target.resolve() == paths[median.id].resolve()
        ):
            failed = True
            raise OSError("simulated final promotion failure")
        return real_atomic_replace(source, target)

    namespace["atomic_replace"] = fail_last_output_promotion

    assert namespace["main"](
        [str(tmp_path / "input.tif"), str(output_dir)]
    ) == 2

    assert failed
    for path, payload in original_payloads.items():
        assert path.read_bytes() == payload
    assert not tuple(output_dir.glob(".vipp-publish-*"))


def test_convenience_folder_loop_returns_only_lightweight_records(tmp_path):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    for name in ("first.tif", "second.tif"):
        (input_dir / name).write_bytes(b"placeholder")
    namespace["load_image"] = lambda _path, **_kwargs: np.ones(
        (4, 5), dtype=np.float32
    )
    def fake_save(_data, path, **kwargs):
        path = Path(path)
        path.write_bytes(b"staged-output")
        if kwargs["provenance"] is not None:
            namespace["_provenance_sidecar_path"](path).write_text(
                "{}",
                encoding="utf-8",
            )
        return path

    namespace["_write_output_uncommitted"] = fake_save

    with pytest.warns(FutureWarning, match="not VIPP durable"):
        records = namespace["batch_process"](input_dir, output_dir)

    assert records == [
        {
            "source_path": str(input_dir / "first.tif"),
            "saved_paths": (
                str(output_dir / "first__threshold.ome.tif"),
            ),
            "workflow_sha256": namespace["WORKFLOW_SHA256"],
        },
        {
            "source_path": str(input_dir / "second.tif"),
            "saved_paths": (
                str(output_dir / "second__threshold.ome.tif"),
            ),
            "workflow_sha256": namespace["WORKFLOW_SHA256"],
        },
    ]
    assert all(
        not isinstance(record, namespace["PipelineResults"])
        for record in records
    )


def test_convenience_folder_loop_releases_prior_outputs_before_next_run(
    tmp_path,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    input_dir.mkdir()
    for name in ("first.tif", "second.tif"):
        (input_dir / name).write_bytes(b"placeholder")
    prior_refs = []
    prior_alive_during_next_load = []
    prior_alive_during_next_run = []

    def load_one(path, **_kwargs):
        if prior_refs:
            gc.collect()
            prior_alive_during_next_load.append(prior_refs[-1]() is not None)
        return path

    namespace["load_image"] = load_one

    class Results(dict):
        image_states = {}
        workflow_sha256 = namespace["WORKFLOW_SHA256"]

    def run_one(*_args, **_kwargs):
        if prior_refs:
            gc.collect()
            prior_alive_during_next_run.append(prior_refs[-1]() is not None)
        output = np.ones((256, 256), dtype=np.float32)
        prior_refs.append(weakref.ref(output))
        return Results(threshold=output)

    namespace["run_pipeline"] = run_one
    def fake_save(_data, path, **kwargs):
        path = Path(path)
        path.write_bytes(b"staged-output")
        if kwargs["provenance"] is not None:
            namespace["_provenance_sidecar_path"](path).write_text(
                "{}",
                encoding="utf-8",
            )
        return path

    namespace["_write_output_uncommitted"] = fake_save

    with pytest.warns(FutureWarning):
        namespace["batch_process"](input_dir, output_dir)

    assert prior_alive_during_next_load == [False]
    assert prior_alive_during_next_run == [False]


def test_exported_run_preserves_native_cancellation_and_cleanup_provenance():
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(namespace["OperationCancelled"]) as caught:
        namespace["run_pipeline"](
            np.ones((4, 5), dtype=np.float32),
            input_metadata={"axes": "YX"},
            cancel_event=cancel_event,
        )

    execution = caught.value.provenance["execution"]
    assert execution["outcome"] == "cancelled"
    assert execution["failure"]["kind"] == "cancelled"
    assert execution["cleanup_succeeded"] is True


def test_exported_run_withholds_outputs_when_cleanup_cannot_be_proven():
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    real_execute = namespace["execute_pipeline_request"]

    def unsafe_cleanup(*args, **kwargs):
        result = real_execute(*args, **kwargs)
        return replace(
            result,
            execution_report=replace(
                result.execution_report,
                cleanup_succeeded=False,
            ),
        )

    namespace["execute_pipeline_request"] = unsafe_cleanup

    with pytest.raises(
        namespace["PipelineExecutionError"],
        match="cleanup could not be proven",
    ) as caught:
        namespace["run_pipeline"](
            np.ones((8, 9), dtype=np.float32),
            input_metadata={"axes": "YX"},
        )

    execution = caught.value.provenance["execution"]
    assert execution["outcome"] == "failed"
    assert execution["failure"]["kind"] == "cleanup_failure"
    assert execution["cleanup_succeeded"] is False
    assert caught.value.execution_report.cleanup_succeeded is False


def test_generated_failure_provenance_preserves_structured_fallback_records():
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    fallback = {
        "segment_id": "segment-0",
        "runtime_id": "cuda-cupy",
        "node_ids": ["gaussian"],
        "reason": "out_of_memory",
        "reason_code": "device_out_of_memory",
        "cpu_retry_succeeded": False,
        "cleanup_succeeded": True,
    }

    provenance = namespace["_build_failure_provenance"](
        ComputeRequest(mode="auto"),
        {
            "kind": "execution_error",
            "error_type": "MemoryError",
            "message": "CPU retry failed.",
            "cleanup_succeeded": True,
            "fallback_records": [fallback],
        },
    )

    execution = provenance["execution"]
    assert execution["outcome"] == "failed"
    assert execution["cleanup_succeeded"] is True
    assert execution["fallback_records"] == [fallback]
    assert execution["failure"]["fallback_records"] == [fallback]


def test_exported_run_classifies_unstructured_executor_errors():
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )

    def fail(*_args, **_kwargs):
        raise ValueError("unstructured failure")

    namespace["execute_pipeline_request"] = fail
    with pytest.raises(ValueError, match="unstructured failure") as caught:
        namespace["run_pipeline"](
            np.ones((4, 5), dtype=np.float32),
            input_metadata={"axes": "YX"},
        )

    execution = caught.value.provenance["execution"]
    assert execution["outcome"] == "failed"
    assert execution["failure"]["error_type"] == "ValueError"


def test_export_uses_unique_variables_for_colliding_node_identifiers():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    first = pipeline.add_node("linear_scale_offset")
    second = pipeline.add_node("linear_scale_offset")
    combined = pipeline.add_node("add_images")
    pipeline.set_param(first.id, "alpha", 2.0)
    pipeline.set_param(first.id, "beta", 0.0)
    pipeline.set_param(second.id, "alpha", 3.0)
    pipeline.set_param(second.id, "beta", 0.0)
    renamed_ids = {first.id: "branch-a", second.id: "branch a"}
    renamed_nodes = [
        replace(node, id=renamed_ids.get(node.id, node.id))
        for node in pipeline.nodes.values()
    ]
    pipeline.restore_graph(
        renamed_nodes,
        [
            GraphConnection("input", "branch-a"),
            GraphConnection("input", "branch a"),
            GraphConnection("branch-a", combined.id, target_port=0),
            GraphConnection("branch a", combined.id, target_port=1),
        ],
    )

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.ones((3, 4), dtype=np.float32)

    np.testing.assert_array_equal(namespace["run_pipeline"](image)[combined.id], 5.0)


def test_export_includes_richardson_lucy_deconvolution_call():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    psf_source = pipeline.add_node("input")
    decon = pipeline.add_node("richardson_lucy_deconvolution")
    pipeline.set_param(decon.id, "spatial_mode", "2D YX")
    pipeline.set_param(decon.id, "iterations", 2)
    pipeline.set_param(decon.id, "resolved_spatial_ndim", 2)
    pipeline.connect("input", decon.id, target_port=0)
    pipeline.connect(psf_source.id, decon.id, target_port=1)
    image = np.zeros((9, 9), dtype=np.float32)
    image[4, 4] = 1.0
    psf = np.zeros((3, 3), dtype=np.float32)
    psf[1, 1] = 1.0

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image, psf)

    _assert_embedded_operation(code, "richardson_lucy_deconvolution")
    assert '"resolved_spatial_ndim":2' in code
    assert results[decon.id].dtype == np.float32
    assert results[decon.id].shape == image.shape


def test_export_compiles_named_tunnel_connections_as_normal_inputs():
    pipeline = _starter_pipeline()
    median = pipeline.add_node("median_filter")
    pipeline.add_output_tunnel("Raw", "input", 0)
    result = pipeline.connect_to_tunnel("Raw", median.id)
    assert result.success

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.random.rand(4, 8, 8).astype(np.float32)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "median_filter")
    assert results[median.id].shape == image.shape


def test_export_prefers_explicit_batch_output_nodes():
    pipeline = _starter_pipeline()
    marker = pipeline.add_node("batch_output")
    pipeline.set_param(marker.id, "tag", "blurred")
    pipeline.connect("gaussian", marker.id)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        np.random.rand(4, 8, 8).astype(np.float32)
    )

    _assert_embedded_operation(code, "batch_output")
    assert namespace["OUTPUT_NODES"] == (marker.id,)
    assert results[marker.id].shape == (4, 8, 8)
    assert "threshold" in results


def test_export_includes_subtract_background_node():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("subtract_background")
    pipeline.set_param(node.id, "radius", 7)
    pipeline.connect("input", node.id)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.zeros((21, 21), dtype=np.uint8)
    image[10, 10] = 200
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )

    _assert_embedded_operation(code, "subtract_background")
    assert '"radius":7' in code
    assert results[node.id].shape == image.shape
    assert results[node.id].dtype == image.dtype


def test_exported_touching_object_separation_pipeline_executes():
    pipeline = PrototypePipeline()
    distance = pipeline.add_node("euclidean_distance_transform")
    markers = pipeline.add_node("h_maxima_markers")
    watershed = pipeline.add_node("marker_controlled_watershed")
    pipeline.connect("input", distance.id)
    pipeline.connect(distance.id, markers.id)
    pipeline.connect(distance.id, watershed.id, target_port=0)
    pipeline.connect(markers.id, watershed.id, target_port=1)
    pipeline.connect("input", watershed.id, target_port=2)

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    yy, xx = np.mgrid[:48, :64]
    image = (((yy - 24) ** 2 + (xx - 22) ** 2 <= 13**2) | (
        (yy - 24) ** 2 + (xx - 42) ** 2 <= 13**2
    ))
    results = namespace["run_pipeline"](image)

    _assert_embedded_operation(code, "euclidean_distance_transform")
    _assert_embedded_operation(code, "h_maxima_markers")
    _assert_embedded_operation(code, "marker_controlled_watershed")
    assert results[watershed.id].dtype == np.int32
    assert int(results[watershed.id].max()) == 2


def test_exported_intensity_measurement_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects_intensity")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    pipeline.connect("input", measurements.id)

    image = np.zeros((7, 7), dtype=np.float32)
    image[1:3, 1:4] = 10
    image[4:6, 4:6] = 20
    pipeline.run(image, input_metadata={"axes": "YX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )
    table = results[measurements.id]
    records = table.records()

    _assert_embedded_operation(code, "measure_objects_intensity")
    assert table.row_count == 2
    assert records[0]["intensity_mean"] == 10.0
    assert records[1]["intensity_mean"] == 20.0


def test_exported_label_volume_pipeline_executes():
    pipeline = _starter_pipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    filtered = pipeline.add_node("filter_labels_by_volume")
    relabeled = pipeline.add_node("relabel_sequential")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(filtered.id, "min_volume", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, filtered.id)
    pipeline.connect(filtered.id, relabeled.id)

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 1:4, 1:4] = 10
    image[1, 7, 7] = 10
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "label_connected_components")
    assert '"resolved_spatial_ndim":3' in code
    assert set(np.unique(results[relabeled.id])) == {0, 1}


def test_exported_label_property_filter_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    filtered = pipeline.add_node("filter_labels_by_property")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(filtered.id, "property_column", "area_pixels")
    pipeline.set_param(filtered.id, "min_value", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    pipeline.connect(labels.id, filtered.id, target_port=0)
    pipeline.connect(measurements.id, filtered.id, target_port=1)

    image = np.zeros((8, 8), dtype=np.float32)
    image[1:4, 1:4] = 10
    image[6, 6] = 10
    pipeline.run(image, input_metadata={"axes": "YX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )

    _assert_embedded_operation(code, "filter_labels_by_property")
    assert '"resolved_spatial_ndim":2' in code
    assert set(np.unique(results[filtered.id])) == {0, 1}


def test_exported_clear_border_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    cleared = pipeline.add_node("clear_border_objects")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, cleared.id)

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 0:3, 0:3] = 10
    image[1, 4:7, 4:7] = 10
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "clear_border_objects")
    assert '"resolved_spatial_ndim":3' in code
    assert set(np.unique(results[cleared.id])) == {0, 2}


def test_exported_fill_holes_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    filled = pipeline.add_node("fill_holes")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(filled.id, "spatial_mode", "3D ZYX volume")
    pipeline.set_param(filled.id, "max_hole_size", 1)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, filled.id)

    mask = np.ones((3, 7, 7), dtype=bool)
    mask[1, 3, 3] = False
    mask[0, 1, 1] = False
    pipeline.run(mask.astype(np.float32), input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        mask.astype(np.float32),
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "fill_holes")
    assert '"max_hole_size":1' in code
    assert '"resolved_spatial_ndim":3' in code
    assert results[filled.id][1, 3, 3]
    assert not results[filled.id][0, 1, 1]


def test_exported_remove_small_objects_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    filtered = pipeline.add_node("remove_small_objects")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(filtered.id, "min_size", 5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, filtered.id)

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 1:4, 1:4] = 1
    image[1, 7, 7] = 1
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )

    _assert_embedded_operation(code, "remove_small_objects")
    assert '"resolved_spatial_ndim":3' in code
    assert results[filtered.id][:, 1:4, 1:4].all()
    assert not results[filtered.id][1, 7, 7]


def test_exported_measure_objects_pipeline_executes_and_saves_table(tmp_path):
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(measurements.id, "include_axis_descriptors", True)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)

    image = np.zeros((3, 9, 9), dtype=np.float32)
    image[:, 1:4, 1:4] = 1
    image[1, 7, 7] = 1
    pipeline.run(image, input_metadata={"axes": "ZYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZYX"},
    )
    table = results[measurements.id]
    output_path = tmp_path / "measurements.ome.tif"

    namespace["save_image"](table, output_path, provenance=results)

    _assert_embedded_operation(code, "measure_objects")
    assert '"include_axis_descriptors":true' in code
    assert "from napari_vipp.core.tables import" in code
    assert table.row_count == 2
    assert table.columns[:2] == ("label_id", "volume_voxels")
    assert "major_axis_length_voxels" in table.columns
    csv_path = tmp_path / "measurements.ome.csv"
    assert csv_path.exists()
    assert csv_path.read_text(encoding="utf-8").startswith(
        "label_id,volume_voxels"
    )
    sidecar = tmp_path / "measurements.ome.csv.vipp-provenance.json"
    assert sidecar.exists()
    provenance = json.loads(sidecar.read_text(encoding="utf-8"))
    assert provenance["workflow"]["sha256"] == results.workflow_sha256
    assert provenance["execution"]["cleanup_succeeded"]


def test_exported_full_run_explicitly_includes_authored_manual_nodes(monkeypatch):
    from napari_vipp.core import execution

    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    captured = []
    captured_kwargs = []
    real_execute = execution.execute_pipeline_request

    def capture_request(request, **kwargs):
        captured.append(request)
        captured_kwargs.append(kwargs)
        return real_execute(request, **kwargs)

    monkeypatch.setattr(execution, "execute_pipeline_request", capture_request)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<exported>", "exec"),
        namespace,
    )
    image = np.zeros((9, 9), dtype=np.float32)
    image[2:6, 2:6] = 1
    cancel_event = threading.Event()
    progress_events = []

    def progress(*args):
        progress_events.append(args)

    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
        progress_callback=progress,
        cancel_event=cancel_event,
    )

    assert results[measurements.id].row_count == 1
    assert len(captured) == 1
    assert captured[0].manual_node_ids == frozenset({measurements.id})
    assert captured[0].cancel_event is cancel_event
    assert callable(captured_kwargs[0]["node_started_callback"])
    assert callable(captured_kwargs[0]["node_finished_callback"])
    assert callable(captured_kwargs[0]["progress_callback"])
    assert captured_kwargs[0]["progress_callback"] is not progress
    assert captured_kwargs[0]["raise_errors"] is True
    assert any(
        node_id == measurements.id and current == total == 1
        for node_id, current, total, _message in progress_events
    )


def test_exported_merged_table_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    morphology = pipeline.add_node("measure_objects")
    intensity = pipeline.add_node("measure_objects_intensity")
    merged = pipeline.add_node("merge_tables")
    annotated = pipeline.add_node("add_metadata_columns")
    selected = pipeline.add_node("select_table_columns")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(annotated.id, "metadata_columns", "condition=demo")
    pipeline.set_param(selected.id, "columns", "label_id,intensity_mean,condition")
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, morphology.id)
    pipeline.connect(labels.id, intensity.id, target_port=0)
    pipeline.connect("input", intensity.id, target_port=1)
    pipeline.connect(morphology.id, merged.id, target_port=0)
    pipeline.connect(intensity.id, merged.id, target_port=1)
    pipeline.connect(merged.id, annotated.id)
    pipeline.connect(annotated.id, selected.id)

    image = np.zeros((7, 7), dtype=np.float32)
    image[1:3, 1:4] = 10
    image[4:6, 4:6] = 20
    pipeline.run(image, input_metadata={"axes": "YX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )
    table = results[selected.id]

    _assert_embedded_operation(code, "merge_tables")
    _assert_embedded_operation(code, "add_metadata_columns")
    _assert_embedded_operation(code, "select_table_columns")
    assert table.row_count == 2
    assert table.columns == ("label_id", "intensity_mean", "condition")
    assert table.records()[0]["condition"] == "demo"


def test_exported_summary_table_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    annotated = pipeline.add_node("add_metadata_columns")
    summarized = pipeline.add_node("summarize_measurements")
    pipeline.set_param(threshold.id, "threshold", 5)
    pipeline.set_param(annotated.id, "metadata_columns", "condition=demo")
    pipeline.set_param(summarized.id, "group_by", "condition")
    pipeline.set_param(summarized.id, "value_columns", "area_pixels")
    pipeline.set_param(summarized.id, "statistics", "mean,min,max")
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, labels.id)
    pipeline.connect(labels.id, measurements.id)
    pipeline.connect(measurements.id, annotated.id)
    pipeline.connect(annotated.id, summarized.id)

    image = np.zeros((2, 12, 12), dtype=np.float32)
    image[0, 1:4, 1:5] = 10
    image[0, 7:10, 7:11] = 10
    image[1, 2:7, 2:6] = 10
    pipeline.run(image, input_metadata={"axes": "TYX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "TYX"},
    )
    table = results[summarized.id]

    _assert_embedded_operation(code, "add_metadata_columns")
    _assert_embedded_operation(code, "summarize_measurements")
    assert table.row_count == 1
    assert table.records()[0]["condition"] == "demo"
    assert table.records()[0]["row_count"] == 3
    assert table.records()[0]["area_pixels_mean"] == 14.666666666666666


def test_exported_skeleton_analysis_pipeline_executes():
    pipeline = PrototypePipeline()
    threshold = pipeline.add_node("binary_threshold")
    skeleton = pipeline.add_node("skeletonize")
    measurements = pipeline.add_node("analyze_skeleton")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.connect("input", threshold.id)
    pipeline.connect(threshold.id, skeleton.id)
    pipeline.connect(skeleton.id, measurements.id)

    image = np.zeros((7, 7), dtype=np.float32)
    image[1:6, 3] = 1
    image[3, 1:6] = 1
    pipeline.run(image, input_metadata={"axes": "YX"})

    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
    )
    table = results[measurements.id]
    record = table.records()[0]

    _assert_embedded_operation(code, "skeletonize")
    _assert_embedded_operation(code, "analyze_skeleton")
    assert table.row_count == 1
    assert record["endpoint_voxel_count"] == 4
    assert record["branch_count"] == 4
    assert record["graph_node_count"] == 5
    assert record["graph_edge_count"] == 4
    assert record["voxel_graph_edge_count"] == 8


def test_exported_extract_channel_uses_shared_explicit_axis_semantics():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    extracted = pipeline.add_node("extract_channel")
    pipeline.set_param(extracted.id, "channel", 1)
    pipeline.connect("input", extracted.id)
    image = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    image[:, 0] = 10
    image[:, 1] = 42
    image[:, 2] = 90

    native = pipeline.run(
        image,
        input_metadata={"axes": "ZCYX"},
    )[extracted.id]
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    results = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "ZCYX"},
    )
    state = image_state_from_array(image, layer_metadata={"axes": "ZCYX"})
    series = ImageSeriesInfo(0, "0", "image", image.shape, "uint16", "ZCYX")
    dataset = ImageDataset(
        image,
        state,
        SourceInspection("memory://image", "test", (series,)),
        series,
    )
    dataset_results = namespace["run_pipeline"](dataset)

    np.testing.assert_array_equal(results[extracted.id], native)
    np.testing.assert_array_equal(dataset_results[extracted.id], native)
    assert results[extracted.id].shape == (2, 4, 5)
    assert results.image_states[extracted.id].axis_order == "ZYX"
    assert results.image_states[extracted.id].axes_explicit
    with pytest.raises(AmbiguousAxisError, match="explicit channel axis"):
        namespace["run_pipeline"](image)


@pytest.mark.parametrize(
    "intensity_mapping",
    [COMPOSITE_RGB_PRESERVE_VALUES, COMPOSITE_RGB_PERCENTILE_1_99],
)
def test_exported_composite_matches_native_values_colors_and_provenance(
    intensity_mapping,
):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    composite = pipeline.add_node("composite_to_rgb")
    pipeline.set_param(composite.id, "intensity_mapping", intensity_mapping)
    pipeline.connect("input", composite.id)
    image = np.arange(2 * 2 * 4 * 5, dtype=np.uint16).reshape(2, 2, 4, 5)
    image[:, 1] *= 3
    state = image_state_from_array(
        image,
        layer_metadata={"axes": "ZCYX"},
        channels=(
            ChannelMetadata(name="yellow", color=0xFFFF00),
            ChannelMetadata(name="cyan", color=0x00FFFF),
        ),
    )
    payload = SourcePayload(image, image_state=state)

    native = pipeline.run(
        image,
        source_payloads={"input": payload},
    )[composite.id]
    native_state = pipeline.output_states[composite.id]
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<exported>", "exec"),
        namespace,
    )

    exported = namespace["run_pipeline"](
        image,
        source_image_states={"input": state},
    )

    np.testing.assert_array_equal(exported[composite.id], native)
    assert exported.image_states[composite.id].to_dict() == native_state.to_dict()
    assert exported.image_states[composite.id].axis_order == "Z,Y,X,rgb"


def test_exported_mask_uses_per_source_semantics_for_broadcasting():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    mask_source = pipeline.add_node("input")
    threshold = pipeline.add_node("binary_threshold")
    masked = pipeline.add_node("mask_image")
    pipeline.set_param(threshold.id, "threshold", 0.5)
    pipeline.set_param(masked.id, "outside_value", -5)
    pipeline.connect(mask_source.id, threshold.id)
    pipeline.connect("input", masked.id, target_port=0)
    pipeline.connect(threshold.id, masked.id, target_port=1)
    image = np.arange(2 * 2 * 3 * 4, dtype=np.int16).reshape(2, 2, 3, 4)
    mask = np.zeros((2, 3, 4), dtype=np.float32)
    mask[0, :, 0] = 1
    mask[1, :, -1] = 1

    native = pipeline.run(
        image,
        input_metadata={"axes": "TZYX"},
        source_payloads={
            mask_source.id: SourcePayload(mask, {"axes": "TYX"}),
        },
    )[masked.id]
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    results = namespace["run_pipeline"](
        image,
        mask,
        input_metadata={"axes": "TZYX"},
        source_metadata={mask_source.id: {"axes": "TYX"}},
    )

    np.testing.assert_array_equal(results[masked.id], native)
    assert results.image_states[masked.id].axis_order == "TZYX"
    with pytest.raises(ValueError, match="explicit axis semantics"):
        namespace["run_pipeline"](image, mask)


def test_exported_multi_source_call_rejects_a_missing_binding():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second = pipeline.add_node("input")
    added = pipeline.add_node("add_images")
    pipeline.connect("input", added.id, target_port=0)
    pipeline.connect(second.id, added.id, target_port=1)
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)

    with pytest.raises(ValueError, match=rf"Source node {second.id!r} has no input"):
        namespace["run_pipeline"](np.ones((3, 4), dtype=np.float32))


def test_exported_source_payload_mapping_can_supply_every_source():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    second = pipeline.add_node("input")
    added = pipeline.add_node("add_images")
    pipeline.connect("input", added.id, target_port=0)
    pipeline.connect(second.id, added.id, target_port=1)
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    first_data = np.ones((3, 4), dtype=np.float32)
    second_data = np.full((3, 4), 2, dtype=np.float32)

    results = namespace["run_pipeline"](
        source_payloads={
            "input": SourcePayload(first_data, {"axes": "YX"}),
            second.id: SourcePayload(second_data, {"axes": "YX"}),
        }
    )

    np.testing.assert_array_equal(results[added.id], 3)


def test_exported_sources_reject_unknown_or_duplicate_bindings():
    pipeline = PrototypePipeline()
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    image = np.ones((3, 4), dtype=np.float32)

    with pytest.raises(ValueError, match="Unknown exported source nodes"):
        namespace["run_pipeline"](
            source_payloads={"typo": SourcePayload(image)},
        )
    with pytest.raises(ValueError, match="supplied both positionally"):
        namespace["run_pipeline"](
            image,
            source_payloads={"input": SourcePayload(image)},
        )


def test_exported_workflow_refuses_an_unvalidated_runtime_version():
    pipeline = PrototypePipeline()
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    namespace["VIPP_VERSION"] = "different-runtime"

    with pytest.raises(RuntimeError, match="active runtime is different-runtime"):
        namespace["run_pipeline"](np.ones((3, 4), dtype=np.float32))


def test_exported_workflow_snapshot_is_revalidated_and_fresh_per_run():
    pipeline = _starter_pipeline()
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    encoded = namespace["_WORKFLOW_JSON"]
    first = namespace["_new_pipeline"]()
    first.set_param("gaussian", "sigma", 9.0)

    second = namespace["_new_pipeline"]()

    assert namespace["_WORKFLOW_JSON"] == encoded
    assert second.nodes["gaussian"].params["sigma"] == 1.2


def test_exported_workflow_fails_closed_when_embedded_json_is_tampered():
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    document = json.loads(namespace["_WORKFLOW_JSON"])
    gaussian = next(node for node in document["nodes"] if node["id"] == "gaussian")
    gaussian["params"]["sigma"] = 9.0
    namespace["_WORKFLOW_JSON"] = json.dumps(document)

    with pytest.raises(RuntimeError, match="scientific integrity check"):
        namespace["run_pipeline"](
            np.ones((4, 5), dtype=np.float32),
            input_metadata={"axes": "YX"},
        )


def test_export_executes_embedded_strict_intent_and_accepts_visible_override():
    from napari_vipp.core.compute_planning import ComputePreflightError

    pipeline = _starter_pipeline()
    request = ComputeRequest(
        mode="custom",
        node_preferences={
            "gaussian": "implementation:unavailable.future.gaussian-v1"
        },
        fallback_policy="strict",
    )

    code = export_pipeline_to_python(pipeline, compute_request=request)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    embedded = json.loads(namespace["_WORKFLOW_JSON"])

    assert embedded["execution"]["compute"] == {
        "mode": "custom",
        "fallback_policy": "strict",
        "node_preferences": {
            "gaussian": "implementation:unavailable.future.gaussian-v1"
        },
        "precision_policy": "scientific-default-v1",
        "workload_policy": "vipp-best-available-v1",
    }
    image = np.ones((8, 9), dtype=np.float32)
    with pytest.raises(
        ComputePreflightError,
        match="Exact implementation .* is unavailable",
    ):
        namespace["run_pipeline"](
            image,
            input_metadata={"axes": "YX"},
        )

    result = namespace["run_pipeline"](
        image,
        input_metadata={"axes": "YX"},
        compute_request=ComputeRequest(
            mode="custom",
            node_preferences={
                "gaussian": "implementation:unavailable.future.gaussian-v1"
            },
            fallback_policy="visible",
        ),
    )

    assert result["gaussian"].shape == (8, 9)
    gaussian = next(
        item
        for item in result.execution_provenance["nodes"]
        if item["node_id"] == "gaussian"
    )
    assert gaussian["actual_implementation"]["implementation_id"] == (
        "cpu-gaussian_blur-v1"
    )
    assert gaussian["fallback_used"]
    assert gaussian["fallback_reason"] == "dependency_unavailable"
    assert json.loads(namespace["_WORKFLOW_JSON"]) == embedded


def test_exported_load_helper_returns_the_verified_frozen_payload():
    code = export_pipeline_to_python(_starter_pipeline())
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    payload = SourcePayload(np.ones((2, 3), dtype=np.uint8), {"axes": "YX"})
    calls = []

    def frozen_snapshot(path, series_index, **kwargs):
        calls.append((path, series_index, kwargs))
        return SimpleNamespace(payload=payload)

    namespace["load_frozen_file_source_snapshot"] = frozen_snapshot
    cancel_event = threading.Event()
    progress = []

    assert (
        namespace["load_image"](
            "source.ome.tif",
            series_index=2,
            progress_callback=lambda *update: progress.append(update),
            cancel_event=cancel_event,
        )
        is payload
    )
    assert len(calls) == 1
    path, series_index, kwargs = calls[0]
    assert (path, series_index) == ("source.ome.tif", 2)
    assert kwargs["cancel_callback"]() is False
    kwargs["progress_callback"](3, 9, "Hashing source bytes")
    assert progress == [("source-load", 3, 9, "Hashing source bytes")]


def test_exported_load_helper_honors_precancel_before_source_hashing():
    code = export_pipeline_to_python(_starter_pipeline())
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    cancel_event = threading.Event()
    cancel_event.set()
    called = False

    def frozen_snapshot(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("cancelled source load reached the loader")

    namespace["load_frozen_file_source_snapshot"] = frozen_snapshot

    with pytest.raises(
        namespace["OperationCancelled"],
        match="before hashing",
    ):
        namespace["load_image"]("source.ome.tif", cancel_event=cancel_event)

    assert not called


def test_exported_save_helper_passes_carried_output_state(tmp_path):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    calibrated = pipeline.add_node("set_pixel_size")
    pipeline.set_param(calibrated.id, "x_size", 0.2)
    pipeline.set_param(calibrated.id, "y_size", 0.3)
    pipeline.set_param(calibrated.id, "unit", "micrometer")
    pipeline.connect("input", calibrated.id)
    code = export_pipeline_to_python(pipeline)
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](
        np.ones((3, 4), dtype=np.float32),
        input_metadata={"axes": "YX"},
    )
    captured: dict[str, object] = {}

    def fake_write_image(data, path, **kwargs):
        captured.update(data=data, path=path, **kwargs)
        Path(path).write_bytes(b"staged-output")
        return path

    namespace["write_image"] = fake_write_image
    output_path = tmp_path / "calibrated.ome.tif"
    namespace["save_image"](
        results[calibrated.id],
        output_path,
        image_state=results.image_states[calibrated.id],
        provenance=results,
        output_node_id=calibrated.id,
    )

    assert captured["image_state"] is results.image_states[calibrated.id]
    assert captured["image_state"].axes[-1].scale == 0.2
    assert captured["image_state"].axes[-2].scale == 0.3
    assert "results.image_states.get(name)" in code
    sidecar = tmp_path / "calibrated.ome.tif.vipp-provenance.json"
    document = json.loads(sidecar.read_text(encoding="utf-8"))
    assert document["workflow"]["sha256"] == results.workflow_sha256
    assert document["output"]["node_id"] == calibrated.id
    assert document["output"]["output_port_index"] == 0
    assert document["output"]["result_context_fingerprint"]
    assert document["provenance_sha256"]
    assert document["execution"]["nodes"][0]["actual_implementation"][
        "implementation_version"
    ] == "1"


def test_exported_save_helper_uses_the_writer_normalized_path(tmp_path):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    normalized = tmp_path / "normalized.npy"

    def normalize_writer(_data, path, **_kwargs):
        staged_normalized = Path(path).with_name(normalized.name)
        staged_normalized.write_bytes(b"staged-output")
        return staged_normalized

    namespace["write_image"] = normalize_writer

    saved = namespace["save_image"](
        np.ones((2, 3), dtype=np.float32),
        tmp_path / "requested",
        provenance={"type": "test-output-provenance"},
    )

    assert saved == normalized
    assert (tmp_path / "normalized.npy.vipp-provenance.json").exists()
    assert not (tmp_path / "requested.vipp-provenance.json").exists()


def test_exported_save_helper_withholds_output_when_provenance_staging_fails(
    tmp_path,
):
    namespace: dict[str, object] = {"__name__": "exported_pipeline"}
    exec(
        compile(
            export_pipeline_to_python(_starter_pipeline()),
            "<exported>",
            "exec",
        ),
        namespace,
    )
    output_path = tmp_path / "output.tif"
    sidecar = namespace["_provenance_sidecar_path"](output_path)
    output_path.write_bytes(b"pre-existing-output")
    sidecar.write_text(
        '{"type":"pre-existing-provenance"}\n',
        encoding="utf-8",
    )

    def staged_writer(_data, path, **_kwargs):
        Path(path).write_bytes(b"new-staged-output")
        return path

    def fail_provenance(_path, _provenance):
        raise OSError("simulated provenance failure")

    namespace["write_image"] = staged_writer
    namespace["save_provenance_sidecar"] = fail_provenance

    with pytest.raises(OSError, match="simulated provenance failure"):
        namespace["save_image"](
            np.ones((2, 3), dtype=np.float32),
            output_path,
            provenance={"type": "exact-output-provenance"},
        )

    assert output_path.read_bytes() == b"pre-existing-output"
    assert json.loads(sidecar.read_text(encoding="utf-8")) == {
        "type": "pre-existing-provenance"
    }
    assert not tuple(tmp_path.glob(".vipp-publish-*"))
