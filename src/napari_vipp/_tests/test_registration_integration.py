"""Durable publication, source revisions and cache boundaries for registration."""

import json
import threading
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    BatchStatus,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import SourceRevisionToken
from napari_vipp.core.transforms import TransformData, TransformState, load_transform
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _series():
    y, x = np.indices((48, 64), dtype=np.float64)
    frames = []
    for dy, dx in ((0, 0), (2, -3), (-1, 2)):
        image = (
            np.exp(-((y - dy - 14) ** 2 / 24 + (x - dx - 17) ** 2 / 36))
            + 0.7 * np.exp(-((y - dy - 31) ** 2 / 32 + (x - dx - 42) ** 2 / 20))
            + 0.4 * np.exp(-((y - dy - 12) ** 2 / 10 + (x - dx - 48) ** 2 / 16))
        )
        frames.append(image.astype(np.float32))
    result = np.stack(frames)
    result.setflags(write=False)
    return result


def _pipeline(*, batch=False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    estimate = pipeline.add_node("estimate_registration")
    pipeline.set_param(estimate.id, "mode", "Time series")
    apply = pipeline.add_node("apply_transform")
    assert pipeline.connect("input", estimate.id).success
    assert pipeline.connect("input", apply.id, target_port=0).success
    assert pipeline.connect(estimate.id, apply.id, target_port=1).success
    output_id = None
    if batch:
        output = pipeline.add_node("batch_output")
        pipeline.set_param(output.id, "tag", "motion")
        pipeline.set_param(output.id, "format", "json")
        assert pipeline.connect(estimate.id, output.id).success
        output_id = output.id
    return pipeline, estimate.id, apply.id, output_id


def _payload(*, revision=1):
    data = _series()
    metadata = {"axes": "TYX"}
    return SourcePayload(
        data,
        metadata,
        "synthetic drift",
        revision_token=SourceRevisionToken(layer_id=571, revision=revision),
    )


def _cache(pipeline):
    return dict(
        cached_outputs=dict(pipeline.outputs),
        cached_output_states=dict(pipeline.output_states),
        cached_node_outputs={
            key: list(value) for key, value in pipeline.node_outputs.items()
        },
        cached_node_output_states={
            key: list(value) for key, value in pipeline.node_output_states.items()
        },
        completed_node_ids=frozenset(pipeline.completed_node_ids),
        cached_execution_states=dict(pipeline.node_execution_states),
        cached_execution_messages=dict(pipeline.node_execution_messages),
        cached_compute_provenance={
            **pipeline.node_cache_lineage,
            **pipeline.node_compute_provenance,
        },
    )


def _request(pipeline, payload, **kwargs):
    return PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline),
        input_data=None,
        input_metadata=None,
        input_name="",
        source_payloads={"input": payload},
        compute_request=ComputeRequest(mode="cpu"),
        manual_node_ids=frozenset(pipeline.manual_node_ids()),
        **kwargs,
    )


def _batch(tmp_path):
    pipeline, estimate, _apply, output = _pipeline(batch=True)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    tifffile.imwrite(
        inputs / "drift.ome.tif",
        _series(),
        ome=True,
        metadata={
            "axes": "TYX",
            "PhysicalSizeX": 0.5,
            "PhysicalSizeY": 0.7,
            "TimeIncrement": 2.0,
        },
        photometric="minisblack",
    )
    workflow = serialize_workflow(pipeline)
    config = BatchConfig(
        workflow_file=Path("workflow.json"),
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Drift series", inputs, "*.ome.tif"),),
        outputs=(
            BatchOutputConfig(
                output,
                "Batch Output",
                "motion",
                "transform",
                "json",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        compute_request=ComputeRequest(mode="cpu"),
        save_python_script=True,
    )
    return workflow, config, estimate


def test_batch_publishes_registration_transform_json_and_provenance(tmp_path):
    workflow, config, estimate = _batch(tmp_path)
    result = run_batch(workflow, config)
    assert result.manifest.items[0].status is BatchStatus.COMPLETED
    assert len(result.saved_paths) == 1
    path = result.saved_paths[0]
    assert path.suffix == ".json" and "motion" in path.stem
    restored = load_transform(path)
    assert restored.is_time_series and len(restored.matrices) == 3
    assert restored.moving_grid.spacing == (0.7, 0.5)
    np.testing.assert_allclose(
        np.asarray(restored.matrices)[1, :2, 2], (-1.4, 1.5), atol=1e-8
    )
    assert estimate in {
        node["node_id"] for node in result.manifest.items[0].execution["nodes"]
    }
    serialized = json.loads(path.read_text(encoding="utf-8"))
    assert serialized["direction"] == "moving-to-reference"
    assert serialized["implementation"]
    assert serialized["moving_grid"]["frame_id"]
    assert result.manifest.workflow_sha256 == config.workflow_sha256


def test_generated_python_saves_transform_with_exact_provenance_sidecar(tmp_path):
    pipeline, estimate, _apply, _output = _pipeline()
    namespace = {"__name__": "registration_export"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<registration export>", "exec"),
        namespace,
    )
    results = namespace["run_pipeline"](source_payloads={"input": _payload()})
    saved = namespace["save_image"](
        results[estimate],
        tmp_path / "drift-transform.tif",
        provenance=results,
        output_node_id=estimate,
    )
    assert saved == tmp_path / "drift-transform.json"
    assert load_transform(saved).to_dict() == results[estimate].to_dict()
    sidecar = json.loads(
        (tmp_path / "drift-transform.json.vipp-provenance.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["workflow"]["sha256"] == results.workflow_sha256
    assert sidecar["output"]["node_id"] == estimate
    assert sidecar["execution"]["cleanup_succeeded"]
    assert not (tmp_path / "drift-transform.tif").exists()


def test_cancelled_transform_publication_preserves_previous_complete_output(tmp_path):
    pipeline, estimate, _apply, _output = _pipeline()
    namespace = {"__name__": "registration_cancelled_export"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<registration export>", "exec"),
        namespace,
    )
    results = namespace["run_pipeline"](source_payloads={"input": _payload()})
    path = tmp_path / "motion.json"
    sidecar = tmp_path / "motion.json.vipp-provenance.json"
    path.write_bytes(b"previous complete transform")
    sidecar.write_bytes(b"previous exact provenance")
    cancel = threading.Event()
    original_save = namespace["save_transform_output"]

    def write_then_cancel(data, staged_path):
        result = original_save(data, staged_path)
        cancel.set()
        return result

    namespace["save_transform_output"] = write_then_cancel
    with pytest.raises(OperationCancelled):
        namespace["save_image"](
            results[estimate],
            path,
            provenance=results,
            output_node_id=estimate,
            cancel_event=cancel,
        )
    assert path.read_bytes() == b"previous complete transform"
    assert sidecar.read_bytes() == b"previous exact provenance"
    assert not tuple(tmp_path.glob(".vipp-publish-*"))


def test_workflow_roundtrip_keeps_dynamic_ports_and_rejects_wrong_domains():
    pipeline, estimate, apply, _output = _pipeline()
    compare = pipeline.add_node("compare_images")
    assert len(pipeline.input_ports(estimate)) == 1
    assert len(pipeline.input_ports(compare.id)) == 2
    assert [port.output_type for port in pipeline.output_ports(estimate)] == [
        "transform",
        "table",
    ]
    assert [port.output_type for port in pipeline.output_ports(apply)] == [
        "image",
        "mask",
    ]
    assert not pipeline.connect("input", apply, target_port=1).success
    assert not pipeline.connect(estimate, apply, target_port=1, source_port=1).success
    assert not pipeline.connect(estimate, compare.id, target_port=0).success
    assert not pipeline.connect("input", estimate, target_port=1).success
    pipeline.set_param(estimate, "mode", "Two images")
    pipeline.set_param(compare.id, "use_mask", True)
    assert len(pipeline.input_ports(estimate)) == 2
    assert len(pipeline.input_ports(compare.id)) == 3
    assert pipeline.connect("input", estimate, target_port=1).success
    assert pipeline.connect(apply, compare.id, target_port=2, source_port=1).success
    restored = deserialize_workflow(serialize_workflow(pipeline))
    clone = PrototypePipeline()
    clone.restore_graph(
        restored["nodes"], restored["connections"], restored["output_tunnels"]
    )
    assert len(clone.input_ports(estimate)) == 2
    assert len(clone.input_ports(compare.id)) == 3
    clone.set_param(estimate, "mode", "Time series")
    clone.set_param(compare.id, "use_mask", False)
    assert len(clone.input_ports(estimate)) == 1
    assert len(clone.input_ports(compare.id)) == 2
    assert all(c.target_port == 0 for c in clone.connections if c.target_id == estimate)
    assert not any(
        c.target_id == compare.id and c.target_port == 2 for c in clone.connections
    )


def test_transform_cache_deepcopy_and_parameter_dirty_rerun_preserve_old_result():
    pipeline, estimate, apply, _output = _pipeline()
    payload = _payload()
    first = execute_pipeline_request(
        _request(pipeline, payload), raise_errors=True
    ).pipeline
    transform = first.outputs[estimate]
    assert isinstance(transform, TransformData)
    assert isinstance(first.output_states[estimate], TransformState)
    assert deepcopy(transform).to_dict() == transform.to_dict()
    initial_document = transform.to_dict()
    # An Apply-only edit must reuse the scientific transform, not estimate again.
    first.set_param(apply, "outside_value", -5.0)
    started = []
    repeated = execute_pipeline_request(
        _request(first, payload, dirty_node_ids=frozenset({apply}), **_cache(first)),
        node_started_callback=started.append,
        raise_errors=True,
    ).pipeline
    assert estimate not in started
    assert repeated.outputs[estimate].to_dict() == initial_document
    assert np.any(repeated.outputs[apply] == -5)
    repeated.set_param(estimate, "reference_time", 1)
    changed = execute_pipeline_request(
        _request(
            repeated,
            payload,
            dirty_node_ids=frozenset({estimate, apply}),
            **_cache(repeated),
        ),
        raise_errors=True,
    ).pipeline
    assert changed.outputs[estimate].reference_time == 1
    np.testing.assert_array_equal(changed.outputs[estimate].matrices[1], np.eye(3))
    assert transform.to_dict() == initial_document


def test_changed_source_revision_does_not_reuse_stale_registration_frame():
    pipeline, estimate, apply, _output = _pipeline()
    first = execute_pipeline_request(
        _request(pipeline, _payload()), raise_errors=True
    ).pipeline
    previous = first.outputs[estimate]
    revised_data = _series().copy()
    revised_data[2] = np.roll(revised_data[2], 1, axis=1)
    revised_data.setflags(write=False)
    revised_payload = replace(_payload(revision=2), data=revised_data)
    started = []
    changed = execute_pipeline_request(
        _request(
            first,
            revised_payload,
            dirty_node_ids=frozenset({apply}),
            **_cache(first),
        ),
        node_started_callback=started.append,
        raise_errors=True,
    ).pipeline
    assert estimate in started
    assert (
        previous.moving_grid.frame_id != changed.outputs[estimate].moving_grid.frame_id
    )
    assert first.outputs[estimate] is previous


def test_cancelled_estimation_never_exposes_partial_transform_or_diagnostics():
    pipeline, estimate, _apply, _output = _pipeline()
    cancel = threading.Event()
    seen = []

    def progress(node_id, current, total, message):
        seen.append((node_id, current, total, message))
        if node_id == estimate and current >= 1:
            cancel.set()

    result = execute_pipeline_request(
        _request(pipeline, _payload(), cancel_event=cancel), progress_callback=progress
    )
    assert seen and cancel.is_set() and result.cancelled
    assert result.failure.kind == "cancelled"
    assert result.pipeline is None or not isinstance(
        result.pipeline.outputs.get(estimate), TransformData
    )
    if result.pipeline is not None:
        assert all(
            value is None for value in result.pipeline.node_outputs.get(estimate, ())
        )


def test_cancelled_batch_never_publishes_partial_transform(tmp_path, monkeypatch):
    import napari_vipp.core.batch as batch_module

    workflow, config, estimate = _batch(tmp_path)
    cancelled = threading.Event()
    original = batch_module.execute_pipeline_request

    def cancellable(request, **kwargs):
        old_progress = kwargs.get("progress_callback")

        def progress(node_id, current, total, message):
            if old_progress:
                old_progress(node_id, current, total, message)
            if node_id == estimate and current >= 1:
                cancelled.set()

        return original(request, **(kwargs | {"progress_callback": progress}))

    monkeypatch.setattr(batch_module, "execute_pipeline_request", cancellable)
    result = run_batch(workflow, config, cancel_event=cancelled)
    assert result.cancelled
    assert result.manifest.items[0].status is BatchStatus.CANCELLED
    assert not result.saved_paths
    assert not list(config.output_dir.glob("*__motion.json"))


def test_carried_labels_remain_labels_in_source_and_apply_output():
    from napari_vipp._sample_data import make_registration_sample_data

    samples = make_registration_sample_data()
    image, image_kwargs, _kind = samples[-2]
    labels, label_kwargs, _kind = samples[-1]
    pipeline, estimate, apply, _output = _pipeline()
    second = pipeline.add_node("input")
    label_apply = pipeline.add_node("apply_transform")
    assert pipeline.connect(second.id, label_apply.id, target_port=0).success
    assert pipeline.connect(estimate, label_apply.id, target_port=1).success
    image_payload = SourcePayload(image, image_kwargs["metadata"], image_kwargs["name"])
    request = replace(
        _request(pipeline, image_payload),
        source_payloads={
            "input": image_payload,
            second.id: SourcePayload(
                labels, label_kwargs["metadata"], label_kwargs["name"]
            ),
        },
    )
    result = execute_pipeline_request(request, raise_errors=True).pipeline
    assert result.output_states[second.id].kind == "label image"
    assert result.output_states[label_apply.id].kind == "label image"
    assert result.outputs[label_apply.id].dtype == labels.dtype
    assert set(np.unique(result.outputs[label_apply.id])) <= set(np.unique(labels))
    assert result.node_outputs[apply][1].dtype == bool
    assert result.node_outputs[label_apply.id][1].dtype == bool
