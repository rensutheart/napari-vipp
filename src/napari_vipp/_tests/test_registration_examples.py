"""Execute shipped registration graphs, not just isolated backend functions."""

from pathlib import Path

import numpy as np
import pytest

from napari_vipp._sample_data import make_registration_sample_data
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.registration_samples import (
    drift_series,
    landmark_errors,
    rigid_volume_pair,
    translation_pair,
)
from napari_vipp.core.workflow import load_workflow, serialize_workflow
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path

ROOT = Path(__file__).resolve().parents[3]


def run_registration_example(example_id):
    spec = _example_workflow_by_id(example_id)
    document = load_workflow(_example_workflow_path(spec))
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    samples = {
        kwargs["name"]: (data, kwargs)
        for data, kwargs, _kind in make_registration_sample_data()
    }
    payloads = {}
    for node in pipeline.nodes.values():
        if node.operation_id == "input":
            data, kwargs = samples[node.params["sample_name"]]
            payloads[node.id] = SourcePayload(data, kwargs["metadata"], kwargs["name"])
    original = payloads["input"]
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=original.data,
            input_metadata=original.metadata,
            input_name=original.name,
            source_payloads=payloads,
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        ),
        raise_errors=True,
    )
    assert not result.error and not result.cancelled
    return result.pipeline, document, payloads


@pytest.mark.parametrize(
    "example_id,factory,tolerance",
    [
        ("registration-translation", translation_pair, 0.12),
        ("registration-rigid-3d", rigid_volume_pair, 0.20),
        ("registration-time-series", drift_series, 0.12),
    ],
)
def test_shipped_registration_examples_match_analytical_motion(
    example_id, factory, tolerance, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    pipeline, document, payloads = run_registration_example(example_id)
    phantom = factory()
    transform = pipeline.outputs["estimate"]
    assert len(transform.matrices) == len(phantom.moving_to_reference)
    for time_index, matrix in enumerate(transform.matrices):
        assert landmark_errors(matrix, phantom, time_index=time_index).max() < tolerance
    assert pipeline.node_outputs["estimate"][1].row_count == len(transform.matrices)
    coverage = pipeline.node_outputs["apply"][1]
    assert coverage.dtype == bool
    assert 0.6 < coverage.mean() < 1
    assert pipeline.outputs["apply"].shape == phantom.moving.shape
    assert set(document["positions"]) == set(pipeline.nodes)
    spec = _example_workflow_by_id(example_id)
    assert (ROOT / "examples" / spec.filename).read_bytes() == _example_workflow_path(
        spec
    ).read_bytes()
    assert all(not payload.data.flags.writeable for payload in payloads.values())
    assert {path.name for path in tmp_path.iterdir()} <= {".napari-vipp-test-state"}

    if example_id == "registration-time-series":
        labels = pipeline.outputs["apply_labels"]
        assert labels.dtype == phantom.labels.dtype
        assert set(np.unique(labels)) <= set(np.unique(phantom.labels))
        for time_index in range(1, 6):
            for channel in range(2):
                mask = coverage[time_index, channel]
                before = phantom.moving[time_index, channel][mask]
                after = pipeline.outputs["apply"][time_index, channel][mask]
                reference = phantom.moving[0, channel][mask]
                assert (
                    np.mean((after - reference) ** 2)
                    < np.mean((before - reference) ** 2) / 20
                )
    else:
        assert pipeline.outputs["compare_before"].row_count > 0
        assert pipeline.outputs["compare_after"].row_count > 0
        pairs = {
            (c.source_id, c.target_id, c.source_port, c.target_port)
            for c in pipeline.connections
        }
        assert ("apply", "compare_before", 1, 2) in pairs
        assert ("apply", "compare_after", 1, 2) in pairs


def test_registration_examples_use_existing_non_mutating_sources_and_safe_models():
    for example_id in (
        "registration-translation",
        "registration-rigid-3d",
        "registration-time-series",
    ):
        spec = _example_workflow_by_id(example_id)
        workflow = load_workflow(_example_workflow_path(spec))
        assert not any(node.operation_id == "save_output" for node in workflow["nodes"])
        assert all(node.params.get("model") != "Affine" for node in workflow["nodes"])
        assert len(workflow["notes"]) >= 3


@pytest.mark.parametrize(
    "example_id", ["registration-translation", "registration-time-series"]
)
def test_shipped_registration_examples_export_the_same_known_motion(
    example_id, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    pipeline, _document, payloads = run_registration_example(example_id)
    namespace = {"__name__": "registration_example_export"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<registration example>", "exec"),
        namespace,
    )
    outputs = namespace["run_pipeline"](
        source_payloads=payloads, compute_request=ComputeRequest(mode="cpu")
    )
    np.testing.assert_allclose(
        outputs["estimate"].matrices, pipeline.outputs["estimate"].matrices
    )
    np.testing.assert_array_equal(outputs["apply"], pipeline.outputs["apply"])
    if example_id == "registration-time-series":
        np.testing.assert_array_equal(
            outputs["apply_labels"], pipeline.outputs["apply_labels"]
        )
