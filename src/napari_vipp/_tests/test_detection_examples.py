"""Known construction truth, independent of the matching implementation."""

from pathlib import Path

import numpy as np
import pytest

from napari_vipp._sample_data import make_detection_sample_data
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
from napari_vipp.core.workflow import load_workflow, serialize_workflow
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path

ROOT = Path(__file__).resolve().parents[3]


def run_detection_example(dimensions, *, separation=None):
    spec = _example_workflow_by_id(f"template-detection-{dimensions}d")
    document = load_workflow(_example_workflow_path(spec))
    pipeline = PrototypePipeline()
    pipeline.restore_graph(
        document["nodes"], document["connections"], document["output_tunnels"]
    )
    if separation is not None:
        pipeline.nodes["peaks"].params["minimum_separation"] = separation
    data, kwargs, _kind = make_detection_sample_data()[dimensions - 2]
    original = data.copy()
    payload = SourcePayload(data, kwargs["metadata"], kwargs["name"])
    payloads = {"input": payload}
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=data,
            input_metadata=payload.metadata,
            input_name=payload.name,
            source_payloads=payloads,
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        ),
        raise_errors=True,
    )
    assert not result.error and not result.cancelled
    np.testing.assert_array_equal(data, original)
    return (
        result.pipeline,
        document,
        payloads,
        kwargs["metadata"]["detection_ground_truth"],
    )


def test_detection_phantoms_have_seeded_independent_construction_truth():
    first, again = make_detection_sample_data(), make_detection_sample_data()
    different = make_detection_sample_data(seed=20260929)
    for (data, kwargs, kind), duplicate, changed in zip(
        first, again, different, strict=True
    ):
        truth = kwargs["metadata"]["detection_ground_truth"]
        assert kind == "image" and data.dtype == np.float32
        assert data.shape[:2] == (2, 2)
        assert kwargs["metadata"]["channel_names"][1] == "Repeated pattern"
        np.testing.assert_array_equal(data, duplicate[0])
        assert not np.array_equal(data, changed[0])
        assert (
            truth["centers"]
            == changed[1]["metadata"]["detection_ground_truth"]["centers"]
        )
        assert truth["missing_center"] not in truth["centers"]
        assert truth["border_center"] not in truth["centers"]
        missing = (1, 1, *truth["missing_center"])
        assert data[missing] < 0.15
        assert data[(1, 1, *truth["centers"][0])] > 0.8
        assert data[(1, 0, *truth["centers"][0])] < 0.15


@pytest.mark.parametrize("dimensions", [2, 3])
def test_detection_examples_recover_exact_centers_and_calibration(
    dimensions, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    pipeline, document, payloads, truth = run_detection_example(dimensions)
    records = pipeline.outputs["peaks"].records()
    axes = truth["spatial_axis_order"].lower()
    found = {tuple(row[f"{axis}_index"] for axis in axes) for row in records}
    assert found == {tuple(center) for center in truth["centers"]}
    assert tuple(truth["missing_center"]) not in found
    assert tuple(truth["border_center"]) not in found
    assert all(0.8 <= row["score"] <= 1 for row in records)
    for row in records:
        for axis, spacing, origin, extent in zip(
            axes,
            truth["spacing"],
            truth["origin"],
            truth["template_shape"],
            strict=True,
        ):
            assert row[f"{axis}_physical"] == pytest.approx(
                origin + spacing * row[f"{axis}_index"]
            )
            assert (
                row[f"template_{axis}_start"] == row[f"{axis}_index"] - (extent - 1) / 2
            )
            assert (
                row[f"template_{axis}_stop"] - row[f"template_{axis}_start"] == extent
            )
    source_shape = payloads["input"].data.shape[2:]
    scores, valid = pipeline.node_outputs["match"]
    assert scores.shape == tuple(
        n - k + 1 for n, k in zip(source_shape, truth["template_shape"], strict=True)
    )
    assert valid.dtype == bool and valid.shape == scores.shape
    assert pipeline.outputs["template"].shape == tuple(truth["template_shape"])
    assert pipeline.outputs["scores"] is not None
    assert set(document["positions"]) == set(pipeline.nodes)
    assert len(document["notes"]) >= 4
    assert not any(
        node.operation_id == "save_output" for node in pipeline.nodes.values()
    )
    assert {path.name for path in tmp_path.iterdir()} <= {".napari-vipp-test-state"}
    spec = _example_workflow_by_id(f"template-detection-{dimensions}d")
    assert (ROOT / "examples" / spec.filename).read_bytes() == _example_workflow_path(
        spec
    ).read_bytes()


@pytest.mark.parametrize("dimensions,separation", [(2, 13.0), (3, 7.0)])
def test_detection_examples_explain_nearby_peak_suppression(dimensions, separation):
    pipeline, _, _, truth = run_detection_example(dimensions, separation=separation)
    assert pipeline.outputs["peaks"].row_count == len(truth["centers"]) - 1


@pytest.mark.parametrize("dimensions", [2, 3])
def test_detection_examples_export_same_table_and_scores(
    dimensions, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    pipeline, _, payloads, _ = run_detection_example(dimensions)
    namespace = {"__name__": "detection_example_export"}
    exec(
        compile(export_pipeline_to_python(pipeline), "<detection example>", "exec"),
        namespace,
    )
    outputs = namespace["run_pipeline"](
        source_payloads=payloads, compute_request=ComputeRequest(mode="cpu")
    )
    assert outputs["peaks"].records() == pipeline.outputs["peaks"].records()
    np.testing.assert_array_equal(outputs["match"], pipeline.outputs["match"])
