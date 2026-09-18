"""Statistics preserves workflows and ordinary table/plot/export boundaries."""

from __future__ import annotations

import csv
import json
import runpy
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.batch import (
    scientific_workflow_document,
    scientific_workflow_hash,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.operation_search import operation_search_aliases
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.result_plots import PlotData
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import (
    canonical_workflow_document,
    deserialize_workflow,
    serialize_workflow,
)

MODERN_FIELDS = {
    "summary_version",
    "summary_level",
    "image_column",
    "sample_column",
    "sample_weighting",
    "missing_policy",
}


def _pipeline():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source = pipeline.nodes["input"]
    threshold = pipeline.add_node("binary_threshold")
    threshold.params["threshold"] = 5
    labels = pipeline.add_node("label_connected_components")
    measurements = pipeline.add_node("measure_objects")
    annotations = pipeline.add_node("add_metadata_columns")
    annotations.params["metadata_columns"] = "condition=demo, sample_id=sample-a"
    summary = pipeline.add_node("summarize_measurements")
    summary.params.update(
        value_columns="area_pixels",
        statistics="count,mean,std",
        group_by="condition",
        image_column="t_index",
        sample_column="sample_id",
    )
    for first, second in zip(
        [source, threshold, labels, measurements, annotations],
        [threshold, labels, measurements, annotations, summary],
        strict=True,
    ):
        assert pipeline.connect(first.id, second.id).success
    image = np.zeros((2, 16, 16), dtype=np.float32)
    image[0, 1:3, 1:3] = image[0, 7:11, 7:11] = 10  # areas 4 and 16
    image[1, 2:8, 2:8] = 10  # area 36
    image.setflags(write=False)
    return pipeline, summary, image


def test_new_node_has_descriptive_defaults_and_old_name_remains_searchable():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("summarize_measurements")
    assert node.title == "Statistics"
    assert node.params["summary_version"] == 2
    assert node.params["group_by"] == ""
    assert node.params["summary_level"] == "Objects"
    assert node.params["image_column"] == node.params["sample_column"] == ""
    assert "summarize measurements" in operation_search_aliases(node.operation_id)


def test_legacy_restore_retains_parameters_schema_and_batch_hash():
    pipeline, summary, _ = _pipeline()
    document = serialize_workflow(pipeline)
    saved = next(n for n in document["nodes"] if n["id"] == summary.id)
    for key in MODERN_FIELDS:
        saved["params"].pop(key)
    saved["params"]["group_by"] = "auto"
    before = deepcopy(document)
    restored = canonical_workflow_document(document)
    current = next(n for n in restored["nodes"] if n["id"] == summary.id)
    assert current["params"]["summary_version"] == 1
    assert current["params"]["group_by"] == "auto"
    assert document == before
    assert scientific_workflow_hash(document) == scientific_workflow_hash(restored)
    scientific_node = next(
        n
        for n in scientific_workflow_document(restored)["nodes"]
        if n["id"] == summary.id
    )
    assert not MODERN_FIELDS.intersection(scientific_node["params"])
    upgraded = deepcopy(restored)
    next(n for n in upgraded["nodes"] if n["id"] == summary.id)["params"][
        "summary_version"
    ] = 2
    assert scientific_workflow_hash(upgraded) != scientific_workflow_hash(restored)


@pytest.mark.parametrize("version", [0, 3, True, 1.5, "2"])
def test_unknown_or_malformed_version_is_not_silently_restored(version):
    pipeline, summary, _ = _pipeline()
    document = serialize_workflow(pipeline)
    next(n for n in document["nodes"] if n["id"] == summary.id)["params"][
        "summary_version"
    ] = version
    with pytest.raises(ValueError, match="Statistics version"):
        deserialize_workflow(document)


def test_partial_v2_recipe_is_rejected_but_legacy_additions_are_migrated():
    pipeline, summary, _ = _pipeline()
    document = serialize_workflow(pipeline)
    params = next(n for n in document["nodes"] if n["id"] == summary.id)["params"]
    params.pop("sample_weighting")
    with pytest.raises(ValueError, match="missing required parameters"):
        deserialize_workflow(document)
    params["summary_version"] = 1
    assert (
        next(
            n for n in deserialize_workflow(document)["nodes"] if n.id == summary.id
        ).params["sample_weighting"]
        == "Equal images"
    )


@pytest.mark.parametrize(
    ("level", "expected", "n"),
    [("Objects", 56 / 3, 3), ("Image averages", 23, 2), ("Sample averages", 23, 1)],
)
def test_shared_execution_levels_preserve_units_counts_and_input(level, expected, n):
    pipeline, summary, image = _pipeline()
    summary.params["summary_level"] = level
    before = image.copy()
    request = PipelineRunRequest(
        run_id=1,
        workflow=serialize_workflow(pipeline),
        input_data=image,
        input_metadata={"axes": "TYX"},
        input_name="Synthetic fields",
        source_payloads={},
        compute_request=ComputeRequest(mode="cpu"),
        manual_node_ids=frozenset(pipeline.nodes),
    )
    result = execute_pipeline_request(request, raise_errors=True)
    assert not result.error
    table = result.pipeline.outputs[summary.id]
    assert isinstance(table, TableData)
    row = table.records()[0]
    assert row["area_pixels_mean"] == pytest.approx(expected)
    assert row["area_pixels_n"] == n
    assert row["area_pixels_object_total"] == row["area_pixels_object_valid"] == 3
    assert row["area_pixels_object_excluded"] == 0
    assert table.unit_for("area_pixels_mean")
    assert summary.id in result.pipeline.node_compute_provenance
    np.testing.assert_array_equal(image, before)
    if level == "Sample averages":
        assert row["area_pixels_std"] is None


def test_summary_flows_to_plot_and_generated_python_without_embedding_measurements():
    pipeline, summary, image = _pipeline()
    plot = pipeline.add_node("plot_results")
    plot.params.update(y_column="area_pixels_mean", group_column="condition")
    assert pipeline.connect(summary.id, plot.id).success
    code = export_pipeline_to_python(pipeline)
    namespace = {"__name__": "statistics_export_test"}
    exec(compile(code, "<statistics-export>", "exec"), namespace)
    results = namespace["run_pipeline"](image, input_metadata={"axes": "TYX"})
    assert isinstance(results[summary.id], TableData)
    assert isinstance(results[plot.id], PlotData)
    assert results[plot.id].series[0].y == pytest.approx((56 / 3,))
    assert "statistics_recipe" in results[summary.id].columns
    assert '"rows"' not in json.dumps(serialize_workflow(pipeline))


@pytest.mark.parametrize("file_format", ["csv", "tsv"])
def test_batch_table_staging_keeps_descriptive_evidence(tmp_path, file_format):
    from napari_vipp.core.batch import (
        BatchOutputPlan,
        ExistingFilePolicy,
        _save_planned_output,
    )

    pipeline, summary, image = _pipeline()
    summary.params["summary_level"] = "Sample averages"
    output = pipeline.add_node("batch_output")
    output.params.update(format=file_format, tag="statistics")
    assert pipeline.connect(summary.id, output.id).success
    pipeline.run(image, input_metadata={"axes": "TYX"})
    path = tmp_path / f"summary.{file_format}"
    plan = BatchOutputPlan(
        output.id,
        "Statistics",
        "statistics",
        "table",
        file_format,
        path,
        ExistingFilePolicy.ERROR,
    )
    staged = _save_planned_output(pipeline, plan)
    assert not path.exists()
    with staged.saved_temporary_path.open(encoding="utf-8", newline="") as stream:
        rows = list(
            csv.DictReader(stream, delimiter="," if file_format == "csv" else "\t")
        )
    assert rows[0]["area_pixels_n"] == "1"
    assert rows[0]["area_pixels_std"] == ""
    assert rows[0]["summary_level"] == "Sample averages"
    assert "standard deviation undefined for n < 2" in rows[0]["area_pixels_status"]


def test_review_demo_measures_real_objects_and_distinguishes_weighting(tmp_path):
    from napari_vipp.core.measurement_collection import load_measurement_collection

    generator = runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "create_statistics_review_demo.py"
        )
    )
    paths = generator["create_demo"](tmp_path / "statistics-demo")
    collection = load_measurement_collection(paths["collection"])
    assert collection.table.row_count == 258
    assert [item.row_count for item in collection.items] == [24, 40, 60, 30, 46, 58]
    assert all(item.included and item.result_sha256 for item in collection.items)
    document = json.loads(paths["workflow"].read_text(encoding="utf-8"))
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=4,
            workflow=document,
            input_data=None,
            input_metadata={},
            input_name="",
            source_payloads={},
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(node["id"] for node in document["nodes"]),
        ),
        raise_errors=True,
    )
    assert not result.error and not result.cancelled
    raw_rows = collection.table.records()
    summaries = []
    for node in document["nodes"]:
        if node["operation_id"] != "summarize_measurements":
            continue
        params = node["params"]
        table = result.pipeline.outputs[node["id"]]
        assert table.unit_for("area_physical_mean") == "µm^2"
        rows = table.records()
        assert len(rows) == 2
        summaries.append(rows)
        for row in rows:
            objects = [r for r in raw_rows if r["condition"] == row["condition"]]
            samples = {}
            for obj in objects:
                samples.setdefault(obj["sample_id"], {}).setdefault(
                    obj["image_id"], []
                ).append(obj["area_physical"])
            image_values = [
                np.mean(values)
                for images in samples.values()
                for values in images.values()
            ]
            if params["summary_level"] == "Objects":
                expected = [obj["area_physical"] for obj in objects]
            elif params["summary_level"] == "Image averages":
                expected = image_values
            elif params["sample_weighting"] == "Equal images":
                expected = [
                    np.mean([np.mean(values) for values in images.values()])
                    for images in samples.values()
                ]
            else:
                expected = [
                    np.mean([v for values in images.values() for v in values])
                    for images in samples.values()
                ]
            assert row["area_physical_n"] == len(expected)
            assert row["area_physical_mean"] == pytest.approx(np.mean(expected))
            assert row["area_physical_std"] == pytest.approx(np.std(expected, ddof=1))
            assert row["area_physical_object_total"] == len(objects)
            assert row["area_physical_object_excluded"] == 0
            assert row["image_count"] == 3 and row["sample_count"] == 2
    assert len(summaries) == 4
    assert summaries[2][0]["area_physical_mean"] != pytest.approx(
        summaries[3][0]["area_physical_mean"]
    )
    with pytest.raises(FileExistsError):
        generator["create_demo"](tmp_path / "statistics-demo")
