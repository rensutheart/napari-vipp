"""Known, single-image structure behind the guided measurement-plot example."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from napari_vipp._sample_data import _measurement_plot_sample
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.result_plots import PlotData, PlotState
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path


def _document():
    spec = _example_workflow_by_id("plot-morphology")
    assert spec is not None
    return json.loads(_example_workflow_path(spec).read_text(encoding="utf-8"))


def test_plot_sample_has_sixty_isolated_calibrated_objects():
    data, kwargs, kind = _measurement_plot_sample()
    repeated, _, _ = _measurement_plot_sample()
    assert kind == "image" and data.dtype == np.uint16
    np.testing.assert_array_equal(data, repeated)
    labels, count = ndimage.label(data > 1000)
    assert count == 60
    assert not np.any(labels[[0, -1], :])
    assert not np.any(labels[:, [0, -1]])
    sizes = np.bincount(labels.ravel())[1:]
    assert sizes.max() / sizes.min() > 4
    scale = kwargs["metadata"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"]
    assert scale == [0.5, 0.5]
    assert "Not biological data" in kwargs["metadata"]["description"]


def test_plot_example_measurements_are_useful_without_a_batch(tmp_path, monkeypatch):
    document = _document()
    # Isolate the scientific measurement fixture from the plot renderer. The
    # end-to-end test below additionally exercises all four real plot results.
    plot_ids = {
        node["id"]
        for node in document["nodes"]
        if node["operation_id"] == "plot_results"
    }
    document["nodes"] = [
        node for node in document["nodes"] if node["id"] not in plot_ids
    ]
    document["connections"] = [
        edge for edge in document["connections"] if edge["target"] not in plot_ids
    ]
    document["positions"] = {
        node_id: position
        for node_id, position in document["positions"].items()
        if node_id not in plot_ids
    }
    document["metadata"] = {}
    data, kwargs, _kind = _measurement_plot_sample()
    original = data.copy()
    data.setflags(write=False)
    monkeypatch.chdir(tmp_path)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=document,
            input_data=data,
            input_metadata=kwargs["metadata"],
            input_name=kwargs["name"],
            source_payloads={},
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(node["id"] for node in document["nodes"]),
        ),
        raise_errors=True,
    )
    assert not result.error and not result.cancelled
    rows = result.pipeline.outputs["annotated"].records()
    assert len(rows) == 60
    assert {row["label_id"] for row in rows} == set(range(1, 61))
    assert {row["image_id"] for row in rows} == {"synthetic_field_01"}
    area = np.array([row["area_physical"] for row in rows])
    intensity = np.array([row["intensity_mean"] for row in rows])
    for row in rows:
        assert row["area_physical"] == pytest.approx(row["area_pixels"] * 0.25)
        assert 0 <= row["eccentricity"] <= 1
        assert row["axis_ratio_major_minor"] >= 1
    assert np.corrcoef(area, intensity)[0, 1] > 0.98
    assert area.max() / area.min() > 4
    np.testing.assert_array_equal(data, original)
    assert not list(tmp_path.iterdir())


def test_plot_example_has_four_annotated_views_and_no_automatic_files():
    document = _document()
    plots = {
        node["id"]: node["params"]
        for node in document["nodes"]
        if node["operation_id"] == "plot_results"
    }
    assert set(plots) == {
        "plot_area",
        "plot_area_intensity",
        "plot_elongation",
        "plot_circularity",
    }
    assert plots["plot_area"]["y_column"] == "area_physical"
    assert plots["plot_area_intensity"]["x_column"] == "area_physical"
    assert plots["plot_area_intensity"]["y_column"] == "intensity_mean"
    assert plots["plot_elongation"]["summary"] == "Median"
    assert plots["plot_circularity"]["distribution"] == "Cumulative"
    assert all(params["point_unit"] == "Objects" for params in plots.values())
    assert not any(
        node["operation_id"] in {"save_output", "batch_output", "table_source"}
        for node in document["nodes"]
    )
    assert len(document["notes"]) == 4
    text = " ".join(note["text"] for note in document["notes"])
    assert "not an independent biological sample" in text
    assert "deliberately" in text


def test_plot_example_executes_all_four_typed_plots(tmp_path, monkeypatch):
    document = _document()
    data, kwargs, _kind = _measurement_plot_sample()
    data.setflags(write=False)
    monkeypatch.chdir(tmp_path)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=2,
            workflow=document,
            input_data=data,
            input_metadata=kwargs["metadata"],
            input_name=kwargs["name"],
            source_payloads={},
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(node["id"] for node in document["nodes"]),
        ),
        raise_errors=True,
    )
    assert not result.error and not result.cancelled
    for node in document["nodes"]:
        if node["operation_id"] != "plot_results":
            continue
        plot = result.pipeline.outputs[node["id"]]
        state = result.pipeline.node_output_states[node["id"]][0]
        assert isinstance(plot, PlotData)
        assert isinstance(state, PlotState) and state.kind == "plot"
        assert plot.counts.input_rows == plot.counts.eligible_rows == 60
        assert plot.counts.excluded_rows == 0
        assert plot.counts.plotted_points == 60
        assert len(plot.series) == 1
        assert plot.plotted_table.row_count == 60
    histogram = result.pipeline.outputs["plot_area"]
    assert sum(histogram.series[0].histogram_values) == 60
    cumulative = result.pipeline.outputs["plot_circularity"]
    assert cumulative.series[0].ecdf_y[-1] == 100.0
    assert not list(tmp_path.iterdir())


def test_grouped_review_demo_has_real_collection_and_six_image_means(tmp_path):
    from napari_vipp.core.measurement_collection import load_measurement_collection

    generator = runpy.run_path(
        str(
            Path(__file__).resolve().parents[3]
            / "scripts"
            / "create_plot_review_demo.py"
        )
    )
    paths = generator["create_demo"](tmp_path / "new-demo")
    collection = load_measurement_collection(paths["collection"])
    assert collection.table.row_count == 258
    assert [item.row_count for item in collection.items] == [24, 40, 60, 30, 46, 58]
    assert all(item.included and item.result_sha256 for item in collection.items)
    assert collection.table.unit_for("area_physical") == "µm^2"
    document = json.loads(paths["workflow"].read_text(encoding="utf-8"))
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=3,
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
    assert not result.error
    for node_id in ("plot_area", "plot_area_intensity", "plot_circularity"):
        plot = result.pipeline.outputs[node_id]
        assert plot.counts.eligible_rows == plot.counts.plotted_points == 258
        assert [len(series.y) for series in plot.series] == [124, 134]
    means = result.pipeline.outputs["plot_elongation"]
    assert means.counts.eligible_rows == 258
    assert means.counts.plotted_points == 6
    assert [len(series.y) for series in means.series] == [3, 3]
    expected = {}
    for row in collection.table.records():
        expected.setdefault(row["image_id"], []).append(row["area_physical"])
    np.testing.assert_allclose(
        sorted(value for series in means.series for value in series.y),
        sorted(np.mean(values) for values in expected.values()),
    )
    with pytest.raises(FileExistsError):
        generator["create_demo"](tmp_path / "new-demo")
