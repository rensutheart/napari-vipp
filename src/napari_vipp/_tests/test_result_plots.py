"""Known answers for descriptive plots and their typed execution boundary."""

import hashlib
import json
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from napari_vipp.core.compute import ComputeMode, ComputeRequest
from napari_vipp.core.execution import (
    PipelineRunRequest,
    _scientific_array_identity,
    execute_pipeline_request,
)
from napari_vipp.core.metadata import (
    format_compact_metadata,
    format_detailed_metadata,
    metadata_history_items,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.preview import make_preview
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.result_plots import (
    DISPLAY_POINT_LIMIT,
    PlotData,
    PlotRecipe,
    PlotState,
    build_plot_result,
    measurement_label,
    numeric_columns,
    plot_results,
    plot_state_from_data,
)
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


@pytest.fixture
def table():
    return TableData(
        ("label_id", "area", "intensity", "condition", "image_id"),
        (
            (1, 2.0, 10.0, "A", "a"),
            (2, 4.0, 20.0, "A", "a"),
            (1, 12.0, 30.0, "A", "b"),
            (1, 8.0, 40.0, "B", "c"),
        ),
        name="Objects",
        source_name="Example",
        column_units=(("area", "micrometer^2"), ("intensity", "a.u.")),
    )


def test_default_is_numeric_not_label_and_preserves_units(table):
    assert numeric_columns(table) == ("area", "intensity")
    data = build_plot_result(table)
    assert data.recipe.y_column == "area"
    assert data.series[0].y == (2.0, 4.0, 12.0, 8.0)
    assert data.y_label == "Area (micrometer^2)"
    assert data.counts.plotted_points == 4
    assert data.nbytes > 0


def test_group_means_full_data_not_biological_replicates(table):
    data = build_plot_result(table, y_column="area", group_column="condition")
    assert data.series[0].mean == 6
    assert data.series[0].median == 4
    assert data.series[1].mean == 8
    assert "independent biological" in data.warnings[-1]


def test_image_means_equal_weight_and_repeat_label_ids_remain_distinct(table):
    data = build_plot_result(
        table,
        y_column="area",
        group_column="condition",
        image_column="image_id",
        point_unit="Mean per image",
    )
    assert data.series[0].y == (3, 12)
    assert data.series[0].mean == 7.5  # not pooled object mean 6
    assert data.series[0].source_rows == ((0, 1), (2,))
    assert data.counts.eligible_rows == 4
    assert data.counts.plotted_points == 3
    rows = data.plotted_table.records()
    assert rows[0]["source:label_id"] == (1, 2)
    assert rows[1]["source:label_id"] == 1
    assert rows[0]["source:image_id"] == "a"
    assert rows[1]["source:image_id"] == "b"


def test_histogram_edges_shared_and_exact_percent(table):
    data = build_plot_result(
        table,
        plot_type="Distribution",
        y_column="area",
        group_column="condition",
        bins=2,
        normalization="Percent",
    )
    first, second = data.series
    assert first.bin_edges == second.bin_edges == (2, 7, 12)
    assert first.histogram_values == pytest.approx((200 / 3, 100 / 3))
    assert second.histogram_values == (0, 100)
    assert data.x_label == "Area (micrometer^2)"
    assert data.y_label == "Within-group percent (%)"


def test_ecdf_ties_have_exact_mass_and_endpoint():
    table = TableData(("area",), ((1,), (1,), (3,), (5,)))
    data = build_plot_result(table, plot_type="Distribution", distribution="Cumulative")
    assert data.series[0].ecdf_x == (1, 3, 5)
    assert data.series[0].ecdf_y == (50, 75, 100)


def test_scatter_pairs_and_exclusion_reasons():
    table = TableData(
        ("area", "intensity"),
        (
            (1, 2),
            (None, 4),
            (5, np.nan),
            (np.inf, 4),
            (3, "bad"),
            (-2, 6),
            (4, 0),
            (None, np.inf),
        ),
    )
    data = build_plot_result(
        table,
        plot_type="Scatter",
        x_column="area",
        y_column="intensity",
        log_x=True,
        log_y=True,
    )
    assert data.counts.input_rows == 8
    assert data.counts.missing_rows == 2
    assert data.counts.nonfinite_rows == 2
    assert data.counts.nonnumeric_rows == 1
    assert data.counts.nonpositive_rows == 2
    assert data.counts.eligible_rows == 1
    assert data.series[0].x == (1,)
    assert data.series[0].y == (2,)
    assert data.series[0].source_rows == ((0,),)


@pytest.mark.parametrize("value", [0.0, 1.0, -7.0])
def test_constant_histogram_retains_every_value(value):
    data = build_plot_result(
        TableData(("area",), ((value,),) * 8), plot_type="Distribution", bins=20
    )
    assert sum(data.series[0].histogram_values) == 8
    assert data.series[0].bin_edges[0] < value < data.series[0].bin_edges[-1]


def test_log_distribution_excludes_only_measured_nonpositive():
    data = build_plot_result(
        TableData(("area",), ((-1,), (0,), (1,), (10,), (100,))),
        plot_type="Distribution",
        log_x=True,
        bins=2,
    )
    assert data.counts.nonpositive_rows == 2
    assert data.series[0].bin_edges == (1, 50.5, 100)
    assert sum(data.series[0].histogram_values) == 3


@pytest.mark.parametrize("rows", [(), ((None,),), ((np.nan,),), ((np.inf,),)])
def test_empty_and_all_missing_are_valid_empty_plot(rows):
    data = build_plot_result(TableData(("area",), rows), y_column="area")
    assert data.counts.plotted_points == 0
    assert data.series == ()
    assert "No eligible values" in data.warnings[-1]


def test_no_numeric_auto_does_not_pick_numeric_strings():
    table = TableData(("label_id", "sample"), ((1, "123"),))
    with pytest.raises(ValueError, match="numeric measurement"):
        build_plot_result(table)


def test_immutable_detached_table_and_exact_identity(table):
    rows = [list(row) for row in table.rows]
    mutable = replace(table, rows=rows)
    data = build_plot_result(mutable)
    rows[0][1] = 999
    assert data.source_table.rows[0][1] == 2
    with pytest.raises(FrozenInstanceError):
        data.recipe.title = "changed"
    with pytest.raises(TypeError):
        data.series[0].y[0] = 9
    baseline = _scientific_array_identity(data, cancel_callback=None)
    assert baseline != _scientific_array_identity(
        build_plot_result(table, title="Other"), cancel_callback=None
    )


def test_large_identity_integer_preserved_but_not_coerced_to_measurement(table):
    table = replace(table, rows=((2**63 + 3, 2.0, 3.0, "a", "image"),))
    data = build_plot_result(table)
    assert data.plotted_table.records()[0]["source:label_id"] == 2**63 + 3
    with pytest.raises(ValueError, match="precision"):
        build_plot_result(table, y_column="label_id")


def test_display_sampling_does_not_sample_summaries_or_export_rows():
    n = DISPLAY_POINT_LIMIT + 123
    table = TableData(("area",), tuple((i,) for i in range(n)))
    data = build_plot_result(table)
    assert len(data.series[0].display_indices) == DISPLAY_POINT_LIMIT
    assert len(data.series[0].y) == n == data.plotted_table.row_count
    assert data.series[0].mean == (n - 1) / 2
    assert (
        data.series[0].display_indices
        == build_plot_result(table).series[0].display_indices
    )
    assert "deterministic sample" in " ".join(data.warnings)


def test_cancellation_before_and_during_preparation(table):
    with pytest.raises(OperationCancelled):
        plot_results(table, progress=ProgressContext(cancelled=lambda: True))
    calls = 0

    def cancel():
        nonlocal calls
        calls += 1
        return calls > 3

    large = TableData(("area",), ((1,),) * 5000)
    with pytest.raises(OperationCancelled):
        build_plot_result(large, cancel_callback=cancel)


def test_recipe_json_roundtrip_and_validation():
    recipe = PlotRecipe(y_column="area", bins=10, title="Objects")
    assert PlotRecipe.from_params(json.loads(json.dumps(recipe.to_params()))) == recipe
    for kwargs in (
        {"recipe_version": 2},
        {"bins": 3.5},
        {"point_unit": "Mean per image"},
        {"log_x": "true"},
        {"point_size": np.nan},
    ):
        with pytest.raises(ValueError):
            PlotRecipe(**kwargs)


def test_conflicting_image_groups_are_not_silently_split(table):
    table = replace(table, rows=(table.rows[0], (*table.rows[1][:3], "B", "a")))
    with pytest.raises(ValueError, match="one mean per image") as error:
        build_plot_result(
            table,
            group_column="condition",
            image_column="image_id",
            point_unit="Mean per image",
        )
    assert "Group by is set to 'Condition'" in str(error.value)
    assert "image 'a'" in str(error.value)


def test_varying_object_measurement_explains_image_mean_conflict_and_recovery():
    table = TableData(
        ("major_axis_length_pixels", "minor_axis_length_pixels", "image_id"),
        (
            (42.0, 19.0, "synthetic field 01"),
            (36.0, 20.0, "synthetic field 01"),
            (21.0, 19.0, "synthetic field 01"),
        ),
        column_units=(
            ("major_axis_length_pixels", "pixels"),
            ("minor_axis_length_pixels", "pixels"),
        ),
    )
    original_rows = table.rows
    recipe = PlotRecipe(
        y_column="major_axis_length_pixels",
        group_column="minor_axis_length_pixels",
        image_column="image_id",
        point_unit="Mean per image",
    )
    with pytest.raises(ValueError) as error:
        build_plot_result(table, recipe=recipe)
    title, explanation, recovery, identity_hint = str(error.value).split("\n\n")
    assert title == "Cannot calculate one mean per image with this grouping."
    assert "Group by is set to 'Minor axis length (pixels)'" in explanation
    assert "image 'synthetic field 01'" in explanation
    assert "Each image must belong to one group" in explanation
    assert "Each point represents to Objects" in recovery
    assert "Group by to None" in recovery
    assert "treatment, with one value per image" in recovery
    assert "same ID refers to different images" in identity_hint
    assert "Image identity column" in identity_hint

    objects = build_plot_result(table, recipe=replace(recipe, point_unit="Objects"))
    assert objects.counts.plotted_points == 3
    assert tuple(series.name for series in objects.series) == ("19.0", "20.0")
    assert objects.series[0].y == (42.0, 21.0)
    assert objects.series[0].source_rows == ((0,), (2,))
    assert objects.series[1].y == (36.0,)
    image_mean = build_plot_result(table, recipe=replace(recipe, group_column=""))
    assert image_mean.counts.plotted_points == 1
    assert image_mean.series[0].y == (33.0,)
    assert image_mean.series[0].source_rows == ((0, 1, 2),)
    assert table.rows is original_rows
    assert recipe.point_unit == "Mean per image"
    assert recipe.group_column == "minor_axis_length_pixels"


def test_plot_state_never_becomes_an_image(table):
    result = build_plot_result(table)
    state = plot_state_from_data(result, history=("Plot Results",))
    assert state.kind == "plot"
    assert format_compact_metadata(state).startswith("PLOT:")
    assert "Points: 4" in format_detailed_metadata(result)
    assert metadata_history_items(state) == ["Plot Results"]
    assert make_preview(result) is None


@pytest.fixture
def plot_pipeline(tmp_path, table):
    from napari_vipp.core.measurement_collection import (
        CollectionItem,
        MeasurementOutput,
        MeasurementPreview,
        collect_measurements,
        save_measurement_collection,
    )

    preview = MeasurementPreview(
        MeasurementOutput("out", "Objects", "objects"),
        (
            CollectionItem(
                "sample-a",
                1,
                "image-a",
                "ready",
                row_count=4,
                result_sha256="c" * 64,
                table=table,
            ),
        ),
        "run",
        "a" * 64,
        "b" * 64,
    )
    path = tmp_path / "objects.vipp-results.json"
    save_measurement_collection(collect_measurements(preview), path)
    pipeline = PrototypePipeline()
    pipeline.restore_graph([], [])
    source = pipeline.add_node("table_source")
    pipeline.set_param(source.id, "dataset_path", str(path))
    pipeline.set_param(
        source.id, "dataset_sha256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    plot = pipeline.add_node("plot_results")
    pipeline.set_param(plot.id, "y_column", "area")
    assert pipeline.connect(source.id, plot.id).success
    return pipeline, source.id, plot.id


@pytest.mark.parametrize("mode", [ComputeMode.CPU, ComputeMode.AUTO])
def test_typed_headless_table_source_pipeline(plot_pipeline, mode):
    pipeline, source, plot = plot_pipeline
    restored = deserialize_workflow(serialize_workflow(pipeline))
    assert restored["nodes"][-1].params["y_column"] == "area"
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={},
            compute_request=ComputeRequest(mode=mode),
        ),
        raise_errors=True,
    )
    assert not result.error
    assert isinstance(result.pipeline.outputs[plot], PlotData)
    assert isinstance(result.pipeline.output_states[plot], PlotState)
    assert result.pipeline.outputs[plot].counts.plotted_points == 4
    assert plot in result.pipeline.node_compute_provenance
    assert source in result.pipeline.node_compute_provenance


def test_pipeline_edit_invalidates_only_plot_and_bypass_forbidden(plot_pipeline):
    pipeline, source, plot = plot_pipeline
    pipeline.run(None)
    original = pipeline.outputs[source]
    pipeline.set_param(plot, "plot_type", "Distribution")
    pipeline.set_param(plot, "bins", 4)
    pipeline.run(None, dirty_node_ids={plot})
    assert pipeline.outputs[source] is original
    assert pipeline.outputs[plot].recipe.bins == 4
    assert not pipeline.operation_spec("plot_results").supports_bypass
    image = pipeline.add_node("gaussian_blur")
    assert not pipeline.connect(plot, image.id).success


def test_axis_interval_edit_reuses_the_upstream_measurements(plot_pipeline):
    pipeline, source, plot = plot_pipeline
    pipeline.run(None)
    original = pipeline.outputs[source]
    original_series = pipeline.outputs[plot].series
    pipeline.set_param(plot, "y_tick_interval", "2")
    pipeline.run(None, dirty_node_ids={plot})
    assert pipeline.outputs[source] is original
    assert pipeline.outputs[plot].recipe.y_tick_interval == "2"
    assert pipeline.outputs[plot].series == original_series


@pytest.fixture
def image_plot_pipeline():
    pipeline = PrototypePipeline()
    pipeline.restore_graph([], [])
    nodes = [
        pipeline.add_node(operation)
        for operation in (
            "input",
            "binary_threshold",
            "label_connected_components",
            "measure_objects",
            "plot_results",
            "batch_output",
        )
    ]
    for first, second in zip(nodes, nodes[1:], strict=False):
        assert pipeline.connect(first.id, second.id).success
    image = np.zeros((24, 24), dtype=np.uint16)
    image[2:5, 2:5] = image[8:13, 8:13] = image[15:21, 15:21] = 1000
    return pipeline, nodes[-2].id, nodes[-1].id, image


@pytest.mark.parametrize("file_format", ["png", "tiff", "svg", "pdf"])
def test_batch_stages_plot_and_never_publishes_before_commit(
    image_plot_pipeline,
    tmp_path,
    file_format,
):
    from napari_vipp.core.batch import (
        BatchOutputPlan,
        ExistingFilePolicy,
        _save_planned_output,
    )
    from napari_vipp.core.batch_setup import _batch_output_config

    pipeline, plot, output, image = image_plot_pipeline
    pipeline.set_param(output, "format", file_format)
    pipeline.run(image)
    config = _batch_output_config(pipeline, output)
    assert config.kind == "plot"
    path = tmp_path / (
        "result.tif" if file_format == "tiff" else f"result.{file_format}"
    )
    plan = BatchOutputPlan(
        output, "Plot", "plot", "plot", file_format, path, ExistingFilePolicy.ERROR
    )
    staged = _save_planned_output(pipeline, plan)
    assert not path.exists()
    assert staged.saved_temporary_path.stat().st_size > 1000
    assert isinstance(pipeline.outputs[plot], PlotData)


def test_generated_python_executes_plot_and_writes_vector_output(
    image_plot_pipeline,
    tmp_path,
):
    from napari_vipp.core.export import export_pipeline_to_python

    pipeline, plot, output, image = image_plot_pipeline
    pipeline.set_param(plot, "y_tick_interval", "5")
    pipeline.set_param(output, "format", "svg")
    code = export_pipeline_to_python(pipeline)
    namespace = {"__name__": "exported_plot_test"}
    exec(compile(code, "<exported>", "exec"), namespace)
    results = namespace["run_pipeline"](image)
    assert isinstance(results[plot], PlotData)
    assert results[plot].recipe.y_tick_interval == "5"
    assert results[output].counts.plotted_points == 3
    path = namespace["_automatic_output_path"](
        tmp_path, "image", output, results[output]
    )
    assert path.suffix == ".svg"
    saved = namespace["_write_output_uncommitted"](
        results[output],
        path,
        output_node_id=output,
    )
    assert saved.read_text(encoding="utf-8").lstrip().startswith("<?xml")


def test_selected_finite_extremes_do_not_overflow_summary():
    data = build_plot_result(TableData(("area",), ((1e308,), (1e308,))))
    assert data.series[0].mean == data.series[0].median == 1e308


def test_log_image_means_do_not_drop_negative_contributors():
    table = TableData(("area", "image_id"), ((-2, "a"), (4, "a"), (-4, "b"), (2, "b")))
    data = build_plot_result(
        table,
        y_column="area",
        point_unit="Mean per image",
        image_column="image_id",
        log_y=True,
    )
    assert data.series[0].y == (1,)
    assert data.series[0].source_rows == ((0, 1),)
    assert data.counts.nonpositive_points == 1  # mean of image b, not its two rows
    assert data.counts.nonpositive_rows == 2
    assert data.counts.eligible_rows == 2
    assert data.counts.finite_rows == 4


@pytest.mark.parametrize(
    ("column", "unit", "expected"),
    [
        ("major_axis_length_pixels", "pixels", "Major axis length (pixels)"),
        ("minor axis length (pixels)", "pixels", "Minor axis length (pixels)"),
        ("area_micrometer^2", "micrometer^2", "Area (micrometer^2)"),
        ("DNA_intensity", "a.u.", "DNA intensity (a.u.)"),
        ("major_axis_length_pixels", "", "Major axis length pixels"),
        ("major_axis_length_pixels", "µm", "Major axis length pixels (µm)"),
        ("distance_Mm", "mm", "Distance Mm (mm)"),
        ("pixels", "pixels", "Pixels (pixels)"),
    ],
)
def test_measurement_labels_are_display_only_and_only_strip_matching_unit(
    column, unit, expected
):
    table = TableData((column,), ((1.23456789012345,),), column_units=((column, unit),))
    assert measurement_label(table, column) == expected
    result = build_plot_result(table, y_column=column)
    assert result.y_label == expected
    assert result.recipe.y_column == column
    assert result.source_table == table
    assert result.series[0].y == (1.23456789012345,)


def test_many_numeric_groups_explain_scatter_without_changing_categories():
    values = tuple(19.0 + i * 0.000000000001 for i in range(60))
    table = TableData(
        ("major_axis_length_pixels", "minor_axis_length_pixels"),
        tuple((float(i), value) for i, value in enumerate(values)),
        column_units=(
            ("major_axis_length_pixels", "pixels"),
            ("minor_axis_length_pixels", "pixels"),
        ),
    )
    result = build_plot_result(
        table,
        y_column="major_axis_length_pixels",
        group_column="minor_axis_length_pixels",
    )
    assert result.recipe.plot_type == "Compare groups"
    assert result.x_label == "Minor axis length (pixels)"
    assert result.y_label == "Major axis length (pixels)"
    assert len(result.series) == 60
    assert tuple(series.name for series in result.series) == tuple(map(str, values))
    assert (
        tuple(
            row["source:minor_axis_length_pixels"]
            for row in result.plotted_table.records()
        )
        == values
    )
    warning = result.warnings[0]
    assert "60 numeric groups" in warning
    assert "equal spacing" in warning
    assert "Scatter" in warning
    assert "X measurement to Minor axis length (pixels)" in warning
    assert "Group by to None" in warning
    scatter = build_plot_result(
        table,
        plot_type="Scatter",
        x_column="minor_axis_length_pixels",
        y_column="major_axis_length_pixels",
    )
    assert scatter.series[0].x == values
    assert not any("numeric groups" in warning for warning in scatter.warnings)


@pytest.mark.parametrize(
    "groups",
    [tuple(range(12)), (1, 2, 3) * 20, tuple(str(i) for i in range(60)), (True, False)],
)
def test_small_or_textual_groups_do_not_suggest_numeric_scatter(groups):
    table = TableData(("area", "group"), tuple((1.0, group) for group in groups))
    result = build_plot_result(table, y_column="area", group_column="group")
    assert not any("numeric groups" in warning for warning in result.warnings)
