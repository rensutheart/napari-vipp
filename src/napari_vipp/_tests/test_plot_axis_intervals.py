"""Explicit tick spacing remains presentation-only, bounded and exportable."""

import json
from dataclasses import replace

import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg

from napari_vipp.core.batch import (
    scientific_workflow_document,
    scientific_workflow_hash,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.plot_rendering import build_plot_figure, export_plot_result
from napari_vipp.core.result_plots import (
    PlotRecipe,
    build_plot_result,
    interval_ticks,
)
from napari_vipp.core.tables import TableData
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow
from napari_vipp.ui.result_plots import PlotRecipeControls, PlotResultsPanel


def _table(*, unit="µm²", kind="measurements"):
    return TableData(
        ("area", "length", "image_id"),
        ((10, 1.2, "one"), (20, 2.7, "one"), (45, 3.8, "two")),
        column_units=(("area", unit), ("length", "µm")),
        table_kind=kind,
    )


def _result(**params):
    return build_plot_result(
        _table(),
        **dict(plot_type="Scatter", x_column="length", y_column="area", **params),
    )


@pytest.mark.parametrize("axis", ["x", "y"])
@pytest.mark.parametrize(
    "value", [0, -1, "0", "-0.5", "nan", "inf", "", None, True, [], "bad"]
)
def test_invalid_interval_is_rejected_without_rounding_or_clamping(axis, value):
    with pytest.raises(ValueError, match="positive finite"):
        PlotRecipe(plot_type="Scatter", **{f"{axis}_tick_interval": value})


@pytest.mark.parametrize("value", ["Auto", "auto", " Auto "])
def test_old_and_automatic_recipes_keep_defaults(value):
    recipe = PlotRecipe.from_params({"y_column": "area", "y_tick_interval": value})
    assert recipe.x_tick_interval == "Auto"
    assert PlotRecipe.from_params(recipe.to_params()) == recipe


def test_workflow_roundtrip_preserves_intervals_and_old_defaults():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("plot_results")
    node.params.update(plot_type="Scatter", x_tick_interval="2e-1", y_tick_interval="5")
    restored = deserialize_workflow(serialize_workflow(pipeline))
    found = next(item for item in restored["nodes"] if item.id == node.id)
    assert found.params["x_tick_interval"] == "2e-1"
    assert found.params["y_tick_interval"] == "5"
    old_workflow = serialize_workflow(pipeline)
    old_node = next(item for item in old_workflow["nodes"] if item["id"] == node.id)
    old_node["params"].pop("x_tick_interval")
    old_node["params"].pop("y_tick_interval")
    restored = deserialize_workflow(old_workflow)
    found = next(item for item in restored["nodes"] if item.id == node.id)
    assert found.params["x_tick_interval"] == found.params["y_tick_interval"] == "Auto"


def test_auto_migration_retains_old_batch_hash_but_custom_interval_changes_it():
    pipeline = PrototypePipeline()
    node = pipeline.add_node("plot_results")
    current = serialize_workflow(pipeline)
    old = json.loads(json.dumps(current))
    old_node = next(item for item in old["nodes"] if item["id"] == node.id)
    old_node["params"].pop("x_tick_interval")
    old_node["params"].pop("y_tick_interval")
    assert scientific_workflow_document(current) == scientific_workflow_document(old)
    assert scientific_workflow_hash(current) == scientific_workflow_hash(old)
    pipeline.set_param(node.id, "y_tick_interval", "5")
    assert scientific_workflow_hash(
        serialize_workflow(pipeline)
    ) != scientific_workflow_hash(old)


@pytest.mark.parametrize("compact", [True, False])
def test_numeric_spacing_is_shared_by_labels_and_grid_without_changing_data(compact):
    automatic = _result()
    result = _result(x_tick_interval="0.5", y_tick_interval=10)
    assert result.series == automatic.series
    assert result.plotted_table == automatic.plotted_table
    assert result.source_table == automatic.source_table
    figure = build_plot_figure(result, compact=compact)
    canvas = FigureCanvasAgg(figure)
    for size in ((3.5, 3), (8, 6)):
        figure.set_size_inches(size)
        canvas.draw()
        axes = figure.axes[0]
        np.testing.assert_allclose(np.diff(axes.get_xticks()), 0.5)
        np.testing.assert_allclose(np.diff(axes.get_yticks()), 10)
        assert all(line.get_visible() for line in axes.get_xgridlines())
        assert all(line.get_visible() for line in axes.get_ygridlines())
        assert not len(axes.xaxis.get_minorticklocs())


def test_histogram_bins_are_independent_from_axis_intervals():
    result = build_plot_result(
        _table(),
        plot_type="Distribution",
        y_column="area",
        bins=4,
        x_tick_interval="5",
        y_tick_interval="1",
    )
    automatic = build_plot_result(
        _table(),
        plot_type="Distribution",
        y_column="area",
        bins=4,
    )
    assert result.series == automatic.series
    figure = build_plot_figure(result)
    FigureCanvasAgg(figure).draw()
    np.testing.assert_allclose(np.diff(figure.axes[0].get_xticks()), 5)
    np.testing.assert_allclose(np.diff(figure.axes[0].get_yticks()), 1)


@pytest.mark.parametrize("interval", [0.5, "1.1"])
def test_counts_reject_fractional_intervals(interval):
    with pytest.raises(ValueError, match="counts.*whole number"):
        build_plot_result(
            _table(),
            plot_type="Distribution",
            y_column="area",
            y_tick_interval=interval,
        )
    with pytest.raises(ValueError, match="counts.*whole number"):
        build_plot_result(
            _table(unit="count"), y_column="area", y_tick_interval=interval
        )


def test_fractional_aggregate_counts_and_percentages_allow_fractional_intervals():
    mean = build_plot_result(
        _table(unit="count"),
        y_column="area",
        point_unit="Mean per image",
        image_column="image_id",
        y_tick_interval="0.5",
    )
    assert mean.series[0].y == (15, 45)
    build_plot_result(
        _table(),
        y_column="area",
        plot_type="Distribution",
        normalization="Percent",
        y_tick_interval="0.75",
    )


def test_categorical_and_log_axes_require_auto():
    with pytest.raises(ValueError, match="categories"):
        PlotRecipe(x_tick_interval="2")
    for axis in ("x", "y"):
        with pytest.raises(ValueError, match="logarithmic axis"):
            PlotRecipe(
                plot_type="Scatter",
                **{f"log_{axis}": True, f"{axis}_tick_interval": "2"},
            )


@pytest.mark.parametrize("interval", ["1e-300", "0.0001"])
def test_tiny_interval_cannot_allocate_unbounded_ticks(interval):
    with pytest.raises(ValueError, match="more than 200 ticks"):
        _result(x_tick_interval=interval)


def test_large_offsets_and_intermediate_overflow_are_bounded():
    with pytest.raises(ValueError, match="more than 200 ticks"):
        interval_ticks(1e300, 1e300 + 1e285, 1e-300, axis="X")
    with pytest.raises(ValueError, match="plotting precision"):
        interval_ticks(1e16, 1e16 + 2, 0.1, axis="X")


def test_auto_margin_is_included_in_interval_budget():
    # Raw range has only 191 intervals; normal plot margins exceed the limit.
    table = TableData(("area",), ((0.0,), (19.0,)))
    with pytest.raises(ValueError, match="more than 200 ticks"):
        build_plot_result(table, y_tick_interval="0.1")


def test_no_grid_retains_custom_ticks():
    figure = build_plot_figure(_result(show_grid=False, x_tick_interval="1"))
    FigureCanvasAgg(figure).draw()
    np.testing.assert_allclose(np.diff(figure.axes[0].get_xticks()), 1)
    assert not any(line.get_visible() for line in figure.axes[0].get_xgridlines())


def test_export_records_and_uses_the_same_intervals(tmp_path):
    target = tmp_path / "intervals.svg"
    result = _result(x_tick_interval="0.5", y_tick_interval="10")
    export_plot_result(result, target, include_data=True)
    settings = json.loads((tmp_path / "intervals-plot-settings.json").read_text())
    assert settings["recipe"]["x_tick_interval"] == "0.5"
    assert settings["recipe"]["y_tick_interval"] == "10"
    assert "<!-- 1.5 -->" in target.read_text()


def test_shared_controls_keep_explicit_values_in_inspector_and_window(qtbot):
    result = _result()
    panel = PlotResultsPanel(result.source_table, result.recipe.to_params(), result)
    qtbot.addWidget(panel)
    window = panel.open_plot()
    emitted = []
    panel.params_changed.connect(emitted.append)
    interval = window.controls.controls["x_tick_interval"]
    interval.auto.setChecked(False)
    interval.edit.setText("0.25")
    interval.edit.editingFinished.emit()
    assert emitted[-1]["x_tick_interval"] == "0.25"
    assert panel.controls.controls["x_tick_interval"].value() == "0.25"
    assert "µm" in interval.hint.text()
    assert not window.export_button.isEnabled()
    panel.close_plot()


def test_log_switch_explicitly_selects_auto_and_retains_manual_text(qtbot):
    controls = PlotRecipeControls()
    qtbot.addWidget(controls)
    recipe = PlotRecipe(plot_type="Scatter", x_tick_interval="0.25")
    controls.set_state(_table(), recipe.to_params())
    emitted = []
    controls.params_changed.connect(emitted.append)
    interval = controls.controls["x_tick_interval"]
    controls.controls["log_x"].setChecked(True)
    assert emitted[-1]["x_tick_interval"] == "Auto"
    assert not interval.edit.isEnabled()
    assert "Log axis" in interval.hint.text()
    controls.controls["log_x"].setChecked(False)
    interval.auto.setChecked(False)
    assert emitted[-1]["x_tick_interval"] == "0.25"


def test_imported_incompatible_interval_is_not_silently_changed(qtbot):
    controls = PlotRecipeControls()
    qtbot.addWidget(controls)
    emitted = []
    controls.params_changed.connect(emitted.append)
    controls.set_state(_table(), {"x_tick_interval": "2"})
    interval = controls.controls["x_tick_interval"]
    assert interval.value() == "2"
    assert not interval.edit.isEnabled()
    assert interval.auto.isEnabled()  # Explicit repair remains available.
    assert emitted == []
    interval.auto.setChecked(True)
    assert emitted[-1]["x_tick_interval"] == "Auto"


@pytest.mark.parametrize(
    "kind", ["Descriptive Statistics v2", "Grouped measurement summary"]
)
def test_summary_table_explains_rows_and_disallows_image_reaggregation(qtbot, kind):
    table = _table(kind=kind)
    result = build_plot_result(table, y_column="area")
    panel = PlotResultsPanel(table, result.recipe.to_params(), result)
    qtbot.addWidget(panel)
    unit = panel.controls.controls["point_unit"]
    assert unit.currentText() == "One summary row"
    assert not unit.model().item(1).isEnabled()
    assert "3 summary rows" in panel.summary.text()
    assert "error bar" in panel.controls.note.text()
    assert result.series[0].name == "All summary rows"
    with pytest.raises(ValueError, match="already a summary table"):
        build_plot_result(
            table, y_column="area", point_unit="Mean per image", image_column="image_id"
        )
    panel.set_state(
        table=table,
        params={
            **result.recipe.to_params(),
            "point_unit": "Mean per image",
            "image_column": "image_id",
        },
    )
    assert unit.currentData() == "Mean per image"
    assert "Choose One summary row" in panel.controls.note.text()
    panel.set_state(
        table=replace(table, table_kind="measurements"),
        params=result.recipe.to_params(),
    )
    assert unit.currentText() == "Objects"
    assert unit.model().item(1).isEnabled()


@pytest.mark.parametrize(
    "kind", ["Descriptive Statistics v2", "Grouped measurement summary"]
)
def test_summary_plot_requires_explicit_measurements_not_numeric_record_fields(
    qtbot, kind
):
    table = TableData(
        ("summary_version", "row_count", "area_mean", "length_median"),
        ((2, 40, 20.5, 1.2), (2, 38, 29.4, 2.3)),
        table_kind=kind,
    )
    for column in ("", "auto"):
        with pytest.raises(
            ValueError, match="Choose a summary measurement for the Y axis"
        ):
            build_plot_result(table, y_column=column)
        with pytest.raises(
            ValueError, match="Choose a summary measurement for the X axis"
        ):
            build_plot_result(
                table, y_column="area_mean", plot_type="Scatter", x_column=column
            )
    result = build_plot_result(
        table, y_column="area_mean", x_column="length_median", plot_type="Scatter"
    )
    assert result.series[0].y == (20.5, 29.4)
    assert result.series[0].x == (1.2, 2.3)
    controls = PlotRecipeControls()
    qtbot.addWidget(controls)
    controls.set_state(table, {})
    assert controls.controls["y_column"].currentText() == "Choose a summary measurement"
    assert controls.controls["y_column"].currentData() == "auto"
