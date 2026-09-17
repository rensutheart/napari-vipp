"""Known-answer descriptive statistics, experimental-unit, and safety contracts."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from napari_vipp.core.operations import summarize_measurements
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.statistics import (
    StatisticsRecipe,
    measurement_columns,
    summarize_statistics,
)
from napari_vipp.core.tables import TableData, table_from_columns


@pytest.fixture
def unequal_table():
    return table_from_columns(
        {
            "condition": ["A"] * 5,
            "image_id": ["a", "a", "a", "b", "c"],
            "sample_id": ["one", "one", "one", "one", "two"],
            "area": [0, 0, 0, 12, 24],
        },
        source_name="Analytical phantom",
        column_units={"area": "µm²"},
    )


@pytest.mark.parametrize(
    ("level", "weighting", "expected", "n", "std"),
    [
        ("Objects", "Equal images", 7.2, 5, math.sqrt(115.2)),
        ("Image averages", "Equal images", 12, 3, 12),
        ("Sample averages", "Equal images", 15, 2, math.sqrt(162)),
        ("Sample averages", "Equal objects", 13.5, 2, math.sqrt(220.5)),
    ],
)
def test_unequal_object_image_sample_counts(
    unequal_table, level, weighting, expected, n, std
):
    output = summarize_statistics(
        unequal_table,
        summary_level=level,
        sample_weighting=weighting,
        image_column="image_id",
        sample_column="sample_id",
        group_by="condition",
        value_columns="area",
    )
    row = output.records()[0]
    assert row["area_mean"] == pytest.approx(expected)
    assert row["area_std"] == pytest.approx(std)
    assert row["area_n"] == row["area_count"] == n
    assert row["area_object_total"] == row["area_object_valid"] == 5
    assert row["area_image_total"] == row["area_image_valid"] == 3
    assert row["area_sample_total"] == row["area_sample_valid"] == 2
    assert row["area_object_excluded"] == 0
    assert row["image_count"] == 3
    assert row["sample_count"] == 2
    assert row["area_status"] == "ok"
    assert output.unit_for("area_mean") == "µm²"
    assert output.unit_for("area_count") == ""
    assert output.unit_for("area_object_total") == ""
    assert output.source_name == "Analytical phantom"
    recipe = json.loads(row["statistics_recipe"])
    assert recipe["summary_version"] == 2
    assert recipe["resolved_group_columns"] == ["condition"]
    assert recipe["resolved_value_columns"] == ["area"]
    assert recipe["quantile_method"] == "linear"
    assert recipe["std_ddof"] == 1


def test_all_statistics_known_answer_and_units():
    table = table_from_columns({"x": [1, 2, 3, 4]}, column_units={"x": "nm"})
    output = summarize_statistics(
        table, statistics="count,mean,median,std,q25,q75,iqr,min,max,sum"
    )
    row = output.records()[0]
    expected = dict(
        count=4,
        mean=2.5,
        median=2.5,
        std=math.sqrt(5 / 3),
        q25=1.75,
        q75=3.25,
        iqr=1.5,
        min=1,
        max=4,
        sum=10,
    )
    for stat, value in expected.items():
        assert row[f"x_{stat}"] == pytest.approx(value)
        assert output.unit_for(f"x_{stat}") == ("" if stat == "count" else "nm")


def test_missing_values_are_per_measurement_and_all_invalid_groups_survive():
    table = TableData(
        ("group", "image", "sample", "x", "y"),
        (
            ("A", "a", "s1", 1, None),
            ("A", "a", "s1", None, 2),
            ("A", "b", "s1", float("nan"), 4),
            ("A", "b", "s1", float("inf"), 6),
            ("A", "b", "s1", "2", 8),
            ("A", "b", "s1", True, 10),
            ("A", "b", "s1", " ", 12),
            ("B", "c", "s2", None, None),
        ),
    )
    result = summarize_statistics(
        table,
        group_by="group",
        value_columns="x,y",
        summary_level="Image averages",
        image_column="image",
        sample_column="sample",
        statistics="mean,std,sum,count",
    ).records()
    a, b = result
    assert a["x_object_total"] == 7
    assert a["x_object_valid"] == 1
    assert a["x_object_excluded"] == 6
    assert a["x_missing_count"] == 2
    assert a["x_nonfinite_count"] == 2
    assert a["x_nonnumeric_count"] == 2
    assert a["x_n"] == a["x_image_valid"] == 1
    assert a["x_image_total"] == 2
    assert a["x_std"] is None
    assert "undefined" in a["x_status"]
    assert a["y_n"] == a["y_image_valid"] == 2
    assert a["y_mean"] == 5  # image means 2 and 8, not the pooled-object mean 7.
    assert b["x_object_total"] == b["x_object_excluded"] == 1
    assert b["x_object_valid"] == b["x_n"] == b["x_count"] == 0
    assert b["x_mean"] is b["x_std"] is b["x_sum"] is None
    assert "No contributing observations" in b["x_status"]


def test_sample_means_ignore_images_without_valid_objects_per_measurement():
    table = table_from_columns(
        {"image": ["a", "b", "c"], "sample": ["one", "one", "two"], "x": [6, None, 24]}
    )
    row = summarize_statistics(
        table,
        summary_level="Sample averages",
        image_column="image",
        sample_column="sample",
    ).records()[0]
    assert row["x_mean"] == 15
    assert row["x_image_total"] == 3
    assert row["x_image_valid"] == 2
    assert row["x_sample_valid"] == row["x_n"] == 2


@pytest.mark.parametrize("value", [None, "", "bad", True, float("nan"), float("inf")])
def test_stop_and_review_reports_each_invalid_kind(value):
    with pytest.raises(ValueError, match="Stop and review"):
        summarize_statistics(
            TableData(("x",), ((value,),)),
            value_columns="x",
            missing_policy="Stop and review",
        )


def test_singleton_std_undefined_and_other_statistics_defined():
    row = summarize_statistics(
        TableData(("x",), ((8,),)), statistics="mean,std,iqr,sum"
    ).records()[0]
    assert row["x_mean"] == row["x_sum"] == 8
    assert row["x_iqr"] == 0
    assert row["x_std"] is None
    assert row["x_n"] == 1


def test_empty_table_explicit_measurement_gives_ungrouped_n_zero():
    output = summarize_statistics(
        TableData(("x",), ()), value_columns="x", statistics="count,mean,std,sum"
    )
    row = output.records()[0]
    assert (
        row["row_count"] == row["x_n"] == row["x_object_total"] == row["x_count"] == 0
    )
    assert row["x_mean"] is row["x_std"] is row["x_sum"] is None
    grouped = summarize_statistics(
        TableData(("group", "x"), ()), group_by="group", value_columns="x"
    )
    assert grouped.rows == ()
    assert "x_n" in grouped.columns


def test_explicit_non_numeric_column_reported_not_discarded():
    row = summarize_statistics(
        TableData(("x",), (("text",),)), value_columns="x"
    ).records()[0]
    assert row["x_nonnumeric_count"] == row["x_object_excluded"] == 1
    assert row["x_n"] == 0
    with pytest.raises(ValueError, match="could not find"):
        summarize_statistics(TableData(("x",), ((1,),)), value_columns="missing")


def test_auto_ignores_metadata_identifiers_strings_bools_but_keeps_all_nonfinite():
    names = (
        "label",
        "label_id",
        "object_id",
        "group",
        "replicate",
        "image",
        "sample",
        "vipp_row",
        "_vipp_row",
        "t_index",
        "source_name",
        "x",
        "nan_only",
        "text",
        "bool",
    )
    table = TableData(names, ((1,) * 11 + (2, float("nan"), "42", True),))
    assert measurement_columns(table) == ("x", "nan_only")
    assert "nan_only_n" in summarize_statistics(table).columns


def test_auto_no_measurements_is_actionable_and_empty_selection_not_auto():
    table = TableData(("x",), (("12",),))
    with pytest.raises(ValueError, match="choose value columns explicitly"):
        summarize_statistics(table)
    with pytest.raises(ValueError, match="No numeric measurement columns selected"):
        summarize_statistics(TableData(("x",), ((1,),)), value_columns="")


def test_explicit_no_groups_and_auto_are_distinct():
    table = table_from_columns(
        {"condition": ["A", "B"], "sample_id": ["s1", "s2"], "x": [1, 3]}
    )
    assert summarize_statistics(table, group_by="").row_count == 1
    assert summarize_statistics(table, group_by="auto").row_count == 2
    assert "sample_id" not in summarize_statistics(table, group_by="auto").columns
    with pytest.raises(ValueError, match="Choose a sample identity"):
        summarize_statistics(table, summary_level="Sample averages")


@pytest.mark.parametrize("column", ["group", "image", "sample"])
@pytest.mark.parametrize("missing", [None, "", float("nan"), float("inf")])
def test_missing_identity_or_group_rejected_even_on_invalid_measurement(
    column, missing
):
    row = {"group": "A", "image": "i", "sample": "s", "x": None}
    row[column] = missing
    table = TableData(tuple(row), (tuple(row.values()),))
    with pytest.raises(ValueError, match="identity/group|identity"):
        summarize_statistics(
            table,
            group_by="group",
            image_column="image",
            sample_column="sample",
            value_columns="x",
        )


@pytest.mark.parametrize("column", ["image", "sample"])
def test_boolean_id_rejected(column):
    with pytest.raises(ValueError, match="booleans"):
        summarize_statistics(
            TableData((column, "x"), ((True, 1),)), **{f"{column}_column": column}
        )


@pytest.mark.parametrize("kind", ["image", "sample"])
def test_ids_cannot_straddle_groups(kind):
    table = TableData(("group", kind, "x"), (("A", "same", 1), ("B", "same", 2)))
    with pytest.raises(ValueError, match="spans multiple groups"):
        summarize_statistics(
            table,
            group_by="group",
            summary_level="Image averages" if kind == "image" else "Sample averages",
            sample_weighting="Equal objects",
            **{f"{kind}_column": kind},
        )


def test_images_cannot_straddle_samples():
    table = TableData(("image", "sample", "x"), (("a", "one", 1), ("a", "two", 2)))
    with pytest.raises(ValueError, match="spans multiple samples"):
        summarize_statistics(table, image_column="image", sample_column="sample")


def test_huge_and_typed_identities_are_exact():
    table = TableData(
        ("image", "x"), ((2**60 + 1, 1), (2**60 + 2, 3), (1, 5), (1.0, 7), ("1", 9))
    )
    row = summarize_statistics(
        table, summary_level="Image averages", image_column="image"
    ).records()[0]
    assert row["x_n"] == row["image_count"] == 5
    assert row["x_mean"] == 5


def test_float64_inexact_integer_measurement_rejected_but_exact_large_integer_allowed():
    with pytest.raises(ValueError, match="not exactly representable"):
        summarize_statistics(TableData(("x",), ((2**53 + 1,),)))
    assert (
        summarize_statistics(TableData(("x",), ((2**60,),))).records()[0]["x_mean"]
        == 2**60
    )


def test_safe_extreme_mean_and_overflow_is_error_not_inf():
    huge = np.finfo(np.float64).max
    table = TableData(("x",), ((huge,), (huge,)))
    assert (
        summarize_statistics(table, statistics="mean,std").records()[0]["x_mean"]
        == huge
    )
    with pytest.raises(ValueError, match="exceeds float64 range"):
        summarize_statistics(table, statistics="sum")
    with pytest.raises(ValueError, match="exceeds float64 range"):
        summarize_statistics(TableData(("x",), ((huge,), (-huge,))), statistics="std")


def test_mean_std_preserve_representable_small_spread_at_large_offset():
    table = TableData(("x",), ((1e16,), (1e16 + 2,), (1e16 + 4,)))
    row = summarize_statistics(table, statistics="mean,std").records()[0]
    assert row["x_mean"] == 1e16 + 2
    assert row["x_std"] == 2
    pair = summarize_statistics(
        TableData(("x",), ((1e16,), (1e16 + 2,))), statistics="std"
    ).records()[0]
    assert pair["x_std"] == pytest.approx(math.sqrt(2))


def test_sum_handles_intermediate_overflow_when_final_result_is_representable():
    huge = float(np.finfo(np.float64).max)
    table = TableData(("x",), ((huge,), (huge,), (-huge,)))
    assert summarize_statistics(table, statistics="sum").records()[0]["x_sum"] == huge


def test_quantiles_preserve_equal_nonzero_subnormal_endpoints():
    tiny = float(np.nextafter(0.0, 1.0))
    table = TableData(("x",), ((tiny,), (tiny,)))
    row = summarize_statistics(table, statistics="mean,median,q25,q75").records()[0]
    assert row["x_mean"] == row["x_median"] == row["x_q25"] == row["x_q75"] == tiny


def test_quantiles_round_once_after_subnormal_interpolation():
    tiny = float(np.nextafter(0.0, 1.0))
    table = TableData(("x",), ((tiny,), (2 * tiny,)))
    row = summarize_statistics(table, statistics="median,q25,q75").records()[0]
    # 1.5 * tiny is the halfway tie, rounded to the even significand 2 * tiny.
    assert row["x_median"] == row["x_q75"] == 2 * tiny
    assert row["x_q25"] == tiny


def test_quantiles_do_not_overflow_with_opposite_extreme_endpoints():
    huge = float(np.finfo(np.float64).max)
    table = TableData(("x",), ((-huge,), (huge,)))
    row = summarize_statistics(table, statistics="median,q25,q75,iqr").records()[0]
    assert row["x_median"] == 0
    assert row["x_q25"] == -huge / 2
    assert row["x_q75"] == huge / 2
    assert row["x_iqr"] == huge


def test_optional_identity_counts_do_not_prohibit_object_categories():
    table = TableData(
        ("group", "image", "sample", "x"),
        (("A", "a", "one", 1), ("B", "a", "one", 3)),
    )
    result = summarize_statistics(
        table, group_by="group", image_column="image", sample_column="sample"
    ).records()
    assert [row["x_n"] for row in result] == [1, 1]
    assert [row["sample_count"] for row in result] == [1, 1]


def test_image_averages_allow_sample_to_span_groups_without_sample_aggregation():
    table = TableData(
        ("group", "image", "sample", "x"),
        (("A", "a", "one", 1), ("B", "b", "one", 3)),
    )
    result = summarize_statistics(
        table,
        group_by="group",
        image_column="image",
        sample_column="sample",
        summary_level="Image averages",
    ).records()
    assert [row["x_n"] for row in result] == [1, 1]


def test_read_only_inputs_are_unchanged_and_result_immutable(unequal_table):
    values = np.array([1.0, 2.0, 3.0])
    values.flags.writeable = False
    table = table_from_columns({"x": values})
    before = table.rows
    result = summarize_statistics(table)
    assert table.rows == before
    np.testing.assert_array_equal(values, [1, 2, 3])
    assert not values.flags.writeable
    with pytest.raises(FrozenInstanceError):
        result.name = "changed"
    assert isinstance(result.rows, tuple)


def test_comma_names_roundtrip_json_lists_and_name_conflicts_fail():
    table = TableData(("group,name", "area,um"), (("A", 3),))
    result = summarize_statistics(
        table, group_by='["group,name"]', value_columns='["area,um"]'
    )
    assert result.records()[0]["area,um_mean"] == 3
    with pytest.raises(ValueError, match="names conflict"):
        summarize_statistics(
            TableData(("x_mean", "x"), (("A", 1),)),
            group_by="x_mean",
            value_columns="x",
        )


@pytest.mark.parametrize("version", [0, 3, True, 2.0, "2", None])
def test_invalid_versions_fail(version):
    with pytest.raises(ValueError, match="summary_version"):
        StatisticsRecipe(summary_version=version)
    with pytest.raises(ValueError, match="summary_version"):
        summarize_measurements(TableData(("x",), ((1,),)), summary_version=version)


@pytest.mark.parametrize(
    "params",
    [
        {"summary_level": "biological"},
        {"missing_policy": "drop"},
        {"statistics": ""},
        {"statistics": "sem"},
        {"group_by": 1},
        {"sample_weighting": "auto"},
    ],
)
def test_recipe_rejects_invalid_modern_params(params):
    with pytest.raises(ValueError):
        StatisticsRecipe.from_params(params)


def test_recipe_identity_incomplete_editing_valid_but_execution_rejected():
    recipe = StatisticsRecipe(summary_level="Sample averages")
    assert StatisticsRecipe.from_params(recipe.to_params()) == recipe
    with pytest.raises(FrozenInstanceError):
        recipe.sample_column = "sample"
    with pytest.raises(ValueError, match="Choose a sample identity"):
        summarize_statistics(TableData(("x",), ((1,),)), recipe=recipe)


def test_equal_objects_sample_mode_does_not_require_image_id():
    table = TableData(("sample", "x"), (("one", 1), ("one", 3), ("two", 8)))
    row = summarize_statistics(
        table,
        summary_level="Sample averages",
        sample_column="sample",
        sample_weighting="Equal objects",
    ).records()[0]
    assert row["x_mean"] == 5
    assert row["x_n"] == 2
    assert "image_count" not in row


def test_legacy_output_unchanged_by_default_or_explicit_version():
    table = TableData(("condition", "x"), (("A", 1), ("B", 4)))
    expected = TableData(
        ("condition", "row_count", "x_mean", "x_std"),
        (("A", 1, 1.0, 0.0), ("B", 1, 4.0, 0.0)),
        name="Measurement summary",
        table_kind="Grouped measurement summary",
    )
    assert summarize_measurements(table, statistics="mean,std") == expected
    assert (
        summarize_measurements(table, statistics="mean,std", summary_version=1)
        == expected
    )
    assert (
        summarize_statistics(table, summary_version=1, statistics="mean,std")
        == expected
    )
    modern = summarize_measurements(
        table, summary_version=2, group_by="", statistics="mean,std"
    )
    assert modern.row_count == 1
    assert modern.records()[0]["x_mean"] == 2.5


def test_cancellation_before_start_and_during_scan():
    table = TableData(("x",), tuple((1,) for _ in range(4096)))
    with pytest.raises(OperationCancelled):
        summarize_statistics(table, progress=ProgressContext(cancelled=lambda: True))
    updates = []
    progress = ProgressContext(
        cancelled=lambda: len(updates) >= 2, reporter=updates.append
    )
    with pytest.raises(OperationCancelled):
        summarize_statistics(table, progress=progress)
    assert len(updates) == 2


def test_duplicate_names_and_malformed_rows_rejected():
    with pytest.raises(ValueError, match="unique input column"):
        summarize_statistics(TableData(("x", "x"), ((1, 2),)))
    with pytest.raises(ValueError, match="rows do not match"):
        summarize_statistics(TableData(("x",), ((1, 2),)))
