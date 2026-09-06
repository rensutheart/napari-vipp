from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.operations import (
    intensity_histogram,
    intensity_histogram_table_columns,
)
from napari_vipp.core.pipeline import (
    MANUAL_AUTO_RECALCULATE_PARAM,
    NODE_LIBRARY_BY_ID,
    PrototypePipeline,
    graph_node_from_persisted_params,
)
from napari_vipp.core.progress import OperationCancelled, ProgressContext
from napari_vipp.core.tables import table_state_from_data
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow

EXPECTED_COLUMNS = (
    "bin_index",
    "bin_left",
    "bin_right",
    "bin_center",
    "bin_width",
    "count",
    "fraction",
    "density",
    "cumulative_count",
    "cumulative_fraction",
)


def _column(table, name: str, *, dtype=None) -> np.ndarray:
    index = table.columns.index(name)
    return np.asarray([row[index] for row in table.rows], dtype=dtype)


def test_intensity_histogram_catalog_contract_is_one_input_manual_measurement():
    spec = NODE_LIBRARY_BY_ID["intensity_histogram"]

    assert spec.title == "Intensity Histogram"
    assert spec.category == "Measurements"
    assert spec.input_type == "array"
    assert spec.output_type == "table"
    assert spec.max_inputs == 1
    assert spec.execution_policy == "manual"
    assert tuple(parameter.name for parameter in spec.parameters) == (
        "bin_count",
        "range_mode",
        "custom_min",
        "custom_max",
        "bin_spacing",
    )


def test_intensity_histogram_defaults_to_auto_recalculate_with_persisted_opt_out():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    histogram = pipeline.add_node("intensity_histogram")

    assert histogram.params[MANUAL_AUTO_RECALCULATE_PARAM] is True
    assert pipeline.node_auto_recalculate(histogram.id)

    pipeline.set_node_auto_recalculate(histogram.id, False)
    payload = deserialize_workflow(serialize_workflow(pipeline))
    restored = PrototypePipeline()
    restored.restore_graph(
        payload["nodes"],
        payload["connections"],
        payload.get("output_tunnels", ()),
    )

    assert restored.nodes[histogram.id].params[
        MANUAL_AUTO_RECALCULATE_PARAM
    ] is False
    assert not restored.node_auto_recalculate(histogram.id)


def test_legacy_histogram_without_auto_preference_uses_new_default():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    histogram = pipeline.add_node("intensity_histogram")
    document = serialize_workflow(pipeline)
    saved_histogram = next(
        node for node in document["nodes"] if node["id"] == histogram.id
    )
    saved_histogram["params"].pop(MANUAL_AUTO_RECALCULATE_PARAM)

    payload = deserialize_workflow(document)
    restored = PrototypePipeline()
    restored.restore_graph(
        payload["nodes"],
        payload["connections"],
        payload.get("output_tunnels", ()),
    )

    assert MANUAL_AUTO_RECALCULATE_PARAM not in restored.nodes[histogram.id].params
    assert restored.node_auto_recalculate(histogram.id)


def test_histogram_auto_recalculate_preference_must_be_boolean():
    spec = NODE_LIBRARY_BY_ID["intensity_histogram"]
    params = {parameter.name: parameter.default for parameter in spec.parameters}
    params[MANUAL_AUTO_RECALCULATE_PARAM] = "false"

    with pytest.raises(ValueError, match="(?i)boolean"):
        graph_node_from_persisted_params(
            "histogram",
            spec.id,
            params,
            index=0,
        )


def test_intensity_histogram_declares_stable_analysis_ready_columns():
    assert intensity_histogram_table_columns() == EXPECTED_COLUMNS


def test_custom_linear_histogram_accounts_for_every_input_value():
    data = np.asarray(
        [-1.0, 0.0, 0.5, 1.0, 2.0, np.nan, np.inf, -np.inf],
        dtype=np.float64,
    )

    table = intensity_histogram(
        data,
        bin_count=2,
        range_mode="Custom range",
        custom_min=0.0,
        custom_max=1.0,
        bin_spacing="Linear",
        source_name="sample-a.tif",
    )

    assert table.columns == EXPECTED_COLUMNS
    assert table.name == "Intensity histogram"
    assert table.table_kind == "Intensity histogram bins"
    assert table.source_name == "sample-a.tif"
    np.testing.assert_allclose(_column(table, "bin_left"), [0.0, 0.5])
    np.testing.assert_allclose(_column(table, "bin_right"), [0.5, 1.0])
    np.testing.assert_array_equal(_column(table, "bin_index"), [1, 2])
    np.testing.assert_array_equal(_column(table, "count"), [1, 2])
    np.testing.assert_allclose(_column(table, "fraction"), [1 / 3, 2 / 3])
    np.testing.assert_allclose(_column(table, "density"), [2 / 3, 4 / 3])
    np.testing.assert_array_equal(_column(table, "cumulative_count"), [1, 3])
    np.testing.assert_allclose(
        _column(table, "cumulative_fraction"),
        [1 / 3, 1.0],
    )

    metadata = table.histogram_metadata
    assert metadata is not None
    assert metadata.input_value_count == 8
    assert metadata.finite_value_count == 5
    assert metadata.nan_value_count == 1
    assert metadata.positive_infinite_value_count == 1
    assert metadata.negative_infinite_value_count == 1
    assert metadata.binned_value_count == 3
    assert metadata.underflow_count == 1
    assert metadata.overflow_count == 1
    assert metadata.nonpositive_excluded_count == 0
    assert metadata.effective_minimum == 0.0
    assert metadata.effective_maximum == 1.0
    assert metadata.bin_count == 2
    assert metadata.bin_spacing == "Linear"


def test_data_range_includes_the_rightmost_value_in_the_last_bin():
    table = intensity_histogram(
        np.asarray([0.0, 1.0, 2.0, 3.0]),
        bin_count=3,
        range_mode="Data range",
        bin_spacing="Linear",
    )

    np.testing.assert_allclose(_column(table, "bin_left"), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(_column(table, "bin_right"), [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(_column(table, "count"), [1, 1, 2])
    assert table.histogram_metadata.binned_value_count == 4
    assert table.histogram_metadata.underflow_count == 0
    assert table.histogram_metadata.overflow_count == 0


def test_logarithmic_bins_use_true_geometric_edges_and_exact_density_area():
    data = np.asarray(
        [1, 2, 9, 10, 11, 99, 100, 101, 999, 1000, 0, -1],
        dtype=np.float64,
    )

    table = intensity_histogram(
        data,
        bin_count=3,
        range_mode="Data range",
        bin_spacing="Logarithmic",
    )

    left = _column(table, "bin_left", dtype=np.float64)
    right = _column(table, "bin_right", dtype=np.float64)
    width = _column(table, "bin_width", dtype=np.float64)
    density = _column(table, "density", dtype=np.float64)
    np.testing.assert_allclose(left, [1.0, 10.0, 100.0])
    np.testing.assert_allclose(right, [10.0, 100.0, 1000.0])
    np.testing.assert_allclose(width, right - left)
    np.testing.assert_array_equal(_column(table, "count"), [3, 3, 4])
    assert np.sum(density * width) == pytest.approx(1.0)
    assert _column(table, "fraction", dtype=np.float64).sum() == pytest.approx(1.0)
    assert _column(table, "cumulative_fraction", dtype=np.float64)[-1] == (
        pytest.approx(1.0)
    )
    assert _column(table, "cumulative_count")[-1] == 10

    metadata = table.histogram_metadata
    assert metadata.finite_value_count == 12
    assert metadata.nonpositive_excluded_count == 2
    assert metadata.binned_value_count == 10
    assert metadata.underflow_count == 0
    assert metadata.overflow_count == 0
    assert metadata.effective_minimum == pytest.approx(1.0)
    assert metadata.effective_maximum == pytest.approx(1000.0)
    assert metadata.bin_spacing == "Logarithmic"


def test_custom_logarithmic_range_separates_nonpositive_underflow_and_overflow():
    table = intensity_histogram(
        np.asarray([-2.0, 0.0, 0.5, 1.0, 10.0, 100.0, 101.0]),
        bin_count=2,
        range_mode="Custom range",
        custom_min=1.0,
        custom_max=100.0,
        bin_spacing="Logarithmic",
    )

    metadata = table.histogram_metadata
    assert metadata.nonpositive_excluded_count == 2
    assert metadata.underflow_count == 1
    assert metadata.overflow_count == 1
    assert metadata.binned_value_count == 3
    np.testing.assert_array_equal(_column(table, "count"), [1, 2])


def test_constant_data_retains_requested_bins_with_a_non_degenerate_range():
    table = intensity_histogram(
        np.full(5, 7.0),
        bin_count=4,
        range_mode="Data range",
        bin_spacing="Linear",
    )

    assert table.row_count == 4
    np.testing.assert_allclose(
        np.concatenate(
            (
                _column(table, "bin_left")[:1],
                _column(table, "bin_right"),
            )
        ),
        np.linspace(6.5, 7.5, 5),
    )
    counts = _column(table, "count")
    assert counts.sum() == 5
    assert np.count_nonzero(counts) == 1
    assert table.histogram_metadata.effective_minimum == pytest.approx(6.5)
    assert table.histogram_metadata.effective_maximum == pytest.approx(7.5)


@pytest.mark.parametrize(
    ("data", "expected_counts"),
    [
        (np.asarray([], dtype=np.float32), (0, 0, 0, 0)),
        (np.asarray([np.nan, np.inf, -np.inf]), (3, 0, 1, 1)),
    ],
)
def test_data_range_without_finite_values_returns_an_empty_descriptive_table(
    data,
    expected_counts,
):
    table = intensity_histogram(
        data,
        bin_count=8,
        range_mode="Data range",
        bin_spacing="Linear",
    )

    assert table.columns == EXPECTED_COLUMNS
    assert table.row_count == 0
    metadata = table.histogram_metadata
    assert (
        metadata.input_value_count,
        metadata.finite_value_count,
        metadata.nan_value_count,
        metadata.positive_infinite_value_count,
    ) == expected_counts
    assert metadata.negative_infinite_value_count == (1 if data.size else 0)
    assert metadata.binned_value_count == 0
    assert metadata.effective_minimum is None
    assert metadata.effective_maximum is None


def test_custom_range_can_represent_an_all_zero_count_histogram():
    table = intensity_histogram(
        np.asarray([np.nan, np.inf, -np.inf]),
        bin_count=4,
        range_mode="Custom range",
        custom_min=-1.0,
        custom_max=1.0,
        bin_spacing="Linear",
    )

    assert table.row_count == 4
    np.testing.assert_array_equal(_column(table, "count"), np.zeros(4, dtype=int))
    assert np.isnan(_column(table, "fraction", dtype=np.float64)).all()
    assert np.isnan(_column(table, "density", dtype=np.float64)).all()
    assert np.isnan(_column(table, "cumulative_fraction", dtype=np.float64)).all()
    np.testing.assert_array_equal(
        _column(table, "cumulative_count"),
        np.zeros(4, dtype=int),
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"bin_count": 1}, "bin count"),
        ({"range_mode": "Unknown"}, "range mode"),
        (
            {
                "range_mode": "Custom range",
                "custom_min": 1.0,
                "custom_max": 1.0,
            },
            "maximum",
        ),
        (
            {
                "range_mode": "Custom range",
                "custom_min": 0.0,
                "custom_max": 10.0,
                "bin_spacing": "Logarithmic",
            },
            "positive|greater than zero",
        ),
        ({"bin_spacing": "Unknown"}, "spacing"),
    ],
)
def test_invalid_histogram_configuration_fails_with_actionable_errors(
    kwargs,
    message,
):
    with pytest.raises(ValueError, match=rf"(?i){message}"):
        intensity_histogram(np.arange(8, dtype=np.float32), **kwargs)


def test_histogram_metadata_is_carried_into_table_state_and_serialization():
    table = intensity_histogram(
        np.asarray([0.0, 1.0, np.nan, np.inf]),
        bin_count=2,
    )

    state = table_state_from_data(table)

    assert state.histogram_metadata == table.histogram_metadata
    serialized = state.to_dict()["histogram_metadata"]
    assert serialized["input_value_count"] == 4
    assert serialized["finite_value_count"] == 2
    assert serialized["nan_value_count"] == 1
    assert serialized["positive_infinite_value_count"] == 1
    assert serialized["binned_value_count"] == 2


def test_intensity_histogram_workflow_round_trip_preserves_authored_settings():
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    histogram = pipeline.add_node("intensity_histogram")
    assert pipeline.connect("input", histogram.id).success
    for name, value in {
        "bin_count": 1024,
        "range_mode": "Custom range",
        "custom_min": 0.25,
        "custom_max": 4096.0,
        "bin_spacing": "Logarithmic",
    }.items():
        pipeline.set_param(histogram.id, name, value)

    payload = deserialize_workflow(serialize_workflow(pipeline))
    restored = PrototypePipeline()
    restored.restore_graph(
        payload["nodes"],
        payload["connections"],
        payload.get("output_tunnels", ()),
    )

    assert restored.nodes[histogram.id].params == pipeline.nodes[histogram.id].params
    assert restored.connections == pipeline.connections


def test_intensity_histogram_honors_cooperative_cancellation():
    progress = ProgressContext(cancelled=lambda: True)

    with pytest.raises(OperationCancelled):
        intensity_histogram(
            np.arange(1_000_000, dtype=np.float32),
            progress=progress,
        )
