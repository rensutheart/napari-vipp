from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    DEFERRED_VALUE_RANGE,
    image_state_from_array,
    transform_image_state,
)


def test_large_metadata_value_range_and_pattern_use_all_finite_values():
    data = np.zeros(600_123, dtype=np.float32)
    data[500_001] = -17.0
    data[511_111] = 7.0
    data[-1] = 1_000.0
    data[522_222] = np.nan

    state = image_state_from_array(data)

    assert state.value_range == "-17 to 1000"
    assert state.value_pattern == ""


def test_large_two_level_numeric_metadata_remains_binary_valued():
    data = np.zeros(600_123, dtype=np.uint16)
    data[500_001:] = 65_535

    state = image_state_from_array(data)

    assert state.value_range == "0 to 65535"
    assert state.value_pattern == "binary-valued"


def test_metadata_value_range_preserves_wide_integer_extrema_exactly():
    base = 2**60
    data = np.array([base, base + 1, base + 3], dtype=np.int64)

    state = image_state_from_array(data)

    assert state.value_range == f"{base} to {base + 3}"


def test_metadata_statistics_can_be_deferred_until_background_execution():
    data = np.arange(16, dtype=np.float32)

    state = image_state_from_array(data, defer_statistics=True)

    assert state.value_range == DEFERRED_VALUE_RANGE
    assert state.value_pattern == ""


@pytest.mark.parametrize(
    ("data", "expected_mode"),
    [
        (
            np.zeros((3, 4), dtype=np.uint16),
            "native integer levels (uint16; Float histogram bins ignored)",
        ),
        (
            np.zeros((3, 4), dtype=np.float32),
            "4096 equal-width float bins (float32)",
        ),
    ],
)
def test_histogram_threshold_history_records_scope_and_dtype_mode(
    data,
    expected_mode,
):
    input_state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    output_state = transform_image_state(
        np.zeros(data.shape, dtype=bool),
        input_state,
        operation_id="otsu_threshold",
        operation_title="Otsu Threshold",
        params={
            "threshold_scope": "Slice histogram",
            "histogram_bins": 4_096,
        },
    )

    history = output_state.history[-1]
    assert history.startswith("Otsu Threshold: Slice histogram; ")
    assert expected_mode in history
    assert history.endswith("non-finite values excluded and set to background")


def test_boolean_threshold_history_records_passthrough_instead_of_algorithm():
    data = np.zeros((3, 4), dtype=bool)
    input_state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    output_state = transform_image_state(
        data.copy(),
        input_state,
        operation_id="triangle_threshold",
        operation_title="Triangle Threshold",
        params={"threshold_scope": "Stack histogram", "histogram_bins": 256},
    )

    assert output_state.history[-1] == (
        "Triangle Threshold: existing boolean segmentation preserved; "
        "automatic threshold bypassed"
    )


def test_li_threshold_history_records_raw_finite_value_policy():
    data = np.zeros((3, 4), dtype=np.float32)
    input_state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    output_state = transform_image_state(
        np.zeros(data.shape, dtype=bool),
        input_state,
        operation_id="li_threshold",
        operation_title="Li Threshold",
        params={"threshold_scope": "Stack histogram"},
    )

    assert output_state.history[-1] == (
        "Li Threshold: Stack histogram; raw finite-value iteration; "
        "non-finite values excluded and set to background"
    )


@pytest.mark.parametrize(
    ("operation_id", "title", "params", "expected"),
    [
        (
            "rescale_intensity",
            "Rescale Intensity",
            {
                "cutoff_mode": "Percentiles",
                "in_low_percentile": 1.0,
                "in_high_percentile": 99.0,
                "in_low_value": -5.0,
                "in_high_value": 5.0,
                "out_min": 0.0,
                "out_max": 255.0,
            },
            "Rescale Intensity: finite-value percentiles 1..99; output 0..255",
        ),
        (
            "rescale_intensity",
            "Rescale Intensity",
            {
                "cutoff_mode": "Values",
                "in_low_percentile": 0.0,
                "in_high_percentile": 100.0,
                "in_low_value": -5.0,
                "in_high_value": 5.0,
                "out_min": -1.0,
                "out_max": 1.0,
            },
            "Rescale Intensity: explicit input values -5..5; output -1..1",
        ),
        (
            "clip_intensity",
            "Clip",
            {"cutoff_mode": "Data range", "minimum": 2.0, "maximum": 4.0},
            "Clip: full data range preserved (no clipping)",
        ),
        (
            "clip_intensity",
            "Clip",
            {"cutoff_mode": "Values", "minimum": 2.0, "maximum": 4.0},
            "Clip: explicit values 2..4",
        ),
    ],
)
def test_intensity_cutoff_history_records_only_active_mode(
    operation_id,
    title,
    params,
    expected,
):
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    input_state = image_state_from_array(data, layer_metadata={"axes": "YX"})

    output_state = transform_image_state(
        data.copy(),
        input_state,
        operation_id=operation_id,
        operation_title=title,
        params=params,
    )

    assert output_state.history[-1] == expected
