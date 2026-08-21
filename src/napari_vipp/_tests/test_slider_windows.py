from __future__ import annotations

import pytest

from napari_vipp.core.pipeline import HISTOGRAM_BINS_PARAMETER, NODE_LIBRARY
from napari_vipp.ui.controls import ParameterBounds, ParameterControl

_OPERATION_SPECS = {spec.id: spec for spec in NODE_LIBRARY}


def _parameter(operation_id: str, parameter_name: str):
    return next(
        parameter
        for parameter in _OPERATION_SPECS[operation_id].parameters
        if parameter.name == parameter_name
    )


@pytest.mark.parametrize(
    (
        "operation_id",
        "parameter_name",
        "entry_range",
        "slider_window",
    ),
    (
        ("prepare_validate_psf", "minimum_valid_sum", (0.0, 1.0), (0.0, 1e-9)),
        (
            "richardson_lucy_deconvolution",
            "filter_epsilon",
            (0.0, 1.0),
            (0.0, 1e-9),
        ),
        ("ratio_image", "epsilon", (0.0, 1.0), (0.0, 1e-4)),
        (
            "sauvola_threshold",
            "dynamic_range",
            (0.0, 1_000_000.0),
            (0.0, 255.0),
        ),
        ("expand_labels", "distance", (0.0, 10_000.0), (0.0, 100.0)),
        (
            "filter_labels_by_property",
            "min_value",
            (-1_000_000_000.0, 1_000_000_000.0),
            (-1_000.0, 1_000.0),
        ),
        (
            "filter_labels_by_property",
            "max_value",
            (-1_000_000_000.0, 1_000_000_000.0),
            (-1_000.0, 1_000.0),
        ),
        (
            "measure_3d_mesh_morphology",
            "minimum_voxel_count",
            (1, 100_000),
            (1, 1_000),
        ),
        (
            "prune_skeleton_branches",
            "min_branch_length",
            (0.0, 100_000.0),
            (0.0, 100.0),
        ),
        (
            "calculate_weighted_image",
            "offset",
            (-100_000.0, 100_000.0),
            (-1_000.0, 1_000.0),
        ),
    ),
)
def test_extreme_parameters_keep_full_entry_range_and_practical_slider_window(
    operation_id,
    parameter_name,
    entry_range,
    slider_window,
):
    parameter = _parameter(operation_id, parameter_name)

    assert (parameter.minimum, parameter.maximum) == entry_range
    assert (parameter.slider_minimum, parameter.slider_maximum) == slider_window
    assert parameter.slider_minimum <= parameter.default <= parameter.slider_maximum
    assert (
        parameter.slider_minimum > parameter.minimum
        or parameter.slider_maximum < parameter.maximum
    )


def test_histogram_bins_keeps_scientific_entry_limit_with_practical_slider_window():
    parameter = HISTOGRAM_BINS_PARAMETER

    assert (parameter.minimum, parameter.maximum) == (2, 65_536)
    assert (parameter.slider_minimum, parameter.slider_maximum) == (2, 4_096)
    assert parameter.slider_minimum <= parameter.default <= parameter.slider_maximum


def test_tiny_scientific_window_keeps_precision_and_wider_entry_range(qtbot):
    parameter = _parameter("richardson_lucy_deconvolution", "filter_epsilon")
    bounds = ParameterBounds(
        parameter.slider_minimum,
        parameter.slider_maximum,
        parameter.step,
        parameter.decimals,
        expandable=False,
        entry_minimum=parameter.minimum,
        entry_maximum=parameter.maximum,
    )

    control = ParameterControl(parameter, parameter.default, bounds)
    qtbot.addWidget(control)

    assert control.slider.minimum() == 0
    assert control.slider.maximum() == 1_000
    assert control.slider.value() == 1
    assert control.value_box.minimum() == 0.0
    assert control.value_box.maximum() == 1.0

    control.slider.setValue(2)

    assert control.value() == pytest.approx(2e-12)

    control.value_box.setValue(1e-6)

    assert control.slider.value() == control.slider.maximum()
    assert control.value() == pytest.approx(1e-6)
