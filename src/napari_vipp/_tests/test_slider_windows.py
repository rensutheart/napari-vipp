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
        ("build_tracks", "maximum_displacement", (1e-6, 1e12), (1e-6, 100.0)),
        ("build_tracks", "maximum_gap", (0, 1_000_000), (0, 10)),
        ("find_peaks", "minimum_value", (-1e12, 1e12), (-1.0, 1.0)),
        ("find_peaks", "minimum_separation", (0.0, 1e12), (0.0, 100.0)),
        ("find_peaks", "maximum_detections", (1, 1_000_000), (1, 10_000)),
        ("find_peaks", "border_exclusion", (0, 1_000_000), (0, 100)),
        ("detect_spots_per_frame", "minimum_value", (-1e12, 1e12), (-1.0, 1.0)),
        ("detect_spots_per_frame", "minimum_separation", (0.0, 1e12), (0.0, 100.0)),
        ("detect_spots_per_frame", "maximum_detections", (1, 1_000_000), (1, 10_000)),
        ("detect_spots_per_frame", "border_exclusion", (0, 1_000_000), (0, 100)),
        ("compare_images", "data_range", (1e-12, 1e12), (1.0, 65_535.0)),
        ("estimate_registration", "reference_channel", (0, 9999), (0, 15)),
        ("estimate_registration", "reference_time", (0, 1_000_000), (0, 100)),
        ("estimate_registration", "iterations", (1, 5000), (1, 1000)),
        ("apply_transform", "outside_value", (-1e12, 1e12), (-255.0, 255.0)),
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


@pytest.mark.parametrize(
    "operation_id,parameter_name,typed_value,mode",
    (
        ("build_tracks", "maximum_displacement", 250.123456, None),
        ("build_tracks", "maximum_gap", 25, None),
        ("find_peaks", "minimum_value", -1000.123456, None),
        ("find_peaks", "minimum_separation", 250.123456, None),
        ("find_peaks", "maximum_detections", 25000, None),
        ("find_peaks", "border_exclusion", 250, None),
        ("detect_spots_per_frame", "minimum_value", 1000.123456, None),
        ("detect_spots_per_frame", "minimum_separation", 250.123456, None),
        ("detect_spots_per_frame", "maximum_detections", 25000, None),
        ("detect_spots_per_frame", "border_exclusion", 250, None),
        ("compare_images", "data_range", 0.123456789012, None),
        ("compare_images", "data_range", 100000.123456, None),
        ("estimate_registration", "reference_channel", 25, None),
        ("estimate_registration", "reference_time", 250, "Time series"),
        ("estimate_registration", "iterations", 2500, "Rigid"),
        ("apply_transform", "outside_value", -1000.125, None),
    ),
)
def test_new_node_slider_window_preserves_typed_values_when_reopening_inspector(
    qtbot, operation_id, parameter_name, typed_value, mode
):
    import numpy as np
    from qtpy.QtCore import Qt

    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette(operation_id)
    if mode is not None:
        node.params["mode" if mode == "Time series" else "model"] = mode
        widget._render_parameters(node.id)
    parameter = _parameter(operation_id, parameter_name)
    control = widget._parameter_widgets[parameter_name]
    slider_range = (control.slider.minimum(), control.slider.maximum())

    control.slider.setValue(control.slider.maximum())
    assert control.value() == pytest.approx(parameter.slider_maximum)
    assert node.params[parameter_name] == control.value()
    assert control.value_box.minimum() == parameter.minimum
    assert control.value_box.maximum() == parameter.maximum

    control.value_box.lineEdit().selectAll()
    qtbot.keyClicks(control.value_box.lineEdit(), str(typed_value))
    qtbot.keyClick(control.value_box.lineEdit(), Qt.Key_Return)
    assert control.value() == typed_value
    assert node.params[parameter_name] == typed_value
    assert (control.slider.minimum(), control.slider.maximum()) == slider_range
    pinned_end = (
        control.slider.minimum()
        if typed_value < parameter.slider_minimum
        else control.slider.maximum()
    )
    assert control.slider.value() == pinned_end

    widget._render_parameters(node.id)
    reopened = widget._parameter_widgets[parameter_name]
    assert reopened.value() == typed_value
    assert node.params[parameter_name] == typed_value
    assert (reopened.slider.minimum(), reopened.slider.maximum()) == slider_range
    assert reopened.slider.value() == pinned_end
    widget._debounce_timer.stop()
