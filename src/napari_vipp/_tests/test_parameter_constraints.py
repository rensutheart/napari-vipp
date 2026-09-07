from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from qtpy.QtCore import Qt

from napari_vipp.core.pipeline import NODE_LIBRARY, ParameterSpec
from napari_vipp.ui.controls import (
    NumericEntryControl,
    ParameterBounds,
    ParameterControl,
)
from napari_vipp.ui.parameter_constraints import (
    HARD_RANGE_NAMES,
    ORDERED_PARAMETERS,
    constrain_parameter_bounds,
)

OPS = {operation.id: operation for operation in NODE_LIBRARY}
ODD_PARAMETERS = [
    (operation.id, spec)
    for operation in NODE_LIBRARY
    for spec in operation.parameters
    if spec.odd_only
]


def bounds_for(spec):
    return ParameterBounds(spec.minimum, spec.maximum, spec.step, spec.decimals, True)


@pytest.mark.parametrize(
    "operation_id,spec",
    ODD_PARAMETERS,
    ids=[f"{op}-{spec.name}" for op, spec in ODD_PARAMETERS],
)
def test_every_odd_parameter_only_emits_odd_values(qtbot, operation_id, spec):
    control = ParameterControl(spec, spec.default, bounds_for(spec))
    qtbot.addWidget(control)
    events = []
    control.valueChanged.connect(events.append)
    for position in range(control.slider.minimum(), control.slider.maximum() + 1):
        control.slider.setValue(position)
        assert control.value() % 2 == 1
    control.value_box.setValue(4)
    assert control.value() % 2 == 1
    control.value_box.setValue(23)
    assert control.value() == 23
    control.value_box.resetToDefault()
    assert control.value() == spec.default
    control.value_box.stepUp()
    assert control.value() == spec.default + 2
    assert all(value % 2 == 1 for value in events)


@pytest.mark.parametrize("control_type", [ParameterControl, NumericEntryControl])
def test_odd_typed_entry_and_loaded_values(qtbot, control_type):
    spec = next(
        p for p in OPS["non_local_means_filter"].parameters if p.name == "patch_size"
    )
    control = control_type(spec, 4, bounds_for(spec))
    qtbot.addWidget(control)
    # Presenting a legacy even value does not silently replace it.
    assert control.value() == 4
    control.value_box.stepUp()
    assert control.value() == 5
    control.show()
    editor = control.value_box.lineEdit()
    editor.selectAll()
    qtbot.keyClicks(editor, "4")
    qtbot.keyClick(editor, Qt.Key_Return)
    assert control.value() == 5
    editor.selectAll()
    qtbot.keyClicks(editor, "7")
    qtbot.keyClick(editor, Qt.Key_Return)
    assert control.value() == 7


def test_positive_tiny_minimum_survives_wide_slider_mapping(qtbot):
    spec = next(
        p
        for p in OPS["normalize_image"].parameters
        if p.name == "reference_standard_deviation"
    )
    control = ParameterControl(spec, 1.0, bounds_for(spec))
    qtbot.addWidget(control)
    assert control.value_box.minimum() == 1e-12
    for position in range(control.slider.minimum(), control.slider.maximum() + 1):
        control.slider.setValue(position)
        assert control.value() > 0
    control.value_box.setValue(0)
    assert control.value() == 1e-12


@pytest.mark.parametrize("control_type", [ParameterControl, NumericEntryControl])
def test_odd_arrow_steps_cannot_land_on_even_range_end(qtbot, control_type):
    spec = ParameterSpec("size", "Size", "int", 5, 3, 16, 2, odd_only=True)
    control = control_type(spec, 15, ParameterBounds(3, 16, 2, 0))
    qtbot.addWidget(control)
    control.value_box.stepUp()
    assert control.value() == 15


def test_log_histogram_requires_positive_custom_limits(qtbot):
    spec = next(
        p for p in OPS["intensity_histogram"].parameters if p.name == "custom_min"
    )
    bounds = constrain_parameter_bounds(
        "intensity_histogram",
        spec,
        {"bin_spacing": "Logarithmic", "custom_min": 0.1, "custom_max": 1.0},
        bounds_for(spec),
    )
    control = NumericEntryControl(spec, 0.1, bounds)
    qtbot.addWidget(control)
    control.value_box.setValue(0)
    assert control.value() > 0


def test_every_numeric_node_control_has_finite_slider_values(qtbot):
    from napari_vipp._widget import VippWidget

    for operation in NODE_LIBRARY:
        params = {p.name: p.default for p in operation.parameters}
        for spec in operation.parameters:
            if spec.kind not in {"int", "float"}:
                continue
            bounds = constrain_parameter_bounds(
                operation.id,
                spec,
                params,
                VippWidget._declared_parameter_bounds(spec),
            )
            control = ParameterControl(spec, spec.default, bounds)
            qtbot.addWidget(control)
            for position in (
                control.slider.minimum(),
                control.slider.maximum() // 2,
                control.slider.maximum(),
            ):
                control.slider.setValue(position)
                assert np.isfinite(control.value()), (operation.id, spec.name)
                if spec.odd_only:
                    assert control.value() % 2 == 1


@pytest.mark.parametrize(
    "operation_id,pair",
    [(op, pair) for op, pairs in ORDERED_PARAMETERS.items() for pair in pairs],
)
def test_paired_limits_use_core_ordering(qtbot, operation_id, pair):
    operation = OPS[operation_id]
    params = {p.name: p.default for p in operation.parameters}
    low, high, strict = pair
    if operation_id == "born_wolf_psf":
        params.update(numerical_aperture=1.2, refractive_index=1.5)
    for name in (low, high):
        spec = next(p for p in operation.parameters if p.name == name)
        bounds = constrain_parameter_bounds(
            operation_id, spec, params, bounds_for(spec)
        )
        control = ParameterControl(spec, params[name], bounds)
        qtbot.addWidget(control)
        if name == low:
            control.value_box.setValue(params[high] + 100)
            assert (
                control.value() < params[high]
                if strict
                else control.value() <= params[high]
            )
        else:
            control.value_box.setValue(params[low] - 100)
            assert (
                control.value() > params[low]
                if strict
                else control.value() >= params[low]
            )
        for position in np.linspace(
            control.slider.minimum(), control.slider.maximum(), 30, dtype=int
        ):
            control.slider.setValue(int(position))
            if name == low:
                assert (
                    control.value() < params[high]
                    if strict
                    else control.value() <= params[high]
                )
            else:
                assert (
                    control.value() > params[low]
                    if strict
                    else control.value() >= params[low]
                )


def test_all_declared_hard_limits_are_honored(qtbot):
    for operation in NODE_LIBRARY:
        params = {p.name: p.default for p in operation.parameters}
        for spec in operation.parameters:
            if spec.name not in HARD_RANGE_NAMES:
                continue
            bounds = constrain_parameter_bounds(
                operation.id, spec, params, bounds_for(spec)
            )
            control = ParameterControl(spec, spec.default, bounds)
            qtbot.addWidget(control)
            assert control.value_box.minimum() >= spec.minimum
            assert control.value_box.maximum() <= spec.maximum


def test_strict_peer_rounding_and_valid_special_cases():
    spec = ParameterSpec("low_sigma", "Low sigma", "float", 1.0, 0, 50, 0.1, 2)
    bounds = constrain_parameter_bounds(
        "difference_of_gaussians", spec, {"high_sigma": 1.001}, bounds_for(spec)
    )
    assert bounds.entry_maximum == 1.0
    for op, name, params in (
        ("filter_labels_by_volume", "max_volume", {"min_volume": 10, "max_volume": 0}),
    ):
        spec = next(p for p in OPS[op].parameters if p.name == name)
        assert constrain_parameter_bounds(
            op, spec, params, bounds_for(spec)
        ) == bounds_for(spec)


@pytest.mark.parametrize("logarithmic", [False, True])
def test_ordinary_step_is_not_a_scientific_multiple(qtbot, logarithmic):
    spec = ParameterSpec("count", "Count", "int", 100, 1, 500, 25)
    control = ParameterControl(
        spec, 101, replace(bounds_for(spec), logarithmic=logarithmic)
    )
    qtbot.addWidget(control)
    control.value_box.setValue(103)
    assert control.value() == 103


def test_reset_does_not_substitute_a_clamped_default(qtbot):
    spec = next(
        p for p in OPS["difference_of_gaussians"].parameters if p.name == "high_sigma"
    )
    bounds = constrain_parameter_bounds(
        "difference_of_gaussians",
        spec,
        {"low_sigma": 5.0},
        bounds_for(spec),
    )
    control = ParameterControl(spec, 8.0, bounds)
    qtbot.addWidget(control)
    menu, reset = control.value_box._create_context_menu()
    assert not reset.isEnabled()
    control.value_box.resetToDefault()
    assert control.value() == 8.0
    menu.deleteLater()


@pytest.mark.parametrize(
    "operation_id,low,high,new_low",
    [
        ("rescale_intensity", "in_low_percentile", "in_high_percentile", 90.0),
        ("rescale_intensity", "out_min", "out_max", 0.75),
        ("normalize_image", "low_percentile", "high_percentile", 90.0),
        ("difference_of_gaussians", "low_sigma", "high_sigma", 2.5),
        ("canny_edges", "low_quantile", "high_quantile", 0.15),
        ("hysteresis_threshold", "low_threshold", "high_threshold", 0.6),
    ],
)
def test_inspector_refreshes_peer_limits_without_editing_peer(
    qtbot,
    operation_id,
    low,
    high,
    new_low,
):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette(operation_id)
    if operation_id == "normalize_image":
        widget._on_param_changed("method", "percentile")
    old_high = node.params[high]
    widget._parameter_widgets[low].value_box.setValue(new_low)
    assert node.params[low] == new_low
    assert node.params[high] == old_high
    high_control = widget._parameter_widgets[high]
    assert high_control.value_box.minimum() >= new_low
    high_control.value_box.setValue(0)
    strict = operation_id in {"normalize_image", "difference_of_gaussians"}
    assert node.params[high] > new_low if strict else node.params[high] >= new_low
    assert widget._parameter_widgets[low].value_box.maximum() <= node.params[high]


def test_nlm_slider_edit_reaches_core_as_odd(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget
    from napari_vipp.core.operations import non_local_means_filter

    widget = VippWidget(_Viewer(np.zeros((8, 8), dtype=np.float32)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("non_local_means_filter")
    control = widget._parameter_widgets["patch_size"]
    control.slider.setValue(4)
    assert node.params["patch_size"] % 2
    result = non_local_means_filter(
        np.zeros((8, 8), dtype=np.float32),
        **{
            key: node.params[key]
            for key in ("patch_size", "patch_distance", "h", "fast_mode")
        },
    )
    assert result.shape == (8, 8)
