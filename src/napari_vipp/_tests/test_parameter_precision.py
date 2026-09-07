"""Catalog-wide protection against rounding fractional controls to integers."""

from decimal import Decimal, localcontext

import numpy as np
import pytest

from napari_vipp.core.pipeline import NODE_LIBRARY
from napari_vipp.ui.controls import NumericEntryControl, ParameterControl

NUMERIC_PARAMETERS = [
    pytest.param(operation.id, spec, id=f"{operation.id}-{spec.name}")
    for operation in NODE_LIBRARY
    for spec in operation.parameters
    if spec.kind in {"int", "float"}
]
FRACTIONAL_MESH_PARAMETERS = [
    ("simplify_mesh", "target_percent", 12.35, 0.1),
    ("simplify_mesh", "aggressiveness", 7.3, 0.1),
    ("smooth_mesh", "strength", 0.37, 0.01),
    ("filter_mesh_objects", "minimum", 0.003125, 0.01),
    ("filter_mesh_objects", "maximum", 0.012345, 0.01),
]


@pytest.mark.parametrize("operation_id,spec", NUMERIC_PARAMETERS)
def test_catalog_precision_represents_defaults_limits_and_steps(operation_id, spec):
    # Inspect declarative specs, not a hand-picked list, so newly added nodes
    # cannot accidentally repeat a missing-decimals bug. Integer/odd controls
    # intentionally remain whole-number-only.
    decimals = spec.decimals if spec.kind == "float" else 0
    values = (spec.default, spec.minimum, spec.maximum, spec.step)
    values += tuple(
        v for v in (spec.slider_minimum, spec.slider_maximum) if v is not None
    )
    with localcontext() as context:
        context.prec = 400  # Includes float64 limits plus fine decimal places.
        for value in values:
            number = Decimal(str(value))
            assert number == round(number, decimals), (operation_id, spec.name, value)
    if spec.odd_only:
        assert spec.kind == "int"
        assert spec.step == 2
        assert spec.default % 2 == 1


@pytest.mark.parametrize("control_type", [ParameterControl, NumericEntryControl])
@pytest.mark.parametrize("operation_id,spec", NUMERIC_PARAMETERS)
def test_numeric_editors_keep_declared_precision(
    qtbot, control_type, operation_id, spec
):
    from napari_vipp._widget import VippWidget

    control = control_type(
        spec, spec.default, VippWidget._declared_parameter_bounds(spec)
    )
    qtbot.addWidget(control)
    assert control.value() == spec.default, (operation_id, spec.name)
    assert control.value_box.singleStep() == spec.step
    events = []
    control.valueChanged.connect(events.append)
    # At zero, fine steps remain distinguishable even for a huge entry range.
    base = 0 if spec.minimum <= 0 <= spec.maximum else spec.default
    if spec.odd_only:
        base = spec.default
    if base + spec.step <= control.value_box.maximum():
        control.value_box.setValue(base)
        control.value_box.stepUp()
        with localcontext() as context:
            context.prec = 400
            expected = float(Decimal(str(base)) + Decimal(str(spec.step)))
        # Decimal entry must survive; tolerate only binary float addition noise,
        # not a loose absolute tolerance that could hide tiny steps becoming zero.
        assert control.value() == pytest.approx(
            expected, rel=0, abs=2 * abs(np.spacing(expected))
        ), (operation_id, spec.name)
        assert events[-1] == control.value()
        if spec.kind == "int":
            assert isinstance(control.value(), int)
            if spec.odd_only:
                assert control.value() % 2 == 1
    control.value_box.resetToDefault()
    assert control.value() == spec.default


@pytest.mark.parametrize(
    "operation,name,value,expected_step",
    FRACTIONAL_MESH_PARAMETERS,
)
def test_mesh_inspector_keeps_fractional_edits_and_rerender(
    qtbot, operation, name, value, expected_step
):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.zeros((8, 8), np.float32)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette(operation)
    control = widget._parameter_widgets[name]
    assert control.value_box.singleStep() == expected_step
    control.value_box.setValue(value)
    assert node.params[name] == value
    widget._render_parameters(node.id)
    control = widget._parameter_widgets[name]
    assert control.value() == value
    if operation == "simplify_mesh":
        control.slider.setValue(control._to_slider(value))
        assert node.params[name] == value
        if name == "target_percent":
            control.value_box.setValue(0)
            assert node.params[name] == 0.01


@pytest.mark.parametrize("operation,name,value,_step", FRACTIONAL_MESH_PARAMETERS)
def test_batch_override_entry_accepts_fractional_mesh_values(
    qtbot, operation, name, value, _step
):
    from napari_vipp._tests.test_ui_batch_parameter_overrides import _source_item
    from napari_vipp.ui.batch_overrides import (
        BatchOverrideParameterSpec,
        BatchOverrideSourceItem,
        BatchParameterOverrideEditor,
    )

    spec = next(
        p
        for op in NODE_LIBRARY
        if op.id == operation
        for p in op.parameters
        if p.name == name
    )
    source = _source_item("fractional-control")
    editor = BatchParameterOverrideEditor()
    qtbot.addWidget(editor)
    binding = BatchOverrideParameterSpec(
        "mesh-node", "Mesh node", operation, spec, spec.default
    )
    assert editor.configure(
        [BatchOverrideSourceItem("input", "Sample", source)], [binding]
    )
    cell = editor.editor_for("input", source, "mesh-node", name)
    qtbot.keyClicks(cell, str(value))
    assert cell.text() == str(value)
    assert editor.overrides()[0].values[0].value == value
