"""Explicit intensity inversion preserves ordered bounds and exact arithmetic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    run_batch,
    scientific_workflow_hash,
)
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.operations import rescale_intensity
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _pipeline(invert=False):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("rescale_intensity")
    for name, value in dict(
        cutoff_mode="Values",
        in_low_value=0,
        in_high_value=4,
        out_min=10,
        out_max=30,
        invert_intensity=invert,
    ).items():
        pipeline.set_param(node.id, name, value)
    assert pipeline.connect("input", node.id).success
    return pipeline, node


@pytest.mark.parametrize("mode", ("Percentiles", "Values"))
@pytest.mark.parametrize("invert", (False, True))
@pytest.mark.parametrize(
    "dtype", (np.uint8, np.uint16, np.int16, np.float32, np.float64)
)
@pytest.mark.parametrize("shape", ((3, 5), (2, 3, 5), (2, 2, 3, 4, 5)))
def test_mapping_preserves_axes_dtype_and_readonly_buffers(mode, invert, dtype, shape):
    data = (np.arange(np.prod(shape)).reshape(shape) % 5).astype(dtype)[..., ::-1]
    before = data.copy()
    data.setflags(write=False)
    expected = (30 - 5 * data if invert else 10 + 5 * data).astype(dtype)
    actual = rescale_intensity(
        data,
        cutoff_mode=mode,
        in_low_value=0,
        in_high_value=4,
        out_min=10,
        out_max=30,
        invert_intensity=invert,
    )
    np.testing.assert_array_equal(actual, expected, strict=True)
    np.testing.assert_array_equal(data, before, strict=True)
    assert not np.shares_memory(actual, data)
    assert not data.flags.writeable


@pytest.mark.parametrize(
    "dtype,base",
    (
        (np.int64, np.iinfo(np.int64).min),
        (np.int64, np.iinfo(np.int64).max - 4),
        (np.uint64, np.iinfo(np.uint64).max - 4),
    ),
)
def test_inversion_keeps_adjacent_wide_integer_levels(dtype, base):
    data = np.asarray([int(base) + i for i in range(5)], dtype=dtype)
    actual = rescale_intensity(
        data, out_min=int(base), out_max=int(base) + 4, invert_intensity=True
    )
    np.testing.assert_array_equal(actual, data[::-1], strict=True)


@pytest.mark.parametrize("mode", ("Values", "Percentiles"))
@pytest.mark.parametrize("invert", (False, True))
def test_float_clipping_and_nonfinite_policy(mode, invert):
    data = np.array([[-np.inf, -4, 0, 2, 4, 8, np.inf, np.nan]], np.float32)
    actual = rescale_intensity(
        data,
        cutoff_mode=mode,
        in_low_value=0,
        in_high_value=4,
        out_min=10,
        out_max=30,
        invert_intensity=invert,
    )
    low, high = (0, 4) if mode == "Values" else (-4, 8)
    expected = np.clip((data.astype(np.float64) - low) / (high - low), 0, 1)
    expected = expected * (-20 if invert else 20) + (30 if invert else 10)
    np.testing.assert_array_equal(actual, expected.astype(data.dtype), strict=True)


@pytest.mark.parametrize("dtype", (np.int16, np.float32))
@pytest.mark.parametrize("invert", (False, True))
def test_equal_input_and_output_bounds_are_defined(dtype, invert):
    data = np.full((2, 3), 4, dtype=dtype)
    actual = rescale_intensity(data, out_min=10, out_max=30, invert_intensity=invert)
    np.testing.assert_array_equal(actual, np.full_like(data, 30 if invert else 10))
    np.testing.assert_array_equal(
        rescale_intensity(
            np.arange(6, dtype=dtype), out_min=10, out_max=10, invert_intensity=invert
        ),
        np.full(6, 10, dtype=dtype),
    )


@pytest.mark.parametrize("mode", ("Values", "Percentiles"))
@pytest.mark.parametrize("invert", (False, True))
def test_boolean_inversion_is_explicit_and_does_not_alias(mode, invert):
    data = np.array([[True, False]])
    data.setflags(write=False)
    actual = rescale_intensity(
        data, cutoff_mode=mode, in_low_value=0, in_high_value=1, invert_intensity=invert
    )
    np.testing.assert_array_equal(actual, ~data if invert else data, strict=True)
    assert not np.shares_memory(actual, data)


@pytest.mark.parametrize("dtype", (bool, np.uint8, np.float32))
@pytest.mark.parametrize("invert", (False, True))
def test_reversed_bounds_are_rejected_even_with_inversion(dtype, invert):
    with pytest.raises(ValueError, match="ascending order.*Invert intensity"):
        rescale_intensity(
            np.zeros((2, 3), dtype=dtype),
            out_min=131,
            out_max=0,
            invert_intensity=invert,
        )


@pytest.mark.parametrize("invert", ("False", "True", 0, 1, None, [], np.nan))
def test_inversion_requires_an_actual_boolean(invert):
    with pytest.raises(ValueError, match="must be a boolean"):
        rescale_intensity(np.zeros((2, 3)), invert_intensity=invert)


@pytest.mark.parametrize("invert", (False, True))
def test_workflow_export_history_and_cache_include_inversion(invert):
    pipeline, node = _pipeline(invert)
    document = serialize_workflow(pipeline)
    original_hash = scientific_workflow_hash(document)
    payload = deserialize_workflow(json.loads(json.dumps(document)))
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    data = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
    expected = 30 - 5 * data if invert else 10 + 5 * data
    np.testing.assert_array_equal(
        restored.run(data, input_metadata={"axes": "ZYX"})[node.id],
        expected,
    )
    state = restored.output_states[node.id]
    assert state.axes == restored.output_states["input"].axes
    assert ("inverted intensity" in state.history[-1]) == invert
    namespace = {"__name__": "exported_pipeline"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    np.testing.assert_array_equal(namespace["run_pipeline"](data)[node.id], expected)
    restored.set_param(node.id, "invert_intensity", not invert)
    assert scientific_workflow_hash(serialize_workflow(restored)) != original_hash
    np.testing.assert_array_equal(restored.run(data)[node.id], 40 - expected)


@pytest.mark.parametrize("reversed_bounds", (False, True))
def test_legacy_workflows_default_off_and_preserve_authored_bounds(reversed_bounds):
    pipeline, node = _pipeline()
    document = serialize_workflow(pipeline)
    record = next(n for n in document["nodes"] if n["id"] == node.id)
    record["params"].pop("invert_intensity")
    if reversed_bounds:
        record["params"].update(out_min=30, out_max=10)
    before = json.loads(json.dumps(document))
    payload = deserialize_workflow(document)
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    params = restored.nodes[node.id].params
    assert params["invert_intensity"] is False
    assert params["out_min"] == (30 if reversed_bounds else 10)
    assert params["out_max"] == (10 if reversed_bounds else 30)
    assert document == before
    data = np.arange(5, dtype=np.float32).reshape(1, 5)
    if reversed_bounds:
        with pytest.raises(ValueError, match="Invert intensity"):
            restored.run(data)
    else:
        np.testing.assert_array_equal(restored.run(data)[node.id], 10 + 5 * data)


@pytest.mark.parametrize("invert", (False, True))
def test_batch_saves_the_selected_mapping(tmp_path, invert):
    pipeline, node = _pipeline(invert)
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "scaled")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect(node.id, output.id).success
    workflow = serialize_workflow(pipeline)
    before = json.loads(json.dumps(workflow))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data = np.arange(5, dtype=np.float32).reshape(1, 5)
    np.save(inputs / "field.npy", data)
    config = BatchConfig(
        workflow_file=tmp_path / "workflow.json",
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=tmp_path / "outputs",
        sources=(BatchSourceConfig("input", "Image Source", inputs, "*.npy"),),
        outputs=(
            BatchOutputConfig(
                output.id,
                "Batch Output",
                "scaled",
                "image",
                "npy",
                "",
                "{source_stem}__{tag}",
            ),
        ),
        default_image_format="npy",
        save_python_script=False,
    )
    result = run_batch(workflow, config)
    assert result.manifest_path.exists()
    np.testing.assert_array_equal(
        np.load(config.output_dir / "field__scaled.npy"),
        30 - 5 * data if invert else 10 + 5 * data,
    )
    assert workflow == before


@pytest.mark.parametrize("dtype", (np.uint8, np.float32))
def test_inspector_checkbox_ordered_controls_and_undo(qtbot, tmp_path, dtype):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.arange(256, dtype=dtype).reshape(1, 16, 16)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("rescale_intensity")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    widget._parameter_widgets["out_max"].value_box.setValue(200)
    widget._parameter_widgets["out_min"].value_box.setValue(131)
    assert node.params["out_min"] == 131
    assert node.params["out_max"] == 200
    assert widget._parameter_widgets["out_max"].value_box.minimum() >= 131
    widget._parameter_widgets["out_max"].value_box.setValue(0)
    assert node.params["out_max"] == 131
    widget._parameter_widgets["out_min"].value_box.setValue(200)
    assert node.params["out_min"] == 131
    bounds = (node.params["out_min"], node.params["out_max"])
    checkbox = widget._parameter_widgets["invert_intensity"].checkbox
    assert not checkbox.isChecked()
    checkbox.setChecked(True)
    assert node.params["invert_intensity"] is True
    assert (node.params["out_min"], node.params["out_max"]) == bounds
    widget.undo()
    assert widget.pipeline.nodes[node.id].params["invert_intensity"] is False
    widget.redo()
    assert widget.pipeline.nodes[node.id].params["invert_intensity"] is True
    panel = widget.inspector_panel
    panel.setParent(None)
    qtbot.addWidget(panel)
    panel.resize(460, 800)
    panel.show()
    qtbot.wait(50)
    assert widget.parameter_group.grab().save(str(tmp_path / "rescale-controls.png"))
