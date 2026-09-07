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
from napari_vipp.core.operations import binary_threshold
from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, PrototypePipeline
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow


def _pipeline(foreground="Above"):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    node = pipeline.add_node("binary_threshold")
    pipeline.set_param(node.id, "threshold", 2.0)
    pipeline.set_param(node.id, "foreground", foreground)
    assert pipeline.connect("input", node.id).success
    return pipeline, node


def _expected_mask(data, foreground):
    if foreground == "In range":
        return (data >= 0.25) & (data <= 0.75)
    if foreground == "Outside range":
        return (data < 0.25) | (data > 0.75)
    return data > 2.0 if foreground == "Above" else data < 2.0


@pytest.mark.parametrize("foreground", ("Above", "Below"))
@pytest.mark.parametrize("dtype", (bool, np.int16, np.uint16, np.float32, np.float64))
@pytest.mark.parametrize("shape", ((2, 3), (2, 3, 4), (2, 2, 3, 4, 5)))
def test_strict_comparison_preserves_dtype_axes_and_readonly_input(
    foreground, dtype, shape
):
    data = (np.arange(np.prod(shape)).reshape(shape) % 3).astype(dtype)
    data = data[..., ::-1]
    before = data.copy()
    data.setflags(write=False)
    expected = data > 1.0 if foreground == "Above" else data < 1.0

    result = binary_threshold(data, 1.0, foreground=foreground)

    np.testing.assert_array_equal(result, expected, strict=True)
    np.testing.assert_array_equal(data, before, strict=True)
    assert not result[data == 1].any()
    assert not np.shares_memory(result, data)
    assert not data.flags.writeable


@pytest.mark.parametrize(
    ("foreground", "expected"),
    (
        ("Above", [False, False, False, False, True, True, False]),
        ("Below", [True, True, False, False, False, False, False]),
    ),
)
def test_ieee_values_are_compared_not_logically_inverted(foreground, expected):
    data = np.array([-np.inf, -1, -0.0, 0.0, 1, np.inf, np.nan], dtype=np.float32)
    np.testing.assert_array_equal(
        binary_threshold(data, 0.0, foreground=foreground), expected
    )


@pytest.mark.parametrize("foreground", ("Above", "Below"))
@pytest.mark.parametrize("dtype", (np.int64, np.uint64))
def test_wide_integers_retain_existing_numpy_scalar_comparison(foreground, dtype):
    data = np.array([0, 2**53 - 1, 2**53, np.iinfo(dtype).max], dtype=dtype)
    expected = data > float(2**53) if foreground == "Above" else data < float(2**53)
    np.testing.assert_array_equal(
        binary_threshold(data, float(2**53), foreground=foreground), expected
    )


@pytest.mark.parametrize("foreground", ("Above", "Below"))
def test_rgb_luma_reduction_is_explicit_and_direction_independent(foreground):
    grey = np.array([[0.0, 0.25], [0.75, 1.0]])
    rgb = np.repeat(grey[None, :, :], 3, axis=0)
    expected = grey > 0.5 if foreground == "Above" else grey < 0.5
    np.testing.assert_array_equal(
        binary_threshold(rgb, 0.5, channel_axis=0, foreground=foreground), expected
    )


@pytest.mark.parametrize("foreground", ("above", "invalid", "", None, 0))
def test_invalid_foreground_is_not_silently_changed(foreground):
    with pytest.raises(ValueError, match="foreground must be"):
        binary_threshold(np.zeros((2, 3)), foreground=foreground)


@pytest.mark.parametrize("foreground", ("Above", "Below", "In range", "Outside range"))
def test_workflow_roundtrip_and_generated_python_preserve_direction(foreground):
    pipeline, node = _pipeline(foreground)
    data = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    expected = _expected_mask(data, foreground)
    document = serialize_workflow(pipeline)
    payload = deserialize_workflow(json.loads(json.dumps(document)))
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    np.testing.assert_array_equal(
        restored.run(data, input_metadata={"axes": "ZYX"})[node.id], expected
    )
    assert restored.nodes[node.id].params["foreground"] == foreground
    assert restored.output_states[node.id].axes == restored.output_states["input"].axes
    assert restored.output_states[node.id].kind == "binary mask"
    assert foreground in restored.output_states[node.id].history[-1]
    namespace = {"__name__": "exported_pipeline"}
    exec(compile(export_pipeline_to_python(pipeline), "<exported>", "exec"), namespace)
    np.testing.assert_array_equal(
        namespace["run_pipeline"](data, input_metadata={"axes": "ZYX"})[node.id],
        expected,
    )


def test_old_workflow_without_direction_keeps_above_and_new_choice_changes_identity():
    pipeline, node = _pipeline()
    document = serialize_workflow(pipeline)
    old_hash = scientific_workflow_hash(document)
    for record in document["nodes"]:
        record["params"].pop("foreground", None)
        record["params"].pop("low_threshold", None)
        record["params"].pop("high_threshold", None)
    payload = deserialize_workflow(document)
    restored = PrototypePipeline()
    restored.restore_graph(payload["nodes"], payload["connections"])
    assert restored.nodes[node.id].params["foreground"] == "Above"
    data = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(restored.run(data)[node.id], data > 2)
    assert scientific_workflow_hash(serialize_workflow(restored)) == old_hash
    restored.set_param(node.id, "foreground", "Below")
    assert scientific_workflow_hash(serialize_workflow(restored)) != old_hash
    np.testing.assert_array_equal(restored.run(data)[node.id], data < 2)


@pytest.mark.parametrize("foreground", ("Above", "Below", "In range", "Outside range"))
def test_durable_batch_uses_authored_direction_without_mutating_workflow(
    tmp_path, foreground
):
    pipeline, node = _pipeline(foreground)
    output = pipeline.add_node("batch_output")
    pipeline.set_param(output.id, "tag", "mask")
    pipeline.set_param(output.id, "format", "npy")
    assert pipeline.connect(node.id, output.id).success
    workflow = serialize_workflow(pipeline)
    before = json.loads(json.dumps(workflow))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    data = np.linspace(0, 3, 12, dtype=np.float32).reshape(3, 4)
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
                "mask",
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
        np.load(config.output_dir / "field__mask.npy"),
        _expected_mask(data, foreground),
    )
    assert workflow == before


def test_inspector_choice_hint_histogram_and_undo_redo(qtbot):
    from napari_vipp._tests.test_widget import _Viewer
    from napari_vipp._widget import VippWidget

    widget = VippWidget(_Viewer(np.arange(256, dtype=np.uint8).reshape(1, 16, 16)))
    qtbot.addWidget(widget)
    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)
    widget.graph_view.select_node(node.id)
    control = widget._parameter_widgets["foreground"]
    assert [control.combo.itemText(i) for i in range(control.combo.count())] == [
        "Above",
        "Below",
        "In range",
        "Outside range",
    ]
    assert "strictly above" in widget.histogram_interaction_hint.text()
    control.combo.setCurrentText("Below")
    assert widget.pipeline.nodes[node.id].params["foreground"] == "Below"
    assert "strictly below" in widget.histogram_interaction_hint.text()
    assert "Equal values stay background" in widget.histogram_interaction_hint.text()
    widget.undo()
    assert widget.pipeline.nodes[node.id].params["foreground"] == "Above"
    widget.redo()
    assert widget.pipeline.nodes[node.id].params["foreground"] == "Below"
    widget._on_input_histogram_marker_changed("threshold", 64.0)
    assert widget.pipeline.nodes[node.id].params["foreground"] == "Below"
    assert widget.pipeline.nodes[node.id].params["threshold"] == 64.0


def test_parameter_declares_cutoff_and_range_modes():
    spec = next(
        p
        for p in NODE_LIBRARY_BY_ID["binary_threshold"].parameters
        if p.name == "foreground"
    )
    assert spec.default == "Above"
    assert spec.choices == ("Above", "Below", "In range", "Outside range")
    assert "Equal values" in spec.tooltip
