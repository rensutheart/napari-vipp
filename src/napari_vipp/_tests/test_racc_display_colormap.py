"""RACC palettes are presentation-only and consistent on every VIPP refresh."""

from copy import deepcopy

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget
from napari_vipp.core.execution import PipelinePresentationShadowResult
from napari_vipp.core.metadata import ChannelMetadata, image_state_from_array
from napari_vipp.core.preview import normalize_thumbnail_with_colormap
from napari_vipp.core.workflow import serialize_workflow

RACC_OPERATIONS = ("racc_index", "masked_racc_index")


def _cached_result(widget, node, *, dtype="float32", color=None):
    # Exercise the accepted-result display boundary without changing or running
    # the scientific operation. Shape ends in 3 to catch napari RGB guessing.
    values = np.linspace(0, 1, 24).reshape(8, 3)
    output = (values * 255).astype(dtype) if dtype == "uint8" else values.astype(dtype)
    output.setflags(write=False)
    state = image_state_from_array(
        output,
        layer_metadata={"axes": "YX"},
        channels=(ChannelMetadata(name="Channel 1", color=color),),
    )
    widget.pipeline.outputs[node.id] = output
    widget.pipeline.output_states[node.id] = state
    widget.pipeline.node_outputs[node.id] = [output]
    widget.pipeline.node_output_states[node.id] = [state]
    return output, state


def _widget(qtbot, operation, *, dtype="float32", color=None, real_viewer=False):
    viewer = ViewerModel() if real_viewer else _Viewer()
    if real_viewer:
        viewer.add_image(np.zeros((8, 3)), name="input", rgb=False)
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    widget.auto_recalculate_checkbox.setChecked(False)
    widget.thumbnail_scope_combo.setCurrentText("Slice")
    widget.thumbnail_contrast_combo.setCurrentText("Min-max")
    node = widget.add_node_from_palette(operation)
    output, state = _cached_result(widget, node, dtype=dtype, color=color)
    return widget, node, output, state


def _assert_thumbnail(widget, node_id, output, colormap, *, contrast_limits=None):
    widget._update_thumbnails()
    rendered = widget.graph_view._cards[node_id].preview.source_pixmap().toImage()
    expected = normalize_thumbnail_with_colormap(
        output,
        size=widget._thumbnail_render_size(),
        colormap=colormap,
        contrast_mode="Min-max",
        contrast_reference=output,
        contrast_limits=contrast_limits,
    )
    assert (rendered.width(), rendered.height()) == (
        expected.shape[1], expected.shape[0],
    )
    for y, x in ((0, 0), (expected.shape[0] // 2, expected.shape[1] // 2), (-1, -1)):
        y %= rendered.height()
        x %= rendered.width()
        assert rendered.pixelColor(x, y).getRgb()[:3] == tuple(expected[y, x])


@pytest.mark.parametrize("operation", RACC_OPERATIONS)
@pytest.mark.parametrize("dtype", ("float32", "uint8"))
def test_racc_magma_overrides_thumbnail_and_inherited_channel_palette(
    qtbot, monkeypatch, operation, dtype,
):
    widget, node, output, state = _widget(
        qtbot, operation, dtype=dtype, color=0x0000FF,
    )
    original = output.copy()
    metadata = deepcopy(state.to_dict())
    workflow = serialize_workflow(widget.pipeline)
    monkeypatch.setattr(
        widget.pipeline, "run",
        lambda *_a, **_k: pytest.fail("Palette refresh recalculated scientific data"),
    )

    for palette in ("Gray", "Red", "Viridis"):
        widget.thumbnail_colormap_combo.setCurrentText(palette)
        _assert_thumbnail(widget, node.id, output, "Magma")
        widget.inspect_node(node.id)
        assert widget.viewer.layers["VIPP Inspect"].colormap == "magma"
        widget._set_active_pin_layer(node.id, output)
        assert widget._active_pinned_layers()[0].colormap == "magma"

    assert widget.pipeline.outputs[node.id] is output
    assert widget.pipeline.output_states[node.id] is state
    assert state.to_dict() == metadata
    assert serialize_workflow(widget.pipeline) == workflow
    np.testing.assert_array_equal(output, original)
    for layer in (
        widget.viewer.layers["VIPP Inspect"], *widget._active_pinned_layers(),
    ):
        assert np.shares_memory(layer.data, output)
        assert not layer.data.flags.writeable
        assert not layer.rgb


@pytest.mark.parametrize("operation", RACC_OPERATIONS)
def test_racc_refresh_overrides_saved_colormap_only(qtbot, monkeypatch, operation):
    widget, node, output, _state = _widget(qtbot, operation)
    widget.inspect_node(node.id)
    inspect = widget.viewer.layers["VIPP Inspect"]
    inspect.colormap = "red"
    inspect.opacity = 0.4
    inspect.gamma = 1.7
    inspect.visible = False
    inspect.contrast_limits = (0.15, 0.65)
    widget._remember_current_inspect_display_profiles()
    documents = widget._inspect_display_profile_documents({node.id})
    assert documents[0]["settings"]["colormap"] == "red"
    widget._discard_inspect_layers()
    widget._load_inspect_display_profiles(documents, {node.id})
    monkeypatch.setattr(
        widget.pipeline, "run", lambda *_a, **_k: pytest.fail("Display recalculated"),
    )

    widget.inspect_node(node.id)
    restored = widget.viewer.layers["VIPP Inspect"]
    assert restored.colormap == "magma"
    assert restored.opacity == 0.4
    assert restored.gamma == 1.7
    assert restored.visible is False
    assert restored.contrast_limits == (0.15, 0.65)

    # napari controls remain usable; the next VIPP refresh chooses RACC Magma.
    restored.colormap = "gray"
    assert restored.colormap == "gray"
    widget._refresh_inspection_layer_if_active()
    assert widget.viewer.layers["VIPP Inspect"] is restored
    assert restored.colormap == "magma"
    assert restored.opacity == 0.4
    assert restored.contrast_limits == (0.15, 0.65)
    widget._set_active_pin_layer(node.id, output)
    pinned = widget._active_pinned_layers()[0]
    pinned.colormap = "red"
    pinned.opacity = 0.3
    widget._refresh_pinned_layer_if_active()
    assert widget._active_pinned_layers()[0] is pinned
    assert pinned.colormap == "magma"
    assert pinned.opacity == 0.3

    widget._reset_selected_inspect_display()
    assert widget.viewer.layers["VIPP Inspect"].colormap == "magma"
    assert widget.viewer.layers["VIPP Inspect"].opacity == 1.0
    assert widget.pipeline.outputs[node.id] is output


@pytest.mark.parametrize("operation", RACC_OPERATIONS)
def test_real_napari_accepts_magma_on_racc_creation_and_reuse(qtbot, operation):
    widget, node, output, _state = _widget(qtbot, operation, real_viewer=True)
    widget.inspect_node(node.id)
    inspect = widget.viewer.layers["VIPP Inspect"]
    assert inspect.colormap.name == "magma"
    assert not inspect.rgb
    widget._set_active_pin_layer(node.id, output)
    assert widget._active_pinned_layers()[0].colormap.name == "magma"
    inspect.colormap = "red"
    widget._refresh_inspection_layer_if_active()
    assert widget.viewer.layers["VIPP Inspect"] is inspect
    assert inspect.colormap.name == "magma"
    np.testing.assert_array_equal(inspect.data, output)


def test_racc_palette_does_not_override_other_nodes_or_scatter_density(qtbot):
    widget, racc, _output, _state = _widget(qtbot, "racc_index")
    widget.inspect_node(racc.id)
    inspect = widget.viewer.layers["VIPP Inspect"]
    ordinary = widget.add_node_from_palette("rescale_intensity")
    output, _state = _cached_result(widget, ordinary)
    widget.thumbnail_colormap_combo.setCurrentText("Viridis")
    _assert_thumbnail(widget, ordinary.id, output, "Viridis")
    widget.inspect_node(ordinary.id)
    assert widget.viewer.layers["VIPP Inspect"] is inspect
    assert inspect.colormap == "gray"
    inspect.colormap = "red"
    widget.inspect_node(racc.id)
    assert inspect.colormap == "magma"
    widget.inspect_node(ordinary.id)
    assert inspect.colormap == "red"
    assert widget.colocalization_scatter_colormap_combo.currentText() == "Viridis"

    _cached_result(widget, ordinary, color=0x0000FF)
    widget.inspect_node(ordinary.id)
    assert inspect.colormap == "blue"


@pytest.mark.parametrize("operation", RACC_OPERATIONS)
@pytest.mark.parametrize("axes", ("CYX", "Y,X,RGB"))
def test_multicomponent_racc_remains_scalar_magma_without_changing_axes(
    qtbot, monkeypatch, operation, axes,
):
    widget, node, _output, _state = _widget(qtbot, operation, real_viewer=True)
    plane = np.linspace(0, 1, 40).reshape(8, 5)
    values = np.stack((plane, plane * 0.5, plane * 0.25))
    if axes == "CYX":
        output = values.astype(np.float32)
        step = (1, 0, 0)
        expected = output[1]
    else:
        output = np.moveaxis(values * 255, 0, -1).astype(np.uint8)
        step = (0, 0, 1)
        expected = output[..., 1]
    output.setflags(write=False)
    state = image_state_from_array(
        output,
        layer_metadata={"axes": axes},
        channels=tuple(
            ChannelMetadata(name=f"Channel {i}", color=color)
            for i, color in enumerate((0xFF0000, 0x00FF00, 0x0000FF))
        ),
    )
    widget.pipeline.outputs[node.id] = output
    widget.pipeline.output_states[node.id] = state
    widget.pipeline.node_outputs[node.id] = [output]
    widget.pipeline.node_output_states[node.id] = [state]
    original_state = deepcopy(state.to_dict())
    original_data = output.copy()
    monkeypatch.setattr(widget, "_current_step", lambda: step)
    monkeypatch.setattr(widget, "_current_step_nsteps", lambda: output.shape)
    monkeypatch.setattr(
        widget.pipeline, "run", lambda *_a, **_k: pytest.fail("Display recalculated"),
    )

    _assert_thumbnail(widget, node.id, expected, "Magma")
    assert not widget._thumbnail_preview_consumes_contrast(node.id, output, state)
    request = widget._thumbnail_contrast_limit_request(
        node.id, output, state, "Min-max", "Stack", "image",
    )
    assert request is not None  # uint8 indices are not pre-encoded RGB pixels.
    assert request.channel_axis is None
    limits = (float(output.min()), float(output.max()))
    widget._thumbnail_contrast_limit_cache[request.key] = limits
    widget.thumbnail_scope_combo.setCurrentText("Stack")
    _assert_thumbnail(widget, node.id, expected, "Magma", contrast_limits=limits)

    widget.inspect_node(node.id)
    widget._set_active_pin_layer(node.id, output)
    assert not widget._colored_channel_axis_layers("VIPP Inspect")
    assert not widget._rgb_channel_layers("VIPP Inspect")
    for layer in (
        widget.viewer.layers["VIPP Inspect"], *widget._active_pinned_layers(),
    ):
        assert not layer.rgb
        assert layer.colormap.name == "magma"
        assert layer.data.shape == output.shape
        assert layer.metadata["vipp_image_state"] == original_state
        assert np.shares_memory(layer.data, output)
        assert not layer.data.flags.writeable
    assert widget.pipeline.output_states[node.id] is state
    assert state.to_dict() == original_state
    np.testing.assert_array_equal(output, original_data)


@pytest.mark.parametrize("operation", RACC_OPERATIONS)
@pytest.mark.parametrize(
    ("color", "source_colormap"), ((None, "gray"), (0x0000FF, "blue")),
)
def test_bypassed_racc_keeps_source_palette_but_shadow_thumbnail_is_magma(
    qtbot, operation, color, source_colormap,
):
    widget, node, output, state = _widget(qtbot, operation, color=color)
    widget.inspect_node(node.id)
    assert widget.viewer.layers["VIPP Inspect"].colormap == "magma"
    assert widget.pipeline.connect("input", node.id).success
    assert widget.pipeline.set_node_execution_mode(node.id, "bypass")
    shadow = 1 - output
    shadow.setflags(write=False)
    widget._bypass_shadow_results[node.id] = PipelinePresentationShadowResult(
        run_id=1,
        node_id=node.id,
        operation_id=operation,
        output=shadow,
        output_state=state,
        node_outputs=(shadow,),
        node_output_states=(state,),
    )

    _assert_thumbnail(widget, node.id, shadow, "Magma")
    widget.inspect_node(node.id)
    widget._set_active_pin_layer(node.id, output)
    for layer in (
        widget.viewer.layers["VIPP Inspect"], *widget._active_pinned_layers(),
    ):
        assert layer.colormap == source_colormap
        assert np.shares_memory(layer.data, output)
        np.testing.assert_array_equal(layer.data, output)
    assert widget.pipeline.outputs[node.id] is output
    assert widget.pipeline.output_states[node.id] is state
