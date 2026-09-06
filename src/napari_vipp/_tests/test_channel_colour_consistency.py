from __future__ import annotations

from dataclasses import replace

import numpy as np
from napari.components import ViewerModel

from napari_vipp._tests.test_widget import _Viewer
from napari_vipp._widget import VippWidget, _histogram_summary
from napari_vipp.core.metadata import (
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
)
from napari_vipp.core.workflow import deserialize_workflow, serialize_workflow
from napari_vipp.ui.plots import _histogram_series_colors


def _qcolor_rgb(color) -> tuple[float, float, float]:
    red, green, blue, _alpha = color.getRgbF()
    return float(red), float(green), float(blue)


def _layer_colormap_name(layer) -> str:
    colormap = getattr(layer, "colormap", None)
    return str(getattr(colormap, "name", colormap)).casefold()


def _inspect_layers(viewer: ViewerModel) -> list:
    return [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "inspect"
    ]


def _generated_channel_layers(viewer, kind: str) -> list:
    return [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == kind
        and layer.metadata.get("display_channel_axis_as_layers") is True
    ]


def _assert_red_green(colors) -> None:
    red, green = (_qcolor_rgb(color) for color in colors)
    assert red[0] > red[1] and red[0] > red[2]
    assert green[1] > green[0] and green[1] > green[2]


def _assert_red_cyan(colors) -> None:
    red, cyan = (_qcolor_rgb(color) for color in colors)
    assert red[0] > red[1] and red[0] > red[2]
    assert cyan[1] > cyan[0] and cyan[2] > cyan[0]


def test_histogram_series_follow_image_state_channel_colours():
    data = np.zeros((2, 6, 7), dtype=np.uint16)
    data[0] = 10
    data[1] = 20
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
        channels=(
            ChannelMetadata(name="Channel 1", color=0xFF0000),
            ChannelMetadata(name="Channel 2", color=0x00FF00),
        ),
    )

    counts, _x_range, colors = _histogram_summary(
        data,
        state=state,
        scope="Stack",
    )

    assert counts is not None
    assert counts.shape[0] == 2
    assert colors is not None
    np.testing.assert_allclose(_qcolor_rgb(colors[0]), (1.0, 0.0, 0.0), atol=1 / 255)
    np.testing.assert_allclose(_qcolor_rgb(colors[1]), (0.0, 1.0, 0.0), atol=1 / 255)


def test_rgba_histogram_defaults_to_rgb_with_neutral_alpha():
    colors = _histogram_series_colors(4, "rgba")
    rgb = np.asarray([_qcolor_rgb(color) for color in colors])

    assert rgb.shape == (4, 3)
    assert rgb[0, 0] > rgb[0, 1] and rgb[0, 0] > rgb[0, 2]
    assert rgb[1, 1] > rgb[1, 0] and rgb[1, 1] > rgb[1, 2]
    assert rgb[2, 2] > rgb[2, 0] and rgb[2, 2] > rgb[2, 1]
    assert np.ptp(rgb[3]) < 0.05


def test_channel_axis_inspection_uses_selected_additive_colours(qtbot):
    source = np.zeros((6, 7), dtype=np.uint16)
    viewer = ViewerModel()
    viewer.add_image(source, name="Source", metadata={"axes": "YX"})
    widget = VippWidget(viewer, defer_initial_run=True)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    qtbot.addWidget(widget)

    data = np.zeros((2, 6, 7), dtype=np.uint16)
    data[0] = 11
    data[1] = 29
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
        channels=(
            ChannelMetadata(name="Red signal", color=0xFF0000),
            ChannelMetadata(name="Green signal", color=0x00FF00),
        ),
    )

    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        data,
        metadata={
            "napari_vipp_kind": "inspect",
            "vipp_image_state": state.to_dict(),
        },
        role="inspect",
    )

    layers = _inspect_layers(viewer)
    assert len(layers) == 2
    assert {_layer_colormap_name(layer) for layer in layers} == {"red", "green"}
    assert all(layer.blending == "additive" for layer in layers)
    assert all(tuple(layer.axis_labels) == ("Y", "X") for layer in layers)
    assert {int(np.asarray(layer.data)[0, 0]) for layer in layers} == {11, 29}


def test_true_rgb_inspection_remains_one_rgb_layer(qtbot):
    source = np.zeros((6, 7), dtype=np.uint16)
    viewer = ViewerModel()
    viewer.add_image(source, name="Source", metadata={"axes": "YX"})
    widget = VippWidget(viewer, defer_initial_run=True)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    qtbot.addWidget(widget)

    data = np.zeros((6, 7, 3), dtype=np.uint8)
    data[..., 0] = 17
    data[..., 1] = 31
    data[..., 2] = 47
    state = replace(
        image_state_from_array(
            data,
            axes=(
                AxisMetadata("y", "space"),
                AxisMetadata("x", "space"),
                AxisMetadata("rgb", "channel"),
            ),
        ),
        kind="RGB image",
    )
    widget.pipeline.outputs["input"] = data
    widget.pipeline.output_states["input"] = state
    widget.pipeline.node_outputs["input"] = [data]
    widget.pipeline.node_output_states["input"] = [state]

    widget._set_or_add_generated_layer(
        "VIPP Inspect",
        data,
        metadata={
            "napari_vipp_kind": "inspect",
            "node_id": "input",
            "output_port": 0,
            "vipp_image_state": state.to_dict(),
        },
        role="inspect",
    )

    layers = _inspect_layers(viewer)
    assert len(layers) == 1
    assert layers[0].rgb is True
    assert tuple(layers[0].axis_labels) == ("Y", "X")
    np.testing.assert_array_equal(np.asarray(layers[0].data), data)


def test_combine_channels_colour_edit_repaints_cached_presentations(
    qtbot,
    monkeypatch,
):
    data = np.zeros((2, 2, 3, 4, 5), dtype=np.uint16)
    data[:, 0] = 11
    data[:, 1] = 29
    viewer = _Viewer(data, metadata={"axes": "TCZYX"})
    widget = VippWidget(viewer)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    qtbot.addWidget(widget)

    first = widget.add_node_from_palette("extract_channel")
    second = widget.add_node_from_palette("extract_channel")
    combined = widget.add_node_from_palette("combine_channels")
    widget._connect_nodes("input", first.id)
    widget._connect_nodes("input", second.id)
    widget.pipeline.set_param(first.id, "channel", 0)
    widget.pipeline.set_param(second.id, "channel", 1)
    widget._connect_nodes(first.id, combined.id, target_port=0)
    widget._connect_nodes(second.id, combined.id, target_port=1)
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)
    widget._select_node(combined.id)
    qtbot.waitUntil(
        lambda: widget.graph_view.node_has_thumbnail(combined.id),
        timeout=5_000,
    )

    output = widget.pipeline.outputs[combined.id]
    output_before = np.asarray(output).copy()
    assert widget.pipeline.nodes[combined.id].params["channel_colors"] == "Red,Green"
    _assert_red_green(widget.histogram_plot._series_colors)

    widget.inspect_node(combined.id)
    widget.pin_node(combined.id)
    for kind in ("inspect", "pinned"):
        layers = _generated_channel_layers(viewer, kind)
        assert len(layers) == 2
        assert {str(layer.colormap) for layer in layers} == {"red", "green"}
        assert all(layer.blending == "additive" for layer in layers)

    # Populate the background-result cache, then require the colour-only edit
    # to reuse those numerical counts instead of scanning the image again.
    widget._selection_diagnostics_initializing = True
    try:
        widget._update_histogram()
    finally:
        widget._selection_diagnostics_initializing = False
    qtbot.waitUntil(
        lambda: (
            widget._active_output_histogram_run_id is None
            and bool(widget._output_histogram_cache)
        ),
        timeout=5_000,
    )
    cached_counts = widget.histogram_plot._series_counts.copy()

    thumbnail_palettes: list[list[str]] = []

    def record_preview(
        data,
        mode,
        current_step,
        current_step_nsteps=None,
        state=None,
        channel_colors=None,
        contrast_mode="Percentile",
        contrast_scope="Slice",
        contrast_limits=None,
        preview_size=None,
    ):
        del (
            data,
            mode,
            current_step,
            current_step_nsteps,
            state,
            contrast_mode,
            contrast_scope,
            contrast_limits,
            preview_size,
        )
        if channel_colors is not None:
            thumbnail_palettes.append(list(channel_colors))
        return None

    monkeypatch.setattr("napari_vipp._widget.make_preview", record_preview)

    def unexpected_histogram_scan(*_args, **_kwargs):
        raise AssertionError("a colour-only edit rescanned unchanged image data")

    monkeypatch.setattr(
        "napari_vipp._widget._histogram_summary",
        unexpected_histogram_scan,
    )
    widget._selection_diagnostics_initializing = True
    try:
        widget._on_channel_color_changed(1, "Cyan")
    finally:
        widget._selection_diagnostics_initializing = False
        widget._debounce_timer.stop()

    assert ["Red", "Cyan"] in thumbnail_palettes
    assert widget.pipeline.outputs[combined.id] is output
    np.testing.assert_array_equal(np.asarray(output), output_before)
    np.testing.assert_array_equal(
        widget.histogram_plot._series_counts,
        cached_counts,
    )
    _assert_red_cyan(widget.histogram_plot._series_colors)
    for kind in ("inspect", "pinned"):
        layers = _generated_channel_layers(viewer, kind)
        assert len(layers) == 2
        assert {str(layer.colormap) for layer in layers} == {"red", "cyan"}
        assert all(layer.blending == "additive" for layer in layers)

    metadata = widget._workflow_metadata()
    document = serialize_workflow(widget.pipeline, metadata=metadata)
    restored = deserialize_workflow(document)
    profiles = restored["metadata"]["vipp"]["inspector"]["display_profiles"]
    combine_profiles = [
        profile for profile in profiles if profile["node_id"] == combined.id
    ]
    assert {profile["display_channel_index"] for profile in combine_profiles} == {
        0,
        1,
    }
