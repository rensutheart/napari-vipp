from __future__ import annotations

import numpy as np
from napari.components import ViewerModel

from napari_vipp._widget import VippWidget
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import SourcePayload


def _selected_inspect_layer(widget: VippWidget):
    return next(
        layer
        for layer in widget.viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "inspect"
        and layer.metadata.get("node_id") == widget._selected_node_id
    )


def test_leaving_tczyx_crop_for_yx_projection_removes_only_crop_surfaces(qtbot):
    source_data = np.zeros((2, 3, 4, 16, 18), dtype=np.uint16)
    source_state = image_state_from_array(
        source_data,
        axes=(
            AxisMetadata("t", "time"),
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
    )
    viewer = ViewerModel()
    annotation_data = np.zeros(source_data.shape[-2:], dtype=np.uint16)
    annotation = viewer.add_labels(
        annotation_data,
        name="Research annotation",
        metadata={"owner": "user", "axes": "YX"},
    )
    widget = VippWidget(viewer, defer_initial_run=True)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    qtbot.addWidget(widget)
    widget._sample_payload_cache = {
        "TCZYX regression sample": SourcePayload(
            source_data,
            {"vipp_image_state": source_state.to_dict()},
            "TCZYX regression sample",
            source_state,
        )
    }
    widget.pipeline.nodes["input"].params.update(
        source_mode="sample",
        sample_name="TCZYX regression sample",
    )

    crop = widget.add_node_from_palette("crop_stack")
    selector = widget.add_node_from_palette("select_axis_slice")
    projection = widget.add_node_from_palette("mip")
    widget.pipeline.nodes[selector.id].params.update(
        range_mode=True,
        remove_axes="0,1",
        remove_indices="0,0",
    )
    widget.pipeline.set_param(projection.id, "axis", 0)
    widget._connect_nodes("input", crop.id)
    widget._connect_nodes("input", selector.id)
    widget._connect_nodes(selector.id, projection.id)
    widget.run_pipeline(force_sync=True)

    assert widget.pipeline.output_states[crop.id].axis_order == "TCZYX"
    assert widget.pipeline.output_states[projection.id].axis_order == "YX"
    crop_output = widget.pipeline.outputs[crop.id]

    widget.graph_view.select_node(crop.id)
    qtbot.waitUntil(
        lambda: bool(widget._owned_crop_presentation_layers("crop_source"))
        and bool(widget._owned_crop_presentation_layers("crop_roi")),
        timeout=5_000,
    )
    first_crop_layers = tuple(widget._owned_crop_presentation_layers())
    assert tuple(viewer.dims.axis_labels) == ("T", "C", "Z", "Y", "X")
    # The ROI shapes layer reaches the far pixel boundary and can extend the
    # spatial world range by one. Its non-spatial T/C/Z extents remain exact.
    assert tuple(viewer.dims.nsteps[:3]) == source_data.shape[:3]

    widget.graph_view.select_node(projection.id)
    # Graph selection publishes its napari presentation after the originating
    # mouse/input event returns. The stale Crop layers may still be present
    # here, but must not survive the deferred refresh.
    qtbot.waitUntil(
        lambda: not widget._owned_crop_presentation_layers()
        and _selected_inspect_layer(widget).data.ndim == 2,
        timeout=5_000,
    )

    inspect = _selected_inspect_layer(widget)
    assert tuple(inspect.axis_labels) == ("Y", "X")
    assert viewer.dims.ndim == 2
    assert tuple(viewer.dims.axis_labels) == ("Y", "X")
    assert tuple(viewer.dims.nsteps) == source_data.shape[-2:]
    assert annotation in viewer.layers
    assert annotation.data is annotation_data
    assert widget.pipeline.outputs[crop.id] is crop_output

    widget.graph_view.select_node(crop.id)
    qtbot.waitUntil(
        lambda: bool(widget._owned_crop_presentation_layers("crop_source"))
        and bool(widget._owned_crop_presentation_layers("crop_roi")),
        timeout=5_000,
    )

    recreated_crop_layers = tuple(widget._owned_crop_presentation_layers())
    assert recreated_crop_layers
    assert not any(layer in recreated_crop_layers for layer in first_crop_layers)
    assert viewer.dims.ndim == 5
    assert tuple(viewer.dims.axis_labels) == ("T", "C", "Z", "Y", "X")
    assert tuple(viewer.dims.nsteps[:3]) == source_data.shape[:3]
    assert annotation in viewer.layers
    assert annotation.data is annotation_data
    assert widget.pipeline.outputs[crop.id] is crop_output
