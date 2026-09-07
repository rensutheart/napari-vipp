"""Bypass refreshes must not restore image placeholders on mesh/table cards."""

from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.execution import PipelinePresentationShadowResult
from napari_vipp.core.meshes import MeshData, MeshState
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.tables import TableData, TableState


def _bypassed_widget(qtbot, source_operation, operation):
    from napari.components import ViewerModel

    from napari_vipp._widget import VippWidget

    viewer = ViewerModel()
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    widget.thumbnail_scope_combo.setCurrentText("Slice")
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    source = (
        pipeline.nodes["input"]
        if source_operation == "input"
        else pipeline.add_node(source_operation)
    )
    node = pipeline.add_node(operation)
    assert pipeline.connect(source.id, node.id).success
    assert pipeline.set_node_execution_mode(node.id, "bypass")
    widget.pipeline = pipeline
    widget._build_graph_from_pipeline()
    return widget, node


def _set_output(widget, node_id, data, state):
    widget.pipeline.outputs[node_id] = data
    widget.pipeline.output_states[node_id] = state
    widget.pipeline.node_outputs[node_id] = [data]
    widget.pipeline.node_output_states[node_id] = [state]


@pytest.mark.parametrize("kind", ["mesh", "table"])
def test_non_image_bypass_refresh_keeps_preview_hidden(qtbot, kind):
    if kind == "mesh":
        widget, node = _bypassed_widget(
            qtbot, "mask_to_3d_mesh", "split_mesh_objects"
        )
        state = MeshState(
            3, 1, tuple(AxisMetadata(n, "space") for n in "zyx"), (2, 2, 2), "test"
        )
        data = MeshData(
            np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]]),
            np.array([[0, 1, 2]]),
            state,
        )
        shadow_data = replace(data, state=replace(state, source_name="what-if"))
        shadow_state = shadow_data.state
    else:
        widget, node = _bypassed_widget(
            qtbot, "measure_objects", "add_metadata_columns"
        )
        data = TableData(("label",), ((1,),), name="measurements")
        state = TableState(1, 1, ("label",), source_name="measurements")
        shadow_data = TableData(("label",), ((2,),), name="what-if")
        shadow_state = replace(state, source_name="what-if")

    card = widget.graph_view._cards[node.id]
    assert card.preview.isHidden()
    for status in ("empty", "ready", "pending", "error", "success"):
        _set_output(
            widget,
            node.id,
            None if status == "empty" else data,
            None if status == "empty" else state,
        )
        widget._bypass_shadow_pending_node_ids.clear()
        widget._bypass_shadow_errors.clear()
        widget._bypass_shadow_results.clear()
        if status == "pending":
            widget._bypass_shadow_pending_node_ids.add(node.id)
        elif status == "error":
            widget._bypass_shadow_errors[node.id] = "old preview failure"
        elif status == "success":
            # Defensive refresh of a cached result from an older worker/session.
            widget._bypass_shadow_results[node.id] = PipelinePresentationShadowResult(
                run_id=1,
                node_id=node.id,
                operation_id=node.operation_id,
                output=shadow_data,
                output_state=shadow_state,
                node_outputs=(shadow_data,),
                node_output_states=(shadow_state,),
            )

        widget._update_thumbnails()

        assert card.preview.isHidden(), status
        assert not card.preview.has_source_pixmap(), status
        assert card.preview.text() == "", status
        if status != "empty":
            assert widget.pipeline.outputs[node.id] is data
            assert widget.pipeline.output_states[node.id] is state

    assert not widget._node_allows_bypass_presentation_shadow(node.id)
    assert widget._node_thumbnail_display_payload(node.id)[0] is data
    assert widget._presentation_shadow_data_kind(
        node.id, shadow_data, shadow_state, 0
    ) == kind


def test_image_bypass_refresh_keeps_what_if_thumbnail(qtbot):
    widget, node = _bypassed_widget(qtbot, "input", "gaussian_blur")
    data = np.arange(64, dtype=np.float32).reshape(8, 8)
    state = image_state_from_array(data, layer_metadata={"axes": "YX"})
    shadow_data = data + 1
    _set_output(widget, node.id, data, state)
    widget._bypass_shadow_results[node.id] = PipelinePresentationShadowResult(
        run_id=1,
        node_id=node.id,
        operation_id=node.operation_id,
        output=shadow_data,
        output_state=state,
        node_outputs=(shadow_data,),
        node_output_states=(state,),
    )

    widget._update_thumbnails()

    card = widget.graph_view._cards[node.id]
    assert widget._node_allows_bypass_presentation_shadow(node.id)
    assert widget._node_thumbnail_display_payload(node.id)[0] is shadow_data
    assert not card.preview.isHidden()
    assert card.preview.has_source_pixmap()
    assert widget.pipeline.outputs[node.id] is data
    assert widget.pipeline.output_states[node.id] is state
