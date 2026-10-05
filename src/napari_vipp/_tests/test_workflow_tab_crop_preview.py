"""Real napari tab switches must retire outgoing presentation-only crop layers."""

import numpy as np
import pytest
from napari.components import ViewerModel
from qtpy.QtWidgets import QApplication

from napari_vipp._tests.test_workflow_tabs import _editor_snapshot
from napari_vipp._widget import CROP_ROI_LAYER_NAME, VippWidget
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.pipeline import EXECUTION_READY
from napari_vipp.ui.napari_compat import viewer_camera


def _publish(pipeline, node_id, data, state):
    pipeline.outputs[node_id] = data
    pipeline.node_outputs[node_id] = [data]
    pipeline.output_states[node_id] = state
    pipeline.node_output_states[node_id] = [state]
    pipeline.completed_node_ids.add(node_id)
    pipeline.node_execution_states[node_id] = EXECUTION_READY


def _state(data, *, translate=(3.0, 5.0, 7.0)):
    rank = data.ndim
    return image_state_from_array(
        data,
        axes=tuple(
            AxisMetadata(axis, "space", scale=scale, translation=offset, unit="um")
            for axis, scale, offset in zip(
                ("z", "y", "x")[-rank:], (2.0, 0.5, 0.25)[-rank:],
                translate[-rank:], strict=True,
            )
        ),
    )


def _switch(qtbot, widget, session, *, settle=True):
    qtbot.waitUntil(lambda: not widget._workflow_tab_switch_block_reason())
    index = widget._workflow_tabs.index_of(session.session_id)
    widget.workflow_tab_bar.setCurrentIndex(index)
    assert widget._workflow_tabs.current is session
    if settle:
        QApplication.processEvents()
        qtbot.waitUntil(lambda: (
            widget._active_viewer_layer() is not None
            and getattr(widget, "_pending_workflow_tab_view_restore", None) is None
        ))


def _small_and_large_tabs(
    qtbot, monkeypatch, *, rank=3, small_camera=None, both_crops=False,
):
    viewer = ViewerModel()
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    runs = []
    monkeypatch.setattr(widget, "run_pipeline", lambda *a, **k: runs.append((a, k)))
    widget.pipeline.reset_empty_graph()
    small_shape = (4, 8, 12)[-rank:]
    small = np.arange(np.prod(small_shape), dtype=np.float32).reshape(small_shape)
    small.setflags(write=False)
    _publish(widget.pipeline, "input", small, _state(small))
    selected = "input"
    if both_crops:
        small_crop = widget.pipeline.add_node("crop_stack")
        assert widget.pipeline.connect("input", small_crop.id).success
        _publish(widget.pipeline, small_crop.id, small, _state(small))
        selected = small_crop.id
    widget.graph_view.build_graph(
        widget.pipeline.nodes.values(), widget.pipeline.connections,
    )
    widget.graph_view.select_node(selected)
    QApplication.processEvents()
    if small_camera is not None:
        viewer_camera(viewer).center, viewer_camera(viewer).zoom = small_camera
    first = widget._workflow_tabs.current
    first.mark_clean(widget._current_history_snapshot())

    second = widget._workflow_tabs.create_blank(make_current=False)
    large_shape = (16, 64, 96)[-rank:]
    large = np.arange(np.prod(large_shape), dtype=np.float32).reshape(large_shape)
    large.setflags(write=False)
    _publish(second.pipeline, "input", large, _state(large))
    crop = second.pipeline.add_node("crop_stack")
    assert second.pipeline.connect("input", crop.id).success
    crop.params.update(top=26, bottom=30, left=42, right=42)
    if rank == 3:
        crop.params.update(z_start=5, z_end=7)
    cropped = large[(slice(5, 9), slice(26, 34), slice(42, 54))[-rank:]].copy()
    cropped.setflags(write=False)
    _publish(
        second.pipeline,
        crop.id,
        cropped,
        _state(cropped, translate=(13.0, 18.0, 17.5)),
    )
    second.mark_clean(_editor_snapshot(second.pipeline, selected=crop.id))
    widget.workflow_tab_bar.sync_from_model(widget._workflow_tabs)
    _switch(qtbot, widget, second)
    qtbot.waitUntil(lambda: len(widget._owned_crop_presentation_layers()) == 2)
    return widget, viewer, first, second, small, large, runs


@pytest.mark.parametrize("rank,ndisplay", ((2, 2), (3, 2), (3, 3)))
@pytest.mark.parametrize("hide_roi", (False, True))
def test_tab_switch_retires_crop_source_and_roi_before_installing_small_workflow(
    qtbot, monkeypatch, rank, ndisplay, hide_roi,
):
    widget, viewer, small_tab, large_tab, small, large, runs = _small_and_large_tabs(
        qtbot, monkeypatch, rank=rank,
    )
    if rank == 3:
        viewer.dims.set_current_step(0, 15)
    viewer.dims.ndisplay = ndisplay
    outgoing = widget._owned_crop_presentation_layers()
    if hide_roi:
        widget._owned_crop_presentation_layers("crop_roi")[0].visible = False
    assert {layer.metadata["napari_vipp_kind"] for layer in outgoing} == {
        "crop_source", "crop_roi",
    }
    assert all(
        layer.metadata["session_id"] == large_tab.session_id for layer in outgoing
    )
    small_before, large_before = small.copy(), large.copy()

    _switch(qtbot, widget, small_tab, settle=False)

    assert not widget._owned_crop_presentation_layers(), [
        (layer.name, layer.visible, layer.metadata)
        for layer in widget._owned_crop_presentation_layers()
    ]
    assert all(layer not in viewer.layers for layer in outgoing)
    QApplication.processEvents()
    assert not widget._owned_crop_presentation_layers()
    assert tuple(viewer.dims.nsteps) == small.shape
    assert widget._active_viewer_layer().data.shape == small.shape
    assert small_tab.pipeline.outputs["input"] is small
    assert large_tab.pipeline.outputs["input"] is large
    np.testing.assert_array_equal(small, small_before)
    np.testing.assert_array_equal(large, large_before)
    assert not small.flags.writeable and not large.flags.writeable
    for _repeat in range(2):
        _switch(qtbot, widget, large_tab)
        assert len(widget._owned_crop_presentation_layers()) == 2
        _switch(qtbot, widget, small_tab)
        assert not widget._owned_crop_presentation_layers()
        assert tuple(viewer.dims.nsteps) == small.shape
        origin = np.array((3, 5, 7)[-rank:])
        scale = np.array((2, 0.5, 0.25)[-rank:])
        np.testing.assert_allclose(viewer.layers.extent.world, np.stack((
            origin, origin + (np.array(small.shape) - 1) * scale,
        )))
    assert runs == []


def test_switching_workflow_tabs_restores_each_retained_camera(qtbot, monkeypatch):
    widget, viewer, small_tab, large_tab, *_arrays, runs = _small_and_large_tabs(
        qtbot, monkeypatch,
    )
    camera = viewer_camera(viewer)
    _switch(qtbot, widget, small_tab)
    viewer.dims.ndisplay = 2
    viewer.dims.order = (0, 2, 1)
    viewer.dims.point = (7.0, 6.0, 8.0)
    camera.center, camera.zoom = (5.0, 6.0, 8.0), 3.25
    camera.angles, camera.perspective = (10.0, 20.0, 30.0), 12.0
    small_view = {
        "ndisplay": 2, "order": (0, 2, 1), "point": (7.0, 6.0, 8.0),
        "center": (5.0, 6.0, 8.0), "zoom": 3.25,
        "angles": (10.0, 20.0, 30.0), "perspective": 12.0,
    }
    _switch(qtbot, widget, large_tab)
    viewer.dims.ndisplay = 3
    viewer.dims.order = (2, 0, 1)
    viewer.dims.point = (23.0, 17.0, 19.0)
    camera.center, camera.zoom = (13.0, 18.0, 23.0), 1.75
    camera.angles, camera.perspective = (35.0, 45.0, 55.0), 28.0
    large_view = {
        "ndisplay": 3, "order": (2, 0, 1), "point": (23.0, 17.0, 19.0),
        "center": (13.0, 18.0, 23.0), "zoom": 1.75,
        "angles": (35.0, 45.0, 55.0), "perspective": 28.0,
    }

    for session, expected in ((small_tab, small_view), (large_tab, large_view)) * 2:
        _switch(qtbot, widget, session)
        assert viewer.dims.ndisplay == expected["ndisplay"]
        assert viewer.dims.order == expected["order"]
        np.testing.assert_allclose(viewer.dims.point, expected["point"])
        np.testing.assert_allclose(camera.center, expected["center"])
        assert camera.zoom == pytest.approx(expected["zoom"])
        np.testing.assert_allclose(camera.angles, expected["angles"])
        assert camera.perspective == pytest.approx(expected["perspective"])

    widget._schedule_selected_viewer_refresh(
        widget._selected_node_id, select_layer=True,
    )
    QApplication.processEvents()
    QApplication.processEvents()
    assert widget._pending_workflow_tab_view_restore is None
    np.testing.assert_allclose(camera.center, large_view["center"])
    assert camera.zoom == pytest.approx(large_view["zoom"])
    np.testing.assert_allclose(camera.angles, large_view["angles"])
    assert camera.perspective == pytest.approx(large_view["perspective"])
    assert runs == []


def test_synchronous_dims_notification_cannot_recreate_retiring_crop_roi(
    qtbot, monkeypatch,
):
    widget, viewer, small_tab, _large_tab, *_arrays, runs = _small_and_large_tabs(
        qtbot, monkeypatch,
    )
    original_remove = widget._remove_layer
    injected = []

    def remove_with_notification(layer):
        original_remove(layer)
        if layer.metadata.get("napari_vipp_kind") == "crop_roi" and not injected:
            # Exercise the allowed synchronous callback ordering explicitly.
            # Ordinary ViewerModel removal cases are tested above without this.
            injected.append(layer)
            viewer.dims.events.point()

    monkeypatch.setattr(widget, "_remove_layer", remove_with_notification)

    _switch(qtbot, widget, small_tab, settle=False)

    assert len(injected) == 1
    assert not widget._owned_crop_presentation_layers()
    QApplication.processEvents()
    assert not widget._owned_crop_presentation_layers()
    assert runs == []


@pytest.mark.parametrize("rank", (2, 3))
def test_same_crop_node_id_in_two_tabs_uses_only_incoming_source_and_roi(
    qtbot, monkeypatch, rank,
):
    widget, viewer, small_tab, large_tab, small, large, runs = _small_and_large_tabs(
        qtbot, monkeypatch, rank=rank, both_crops=True,
    )
    small_crop = next(node for node in small_tab.pipeline.nodes.values()
                      if node.operation_id == "crop_stack")
    large_crop = next(node for node in large_tab.pipeline.nodes.values()
                      if node.operation_id == "crop_stack")
    assert small_crop.id == large_crop.id

    for session, source in ((small_tab, small), (large_tab, large)) * 2:
        outgoing = widget._owned_crop_presentation_layers()
        widget._schedule_selected_viewer_refresh(
            widget._selected_node_id, select_layer=True,
        )
        _switch(qtbot, widget, session)
        owned = widget._owned_crop_presentation_layers()
        assert len(owned) == 2
        assert all(layer not in viewer.layers for layer in outgoing)
        assert all(
            layer.metadata["session_id"] == session.session_id for layer in owned
        )
        assert widget._owned_crop_presentation_layers("crop_source")[0].data is source
        assert tuple(widget._active_viewer_layer().data.shape) == source.shape
    assert runs == []


def test_tab_switch_preserves_user_shapes_with_crop_roi_name(qtbot, monkeypatch):
    widget, viewer, small_tab, _large_tab, *_arrays, runs = _small_and_large_tabs(
        qtbot, monkeypatch,
    )
    polygon = np.array(((5, 6, 8), (5, 6, 9), (5, 7, 9), (5, 7, 8)), dtype=float)
    user_roi = viewer.add_shapes(
        [polygon], name=CROP_ROI_LAYER_NAME, shape_type="polygon",
        metadata={"owner": "user"},
    )
    user_roi.visible = False

    _switch(qtbot, widget, small_tab)

    assert user_roi in viewer.layers
    assert user_roi.metadata == {"owner": "user"}
    assert not user_roi.visible
    np.testing.assert_array_equal(user_roi.data[0], polygon)
    assert not widget._owned_crop_presentation_layers()
    assert runs == []


def test_queued_outgoing_crop_refresh_cannot_publish_after_tab_switch(
    qtbot, monkeypatch,
):
    widget, _viewer, small_tab, large_tab, small, _large, runs = _small_and_large_tabs(
        qtbot, monkeypatch,
    )
    selected = widget._selected_node_id
    widget._schedule_selected_viewer_refresh(selected, select_layer=True)
    _switch(qtbot, widget, small_tab, settle=False)
    # Leave both old and new queued viewer callbacks to the real event loop.
    QApplication.processEvents()
    QApplication.processEvents()

    assert widget._workflow_tabs.current is small_tab
    assert widget._workflow_tabs.current is not large_tab
    assert widget._selected_node_id == "input"
    assert not widget._owned_crop_presentation_layers()
    assert widget._active_viewer_layer().data.shape == small.shape
    assert runs == []


def test_first_visit_frames_selected_small_source_not_hidden_large_user_image(
    qtbot, monkeypatch,
):
    widget, viewer, _small_tab, _large_tab, *_arrays, runs = _small_and_large_tabs(
        qtbot, monkeypatch,
    )
    hidden_data = np.zeros((20, 128, 128), dtype=np.uint8)
    hidden = viewer.add_image(
        hidden_data, name="User image kept hidden", visible=False,
        scale=(4.0, 2.0, 3.0), translate=(-100.0, -100.0, -100.0),
    )
    target = widget._workflow_tabs.create_blank(make_current=False)
    small = np.ones((3, 7, 11), dtype=np.float32)
    small.setflags(write=False)
    state = image_state_from_array(small, axes=(
        AxisMetadata("z", "space", scale=3.0, translation=10.0, unit="um"),
        AxisMetadata("y", "space", scale=0.4, translation=20.0, unit="um"),
        AxisMetadata("x", "space", scale=0.2, translation=30.0, unit="um"),
    ))
    _publish(target.pipeline, "input", small, state)
    target.mark_clean(_editor_snapshot(target.pipeline, selected="input"))
    widget.workflow_tab_bar.sync_from_model(widget._workflow_tabs)

    _switch(qtbot, widget, target)

    camera = viewer_camera(viewer)
    np.testing.assert_allclose(camera.center, (0.0, 21.2, 31.0))
    assert camera.zoom > 10.0  # selected 2.8-by-2.2 world-unit footprint
    assert 10.0 <= viewer.dims.point[0] <= 16.0
    assert hidden in viewer.layers and not hidden.visible
    assert hidden.data is hidden_data
    assert target.pipeline.outputs["input"] is small
    assert widget._active_viewer_layer().data.shape == small.shape
    assert runs == []


def test_closing_current_crop_tab_retires_layers_and_restores_remaining_view(
    qtbot, monkeypatch,
):
    expected_camera = ((5.0, 6.0, 8.0), 3.25)
    widget, viewer, small_tab, large_tab, small, _large, runs = _small_and_large_tabs(
        qtbot, monkeypatch, small_camera=expected_camera,
    )
    outgoing = widget._owned_crop_presentation_layers()
    widget._schedule_selected_viewer_refresh(
        widget._selected_node_id, select_layer=True,
    )
    large_tab.mark_clean(
        widget._current_history_snapshot(),
        persistence_token=widget._workflow_tab_persistence_token(),
    )
    qtbot.waitUntil(lambda: not widget._workflow_tab_switch_block_reason())
    index = widget._workflow_tabs.index_of(large_tab.session_id)

    widget.workflow_tab_bar.tabCloseRequested.emit(index)
    QApplication.processEvents()
    qtbot.waitUntil(lambda: widget._pending_workflow_tab_view_restore is None)

    assert len(widget._workflow_tabs) == 1
    assert widget._workflow_tabs.current is small_tab
    assert all(layer not in viewer.layers for layer in outgoing)
    assert not widget._owned_crop_presentation_layers()
    assert tuple(viewer.dims.nsteps) == small.shape
    np.testing.assert_allclose(viewer_camera(viewer).center, expected_camera[0])
    assert viewer_camera(viewer).zoom == pytest.approx(expected_camera[1])
    assert small_tab.pipeline.outputs["input"] is small
    assert runs == []


def test_restored_slice_is_clamped_to_changed_source_not_hidden_global_extent(
    qtbot, monkeypatch,
):
    widget, viewer, small_tab, large_tab, small, _large, runs = _small_and_large_tabs(
        qtbot, monkeypatch,
    )
    _switch(qtbot, widget, small_tab)
    viewer.dims.set_current_step(0, 3)
    assert viewer.dims.point[0] == pytest.approx(9.0)
    _switch(qtbot, widget, large_tab)
    viewer.add_image(
        np.zeros((20, 64, 96), dtype=np.uint8), name="Hidden global extent",
        visible=False, scale=(4.0, 2.0, 3.0),
    )
    reduced = small[:2].copy()
    reduced.setflags(write=False)
    _publish(small_tab.pipeline, "input", reduced, _state(reduced))

    _switch(qtbot, widget, small_tab)

    assert viewer.layers.extent.world[1, 0] > 9.0
    assert viewer.dims.point[0] == pytest.approx(5.0)
    assert widget._active_viewer_layer().data.shape == reduced.shape
    assert small_tab.pipeline.outputs["input"] is reduced
    assert not small.flags.writeable and not reduced.flags.writeable
    assert runs == []
