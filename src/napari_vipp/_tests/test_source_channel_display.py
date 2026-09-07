"""Image Source display choices never change scientific channels or pixels."""

from __future__ import annotations

import json
from copy import deepcopy

import numpy as np
import pytest
from napari.components import ViewerModel

from napari_vipp._widget import VippWidget
from napari_vipp.core.metadata import ChannelMetadata, image_state_from_array
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import load_workflow, save_workflow


def _widget(qtbot, *, axes="ZCYX", colors=True):
    shape = tuple({"Z": 5, "C": 2, "T": 3, "Y": 8, "X": 9}[a] for a in axes)
    data = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape)
    state = image_state_from_array(
        data,
        layer_metadata={"axes": axes},
        channels=(
            ChannelMetadata(name="CH1", color=0x0000FF if colors else None),
            ChannelMetadata(name="CH3", color=0x00FF00 if colors else None),
        )
        if "C" in axes
        else (),
    )
    data.setflags(write=False)
    viewer = ViewerModel()
    viewer.add_image(
        data,
        rgb=False,
        name="OIR-style source",
        metadata={
            "vipp_image_state": state.to_dict(),
        },
    )
    widget = VippWidget(viewer, defer_initial_run=True)
    qtbot.addWidget(widget)
    widget._should_run_pipeline_in_background = lambda *_args, **_kwargs: False
    widget.run_pipeline(force_sync=True)
    widget.graph_view.select_node("input")
    return widget, viewer, data


@pytest.mark.parametrize("axes", ["CYX", "ZCYX", "TCZYX"])
@pytest.mark.parametrize("colors", [False, True])
@pytest.mark.parametrize("ndisplay", [2, 3])
def test_source_switches_immediately_without_scientific_changes(
    qtbot,
    monkeypatch,
    axes,
    colors,
    ndisplay,
):
    widget, viewer, data = _widget(qtbot, axes=axes, colors=colors)
    viewer.dims.ndisplay = ndisplay
    control = widget._parameter_widgets["source_channel_display"]
    assert control.value() == "Stack (C slider)"
    assert control.isEnabled()
    assert not widget._colored_channel_axis_layers("VIPP Inspect")
    state = widget.pipeline.output_states["input"]
    before_state = state.to_dict()
    before_params = deepcopy(widget.pipeline.nodes["input"].params)
    output = widget.pipeline.outputs["input"]
    before_data = output.copy()
    before_dirty = set(widget._pending_dirty_node_ids)
    monkeypatch.setattr(
        widget, "run_pipeline", lambda *_a, **_k: pytest.fail("Recalculated")
    )
    for axis in widget._view_dim_axes():
        widget._on_view_dim_changed(axis.step_axis, min(axis.size - 1, 3))
    before_positions = {a.name: a.value for a in widget._view_dim_axes()}

    for _ in range(2):
        control.combo.setCurrentText("Separate coloured layers")
        layers = widget._colored_channel_axis_layers("VIPP Inspect")
        assert len(layers) == 2
        for index, layer in enumerate(layers):
            np.testing.assert_array_equal(
                layer.data, np.take(output, index, axis=axes.index("C"))
            )
            assert not layer.data.flags.writeable
            assert "C" not in layer.axis_labels
            expected_color = (0, 0, 1) if index == 0 else (0, 1, 0)
            np.testing.assert_array_equal(
                layer.metadata["display_channel_color"], expected_color
            )
        assert "C" not in {a.label for a in widget._view_dim_axes()}
        assert "C" not in viewer.dims.axis_labels
        assert widget.pipeline.outputs["input"] is output
        assert widget.pipeline.output_states["input"] is state
        assert widget.pipeline.nodes["input"].params == before_params
        assert widget._pending_dirty_node_ids == before_dirty

        control.combo.setCurrentText("Stack (C slider)")
        assert not widget._colored_channel_axis_layers("VIPP Inspect")
        inspect = viewer.layers["VIPP Inspect"]
        assert inspect.data.shape == output.shape
        assert not inspect.data.flags.writeable
        assert {a.name: a.value for a in widget._view_dim_axes()} == before_positions
        channel = next(a for a in widget._view_dim_axes() if a.label == "C")
        widget._on_view_dim_changed(channel.step_axis, 0)
        c_axis = tuple(viewer.dims.axis_labels).index("C")
        assert viewer.dims.current_step[c_axis] == 0
        viewer.dims.set_current_step(c_axis, 1)
        assert next(a.value for a in widget._view_dim_axes() if a.label == "C") == 1
        if ndisplay == 3 and "Z" in axes:
            assert {viewer.dims.axis_labels[i] for i in viewer.dims.displayed} == {
                "Z",
                "Y",
                "X",
            }
    assert state.to_dict() == before_state
    np.testing.assert_array_equal(output, before_data)
    np.testing.assert_array_equal(data, before_data)
    with widget._suspend_viewer_layer_change_handling():
        viewer.layers.clear()


def test_source_choice_survives_axis_roundtrip_and_colour_edits(qtbot):
    widget, viewer, _data = _widget(qtbot, colors=False)
    # An authored colour no longer implicitly splits a source into layers.
    widget._on_metadata_channel_color_changed(0, "Magenta")
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)
    assert not widget._colored_channel_axis_layers("VIPP Inspect")
    original = widget._image_source_value(widget.pipeline.nodes["input"])
    widget._on_image_source_changed({**original, "axis_declaration": "ZCYX -> TCYX"})
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)
    widget._on_image_source_changed(original)
    widget._debounce_timer.stop()
    widget.run_pipeline(force_sync=True)
    assert [a.name for a in widget.pipeline.output_states["input"].axes] == list("zcyx")
    assert "C" in {a.label for a in widget._view_dim_axes()}
    assert not widget._colored_channel_axis_layers("VIPP Inspect")
    viewer.layers.clear()


def test_display_choice_persists_only_in_workflow_inspector_metadata(qtbot, tmp_path):
    widget, viewer, _data = _widget(qtbot)
    widget._on_source_channel_display_changed("input", "layers")
    snapshot = widget._current_history_snapshot()
    assert snapshot.source_channel_displays == (("input", "layers"),)
    path = tmp_path / "channel-display.json"
    save_workflow(path, widget.pipeline, metadata=widget._workflow_metadata())
    document = load_workflow(path)
    assert document["metadata"]["vipp"]["inspector"]["source_channel_displays"] == {
        "input": "layers"
    }
    assert (
        "source_channel_display"
        not in json.loads(path.read_text())["nodes"][0]["params"]
    )
    widget._source_channel_displays.clear()
    widget.load_workflow_file(path)
    assert widget._source_channel_displays == {"input": "layers"}
    assert (
        widget._parameter_widgets["source_channel_display"].value()
        == "Separate coloured layers"
    )
    viewer.layers.clear()


def test_no_channel_axis_disables_display_choice(qtbot):
    widget, viewer, _data = _widget(qtbot, axes="ZYX")
    assert not widget._parameter_widgets["source_channel_display"].isEnabled()
    viewer.layers.clear()


def test_source_channel_display_updates_its_pin(qtbot):
    widget, viewer, _data = _widget(qtbot)
    widget.pin_node("input")
    assert len(widget._active_pinned_layers()) == 1
    widget._on_source_channel_display_changed("input", "layers")
    assert len(widget._active_pinned_layers()) == 2
    widget._on_source_channel_display_changed("input", "stack")
    layers = widget._active_pinned_layers()
    assert len(layers) == 1
    assert "C" in layers[0].axis_labels
    viewer.layers.clear()


def test_source_channel_choice_is_independent_per_workflow_tab(qtbot, monkeypatch):
    widget, viewer, _data = _widget(qtbot)
    monkeypatch.setattr(widget, "run_pipeline", lambda *_a, **_k: None)
    first = widget._workflow_tabs.current
    widget._on_source_channel_display_changed("input", "layers")
    widget._new_workflow()
    second = widget._workflow_tabs.current
    assert not widget._source_channel_displays
    assert (
        widget._parameter_widgets["source_channel_display"].value()
        == "Stack (C slider)"
    )
    assert widget._activate_workflow_tab(
        widget._workflow_tabs.index_of(first.session_id),
        check_safety=False,
    )
    assert widget._source_channel_displays == {"input": "layers"}
    assert widget._activate_workflow_tab(
        widget._workflow_tabs.index_of(second.session_id),
        check_safety=False,
    )
    assert not widget._source_channel_displays
    viewer.layers.clear()


@pytest.mark.parametrize(
    "choices", [[], {"input": "invalid"}, {"input": []}, {"input": None}]
)
def test_invalid_saved_channel_displays_are_rejected(tmp_path, choices):
    with pytest.raises(ValueError, match="Source channel displays"):
        save_workflow(
            tmp_path / "invalid.json",
            PrototypePipeline(),
            metadata={
                "vipp": {"inspector": {"source_channel_displays": choices}},
            },
        )


@pytest.mark.parametrize("width", [340, 620])
@pytest.mark.parametrize("theme", ["dark", "light"])
def test_source_channel_choice_layout(qtbot, qapp, tmp_path, width, theme):
    from napari._qt.qt_resources import get_stylesheet
    from qtpy.QtGui import QFont

    widget, viewer, _data = _widget(qtbot)
    panel = widget.inspector_panel
    panel.setParent(None)
    qtbot.addWidget(panel)
    panel.setFont(QFont("Segoe UI", 12))
    panel.setStyleSheet(get_stylesheet(theme, extra_variables={"font_size": "12pt"}))
    panel.resize(width, 650)
    panel.show()
    control = widget._parameter_widgets["source_channel_display"]
    control.combo.setCurrentText("Separate coloured layers")
    qapp.processEvents()
    panel.ensureWidgetVisible(control)
    qtbot.wait(50)
    assert (
        panel.viewport()
        .rect()
        .contains(control.mapTo(panel.viewport(), control.rect().bottomRight()))
    )
    assert control.combo.width() > 170
    assert panel.grab().save(str(tmp_path / "source-channel-display.png"))
    viewer.layers.clear()
