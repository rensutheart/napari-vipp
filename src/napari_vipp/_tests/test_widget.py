from __future__ import annotations

import numpy as np

from napari_vipp._widget import VippWidget


class _Event:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback

    def emit(self):
        if self.callback is not None:
            self.callback()


class _LayerEvents:
    def __init__(self):
        self.inserted = _Event()
        self.removed = _Event()


class _DimsEvents:
    def __init__(self):
        self.current_step = _Event()


class _Dims:
    def __init__(self):
        self.current_step = (0, 0, 0)
        self.events = _DimsEvents()


class _Layer:
    def __init__(self, data, name, metadata=None, layer_type="image"):
        self.data = data
        self.name = name
        self.metadata = metadata or {}
        self.layer_type = layer_type
        self.blending = None
        self.colormap = None
        self.contrast_limits = None
        self.visible = True


class _LayerList(list):
    def __init__(self, layers):
        super().__init__(layers)
        self.events = _LayerEvents()

    def __getitem__(self, item):
        if isinstance(item, str):
            for layer in self:
                if layer.name == item:
                    return layer
            raise KeyError(item)
        return super().__getitem__(item)

    def move(self, source, target):
        if source < target:
            target -= 1
        if source == target:
            return False
        layer = self.pop(source)
        self.insert(target, layer)
        return True


class _Viewer:
    def __init__(self, data=None):
        if data is None:
            data = np.zeros((4, 16, 18), dtype=np.float32)
        self.layers = _LayerList(
            [_Layer(data, "input volume")]
        )
        self.dims = _Dims()

    def add_image(self, data, **kwargs):
        layer = _Layer(
            data,
            kwargs["name"],
            metadata=kwargs.get("metadata"),
            layer_type="image",
        )
        layer.blending = kwargs.get("blending")
        layer.colormap = kwargs.get("colormap")
        layer.contrast_limits = kwargs.get("contrast_limits")
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _Layer(
            data,
            kwargs["name"],
            metadata=kwargs.get("metadata"),
            layer_type="labels",
        )
        self.layers.append(layer)
        return layer


def test_widget_builds_graph_and_inspects_node(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    assert widget.layer_combo.count() == 1
    assert "gaussian" in widget.pipeline.outputs

    widget.inspect_node("gaussian")

    inspect_layer = viewer.layers["VIPP Inspect"]
    assert inspect_layer.metadata["node_id"] == "gaussian"
    assert inspect_layer.data.shape == viewer.layers["input volume"].data.shape


def test_widget_pins_threshold_as_labels(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("threshold")

    pinned = viewer.layers["VIPP Pinned: Otsu Threshold"]
    assert pinned.metadata["napari_vipp_kind"] == "pinned"
    assert pinned.data.dtype == np.uint8
    assert widget.graph_view._cards["threshold"].pin_button.text() == "Unpin"
    assert widget.pin_button.text() == "Pin selected"


def test_pin_toggles_active_node_layer(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("threshold")
    widget.pin_node("threshold")

    pinned_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "pinned"
    ]
    assert pinned_layers == []
    assert widget._active_pinned_node_id is None
    assert len(viewer.layers) == 1
    assert widget.graph_view._cards["threshold"].pin_button.text() == "Pin"
    assert widget.status_label.text() == "Unpinned 'Otsu Threshold'."


def test_nodes_without_parameters_hide_parameter_group(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget._select_node("threshold")

    assert widget.parameter_group.isHidden()


def test_palette_adds_node_and_connects_branch(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("median_filter")
    widget._connect_nodes("input", node.id)

    assert node.id in widget.pipeline.nodes
    assert (("input", node.id)) in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert widget.pipeline.outputs[node.id] is not None


def test_incompatible_connection_is_rejected(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("gaussian_blur")
    widget._connect_nodes("threshold", node.id)

    assert ("threshold", node.id) not in {
        (connection.source_id, connection.target_id)
        for connection in widget.pipeline.connections
    }
    assert "Cannot connect mask output to image input" in widget.status_label.text()


def test_only_one_node_is_actively_pinned(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("gaussian")
    widget.pin_node("threshold")

    pinned_layers = [
        layer
        for layer in viewer.layers
        if layer.metadata.get("napari_vipp_kind") == "pinned"
    ]
    assert len(pinned_layers) == 1
    assert pinned_layers[0].metadata["node_id"] == "threshold"
    assert widget._active_pinned_node_id == "threshold"
    assert widget.graph_view._cards["threshold"]._pinned
    assert not widget.graph_view._cards["gaussian"]._pinned
    assert viewer.layers[-1] is pinned_layers[0]


def test_selecting_another_node_does_not_clear_pin(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("threshold")
    widget.pin_node("threshold")
    widget.graph_view.select_node("gaussian")

    assert widget._active_pinned_node_id == "threshold"
    assert widget.graph_view._cards["threshold"]._pinned
    assert widget.graph_view._cards["threshold"].pin_button.text() == "Unpin"
    assert widget.graph_view._cards["gaussian"].pin_button.text() == "Pin"
    assert widget.pin_button.text() == "Pin selected"


def test_selected_pinned_node_shows_unpin_in_inspector(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("threshold")
    widget.pin_node("threshold")

    assert widget.pin_button.text() == "Unpin selected"


def test_active_pin_stays_on_top_after_inspect(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.pin_node("threshold")
    widget.inspect_node("gaussian")

    assert viewer.layers[-1].metadata["napari_vipp_kind"] == "pinned"
    assert viewer.layers[-1].metadata["node_id"] == "threshold"
    assert viewer.layers[-2].metadata["napari_vipp_kind"] == "inspect"


def test_inspect_shows_mask_as_standalone_image(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.inspect_node("gaussian")
    first_inspect = viewer.layers["VIPP Inspect"]
    assert first_inspect.layer_type == "image"
    assert first_inspect.metadata["display_kind"] == "image"

    widget.inspect_node("threshold")
    second_inspect = viewer.layers["VIPP Inspect"]

    assert second_inspect is first_inspect
    assert second_inspect.layer_type == "image"
    assert second_inspect.metadata["display_kind"] == "image"
    assert second_inspect.metadata["data_kind"] == "mask"
    assert second_inspect.metadata["node_id"] == "threshold"
    assert second_inspect.contrast_limits == (0, 1)
    assert second_inspect.blending == "opaque"


def test_inspecting_active_mask_pin_keeps_pin_overlay_on_mask_image(qtbot):
    viewer = _Viewer()
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    widget.graph_view.select_node("threshold")
    widget.pin_node("threshold")
    widget.inspect_node("threshold")

    inspect = viewer.layers["VIPP Inspect"]
    pinned = viewer.layers["VIPP Pinned: Otsu Threshold"]

    assert inspect.layer_type == "image"
    assert inspect.metadata["data_kind"] == "mask"
    assert pinned.layer_type == "labels"
    assert pinned.metadata["display_kind"] == "labels"
    assert viewer.layers[-2] is inspect
    assert viewer.layers[-1] is pinned


def test_binary_threshold_uses_uint8_slider_range(qtbot):
    viewer = _Viewer(np.arange(16, dtype=np.uint8).reshape(4, 4))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["threshold"]
    assert control.slider.minimum() == 0
    assert control.slider.maximum() == 255
    assert control.value_box.minimum() == 0
    assert control.value_box.maximum() == 255

    control.slider.setValue(128)

    assert widget.pipeline.nodes[node.id].params["threshold"] == 128


def test_binary_threshold_uses_unit_float_slider_range(qtbot):
    data = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    viewer = _Viewer(data)
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("binary_threshold")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["threshold"]
    assert control.value_box.minimum() == 0.0
    assert control.value_box.maximum() == 1.0
    assert control.slider.maximum() == 1000


def test_projection_axis_slider_uses_input_dimensionality(qtbot):
    viewer = _Viewer(np.zeros((2, 3, 4, 5), dtype=np.float32))
    widget = VippWidget(viewer)
    qtbot.addWidget(widget)

    node = widget.add_node_from_palette("mip")
    widget._connect_nodes("input", node.id)

    control = widget._parameter_widgets["axis"]
    assert control.slider.minimum() == 0
    assert control.slider.maximum() == 3
    assert control.value_box.maximum() == 3
