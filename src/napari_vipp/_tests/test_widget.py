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
    def __init__(self, data, name, metadata=None):
        self.data = data
        self.name = name
        self.metadata = metadata or {}
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


class _Viewer:
    def __init__(self):
        self.layers = _LayerList(
            [_Layer(np.zeros((4, 16, 18), dtype=np.float32), "input volume")]
        )
        self.dims = _Dims()

    def add_image(self, data, **kwargs):
        layer = _Layer(data, kwargs["name"], metadata=kwargs.get("metadata"))
        self.layers.append(layer)
        return layer

    def add_labels(self, data, **kwargs):
        layer = _Layer(data, kwargs["name"], metadata=kwargs.get("metadata"))
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
