from __future__ import annotations

import numpy as np
import pytest

from napari_vipp.core.metadata import (
    AXIS_CONFIDENCE_EXPLICIT,
    AXIS_CONFIDENCE_INFERRED,
    AXIS_CONFIDENCE_MIXED,
    image_state_from_array,
)
from napari_vipp.ui.source_adapter import (
    LiveLayerSourceAdapter,
    apply_live_layer_axis_transform,
)


class _Signal:
    def __init__(self) -> None:
        self.callbacks = []

    def connect(self, callback) -> None:
        self.callbacks.append(callback)

    def disconnect(self, callback) -> None:
        self.callbacks.remove(callback)

    def emit(self) -> None:
        for callback in tuple(self.callbacks):
            callback(object())


class _Events:
    def __init__(self) -> None:
        self.data = _Signal()
        self.metadata = _Signal()
        self.rgb = _Signal()
        self.scale = _Signal()
        self.axis_labels = _Signal()


class _Layer:
    def __init__(self, data) -> None:
        self.data = data
        self.metadata = {"nested": {"axis": "YX"}}
        self.name = "source"
        self.rgb = False
        self.scale = np.array([0.5, 0.25])
        self.translate = np.array([10.0, 20.0])
        self.rotate = np.eye(2)
        self.shear = np.array([0.0])
        self.affine = np.eye(3)
        self.units = ("micrometer", "micrometer")
        self.axis_labels = ("y", "x")
        self.events = _Events()


def test_numpy_source_snapshot_is_owned_read_only_and_cached_per_revision():
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)
    layer = _Layer(source)
    adapter = LiveLayerSourceAdapter()

    first = adapter.snapshot(layer)
    second = adapter.snapshot(layer)

    assert first is second
    assert first.rgb is False
    assert first.data_is_detached
    assert not np.shares_memory(first.data, source)
    assert not first.data.flags.writeable
    np.testing.assert_array_equal(first.data, source)
    assert first.scale == (0.5, 0.25)
    assert first.translate == (10.0, 20.0)
    assert first.rotate == ((1.0, 0.0), (0.0, 1.0))
    assert first.affine == ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert first.units == ("micrometer", "micrometer")
    assert first.axis_labels == ("y", "x")


def test_snapshot_metadata_and_data_do_not_follow_live_mutation():
    source = np.arange(6, dtype=np.int16).reshape(2, 3)
    layer = _Layer(source)
    adapter = LiveLayerSourceAdapter()

    snapshot = adapter.snapshot(layer)
    source[:] = -1
    layer.metadata["nested"]["axis"] = "changed"

    np.testing.assert_array_equal(snapshot.data, np.arange(6).reshape(2, 3))
    assert snapshot.metadata == {"nested": {"axis": "YX"}}


def test_source_event_advances_revision_and_creates_one_new_snapshot():
    invalidated = []
    layer = _Layer(np.zeros((2, 2), dtype=np.uint8))
    adapter = LiveLayerSourceAdapter(invalidated.append)
    first = adapter.snapshot(layer)

    layer.data = np.full((2, 2), 7, dtype=np.uint8)
    layer.events.data.emit()
    second = adapter.snapshot(layer)

    assert invalidated == [layer]
    assert second is adapter.snapshot(layer)
    assert second is not first
    assert second.token.revision == first.token.revision + 1
    assert not adapter.token_is_current(first.token)
    assert adapter.token_is_current(second.token)
    np.testing.assert_array_equal(second.data, 7)


def test_metadata_and_transform_events_also_invalidate_source_revision():
    layer = _Layer(np.zeros((2, 2), dtype=np.uint8))
    adapter = LiveLayerSourceAdapter()
    initial = adapter.snapshot(layer)

    layer.events.metadata.emit()
    after_metadata = adapter.snapshot(layer)
    layer.events.scale.emit()
    after_scale = adapter.snapshot(layer)

    assert after_metadata.token.revision == initial.token.revision + 1
    assert after_scale.token.revision == after_metadata.token.revision + 1
    assert not adapter.tokens_are_current((initial.token, after_metadata.token))
    assert adapter.tokens_are_current((after_scale.token,))


def test_rgb_and_axis_label_events_invalidate_source_revision_when_available():
    layer = _Layer(np.zeros((2, 2), dtype=np.uint8))
    adapter = LiveLayerSourceAdapter()
    initial = adapter.snapshot(layer)

    layer.rgb = True
    layer.events.rgb.emit()
    after_rgb = adapter.snapshot(layer)
    layer.axis_labels = ("y", "x")
    layer.events.axis_labels.emit()
    after_axis_labels = adapter.snapshot(layer)

    assert after_rgb.rgb is True
    assert after_rgb.token.revision == initial.token.revision + 1
    assert after_axis_labels.token.revision == after_rgb.token.revision + 1


def test_explicit_refresh_can_invalidate_without_scheduling_callback():
    layer = _Layer(np.zeros((2, 2), dtype=np.uint8))
    invalidated = []
    adapter = LiveLayerSourceAdapter(invalidated.append)
    initial = adapter.snapshot(layer)

    adapter.invalidate(layer, notify=False)
    refreshed = adapter.snapshot(layer)

    assert invalidated == []
    assert refreshed.token.revision == initial.token.revision + 1


def test_sync_layers_and_shutdown_disconnect_external_events():
    first = _Layer(np.zeros((1,), dtype=np.uint8))
    second = _Layer(np.ones((1,), dtype=np.uint8))
    invalidated = []
    adapter = LiveLayerSourceAdapter(invalidated.append)
    adapter.sync_layers((first, second))
    first_snapshot = adapter.snapshot(first)

    adapter.sync_layers((second,))
    first.events.data.emit()

    assert invalidated == []
    assert not adapter.token_is_current(first_snapshot.token)

    second_snapshot = adapter.snapshot(second)
    adapter.shutdown()
    second.events.data.emit()

    assert invalidated == []
    assert not adapter.token_is_current(second_snapshot.token)


def test_lazy_or_external_data_is_identified_as_not_detached():
    external = object()
    layer = _Layer(external)

    snapshot = LiveLayerSourceAdapter().snapshot(layer)

    assert snapshot.data is external
    assert not snapshot.data_is_detached


def _semantic_snapshot(
    data,
    *,
    axis_labels,
    rgb: bool = False,
    scale=None,
    translate=None,
    units=None,
):
    layer = _Layer(data)
    displayed_ndim = np.ndim(data) - (1 if rgb else 0)
    layer.rgb = rgb
    layer.axis_labels = axis_labels
    layer.scale = scale
    layer.translate = translate
    layer.units = units
    layer.rotate = np.eye(displayed_ndim)
    layer.shear = np.zeros(max(displayed_ndim - 1, 1))
    layer.affine = np.eye(displayed_ndim + 1)
    return LiveLayerSourceAdapter().snapshot(layer)


def test_meaningful_axis_labels_create_mixed_explicit_semantics():
    data = np.zeros((2, 5, 7), dtype=np.float32)
    state = image_state_from_array(data)
    snapshot = _semantic_snapshot(
        data,
        axis_labels=("-3", "Y", "X"),
    )

    aligned = apply_live_layer_axis_transform(state, snapshot)

    assert aligned is not None
    assert aligned.axis_confidence == AXIS_CONFIDENCE_MIXED
    assert [axis.name for axis in aligned.axes] == ["z", "y", "x"]
    assert [axis.confidence for axis in aligned.axes] == [
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_EXPLICIT,
        AXIS_CONFIDENCE_EXPLICIT,
    ]


@pytest.mark.parametrize(
    "axis_labels",
    (
        ("-3", "-2", "-1"),
        ("axis 0", "dim_1", ""),
    ),
)
def test_default_axis_labels_do_not_claim_semantic_confidence(
    axis_labels,
):
    data = np.zeros((2, 5, 7), dtype=np.float32)
    state = image_state_from_array(data)
    snapshot = _semantic_snapshot(data, axis_labels=axis_labels)

    aligned = apply_live_layer_axis_transform(state, snapshot)

    assert aligned is not None
    assert aligned.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert not aligned.axes_explicit


def test_unrecognized_axis_labels_remain_display_names_not_semantic_claims():
    data = np.zeros((2, 5, 7), dtype=np.float32)
    state = image_state_from_array(data)
    snapshot = _semantic_snapshot(
        data,
        axis_labels=("row", "column", "foo"),
    )

    aligned = apply_live_layer_axis_transform(state, snapshot)

    assert aligned is not None
    assert [axis.name for axis in aligned.axes] == ["row", "column", "foo"]
    assert aligned.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert not aligned.axes_explicit


def test_rgb_declaration_only_promotes_final_colour_axis_and_maps_transforms():
    data = np.zeros((2, 5, 7, 3), dtype=np.float32)
    state = image_state_from_array(data)
    snapshot = _semantic_snapshot(
        data,
        axis_labels=("-3", "-2", "-1"),
        rgb=True,
        scale=(2.0, 0.5, 0.25),
        translate=(3.0, 4.0, 5.0),
        units=("second", "micrometer", "micrometer"),
    )

    aligned = apply_live_layer_axis_transform(state, snapshot)

    assert aligned is not None
    assert aligned.axis_confidence == AXIS_CONFIDENCE_MIXED
    assert [axis.confidence for axis in aligned.axes] == [
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_INFERRED,
        AXIS_CONFIDENCE_EXPLICIT,
    ]
    assert aligned.axes[-1].name == "rgb"
    assert aligned.axes[-1].type == "channel"
    assert tuple(axis.scale for axis in aligned.axes) == (2.0, 0.5, 0.25, 1.0)
    assert tuple(axis.translation for axis in aligned.axes) == (3.0, 4.0, 5.0, 0.0)
    assert tuple(axis.unit for axis in aligned.axes) == (
        "second",
        "micrometer",
        "micrometer",
        None,
    )


def test_rgb_false_does_not_promote_shape_inferred_colour_guess():
    data = np.zeros((5, 7, 3), dtype=np.uint8)
    state = image_state_from_array(data)
    snapshot = _semantic_snapshot(
        data,
        axis_labels=("-3", "-2", "-1"),
        rgb=False,
    )

    aligned = apply_live_layer_axis_transform(state, snapshot)

    assert aligned is not None
    assert aligned.axis_confidence == AXIS_CONFIDENCE_INFERRED
    assert not aligned.axes[-1].is_explicit


def test_authoritative_carried_ome_axes_survive_live_labels_and_transform():
    data = np.zeros((2, 5, 7), dtype=np.float32)
    ome_state = image_state_from_array(
        data,
        layer_metadata={
            "ome": {
                "multiscales": [
                    {
                        "axes": [
                            {"name": "z", "type": "space"},
                            {"name": "y", "type": "space"},
                            {"name": "x", "type": "space"},
                        ]
                    }
                ]
            }
        },
    )
    carried = image_state_from_array(
        data,
        layer_metadata={"vipp_image_state": ome_state.to_dict()},
    )
    snapshot = _semantic_snapshot(
        data,
        axis_labels=("time", "row", "column"),
        scale=(2.0, 0.5, 0.25),
    )

    aligned = apply_live_layer_axis_transform(carried, snapshot)

    assert aligned is not None
    assert aligned.axes_explicit
    assert [axis.name for axis in aligned.axes] == ["z", "y", "x"]
    assert tuple(axis.scale for axis in aligned.axes) == (2.0, 0.5, 0.25)
    assert aligned.metadata_source.startswith("VIPP carried state")


def test_authoritative_yxc_accepts_rgb_display_transform_lengths():
    data = np.zeros((5, 7, 3), dtype=np.uint8)
    state = image_state_from_array(data, layer_metadata={"axes": "YXC"})
    snapshot = _semantic_snapshot(
        data,
        axis_labels=("y", "x"),
        rgb=True,
        scale=(0.5, 0.25),
        translate=(4.0, 6.0),
        units=("micrometer", "micrometer"),
    )

    aligned = apply_live_layer_axis_transform(state, snapshot)

    assert aligned is not None
    assert aligned.axis_order == "YXC"
    assert aligned.axes_explicit
    assert tuple(axis.scale for axis in aligned.axes) == (0.5, 0.25, 1.0)
    assert tuple(axis.translation for axis in aligned.axes) == (4.0, 6.0, 0.0)


@pytest.mark.parametrize(
    ("shape", "rgb", "axis_labels", "expected_axes"),
    (
        ((2, 5, 7), False, ("y", "x"), 3),
        ((2, 5, 7, 3), True, ("z", "y", "x", "rgb"), 3),
    ),
)
def test_invalid_axis_label_length_is_rejected(
    shape,
    rgb,
    axis_labels,
    expected_axes,
):
    data = np.zeros(shape, dtype=np.float32)
    state = image_state_from_array(data)
    snapshot = _semantic_snapshot(data, axis_labels=axis_labels, rgb=rgb)

    with pytest.raises(
        ValueError,
        match=(
            rf"{len(axis_labels)} axis labels for {expected_axes} "
            "displayed data axes"
        ),
    ):
        apply_live_layer_axis_transform(state, snapshot)
