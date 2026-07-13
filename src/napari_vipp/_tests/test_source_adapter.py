from __future__ import annotations

import numpy as np

from napari_vipp.ui.source_adapter import LiveLayerSourceAdapter


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
        self.scale = _Signal()


class _Layer:
    def __init__(self, data) -> None:
        self.data = data
        self.metadata = {"nested": {"axis": "YX"}}
        self.name = "source"
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
