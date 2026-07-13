"""Revisioned, owned snapshots of live napari layer sources.

Pipeline execution may run after the GUI event that started it.  Passing a
live layer array across that boundary would let an edit produce a calculation
from more than one source revision.  This module therefore owns a detached,
read-only NumPy snapshot for each observed layer revision and records the
events that invalidate it.

The implementation deliberately uses napari's public event interface by duck
typing rather than importing napari.  This keeps the boundary independently
testable and avoids making the core scientific package depend on the viewer.
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

SOURCE_REVISION_EVENTS = (
    "data",
    "set_data",
    "metadata",
    "name",
    "scale",
    "translate",
    "rotate",
    "shear",
    "affine",
    "units",
    "axis_labels",
    "labels_update",
)


@dataclass(frozen=True, slots=True)
class SourceRevisionToken:
    """Identity of one observed revision of a live viewer layer."""

    layer_id: int
    revision: int


@dataclass(frozen=True, slots=True, eq=False)
class LiveLayerSnapshot:
    """One stable source revision captured on the GUI thread."""

    token: SourceRevisionToken
    data: Any
    metadata: dict[str, Any]
    name: str
    scale: tuple[float, ...] | None
    translate: tuple[float, ...] | None
    rotate: tuple[tuple[float, ...], ...] | None
    shear: tuple[float, ...] | None
    affine: tuple[tuple[float, ...], ...] | None
    units: tuple[str, ...] | None
    axis_labels: tuple[str, ...] | None
    data_is_detached: bool


@dataclass(slots=True)
class _TrackedLayer:
    layer: Any
    revision: int
    connections: list[tuple[Any, Callable[..., None]]]
    snapshot: LiveLayerSnapshot | None = None


class LiveLayerSourceAdapter:
    """Track live layers and cache exactly one snapshot per source revision."""

    def __init__(self, on_invalidated: Callable[[Any], None] | None = None) -> None:
        self._on_invalidated = on_invalidated
        self._tracked: dict[int, _TrackedLayer] = {}
        self._closed = False

    def sync_layers(self, layers) -> None:
        """Observe exactly ``layers``, disconnecting sources no longer selected."""
        if self._closed:
            return
        unique_layers = {id(layer): layer for layer in layers}
        for layer_id, tracked in tuple(self._tracked.items()):
            layer = unique_layers.get(layer_id)
            if layer is not tracked.layer:
                self._disconnect(tracked)
                self._tracked.pop(layer_id, None)
        for layer_id, layer in unique_layers.items():
            tracked = self._tracked.get(layer_id)
            if tracked is not None and tracked.layer is layer:
                continue
            self._tracked[layer_id] = self._track(layer)

    def snapshot(self, layer: Any) -> LiveLayerSnapshot:
        """Return the cached owned snapshot for the layer's current revision."""
        if self._closed:
            raise RuntimeError("Live source adapter has been shut down.")
        layer_id = id(layer)
        tracked = self._tracked.get(layer_id)
        if tracked is None or tracked.layer is not layer:
            if tracked is not None:
                self._disconnect(tracked)
            tracked = self._track(layer)
            self._tracked[layer_id] = tracked
        if tracked.snapshot is None:
            tracked.snapshot = self._capture(tracked)
        return tracked.snapshot

    def token_is_current(self, token: SourceRevisionToken) -> bool:
        """Return whether ``token`` still names the observed live revision."""
        tracked = self._tracked.get(token.layer_id)
        return bool(
            not self._closed
            and tracked is not None
            and tracked.revision == token.revision
        )

    def tokens_are_current(self, tokens) -> bool:
        """Return whether every supplied source revision is still current."""
        return all(self.token_is_current(token) for token in tokens)

    def invalidate(self, layer: Any) -> SourceRevisionToken | None:
        """Advance a tracked layer revision and discard its cached snapshot."""
        if self._closed:
            return None
        tracked = self._tracked.get(id(layer))
        if tracked is None or tracked.layer is not layer:
            return None
        tracked.revision += 1
        tracked.snapshot = None
        if self._on_invalidated is not None:
            self._on_invalidated(layer)
        return SourceRevisionToken(id(layer), tracked.revision)

    def shutdown(self) -> None:
        """Disconnect every live-layer event and make the adapter terminal."""
        if self._closed:
            return
        self._closed = True
        for tracked in tuple(self._tracked.values()):
            self._disconnect(tracked)
        self._tracked.clear()

    def _track(self, layer: Any) -> _TrackedLayer:
        tracked = _TrackedLayer(layer=layer, revision=0, connections=[])
        events = getattr(layer, "events", None)
        if events is None:
            return tracked

        def on_change(_event=None, *, observed_layer=layer) -> None:
            self.invalidate(observed_layer)

        seen_signals: set[int] = set()
        for event_name in SOURCE_REVISION_EVENTS:
            signal = getattr(events, event_name, None)
            if signal is None or id(signal) in seen_signals:
                continue
            seen_signals.add(id(signal))
            try:
                signal.connect(on_change)
            except Exception:
                continue
            tracked.connections.append((signal, on_change))
        return tracked

    @staticmethod
    def _disconnect(tracked: _TrackedLayer) -> None:
        for signal, callback in reversed(tracked.connections):
            try:
                signal.disconnect(callback)
            except Exception:
                pass
        tracked.connections.clear()
        tracked.snapshot = None

    @staticmethod
    def _capture(tracked: _TrackedLayer) -> LiveLayerSnapshot:
        layer = tracked.layer
        data, data_is_detached = _snapshot_data(getattr(layer, "data", None))
        raw_metadata = getattr(layer, "metadata", None)
        metadata = deepcopy(raw_metadata) if isinstance(raw_metadata, dict) else {}
        return LiveLayerSnapshot(
            token=SourceRevisionToken(id(layer), tracked.revision),
            data=data,
            metadata=metadata,
            name=str(getattr(layer, "name", "")),
            scale=_float_tuple(getattr(layer, "scale", None)),
            translate=_float_tuple(getattr(layer, "translate", None)),
            rotate=_matrix_tuple(getattr(layer, "rotate", None)),
            shear=_float_tuple(getattr(layer, "shear", None)),
            affine=_affine_matrix_tuple(getattr(layer, "affine", None)),
            units=_string_tuple(getattr(layer, "units", None)),
            axis_labels=_string_tuple(getattr(layer, "axis_labels", None)),
            data_is_detached=data_is_detached,
        )


def _snapshot_data(data: Any) -> tuple[Any, bool]:
    """Detach writable in-memory NumPy data without realizing lazy arrays."""
    if not isinstance(data, np.ndarray):
        return data, False
    snapshot = np.array(data, copy=True, order="K", subok=True)
    try:
        snapshot.setflags(write=False)
    except ValueError:
        return snapshot, False
    detached = not np.shares_memory(snapshot, data) and not snapshot.flags.writeable
    return snapshot, detached


def _float_tuple(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return tuple(float(item) for item in arr)


def _matrix_tuple(value: Any) -> tuple[tuple[float, ...], ...] | None:
    if value is None:
        return None
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if arr.ndim == 0:
        return ((float(arr),),)
    if arr.ndim == 1:
        return (tuple(float(item) for item in arr),)
    if arr.ndim != 2:
        return None
    return tuple(tuple(float(item) for item in row) for row in arr)


def _affine_matrix_tuple(value: Any) -> tuple[tuple[float, ...], ...] | None:
    matrix = getattr(value, "affine_matrix", value)
    return _matrix_tuple(matrix)


def _string_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    try:
        return tuple(str(item) for item in value)
    except TypeError:
        return None


__all__ = [
    "SOURCE_REVISION_EVENTS",
    "LiveLayerSnapshot",
    "LiveLayerSourceAdapter",
    "SourceRevisionToken",
]
