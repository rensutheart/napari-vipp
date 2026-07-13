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
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from napari_vipp.core.metadata import ImageState

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

    def invalidate(
        self,
        layer: Any,
        *,
        notify: bool = True,
    ) -> SourceRevisionToken | None:
        """Advance a tracked layer revision and discard its cached snapshot."""
        if self._closed:
            return None
        tracked = self._tracked.get(id(layer))
        if tracked is None or tracked.layer is not layer:
            return None
        tracked.revision += 1
        tracked.snapshot = None
        if notify and self._on_invalidated is not None:
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


def apply_live_layer_axis_transform(
    state: ImageState | None,
    snapshot: LiveLayerSnapshot,
) -> ImageState | None:
    """Carry an axis-aligned napari layer transform into scientific metadata."""
    if state is None:
        return None
    ndim = len(state.axes)
    _require_supported_axis_transform(snapshot, ndim)

    scale = snapshot.scale
    translate = snapshot.translate
    units = snapshot.units
    if scale is not None and len(scale) != ndim:
        raise ValueError(
            f"Live napari source '{snapshot.name}' has {len(scale)} scale values "
            f"for {ndim} data axes."
        )
    if translate is not None and len(translate) != ndim:
        raise ValueError(
            f"Live napari source '{snapshot.name}' has {len(translate)} translation "
            f"values for {ndim} data axes."
        )
    if units is not None and len(units) != ndim:
        raise ValueError(
            f"Live napari source '{snapshot.name}' has {len(units)} units for "
            f"{ndim} data axes."
        )
    if scale is not None and any(
        not np.isfinite(value) or value <= 0 for value in scale
    ):
        raise ValueError(
            f"Live napari source '{snapshot.name}' has a non-positive or non-finite "
            "axis scale. VIPP requires a positive axis-aligned physical grid."
        )

    use_scale = bool(
        scale is not None
        and (
            any(not np.isclose(value, 1.0) for value in scale)
            or all(np.isclose(axis.scale, 1.0) for axis in state.axes)
        )
    )
    use_translate = bool(
        translate is not None
        and (
            any(not np.isclose(value, 0.0) for value in translate)
            or all(np.isclose(axis.translation, 0.0) for axis in state.axes)
        )
    )
    physical_units = (
        tuple(_normalized_axis_unit(unit) for unit in units)
        if units is not None
        else ()
    )
    use_units = bool(physical_units and any(physical_units))
    if not (use_scale or use_translate or use_units):
        return state

    axes = tuple(
        replace(
            axis,
            scale=(
                float(scale[index])
                if use_scale and scale is not None
                else axis.scale
            ),
            translation=(
                float(translate[index])
                if use_translate and translate is not None
                else axis.translation
            ),
            unit=(
                physical_units[index]
                if use_units and physical_units[index]
                else axis.unit
            ),
        )
        for index, axis in enumerate(state.axes)
    )
    source = state.metadata_source
    if "napari layer transform" not in source.lower():
        source = f"{source}; napari layer transform"
    return replace(state, axes=axes, metadata_source=source)


def _require_supported_axis_transform(
    snapshot: LiveLayerSnapshot,
    ndim: int,
) -> None:
    if snapshot.shear is not None and any(
        not np.isclose(value, 0.0) for value in snapshot.shear
    ):
        raise ValueError(
            f"Live napari source '{snapshot.name}' has shear. VIPP does not "
            "silently discard non-axis-aligned transforms; resample or register "
            "the image explicitly before calculating."
        )
    if snapshot.rotate is not None and not _matrix_is_identity(
        snapshot.rotate,
        ndim,
        allow_zero_scalar=True,
    ):
        raise ValueError(
            f"Live napari source '{snapshot.name}' has rotation. VIPP does not "
            "silently discard non-axis-aligned transforms; resample or register "
            "the image explicitly before calculating."
        )
    if snapshot.affine is not None and not _matrix_is_identity(
        snapshot.affine,
        ndim + 1,
    ):
        raise ValueError(
            f"Live napari source '{snapshot.name}' has an additional affine "
            "transform. VIPP requires explicit resampling or registration before "
            "calculation."
        )


def _matrix_is_identity(
    matrix: tuple[tuple[float, ...], ...],
    dimension: int,
    *,
    allow_zero_scalar: bool = False,
) -> bool:
    try:
        arr = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError):
        return False
    if allow_zero_scalar and arr.size == 1:
        return bool(np.isclose(float(arr.reshape(-1)[0]), 0.0))
    return bool(
        arr.shape == (dimension, dimension)
        and np.allclose(arr, np.eye(dimension))
    )


def _normalized_axis_unit(unit: str) -> str | None:
    normalized = str(unit).strip()
    if normalized.lower() in {"", "pixel", "pixels", "px", "dimensionless"}:
        return None
    return normalized


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
    "apply_live_layer_axis_transform",
]
