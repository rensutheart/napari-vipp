"""Qt-free contracts for exact scientific source-window reads.

These records intentionally do not reuse the presentation-preview contracts.
A source window is always an exact level-0 scientific read whose pixels may be
fed into graph execution.  Reader implementations must slice their lazy source
before computing and must return an owned, C-contiguous, read-only array.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from napari_vipp.core.metadata import AxisDeclaration, ImageState
from napari_vipp.core.progress import OperationCancelled

if TYPE_CHECKING:
    from napari_vipp.core.io.model import ImageSeriesInfo, SourceInspection


SourceWindowProgressCallback = Callable[[int, int, str], None]


@dataclass(frozen=True, slots=True)
class SourceWindowReadEstimate:
    """Conservative decoded-memory estimate for one exact chunked read."""

    requested_decoded_bytes: int
    estimated_touched_chunk_decoded_bytes: int
    estimated_touched_chunk_count: int
    estimated_peak_bytes: int
    basis: str

    def __post_init__(self) -> None:
        for name in (
            "requested_decoded_bytes",
            "estimated_touched_chunk_decoded_bytes",
            "estimated_touched_chunk_count",
            "estimated_peak_bytes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"source window {name} must be non-negative.")
        if self.estimated_touched_chunk_decoded_bytes < self.requested_decoded_bytes:
            raise ValueError(
                "touched chunk bytes cannot be below requested decoded bytes."
            )
        minimum_peak = (
            self.estimated_touched_chunk_decoded_bytes
            + 2 * self.requested_decoded_bytes
        )
        if self.estimated_peak_bytes < minimum_peak:
            raise ValueError(
                "source window peak estimate must include touched chunks, the "
                "assembled ROI, and the detached publication copy."
            )
        basis = str(self.basis).strip()
        if not basis:
            raise ValueError("source window read estimates require a basis.")
        object.__setattr__(self, "basis", basis)

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_decoded_bytes": self.requested_decoded_bytes,
            "estimated_touched_chunk_decoded_bytes": (
                self.estimated_touched_chunk_decoded_bytes
            ),
            "estimated_touched_chunk_count": self.estimated_touched_chunk_count,
            "estimated_peak_bytes": self.estimated_peak_bytes,
            "basis": self.basis,
        }


def estimate_exact_window_read(
    source_shape: Sequence[int],
    source_dtype,
    selection: tuple[slice, ...],
    *,
    chunk_grid: Sequence[Sequence[int]] | None = None,
) -> SourceWindowReadEstimate:
    """Estimate a rank-preserving exact read from a declared chunk grid.

    The upper bound assumes all intersecting decoded chunks coexist with both
    the assembled ROI and the detached read-only publication copy.  Keeping
    this arithmetic reader-neutral lets preflight planning use the same facts
    as the OME-Zarr reader without opening or computing the source again.
    """

    shape = tuple(int(size) for size in source_shape)
    normalized = SourceWindowRequest(selection).normalized_selection(shape)
    dtype = np.dtype(source_dtype)
    requested_shape = tuple(
        int(selector.stop) - int(selector.start) for selector in normalized
    )
    requested = int(np.prod(requested_shape, dtype=np.int64)) * dtype.itemsize
    chunks = tuple(tuple(int(size) for size in axis) for axis in (chunk_grid or ()))
    if not chunks:
        touched = requested
        count = 1
        basis = "unchunked exact read plus assembled ROI and detached publication copy"
    else:
        if len(chunks) != len(shape):
            raise ValueError("source chunk grid rank must match the source shape.")
        touched_axes: list[tuple[int, ...]] = []
        for axis_chunks, selector, axis_size in zip(
            chunks,
            normalized,
            shape,
            strict=True,
        ):
            if not axis_chunks or any(size <= 0 for size in axis_chunks):
                raise ValueError("source chunk grid sizes must be positive.")
            if sum(axis_chunks) != axis_size:
                raise ValueError(
                    "source chunk grid must cover the complete source shape."
                )
            touched_axes.append(
                _source_window_touched_chunk_sizes(axis_chunks, selector)
            )
        count = int(np.prod([len(values) for values in touched_axes], dtype=np.int64))
        touched = (
            int(np.prod([sum(values) for values in touched_axes], dtype=np.int64))
            * dtype.itemsize
        )
        basis = (
            "conservative upper bound from the declared chunk grid: all "
            "intersecting decoded chunks plus assembled ROI and detached "
            "publication copy; compressed storage and codec scratch are not exposed"
        )
    return SourceWindowReadEstimate(
        requested_decoded_bytes=requested,
        estimated_touched_chunk_decoded_bytes=touched,
        estimated_touched_chunk_count=count,
        estimated_peak_bytes=touched + 2 * requested,
        basis=basis,
    )


def _source_window_touched_chunk_sizes(
    chunks: tuple[int, ...],
    selector: slice,
) -> tuple[int, ...]:
    start = int(selector.start)
    stop = int(selector.stop)
    cursor = 0
    touched: list[int] = []
    for chunk_size in chunks:
        chunk_stop = cursor + int(chunk_size)
        if cursor < stop and chunk_stop > start:
            touched.append(int(chunk_size))
        cursor = chunk_stop
        if cursor >= stop:
            break
    if not touched:
        raise IndexError("Exact window does not intersect the source chunk grid.")
    return tuple(touched)


@dataclass(frozen=True, slots=True)
class SourceWindowRequest:
    """One full-rank unit-step window in level-0 pixel coordinates.

    ``selection`` must contain one :class:`slice` for every source dimension.
    Integer indexing is deliberately excluded so an exact window never drops
    an axis.  Bounds are normalized and checked against the selected source by
    :meth:`normalized_selection` before any pixels are read.

    By default T and C are required to remain complete.  The opt-out exists for
    future explicit callers, but Crop Stack pushdown must retain the default.
    ``source_revision`` and ``source_item_digest`` are opaque verified-source
    facts supplied by the caller and become part of the immutable read identity.
    The reader does not claim to establish those facts itself.
    """

    selection: tuple[slice, ...]
    analysis_level: int = 0
    axis_declaration: AxisDeclaration | None = None
    preserve_time_and_channels: bool = True
    source_revision: str = ""
    source_item_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.selection, tuple) or not self.selection:
            raise ValueError(
                "source window selection must be a non-empty tuple of slices."
            )
        normalized: list[slice] = []
        for index, selector in enumerate(self.selection):
            if not isinstance(selector, slice):
                raise TypeError(
                    "source window selection must preserve rank; "
                    f"dimension {index} is not a slice."
                )
            if selector.step not in {None, 1}:
                raise ValueError("source windows support unit-step slices only.")
            start = _optional_non_negative_int(selector.start, "slice start")
            stop = _optional_non_negative_int(selector.stop, "slice stop")
            if start is not None and stop is not None and stop <= start:
                raise ValueError("source window slices must have positive extents.")
            normalized.append(slice(start, stop, 1))
        if isinstance(self.analysis_level, bool) or int(self.analysis_level) != 0:
            raise ValueError("scientific source-window reads are fixed to level 0.")
        object.__setattr__(self, "selection", tuple(normalized))
        object.__setattr__(self, "analysis_level", 0)
        object.__setattr__(
            self,
            "axis_declaration",
            AxisDeclaration.from_value(self.axis_declaration),
        )
        object.__setattr__(
            self,
            "preserve_time_and_channels",
            bool(self.preserve_time_and_channels),
        )
        object.__setattr__(self, "source_revision", str(self.source_revision))
        object.__setattr__(
            self,
            "source_item_digest",
            str(self.source_item_digest),
        )

    def normalized_selection(
        self,
        shape: Sequence[int],
    ) -> tuple[slice, ...]:
        """Return explicit in-bounds selectors for one exact source shape."""
        source_shape = tuple(int(size) for size in shape)
        if len(source_shape) != len(self.selection):
            raise ValueError(
                "source window rank does not match the selected source: "
                f"{len(self.selection)} selectors for {len(source_shape)} dimensions."
            )
        if any(size <= 0 for size in source_shape):
            raise ValueError("source window reads require positive source dimensions.")
        result: list[slice] = []
        for index, (selector, size) in enumerate(
            zip(self.selection, source_shape, strict=True)
        ):
            start = 0 if selector.start is None else int(selector.start)
            stop = size if selector.stop is None else int(selector.stop)
            if start >= size or stop > size or stop <= start:
                raise IndexError(
                    "source window slice is outside the selected source at "
                    f"dimension {index}: [{start}:{stop}] for size {size}."
                )
            result.append(slice(start, stop, 1))
        return tuple(result)


@dataclass(frozen=True, slots=True)
class SourceWindowIdentity:
    """Stable identity of one exact scientific source-window read."""

    source_uri: str
    source_format: str
    series_index: int
    item_key: str
    reader_key: str
    reader_version: str
    source_shape: tuple[int, ...]
    source_dtype: str
    axis_names: tuple[str, ...]
    bounds: tuple[tuple[int, int], ...]
    read_estimate: SourceWindowReadEstimate | None = None
    source_revision: str = ""
    source_item_digest: str = ""
    analysis_level: int = 0

    def __post_init__(self) -> None:
        shape = tuple(int(size) for size in self.source_shape)
        bounds = tuple((int(start), int(stop)) for start, stop in self.bounds)
        axes = tuple(str(name).strip().casefold() for name in self.axis_names)
        if not shape or any(size <= 0 for size in shape):
            raise ValueError("source window identity requires a positive source shape.")
        if len(bounds) != len(shape) or len(axes) != len(shape):
            raise ValueError(
                "source window identity shape, axes, and bounds must have equal rank."
            )
        for index, ((start, stop), size) in enumerate(
            zip(bounds, shape, strict=True)
        ):
            if start < 0 or stop <= start or stop > size:
                raise ValueError(
                    "source window identity has invalid bounds at dimension "
                    f"{index}: [{start}:{stop}] for size {size}."
                )
        if isinstance(self.series_index, bool) or int(self.series_index) < 0:
            raise ValueError("source window series index must be non-negative.")
        if isinstance(self.analysis_level, bool) or int(self.analysis_level) != 0:
            raise ValueError("scientific source-window identity must use level 0.")
        if self.read_estimate is not None and not isinstance(
            self.read_estimate,
            SourceWindowReadEstimate,
        ):
            raise TypeError(
                "source window read_estimate must be a SourceWindowReadEstimate."
            )
        object.__setattr__(self, "source_uri", str(self.source_uri))
        object.__setattr__(self, "source_format", str(self.source_format))
        object.__setattr__(self, "series_index", int(self.series_index))
        object.__setattr__(self, "item_key", str(self.item_key))
        object.__setattr__(self, "reader_key", str(self.reader_key))
        object.__setattr__(self, "reader_version", str(self.reader_version))
        object.__setattr__(self, "source_shape", shape)
        object.__setattr__(self, "source_dtype", np.dtype(self.source_dtype).name)
        object.__setattr__(self, "axis_names", axes)
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "source_revision", str(self.source_revision))
        object.__setattr__(
            self,
            "source_item_digest",
            str(self.source_item_digest),
        )
        object.__setattr__(self, "analysis_level", 0)

    @property
    def output_shape(self) -> tuple[int, ...]:
        return tuple(stop - start for start, stop in self.bounds)

    @property
    def selection(self) -> tuple[slice, ...]:
        return tuple(slice(start, stop, 1) for start, stop in self.bounds)

    def to_dict(self, *, include_source_uri: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema": "vipp-source-window-v1",
            "source_format": self.source_format,
            "series_index": self.series_index,
            "item_key": self.item_key,
            "reader_key": self.reader_key,
            "reader_version": self.reader_version,
            "source_shape": list(self.source_shape),
            "source_dtype": self.source_dtype,
            "axis_names": list(self.axis_names),
            "bounds": [list(bound) for bound in self.bounds],
            "read_estimate": (
                None if self.read_estimate is None else self.read_estimate.to_dict()
            ),
            "source_revision": self.source_revision,
            "source_item_digest": self.source_item_digest,
            "analysis_level": self.analysis_level,
        }
        if include_source_uri:
            payload["source_uri"] = self.source_uri
        return payload

    @property
    def digest(self) -> str:
        encoded = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(b"vipp-source-window-v1\0" + encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class SourceWindowControl:
    """Cooperative cancellation and progress callbacks for a scientific read."""

    cancelled: Callable[[], bool] | None = None
    reporter: SourceWindowProgressCallback | None = None
    preflight: Callable[[SourceWindowReadEstimate], None] | None = None

    def __post_init__(self) -> None:
        for name in ("cancelled", "reporter", "preflight"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"source window {name} must be callable or None.")

    def check_active(self) -> None:
        if self.cancelled is not None and bool(self.cancelled()):
            raise OperationCancelled("Exact scientific source-window read cancelled.")

    def report(self, current: int, total: int, message: str) -> None:
        self.check_active()
        bounded_total = max(int(total), 0)
        bounded_current = max(int(current), 0)
        if bounded_total:
            bounded_current = min(bounded_current, bounded_total)
        if self.reporter is not None:
            self.reporter(bounded_current, bounded_total, str(message))
        self.check_active()

    def preflight_read(self, estimate: SourceWindowReadEstimate) -> None:
        """Run a caller-owned host-memory gate immediately before compute."""
        if not isinstance(estimate, SourceWindowReadEstimate):
            raise TypeError("source window preflight requires a read estimate.")
        self.check_active()
        if self.preflight is not None:
            self.preflight(estimate)
        self.check_active()


@dataclass(frozen=True, slots=True)
class SourceWindowResult:
    """Owned exact level-0 pixels and their normalized cropped metadata."""

    data: np.ndarray
    image_state: ImageState
    identity: SourceWindowIdentity
    inspection: SourceInspection
    selected_series: ImageSeriesInfo

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("source window data must be an owned NumPy array.")
        if not isinstance(self.image_state, ImageState):
            raise TypeError("source window image_state must be an ImageState.")
        if not isinstance(self.identity, SourceWindowIdentity):
            raise TypeError("source window identity must be a SourceWindowIdentity.")
        if tuple(self.data.shape) != self.identity.output_shape:
            raise ValueError("source window data shape does not match its identity.")
        if tuple(self.image_state.shape) != tuple(self.data.shape):
            raise ValueError("source window data and ImageState shapes must agree.")
        if np.dtype(self.data.dtype).name != self.identity.source_dtype:
            raise ValueError("source window dtype does not match its identity.")
        estimate = self.identity.read_estimate
        if (
            estimate is not None
            and estimate.requested_decoded_bytes != self.data.nbytes
        ):
            raise ValueError(
                "source window requested-byte estimate does not match its data."
            )
        if not self.data.flags.owndata or not self.data.flags.c_contiguous:
            raise ValueError(
                "source window data must be owned and C-contiguous before publication."
            )
        if self.data.flags.writeable:
            raise ValueError("source window data must be read-only before publication.")
        if tuple(axis.name.casefold() for axis in self.image_state.axes) != (
            self.identity.axis_names
        ):
            raise ValueError(
                "source window ImageState axes do not match its read identity."
            )
        if self.selected_series.key != self.identity.item_key:
            raise ValueError("source window selected item key does not match identity.")
        if int(self.selected_series.index) != int(self.identity.series_index):
            raise ValueError(
                "source window selected series index does not match identity."
            )
        if tuple(self.selected_series.shape) != self.identity.source_shape:
            raise ValueError(
                "source window selected item shape does not match identity."
            )
        if np.dtype(self.selected_series.dtype).name != self.identity.source_dtype:
            raise ValueError(
                "source window selected item dtype does not match identity."
            )
        inspection_keys = {
            (int(item.index), str(item.key)) for item in self.inspection.series
        }
        if (self.identity.series_index, self.identity.item_key) not in inspection_keys:
            raise ValueError(
                "source window selected item is absent from its inspection."
            )


@dataclass(frozen=True, slots=True)
class ExactSourceWindowData:
    """Logical full source backed by one already-materialized exact window.

    The graph must continue to see the Image Source's complete declared shape,
    while the direct Crop Stack consumes only the region that its authored
    margins retain.  This wrapper prevents any accidental full-array coercion
    and publishes the owned window only when the requested selection matches
    the reader-verified identity exactly.
    """

    data: np.ndarray
    window_state: ImageState
    identity: SourceWindowIdentity

    def __post_init__(self) -> None:
        if not isinstance(self.data, np.ndarray):
            raise TypeError("exact source-window data must be a NumPy array.")
        if not isinstance(self.window_state, ImageState):
            raise TypeError("exact source-window state must be an ImageState.")
        if not isinstance(self.identity, SourceWindowIdentity):
            raise TypeError("exact source-window identity is required.")
        if tuple(self.data.shape) != self.identity.output_shape:
            raise ValueError("exact source-window data shape does not match identity.")
        if tuple(self.window_state.shape) != tuple(self.data.shape):
            raise ValueError("exact source-window state does not match its data.")
        if np.dtype(self.data.dtype).name != self.identity.source_dtype:
            raise ValueError("exact source-window dtype does not match identity.")
        if self.data.flags.writeable:
            raise ValueError("exact source-window data must be read-only.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the complete logical source shape, not the retained ROI."""

        return self.identity.source_shape

    @property
    def dtype(self) -> np.dtype:
        return np.dtype(self.identity.source_dtype)

    @property
    def ndim(self) -> int:
        return len(self.identity.source_shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.identity.source_shape, dtype=np.int64))

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def window_nbytes(self) -> int:
        return int(self.data.nbytes)

    def materialize_exact(self, selection: Sequence[slice]) -> np.ndarray:
        """Return the owned window only for its exact full-rank selection."""

        requested = _normalized_selection_tuple(selection, self.shape)
        if requested != self.identity.selection:
            raise RuntimeError(
                "The source-window execution plan is stale: Crop Stack requested "
                f"{_selection_text(requested)}, but the verified reader supplied "
                f"{_selection_text(self.identity.selection)}. Reload the source "
                "with the current workflow before calculating."
            )
        return self.data

    def __getitem__(self, selection: object) -> np.ndarray:
        selectors = selection if isinstance(selection, tuple) else (selection,)
        if any(not isinstance(selector, slice) for selector in selectors):
            raise TypeError(
                "Exact source-window data supports only full-rank slice access."
            )
        return self.materialize_exact(selectors)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        del dtype, copy
        raise RuntimeError(
            "The complete Image Source was intentionally not materialized. "
            "Run its directly connected Crop Stack, or remove/bypass that crop "
            "and explicitly reload the full source."
        )


def _normalized_selection_tuple(
    selection: Sequence[slice],
    shape: Sequence[int],
) -> tuple[slice, ...]:
    selectors = tuple(selection)
    source_shape = tuple(int(size) for size in shape)
    if len(selectors) != len(source_shape):
        raise ValueError(
            "exact source-window access must provide one slice per dimension."
        )
    normalized: list[slice] = []
    for selector, size in zip(selectors, source_shape, strict=True):
        if not isinstance(selector, slice):
            raise TypeError("exact source-window access supports slices only.")
        start, stop, step = selector.indices(size)
        if step != 1 or stop <= start:
            raise ValueError(
                "exact source-window access requires positive unit-step slices."
            )
        normalized.append(slice(start, stop, 1))
    return tuple(normalized)


def _selection_text(selection: Sequence[slice]) -> str:
    return "[" + ", ".join(
        f"{int(selector.start)}:{int(selector.stop)}" for selector in selection
    ) + "]"


def _optional_non_negative_int(value: object, label: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError(f"source window {label} must be an integer or None.")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"source window {label} must be an integer or None."
        ) from exc
    if parsed != value:
        raise TypeError(f"source window {label} must be an exact integer.")
    if parsed < 0:
        raise ValueError(f"source window {label} must be non-negative.")
    return parsed


__all__ = [
    "ExactSourceWindowData",
    "SourceWindowControl",
    "SourceWindowIdentity",
    "SourceWindowProgressCallback",
    "SourceWindowReadEstimate",
    "SourceWindowRequest",
    "SourceWindowResult",
    "estimate_exact_window_read",
]
