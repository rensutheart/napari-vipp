"""Qt-free contracts for source-backed presentation previews."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from napari_vipp.core.metadata import AxisDeclaration, ImageState
from napari_vipp.core.progress import OperationCancelled


class SourcePreviewCancelled(OperationCancelled):
    """Raised when the requested presentation preview is cancelled."""


class StaleSourcePreviewGeneration(OperationCancelled):
    """Raised when a newer presentation-preview generation supersedes work."""


@dataclass(frozen=True, slots=True)
class SourcePreviewProgress:
    """One generation-qualified presentation-preview progress update."""

    generation: int
    current: int
    total: int
    message: str

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ValueError("source preview progress generation must be non-negative.")
        total = max(int(self.total), 0)
        current = max(int(self.current), 0)
        if total:
            current = min(current, total)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "current", current)
        object.__setattr__(self, "total", total)
        object.__setattr__(self, "message", str(self.message))


@dataclass(frozen=True, slots=True)
class SourcePreviewControl:
    """Cooperative cancellation, staleness, and progress callbacks for a read."""

    generation: int
    cancelled: Callable[[], bool] | None = None
    current_generation: Callable[[], int] | None = None
    reporter: Callable[[SourcePreviewProgress], None] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ValueError("source preview generation must be non-negative.")
        object.__setattr__(self, "generation", int(self.generation))
        for name in ("cancelled", "current_generation", "reporter"):
            value = getattr(self, name)
            if value is not None and not callable(value):
                raise TypeError(f"source preview {name} must be callable or None.")

    def check_active(self) -> None:
        """Fail before obsolete work can continue or publish a result."""
        if self.cancelled is not None and bool(self.cancelled()):
            raise SourcePreviewCancelled(
                f"Source preview generation {self.generation} was cancelled."
            )
        if self.current_generation is not None:
            current = int(self.current_generation())
            if current != self.generation:
                raise StaleSourcePreviewGeneration(
                    f"Source preview generation {self.generation} was superseded "
                    f"by generation {current}."
                )

    def report(self, current: int, total: int, message: str) -> None:
        """Report bounded progress while retaining publication safety."""
        self.check_active()
        total = max(int(total), 0)
        current = max(int(current), 0)
        if total:
            current = min(current, total)
        if self.reporter is not None:
            self.reporter(
                SourcePreviewProgress(
                    generation=self.generation,
                    current=current,
                    total=total,
                    message=str(message),
                )
            )
        self.check_active()


class SourcePreviewGenerationCoordinator:
    """Thread-safe owner of the newest presentation-preview generation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._current_generation = 0
        self._cancelled_generations: set[int] = set()

    @property
    def current_generation(self) -> int:
        with self._lock:
            return self._current_generation

    def begin(
        self,
        *,
        reporter: Callable[[SourcePreviewProgress], None] | None = None,
    ) -> SourcePreviewControl:
        """Start a new generation and make all older controls stale."""
        with self._lock:
            self._current_generation += 1
            generation = self._current_generation
            self._cancelled_generations.clear()
        return SourcePreviewControl(
            generation=generation,
            cancelled=lambda: self.is_cancelled(generation),
            current_generation=lambda: self.current_generation,
            reporter=reporter,
        )

    def cancel(self, generation: int | None = None) -> None:
        """Cancel the given generation, or the current generation by default."""
        with self._lock:
            selected = (
                self._current_generation if generation is None else int(generation)
            )
            if selected < 0:
                raise ValueError("source preview generation must be non-negative.")
            self._cancelled_generations.add(selected)

    def is_cancelled(self, generation: int) -> bool:
        with self._lock:
            return int(generation) in self._cancelled_generations

    def may_publish(self, generation: int) -> bool:
        """Return whether a completed result still belongs to the active request."""
        with self._lock:
            selected = int(generation)
            return (
                selected == self._current_generation
                and selected not in self._cancelled_generations
            )


@dataclass(frozen=True, slots=True)
class SourcePreviewRequest:
    """A bounded presentation request expressed in analysis-level coordinates.

    ``yx_region`` is ``(y_start, y_stop, x_start, x_stop)`` in level-0 pixel
    coordinates.  T, Z, and C indices refer to their semantic axes and are
    mapped through each level's declared coordinate transform and sliced before
    any pixels are computed. ``retain_z`` instead keeps the selected preview
    level's complete Z extent, while T and C remain plane selections.
    ``max_decoded_bytes`` bounds both the materialized preview and the decoded
    source chunks it may touch. ``level`` chooses an exact declared
    presentation level; the default selects a suitable safe lower-resolution
    level automatically.
    """

    display_shape_yx: tuple[int, int] = (512, 512)
    t_index: int | None = None
    z_index: int | None = None
    c_index: int | None = None
    yx_region: tuple[int, int, int, int] | None = None
    level: int | None = None
    axis_declaration: AxisDeclaration | None = None
    retain_z: bool = False
    max_decoded_bytes: int = 64 * 1024**2

    def __post_init__(self) -> None:
        display = _positive_pair(self.display_shape_yx, "display_shape_yx")
        object.__setattr__(self, "display_shape_yx", display)
        for name in ("t_index", "z_index", "c_index", "level"):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"source preview {name} must be non-negative.")
            object.__setattr__(self, name, int(value))
        if not isinstance(self.retain_z, bool):
            raise TypeError("source preview retain_z must be a Boolean.")
        if self.retain_z and self.z_index is not None:
            raise ValueError(
                "source preview retain_z cannot be combined with z_index."
            )
        maximum = self.max_decoded_bytes
        if isinstance(maximum, bool) or int(maximum) <= 0:
            raise ValueError(
                "source preview max_decoded_bytes must be a positive integer."
            )
        object.__setattr__(self, "max_decoded_bytes", int(maximum))
        if self.yx_region is not None:
            if len(self.yx_region) != 4:
                raise ValueError("yx_region must contain y0, y1, x0, and x1.")
            region = tuple(int(value) for value in self.yx_region)
            if any(value < 0 for value in region):
                raise ValueError("yx_region coordinates must be non-negative.")
            y0, y1, x0, x1 = region
            if y1 <= y0 or x1 <= x0:
                raise ValueError("yx_region must have positive Y and X extents.")
            object.__setattr__(self, "yx_region", region)
        object.__setattr__(
            self,
            "axis_declaration",
            AxisDeclaration.from_value(self.axis_declaration),
        )


@dataclass(frozen=True, slots=True)
class SourcePreviewReadMetrics:
    """Truthful measured or explicitly estimated I/O facts for one preview."""

    requested_decoded_bytes: int
    estimated_decoded_bytes_read: int | None = None
    estimated_objects_read: int | None = None
    basis: str = ""

    def __post_init__(self) -> None:
        for name in (
            "requested_decoded_bytes",
            "estimated_decoded_bytes_read",
            "estimated_objects_read",
        ):
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError(f"source preview metric {name} must be non-negative.")
            object.__setattr__(self, name, int(value))
        if (
            self.estimated_decoded_bytes_read is not None
            or self.estimated_objects_read is not None
        ) and not str(self.basis).strip():
            raise ValueError("estimated source preview metrics require a basis.")
        object.__setattr__(self, "basis", str(self.basis).strip())


@dataclass(frozen=True, slots=True)
class SourcePreviewResult:
    """A presentation-only array that cannot be mistaken for analysis data."""

    data: np.ndarray
    image_state: ImageState
    preview_level: int
    level_count: int
    message: str
    metrics: SourcePreviewReadMetrics
    generation: int
    analysis_level: int = 0
    presentation_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.image_state, ImageState):
            raise TypeError("source preview image_state must be an ImageState.")
        if not isinstance(self.metrics, SourcePreviewReadMetrics):
            raise TypeError(
                "source preview metrics must be a SourcePreviewReadMetrics."
            )
        data = np.asarray(self.data)
        if tuple(data.shape) != tuple(self.image_state.shape):
            raise ValueError("source preview data and ImageState shapes must agree.")
        preview_level = int(self.preview_level)
        level_count = int(self.level_count)
        if level_count <= 0 or preview_level < 0 or preview_level >= level_count:
            raise ValueError("source preview level must identify a declared level.")
        if isinstance(self.analysis_level, bool) or int(self.analysis_level) != 0:
            raise ValueError("0.14.0a1 analysis is fixed to OME-Zarr level 0.")
        if self.presentation_only is not True:
            raise ValueError("source preview results are presentation-only.")
        if isinstance(self.generation, bool) or int(self.generation) < 0:
            raise ValueError("source preview generation must be non-negative.")
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "preview_level", preview_level)
        object.__setattr__(self, "level_count", level_count)
        object.__setattr__(self, "generation", int(self.generation))
        object.__setattr__(self, "message", str(self.message))

    @property
    def uses_lower_resolution(self) -> bool:
        return self.preview_level > self.analysis_level

    @property
    def has_declared_lower_resolution(self) -> bool:
        return self.level_count > 1

    @property
    def preserves_label_semantics(self) -> bool:
        return self.image_state.kind == "label image"

    @property
    def intensity_statistics_allowed(self) -> bool:
        """Labels must never enter intensity-statistics preview paths."""
        return not self.preserves_label_semantics


def _positive_pair(value: tuple[int, int], label: str) -> tuple[int, int]:
    if len(value) != 2:
        raise ValueError(f"{label} must contain exactly two values.")
    pair = tuple(int(item) for item in value)
    if any(item <= 0 for item in pair):
        raise ValueError(f"{label} values must be positive.")
    return pair
