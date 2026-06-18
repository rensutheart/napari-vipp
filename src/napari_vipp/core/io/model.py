"""Normalized headless image I/O records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from napari_vipp.core.metadata import ImageState


@dataclass(frozen=True)
class AnalysisLabel:
    """One label image to package with a reference OME-Zarr image."""

    name: str
    data: Any
    image_state: ImageState | None = None
    source_node_id: str = ""


@dataclass(frozen=True)
class ImageSeriesInfo:
    """One selectable image or series inside a source container."""

    index: int
    key: str
    name: str
    shape: tuple[int, ...]
    dtype: str
    axes: str
    kind: str = "image"

    @property
    def label(self) -> str:
        name = self.name or f"Series {self.index + 1}"
        dimensions = " x ".join(str(size) for size in self.shape)
        return f"{name} | {self.axes}: {dimensions} | {self.dtype}"


@dataclass(frozen=True)
class SourceInspection:
    """Structure discovered without loading a complete source image."""

    uri: str
    format: str
    series: tuple[ImageSeriesInfo, ...]
    original_metadata: Any = None


@dataclass(frozen=True)
class ImageDataset:
    """One complete image item plus normalized and original metadata."""

    data: Any
    image_state: ImageState
    inspection: SourceInspection
    selected_series: ImageSeriesInfo
    original_metadata: Any = None
    multiscale_levels: tuple[Any, ...] = ()
    associated_labels: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
