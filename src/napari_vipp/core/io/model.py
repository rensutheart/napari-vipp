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
    image_state: ImageState | None = None
    reader_key: str = ""
    reader_version: str = ""
    capabilities: tuple[str, ...] = ()
    estimated_decoded_bytes: int | None = None
    level_shapes: tuple[tuple[int, ...], ...] = ()
    analysis_chunk_grid: tuple[tuple[int, ...], ...] = ()

    @property
    def label(self) -> str:
        name = self.name or f"Series {self.index + 1}"
        dimensions = " x ".join(str(size) for size in self.shape)
        details = [f"{self.axes}: {dimensions}", self.dtype, self.kind]
        if self.reader_key:
            details.append(self.reader_key)
        if len(self.level_shapes) > 1:
            details.append(f"{len(self.level_shapes)} levels")
        return f"{name} | " + " | ".join(details)


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
