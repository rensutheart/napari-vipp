"""Verified, detached snapshots for local interactive image sources.

This module owns the scientific boundary between a mutable local file/store and
the pipeline. It deliberately contains no Qt behavior; scheduling and cache
policy remain responsibilities of the caller.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import numpy as np

from napari_vipp.core.io import (
    ImageDataset,
    SourceInspection,
    read_image,
)
from napari_vipp.core.metadata import image_state_from_array
from napari_vipp.core.pipeline import SourcePayload
from napari_vipp.core.source_identity import (
    LocalSourceIdentity,
    SourceChangedError,
    capture_local_source_identity,
    verify_local_source_identity,
)

FILE_SOURCE_SNAPSHOT_POLICY = "pinned until Refresh"


class ImageReader(Protocol):
    """Callable shape accepted by the frozen-source loader."""

    def __call__(
        self,
        path: str | Path,
        *,
        series_index: int = 0,
    ) -> ImageDataset: ...


@dataclass(frozen=True, slots=True)
class SourceFileSnapshot:
    """One exact local source revision detached for pipeline ownership."""

    payload: SourcePayload
    inspection: SourceInspection
    identity: LocalSourceIdentity


@dataclass(frozen=True, slots=True)
class VerifiedSourceInspection:
    """Source structure paired with the exact revision that was inspected."""

    inspection: SourceInspection
    identity: LocalSourceIdentity


def load_frozen_file_source_snapshot(
    path: str | Path,
    series_index: int,
    *,
    expected_identity: LocalSourceIdentity | None = None,
    reader: ImageReader | None = None,
) -> SourceFileSnapshot:
    """Read one exact local revision into an owned, read-only NumPy array.

    The complete file or directory tree is hashed before the reader opens it
    and verified again only after lazy data has been fully materialized. A
    caller may inject its reader; this keeps UI monkeypatching and alternate
    reader dispatch outside the core scientific boundary.
    """
    source = Path(path).expanduser().resolve(strict=False)
    identity = capture_local_source_identity(source)
    if expected_identity is not None and identity != expected_identity:
        raise SourceChangedError(
            "Local scientific source changed after its interactive snapshot "
            f"was pinned: {source}. Press Refresh to load the new revision."
        )

    selected_reader = read_image if reader is None else reader
    dataset = selected_reader(source, series_index=int(series_index))
    data = np.array(
        np.asarray(dataset.data),
        copy=True,
        order="K",
        subok=False,
    )
    data.setflags(write=False)
    verify_local_source_identity(source, identity)

    source_state = dataset.image_state
    snapshot_state = image_state_from_array(
        data,
        axes=source_state.axes,
        metadata_source=source_state.metadata_source,
        source_name=source_state.source_name,
        history=source_state.history,
        channels=source_state.channels,
        acquisition=source_state.acquisition,
        source=source_state.source,
    )
    if snapshot_state is not None:
        snapshot_state = replace(snapshot_state, kind=source_state.kind)

    payload = SourcePayload(
        data,
        {
            "vipp_source_path": str(source),
            "vipp_source_identity": identity.to_dict(),
            "vipp_source_snapshot_policy": FILE_SOURCE_SNAPSHOT_POLICY,
        },
        dataset.selected_series.name,
        snapshot_state,
        identity,
    )
    return SourceFileSnapshot(payload, dataset.inspection, identity)


__all__ = [
    "FILE_SOURCE_SNAPSHOT_POLICY",
    "ImageReader",
    "SourceFileSnapshot",
    "VerifiedSourceInspection",
    "load_frozen_file_source_snapshot",
]
