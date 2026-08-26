from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from napari_vipp.core.io import numpy_io
from napari_vipp.core.io.model import ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import AxisMetadata, image_state_from_array
from napari_vipp.core.source_identity import SourceChangedError
from napari_vipp.core.source_inspection import (
    SourceInspectionPhase,
    inspect_local_source_item,
)


def test_metadata_inspection_resolves_npz_without_reading_pixels(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "arrays.npz"
    np.savez_compressed(
        path,
        thumbnail=np.zeros((8, 10), dtype=np.uint8),
        volume=np.zeros((2, 5, 6), dtype=np.uint16),
    )
    monkeypatch.setattr(
        numpy_io.np,
        "load",
        lambda *_args, **_kwargs: pytest.fail("metadata inspection decoded pixels"),
    )
    progress = []

    resolved = inspect_local_source_item(
        path,
        item_key="volume",
        progress_callback=progress.append,
    )

    assert resolved.selected_series.key == "volume"
    assert resolved.selected_series.index == 1
    assert resolved.effective_image_state.axis_order == "ZYX"
    assert resolved.source_item.selector.key == "volume"
    assert resolved.source_item.resolved.shape == (2, 5, 6)
    phases = [update.phase for update in progress]
    assert phases[0] is SourceInspectionPhase.IDENTITY
    assert SourceInspectionPhase.HEADER in phases
    assert SourceInspectionPhase.NORMALIZE in phases
    assert phases[-1] is SourceInspectionPhase.VERIFY
    assert phases.index(SourceInspectionPhase.HEADER) < phases.index(
        SourceInspectionPhase.NORMALIZE
    )


def test_saved_source_item_key_survives_inspection_order_change(tmp_path) -> None:
    path = tmp_path / "container.fake"
    path.write_bytes(b"stable scientific bytes")

    first = _fake_inspection(path, ((0, "image"), (1, "labels/cells")))
    resolved = inspect_local_source_item(
        path,
        series_index=1,
        item_key="labels/cells",
        inspector=lambda _path: first,
        state_inspector=_fake_state_inspector,
    )
    second = _fake_inspection(path, ((0, "labels/cells"), (1, "image")))

    reopened = inspect_local_source_item(
        path,
        series_index=1,
        expected_source_item=resolved.source_item,
        inspector=lambda _path: second,
        state_inspector=_fake_state_inspector,
    )

    assert reopened.selected_series.key == "labels/cells"
    assert reopened.selected_series.index == 0
    assert reopened.source_item.selector.key == "labels/cells"
    assert reopened.source_item.reader == resolved.source_item.reader
    assert reopened.source_item.resolved == resolved.source_item.resolved


def test_metadata_inspection_rejects_source_mutation_before_publication(
    tmp_path,
) -> None:
    path = tmp_path / "changing.fake"
    path.write_bytes(b"before")
    inspection = _fake_inspection(path, ((0, "image"),))

    def mutate_then_normalize(source, *, inspection, series_index):
        source.write_bytes(b"after")
        return _fake_state_inspector(
            source,
            inspection=inspection,
            series_index=series_index,
        )

    with pytest.raises(SourceChangedError, match="changed during execution"):
        inspect_local_source_item(
            path,
            inspector=lambda _path: inspection,
            state_inspector=mutate_then_normalize,
        )


def _fake_inspection(
    path: Path,
    ordered_items: tuple[tuple[int, str], ...],
) -> SourceInspection:
    series = tuple(
        ImageSeriesInfo(
            index=index,
            key=key,
            name=key,
            shape=(8, 10),
            dtype="uint16",
            axes="YX",
            kind="labels" if key.startswith("labels/") else "image",
            reader_key="fixture-reader",
            reader_version="1.0",
            capabilities=("pixel_lazy_inspection", "decoded_size_estimate"),
            estimated_decoded_bytes=160,
            level_shapes=((8, 10),),
        )
        for index, key in ordered_items
    )
    return SourceInspection(str(path), "fixture", series)


def _fake_state_inspector(
    _path,
    *,
    inspection: SourceInspection,
    series_index: int,
):
    selected = next(item for item in inspection.series if item.index == series_index)
    data = np.zeros(selected.shape, dtype=np.dtype(selected.dtype))
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("y", "space", source_axis=0),
            AxisMetadata("x", "space", source_axis=1),
        ),
        metadata_source="fixture header",
    )
    assert state is not None
    if selected.kind == "labels":
        from dataclasses import replace

        state = replace(state, kind="label image")
    return state
