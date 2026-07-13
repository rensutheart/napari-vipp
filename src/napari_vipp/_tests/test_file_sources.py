from __future__ import annotations

import os
from dataclasses import replace

import numpy as np
import pytest

from napari_vipp.core.file_sources import (
    FILE_SOURCE_SNAPSHOT_POLICY,
    load_frozen_file_source_snapshot,
)
from napari_vipp.core.io import (
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
)
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisMetadata,
    ChannelMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.source_identity import (
    SourceChangedError,
    capture_local_source_identity,
)


def _dataset(path, data, *, name: str = "Selected series") -> ImageDataset:
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("y", "space", unit="micrometer", scale=0.5),
            AxisMetadata("x", "space", unit="micrometer", scale=0.25),
        ),
        metadata_source="verified reader metadata",
        source_name=name,
        history=("reader normalized source",),
        channels=(ChannelMetadata(name="DNA", fluor="DAPI"),),
        acquisition=AcquisitionMetadata(
            objective="Plan Apo",
            objective_na=1.4,
        ),
        source=SourceMetadata(
            uri=str(path),
            format="test-format",
            series_index=2,
            series_name=name,
            source_uuid="source-uuid",
        ),
    )
    assert state is not None
    state = replace(state, kind="label image")
    series = ImageSeriesInfo(
        2,
        "series-2",
        name,
        tuple(data.shape),
        str(data.dtype),
        "YX",
        kind="labels",
    )
    inspection = SourceInspection(
        str(path),
        "test-format",
        (series,),
        original_metadata={"scientific": "metadata"},
    )
    return ImageDataset(
        data,
        state,
        inspection,
        series,
        provenance={"reader": "test-reader"},
    )


def test_frozen_file_snapshot_is_owned_read_only_and_preserves_state(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable scientific bytes")
    backing = np.arange(20, dtype=np.uint16).reshape(4, 5)
    calls = []

    def reader(path, *, series_index=0):
        calls.append((path, series_index))
        return _dataset(path, backing)

    snapshot = load_frozen_file_source_snapshot(
        source,
        2,
        reader=reader,
    )

    frozen = snapshot.payload.data
    assert calls == [(source.resolve(), 2)]
    assert isinstance(frozen, np.ndarray)
    assert frozen.flags.owndata
    assert not frozen.flags.writeable
    assert not np.shares_memory(frozen, backing)
    np.testing.assert_array_equal(frozen, backing)
    assert snapshot.payload.revision_token is snapshot.identity
    assert snapshot.payload.metadata == {
        "vipp_source_path": str(source.resolve()),
        "vipp_source_identity": snapshot.identity.to_dict(),
        "vipp_source_snapshot_policy": FILE_SOURCE_SNAPSHOT_POLICY,
    }
    assert snapshot.inspection.original_metadata == {"scientific": "metadata"}

    state = snapshot.payload.image_state
    assert state is not None
    assert state.kind == "label image"
    assert state.metadata_source == "verified reader metadata"
    assert state.source_name == "Selected series"
    assert state.history == ("reader normalized source",)
    assert state.channels == (ChannelMetadata(name="DNA", fluor="DAPI"),)
    assert state.acquisition.objective == "Plan Apo"
    assert state.acquisition.objective_na == 1.4
    assert state.source.format == "test-format"
    assert state.source.source_uuid == "source-uuid"
    assert state.value_range == "0 to 19"

    expected = frozen.copy()
    backing[:] = 0
    np.testing.assert_array_equal(frozen, expected)
    with pytest.raises(ValueError, match="read-only"):
        frozen[0, 0] = 1


def test_frozen_file_snapshot_rejects_expected_revision_mismatch(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"revision-A")
    expected = capture_local_source_identity(source)
    source.write_bytes(b"revision-B")
    reader_called = False

    def reader(path, *, series_index=0):
        nonlocal reader_called
        reader_called = True
        data = np.ones((2, 3), dtype=np.uint8)
        return _dataset(path, data)

    with pytest.raises(SourceChangedError, match="Press Refresh"):
        load_frozen_file_source_snapshot(
            source,
            0,
            expected_identity=expected,
            reader=reader,
        )

    assert not reader_called


def test_frozen_file_snapshot_rejects_directory_mutation_during_read(tmp_path):
    source = tmp_path / "source.zarr"
    source.mkdir()
    chunk = source / "chunk.bin"
    chunk.write_bytes(b"chunk-A")
    root_stat = source.stat()

    def mutating_reader(path, *, series_index=0):
        chunk.write_bytes(b"chunk-B")
        os.utime(
            source,
            ns=(root_stat.st_atime_ns, root_stat.st_mtime_ns),
        )
        data = np.ones((2, 3), dtype=np.uint8)
        return _dataset(path, data)

    with pytest.raises(SourceChangedError, match="changed during execution"):
        load_frozen_file_source_snapshot(
            source,
            0,
            reader=mutating_reader,
        )

    assert source.stat().st_mtime_ns == root_stat.st_mtime_ns
    assert source.stat().st_size == root_stat.st_size
