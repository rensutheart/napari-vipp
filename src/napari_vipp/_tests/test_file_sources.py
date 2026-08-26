from __future__ import annotations

import os
import threading
from dataclasses import replace

import numpy as np
import pytest
from ome_zarr.format import FormatV05
from ome_zarr.writer import write_image as write_ome_zarr_image

from napari_vipp.core import file_sources
from napari_vipp.core.file_sources import (
    FILE_SOURCE_SNAPSHOT_POLICY,
    load_frozen_file_source_snapshot,
)
from napari_vipp.core.host_memory import HostMemorySnapshot, HostMemorySource
from napari_vipp.core.io import (
    ImageDataset,
    ImageSeriesInfo,
    SourceInspection,
    inspect_image_source,
    inspect_image_state,
)
from napari_vipp.core.metadata import (
    AcquisitionMetadata,
    AxisDeclaration,
    AxisMetadata,
    ChannelMetadata,
    SourceMetadata,
    apply_axis_declaration,
    image_state_from_array,
)
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import (
    SourceChangedError,
    capture_local_source_bundle,
    capture_local_source_identity,
)
from napari_vipp.core.source_resolution import resolve_source_item


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
        "vipp_source_series_index": 2,
        "vipp_source_item_key": "series-2",
        "vipp_source_snapshot_policy": FILE_SOURCE_SNAPSHOT_POLICY,
        "vipp_source_item_digest": snapshot.source_item.digest,
        "vipp_source_item": snapshot.source_item.to_public_dict(),
    }
    assert snapshot.payload.source_item is snapshot.source_item
    assert str(source.resolve()) not in str(
        snapshot.payload.metadata["vipp_source_item"]
    )
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


def test_frozen_file_snapshot_cancels_during_initial_full_hash(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"a" * (3 * 1024 * 1024))
    cancel_event = threading.Event()
    progress = []
    reader_called = False

    def reader(path, *, series_index=0):
        nonlocal reader_called
        reader_called = True
        return _dataset(path, np.ones((2, 3), dtype=np.uint8))

    def on_progress(current, total, message):
        progress.append((current, total, message))
        if "Source validation 1/3" in message and current >= 1024 * 1024:
            cancel_event.set()

    with pytest.raises(OperationCancelled, match="validating a source identity"):
        load_frozen_file_source_snapshot(
            source,
            0,
            reader=reader,
            cancel_callback=cancel_event.is_set,
            progress_callback=on_progress,
        )

    assert progress
    assert not reader_called


def test_frozen_file_snapshot_cancels_during_chunked_materialization(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable scientific bytes")
    backing = np.arange(64, dtype=np.uint8).reshape(8, 8)
    cancel_event = threading.Event()
    progress = []
    monkeypatch.setattr(file_sources, "_MATERIALIZE_CHUNK_BYTES", 16)

    def on_progress(current, total, message):
        progress.append((current, total, message))
        if (
            "Source materialization 2/3" in message
            and 0 < current < total
        ):
            cancel_event.set()

    with pytest.raises(OperationCancelled, match="materializing image data"):
        load_frozen_file_source_snapshot(
            source,
            0,
            reader=lambda path, *, series_index=0: _dataset(path, backing),
            cancel_callback=cancel_event.is_set,
            progress_callback=on_progress,
        )

    materialization_updates = [
        update
        for update in progress
        if "Source materialization 2/3" in update[2]
    ]
    assert materialization_updates[0][:2] == (0, backing.nbytes)
    assert 0 < materialization_updates[-1][0] < backing.nbytes


def test_frozen_file_snapshot_cancels_during_full_reverification(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable scientific bytes")
    cancel_event = threading.Event()
    progress = []
    reader_called = False

    def reader(path, *, series_index=0):
        nonlocal reader_called
        reader_called = True
        return _dataset(path, np.ones((2, 3), dtype=np.uint8))

    def on_progress(current, total, message):
        progress.append((current, total, message))
        if "Source reverification 3/3" in message:
            cancel_event.set()

    with pytest.raises(OperationCancelled, match="validating a source identity"):
        load_frozen_file_source_snapshot(
            source,
            0,
            reader=reader,
            cancel_callback=cancel_event.is_set,
            progress_callback=on_progress,
        )

    assert reader_called
    assert any("Source materialization 2/3" in item[2] for item in progress)
    assert any("Source reverification 3/3" in item[2] for item in progress)


def test_saved_source_item_rebinds_by_key_after_series_order_reversal(tmp_path):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable scientific bytes")
    data_by_key = {
        "scene-a": np.full((2, 3), 1, dtype=np.uint16),
        "scene-b": np.full((2, 3), 7, dtype=np.uint16),
    }

    def inspection(order):
        series = tuple(
            ImageSeriesInfo(
                index=index,
                key=key,
                name=key,
                shape=(2, 3),
                dtype="uint16",
                axes="YX",
                reader_key="fixture-reader",
                reader_version="1.0",
            )
            for index, key in enumerate(order)
        )
        return SourceInspection(str(source), "fixture-format", series)

    forward = inspection(("scene-a", "scene-b"))
    reversed_items = inspection(("scene-b", "scene-a"))

    def reader_for(current):
        def read(path, *, series_index=0):
            selected = current.series[series_index]
            dataset = _dataset(
                path,
                data_by_key[selected.key],
                name=selected.name,
            )
            return replace(
                dataset,
                inspection=current,
                selected_series=selected,
            )

        return read

    first = load_frozen_file_source_snapshot(
        source,
        1,
        reader=reader_for(forward),
    )
    reopened = load_frozen_file_source_snapshot(
        source,
        1,
        expected_source_item=first.source_item,
        reader=reader_for(reversed_items),
        inspector=lambda _path: reversed_items,
    )

    assert reopened.source_item == first.source_item
    assert reopened.payload.metadata["vipp_source_series_index"] == 0
    assert reopened.payload.metadata["vipp_source_item_key"] == "scene-b"
    np.testing.assert_array_equal(reopened.payload.data, data_by_key["scene-b"])


def test_inspected_qyx_source_item_survives_full_ome_zarr_load(tmp_path):
    source = tmp_path / "unknown-axis.ome.zarr"
    values = np.arange(3 * 16 * 20, dtype=np.uint16).reshape(3, 16, 20)
    write_ome_zarr_image(
        values,
        str(source),
        fmt=FormatV05(),
        axes=(
            {"name": "q", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ),
        scale={"q": 2.0, "y": 0.4, "x": 0.4},
        scale_factors=(),
    )
    declaration = AxisDeclaration("QYX", "ZYX")
    inspection = inspect_image_source(source)
    raw_state = inspect_image_state(
        source,
        inspection=inspection,
        series_index=0,
    )
    effective_state = apply_axis_declaration(
        raw_state,
        declaration,
        declaration_source="Image Source",
    )
    saved = resolve_source_item(
        capture_local_source_bundle(source),
        inspection,
        series_index=0,
        image_state=effective_state,
        axis_declaration=declaration,
    )
    saved_digest = saved.digest

    loaded = load_frozen_file_source_snapshot(
        source,
        expected_source_item=saved,
    )

    assert loaded.source_item is saved
    assert loaded.source_item.digest == saved_digest
    assert loaded.payload.source_item is saved
    assert loaded.payload.image_state.axis_order == "ZYX"
    np.testing.assert_array_equal(loaded.payload.data, values)


def test_source_memory_preflight_refuses_before_eager_reader_opens(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable scientific bytes")
    selected = ImageSeriesInfo(
        index=0,
        key="image",
        name="large image",
        shape=(100, 100),
        dtype="uint16",
        axes="YX",
        reader_key="fixture-reader",
        estimated_decoded_bytes=20_000,
        capabilities=("decoded_size_estimate",),
    )
    inspection = SourceInspection(
        str(source),
        "fixture-format",
        (selected,),
    )
    reader_called = False

    def reader(path, *, series_index=0):
        nonlocal reader_called
        reader_called = True
        return _dataset(path, np.zeros((100, 100), dtype=np.uint16))

    monkeypatch.setattr(file_sources, "_MEMORY_PREFLIGHT_MIN_BYTES", 1)
    monkeypatch.setattr(
        file_sources,
        "capture_host_memory",
        lambda: HostMemorySnapshot(
            platform="linux",
            source=HostMemorySource.LINUX_PROC_MEMINFO,
            physical_total_bytes=32_000,
            physical_available_bytes=24_000,
        ),
    )

    with pytest.raises(MemoryError, match="memory preflight refused"):
        load_frozen_file_source_snapshot(
            source,
            inspector=lambda _path: inspection,
            reader=reader,
        )

    assert not reader_called


def test_saved_source_item_memory_preflight_runs_with_injected_reader(
    tmp_path,
    monkeypatch,
):
    source = tmp_path / "source.fake"
    source.write_bytes(b"stable scientific bytes")
    selected = ImageSeriesInfo(
        index=0,
        key="image",
        name="large image",
        shape=(100, 100),
        dtype="uint16",
        axes="YX",
        reader_key="fixture-reader",
        estimated_decoded_bytes=20_000,
        capabilities=("decoded_size_estimate",),
    )
    inspection = SourceInspection(str(source), "fixture-format", (selected,))
    state = _dataset(source, np.zeros((100, 100), dtype=np.uint16)).image_state
    source_item = resolve_source_item(
        capture_local_source_bundle(source, source_format=inspection.format),
        inspection,
        item_key="image",
        image_state=state,
    )
    reader_called = False

    def reader(path, *, series_index=0):
        nonlocal reader_called
        reader_called = True
        return _dataset(path, np.zeros((100, 100), dtype=np.uint16))

    monkeypatch.setattr(file_sources, "_MEMORY_PREFLIGHT_MIN_BYTES", 1)
    monkeypatch.setattr(
        file_sources,
        "capture_host_memory",
        lambda: HostMemorySnapshot(
            platform="linux",
            source=HostMemorySource.LINUX_PROC_MEMINFO,
            physical_total_bytes=32_000,
            physical_available_bytes=24_000,
        ),
    )

    with pytest.raises(MemoryError, match="memory preflight refused"):
        load_frozen_file_source_snapshot(
            source,
            expected_source_item=source_item,
            reader=reader,
        )

    assert not reader_called
