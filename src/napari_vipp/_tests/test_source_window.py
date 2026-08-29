from __future__ import annotations

import numpy as np
import pytest
from ome_zarr.format import FormatV04, FormatV05
from ome_zarr.writer import write_image as write_ome_zarr_image

import napari_vipp.core.file_sources as file_sources
import napari_vipp.core.io.ome_zarr as ome_zarr_io
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.host_memory import HostMemorySnapshot, HostMemorySource
from napari_vipp.core.io import (
    inspect_image_source,
    inspect_image_state,
    read_image_exact_window,
)
from napari_vipp.core.io.ome_zarr import read_ome_zarr_exact_window
from napari_vipp.core.metadata import AxisDeclaration
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.progress import OperationCancelled
from napari_vipp.core.source_identity import capture_local_source_bundle
from napari_vipp.core.source_resolution import resolve_source_item
from napari_vipp.core.source_window import (
    ExactSourceWindowData,
    SourceWindowControl,
    SourceWindowReadEstimate,
    SourceWindowRequest,
)
from napari_vipp.core.source_window_planning import plan_exact_source_crop_window


def _write_tczyx(path, fmt) -> np.ndarray:
    data = np.arange(2 * 3 * 4 * 24 * 30, dtype=np.uint16).reshape(
        2,
        3,
        4,
        24,
        30,
    )
    write_ome_zarr_image(
        data,
        str(path),
        fmt=fmt,
        axes=(
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ),
        scale={"t": 1, "c": 1, "z": 2, "y": 0.4, "x": 0.25},
        scale_factors=(),
        storage_options={"chunks": (1, 1, 2, 8, 10)},
    )
    return data


@pytest.mark.parametrize("fmt", [FormatV04(), FormatV05()])
def test_exact_window_reads_owned_level_zero_pixels_and_shifted_state(tmp_path, fmt):
    path = tmp_path / f"window-{fmt.version}.ome.zarr"
    source = _write_tczyx(path, fmt)
    request = SourceWindowRequest(
        (
            slice(None),
            slice(None),
            slice(1, 3),
            slice(5, 20),
            slice(7, 27),
        ),
        source_revision="sha256:revision",
        source_item_digest="sha256:item",
    )

    result = read_image_exact_window(path, request=request)

    np.testing.assert_array_equal(result.data, source[request.selection])
    assert result.data.flags.owndata
    assert result.data.flags.c_contiguous
    assert not result.data.flags.writeable
    assert result.image_state.shape == (2, 3, 2, 15, 20)
    assert result.image_state.dtype == "uint16"
    assert tuple(axis.name for axis in result.image_state.axes) == (
        "t",
        "c",
        "z",
        "y",
        "x",
    )
    assert [axis.translation for axis in result.image_state.axes] == [
        0,
        0,
        pytest.approx(2.0),
        pytest.approx(2.0),
        pytest.approx(1.75),
    ]
    assert result.identity.analysis_level == 0
    assert result.identity.source_shape == source.shape
    assert result.identity.output_shape == result.data.shape
    assert result.identity.bounds == (
        (0, 2),
        (0, 3),
        (1, 3),
        (5, 20),
        (7, 27),
    )
    assert result.identity.reader_key == "ome-zarr"
    assert result.identity.reader_version != ""
    assert result.identity.source_revision == "sha256:revision"
    assert result.identity.source_item_digest == "sha256:item"
    estimate = result.identity.read_estimate
    assert estimate is not None
    assert estimate.requested_decoded_bytes == result.data.nbytes
    assert estimate.estimated_touched_chunk_decoded_bytes >= result.data.nbytes
    assert estimate.estimated_touched_chunk_count >= 1
    assert estimate.estimated_peak_bytes == (
        estimate.estimated_touched_chunk_decoded_bytes + 2 * result.data.nbytes
    )
    assert result.identity.to_dict()["read_estimate"] == estimate.to_dict()
    assert "exact_region_read" in result.selected_series.capabilities


def test_exact_window_preserves_complete_time_and_channels_by_default(tmp_path):
    path = tmp_path / "preserve-tc.ome.zarr"
    _write_tczyx(path, FormatV04())
    request = SourceWindowRequest(
        (
            slice(None),
            slice(1, 2),
            slice(None),
            slice(None),
            slice(None),
        )
    )

    with pytest.raises(ValueError, match="preserve complete T and C"):
        read_ome_zarr_exact_window(path, request=request)

    explicit_future_request = SourceWindowRequest(
        request.selection,
        preserve_time_and_channels=False,
    )
    result = read_ome_zarr_exact_window(path, request=explicit_future_request)
    assert result.data.shape == (2, 1, 4, 24, 30)


def test_exact_window_applies_explicit_axis_declaration_before_tc_guard(tmp_path):
    path = tmp_path / "declared-qyx.ome.zarr"
    data = np.arange(4 * 20 * 30, dtype=np.uint16).reshape(4, 20, 30)
    write_ome_zarr_image(
        data,
        str(path),
        fmt=FormatV04(),
        axes=(
            {"name": "q", "type": "unknown"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ),
        scale_factors=(),
    )
    request = SourceWindowRequest(
        (slice(1, 3), slice(2, 18), slice(4, 24)),
        axis_declaration=AxisDeclaration("QYX", "ZYX"),
    )

    result = read_ome_zarr_exact_window(path, request=request)

    np.testing.assert_array_equal(result.data, data[1:3, 2:18, 4:24])
    assert result.image_state.axis_order == "ZYX"
    assert result.identity.axis_names == ("z", "y", "x")
    assert result.image_state.axes[0].translation == pytest.approx(1.0)


def test_source_window_contract_rejects_rank_drops_strides_and_bad_bounds():
    with pytest.raises(TypeError, match="preserve rank"):
        SourceWindowRequest((slice(None), 1))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unit-step"):
        SourceWindowRequest((slice(None, None, 2),))
    with pytest.raises(ValueError, match="fixed to level 0"):
        SourceWindowRequest((slice(None),), analysis_level=1)

    request = SourceWindowRequest((slice(None), slice(2, 8)))
    with pytest.raises(ValueError, match="rank does not match"):
        request.normalized_selection((10,))
    with pytest.raises(IndexError, match="outside"):
        request.normalized_selection((10, 6))


def test_exact_window_checks_cancellation_before_reading_chunks(tmp_path):
    path = tmp_path / "cancelled.ome.zarr"
    _write_tczyx(path, FormatV04())
    control = SourceWindowControl(cancelled=lambda: True)

    with pytest.raises(OperationCancelled, match="source-window read cancelled"):
        read_image_exact_window(
            path,
            request=SourceWindowRequest(
                tuple(slice(None) for _ in range(5))
            ),
            control=control,
        )


def test_exact_window_registry_never_falls_back_for_unsupported_format(tmp_path):
    path = tmp_path / "image.npy"
    np.save(path, np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(ValueError, match="OME-Zarr 0.4/0.5"):
        read_image_exact_window(
            path,
            request=SourceWindowRequest((slice(None), slice(None))),
        )


def test_exact_loader_and_crop_match_ordinary_full_source_execution(tmp_path):
    path = tmp_path / "pipeline-parity.ome.zarr"
    source = _write_tczyx(path, FormatV05())
    inspection = inspect_image_source(path)
    full_state = inspect_image_state(path, inspection=inspection)
    source_item = resolve_source_item(
        capture_local_source_bundle(path),
        inspection,
        image_state=full_state,
    )

    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    crop = pipeline.add_node("crop_stack")
    assert pipeline.connect("input", crop.id).success
    for name, value in {
        "z_start": 1,
        "z_end": 1,
        "top": 5,
        "bottom": 4,
        "left": 7,
        "right": 3,
    }.items():
        pipeline.set_param(crop.id, name, value)

    decision = plan_exact_source_crop_window(
        pipeline,
        "input",
        source_item,
        full_state,
    )
    assert decision.plan is not None
    snapshot = load_frozen_file_source_snapshot(
        path,
        expected_source_item=source_item,
        exact_window_request=decision.plan.request,
    )
    assert isinstance(snapshot.payload.data, ExactSourceWindowData)
    assert snapshot.payload.data.shape == source.shape
    assert snapshot.payload.data.data.shape == (2, 3, 2, 15, 20)

    pipeline.run(None, source_payloads={"input": snapshot.payload})

    np.testing.assert_array_equal(
        pipeline.outputs[crop.id],
        source[:, :, 1:3, 5:20, 7:27],
    )
    output_state = pipeline.output_states[crop.id]
    assert output_state.shape == (2, 3, 2, 15, 20)
    assert tuple(axis.translation for axis in output_state.axes) == pytest.approx(
        (0.0, 0.0, 2.0, 2.0, 1.75)
    )
    assert snapshot.payload.metadata["vipp_source_read_strategy"] == (
        "exact-level-0-window"
    )
    assert snapshot.payload.metadata["vipp_source_window_read_estimate"] == (
        snapshot.payload.data.identity.read_estimate.to_dict()
    )


def test_pathological_large_chunk_estimate_dominates_tiny_roi():
    class HugeChunkArray:
        shape = (100_000, 100_000)
        dtype = np.dtype("uint16")
        chunks = ((100_000,), (100_000,))

    estimate = ome_zarr_io._exact_window_read_estimate(
        HugeChunkArray(),
        (slice(12, 13), slice(34, 35)),
    )

    assert estimate.requested_decoded_bytes == 2
    assert estimate.estimated_touched_chunk_count == 1
    assert estimate.estimated_touched_chunk_decoded_bytes == 20_000_000_000
    assert estimate.estimated_peak_bytes == 20_000_000_004


def test_exact_loader_refuses_pathological_touched_chunk_before_compute(
    tmp_path,
    monkeypatch,
):
    path = tmp_path / "preflight-window.ome.zarr"
    _write_tczyx(path, FormatV04())
    inspection = inspect_image_source(path)
    full_state = inspect_image_state(path, inspection=inspection)
    source_item = resolve_source_item(
        capture_local_source_bundle(path),
        inspection,
        image_state=full_state,
    )
    request = SourceWindowRequest(
        (
            slice(None),
            slice(None),
            slice(1, 2),
            slice(5, 6),
            slice(7, 8),
        )
    )
    requested_bytes = 2 * 3 * 1 * 1 * 1 * np.dtype("uint16").itemsize
    huge_estimate = SourceWindowReadEstimate(
        requested_decoded_bytes=requested_bytes,
        estimated_touched_chunk_decoded_bytes=4 * 1024**3,
        estimated_touched_chunk_count=1,
        estimated_peak_bytes=4 * 1024**3 + 2 * requested_bytes,
        basis="synthetic pathological single-chunk fixture",
    )
    compute_called = False

    def unexpected_compute(*_args, **_kwargs):
        nonlocal compute_called
        compute_called = True
        raise AssertionError("chunk computation must not start after preflight refusal")

    monkeypatch.setattr(
        ome_zarr_io,
        "_exact_window_read_estimate",
        lambda *_args, **_kwargs: huge_estimate,
    )
    monkeypatch.setattr(ome_zarr_io, "_compute_exact_window", unexpected_compute)
    monkeypatch.setattr(file_sources, "_MEMORY_PREFLIGHT_MIN_BYTES", 1)
    monkeypatch.setattr(
        file_sources,
        "capture_host_memory",
        lambda: HostMemorySnapshot(
            platform="linux",
            source=HostMemorySource.LINUX_PROC_MEMINFO,
            physical_total_bytes=8 * 1024**3,
            physical_available_bytes=2 * 1024**3,
        ),
    )

    with pytest.raises(MemoryError, match="intersecting decoded chunks"):
        load_frozen_file_source_snapshot(
            path,
            expected_source_item=source_item,
            exact_window_request=request,
        )

    assert not compute_called
