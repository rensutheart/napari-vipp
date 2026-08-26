from __future__ import annotations

import dask.array as da
import numpy as np
import pytest
import zarr
from dask.callbacks import Callback as DaskCallback
from ome_zarr.format import FormatV04, FormatV05
from ome_zarr.writer import write_image as write_ome_zarr_image
from ome_zarr.writer import write_labels as write_ome_zarr_labels

from napari_vipp.core.io.ome_zarr import (
    OmeZarrLevelInfo,
    _choose_preview_level,
    _preview_selection,
    enumerate_ome_zarr_levels,
    read_ome_zarr,
    read_ome_zarr_presentation_preview,
)
from napari_vipp.core.metadata import AxisDeclaration, AxisMetadata
from napari_vipp.core.source_preview import (
    SourcePreviewCancelled,
    SourcePreviewControl,
    SourcePreviewGenerationCoordinator,
    SourcePreviewRequest,
    StaleSourcePreviewGeneration,
)


def _write_multiscale(path, fmt, *, storage_options=None) -> np.ndarray:
    data = np.arange(2 * 3 * 4 * 64 * 80, dtype=np.uint16).reshape(
        2,
        3,
        4,
        64,
        80,
    )
    write_ome_zarr_image(
        data,
        str(path),
        fmt=fmt,
        axes=(
            {"name": "t", "type": "time"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ),
        scale={"t": 1, "c": 1, "z": 2, "y": 0.4, "x": 0.4},
        scale_factors=(2, 4),
        storage_options=storage_options,
    )
    return data


@pytest.mark.parametrize("fmt", [FormatV04(), FormatV05()])
def test_enumerates_declared_levels_transforms_and_chunks(tmp_path, fmt):
    path = tmp_path / f"levels-{fmt.version}.ome.zarr"
    _write_multiscale(path, fmt)

    levels = enumerate_ome_zarr_levels(path)

    assert [level.index for level in levels] == [0, 1, 2]
    assert [level.path for level in levels] == ["s0", "s1", "s2"]
    assert [level.shape for level in levels] == [
        (2, 3, 4, 64, 80),
        (2, 3, 4, 32, 40),
        (2, 3, 4, 16, 20),
    ]
    assert tuple(axis.name for axis in levels[0].axes) == (
        "t",
        "c",
        "z",
        "y",
        "x",
    )
    assert [axis.scale for axis in levels[2].axes] == [1, 1, 2, 1.6, 1.6]
    assert [axis.translation for axis in levels[2].axes] == [
        0,
        0,
        0,
        pytest.approx(0.6),
        pytest.approx(0.6),
    ]
    assert [item.type for item in levels[1].coordinate_transformations] == [
        "scale",
        "translation",
    ]
    assert levels[0].chunk_shape is not None
    assert levels[0].estimated_decoded_bytes == 2 * 3 * 4 * 64 * 80 * 2


def test_enumerates_multiscale_scope_identity_transform(tmp_path):
    path = tmp_path / "global-transform.ome.zarr"
    _write_multiscale(path, FormatV05())
    root = zarr.open_group(str(path), mode="a")
    attrs = root.attrs.asdict()
    ome = attrs["ome"]
    ome["multiscales"][0]["coordinateTransformations"] = [{"type": "identity"}]
    root.attrs["ome"] = ome

    levels = enumerate_ome_zarr_levels(path)

    assert all(
        level.coordinate_transformations[-1].type == "identity"
        and level.coordinate_transformations[-1].scope == "multiscale"
        for level in levels
    )
    assert [axis.scale for axis in levels[2].axes] == [1, 1, 2, 1.6, 1.6]


def test_sparse_pyramid_never_falls_back_to_enormous_analysis_level():
    axes = (
        AxisMetadata("y", "space"),
        AxisMetadata("x", "space"),
    )
    levels = (
        OmeZarrLevelInfo(0, "0", (100_000, 100_000), "uint16", axes, (), ()),
        OmeZarrLevelInfo(1, "1", (256, 256), "uint16", axes, (), ()),
    )

    selected = _choose_preview_level(
        levels,
        SourcePreviewRequest(display_shape_yx=(512, 512)),
    )

    assert selected.index == 1


@pytest.mark.parametrize(
    ("axis_name", "axis_type", "request_field"),
    [
        ("t", "time", "t_index"),
        ("z", "space", "z_index"),
        ("c", "channel", "c_index"),
    ],
)
def test_indexed_axes_map_from_analysis_coordinates_into_selected_level(
    axis_name,
    axis_type,
    request_field,
):
    analysis_axes = (
        AxisMetadata(axis_name, axis_type, scale=2.0),
        AxisMetadata("y", "space"),
        AxisMetadata("x", "space"),
    )
    preview_axes = (
        AxisMetadata(axis_name, axis_type, scale=4.0, translation=1.0),
        AxisMetadata("y", "space", scale=2.0, translation=0.5),
        AxisMetadata("x", "space", scale=2.0, translation=0.5),
    )
    levels = (
        OmeZarrLevelInfo(0, "0", (6, 64, 80), "uint16", analysis_axes, (), ()),
        OmeZarrLevelInfo(1, "1", (3, 32, 40), "uint16", preview_axes, (), ()),
    )
    request_args = {request_field: 4}
    automatic = SourcePreviewRequest(**request_args)
    explicit = SourcePreviewRequest(level=1, **request_args)

    assert _choose_preview_level(levels, automatic).index == 1
    assert _choose_preview_level(levels, explicit).index == 1
    assert _preview_selection(levels, levels[1], explicit)[0] == 2


def test_auto_preview_never_reads_level_zero_when_lower_levels_do_not_cover_plane():
    analysis_axes = (
        AxisMetadata("z", "space"),
        AxisMetadata("y", "space"),
        AxisMetadata("x", "space"),
    )
    displaced_axes = (
        AxisMetadata("z", "space", translation=100.0),
        AxisMetadata("y", "space", scale=2.0, translation=0.5),
        AxisMetadata("x", "space", scale=2.0, translation=0.5),
    )
    levels = (
        OmeZarrLevelInfo(0, "0", (4, 64, 80), "uint16", analysis_axes, (), ()),
        OmeZarrLevelInfo(1, "1", (2, 32, 40), "uint16", displaced_axes, (), ()),
    )

    with pytest.raises(IndexError, match="No declared lower-resolution level"):
        _choose_preview_level(levels, SourcePreviewRequest(z_index=3))
    with pytest.raises(IndexError, match="no corresponding pixel"):
        _choose_preview_level(
            levels,
            SourcePreviewRequest(z_index=3, level=1),
        )


def test_preview_applies_reviewed_positional_axis_declaration_before_compute(
    tmp_path,
):
    path = tmp_path / "unknown-axis.ome.zarr"
    data = np.arange(3 * 64 * 80, dtype=np.uint16).reshape(3, 64, 80)
    write_ome_zarr_image(
        data,
        str(path),
        fmt=FormatV05(),
        axes=(
            {"name": "q", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ),
        scale_factors=(2, 4),
    )

    result = read_ome_zarr_presentation_preview(
        path,
        request=SourcePreviewRequest(
            display_shape_yx=(8, 10),
            z_index=2,
            axis_declaration=AxisDeclaration("QYX", "ZYX"),
        ),
    )

    assert result.preview_level > 0
    assert result.data.ndim == 2
    assert result.image_state.axis_order == "YX"
    assert "explicit axis declaration" in result.image_state.metadata_source


@pytest.mark.parametrize("fmt", [FormatV04(), FormatV05()])
def test_lower_level_slices_tzc_and_region_before_compute(
    tmp_path,
    fmt,
    monkeypatch,
):
    path = tmp_path / f"bounded-{fmt.version}.ome.zarr"
    _write_multiscale(path, fmt)
    loaded = read_ome_zarr(path)
    expected = loaded.multiscale_levels[2][1, 2, 3, 2:14, 2:18].compute()

    computed_shapes: list[tuple[int, ...]] = []
    original_compute = da.Array.compute

    def tracked_compute(self, *args, **kwargs):
        computed_shapes.append(tuple(int(size) for size in self.shape))
        return original_compute(self, *args, **kwargs)

    monkeypatch.setattr(da.Array, "compute", tracked_compute)
    result = read_ome_zarr_presentation_preview(
        path,
        request=SourcePreviewRequest(
            display_shape_yx=(8, 10),
            t_index=1,
            c_index=2,
            z_index=3,
            yx_region=(8, 56, 10, 70),
        ),
    )

    assert computed_shapes == [(12, 16)]
    np.testing.assert_array_equal(result.data, expected)
    assert result.preview_level == 2
    assert result.analysis_level == 0
    assert result.presentation_only
    assert result.message == ("Preview level 2 - analysis remains full resolution")
    assert result.image_state.axis_order == "YX"
    assert result.image_state.axes[0].translation == pytest.approx(3.8)
    assert result.image_state.axes[1].translation == pytest.approx(3.8)
    assert result.metrics.requested_decoded_bytes == 12 * 16 * 2
    assert result.metrics.estimated_objects_read == 1
    assert "estimated" in result.metrics.basis


def test_region_preview_reads_only_intersecting_lower_level_chunks(tmp_path):
    path = tmp_path / "chunked.ome.zarr"
    _write_multiscale(
        path,
        FormatV05(),
        storage_options={"chunks": (1, 1, 1, 8, 8)},
    )

    result = read_ome_zarr_presentation_preview(
        path,
        request=SourcePreviewRequest(
            display_shape_yx=(12, 20),
            t_index=0,
            c_index=0,
            z_index=0,
            yx_region=(16, 48, 16, 64),
        ),
    )

    assert result.preview_level == 1
    assert result.data.shape == (16, 24)
    assert result.metrics.estimated_objects_read == 6
    assert result.metrics.requested_decoded_bytes == 16 * 24 * 2
    assert result.metrics.estimated_decoded_bytes_read == 16 * 24 * 2
    assert result.metrics.estimated_decoded_bytes_read < (
        enumerate_ome_zarr_levels(path)[1].estimated_decoded_bytes
    )


@pytest.mark.parametrize("level", [1, 2])
def test_explicit_preview_level_reads_the_requested_declared_level(
    tmp_path,
    level,
):
    path = tmp_path / f"explicit-level-{level}.ome.zarr"
    _write_multiscale(path, FormatV05())
    loaded = read_ome_zarr(path)
    expected = loaded.multiscale_levels[level][1, 2, 3].compute()

    result = read_ome_zarr_presentation_preview(
        path,
        request=SourcePreviewRequest(
            t_index=1,
            c_index=2,
            z_index=3,
            level=level,
        ),
    )

    assert result.preview_level == level
    np.testing.assert_array_equal(result.data, expected)


def test_explicit_preview_level_rejects_undeclared_level(tmp_path):
    path = tmp_path / "explicit-level-invalid.ome.zarr"
    _write_multiscale(path, FormatV05())

    with pytest.raises(IndexError, match="outside 0..2"):
        read_ome_zarr_presentation_preview(
            path,
            request=SourcePreviewRequest(level=3),
        )


def test_z_downsampled_level_reads_physically_corresponding_plane(tmp_path):
    path = tmp_path / "z-downsampled.ome.zarr"
    _write_multiscale(path, FormatV05())
    root = zarr.open_group(str(path), mode="a")
    del root["s2"]
    del root["s1"]
    target = np.arange(2 * 3 * 2 * 32 * 40, dtype=np.uint16).reshape(
        2,
        3,
        2,
        32,
        40,
    )
    target += 10_000
    root.create_array(
        "s1",
        data=target,
        chunks=(1, 1, 1, 16, 20),
    )
    attributes = root.attrs.asdict()
    datasets = attributes["ome"]["multiscales"][0]["datasets"][:2]
    datasets[1]["coordinateTransformations"] = [
        {"type": "scale", "scale": [1.0, 1.0, 4.0, 0.8, 0.8]},
        {"type": "translation", "translation": [0.0, 0.0, 1.0, 0.2, 0.2]},
    ]
    attributes["ome"]["multiscales"][0]["datasets"] = datasets
    root.attrs["ome"] = attributes["ome"]

    result = read_ome_zarr_presentation_preview(
        path,
        request=SourcePreviewRequest(
            t_index=1,
            c_index=2,
            z_index=3,
            level=1,
        ),
    )

    assert result.preview_level == 1
    np.testing.assert_array_equal(result.data, target[1, 2, 1])


def test_single_level_truthfully_reports_no_lower_preview(tmp_path):
    path = tmp_path / "single.ome.zarr"
    data = np.arange(32 * 40, dtype=np.uint16).reshape(32, 40)
    write_ome_zarr_image(
        data,
        str(path),
        fmt=FormatV04(),
        axes=(
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ),
        scale_factors=(),
    )

    result = read_ome_zarr_presentation_preview(
        path,
        request=SourcePreviewRequest(display_shape_yx=(8, 8)),
    )

    np.testing.assert_array_equal(result.data, data)
    assert result.preview_level == 0
    assert result.level_count == 1
    assert not result.uses_lower_resolution
    assert not result.has_declared_lower_resolution
    assert result.message == (
        "Full-resolution preview - no lower-resolution level declared"
    )


def test_multiscale_label_preview_preserves_label_semantics(tmp_path):
    path = tmp_path / "labels.ome.zarr"
    image = np.zeros((64, 80), dtype=np.uint16)
    labels = np.zeros((64, 80), dtype=np.int32)
    labels[8:40, 12:52] = 7
    write_ome_zarr_image(
        image,
        str(path),
        fmt=FormatV04(),
        axes=(
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ),
        scale_factors=(2, 4),
    )
    write_ome_zarr_labels(
        labels,
        str(path),
        name="segmentation",
        fmt=FormatV04(),
        axes=(
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ),
        scale_factors=(2, 4),
        label_metadata={"source": {"image": "../../"}},
    )

    expected = read_ome_zarr(path, series_index=1).multiscale_levels[3].compute()
    result = read_ome_zarr_presentation_preview(
        path,
        series_index=1,
        request=SourcePreviewRequest(display_shape_yx=(8, 10)),
    )

    np.testing.assert_array_equal(result.data, expected)
    assert result.preview_level == 3
    assert result.image_state.kind == "label image"
    assert result.image_state.value_range == "not computed for label preview"
    assert result.preserves_label_semantics
    assert not result.intensity_statistics_allowed


def test_cancelled_preview_stops_before_pixel_compute(tmp_path, monkeypatch):
    path = tmp_path / "cancelled.ome.zarr"
    _write_multiscale(path, FormatV04())
    coordinator = SourcePreviewGenerationCoordinator()
    control = coordinator.begin()
    coordinator.cancel(control.generation)

    monkeypatch.setattr(
        da.Array,
        "compute",
        lambda *_args, **_kwargs: pytest.fail("cancelled work decoded pixels"),
    )
    with pytest.raises(SourcePreviewCancelled):
        read_ome_zarr_presentation_preview(path, control=control)
    assert not coordinator.may_publish(control.generation)


def test_preview_cancellation_is_checked_between_dask_tasks(tmp_path):
    path = tmp_path / "cancel-between-chunks.ome.zarr"
    _write_multiscale(
        path,
        FormatV05(),
        storage_options={"chunks": (1, 1, 1, 8, 8)},
    )
    completed_tasks = []
    control = SourcePreviewControl(
        generation=1,
        cancelled=lambda: bool(completed_tasks),
    )

    with DaskCallback(
        posttask=lambda key, *_args: completed_tasks.append(key),
    ):
        with pytest.raises(SourcePreviewCancelled):
            read_ome_zarr_presentation_preview(
                path,
                request=SourcePreviewRequest(
                    display_shape_yx=(12, 20),
                    t_index=0,
                    c_index=0,
                    z_index=0,
                    yx_region=(16, 48, 16, 64),
                ),
                control=control,
            )

    assert completed_tasks


def test_preview_reports_generation_qualified_progress(tmp_path):
    path = tmp_path / "progress.ome.zarr"
    _write_multiscale(path, FormatV04())
    updates = []
    coordinator = SourcePreviewGenerationCoordinator()
    control = coordinator.begin(reporter=updates.append)

    result = read_ome_zarr_presentation_preview(
        path,
        request=SourcePreviewRequest(display_shape_yx=(8, 10)),
        control=control,
    )

    assert [update.current for update in updates] == [0, 1, 2, 3, 4]
    assert {update.total for update in updates} == {4}
    assert {update.generation for update in updates} == {control.generation}
    assert updates[-1].message == result.message
    assert coordinator.may_publish(result.generation)


def test_new_generation_stops_superseded_read_before_compute(tmp_path, monkeypatch):
    path = tmp_path / "stale.ome.zarr"
    _write_multiscale(path, FormatV04())
    coordinator = SourcePreviewGenerationCoordinator()

    def supersede_on_read(progress) -> None:
        if progress.current == 2:
            coordinator.begin()

    stale_control = coordinator.begin(reporter=supersede_on_read)
    monkeypatch.setattr(
        da.Array,
        "compute",
        lambda *_args, **_kwargs: pytest.fail("stale work decoded pixels"),
    )

    with pytest.raises(StaleSourcePreviewGeneration):
        read_ome_zarr_presentation_preview(path, control=stale_control)
    assert not coordinator.may_publish(stale_control.generation)


def test_preview_rejects_requested_semantic_axis_that_is_absent(tmp_path):
    path = tmp_path / "yx-only.ome.zarr"
    write_ome_zarr_image(
        np.zeros((16, 20), dtype=np.uint8),
        str(path),
        fmt=FormatV05(),
        axes=(
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ),
        scale_factors=(),
    )

    with pytest.raises(ValueError, match="no C axis"):
        read_ome_zarr_presentation_preview(
            path,
            request=SourcePreviewRequest(c_index=0),
        )


def test_preview_rejects_unsupported_local_ome_zarr_version(tmp_path):
    path = tmp_path / "old.ome.zarr"
    root = zarr.open_group(str(path), mode="w", zarr_format=2)
    root.create_array("0", data=np.zeros((16, 20), dtype=np.uint8))
    root.attrs["multiscales"] = [
        {
            "version": "0.3",
            "axes": ["y", "x"],
            "datasets": [{"path": "0"}],
        }
    ]

    with pytest.raises(ValueError, match="0.4 and 0.5"):
        enumerate_ome_zarr_levels(path)
