from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from tifffile import TiffFile, TiffWriter

from napari_vipp.core.io import (
    AnalysisLabel,
    inspect_image_source,
    read_image,
    write_image,
    write_ome_zarr_analysis_dataset,
)
from napari_vipp.core.metadata import (
    AxisMetadata,
    ChannelMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload


def test_ome_tiff_round_trip_preserves_axes_scale_and_channels(tmp_path):
    data = np.arange(2 * 3 * 4 * 8 * 9, dtype=np.uint16).reshape(
        2, 3, 4, 8, 9
    )
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("t", "time", "s", 2.5),
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", "µm", 1.2),
            AxisMetadata("y", "space", "µm", 0.4),
            AxisMetadata("x", "space", "µm", 0.4),
        ),
        source_name="Processed fluorescence",
        channels=(
            ChannelMetadata(
                name="DAPI",
                emission_wavelength=461,
                emission_wavelength_unit="nm",
            ),
            ChannelMetadata(name="FITC"),
            ChannelMetadata(name="TRITC"),
        ),
        source=SourceMetadata(
            uri="source.ome.tif",
            format="ome-tiff",
            series_name="Acquisition 1",
        ),
        history=("Gaussian Blur", "Otsu Threshold"),
    )
    path = tmp_path / "processed.ome.tif"

    write_image(data, path, format="ome-tiff", image_state=state)
    inspection = inspect_image_source(path)
    loaded = read_image(path)

    assert inspection.format == "ome-tiff"
    assert inspection.series[0].axes == "TCZYX"
    assert loaded.image_state.axis_order == "TCZYX"
    assert [axis.scale for axis in loaded.image_state.axes] == [
        2.5,
        1.0,
        1.2,
        0.4,
        0.4,
    ]
    assert [channel.name for channel in loaded.image_state.channels] == [
        "DAPI",
        "FITC",
        "TRITC",
    ]
    assert np.array_equal(loaded.data, data)
    with TiffFile(path) as tif:
        assert "napari-vipp" in tif.ome_metadata
        assert "Gaussian Blur" in tif.ome_metadata


def test_imagej_tiff_round_trip_writes_hyperstack_calibration(tmp_path):
    data = np.zeros((2, 3, 4, 8, 9), dtype=bool)
    data[:, 1, :, 2:5, 3:6] = True
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("t", "time", "s", 2.5),
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", "µm", 1.2),
            AxisMetadata("y", "space", "µm", 0.4),
            AxisMetadata("x", "space", "µm", 0.4),
        ),
    )
    path = tmp_path / "mask.tif"

    write_image(data, path, format="imagej-tiff", image_state=state)
    loaded = read_image(path)

    with TiffFile(path) as tif:
        assert tif.is_imagej
        assert tif.series[0].axes == "TZCYX"
        assert tif.imagej_metadata["unit"] == "micron"
        assert tif.imagej_metadata["spacing"] == pytest.approx(1.2)
        assert tif.imagej_metadata["finterval"] == pytest.approx(2.5)
        assert set(np.unique(tif.series[0].asarray())) == {0, 255}
    scales = {axis.name: axis.scale for axis in loaded.image_state.axes}
    assert scales["x"] == pytest.approx(0.4)
    assert scales["y"] == pytest.approx(0.4)
    assert scales["z"] == pytest.approx(1.2)


def test_tiff_inspection_and_reader_select_independent_series(tmp_path):
    first = np.arange(30, dtype=np.uint8).reshape(5, 6)
    second = np.arange(56, dtype=np.uint16).reshape(7, 8)
    path = tmp_path / "two-series.tif"
    with TiffWriter(path) as tif:
        tif.write(first, metadata={"axes": "YX"})
        tif.write(second, metadata={"axes": "YX"})

    inspection = inspect_image_source(path)
    loaded = read_image(path, series_index=1)

    assert inspection.format == "tiff"
    assert len(inspection.series) == 2
    assert inspection.series[1].shape == (7, 8)
    assert loaded.selected_series.index == 1
    assert np.array_equal(loaded.data, second)


@pytest.mark.parametrize(
    ("format", "expected_format"),
    [("ome-zarr", "ome-zarr-0.4"), ("ome-zarr-0.5", "ome-zarr-0.5")],
)
def test_ome_zarr_round_trip_is_lazy_and_preserves_scale(
    tmp_path,
    format,
    expected_format,
):
    data = np.arange(2 * 3 * 8 * 9, dtype=np.uint16).reshape(2, 3, 8, 9)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", "µm", 1.2),
            AxisMetadata("y", "space", "µm", 0.4),
            AxisMetadata("x", "space", "µm", 0.4),
        ),
        source_name="Zarr fluorescence",
        channels=(
            ChannelMetadata(name="DAPI"),
            ChannelMetadata(name="FITC"),
        ),
        history=("Gaussian Blur",),
    )
    path = tmp_path / f"roundtrip-{expected_format}.ome.zarr"

    write_image(data, path, format=format, image_state=state)
    inspection = inspect_image_source(path)
    loaded = read_image(path)

    assert inspection.format == expected_format
    assert inspection.series[0].axes == "CZYX"
    assert hasattr(loaded.data, "compute")
    assert loaded.image_state.value_range == "not computed (lazy)"
    assert [channel.name for channel in loaded.image_state.channels] == [
        "DAPI",
        "FITC",
    ]
    assert [axis.scale for axis in loaded.image_state.axes] == [
        1.0,
        1.2,
        0.4,
        0.4,
    ]
    assert np.array_equal(loaded.data.compute(), data)


def test_ome_zarr_analysis_dataset_round_trip_includes_label_group(tmp_path):
    image = np.zeros((2, 4, 8, 9), dtype=np.uint16)
    image_state = image_state_from_array(
        image,
        axes=(
            AxisMetadata("c", "channel"),
            AxisMetadata("z", "space", "µm", 1.2),
            AxisMetadata("y", "space", "µm", 0.4),
            AxisMetadata("x", "space", "µm", 0.4),
        ),
        source_name="Reference image",
        channels=(ChannelMetadata(name="DAPI"), ChannelMetadata(name="FITC")),
    )
    labels = np.zeros((4, 8, 9), dtype=np.int32)
    labels[:, 2:4, 3:5] = 1
    label_state = replace(
        image_state_from_array(labels, axes=image_state.axes[1:]),
        kind="label image",
    )
    path = tmp_path / "analysis.ome.zarr"

    write_ome_zarr_analysis_dataset(
        image,
        path,
        labels=(AnalysisLabel("Nuclei Labels", labels, label_state, "labels"),),
        image_state=image_state,
    )
    inspection = inspect_image_source(path)

    assert [(series.name, series.kind) for series in inspection.series] == [
        ("Reference image", "image"),
        ("Nuclei_Labels", "labels"),
    ]
    loaded_image = read_image(path, series_index=0)
    loaded_labels = read_image(path, series_index=1)
    assert loaded_image.associated_labels == ("Nuclei_Labels",)
    assert loaded_labels.image_state.kind == "label image"
    assert loaded_labels.image_state.axis_order == "ZYX"
    assert np.array_equal(loaded_labels.data.compute(), labels)


def test_ome_zarr_analysis_dataset_rejects_mismatched_label_shape(tmp_path):
    image = np.zeros((4, 8, 9), dtype=np.uint16)
    image_state = image_state_from_array(
        image,
        axes=(
            AxisMetadata("z", "space", "µm", 1.2),
            AxisMetadata("y", "space", "µm", 0.4),
            AxisMetadata("x", "space", "µm", 0.4),
        ),
    )
    labels = np.zeros((4, 8, 8), dtype=np.int32)
    label_state = replace(
        image_state_from_array(labels, axes=image_state.axes),
        kind="label image",
    )

    with pytest.raises(ValueError, match="reference image"):
        write_ome_zarr_analysis_dataset(
            image,
            tmp_path / "bad.ome.zarr",
            labels=(AnalysisLabel("bad labels", labels, label_state, "labels"),),
            image_state=image_state,
        )


def test_pipeline_source_payload_accepts_prebuilt_image_state():
    data = np.zeros((4, 8, 9), dtype=np.uint16)
    state = image_state_from_array(
        data,
        axes=(
            AxisMetadata("z", "space", "µm", 1.5),
            AxisMetadata("y", "space", "µm", 0.3),
            AxisMetadata("x", "space", "µm", 0.3),
        ),
        metadata_source="OME-XML",
    )
    pipeline = PrototypePipeline()

    pipeline.run(
        None,
        source_payloads={
            "input": SourcePayload(data, name="OME source", image_state=state)
        },
    )

    assert pipeline.output_states["input"] is state


def test_auto_format_keeps_int32_labels_in_conventional_tiff(tmp_path):
    labels = np.array([[0, 70000], [2, 3]], dtype=np.int32)
    state = image_state_from_array(labels)
    state = replace(state, kind="label image")
    path = tmp_path / "labels.tif"

    write_image(labels, path, image_state=state)

    with TiffFile(path) as tif:
        assert not tif.is_imagej
        assert tif.ome_metadata is None
        assert tif.asarray().dtype == np.int32
        assert int(tif.asarray().max()) == 70000


def test_standalone_label_image_is_not_miswritten_as_ome_zarr(tmp_path):
    labels = np.array([[0, 1], [2, 3]], dtype=np.int32)
    state = replace(image_state_from_array(labels), kind="label image")

    with pytest.raises(ValueError, match="Export OME Analysis Dataset"):
        write_image(
            labels,
            tmp_path / "labels.ome.zarr",
            format="ome-zarr",
            image_state=state,
        )
