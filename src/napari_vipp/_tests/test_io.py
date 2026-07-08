from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import imageio.v3 as iio
import numpy as np
import pytest
from tifffile import TiffFile, TiffWriter

import napari_vipp.core.io.microscope as microscope_io
from napari_vipp.core.io import (
    AnalysisLabel,
    detect_deconvolution_metadata,
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


def test_common_raster_sources_read_png_and_jpeg(tmp_path):
    gray = np.arange(6 * 7, dtype=np.uint8).reshape(6, 7)
    gray_path = tmp_path / "gray.png"
    iio.imwrite(gray_path, gray)

    gray_inspection = inspect_image_source(gray_path)
    gray_loaded = read_image(gray_path)

    assert gray_inspection.format == "png"
    assert gray_inspection.series[0].axes == "YX"
    assert gray_loaded.image_state.kind == "intensity image"
    assert gray_loaded.image_state.axis_order == "YX"
    assert np.array_equal(gray_loaded.data, gray)

    rgb = np.zeros((6, 7, 3), dtype=np.uint8)
    rgb[..., 0] = 255
    rgb[..., 1] = np.arange(7, dtype=np.uint8) * 20
    rgb[..., 2] = np.arange(6, dtype=np.uint8)[:, None] * 30
    png_path = tmp_path / "rgb.png"
    jpeg_path = tmp_path / "rgb.jpg"
    iio.imwrite(png_path, rgb)
    iio.imwrite(jpeg_path, rgb)

    png_inspection = inspect_image_source(png_path)
    png_loaded = read_image(png_path)
    jpeg_loaded = read_image(jpeg_path)

    assert png_inspection.format == "png"
    assert png_inspection.series[0].axes == "Y,X,rgb"
    assert png_inspection.series[0].shape == rgb.shape
    assert png_loaded.image_state.kind == "RGB image"
    assert png_loaded.image_state.axis_order == "Y,X,rgb"
    assert png_loaded.image_state.source.format == "png"
    assert np.array_equal(png_loaded.data, rgb)
    assert jpeg_loaded.image_state.kind == "RGB image"
    assert jpeg_loaded.image_state.source.format == "jpeg"
    assert jpeg_loaded.data.shape == rgb.shape
    with pytest.raises(IndexError):
        read_image(png_path, series_index=1)


def test_write_image_saves_2d_raster_formats(tmp_path):
    gray = (np.arange(6 * 7, dtype=np.uint16).reshape(6, 7) * 1000)
    png_path = tmp_path / "gray.png"

    saved = write_image(gray, png_path, format="png")
    loaded_gray = iio.imread(png_path)

    assert saved == png_path
    assert loaded_gray.dtype == np.uint16
    assert np.array_equal(loaded_gray, gray)

    rgb = np.zeros((6, 7, 3), dtype=np.float32)
    rgb[..., 1] = 1.0
    jpeg_path = tmp_path / "rgb.jpg"

    write_image(rgb, jpeg_path, format="jpeg")
    loaded_rgb = iio.imread(jpeg_path)

    assert loaded_rgb.dtype == np.uint8
    assert loaded_rgb.shape == rgb.shape


def test_write_image_rejects_stack_raster_export(tmp_path):
    path = tmp_path / "stack.png"

    with pytest.raises(ValueError, match="2D"):
        write_image(np.zeros((3, 6, 7), dtype=np.uint8), path, format="png")


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


def test_ome_zarr_analysis_dataset_repairs_missing_label_metadata(
    tmp_path,
    monkeypatch,
):
    import ome_zarr.writer as ome_writer

    image = np.zeros((4, 8, 9), dtype=np.uint16)
    image_state = image_state_from_array(
        image,
        axes=(
            AxisMetadata("z", "space"),
            AxisMetadata("y", "space"),
            AxisMetadata("x", "space"),
        ),
        source_name="Reference image",
    )
    labels = np.zeros((4, 8, 9), dtype=np.int32)
    labels[:, 2:4, 3:5] = 1
    label_state = replace(
        image_state_from_array(labels, axes=image_state.axes),
        kind="label image",
    )
    path = tmp_path / "analysis.ome.zarr"

    monkeypatch.setattr(
        ome_writer,
        "write_label_metadata",
        lambda *_args, **_kwargs: None,
    )

    write_ome_zarr_analysis_dataset(
        image,
        path,
        labels=(AnalysisLabel("Nuclei Labels", labels, label_state, "labels"),),
        image_state=image_state,
    )
    inspection = inspect_image_source(path)
    loaded_labels = read_image(path, series_index=1)

    assert [(series.name, series.kind) for series in inspection.series] == [
        ("Reference image", "image"),
        ("Nuclei_Labels", "labels"),
    ]
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


def test_nd2_microscope_reader_normalizes_metadata(monkeypatch, tmp_path):
    path = tmp_path / "source.nd2"
    path.write_bytes(b"fake nd2")

    class FakeND2File:
        shape = (2, 3, 4, 5)
        dtype = np.dtype("uint16")
        sizes = {"T": 2, "C": 3, "Y": 4, "X": 5}
        attributes = {"bitsPerComponentSignificant": 16}
        experiment = ()
        text_info = {
            "Description": "NIS Elements Richardson-Lucy deconvolution"
        }

        metadata = SimpleNamespace(
            channels=(
                SimpleNamespace(
                    channel=SimpleNamespace(
                        name="DAPI",
                        colorRGBA=0x00FF0000,
                        excitationLambdaNm=405.0,
                        emissionLambdaNm=461.0,
                    )
                ),
            )
        )

        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def voxel_size(self):
            return SimpleNamespace(x=0.21, y=0.22, z=1.0)

        def unstructured_metadata(self):
            return {
                "ObjectiveName": "Plan Apo 60x Oil",
                "objectiveNumericalAperture": 1.4,
                "objectiveMagnification": 60,
                "refractiveIndex": 1.515,
            }

    fake_nd2 = SimpleNamespace(
        ND2File=FakeND2File,
        imread=lambda *_args, **_kwargs: np.zeros(
            (2, 3, 4, 5),
            dtype=np.uint16,
        ),
    )

    def fake_import(name):
        if name == "nd2":
            return fake_nd2
        raise ImportError(name)

    monkeypatch.setattr(microscope_io, "import_module", fake_import)

    inspection = inspect_image_source(path)
    loaded = read_image(path)

    assert inspection.format == "nikon-nd2"
    assert inspection.series[0].axes == "TCYX"
    assert loaded.image_state.axis_order == "TCYX"
    assert [axis.scale for axis in loaded.image_state.axes] == [
        1.0,
        1.0,
        0.22,
        0.21,
    ]
    assert loaded.image_state.channels[0].name == "DAPI"
    assert loaded.image_state.channels[0].emission_wavelength == 461.0
    assert loaded.image_state.acquisition.objective_na == 1.4
    assert loaded.image_state.acquisition.refractive_index == 1.515
    assert loaded.image_state.acquisition.deconvolution_applied is True
    assert loaded.image_state.acquisition.deconvolution_method == "Richardson-Lucy"


def test_microscope_reader_reports_missing_optional_dependency(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "source.nd2"
    path.write_bytes(b"fake nd2")

    def fake_import(name):
        raise ImportError(name)

    monkeypatch.setattr(microscope_io, "import_module", fake_import)

    with pytest.raises(ImportError, match="optional dependency"):
        inspect_image_source(path)


def test_czi_missing_optional_dependency_prefers_format_specific_hint(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "source.czi"
    path.write_bytes(b"fake czi")

    def fake_import(name):
        raise ImportError(name)

    monkeypatch.setattr(microscope_io, "import_module", fake_import)

    with pytest.raises(ImportError) as exc:
        inspect_image_source(path)

    message = str(exc.value)
    assert "napari-vipp[czi]" in message
    assert "napari-vipp[bioformats]" in message


def test_nested_microscope_metadata_does_not_break_acquisition_detection():
    acquisition = microscope_io._acquisition_from_metadata(
        {
            "objective": {"Name": "Plan-Apochromat 63x"},
            "objectiveName": "Plan-Apochromat 63x",
            "objectiveNumericalAperture": {"value": 1.4},
            "numericalAperture": 1.35,
        }
    )

    assert acquisition.objective == "Plan-Apochromat 63x"
    assert acquisition.objective_na == 1.35


def test_xarray_czi_metadata_normalizes_channels_and_axes():
    class Coord:
        def __init__(self, values):
            self.values = np.asarray(values)

    fake = SimpleNamespace(
        attrs={
            "channels": {
                "Ch1": {
                    "Fluor": "Hoechst 33342",
                    "Color": "#0000FF",
                    "ExcitationWavelength": 405.0,
                    "EmissionWavelength": 469.91,
                },
                "Ch2": {
                    "Fluor": "Alexa Fluor 568",
                    "Color": "#FF0000",
                    "ExcitationWavelength": 561.0,
                    "DetectionWavelength": (592.01, 712.0),
                },
            },
            "coord_scales": {
                "C": 182.095,
                "Z": 2e-7,
                "Y": 1.660531641927678e-7,
                "X": 1.660531641927678e-7,
            },
            "coord_units": {
                "C": "nanometer",
                "Z": "meter",
                "Y": "meter",
                "X": "meter",
            },
        },
        coords={
            "C": Coord([469.91, 652.005]),
            "T": Coord([7404.34473018, 7415.62325915]),
            "Z": Coord([0.0, 2e-7]),
            "Y": Coord([0.0, 1.660531641927678e-7]),
            "X": Coord([0.0, 1.660531641927678e-7]),
        },
    )

    axes = microscope_io._axes_from_xarray(
        fake,
        ("C", "T", "Z", "Y", "X"),
        (2, 2, 2, 2, 2),
    )
    channels = microscope_io._channels_from_xarray(fake)

    assert [(axis.name, axis.type, axis.unit) for axis in axes] == [
        ("c", "channel", None),
        ("t", "time", "second"),
        ("z", "space", "micrometer"),
        ("y", "space", "micrometer"),
        ("x", "space", "micrometer"),
    ]
    assert [axis.scale for axis in axes] == pytest.approx(
        [
            1.0,
            11.27852897,
            0.2,
            0.1660531641927678,
            0.1660531641927678,
        ]
    )
    assert channels[0].name == "Hoechst 33342"
    assert channels[0].color == 0x0000FF
    assert channels[0].excitation_wavelength == 405.0
    assert channels[0].emission_wavelength == 469.91
    assert channels[1].name == "Alexa Fluor 568"
    assert channels[1].color == 0xFF0000
    assert channels[1].emission_wavelength == pytest.approx(652.005)


def test_deconvolution_metadata_detection_is_conservative():
    assert detect_deconvolution_metadata(
        {"Processing": "Huygens deconvolution"}
    ) == (True, "Huygens")
    assert detect_deconvolution_metadata(
        {"Processing": "no deconvolution applied"}
    ) == (False, "")
    assert detect_deconvolution_metadata({"Processing": "raw acquisition"}) == (
        None,
        "",
    )
