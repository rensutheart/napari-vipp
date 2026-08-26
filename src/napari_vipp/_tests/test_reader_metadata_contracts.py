from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import napari_vipp.core.io.microscope as microscope_io
import napari_vipp.core.io.registry as io_registry
from napari_vipp.core.io.errors import ImageSourceError, ImageSourceErrorCode
from napari_vipp.core.io.model import ImageDataset, ImageSeriesInfo, SourceInspection
from napari_vipp.core.metadata import image_state_from_array


def test_registry_normalizes_reader_capabilities_for_sourceitem_contract(
    monkeypatch,
    tmp_path,
):
    path = tmp_path / "source.nd2"
    path.write_bytes(b"reader-double")
    selected = ImageSeriesInfo(
        0,
        "0",
        "Lazy image",
        (8, 9),
        "uint16",
        "YX",
        capabilities=(
            "pixel-lazy-inspection",
            "lazy-data",
            "decoded-size-estimate",
        ),
    )
    monkeypatch.setattr(
        io_registry,
        "inspect_microscope",
        lambda _path: SourceInspection(
            str(path),
            "nikon-nd2",
            (selected,),
        ),
    )

    annotated = io_registry.inspect_image_source(path).series[0]

    assert "pixel_lazy_inspection" in annotated.capabilities
    assert "lazy_data" in annotated.capabilities
    assert "decoded_size_estimate" in annotated.capabilities
    assert not any("-" in value for value in annotated.capabilities)


def test_nd2_inspection_reports_lazy_contract_and_decoded_size(
    monkeypatch,
    tmp_path,
):
    class FakeND2File:
        shape = (2, 3, 8, 9)
        dtype = np.dtype("uint16")
        sizes = MappingProxyType({"Z": 2, "C": 3, "Y": 8, "X": 9})
        metadata = SimpleNamespace(channels=())
        attributes = None
        experiment = None
        text_info = None

        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def voxel_size(self):
            return SimpleNamespace(x=0.2, y=0.2, z=0.8)

        def unstructured_metadata(self):
            return {}

    fake_module = SimpleNamespace(ND2File=FakeND2File, __version__="1.2.3")
    monkeypatch.setattr(
        microscope_io,
        "import_module",
        lambda name: fake_module if name == "nd2" else None,
    )

    selected = microscope_io._inspect_nd2(tmp_path / "source.nd2").series[0]

    assert selected.reader_key == "nd2"
    assert selected.reader_version == "1.2.3"
    assert selected.capabilities == (
        "pixel-lazy-inspection",
        "lazy-data",
        "decoded-size-estimate",
    )
    assert selected.estimated_decoded_bytes == 2 * 3 * 8 * 9 * 2


def test_lif_inspection_and_read_share_metadata_without_inspection_decode(
    monkeypatch,
    tmp_path,
):
    decode_calls: list[str] = []
    image = SimpleNamespace(
        shape=(5, 3, 8, 9),
        dtype=np.dtype("uint8"),
        dims=("Z", "C", "Y", "X"),
        name="Anisotropic volume",
        coords={
            "Z": np.arange(5) * 0.8e-6,
            "C": np.asarray(["Green", "Red", "Blue"]),
            "Y": np.arange(8) * 0.1e-6,
            "X": np.arange(9) * 0.1e-6,
        },
        _dimensions=(
            SimpleNamespace(label="Z", unit="m"),
            SimpleNamespace(label="Y", unit="m"),
            SimpleNamespace(label="X", unit="m"),
        ),
        _channel_names=("Green", "Red", "Blue"),
        attrs={
            "HardwareSetting": {
                "ObjectiveName": "HC PL APO 63x/1.40 OIL",
                "Magnification": 63,
                "NumericalAperture": 1.4,
                "Immersion": "OIL",
                "RefractionIndex": 1.518,
            }
        },
    )

    def asarray():
        decode_calls.append("decoded")
        return np.zeros(image.shape, dtype=image.dtype)

    image.asarray = asarray

    class FakeLifFile:
        images = (image,)
        xml_header = "<LMSDataContainerHeader />"

        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    fake_module = SimpleNamespace(LifFile=FakeLifFile, __version__="test")
    monkeypatch.setattr(microscope_io, "_optional_import", lambda *_args: fake_module)
    path = tmp_path / "source.lif"

    inspection = microscope_io._inspect_lif(path)
    inspected = inspection.series[0]

    assert decode_calls == []
    assert inspected.capabilities == microscope_io._EAGER_READER_CAPABILITIES
    assert inspected.image_state is not None
    assert [axis.scale for axis in inspected.image_state.axes] == pytest.approx(
        [0.8, 1.0, 0.1, 0.1]
    )
    assert [axis.unit for axis in inspected.image_state.axes] == [
        "micrometer",
        None,
        "micrometer",
        "micrometer",
    ]
    assert [channel.name for channel in inspected.image_state.channels] == [
        "Green",
        "Red",
        "Blue",
    ]
    assert inspected.image_state.acquisition.objective_magnification == 63
    assert inspected.image_state.acquisition.objective_na == 1.4
    assert inspected.image_state.acquisition.refractive_index == 1.518

    loaded = microscope_io._read_lif(path)

    assert decode_calls == ["decoded"]
    assert loaded.image_state.axes == inspected.image_state.axes
    assert loaded.image_state.channels == inspected.image_state.channels
    assert loaded.image_state.acquisition == inspected.image_state.acquisition


def test_czi_same_shape_scenes_keep_stable_keys_metadata_and_pixels(
    monkeypatch,
    tmp_path,
):
    class FakeScene:
        shape = (2, 4, 5)
        dtype = np.dtype("uint8")
        dims = ("C", "Y", "X")
        coord_scales = {"Y": 0.1e-6, "X": 0.1e-6}
        coord_units = {"Y": "meter", "X": "meter"}
        coords = {"C": np.asarray(["DAPI", "FITC"])}
        channels = {
            "DAPI": {"ExcitationWavelength": 405, "EmissionWavelength": 461},
            "FITC": {"ExcitationWavelength": 488, "EmissionWavelength": 520},
        }
        objective = {
            "Name": "Plan-Apochromat",
            "NominalMagnification": 5,
            "LensNA": 0.35,
            "Immersion": "Air",
        }

        def __init__(self, key):
            self.key = key
            self.name = f"Scene {key}"
            self.attrs = {
                "coord_scales": self.coord_scales,
                "coord_units": self.coord_units,
                "channels": self.channels,
                "objective": self.objective,
            }

        def asarray(self):
            return np.full(self.shape, self.key, dtype=self.dtype)

    scenes = {3: FakeScene(3), 9: FakeScene(9)}

    class FakeCziFile:
        def __init__(self, _path):
            self.scenes = scenes

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def metadata(self):
            return "<Metadata />"

    fake_module = SimpleNamespace(CziFile=FakeCziFile, __version__="test")
    monkeypatch.setattr(microscope_io, "_optional_import", lambda *_args: fake_module)
    path = tmp_path / "source.czi"

    inspection = microscope_io._inspect_czi(path)

    assert [item.key for item in inspection.series] == ["3", "9"]
    assert [item.shape for item in inspection.series] == [(2, 4, 5), (2, 4, 5)]
    assert all(
        item.capabilities == microscope_io._EAGER_READER_CAPABILITIES
        for item in inspection.series
    )
    assert inspection.series[1].image_state is not None
    assert [axis.scale for axis in inspection.series[1].image_state.axes] == (
        pytest.approx([1.0, 0.1, 0.1])
    )
    assert [channel.name for channel in inspection.series[1].image_state.channels] == [
        "DAPI",
        "FITC",
    ]
    assert inspection.series[1].image_state.acquisition.objective_na == 0.35

    loaded = microscope_io._read_czi(path, 1)

    assert np.all(loaded.data == 9)
    assert loaded.selected_series.key == "9"
    assert loaded.image_state.axes == inspection.series[1].image_state.axes
    assert loaded.image_state.channels == inspection.series[1].image_state.channels
    assert loaded.image_state.acquisition == (
        inspection.series[1].image_state.acquisition
    )


def test_czi_missing_scene_is_not_retargeted_by_ordinal_position():
    selected = ImageSeriesInfo(1, "9", "Scene 9", (2, 4, 5), "uint8", "CYX")
    czi = SimpleNamespace(scenes={3: object(), 7: object()})

    with pytest.raises(ValueError, match="will not substitute"):
        microscope_io._czi_scene(czi, selected)


def test_oir_inspection_preserves_tcyx_calibration_channels_and_objective(
    monkeypatch,
    tmp_path,
):
    channel_records = (
        SimpleNamespace(
            name="unused",
            start_wavelength=None,
            end_wavelength=None,
        ),
        SimpleNamespace(name="CH2", start_wavelength=500.0, end_wavelength=540.0),
        SimpleNamespace(name="CH3", start_wavelength=570.0, end_wavelength=620.0),
    )

    class FakeOirFile:
        shape = (8, 2, 6, 7)
        dtype = np.dtype("uint16")
        dims = ("T", "C", "Y", "X")
        name = "sequence.oir"
        coord_scales = {"T": 10.0, "Y": 2.5, "X": 2.5}
        coord_units = {"T": "second", "Y": "micrometer", "X": "micrometer"}
        coord_offsets = {"T": 0.0, "Y": 0.0, "X": 0.0}
        coords = {"C": np.asarray(["CH2", "CH3"])}
        channels = channel_records
        attrs = {
            "objectiveName": "UPLXAPO10X",
            "magnification": 10.0,
            "naValue": 0.4,
            "immersion": "DRY",
        }
        xml_metadata = {}

        def __init__(self, _path):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

        def asarray(self):
            return np.zeros(self.shape, dtype=self.dtype)

    fake_module = SimpleNamespace(OirFile=FakeOirFile, __version__="test")
    monkeypatch.setattr(microscope_io, "_optional_import", lambda *_args: fake_module)
    path = tmp_path / "sequence.oir"

    inspection = microscope_io._inspect_oir(path)
    inspected = inspection.series[0]

    assert inspected.image_state is not None
    assert inspected.image_state.axis_order == "TCYX"
    assert [axis.scale for axis in inspected.image_state.axes] == [10.0, 1.0, 2.5, 2.5]
    assert [axis.unit for axis in inspected.image_state.axes] == [
        "second",
        None,
        "micrometer",
        "micrometer",
    ]
    assert [channel.name for channel in inspected.image_state.channels] == [
        "CH2",
        "CH3",
    ]
    assert inspected.image_state.channels[1].emission_wavelength == 595.0
    assert inspected.image_state.acquisition.objective_magnification == 10.0
    assert inspected.image_state.acquisition.objective_immersion == "DRY"

    loaded = microscope_io._read_oir(path)

    assert loaded.image_state.axes == inspected.image_state.axes
    assert loaded.image_state.channels == inspected.image_state.channels
    assert loaded.image_state.acquisition == inspected.image_state.acquisition


def test_lsm_main_and_rgb_thumbnail_have_metadata_parity(monkeypatch, tmp_path):
    main = ImageSeriesInfo(0, "0", "Series 1", (4, 8, 9), "uint16", "CYX")
    thumbnail = ImageSeriesInfo(1, "1", "Series 2", (3, 4, 5), "uint8", "SYX")
    tiff_inspection = SourceInspection(
        "source.lsm",
        "tiff",
        (main, thumbnail),
        None,
    )
    metadata = {
        "VoxelSizeX": 0.2e-6,
        "VoxelSizeY": 0.2e-6,
        "OriginX": 0.0,
        "OriginY": 0.0,
        "ChannelColors": {
            "ColorNames": ["DAPI", "FITC", "TRITC", "Cy5"],
            "Colors": [
                [0, 0, 255, 0],
                [0, 255, 0, 0],
                [255, 0, 0, 0],
                [255, 0, 255, 0],
            ],
        },
        "ScanInformation": {
            "Objective": "Plan-Apochromat 10x/0.45",
            "Tracks": [
                {
                    "DetectionChannels": [
                        {"ChannelName": "DAPI", "DyeName": "DAPI"},
                        {"ChannelName": "FITC", "DyeName": "FITC"},
                        {"ChannelName": "TRITC", "DyeName": "TRITC"},
                        {"ChannelName": "Cy5", "DyeName": "Cy5"},
                    ]
                }
            ],
        },
    }
    fake_tif_series = (
        SimpleNamespace(
            attrs={
                "coord_scales": {"Y": 0.2, "X": 0.2},
                "coord_units": {"Y": "micrometer", "X": "micrometer"},
            }
        ),
        SimpleNamespace(attrs={}),
    )

    class FakeTiffFile:
        def __init__(self, _path):
            self.lsm_metadata = metadata
            self.series = fake_tif_series

        def __enter__(self):
            return self

        def __exit__(self, *_exc):
            return False

    monkeypatch.setattr(microscope_io, "inspect_tiff", lambda _path: tiff_inspection)
    monkeypatch.setattr(microscope_io, "TiffFile", FakeTiffFile)
    path = tmp_path / "source.lsm"

    inspection = microscope_io._inspect_lsm(path)
    primary, rgb = inspection.series

    assert primary.image_state is not None
    assert rgb.image_state is not None
    assert primary.image_state.axis_order == "CYX"
    assert "".join(axis.name.upper() for axis in rgb.image_state.axes) == "RGBYX"
    assert [axis.scale for axis in primary.image_state.axes] == [1.0, 0.2, 0.2]
    assert [channel.name for channel in primary.image_state.channels] == [
        "DAPI",
        "FITC",
        "TRITC",
        "Cy5",
    ]
    assert primary.image_state.acquisition.objective_magnification == 10.0
    assert primary.image_state.acquisition.objective_na == 0.45
    assert rgb.image_state.channels == ()
    assert [axis.unit for axis in rgb.image_state.axes] == [None, None, None]
    assert [axis.scale for axis in rgb.image_state.axes] == [1.0, 1.0, 1.0]
    assert primary.estimated_decoded_bytes == 4 * 8 * 9 * 2
    assert rgb.estimated_decoded_bytes == 3 * 4 * 5

    raw_state = image_state_from_array(np.zeros(primary.shape, dtype=np.uint16))
    assert raw_state is not None
    raw = ImageDataset(
        np.zeros(primary.shape, dtype=np.uint16),
        raw_state,
        tiff_inspection,
        main,
    )
    monkeypatch.setattr(microscope_io, "read_tiff", lambda *_args: raw)
    loaded = microscope_io._read_lsm(path)

    assert loaded.image_state.axes == primary.image_state.axes
    assert loaded.image_state.channels == primary.image_state.channels
    assert loaded.image_state.acquisition == primary.image_state.acquisition


def test_bioio_reports_logical_decoded_size_and_no_unproven_pyramid(
    monkeypatch,
    tmp_path,
):
    class FakeBioImage:
        scenes = ("Resolution Level 1",)
        shape = (1, 2, 64, 109, 143)
        dtype = np.dtype("uint8")
        dims = SimpleNamespace(order="TCZYX")
        physical_pixel_sizes = SimpleNamespace(X=0.1646, Y=0.1646, Z=0.2)
        channel_names = ("CollagenIV", "GFAP")
        metadata = {"objectiveName": "63x, 1.3NA"}

        def __init__(self, _path):
            self.current_scene = self.scenes[0]

        def set_scene(self, scene):
            if scene != self.scenes[0]:
                raise IndexError(scene)

        @property
        def dask_data(self):
            raise AssertionError("inspection must not request BioIO pixel data")

    fake_module = SimpleNamespace(BioImage=FakeBioImage, __version__="test")
    monkeypatch.setattr(microscope_io, "_optional_bioio", lambda _suffix: fake_module)

    selected = microscope_io._inspect_bioio(
        tmp_path / "source.ims",
        "imaris-ims",
    ).series[0]

    assert selected.estimated_decoded_bytes == 1 * 2 * 64 * 109 * 143
    assert selected.capabilities == microscope_io._LAZY_READER_CAPABILITIES
    assert selected.reader_key == "bioio-bioformats"
    assert "level-enumeration" not in selected.capabilities
    assert selected.image_state is not None
    assert [axis.scale for axis in selected.image_state.axes] == [
        1.0,
        1.0,
        0.2,
        0.1646,
        0.1646,
    ]
    assert [channel.name for channel in selected.image_state.channels] == [
        "CollagenIV",
        "GFAP",
    ]


def test_bioio_java_initialization_failure_is_actionable(monkeypatch, tmp_path):
    class FailingBioImage:
        def __init__(self, _path):
            raise RuntimeError("JVM was not found while starting BioFormats")

    fake_module = SimpleNamespace(BioImage=FailingBioImage, __version__="test")
    monkeypatch.setattr(microscope_io, "_optional_bioio", lambda _suffix: fake_module)

    with pytest.raises(ImageSourceError) as exc:
        microscope_io._inspect_bioio(tmp_path / "source.vsi", "olympus-vsi")

    assert exc.value.code is ImageSourceErrorCode.JAVA_BIOFORMATS_READINESS
    assert exc.value.stage == "inspect"
    assert exc.value.backend == "bioio-bioformats"
    assert "Java runtime" in exc.value.display_text
    assert "restart VIPP" in exc.value.display_text


def test_bioformats_plugin_absence_is_reported_before_vsi_inspection(monkeypatch):
    bioio = SimpleNamespace(BioImage=object, __version__="test")

    def fake_import(name):
        if name == "bioio":
            return bioio
        if name == "bioio_bioformats":
            raise ModuleNotFoundError(name)
        raise AssertionError(name)

    monkeypatch.setattr(microscope_io, "import_module", fake_import)

    with pytest.raises(microscope_io.OptionalMicroscopeReaderError) as exc:
        microscope_io._optional_bioio(".vsi")

    assert exc.value.module_name == "bioio_bioformats"
    assert exc.value.install_command == 'pip install "napari-vipp[bioformats]"'
    assert "bioio_bioformats" in str(exc.value)
