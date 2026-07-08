from __future__ import annotations

from napari_vipp._sample_data import make_sample_data


def test_sample_data_includes_grayscale_multichannel_and_timelapse():
    samples = make_sample_data()

    names = [metadata["name"] for _data, metadata, _layer_type in samples]
    shapes = [data.shape for data, _metadata, _layer_type in samples]
    axis_orders = [
        metadata["metadata"]["vipp_axis_order"]
        for _data, metadata, _layer_type in samples
    ]
    preferred_flags = [
        metadata["metadata"].get("napari_vipp_preferred_input", False)
        for _data, metadata, _layer_type in samples
    ]

    assert names == [
        "VIPP synthetic volume",
        "VIPP synthetic multichannel volume",
        "VIPP synthetic time-lapse multichannel",
        "VIPP synthetic measurement summary",
        "VIPP synthetic object morphology",
        "VIPP synthetic 3D mesh morphology",
        "VIPP synthetic skeleton network",
        "VIPP synthetic advanced skeleton network",
        "VIPP synthetic colocalization",
        "VIPP synthetic deconvolution image",
        "VIPP synthetic measured PSF",
        "VIPP synthetic 3D deconvolution volume",
        "VIPP synthetic 3D measured PSF",
    ]
    assert shapes[0] == (12, 96, 128)
    assert shapes[1] == (3, 12, 96, 128)
    assert shapes[2] == (5, 3, 12, 96, 128)
    assert shapes[3] == (3, 64, 64)
    assert shapes[4] == (80, 104)
    assert shapes[5] == (24, 84, 104)
    assert shapes[6] == (11, 64, 64)
    assert shapes[7] == (2, 17, 96, 96)
    assert shapes[8] == (2, 10, 80, 96)
    assert shapes[9] == (96, 96)
    assert shapes[10] == (15, 15)
    assert shapes[11] == (9, 48, 56)
    assert shapes[12] == (5, 9, 9)
    assert axis_orders == [
        "ZYX",
        "CZYX",
        "TCZYX",
        "TYX",
        "YX",
        "ZYX",
        "ZYX",
        "TZYX",
        "CZYX",
        "YX",
        "YX",
        "ZYX",
        "ZYX",
    ]
    assert preferred_flags == [
        False,
        False,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
        False,
    ]
    assert samples[0][1]["visible"] is False
    assert samples[1][1]["visible"] is False

    mesh_metadata = samples[5][1]["metadata"]
    mesh_scale = mesh_metadata["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"]
    assert mesh_scale == [2.0, 0.5, 0.5]

    advanced_metadata = samples[7][1]["metadata"]
    advanced_scale = advanced_metadata["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"]
    assert advanced_scale == [1.0, 1.2, 0.25, 0.25]

    deconvolution_metadata = samples[9][1]["metadata"]
    deconvolution_scale = deconvolution_metadata["ome"]["multiscales"][0]["datasets"][
        0
    ]["coordinateTransformations"][0]["scale"]
    assert deconvolution_scale == [0.12, 0.12]

    deconvolution_3d_metadata = samples[11][1]["metadata"]
    deconvolution_3d_scale = deconvolution_3d_metadata["ome"]["multiscales"][0][
        "datasets"
    ][0]["coordinateTransformations"][0]["scale"]
    assert deconvolution_3d_scale == [0.35, 0.12, 0.12]


def test_mesh_morphology_sample_empty_slices_have_empty_background():
    data, _metadata, _layer_type = next(
        sample
        for sample in make_sample_data()
        if sample[1]["name"] == "VIPP synthetic 3D mesh morphology"
    )

    assert data[0].max() == 0
    assert data[1].max() == 0
    assert data[-1].max() == 0
    assert data.max() > 50_000
