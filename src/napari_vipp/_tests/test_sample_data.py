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
