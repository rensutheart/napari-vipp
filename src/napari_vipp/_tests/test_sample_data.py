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

    assert names == [
        "VIPP synthetic volume",
        "VIPP synthetic multichannel volume",
        "VIPP synthetic time-lapse multichannel",
    ]
    assert shapes[0] == (12, 96, 128)
    assert shapes[1] == (3, 12, 96, 128)
    assert shapes[2] == (5, 3, 12, 96, 128)
    assert axis_orders == ["ZYX", "CZYX", "TCZYX"]
