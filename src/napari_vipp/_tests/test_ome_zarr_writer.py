from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import zarr

from napari_vipp.core.io import read_image, write_image
from napari_vipp.core.metadata import (
    AxisMetadata,
    ChannelMetadata,
    image_state_from_array,
)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_ome_zarr_writer_preserves_native_channel_metadata(tmp_path, version, dtype):
    data = np.arange(2 * 3 * 4, dtype=dtype).reshape(2, 3, 4)
    expected = data.copy()
    data.flags.writeable = False
    state = replace(
        image_state_from_array(
            data,
            axes=(
                AxisMetadata("c", "channel"),
                AxisMetadata("y", "space", "micrometer", 0.4),
                AxisMetadata("x", "space", "micrometer", 0.6),
            ),
            source_name="Fluorescence",
            channels=(
                ChannelMetadata(name="DAPI", color=0x1234AB),
                ChannelMetadata(name="FITC", color=0x00FF00),
            ),
        ),
        history=("Channel export regression",),
    )
    path = tmp_path / "channels.ome.zarr"

    write_image(data, path, format=f"ome-zarr-{version}", image_state=state)

    attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    native = attrs if version == "0.4" else attrs["ome"]
    maximum = 65535 if dtype == np.uint16 else 1.0
    assert native["omero"] == {
        "channels": [
            {
                "label": name,
                "color": color,
                "active": True,
                "window": {"min": 0, "start": 0, "max": maximum, "end": maximum},
            }
            for name, color in [("DAPI", "1234AB"), ("FITC", "00FF00")]
        ],
        "rdefs": {"model": "color"},
    }
    assert native["vipp"]["history"] == list(state.history)
    datasets = native["multiscales"][0]["datasets"]
    assert len(datasets) == 1
    transforms = {
        transform["type"]: transform[transform["type"]]
        for transform in datasets[0]["coordinateTransformations"]
    }
    assert transforms["scale"] == [1.0, 0.4, 0.6]
    assert transforms.get("translation", [0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]
    loaded = read_image(path)
    assert loaded.data.dtype == data.dtype
    assert np.array_equal(loaded.data.compute(), expected)
    assert [channel.name for channel in loaded.image_state.channels] == [
        "DAPI",
        "FITC",
    ]
    assert [axis.scale for axis in loaded.image_state.axes] == [1.0, 0.4, 0.6]
    assert [axis.unit for axis in loaded.image_state.axes] == [
        None,
        "micrometer",
        "micrometer",
    ]
    assert np.array_equal(data, expected)


@pytest.mark.parametrize("version", ["0.4", "0.5"])
def test_ome_zarr_writer_keeps_default_scalar_channel_metadata(tmp_path, version):
    data = np.array([[False, True], [True, False]])
    path = tmp_path / "scalar.ome.zarr"

    write_image(data, path, format=f"ome-zarr-{version}")

    attrs = zarr.open_group(str(path), mode="r").attrs.asdict()
    native = attrs if version == "0.4" else attrs["ome"]
    assert native["omero"]["channels"] == [
        {
            "label": "Channel 1",
            "color": "FFFFFF",
            "active": True,
            "window": {"min": 0, "start": 0, "max": 1, "end": 1},
        }
    ]
    loaded = read_image(path)
    assert loaded.data.dtype == data.dtype
    assert np.array_equal(loaded.data.compute(), data)
