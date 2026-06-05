"""Sample data for the VIPP prototype."""

from __future__ import annotations

import numpy as np


def make_sample_data():
    """Return synthetic fluorescence samples covering Z, C, and T axes."""
    z, y, x = np.indices((12, 96, 128), dtype=np.float32)
    sphere = _sphere(z, y, x)
    tube = _tube(y, x)
    gradient = (x / x.max()) * 0.25
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.035, size=sphere.shape)
    data = np.clip(sphere * 0.75 + tube * 0.45 + gradient + noise, 0, 1)
    data = (data * 255).astype(np.uint8)

    metadata = {
        "name": "VIPP synthetic volume",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            **_ome_image_metadata("ZYX", data.shape),
        },
    }
    return [
        (data, metadata, "image"),
        _multichannel_volume_sample(z, y, x, rng),
        _time_lapse_sample(z, y, x, rng),
    ]


def _multichannel_volume_sample(z, y, x, rng):
    nuclei = _sphere(z, y, x, center=(6, 48, 62), radii=(15, 420, 520))
    neurite = _tube(y, x, center_y=42, amplitude=9, period=12, width=45)
    puncta = _puncta(z, y, x)
    channels = [
        np.clip(nuclei * 0.85 + rng.normal(0, 0.025, nuclei.shape), 0, 1),
        np.clip(neurite * 0.75 + rng.normal(0, 0.025, nuclei.shape), 0, 1),
        np.clip(puncta * 0.9 + nuclei * 0.18 + rng.normal(0, 0.02, nuclei.shape), 0, 1),
    ]
    data = (np.stack(channels, axis=0) * 65535).astype(np.uint16)
    metadata = {
        "name": "VIPP synthetic multichannel volume",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "channel_names": [
                "DAPI-like nuclei",
                "FITC-like neurites",
                "TRITC-like puncta",
            ],
            **_ome_image_metadata("CZYX", data.shape),
        },
    }
    return data, metadata, "image"


def _time_lapse_sample(z, y, x, rng):
    frames = []
    for time_index in range(5):
        shift = time_index * 4
        nuclei = _sphere(
            z,
            y,
            x,
            center=(6, 46 + time_index, 52 + shift),
            radii=(15, 360, 430),
        )
        neurite = _tube(
            y,
            x,
            center_y=38 + time_index * 2,
            amplitude=8 + time_index,
            period=11,
            width=44,
        )
        reporter = np.exp(-((x - 82 + shift) ** 2 + (y - 58) ** 2) / 340)
        channels = [
            np.clip(nuclei * 0.8 + rng.normal(0, 0.025, nuclei.shape), 0, 1),
            np.clip(neurite * 0.65 + rng.normal(0, 0.025, nuclei.shape), 0, 1),
            np.clip(reporter * 0.75 + rng.normal(0, 0.02, nuclei.shape), 0, 1),
        ]
        frames.append(np.stack(channels, axis=0))
    data = (np.stack(frames, axis=0) * 65535).astype(np.uint16)
    metadata = {
        "name": "VIPP synthetic time-lapse multichannel",
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": True,
            "channel_names": [
                "DAPI-like nuclei",
                "FITC-like neurites",
                "mCherry-like reporter",
            ],
            **_ome_image_metadata("TCZYX", data.shape),
        },
    }
    return data, metadata, "image"


def _ome_image_metadata(axis_order: str, shape: tuple[int, ...]) -> dict:
    axes = [_axis_metadata(name) for name in axis_order]
    scale = [
        1.0 if axis["type"] in {"time", "channel"} else 0.45
        for axis in axes
    ]
    return {
        "ome": {
            "version": "0.5",
            "multiscales": [
                {
                    "name": axis_order,
                    "axes": axes,
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [
                                {"type": "scale", "scale": scale}
                            ],
                        }
                    ],
                }
            ],
        },
        "vipp_axis_order": axis_order,
        "vipp_shape": tuple(int(size) for size in shape),
    }


def _axis_metadata(name: str) -> dict[str, str]:
    normalized = name.lower()
    if normalized == "t":
        return {"name": "t", "type": "time", "unit": "second"}
    if normalized == "c":
        return {"name": "c", "type": "channel"}
    return {"name": normalized, "type": "space", "unit": "micrometer"}


def _sphere(z, y, x, center=(6, 48, 62), radii=(15, 420, 520)):
    cz, cy, cx = center
    rz, ry, rx = radii
    return ((z - cz) ** 2 / rz + (y - cy) ** 2 / ry + (x - cx) ** 2 / rx) < 1


def _tube(y, x, center_y=42, amplitude=9, period=12, width=45):
    return np.exp(-((y - center_y - amplitude * np.sin(x / period)) ** 2) / width)


def _puncta(z, y, x):
    centers = [(4, 35, 48), (7, 58, 76), (8, 46, 92), (5, 67, 56)]
    signal = np.zeros_like(z, dtype=np.float32)
    for cz, cy, cx in centers:
        signal += np.exp(-((z - cz) ** 2 / 3 + (y - cy) ** 2 / 46 + (x - cx) ** 2 / 46))
    return np.clip(signal, 0, 1)
