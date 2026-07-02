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
        _measurement_summary_sample(),
        _skeleton_network_sample(),
        _advanced_skeleton_network_sample(),
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


def _measurement_summary_sample():
    """Return a small time series with deterministic object areas."""
    data = np.zeros((3, 64, 64), dtype=np.uint16)
    # Areas per timepoint are:
    # t0: 24, 30
    # t1: 12, 20, 28
    # t2: 40
    objects = (
        ((0, slice(8, 12), slice(8, 14)), 24_000),
        ((0, slice(30, 36), slice(20, 25)), 26_000),
        ((1, slice(8, 11), slice(8, 12)), 28_000),
        ((1, slice(24, 28), slice(24, 29)), 30_000),
        ((1, slice(42, 46), slice(40, 47)), 32_000),
        ((2, slice(18, 23), slice(18, 26)), 34_000),
    )
    for index, value in objects:
        data[index] = value

    yy, xx = np.indices(data.shape[-2:], dtype=np.uint16)
    background = ((yy + xx) % 32).astype(np.uint16)
    data += background[None, :, :]

    metadata = {
        "name": "VIPP synthetic measurement summary",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Time-series object sample with known per-timepoint object "
                "counts and areas for validating measurement summaries."
            ),
            **_ome_image_metadata("TYX", data.shape),
        },
    }
    return data, metadata, "image"


def _skeleton_network_sample():
    """Return a sparse 3D network with known endpoints, branches, and spur."""
    data = np.zeros((11, 64, 64), dtype=np.uint16)
    signal = np.uint16(48_000)

    center = (5, 32, 32)
    data[center[0], center[1], 14:51] = signal
    data[center[0], 12:45, center[2]] = signal
    data[2:6, center[1], center[2]] = signal
    data[6, center[1], center[2]] = signal  # short terminal spur

    data[8, 50, 45:52] = signal  # separate linear component
    data[9, 10, 10] = signal  # isolated skeleton voxel

    metadata = {
        "name": "VIPP synthetic skeleton network",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Sparse 3D skeleton-style network with a junction, terminal "
                "branches, a short spur, a separate component, and an isolated "
                "voxel for validating skeleton QC and pruning nodes."
            ),
            **_ome_image_metadata("ZYX", data.shape),
        },
    }
    return data, metadata, "image"


def _advanced_skeleton_network_sample():
    """Return a multi-timepoint 3D skeleton with loops, branches, and fragments."""
    data = np.zeros((2, 17, 96, 96), dtype=np.uint16)
    signal = np.uint16(52_000)

    # Frame 0: a connected loop/cross network with 3D branches, a separate
    # Y-shaped fragment, a pure cycle, short spurs, and an isolated voxel.
    _draw_rectangle_zyx(data[0], z=8, y0=22, y1=66, x0=22, x1=72, value=signal)
    _draw_line_zyx(data[0], (8, 22, 47), (8, 66, 47), signal)
    _draw_line_zyx(data[0], (8, 44, 22), (8, 44, 72), signal)
    _draw_line_zyx(data[0], (3, 44, 47), (13, 44, 47), signal)
    _draw_line_zyx(data[0], (8, 44, 47), (3, 33, 34), signal)
    _draw_line_zyx(data[0], (8, 44, 47), (13, 58, 64), signal)
    _draw_line_zyx(data[0], (8, 22, 47), (8, 14, 47), signal)
    _draw_line_zyx(data[0], (8, 66, 72), (8, 70, 76), signal)

    _draw_line_zyx(data[0], (5, 76, 12), (5, 76, 34), signal)
    _draw_line_zyx(data[0], (5, 76, 23), (5, 66, 18), signal)
    _draw_line_zyx(data[0], (5, 76, 23), (9, 83, 29), signal)
    _draw_rectangle_zyx(data[0], z=13, y0=10, y1=22, x0=76, x1=88, value=signal)
    _draw_line_zyx(data[0], (2, 88, 6), (2, 88, 15), signal)
    _draw_line_zyx(data[0], (2, 88, 11), (4, 92, 11), signal)
    _draw_line_zyx(data[0], (1, 5, 5), (1, 5, 13), signal)
    _draw_line_zyx(data[0], (1, 5, 9), (1, 10, 9), signal)
    data[0, 15, 88, 88] = signal

    # Frame 1: a shifted, more fragmented version with two looped components,
    # a long 3D trunk, several terminal branches, and isolated noise voxels.
    _draw_rectangle_zyx(data[1], z=7, y0=18, y1=58, x0=18, x1=58, value=signal)
    _draw_line_zyx(data[1], (7, 38, 18), (7, 38, 58), signal)
    _draw_line_zyx(data[1], (2, 38, 38), (14, 38, 38), signal)
    _draw_line_zyx(data[1], (7, 38, 38), (2, 24, 24), signal)
    _draw_line_zyx(data[1], (7, 38, 38), (12, 52, 54), signal)
    _draw_line_zyx(data[1], (7, 18, 38), (7, 10, 32), signal)
    _draw_line_zyx(data[1], (7, 58, 58), (7, 64, 66), signal)

    _draw_rectangle_zyx(data[1], z=11, y0=58, y1=78, x0=64, x1=84, value=signal)
    _draw_line_zyx(data[1], (11, 68, 64), (11, 68, 84), signal)
    _draw_line_zyx(data[1], (11, 68, 74), (15, 84, 74), signal)
    _draw_line_zyx(data[1], (3, 78, 18), (3, 88, 18), signal)
    data[1, 1, 88, 88] = signal
    data[1, 15, 8, 86] = signal

    metadata_block = _ome_image_metadata("TZYX", data.shape)
    metadata_block["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"] = [1.0, 1.2, 0.25, 0.25]
    metadata = {
        "name": "VIPP synthetic advanced skeleton network",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Two-timepoint sparse 3D skeleton-style network with loops, "
                "junction-rich grid components, 3D terminal branches, separate "
                "fragments, short spurs, isolated voxels, and anisotropic "
                "spatial calibration for stress-testing skeleton graph outputs."
            ),
            **metadata_block,
        },
    }
    return data, metadata, "image"


def _draw_line_zyx(
    target: np.ndarray,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
    value: np.uint16,
) -> None:
    start_array = np.asarray(start, dtype=np.float32)
    end_array = np.asarray(end, dtype=np.float32)
    steps = int(np.max(np.abs(end_array - start_array))) + 1
    for point in np.linspace(start_array, end_array, steps):
        z_index, y_index, x_index = np.rint(point).astype(int)
        target[z_index, y_index, x_index] = value


def _draw_rectangle_zyx(
    target: np.ndarray,
    *,
    z: int,
    y0: int,
    y1: int,
    x0: int,
    x1: int,
    value: np.uint16,
) -> None:
    _draw_line_zyx(target, (z, y0, x0), (z, y0, x1), value)
    _draw_line_zyx(target, (z, y1, x0), (z, y1, x1), value)
    _draw_line_zyx(target, (z, y0, x0), (z, y1, x0), value)
    _draw_line_zyx(target, (z, y0, x1), (z, y1, x1), value)


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
