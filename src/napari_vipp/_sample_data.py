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
        _object_morphology_sample(),
        _mesh_morphology_sample(),
        _skeleton_network_sample(),
        _advanced_skeleton_network_sample(),
        _colocalization_sample(rng),
        _deconvolution_image_sample(),
        _measured_psf_sample(),
        _deconvolution_volume_sample(),
        _measured_psf_3d_sample(),
    ]


def _multichannel_volume_sample(z, y, x, rng):
    nuclei = _sphere(z, y, x, center=(6, 48, 62), radii=(15, 420, 520))
    neurite = _tube(y, x, center_y=42, amplitude=9, period=12, width=45)
    puncta = _puncta(z, y, x)
    channels = [
        np.clip(nuclei * 0.85 + rng.normal(0, 0.025, nuclei.shape), 0, 1),
        np.clip(neurite * 0.75 + rng.normal(0, 0.025, nuclei.shape), 0, 1),
        np.clip(puncta * 0.9 + rng.normal(0, 0.02, nuclei.shape), 0, 1),
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


def _object_morphology_sample():
    """Return separated 2D objects with varied shapes for morphology metrics."""
    yy, xx = np.indices((80, 104), dtype=np.float32)
    data = np.zeros((80, 104), dtype=np.uint16)

    circle = (yy - 20) ** 2 + (xx - 20) ** 2 <= 8**2
    ellipse = ((yy - 20) / 6) ** 2 + ((xx - 62) / 15) ** 2 <= 1
    rectangle = (yy >= 46) & (yy < 64) & (xx >= 10) & (xx < 31)
    concave = ((yy >= 43) & (yy < 68) & (xx >= 58) & (xx < 65)) | (
        (yy >= 61) & (yy < 68) & (xx >= 58) & (xx < 88)
    )

    data[circle] = 36_000
    data[ellipse] = 42_000
    data[rectangle] = 48_000
    data[concave] = 54_000
    background = ((yy + xx) % 64).astype(np.uint16)
    data += background

    metadata = {
        "name": "VIPP synthetic object morphology",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Separated 2D circle, ellipse, rectangle, and concave objects "
                "for validating derived object morphology, circularity, and Hu "
                "moment measurements."
            ),
            **_ome_image_metadata("YX", data.shape),
        },
    }
    return data, metadata, "image"


def _mesh_morphology_sample():
    """Return separated 3D objects for mesh/surface morphology validation."""
    z, y, x = np.indices((24, 84, 104), dtype=np.float32)
    data = np.zeros((24, 84, 104), dtype=np.uint16)

    sphere = ((z - 7) / 3.0) ** 2 + ((y - 22) / 10.0) ** 2 + (
        (x - 22) / 10.0
    ) ** 2 <= 1
    ellipsoid = ((z - 13) / 5.0) ** 2 + ((y - 22) / 7.0) ** 2 + (
        (x - 68) / 15.0
    ) ** 2 <= 1
    cuboid = (
        (z >= 5)
        & (z < 12)
        & (y >= 50)
        & (y < 68)
        & (x >= 10)
        & (x < 36)
    )
    lobe_left = ((z - 16) / 3.0) ** 2 + ((y - 56) / 9.0) ** 2 + (
        (x - 62) / 9.0
    ) ** 2 <= 1
    lobe_right = ((z - 16) / 3.0) ** 2 + ((y - 56) / 9.0) ** 2 + (
        (x - 82) / 9.0
    ) ** 2 <= 1
    bridge = (
        (z >= 14)
        & (z <= 18)
        & (y >= 52)
        & (y <= 60)
        & (x >= 62)
        & (x <= 82)
    )

    data[sphere] = 34_000
    data[ellipsoid] = 40_000
    data[cuboid] = 46_000
    data[lobe_left | lobe_right | bridge] = 52_000
    data[2, 78, 96] = 58_000
    data[2, 78, 97] = 58_000

    foreground_texture = ((z * 3 + y + x) % 96).astype(np.uint16)
    data += foreground_texture * (data > 0)

    metadata_block = _ome_image_metadata("ZYX", data.shape)
    metadata_block["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"] = [2.0, 0.5, 0.5]
    metadata = {
        "name": "VIPP synthetic 3D mesh morphology",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Separated 3D sphere-like, ellipsoid, cuboid, concave dumbbell, "
                "and tiny objects with anisotropic Z/Y/X scale for validating "
                "mesh surface, volume, convex hull, and sphericity metrics."
            ),
            **metadata_block,
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


def _colocalization_sample(rng):
    """Return two-channel 3D structures with known partial colocalization."""
    z, y, x = np.indices((10, 80, 96), dtype=np.float32)
    data = np.zeros((2, 10, 80, 96), dtype=np.float32)

    channel_1_shared = (
        ((z - 4.5) / 2.4) ** 2
        + ((y - 39) / 13.0) ** 2
        + ((x - 42) / 15.0) ** 2
        <= 1
    )
    channel_2_shared = (
        ((z - 4.5) / 2.4) ** 2
        + ((y - 41) / 13.0) ** 2
        + ((x - 50) / 15.0) ** 2
        <= 1
    )
    channel_1_only = (
        ((z - 3.0) / 1.8) ** 2
        + ((y - 20) / 7.0) ** 2
        + ((x - 24) / 7.0) ** 2
        <= 1
    )
    channel_2_only = (
        ((z - 6.5) / 2.0) ** 2
        + ((y - 58) / 8.0) ** 2
        + ((x - 72) / 9.0) ** 2
        <= 1
    )
    channel_1_puncta = (
        ((z - 7) / 1.1) ** 2
        + ((y - 24) / 3.0) ** 2
        + ((x - 70) / 3.0) ** 2
        <= 1
    )
    channel_2_puncta = (
        ((z - 7) / 1.1) ** 2
        + ((y - 27) / 3.0) ** 2
        + ((x - 73) / 3.0) ** 2
        <= 1
    )

    data[0, channel_1_shared] = 0.72
    data[1, channel_2_shared] = 0.78
    data[0, channel_1_only] = 0.86
    data[1, channel_2_only] = 0.82
    data[0, channel_1_puncta] = 0.95
    data[1, channel_2_puncta] = 0.90

    gradient = (x / max(float(x.max()), 1.0)) * 0.05
    data[0] += gradient
    data[1] += np.flip(gradient, axis=-1)
    data += rng.normal(0.0, 0.018, size=data.shape)
    data = (np.clip(data, 0.0, 1.0) * 65535).astype(np.uint16)

    metadata_block = _ome_image_metadata("CZYX", data.shape)
    metadata_block["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"] = [1.0, 1.0, 0.35, 0.35]
    metadata = {
        "name": "VIPP synthetic colocalization",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "channel_names": [
                "Channel 1 partial-overlap objects",
                "Channel 2 partial-overlap objects",
            ],
            "description": (
                "Two-channel 3D fluorescence sample with one partially "
                "overlapping object pair, single-channel-only objects, offset "
                "puncta, background gradients, and noise for validating "
                "colocalization thresholds, overlays, scatter plots, and RACC."
            ),
            **metadata_block,
        },
    }
    return data, metadata, "image"


def _deconvolution_image_sample():
    """Return a blurred/noisy 2D image for PSF-aware restoration review."""
    yy, xx = np.indices((96, 96), dtype=np.float32)
    clean = np.zeros((96, 96), dtype=np.float32)
    objects = (
        (28.0, 28.0, 4.5, 1.00),
        (34.0, 62.0, 6.0, 0.85),
        (62.0, 38.0, 5.0, 0.75),
        (68.0, 70.0, 3.5, 0.95),
    )
    for cy, cx, radius, intensity in objects:
        clean += intensity * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / radius**2)
    clean += 0.08 * (xx / max(float(xx.max()), 1.0))
    psf = _synthetic_deconvolution_psf()
    blurred = _convolve_same_reflect(clean, psf)
    rng = np.random.default_rng(123)
    noisy = np.clip(
        blurred + rng.normal(0.0, 0.018, size=blurred.shape).astype(np.float32),
        0.0,
        None,
    )
    data = (noisy / max(float(noisy.max()), 1e-6) * 65535).astype(np.uint16)
    metadata = {
        "name": "VIPP synthetic deconvolution image",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Blurred and noisy 2D bead/object image for reviewing "
                "Richardson-Lucy and Richardson-Lucy TV deconvolution with a "
                "separate measured PSF sample."
            ),
            **_ome_image_metadata("YX", data.shape),
        },
    }
    metadata["metadata"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"] = [0.12, 0.12]
    return data, metadata, "image"


def _measured_psf_sample():
    """Return a compact measured-PSF-like kernel with mild background error."""
    clean = _synthetic_deconvolution_psf(size=15, sigma=1.45)
    yy, xx = np.indices(clean.shape, dtype=np.float32)
    ring = 0.004 * np.sin(xx * 0.9) - 0.003 * np.cos(yy * 0.7)
    measured = (clean + ring).astype(np.float32)
    metadata = {
        "name": "VIPP synthetic measured PSF",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Compact 2D measured-PSF-style kernel with small background "
                "offsets so Prepare / Validate PSF can clip, center, and "
                "normalize it before deconvolution."
            ),
            **_ome_image_metadata("YX", measured.shape),
        },
    }
    metadata["metadata"]["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"] = [0.12, 0.12]
    return measured.astype(np.float32, copy=False), metadata, "image"


def _deconvolution_volume_sample():
    """Return a blurred/noisy 3D volume for PSF-aware restoration review."""
    z, y, x = np.indices((9, 48, 56), dtype=np.float32)
    clean = np.zeros((9, 48, 56), dtype=np.float32)
    objects = (
        (3.0, 17.0, 18.0, 1.5, 4.5, 4.0, 1.00),
        (5.0, 30.0, 34.0, 1.8, 5.0, 6.0, 0.85),
        (6.0, 18.0, 42.0, 1.3, 3.5, 3.5, 0.70),
    )
    for cz, cy, cx, rz, ry, rx, intensity in objects:
        clean += intensity * np.exp(
            -(
                ((z - cz) ** 2) / max(rz**2, 1e-6)
                + ((y - cy) ** 2) / max(ry**2, 1e-6)
                + ((x - cx) ** 2) / max(rx**2, 1e-6)
            )
        )
    clean += 0.05 * (x / max(float(x.max()), 1.0))
    clean += 0.03 * (z / max(float(z.max()), 1.0))
    psf = _synthetic_deconvolution_psf_3d()
    blurred = _convolve_same_reflect(clean, psf)
    rng = np.random.default_rng(321)
    noisy = np.clip(
        blurred + rng.normal(0.0, 0.014, size=blurred.shape).astype(np.float32),
        0.0,
        None,
    )
    data = (noisy / max(float(noisy.max()), 1e-6) * 65535).astype(np.uint16)
    metadata_block = _ome_image_metadata("ZYX", data.shape)
    metadata_block["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"] = [0.35, 0.12, 0.12]
    metadata = {
        "name": "VIPP synthetic 3D deconvolution volume",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Small blurred and noisy ZYX volume with anisotropic Z/Y/X "
                "scale for reviewing volumetric Richardson-Lucy and "
                "Richardson-Lucy TV deconvolution."
            ),
            **metadata_block,
        },
    }
    return data, metadata, "image"


def _measured_psf_3d_sample():
    """Return a compact 3D measured-PSF-like kernel with mild background error."""
    clean = _synthetic_deconvolution_psf_3d(
        z_size=5,
        xy_size=9,
        sigma_z=0.95,
        sigma_xy=1.35,
    )
    z, y, x = np.indices(clean.shape, dtype=np.float32)
    ripple = 0.0025 * np.sin(x * 1.1) - 0.002 * np.cos(y * 0.8)
    ripple += 0.0015 * (z - z.mean())
    measured = (clean + ripple).astype(np.float32)
    metadata_block = _ome_image_metadata("ZYX", measured.shape)
    metadata_block["ome"]["multiscales"][0]["datasets"][0][
        "coordinateTransformations"
    ][0]["scale"] = [0.35, 0.12, 0.12]
    metadata = {
        "name": "VIPP synthetic 3D measured PSF",
        "visible": False,
        "metadata": {
            "napari_vipp_sample": True,
            "napari_vipp_preferred_input": False,
            "description": (
                "Compact ZYX measured-PSF-style kernel with slight background "
                "offsets for validating 3D PSF preparation and deconvolution."
            ),
            **metadata_block,
        },
    }
    return measured.astype(np.float32, copy=False), metadata, "image"


def _synthetic_deconvolution_psf(
    *,
    size: int = 15,
    sigma: float = 1.7,
) -> np.ndarray:
    coords = np.arange(size, dtype=np.float32) - size // 2
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    psf = np.exp(-(yy**2 + xx**2) / (2.0 * float(sigma) ** 2))
    psf += 0.03 * np.exp(-(yy**2 + xx**2) / (2.0 * (float(sigma) * 2.2) ** 2))
    psf = np.maximum(psf, 0.0)
    return (psf / psf.sum(dtype=np.float64)).astype(np.float32)


def _synthetic_deconvolution_psf_3d(
    *,
    z_size: int = 5,
    xy_size: int = 9,
    sigma_z: float = 1.15,
    sigma_xy: float = 1.55,
) -> np.ndarray:
    z_coords = np.arange(z_size, dtype=np.float32) - z_size // 2
    xy_coords = np.arange(xy_size, dtype=np.float32) - xy_size // 2
    zz, yy, xx = np.meshgrid(z_coords, xy_coords, xy_coords, indexing="ij")
    psf = np.exp(
        -(
            (zz**2) / (2.0 * float(sigma_z) ** 2)
            + (yy**2 + xx**2) / (2.0 * float(sigma_xy) ** 2)
        )
    )
    psf += 0.025 * np.exp(
        -(
            (zz**2) / (2.0 * (float(sigma_z) * 1.8) ** 2)
            + (yy**2 + xx**2) / (2.0 * (float(sigma_xy) * 2.2) ** 2)
        )
    )
    psf = np.maximum(psf, 0.0)
    return (psf / psf.sum(dtype=np.float64)).astype(np.float32)


def _convolve_same_reflect(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    pad_width = tuple((size // 2, size // 2) for size in kernel.shape)
    padded = np.pad(image, pad_width, mode="reflect")
    output = np.zeros_like(image, dtype=np.float32)
    base_slices = tuple(slice(0, size) for size in image.shape)
    for kernel_index in np.ndindex(kernel.shape):
        slices = tuple(
            slice(offset, offset + base_slice.stop)
            for offset, base_slice in zip(kernel_index, base_slices, strict=True)
        )
        output += kernel[kernel_index] * padded[slices]
    return output


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
    centers = [
        (4, 35, 48, 1.0, 3.0, 46.0, 46.0),
        (7, 58, 76, 1.0, 3.0, 46.0, 46.0),
        (8, 46, 92, 1.0, 3.0, 46.0, 46.0),
        (5, 67, 56, 0.8, 1.0, 8.0, 8.0),
        (6, 48, 2, 0.9, 3.0, 46.0, 46.0),
    ]
    signal = np.zeros_like(z, dtype=np.float32)
    for cz, cy, cx, amplitude, z_width, y_width, x_width in centers:
        signal += amplitude * np.exp(
            -(
                (z - cz) ** 2 / z_width
                + (y - cy) ** 2 / y_width
                + (x - cx) ** 2 / x_width
            )
        )
    return np.clip(signal, 0, 1)
