"""Sample data for the VIPP prototype."""

from __future__ import annotations

import numpy as np


def make_sample_data():
    """Return a small 3D fluorescence-like sample volume for napari."""
    z, y, x = np.indices((12, 96, 128), dtype=np.float32)
    sphere = ((z - 6) ** 2 / 15 + (y - 48) ** 2 / 420 + (x - 62) ** 2 / 520) < 1
    tube = np.exp(-((y - 42 - 9 * np.sin(x / 12)) ** 2) / 45)
    gradient = (x / x.max()) * 0.25
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.035, size=sphere.shape)
    data = np.clip(sphere * 0.75 + tube * 0.45 + gradient + noise, 0, 1)
    data = (data * 255).astype(np.uint8)

    metadata = {
        "name": "VIPP synthetic volume",
        "metadata": {"axes": "ZYX", "napari_vipp_sample": True},
    }
    return [(data, metadata, "image")]
