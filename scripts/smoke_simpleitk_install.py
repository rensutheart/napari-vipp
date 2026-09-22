"""Verify the installed native SimpleITK dependency and CPU median boundary."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import version
from pathlib import Path


def smoke_simpleitk_install(*, require_installed=False):
    """Fail rather than silently accept a missing/broken native dependency."""
    import numpy as np
    import SimpleITK as sitk
    from scipy import ndimage as ndi

    import napari_vipp
    from napari_vipp.core.operations import median_filter
    from napari_vipp.core.simpleitk_filters import median_filter_backend

    repository = Path(__file__).resolve().parents[1]
    installed = Path(napari_vipp.__file__).resolve()
    if require_installed and repository in installed.parents:
        raise RuntimeError(f"Smoke imported the source checkout: {installed}")
    if version("simpleitk") != "2.5.6":
        raise RuntimeError("SimpleITK is outside the qualified exact 2.5.6 pin.")

    rng = np.random.default_rng(20260922)
    cases = []
    for shape in ((256, 257), (3, 256, 257)):
        source = rng.integers(0, 65536, size=shape, dtype=np.uint16)
        original = source.copy()
        source.setflags(write=False)
        # The explicit reflected halo makes the ITK neighbourhood at the edge
        # match SciPy's half-sample-symmetric reflect mode. Z is not filtered.
        halo = [(0, 0)] * (source.ndim - 2) + [(2, 2), (2, 2)]
        padded = np.pad(source, halo, mode="symmetric")
        itk_image = sitk.GetImageFromArray(padded, isVector=False)
        radius = [2, 2] + [0] * (source.ndim - 2)
        native = sitk.GetArrayFromImage(sitk.Median(itk_image, radius))[
            (..., slice(2, -2), slice(2, -2))
        ]
        expected = ndi.median_filter(
            source, size=(1,) * (source.ndim - 2) + (5, 5), mode="reflect"
        )
        assert (
            median_filter_backend(
                source, size=5, xy_axes=(source.ndim - 2, source.ndim - 1)
            )
            == "simpleitk"
        ), "Installed smoke must exercise the accelerated operation."
        actual = median_filter(source, size=5)
        np.testing.assert_array_equal(native, expected)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(source, original)
        assert actual.dtype == source.dtype
        cases.append({"shape": list(shape), "dtype": str(source.dtype), "size": 5})

    return {
        "status": "passed",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "installed_module": str(installed),
        "versions": {name: version(name) for name in ("simpleitk", "numpy", "scipy")},
        "itk_version": sitk.Version_ITKVersionString(),
        "checks": cases,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-installed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            smoke_simpleitk_install(require_installed=args.require_installed), indent=2
        )
    )
