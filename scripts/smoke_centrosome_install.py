"""Check installed Centrosome native kernels and VIPP wrappers without Qt."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

NATIVE_MODULES = (
    "_propagate",
    "_cpmorphology2",
    "_convex_hull",
    "_filter",
    "_lapjv",
    "_fastemd",
)


def smoke_centrosome_install(*, require_installed=False):
    """Fail on absent/wrong-version kernels, checkout leakage or a broken ABI."""
    import centrosome
    import numpy as np

    import napari_vipp
    from napari_vipp.core.cellprofiler_compartments import cellprofiler_smooth
    from napari_vipp.core.cellprofiler_propagation import cellprofiler_propagation

    if version("centrosome") != "1.3.4":
        raise RuntimeError("The installed Centrosome version must be exactly 1.3.4.")
    modules = {
        "napari_vipp": napari_vipp,
        "centrosome": centrosome,
        **{
            f"centrosome.{name}": importlib.import_module(f"centrosome.{name}")
            for name in NATIVE_MODULES
        },
    }
    paths = {name: Path(module.__file__).resolve() for name, module in modules.items()}
    if require_installed:
        prefix = Path(sys.prefix).resolve()
        repository = Path(__file__).resolve().parents[1]
        for name, path in paths.items():
            if repository in path.parents or not path.is_relative_to(prefix):
                raise RuntimeError(f"Smoke did not import installed {name}: {path}")

    image = np.zeros((5, 12), dtype=np.float64)
    seeds = np.zeros(image.shape, dtype=np.int32)
    seeds[2, 2], seeds[2, 9] = 7, 19
    mask = np.ones(image.shape, dtype=bool)
    originals = [array.copy() for array in (image, seeds, mask)]
    for array in (image, seeds, mask):
        array.setflags(write=False)
    actual = cellprofiler_propagation([image, seeds, mask], regularization=1.0)
    expected = np.full(image.shape, 19, dtype=np.int32)
    expected[:, :6] = 7
    np.testing.assert_array_equal(actual, expected)
    for array, original in zip((image, seeds, mask), originals, strict=True):
        np.testing.assert_array_equal(array, original)
    constant = np.full((8, 11), 0.25, dtype=np.float32)
    constant.setflags(write=False)
    np.testing.assert_allclose(
        cellprofiler_smooth(constant), constant, rtol=1e-7, atol=0
    )
    return {
        "status": "passed",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "versions": {
            name: version(name)
            for name in ("napari-vipp", "centrosome", "numpy", "scipy", "scikit-image")
        },
        "module_paths": {name: str(path) for name, path in paths.items()},
        "checks": [
            "native-extension imports",
            "two-seed propagation",
            "constant smoothing",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-installed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            smoke_centrosome_install(require_installed=args.require_installed), indent=2
        )
    )
