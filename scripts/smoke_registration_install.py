"""Check native SimpleITK registration and transform resources in an installation."""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
from dataclasses import replace
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path


def smoke_registration_install(*, require_installed=False):
    import numpy as np
    import SimpleITK as sitk

    import napari_vipp
    from napari_vipp.core.metadata import (
        AxisMetadata,
        SourceMetadata,
        image_state_from_array,
    )
    from napari_vipp.core.registration import apply_transform, estimate_registration
    from napari_vipp.core.registration_samples import landmark_errors, rigid_volume_pair
    from napari_vipp.core.transforms import load_transform, save_transform_output

    repository = Path(__file__).resolve().parents[1]
    installed = Path(napari_vipp.__file__).resolve()
    if require_installed and repository in installed.parents:
        raise RuntimeError(f"Smoke imported the source checkout: {installed}")
    if version("SimpleITK") != "2.5.6":
        raise RuntimeError("SimpleITK is outside the qualified exact 2.5.6 pin.")
    for resource in (
        "synthetic-registration-translation.json",
        "synthetic-registration-rigid-3d.json",
        "synthetic-registration-time-series.json",
    ):
        if not files("napari_vipp").joinpath("examples", resource).is_file():
            raise RuntimeError(f"Installed registration example is missing: {resource}")
    phantom = rigid_volume_pair()
    axes = tuple(
        AxisMetadata(name, "space", "micrometer", spacing, origin)
        for name, spacing, origin in zip(
            "zyx", phantom.spacing, phantom.origin, strict=True
        )
    )
    moving_state = image_state_from_array(
        phantom.moving,
        axes=axes,
        source_name="moving",
        source=SourceMetadata(source_uuid="smoke-moving-frame"),
    )
    reference_state = image_state_from_array(
        phantom.reference,
        axes=axes,
        source_name="reference",
        source=SourceMetadata(source_uuid="smoke-reference-frame"),
    )
    original = phantom.moving.copy()
    transform, diagnostics = estimate_registration(
        phantom.moving,
        phantom.reference,
        moving_state=moving_state,
        reference_state=reference_state,
        model="Rigid",
        iterations=150,
    )
    errors = landmark_errors(transform.matrices[0], phantom)
    assert max(errors) < 0.4, errors
    aligned, valid = apply_transform(
        phantom.moving, transform, image_state=moving_state
    )
    assert aligned.dtype == np.float64 and valid.dtype == bool
    assert diagnostics.records()[0]["correlation"] > 0.99
    np.testing.assert_array_equal(phantom.moving, original)
    # IDs beyond float64's exact range must never enter an interpolating cast.
    wide_id = np.uint64(2**63 + 19)
    labels = np.where(phantom.moving > 0.25, wide_id, np.uint64(0))
    labels.setflags(write=False)
    label_state = replace(
        moving_state, dtype="uint64", kind="label image", bit_depth="64-bit integer"
    )
    registered_labels, _ = apply_transform(labels, transform, image_state=label_state)
    assert registered_labels.dtype == np.uint64
    np.testing.assert_array_equal(np.unique(registered_labels), [np.uint64(0), wide_id])
    with tempfile.TemporaryDirectory(prefix="vipp-registration-smoke-") as folder:
        target = save_transform_output(transform, Path(folder) / "alignment.json")
        assert load_transform(target) == transform
    return {
        "status": "passed",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "installed_module": str(installed),
        "versions": {
            name: version(name) for name in ("SimpleITK", "numpy", "scikit-image")
        },
        "itk_version": sitk.Version_ITKVersionString(),
        "model": "Rigid",
        "shape": list(phantom.moving.shape),
        "maximum_landmark_error_micrometer": float(max(errors)),
        "checks": [
            "native optimizer",
            "3D anisotropic grid",
            "resampling",
            "input preservation",
            "exact wide-integer label IDs",
            "transform JSON",
            "packaged examples",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-installed", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            smoke_registration_install(require_installed=args.require_installed),
            indent=2,
        )
    )
