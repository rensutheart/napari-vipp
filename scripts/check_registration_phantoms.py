"""Record known-motion accuracy and before/after panels, without Qt or real data.

Run with --output-dir outside the repository. This is a small synthetic smoke
qualification, not a claim that arbitrary biological images will register.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from napari_vipp.core.metadata import (
    AxisMetadata,
    SourceMetadata,
    image_state_from_array,
)
from napari_vipp.core.registration import apply_transform, estimate_registration
from napari_vipp.core.registration_samples import (
    affine_pair,
    drift_series,
    landmark_errors,
    rigid_volume_pair,
    translation_pair,
)


def state_for(phantom, data, *, reference=False):
    order = "ZYX"[-len(phantom.spacing) :] if reference else phantom.axes
    spatial = "ZYX"[-len(phantom.spacing) :]
    scale = dict(zip(spatial, phantom.spacing, strict=True))
    origin = dict(zip(spatial, phantom.origin, strict=True))
    axes = tuple(
        AxisMetadata(
            name=axis.lower(),
            type="time" if axis == "T" else "channel" if axis == "C" else "space",
            unit="second" if axis == "T" else None if axis == "C" else "micrometer",
            scale=scale.get(axis, 2.0 if axis == "T" else 1.0),
            translation=origin.get(axis, 0.0),
        )
        for axis in order
    )
    return image_state_from_array(
        data,
        axes=axes,
        source_name=phantom.name + (" reference" if reference else " moving"),
        source=SourceMetadata(source_uuid=phantom.name + str(reference)),
    )


def _project(data):
    return np.max(data, axis=0) if data.ndim == 3 else data


def run_case(phantom, model, output):
    is_series = phantom.axes.startswith("TC")
    moving_state = state_for(phantom, phantom.moving)
    reference_state = state_for(phantom, phantom.reference, reference=True)
    started = time.perf_counter()
    transform, diagnostics = estimate_registration(
        phantom.moving,
        None if is_series else phantom.reference,
        moving_state=moving_state,
        reference_state=None if is_series else reference_state,
        mode="Time series" if is_series else "Two images",
        model=model,
        precision=20,
    )
    elapsed_estimate = time.perf_counter() - started
    started = time.perf_counter()
    aligned, coverage = apply_transform(
        phantom.moving,
        transform,
        image_state=moving_state,
        interpolation="Linear",
    )
    elapsed_apply = time.perf_counter() - started
    errors = [
        landmark_errors(matrix, phantom, time_index=index).tolist()
        for index, matrix in enumerate(transform.matrices)
    ]
    rmse = []
    for time_index in range(len(transform.matrices)):
        for channel in range(2 if is_series else 1):
            reference = phantom.moving[0, channel] if is_series else phantom.reference
            before = (
                phantom.moving[time_index, channel] if is_series else phantom.moving
            )
            after = aligned[time_index, channel] if is_series else aligned
            mask = coverage[time_index, channel] if is_series else coverage
            rmse.append(
                dict(
                    time=time_index,
                    channel=channel,
                    valid_voxels=int(mask.sum()),
                    before=float(
                        np.sqrt(np.mean((before[mask] - reference[mask]) ** 2))
                    ),
                    after=float(np.sqrt(np.mean((after[mask] - reference[mask]) ** 2))),
                )
            )
    record = dict(
        fixture=phantom.name,
        shape=list(phantom.moving.shape),
        axes=phantom.axes,
        spacing=list(phantom.spacing),
        origin=list(phantom.origin),
        model=model,
        landmark_error_unit="micrometer",
        maximum_landmark_error=float(np.max(errors)),
        landmark_errors=errors,
        valid_fraction=float(coverage.mean()),
        estimate_seconds=elapsed_estimate,
        apply_seconds=elapsed_apply,
        known_matrices=phantom.moving_to_reference,
        estimated_transform=transform.to_dict(),
        same_coverage_rmse=rmse,
        diagnostics=diagnostics.records(),
    )
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    frame = len(transform.matrices) - 1
    before = phantom.moving[frame, 0] if is_series else phantom.moving
    after = aligned[frame, 0] if is_series else aligned
    mask = coverage[frame, 0] if is_series else coverage
    reference = phantom.reference
    figure, axes = plt.subplots(1, 4, figsize=(13, 4), layout="constrained")
    for axis, values, title in zip(
        axes,
        (reference, before, after, mask),
        ("Reference", "Before", "After", "Valid coverage"),
        strict=True,
    ):
        axis.imshow(_project(values), cmap="gray", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_axis_off()
    figure.suptitle(
        f"{phantom.name} · {model} · "
        f"max landmark error {record['maximum_landmark_error']:.4g} µm\n"
        "3D panels use maximum-intensity projection; metrics use full volumes"
    )
    figure.savefig(output / f"{phantom.name}.png", dpi=130)
    plt.close(figure)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for factory, model in (
        (translation_pair, "Translation"),
        (rigid_volume_pair, "Rigid"),
        (affine_pair, "Affine"),
        (drift_series, "Translation"),
    ):
        record = run_case(factory(), model, args.output_dir)
        records.append(record)
        print(
            f"{record['fixture']}: maximum landmark error "
            f"{record['maximum_landmark_error']:.6g} micrometres; "
            f"estimate {record['estimate_seconds']:.2f}s",
            flush=True,
        )
    report = {
        "scope": "Four analytical fixtures; no real-data or general accuracy claim.",
        "method": (
            "Independent continuous object evaluation; "
            "no resampling to manufacture moving data."
        ),
        "comparison": "RMSE before and after uses the same valid-coverage region.",
        "results": records,
    }
    (args.output_dir / "results.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
