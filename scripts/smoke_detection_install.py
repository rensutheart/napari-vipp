"""Verify packaged detection workflows against independently planted locations."""

from __future__ import annotations

import argparse
import json
import platform
from importlib.metadata import distribution, version
from importlib.resources import files
from pathlib import Path


def smoke_detection_install(*, expected_package_root=None, require_installed=False):
    import numpy as np

    import napari_vipp
    from napari_vipp._sample_data import make_detection_sample_data
    from napari_vipp.core.compute import ComputeRequest
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.pipeline import PrototypePipeline, SourcePayload
    from napari_vipp.core.workflow import load_workflow, serialize_workflow

    imported = Path(napari_vipp.__file__).resolve()
    if expected_package_root is not None:
        root = Path(expected_package_root).resolve()
        if root not in imported.parents:
            raise RuntimeError(f"Wrong installed package: {imported}, expected {root}")
    distribution_root = Path(distribution("napari-vipp").locate_file("")).resolve()
    if require_installed and distribution_root not in imported.parents:
        raise RuntimeError(
            f"Imported code {imported} does not belong to installed distribution "
            f"{distribution_root}"
        )
    results = []
    samples = make_detection_sample_data()
    for dimensions in (2, 3):
        resource = files("napari_vipp").joinpath(
            "examples", f"synthetic-template-detection-{dimensions}d.json"
        )
        if not resource.is_file():
            raise RuntimeError(f"Missing packaged example: {resource}")
        workflow = load_workflow(Path(str(resource)))
        pipeline = PrototypePipeline()
        pipeline.restore_graph(
            workflow["nodes"], workflow["connections"], workflow["output_tunnels"]
        )
        data, kwargs, _kind = samples[dimensions - 2]
        original = data.copy()
        truth = kwargs["metadata"]["detection_ground_truth"]
        request = PipelineRunRequest(
            run_id=dimensions,
            workflow=serialize_workflow(pipeline),
            input_data=None,
            input_metadata=None,
            input_name="",
            source_payloads={
                "input": SourcePayload(data, kwargs["metadata"], kwargs["name"])
            },
            compute_request=ComputeRequest(mode="cpu"),
            manual_node_ids=frozenset(pipeline.manual_node_ids()),
        )
        result = execute_pipeline_request(request, raise_errors=True).pipeline
        table = result.outputs["peaks"]
        axes = truth["spatial_axis_order"].lower()
        found = {
            tuple(row[f"{axis}_index"] for axis in axes) for row in table.records()
        }
        expected = {tuple(center) for center in truth["centers"]}
        if found != expected:
            raise AssertionError(f"{dimensions}D detections: {found} != {expected}")
        np.testing.assert_array_equal(data, original)
        for row in table.records():
            for axis, scale, origin in zip(
                axes, truth["spacing"], truth["origin"], strict=True
            ):
                np.testing.assert_allclose(
                    row[f"{axis}_physical"],
                    origin + scale * row[f"{axis}_index"],
                    rtol=1e-12,
                    atol=1e-12,
                )
        score, valid = result.node_outputs["match"]
        assert valid.dtype == bool and score.shape == valid.shape
        evidence = table.detection_metadata
        assert evidence.accepted_count == evidence.returned_count == len(expected)
        assert not evidence.truncated
        results.append(
            {
                "dimensions": dimensions,
                "input_shape": list(data.shape),
                "score_shape": list(score.shape),
                "expected_and_found_centres": sorted(found),
                "false_positives": len(found - expected),
                "missed": len(expected - found),
                "coordinate_index_error": 0,
                "valid_score_count": int(np.count_nonzero(valid)),
                "detection_evidence": evidence.to_dict(),
            }
        )
    return {
        "status": "passed",
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "imported_package": str(imported),
        "distribution_metadata_root": str(distribution_root),
        "installed_distribution_verified": require_installed,
        "versions": {
            name: version(name)
            for name in ("napari-vipp", "numpy", "scipy", "scikit-image")
        },
        "cases": results,
        "limits": [
            "Synthetic known patterns, not acquired-image detection accuracy",
            "Fixed supplied template size/orientation; complete placements only",
            "Native operating system shown above only",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-package-root")
    parser.add_argument("--require-installed", action="store_true")
    parser.add_argument("--output-json", type=Path)
    options = parser.parse_args()
    evidence = smoke_detection_install(
        expected_package_root=options.expected_package_root,
        require_installed=options.require_installed,
    )
    text = json.dumps(evidence, indent=2)
    if options.output_json is not None:
        options.output_json.write_text(text + "\n", encoding="utf-8")
    print(text)
