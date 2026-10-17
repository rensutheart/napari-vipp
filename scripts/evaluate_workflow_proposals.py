"""Small reproducible authoring comparison; no scientific pipeline execution.

Without --model this validates only the prepared, agent-authored reference.
With --model it makes at most two proposal calls for each of three synthetic
tasks to an already-running trusted loopback server. It never starts/pulls a
model, sends images, uses cloud credentials or writes a report automatically.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from napari_vipp.core.metadata import AxisMetadata, image_state_from_array  # noqa: E402
from napari_vipp.core.pipeline import PrototypePipeline  # noqa: E402
from napari_vipp.core.workflow_proposal_model import (  # noqa: E402
    ModelProposalError,
    ModelSettings,
    generate_proposal,
)
from napari_vipp.core.workflow_proposals import (  # noqa: E402
    build_catalogue,
    capture_proposal_context,
    reference_proposal,
    validate_proposal,
)

TASKS = (
    {
        "id": "bright_objects_2d",
        "shape": (48, 64),
        "scales": (0.2, 0.2),
        "objective": "Suggest a starting workflow to smooth this 2D intensity "
        "image, identify bright isolated objects, measure the objects and "
        "review original intensity beside segmentation. Do not run it.",
    },
    {
        "id": "anisotropic_volume_3d",
        "shape": (12, 40, 48),
        "scales": (0.8, 0.2, 0.2),
        "objective": "Suggest a 3D starting workflow for bright isolated objects. "
        "Use isotropic Gaussian smoothing with a physical sigma of 0.4 um, "
        "expressing it explicitly in the supported voxel parameters using the "
        "supplied calibration. Use one volume-wide Otsu threshold, label 3D "
        "objects, measure them and review intensity with segmentation. Do not run it.",
    },
    {
        "id": "unsupported_lineage_request",
        "shape": (32, 40),
        "scales": (0.2, 0.2),
        "objective": "Track cell divisions and reconstruct parent-daughter "
        "lineages over time from this single image. Ask me if that is not "
        "possible with this source or the supplied catalogue. Do not invent data.",
    },
)


def make_context(task):
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    shape = task["shape"]
    coordinates = np.indices(shape, dtype=np.float32)
    data = np.zeros(shape, dtype=np.float32)
    for fraction in (0.30, 0.68):
        radius = sum(
            ((axis - fraction * size) * scale / 0.8) ** 2
            for axis, size, scale in zip(
                coordinates, shape, task["scales"], strict=True
            )
        )
        data += np.exp(-radius).astype(np.float32)
    data.flags.writeable = False
    axes = tuple(
        AxisMetadata(name, "space", "um", scale=scale)
        for name, scale in zip(
            "zyx" if len(shape) == 3 else "yx", task["scales"], strict=True
        )
    )
    pipeline.outputs["input"] = data
    pipeline.output_states["input"] = image_state_from_array(data, axes=axes)
    return capture_proposal_context(pipeline, "input")


def curated_reference(context, task_id):
    reference = reference_proposal(context)
    if task_id == "anisotropic_volume_3d":
        blur = next(node for node in reference["nodes"] if node["key"] == "blur")
        blur["params"].update(sigma_z=0.5, sigma_y=2.0, sigma_x=2.0)
        reference["assumptions"].append(
            "Explicit 0.4 um sigma divided by known Z/Y/X voxel scale: 0.5/2/2."
        )
    elif task_id == "unsupported_lineage_request":
        reference.update(
            summary="One scalar image and this catalogue cannot establish lineage.",
            assumptions=[],
            questions=[
                "Can you provide a time series and an approved tracking/lineage "
                "workflow? Those operations are outside this prototype catalogue."
            ],
            nodes=[],
            connections=[],
        )
    return reference


def intent_checks(proposal, task_id):
    """Narrow predeclared checks, not an LLM judge or biological accuracy score."""
    if task_id == "unsupported_lineage_request":
        return {
            "asks_question": bool(proposal["questions"]),
            "does_not_offer_a_runnable_lineage_graph": not proposal["nodes"],
        }
    by_operation = {node["operation_id"]: node for node in proposal["nodes"]}
    checks = {
        "includes_threshold_labels_measurement_review": all(
            name in by_operation
            for name in (
                "otsu_threshold",
                "label_connected_components",
                "measure_objects",
                "review_images",
            )
        ),
        "explicitly_records_assumptions": bool(proposal["assumptions"]),
    }

    def linked(source, target, port):
        return any(
            edge["source"] == source
            and edge["target"] == target
            and edge["source_port"] == 0
            and edge["target_port"] == port
            for edge in proposal["connections"]
        )

    mask = by_operation.get("otsu_threshold", {}).get("key")
    labels = by_operation.get("label_connected_components", {}).get("key")
    measure = by_operation.get("measure_objects", {}).get("key")
    review = by_operation.get("review_images", {}).get("key")
    checks["measures_segmented_labels_and_reviews_against_original"] = all(
        value is not None for value in (mask, labels, measure, review)
    ) and all(
        (
            linked(mask, labels, 0),
            linked(labels, measure, 0),
            linked("input", review, 0),
            linked(labels, review, 1),
        )
    )
    if task_id == "anisotropic_volume_3d":
        params = by_operation.get("gaussian_blur_3d", {}).get("params", {})
        checks["requested_physical_sigma_converted_explicitly"] = all(
            type(params.get(key)) in (int, float)
            and math.isclose(params[key], expected, rel_tol=1e-9)
            for key, expected in (("sigma_z", 0.5), ("sigma_y", 2.0), ("sigma_x", 2.0))
        )
        checks["uses_one_volume_threshold"] = (
            by_operation.get("otsu_threshold", {})
            .get("params", {})
            .get("threshold_scope")
            == "Stack histogram"
        )
    else:
        checks["uses_2d_smoothing"] = "gaussian_blur" in by_operation
    return checks


def evaluate(settings=None):
    rows = []
    for task in TASKS:
        context = make_context(task)
        reference = curated_reference(context, task["id"])
        prepared = validate_proposal(reference, context)
        row = {
            "task": task["id"],
            "objective": task["objective"],
            "source_summary": context.source_summary,
            "reference": {
                "origin": "prepared agent-authored fixture; not an API model response",
                "contract_valid": True,
                "can_apply": prepared.can_apply,
                "intent_checks": intent_checks(reference, task["id"]),
                "proposal": reference,
            },
            "local": {"status": "not run; --model opt-in required"},
        }
        if settings is not None:
            started = time.monotonic()
            try:
                result = generate_proposal(context, task["objective"], settings)
                row["local"] = {
                    "status": "completed",
                    "first_attempt_contract_valid": result.attempts[0].valid,
                    "final_contract_valid": True,
                    "can_apply": result.prepared.can_apply,
                    "intent_checks": intent_checks(result.proposal, task["id"]),
                    "request_sha256": result.request_sha256,
                    "attempts": [asdict(attempt) for attempt in result.attempts],
                    "proposal": result.proposal,
                }
            except ModelProposalError as exc:
                row["local"] = {
                    "status": "failed",
                    "error": str(exc),
                    "attempts": [asdict(attempt) for attempt in exc.attempts],
                }
            row["local"]["elapsed_seconds"] = time.monotonic() - started
        rows.append(row)
    return {
        "kind": "napari-vipp-proposal-authoring-comparison",
        "version": 1,
        "catalogue": build_catalogue(),
        "settings": asdict(settings) if settings else None,
        "inference_preset": {
            "json_output": True,
            "tools": False,
            "ollama": {
                "thinking": False,
                "context_tokens": 16384,
                "seed": 0,
                "allow_prompt_truncation": False,
            },
        },
        "limits": "Three synthetic authoring cases; one run per case; max two "
        "generation attempts per case. No tool use or scientific execution.",
        "interpretation": "Prepared references are not cloud/API performance. "
        "Graph validation and narrow intent checks do not establish scientific "
        "correctness. No general Qwen-versus-frontier ranking can be inferred.",
        "tasks": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", help="Explicit opt-in: installed local model identifier"
    )
    parser.add_argument(
        "--kind", choices=("ollama", "openai-compatible"), default="ollama"
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--max-attempts", type=int, choices=(1, 2), default=2)
    parser.add_argument("--timeout-seconds", type=float, default=60)
    args = parser.parse_args()
    settings = None
    if args.model:
        settings = ModelSettings(
            kind=args.kind,
            base_url=args.base_url,
            model=args.model,
            max_attempts=args.max_attempts,
            timeout_seconds=args.timeout_seconds,
        )
        settings.validate()
    print(json.dumps(evaluate(settings), ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
