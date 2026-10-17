"""Freeze LIMITED recorded buffers after an acceptance-recorder failure.

No provider, GUI, scientific execution, repair or candidate selection occurs.
This cannot recover absent output states, table values, mesh-object metadata,
cleanup reports or final UI-adoption receipts. Default mode is read-only.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from qualify_private import (
    PrivatePackageImportGuard,
    assert_private_imports,
)
from replay_workflow_investigation_selection import read_json, run_inventory
from run_workflow_feedback_refinement import (
    EXPECTED_BASELINE_LOCK,
    assert_feedback_request,
    bind_additive_graph,
    verify_baseline,
)
from workflow_investigation_evidence import (
    RAW_PIXELS_SHA256,
    EvidenceJournal,
    InferenceCallGuard,
    SampleGeneratorCallGuard,
    json_bytes,
    sha256,
)


def inventory_hash(inventory):
    return sha256(json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode())


def verify_events(root):
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text("utf-8").splitlines()
    ]
    previous = "0" * 64
    for number, event in enumerate(events, 1):
        if event["sequence"] != number or event["previous_event_sha256"] != previous:
            raise ValueError("The recorder event chain changed.")
        previous = sha256(json_bytes(event))
    starts = [item for item in events if item["kind"] == "explicit_one_round_start"]
    runs = [item for item in events if item["kind"] == "normal_product_run"]
    adds = [item for item in events if item["kind"] == "normal_product_add"]
    failures = [item for item in events if item["kind"] == "acceptance_failed"]
    if (
        len(starts) != 1
        or [item["details"]["phase"] for item in runs] != ["baseline", "refinement"]
        or [item["details"]["phase"] for item in adds] != ["baseline", "refinement"]
        or not all(item["details"]["accepted"] is True for item in runs)
        or len(failures) != 1
        or failures[0]["details"]["message"]
        != "The owned normal run failed or exceeded 300 seconds."
    ):
        raise ValueError(
            "This is not the one authorized run with the recorded readiness failure."
        )
    start = datetime.fromisoformat(starts[0]["utc"]).timestamp()
    deadline = start + 1800
    requests = [item for item in events if item["kind"] == "product_model_request"]
    responses = [item for item in events if item["kind"] == "product_model_response"]
    expected = [
        (kind, f"requests/request-{number:03d}")
        for number, kind in enumerate(("assessment", "revision", "assessment"), 1)
    ]
    for records in (requests, responses):
        if [
            (item["details"]["request_kind"], item["details"]["prefix"])
            for item in records
        ] != expected:
            raise ValueError("The three bounded model phases changed.")
    calls = [item for item in events if item["kind"] == "actual_model_call"]
    if not 3 <= len(calls) <= 6 or len(
        {item["details"]["prefix"] for item in calls}
    ) != len(calls):
        raise ValueError("The actual native turn bound changed.")
    for event in [*requests, *responses, *calls]:
        if not start <= datetime.fromisoformat(event["utc"]).timestamp() < deadline:
            raise ValueError("A model phase escaped the original bounded deadline.")
    return events, previous, deadline


def collect_node_buffers(root, model_key_mapping):
    """Identify persisted byte payloads by exact model-owned node IDs, not order."""
    result = []
    for key, node_id in model_key_mapping.items():
        files = []
        for path in sorted(
            (root / "final-workflow/outputs").glob(f"{node_id}-port-*.npy")
        ):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError("A recorded scientific payload is redirected.")
            array = np.load(path, allow_pickle=False)
            files.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "file_sha256": sha256(path.read_bytes()),
                    "dtype": array.dtype.str,
                    "shape": list(array.shape),
                    "raw_c_order_sha256": sha256(array.tobytes(order="C")),
                }
            )
        result.append(
            {
                "model_key": key,
                "node_id": node_id,
                "files": files,
                "complete_output_state_available": False,
            }
        )
    return result


def verify_recording(args):
    root = args.recorded_run.resolve(strict=True)
    before = run_inventory(root)
    if inventory_hash(before) != args.expected_inventory_sha256:
        raise ValueError("The independently pinned recorded-run inventory changed.")
    if any(
        (root / name).exists()
        for name in (
            "receipt.json",
            "selected-result-lock.json",
            "final-workflow/full-state.json",
        )
    ):
        raise ValueError(
            "Limited adapter requires incomplete recording and no full states."
        )
    failure = read_json(root / "terminal-write-failure.json")
    if failure != {
        "reason": "Out of range float values are not JSON compliant: nan",
        "scientific_success_established": False,
        "status": "incomplete",
        "type": "ValueError",
    }:
        raise ValueError(
            "The recorded terminal failure is not the expected strict-JSON failure."
        )
    events, chain, deadline = verify_events(root)
    protocol = read_json(root / "protocol.json")
    if (
        protocol["baseline_lock_sha256"] != EXPECTED_BASELINE_LOCK
        or protocol["maximum_additional_scientific_runs"] != 1
        or protocol["recorded_assessment_continuation"] is not None
    ):
        raise ValueError("This is not the fresh explicitly authorized one-round retry.")
    baseline = Path(protocol["baseline_replay"]).resolve(strict=True)
    prior_binding = read_json(baseline / "continuation-binding.json")
    prior_finalization = read_json(baseline / "finalization-provenance.json")
    baseline_args = SimpleNamespace(
        baseline_replay=baseline,
        recorded_run=Path(prior_binding["original_run"]),
        finalization=Path(prior_finalization["root"]),
        qualified_source=args.qualified_source,
        qualification_receipt=args.qualification_receipt,
    )
    bundle, _baseline, baseline_inventory, qualification = verify_baseline(
        baseline_args
    )
    if (
        Path(protocol["qualified_source"]).resolve(strict=True)
        != args.qualified_source.resolve(strict=True)
        or protocol["qualified_wheel_sha256"] != qualification["wheel_sha256"]
        or read_json(root / "qualification-receipt.json") != qualification
        or protocol["include_input_previews"] is not True
        or protocol["include_output_previews"] is not True
    ):
        raise ValueError("The recorded package or both preview permissions changed.")
    if (
        protocol["baseline_inventory"] != baseline_inventory
        or protocol["original_run_inventory"] != bundle.before
        or protocol["original_finalization_inventory"] != bundle.finalization_before
    ):
        raise ValueError("The prior immutable scientific baseline binding changed.")
    baseline_roles = read_json(root / "baseline/exact-output-verification.json")
    if {item["role"] for item in baseline_roles} != {
        item[0]["role"] for item in bundle.expected
    }:
        raise ValueError("The recorded seven-role baseline verification changed.")
    recipe = read_json(root / "refinement/proposal.json")
    response = read_json(root / "requests/request-002/validated-response.json")
    if response["proposal"] != recipe:
        raise ValueError(
            "The model's validated proposal differs from the added recipe."
        )
    prepared = read_json(root / "refinement/prepared-workflow.json")
    final_graph = read_json(root / "final-workflow/workflow.json")
    owned = read_json(root / "refinement/binding.json")
    dispatch = read_json(root / "refinement/dispatch.json")
    request = read_json(root / "requests/request-003/public-request.json")
    assessment_response = read_json(
        root / "requests/request-003/validated-response.json"
    )
    assert_feedback_request(request)
    first_request = read_json(root / "requests/request-001/public-request.json")
    if (
        request["image_permissions"] != {"input_images": True, "output_images": True}
        or first_request["image_permissions"] != request["image_permissions"]
        or first_request["source"] != request["source"]
    ):
        raise ValueError("Captured source identity or preview permissions changed.")
    source_path = root / "final-workflow/outputs/input-port-0.npy"
    source = np.load(source_path, allow_pickle=False)
    if (
        source.dtype != np.uint8
        or source.shape != (12, 96, 128)
        or sha256(source.tobytes(order="C")) != RAW_PIXELS_SHA256
    ):
        raise ValueError("Persisted source pixels differ from the approved input.")
    source_buffer = {
        "path": source_path.relative_to(root).as_posix(),
        "file_sha256": sha256(source_path.read_bytes()),
        "raw_c_order_sha256": RAW_PIXELS_SHA256,
        "dtype": source.dtype.str,
        "shape": list(source.shape),
    }
    if dispatch["run_id"] != request["run_id"] or dispatch["run_id"] != "background-2":
        raise ValueError(
            "The captured final assessment does not bind the recorded owned Run."
        )
    qualified_root = args.qualified_source.resolve(strict=True)
    package = qualified_root / qualification["package_directory"]
    assert_private_imports(qualified_root)
    sys.path.insert(0, str(package.parent))
    importer = PrivatePackageImportGuard(
        qualified_root, module_file_sha256=qualification["module_file_sha256"]
    )
    sys.meta_path.insert(0, importer)
    sample_guard, inference_guard = (
        SampleGeneratorCallGuard(package / "_sample_data.py"),
        InferenceCallGuard(package),
    )
    sample_guard.install()
    inference_guard.install()
    try:
        from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID, SPATIAL_OPERATIONS
        from napari_vipp.core.workflow_assessment_model import validate_assessment

        _binding, mapping = bind_additive_graph(
            recipe, prepared, final_graph, owned["added_node_ids"], SPATIAL_OPERATIONS
        )
        expected_ids = [mapping[item["key"]] for item in recipe["nodes"]]
        if expected_ids != owned["added_node_ids"]:
            raise ValueError("The model keys no longer preserve the exact Add order.")
        aliases = {f"n{index}": item for index, item in enumerate(recipe["nodes"], 1)}
        public_nodes = request["candidate_graph"]["nodes"]
        if [node["key"] for node in public_nodes] != list(aliases):
            raise ValueError("The assessment graph aliases changed.")
        key_aliases = {item["key"]: alias for alias, item in aliases.items()}
        expected_wires = sorted(
            (
                "input" if edge["source"] == "input" else key_aliases[edge["source"]],
                key_aliases[edge["target"]],
                edge["source_port"],
                edge["target_port"],
            )
            for edge in recipe["connections"]
        )
        public_wires = sorted(
            (edge["source"], edge["target"], edge["source_port"], edge["target_port"])
            for edge in request["candidate_graph"]["connections"]
        )
        if public_wires != expected_wires:
            raise ValueError(
                "The captured assessment wiring differs from the model recipe."
            )
        graph_nodes = {node["id"]: node for node in final_graph["nodes"]}
        for node in public_nodes:
            item = aliases[node["key"]]
            actual = graph_nodes[mapping[item["key"]]]
            if node["operation_id"] != item["operation_id"] or any(
                key not in actual["params"] or actual["params"][key] != value
                for key, value in node["params"].items()
            ):
                raise ValueError(
                    "The assessed node lineage or supplied parameters changed."
                )
        context = SimpleNamespace(
            assessment_id=request["assessment_id"],
            run_id=request["run_id"],
            _evidence=request,
        )
        report = validate_assessment(
            json.dumps(assessment_response["assessment"]), context
        )
        buffers = collect_node_buffers(root, mapping)
        descriptions = {item["key"]: item for item in recipe["nodes"]}
        for item in buffers:
            authored = descriptions[item["model_key"]]
            spec = NODE_LIBRARY_BY_ID[authored["operation_id"]]
            item.update(
                operation_id=authored["operation_id"],
                model_reason=authored["reason"],
                model_parameters=authored["params"],
                catalogue_output_type=spec.output_type,
                catalogue_presentation_only=spec.presentation_only,
            )
            if spec.presentation_only and spec.output_type == "none":
                if spec.id != "review_images" or item["files"]:
                    raise ValueError(
                        "An unexpected presentation-only payload was recorded."
                    )
        if sample_guard.calls or inference_guard.calls:
            raise ValueError("Offline descriptor isolation was violated.")
        guard_receipt = {
            "sample": sample_guard.receipt(),
            "inference": inference_guard.receipt(),
        }
    finally:
        sample_guard.close()
        inference_guard.close()
        sys.meta_path.remove(importer)
    if run_inventory(root) != before:
        raise ValueError("The recorded run changed during read-only verification.")
    return {
        "root": root,
        "inventory": before,
        "inventory_sha256": inventory_hash(before),
        "event_chain_sha256": chain,
        "deadline_utc": deadline,
        "recipe": recipe,
        "mapping": mapping,
        "buffers": buffers,
        "dispatch": dispatch,
        "report": report.to_dict(),
        "request": request,
        "source_buffer": source_buffer,
        "qualification": qualification,
        "failure": failure,
        "baseline_lock_sha256": EXPECTED_BASELINE_LOCK,
        "guards": guard_receipt,
        "model_calls_recorded": sum(
            item["kind"] == "actual_model_call" for item in events
        ),
    }


def write_descriptor(bundle, output):
    if not time.time() < bundle["deadline_utc"]:
        raise ValueError("The original retry recording deadline expired.")
    output = output.resolve()
    if output.is_relative_to(bundle["root"]):
        raise ValueError("The limited descriptor must remain outside the original run.")
    journal = EvidenceJournal(output)
    limitations = [
        "The original recording remains incomplete after a presentation-node "
        "readiness timeout and strict-JSON NaN failure.",
        "No final full-state manifest or ordinary completion receipt was persisted; "
        "scientific node states cannot be independently asserted from this bundle.",
        "No raw fresh execution cleanup report or active/pending execution "
        "snapshot was persisted.",
        "The final model report was strictly validated for guarded captured "
        "request003; final UI adoption is not proven by the provider observer alone.",
        "NPY payloads bind exact recorded filenames/model node IDs and file/raw "
        "hashes, not a missing per-port snapshot fingerprint.",
        "Complete per-output carried metadata, MeshObject names/IDs/colours/"
        "provenance and unpersisted table values cannot be recovered.",
        "Source grid information is captured request evidence, not a substitute "
        "for absent per-output metadata.",
        "This locks the single authorized recorded refinement, not a model-ranked "
        "winner or a mechanically completed acceptance Run.",
    ]
    descriptor = {
        "kind": "napari-vipp-limited-feedback-recording",
        "version": 1,
        "qualification": "limited-recorded-buffer-and-assessment-binding",
        "original_run": str(bundle["root"]),
        "original_inventory_sha256": bundle["inventory_sha256"],
        "original_file_inventory": bundle["inventory"],
        "event_chain_sha256": bundle["event_chain_sha256"],
        "source_grid_evidence": bundle["request"]["source"],
        "persisted_source_buffer": bundle["source_buffer"],
        "qualified_wheel_sha256": bundle["qualification"]["wheel_sha256"],
        "finalizer_source_sha256": sha256(Path(__file__).read_bytes()),
        "model_key_to_node_id": bundle["mapping"],
        "recorded_node_buffers": bundle["buffers"],
        "exact_model_recipe": bundle["recipe"],
        "owned_dispatch_record": bundle["dispatch"],
        "strict_validated_model_assessment": bundle["report"],
        "original_terminal_failure": bundle["failure"],
        "model_calls_recorded": bundle["model_calls_recorded"],
        "new_model_calls": 0,
        "new_scientific_runs": 0,
        "mechanical_acceptance_pass": False,
        "scientific_accuracy_established": False,
        "limitations": limitations,
        "isolation": bundle["guards"],
    }
    journal.document("limited-recording.json", descriptor)
    if run_inventory(bundle["root"]) != bundle["inventory"]:
        raise ValueError("The recorded run changed before the limited snapshot lock.")
    if not time.time() < bundle["deadline_utc"]:
        raise ValueError("The original retry recording deadline expired before lock.")
    return journal.lock_selection(
        {
            "candidate_id": "recorded-refinement1",
            "run_id": bundle["dispatch"]["run_id"],
            "role_mapping": [
                {
                    "role": "model-node:" + item["model_key"],
                    "node_id": item["node_id"],
                    "recorded_files": item["files"],
                }
                for item in bundle["buffers"]
                if item["files"]
            ],
            "selection_basis": (
                "single authorized recorded candidate, not a model-ranked winner"
            ),
            "qualification": descriptor["qualification"],
            "mechanical_acceptance_pass": False,
            "original_inventory_sha256": bundle["inventory_sha256"],
            "limitations": limitations,
        }
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorded-run", type=Path, required=True)
    parser.add_argument("--expected-inventory-sha256", required=True)
    parser.add_argument("--qualified-source", type=Path, required=True)
    parser.add_argument("--qualification-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-approved", action="store_true")
    args = parser.parse_args(argv)
    bundle = verify_recording(args)
    if not args.write_approved:
        print(
            json.dumps(
                {
                    "limited_recording_verified": True,
                    "model_keys": len(bundle["buffers"]),
                    "mechanical_acceptance_pass": False,
                    "new_model_calls": 0,
                    "new_scientific_runs": 0,
                }
            )
        )
        return 0
    if args.output is None:
        parser.error("Explicit write mode requires a fresh --output.")
    print(json.dumps(write_descriptor(bundle, args.output)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
