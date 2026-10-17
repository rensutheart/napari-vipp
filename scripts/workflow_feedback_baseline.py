"""Read-only binding of the model's own prior limited recording as a baseline."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from replay_workflow_investigation_selection import read_json, run_inventory
from workflow_investigation_evidence import (
    RAW_PIXELS_SHA256,
    json_bytes,
    recipe_node_mapping,
    sha256,
)

PRIOR_INVENTORY_SHA256 = (
    "0e6c29e0f19fcb875e111c34b82a1ce64eab05c2bdd8cda2cf91cb9a0ef08477"
)
PRIOR_LIMITED_LOCK_SHA256 = (
    "d1221fb513427027373ec24d810d5704332d94fc8760e8ab1816a29ba0262750"
)


def source_local_graph(graph, node_ids):
    wanted = {"input", *node_ids}
    nodes = [node for node in graph["nodes"] if node["id"] in wanted]
    if {node["id"] for node in nodes} != wanted:
        raise ValueError("The exact prior source-local node set is incomplete.")
    edges = [edge for edge in graph["connections"] if edge["target"] in wanted]
    if any(edge["source"] not in wanted for edge in edges):
        raise ValueError("The model recipe depends on an unrecorded earlier branch.")
    return {**graph, "nodes": nodes, "connections": edges}


def adapt_own_model_baseline(original, recorded_run, limited_lock):
    root, lock_path = (
        Path(recorded_run).resolve(strict=True),
        Path(limited_lock).resolve(strict=True),
    )
    inventory = run_inventory(root)
    digest = sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    )
    if (
        digest != PRIOR_INVENTORY_SHA256
        or sha256(lock_path.read_bytes()) != PRIOR_LIMITED_LOCK_SHA256
    ):
        raise ValueError(
            "The pinned prior own-model recording or limited lock changed."
        )
    lock = read_json(lock_path)
    descriptor_path = lock_path.parent / "limited-recording.json"
    for artifact in lock["artifacts"]:
        path = (lock_path.parent / artifact["path"]).resolve(strict=True)
        if (
            not path.is_relative_to(lock_path.parent)
            or sha256(path.read_bytes()) != artifact["sha256"]
        ):
            raise ValueError("A limited-lock descriptor artifact changed.")
    descriptor = read_json(descriptor_path)
    if (
        descriptor["original_file_inventory"] != inventory
        or descriptor["mechanical_acceptance_pass"] is not False
        or descriptor["original_inventory_sha256"] != digest
    ):
        raise ValueError(
            "The limited recording identity or disclosed qualification changed."
        )
    recipe = read_json(root / "refinement/proposal.json")
    if (
        recipe
        != read_json(root / "requests/request-002/validated-response.json")["proposal"]
        or recipe != descriptor["exact_model_recipe"]
    ):
        raise ValueError("The own-model baseline is not its exact validated proposal.")
    mapping = descriptor["model_key_to_node_id"]
    prepared = source_local_graph(
        read_json(root / "refinement/prepared-workflow.json"), mapping.values()
    )
    executed = source_local_graph(
        read_json(root / "final-workflow/workflow.json"), mapping.values()
    )
    if (
        recipe_node_mapping(recipe, prepared, recipe["context_id"]) != mapping
        or recipe_node_mapping(recipe, executed, recipe["context_id"]) != mapping
    ):
        raise ValueError(
            "The full prior model recipe/parameter/wiring binding changed."
        )
    expected = []
    seen_files = set()
    for output in descriptor["recorded_node_buffers"]:
        key, node = output["model_key"], output["node_id"]
        if mapping[key] != node:
            raise ValueError("A persisted buffer's explicit model key changed.")
        arrays = {}
        for item in output["files"]:
            path = (root / item["path"]).resolve(strict=True)
            if (
                not path.is_relative_to(root)
                or sha256(path.read_bytes()) != item["file_sha256"]
            ):
                raise ValueError("A prior model buffer file changed.")
            array = np.load(path, allow_pickle=False)
            if (
                array.dtype.str != item["dtype"]
                or list(array.shape) != item["shape"]
                or sha256(array.tobytes(order="C")) != item["raw_c_order_sha256"]
            ):
                raise ValueError("A prior model scientific buffer changed.")
            arrays[path.name] = array
            seen_files.add(item["path"])
        if not arrays:
            continue  # Prior absent table/presentation data are not invented.
        base = f"{node}-port-0"
        role = {"role": "model-node:" + key, "node": key, "port": 0}
        if f"{base}.npy" in arrays and len(arrays) == 1:
            expected.append(
                (
                    role,
                    node,
                    {"array": {}, "prior_metadata_unavailable": True},
                    arrays[f"{base}.npy"],
                )
            )
        elif set(arrays) == {
            f"{base}-vertices.npy",
            f"{base}-faces.npy",
            f"{base}-face-object-ids.npy",
        }:
            data = SimpleNamespace(
                vertices=arrays[f"{base}-vertices.npy"],
                faces=arrays[f"{base}-faces.npy"],
                face_object_ids=arrays[f"{base}-face-object-ids.npy"],
            )
            expected.append(
                (
                    role,
                    node,
                    {
                        "persisted_mesh_arrays": True,
                        "prior_objects_and_metadata_unavailable": True,
                    },
                    data,
                )
            )
        else:
            raise ValueError(
                "The prior port's complete persisted buffer set is unsupported."
            )
    actual_owned_files = {
        path.relative_to(root).as_posix()
        for path in (root / "final-workflow/outputs").glob("*.npy")
        if any(path.name.startswith(node + "-port-") for node in mapping.values())
    }
    if seen_files != actual_owned_files or not expected:
        raise ValueError("Not every persisted model-owned scientific buffer was bound.")
    adapted = dict(vars(original))
    all_persisted = {
        path.relative_to(root).as_posix()
        for path in (root / "final-workflow/outputs").glob("*.npy")
    }
    excluded_contextual = sorted(
        all_persisted - seen_files - {"final-workflow/outputs/input-port-0.npy"}
    )
    persisted_source = np.load(
        root / "final-workflow/outputs/input-port-0.npy", allow_pickle=False
    )
    if (
        persisted_source.dtype != np.uint8
        or persisted_source.shape != (12, 96, 128)
        or sha256(persisted_source.tobytes(order="C")) != RAW_PIXELS_SHA256
    ):
        raise ValueError(
            "Prior persisted source pixels differ from the approved source."
        )
    adapted.update(
        recipe=recipe,
        prepared_graph=prepared,
        executed_graph=executed,
        expected=expected,
        candidate_id="prior-model-refinement1",
        recorded_buffers_only=True,
        persisted_source=persisted_source,
        own_model_baseline={
            "root": str(root),
            "inventory": inventory,
            "inventory_sha256": digest,
            "limited_root": str(lock_path.parent),
            "limited_inventory": run_inventory(lock_path.parent),
            "limited_lock_sha256": PRIOR_LIMITED_LOCK_SHA256,
            "fresh_full_metadata_is_new_recording": True,
            "old_missing_tables_metadata_objects_not_claimed_exact": True,
            "persisted_buffers_bound": len(seen_files),
            "persisted_source_buffers_bound": 1,
            "persisted_buffers_bound_including_source": len(seen_files) + 1,
            "older_contextual_branch_files_not_replayed": excluded_contextual,
            "older_contextual_branch_files_not_replayed_count": len(
                excluded_contextual
            ),
            "recipe_sha256": sha256(json_bytes(recipe)),
        },
    )
    return SimpleNamespace(**adapted)
