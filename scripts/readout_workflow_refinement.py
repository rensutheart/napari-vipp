"""Post-terminal own-output comparison only; default prints the plan, reads no run.

No provider, Qt, pipeline execution, image rendering, reference workflow, repair
or scientific ranking is available. Every supplied run needs an external exact
selection-lock hash and a genuinely finished recorded session before array reads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from replay_workflow_investigation_selection import artifact_path, run_inventory
from workflow_investigation_evidence import (
    RAW_PIXELS_SHA256,
    EvidenceJournal,
    read_recording_json,
    sha256,
)
from workflow_microedit_evidence import verified_lock

EVIDENCE_ROOT = Path(
    "C:/Users/rensu/Documents/VIPP/VIPP-local-tests/ai-assistant-efficient-refinement-20261017"
)
MAX_ARRAY_BYTES = 64 * 1024 * 1024
MAX_MASK_ELEMENTS = 4_000_000
MAX_MESH_FACES = 200_000
EXPECTED_PROTECTED_HASH = (
    "30c5b72fcdab922128d2e8c3a508d8088afb4628668dd3c0a3b995cc7b08b09f"
)
PLAN = {
    "mode": "prepare only; no run read or calculation",
    "gate": (
        "Externally pinned terminal selection lock, full artifact/event "
        "verification, genuine FINISH."
    ),
    "scope": (
        "Accepted baseline and every recorded own-model candidate "
        "from each supplied run; unranked."
    ),
    "metrics": [
        "Actual mask foreground and baseline additions/removals",
        "Exact 2340-voxel protected foreground retention",
        "Changes outside exact protected voxels (not a claimed anatomical centre ROI)",
        "Vertical per-(Z,X) first-to-last Y extent; includes gaps, not true thickness",
        "Recorded mesh diagnostics and raw-buffer boundary/nonmanifold edge check",
        "Model assessments/FINISH reasons and lookup/trial counters",
    ],
    "previews": (
        "Index existing saved PNG boards and exact action/view metadata; no new render."
    ),
    "forbidden": [
        "active-run reads",
        "human/reference workflows",
        "new pipeline/model calls",
        "repair",
        "quality score",
        "winner selection",
    ],
}


def canonical(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def array_hash(value):
    header = canonical({"shape": list(value.shape), "dtype": value.dtype.str})
    return hashlib.sha256(header + value.tobytes(order="C")).hexdigest()


def grid(state):
    shape = state.get("shape", state.get("source_shape"))
    axes = state.get("axes", state.get("spatial_axes", state.get("source_axes")))
    if (
        type(shape) is not list
        or type(axes) is not list
        or len(shape) != 3
        or len(axes) != 3
        or any(type(size) is not int or size <= 0 for size in shape)
    ):
        raise ValueError("The recorded source grid is not explicit native 3D.")
    result = []
    units = {
        "µm": "micrometer",
        "μm": "micrometer",
        "um": "micrometer",
        "micrometer": "micrometer",
    }
    for index, axis in enumerate(axes):
        scale, origin = axis.get("scale"), axis.get("translation")
        if (
            axis.get("name", "").upper() != "ZYX"[index]
            or axis.get("type") != "space"
            or type(scale) not in {float, int}
            or not np.isfinite(scale)
            or scale <= 0
            or type(origin) not in {float, int}
            or not np.isfinite(origin)
            or axis.get("unit") not in units
        ):
            raise ValueError("Native axes, units, scale or origin are ambiguous.")
        result.append(
            {
                "name": "ZYX"[index],
                "scale": scale,
                "origin": origin,
                "unit": units[axis["unit"]],
            }
        )
    return {"shape": shape, "axes": result}


def array_grid(value, state):
    carried = grid(state)
    if carried["shape"] != list(value.shape):
        raise ValueError("The carried image shape differs from its actual buffer.")
    return carried


def read_array(root, manifest):
    path = artifact_path(root, manifest["path"])
    if (
        path.suffix != ".npy"
        or path.stat().st_size > MAX_ARRAY_BYTES + 65536
        or sha256(path.read_bytes()) != manifest["sha256"]
    ):
        raise ValueError("A recorded bounded NPY artifact changed.")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if (
        type(value) is not np.memmap
        or value.nbytes > MAX_ARRAY_BYTES
        or str(value.dtype) != manifest["dtype"]
        or list(value.shape) != manifest["shape"]
        or sha256(value.tobytes(order="C")) != manifest["raw_sha256"]
    ):
        raise ValueError("A recorded array's dtype/shape/raw bytes changed.")
    value.flags.writeable = False
    return value


def foreground(value):
    if (
        value.ndim != 3
        or value.size > MAX_MASK_ELEMENTS
        or value.dtype.kind not in "biu"
        or np.any(value < 0)
    ):
        raise ValueError(
            "Mask comparison requires Boolean or nonnegative integer foreground."
        )
    return value > 0


def mask_changes(baseline, candidate, protected):
    if baseline.shape != candidate.shape or protected.shape != baseline.shape:
        raise ValueError("Mask/protected grids differ; no resampling is allowed.")
    added, removed = candidate & ~baseline, baseline & ~candidate
    return {
        "foreground_count": int(candidate.sum()),
        "baseline_foreground_count": int(baseline.sum()),
        "added_foreground": int(added.sum()),
        "removed_foreground": int(removed.sum()),
        "protected_count": int(protected.sum()),
        "protected_retained": int((candidate & protected).sum()),
        "protected_missing": int((~candidate & protected).sum()),
        "added_outside_exact_protected_voxels": int((added & ~protected).sum()),
        "removed_outside_exact_protected_voxels": int((removed & ~protected).sum()),
        "meaning": (
            "Exact foreground changes, not accuracy, wanted signal or improvement."
        ),
    }


def assert_protected_baseline(baseline, protected):
    if baseline.shape != protected.shape or np.any(protected & ~baseline):
        raise ValueError("The exact protected foreground is not in the accepted band.")


def vertical_span_proxy(baseline, candidate, protected, y_scale):
    """Only paired nonempty native (Z,X) columns, with explicit population shifts."""

    def extents(mask):
        present = mask.any(axis=1)
        first = mask.argmax(axis=1)
        last = mask.shape[1] - 1 - mask[:, ::-1, :].argmax(axis=1)
        return present, np.where(present, last - first + 1, 0)

    old_present, old = extents(baseline)
    new_present, new = extents(candidate)
    paired = old_present & new_present
    outside = paired & ~protected.any(axis=1)
    regions = {}
    for name, population in (
        ("all_paired_columns", paired),
        ("paired_columns_without_any_protected_voxel", outside),
    ):
        regions[name] = {
            "column_count": int(population.sum()),
            "baseline_mean_cells": float(old[population].mean())
            if population.any()
            else None,
            "candidate_mean_cells": float(new[population].mean())
            if population.any()
            else None,
            "mean_delta_micrometers": float(
                (new[population] - old[population]).mean() * y_scale
            )
            if population.any()
            else None,
        }
    return {
        "definition": (
            "Y-axis first-to-last foreground cell extent per native (Z,X), "
            "including internal gaps."
        ),
        "not": (
            "Not perpendicular/geometric/anatomical thickness, noise score "
            "or scientific improvement."
        ),
        "paired_population": regions,
        "baseline_nonempty_columns": int(old_present.sum()),
        "candidate_nonempty_columns": int(new_present.sum()),
        "lost_nonempty_columns": int((old_present & ~new_present).sum()),
        "new_nonempty_columns": int((new_present & ~old_present).sum()),
        "y_scale_micrometers_per_cell": y_scale,
    }


def edge_check(vertices, faces):
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or faces.ndim != 2
        or faces.shape[1] != 3
        or faces.dtype.kind not in "iu"
        or not len(vertices)
        or not len(faces)
        or not np.isfinite(vertices).all()
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("Recorded mesh geometry is empty or invalid.")
    if len(faces) > MAX_MESH_FACES:
        return {
            "recomputed": False,
            "reason": "Readout-only 200000-face check bound; use recorded diagnostics.",
        }
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])), axis=1
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "recomputed": True,
        "boundary_edge_count": int((counts == 1).sum()),
        "nonmanifold_edge_count": int((counts > 2).sum()),
        "meaning": (
            "Raw indexed-edge incidence only; "
            "no solid/manifold/anatomical validity claim."
        ),
    }


def load_terminal(root, lock_hash):
    root, lock = verified_lock(root, lock_hash)
    bound = {row["path"] for row in lock["artifacts"]}

    def document(relative, *, recording=False):
        if relative not in bound:
            raise ValueError("Readout requires a lock-bound artifact: " + relative)
        path = artifact_path(root, relative)
        return (
            read_recording_json(path)
            if recording
            else json.loads(path.read_text("utf-8"))
        )

    receipt, session = document("receipt.json"), document("session/terminal.json")
    protocol = document("protocol.json")
    if (
        receipt.get("selected_result_locked") is not True
        or session.get("finished") is not True
        or protocol.get("reference_material_read") is not False
        or lock.get("reference_material_read") is not False
    ):
        raise ValueError(
            "Only genuinely terminal locked own-model recordings may be compared."
        )
    actions = []
    for name in sorted(bound):
        if re.fullmatch(r"actions/action-\d{3}/action.json", name):
            action = document(name)
            actions.append((name.rsplit("/", 1)[0], action))
    finishes = [
        (prefix, action) for prefix, action in actions if action["action"] == "finish"
    ]
    if len(finishes) != 1 or actions[-1] != finishes[0]:
        raise ValueError(
            "A genuine final recorded FINISH is required; no default selection."
        )
    nomination = finishes[0][1]["arguments"]["candidate_id"]
    finish_result = document(finishes[0][0] + "/result.json")
    if (
        finish_result.get("status") != "ok"
        or lock["selection"]["candidate_id"] != nomination
        or receipt["counts"]["actions"] != len(actions)
    ):
        raise ValueError("FINISH, lock and actual action counts disagree.")
    events = [
        json.loads(line)
        for line in (root / "events.jsonl").read_text("utf-8").splitlines()
    ]
    return {
        "root": root,
        "lock": lock,
        "lock_hash": lock_hash,
        "bound": bound,
        "document": document,
        "actions": actions,
        "session": session,
        "receipt": receipt,
        "events": events,
        "nomination": nomination,
        "before": run_inventory(root),
    }


def candidate_records(run):
    root, document = run["root"], run["document"]
    prefix = run["actions"][-1][0] + "/candidates/"
    names = sorted(
        name
        for name in run["bound"]
        if name.startswith(prefix) and name.endswith("/candidate.json")
    )
    result = []
    for name in names:
        folder = name.rsplit("/", 1)[0]
        evidence = document(name)
        roles, mapping = (
            document(folder + "/output-roles.json"),
            document(folder + "/key-to-node.json"),
        )
        recipe = document(folder + "/proposal.json")
        if (
            set(mapping) != {node["key"] for node in recipe["nodes"]}
            or len(set(mapping.values())) != len(mapping)
            or len({row["role"] for row in roles}) != len(roles)
        ):
            raise ValueError("Recorded semantic role/key/node mappings are ambiguous.")
        path = folder + "/trial-output-manifest.json"
        if path not in run["bound"]:
            result.append(
                {
                    "candidate_id": evidence["candidate_id"],
                    "status": evidence["status"],
                    "unavailable": (
                        "No lock-bound completed output manifest; "
                        "not substituted by another candidate."
                    ),
                    "evidence": evidence,
                }
            )
            continue
        ports = document(path, recording=True)
        cache = {(port["node_id"], port["port"]): port for port in ports}
        if len(cache) != len(ports):
            raise ValueError("A captured output manifest duplicates a port.")
        source = cache[("input", 0)]
        pixels = read_array(root, source["array"])
        if (
            pixels.dtype != np.uint8
            or pixels.shape != (12, 96, 128)
            or sha256(pixels.tobytes(order="C")) != RAW_PIXELS_SHA256
        ):
            raise ValueError(
                "A recorded candidate is not bound to the approved source pixels."
            )
        source_grid = array_grid(pixels, source["state"])
        bindings = {row["role"]: row for row in evidence["role_bindings"]}
        if len(bindings) != len(evidence["role_bindings"]):
            raise ValueError("Public candidate role bindings are not unique.")
        outputs = {}
        for role in roles:
            key, port = role["node"], role["port"]
            if key not in mapping or type(port) is not int or port < 0:
                raise ValueError("A role has an invalid typed node key/port.")
            binding = bindings[role["role"]]
            if (
                binding["node_key"] != key
                or type(binding["port"]) is not int
                or binding["port"] != port
            ):
                raise ValueError(
                    "The public role key/port differs from its actual captured role."
                )
            saved = cache[(mapping[key], port)]
            if role["role"] in {"band_mask", "circle_mask"}:
                value = read_array(root, saved["array"])
                if (
                    array_grid(value, saved["state"]) != source_grid
                    or array_hash(value) != bindings[role["role"]]["output_sha256"]
                ):
                    raise ValueError(
                        "An actual mask/grid differs from its public role binding."
                    )
                outputs[role["role"]] = foreground(value)
            elif role["role"] in {"band_mesh", "circle_mesh"}:
                mesh = saved["mesh"]
                vertices, faces = (
                    read_array(root, mesh["vertices"]),
                    read_array(root, mesh["faces"]),
                )
                object_ids = read_array(root, mesh["face_object_ids"])
                objects = tuple(
                    (row["object_id"], row["color"]) for row in mesh["objects"]
                )
                digest = sha256(
                    canonical(
                        [
                            array_hash(vertices),
                            array_hash(faces),
                            array_hash(object_ids),
                            objects,
                        ]
                    )
                )
                if (
                    digest != bindings[role["role"]]["output_sha256"]
                    or grid(mesh["state"]) != source_grid
                ):
                    raise ValueError(
                        "An actual mesh/grid differs from its recorded role binding."
                    )
                diagnostic = next(
                    row["diagnostics"]
                    for row in evidence["outputs"]
                    if row["role"] == role["role"]
                )
                outputs[role["role"]] = {
                    "recorded_diagnostics": diagnostic,
                    "raw_buffer_edge_check": edge_check(vertices, faces),
                }
        result.append(
            {
                "candidate_id": evidence["candidate_id"],
                "status": evidence["status"],
                "evidence": evidence,
                "roles": roles,
                "outputs": outputs,
                "cache": cache,
                "source_grid": source_grid,
                "candidate_record": name,
            }
        )
    return result


def summarize_run(run):
    candidates = candidate_records(run)
    baseline = next(row for row in candidates if row["candidate_id"] == "candidate0")
    config = run["document"]("preservation-configuration.json")
    primary_port = baseline["cache"][tuple(config["primary_ref"])]
    region_port = baseline["cache"][tuple(config["region_ref"])]
    if any(
        grid(port["state"]) != baseline["source_grid"]
        for port in (primary_port, region_port)
    ):
        raise ValueError("The recorded protected input grids differ from the source.")
    primary, region = (
        read_array(run["root"], port["array"])
        for port in (primary_port, region_port)
    )
    if any(
        array_grid(value, port["state"]) != baseline["source_grid"]
        for value, port in ((primary, primary_port), (region, region_port))
    ):
        raise ValueError("Actual protected input shapes differ from the source grid.")
    protected = foreground(primary) & foreground(region)
    contract = config["product_contract"]
    if (
        int(protected.sum()) != 2340
        or contract["protected_count"] != 2340
        or array_hash(protected) != contract["protected_sha256"]
        or contract["protected_sha256"] != EXPECTED_PROTECTED_HASH
    ):
        raise ValueError("The explicit recorded product protected region changed.")
    assert_protected_baseline(baseline["outputs"]["band_mask"], protected)
    metrics = []
    for candidate in candidates:
        if "unavailable" in candidate:
            metrics.append(
                {k: candidate[k] for k in ("candidate_id", "status", "unavailable")}
            )
            continue
        if candidate["source_grid"] != baseline["source_grid"]:
            raise ValueError(
                "Candidate grids differ; no implicit alignment is allowed."
            )
        out, old = candidate["outputs"], baseline["outputs"]
        missing = sorted({"band_mask", "circle_mask"} - out.keys())
        if missing:
            metrics.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "status": candidate["status"],
                    "unavailable": (
                        "Required comparison mask roles not declared; no substitution."
                    ),
                    "missing_roles": missing,
                }
            )
            continue
        metrics.append(
            {
                "candidate_id": candidate["candidate_id"],
                "status": candidate["status"],
                "selected_by_model": candidate["candidate_id"] == run["nomination"],
                "candidate_record": candidate["candidate_record"],
                "band": mask_changes(old["band_mask"], out["band_mask"], protected),
                "circle": mask_changes(
                    old["circle_mask"], out["circle_mask"], np.zeros_like(protected)
                ),
                "vertical_y_span_proxy": vertical_span_proxy(
                    old["band_mask"],
                    out["band_mask"],
                    protected,
                    baseline["source_grid"]["axes"][1]["scale"],
                ),
                "band_mesh": out.get("band_mesh"),
                "circle_mesh": out.get("circle_mesh"),
                "model_preservation_result": candidate["evidence"].get(
                    "preservation_check"
                ),
            }
        )
    counts = Counter(action["action"] for _, action in run["actions"])
    management = {
        name: counts[name]
        for name in (
            "find_nodes",
            "describe_node",
            "describe_nodes",
            "describe_candidate",
            "read_notebook",
            "pin_notebook",
            "pin_previews",
        )
        if counts[name]
    }
    starts = [
        event
        for event in run["events"]
        if event["kind"] == "explicit_investigative_refinement_start"
    ]
    finish_times = [
        event
        for event in run["events"]
        if event["kind"] == "product_action"
        and event["details"]["action"]["action"] == "finish"
    ]
    duration = (
        datetime.fromisoformat(finish_times[0]["utc"])
        - datetime.fromisoformat(starts[0]["utc"])
    ).total_seconds()
    preview_boards = []
    for prefix, action in run["actions"]:
        manifest_path = prefix + "/result-previews/manifest.json"
        if manifest_path in run["bound"] and action["action"] in {
            "inspect_source",
            "inspect_candidate",
            "pin_previews",
        }:
            preview_boards.append(
                {
                    "action": action,
                    "manifest": manifest_path,
                    "board": run["document"](manifest_path),
                    "note": (
                        "Full saved board may contain protected/pinned views; "
                        "do not rebind ownership by file order."
                    ),
                }
            )
    report = {
        "root": str(run["root"]),
        "lock_sha256": run["lock_hash"],
        "nomination": run["nomination"],
        "source_grid": baseline["source_grid"],
        "protected_count": 2340,
        "candidates": metrics,
        "efficiency": {
            "actual_counts": run["receipt"]["counts"],
            "action_types": dict(counts),
            "discovery_and_record_management": management,
            "discovery_and_record_management_total": sum(management.values()),
            "trial_action_indices": [
                int(prefix[-3:])
                for prefix, action in run["actions"]
                if action["action"] in {"trial_edit", "trial_candidate"}
            ],
            "seconds_from_start_to_finish_action": duration,
            "not": (
                "Not a causal performance benchmark; "
                "model choices and workload may differ."
            ),
        },
        "model_experiment_assessments": run["session"]["experiment_ledger"],
        "model_finish": run["actions"][-1][1]["arguments"],
        "saved_preview_boards": preview_boards,
    }
    return report, baseline["outputs"]["band_mask"], baseline["outputs"]["circle_mask"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", nargs=2, action="append", metavar=("ROOT", "LOCK_SHA256")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--terminal-read-approved", action="store_true")
    args = parser.parse_args(argv)
    if not args.terminal_read_approved:
        print(json.dumps(PLAN))
        return 0
    if not args.run or len(args.run) > 2 or args.output is None or args.output.exists():
        parser.error(
            "Use one or two explicitly locked terminal runs "
            "and a fresh output directory."
        )
    audit_root = (EVIDENCE_ROOT / "audit").resolve()
    if not args.output.resolve().is_relative_to(audit_root):
        parser.error(
            "Comparison output must stay in this experiment's separate audit tree."
        )
    for root, digest in args.run:
        if Path(root).resolve().parent != (
            EVIDENCE_ROOT / "live"
        ).resolve() or not re.fullmatch(r"[0-9a-f]{64}", digest):
            parser.error("Only explicitly pinned own live-run roots are in scope.")
    runs = [load_terminal(Path(root), digest) for root, digest in args.run]
    reports, baseline = [], None
    for run in runs:
        report, band, circle = summarize_run(run)
        if baseline is not None and (
            not np.array_equal(baseline[0], band)
            or not np.array_equal(baseline[1], circle)
        ):
            raise ValueError("The accepted baselines differ across runs.")
        baseline = (band, circle)
        reports.append(report)
    for run in runs:
        if run_inventory(run["root"]) != run["before"]:
            raise ValueError("Input evidence changed during the readout.")
    journal = EvidenceJournal(args.output)
    journal.document(
        "readout.json",
        {
            "kind": "own-model-post-terminal-readout",
            "version": 1,
            "unranked_all_candidates": True,
            "reference_material_read": False,
            "scientific_accuracy_or_quality_score": None,
            "no_new_model_pipeline_render_repair_or_selection": True,
            "runs": reports,
        },
    )
    journal.event(
        "post_terminal_readout_completed",
        source_file_sha256=sha256(Path(__file__).read_bytes()),
        runs=[
            {"root": str(run["root"]), "lock_sha256": run["lock_hash"]} for run in runs
        ],
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "runs": len(reports),
                "reference_material_read": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
