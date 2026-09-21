"""Run the archived Marcotti et al. compartment workflow through VIPP.

The image graph reproduces the recorded CellProfiler 4.2.6 algorithm profile.
The authors' separate Python-analysis step is represented by a keyed compartment
join and their nuclear/(nuclear+cytoplasmic) mean-intensity formula. Downloaded
sources are never modified; review inputs preserve every source pixel exactly.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def digest(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def document_digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def execution_contract():
    """Identify the actual driver, source tree and scientific runtime."""
    source = REPO / "src/napari_vipp"
    return {
        "script_sha256": digest(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "versions": {
            package: importlib.metadata.version(package)
            for package in (
                "numpy",
                "scipy",
                "scikit-image",
                "centrosome",
                "pandas",
                "tifffile",
                "napari",
                "napari-vipp",
            )
        },
        "source_sha256": {
            path.relative_to(REPO).as_posix(): digest(path)
            for path in sorted(source.rglob("*.py"))
            if "_tests" not in path.parts
        },
    }


def reference_directory(case, reference_root):
    root = Path(reference_root)
    if case["Case"] == "idr0139":
        name = f"{case['Well']}_F{int(case['field']):03d}"
        candidates = [root / "arrays40" / name, root / "arrays" / name]
    else:
        well = case["FileName"].split("-")[0]
        base = root / "yap" / f"LM2_GEFGAP_ONTARGETPlus_{case['Plate']}"
        name = f"{well}_F{int(case['field'])}"
        candidates = [base / "all-arrays" / name, base / "arrays" / name]
    return next((path for path in candidates if path.is_dir()), None)


def reusable_record(case, out, contract, reference_root):
    """Reuse only fully identified, unchanged outputs from the current recipe."""
    receipt = out / "result.json"
    if not receipt.exists():
        return None
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        recipe, _ = build_workflow(case["protein"], case["Case"] == "idr0139")
        ref = reference_directory(case, reference_root)
        ref_path = str(ref) if ref is not None else None
        if (
            record.get("execution_contract") != contract
            or record.get("case_fingerprint") != document_digest(case)
            or record.get("recipe_sha256") != document_digest(recipe)
            or record.get("reference_directory") != ref_path
            or not record.get("artifact_sha256")
            or [source["path"] for source in record["sources"]] != case["paths"]
        ):
            return None
        for source in record["sources"]:
            if digest(source["path"]) != source["sha256"]:
                return None
        for name, checksum in record["artifact_sha256"].items():
            if Path(name).name != name or digest(out / name) != checksum:
                return None
        for name, checksum in record.get("reference_sha256", {}).items():
            if ref is None or Path(name).name != name or digest(ref / name) != checksum:
                return None
        return record
    except (OSError, ValueError, KeyError, TypeError):
        return None


def build_workflow(protein="Fascin", secondary_protein=True):
    """Build an ordinary saveable VIPP graph, with explicit port connections."""
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import serialize_workflow

    graph = PrototypePipeline()
    graph.reset_empty_graph()
    nodes = {"input": "input"}
    positions = {"input": [0, 0]}

    def add(role, operation, upstream=None, position=(0, 0), **params):
        node = graph.add_node(operation)
        nodes[role] = node.id
        positions[node.id] = list(position)
        for name, value in params.items():
            graph.set_param(node.id, name, value)
        if upstream:
            connect(upstream, role)
        return node.id

    def connect(source, target, target_port=0, source_port=0):
        result = graph.connect(
            nodes[source],
            nodes[target],
            target_port=target_port,
            source_port=source_port,
        )
        assert result.success, result.message

    add(
        "float",
        "convert_dtype",
        "input",
        (390, 0),
        output_dtype="float32",
        scaling="preserve",
    )
    add(
        "normalized",
        "rescale_intensity",
        "float",
        (780, 0),
        cutoff_mode="Values",
        in_low_value=0.0,
        in_high_value=65535.0,
        out_min=0.0,
        out_max=1.0,
        invert_intensity=False,
    )
    channels = ["DNA", "Actin", protein]
    if secondary_protein:
        channels.append("NuclearActin")
    for index, channel in enumerate(channels):
        add(
            channel, "extract_channel", "normalized", (1170, index * 450), channel=index
        )
    add("SmoothedDNA", "cellprofiler_smooth", "DNA", (1560, 0), artifact_diameter=2.0)
    add(
        "SmoothedActin",
        "cellprofiler_smooth",
        "Actin",
        (1560, 450),
        artifact_diameter=2.0,
    )
    add(
        "Nuclei",
        "cellprofiler_primary_objects",
        "SmoothedDNA",
        (1950, 0),
        min_diameter=15,
        max_diameter=50,
        threshold_smoothing=1.3488,
    )
    add(
        "ActinMask",
        "cellprofiler_threshold",
        "SmoothedActin",
        (1950, 850),
        smoothing_scale=0.0,
    )
    add("Seeds", "cellprofiler_propagation_seeds", position=(2340, 0))
    connect("Nuclei", "Seeds", source_port=1)
    connect("Nuclei", "Seeds", target_port=1)
    add("Grown", "cellprofiler_propagation", position=(2730, 450), regularization=0.05)
    for port, role in enumerate(("SmoothedActin", "Seeds", "ActinMask")):
        connect(role, "Grown", target_port=port)
    add("Cells", "cellprofiler_finish_cells", position=(3120, 450), fill_holes=True)
    connect("Grown", "Cells")
    connect("Nuclei", "Cells", target_port=1)
    add("Cytoplasm", "cellprofiler_cytoplasm", position=(3510, 850), shrink_nuclei=True)
    connect("Cells", "Cytoplasm")
    connect("Nuclei", "Cytoplasm", target_port=1)
    for index, compartment in enumerate(("Nuclei", "Cells", "Cytoplasm")):
        for pindex, signal in enumerate(channels[2:]):
            role = f"{compartment}_{signal}"
            add(
                role,
                "measure_objects_intensity",
                position=(3900 + pindex * 390, index * 550),
                spatial_mode="2D YX",
            )
            connect(compartment, role)
            connect(signal, role, target_port=1)
    notes = [
        {
            "id": "paper-profile",
            "position": [0, -620],
            "width": 1120,
            "text": "Marcotti et al. 2026 — recorded CellProfiler 4.2.6 "
            "compartment profile. Original uint16 intensities divided by 65535 "
            "as float32; Gaussian diameter 2; Li threshold; Shape/Shape nuclei "
            "15–50 px, border and size exclusion; unedited border seeds retained "
            "during Propagation (regularization 0.05); cell holes filled; "
            "shrunken nuclear subtraction. All parameters are in pixels. "
            "Postprocessing matches the authors' Python workflow: join label_id "
            "within field, then nuclear mean/(nuclear mean+cytoplasmic mean). "
            "Published-table agreement must be evaluated separately from "
            "agreement with CellProfiler on these TIFFs.",
        }
    ]
    return serialize_workflow(graph, positions=positions, notes=notes), nodes


def cases_from_reference(root):
    import pandas as pd

    cells = pd.read_csv(root / "validation/statistics-paper/reference/cells.csv")
    images = cells.drop_duplicates(["Case", "Plate", "ImageNumber"])
    tables = root / "statistics/reference/data_subsets/cell_profiler_outputs"
    result = []
    for (case, plate), subset in images.groupby(["Case", "Plate"], sort=False):
        disk_plate = f"LM2_GEFGAP_ONTARGETPlus_{plate}" if case == "idr0028" else plate
        directory = (
            tables / "idr0139"
            if case == "idr0139"
            else tables / "idr0028/screenB" / disk_plate
        )
        lookup = pd.read_csv(directory / "Image.csv").set_index("ImageNumber")
        for row in subset.to_dict("records"):
            image = lookup.loc[row["ImageNumber"]]
            record = {
                key: row[key]
                for key in (
                    "Case",
                    "Plate",
                    "Well",
                    "Replicate",
                    "Treatment",
                    "ImageNumber",
                    "FileName",
                )
            }
            record["AuthorCount"] = int(image["Count_Nuclei"])
            record["author_md5"] = {}
            if case == "idr0139":
                paths = []
                for channel in ("DNA", "Actin", "Fascin", "NuclearActin"):
                    filename = image[f"FileName_{channel}"]
                    path = root / "statistics/idr0139/raw" / filename
                    if not path.exists():
                        path = root / "statistics/idr0139-superplots/raw" / filename
                    paths.append(str(path))
                    record["author_md5"][channel] = str(image[f"MD5Digest_{channel}"])
                record.update(
                    paths=paths,
                    channels=["DNA", "Actin", "Fascin", "NuclearActin"],
                    protein="Fascin",
                    field=row["FileName"].split("F")[1][:3],
                )
            else:
                path = root / "statistics/idr0028/raw" / disk_plate / row["FileName"]
                record.update(
                    paths=[str(path)],
                    channels=["DNA", "Actin", "YAPTAZ"],
                    protein="YAPTAZ",
                    field=str(int(image["Metadata_Field"])),
                )
                record["author_md5"]["multichannel"] = str(image["MD5Digest_Hoechst"])
            record["key"] = (
                f"{case}/{plate}/{record['Well']}_F{int(record['field']):03d}"
            )
            result.append(record)
    return result


def run_case(case, output, reference_root, contract):
    import numpy as np
    import pandas as pd
    import tifffile

    from napari_vipp.core.compute import ComputeMode, ComputeRequest
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.export import export_pipeline_to_python
    from napari_vipp.core.file_sources import load_frozen_file_source_snapshot

    started = time.perf_counter()
    assert execution_contract() == contract, "Code/runtime changed before execution"
    out = Path(output) / case["key"]
    out.mkdir(parents=True, exist_ok=True)
    sources = [
        {"path": path, "sha256": digest(path), "md5": digest(path, "md5")}
        for path in case["paths"]
    ]
    arrays = [tifffile.imread(path) for path in case["paths"]]
    if case["Case"] == "idr0139":
        image = np.stack(arrays)
        assert image.shape == (4, 996, 996)
    else:
        assert arrays[0].shape == (4, 500, 667)
        # Archived CP Frame_* identifies pages0,2,3 as DNA,Actin,YAPTAZ.
        image = arrays[0][[0, 2, 3]]
    assert image.dtype == np.uint16
    source_keys = case["channels"] if len(sources) > 1 else ["multichannel"]
    for source, key in zip(sources, source_keys, strict=True):
        source["author_md5"] = case["author_md5"][key]
        source["author_file_bytes_identical"] = source["md5"] == source["author_md5"]
    input_path = out / "input-channels.ome.tif"
    metadata = {"axes": "CYX", "Channel": {"Name": case["channels"]}}
    if case["Case"] == "idr0028":
        metadata.update(
            PhysicalSizeX=0.6459,
            PhysicalSizeY=0.6459,
            PhysicalSizeXUnit="µm",
            PhysicalSizeYUnit="µm",
        )
    tifffile.imwrite(
        input_path,
        image,
        metadata=metadata,
        compression="zlib",
        photometric="minisblack",
    )
    snapshot = load_frozen_file_source_snapshot(input_path)
    np.testing.assert_array_equal(snapshot.payload.data, image)
    document, nodes = build_workflow(case["protein"], case["Case"] == "idr0139")
    recipe_sha256 = document_digest(document)
    params = document["nodes"][0]["params"]
    params.update(
        source_mode="file path",
        file_path=str(input_path),
        _vipp_source_item=snapshot.payload.source_item.to_dict(),
    )
    write_json(out / "workflow.json", document)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=document,
            input_data=snapshot.payload.data,
            input_metadata=snapshot.payload.metadata,
            input_name=case["key"],
            source_payloads={"input": snapshot.payload},
            compute_request=ComputeRequest(mode=ComputeMode.CPU),
            manual_node_ids=frozenset(nodes.values()),
        )
    )
    if result.error:
        raise RuntimeError(f"{case['key']}: {result.error}")
    graph = result.pipeline
    assert graph is not None
    values = {role: graph.outputs[node_id] for role, node_id in nodes.items()}
    labels = {name: values[name] for name in ("Nuclei", "Cells", "Cytoplasm")}
    intermediates = {
        name: values[name]
        for name in (
            "DNA",
            "Actin",
            "SmoothedDNA",
            "SmoothedActin",
            "ActinMask",
            "Seeds",
            "Grown",
            "Nuclei",
            "Cells",
            "Cytoplasm",
            case["protein"],
        )
    }
    intermediates["UneditedNuclei"] = graph.node_outputs[nodes["Nuclei"]][1]
    np.savez_compressed(out / "intermediates.npz", **intermediates)
    frame = None
    for compartment in labels:
        for signal in case["channels"][2:]:
            table = pd.DataFrame(values[f"{compartment}_{signal}"].records())
            table.to_csv(
                out / f"{compartment}-{signal}.csv", index=False, float_format="%.17g"
            )
            if signal != case["protein"]:
                continue
            selected = table[["label_id", "intensity_mean"]].rename(
                columns={
                    "label_id": "ObjectNumber",
                    "intensity_mean": f"{compartment}Mean",
                }
            )
            frame = (
                selected
                if frame is None
                else frame.merge(
                    selected, on="ObjectNumber", how="outer", validate="one_to_one"
                )
            )
    assert frame is not None and frame.notna().all().all()
    frame["Ratio"] = frame["NucleiMean"] / (
        frame["NucleiMean"] + frame["CytoplasmMean"]
    )
    assert np.isfinite(frame["Ratio"]).all()
    for key in (
        "Case",
        "Plate",
        "Well",
        "Replicate",
        "Treatment",
        "ImageNumber",
        "FileName",
    ):
        frame[key] = case[key]
    frame.to_csv(out / "cells.csv", index=False, float_format="%.17g")
    comparison = {}
    reference_sha256 = {}
    ref = reference_directory(case, reference_root)
    if ref is not None:
        mapping = {
            "DNA": "DNA",
            "Actin": "Actin",
            "SmoothedDNA": "SmoothedDNA",
            "SmoothedActin": "SmoothedActin",
            "ActinMask": "Cells_threshold_mask_replayed",
            "Nuclei": "Nuclei_segmented",
            "UneditedNuclei": "Nuclei_unedited_segmented",
            "Cells": "Cells_segmented",
            "Cytoplasm": "Cytoplasm_segmented",
            case["protein"]: case["protein"],
        }
        if case["Case"] == "idr0028":
            mapping["DNA"] = "Hoechst"
        for name, reference in mapping.items():
            reference_path = ref / f"{reference}.npy"
            checksum = digest(reference_path)
            expected = np.load(reference_path, allow_pickle=False)
            comparison[name] = {
                "exact": bool(np.array_equal(intermediates[name], expected)),
                "different_pixels": int(
                    np.count_nonzero(intermediates[name] != expected)
                ),
            }
            assert digest(reference_path) == checksum, "Reference changed while read"
            reference_sha256[reference_path.name] = checksum
        # Different NumPy builds may move a float32 Li threshold by a few ULPs.
        # Retain that intermediate evidence and independently require the final
        # compartment label maps to agree pixel for pixel.
        assert all(comparison[name]["exact"] for name in labels), comparison
    (out / "workflow.py").write_text(export_pipeline_to_python(graph), encoding="utf-8")
    record = {
        **case,
        "sources": sources,
        "execution_contract": contract,
        "case_fingerprint": document_digest(case),
        "recipe_sha256": recipe_sha256,
        "reference_directory": str(ref) if ref is not None else None,
        "reference_sha256": reference_sha256,
        "executed_utc": datetime.now(UTC).isoformat(),
        "vipp_count": len(frame),
        "count_difference_from_author": len(frame) - case["AuthorCount"],
        "ratio_mean": float(frame["Ratio"].mean()),
        "elapsed_seconds": time.perf_counter() - started,
        "independent_cellprofiler_comparison": comparison,
        "workflow_sha256": digest(out / "workflow.json"),
        "review_input_sha256": digest(input_path),
        "source_pixels_preserved": True,
    }
    assert all(digest(source["path"]) == source["sha256"] for source in sources)
    record["original_sources_reverified"] = True
    record["artifact_sha256"] = {
        path.name: digest(path)
        for path in sorted(out.iterdir())
        if path.is_file() and path.name != "result.json"
    }
    assert execution_contract() == contract, "Code/runtime changed during execution"
    write_json(out / "result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("D:/VIPP-paper-reproductions")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--case", choices=("idr0139", "idr0028"))
    parser.add_argument("--well", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output or args.root / "validation/statistics-paper/vipp"
    output.mkdir(parents=True, exist_ok=True)
    cases = cases_from_reference(args.root)
    if args.case:
        cases = [case for case in cases if case["Case"] == args.case]
    if args.well:
        cases = [case for case in cases if case["Well"] in args.well]
    if args.limit:
        cases = cases[: args.limit]
    contract = execution_contract()
    reference = args.root / "validation/statistics-paper/cellprofiler-reference"
    pending, records = [], []
    for case in cases:
        record = (
            reusable_record(case, output / case["key"], contract, reference)
            if args.resume
            else None
        )
        if record is not None:
            records.append(record)
        else:
            pending.append(case)
    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(run_case, case, output, reference, contract): case
            for case in pending
        }
        for future in as_completed(jobs):
            try:
                record = future.result()
            except Exception as error:
                failures.append({"key": jobs[future]["key"], "error": str(error)})
                write_json(output / "failures.json", failures)
                print(f"FAILED {jobs[future]['key']}: {error}", flush=True)
                continue
            records.append(record)
            print(
                f"{len(records)}/{len(cases)} {record['key']}: "
                f"VIPP {record['vipp_count']}, author {record['AuthorCount']}",
                flush=True,
            )
            write_json(
                output / "progress.json", sorted(records, key=lambda item: item["key"])
            )
    import pandas as pd

    if failures:
        raise RuntimeError(
            f"{len(failures)} fields failed; inspect failures.json before resuming."
        )
    order = {case["key"]: index for index, case in enumerate(cases)}
    records.sort(key=lambda record: order[record["key"]])
    for case in cases:
        assert reusable_record(case, output / case["key"], contract, reference), (
            f"Inputs, code, recipe or outputs changed before aggregation: {case['key']}"
        )
    combined = pd.concat(
        [pd.read_csv(output / record["key"] / "cells.csv") for record in records],
        ignore_index=True,
    )
    combined.to_csv(output / "cells.csv", index=False, float_format="%.17g")
    write_json(
        output / "results.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "versions": {
                package: importlib.metadata.version(package)
                for package in (
                    "numpy",
                    "scipy",
                    "scikit-image",
                    "centrosome",
                    "pandas",
                    "napari-vipp",
                )
            },
            "source_checkout": str(REPO),
            "script_sha256": digest(__file__),
            "scientific_source_hashes": {
                name: digest(REPO / "src/napari_vipp/core" / name)
                for name in (
                    "cellprofiler_compartments.py",
                    "cellprofiler_propagation.py",
                    "operations.py",
                    "measurements.py",
                    "pipeline.py",
                )
            },
            "field_count": len(records),
            "cell_count": len(combined),
            "cases": sorted(records, key=lambda item: item["key"]),
        },
    )
    print(
        f"Saved {len(records)} fields and {len(combined)} cells to {output}", flush=True
    )


if __name__ == "__main__":
    main()
