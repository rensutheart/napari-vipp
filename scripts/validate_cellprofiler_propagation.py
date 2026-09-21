"""Reproduce full-field Propagation parity on acquired IDR0139 images.

This is implementation evidence, not reproduction of the paper's complete
CellProfiler pipeline or a comparison with biological ground-truth masks.
Source pixels and derived images are written only to the external output root.
Run from an environment with VIPP and its dependencies installed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import threading
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import psutil
import tifffile
from centrosome.propagate import propagate
from PIL import Image, ImageDraw
from skimage.segmentation import find_boundaries

from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
from napari_vipp.core.export import export_pipeline_to_python
from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow

PARAMETERS = {
    "input_channels": ["DNA", "Actin"],
    "intensity_conversion": "Convert Dtype float32, preserve numeric values",
    "intensity_scaling": {"input": [0, 65535], "output": [0, 1]},
    "gaussian_sigma_pixels": 2 / 2.35,
    "seed_threshold": "Li, whole image",
    "minimum_seed_area_pixels": 100,
    "seed_holes": "fill all, face connectivity",
    "seed_connectivity": "full (8 neighbors)",
    "foreground_threshold": "Li on smoothed, explicitly scaled Actin",
    "regularization": 0.05,
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def measured(call):
    """Sample total process RSS; distinguish baseline and incremental peak."""
    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.01):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        output = call()
    finally:
        seconds = time.perf_counter() - started
        samples.append(process.memory_info().rss)
        stop.set()
        thread.join()
    return output, {
        "seconds": seconds,
        "rss_baseline_bytes": baseline,
        "rss_peak_bytes": max(samples),
        "rss_increment_bytes": max(samples) - baseline,
        "rss_sampling_interval_seconds": 0.01,
    }


def make_pipeline(input_path):
    graph = PrototypePipeline()
    graph.reset_empty_graph()
    graph.nodes["input"].params.update(
        source_mode="file path", file_path=str(input_path)
    )
    positions = {"input": (0, 140)}
    ids = {}

    def add(operation, key, position, **params):
        node = graph.add_node(operation)
        for name, value in params.items():
            graph.set_param(node.id, name, value)
        ids[key] = node.id
        positions[node.id] = position
        return node.id

    def connect(source, target, port=0):
        result = graph.connect(source, target, target_port=port)
        if not result.success:
            raise RuntimeError(f"Cannot connect {source} to {target}: {result}")

    for channel, key, y in ((0, "dna", 0), (1, "actin", 430)):
        channel_id = add("extract_channel", key, (350, y), channel=channel)
        float_id = add(
            "convert_dtype",
            f"{key}_float",
            (700, y),
            output_dtype="float32",
            scaling="preserve",
        )
        scaled_id = add(
            "rescale_intensity",
            f"{key}_scaled",
            (1050, y),
            cutoff_mode="Values",
            in_low_value=0.0,
            in_high_value=65535.0,
            out_min=0.0,
            out_max=1.0,
        )
        smooth_id = add(
            "gaussian_blur",
            f"{key}_smoothed",
            (1400, y),
            sigma=PARAMETERS["gaussian_sigma_pixels"],
        )
        threshold_id = add("li_threshold", f"{key}_threshold", (1750, y))
        for source, target in (
            ("input", channel_id),
            (channel_id, float_id),
            (float_id, scaled_id),
            (scaled_id, smooth_id),
            (smooth_id, threshold_id),
        ):
            connect(source, target)
    cleaned = add(
        "remove_small_objects",
        "seed_cleaned",
        (2100, 0),
        min_size=100,
        spatial_mode="2D YX",
        connectivity="Full connectivity",
    )
    filled = add(
        "fill_holes",
        "seed_filled",
        (2450, 0),
        max_hole_size=0,
        spatial_mode="2D YX",
        connectivity="Face connected",
    )
    seeds = add(
        "label_connected_components",
        "seeds",
        (2800, 0),
        spatial_mode="2D YX",
        connectivity="Full connectivity",
    )
    grown = add(
        "cellprofiler_propagation",
        "propagation",
        (3150, 230),
        regularization=PARAMETERS["regularization"],
    )
    for source, target, port in (
        (ids["dna_threshold"], cleaned, 0),
        (cleaned, filled, 0),
        (filled, seeds, 0),
        (ids["actin_smoothed"], grown, 0),
        (seeds, grown, 1),
        (ids["actin_threshold"], grown, 2),
    ):
        connect(source, target, port)
    document = serialize_workflow(
        graph, positions=positions, compute_request=ComputeRequest(mode="cpu")
    )
    return graph, ids, document


def display_gray(data):
    """Display-only percentile stretch, never passed into scientific execution."""
    lower, upper = np.percentile(data, [1, 99.5])
    gray = np.clip((data.astype(float) - lower) / max(upper - lower, 1), 0, 1)
    return np.repeat(np.rint(gray[..., None] * 255).astype(np.uint8), 3, axis=2)


def save_overlay(path, dna, actin, seeds, mask, labels, label):
    base = display_gray(actin)
    nuclei = display_gray(dna)
    nuclei[find_boundaries(seeds, mode="outer")] = [70, 245, 235]
    masked = base.copy()
    masked[~mask] = masked[~mask] // 4
    grown = base.copy()
    grown[find_boundaries(labels, mode="inner")] = [255, 190, 20]
    grown[find_boundaries(seeds, mode="outer")] = [70, 245, 235]
    panels = [nuclei, masked, grown]
    titles = ["DNA / seed outlines", "Actin / foreground mask", "Actin / grown regions"]
    width, height = 600, 600
    canvas = Image.new("RGB", (width * 3, height + 92), "#17202a")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (16, 12), f"{label} | full 996 x 996 field | fixed settings", fill="white"
    )
    draw.text(
        (16, 31),
        "Cyan: seed boundary. Gold: propagated boundary. Display stretch only.",
        fill="white",
    )
    for index, (panel, title) in enumerate(zip(panels, titles, strict=True)):
        image = Image.fromarray(panel).resize((width, height), Image.Resampling.LANCZOS)
        canvas.paste(image, (index * width, 92))
        draw.text((index * width + 16, 68), title, fill="white")
    canvas.save(path)


def run_field(record, output_root):
    name = f"{record['well']}_F{record['field']}"
    destination = output_root / name
    destination.mkdir(parents=True, exist_ok=True)
    source_records = {}
    channels = []
    for channel in PARAMETERS["input_channels"]:
        source = record["channels"][channel]
        source_hash = sha256(source["path"])
        if source_hash != source["sha256"]:
            raise ValueError(f"Source checksum mismatch: {source['path']}")
        array = tifffile.imread(source["path"])
        if array.shape != (996, 996) or array.dtype != np.uint16:
            raise ValueError("Expected the verified full-field uint16 YX source.")
        source_records[channel] = dict(source, verified_sha256=source_hash)
        channels.append(array)
    data = np.stack(channels, axis=0)
    data.setflags(write=False)
    array_hash = hashlib.sha256(data.tobytes()).hexdigest()
    input_path = destination / "DNA_Actin_CYX.ome.tif"
    tifffile.imwrite(
        input_path,
        data,
        ome=True,
        photometric="minisblack",
        metadata={"axes": "CYX", "Channel": {"Name": ["DNA", "Actin"]}},
    )
    snapshot = load_frozen_file_source_snapshot(input_path)
    np.testing.assert_array_equal(snapshot.payload.data, data)
    data = snapshot.payload.data
    graph, ids, workflow = make_pipeline(input_path)
    source_item = snapshot.payload.source_item.to_dict()
    graph.nodes["input"].params["_vipp_source_item"] = source_item
    workflow["nodes"][0]["params"]["_vipp_source_item"] = source_item
    write_json(destination / "workflow.json", workflow)
    exported_path = destination / "workflow.py"
    exported_path.write_text(export_pipeline_to_python(graph), encoding="utf-8")
    result, performance = measured(
        lambda: execute_pipeline_request(
            PipelineRunRequest(
                run_id=1,
                workflow=workflow,
                input_data=data,
                input_metadata=snapshot.payload.metadata,
                input_name=name,
                source_payloads={"input": snapshot.payload},
                compute_request=ComputeRequest(mode="cpu"),
                manual_node_ids=frozenset(graph.manual_node_ids()),
            ),
            raise_errors=True,
        )
    )
    if result.error or result.cancelled:
        raise RuntimeError(result.error or "Unexpected cancellation.")
    outputs = result.pipeline.outputs
    guidance = outputs[ids["actin_smoothed"]]
    seeds = outputs[ids["seeds"]]
    mask = outputs[ids["actin_threshold"]]
    grown = outputs[ids["propagation"]]
    if not np.any(seeds) or not np.any(mask) or np.unique(grown).size < 3:
        raise AssertionError(
            "The acquired example must contain multiple grown regions."
        )
    reference_result, reference_performance = measured(
        lambda: propagate(guidance, seeds, mask, PARAMETERS["regularization"])
    )
    reference, _distances = reference_result
    np.testing.assert_array_equal(grown, reference)
    if hashlib.sha256(data.tobytes()).hexdigest() != array_hash:
        raise AssertionError("Source input was mutated during graph execution.")
    namespace = {"__name__": "propagation_example_export"}
    exec(
        compile(exported_path.read_text(encoding="utf-8"), str(exported_path), "exec"),
        namespace,
    )
    exported_outputs, exported_performance = measured(
        lambda: namespace["run_pipeline"](
            namespace["load_image"](input_path, source_node_id="input"),
            input_name=name,
        )
    )
    np.testing.assert_array_equal(exported_outputs[ids["propagation"]], reference)
    artifacts = {}
    for key, array in (
        ("guidance", guidance),
        ("seeds", seeds),
        ("foreground", mask),
        ("propagation", grown),
        ("centrosome_reference", reference),
    ):
        path = destination / f"{key}.npy"
        np.save(path, array)
        artifacts[key] = {
            "file": path.name,
            "sha256": sha256(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
    save_overlay(destination / "overlay.png", *channels, seeds, mask, grown, name)
    metrics = {
        "field": name,
        "condition": record["condition"],
        "author_image_number": record["author_image_number"],
        "sources": source_records,
        "parameters": PARAMETERS,
        "shape_yx": list(grown.shape),
        "positive_seed_ids": int(np.unique(seeds).size - 1),
        "positive_output_ids": int(np.unique(grown).size - 1),
        "foreground_pixels": int(np.count_nonzero(mask)),
        "seed_pixels_outside_foreground": int(np.count_nonzero((seeds > 0) & ~mask)),
        "unreached_foreground_pixels": int(np.count_nonzero(mask & (grown == 0))),
        "centrosome_mismatched_pixels": int(np.count_nonzero(grown != reference)),
        "generated_python_mismatched_pixels": int(
            np.count_nonzero(exported_outputs[ids["propagation"]] != reference)
        ),
        "immutable_input_passed": True,
        "canonical_file_source_exact_pixels": True,
        "workflow_sha256": sha256(destination / "workflow.json"),
        "exported_python_sha256": sha256(exported_path),
        "derived_input_sha256": sha256(input_path),
        "shared_graph_performance": performance,
        "direct_centrosome_performance": reference_performance,
        "generated_python_performance": exported_performance,
        "output_history": list(
            result.pipeline.output_states[ids["propagation"]].history
        ),
        "artifacts": artifacts,
    }
    write_json(destination / "evidence.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("D:/VIPP-paper-reproductions/statistics/idr0139"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("D:/VIPP-paper-reproductions/validation/cellprofiler-propagation"),
    )
    args = parser.parse_args()
    manifest = json.loads((args.dataset_root / "verified-image-sets.json").read_text())
    args.output_root.mkdir(parents=True, exist_ok=True)
    selected = [
        next(
            record
            for record in manifest["image_sets"]
            if record["well"] == well and record["field"] == "001"
        )
        for well in ("J05", "O02", "E22", "L08")
    ]
    source_directory = (
        "https://ftp.ebi.ac.uk/pub/databases/IDR/idr0139-lawson-fascin/"
        "20220707-box/1093711385/"
    )
    for record in selected:
        for source in record["channels"].values():
            filename = Path(source["path"].replace("\\", "/")).name
            source["path"] = str((args.dataset_root / "raw" / filename).resolve())
            source["source_url"] = source_directory + filename
    write_json(args.output_root / "summary.json", {"status": "running"})
    results = []
    try:
        for record in selected:
            print(f"Validating {record['well']} field {record['field']}", flush=True)
            results.append(run_field(record, args.output_root))
    except Exception as error:
        write_json(
            args.output_root / "summary.json",
            {
                "status": "failed",
                "error": str(error),
                "completed_fields": results,
            },
        )
        raise
    report = {
        "schema_version": 1,
        "date_utc": datetime.now(UTC).isoformat(),
        "status": "passed",
        "scope": "Exact Propagation parity on identical graph-generated inputs",
        "not_claimed": [
            "Full CellProfiler pipeline equivalence",
            "Biological segmentation accuracy",
            "Paper result reproduction",
            "Timing or memory guarantees",
        ],
        "calibration": "Physical calibration unknown; spatial parameters are pixels.",
        "parameter_policy": (
            "One frozen parameter set used unchanged on four first fields; no tuning."
        ),
        "source_attribution": "IDR0139, Lawson et al., CC BY 4.0",
        "source_study": "https://idr.openmicroscopy.org/study/idr0139/",
        "script_sha256": sha256(Path(__file__)),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in (
                "napari-vipp",
                "centrosome",
                "numpy",
                "scipy",
                "scikit-image",
                "tifffile",
                "psutil",
                "Pillow",
            )
        },
        "rss_caveat": (
            "Total process RSS sampled every 10 ms; increments are per call. "
            "Sequential runs share caches; this is not isolated peak allocation."
            " The sampler can miss short peaks or compiled calls holding the GIL."
        ),
        "fields": results,
    }
    write_json(args.output_root / "summary.json", report)
    print(
        f"PASS: {len(results)} full fields; all graph/export pixels match Centrosome.",
        flush=True,
    )
    print(args.output_root / "summary.json", flush=True)


if __name__ == "__main__":
    main()
