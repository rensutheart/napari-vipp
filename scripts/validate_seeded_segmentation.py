"""Reproduce bounded 3D watershed and Random Walker evidence on local images.

No pixels are downloaded or redistributed by this script. Outputs include the
executed workflow, original-input hashes, parameters, scientific checks, timing,
sampled process RSS, and inspectable overlays. Random Walker runs in a child
process with a wall-time and process-RSS ceiling; it is not a VIPP node.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import threading
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def array_digest(array):
    import numpy as np

    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path, document):
    Path(path).write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def measured(call):
    """Sample this process's working-set RSS, including native allocations."""
    import psutil

    process = psutil.Process()
    baseline = process.memory_info().rss
    samples = [baseline]
    stop = threading.Event()

    def sample():
        while not stop.wait(0.01):
            samples.append(process.memory_info().rss)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    start = time.perf_counter()
    try:
        value = call()
    finally:
        elapsed = time.perf_counter() - start
        samples.append(process.memory_info().rss)
        stop.set()
        thread.join()
    return value, {
        "elapsed_seconds": elapsed,
        "baseline_process_rss_bytes": baseline,
        "peak_sampled_process_rss_bytes": max(samples),
        "peak_increment_over_baseline_bytes": max(samples) - baseline,
        "rss_sample_interval_seconds": 0.01,
    }


def build_workflow(*, phantom=False, sample=False):
    from napari_vipp.core.pipeline import PrototypePipeline
    from napari_vipp.core.workflow import serialize_workflow

    graph = PrototypePipeline()
    graph.reset_empty_graph()
    if sample:
        graph.set_param("input", "source_mode", "sample")
        graph.set_param("input", "sample_name", "VIPP synthetic volume")
    nodes = {"input": "input"}

    def add(role, operation, upstream=None, **params):
        node = graph.add_node(operation)
        nodes[role] = node.id
        for name, value in params.items():
            graph.set_param(node.id, name, value)
        if upstream:
            result = graph.connect(nodes[upstream], node.id)
            assert result.success, result.message
        return node.id

    if phantom:
        add("mask", "binary_threshold", "input", threshold=0.5)
    else:
        add(
            "float",
            "convert_dtype",
            "input",
            output_dtype="float32",
            scaling="preserve",
        )
        add(
            "smooth",
            "gaussian_blur_3d",
            "float",
            sigma_z=0.5,
            sigma_y=1.0,
            sigma_x=1.0,
            lock_xy=True,
        )
        add(
            "threshold",
            "otsu_threshold",
            "smooth",
            threshold_scope="Stack histogram",
            histogram_bins=256,
        )
        add(
            "mask",
            "remove_small_objects",
            "threshold",
            min_size=16,
            connectivity="Face connected",
            spatial_mode="3D ZYX",
        )
    add("distance", "euclidean_distance_transform", "mask", spatial_mode="3D ZYX")
    add(
        "markers",
        "h_maxima_markers",
        "distance",
        h=1.0,
        connectivity="Full connectivity",
        spatial_mode="3D ZYX",
    )
    watershed = add(
        "labels",
        "marker_controlled_watershed",
        image_mode="Distance map (invert)",
        compactness=0.0,
        watershed_line=False,
        spatial_mode="3D ZYX",
    )
    for port, role in enumerate(("distance", "markers", "mask")):
        result = graph.connect(nodes[role], watershed, target_port=port)
        assert result.success, result.message
    positions = {
        node_id: (index * 390.0, 100.0 if role != "markers" else 460.0)
        for index, (role, node_id) in enumerate(nodes.items())
    }
    note = {
        "id": "scientific-contract",
        "position": [0.0, -500.0],
        "width": 720.0,
        "text": "3D seeded watershed demonstration (CPU). Convert to float32 preserves "
        "native intensity values; Gaussian sigma is Z/Y/X = 0.5/1/1 voxels. "
        "One stack-wide Otsu histogram; remove components <16 voxels "
        "(face connectivity); "
        "EDT and H-Maxima h=1 use voxel-index distances, not physical micrometers. "
        "Watershed uses 3D face connectivity and no watershed line. "
        "Physical calibration is carried unchanged but does not weight EDT "
        "or watershed. This demonstrates segmentation, not the complete "
        "mitochondria paper or Fiji parity.",
    }
    if phantom:
        note["text"] = (
            "Analytic union of two equal-radius spheres. Threshold at 0.5, "
            "3D voxel-index EDT, H-Maxima h=1/full connectivity, then 3D "
            "marker-controlled watershed with face connectivity. The known "
            "division is the halfway X plane. Physical calibration is carried "
            "but does not weight distances. No image preprocessing is used."
        )
    return serialize_workflow(graph, positions=positions, notes=[note]), nodes


def run_vipp(image, spacing, document):
    from napari_vipp.core.compute import ComputeMode, ComputeRequest
    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.metadata import AxisMetadata, image_state_from_array

    axes = tuple(
        AxisMetadata(name, "space", "micrometer", scale, source_axis=index)
        for index, (name, scale) in enumerate(zip("zyx", spacing, strict=True))
    )
    state = image_state_from_array(image, axes=axes)
    result = execute_pipeline_request(
        PipelineRunRequest(
            run_id=1,
            workflow=document,
            input_data=image,
            input_metadata={"vipp_image_state": state.to_dict()},
            input_name="validation input",
            source_payloads={},
            compute_request=ComputeRequest(mode=ComputeMode.CPU),
        )
    )
    if result.error:
        raise RuntimeError(result.error)
    assert result.pipeline is not None
    return result.pipeline, axes


def phantom():
    import numpy as np

    z, y, x = np.indices((35, 49, 64))
    sphere1 = (z - 17) ** 2 + (y - 24) ** 2 + (x - 22) ** 2 <= 15**2
    sphere2 = (z - 17) ** 2 + (y - 24) ** 2 + (x - 41) ** 2 <= 15**2
    mask = sphere1 | sphere2
    truth = np.where(mask, np.where(x <= 31, 1, 2), 0).astype(np.int32)
    return mask.astype(np.float32), truth


def overlay(path, image, mask, labels, title, other=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.segmentation import find_boundaries

    z = image.shape[0] // 2
    cuts = [
        (image[z], mask[z], labels[z], "central XY"),
        (
            image[:, image.shape[1] // 2],
            mask[:, mask.shape[1] // 2],
            labels[:, labels.shape[1] // 2],
            "central XZ; index aspect",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6), constrained_layout=True)
    lo, hi = np.percentile(image, (1, 99.8))
    for row, (raw, foreground, objects, view) in enumerate(cuts):
        axes[row, 0].imshow(raw, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        axes[row, 0].set_title(f"{view}: raw (display 1–99.8 percentile)")
        axes[row, 1].imshow(raw, cmap="gray", vmin=lo, vmax=hi, aspect="auto")
        rgba = np.zeros((*raw.shape, 4))
        rgba[foreground] = (0.2, 0.6, 1, 0.23)
        rgba[find_boundaries(objects)] = (1, 0.3, 0, 0.9)
        axes[row, 1].imshow(rgba, aspect="auto")
        axes[row, 1].set_title(f"{view}: foreground blue, label boundaries orange")
    if other is not None:
        axes[0, 1].set_title(
            f"{title}; disagreement {np.count_nonzero(other != labels):,} voxels"
        )
    for axis in axes.flat:
        axis.set_axis_off()
    fig.suptitle(title)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def validate_case(name, image, spacing, out, *, source=None, truth=None):
    import numpy as np
    import tifffile
    from skimage.segmentation import watershed

    from napari_vipp.core.execution import PipelineRunRequest, execute_pipeline_request
    from napari_vipp.core.export import export_pipeline_to_python
    from napari_vipp.core.file_sources import load_frozen_file_source_snapshot
    from napari_vipp.core.operations import marker_controlled_watershed

    out.mkdir(parents=True)
    image.setflags(write=False)
    original_hash = array_digest(image)
    document, nodes = build_workflow(phantom=truth is not None)
    write_json(out / "workflow.json", document)
    (graph, axes), timing = measured(lambda: run_vipp(image, spacing, document))
    values = {role: graph.outputs[node_id] for role, node_id in nodes.items()}
    mask, distance, markers, labels = (
        values[key] for key in ("mask", "distance", "markers", "labels")
    )
    reference, reference_timing = measured(
        lambda: watershed(
            -distance, markers, mask=mask, compactness=0, watershed_line=False
        )
    )
    repeat, repeat_timing = measured(
        lambda: marker_controlled_watershed(
            [distance, markers, mask], spatial_mode="3D ZYX"
        )
    )
    slices = marker_controlled_watershed(
        [distance, markers, mask], spatial_mode="2D YX"
    )
    assert np.array_equal(labels, reference), (
        f"{name}: shared execution differs from reference"
    )
    assert np.array_equal(labels, repeat), f"{name}: direct repeat differs"
    assert graph.output_states[nodes["labels"]].axes == axes
    assert array_digest(image) == original_hash
    assert np.all(labels[~mask] == 0)
    assert np.array_equal(labels[markers > 0], markers[markers > 0])
    record = {
        "name": name,
        "source": source,
        "shape_zyx": list(image.shape),
        "dtype": image.dtype.name,
        "input_array_sha256": original_hash,
        "spacing_zyx_um": list(spacing),
        "workflow_sha256": digest(out / "workflow.json"),
        "algorithm_spacing": "voxel-index EDT and watershed; "
        "physical calibration carried only",
        "full_workflow": timing,
        "direct_skimage_watershed": reference_timing,
        "direct_vipp_watershed_repeat": repeat_timing,
        "mask_voxels": int(mask.sum()),
        "seed_label_count": int(np.unique(markers[markers > 0]).size),
        "output_label_count": int(np.unique(labels[labels > 0]).size),
        "unseeded_foreground_voxels": int(np.count_nonzero(mask & (labels == 0))),
        "exact_skimage_parity": True,
        "exact_direct_repeat": True,
        "input_unchanged": True,
        "calibration_preserved": True,
        "seed_ids_preserved": True,
        "slice_vs_volume_different_voxels": int(np.count_nonzero(slices != labels)),
        "label_array_sha256": array_digest(labels),
        "label_history": list(graph.output_states[nodes["labels"]].history),
    }
    if truth is not None:
        # Expected geometry uses a halfway plane; IDs follow left/right seed order.
        record["synthetic_truth_voxel_accuracy_in_foreground"] = float(
            np.mean(labels[mask] == truth[mask])
        )
        assert record["synthetic_truth_voxel_accuracy_in_foreground"] == 1.0
    metadata = {
        "axes": "ZYX",
        "PhysicalSizeZ": spacing[0],
        "PhysicalSizeY": spacing[1],
        "PhysicalSizeX": spacing[2],
        "PhysicalSizeZUnit": "µm",
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeXUnit": "µm",
    }
    tifffile.imwrite(
        out / "watershed-labels.ome.tif", labels, metadata=metadata, compression="zlib"
    )
    # These review inputs preserve every numerical value while carrying the
    # explicitly sourced calibration into a self-contained local GUI workflow.
    review_input = out / "input-calibrated.ome.tif"
    tifffile.imwrite(review_input, image, metadata=metadata, compression="zlib")
    snapshot = load_frozen_file_source_snapshot(review_input)
    assert np.array_equal(snapshot.payload.data, image)
    review_document = json.loads(json.dumps(document))
    input_params = review_document["nodes"][0]["params"]
    input_params["file_path"] = str(review_input)
    input_params["source_mode"] = "file path"
    input_params["_vipp_source_item"] = snapshot.payload.source_item.to_dict()
    write_json(out / "review-workflow.json", review_document)
    repeated, repeated_timing = measured(
        lambda: execute_pipeline_request(
            PipelineRunRequest(
                run_id=2,
                workflow=review_document,
                input_data=snapshot.payload.data,
                input_metadata=snapshot.payload.metadata,
                input_name=name,
                source_payloads={"input": snapshot.payload},
            )
        )
    )
    assert not repeated.error, repeated.error
    assert np.array_equal(repeated.pipeline.outputs[nodes["labels"]], labels)
    record["exact_file_workflow_repeat"] = True
    record["file_workflow_repeat"] = repeated_timing
    record["review_input_sha256"] = digest(review_input)
    record["review_workflow_sha256"] = digest(out / "review-workflow.json")
    (out / "workflow.py").write_text(
        export_pipeline_to_python(repeated.pipeline), encoding="utf-8"
    )
    np.savez_compressed(
        out / "intermediates.npz", mask=mask, distance=distance, markers=markers
    )
    overlay(out / "overlay.png", image, mask, labels, name)
    write_json(out / "result.json", record)
    return record, values


def random_walker_worker(request_file):
    """Kept in a separate process so the parent can enforce resource ceilings."""
    import numpy as np
    from skimage.segmentation import random_walker

    request = json.loads(Path(request_file).read_text(encoding="utf-8"))
    arrays = np.load(request["arrays"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result, timing = measured(
            lambda: random_walker(
                arrays["image"],
                arrays["markers"],
                beta=request["beta"],
                mode="cg_j",
                tol=0.001,
                copy=True,
                spacing=tuple(request["spacing"]),
                return_full_prob=False,
                channel_axis=None,
            )
        )
    np.save(request["output"], result)
    write_json(
        request["receipt"],
        {
            **timing,
            "warnings": [str(item.message) for item in caught],
            "output_sha256": digest(request["output"]),
        },
    )


def benchmark_random_walker(
    name,
    image,
    spacing,
    values,
    out,
    *,
    timeout=45,
    rss_limit=3 * 1024**3,
    truth=None,
):
    import numpy as np
    import psutil

    out.mkdir(parents=True)
    labels = values["labels"]
    original_seeds = values["markers"]
    ids = np.unique(original_seeds[original_seeds > 0])
    # Zero output marks foreground components that have no usable marker. Exclude
    # them explicitly for the RW solve, matching watershed's reached domain.
    markers = np.where(labels > 0, 0, -1).astype(np.int32)
    for new_id, source_id in enumerate(ids, 1):
        markers[original_seeds == source_id] = new_id
    np.savez_compressed(
        out / "input.npz", image=image.astype(np.float32), markers=markers
    )
    request = {
        "arrays": str(out / "input.npz"),
        "output": str(out / "labels.npy"),
        "receipt": str(out / "worker-result.json"),
        "beta": 130,
        "spacing": list(spacing),
    }
    write_json(out / "request.json", request)
    log = (out / "worker.log").open("w", encoding="utf-8")
    started = time.perf_counter()
    process = subprocess.Popen(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            str(out / "request.json"),
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    observer = psutil.Process(process.pid)
    peak = 0
    status = "completed"
    while process.poll() is None:
        try:
            # Windows venv launchers may delegate to a child python.exe.
            family = [observer, *observer.children(recursive=True)]
            peak = max(peak, sum(member.memory_info().rss for member in family))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if time.perf_counter() - started > timeout:
            status = "wall_time_limit"
        elif peak > rss_limit:
            status = "rss_limit"
        if status != "completed":
            for child in observer.children(recursive=True):
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            process.kill()
            process.wait()
            break
        time.sleep(0.01)
    return_code = process.wait()
    log.close()
    if return_code and status == "completed":
        status = "error"
    record = {
        "name": name,
        "shape_zyx": list(image.shape),
        "seed_label_count": len(ids),
        "active_voxels": int((markers >= 0).sum()),
        "status": status,
        "wall_seconds_including_imports_and_io": time.perf_counter() - started,
        "peak_worker_process_tree_rss_bytes_including_imports_and_io": peak,
        "wall_time_limit_seconds": timeout,
        "process_tree_rss_limit_bytes": rss_limit,
        "rss_sample_interval_seconds": 0.01,
        "mode": "cg_j",
        "beta": 130,
        "tol": 0.001,
        "spacing_zyx_um": list(spacing),
        "input_intensity": "float32 preserving native values",
        "inactive_policy": "-1 wherever watershed is zero; "
        "remap seeds densely and restore IDs after solve",
        "return_code": return_code,
    }
    if status == "completed":
        record["solve"] = json.loads(
            (out / "worker-result.json").read_text(encoding="utf-8")
        )
        result = np.load(out / "labels.npy")
        restored = np.zeros(result.shape, dtype=np.int32)
        for new_id, source_id in enumerate(ids, 1):
            restored[result == new_id] = source_id
        record["different_from_watershed_voxels"] = int(
            np.count_nonzero(restored != labels)
        )
        record["seed_ids_preserved"] = bool(
            np.array_equal(
                restored[original_seeds > 0], original_seeds[original_seeds > 0]
            )
        )
        record["unassigned_active_voxels"] = int(
            np.count_nonzero((markers >= 0) & (result <= 0))
        )
        if truth is not None:
            record["synthetic_truth_voxel_accuracy_in_foreground"] = float(
                np.mean(restored[truth > 0] == truth[truth > 0])
            )
        overlay(
            out / "overlay.png",
            image,
            values["mask"],
            restored,
            f"Random Walker {name}",
            other=labels,
        )
    write_json(out / "result.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("D:/VIPP-paper-reproductions/mitochondria"),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--write-example", type=Path)
    parser.add_argument("--skip-random-walker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        random_walker_worker(args.worker)
        return
    if args.write_example:
        document, _nodes = build_workflow(sample=True)
        write_json(args.write_example, document)
        return

    import psutil
    import tifffile

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = (
        args.output_dir
        or args.data_root.parent / "validation" / "seeded-segmentation" / stamp
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "validation-driver.py").write_bytes(Path(__file__).read_bytes())
    report = {
        "created_utc": stamp,
        "script_sha256": digest(__file__),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "total_ram_bytes": psutil.virtual_memory().total,
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-image",
                "tifffile",
                "psutil",
                "napari-vipp",
            )
        },
        "acquisition_manifest_sha256": digest(args.data_root / "manifest.json"),
        "scope": "Implementation parity and feasibility; "
        "no acquired-image ground truth or Fiji validation",
        "watershed": [],
        "random_walker": [],
    }
    image, truth = phantom()
    record, phantom_values = validate_case(
        "analytic-overlapping-spheres",
        image,
        (2.0, 0.5, 0.5),
        output / "analytic-overlapping-spheres",
        truth=truth,
    )
    report["watershed"].append(record)
    if not args.skip_random_walker:
        report["random_walker"].append(
            benchmark_random_walker(
                "analytic-overlapping-spheres",
                image,
                (2.0, 0.5, 0.5),
                phantom_values,
                output / "analytic-random-walker",
                truth=truth,
            )
        )
    print(
        json.dumps(
            {
                "case": record["name"],
                "labels": record["output_label_count"],
                "seconds": record["full_workflow"]["elapsed_seconds"],
            }
        ),
        flush=True,
    )
    cases = [
        (
            "nellie-yeast-t000",
            args.data_root / "derived/nellie_yeast_t000_ZYX.ome.tif",
            (0.25, 0.0655, 0.0655),
        )
    ]
    cases.extend(
        (path.stem.replace(" ", "_"), path, (0.2, 0.1667, 0.1667))
        for path in sorted((args.data_root / "raw/mitograph-mammalian").glob("*.tif"))
    )
    for index, (name, path, spacing) in enumerate(cases):
        image = tifffile.imread(path)
        record, values = validate_case(
            name,
            image,
            spacing,
            output / name,
            source={"path": str(path), "sha256": digest(path)},
        )
        report["watershed"].append(record)
        print(
            json.dumps(
                {
                    "case": name,
                    "labels": record["output_label_count"],
                    "seconds": record["full_workflow"]["elapsed_seconds"],
                }
            ),
            flush=True,
        )
        write_json(output / "report.json", report)
        if not args.skip_random_walker and index in (0, 4):
            center = tuple(size // 2 for size in image.shape)
            selection = (
                slice(None),
                slice(center[1] - 48, center[1] + 48),
                slice(center[2] - 48, center[2] + 48),
            )
            crop = image[selection].copy()
            crop_record, crop_values = validate_case(
                name + "-center96",
                crop,
                spacing,
                output / (name + "-center96"),
                source={
                    "path": str(path),
                    "sha256": digest(path),
                    "crop_zyx": [
                        [0, image.shape[0]],
                        [center[1] - 48, center[1] + 48],
                        [center[2] - 48, center[2] + 48],
                    ],
                },
            )
            report["watershed"].append(crop_record)
            benchmark = benchmark_random_walker(
                name + "-center96",
                crop,
                spacing,
                crop_values,
                output / (name + "-random-walker-crop"),
            )
            report["random_walker"].append(benchmark)
            print(
                json.dumps({"random_walker": name, "status": benchmark["status"]}),
                flush=True,
            )
            if index == 0 and benchmark["status"] == "completed":
                benchmark = benchmark_random_walker(
                    name + "-full",
                    image,
                    spacing,
                    values,
                    output / (name + "-random-walker-full"),
                )
                report["random_walker"].append(benchmark)
            write_json(output / "report.json", report)
    print(str(output), flush=True)
    write_json(output / "report.json", report)


if __name__ == "__main__":
    main()
