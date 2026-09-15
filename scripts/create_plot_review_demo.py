"""Generate a real six-image batch and collected-table plotting review fixture.

This development helper writes only into a new, explicitly requested directory.
It runs the ordinary batch/collection APIs so the collection contains genuine
output hashes and run evidence rather than fabricated verification records.
Synthetic conditions demonstrate UI grouping, not biological replication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path

import tifffile

from napari_vipp._sample_data import _measurement_plot_sample
from napari_vipp.core.batch import (
    BatchConfig,
    BatchOutputConfig,
    BatchSourceConfig,
    atomic_write_json,
    run_batch,
    save_batch_config,
    scientific_workflow_hash,
)
from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.measurement_collection import (
    collect_measurements,
    inspect_collection,
    save_measurement_collection,
)
from napari_vipp.ui.examples import _example_workflow_by_id, _example_workflow_path


def create_demo(root: Path) -> dict[str, Path]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root / "inputs"
    inputs.mkdir()
    # Unequal yields make object pooling and equally weighted image means
    # observably different. There are three invented fields per condition.
    specifications = (
        ("Control", 24, 0.92, 0.96),
        ("Control", 40, 1.0, 1.0),
        ("Control", 60, 1.04, 1.04),
        ("Treated", 30, 1.06, 1.12),
        ("Treated", 46, 1.09, 1.18),
        ("Treated", 58, 1.12, 1.24),
    )
    for index, (condition, count, size, intensity) in enumerate(specifications, 1):
        data, _kwargs, _kind = _measurement_plot_sample(
            seed=20260915 + index,
            object_count=count,
            size_factor=size,
            intensity_factor=intensity,
        )
        tifffile.imwrite(
            inputs / f"{index:02d}_{condition.lower()}_field.ome.tif",
            data,
            ome=True,
            metadata={
                "axes": "YX",
                "PhysicalSizeX": 0.5,
                "PhysicalSizeY": 0.5,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeYUnit": "µm",
            },
        )
    spec = _example_workflow_by_id("plot-morphology")
    original = json.loads(_example_workflow_path(spec).read_text(encoding="utf-8"))
    workflow = deepcopy(original)
    retained = {
        node["id"]
        for node in workflow["nodes"]
        if node["operation_id"] != "plot_results" and node["id"] != "annotated"
    }
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] in retained]
    workflow["connections"] = [
        edge for edge in workflow["connections"] if edge["target"] in retained
    ]
    workflow["positions"] = {
        node_id: position
        for node_id, position in workflow["positions"].items()
        if node_id in retained
    }
    workflow["metadata"] = {}
    workflow["notes"] = []
    workflow["nodes"][0]["params"].update(
        source_mode="file path",
        sample_name="",
        file_path=str(inputs / "01_control_field.ome.tif"),
    )
    output_params = {
        "tag": "objects",
        "format": "tsv",
        "subfolder": "tables",
        "filename_template": "{source_stem}__{tag}",
        "overwrite": "batch default",
    }
    workflow["nodes"].append(
        {
            "id": "batch_output",
            "operation_id": "batch_output",
            "params": output_params,
        }
    )
    workflow["connections"].append(
        {
            "source": "merged",
            "target": "batch_output",
            "source_port": 0,
            "target_port": 0,
        }
    )
    workflow["positions"]["batch_output"] = [2650, 500]
    workflow_path = root / "Measure six synthetic fields.json"
    atomic_write_json(workflow_path, workflow)
    config = BatchConfig(
        workflow_file=workflow_path,
        workflow_sha256=scientific_workflow_hash(workflow),
        output_dir=root / "outputs",
        sources=(BatchSourceConfig("input", "Synthetic fields", inputs, "*.ome.tif"),),
        outputs=(
            BatchOutputConfig(
                "batch_output",
                "Batch Output",
                "objects",
                "table",
                "tsv",
                "tables",
                "{source_stem}__{tag}",
            ),
        ),
        compute_request=ComputeRequest(mode="cpu"),
        continue_on_error=False,
    )
    config_path = root / "batch-config.json"
    save_batch_config(config_path, config)
    result = run_batch(
        workflow, config, workflow_path=workflow_path, config_path=config_path
    )
    manifest = result.manifest.to_dict()
    preview = inspect_collection(manifest, "batch_output")
    annotations = {
        item.key: {
            "condition": specifications[index][0],
            "image_id": f"synthetic_field_{index + 1:02d}",
            "data_origin": "Synthetic demonstration; not biological samples",
        }
        for index, item in enumerate(preview.items)
    }
    collection = collect_measurements(preview, annotations=annotations)
    expected_count = sum(entry[1] for entry in specifications)
    if collection.table.row_count != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} objects in generated collection."
        )
    collection_path = root / "Synthetic grouped measurements.vipp-results.json"
    save_measurement_collection(collection, collection_path)
    digest = hashlib.sha256(collection_path.read_bytes()).hexdigest()
    plot_nodes = [
        deepcopy(node)
        for node in original["nodes"]
        if node["operation_id"] == "plot_results"
    ]
    for node in plot_nodes:
        node["params"].update(group_column="condition", image_column="image_id")
    plot_nodes[0]["params"].update(
        normalization="Percent", title="Area distributions · two synthetic conditions"
    )
    plot_nodes[1]["params"].update(title="Area and intensity · 258 synthetic objects")
    plot_nodes[2]["params"].update(
        y_column="area_physical",
        point_unit="Mean per image",
        summary="Mean",
        title="Mean object area per image · three fields per condition",
    )
    plot_nodes[3]["params"].update(title="Circularity · cumulative distributions")
    grouped = {
        "type": original["type"],
        "version": original["version"],
        "metadata": {
            "vipp": {
                "inspector": {
                    "selected_node_id": "plot_elongation",
                    "right_panel_visible": True,
                }
            }
        },
        "nodes": [
            {
                "id": "table_source",
                "operation_id": "table_source",
                "params": {
                    "dataset_path": str(collection_path),
                    "dataset_sha256": digest,
                },
            },
            *plot_nodes,
        ],
        "connections": [
            {
                "source": "table_source",
                "target": node["id"],
                "source_port": 0,
                "target_port": 0,
            }
            for node in plot_nodes
        ],
        "positions": {
            "table_source": [0, 660],
            **{node["id"]: [650, index * 460] for index, node in enumerate(plot_nodes)},
        },
        "notes": [
            {
                "id": "synthetic",
                "text": (
                    "SIX SYNTHETIC IMAGES, TWO DEMONSTRATION CONDITIONS\n"
                    "258 measured objects from unequal yields (24/40/60 and 30/46/58). "
                    "Compare pooled objects with one mean per image. Control/Treated "
                    "are invented labels; these are not biological replicates. All "
                    "measurements came from the saved real batch run in this folder."
                ),
                "position": [0, -240],
                "width": 580,
            },
            {
                "id": "compare",
                "text": (
                    "COMPARE OBSERVATIONAL LEVELS\nThe third plot starts with one "
                    "mean area per image (6 points). Change Each point represents to "
                    "Objects to see 258 points; images with more objects then "
                    "contribute more observations. Histogram percentages use "
                    "shared bins."
                ),
                "position": [650, -240],
                "width": 580,
            },
        ],
    }
    grouped_path = root / "Grouped measurement plots.json"
    atomic_write_json(grouped_path, grouped)
    return {
        "workflow": grouped_path,
        "collection": collection_path,
        "manifest": result.manifest_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="A new directory; existing paths are refused."
    )
    for name, path in create_demo(parser.parse_args().root).items():
        print(f"{name}: {path}")
