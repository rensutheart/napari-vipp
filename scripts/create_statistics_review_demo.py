"""Create measured synthetic data for reviewing descriptive Statistics.

The six images are actually measured by the ordinary batch executor. The four
sample IDs are invented annotations to demonstrate unequal objects/images per
sample, not a claim of biological independence. Existing directories are refused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import runpy
from pathlib import Path

from napari_vipp.core.batch import atomic_write_json
from napari_vipp.core.measurement_collection import (
    collect_measurements,
    inspect_collection,
    save_measurement_collection,
)
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import serialize_workflow


def create_demo(root: Path) -> dict[str, Path]:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    image_demo = runpy.run_path(
        str(Path(__file__).with_name("create_plot_review_demo.py"))
    )["create_demo"](root / "image-analysis")
    manifest = json.loads(image_demo["manifest"].read_text(encoding="utf-8"))
    preview = inspect_collection(manifest, "batch_output")
    conditions = ("Control",) * 3 + ("Treated",) * 3
    sample_ids = (
        "control-a",
        "control-a",
        "control-b",
        "treated-a",
        "treated-a",
        "treated-b",
    )
    annotations = {
        item.key: {
            "condition": conditions[index],
            "image_id": f"synthetic-field-{index + 1:02d}",
            "sample_id": sample_ids[index],
            "data_origin": "Synthetic demonstration; not biological samples",
        }
        for index, item in enumerate(preview.items)
    }
    collection = collect_measurements(preview, annotations=annotations)
    collection_path = root / "Statistics measurements.vipp-results.json"
    save_measurement_collection(collection, collection_path)

    pipeline = PrototypePipeline()
    pipeline.restore_graph([], [])
    source = pipeline.add_node("table_source")
    source.params.update(
        dataset_path=str(collection_path),
        dataset_sha256=hashlib.sha256(collection_path.read_bytes()).hexdigest(),
    )
    positions = {source.id: (0, 520)}
    summaries = []
    levels = (
        ("Objects", "Equal images"),
        ("Image averages", "Equal images"),
        ("Sample averages", "Equal images"),
        ("Sample averages", "Equal objects"),
    )
    for index, (level, weighting) in enumerate(levels):
        node = pipeline.add_node("summarize_measurements")
        node.params.update(
            summary_version=2,
            value_columns=json.dumps(["area_physical", "intensity_mean"]),
            group_by="condition",
            statistics="count,mean,median,std,min,max,q25,q75,iqr",
            summary_level=level,
            image_column="image_id",
            sample_column="sample_id",
            sample_weighting=weighting,
            missing_policy="Exclude and report",
        )
        assert pipeline.connect(source.id, node.id).success
        positions[node.id] = (560, index * 390)
        summaries.append(node)

    plot = pipeline.add_node("plot_results")
    plot.params.update(
        y_column="area_physical_mean",
        group_column="condition",
        summary="None",
        title="One descriptive group mean per condition · synthetic data",
    )
    assert pipeline.connect(summaries[2].id, plot.id).success
    positions[plot.id] = (1100, 780)
    document = serialize_workflow(pipeline, positions)
    document["notes"] = [
        {
            "id": "overview",
            "position": [0, -280],
            "width": 520,
            "text": (
                "DESCRIPTIVE STATISTICS — REAL MEASUREMENTS, SYNTHETIC SAMPLES\n"
                "258 objects across six generated images and four invented sample IDs. "
                "Objects per image: 24 / 40 / 60 (Control), 30 / 46 / 58 (Treated). "
                "Within each condition the first two images share one sample ID; "
                "the third belongs to a second sample. "
                "These are not biological replicates."
            ),
        },
        {
            "id": "levels",
            "position": [560, -280],
            "width": 490,
            "text": (
                "COMPARE THE FOUR BRANCHES\n"
                "Top to bottom: pooled objects; image averages; sample averages with "
                "equal image weighting; sample averages with equal object weighting. "
                "The same data can give different means because these answer different "
                "questions. Check the contributing n and units in each summary."
            ),
        },
        {
            "id": "output",
            "position": [1100, 530],
            "width": 420,
            "text": (
                "SUMMARY TABLES CAN FEED PLOTS\n"
                "This plot has one group mean per condition, not the individual "
                "sample values. SD describes spread; no confidence intervals, "
                "significance tests or biological conclusions are calculated."
            ),
        },
    ]
    document["metadata"] = {
        "vipp": {
            "inspector": {
                "selected_node_id": summaries[2].id,
                "right_panel_visible": True,
            }
        }
    }
    workflow_path = root / "Descriptive statistics review.json"
    atomic_write_json(workflow_path, document)
    return {
        "workflow": workflow_path,
        "collection": collection_path,
        "manifest": image_demo["manifest"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root", type=Path, help="A new directory; existing paths are refused."
    )
    for name, path in create_demo(parser.parse_args().root).items():
        print(f"{name}: {path}")
