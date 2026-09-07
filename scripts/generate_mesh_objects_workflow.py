"""Generate the small, calibrated mesh-object acceptance workflow.

No image writer is included. The Batch Output is only an explicit 3MF output
declaration; opening or calculating the workflow does not write a file.
"""

from __future__ import annotations

from pathlib import Path

from napari_vipp.core.compute import ComputeRequest
from napari_vipp.core.pipeline import PrototypePipeline
from napari_vipp.core.workflow import save_workflow

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FILENAME = "synthetic-mesh-objects.json"


def build_workflow():
    """Return a deterministic graph, positions, notes and initial inspector state."""
    pipeline = PrototypePipeline()
    pipeline.reset_empty_graph()
    pipeline.nodes["input"].params.update(
        source_mode="sample",
        sample_name="VIPP synthetic 3D mesh morphology",
        file_path="",
        layer_name="",
        binding_mode="single item",
    )
    positions = {"input": (0.0, 0.0)}

    def add(operation, x, y, **parameters):
        node = pipeline.add_node(operation)
        node.params.update(parameters)
        positions[node.id] = (float(x), float(y))
        return node.id

    def connect(source, target, port=0):
        result = pipeline.connect(source, target, target_port=port)
        if not result.success:
            raise RuntimeError(result.message)

    threshold = add("binary_threshold", 350, 0, threshold=1000, foreground="Above")
    mesh = add("mask_to_3d_mesh", 700, 0, object_mode="Connected objects")
    split = add("split_mesh_objects", 1050, 0)
    colors = add("color_mesh_objects", 1400, 0, color_by="mesh_volume_physical")
    large = add("filter_mesh_objects", 1750, 0, minimum=10, maximum=1e12)
    tiny = add(
        "filter_mesh_objects", 1750, 340, minimum=10, maximum=1e12, keep="Outside range"
    )
    combined = add("combine_meshes", 2100, 120, input_count=2)
    smoothed = add("smooth_mesh", 2450, 120, iterations=3, strength=0.3)
    simplified = add("simplify_mesh", 2800, 120, target_percent=65.0)
    measure = add(
        "measure_3d_mesh_morphology", 3150, 0, include_convex_hull_metrics=False
    )
    output = add(
        "batch_output", 3150, 340, format="3mf", tag="mesh_objects", overwrite="no"
    )
    for source, target in (
        ("input", threshold),
        (threshold, mesh),
        (mesh, split),
        (split, colors),
        (colors, large),
        (colors, tiny),
        (large, combined),
        (combined, smoothed),
        (smoothed, simplified),
        (simplified, measure),
        (simplified, output),
    ):
        connect(source, target)
    connect(tiny, combined, 1)
    notes = [
        dict(
            id="mesh_start",
            position=(0.0, -250.0),
            width=670.0,
            text="1. CALCULATE ALL, THEN INSPECT THE MESHES\n"
            "Five synthetic objects; explicit Z/Y/X, 2 x 0.5 x 0.5 micrometres. "
            "Connected objects retains foreground IDs, including cavity walls. "
            "Manual mesh nodes calculate only when requested.",
        ),
        dict(
            id="mesh_identity",
            position=(1050.0, -250.0),
            width=670.0,
            text="2. IDENTITIES, COLOURS AND TWO DISJOINT GROUPS\n"
            "Split uses shared triangle edges; no implicit union. Colour uses "
            "current mesh volume. In range keeps four objects >= 10 um^3; "
            "Outside range keeps the tiny object. Combine reunites all five "
            "without duplicate objects, welding or lost colours.",
        ),
        dict(
            id="mesh_refine",
            position=(2450.0, -250.0),
            width=1000.0,
            text="3. COMPARE ORIGINAL AND REFINED GEOMETRY\n"
            "Smoothing and simplification create new meshes; the upstream "
            "originals remain available. Inspect shape and updated measurements. "
            "Export a selected mesh as 3MF to preserve units, objects and colours. "
            "The Batch Output only declares an output: Calculate all saves nothing.",
        ),
    ]
    metadata = {
        "vipp": {
            "inspector": {
                "selected_node_id": colors,
                "right_panel_visible": True,
            }
        }
    }
    return pipeline, positions, notes, metadata


def main():
    pipeline, positions, notes, metadata = build_workflow()
    for directory in (
        REPOSITORY_ROOT / "examples",
        REPOSITORY_ROOT / "src" / "napari_vipp" / "examples",
    ):
        target = save_workflow(
            directory / FILENAME,
            pipeline,
            positions=positions,
            notes=notes,
            metadata=metadata,
            compute_request=ComputeRequest(mode="cpu"),
        )
        print(f"Generated {target}")


if __name__ == "__main__":
    main()
