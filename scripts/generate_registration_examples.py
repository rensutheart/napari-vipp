"""Regenerate the small, annotated registration workflows in both resource trees."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def source(node_id, sample):
    return node(node_id, "input", source_mode="sample", sample_name=sample)


def node(node_id, operation, **params):
    from napari_vipp.core.pipeline import NODE_LIBRARY_BY_ID

    defaults = {
        item.name: item.default for item in NODE_LIBRARY_BY_ID[operation].parameters
    }
    return {"id": node_id, "operation_id": operation, "params": defaults | params}


def wire(source_id, target, target_port=0, source_port=0, tunnel=""):
    result = {
        "source": source_id,
        "target": target,
        "target_port": target_port,
        "source_port": source_port,
    }
    if tunnel:
        result["tunnel"] = tunnel
    return result


def note(node_id, text, position, width=370):
    return {
        "id": "note_" + node_id,
        "attached_node": node_id,
        "text": text,
        "position": position,
        "width": width,
    }


def document(nodes, connections, positions, notes):
    tunnels = {
        item["tunnel"]: {
            "name": item["tunnel"],
            "source": item["source"],
            "source_port": item["source_port"],
        }
        for item in connections
        if item.get("tunnel")
    }
    return {
        "type": "napari-vipp-workflow",
        "version": 3,
        "metadata": {
            "vipp": {
                "inspector": {
                    "selected_node_id": "estimate",
                    "right_panel_visible": True,
                }
            }
        },
        "nodes": nodes,
        "connections": connections,
        "positions": positions,
        "tunnels": list(tunnels.values()),
        "notes": notes,
        "view": {"zoom": 0.8, "pan": [0.0, 0.0]},
    }


def pair_example(*, rigid):
    title = "3D rigid" if rigid else "2D translation"
    model = "Rigid" if rigid else "Translation"
    nodes = [
        source("input", f"VIPP registration {title} moving"),
        source("reference", f"VIPP registration {title} reference"),
        node(
            "estimate",
            "estimate_registration",
            mode="Two images",
            model=model,
            precision=20,
            iterations=200,
        ),
        node("apply", "apply_transform", interpolation="Linear", outside_value=0.0),
        node(
            "compare_after",
            "compare_images",
            use_mask=True,
            data_range=1.0,
            window_size=7,
        ),
        node(
            "compare_before",
            "compare_images",
            use_mask=True,
            data_range=1.0,
            window_size=7,
        ),
    ]
    connections = [
        wire("input", "estimate", 0),
        wire("reference", "estimate", 1),
        wire("input", "apply", 0, tunnel="Original moving image"),
        wire("estimate", "apply", 1),
        wire("reference", "compare_after", 0, tunnel="Reference image"),
        wire("apply", "compare_after", 1, tunnel="Aligned image"),
        wire("apply", "compare_after", 2, 1, tunnel="Same valid coverage"),
        wire("reference", "compare_before", 0, tunnel="Reference image"),
        wire("input", "compare_before", 1, tunnel="Original moving image"),
        wire("apply", "compare_before", 2, 1, tunnel="Same valid coverage"),
    ]
    positions = {
        "input": [0, 80],
        "reference": [0, 650],
        "estimate": [620, 350],
        "apply": [1520, 350],
        "compare_after": [2260, 80],
        "compare_before": [2260, 650],
    }
    ground_truth = (
        "KNOWN 3D MOTION\nThe entire volume is rotated about all three physical "
        "axes (3, -4 and 6 degrees in the authored ZYX rotation order), then "
        "displaced by (Z=0.8, Y=-0.7, X=1.2) micrometres. Voxel spacing is "
        "(1.1, 0.4, 0.3) micrometres. Translation alone cannot undo this motion."
        if rigid
        else "KNOWN SUBPIXEL MOTION\nObjects in the moving image are displaced "
        "by Y=+4.25 and X=-6.5 pixels. The correcting transform should therefore "
        "shift by Y=-4.25 and X=+6.5 pixels, or Y=-1.7 and X=+1.95 micrometres. "
        "The moving image also has 15% increased intensity, a small offset "
        "and light noise."
    )
    notes = [
        note("input", ground_truth, [0, -260], 460),
        note(
            "estimate",
            "1. ESTIMATE, THEN APPLY\nCalculate Estimate Registration and "
            "inspect its diagnostics output. The first output is a reusable "
            "transform; Apply Transform resamples the original image once onto "
            "the reference grid. All processing is CPU in this first implementation.",
            [640, -240],
            450,
        ),
        note(
            "apply",
            "2. VALID COVERAGE\nApply Transform also returns a Boolean mask "
            "describing valid source support. Outside fill is 0 and must not "
            "be mistaken for measured background. Intensities use linear "
            "interpolation; labels should use nearest neighbour.",
            [1530, -240],
            450,
        ),
        note(
            "compare_after",
            "3. SAME-REGION COMPARISON\nThe upper comparison uses the aligned "
            "image; the lower uses the original moving image. Both use the SAME "
            "coverage mask from Apply Transform. SSIM and PSNR use an explicit "
            "nominal intensity range of 1. Scores describe agreement, not "
            "biological validity.",
            [2820, 100],
            440,
        ),
    ]
    return document(nodes, connections, positions, notes)


def time_example():
    nodes = [
        source("input", "VIPP registration XYZ drift time series"),
        source("labels", "VIPP registration XYZ drift labels"),
        node(
            "estimate",
            "estimate_registration",
            mode="Time series",
            model="Translation",
            channel=0,
            reference_time=0,
            precision=20,
        ),
        node("apply", "apply_transform", interpolation="Linear", outside_value=0.0),
        node(
            "apply_labels",
            "apply_transform",
            interpolation="Nearest",
            outside_value=0.0,
        ),
    ]
    connections = [
        wire("input", "estimate"),
        wire("input", "apply", tunnel="Original two-channel series"),
        wire("estimate", "apply", 1),
        wire("labels", "apply_labels"),
        wire("estimate", "apply_labels", 1, tunnel="Shared XYZ drift transforms"),
    ]
    positions = {
        "input": [0, 100],
        "estimate": [640, 100],
        "apply": [1540, 100],
        "labels": [0, 720],
        "apply_labels": [1540, 720],
    }
    notes = [
        note(
            "input",
            "1. SIX FULL VOLUMES, TWO CHANNELS\nEach timepoint is a complete "
            "XYZ volume. The second channel has different object intensities "
            "but exactly the same stage motion. There is no independent "
            "biological movement in this analytical sample.",
            [0, -220],
            440,
        ),
        note(
            "estimate",
            "2. ONE TRANSFORM PER TIMEPOINT\nUse channel 0 for estimation and "
            "timepoint 0 as the reference. The known apparent displacements in "
            "ZYX pixels are: (0,0,0), (0.6,-1.25,2.5), (1.2,-2.5,4.5), "
            "(-0.4,-0.75,2), (0.8,1.5,-2), (1.5,2.25,-3.5). Correcting "
            "shifts have the opposite sign.",
            [640, -240],
            460,
        ),
        note(
            "apply",
            "3. REUSE ACROSS CHANNELS\nApply the estimated XYZ motion to both "
            "original channels. Time and channel axes are never spatial "
            "dimensions, and individual Z slices are never registered "
            "separately. Scrub T before and after correction.",
            [1540, -220],
            450,
        ),
        note(
            "labels",
            "4. LABELS FOLLOW THE SAME COORDINATE FRAME\nThe companion TZYX "
            "labels share the same source coordinate identity as the two-channel "
            "image. Reuse the transform series with nearest-neighbour "
            "interpolation to preserve integer object IDs. Fractional shifts "
            "can still change voxelized object boundaries.",
            [0, 1100],
            480,
        ),
        note(
            "apply_labels",
            "VALIDITY AND LIMITATIONS\nBoth Apply Transform nodes produce "
            "coverage masks. Keep them when comparing or measuring aligned "
            "data. Registration diagnostics are not confidence probabilities, "
            "and successful alignment does not prove biological equivalence.",
            [1540, 1100],
            430,
        ),
    ]
    return document(nodes, connections, positions, notes)


def main():
    workflows = {
        "synthetic-registration-translation.json": pair_example(rigid=False),
        "synthetic-registration-rigid-3d.json": pair_example(rigid=True),
        "synthetic-registration-time-series.json": time_example(),
    }
    for filename, workflow in workflows.items():
        rendered = json.dumps(workflow, indent=2, ensure_ascii=False) + "\n"
        for directory in (ROOT / "examples", ROOT / "src/napari_vipp/examples"):
            (directory / filename).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
