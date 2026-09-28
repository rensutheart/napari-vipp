"""Regenerate annotated known-answer detection graphs in both resource trees."""

from __future__ import annotations

import json
from pathlib import Path

from generate_registration_examples import document, node, note, source, wire

ROOT = Path(__file__).resolve().parents[1]


def detection_example(dimensions):
    volume = dimensions == 3
    crop = (
        dict(z_start=2, z_end=14, top=10, bottom=45, left=10, right=59)
        if volume
        else dict(top=14, bottom=61, left=15, right=86)
    )
    nodes = [
        source("input", f"VIPP synthetic {dimensions}D template detection"),
        node("time", "select_axis_slice", axis=0, index=1,
             range_mode=True, remove_axes="0", remove_indices="1"),
        node("channel", "extract_channel", channel=1),
        node("template", "crop_stack", **crop),
        node("match", "template_match"),
        node(
            "peaks",
            "find_peaks",
            use_mask=True,
            minimum_value=0.8,
            minimum_separation=5.0 if volume else 8.0,
            separation_units="Physical (micrometers)" if volume else "Pixels",
            maximum_detections=100,
            border_exclusion=0,
        ),
        node(
            "scores",
            "plot_results",
            plot_type="Distribution",
            y_column="score",
            distribution="Histogram",
            bins=10,
            title="Template correlation at retained detections",
        ),
    ]
    connections = [
        wire("input", "time"),
        wire("time", "channel"),
        wire("channel", "template"),
        wire("channel", "match", 0, tunnel="Selected source image"),
        wire("template", "match", 1),
        wire("match", "peaks", 0),
        wire("match", "peaks", 1, 1, tunnel="Valid template support"),
        wire("peaks", "scores"),
    ]
    positions = {
        "input": [0, 100],
        "time": [570, 100],
        "channel": [1140, 100],
        "template": [1710, 100],
        "match": [2280, 100],
        "peaks": [2850, 100],
        "scores": [3570, 100],
    }
    truth = (
        "Four source-index ZYX centers: (5,14,15), (5,14,31), (12,40,24), "
        "(16,44,61). Spacing is (1.5,0.5,0.4) micrometres, origin (-3,4,11). "
        "The first pair is 6.4 micrometres apart."
        if volume
        else "Five source-index YX centers: (20,20), (20,65), (47,34), (47,46), "
        "(72,80). The nearby pair is 12 pixels apart; amplitudes differ."
    )
    notes = [
        note(
            "input",
            "1. EXPLICIT INPUT\nSelect T=1, then Repeated pattern "
            "(channel 1). Channel 0 is independent noise; T=0 is a different "
            "arrangement. These are seeded synthetic images, not biological data.",
            [0, -260],
            460,
        ),
        note(
            "template",
            "2. EXISTING CROP STACK\nCrop the first complete "
            f"pattern to {'7 x 9 x 11' if volume else '13 x 11'} samples. "
            "No resizing, rotating, segmentation or template library is used. "
            "Template origin is independent; its pixel/voxel sampling must match.",
            [1630, -260],
            460,
        ),
        note(
            "match",
            "3. SCORE SUPPORT\nThe smaller score map contains only "
            "full-template placements. Its origin is offset to template centers. "
            "Keep the Boolean valid-support output connected to Find Peaks: "
            "undefined correlations are not measured zeros.",
            [2260, -260],
            460,
        ),
        note(
            "peaks",
            "4. KNOWN ANSWER\n" + truth + " Inspect the detection "
            "table and source-coordinate overlay. The clipped border copy and "
            "the deliberately missing site are not expected detections.",
            [2880, -300],
            500,
        ),
        note(
            "scores",
            "5. SCORES, NOT CONFIDENCE\nOpen the score histogram. "
            "The template's own location is included, so its near-one score is "
            "not an independent validation result. Correlation is not probability. "
            "Calculation writes no files and produces no segmentation labels.",
            [3590, -260],
            470,
        ),
    ]
    result = document(nodes, connections, positions, notes)
    result["metadata"]["vipp"]["inspector"]["selected_node_id"] = "peaks"
    result["view"]["zoom"] = 0.55
    return result


def main():
    for dimensions in (2, 3):
        payload = json.dumps(detection_example(dimensions), indent=2) + "\n"
        for folder in (ROOT / "examples", ROOT / "src/napari_vipp/examples"):
            (folder / f"synthetic-template-detection-{dimensions}d.json").write_text(
                payload, encoding="utf-8", newline="\n"
            )


if __name__ == "__main__":
    main()
