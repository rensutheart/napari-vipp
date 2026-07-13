"""Example workflow registry and resource lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExampleWorkflowSpec:
    id: str
    category: str
    title: str
    filename: str
    samples: tuple[str, ...]
    description: str
    generated_batch_demo: bool = False


EXAMPLE_WORKFLOWS: tuple[ExampleWorkflowSpec, ...] = (
    ExampleWorkflowSpec(
        "batch-provenance",
        "Batch & Reproducibility",
        "Deterministic Batch & Provenance",
        "synthetic-batch-provenance.json",
        ("Ready-to-run paired .npy demo",),
        "Open a ready-to-run three-item batch demo with paired sources, "
        "collision-aware preview, explicit image/label/table outputs, saved "
        "config and runner files, manifests, archives, per-item provenance, "
        "and automatic ground-truth validation.",
        generated_batch_demo=True,
    ),
    ExampleWorkflowSpec(
        "label-cleanup",
        "Segmentation & Labels",
        "Red-Channel Label Cleanup",
        "otsu-red-channel-labels.json",
        ("VIPP synthetic multichannel volume",),
        "Split the red/TRITC-like channel, blur, threshold, fill holes, "
        "label objects, clear borders, and filter labels by volume.",
    ),
    ExampleWorkflowSpec(
        "object-intensity",
        "Measurements & Tables",
        "Object Intensity Measurements",
        "red-channel-object-intensity-measurements.json",
        ("VIPP synthetic multichannel volume",),
        "Measure cleaned object labels together with matching red-channel "
        "intensity values.",
    ),
    ExampleWorkflowSpec(
        "merged-measurements",
        "Measurements & Tables",
        "Merged Measurement Table",
        "red-channel-merged-measurement-table.json",
        ("VIPP synthetic multichannel volume",),
        "Build a PCA-oriented table from object morphology, object intensity, "
        "and metadata columns.",
    ),
    ExampleWorkflowSpec(
        "summary-table",
        "Measurements & Tables",
        "Grouped Measurement Summary",
        "synthetic-measurement-summary.json",
        ("VIPP synthetic measurement summary",),
        "Summarize known object counts and areas across timepoints.",
    ),
    ExampleWorkflowSpec(
        "derived-morphology",
        "Measurements & Tables",
        "Derived 2D Object Morphology",
        "synthetic-derived-object-morphology.json",
        ("VIPP synthetic object morphology",),
        "Calculate 2D morphology, circularity, perimeter-area ratio, and Hu "
        "moments.",
    ),
    ExampleWorkflowSpec(
        "mesh-morphology",
        "Measurements & Tables",
        "3D Mesh Morphology",
        "synthetic-3d-mesh-morphology.json",
        ("VIPP synthetic 3D mesh morphology",),
        "Measure anisotropic 3D objects with surface area, mesh volume, convex "
        "hull metrics, and sphericity.",
    ),
    ExampleWorkflowSpec(
        "skeleton-qc",
        "Skeletons & Networks",
        "Skeleton QC",
        "synthetic-skeleton-qc.json",
        ("VIPP synthetic skeleton network",),
        "Inspect skeleton keypoints, components, branches, pruning, graph "
        "tables, and network summaries.",
    ),
    ExampleWorkflowSpec(
        "advanced-skeleton",
        "Skeletons & Networks",
        "Advanced Skeleton Network",
        "synthetic-advanced-skeleton-network.json",
        ("VIPP synthetic advanced skeleton network",),
        "Review time-indexed 3D skeleton/network analysis with loops, fragments, "
        "spurs, and anisotropic calibration.",
    ),
    ExampleWorkflowSpec(
        "racc-colocalization",
        "Colocalization & Association",
        "RACC Colocalization",
        "synthetic-colocalization-racc.json",
        ("VIPP synthetic colocalization",),
        "Inspect red/green channel tunnels, ROI masks, scatter thresholds, "
        "Manders/Pearson metrics, and RACC output.",
    ),
    ExampleWorkflowSpec(
        "object-colocalization",
        "Colocalization & Association",
        "Object Colocalization Association",
        "synthetic-object-colocalization-association.json",
        ("VIPP synthetic colocalization",),
        "Combine thresholded channel labels, object colocalization rows, label "
        "overlap, nearest distances, and event localization.",
    ),
    ExampleWorkflowSpec(
        "deconvolution-2d",
        "Restoration & PSF",
        "2D Richardson-Lucy / TV Deconvolution",
        "synthetic-deconvolution-rl-tv.json",
        ("VIPP synthetic deconvolution image", "VIPP synthetic measured PSF"),
        "Prepare a measured PSF and compare ordinary Richardson-Lucy with "
        "Richardson-Lucy TV restoration.",
    ),
    ExampleWorkflowSpec(
        "deconvolution-3d",
        "Restoration & PSF",
        "3D Richardson-Lucy / TV Deconvolution",
        "synthetic-3d-deconvolution-rl-tv.json",
        (
            "VIPP synthetic 3D deconvolution volume",
            "VIPP synthetic 3D measured PSF",
        ),
        "Run volumetric PSF-aware restoration with matched 3D image and PSF "
        "sources.",
    ),
)


def _example_workflow_by_id(example_id: str) -> ExampleWorkflowSpec | None:
    for spec in EXAMPLE_WORKFLOWS:
        if spec.id == example_id:
            return spec
    return None


def _example_workflow_dir() -> Path:
    candidates = (
        Path(__file__).resolve().parents[1] / "examples",
        Path(__file__).resolve().parents[3] / "examples",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return candidates[-1]


def _example_workflow_path(spec: ExampleWorkflowSpec) -> Path:
    return _example_workflow_dir() / spec.filename


__all__ = [
    "EXAMPLE_WORKFLOWS",
    "ExampleWorkflowSpec",
    "_example_workflow_by_id",
    "_example_workflow_dir",
    "_example_workflow_path",
]
