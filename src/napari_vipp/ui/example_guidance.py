"""Curated, read-only guidance for the bundled example chooser.

These descriptions support choosing and exploring a workflow. They are display
metadata only and never change the executable example or its saved parameters.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class ExampleGuidance:
    """Plain-text guidance with action bullets and optional method details."""

    purpose: str
    data: str
    explore: tuple[str, ...]
    results: tuple[str, ...]
    try_this: str
    caution: str
    method_name: str = ""
    method_description: str = ""
    method_meaning: str = ""
    paper_authors: str = ""
    paper_journal: str = ""
    paper_url: str = ""
    chooser_category: str = ""


EXAMPLE_GUIDANCE_BY_ID: Mapping[str, ExampleGuidance] = MappingProxyType(
    {
        "label-cleanup": ExampleGuidance(
            purpose="Turn the red channel into a clean set of labelled 3D objects.",
            data="Synthetic multichannel volume",
            explore=(
                (
                    "Compare the red-channel image with the foreground mask created "
                    "by thresholding."
                ),
                "Inspect the labelled objects after holes in the mask are filled.",
                "Filter objects by volume and whether they touch the image edge.",
            ),
            results=("Cleaned object labels",),
            try_this=(
                "Change the minimum volume in the label filter. Calculate again and "
                "compare how many objects are kept or removed."
            ),
            caution=(
                "Check the spatial calibration before interpreting physical object "
                "volumes."
            ),
        ),
        "separate-overlapping-objects": ExampleGuidance(
            purpose=(
                "Recover a rounded object and a curved band from one overlapping "
                "signal, then make a mesh for each."
            ),
            data="Synthetic volume with two overlapping structures",
            explore=(
                "Compare the regions selected by different intensity thresholds.",
                "Use XOR to keep regions found in only one of the two masks.",
                "Use OR to join the pieces of the curved band into one mask.",
                "Inspect the separate 3D surfaces made from the two object masks.",
            ),
            results=(
                "Two object masks",
                "Separate coloured meshes",
                "Mesh measurements",
            ),
            try_this=(
                "Compare the masks before and after Logical OR, then inspect the two "
                "meshes to see which structure each represents."
            ),
            caution=(
                "The saved thresholds and cleanup settings are tailored to this sample."
            ),
        ),
        "gpu-segmentation": ExampleGuidance(
            purpose=(
                "Follow a segmentation workflow that can use the GPU while keeping "
                "CPU fallbacks visible."
            ),
            data="Synthetic GPU segmentation cleanup sample",
            explore=(
                "Follow the selected channel through filtering and thresholding.",
                "Follow mask cleanup through to separately labelled 3D objects.",
                "Check each node's CPU/GPU assignment to see where processing runs.",
            ),
            results=(
                "Binary mask",
                "3D object labels",
                "Per-node compute assignments",
            ),
            try_this=(
                "Compare the compute assignments with CPU and Prefer GPU selected. "
                "Follow the canvas notes to see where processing can use the GPU."
            ),
            caution=(
                "Prefer GPU does not mean every node will run on the GPU; supported "
                "hardware and operations determine the assignments."
            ),
        ),
        "object-intensity": ExampleGuidance(
            purpose="Measure red-channel signal inside each segmented object.",
            data="Synthetic multichannel volume",
            explore=(
                "Inspect the cleaned labels that define each object's boundary.",
                "Check which red-channel image supplies the intensity values.",
                "Read the intensity measurements for each object in the results table.",
            ),
            results=(
                "Object labels",
                "Per-object intensity table",
            ),
            try_this=(
                "Choose an object in the label image and find its label ID in the "
                "intensity table. Compare its intensity with another object."
            ),
            caution=(
                "Measurements depend on both the object boundaries and the image "
                "used for intensity values."
            ),
        ),
        "merged-measurements": ExampleGuidance(
            purpose=(
                "Bring object shape, intensity and experiment metadata together in "
                "one table."
            ),
            data="Synthetic multichannel volume",
            explore=(
                (
                    "Compare shape and intensity measurements made from the same "
                    "object labels."
                ),
                "Join the two tables by matching each object's label ID.",
                "Add condition and replicate columns to describe the experiment.",
            ),
            results=("Combined object-level measurement table",),
            try_this=(
                "Edit the condition in Add Metadata Columns and calculate again. "
                "Find the new value beside the existing shape and intensity "
                "measurements."
            ),
            caution=(
                "The workflow prepares a table for further analysis; it does not "
                "perform PCA."
            ),
        ),
        "summary-table": ExampleGuidance(
            purpose=(
                "Explore measurements, summaries and plots together in the "
                "Results Workspace."
            ),
            data=(
                "Synthetic measurement-summary sample with known object counts and "
                "areas"
            ),
            explore=(
                (
                    "Select Add Metadata Columns or Statistics, then open the "
                    "Results Workspace to see the table beside its controls."
                ),
                (
                    "Group the six object rows across three timepoints to "
                    "calculate counts and mean, minimum and maximum areas."
                ),
                (
                    "Check included values and see why one object cannot define "
                    "a standard deviation."
                ),
                (
                    "Compare the original-data plot, with six object points, "
                    "with the summary plot, with one mean per timepoint."
                ),
            ),
            results=(
                "Object-level measurements",
                "Grouped summary table",
                "Original-object and summary-mean plots",
            ),
            try_this=(
                "In the workspace's Plots tab, switch between the original "
                "measurements and summary table. Check which rows each point "
                "represents. In Appearance, try a Y axis label interval of 5."
            ),
            caution=(
                "Neither object rows nor summary rows automatically represent "
                "independent biological replicates. A summary's SD column does "
                "not automatically add error bars to a plot."
            ),
        ),
        "derived-morphology": ExampleGuidance(
            purpose="Describe how round, elongated or irregular a 2D object is.",
            data="Synthetic 2D object-morphology image",
            explore=(
                "Inspect object area and boundary length relative to that area.",
                "Use circularity and axis ratios to compare roundness and elongation.",
                "Inspect Hu moments, numerical descriptions of object shape.",
            ),
            results=(
                "Labelled 2D objects",
                "Shape descriptor table",
            ),
            try_this=(
                "Compare a rounded object with an elongated one. Look at circularity "
                "and the major-to-minor axis ratio in the table."
            ),
            caution="These are 2D shape descriptors, not 3D surface measurements.",
        ),
        "plot-morphology": ExampleGuidance(
            purpose=(
                "Turn measurements from one image into readable shape and "
                "intensity plots."
            ),
            data=(
                "One synthetic 2D image with 60 separated ellipses; "
                "0.5 micrometer per pixel"
            ),
            explore=(
                "Follow thresholding and labelling to one measurement row per object.",
                "Compare the area histogram with individual elongation measurements.",
                "Explore the designed relationship between area and brightness.",
                "Open a plot window, change its settings and export a sized figure.",
            ),
            results=(
                "60 labelled objects",
                "Joined shape and intensity measurements",
                "Histogram, scatter, individual-point and cumulative plots",
            ),
            try_this=(
                "Select the area-intensity scatter node and choose Open plot. "
                "Change the Y measurement to eccentricity to compare a different "
                "relationship. The source image does not need recalculation."
            ),
            caution=(
                "These are invented demonstration data. The 60 objects come from "
                "one image, not 60 independent biological samples; size and "
                "brightness were deliberately linked in the sample generator."
            ),
        ),
        "mesh-morphology": ExampleGuidance(
            purpose=(
                "Measure 3D object shape using both voxels and reconstructed surfaces."
            ),
            data="Synthetic 3D objects with unequal voxel spacing across axes",
            explore=(
                (
                    "Compare object volumes estimated from voxels and from "
                    "reconstructed surfaces."
                ),
                (
                    "Inspect surface area and sphericity, a measure of how "
                    "sphere-like an object is."
                ),
                (
                    "Compare each object with its convex hull, an outer envelope "
                    "without inward dents."
                ),
                "Check each object's mesh status beside its measurements.",
            ),
            results=("Combined voxel and mesh morphology table",),
            try_this=(
                "Compare volume_physical with mesh_volume_physical for one object, "
                "then inspect its sphericity and mesh status."
            ),
            caution=(
                "Voxel and surface-based volumes are different estimates and need "
                "not be identical."
            ),
        ),
        "racc-colocalization": ExampleGuidance(
            purpose=(
                "Explore RACC colour mapping across a two-channel image and within a "
                "selected region."
            ),
            data="Synthetic two-channel 3D image",
            explore=(
                (
                    "Compare whole-image and region-restricted RACC maps using the "
                    "Magma colour scheme."
                ),
                (
                    "Review the manual signal cutoffs: channel 1 at 43,970.51 and "
                    "channel 2 at 48,073.03."
                ),
                (
                    "Inspect the region selected from the red channel at a cutoff of "
                    "30,000."
                ),
                (
                    "Change Theta to see how strongly signal pairs away from the "
                    "fitted relationship are reduced."
                ),
            ),
            results=(
                "Whole-image RACC map",
                "ROI-restricted RACC map",
            ),
            try_this=(
                "In RACC Index, increase Theta from 45° to 60° and recalculate. "
                "Inspect which regions become less prominent as intensity pairs "
                "farther from the fitted line receive a stronger penalty."
            ),
            caution=(
                "Whole-image and ROI results fit different voxel populations; their "
                "colours should not be treated as a shared quantitative scale."
            ),
            method_name="Regression adjusted colocalisation colour mapping (RACC)",
            method_description=(
                "The method fits a line through the paired channel intensities using "
                "Deming regression. It emphasises voxels with stronger combined "
                "signal that lie close to this relationship, while reducing the "
                "contribution of pairs that deviate from it. The resulting colour "
                "map shows how these intensity relationships vary across the image."
            ),
            method_meaning=(
                "Unlike a yes/no overlap mask, RACC provides a graded view. It is a "
                "qualitative visualisation, not a probability of colocalisation or a "
                "replacement for Pearson and Manders measurements."
            ),
            paper_authors="Theart, Loos & Niesler (2019)",
            paper_journal="PLOS ONE 14(11): e0225141.",
            paper_url="https://doi.org/10.1371/journal.pone.0225141",
        ),
        "colocalization-overlap": ExampleGuidance(
            purpose=(
                "Review thresholds, measure colocalization and turn shared voxels "
                "into measurable overlap regions."
            ),
            data="Synthetic two-channel 3D image",
            explore=(
                "Inspect the automatic intensity cutoffs found by Costes thresholding.",
                (
                    "Compare white overlap views for the whole image and the "
                    "selected red-channel region."
                ),
                (
                    "Compare Pearson (signal correlation) and Manders (fractions of "
                    "signal overlapping) measurements."
                ),
                (
                    "Count connected overlap regions after removing small ones from "
                    "the whole-image mask."
                ),
            ),
            results=(
                "Overlap overlays",
                "Whole-image and ROI metric tables",
                "Boolean masks",
                "Labelled regions",
                "Object measurements",
            ),
            try_this=(
                "In Remove Small Objects, lower the minimum size from 20 to 1 voxel. "
                "Calculate again and compare the number of rows in Measure Objects."
            ),
            caution=(
                "These counts describe connected overlap regions—not the original "
                "organelles. The counting branch is not restricted to the ROI."
            ),
        ),
        "object-colocalization": ExampleGuidance(
            purpose=(
                "Find which objects from two channels overlap, and how far apart the "
                "nearest objects are."
            ),
            data="Synthetic two-channel 3D image",
            explore=(
                "Threshold and label the objects in each channel separately.",
                "Find which labelled objects share some of the same voxels.",
                (
                    "Compare overlapping objects with their nearest neighbours and "
                    "the distances between them."
                ),
                (
                    "Use Event Localization to assign each object to the region "
                    "with its largest overlap."
                ),
            ),
            results=(
                "Two sets of object labels",
                "Association and distance tables",
            ),
            try_this=(
                "Choose an object and compare its overlap-association row with its "
                "nearest-object distance. Check whether the closest object also "
                "overlaps it."
            ),
            caution=(
                "Proximity, voxel overlap and object association describe different "
                "relationships."
            ),
        ),
        "deconvolution-2d": ExampleGuidance(
            purpose=(
                "Compare two ways to restore a blurred 2D image using its "
                "point-spread function."
            ),
            data="Synthetic blurred image and measured point-spread function (PSF)",
            explore=(
                (
                    "Inspect the prepared point-spread function (PSF), the blur "
                    "pattern used for restoration."
                ),
                (
                    "Compare ordinary Richardson–Lucy with its noise-controlling TV "
                    "(total variation) version."
                ),
                (
                    "Explore how iteration count and TV strength change the restored "
                    "image."
                ),
            ),
            results=(
                "Prepared PSF",
                "Two restored 2D images",
            ),
            try_this=(
                "Compare both results at 25 iterations, then change the TV "
                "regularisation and inspect the balance between fine detail and "
                "smoothness."
            ),
            caution=(
                "More iterations or stronger regularisation do not automatically "
                "produce a more faithful image."
            ),
        ),
        "deconvolution-3d": ExampleGuidance(
            purpose=(
                "Restore a full image volume using a matching 3D point-spread function."
            ),
            data="Synthetic 3D deconvolution volume and measured 3D PSF",
            explore=(
                (
                    "Inspect the prepared 3D point-spread function (PSF), the blur "
                    "pattern used for restoration."
                ),
                (
                    "Compare ordinary Richardson–Lucy with its noise-controlling TV "
                    "(total variation) version."
                ),
                (
                    "Move through the depth of both restored volumes to inspect more "
                    "than one slice."
                ),
            ),
            results=(
                "Prepared 3D PSF",
                "Two restored volumes",
            ),
            try_this=(
                "Move through Z and compare the original with both restored volumes. "
                "Inspect the same feature at several depths."
            ),
            caution=(
                "The image and PSF must have compatible spatial calibration for "
                "meaningful restoration."
            ),
        ),
        "per-label-skeleton": ExampleGuidance(
            purpose=(
                "Measure each labelled object's skeleton and morphology together "
                "without losing the original object identity."
            ),
            data="Synthetic 3D network with three labels and 0.45 micrometer voxels",
            explore=(
                "Compare filtered labels with skeleton labels; each object "
                "keeps its original ID.",
                "Inspect component counts and isolated nodes in the per-label "
                "summary and component detail tables.",
                "Join the measurements by label ID and plot object volume "
                "against skeleton length.",
            ),
            results=(
                "Skeleton image with original labels",
                "One skeleton summary row per original object",
                "Component detail table and merged morphology table",
                "Object volume and skeleton length scatter plot",
            ),
            try_this=(
                "Raise the minimum label volume from 1 to 2 voxels. The isolated "
                "voxel disappears from both measurement branches together."
            ),
            caution=(
                "This sparse sample demonstrates identity and graph measurements. "
                "It does not validate segmentation of acquired images. Objects "
                "within one image are not independent biological samples."
            ),
        ),
        "skeleton-qc": ExampleGuidance(
            purpose=(
                "Inspect a skeleton's branches, junctions and disconnected "
                "components before trusting its measurements."
            ),
            data="Synthetic 3D skeleton network",
            explore=(
                "Locate branch ends and junctions in the labelled skeleton views.",
                "Read the measurements for individual branches and the whole network.",
                (
                    "Compare the network before and after pruning, which removes "
                    "short branches."
                ),
            ),
            results=(
                "Keypoint and branch labels",
                "Pruned skeleton",
                "Network measurements",
            ),
            try_this=(
                "Change the minimum branch length in the pruning node. Compare the "
                "skeleton analysis before and after pruning."
            ),
            caution=(
                "The input is already skeletonised; pruning deliberately changes the "
                "network being measured."
            ),
        ),
        "advanced-skeleton": ExampleGuidance(
            purpose=(
                "Explore how loops, fragments and short spurs affect a 3D network "
                "over time."
            ),
            data="Synthetic time-indexed 3D skeleton with anisotropic calibration",
            explore=(
                (
                    "Use coloured branch overlays to locate loops, fragments and "
                    "short spurs."
                ),
                (
                    "Compare branch measurements and network summaries at different "
                    "timepoints."
                ),
                "Inspect how pruning short branches changes the original skeleton.",
            ),
            results=(
                "Graph overlays",
                "Original and pruned skeletons",
                "Time-indexed network tables",
            ),
            try_this=(
                "Choose a timepoint and compare the original and pruned overlays. "
                "Look for short spurs that disappear and loops that remain."
            ),
            caution=(
                "Lengths depend on voxel spacing, and pruning can change network "
                "connectivity."
            ),
        ),
        "mesh-objects": ExampleGuidance(
            purpose=(
                "Separate a mesh into objects, give them colours, then refine and "
                "measure their surfaces."
            ),
            data="Synthetic 3D mesh morphology sample",
            explore=(
                "Split the starting mesh into five separate objects.",
                "Colour the objects by the number of triangles making up each surface.",
                "Filter objects and combine the retained meshes.",
                "Compare the original surfaces with smoothed and simplified versions.",
            ),
            results=(
                "Coloured 3D meshes",
                "Refined surfaces",
                "Object measurements",
            ),
            try_this=(
                "Change the smoothing strength and compare the surface with the "
                "original. Then inspect the triangle count after simplification."
            ),
            caution=(
                "Smoothing and simplification change geometry, so they can also "
                "change measurements."
            ),
        ),
        "batch-provenance": ExampleGuidance(
            purpose=(
                "Practise processing several paired inputs and reviewing their saved "
                "results and run records."
            ),
            data="Three generated pairs of synthetic image files",
            explore=(
                "Check that the paired input files are matched correctly.",
                "Review output names and overwrite warnings before running.",
                (
                    "Run the three-item batch to create image, label and measurement "
                    "files."
                ),
                (
                    "Inspect the run records and checks against the sample's known "
                    "results."
                ),
            ),
            results=(
                "Image, label and measurement files",
                "Batch settings",
                "Run records",
            ),
            try_this=(
                "Choose a folder for the demo working copy. Review the paired items "
                "in the batch workspace, run the demo, then inspect its results."
            ),
            caution=(
                "This example creates a working folder and writes result files when "
                "run."
            ),
        ),
        "exhaustive-inspector": ExampleGuidance(
            purpose=(
                "Inspect image and measurement tools in one comprehensive "
                "testing workflow."
            ),
            data=(
                "Eight synthetic samples spanning images, volumes, skeletons and "
                "deconvolution"
            ),
            explore=(
                (
                    "Browse the nine labelled lanes to find different types of "
                    "processing nodes."
                ),
                "Compare the inspector controls for images, masks, meshes and tables.",
                "Follow the canvas notes to review the saved display settings.",
            ),
            results=(
                "Images",
                "Masks and labels",
                "Meshes",
                "Measurement tables",
            ),
            try_this=(
                "Select an image node, a measurement node and a mesh node. Calculate "
                "their results and compare the available inspector sections."
            ),
            caution=(
                "This is an interface-testing collection, not one recommended "
                "analysis pipeline. Table Source needs a saved batch measurement "
                "dataset and is reviewed separately."
            ),
            chooser_category="Developer & testing workflows",
        ),
        "graph-authoring": ExampleGuidance(
            purpose="Test workflow editing without rebuilding a graph from scratch.",
            data="Synthetic object-morphology image",
            explore=(
                (
                    "Insert a node before a shared tunnel, the shortcut used by "
                    "several connections."
                ),
                "Copy settings from one node to another.",
                "Duplicate or move a group of connected nodes.",
                "Undo and redo edits using the numbered canvas checks.",
            ),
            results=(
                (
                    "Editable example branches for comparing graph changes and undo "
                    "behaviour"
                ),
            ),
            try_this=(
                "Follow the first canvas note to insert Invert before the shared "
                "tunnel, then undo the change."
            ),
            caution=(
                "The GPU-specific check requires a compatible GPU; the other editing "
                "checks do not."
            ),
            chooser_category="Developer & testing workflows",
        ),
        "responsive-crop": ExampleGuidance(
            purpose=(
                "Test interactive cropping while preserving channels, timepoints and "
                "calibration."
            ),
            data="Synthetic 3D time series with three channels",
            explore=(
                "Drag a crop margin to see the selection outline update.",
                (
                    "Check the cropped image's size and starting position in "
                    "calibrated image coordinates."
                ),
                "Test that one undo restores a completed crop change.",
            ),
            results=(
                "Cropped volume",
                "Updated spatial origin",
                "Retained time and channel axes",
            ),
            try_this=(
                "Drag a crop margin, release it to calculate, then undo once to "
                "restore the starting value."
            ),
            caution=(
                "This is an interaction test; an unidentified stack axis must not be "
                "assumed to represent depth."
            ),
            chooser_category="Developer & testing workflows",
        ),
        "safe-node-bypass": ExampleGuidance(
            purpose=(
                "Test switching a crop between active processing and unchanged "
                "pass-through."
            ),
            data="Synthetic 3D volume",
            explore=(
                "Compare Run with Bypass, which passes the input through unchanged.",
                "Check the output size in each mode.",
                "Undo and redo the mode change.",
                (
                    "Save and reopen the workflow to check that the chosen mode is "
                    "retained."
                ),
            ),
            results=("Original or cropped volume, depending on the bypass setting",),
            try_this=(
                "Select Crop Stack and turn bypass off. Compare its output "
                "dimensions, then undo to restore bypass."
            ),
            caution=(
                "Use the actual output and metadata to check what a bypassed node "
                "passes through."
            ),
            chooser_category="Developer & testing workflows",
        ),
        "general-node-bypass": ExampleGuidance(
            purpose=(
                "Test bypass behaviour for single-input and multi-input processing "
                "nodes."
            ),
            data="Synthetic blurred image and measured PSF",
            explore=(
                "Compare Gaussian Blur and Richardson–Lucy TV with bypass on and off.",
                "Check which image passes through unchanged when a node is bypassed.",
                "Follow the canvas notes to test cases where bypass is unavailable.",
            ),
            results=("Processed images or unchanged primary-input images",),
            try_this=(
                "Turn bypass off on Richardson–Lucy TV, calculate it, then restore "
                "bypass and compare the output."
            ),
            caution=(
                "Bypassed Richardson–Lucy TV forwards its Image input. The connected "
                "PSF does not affect that pass-through."
            ),
            chooser_category="Developer & testing workflows",
        ),
    }
)


__all__ = ["EXAMPLE_GUIDANCE_BY_ID", "ExampleGuidance"]
