# Colocalization And RACC Plan

Status: implementation record; core nodes complete, follow-up work tracked below

Last reviewed: 2026-07-10

This document tracks the VIPP colocalization phase. The goal is to support
transparent two-channel colocalization workflows, per-object/object-association
tables, and deeper RACC interop.

## Scope And Terminology

Colocalization analysis has three related but distinct use cases:

- **Unrestricted image analysis**: calculate one result for all voxels/pixels in
  two same-shaped channels.
- **Mask/ROI-restricted analysis**: calculate one result inside a binary ROI
  mask. This is useful for nuclei, cytoplasm, mitochondria-rich regions, or a
  user-drawn/segmented compartment.
- **Object-restricted analysis**: calculate one row per label object, such as
  per nucleus, per cell, or per mitochondrion. This should be implemented as a
  separate label-aware node so the table aligns with object measurement tables.

All first-pass threshold parameters are expressed in normalized `0..255`
intensity units. VIPP jointly scales the two channels into this range before
manual thresholding, Costes thresholding, scatter plotting, colocalized-voxel
display, and RACC. This keeps threshold behavior stable across `uint8`,
`uint16`, and normalized float inputs.

## Implemented Batches

Implemented nodes live under **Colocalization & Spatial Analysis**:

| Node | Output | Purpose |
| --- | --- | --- |
| `Colocalization Metrics` | Table | Whole-image two-channel metrics: thresholds, voxel counts, Pearson, Manders M1/M2, overlap coefficients, intensity sums, and Costes diagnostics. Manual/cached by default. |
| `Colocalized Voxels` | RGB image | Live visual threshold feedback. Display modes include white overlay on channel colours, white-only colocalized voxels, and channel-coloured colocalized voxels. |
| `RACC Index` | Image | RACC-like scalar index image using the same threshold normalization and optional Costes thresholds. Manual/cached by default. |
| `Masked Colocalization Metrics` | Table | Same metric family as `Colocalization Metrics`, restricted to a third ROI mask input. Manual/cached by default. |
| `Masked Colocalized Voxels` | RGB image | Same visual threshold feedback as `Colocalized Voxels`, but zeroes voxels outside the ROI mask. |
| `Masked RACC Index` | Image | Same RACC-like scalar image as `RACC Index`, restricted to the ROI mask. Manual/cached by default. |
| `Object Colocalization Metrics` | Table | Label-aware two-channel metrics, one row per object, with `label_id`, leading axis indices, thresholds, voxel counts, Pearson, Manders, overlap coefficients, intensity sums, and Costes diagnostics. Manual/cached by default. |
| `Label Overlap Association` | Table | One row per overlapping reference/target label pair with overlap voxel counts, reference/target overlap fractions, and intersection-over-union. Manual/cached by default. |
| `Nearest Object Distance` | Table | One row per reference label with the nearest target label and centroid distance in pixels, plus physical distance when axis calibration is available. Manual/cached by default. |
| `Event Localization` | Table | Event/puncta objects assigned to dominant overlapping labels, masks, or ROIs with event voxel counts and overlap fractions. Manual/cached by default. |

Selected colocalization threshold nodes show a native inspector scatter panel.
The panel renders channel-1 versus channel-2 scatter density, uses a log-count
display by default, keeps the X/Y plot area square, shows axis start/end
values, draws threshold guide lines in the selected channel colours, and lets
the user drag those guide lines to switch back to manual thresholds. The density
colormap is selectable in the inspector. When `Costes auto` is selected, the
calculated thresholds are written back into the visible threshold controls so
the decision remains transparent.

Bundled validation data:

- `VIPP synthetic colocalization`: a two-channel `CZYX` volume with a partially
  overlapping object pair, channel-specific objects, offset puncta, gradients,
  and noise.

Example workflow:

- `examples/synthetic-colocalization-racc.json`
- `examples/synthetic-object-colocalization-association.json`

## RACC Relationship

The existing RACC napari plugin at
`/Users/rensu/Dropbox/Research/SYSTEMS/RACC/RACC_Napari` has a good
separation between:

- UI-independent numerical core (`_racc.py`);
- scatter rendering (`_scatter.py`);
- napari-layer view helpers (`_views.py`);
- full widget wiring (`_widget.py`).

VIPP now implements the core numerical ideas needed for graph workflows:
joint normalization, Costes thresholds, inspector scatter-density rendering,
ROI restriction, and RACC index calculation. The RACC plugin should remain a
separate napari plugin for the focused interactive RACC workflow unless a
shared library package is later created. The current VIPP implementation should
therefore not silently vendor the entire RACC plugin UI.

Current dependency status: VIPP does not depend on the standalone RACC plugin.
RACC is not listed in `pyproject.toml`; `RACC Index` and `Masked RACC Index`
use VIPP's own implementation in `napari_vipp.core.operations`.

Important policy note: the RACC plugin README includes license/patent notices.
Before copying more RACC code wholesale or making a public release that markets
RACC functionality, the release notes and documentation should explicitly state
the license/patent status and attribution.

Publication-facing method documentation is now captured in
[colocalization-method-notes.md](colocalization-method-notes.md). It documents
the implemented Pearson, Manders, Costes, RACC-like, ROI-restricted,
object-restricted, and object-association assumptions used by VIPP.

## Next Batches

1. **RACC interop**
   - Decide whether to extract a small shared RACC core package or keep VIPP
     and RACC as separate packages with duplicated, attributed numerical
     routines.
   - Consider an action to send selected VIPP channel outputs to the RACC
     plugin when both plugins are installed.

2. **Validation figure/notebook**
   - Add a validation notebook or scripted test figure based on the synthetic
     colocalization sample when preparing manuscript figures or benchmark
     supplements.
