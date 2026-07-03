# Colocalization And RACC Plan

Last reviewed: 2026-07-03

This document tracks the VIPP colocalization phase. The goal is to support
transparent two-channel colocalization workflows first, then extend toward
per-object analysis, and deeper RACC interop.

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

## Implemented First Batches

Implemented nodes live under **Colocalization & Spatial Analysis**:

| Node | Output | Purpose |
| --- | --- | --- |
| `Colocalization Metrics` | Table | Whole-image two-channel metrics: thresholds, voxel counts, Pearson, Manders M1/M2, overlap coefficients, intensity sums, and Costes diagnostics. Manual/cached by default. |
| `Colocalized Voxels` | RGB image | Live visual threshold feedback. Display modes include white overlay on channel colours, white-only colocalized voxels, and channel-coloured colocalized voxels. |
| `RACC Index` | Image | RACC-like scalar index image using the same threshold normalization and optional Costes thresholds. Manual/cached by default. |
| `Masked Colocalization Metrics` | Table | Same metric family as `Colocalization Metrics`, restricted to a third ROI mask input. Manual/cached by default. |
| `Masked Colocalized Voxels` | RGB image | Same visual threshold feedback as `Colocalized Voxels`, but zeroes voxels outside the ROI mask. |
| `Masked RACC Index` | Image | Same RACC-like scalar image as `RACC Index`, restricted to the ROI mask. Manual/cached by default. |

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

Important policy note: the RACC plugin README includes license/patent notices.
Before copying more RACC code wholesale or making a public release that markets
RACC functionality, the release notes and documentation should explicitly state
the license/patent status and attribution.

## Next Batches

1. **Per-object colocalization**
   - Add a label-aware node with inputs: labels, channel 1, channel 2.
   - Output one table row per label object with the same metric family.
   - Use label ids and leading axes so rows merge cleanly with object
     morphology/intensity tables.

2. **Object association and localization**
   - Add label-label overlap and nearest-object distance tables.
   - Add event/puncta localization against labels, masks, or ROIs.
   - Keep these as table outputs so they can merge with object morphology and
     intensity measurements.

3. **RACC interop**
   - Decide whether to extract a small shared RACC core package or keep VIPP
     and RACC as separate packages with duplicated, attributed numerical
     routines.
   - Consider an action to send selected VIPP channel outputs to the RACC
     plugin when both plugins are installed.

4. **Publication documentation**
   - Add method definitions for Pearson, Manders, overlap coefficient, Costes,
     and RACC.
   - Add a validation notebook or scripted test figure based on the synthetic
     colocalization sample.
