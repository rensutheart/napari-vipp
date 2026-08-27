# VIPP Example Workflows

These workflows are checked into the repository as small, deterministic review
graphs. They are intended for regression testing, manual UI review, and future
documentation screenshots.

Launch one of the named review workflows with:

```bash
python scripts/launch_vipp_intensity_workflow.py <name>
```

Current launcher IDs (run the launcher with `--list` for the authoritative
registry):

- `graph-authoring`
- `responsive-crop`
- `batch-provenance`
- `label-cleanup`
- `gpu-segmentation`
- `object-intensity`
- `merged-measurements`
- `summary-table`
- `derived-morphology`
- `mesh-morphology`
- `skeleton-qc`
- `advanced-skeleton`
- `racc-colocalization`
- `object-colocalization`
- `deconvolution-2d`
- `deconvolution-3d`

The older aliases `intensity`, `merged`, `morphology`, `mesh`, `object-coloc`,
and `deconvolution` remain accepted for compatibility.

In the interactive widget, use `Open example...`; it presents these checked-in
workflows as a grouped, searchable template list and opens them with their
sample `Image Source` nodes already configured. Use `Load workflow...` for
custom or external workflow JSON files.

| Workflow | Input sample | Purpose |
| --- | --- | --- |
| `graph-authoring-acceptance.json` | `VIPP synthetic object morphology` | Numbered, on-canvas acceptance recipe for inserting a node before a shared tunnel, copying settings between matching nodes, copying and moving a connected node group, pasting at a chosen location, checking one-step undo/redo, and using **Add conversion** to make a `uint16` Gaussian input GPU eligible on a qualified GPU setup. It opens with Auto compute intent so that qualified systems can show the real tip; CPU-only systems continue normally without it. Open it as `graph-authoring`. |
| `responsive-volume-crop-acceptance.json` | `VIPP synthetic time-lapse multichannel` | Numbered acceptance path for the responsive TCZYX Crop Stack. It verifies explicit-Z controls, an immediate constant-size crop box and current-slice outline during rapid slider movement, one committed calculation and undo after release or idle, exact T/C preservation and physical-origin shifts, draft flushing before calculation/save/export/batch/tab/close boundaries, inferred-QYX rejection, and the explained CPU assignment under Prefer GPU. The authored margins crop `(5, 3, 12, 96, 128)` to `(5, 3, 9, 87, 115)`. Open it as `responsive-crop`. |
| `synthetic-batch-provenance.json` | Ready-to-run paired NumPy demo | End-to-end batch validation with three deterministic sorted-position pairs, explicit NPY/TIFF/TSV outputs, known overlap labels and measurements, portable config/runner files, and full manifest provenance. Select it through `Open example...`, click `Open batch demo...`, choose a working-copy location, use the representative slider or preview-table rows to inspect all three paired fields throughout the graph, then click `Run demo batch` and inspect retained progress and validation in the batch workspace. |
| `otsu-red-channel-labels.json` | `VIPP synthetic multichannel volume` | Label-cleanup review path: split the red/TRITC-like channel, blur, Otsu threshold, fill holes, connected components, clear border objects, and volume filtering. |
| `synthetic-gpu-segmentation-bridge.json` | `VIPP synthetic GPU segmentation cleanup` | Portable, annotated Prefer-GPU path through Extract Channel, an exact float32 Preserve conversion, Gaussian Blur, a fixed threshold with a safe sample-specific margin, boolean Remove Small Objects and Fill Holes cleanup, and 3D Connected Components. The 22-voxel cutoff visibly removes one isolated 19-voxel speck, then Fill Holes restores 31 enclosed cavity voxels. Its notes distinguish GPU eligibility from a guarantee, explain host-first extraction versus a resident no-copy view, and limit the one-round-trip expectation to a single retained terminal output. Open it as `gpu-segmentation`; unsupported regions visibly fall back to CPU. |
| `red-channel-object-intensity-measurements.json` | `VIPP synthetic multichannel volume` | Named multi-input table node review: filtered labels plus matching intensity image, carried through a `Red intensity` tunnel, into `Measure Objects + Intensity`. |
| `red-channel-merged-measurement-table.json` | `VIPP synthetic multichannel volume` | PCA-oriented table assembly path: object morphology, object intensity via `Red intensity` tunnel, table merge, and metadata columns. |
| `synthetic-measurement-summary.json` | `VIPP synthetic measurement summary` | Grouped measurement summaries with known timepoint object counts and areas. |
| `synthetic-derived-object-morphology.json` | `VIPP synthetic object morphology` | Derived 2D morphology, circularity, perimeter/area ratio, Hu moments, and checklist-based column selection. |
| `synthetic-3d-mesh-morphology.json` | `VIPP synthetic 3D mesh morphology` | True-3D mesh morphology on anisotropic objects, including surface area, mesh volume, convex hull metrics, sphericity, and tiny-object status reporting. |
| `synthetic-skeleton-qc.json` | `VIPP synthetic skeleton network` | Compact skeleton QC path using a `Skeleton mask` tunnel: keypoint masks, component/branch labels, pruning, branch tables, graph tables, and overall network summaries. |
| `synthetic-advanced-skeleton-network.json` | `VIPP synthetic advanced skeleton network` | Stress test using a `Skeleton mask` tunnel for time-indexed 3D skeleton/network analysis with loops, disconnected fragments, pruning, graph overlays, branch summaries, and anisotropic physical calibration. |
| `synthetic-colocalization-racc.json` | `VIPP synthetic colocalization` | Two-channel colocalization review path using named red/green channel tunnels: ROI mask, inspector scatter threshold guides, colocalized-voxel RGB views, Pearson/Manders metrics, and RACC index output. |
| `synthetic-object-colocalization-association.json` | `VIPP synthetic colocalization` | Object-aware colocalization and association review path using named red/green channel tunnels: thresholded channel labels, object colocalization rows, label overlap, nearest-object distances, event localization, and merged morphology/colocalization tables. |
| `synthetic-deconvolution-rl-tv.json` | `VIPP synthetic deconvolution image` plus `VIPP synthetic measured PSF` | PSF-aware restoration review path with ordinary RL and RL-TV side by side at 25 iterations. RL-TV uses the conservative production-like `0.002` regularization; compare with zero before increasing it. |
| `synthetic-3d-deconvolution-rl-tv.json` | `VIPP synthetic 3D deconvolution volume` plus `VIPP synthetic 3D measured PSF` | Volumetric PSF-aware review path with one shared, visible `float32` Preserve conversion feeding matched 25-iteration RL/RL-TV branches, a matched ZYX PSF, the authored `1e-12` filter epsilon, and conservative `0.002` TV regularization. The conversion does not rescale intensity. GPU agreement is a backend check, not proof that the PSF, iteration count, or restored structures are scientifically valid. |

## Validation Expectations

The repository test suite loads and runs every workflow above. When adding a new
example workflow, also add:

- a deterministic bundled sample or a clearly documented existing sample;
- a row in this file;
- a focused assertion in `test_example_workflow.py` that checks the expected
  output type and at least one biologically meaningful invariant;
- a launcher shortcut when the workflow is meant for frequent manual review.

## Measurement Phase Examples

The current measurement/morphology phase is represented by:

- `red-channel-object-intensity-measurements.json`;
- `red-channel-merged-measurement-table.json`;
- `synthetic-measurement-summary.json`;
- `synthetic-derived-object-morphology.json`;
- `synthetic-3d-mesh-morphology.json`;
- `synthetic-skeleton-qc.json`;
- `synthetic-advanced-skeleton-network.json`;
- `synthetic-colocalization-racc.json`;
- `synthetic-object-colocalization-association.json`;
- `synthetic-deconvolution-rl-tv.json`;
- `synthetic-3d-deconvolution-rl-tv.json`;
- `synthetic-batch-provenance.json`.

Together these cover object morphology, intensity per object, table merging,
metadata annotation, grouped summaries, skeleton/network measurements, 3D mesh
morphology, first-pass pixel colocalization/RACC outputs with ROI-masked
variants, named channel tunnels, object-aware colocalization/association
tables, and 2D/3D PSF-aware Richardson-Lucy/RL-TV restoration.

The batch example is different from the layer-backed examples. VIPP generates
a portable directory containing two input collections, the workflow, config,
thin runner, exact ground truth, and an empty results folder. The input names
intentionally differ between folders to prove that pairing follows sorted
position. Its expected overlap object counts are 1, 2, and 0 across the three
items.

Graph tunnels in these examples are used as readability aids for reused sources:
`Red intensity` avoids long back-reference wires from a split channel, while
`Skeleton mask` avoids a dense fan-out from one binary skeleton mask into many
QC and measurement nodes. Dense examples also include saved graph notes and
selected-inspector metadata so manual review opens on a meaningful node.
