# VIPP Example Workflows

These workflows are checked into the repository as small, deterministic review
graphs. They are intended for regression testing, manual UI review, and future
documentation screenshots.

Launch one of the named review workflows with:

```bash
python scripts/launch_vipp_intensity_workflow.py <name>
```

Current launcher names:

- `intensity`
- `merged`
- `morphology`
- `mesh`
- `object-coloc`

Other workflows can be loaded from the VIPP toolbar with `Load workflow...`.

| Workflow | Input sample | Purpose |
| --- | --- | --- |
| `otsu-red-channel-labels.json` | `VIPP synthetic multichannel volume` | Label-cleanup review path: blur, Otsu threshold, split channel mask, fill holes, connected components, clear border objects, and volume filtering. |
| `red-channel-object-intensity-measurements.json` | `VIPP synthetic multichannel volume` | Named multi-input table node review: filtered labels plus matching intensity image into `Measure Objects + Intensity`. |
| `red-channel-merged-measurement-table.json` | `VIPP synthetic multichannel volume` | PCA-oriented table assembly path: object morphology, object intensity, table merge, and metadata columns. |
| `synthetic-measurement-summary.json` | `VIPP synthetic measurement summary` | Grouped measurement summaries with known timepoint object counts and areas. |
| `synthetic-derived-object-morphology.json` | `VIPP synthetic object morphology` | Derived 2D morphology, circularity, perimeter/area ratio, Hu moments, and checklist-based column selection. |
| `synthetic-3d-mesh-morphology.json` | `VIPP synthetic 3D mesh morphology` | True-3D mesh morphology on anisotropic objects, including surface area, mesh volume, convex hull metrics, sphericity, and tiny-object status reporting. |
| `synthetic-skeleton-qc.json` | `VIPP synthetic skeleton network` | Compact skeleton QC path: keypoint masks, component/branch labels, pruning, branch tables, graph tables, and overall network summaries. |
| `synthetic-advanced-skeleton-network.json` | `VIPP synthetic advanced skeleton network` | Stress test for time-indexed 3D skeleton/network analysis with loops, disconnected fragments, pruning, graph overlays, branch summaries, and anisotropic physical calibration. |
| `synthetic-colocalization-racc.json` | `VIPP synthetic colocalization` | Two-channel colocalization review path using named red/green channel tunnels: ROI mask, inspector scatter threshold guides, colocalized-voxel RGB views, Pearson/Manders metrics, and RACC index output. |
| `synthetic-object-colocalization-association.json` | `VIPP synthetic colocalization` | Object-aware colocalization and association review path using named red/green channel tunnels: thresholded channel labels, object colocalization rows, label overlap, nearest-object distances, event localization, and merged morphology/colocalization tables. |

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
- `synthetic-object-colocalization-association.json`.

Together these cover object morphology, intensity per object, table merging,
metadata annotation, grouped summaries, skeleton/network measurements, 3D mesh
morphology, first-pass pixel colocalization/RACC outputs with ROI-masked
variants, named channel tunnels, and object-aware colocalization/association
tables.
