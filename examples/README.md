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
- `deconvolution`
- `deconvolution-3d`

In the interactive widget, use `Open example...`; it presents these checked-in
workflows as a grouped, searchable template list and opens them with their
sample `Image Source` nodes already configured. Use `Load workflow...` for
custom or external workflow JSON files.

| Workflow | Input sample | Purpose |
| --- | --- | --- |
| `otsu-red-channel-labels.json` | `VIPP synthetic multichannel volume` | Label-cleanup review path: split the red/TRITC-like channel, blur, Otsu threshold, fill holes, connected components, clear border objects, and volume filtering. |
| `red-channel-object-intensity-measurements.json` | `VIPP synthetic multichannel volume` | Named multi-input table node review: filtered labels plus matching intensity image, carried through a `Red intensity` tunnel, into `Measure Objects + Intensity`. |
| `red-channel-merged-measurement-table.json` | `VIPP synthetic multichannel volume` | PCA-oriented table assembly path: object morphology, object intensity via `Red intensity` tunnel, table merge, and metadata columns. |
| `synthetic-measurement-summary.json` | `VIPP synthetic measurement summary` | Grouped measurement summaries with known timepoint object counts and areas. |
| `synthetic-derived-object-morphology.json` | `VIPP synthetic object morphology` | Derived 2D morphology, circularity, perimeter/area ratio, Hu moments, and checklist-based column selection. |
| `synthetic-3d-mesh-morphology.json` | `VIPP synthetic 3D mesh morphology` | True-3D mesh morphology on anisotropic objects, including surface area, mesh volume, convex hull metrics, sphericity, and tiny-object status reporting. |
| `synthetic-skeleton-qc.json` | `VIPP synthetic skeleton network` | Compact skeleton QC path using a `Skeleton mask` tunnel: keypoint masks, component/branch labels, pruning, branch tables, graph tables, and overall network summaries. |
| `synthetic-advanced-skeleton-network.json` | `VIPP synthetic advanced skeleton network` | Stress test using a `Skeleton mask` tunnel for time-indexed 3D skeleton/network analysis with loops, disconnected fragments, pruning, graph overlays, branch summaries, and anisotropic physical calibration. |
| `synthetic-colocalization-racc.json` | `VIPP synthetic colocalization` | Two-channel colocalization review path using named red/green channel tunnels: ROI mask, inspector scatter threshold guides, colocalized-voxel RGB views, Pearson/Manders metrics, and RACC index output. |
| `synthetic-object-colocalization-association.json` | `VIPP synthetic colocalization` | Object-aware colocalization and association review path using named red/green channel tunnels: thresholded channel labels, object colocalization rows, label overlap, nearest-object distances, event localization, and merged morphology/colocalization tables. |
| `synthetic-deconvolution-rl-tv.json` | `VIPP synthetic deconvolution image` plus `VIPP synthetic measured PSF` | PSF-aware restoration review path: measured PSF source, `Prepare / Validate PSF`, ordinary Richardson-Lucy, and Richardson-Lucy TV deconvolution side by side. |
| `synthetic-3d-deconvolution-rl-tv.json` | `VIPP synthetic 3D deconvolution volume` plus `VIPP synthetic 3D measured PSF` | Volumetric PSF-aware restoration review path: ZYX measured PSF source, `Prepare / Validate PSF`, ordinary 3D Richardson-Lucy, and 3D Richardson-Lucy TV deconvolution side by side. |

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
- `synthetic-3d-deconvolution-rl-tv.json`.

Together these cover object morphology, intensity per object, table merging,
metadata annotation, grouped summaries, skeleton/network measurements, 3D mesh
morphology, first-pass pixel colocalization/RACC outputs with ROI-masked
variants, named channel tunnels, object-aware colocalization/association
tables, and 2D/3D PSF-aware Richardson-Lucy/RL-TV restoration.

Graph tunnels in these examples are used as readability aids for reused sources:
`Red intensity` avoids long back-reference wires from a split channel, while
`Skeleton mask` avoids a dense fan-out from one binary skeleton mask into many
QC and measurement nodes. Dense examples also include saved graph notes and
selected-inspector metadata so manual review opens on a meaningful node.
