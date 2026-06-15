# napari-vipp Planning Notes

Last reviewed: 2026-06-15

For the prioritized algorithm and node catalogue, see
[node-roadmap.md](node-roadmap.md). For implementation details, see
[architecture.md](architecture.md).

## Current Product Direction

The graph is the primary work surface. A useful VIPP feature should form part
of a reproducible bioimage workflow, not merely add another isolated image
filter.

The target first-class workflow families are:

- nuclei and cell segmentation;
- puncta and spot analysis;
- mitochondrial object and network analysis;
- pixel-based and object-based colocalization;
- 2D images and true 3D fluorescence z-stacks.

Registration and deconvolution remain later milestones.

## Implemented Platform Capabilities

### Manual Graph Editing

The graph currently supports:

- a large pan/zoom canvas with movable node cards;
- node creation from the searchable palette;
- node and connection deletion;
- click-to-connect and drag-to-connect wiring;
- visual compatible/incompatible drop feedback;
- cycle and port-type rejection;
- slot-aware multi-input connections;
- per-port multi-output connections;
- dynamic Split Channels output counts;
- connector updates while nodes move;
- save/load of canvas positions.

Remaining graph-editor refinements include automatic insertion when dropping a
node onto an existing wire, alignment/layout tools, graph annotations, and
larger-workflow navigation aids.

### Workflow Persistence

Portable JSON workflow persistence is implemented. Version 1 stores:

- stable node and operation ids;
- parameter values;
- source and target node ids;
- target input slots and source output slots;
- canvas positions;
- workflow type and version.

Loading derives titles, categories, and port contracts from the installed node
library. Unknown operations and dangling connections are skipped so a partially
compatible workflow can still load.

Not yet stored:

- per-node thumbnail visibility;
- inspector and histogram UI state;
- graph notes or annotations;
- environment/package provenance;
- YAML format.

Image Source selections, including file paths and napari layer names, are saved
as ordinary node parameters. File paths are currently literal local paths;
workflow files do not embed input data, rebase paths, or package assets for
portable sharing.

The checked-in example
[`examples/otsu-red-channel-labels.json`](../examples/otsu-red-channel-labels.json)
demonstrates the current format.

### Python Export And Batch Execution

Python export is implemented from the same graph model used by the UI. The
generated script:

- imports pure functions from `napari_vipp.core.operations`;
- reconstructs slot-aware multi-input and multi-output routing;
- exposes `run_pipeline()`;
- exposes a folder-oriented `batch_process()` helper;
- provides a command-line entry point;
- saves terminal graph outputs.

The current exporter is headless but still requires the `napari-vipp` Python
package. It handles array outputs only, and its folder batch helper assumes one
primary image source. A richer batch UI, environment lock file, embedded
provenance, table outputs, multiple independently bound sources, and explicit
iteration over semantic axes remain future work.

### Data State Visibility

Every graph output carries an OME-NGFF-inspired `ImageState` alongside its
array. The state includes:

- shape and dtype;
- semantic axes and axis types;
- units, scale, and origin where available;
- image/mask/labels/RGB/multichannel kind;
- value and bit-depth summaries;
- source and operation history.

OME-NGFF-like `multiscales` metadata is used when available. Plain arrays fall
back to inferred axes, and the UI identifies that inference.

Type conversion and axis subsetting are explicit graph operations. The
`Convert Dtype` node exposes rescale, clip, and preserve-cast behavior.
`Select Axis Slice` supports retaining ranges and removing one or more axes
while updating metadata.

### Label Workflow

The first object-cleanup workflow is implemented:

```text
image
  -> Gaussian Blur
  -> Otsu Threshold
  -> Split Channels
  -> Label Connected Components
  -> Filter Labels By Volume
  -> cleaned labels
```

Current label support includes:

- a distinct `labels` graph type;
- napari Labels inspection and pinning;
- 2D-per-plane and true 3D connected-component labeling;
- face or full connectivity;
- minimum and optional maximum pixel/voxel-volume filtering;
- logarithmic, data-aware volume sliders;
- an incoming object-volume histogram with threshold markers;
- sequential relabeling;
- integer-preserving TIFF and NumPy saving;
- workflow persistence and Python export.

## Legacy VIPP Migration

The main single-input Sharratt/VIPP operations have been ported, including
contrast, filtering, thresholding, morphology, cropping, channel extraction,
and saving.

Broader graph capabilities have also enabled:

- Split Channels and Combine Channels;
- configurable multichannel-to-RGB display;
- image arithmetic and logical operations;
- workflow save/load and Python export;
- image and label histograms;
- first-class label cleanup.

Still requiring new platform types or UI:

- Morphological Properties and other measurements require table outputs;
- spot/peak detection benefits from points outputs;
- marker-controlled watershed requires named heterogeneous input ports;
- Channel Overlap and colocalization should produce scalar/table results rather
  than synthetic images;
- batch processing needs a dedicated UI beyond exported scripts.

## Immediate Implementation Sequence

The next small, coherent milestone is:

1. `Clear Border Objects` for mask and labels inputs.
2. `Remove Small Holes` for binary masks.
3. Calibrated physical area/volume mode using spatial scale metadata.
4. Kept/removed object counts in the volume-filter summary.

After that, implement touching-object separation:

1. named heterogeneous input ports;
2. Euclidean Distance Transform;
3. H-Maxima or local-maxima marker generation;
4. Marker-Controlled Watershed;
5. Expand Labels.

The following platform milestone is `table` outputs, `Measure Objects`, and
`Save Table`.

## Planned Artifacts

| Artifact | Status |
| --- | --- |
| `workflow.json` | Implemented |
| exported `pipeline.py` | Implemented |
| example label workflow JSON | Implemented |
| `batch_config.yaml` | Not implemented |
| measurement CSV/TSV | Blocked on table outputs |
| environment/provenance manifest | Not implemented |
