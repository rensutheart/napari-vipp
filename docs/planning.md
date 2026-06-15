# napari-vipp Planning Notes

For the prioritized algorithm and node catalogue discussion, see
[node-roadmap.md](node-roadmap.md).

## Manual Graph Editing

The graph should behave like Blender shader nodes, Unreal Blueprints, and the
original VIPP/Ryven editor:

- nodes can be freely repositioned on a large pan/zoom canvas;
- users can create space in the graph and insert nodes between existing stages;
- ports should be visible on node edges;
- connections should be made by clicking or dragging from an output port to an
  input port;
- invalid connections should be rejected visually;
- connectors should remain attached while nodes move.

The first prototype supports moving existing node cards and keeps connector
curves attached. Dynamic add/remove/connect interactions are the next graph
editing milestone.

## Pipeline Export

Once a user has manually built and tuned a pipeline, the system should export
that pipeline as runnable Python code. The export should be suitable for:

- batch processing over files, timepoints, channels, or positions;
- sharing by email or repository;
- including in supplementary material for a paper;
- running headless without napari when only computation is needed.

The exported code should be generated from the same graph model used by the UI,
not reverse-engineered from widget state.

## Workflow Save And Load

The graph should also save to a portable JSON or YAML workflow file containing:

- node ids and stable operation ids;
- parameter values;
- input/output connections;
- node positions on the canvas;
- preview settings;
- version metadata;
- optional provenance and notes.

Loading the file in another napari-vipp installation should recreate the same
graph layout, parameters, and connections.

## Data State Visibility

Each graph stage should make the flowing image state visible, not just the
preview image. The metadata model should be OME-NGFF-inspired: each node output
carries axis names, axis types, units, scale/origin transforms, shape, dtype,
value range, and transform history. When an input layer has OME-NGFF
`multiscales` axes, VIPP should use those directly. When the source is only a
plain NumPy array, the UI may fall back to inferred axes, but must explicitly
label that fallback so the user understands that the input axes still need to be
declared or loaded from a richer source.

Type conversion should be explicit in the graph. Converter nodes can turn a
volume into `uint8`, `uint16`, `float32`, or `bool` output using either rescale,
clip, or preserve-cast behavior. This keeps downstream batch/export workflows
honest about when a 16-bit image became an 8-bit display-oriented
representation, or when an intensity image became a logical mask.

Axis subsetting should also be explicit. A generic Select Axis Slice node lets
users select a single timepoint, channel, z-slice, or other axis index and drops
that axis from downstream metadata. Future UI work should label these controls
with axis names rather than only numeric axis indices.

## Legacy VIPP Node Migration

The first Sharratt/VIPP migration pass ports the single-input image-processing
nodes into the napari-vipp operation library:

- contrast stretching, gamma correction, crop, and channel extraction;
- average blur, median blur, Gaussian blur, Gaussian blur 3D, and bilateral
  filtering;
- binary, adaptive mean, adaptive Gaussian, Otsu, and triangle thresholding;
- dilation, erosion, opening, closing, top hat, black hat, morphological
  gradient, fill holes, and volume filtering.

The remaining old VIPP nodes need broader UI/model support before they can be
represented faithfully:

- Split Channels and Merge Channels need true multi-output and multi-input
  graph nodes.
- Channel Overlap needs at least two image inputs.
- Histogram and Morphological Properties need non-image result panels/tables.
- Save Pipeline Output and Batch Process belong with workflow export and batch
  execution rather than the current single-output preview graph.

## Suggested Future Artifacts

- `workflow.json` or `workflow.yaml`: portable GUI workflow.
- `pipeline.py`: exported runnable script.
- `batch_config.yaml`: input/output paths and batch dimensions.
