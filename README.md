# napari-vipp

`napari-vipp` is an early prototype of a napari-native Visual Image Processing
Pipeline (VIPP): an interactive node-graph workflow composer for bioimage
analysis.

The project is exploring a workflow where the graph is the main work surface:
users add processing nodes, connect outputs to inputs, tune parameters, inspect
stage outputs in napari, and compare mask overlays or processing branches while
the pipeline updates live.

This is research/prototype software. The current code is useful for exploring
interaction patterns and metadata-aware pipeline execution, but it is not yet a
released production analysis package.

## Current Status

The prototype currently supports:

- a napari `npe2` plugin manifest and dock widget;
- a large pan/zoom Qt graph canvas;
- draggable node cards with input/output ports and curved connectors;
- adding nodes from a categorized, fuzzy-searchable node library;
- connecting nodes by dragging or click-to-connect from output to input ports;
- per-node thumbnails with global `Slice`, `MIP`, and `Off` preview modes;
- optional per-node thumbnail disabling for heavier workflows;
- selected-node parameter controls in the inspector;
- slider plus numeric entry controls with soft range expansion where useful;
- right-panel histograms with slice/stack and linear/log modes;
- compact node metadata plus detailed selected-node metadata;
- OME-NGFF-inspired image state propagation through the graph;
- mask pinning as a napari overlay layer;
- generated inspect layers for full-resolution napari review.

## Image Metadata

The graph carries an explicit image state object alongside each node output.
That state records:

- shape and dtype;
- axis names and axis types, such as `t`, `c`, `z`, `y`, `x`;
- units, scale, and origin/translation where available;
- value range, bit depth, memory estimate, and binary-value hints;
- source layer and operation history.

When a napari layer provides OME-NGFF-style `multiscales` metadata, VIPP reads
the axis definitions and coordinate transforms. When the source is a plain array
without reliable metadata, VIPP falls back to inferred axes and labels that
fallback explicitly.

This is currently OME-NGFF-inspired internal metadata propagation, not full
OME-Zarr import/export yet.

## Sample Data

The plugin contributes synthetic fluorescence-like sample data:

- `VIPP synthetic volume`: grayscale `ZYX` stack;
- `VIPP synthetic multichannel volume`: `CZYX` volume with three probe-like
  channels;
- `VIPP synthetic time-lapse multichannel`: `TCZYX` time-lapse, multichannel
  stack.

The multichannel samples use separate intensity channels, not baked RGB images.
Graph thumbnails render these as fluorescence-style pseudo-color composites
while preserving the underlying channel axis in the carried metadata.

Open sample data from napari:

```text
File > Open Sample > VIPP synthetic volume
```

## Node Library

The current node catalogue includes:

- Input and axis tools:
  - Crop Stack
  - Select Axis Slice
- Contrast:
  - Contrast Stretching
  - Gamma Correction
- Filtering:
  - Average Blur
  - Gaussian Blur
  - Gaussian Blur 3D
  - Median Filter
  - Bilateral Filtering
- Projection:
  - Maximum Projection
- Segmentation:
  - Otsu Threshold
  - Triangle Threshold
  - Binary Threshold
  - Adaptive Mean Threshold
  - Adaptive Gaussian Threshold
- Morphology:
  - Dilation
  - Erosion
  - Opening
  - Closing
  - Top Hat
  - Black Hat
  - Morphological Gradient
  - Fill Holes
  - Volume Filter
- Channels:
  - Extract Channel
  - Channel Composite
- Utility:
  - Convert Dtype
  - Invert

`Extract Channel` is the current practical split-channel path. `Channel
Composite` creates an RGB display-style output from a multi-channel image. True
multi-output split nodes and true multi-input merge nodes are planned but not
implemented yet.

## Development

Create a local environment and install in editable mode:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Validate and test:

```bash
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest
```

Launch napari manually and open the widget from:

```text
Plugins > VIPP Workflow (napari-vipp)
```

Or launch the local sample app:

```bash
python scripts\launch_vipp_sample.py
```

## Roadmap

Near-term development priorities:

- axis-aware channel selectors that show probe names instead of only numbers;
- true multi-input and multi-output graph nodes;
- workflow save/load to JSON or YAML;
- pipeline export to runnable Python for batch processing;
- OME-Zarr/OME-NGFF import and export support;
- richer non-image outputs, such as measurements and morphology tables;
- batch execution over files, positions, channels, z-slices, and timepoints.

See `docs/planning.md` for more detailed planning notes.
