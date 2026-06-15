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
- explicit image source nodes for napari layers, files, or bundled samples;
- quick selected-output saving plus graph-level save nodes;
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
When the full sample suite is open, the workflow automatically starts from the
`VIPP synthetic time-lapse multichannel` layer so the input metadata should read
as `TCZYX`. The simpler grayscale and `CZYX` examples are still available in the
toolbar input selector and in the graph-level `Image Source` node.

Open sample data from napari:

```text
File > Open Sample > VIPP synthetic microscopy samples
```

## Node Library

The current node catalogue includes:

- Image Data:
  - Source & Output:
    - Image Source
    - Save Image
  - Axes & Regions:
    - Crop Stack
    - Select Axis Slice
  - Channels & Composites:
    - Extract Channel
    - Combine Channels
    - Split Channels
    - Composite → RGB
  - Type & Scaling:
    - Convert Dtype
    - Rescale Intensity
    - Normalize
    - Clip
  - Math & Logic:
    - Calculate New Image
    - Add
    - Subtract
    - Ratio
    - Mask Image
    - Logical AND
    - Logical OR
    - Logical XOR
    - Invert
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
- Label Operations:
  - Label Connected Components
  - Filter Labels By Volume
  - Clear Border Objects
  - Relabel Sequential

The label pipeline converts binary masks into integer object IDs. Connected
components can run over full `ZYX` volumes or independently over `YX` images.
Volume filtering currently uses pixel/voxel counts, preserves retained IDs, and
leaves compact renumbering to the explicit `Relabel Sequential` node. Its
volume sliders use the largest observed object as their data-aware upper bound
and a logarithmic scale for useful control across small and large structures;
the numeric fields still accept exact values up to one billion. Selecting
`Filter Labels By Volume` also shows the incoming object-volume distribution
above the regular histogram. Dashed minimum and enabled maximum markers update
with the filter controls. Its `Log volume axis` toggle is enabled by
default and can be switched off for a linear distribution.

`Clear Border Objects` accepts either a binary mask or integer labels and
preserves that semantic type. It can remove edge objects independently from
each `YX` slice or from the complete `ZYX` volume, with an optional border
buffer. For true 2D inputs, the inspector omits the invalid 3D choice.

`Extract Channel` pulls one selected channel from a multichannel image.
`Split Channels` is its bulk counterpart: it emits one output port per channel
in the image (losslessly, preserving dtype), with the port count following the
true channel count. Each channel also preserves the semantic type of its input,
so splitting a threshold mask produces mask ports that connect directly to
`Label Connected Components`. `Combine Channels` is the inverse multi-input
node: set the expected channel/input count, connect that many upstream images,
and it stacks them into an explicit multichannel output. `Composite → RGB` is
a configurable display node that maps a multichannel composite to a
channel-last RGB image — auto-detecting the channel axis by default, with
optional manual axis and per-plane channel selection. `Calculate New Image` is
a multi-input image-math node that applies comma separated weights to connected
inputs and then adds an offset.

`Image Source` can point to an existing napari layer, a local `.npy` or TIFF
file, or one of the bundled synthetic samples. `Save Image` passes data through
unchanged and, when `Auto-save on update` is set to `on`, writes the node input
to disk every time the graph recomputes. For quick interactive work, the
inspector also provides `Save selected output...` for the currently selected
node; that dialog defaults to TIFF but still allows `.npy`. TIFF output is
written in ImageJ hyperstack format when axis metadata is available, and binary
masks are saved as 8-bit `0`/`255` values.

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

To start directly from the multichannel Otsu-to-label workflow:

```bash
python scripts\launch_vipp_label_workflow.py
```

The same graph can be loaded manually from
`examples/otsu-red-channel-labels.json`.

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
