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
- toolbar graph zoom controls with a `100%` reset plus Ctrl/trackpad wheel zoom;
- draggable node cards with input/output ports and curved connectors;
- adding nodes from a categorized, fuzzy-searchable node library;
- connecting nodes by dragging or click-to-connect from output to input ports;
- explicit image source nodes for napari layers, files, or bundled samples;
- quick selected-output saving plus graph-level save nodes;
- per-node thumbnails with global show/hide, `Slice`/`MIP` preview modes,
  contrast modes (`Percentile`, `Min-max`, `Raw`), and monochrome colormaps;
- optional per-node thumbnail disabling for heavier workflows;
- selected-node parameter controls in the inspector;
- slider plus numeric entry controls with soft range expansion where useful;
- right-panel output histograms plus cutoff-node input histograms with
  slice/stack and linear/log modes;
- compact node metadata plus detailed selected-node metadata;
- normalized axes, channel, acquisition, source, and provenance metadata;
- OME-TIFF, ImageJ TIFF, conventional TIFF, OME-Zarr 0.4/0.5, and common
  raster image import plus 2D raster export;
- adaptive image/series selection for multi-image sources;
- table outputs for object, intensity, skeleton, merged, and annotated results;
- image/mask/label pinning as persistent napari preview layers;
- generated inspect layers for full-resolution napari review.

## Image Metadata

The graph carries an explicit image state object alongside each node output.
That state records:

- shape and dtype;
- axis names and axis types, such as `t`, `c`, `z`, `y`, `x`;
- units, scale, and origin/translation where available;
- value range, bit depth, memory estimate, and binary-value hints;
- source layer and operation history.
- channel names, fluorophore/wavelength fields where available;
- acquisition and stable source identity records.

When a napari layer provides OME-NGFF-style `multiscales` metadata, VIPP reads
the axis definitions and coordinate transforms. When the source is a plain array
without reliable metadata, VIPP falls back to inferred axes and labels that
fallback explicitly.

OME-TIFF metadata and local OME-Zarr 0.4/0.5 images are supported through the
shared headless I/O layer. OME-Zarr label groups, HCS plate browsing, generated
pyramids, remote stores, and full operation-level lazy execution remain future
work. See `docs/io-user-guide.md` for the current format contract.

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
    - Reorder Axes
    - Set Pixel Size / Units
    - Rescale Axes
  - Channels & Composites:
    - Extract Channel
    - Combine Channels
    - Split Channels
    - Composite → RGB
  - Utilities:
    - Convert Dtype
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
- Intensity & Contrast:
  - Linear Scale + Offset
  - Gamma Correction
  - Rescale Intensity
  - Normalize
  - Clip
- Filtering:
  - Smoothing & Denoising:
    - Average Blur
    - Gaussian Blur
    - Gaussian Blur 3D
    - Median Filter
    - Bilateral Filtering
    - Non-Local Means
  - Edge & Detail:
    - Difference of Gaussians
    - Unsharp Mask
    - Sobel Edges
    - Canny Edges
    - Laplace Filter
- Projection:
  - Maximum Projection
  - Project Image (axis-aware dropdown plus multiple projection methods)
  - Orthogonal Projection
- Segmentation:
  - Global Thresholds:
    - Otsu Threshold
    - Triangle Threshold
    - Li Threshold
    - Yen Threshold
    - Isodata Threshold
    - Minimum Threshold
    - Binary Threshold
    - Hysteresis Threshold
  - Local Thresholds:
    - Adaptive Mean Threshold
    - Adaptive Gaussian Threshold
    - Sauvola Threshold
    - Niblack Threshold
- Morphology:
  - Dilation
  - Erosion
  - Opening
  - Closing
  - Top Hat
  - Black Hat
  - Morphological Gradient
  - Fill Holes
  - Remove Small Objects
  - Skeletonize
- Label Operations:
  - Label Connected Components
  - Filter Labels By Volume
  - Filter Labels By Property
  - Clear Border Objects
  - Relabel Sequential
- Measurements:
  - Measure Objects
  - Measure Objects + Intensity
  - Analyze Skeleton
  - Merge Tables
  - Select Table Columns
  - Add Metadata Columns

Histogram-based automatic threshold nodes show `Threshold uses` on stack inputs.
`Stack histogram` computes one cutoff from the whole grayscale input and applies
it to the full image; `Slice histogram` computes a separate cutoff per displayed
plane while still producing a full-stack mask. The control is hidden for 2D
inputs. These nodes also show the input histogram used for threshold selection
with a live marker at the chosen threshold.

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

`Filter Labels By Property` accepts named `Labels` and `Measurements table`
inputs and keeps or removes labels using any numeric table column, including
area/volume, intensity, and skeleton/network measurements. It preserves label
IDs and leaves compact renumbering to `Relabel Sequential`.

`Clear Border Objects` accepts either a binary mask or integer labels and
preserves that semantic type. In 3D it can remove objects touching all `ZYX`
volume boundaries or only the lateral `YX` boundaries, with an optional border
buffer. Timepoints and channels are processed independently.

`Fill Holes` accepts binary masks and defaults to metadata-aware processing:
2D images are filled in `YX`, while true z-stacks are filled as complete `ZYX`
volumes. An advanced 2D-per-slice mode remains available for deliberately
slice-wise segmentations. A maximum size of `0` fills every enclosed hole;
positive values fill only holes up to the selected pixel area or voxel volume.
The inspector hides 3D for true 2D inputs and warns when slice-wise filling is
selected for a z-stack.

`Remove Small Objects` accepts binary masks or integer labels and preserves the
connected semantic type. It removes objects below a minimum pixel area or voxel
volume, supports 2D or 3D processing, and exposes connectivity for mask inputs.
Its logarithmic size control is bounded by the largest observed input object.
For labels-only cleanup with both minimum and maximum cutoffs, use
`Filter Labels By Volume`.

`Measure Objects` accepts a label image and produces a table output instead of
an image. The default measurement set includes label ID, pixel/voxel area or
volume, calibrated physical area or volume when spatial scale metadata is
available, centroid, bounding box, equivalent diameter, extent, and Euler
number. Optional checkboxes add shape descriptors, axis/inertia descriptors,
and 2D boundary descriptors. The 2D boundary group is hidden for true 3D
inputs because perimeter, Crofton perimeter, orientation, and eccentricity are
2D concepts in the current implementation. Table outputs show a row preview in
the inspector and can be saved as CSV or TSV.

`Measure Objects + Intensity` is the first named multi-input measurement node.
It has separate `Labels` and `Intensity image` input ports, then outputs the
basic object morphology columns plus per-label mean, minimum, maximum, sum, and
standard deviation intensity. It exposes the same optional morphology groups as
`Measure Objects`. The example workflow
`examples/red-channel-object-intensity-measurements.json` demonstrates this
pattern.

`Merge Tables` joins two or more table outputs into a single table. In `auto`
mode it joins on stable identity columns such as `t_index` and `label_id`; when
no identity columns are shared, equal-length tables can be joined by row
position. `Select Table Columns` keeps, drops, or reorders table columns while
preserving row order and column units. `Add Metadata Columns` appends constant
treatment, replicate, batch, or condition columns before CSV/TSV export. The
example workflow
`examples/red-channel-merged-measurement-table.json` demonstrates a
PCA-oriented table assembly path.

`Skeletonize` accepts a binary mask and produces a skeleton mask using
metadata-aware 2D or 3D processing. `Analyze Skeleton` accepts a skeleton mask
and outputs a per-component table with skeleton voxel count, endpoint voxels,
junction voxels, isolated nodes, branch/graph edge counts, voxel-graph edge
count, cycle count, per-block component count, component voxel fraction, and
skeleton length in pixel/voxel and physical units when scale metadata is
available. These nodes are generic and are intended for mitochondria, neurites,
vessels, fibers, hyphae, and other curvilinear structures.

`Extract Channel` pulls one selected channel from a multichannel image.
`Split Channels` is its bulk counterpart: it emits one output port per channel
in the image (losslessly, preserving dtype), with the port count following the
true channel count. Each channel also preserves the semantic type of its input,
so splitting a threshold mask produces mask ports that connect directly to
`Label Connected Components`. `Combine Channels` is the inverse multi-input
node: set the expected channel/input count, connect that many upstream images,
and it stacks them into an explicit multichannel output. Channel pseudo-colours
are carried as metadata from OME sources, Image Source overrides, Combine
Channels, or the `Assign Channel Colors` pass-through node. `Composite → RGB`
maps a multichannel composite to a channel-last RGB image. Auto mode preserves
true RGB/RGBA inputs, and otherwise blends all channels by their carried
pseudo-colours, so yellow contributes to red and green and cyan contributes to
green and blue. Manual red/green/blue selectors remain available for forced
single-channel plane mapping. `Calculate New Image` is a multi-input image-math
node that applies comma separated weights to connected inputs and then adds an
offset.

`Mask Image` is a typed two-input node with separate `Image` and `Mask` ports.
The mask port accepts binary masks or labels, treating nonzero values as inside
the mask. Spatial masks can be broadcast over compatible image axes, including
channel-last RGB/RGBA images and common channel-first multichannel arrays, so a
`YX` or `ZYX` mask can mask all colour/channel planes without first splitting
the image.

`Image Source` can point to an existing napari layer, a local `.npy`, TIFF,
OME-Zarr, or common raster source such as PNG/JPEG/BMP/GIF/WebP, or one of the
bundled synthetic samples. `Set Pixel Size / Units` repairs missing or incorrect
input calibration by setting X/Y pixel size, optional Z step size, and the
shared physical unit carried in downstream metadata. Scale-aware nodes such as
`Orthogonal Projection` use this metadata to preserve physical proportions for
anisotropic z-stacks. `Rescale Axes` changes the sampled pixel grid along X/Y/Z
with optional X/Y aspect-ratio locking, nearest-neighbor through spline
interpolation choices, and anti-aliasing for intensity-image downsampling; it
updates physical scale metadata inversely to the requested scale factors.
`Save Image` passes data through unchanged and, when `Auto-save on update` is
set to `on`, writes the node input to disk every time the graph recomputes. For
quick interactive work, the
inspector also provides `Save selected output...` for the currently selected
node; that dialog defaults to TIFF but also allows `.npy` and PNG/JPEG-style
formats when the selected output is 2D. TIFF output is written in ImageJ
hyperstack format when axis metadata is available, and binary masks are saved
as 8-bit `0`/`255` values.

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
`examples/otsu-red-channel-labels.json`. A second review workflow,
`examples/red-channel-object-intensity-measurements.json`, demonstrates the
named `Labels` plus `Intensity image` input slots on `Measure Objects +
Intensity`.

## Roadmap

Near-term development priorities:

- axis-aware channel selectors that show probe names instead of only numbers;
- grouped table summaries by condition/time/source;
- skeleton QC feature masks, branch labels, and short-branch pruning;
- distance transforms, marker generation, and marker-controlled watershed;
- richer 3D mesh morphology and calibrated physical variants for extended
  length/shape measurements;
- fluorescence background correction;
- OME-Zarr pyramids, label colors/properties, and preview-resolution selection;
- plate/well/field browsing, remote reads, batch execution, and memory-aware
  lazy execution;
- richer non-image outputs, including points and scalar summaries;
- batch execution over files, positions, channels, z-slices, and timepoints.

See `docs/planning.md` for broader planning, `docs/io-user-guide.md` for current
I/O behavior, `docs/ome-io-plan.md` for the accepted OME architecture, and
`docs/research-and-publication.md` for the evidence and publication record.
