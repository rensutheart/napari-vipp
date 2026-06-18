# Changelog

## Unreleased

- Added a shared headless image I/O registry used by Image Source, quick save,
  Save Image, and generated Python scripts.
- Added OME-TIFF import/export with series discovery, semantic axes, physical
  scale, channel metadata, source identity, and VIPP workflow provenance.
- Added explicit ImageJ TIFF and conventional TIFF export modes. ImageJ mode
  writes calibrated hyperstacks; conventional TIFF preserves 32-bit label IDs.
- Added local OME-Zarr 0.4/0.5 image import/export with lazy Dask-backed reads,
  semantic axes, scale, channel names, and namespaced VIPP provenance.
- Added graph-aware Export OME Analysis Dataset for a reference image plus
  OME-Zarr `image-label` groups, preserving integer label IDs and label-node
  provenance.
- OME-Zarr label groups now import as VIPP label images rather than ordinary
  intensity images.
- Image Source now adapts to multi-image files/stores with a series selector,
  OME-Zarr folder browser, source summary, and stored single-item/collection
  binding mode.
- Added toolbar graph zoom controls with a calibrated 40%-250% slider, compact
  reset icon for the `100%` default, and synced Ctrl/trackpad wheel zoom that
  can continue beyond the slider range. The `100%` default corresponds to the
  previous calibrated 125% graph size.
- Extended `ImageState` with normalized channel, acquisition, and source
  metadata, and allowed source nodes to inject reader-built state directly.
- Added source-axis tracking to image metadata so thumbnails, slice histograms,
  and current-view labels follow the correct napari sliders after nodes remove
  axes, such as `Split Channels` removing C from `TCZYX`. Napari-layer sources
  now also right-align their axes to the current viewer dimensions, and thumbnail
  refreshes listen to durable dims callbacks.
- Added first-class table outputs, CSV/TSV table saving, table metadata
  summaries, generated-script table export, and an inspector table preview.
- Table-only nodes now hide the per-node thumbnail toggle instead of showing a
  disabled image-preview option.
- Added a first-class `labels` graph type and a new Label Operations palette
  category. Label outputs inspect and pin as napari Labels layers.
- Added `Label Connected Components`, `Filter Labels By Volume`, and
  `Relabel Sequential` nodes. They support explicit 2D/3D spatial processing,
  process leading time/channel dimensions independently, and preserve label IDs
  until relabeling is requested.
- Added `Measure Objects`, which measures label images with
  `skimage.measure.regionprops_table` and outputs one row per object with label
  ID, pixel/voxel size, calibrated physical area/volume when axis scale is
  available, centroid, bounding box, equivalent diameter, extent, and Euler
  number.
- Added named typed input slots and `Measure Objects + Intensity`, which accepts
  separate label and intensity-image inputs and outputs basic object morphology
  plus per-object mean, minimum, maximum, sum, and standard deviation intensity.
- Added `Merge Tables` and `Add Metadata Columns` for assembling measurement
  branches into one PCA-ready table with object-identity joins and explicit
  treatment/replicate/batch annotations.
- Added generic skeleton/network analysis nodes. `Skeletonize` thins binary
  masks in metadata-aware 2D or 3D spatial blocks, and `Analyze Skeleton`
  outputs a per-component table with skeleton voxel count, endpoint voxels,
  junction voxels, isolated nodes, branch/graph edge counts, voxel-graph edge
  count, cycle count, connected-component context, and calibrated length when
  spatial scale metadata is available.
- Added `Clear Border Objects` for binary masks and integer labels. It preserves
  label IDs, supports all-volume or lateral-only boundaries in 3D, and exposes
  an optional data-aware border buffer.
- Expanded `Fill Holes` with metadata-aware 2D/3D processing, an advanced
  per-slice mode, maximum hole area/volume filtering, connectivity control,
  input-aware mode choices, and a warning when slice-wise filling is selected
  for a z-stack.
- Replaced the obsolete `Volume Filter` operation with `Remove Small Objects`.
  The new node accepts masks or labels, preserves their semantic type, supports
  metadata-aware 2D/3D processing and mask connectivity, and uses contextual
  logarithmic area/volume controls.
- Workflow loading now uses one strict versioned schema. Unknown operations,
  malformed records, duplicate ids, invalid positions, and dangling
  connections are rejected instead of being silently ignored.
- Label TIFF saves now preserve 32-bit integer IDs using standard TIFF because
  ImageJ TIFF does not support 32-bit integer label data.
- Made Split Channels preserve upstream image/mask/label port types so
  thresholded channels connect directly to label operations.
- Added a prebuilt Otsu red-channel labeling workflow and manual launch script.
- Made label-volume filter sliders logarithmic and data-aware, using the largest
  incoming object while preserving exact numeric entry up to the hard limit.
- Added an incoming label-volume distribution to the volume-filter inspector,
  with live minimum and enabled maximum threshold markers plus a log-scale
  toggle that defaults on.
- Detached VIPP windows now use standard top-level window controls, including a
  maximize button. Double-clicking the detached title bar toggles maximized
  state instead of re-docking the panel.
- Renamed the `Channel Composite` node to `Combine Channels` to make it the
  clear complement of channel splitting; it still stacks its connected inputs
  into a multichannel image.
- Added a generic `Split Channels` node that emits one output port per channel
  in the image (replacing the fixed three-port `Split RGB`). The split is
  lossless and preserves dtype, and the port count adjusts to the true channel
  count once the node processes an image (a grayscale image yields a single
  port). `Combine Channels` and `Split Channels` are inverse operations and sit
  next to each other in the node palette.
- Added a single configurable `Composite → RGB` display node (merging the two
  earlier RGB nodes) that maps a multichannel composite to a channel-last RGB
  image. By default the channel axis is auto-detected and channels map in order
  (0→R, 1→G, 2→B; single channel→white); the channel axis and per-plane channel
  selections can be set explicitly.
- Added true multi-output support to the graph model, canvas, persistence, and
  Python export: connections now carry a `source_port`, nodes can declare static
  `OutputSpec` ports or a dynamic `output_factory`, and downstream wires resolve
  the selected port (stale wires to removed ports are trimmed automatically).

## 0.1.0

- Initial napari plugin scaffold.
- Added prototype visual workflow widget with node thumbnails and inspect/pin
  behavior.
