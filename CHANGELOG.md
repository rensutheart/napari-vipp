# Changelog

## Unreleased

- Added a first-class `labels` graph type and a new Label Operations palette
  category. Label outputs inspect and pin as napari Labels layers.
- Added `Label Connected Components`, `Filter Labels By Volume`, and
  `Relabel Sequential` nodes. They support explicit 2D/3D spatial processing,
  process leading time/channel dimensions independently, and preserve label IDs
  until relabeling is requested.
- Added `Clear Border Objects` for binary masks and integer labels. It preserves
  label IDs, supports slice-wise `YX` and volumetric `ZYX` processing, exposes
  an optional border buffer, and hides the 3D mode for true 2D inputs.
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
