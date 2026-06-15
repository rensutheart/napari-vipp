# Bioimage Node Roadmap

Status: working discussion document  
Last reviewed: 2026-06-15

This document prioritizes future VIPP nodes for bioimage analysis. It is not a
commitment to reproduce every function in OpenCV, scikit-image, or SciPy.
Instead, the goal is to support complete, understandable analysis workflows
while preserving image axes, physical scale, data type, and provenance.

The current recommendation is to build a strong classical segmentation and
measurement workflow before adding a large catalogue of specialized filters.

## Confirmed Product Scope

The initial bioimage scope should support all of these workflow families:

- nuclei and cell segmentation;
- puncta and spot detection;
- mitochondrial morphology and network analysis;
- colocalization analysis.

Spatial processing must support both:

- true 3D fluorescence `ZYX` volumes, including anisotropic voxel spacing;
- 2D images, including flow-cytometry imaging data.

Registration and deconvolution remain important later milestones, but they are
not part of the current implementation focus.

The immediate focus is:

```text
binary mask
  -> label structures
  -> remove/filter labeled structures by volume
  -> preserve cleaned integer labels for inspection and later measurement
```

Implementation status:

- implemented: first-class `labels` graph/display type;
- implemented: Label Connected Components with 2D/3D spatial modes and
  configurable face/full connectivity;
- implemented: Filter Labels By Volume using pixel/voxel counts while
  preserving retained IDs, with a data-aware slider and input-volume
  distribution;
- implemented: Clear Border Objects for masks and labels, with a border buffer
  and axis-aware 2D/3D processing;
- implemented: Relabel Sequential;
- next: enhance Fill Holes with size-limited and 2D/3D processing, add
  calibrated physical area/volume, and support richer property-based label
  filtering.

## Priority Definitions

| Priority | Meaning |
| --- | --- |
| P0 | Platform capability required by several important nodes |
| P1 | Implement next; high value for common bioimage workflows |
| P2 | Useful after the core segmentation and measurement path works |
| P3 | Specialized, expensive, or dependent on additional product decisions |
| Defer | Do not implement now unless a concrete workflow requires it |

## Decision Principles

1. Prefer complete workflows over library feature parity.
2. Treat `T`, `C`, `Z`, `Y`, and `X` axes explicitly.
3. Make slice-wise versus volumetric processing a visible choice.
4. Use physical spacing when an operation represents distance, size, or volume.
5. Keep binary masks, integer labels, intensity images, tables, and points
   semantically distinct.
6. Prefer SciPy and scikit-image while they cover the requirement. OpenCV
   should become a dependency only for a demonstrated capability or performance
   gap.
7. Every node must remain deterministic, testable headlessly, serializable in a
   workflow, and exportable to Python.
8. Avoid several nearly identical nodes when one well-designed node with a
   method selector is clearer.

## Current Coverage

VIPP already has useful coverage in these areas:

- image sources, saves, workflow persistence, and Python export;
- channel extraction, splitting, combination, and RGB display conversion;
- crop and axis slicing;
- dtype conversion, normalization, clipping, and intensity rescaling;
- image arithmetic, masking, and logical operations;
- Gaussian, mean, median, and bilateral filtering;
- fixed, Otsu, triangle, and local thresholding;
- binary erosion, dilation, opening, closing, fill holes, and small-component
  removal;
- maximum projection;
- image metadata, thumbnails, intensity histograms, label-volume histograms,
  mask inspection, and label inspection;
- connected-component labels, pixel/voxel-volume filtering, and sequential
  relabeling.

The remaining object-analysis gap is measurement and difficult segmentation.
VIPP can now label separated foreground objects and clean them by size, but it
cannot yet:

- remove partial objects at image or ROI borders;
- separate touching objects with distance markers and watershed;
- measure morphology or per-object intensity into a table;
- filter labels by properties other than pixel/voxel volume;
- use calibrated physical area/volume as a filter unit.

## Existing Nodes To Clarify

Before growing the catalogue, several current names or contracts should be
cleaned up.

| Current node | Issue | Recommended direction |
| --- | --- | --- |
| Contrast Stretching | It currently performs scale plus offset and converts to `uint8`; true percentile rescaling already exists separately. | Rename to `Linear Scale + Offset`, preserve a sensible dtype, or deprecate it in favor of image math plus `Rescale Intensity`. |
| Top Hat / Black Hat | These are binary operations, while grayscale white top-hat is a common bioimage background-removal tool. | Rename current nodes `Binary Top Hat` and `Binary Black Hat`; add explicit grayscale morphology later. |
| Volume Filter | It removes small connected foreground components. The name does not reveal that behavior or connectivity. | Rename to `Remove Small Objects`; expose connectivity and physical-size semantics. |
| Fill Holes | It fills all holes across the full n-dimensional input and has no spatial-scope or size controls. | Keep the familiar name, add a maximum hole area/volume, and expose input-aware `2D YX` versus `3D ZYX` processing. |
| Maximum Projection | Axis is numeric and only the maximum reducer is available. | Replace or complement it with an axis-aware `Reduce Axis` node. |
| 2D filtering behavior | Several nodes infer XY and process other axes plane by plane. | Add an explicit processing scope: `XY per plane` or `spatial volume`. |

## P0: Platform Foundations

These are not glamorous nodes, but they prevent important algorithms from being
implemented with misleading types or awkward parameters.

### First-Class Label Images: Implemented

The `labels` port/output type represents non-negative integer object IDs where
`0` is background. Labels inspect and pin as napari Labels layers and are not
treated as ordinary intensity data.

Implemented behavior:

- labels connect to label-aware nodes and generic array nodes;
- labels do not connect silently to binary-mask-only nodes;
- connected-component and filter operations record their parameters in
  operation history;
- 2D-per-plane versus volumetric processing is explicit and exportable;
- save and inspect actions preserve integer IDs.

Nearest-neighbor interpolation remains a requirement for future geometry and
registration nodes that accept labels.

### Named Heterogeneous Input Ports: Not Implemented

The current operation model gives all inputs one type. Marker-controlled
watershed needs ports such as:

| Port | Type |
| --- | --- |
| Elevation | image |
| Markers | labels |
| Mask | mask, optional |

Add an `InputSpec` equivalent to `OutputSpec`, including name, type, title,
optional/required state, and stable slot identity.

### Table Outputs: Not Implemented

Add a `table` output type before object measurement nodes. A table should have:

- stable column names and units;
- one row per object or observation;
- an inspector table view;
- CSV/TSV save support;
- workflow/export support;
- optional linkage from a `label` column back to the displayed Labels layer.

A dictionary of one-dimensional NumPy arrays is a reasonable headless core
representation. Pandas should not be required merely to compute measurements.

### Spatial Scope And Units: Partially Implemented

Label operations currently expose `Auto from axes`, `2D YX`, and `3D ZYX`.
The broader shared contract should become:

- `XY per plane`: repeat independently over non-spatial dimensions;
- `spatial volume`: operate over the recognized spatial axes;
- size units: `pixels/voxels` initially, with `physical units` enabled when
  axis scale metadata is reliable;
- connectivity: explicit and dimension-aware;
- anisotropic spacing: passed to distance, expansion, and measurement functions
  where supported.

### Points Output, Later In P0: Not Implemented

Spot detection and peak finding naturally produce coordinates. A first-class
`points` type should eventually map to a napari Points layer and a table. It is
not required for the first label/watershed milestone because maxima can
initially be emitted as a marker mask or marker-label image.

## P1: Implement Next

### P1A: Binary Masks And Labels

Implement the remaining nodes in the order shown by the planned statuses.

| Status | Node | Input -> Output | Suggested backend | Why it matters |
| --- | --- | --- | --- | --- |
| Implemented | Label Connected Components | mask -> labels | `scipy.ndimage.label` | Converts segmentation masks into distinct objects. |
| Implemented | Filter Labels By Volume | labels -> labels | NumPy label counts | Removes labels outside minimum and optional maximum pixel/voxel volume. |
| Implemented | Relabel Sequential | labels -> labels | `skimage.segmentation.relabel_sequential` | Normalizes sparse IDs after filtering. |
| Implemented | Clear Border Objects | mask/labels -> same semantic type | `skimage.segmentation.clear_border` | Removes partial objects touching image or ROI boundaries. |
| Next | Enhance Fill Holes | mask -> mask | `scipy.ndimage.binary_fill_holes` and `skimage.morphology.remove_small_holes` | Adds size-limited and explicit slice-wise/volumetric filling without introducing a duplicate node. |
| Planned 2 | Remove Small Objects | mask/labels -> same semantic type | `skimage.morphology.remove_small_objects` | Standard cleanup with clear naming and connectivity. |
| Planned 3 | Euclidean Distance Transform | mask -> image | `scipy.ndimage.distance_transform_edt` | Foundation for separating touching objects and measuring thickness. |
| Planned 4 | H-Maxima / Local Maxima Markers | image -> mask or labels | `skimage.morphology.h_maxima` or `local_maxima` | Produces robust watershed seeds. |
| Planned 5 | Marker-Controlled Watershed | image + labels + optional mask -> labels | `skimage.segmentation.watershed` | Core method for separating touching nuclei, cells, and particles. |
| Planned 6 | Expand Labels | labels -> labels | `skimage.segmentation.expand_labels` | Approximates cell regions from nuclear seeds without label overlap. |
| Planned 7 | Find Label Boundaries | labels -> mask | `skimage.segmentation.find_boundaries` | Useful for QC, overlays, and boundary measurements. |

`Volume Filter` can be retained as a workflow-compatible alias while
`Remove Small Objects` becomes the preferred UI name.

`Filter Labels By Volume` is also implemented as the label-preserving cleanup
operation. It supports both minimum and optional maximum size, a data-aware
logarithmic slider, and a volume-distribution inspector.

### What These Segmentation Terms Mean

#### Connected Components

Connected-component labeling takes a binary mask and assigns a distinct
positive integer to each spatially connected foreground structure:

```text
binary mask values: 0, 1
label image values: 0, 1, 2, 3, ...
```

`0` remains background. In 3D, connectivity is evaluated across the `Z`, `Y`,
and `X` spatial axes, so one nucleus spanning several z-slices receives one
label. In 2D, connectivity is evaluated across `Y` and `X`.

Connectivity must be configurable:

- 2D: edge-only neighbors (4-connectivity) or edge/corner neighbors
  (8-connectivity);
- 3D: face-only neighbors (6-connectivity), or progressively more permissive
  edge/corner connectivity (18 or 26).

This is the required first step for volume-based object cleanup because volumes
belong to objects, not to an undifferentiated boolean foreground.

#### Distance Transform

For each foreground pixel or voxel, a Euclidean distance transform reports the
distance to the nearest background. Values are small near object boundaries and
largest near object centers.

For a roughly round nucleus, the distance map resembles a hill whose summit is
near the nucleus center. For true 3D data, physical `Z/Y/X` spacing should be
used so distance is not distorted by thick z-slices.

Distance transforms are not required merely to label already-separated
objects. They become important when:

- touching nuclei or cells form one connected component;
- approximate local thickness is useful;
- watershed markers need to be generated from object interiors.

#### Marker Generation

Markers are seed labels identifying where individual objects are believed to
be. They are commonly generated from robust local maxima in a distance map.

For two touching nuclei:

1. thresholding may produce one merged binary component;
2. the distance map may contain two interior peaks;
3. marker generation converts those peaks into two seed labels;
4. watershed grows the two seeds through the merged mask until they meet;
5. the result is two labeled nuclei.

`H-Maxima` suppresses weak/shallow peaks and is generally more controllable than
using every raw local maximum, which often creates too many watershed objects.

Markers are therefore an optional splitting mechanism, not a prerequisite for
ordinary connected-component labeling.

### P1B: Background And Feature Enhancement

| Order | Node | Input -> Output | Suggested backend | Notes |
| --- | --- | --- | --- | --- |
| 1 | Rolling-Ball Background | image -> background image | `skimage.restoration.rolling_ball` | High value for uneven fluorescence background. Default to per-plane processing; warn that large 3D radii are expensive. |
| 2 | Subtract Background | image + background -> image | Existing subtraction or a named wrapper | Keeping the background visible makes the workflow auditable. |
| 3 | Grayscale White Top-Hat | image -> image | `scipy.ndimage.white_tophat` | Faster approximate background suppression for bright objects. |
| 4 | Difference of Gaussians | image -> image | `skimage.filters.difference_of_gaussians` | Enhances puncta and structures within a size band. |
| 5 | Laplacian of Gaussian | image -> image | `scipy.ndimage.gaussian_laplace` | Useful for blob enhancement and marker generation. |
| 6 | Unsharp Mask | image -> image | `skimage.filters.unsharp_mask` | General sharpening; useful but less central than background correction. |

Rolling-ball background estimation should probably remain separate from
subtraction. This exposes the estimated background and allows users to inspect,
save, or reuse it. A later convenience node may return both `background` and
`corrected` outputs.

### P1C: Thresholding Without Node Explosion

Add one `Automatic Threshold` node with a method selector:

- Otsu;
- Li;
- Yen;
- Isodata;
- Triangle;
- Mean.

Keep existing named nodes loadable for workflow compatibility. New workflows
should use the consolidated node unless a named node improves readability.

Add `Multi-Otsu Classes` separately because its output is not a binary mask. It
should output an integer class image and optionally one mask output per class.

Sauvola and Niblack thresholding are useful for uneven brightfield or document-
like images, but are P2 unless histology and brightfield become primary targets.

### P1D: Axis Reduction

Add a `Reduce Axis` node with:

- axis selected by semantic name;
- method: maximum, minimum, mean, sum, median, or standard deviation;
- keep/remove reduced axis;
- output metadata with the reduced axis handled correctly.

This replaces a family of nearly identical projection nodes and supports common
z-stack, time-series, and channel reductions.

### P1E: Object Measurements

This follows table support and label images.

| Order | Node | Inputs -> Outputs | Suggested backend |
| --- | --- | --- | --- |
| 1 | Measure Objects | labels + optional intensity image -> table | `skimage.measure.regionprops_table` |
| 2 | Filter Labels By Property | labels + table, or labels + intensity -> labels | region properties plus label remapping |
| 3 | Save Table | table -> table | CSV/TSV writer |
| 4 | Summarize Measurements | table -> table/scalars | NumPy/SciPy statistics |

The first `Measure Objects` property set should be intentionally small:

- label ID;
- pixel/voxel count and physical area/volume;
- centroid;
- bounding box;
- mean, minimum, maximum, sum, and standard deviation of intensity;
- equivalent diameter;
- extent and solidity where dimensionally valid.

Properties such as eccentricity, orientation, perimeter, and Feret diameter
need dimension-specific UI labels and should not be presented as universally
meaningful in 3D.

## Next Implementation Recommendation

Enhance `Fill Holes` next. `Clear Border Objects` is implemented with the
planned mask/label type preservation, `Auto`/2D/3D processing, independent
leading-axis blocks, retained label IDs, and a configurable border buffer.
Removed/retained object counts remain a possible inspector enhancement.

After Fill Holes, add calibrated physical area/volume mode. Touching-object
separation should follow as a coordinated platform milestone because watershed
needs heterogeneous named input ports.

Recommended `Fill Holes` contract:

- retain the node name `Fill Holes`;
- input/output: `mask -> mask`;
- maximum hole size: `0` fills all enclosed holes for backward compatibility;
- a positive maximum fills only holes up to that many pixels or voxels;
- spatial mode: `Auto from axes`, `2D YX`, or `3D ZYX`;
- `2D YX` fills each XY slice independently;
- `3D ZYX` fills cavities in the complete spatial volume;
- process leading time and channel axes independently;
- when the connected input has only two spatial axes, omit or disable `3D ZYX`
  rather than presenting an invalid option;
- Auto mode resolves to 2D for 2D images and to 3D for true z-stacks.

## Recommended First Milestone

The first cohesive feature milestone should be split into a required label
cleanup path and an optional touching-object separation path.

### Milestone 1A: Labels And Volume Cleanup

Completed:

1. `labels` type and napari Labels inspection.
2. Label Connected Components.
3. Filter Labels By Volume.
4. Clear Border Objects.
5. Relabel Sequential.

Remaining:

1. Enhance Fill Holes for the pre-label binary-mask path.
2. Calibrated physical area/volume filtering.
3. Explicit kept/removed object counts.

The primary pipeline is:

```text
Image
  -> threshold / segmentation mask
  -> binary mask cleanup
  -> Label Connected Components
  -> Filter Labels By Volume
  -> Relabel Sequential
  -> cleaned Labels
```

`Filter Labels By Volume` should:

- accept integer labels without merging neighboring labels: implemented;
- support minimum and optional maximum volume: implemented;
- offer voxel/pixel count immediately: implemented;
- offer calibrated physical area/volume when trustworthy scale and units exist;
- preserve retained label IDs until an explicit `Relabel Sequential` node:
  implemented;
- report how many labels were kept and removed;
- operate over full `ZYX` volumes for 3D data and `YX` for 2D data:
  implemented;
- later expand to filtering by morphological and intensity properties.

The inspector already reports the incoming object count, median, and largest
volume and displays the input volume distribution with live thresholds. It does
not yet report kept and removed counts separately.

The current `Volume Filter` does not meet this contract: it converts its input
to a boolean mask, labels internally, removes small components, and returns a
boolean mask. It should remain loadable for compatibility but should not be the
long-term object-filter node.

### Milestone 1B: Split Touching Objects

1. Named heterogeneous input ports.
2. Euclidean Distance Transform.
3. H-Maxima Markers.
4. Marker-Controlled Watershed.
5. Expand Labels.

This optional branch is inserted when thresholded structures touch:

```text
binary mask
  -> Distance transform
  -> H-maxima markers
  -> Watershed
  -> Filter Labels By Volume
  -> cleaned Labels
```

The next milestone should add `table`, `Measure Objects`, and `Save Table`.

## P2: Add After Core Object Analysis

### Denoising And Restoration

| Node | Suggested backend | Reason for P2 |
| --- | --- | --- |
| Total Variation Denoise | `skimage.restoration.denoise_tv_chambolle` | Useful nD denoising, but parameter meaning needs clear documentation. |
| Non-Local Means Denoise | `skimage.restoration.denoise_nl_means` | Powerful but computationally expensive and parameter-heavy. |
| Wavelet Denoise | `skimage.restoration.denoise_wavelet` | Useful for fluorescence noise, but dtype and channel semantics need care. |
| Estimate Noise Sigma | `skimage.restoration.estimate_sigma` | Helps configure denoisers and can output a scalar/table value. |
| Richardson-Lucy Deconvolution | `skimage.restoration.richardson_lucy` | Important for microscopy, but requires a PSF input or a defensible PSF generator. |
| Wiener Deconvolution | `skimage.restoration.wiener` | Useful once PSF handling is established. |

Deconvolution should not be shipped as a single sigma slider. The user must be
able to inspect or supply the point-spread function, understand iteration count,
and preserve physical spacing.

### Registration And Drift Correction

| Node | Suggested backend | Notes |
| --- | --- | --- |
| Estimate Translation | `skimage.registration.phase_cross_correlation` | Two images in, transform/shift result out. |
| Apply Translation | `scipy.ndimage.shift` | Interpolation must depend on image versus labels. |
| Register Stack To Reference | repeated phase cross-correlation | Useful for time-lapse drift correction. |
| Affine Transform | `scipy.ndimage.affine_transform` or `skimage.transform.warp` | Requires a first-class transform representation. |

Registration is high-value, but a transform type and interpolation policy should
be designed before exposing several registration algorithms.

### Geometry And Sampling

- resample to target pixel/voxel spacing;
- resize by shape or scale;
- pad/crop to a target shape;
- flip and rotate by right angles;
- arbitrary rotation;
- isotropic resampling for 3D visualization or analysis.

These operations must update scale, translation, and axes correctly. Label
inputs require nearest-neighbor interpolation.

### Shape And Structure

- skeletonize / skeletonize 3D;
- medial axis;
- prune short skeleton branches;
- grayscale erosion, dilation, opening, and closing;
- label erosion and label-safe expansion;
- convex hull per object;
- object boundary distance or thickness.

### Domain-Specific Filters

- Frangi vesselness;
- Sato tubeness;
- Meijering neuriteness;
- Hessian-based ridge filters;
- Gabor filters.

These are valuable for vessels, fibers, and neurites, but should follow a clear
target workflow because their scale parameters and polarity options are easy to
misuse.

### Spot And Blob Detection

- Blob LoG;
- Blob DoG;
- peak local maximum;
- spot intensity measurement;
- count spots per labeled object.

These become much cleaner after `points` and `table` output types exist.

### Colocalization And Segmentation Quality

- Pearson correlation within an optional mask;
- Manders overlap coefficients;
- object overlap / intersection-over-union;
- Dice coefficient;
- adapted Rand error;
- variation of information;
- contingency table between label images.

These should produce scalar or table results, not synthetic images.

## P3: Specialized Or Expensive

- random walker segmentation;
- active contours and morphological snakes;
- graph-cut or graph-based segmentation;
- superpixels such as SLIC;
- optical flow;
- non-rigid registration;
- image stitching and mosaics;
- object tracking across time;
- 3D surface/mesh extraction and measurement;
- frequency-domain notch filtering;
- blind deconvolution;
- learned denoisers;
- Cellpose, StarDist, ilastik, or other model-backed segmentation.

Model-backed nodes should be optional integrations with isolated dependencies,
model provenance, device selection, and reproducible model/version metadata.
They should not make the core plugin installation heavy.

## Defer For Now

The following are common in general computer-vision libraries but are not good
early priorities for a bioimage workflow composer:

- many separate edge operators such as Sobel, Scharr, Prewitt, Roberts, and
  Canny;
- Hough line and circle transforms;
- contour hierarchy operations;
- polygon approximation and rotated boxes;
- ORB/SIFT-like keypoints and descriptors;
- template matching;
- face/object detection APIs;
- camera calibration and video-stream processing;
- broad color-space conversion catalogues;
- artistic transforms, image pyramids, and inpainting;
- duplicate OpenCV versions of operations already provided well by SciPy or
  scikit-image.

One gradient-magnitude node may be useful for watershed or QC, but implementing
every edge detector would add palette noise without completing a bioimage
workflow.

## Suggested Palette Structure

```text
Image Data
  Source & Output
  Axes & Geometry
  Channels & Composites
  Type & Scaling
  Math & Logic

Enhancement
  Background Correction
  Contrast
  Denoising
  Sharpening & Features

Segmentation
  Thresholding
  Marker Generation
  Watershed & Region Methods

Binary Morphology
Label Operations
Measurements
Registration
Restoration
```

`Filtering` can remain during migration, but splitting enhancement by intent
will become easier to navigate than one long filter list.

## Example Target Workflows

### Fluorescent Nuclei

```text
Extract channel
  -> Rolling-ball background
  -> Gaussian blur
  -> Automatic threshold
  -> Remove small objects
  -> Fill Holes, optionally limited by hole area/volume
  -> Distance transform
  -> H-maxima markers
  -> Watershed
  -> Measure objects
```

### Cell Regions From Nuclear Seeds

```text
Nuclear labels
  -> Expand labels
  -> Optional membrane/intensity mask constraint
  -> Measure per-cell channel intensity
```

### Fluorescent Puncta

```text
Extract channel
  -> Background correction
  -> Difference of Gaussians or Laplacian of Gaussian
  -> Peak/blob detection
  -> Points/table
  -> Count spots per cell label
```

### Mitochondrial Morphology

Mitochondria can be analyzed either as separate organelles or as a connected
network. Connected-component labeling and volume filtering are useful for
removing debris and measuring fragmented organelles, but a connected network
may intentionally form one large label.

```text
Extract mitochondrial channel
  -> background correction
  -> ridge/tubeness enhancement or denoising
  -> threshold
  -> binary cleanup
  -> labels for fragment analysis
  -> skeleton/network analysis for branches and connectivity
```

This means future mitochondrial support needs both object measurements and
skeleton/graph measurements; volume filtering alone is not sufficient.

### Colocalization

Pixel-based and object-based colocalization should remain distinct:

```text
Pixel-based:
channel A + channel B + optional ROI mask
  -> Pearson / Manders metrics
  -> result table

Object-based:
labels A + labels B
  -> overlap, nearest-neighbor distance, or object association
  -> result table and optional matched labels
```

Colocalization therefore follows table outputs and benefits from labels, but
does not require registration in its first version when channels are already
spatially aligned.

### Time-Lapse Drift Correction

```text
Select reference frame
  -> Estimate translation per timepoint
  -> Apply translation
  -> Preserve T/C/Z/Y/X metadata
```

## Decisions And Open Discussion

Decided:

1. Label connectivity exposes both face and full connectivity; full
   connectivity is the current default.
2. Object labels are created independently over leading non-spatial axes, while
   a recognized `ZYX` block is treated as one volume.
3. Registration and deconvolution remain later milestones.
4. The initial workflow priority covers nuclei/cells, puncta, mitochondria, and
   colocalization rather than only one of those domains.

Still open:

1. Should `Filter Labels By Volume` default to calibrated units whenever scale
   metadata exists, or keep pixels/voxels as the default with physical units as
   an explicit mode?
2. How should mixed or incompatible spatial units be handled?
3. Which mitochondrial workflow should come first after generic measurements:
   fragmented-organelle object measurements or connected-network skeleton
   analysis?
4. Should `Automatic Threshold` consolidate existing threshold nodes in the
   palette while preserving old operation IDs for saved workflows?
5. Should `Remove Small Objects` replace `Volume Filter` in the visible palette
   while retaining the old operation id for workflow compatibility?

## Library Direction

The current environment has SciPy and scikit-image but not OpenCV. That is a
reasonable default:

- SciPy provides efficient multidimensional filtering, morphology, distance
  transforms, interpolation, and labeling.
- scikit-image provides bioimage-relevant segmentation, labels, region
  measurements, restoration, feature detection, and registration.
- OpenCV is strongest for broad 2D computer vision, contours, real-time/video
  workflows, and highly optimized implementations. Those strengths do not yet
  justify another required dependency for VIPP.

OpenCV can be reconsidered if profiling shows a meaningful performance gap or a
selected node has no adequate SciPy/scikit-image implementation.

## Primary References

- [scikit-image segmentation API](https://scikit-image.org/docs/stable/api/skimage.segmentation.html)
- [scikit-image morphology API](https://scikit-image.org/docs/stable/api/skimage.morphology.html)
- [scikit-image measurement API](https://scikit-image.org/docs/stable/api/skimage.measure.html)
- [scikit-image filters API](https://scikit-image.org/docs/stable/api/skimage.filters.html)
- [scikit-image restoration API](https://scikit-image.org/docs/stable/api/skimage.restoration.html)
- [scikit-image feature API](https://scikit-image.org/docs/stable/api/skimage.feature.html)
- [scikit-image registration API](https://scikit-image.org/docs/stable/api/skimage.registration.html)
- [scikit-image transform API](https://scikit-image.org/docs/stable/api/skimage.transform.html)
- [SciPy multidimensional image processing API](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
- [OpenCV image-processing overview](https://docs.opencv.org/4.x/d7/da8/tutorial_table_of_content_imgproc.html)
