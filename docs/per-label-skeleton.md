# Per-label skeleton implementation contract

Unreleased after 0.16.0a1. This is the developer contract for
`skeletonize_labels` and `analyze_skeleton_per_label`. Public instructions live
in the manual's [skeleton tutorial](https://rensutheart.github.io/vipp-mkdocs/nightly/workflows/skeleton-network-analysis/)
and [node reference](https://rensutheart.github.io/vipp-mkdocs/nightly/reference/skeleton-nodes/).

## Ownership and compatibility

[`core/label_skeleton.py`](../src/napari_vipp/core/label_skeleton.py) owns the
Qt-free label iteration, thinning, subset validation and table construction.
`core/pipeline.py` owns operation registration, optional-input preparation,
spatial/grid validation, progress injection and output-state construction;
`core/metadata.py` preserves label-image metadata and thinning history.

The two nodes are CPU implementations. `skeletonize_labels` returns a new
label array. `analyze_skeleton_per_label` returns two `TableData` outputs:
port 0 `out` (**Objects**) and port 1 `components` (**Skeleton components**).
Analysis is manual/cached. The shared executor owns cancellation, accepted
cache results, compute provenance and publication across interactive, batch
and generated-Python execution.

Existing `skeletonize`, `analyze_skeleton` and `label_skeleton_components` keep
their mask-first semantics. In particular, **Label Skeleton Components**
assigns new IDs to connected skeleton pieces. Those IDs do not identify an
earlier object label. The new route preserves original object identity and
does not reinterpret saved workflows or replace the old component-label node.

## Scientific algorithm

Each positive label is isolated as a Boolean mask inside its bounding box,
padded with one background voxel, then passed to scikit-image's
[`morphology.skeletonize`](https://scikit-image.org/docs/stable/api/skimage.morphology.html#skimage.morphology.skeletonize).
Auto resolves to Zhang/Suen thinning in 2D and Lee/Kashyap/Chu thinning in 3D.
Lee is also accepted in 2D; Zhang is rejected for a 3D block. The official
reference cites Zhang and Suen (1984), *Communications of the ACM* 27(3), and
Lee, Kashyap and Chu (1994), *Computer Vision, Graphics, and Image Processing*
56(6):462–478. VIPP's label separation surrounds that existing binary algorithm;
it is not a new thinning method.

The crop and zero padding preserve the foreground and background-boundary
assumption. Tests compare against independently thinned full-image masks,
including touching labels. No sampling, iteration cap, resampling, bridging,
pruning or label renumbering is introduced. Thinning remains a voxel-grid
operation, not a physical medial-axis calculation for anisotropic data.

Measurement delegates each label mask to the existing
`operations._analyze_skeleton_block`. Component discovery uses full
8-connectivity in 2D or 26-connectivity in 3D. Graph edges retain VIPP's
existing `_valid_skeleton_edge` convention: a diagonal shortcut is omitted
when a lower-order intermediate skeleton voxel already connects that local
neighborhood. Lengths sum the retained graph edges, using Euclidean offset
lengths in voxel coordinates or calibrated spatial coordinates. Component
discovery and graph-edge reduction are distinct stages; do not substitute a
new connectivity rule without separate tests and a documented behavior change.

## Array, axes and calibration contract

- Inputs must be non-negative signed or unsigned integer label arrays with
  at least two dimensions. Zero is background. Boolean, float, negative,
  complex and object inputs fail instead of being cast. Sparse and wide
  unsigned IDs, including values above signed int64 range, remain exact in
  output images and both tables. No allocation is indexed by the largest ID.
- Skeleton output retains the original shape, dtype and axis order, owns its
  buffer and is an ID-preserving voxel subset. Read-only and noncontiguous
  inputs are supported without mutation.
- Spatial processing follows explicit axis semantics. Declared YX/ZYX resolve
  2D/3D blocks; ambiguous QYX volume requests fail. A bare 2D core-array call
  can resolve YX, while an ambiguous higher-rank Auto request cannot.
  Explicit 2D processes each remaining slice independently. Named nontrailing
  spatial axes are handled through temporary axis views; the returned image
  retains the authored axis order. Other axes are independent blocks and
  produce identity columns such as `t_index` or `p_index`.
- Physical lengths require finite positive spacing and recognized compatible
  length units on every spatial axis. Mixed compatible units are converted
  to the resolved X-axis unit. Missing or pixel/voxel units yield only
  `skeleton_length_pixels` (2D) or `skeleton_length_voxels` (3D). Partial,
  incompatible or unknown physical units, invalid spacing and non-finite
  converted spacing fail. Unit spacing must never manufacture a micrometer
  claim. Scale, axes, origin and inherited history are retained through the
  image branch; table columns carry their own measurement units.

## Required originals and optional skeleton

The analysis node requires port 0 `labels` (**Original labels**). Port 1
`skeleton` (**Skeleton (optional)**) is optional; connecting it alone does not
make the node executable. Without it, analysis thins each original label
internally. With it, analysis measures the supplied voxels and ignores the
thinning-method parameter.

Both inputs must be integer labels of the same shape. Every nonzero skeleton
voxel must equal the original label at that exact coordinate. The pipeline
also validates axis semantics, scale, units and origin as one physical-grid
contract, during both preflight and execution. Equal array shapes are not
sufficient. The lower-level array function checks shape and label membership;
callers bypassing the pipeline remain responsible for physical alignment.

The supplied image is trusted to be an already prepared skeleton. Membership
validation does not prove one-voxel thickness, biological correctness or
equivalence to a particular thinning method. It must not be silently thinned
again or remapped to make an invalid supplied skeleton fit.

## Row identity, empty objects and aggregation

The Objects table contains exactly one row per positive original label in
each processed spatial block. It carries `label_id`, the leading-axis index
columns, `skeleton_status` (`ok` or `empty`), `skeleton_component_count`, all
existing additive skeleton count fields and total graph length. Each count
and length is summed over that label's components. Calibrated tables also
contain `skeleton_length_physical` and `physical_unit`.

The component table adds a local `component_id`,
`component_count_in_label` and `component_voxel_fraction_in_label`; its count
and length fields describe the individual component. Several disconnected
fragments carrying one original label remain one object summary with multiple
detail rows. Repeated component-count context must not be summed across
detail rows.

Thinning can erase particular objects, including some small/even Lee inputs.
The original label still produces an `empty` row with zero counts and length,
and no component detail rows. An all-background input produces schema-stable
empty tables. A one-voxel isolate instead produces `ok`, one component, one
isolated node and zero edge length. `isolated_node_count` means degree-zero
skeleton voxels, not every disconnected object or mitochondrial fragment.

Joining morphology to Objects preserves one row per object. Join identity is
the original label **plus every applicable leading-axis and source-image
identifier**; a label ID alone is not globally unique. Source metadata is
preserved, but multi-image collection must still carry explicit image identity.
Joining component details repeats morphology once per component. IDs, status
and repeated context are not quantitative PCA features.

Summed per-label counts describe disjoint, label-constrained graphs. They are
not necessarily the topology of a whole-cell binary skeleton: adjacent labels
may gain connecting edges when identity is removed. Whole-network connectedness
and normalized ratios require their own explicitly defined population and
denominator. This change does not add inferential statistics or independent
biological replication.

## Resource and cancellation boundaries

Bounds discovery scans each spatial block in chunks of at most 1,048,576
values, keeping bounds per observed label. Thinning and graph work then use
one label bounding box at a time, instead of comparing every label against
the full volume. A widely dispersed label can still have a near-full-volume
box. The output label image, per-label bounds, accumulated tables and current
label's skeleton/graph allocations remain resident; this is not an out-of-core
algorithm or a guarantee of fixed peak memory.

Progress and cancellation checks occur during chunk scans, between labels,
and before/after atomic thinning and graph calls. A single library call or
large label graph is not preempted internally. Cancellation raises before
accepting the incomplete result; shared execution must not expose partial
tables as ready or publish them as completed batch outputs. No input or
upstream cached buffer is modified.

## Evidence and limits

The implementation check set recorded on 2026-09-22 comprises:

| Test file | Cases | Contract covered |
| --- | ---: | --- |
| [`test_label_skeleton.py`](../src/napari_vipp/_tests/test_label_skeleton.py) | 64 | Independent scikit-image reference, dtype/range, touching labels, fragments, isolates, Lee empty results, anisotropic units, semantic axes, read-only inputs and chunked cancellation. |
| [`test_label_skeleton_pipeline.py`](../src/napari_vipp/_tests/test_label_skeleton_pipeline.py) | 32 | Optional port, grid preflight, both output histories, joins, workflow/export round trip, CPU provenance, batch exports, cancellation and unchanged binary-component identity. |
| [`test_label_skeleton_example.py`](../src/napari_vipp/_tests/test_label_skeleton_example.py) | 4 | Shipped example, shared object filtering, original-ID join, retained isolate, 0.45 micrometer calibration, plot and generated Python. |

A read-only regression on the three available Control, FFA and Insulin masks
retained all 566, 435 and 333 original labels respectively. Their summaries
retained 58, 23 and 18 empty skeleton results rather than dropping those
objects. These checks establish identity/accounting behavior on those inputs;
they do not validate the segmentation, biological interpretation, acquisition
calibration or statistical separation of the three cells. Broader performance
and release qualification must be reported separately with their actual scope.

The bundled `synthetic-per-label-skeleton.json` and exhaustive inspector
showcase cover the user-facing nodes. Their initial/ready Qt layout checks,
manual content checks and strict manual build are separate presentation and
documentation checks, not scientific validation of acquired images.
