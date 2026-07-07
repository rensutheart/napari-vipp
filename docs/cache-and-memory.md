# Cache And Memory Policy

VIPP is currently an eager, interactive workflow builder. Most nodes calculate
NumPy-like in-memory outputs so that graph thumbnails, node inspection, pinned
overlays, and downstream edits feel immediate. This is useful while designing a
workflow, but it can become expensive on large z-stacks, time series, or
OME-Zarr-derived arrays.

## Interactive Cache Modes

Use the Settings menu to choose the cache mode.

The status bar reports the estimated VIPP cache size, the active strategy in
parentheses, and system RAM. For example: `Cache 220 MB (Smart interactive) |
RAM free 11.4 GB / 31.7 GB`.

| Mode | Intended use | Retained outputs |
| --- | --- | --- |
| Keep all node outputs cached | Default interactive graph design. Best for rapid inspection and branching. | Every calculated node output until the graph/source changes or memory guard intervenes. |
| Smart interactive cache | Large graphs where repeated inspection still matters. Selecting a pruned node restores that node output and thumbnail for inspection. | Selected and pinned nodes, direct working inputs, source nodes, branch points, explicit output nodes, recently inspected nodes, and nodes marked `Keep output cached`. |
| Low-memory mode | Memory-constrained interactive work and batch-like runs. | The selected/pinned working input and result, explicit output nodes, and nodes marked `Keep output cached`. |

Batch collection runs use low-memory retention internally. They keep only the
outputs that must be saved, then clear item-level caches before moving to the
next input.

## Memory Guard

The Settings menu includes `Auto memory guard` and `Cache limit`.

When keep-all mode is active and the estimated VIPP cache exceeds the configured
share of reclaimable memory, VIPP automatically:

1. switches cache mode to `Smart interactive cache`;
2. clears optional helper caches;
3. prunes nonessential node outputs;
4. refreshes thumbnails, inspector views, and cache status;
5. shows a warning explaining what happened.

Reclaimable memory is calculated as:

`currently free RAM + current VIPP cache`

The default cache limit is 90%. In other words, VIPP allows keep-all caching to
use most of the memory that would be available if VIPP cleared its own cache,
while still leaving a little breathing room for napari, Python, and the
operating system. The guard can also trigger if system free RAM falls below a
small safety reserve. If the operating system does not report free memory, VIPP
falls back to a total-RAM estimate.

The cache estimate counts in-memory node outputs and VIPP tables. It is not a
complete Python heap profile, so it should be read as a practical warning
signal rather than an exact peak-memory measurement.

## Per-Node Keep Cached

Every selected node exposes `Keep output cached` in the inspector. Use it for
expensive intermediates that you expect to inspect or reuse repeatedly, such as:

- a slow background-subtracted image feeding several branches;
- a high-quality rescaled or registered reference image;
- a manual measurement table that is expensive to recompute;
- a segmentation mask used by multiple downstream analyses.

This setting is saved in workflow JSON as a hidden VIPP node setting and is
excluded from exported operation calls. It affects cache retention only; it does
not force a node to calculate if the node has no output yet.

## Operation Memory Characteristics

| Operation family | Current behavior | Memory notes |
| --- | --- | --- |
| Image source and OME-Zarr reads | Source metadata can be lazy-capable, and OME-Zarr level reads can remain Dask-backed at the I/O boundary. | Most processing nodes still materialize arrays when they calculate. |
| Pointwise intensity, threshold, clipping, rescaling, and image math | Eager and usually cache-friendly. | Outputs are often the same shape as the input, so keep-all can multiply memory by pipeline length. |
| Filtering, background correction, morphology, distance transform, watershed, and axis rescaling | Eager. | Often memory-heavy; 3D background correction, distance maps, label images, and interpolation can create large temporary arrays. |
| Projection and orthogonal views | Eager. | Usually reduce dimensionality, but orthogonal view generation can increase canvas size depending on physical scaling. |
| Channel split/composite and RGB conversion | Eager. | Split outputs can duplicate channel data; composites may add a channel/RGB axis. |
| Object, mesh, skeleton, and colocalization measurement nodes | Manual/cached where expensive. | Tables are usually smaller than images, but their source images or intermediate masks can dominate memory. |
| Save Image and Batch Output nodes | Explicit terminal/output intent. | Batch execution retains these outputs only long enough to write them. |

## Large-Data Direction

The current policy is pragmatic rather than fully lazy. Before VIPP can be
comfortable on very large OME-Zarr datasets, the next scale work should add:

- operation capability declarations for eager, lazy-safe, memory-heavy, and
  scale-aware nodes;
- preview-resolution controls for thumbnails and histograms;
- sampled or chunked histograms for large arrays;
- OME-Zarr pyramid generation and preview-level selection;
- confirmation before eager-only nodes materialize very large lazy arrays.
