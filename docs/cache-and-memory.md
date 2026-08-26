# Cache And Memory Policy

Last reviewed: 2026-08-05

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
| Smart interactive cache | Large graphs where repeated inspection still matters. Selecting a pruned node restores that node output and thumbnail for inspection. | All calculated manual nodes; temporarily blocked downstream snapshots; selected and pinned nodes; direct working inputs; source nodes; branch points; explicit output nodes; recently inspected nodes; and nodes marked `Keep output cached`. |
| Low-memory mode | Memory-constrained interactive work and batch-like runs. | All calculated manual nodes; temporarily blocked downstream snapshots; the selected/pinned working input and result; explicit output nodes; and nodes marked `Keep output cached`. |

Batch collection runs use low-memory retention internally. They keep only the
outputs that must be saved, then clear item-level caches before moving to the
next input.

Calculated **manual-node** results are a retention invariant in the interactive
graph: Keep-all, Smart, and Low-memory modes all preserve them, including an
expensive deconvolution several hops upstream from the node currently being
edited. This prevents a downstream display/mapping change from silently
discarding work that required an explicit Calculate action.

A branch waiting behind a stale manual-node barrier is also retained as one
coherent snapshot, including automatic descendants. Those nodes use the
`blocked` execution state until the actionable `stale` frontier is recalculated;
normal Smart/Low-memory pruning resumes after the branch becomes current.

Automatic nodes follow the table's ordinary retention sets. A downstream edit
does not invalidate an unchanged automatic upstream node, but Smart/Low-memory
pruning may remove its cached array; VIPP then recomputes that automatic result
when it is required again. Use `Keep output cached` when a particular automatic
intermediate must survive that intentional pruning.

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

### Platform Memory Reporting

The RAM value beside the cache estimate uses an explicit operating-system
branch; VIPP does not assume that one memory API exists everywhere:

| Platform | Provider |
| --- | --- |
| Windows | `GlobalMemoryStatusEx` through the native Windows API, with physical RAM and system commit reported separately. |
| macOS | `host_statistics64` for available pages plus POSIX page-size and physical-page counts. |
| Other POSIX systems | `os.sysconf` page and physical-memory counters. |

Each provider fails closed to an unavailable RAM reading if its native API or
counter is absent. A missing platform counter must not prevent graph execution,
and Windows never enters the POSIX/macOS `sysconf` path. The auto memory guard
then uses the information that is available, including its existing total-RAM
fallback, rather than inventing a cross-platform estimate.

On Windows, `Commit free` is the remaining system commit headroom reported by
`GlobalMemoryStatusEx`; it is not merely free physical RAM or the configured
page-file size. A large NumPy/SciPy allocation can fail when commit is exhausted
even if Windows still reports physical RAM available. VIPP therefore checks
both physical and commit reserves before an optional memory-intensive Auto CPU
timing comparison. If that comparison is unsafe—or commit information is
unavailable on Windows—Auto retains its reviewed safe assignment, reports why
the timing was skipped, and can gather the missing evidence on a later run.

This preflight protects optional evidence collection; it is not a guarantee
that every library allocation will succeed. A real host `MemoryError` is
classified separately. In the interactive graph, a failed or canceled private
run never publishes uncomputed or provenance-unknown processing data over an
earlier valid result. A verified source boundary may be accepted; a completed
processing node may additionally be accepted from a cleanup-failed result, and
only with its matching actual-implementation decision. Prior outputs,
  thumbnails, scientific compute badges, and selected-node thumbnail contrast
  status remain truthful for all other affected work while it is requeued. A
  cleanup failure additionally quarantines new compute in that VIPP process
  until restart.

## Per-Node Keep Cached

Every selected node exposes `Keep output cached` in the inspector. Use it for
expensive intermediates that you expect to inspect or reuse repeatedly, such as:

- a slow background-subtracted image feeding several branches;
- a high-quality rescaled or registered reference image;
- a manual measurement table that is expensive to recompute;
- a segmentation mask used by multiple downstream analyses.

This setting is saved in workflow JSON as a hidden VIPP node setting and remains
embedded in generated workflow exports, but the executor filters it before
calling the scientific operation. It affects cache retention only; it does not
force a node to calculate if the node has no output yet.

## Operation Memory Characteristics

| Operation family | Current behavior | Memory notes |
| --- | --- | --- |
| Image source and OME-Zarr reads | Readers may expose lazy arrays internally, but an interactive file source is fully materialized into one verified, read-only snapshot before it enters the graph. | The complete selected source stays resident until Refresh; OME-Zarr, microscope, and large-file materialization runs on the background queue. |
| Pointwise intensity, threshold, clipping, rescaling, and image math | Eager and usually cache-friendly. | Outputs are often the same shape as the input, so keep-all can multiply memory by pipeline length. |
| Filtering, background correction, morphology, distance transform, watershed, and axis rescaling | Eager. | Often memory-heavy; 3D background correction, distance maps, label images, and interpolation can create large temporary arrays. |
| Projection and orthogonal views | Eager. | Usually reduce dimensionality, but orthogonal view generation can increase canvas size depending on physical scaling. |
| Channel split/composite and RGB conversion | Eager. | Split outputs can duplicate channel data; composites may add a channel/RGB axis. |
| Object, mesh, skeleton, and colocalization measurement nodes | Manual/cached where expensive. | Tables are usually smaller than images, but their source images or intermediate masks can dominate memory. |

Exact interior percentiles require one native-dtype working buffer so that the
requested order statistics can be selected without histogram approximation.
The common integer `0..100` percentile pair uses exact extrema and avoids that
buffer. Integer Rescale then maps in bounded chunks; integer Clip is a native
pointwise clamp and does not allocate a whole-stack float copy.
| Save Image and Batch Output nodes | Explicit terminal/output intent. | Batch execution retains these outputs only long enough to write them. |

## Large-Data Direction

The current policy is pragmatic rather than fully lazy. The 0.14.0a1 candidate
can slice a declared lower local OME-Zarr 0.4/0.5 level for presentation while
keeping analysis at level 0. Before VIPP can be comfortable on very large
scientific graphs, the next scale work should add:

- operation capability declarations for eager, lazy-safe, memory-heavy, and
  scale-aware nodes;
- richer pyramid-aware thumbnail controls beyond automatic lower-level source
  preview and the current Low (90 × 55), Standard (180 × 110), High
  (360 × 220), and Very High (720 × 440) backing-detail controls;
- broader chunked execution beyond the bounded global-threshold and inspector
  histogram paths;
- remote-store and non-OME-Zarr pyramid preview plus pyramid generation;
- confirmation before eager-only nodes materialize very large lazy arrays.

Thumbnail render detail and contrast work are separate budgets. Low, Standard,
High, and Very High retain 90 × 55, 180 × 110, 360 × 220, or 720 × 440 backing
images for the fixed card viewport; High and Very High may improve HiDPI display,
downsampling, or maximum graph zoom without making the card itself larger. Very
High uses four times the backing pixels and approximately four times the pixmap
memory of High. Slice contrast normalizes this spatially sampled current
view, so changing detail can slightly change Slice display limits. Stack
Percentile/Min-max instead summarizes the complete node output, caches its
limits independently of render detail, and reports its own progress. Native
`uint8` and `uint16` Percentile calculations use an exact, bounded histogram
on CPU or eligible CuPy GPU; Min-max uses an exact native reduction.
Auto's conservative cold GPU crossover is 384 MiB for `uint8` and 512 MiB for
`uint16`, becoming 32 MiB once the histogram path is warm. These measured
defaults are heuristics rather than guarantees because hardware, distribution,
startup, residency, and competing work can change the fastest backend. Prefer
GPU is the explicit override. Other dtypes retain the exact NumPy-compatible CPU
percentile path; it may allocate full-array conversion/filter temporaries and
its active NumPy call may temporarily be non-interruptible. The GPU histogram
uploads one complete eligible input and allocates a fixed count table; admission
accounts for both plus conservative overhead. Slice contrast avoids the
full-output scan.
