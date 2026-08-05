# VIPP Operator Tips and Performance

Last reviewed: 2026-08-05

This guide is for day-to-day operation of larger or more complex workflows.
It focuses on responsiveness, stability, and practical tuning.

## Choosing Background Mode

VIPP supports two background-processing behaviors in the toolbar:

- `Run all in BG` off: automatic mode backgrounds known slower operations and
  updates involving at least 32 MiB or four million image values. Smaller,
  quick edits remain inline.
- `Run all in BG` on: all recomputes use background mode.

Use `Run all in BG` on when:

- pipelines are long;
- mid-sized operations still make interaction feel uneven;
- users need visible progress feedback during recompute.

Use `Run all in BG` off when:

- edits are usually small and fast;
- reducing per-run orchestration overhead is more important than progress UI.

The toolbar `Cancel calculation` button appears while a background graph update
is active. Compute mode and per-node backend controls are disabled for the
entire run. Use this explicit button before changing CPU/GPU policy. It cancels
queued reruns, marks the in-flight dirty nodes as pending again, and asks
cooperative operations to stop. The button and compute controls remain in their
stopping state until the worker has synchronized and released CPU/GPU resources;
selecting CPU can therefore never leave an earlier GPU calculation running.
Rolling-ball/subtract-background block
processing, rescale axes, and 3D mesh morphology now report progress and check
for cancellation between internal work units. VIPP still cannot forcibly
terminate a NumPy, SciPy, or scikit-image call that is already executing inside
the worker thread, so CPU use may continue briefly while the current work unit
finishes.

Cancellation retains the last coherent result. After a failed/OOM calculation,
VIPP may accept a verified source boundary. If cleanup itself failed, it may
also accept a completed processing node whose matching actual-implementation
decision is available; an uncomputed or
unreported processing value never replaces a prior valid output. Existing
images, thumbnails, and truthful CPU/GPU badges therefore remain available for
uncompleted work, with pending or previous-result styling where intent differs.
If accelerator cleanup itself fails, all calculation, policy, benchmark,
optimizer, and new batch controls stay disabled until VIPP is restarted.

## Preview and Dims Strategy

`Follow napari dims` controls whether previews and slice-based histograms track
the current napari dim position.

- On: best for normal interactive exploration.
- Off: best for fixed-reference comparison while scrubbing dims.

For heavy scenes, these settings can help reduce UI churn:

- set `Thumbnail detail` to `Low (90 × 55)` while authoring; use Standard
  (180 × 110) or High (360 × 220) when more backing detail is useful on a HiDPI
  display or during downsampling;
- set `Contrast Range` to `Slice` when you do not need stable brightness across
  the whole output;
- set preview mode to `Off` when tuning non-visual parameters;
- keep histogram scope to `Slice` while iterating.

Thumbnail detail changes only the backing image for the fixed card viewport. It
does not change the full output read by Stack contrast. Slice contrast instead
normalizes the spatially sampled current view, so its display limits may shift
slightly between Low, Standard, and High. `Settings > Thumbnail statistics`
controls full-output Stack work. Auto uses eligible exact
`uint8` Percentile histograms on CuPy from a conservative 384-MiB cold crossover
and `uint16` histograms from 512 MiB; both use 32 MiB once warm. These measured
defaults are heuristics, not a guarantee for every distribution or computer.
CPU avoids CUDA, and Prefer GPU is the explicit override with visible fallback.
Min-max uses an exact native reduction on CPU instead of a histogram. Float and
other-dtype percentiles retain the exact NumPy-compatible CPU path. Main compute
CPU always forces these statistics to CPU; main Prefer GPU biases presentation
Auto toward GPU.

Read the node's separate `Stats…`, `Stats · CPU`, `Stats · GPU`,
`Stats · CPU fallback`, or `Stats · error` chip for presentation state; the
ordinary compute badge still identifies what produced the scientific output.
Stack statistics use the toolbar progress and Cancel surfaces and retain
provisional thumbnails if cancelled. CPU integer work stops between bounded
chunks. An active GPU kernel/synchronization or exact float/other-dtype NumPy
percentile can have a non-interruptible inner pass; the progress message
identifies the phase and cancellation takes effect after that pass returns.

Large stack histograms and automatic-threshold markers are calculated in the
background. The inspector briefly shows `calculating...` and reuses the result
when napari emits repeated dimension events or the node is revisited. Choosing
`Slice` reduces the requested scope, but both slice and stack histograms count
all finite pixels in that scope; VIPP does not introduce hidden sampling.

## Practical Workflow Habits

- Add expensive nodes later in graph construction and tune early nodes first.
- Prefer a stable input layer during intensive tuning to keep cache reuse high.
- Use pinned outputs for side-by-side checks without reconfiguring the graph.
- Save workflow snapshots before major parameter sweeps.

## Deconvolution Tuning Order

For blurred or missing structures, change one cause at a time:

1. Read the RL/RL-TV PSF preflight and validate rank, physical sampling,
   centering, and support. Missing calibration is a warning, not permission to
   assume unit pixel spacing.
2. Reduce TV regularization and compare directly with `0`. The default `0.002`
   is conservative; `0.008-0.012` is comparatively strong and may erase real
   dim or fine structures.
3. Check under-convergence. More iterations may recover feature intensity, but
   can also worsen noise, boundary artifacts, or global error.
4. Compare ordinary RL and RL-TV at the same iteration count.
5. Inspect boundary regions and PSF provenance.
6. Only then test advanced numerical guards such as TV epsilon, filter epsilon,
   or denominator floor.

Do not use iteration count to compensate for a miscentered or incorrectly
sampled PSF. A smoother reconstruction, or one with a better global denoising
metric, is not automatically the result that best preserves meaningful dim
structures.

## Troubleshooting Slow Updates

If updates feel slow:

1. Turn `Run all in BG` on if a mid-sized operation falls below the automatic
   cutoff but still pauses interaction.
2. Set thumbnail detail to `Low`.
3. Use `Slice` contrast range to avoid full-output thumbnail statistics.
4. Set preview mode to `Off` and retest.
5. Switch preview mode from `MIP` to `Slice`.
6. Reduce graph fan-out while tuning upstream nodes.
7. Re-enable features one by one to identify the dominant cost.

## Related Docs

- End-user behavior: `docs/user-guide.md`
- Architecture and internal design: `docs/developer-notes.md`
