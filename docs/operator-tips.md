# VIPP Operator Tips and Performance

Last reviewed: 2026-09-05

This guide is for day-to-day operation of larger or more complex workflows.
It focuses on responsiveness, stability, and practical tuning.

## Choosing Background Mode

The right-side gear menu's `Run all in background` setting provides two
background-processing behaviors:

- `Run all in background` off: automatic mode backgrounds known slower
  operations and updates involving at least 32 MiB or four million image
  values. Smaller, quick edits remain inline.
- `Run all in background` on: all recomputes use background mode.

Use `Run all in background` on when:

- pipelines are long;
- mid-sized operations still make interaction feel uneven;
- users need visible progress feedback during recompute.

Use `Run all in background` off when:

- edits are usually small and fast;
- reducing per-run orchestration overhead is more important than progress UI.

The `Stop` button appears beside progress in the status footer while a
background graph update is active. Compute mode and per-node backend controls
are disabled for the entire run. Use this explicit button before changing
CPU/GPU policy. It cancels
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

The gear menu's `Link napari/VIPP sliders` setting controls whether previews
and slice-based histograms track the current napari dim position.

- On: best for normal interactive exploration.
- Off: best for fixed-reference comparison while scrubbing dims.

For heavy scenes, these settings can help reduce UI churn:

- set `Preview > Detail` to `Low (90 × 55)` while authoring; use Standard
  (180 × 110), High (360 × 220), or Very High (720 × 440) when more backing
  detail is useful on a HiDPI display, during downsampling, or at maximum graph
  zoom. Very High uses four times the thumbnail pixels of High;
- set `Preview > Range` to `Slice` when you do not need stable brightness across
  the whole output;
- set `Preview > Mode` to `Off` when tuning non-visual parameters;
- keep histogram scope to `Slice` while iterating.

Thumbnail detail changes only the backing image for the fixed card viewport. It
does not change the full output read by Stack contrast. Slice contrast instead
normalizes the spatially sampled current view, so its display limits may shift
slightly between Low, Standard, High, and Very High. The gear menu's
`Thumbnail statistics` setting controls full-output Stack work. Auto uses
eligible exact `uint8` Percentile histograms on CuPy from a conservative
384-MiB cold crossover and `uint16` histograms from 512 MiB; both use 32 MiB once warm. These measured
defaults are heuristics, not a guarantee for every distribution or computer.
CPU avoids CUDA, and Prefer GPU is the explicit override with visible fallback.
Min-max uses an exact native reduction on CPU instead of a histogram. Float and
other-dtype percentiles retain the exact NumPy-compatible CPU path. Main compute
CPU always forces these statistics to CPU; main Prefer GPU biases presentation
Auto toward GPU.

Select a node and read the compact `Thumbnail contrast` row near the top of its
inspector for presentation state: `Calculating…`, `CPU · NumPy`, `GPU · CuPy`,
`CPU fallback`, or `Error`. The ordinary title-row compute badge still identifies
what produced the scientific output. Hover the inspector row or thumbnail for
details; keyboard What's This help and screen readers receive the same text.
Stack statistics use the status footer's progress bar and `Stop` button and
retain provisional thumbnails if cancelled. CPU integer work stops
between bounded chunks. An active GPU kernel/synchronization or exact
float/other-dtype NumPy percentile can have a non-interruptible inner pass; the
progress message identifies the phase and cancellation takes effect after that
pass returns.

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

## Working Through A Batch

Use the four `Batch workflow` tabs in order: `Setup` for sources and destination,
`Items & outputs` for checking the exact pairing and planned files, `Overrides`
for supported per-sample values, and `Run & results` for execution and evidence.
The footer keeps the current activity and next action visible while changing
tabs; source and destination controls reflow vertically in a narrower window.

- Start with `Check batch`. Source revision, metadata, pairing, and output
  planning run in the background without calculating a graph representative or
  writing batch outputs. `Preview selected` is optional and calculates one
  item through the live graph; it does not run the full collection.
- The first Check stage only lists files; the list appears before metadata and
  exact-content fingerprint checks finish. Follow the active-row indicator and
  checked-file count. Fingerprinting reads all source bytes, so large CZI/TIFF
  files can take much longer than directory listing. File counts are not sample
  counts when a container has multiple images; wait for the checked plan before
  previewing, loading overrides, or running.
- Keep the distinction between an old table and a current checked plan. Source,
  destination, workflow, or override edits require `Check batch`/`Recheck all`
  before Run is available. Run then verifies the entire batch again and stops
  for review if inputs or destinations changed unexpectedly.
- `Recheck selected` only verifies the chosen source revisions and output-file
  presence. It does not recheck the other items or replace full scientific
  preflight. A failed selected check requires a full recheck.
- Both the item browser and override table use 50-row pages. Search and filter
  first; in `Overrides`, choose relevant parameters with `Show columns…` and
  use sample checkboxes, `This page`, or `Select all matching` for multi-sample
  selection. Check the selected count: samples hidden by a filter may still be
  selected.
- Blank override cells inherit the workflow value. In `Edit selected…`, choose
  only the parameters to change, then use `Set value` or `Use workflow value`.
  `Apply to selected samples` commits the validated draft; `Discard edits`
  leaves existing values untouched. Unchosen parameters are preserved.
- `Reset selected…` restores every parameter for the checked samples, including
  hidden columns and selections on other pages. `Deselect` only unchecks samples.
  `Reset all overrides…` at the top restores all sample parameters and all
  Run/Bypass choices. Both resets ask for confirmation; neither changes the
  original workflow. Check the batch again after resetting values.

The per-sample editor currently supports eligible public numeric parameters
only. Choice settings such as a direction selector must be changed in the
shared workflow; they are not available as per-sample controls. `Load overrides`
opens the selected samples in this editor, not a file import. Recheck after
editing, and use an optional representative to judge the result before a large
run.

The compute summary describes the inherited or saved request, not actual GPU
use. Check run evidence for the implementations that really executed. Progress
reports real item and operation checkpoints; a long library call can remain
indeterminate. Elapsed time comes from captured timing or reported timestamps,
and unavailable timings are not filled with estimates.

Use `Stop safely` and wait for cleanup before starting new work. Saved outputs
are retained, and the final result distinguishes failed, partial, cancelled,
skipped, and completed items. Hiding the window does not stop an active batch.
There is no automatic resume control: resolve the cause, recheck the full batch,
and deliberately retain or change the existing-file policy before a new run.
`Skip existing` is file preservation, not a
guarantee that an earlier item's unfinished work resumes. Overwrite and
overwrite-approval rules still apply.

After checking, use **Existing files · batch default** directly in Items &
outputs or Run & results to choose the default Skip or Overwrite policy. Right-click
an individual row in Items & outputs to **Keep existing outputs** or **Rerun and
overwrite outputs** for that item only, independent of the checkbox selection.
**Use batch default** resets one item; **Reset item choices** resets all file
choices, not parameter or node overrides. Individual choices survive save/load
and batch-default changes, and are bound to exact samples and destination paths.
A policy-only
change retains the source checks; it does not need another manual recheck. Ask
before overwrite prompts when you press Run. **Keep existing** means preservation,
not proof that an earlier workflow completed; the Run count excludes items whose
outputs are all being kept. Run still revalidates current disk contents.

Run performs one final collection validation to catch disk changes since Check;
the worker reuses that plan, including individual keep/overwrite decisions.
This validation hashes full source contents, which can take time for large
containers. Per-item verification and guarded output publication remain active
during execution; reusing the plan is not permission to use changed inputs.

Use a source or output's `Find in File Explorer`/`Find in Finder` action to
locate that exact existing file. Linux explicitly opens its containing folder
instead. Missing files cannot be revealed. In `Run & results`, `Refresh file status`
updates presence without rerunning analysis, `Output folder` opens the
directory. The readable run summary appears directly on the page; the footer's
`View run report` returns to it after browsing. To prepare another run, return to
`Setup` and explicitly choose `Check batch` after reviewing settings. The summary
separates saved, kept, failed and cancelled outputs, and shows failure reasons.
Expand `Show all details` to read longer reasons without opening a JSON file.
`Find manifest JSON` locates the archived technical record. Keep that record when
comparing runs: a successful fresh check replaces the previous run view.

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

1. Turn `Run all in background` on if a mid-sized operation falls below the
   automatic cutoff but still pauses interaction.
2. Set `Preview > Detail` to `Low`.
3. Set `Preview > Range` to `Slice` to avoid full-output thumbnail statistics.
4. Set `Preview > Mode` to `Off` and retest.
5. Switch `Preview > Mode` from `MIP` to `Slice`.
6. Reduce graph fan-out while tuning upstream nodes.
7. Re-enable features one by one to identify the dominant cost.

## Related Docs

- End-user behavior: `docs/user-guide.md`
- Architecture and internal design: `docs/developer-notes.md`
