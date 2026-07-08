# VIPP Operator Tips and Performance

This guide is for day-to-day operation of larger or more complex workflows.
It focuses on responsiveness, stability, and practical tuning.

## Choosing Background Mode

VIPP supports two background-processing behaviors in the toolbar:

- `Run all in BG` off: background mode is used for known slower operations.
- `Run all in BG` on: all recomputes use background mode.

Use `Run all in BG` on when:

- images are large;
- pipelines are long;
- users need visible progress feedback during recompute.

Use `Run all in BG` off when:

- edits are usually small and fast;
- reducing per-run orchestration overhead is more important than progress UI.

The toolbar `Cancel` button appears while a background graph update is active.
It cancels queued reruns, marks the in-flight dirty nodes as pending again, and
asks cooperative operations to stop. Rolling-ball/subtract-background block
processing, rescale axes, and 3D mesh morphology now report progress and check
for cancellation between internal work units. VIPP still cannot forcibly
terminate a NumPy, SciPy, or scikit-image call that is already executing inside
the worker thread, so CPU use may continue briefly while the current work unit
finishes.

## Preview and Dims Strategy

`Follow napari dims` controls whether previews and slice-based histograms track
the current napari dim position.

- On: best for normal interactive exploration.
- Off: best for fixed-reference comparison while scrubbing dims.

For heavy scenes, these settings can help reduce UI churn:

- set preview mode to `Off` when tuning non-visual parameters;
- keep histogram scope to `Slice` while iterating.

## Practical Workflow Habits

- Add expensive nodes later in graph construction and tune early nodes first.
- Prefer a stable input layer during intensive tuning to keep cache reuse high.
- Use pinned outputs for side-by-side checks without reconfiguring the graph.
- Save workflow snapshots before major parameter sweeps.

## Troubleshooting Slow Updates

If updates feel slow:

1. Turn `Run all in BG` off and compare interactive latency.
2. Set preview mode to `Off` and retest.
3. Switch preview mode from `MIP` to `Slice`.
4. Reduce graph fan-out while tuning upstream nodes.
5. Re-enable features one by one to identify the dominant cost.

## Related Docs

- End-user behavior: `docs/user-guide.md`
- Architecture and internal design: `docs/developer-notes.md`
