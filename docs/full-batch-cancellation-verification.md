# Full Collection-Batch Cancellation Verification

Status: implemented on `codex/gpu-cross-platform-support`

Cross-platform scope: Windows, macOS, and Linux

Last reviewed: 2026-08-04

## Outcome

Full collection-batch execution is cooperatively cancellable on the GPU branch.
The implementation is not limited to CUDA or Windows: the Batch Workspace,
worker, cancellation token, core batch checkpoints, and provenance behavior are
shared across supported platforms and CPU/GPU execution modes.

The CPU-only `main` branch at commit `33ac31b` still uses the older synchronous,
non-cancellable full-batch path. Do not use that branch to judge the current GPU
branch's batch cancellation behavior.

## Implemented Behavior

- A visible `Cancel batch` control appears while a full batch is running.
- The full batch runs in a `CollectionBatchWorker` instead of blocking the Qt
  GUI thread.
- Cancellation is single-shot and changes the visible state to
  `Cancelling...` while the active operation reaches a safe checkpoint.
- One worker-owned `threading.Event` is passed through the controller into
  `core.batch.run_batch()` and the pipeline execution request.
- Cooperative operations can stop within the current item. A monolithic
  third-party call may finish before the token is observed; no Python or native
  thread is forcibly terminated.
- Item-level and nested operation-level progress remain visible in the retained
  Batch Workspace.
- Completed output remains published and recorded. Private staged output is
  cleaned up when cancellation wins before safe promotion.
- The manifest distinguishes `cancelled` from `failed`, records skipped work,
  and retains runtime-cleanup evidence.
- Stale progress from an older job id is ignored, and the workspace restores
  its controls after the worker reaches a terminal outcome.

## Implementation Landmarks

- `5e91136` — nested batch progress and cancel UI.
- `725fde6` — background batch worker plus progress/cancellation wiring.
- `5992a6d` — real GPU cancellation and cleanup coverage.
- `src/napari_vipp/ui/batch.py` — cancel control and retained terminal state.
- `src/napari_vipp/ui/batch_workers.py` — worker-owned cancellation event and
  typed progress/outcome signals.
- `src/napari_vipp/_widget.py` — active-job routing, stale-job rejection, and
  UI restoration.
- `src/napari_vipp/core/batch.py` — safe cancellation checkpoints, cancelled
  item status, manifest finalization, and execution cleanup.

## Apple M1 Max Verification

On 2026-08-04, commit `e024409` was tested from the GPU branch on an Apple M1
Max (`arm64`) running macOS 26.5.2. These focused tests passed:

- `test_pre_cancelled_batch_is_first_class_and_never_discovers_accelerator`
- `test_worker_carries_cancel_and_both_progress_channels`
- `test_batch_dialog_cancel_is_single_shot_and_waits_for_safe_checkpoint`
- `test_batch_worker_nested_progress_and_safe_cancel_reach_retained_dialog`

This verifies the CPU-safe cancellation path, worker token propagation,
single-shot UI behavior, nested progress, retained cancelled state, cancelled
manifest, and cleanup evidence on Apple Silicon. It does not claim an Apple GPU
runtime; the cancellation architecture is provider-neutral.

## Windows / GPU Evidence

The GPU branch includes a real-provider cancellation test that requests
cancellation during execution and checks cleanup before a partial pipeline or
output can be published. Windows remains subject to the same cooperative
limitation as macOS: an active non-interruptible native call can delay the final
cancelled state until its next safe boundary.

## Manual Cross-Platform Smoke

Use this short smoke test on both the Windows GPU machine and an Apple Silicon
Mac before release:

1. Open a batch with enough synthetic items, or a deliberately slow operation,
   to leave time for interaction.
2. Start the batch and confirm item and nested operation progress update while
   the rest of the GUI remains responsive.
3. Click `Cancel batch` after at least one item completes.
4. Confirm the control changes to `Cancelling...` and only one cancellation
   request is issued.
5. Confirm the run reaches a retained `Batch cancelled` state at a safe
   checkpoint.
6. Open each completed output and confirm no partial final destination exists.
7. Inspect the latest and archived manifests: completed, cancelled, and skipped
   items must agree, and runtime cleanup must be recorded.
8. Preflight and run a new batch successfully in the same application session.

## Known Limitation

Cancellation is cooperative, not immediate process termination. NumPy, SciPy,
scikit-image, file-reader/writer, or accelerator calls without internal
checkpoints may continue briefly. The UI must continue to say `Cancelling...`
until cleanup finishes; forcibly killing the worker would violate output and
accelerator-runtime safety.
