# VIPP 0.16.0a9

VIPP 0.16.0a9 is a focused alpha bug-fix release for workflow recovery, canvas navigation, calculation progress and Crop previews. It combines both post-a8 bug-fix branches. Unfinished assistant/MCP and proposed Apple GPU work are not included.

## Recovery that keeps meaningful work

- An untouched **Untitled** workflow created at startup no longer produces a new recovery entry.
- Loading the starter preview does not count as editing the workflow.
- Editing the starter still enables its normal recovery checkpoints, including changes made during deferred startup.
- Existing historical recovery copies are not deleted by this change. Edited workflows and original saved files remain protected.

## A steadier workflow canvas

- Moving the pointer over the canvas no longer continues an interrupted drag, including while a node calculates.
- Panning requires a mouse press initiated on the canvas and the same button to remain held.
- Releasing the button, losing mouse ownership, hiding the canvas, changing focus or dropping a node ends the previous pan gesture.
- Palette drag-and-drop and inspector slider edits do not start a canvas pan.
- Routine UI resizing no longer pulls the graph toward the pointer.
- Finishing a quick calculation no longer restores an old viewport position over navigation you made while it was running.

## Calculation progress without layout jumps

- The bottom status bar reserves the styled progress controls' height before work begins, so the canvas does not jump when progress appears or clears.
- Switching between status messages and progress removes the outgoing content first, avoiding a brief extra wrapped row.
- Narrow docks and host/theme padding use the same steady-height behavior.
- Progress and Stop remain available; long warnings and actionable errors can still wrap instead of being clipped.

## More reliable Crop previews

- Crop selection guards the complete viewer-layer transition so dimensions and layer-list events cannot re-enter an unfinished update.
- Adjusting Crop margins reuses the unchanged source image instead of publishing it again unnecessarily.
- Unchanged ROI graphics properties are retained; actual data, geometry and calibration changes still update.
- A selected Crop uses its source image and ROI directly, avoiding a temporary Inspect layer that would immediately be removed.

## Compatibility and downloads

Workflow, batch-config and manifest schemas remain **6**. These fixes change UI and recovery orchestration, not scientific kernels, image results, CUDA algorithms or installer engines. Recovery still does not restore scientific caches, Undo history or in-flight work.

Windows x86_64 and separate macOS arm64/x86_64 installers are explicitly **unsigned alpha builds**. macOS packages remain CPU-only and unnotarized. Use the matching checksum files and the [0.16.0a9 manual](https://rensutheart.github.io/vipp-mkdocs/0.16.0a9/). See the [release plan and qualification boundaries](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a9/docs/release-0.16.0a9-plan.md).

[Full source comparison: 0.16.0a8 → 0.16.0a9](https://github.com/rensutheart/napari-vipp/compare/v0.16.0a8...v0.16.0a9)
