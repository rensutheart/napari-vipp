# VIPP 0.16.0a10

VIPP 0.16.0a10 is an alpha patch for reopening recent workflows, recovering from a changed image reader, and editing Crop bounds on Windows. It combines the ready non-AI branch changes since a9. All AI workflow-authoring work remains excluded for 0.17.

## Reopen recent workflows by full path

- Hover over **Open Recent** in the three-dot Workflow actions menu to see your latest successfully opened or saved workflows.
- Full paths distinguish workflows with the same filename in different folders.
- The list holds up to ten workflows, newest first, and stays available between launches.
- Paths on disconnected drives or offline network shares remain listed; opening the menu does not probe those locations.
- Selecting an item follows normal workflow-opening safeguards and opens it in a new tab rather than replacing existing work.
- Failed or cancelled opens and bundled examples do not enter recent history.
- **Clear recent workflows** forgets only the list. It does not delete your workflow files or reset last-used folders.

## A clear action when an image reader changes

- A workflow saved with a different reader implementation or version now shows a short warning explaining the problem.
- **Details** keeps the saved and current reader versions available without filling the status bar with technical text.
- **Refresh file sources** explicitly reloads the current workflow's file sources with the current reader; the saved binding is not silently accepted or changed.
- An old warning cannot refresh a different workflow tab. A failed batch preview that restored another representative does not offer to rebind that other sample.
- After refreshing, check channels, axes, calibration and results: a different reader can interpret the same file differently. Collection verification remains a separate check.

## Live Crop bounds without another OpenGL overlay

- Windows Intel and unrecognised graphics use a guarded Crop display that avoids adding a second image volume or a Shapes overlay to the OpenGL scene.
- The guarded route shows an orange rectangle over a 2D slice or projection, and a wireframe over the 3D input volume, with a compact inspector bounds schematic.
- The uncut input stays visible around the proposed bounds so you can see what will be removed.
- In 2D, a red outline means the displayed slice or projection misses the retained region.
- Rotation, zoom, view changes and selection updates keep the outline registered to the image; leaving Crop hides it.
- A held slider drag changes only the proposed bounds. Releasing applies the scientific crop; it does not copy the preview into the result.
- Grabbing a slider at a nonzero margin no longer commits an extra first pixel or splits the edit into two history steps.
- Refreshing an already ordered viewer no longer needlessly moves managed layers through the layer list.
- Unsupported or unavailable canvas projections hide the outline rather than leaving stale geometry. The bounds schematic remains a separate, clearly labelled editing aid.

## Compatibility, scope and downloads

The existing napari compatibility checks now include the new Crop helpers and guarded presentation regressions. Scientific kernels, full-resolution results, CUDA providers, workflow/batch/manifest schemas, runtime dependency pins and installer engines are unchanged. Crop presentation is not scientific resampling, and this patch does not claim biological validation or every graphics-driver combination.

An internal Apple GPU feasibility proposal is included as planning only. It adds no Apple GPU implementation, dependency or newly supported runtime route. macOS installers remain CPU-only.

Windows x86_64 and separate macOS arm64/x86_64 installers are explicitly **unsigned alpha builds**; macOS packages are also unnotarized. Use the matching checksum files, the [0.16.0a10 manual](https://rensutheart.github.io/vipp-mkdocs/0.16.0a10/), and the [release plan and qualification boundaries](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a10/docs/release-0.16.0a10-plan.md).

[Full source comparison: 0.16.0a9 → 0.16.0a10](https://github.com/rensutheart/napari-vipp/compare/v0.16.0a9...v0.16.0a10)
