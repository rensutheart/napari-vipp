# VIPP 0.16.0a7

VIPP 0.16.0a7 fixes initial and recovered workflow previews and adds a choice of relative or absolute paths when saving workflow JSON. This is an incremental alpha patch; unfinished assistant and MCP features are not included.

## Load and recover useful results

- The welcome sample loads even when interrupted workflow copies are available. **Calculate all** also works for an untouched workflow that has no calculated results yet.
- **Recover selected** loads verified sources and rebuilds automatic processing results, such as Gaussian Blur and Otsu Threshold. These are fresh calculations, not cached arrays restored from JSON. Missing or changed inputs remain errors rather than being replaced with a sample.
- Recovered manual steps remain paused even if their saved Auto Recalculate setting is on. File-writing nodes and dependent branches also wait. Per-node **Calculate** authorizes that manual step; **Calculate all** explicitly resumes the normal workflow. Review output destinations before doing so.
- Opening a workflow at an unavailable manual result shows an available source initially without calculating that manual step. Saved graph selection is preserved.

## Choose how workflow paths are saved

Under **Settings → Workflow paths**, choose **Relative to workflow file** (the existing default) to move the workflow and its data together, or **Absolute file locations** to move only the workflow JSON while the data stay in place. The choice applies on the next Save, Save As or timestamped workflow save, including an attached Batch configuration. It does not move or copy data or change separate Python-export and Batch Save config behavior.

To convert an existing workflow, open it while its sources still resolve, select **Absolute file locations**, and save before moving the JSON. **Save workflow as…** already retains resolved input/output locations in either mode. A file moved alone with broken relative paths must first be reopened from its original location or have its sources relinked; VIPP cannot infer the lost base directory.

## Compatibility and limits

Workflow, batch-config and manifest schemas remain **6**. Scientific kernels, image interpretation, compute-provider implementations, dependency pins and installer engines are unchanged. Recovery does not restore source pixels, scientific caches, Undo history, in-flight work or Batch workspace/configuration; save separate Batch configuration as needed. Absolute paths require the same locations to remain accessible on the destination computer.

Desktop installers remain explicitly unsigned alpha builds. macOS packages are separate CPU-only arm64/x86_64 artifacts and are unnotarized; verify the matching checksums. The [release plan](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a7/docs/release-0.16.0a7-plan.md) and [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a7/docs/release-qualification-baseline.md) distinguish changed-domain checks from carried-forward evidence. See the [versioned manual](https://rensutheart.github.io/vipp-mkdocs/0.16.0a7/).

[Full source comparison: 0.16.0a6 → 0.16.0a7](https://github.com/rensutheart/napari-vipp/compare/v0.16.0a6...v0.16.0a7)
