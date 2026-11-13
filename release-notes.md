# VIPP 0.16.0a8

VIPP 0.16.0a8 improves the workflow editor, recovery, and 3D viewing, and adds faster Windows updates for compatible managed installations. This is an alpha prerelease; unfinished assistant and MCP features are not included.

## Workflows and recovery

- Inspector fields switch together between wide and narrow layouts. Scrolling past sliders, spin boxes, and closed menus no longer edits them. Port names and inspector headings wrap instead of clipping.
- **Connected inputs** identifies what is wired to each port. **Input metadata** keeps received data details separate, and **Node details** presents the selected node's configuration without repeating it in the inspector.
- Node-library search prioritizes direct matches. Source loading shows a compact progress banner, and results tables size columns to their contents by default.
- Select several interrupted sessions with checkboxes and recover them into unsaved tabs, or confirm a discard. Original workflow files stay unchanged; manual and file-writing steps stay paused.

## 3D viewing and large images

- **Viewer graphics** shows the active OpenGL renderer and lets you choose Auto, GPU, or CPU for the next standalone Windows launch. CPU mode needs software OpenGL support. This setting is separate from workflow **Compute**.
- Large intensity volumes can use a reduced 3D preview when needed; masks, labels and analysis results are not downsampled. If a full-resolution layer cannot be displayed in 3D, VIPP reports the problem and offers a 2D route. Review Images improves 3D and oblique navigation.
- Hysteresis Threshold avoids a full-volume float64 temporary while preserving thresholds, connectivity and the output mask.

## Windows setup

- **Automatic** chooses the fastest safe update method. **In-place** updates only the packages that need to change in an eligible managed environment. Close VIPP before proceeding and keep it closed until setup finishes. Unsupported or unverifiable installations still need **Replacement** or **Fresh**.
- **Replacement** stages a separate copy with compatible installed packages; **Fresh** builds a new environment. Setup can reuse verified downloads, and reports measured progress where available.
- A stale, verified Windows Installed apps entry can be backed up and repaired separately. Setup asks before applying the repair or an update. **Open VIPP** uses the installation's own runtime search paths without editing the global PATH.

## Compatibility and limits

Workflow, batch-config and manifest schemas remain **6**. Display changes do not change scientific results or compute selection. Recovery does not restore scientific caches, Undo history or in-flight work.

Desktop installers are explicitly unsigned alpha builds. macOS packages are separate CPU-only arm64/x86_64 artifacts and are unnotarized. Verify the matching checksums and see the [release plan](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a8/docs/release-0.16.0a8-plan.md) and [versioned manual](https://rensutheart.github.io/vipp-mkdocs/0.16.0a8/).

[Full source comparison: 0.16.0a7 → 0.16.0a8](https://github.com/rensutheart/napari-vipp/compare/v0.16.0a7...v0.16.0a8)
