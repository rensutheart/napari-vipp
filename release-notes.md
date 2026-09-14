# VIPP 0.15.0a5

VIPP 0.15.0a5 makes examples easier to explore, adds a reusable colocalization mask, improves histogram and mesh displays, and simplifies desktop launching and updates.

**Alpha software.** Preserve original data, workflows, environments and decisive outputs. Validate analyses on representative data before scientific interpretation.

## Easier examples and colocalization

- Browse 22 examples in a searchable, resizable chooser with clearly separated list and description panels. Each example explains its input, what to explore and what you'll get, with short bullet lists and a suggested experiment. Compact spacing, wrapped text, keyboard opening and light/dark themes keep the chooser readable.
- RACC Colocalization now focuses on the RACC method, its paper and whole-image/selected-region colour maps. Colocalization, Overlap & Object Counts separately demonstrates standard metrics, white overlap views and counting connected overlap regions. RACC uses the Magma colour map, manual channel thresholds of 43,970.51 and 48,073.03, and a red-channel region threshold of 30,000.
- The new CPU-based Colocalization Mask node produces a Boolean mask of voxels at or above both channel thresholds, using native-intensity Manual or shared whole-image Costes thresholds. Connect it to cleanup, connected-component labelling and object measurements. Connected overlap regions are not automatically counts of whole organelles or biological events.
- Example graphs have more room around nodes, tunnels and notes while retaining their general analysis coverage. Existing saved workflow copies are not rearranged.

## Clearer displays

- Intensity histograms in the inspector grow vertically to accommodate titles, channel legends and axes, including narrow inspectors and larger fonts. The plotted data and export calculations are unchanged.
- Mesh display avoids unnecessary transform updates. Display failures retain the calculated result and show a compact warning with Details and Dismiss instead of filling the inspector with a traceback.

## Simpler desktop launch and updates

- Desktop installations use one VIPP shortcut and Auto compute selection; CPU/GPU choices remain available inside VIPP. The redundant CPU and Prefer-GPU shortcuts and the splash's Automatic badge are removed. macOS packages remain CPU-only.
- VIPP checks for updates on every startup unless disabled, retries transient connection failures and distinguishes cached version information from a successful fresh check. The update dialog separates version status, actions and preferences.
- Owned Windows desktop installations offer Download & open update: VIPP downloads the official installer, verifies its SHA-256 and installation ownership, then opens guided setup. It does not silently replace an environment or close unsaved work. Manual/plugin environments and other platforms keep their appropriate manual update routes.

## Compatibility and qualification

Workflow, batch-configuration and manifest schema versions are unchanged. The new node uses the existing schema and is unavailable in older VIPP versions. Existing GPU algorithms and reader routes are unchanged; the new mask node runs on CPU. Recorded-run reproduction still identifies different VIPP versions and requires explicit acknowledgement; verified batch resume retains its existing cross-version restrictions.

This release temporarily requires psygnal below 0.16 to avoid a reproduced event-callback regression affecting mesh displays and crop overlays. All eight affected tests pass with napari 0.9.1 and psygnal 0.15.1. Newer psygnal versions will be reconsidered after compatibility testing; scientific calculations are unchanged by this constraint.

The [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.15.0a5/docs/release-qualification-baseline.md) identifies changed domains, focused checks and carry-forward boundaries. Exact-release CI and artifact checks remain separate from scientific validation of a new dataset.

Use the [0.15.0a5 release page](https://github.com/rensutheart/napari-vipp/releases/tag/v0.15.0a5) for Python distributions, Windows setup and separate Apple Silicon/Intel macOS installers with their checksums, and the [0.15.0a5 manual](https://rensutheart.github.io/vipp-mkdocs/0.15.0a5/) for instructions. Desktop installers are explicitly unsigned alpha builds; macOS packages are also unnotarized.
