# VIPP 0.15.0a4

VIPP 0.15.0a4 is a small presentation-fix release: clearer workflow wires and easier comparison of overlapping histogram channels. It does not change scientific calculations or add new analysis nodes.

**Alpha software.** Preserve original data, workflows, environments and decisive outputs. Validate analyses on representative data before scientific interpretation.

## Clearer workflow wires

- Backward and same-column connections route outside their source and destination cards, including wide multi-input nodes and while dragging.
- Routing considers nearby cards, rounded detours and crowded layouts. Moving or resizing nodes updates their routes without changing workflow connections or calculations.
- Completely overlapping cards can still make an unobstructed path impossible. During dragging, connected cards remain protected; the full obstacle-aware route is refreshed when the drag ends.

## More readable overlapping histograms

- Detailed multichannel histograms draw faint fills first, then continuous stepped outlines. Dense vertical borders around every bin no longer obscure channels drawn earlier.
- PNG/TIFF exports use the same clearer appearance. Single-channel bars, underlying bin values, hover information, logarithmic zero gaps and dense-plot peak preservation remain unchanged.

## Compatibility and qualification

Workflow, batch-configuration and manifest schemas remain unchanged from 0.15.0a3. No GPU algorithms, reader routes, runtime dependencies or installer code change in this release. Recorded-run reproduction still identifies different VIPP versions and requires explicit acknowledgement; verified batch resume retains its existing cross-version restrictions.

The [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.15.0a4/docs/release-qualification-baseline.md) records the focused routing/histogram checks, visual review and carry-forward boundaries. Exact-release CI and artifact checks remain separate from scientific validation of a new dataset.

Use the [0.15.0a4 release page](https://github.com/rensutheart/napari-vipp/releases/tag/v0.15.0a4) for the Python distributions, Windows setup and separate Apple Silicon/Intel macOS installers with their checksums, and the [0.15.0a4 manual](https://rensutheart.github.io/vipp-mkdocs/0.15.0a4/) for instructions. Desktop installers are explicitly unsigned alpha builds; macOS packages are also unnotarized.
