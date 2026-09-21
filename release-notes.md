# VIPP 0.16.0a1

VIPP 0.16.0a1 brings measured objects, descriptive summaries and figures into a connected Results Workspace, alongside reproducible seeded segmentation and CellProfiler compartment workflows.

**Alpha software.** Preserve original images, saved workflows and decisive outputs. Check representative data before scientific interpretation.

## From measurements to results

- Collect verified batch measurement outputs without reprocessing images. Retain source identities, units and the inventory of empty, failed or excluded items. Export CSV/TSV or Excel; optionally save a typed VIPP dataset for Table Source.
- Statistics summarizes objects, image averages or declared sample averages with explicit grouping, weighting and missing-value handling. Choose counts, averages and spread; singleton sample SD is undefined rather than zero. Existing summaries retain their earlier calculation rules until explicitly upgraded.
- Plot Results provides grouped points, distributions, cumulative distributions and scatter plots. Edit a shared recipe in the inspector, plot window or Results Workspace; export sized PNG/TIFF/SVG/PDF figures. Numeric axes offer automatic or explicit tick spacing, and counts use whole-number ticks.
- Results Workspace combines Data, Summary and Plots with a visible Workflow → Data source → Statistics node → Plot selection bar. Source relationships remain explicit, edits stay synchronized, stale outputs cannot be exported as current, and plot previews fit the visible window.
- Give nodes descriptive custom names or use automatic labels. Names remain consistent across the graph and results selectors; renaming does not change calculations or figure titles.
- Use spreadsheet-style Increase Decimal and Decrease Decimal controls throughout Results Workspace and table windows. Round floating-point display values from 0–15 decimal places; integers stay unchanged, hover reveals the original value, and calculations, sorting and exports retain full precision.

## Reproducible seeded segmentation

- Add the CPU-based Grow Regions from Seeds — CellProfiler Propagation node, with explicit aligned 2D inputs and pinned Centrosome behavior. Existing Marker-Controlled Watershed remains the volumetric method.
- Add six bounded CellProfiler compartment-profile stages for smoothing, thresholding, nuclei, propagation seeds, cell finishing and cytoplasm. Preserve retained and unfiltered nuclei separately and make pixel-scale assumptions explicit.
- Include synthetic demonstrations and recorded independent numerical comparisons. Agreement with a reference implementation does not establish biological segmentation accuracy or complete historical paper reproduction; documented Fascin input differences remain unresolved.

## Desktop refinements

- Separate workflow actions from settings and standardize dialog action ordering for each platform.
- Show live Windows package-installation output with a stable scroll position and an explicit Jump to latest action.
- Improve plot labels, integer count axes, busy indicators, compact notices, measurement selection and light/dark readability.

## Scope and compatibility

Statistics remains descriptive: no hypothesis tests, ANOVA, p-values, significance labels or confidence intervals are added. Registration, image comparison and template matching are later 0.16 work, not features of this alpha.

New nodes and parameters require the new version; retain copies before resaving older workflows. Reproduction version acknowledgements and verified-resume restrictions remain in force. macOS remains CPU-only. Desktop installers are unsigned alpha builds, and macOS packages are unnotarized.

See the [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.16.0a1/docs/release-qualification-baseline.md) for evidence boundaries and the release assets for platform checksums. Local demonstrations do not establish biological accuracy or complete historical paper reproduction.
