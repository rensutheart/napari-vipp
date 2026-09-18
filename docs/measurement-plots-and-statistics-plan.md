# Measurement Plots And Statistics: 0.16 Scope

Status: **batch measurement collection**, the first **Plot Results** slice
and descriptive **Statistics** are implemented and unreleased after
**0.15.0a5**. The approved **Results Workspace** joins their interfaces;
Statistics remains descriptive summaries only. This boundary supersedes the
earlier proposal's inference and uncertainty phases; they are not promised next steps.
Use [active planning](planning.md) for release order and the
[Statistics contract](statistics.md) for exact calculation and compatibility
rules. No release date or version bump is implied.

## Product Decision

Help a user explore measured objects, compare clearly defined descriptive
summaries, and export results without losing units, identity or the recipe.
Keep two general-purpose nodes under **Measurements -> Tables**:

| Node | Input and output | Approved role |
| --- | --- | --- |
| **Plot Results** | Measurement or summary table -> typed plot result | Grouped points, histograms, cumulative distributions and scatter; editable pop-out and sized figure export. |
| **Statistics** | Measurement table -> ordinary summary table | Counts, mean, median, sample SD, quartiles, IQR, min/max and sum at a declared observational level. |

Statistics expands **Summarize Measurements** in place. The saved operation ID
remains `summarize_measurements`. New nodes use a versioned descriptive recipe;
old workflows and direct calls retain version 1 until explicitly upgraded.
There is no new result type or separate node per statistic.

## Results Workspace: One Interface, Existing Nodes

The approved nonmodal **Results Workspace** opens from a table-producing,
Statistics or Plot Results node. **Data**, **Summary** and **Plots** place
controls beside the relevant table or figure. The inspector and ordinary
table/plot windows remain available; this is not a new all-in-one operation.

- Start from one table. Merge matching measurements or collect batch rows
  upstream; the workspace must not guess which relationship the user means.
- **Data source** browses exact table outputs, including filtered/merged tables
  and named ports. Viewing a summary scopes **Plots** to its direct connections;
  no connected plot is an empty state, not a fallback to another summary.
  **Plots for** switches this browsing context without editing the graph.
- Opening/browsing creates no nodes. **Add summary** and **Add plot** are
  explicit, undoable creation of ordinary Statistics/Plot Results nodes.
  Existing related nodes can be selected without copying their settings.
- All views of the same node edit one saved recipe and refresh together.
  Independent nodes sharing an input stay independent. Updates invalidate
  the changed node and its descendants, not cached upstream measurements.
- Plot input explicitly distinguishes original measurements from a summary
  table. **Change plot input** changes the graph connection, not a private window
  dataset. State the input row count and the meaning of the plotted values;
  summary rows must not masquerade as individual objects or be implicitly
  averaged again by image.
- Searching a table and hiding columns only change its view. They do not
  exclude observations, change Statistics/Plot Results inputs or silently
  narrow exported scientific data.
- Table export uses the existing full-table CSV/TSV path; figure export uses
  the current saved plot recipe and existing PNG/TIFF/SVG/PDF controls.
  Neither calculation nor opening a window writes data to disk.
- Keep a reserved activity area and show stale/failed state explicitly.
  Current-result export stays unavailable while results are out of date.

Numeric plot axes add **Auto** (default) or custom positive major tick
intervals. Grid lines follow the same ticks in all views and exports. This
does not change bins, measurements, summaries or categorical group spacing;
integer count-axis semantics remain intact. `x_tick_interval` and
`y_tick_interval` are saved PlotRecipe parameters. Log axes and categorical X
require Auto. Custom values must be finite and positive, integral for count
axes, and produce no more than 200 major ticks across the padded plot range.
Sub-precision or overly dense intervals fail explicitly rather than being
rounded, clamped or silently replaced by Auto.

## Explicit Boundary

The 0.16 scope stops at descriptive summaries. No inferential tests, p-values,
ANOVA, significance labels, confidence intervals, bootstrap uncertainty,
automatic test selection, or statistical-assumption diagnostics are included.
Do not add hidden test APIs or disabled test-menu scaffolding in anticipation
of a later phase. Any reconsideration requires a separate deliberate product
and scientific review; it is not a committed follow-up release.

Counts describe eligible observations, images and declared samples. Declaring
a sample ID does not establish biological independence. Neither a thousand
cells nor a hundred fields automatically means that many independent
experiments. Users can export the table and recipe for external analysis.

## Implemented Results Foundation

The [collection contract](measurement-collection.md) covers the explicit
**Collect measurement results** post-run action. It verifies recorded typed
outputs, preserves source/image/object identity, annotations and units, and
retains failed, missing, empty and excluded items in its review inventory.
Collection does not rerun measurements or guess historical CSV types.

Direct collection export supports CSV/TSV with an optional image-summary
companion, or Excel with **Measurements**, **Image summary** and **About this
collection** sheets. Saving a typed `.vipp-results.json` dataset and opening it
through **Table Source** are separate optional actions. The workflow stores the
external dataset path/hash, not its rows. These collection-specific Excel
exports do not imply Excel support for ordinary result tables.

**Merge Tables** is a horizontal join, not vertical collection. A single
image's measurement table can feed either results node directly; a batch
collection is not required. Across-batch analysis remains an explicit
post-run workflow, never hidden execution across per-image nodes.

The first **Plot Results** implementation provides Compare groups (points with
optional mean/median), Distribution (shared-bin histogram or cumulative
distribution), and Scatter. It distinguishes objects from equally weighted
image means. Inspector and nonmodal pop-out edit the same saved recipe, and
headless export produces sized PNG/TIFF or SVG/PDF figures. Display choices do
not rerun upstream segmentation, change measurements or establish independence.
Stale/failed figures cannot be exported as current results.

The bundled `plot-morphology` example contains one calibrated synthetic image
with 60 ellipses and four plot branches. Its designed area/intensity association
is demonstration data, not biological evidence. Paired/time-course,
count/fraction, violin and uncertainty views are not commitments of this scope.

## Approved Descriptive Statistics

Users choose measurements, optional group columns, statistics and an explicit
missing-value policy using connected-table controls. Numeric identifiers are
not automatically treated as measurements. Keep ordinary object exploration
possible without inventing image or biological-sample IDs.

| Level | Values summarized within each group | Weighting |
| --- | --- | --- |
| Objects | Eligible input measurement rows | Every object contributes equally. |
| Image means | Arithmetic mean of eligible objects in each image | Every contributing image mean contributes equally. |
| Sample means: equal images | Mean of image means within each declared sample | Images have equal weight within a sample; sample means have equal weight in the final summary. |
| Sample means: equal objects | Mean of all eligible objects within each declared sample | Objects have equal weight within a sample; sample means have equal weight in the final summary. |

Require explicit relevant identities and reject ambiguous identity/group
relationships. Missing measurements can be excluded and counted or cause an
error, according to the saved policy. Never turn missing values into zeros,
silently parse numeric strings, remove outliers, normalize, transform or
convert units. An empty image contributes no invented object or mean; its
collection inventory remains the place to review it.

Expose object totals, included/excluded measurements and exclusion reasons,
available image/sample counts, and the actual number of summarized units.
Sample SD is undefined with fewer than two eligible units in version 2; keep
that distinct from zero spread in a constant group. Quartiles use the explicit
linear method. The [implementation contract](statistics.md) fixes formulas,
empty-input behavior, output names and recipe migration.

## Persistence, Export And Execution

All scientific choices belong to node parameters and the cache identity.
Store the recipe and counts in the result as well. Output remains wide
`TableData`, preserving distinct measurement units and ordinary table
composition, inspection, CSV/TSV export and Plot Results compatibility.
Calculating the node never writes files. Existing explicit batch publication
and export paths remain responsible for overwrite and atomic-write policies.

The implementation is Qt-free CPU calculation behind the shared operation
registry. Interactive, batch, workflow reopen and generated-Python routes must
use the same versioned rules. No new GPU capability or statistics dependency
is needed. Reproducibility packages retain the workflow recipe, not measurement
tables or figures automatically.

## Acceptance

- Independently check known-answer counts/reductions, linear quartiles and
  sample SD; include signed, constant, tied and very small populations.
- Exercise unequal objects per image and unequal images per sample so pooling
  and equal-image weighting demonstrably differ.
- Verify empty/all-invalid/singleton groups, nonnumeric values, numeric
  strings, non-finite floats and integers unsafe to convert to float64.
- Reject absent/ambiguous required identities and conflicting groups; preserve
  repeated local object IDs in different images without using them as image IDs.
- Verify mixed measurement units, immutable inputs, deterministic results,
  explicit exclusion counts, missing-column errors and collision handling.
- Preserve unversioned/version-1 workflow, direct-call and generated behavior;
  test explicit upgrade, undo/redo and save/reopen of version 2.
- Check the actual inspector controls, result table, ordinary export and
  downstream plotting. Exercise batch/headless execution and stale results.
- Check workspace node creation/reuse, source switching, undo/redo,
  save/reopen, workflow-tab isolation and synchronized inspectors/windows.
  Opening a workspace must not create a node or recalculate an upstream image.
- Check automatic/custom tick intervals in interactive and headless rendering,
  including count, categorical, logarithmic and invalid/dense interval cases.
- Keep the companion manual concise, mark behavior unreleased after 0.15.0a5,
  and document exact evidence without claiming biological validation.

## Further Work Requires A New Decision

This scope is complete when the descriptive path is useful, tested and
documented. Additional plot families or inferential analysis may be discussed
separately, but there is no inference phase to proceed into automatically.
