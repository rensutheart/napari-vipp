# Measurement Plots And Statistics: 0.16 Plan

Status: planned addition to the **0.16 release series**, not implemented.
Requested and reviewed: 2026-09-10. No alpha milestone or release date assigned.
This complements, rather than replaces, the
[registration, comparison, and template-matching plan](registration-and-template-matching-plan.md).
Use [active planning](planning.md) for release order.

## Product Decision

VIPP should help a biologist go from measured objects to an understandable
figure and a small, explicit statistical summary without leaving the workflow.
It should not become a general replacement for Prism, R, Python, or a
statistician. Make exploration, experimental-unit awareness, and reproducible
presentation the core; add a bounded set of inferential methods only after
their scientific contracts are qualified.

The requested interface is **two general-purpose nodes**, not one node per
plot or statistical test:

| Node | Input and output | Role |
| --- | --- | --- |
| **Plot Results** | Measurement or summary table -> typed plot result | Choose fields and a plot family, inspect the figure, open an editable pop-out, and export it. |
| **Statistics** | Measurement table -> results table | Choose measurements, groups, independent samples, summaries, and supported comparisons. Its output can feed Plot Results or ordinary table export. |

Place both under **Measurements -> Tables**, beside the existing table tools.
Statistics should expand the existing **Summarize Measurements** node, not
compete with it. Preserve its saved operation identity and older calculations;
new scientific semantics need a versioned recipe and deliberate migration.
No new top-level category or extra node for each export format is needed.

## Scientific Rationale

[Senft et al. (2023), *A biologist's guide to planning and performing
quantitative bioimaging experiments*](https://doi.org/10.1371/journal.pbio.3002167)
emphasizes choosing measurements and statistical analysis for the biological
question, stating the level of comparison, and showing individual observations
alongside useful summaries. This is the basis for the safeguards below, not a
claim that the paper prescribes this UI or a universal test menu.

[Lord et al. (2020), *SuperPlots: Communicating reproducibility and variability
in cell biology*](https://rupress.org/jcb/article/219/6/e202001064/151717/SuperPlots-Communicating-reproducibility-and)
provides a useful model for showing both within-sample variation and variation
between independent experiments. Plotting every cell does not make every cell
an independent experimental replicate.

The following feature choices are VIPP design proposals. A workflow with 1,000
cells from three independent experiments must not silently become a test with
1,000 independent replicates. Conversely, VIPP must not assume that every
experiment has the same hierarchy: the user declares the relevant independent
unit for their question.

## Data And Batch Collection Come First

Existing `TableData` provides immutable rows, column units, table kind, and
source name. Existing tools select columns, add metadata, merge tables,
and summarize groups. These are useful foundations, but there is not yet a
formal experimental-design contract or a cross-item batch table collector.
**Merge Tables is a horizontal join, not vertical collection of batch rows.**

Plan three explicit scopes:

1. **One image:** connect its measurement table directly to either node.
2. **Each batch item:** the same graph calculates a separate figure/summary
   for each item; explicit Batch Output settings publish those results.
3. **Across the batch:** an explicit **Collect measurement results** action in
   Run & results collects compatible tables into a reusable table dataset.
   The user can annotate samples and conditions, then analyze that dataset
   with the same two nodes in a results workflow. This is a post-run step,
   not hidden cross-item execution inside a per-image node.

Before implementing scope 3, settle the durable table-dataset source/open
contract. Prefer extending existing data-opening/source affordances over a
family of new analysis nodes. A results workflow must be able to reopen the
collection, rerun without Qt, and report missing tables; an in-memory handoff
alone is not sufficient.

Collection must retain source item, source image, local object ID, measurement
operation/settings, and available condition/sample/replicate/time annotations.
Use composite object identities: label 1 in different images is not the same
object. Offer a reviewable annotation grid (and optional imported metadata)
instead of expecting users to edit JSON. Never infer biological replicates
from filenames without an explicit, saved mapping.

Check schemas and units before collection; do not silently mix pixel and
micrometer measurements or coerce incompatible columns. Freeze the exact
included result revisions and identify failed, missing, skipped, stale, or
excluded items. Collection after a resumed run must not duplicate completed
items. Do not silently recalculate images just to open a collected table.

## Shared Analysis Controls

Use column pickers populated from the connected table, with names, units, and
small example values. Keep the ordinary path simple; reveal design controls
when grouping, replicate summaries, or inference needs them:

- **Measurements:** one or more numeric columns; categorical IDs are not
  automatically offered as measurements simply because they contain numbers.
- **Groups / conditions:** optional categorical columns and a saved order.
- **Independent sample:** a column identifying the independently sampled or
  treated unit. Explain with examples such as animal, culture, or experiment.
- **Within each sample:** explicit aggregation across cells and, when present,
  images. Make pooling cells versus first summarizing images distinguishable;
  unequal cell yield must not silently weight experimental replicates.
- **Paired by:** required identity for paired comparisons, not table row order.
- **Missing or invalid values:** an explicit policy with included/excluded
  counts and reasons. Never silently interpret missing measurements as zero.

Allow descriptive, object-level exploration without inventing replicate IDs.
Label it clearly and withhold replicate-level confidence intervals and tests
until the user declares independence or explicitly confirms that rows are
independent. A declaration is recorded user input, not an automatic guarantee.
Show object/image counts and independent-sample counts separately.

No implicit outlier removal, normalization, unit conversion, or analysis on
log-transformed values just because a plot axis uses a logarithmic scale.
Any supported analytical transformation must be explicit, saved, and reflected
in results and labels. Leave general data wrangling outside this initial scope.

## Plot Results

### Initial plot families

One plot-type selector should cover the following, with only relevant controls
visible. Start with a single figure and optional facets, not a full poster editor.

| Family | Typical bioimage use | Important controls |
| --- | --- | --- |
| Grouped points with summaries | Area or intensity across conditions | Individual points, replicate identity, mean/median; optional box-and-whisker summary with its definition stated. |
| Distribution: histogram or ECDF | Object sizes, intensities, distances | Shared bins/range across groups; count versus proportion/density with units; ECDF avoids a bin-width choice. |
| Scatter | Area versus intensity, morphology relationships | X/Y columns, group colour, optional facets; no automatic regression or significance claim. |
| Paired points / ordered lines | Matched samples or measurements over time | Pair/trajectory ID and explicit X/time column; no invented pairing, sorting, or interpolation. |
| Counts / fractions | Objects in defined categories | Explicit category and denominator, with per-sample versus pooled counts distinguished. |

Prefer visible observations and replicate summaries over mean-only bar charts.
Offer a **Show independent samples** presentation inspired by SuperPlots; this
does not imply that every plot requires inference. Violin plots can follow once
bandwidth, sample-size limitations, and small-group handling are explicit.

### Inspector and pop-out

The node inspector selects plot type, fields, groups, and basic appearance and
shows a real plot preview once available. **Open plot...** opens a nonmodal,
resizable window. Its settings remain editable there: titles/axis labels,
units, axis scale/limits, group order, colours, point size/opacity, legend,
summary/error-bar choice, facets, and export size. Disabled options explain
what is missing instead of producing a blank plot.

Inspector and pop-out edit **one saved recipe**, including undo/redo and
save/reopen. Presentation edits apply immediately to cached measurements, with
no upstream segmentation rerun. Changes to bins, aggregation, selection, or
statistical methods invalidate the appropriate derived result. Stale results
remain visibly stale; asynchronous work supports cancellation and cannot
overwrite a newer recipe. Reuse histogram/colocalization pop-out interaction
patterns, not an independent settings store.

Use accessible palettes, distinguish groups with more than colour where
practical, preserve meaningful channel/condition names, and keep units in axis
labels. Large scatter/point displays may use deterministic display sampling,
but must show displayed/eligible counts and record the seed. Scientific
summaries always use the full eligible data, not a display sample.

## Statistics

### Core: descriptive results

Provide a friendly multi-select of count, mean, median, sample SD, quartiles,
IQR, min/max, and sum where meaningful. Report total rows, valid values,
excluded values, and independent-sample n separately. Preserve units and
group/measurement identity in a typed, exportable results table.

Existing Summarize Measurements already provides many of these reductions.
Do not blindly inherit its edge cases: it currently drops values that cannot
become finite floats and returns sample SD = 0 at n = 1. New recipes should
report exclusions and undefined SD with a clear insufficient-data reason;
older saved workflows must retain deliberately versioned behavior.

Uncertainty is optional, never an unlabeled error bar. Distinguish spread
(SD/IQR) from uncertainty in an estimate (confidence interval). A bounded
bootstrap CI may be added with its method, confidence level, iteration count,
seed, and independent resampling unit recorded. Paired samples stay paired;
nested cells are not independently resampled as experimental replicates.

### Bounded inference: only after design and validation gates

Candidate first comparisons are **Welch's two-sample t-test** for two independent
groups and **paired t-test** for declared pairs. Pearson/Spearman association
is useful for measurement relationships, with observational level stated and
no object-level p-value pretending to describe independent experiments.
Rank-based two-group alternatives can follow with their assumptions and null
hypotheses explained; they are not universal assumption-free median tests.

Select a question/method explicitly. Do not choose tests automatically from a
normality test, try every test, or default to significance stars. Show group
estimates, an appropriate effect estimate and available uncertainty, effective
n, method/options, exact p-value when applicable, and useful diagnostics.
Reject or mark undefined insufficient samples, constant inputs, and incomplete
or duplicated pairs according to a declared policy. Multiple emitted tests
need an explicit comparison family and reviewed multiplicity adjustment.

Inferential methods are **not a reason to delay the useful descriptive/plotting
foundation**. Review their exact menu before coding that phase. Mixed-effects
models, general ANOVA/post-hoc suites, survival analysis, power/sample-size
planning, Bayesian models, and automatic statistical advice are out of scope.
Export tidy data and methods so specialists can continue elsewhere.

## Rendering, Export, And Reproducibility

Matplotlib and SciPy are already dependencies. Prefer a shared Matplotlib
renderer and bounded SciPy statistics over adding a large plotting/statistics
framework. Existing histogram/scatter exports render Qt widgets: their export
code alone does not provide headless or vector figures.

Introduce versioned, serializable, **Qt-free** analysis/plot recipes and result
contracts. Retain the exact data revision and derived table, not a live Figure,
widget, or pickled runtime object. A plot is not an image with pretend spatial
axes. Render the same recipe in the pop-out and through headless export.

Plan explicit **PNG/TIFF and SVG/PDF** export with physical/pixel size, DPI,
background, and fonts controlled independently of the current window size.
Offer the plotted/summary data as CSV/TSV and a compact methods/recipe sidecar.
Explain units, n, exclusions, transformations, aggregation, and error bars
without requiring a technical README to interpret the figure.

Calculating a node does not write a file. Inspector export and explicitly
configured Batch Output publication should use the same figure artifact
contract, format filtering, cancellation, overwrite, and atomic-publication
rules. Plot ports, cache accounting, history, workflow persistence, generated
Python/CLI, and Batch Output support are real implementation work, not assumed
to exist already. Statistics stays compatible with ordinary table export.

Reproducibility packages retain plot/statistics recipes, workflow notes,
software versions, and appropriate provenance. Do **not** start bundling raw
measurement tables, source images, generated figures, or other result files
by default; the author still shares those separately. Missing externally
shared table collections must have a clear reconnect/verification path.

## Delivery And Acceptance

Recommended incremental delivery within 0.16:

1. Versioned design/recipe contracts, migration rules, and a reopenable,
   identity-preserving batch-results collection/annotation path.
2. Statistics descriptive mode and Plot Results grouped points, distributions,
   and scatter; shared inspector/pop-out state and headless/raster/vector export.
3. Paired/time/count views, replicate-aware uncertainty, documentation, and
   end-to-end collection examples. Gate individual modes on their contracts.
4. Review and qualify bounded inference; defer any unqualified method rather
   than suggesting that a p-value proves a biological conclusion.

Acceptance must include unequal cells/images per replicate; repeated local
object IDs across images; missing/duplicated pairs; mixed units/schemas;
empty, nonnumeric, NaN, constant, tied, and very small groups; reproducible
jitter/CI seeds; and visible excluded/failed batch items. Verify collection
after resume, workflow reopening, input immutability, stale/cancel behavior,
and that display sampling never changes statistical results.

Review known-answer summaries/comparisons independently, not just against the
same library call. Visually review small/large figures, light/dark themes,
overlapping groups, colour accessibility, and exported sizes across desktop,
plugin, and headless paths. Include a guided synthetic example with objects
nested in images and independent samples, plus the same data exported for
external analysis. User review of the column/sample selectors and pop-out is
required before considering this workflow complete.
