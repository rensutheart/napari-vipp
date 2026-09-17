# Statistics-paper compartment reproduction

Implementation and evidence record for the unreleased CellProfiler compartment
profile. Public task instructions live in the companion
[manual](https://github.com/rensutheart/vipp-mkdocs), on the
`docs/statistics-paper-reproduction` branch.

## Source contract

The target is [Marcotti et al., 2026](https://doi.org/10.1242/jcs.264367),
with the [authors' pinned code](https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility/tree/a81d20c9393485e57395cd4a1be9e7ff83c7f11d)
and [Zenodo v2 measurement tables](https://zenodo.org/records/15584545).
The actual archived `Experiment.csv` records identify CellProfiler **4.2.6**
and contain their executed pipelines. These records take precedence over the
generic pipeline's date/revision or an inferred application version. Stages
5–9 share the same settings in both studies and all three YAP/TAZ plates.

The acquired cohort contains **376 fields in 22 wells**: 40 fascin fields in
ten wells and 336 YAP/TAZ fields in twelve wells. The reference contains
**20,298 matched nuclear/cytoplasmic cell pairs**. These are selected paper
cohorts, not the entirety of either original screen.

Raw TIFFs and numerical evidence are outside Git, under
`D:/VIPP-paper-reproductions/`. Reference input files remain unchanged. Every
review OME-TIFF is a lossless derivative with an explicit CYX declaration and a
source-channel map. IDR0028 pages 0/2/3 are Hoechst/Actin/YAPTAZ despite the raw
TIFF metadata presenting the four acquisition pages as time. IDR0139 names use
DNA `A01...C01`, Actin `A03...C04`, Fascin `A02...C02`, and nuclear actin
`A04...C03`. A-channel and C-channel suffixes are not interchangeable.

## Numerical profile

The graph explicitly converts uint16 to float32, preserving values, then maps
the fixed range 0–65535 to 0–1. This agrees with CellProfiler's image loading
for all 65,536 possible uint16 values; there is no per-image min/max scaling.

| Stage | Recorded behavior |
| --- | --- |
| Smooth DNA and actin | Gaussian artifact diameter 2; sigma 2/2.35; constant-zero filtering normalized by the filtered valid mask; float32 output |
| Nucleus threshold | Global Li/Minimum Cross-Entropy computed before threshold smoothing; tolerance max(minimum distinct spacing/2, 0.5/65536); scale 1.3488 gives sigma 1; correction 1; bounds 0–1; foreground includes equality |
| Nuclei | Shape/Shape declumping; 15–50 pixel diameters; eight connectivity; automatic reduced-resolution maxima; hole filling; size/border exclusion; fixed local random seed 0 |
| Propagation seeds | Retain the unedited border-touching nuclei as competitors; remove excluded interior nuclei; preserve the accepted-nucleus output separately |
| Cells | Global Li actin mask without threshold smoothing; Centrosome propagation at 0.05; fill labelled holes; map to retained-nucleus IDs and remove excluded-border competitors |
| Cytoplasm | Subtract shrunken nuclei; the one-pixel Centrosome nuclear outline remains in cytoplasm |
| Measurement | Existing VIPP Measure Objects + Intensity on each compartment and original normalized signal |
| Per-cell ratio | Join by image and object ID; nuclear mean / (nuclear mean + cytoplasmic mean) |

The ratio follows the authors' executed Python code and archived measurements.
It is not nuclear/whole-cell mean, integrated nuclear/total-cell intensity, or
nuclear/cytoplasmic mean. Compartment areas do not enter this ratio.

Six new nodes expose the missing stages around the existing Propagation node.
They implement this fixed CellProfiler profile, not every configuration offered
by the original application. All require a single, explicitly declared YX
plane. Multichannel, temporal and volumetric inputs need an explicit selection;
multiple label inputs must share a sampled physical grid. Diameters and
distances are pixels even when calibration is carried. Normalized guidance is
finite float32 in [0,1]; exactly one float32 ULP above 1 is accepted unchanged
because the reference Gaussian may produce this bounded rounding overshoot.
There is no clipping or inferred normalization.

Both nucleus outputs retain YX geometry, calibration and label metadata through
planning, execution, workflow saving and exported Python. Inputs are not
mutated. Cancellation is checked before and after blocking native kernels.
Centrosome is included in scientific history and cache dependency identity.
The local RNG does not alter NumPy's global random state. The upstream BSD
notice accompanies the adapted code and packaged `NOTICE`.

## Independent verification

The reference executable is the official Windows CellProfiler 4.2.6 build,
extracted into an isolated local directory. It runs the embedded author
pipelines on the acquired raw TIFFs. Diagnostic array/measurement capture does
not alter image-processing settings. The reference uses Python 3.8.10,
NumPy 1.23.1, SciPy 1.9.0 and scikit-image 0.18.3. VIPP uses its supported modern
Python environment; exact installed versions are recorded with each run.

The committed synthetic fixture is independently generated by that executable,
with its input generator, provenance and licence. Stage-by-stage checks include
smoothed channels, thresholds, unedited and retained nuclei, cells and
cytoplasm. Integration tests check the two-port metadata/preflight contract,
grid and axis rejection, shared execution, CPU registry, cache dependencies,
and exported Python. The exhaustive showcase has a separate CellProfiler lane.

The acquired-image comparison deliberately separates three claims:

1. VIPP versus independent CellProfiler on identical current inputs.
2. Current input file bytes versus the authors' archived MD5 records.
3. VIPP measurements/statistics versus the authors' deposited output tables.

The final independent comparison verifies all **376** fresh per-field execution
receipts and artifact hashes under one code/runtime contract. All **1,128**
final compartment label maps agree at all **455,209,920** compared pixels;
all **93,246** compartment mean intensities agree exactly with the independent
CellProfiler outputs. Label values agree; VIPP's documented int32 output dtype
can differ from CellProfiler's smaller integer representation.

One observed intermediate difference is recorded for fascin J05 F003: modern
float32 reductions shift the Li actin threshold from
0.005051327869296074 to 0.005051319487392902. One mask pixel differs; final
nucleus, cell and cytoplasm labels are unchanged. A native iteration trace
locates the first difference in the initial float32 mean. No field-specific
threshold correction is introduced. A second one-pixel actin-mask difference
occurs at YAP/TAZ plate 2B, C13 F023: thresholds
0.0004991571186110377 and 0.0004991571768186986 straddle an equality boundary.
Its final compartments also agree exactly. Therefore full intermediate bitwise
equivalence across NumPy builds is not claimed.

The YAP/TAZ scalar comparison covers all 9,550 reference cells in 336 fields.
Fascin requires a provenance qualification: only N12 and O02 have all source
files identical to the author records (32/160 channel TIFFs). The remaining
128 hashes are absent from the full pinned author image table; a fresh FTP
download confirms the local acquisition. Author full-table versus deposited
subset comparison is exact, and all 17 shared Zenodo v2/v3 CSVs are byte
identical. These checks do not establish why the other source files differ.
Their object IDs must not be assumed to pair across segmentations.

The final run-level counts, per-cell errors, centroid checks, sampling curves,
figures and source receipts are in the local evidence directory's comparison
report. The repository retains a compact
[qualification receipt](statistics-paper-results.json) with source hashes and
the [primary-well comparisons](statistics-paper-primary-comparison.csv).
No biological ground-truth accuracy or independent experimental
replication is claimed by numerical pipeline agreement.

## Statistical analysis boundary

The original paper performs segmentation in CellProfiler and subsequent
analysis in Python. VIPP therefore provides the image graph and compartment
tables; the companion scripts perform the authors' keyed ratio calculation and
sampling recipes. No general inferential-statistics UI is added here.

The reconstruction preserves the 50/200-cell swarmplot seed schedules,
50/200/500-cell per-well superplots, successive ten-cell cumulative sampling,
100-draw IQR curves at sizes 10–490, and standardized mean differences using
the control sample SD. Sampling membership and curve values are saved.
Cells and fields remain subsamples; the well/plate structure stays explicit.

Published Table 1 counts, means, medians and SD are recoverable from the
reference measurements. All eight quartiles match using Weibull/exclusive
percentiles, while the notebooks use the linear convention. This is a
compatible explanation, not proof of the undocumented table-generation code.
The published SEM column is not explained by per-cell SEM or the SEM of the
four field means. It is reported as unresolved, not silently substituted.

## Reproduction assets

- `scripts/acquire_statistics_superplots.py`: remaining 96 fascin TIFFs with
  source receipts, checksums and complete TIFF decoding.
- `scripts/reproduce_statistics_paper.py`: native graph generation, stable
  source bindings, per-field workflow JSON/exported Python, intermediates,
  compartment tables, provenance and final-source verification.
- `scripts/recompute_statistics_paper_reference.py`: author-table joins,
  descriptive and sampling reference reconstruction.
- `scripts/compare_statistics_paper.py`: numerical comparisons and figures.
- `D:/VIPP-paper-reproductions/validation/statistics-paper/`: executed evidence;
  the top-level README identifies review workflows and exact rerun commands.

Application checks and installed-wheel qualification are recorded in the
handoff evidence accompanying this implementation. The original application
and manual checkouts are intentionally separate from these worktrees.

The full CPU test attempt completed with 10,100 passes, 459 skips and three
change-related integration failures. Correcting the exhaustive source/planning/
bypass fixtures exposed a nondeterministic history entry: runtime progress
objects were being rendered alongside scientific settings. History now records
only the explicit settings for each compartment operation. The final affected
five-file rerun passes 317 tests (one optional real-CUDA test skipped); the
isolated installed wheel passes all 206 compartment tests. The final source was
not subjected to a second complete suite. The full image cohort was freshly
replayed after the metadata fix so its execution records identify the final
implementation. The HDD qualification record preserves the full logs.

The reference-recomputation script was independently replayed into a fresh
directory: all 34 scientific CSVs, three notebook inventories and the descriptive
JSON are byte-identical to the audited reference outputs. The generated notebook
records the actual script and explicit source/output locations.
