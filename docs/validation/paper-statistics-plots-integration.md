# Paper workflows with existing statistics and plots

This unreleased integration combines the recorded CellProfiler compartment
profile from `feat/statistics-paper-reproduction` (`c11c1eb`) with the existing
measurement collection, Statistics, Plot Results, Table Source and Results
Workspace implementation from `feat/collect-measurement-results` (`1f5d692`).
It adds no statistical algorithm and does not change the CellProfiler kernels.

The integration lives in a separate `feat/paper-statistics-plots` worktree.
Neither source branch, its working tree, nor the previously qualified paper
environment was modified.

## Integration adjustments

- Preserve both branches' changelog entries and the nine-lane example catalog.
- Add an object mean-intensity distribution to the exhaustive example so it
  includes the Plot Results operation. Table Source keeps its explicit external
  dataset exception.
- Keep that example's historical summary at `summary_version=1`. Regenerating
  it with the new version-2 defaults would otherwise change its recipe.
- Update the exhaustive graph's direct-edge count and scientific hash. The
  historical graph hash still matches after removing the new display branch
  and compartment lanes.
- Add a bridge regression: official CellProfiler 4.2.6 fixture nuclei feed
  intensity measurements, a descriptive Statistics table and Plot Results.
  The reference labels remain exact; the summary count, mean and sample standard
  deviation match direct calculations from the fixture pixels.

## Checks on 18 September 2026

The initial focused selection covered 1,680 tests across CellProfiler, batch,
measurement collection/export, Table Source, Statistics, plots, Results
Workspace, exhaustive examples and workflow schema goldens:

- 1,658 passed; one optional real-CUDA batch test skipped.
- Three exhaustive-example expectations needed the integration adjustments
  above. The final rerun of all three affected test files, including the new
  bridge regression, passed **139 tests**.
- Eighteen batch UI geometry assertions failed. All eighteen were reproduced
  against the unchanged `1f5d692` source in the same dependency environment:
  four in `test_batch_check_progress_ui`, three in `test_batch_item_commands`,
  four in `test_batch_override_page_style`, three in
  `test_batch_override_reset_actions`, and four in
  `test_batch_results_presentation`. The first four also failed with Qt's native
  Windows platform. They are recorded inherited UI issues, not a passing full
  suite or a claim that the batch UI is completely qualified.
- The separate computational `test_batch.py` run passed **82 tests**.
- The documentation contract suite passed **26 tests**.
- Ruff and npe2 manifest validation passed. Wheel and sdist build succeeded.

No complete repository suite was rerun. No unrelated UI repair is included.
Detailed local logs and JUnit receipts are under
`.cache/paper-statistics-plots/`; the directory is intentionally untracked.

## Local runtime boundary

The new runtime virtual environment has its own installed integration wheel,
XlsxWriter and build tools. A `.pth` file reuses the prior qualified environment's
dependency directory read-only, so the local runtime depends on that directory
remaining present. This is a local review environment, not a portable installer.

Recorded scientific packages are NumPy 2.5.1, SciPy 1.18.0, scikit-image 0.26.0
and Centrosome 1.3.4; the UI uses napari 0.9.1, PyQt6 6.11.0 and Matplotlib
3.11.2. XlsxWriter is 3.2.9. No GPU qualification was performed here.

Paper-specific workflow authoring, cohort batch execution, and public-manual
updates are companion work. This record establishes integration behavior and
does not extend the earlier paper agreement claims to untested outputs.
