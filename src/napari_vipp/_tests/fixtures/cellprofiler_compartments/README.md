# Independent CellProfiler compartment fixture

The CC0 synthetic input generator creates four uint16 stain images, with four
retained nuclei, an excluded border nucleus and a small excluded nucleus.
`cellprofiler-4.2.6-fixture.npz` contains those inputs and reference stage outputs
captured from the official CellProfiler 4.2.6 Windows executable. The VIPP
implementation was not used to create the reference outputs.

The reference used the authors' exact IDR0139 embedded pipeline at
[Enhancing-Reproducibility commit a81d20c](https://github.com/FrancisCrickInstitute/Enhancing-Reproducibility/tree/a81d20c9393485e57395cd4a1be9e7ff83c7f11d),
recovered from `inputs/cell_profiler_outputs/idr0139/Experiment.csv`.
A diagnostic observer recorded arrays without modifying the segmentation;
threshold masks were replayed using the original module's `apply_threshold`
and its measured threshold. `fixture-provenance.json` records per-array hashes,
the fixture hash, parameters measured by CP, runtime versions and the official
installer hash. The fixture and generated inputs are CC0-1.0.

The published profile is Gaussian diameter 2; primary minimum-cross-entropy
threshold with smoothing scale 1.3488; shape declumping and shape watershed;
diameter range 15–50, automatic maxima suppression, reduced-resolution maxima,
size/border rejection and hole filling; Actin minimum-cross-entropy mask;
Propagation regularization 0.05; secondary hole filling and retained-nucleus
remapping; and cytoplasm subtraction with nuclear outline preservation.

The test compares every segmentation stage exactly. This validates agreement
with the named CellProfiler runtime on the fixture, not the biological
correctness of the segmentation or agreement with published image measurements.
