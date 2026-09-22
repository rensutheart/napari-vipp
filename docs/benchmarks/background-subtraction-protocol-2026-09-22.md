# Background subtraction: comparison protocol

Status: completed comparison; see the
[reviewed findings and limitations](background-subtraction-2026-09-22.md).
The original memory sampler observed only the Windows interpreter launcher;
this run's memory measurements and runtime memory-cap claim are invalid.
Operation timers and output comparisons were recorded inside the actual worker.

## Questions

1. How long does the complete Subtract Background operation take on this machine?
2. How much faster or slower are alternative background estimators?
3. How much do corrected values, dim/broad structures and fixed-threshold regions
   change? Agreement with today's VIPP output and recovery of known synthetic
   truth are separate questions.

## Compared methods

| Method | Purpose |
| --- | --- |
| VIPP CPU rolling ball | Current reference operation |
| VIPP GPU rolling ball | Current GPU implementation; include transfer costs |
| SimpleITK flat-ball opening | Alternative estimator with rounded spatial support |
| SimpleITK flat-box opening | Alternative estimator with rectangular support |
| SimpleITK opening by reconstruction | Alternative estimator with reconstruction |
| SciPy flat-box opening | Same-method control for the SimpleITK box implementation |

The three SimpleITK candidates are **not equivalent implementations of rolling
ball**. A shared radius does not make their geometric/intensity conventions
identical. A fast candidate could merit a separately named node or method, but
this investigation does not add one or silently change existing workflows.

## Shared operation policy

Preserve VIPP's original-input subtraction, finite-value preparation, optional
3-by-3 nearest-boundary mean presmoothing, background polarity, negative clipping,
integer rounding and dtype restoration. Use float32 working values except for
float64 input. Do not substitute white top-hat on a smoothed image: that would
also change which image is subtracted from the estimated background.

XY mode invokes each plane independently, including reconstruction. True 3D
uses a volumetric estimator. The benchmark has explicit spatial dimensionality;
it does not infer channels or mix time/channel dimensions into spatial filtering.
The box control uses matched safe-border padding; other methods retain their
documented boundary conventions. Radius is in voxels, not physical units.

## Coverage and measurement

The executable case matrix lives in
`scripts/benchmark_background_comparison.py`. It covers small/medium/large YX
images, XY stacks, true ZYX volumes, multiple radii, presmoothing on/off, uint16
and count-scale/normalized float32 data, both background polarities, and sparse,
crowded, broad, ramp and vignette phantoms. The run's JSON records exact cases.

Run one method at a time in a fresh child process. Fixture construction is outside
timing. Record the first operation call separately, then three warm complete-call
repetitions and their median/range. First-call time includes lazy algorithm imports
and GPU initialization/compilation where applicable, not interpreter startup or
fixture creation; existing disk caches are not deleted. GPU timings include upload,
download and synchronization. These are not GPU-resident pipeline timings.

Use an isolated environment with recorded dependency versions and source hashes.
ITK thread limits are set before worker imports and on filter instances. Record
sampled host-process peak memory separately from GPU memory (not measured).
Do not run the regression suite or competing benchmark jobs simultaneously.

Apply explicit per-worker time and host-memory limits. Preserve partial timing
samples and reference output when available. A timeout is censored work, not a
measured runtime; missing reference outputs cannot produce numerical comparisons
or invented speedup ratios. Checkpoint after each method. Resume only compatible
source/settings/case matrices; retain prior evidence on interruption.

## Scientific differences

Use the same deterministic input for every method. Record output shape/dtype,
exact agreement, mismatch fraction, MAE, RMSE, maximum error, range-normalized
RMSE and correlation. Do not confuse high correlation with equivalence.

Synthetic images have known additive foreground and background before input
noise/quantization. Compare retained foreground, background residual, and each
generated object's retained signal, including the weakest-retained object.
Overlap labels have one owning object; they do not separately attribute summed
signals in overlapping pixels. Retention above one can include residual background.

Apply one fixed truth-derived threshold to every corrected output and compare
mask Dice and connected-region counts. These are sensitivity diagnostics, not a
biological-accuracy claim or a threshold optimized for each method. Noise and
quantization create a common truth-error floor. Selected cases also retain raw
unclipped/unrounded diagnostics so clipping cannot conceal estimator differences.

Representative image comparisons must use shared display ranges for comparable
outputs, signed difference scales and identified slices. Interpret normalized
float results separately from count-scale results: rolling-ball geometry includes
an intensity dimension, unlike flat morphological opening. Do not average those
conditions into a single universal quality or speed score.

## Decision after measurements

Report actual runtimes, variability, transfer policy and incomplete cases; identify
where speed comes from a library implementation versus a different algorithm.
Describe failure modes before proposing defaults. Synthetic evidence alone does
not justify changing a user's existing biological analysis. No production node,
installer, active environment, commit or release is changed by this benchmark.

## Primary references

- [scikit-image rolling-ball explanation and intensity scaling](https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_rolling_ball.html)
- [SimpleITK grayscale opening](https://simpleitk.org/doxygen/v2_5/html/classitk_1_1simple_1_1GrayscaleMorphologicalOpeningImageFilter.html)
- [SimpleITK opening by reconstruction](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1OpeningByReconstructionImageFilter.html)
- [SciPy grayscale opening](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.grey_opening.html)
