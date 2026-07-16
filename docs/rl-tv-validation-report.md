# Richardson–Lucy Total Variation scientific validation

Date: 2026-07-15
Status: **No definite production algorithm defect established; share with caveats**

## Executive finding

The reported pattern—plausible structures becoming blurred or disappearing in the first few iterations, then becoming clearer without fully recovering—is reproduced most strongly by **under-convergence from the constant initialization**, compounded by **TV regularization above the production default**. On the deterministic phantoms, initializing from the observed image reduced 5-iteration MSE by 24% in 2D and 26% in 3D and recovered more point, line, and dim signal. Increasing TV regularization monotonically reduced point/line/dim recovery even when global MSE improved.

The production default `lambda=0.002` was mild on these phantoms: relative to ordinary RL at 25 iterations, dim-structure recovery changed by -0.4 percentage points in 2D and -0.8 points in 3D, while MSE improved by 3.0% and 2.3%. The shipped examples are not default-like: the 2D workflow uses 18 iterations and `lambda=0.012`; the 3D workflow uses only 8 iterations and `lambda=0.008`. Those settings produced substantially lower feature recovery than the 25-iteration default.

PSF errors had the largest conditional impact. A one-pixel lateral PSF shift reduced point recovery from 0.830 to 0.023 in 2D and from 0.545 to 0.077 in 3D. A 25% spatial-sampling mismatch raised 2D MSE by 35% and reduced dim recovery from 0.739 to 0.487. VIPP's pipeline correctly rejects metadata-known sampling and dimensionality mismatches, but direct array calls cannot validate physical sampling, and the RL nodes do not themselves force odd shape or recenter a PSF.

The denominator sign matches the published RL-TV convention, RL-TV with `lambda=0` is bit-identical to ordinary RL, and the default denominator floor, TV epsilon, and filter epsilon were inactive or immaterial in this experiment. No production defaults or algorithms were changed.

## Reproducible evidence

- Analysis and fixture generator: [`scripts/validate_rl_tv_phantoms.py`](../scripts/validate_rl_tv_phantoms.py)
- Harness tests: [`src/napari_vipp/_tests/test_rl_tv_validation.py`](../src/napari_vipp/_tests/test_rl_tv_validation.py)
- Full 100-row results: [`docs/validation/rl-tv/results.csv`](validation/rl-tv/results.csv)
- PSF checks: [`docs/validation/rl-tv/psf-checks.csv`](validation/rl-tv/psf-checks.csv)
- Machine-readable summary and environment: [`docs/validation/rl-tv/summary.json`](validation/rl-tv/summary.json)

Reproduce from the repository root:

```powershell
.\.venv\Scripts\python.exe scripts\validate_rl_tv_phantoms.py
.\.venv\Scripts\python.exe -m pytest -q src\napari_vipp\_tests\test_rl_tv_validation.py
```

The script uses fixed seeds `20260715` and `20260716`. The recorded environment was Python 3.12.7, NumPy 2.4.6, SciPy 1.17.1, and scikit-image 0.26.0 on Windows 11.

## Experimental design

The 2D `64×64` and 3D `17×48×48` float phantoms each contain two point sources, a one-voxel thin line, a dim line near a bright line, and a structure within four samples of the field boundary. The 2D grid is `(0.10, 0.10) µm`; the 3D grid is `(0.30, 0.10, 0.10) µm`. Production Born–Wolf PSFs were generated at matching sampling with shapes `15×15` and `7×15×15`.

The forward model uses normalized PSFs, reflect-boundary convolution, fixed-seed Poisson noise at 100 photons at peak, and Gaussian read noise at 0.4% of the blurred peak. Reflect blur intentionally models signal continuing beyond the acquired field; the restoration comparisons then expose the production zero-extension assumption.

Measured outputs are MSE, PSNR, SSIM, total variation, restored/truth flux ratio, border-band MSE, and truth-normalized recovery for points, thin lines, the bright line, the dim line, and the border structure. Sweeps cover iterations, TV regularization, TV epsilon, denominator floor, filter epsilon, initialization, boundary handling, TV stencil, PSF scaling, PSF centering, PSF support, physical sampling, and photon regime.

## Baseline results

Recovery values are ratios of restored intensity to truth intensity on the exact feature mask; higher is better. MSE and border MSE are lower-is-better.

| Phantom / configuration | Iter. | λ | MSE | PSNR dB | SSIM | Point | Thin line | Dim line | Border MSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2D observed | 0 | 0 | 0.0047545 | 23.229 | 0.8373 | 0.173 | 0.399 | 0.472 | 0.0020327 |
| 2D ordinary RL | 25 | 0 | 0.0013571 | 28.674 | 0.9561 | 0.835 | 0.882 | 0.743 | 0.0019441 |
| 2D RL-TV default | 25 | 0.002 | 0.0013162 | 28.807 | 0.9564 | 0.830 | 0.879 | 0.739 | 0.0018879 |
| 2D shipped example | 18 | 0.012 | 0.0011170 | 29.519 | 0.9519 | 0.747 | 0.827 | 0.653 | 0.0011232 |
| 3D observed | 0 | 0 | 0.0015093 | 28.212 | 0.8156 | 0.062 | 0.102 | 0.209 | 0.0010457 |
| 3D ordinary RL | 25 | 0 | 0.0005363 | 32.706 | 0.9581 | 0.554 | 0.635 | 0.335 | 0.0005539 |
| 3D RL-TV default | 25 | 0.002 | 0.0005241 | 32.806 | 0.9576 | 0.545 | 0.628 | 0.327 | 0.0005326 |
| 3D shipped example | 8 | 0.008 | 0.0007645 | 31.166 | 0.8890 | 0.227 | 0.266 | 0.233 | 0.0003688 |

Total variation was 76.96 for 2D RL, 76.20 for default RL-TV, and 69.22 for the 2D example. In 3D it was 245.16, 242.26, and 163.10 respectively. Thus the example settings have a visibly stronger smoothing/convergence effect than the default.

Flux ratios were 1.09 in 2D and 1.31–1.32 in 3D for observed/RL outputs. This is expected for this deliberately adversarial boundary fixture: reflecting the near-boundary object adds out-of-field signal back into the acquired field. The ratio is therefore a boundary-model diagnostic here, not evidence that RL created flux.

## Ranked diagnosis

| Rank | Candidate | Likelihood for the reported behavior | Measured impact and interpretation |
|---:|---|---|---|
| 1 | Few iterations plus constant initialization | High | At 5 iterations, observed initialization reduced MSE 24% (2D) and 26% (3D). Dim recovery rose 0.407→0.501 and 0.219→0.269; 3D point recovery rose 0.126→0.240. Feature recovery continued to rise through 50 iterations even after MSE began worsening. |
| 2 | TV regularization strength | High for the shipped examples; moderate at the default | From λ=0 to 0.02, dim recovery fell 0.743→0.703 in 2D and 0.335→0.266 in 3D while MSE improved. The default λ=0.002 cost less than one recovery point; λ=0.008–0.012 was materially more suppressive. |
| 3 | PSF centering, sampling, or inadequate support | Very high impact; likelihood depends on PSF provenance | A one-pixel shift caused the largest failure. A 25% sampling mismatch materially degraded both phantoms. A too-small 3D PSF raised MSE 48% and reduced thin-line recovery 0.628→0.382. Generated, metadata-matched Born–Wolf PSFs passed all checks, so this is less likely for the exact shipped workflows than for imported PSFs. |
| 4 | Zero-extension boundary handling | Moderate, boundary- and iteration-dependent | At 25 iterations reflect padding reduced border MSE 37% in 2D and 15% in 3D. At 5 iterations it worsened border MSE 67% and 84%, respectively. It is not a safe unconditional replacement under the current initialization. |
| 5 | Dataset/noise-specific behavior | Unresolved | Photon sweeps changed absolute and feature-specific results non-monotonically. Only synthetic ground truth and one fixed seed per noise level were evaluated; real morphology and PSF aberration remain untested. |
| 6 | TV divergence discretization | Low-to-moderate | A forward/backward-adjoint comparison changed 2D MSE by -6.0% and 3D MSE by -0.6%, but slightly reduced feature recovery. This does not establish the production central-difference stencil as defective. Physical voxel spacing is not used in the production TV stencil, so anisotropic-TV behavior deserves a separate controlled study. |
| 7 | Denominator floor | Very low at current/default-example λ | Instrumented default runs had minimum raw denominators 0.9955 (2D) and 0.9932 (3D) with zero floor activation. Sweeping the floor from 0.001 through 0.5 produced identical results. |
| 8 | TV epsilon or filter epsilon | Very low at defaults | TV epsilon from `1e-12` to `1e-3` was immaterial. Filter epsilon `0`, `1e-12`, and `1e-6` were identical; only the extreme `1e-3` setting changed results. |
| 9 | Scalar PSF normalization | Very low for the reported loss | Multiplying the PSF by 3 and disabling normalization was numerically invariant because the scalar cancels between the forward blur ratio and adjoint correction. Normalization remains advisable for model semantics and validation. |

## Iterations and initialization

The early reconstruction trajectory supports the reported observation directly:

| Phantom | Iterations | MSE | Point recovery | Thin-line recovery | Dim recovery | Border MSE |
|---|---:|---:|---:|---:|---:|---:|
| 2D | 5 | 0.0028017 | 0.376 | 0.576 | 0.407 | 0.0009148 |
| 2D | 10 | 0.0015379 | 0.603 | 0.736 | 0.523 | 0.0007600 |
| 2D | 25 | 0.0013162 | 0.830 | 0.879 | 0.739 | 0.0018879 |
| 2D | 50 | 0.0020148 | 0.889 | 0.915 | 0.840 | 0.0037149 |
| 3D | 5 | 0.0010128 | 0.126 | 0.165 | 0.219 | 0.0005126 |
| 3D | 10 | 0.0006517 | 0.299 | 0.343 | 0.247 | 0.0003425 |
| 3D | 25 | 0.0005241 | 0.545 | 0.628 | 0.327 | 0.0005326 |
| 3D | 50 | 0.0006568 | 0.679 | 0.762 | 0.441 | 0.0008396 |

More iterations recovered exact-mask intensity but eventually worsened global and border MSE—ordinary early-stopping behavior, not proof that lost structures are hallucinated or permanently deleted. Constant `0.5` and constant positive-mean initializations were identical because a uniform positive initialization's scale cancels in normalized multiplicative RL. Observed initialization changed spatial structure, so it accelerated early recovery. At 25 iterations it retained slightly more feature intensity but was not uniformly better in MSE, which argues for a separate behavior-changing evaluation rather than a quiet default change.

## Equation and reference cross-check

The production update is equivalent to

```text
x[k+1] = x[k] * Hᵀ(y / (H x[k]))
                 / max(1 - λ div(∇x / sqrt(|∇x|² + ε²)), floor)
```

The minus sign and denominator placement match Dey et al.'s RL-TV formulation and the [ImageJ Ops deconvolution equation](https://imagej.net/imagej-wiki-static/Ops_Deconvolution). The [Dey et al. paper](https://doi.org/10.1002/jemt.20294) is the primary microscopy reference. The [ImageJ reference implementation](https://github.com/imagej/imagej-ops/blob/master/src/main/java/net/imagej/ops/deconvolve/RichardsonLucyTVUpdate.java) also multiplies by `1 / (1 - regularizationFactor * variation)` but uses a more specialized forward/backward/minmod-like 3D stencil and explicit axis steps, not NumPy's repeated central differences.

The smoothing sign is correct in production: an isolated peak has negative divergence, making the denominator greater than one and suppressing the peak; an isolated trough has positive divergence. This is now covered by a deterministic test. The denominator floor is a local numerical guard not present in the cited equation; it was inactive in all default instrumented runs.

RL-TV with `lambda=0` matched ordinary RL exactly in both 2D and 3D (`max_abs_difference = 0`). This is expected because both production paths use the same constant initialization, normalization, convolution, ratio guard, and non-negativity clamp; the TV branch is skipped at zero. A mismatch against another library would most likely come from initialization, boundary convention, PSF normalization/centering, clipping, or a different stopping implementation—not TV itself.

## PSF validation findings

The generated 2D and 3D PSFs passed sum, non-negativity, odd-shape, peak-at-center, zero centroid-offset, dimensionality, and matched-grid checks. Sums were `0.999999991` and `0.999999988`. Metadata-aware validation rejected a 25% spacing mismatch on every spatial axis and rejected wrong-rank PSFs.

Current behavior should be understood precisely:

- The pipeline validates image/PSF physical sampling when metadata are available and refuses implicit resampling.
- Direct array operation calls do not carry enough metadata to validate physical voxel spacing.
- The RL operations clean non-finite/negative PSF samples and optionally normalize the sum.
- Odd-shape enforcement and centering are provided by **Prepare / Validate PSF**; they are not automatically applied by the RL nodes.
- Born–Wolf generation already creates centered, odd, normalized kernels, so a centering failure is unlikely unless the PSF is shifted after generation or an imported PSF bypasses preparation.

## Recommendations

1. **Default λ:** retain `0.002` provisionally. Do not increase it. The synthetic evidence does not justify lowering the production default either; validate representative real datasets first. Operational guidance should start at λ=0 and add only the minimum regularization needed for noise control. Treat the shipped λ=0.008/0.012 examples as aggressive demonstrations, not recommended defaults.
2. **Initialization:** evaluate observed-image initialization in a separate behavior-changing PR. It materially improves low-iteration recovery, especially the 3D case, but changes convergence and is not uniformly superior at 25 iterations. Consider exposing initialization as an explicit option before changing the default.
3. **Padding:** do not globally replace zero extension with reflect padding yet. Reflect is promising at 25 iterations and near boundaries, but it regresses early constant-initialized runs. A future PR should compare zero, reflect, and preferably non-circulant/normalization-aware edge handling across initialization and iteration count.
4. **PSF sampling and centering:** retain mandatory pipeline grid checks. Add an explicit preflight status or warning for missing physical calibration, even PSF shape, and off-center peak/centroid; direct users toward **Prepare / Validate PSF**. Never resample a PSF implicitly.
5. **TV stencil:** do not change it on this evidence. Open a separate investigation for adjoint-consistent finite differences and physical-spacing-aware TV on anisotropic 3D data, with a reference implementation parity target.
6. **Numerical guards:** keep the present floor and epsilons. They are not credible causes at their defaults. Consider diagnostic reporting if the denominator floor activates, because that would indicate a qualitatively different regime.
7. **Real-data validation:** before any preset or default change, run bead data plus at least three calibrated biological volumes spanning sparse points, dim structures beside bright signal, and boundary-touching objects. Include multiple noise levels or acquisitions and blinded expert review.

## Proposed acceptance thresholds for a later behavior-changing PR

| Area | Proposed threshold |
|---|---|
| λ=0 compatibility | Maximum absolute difference from ordinary RL ≤ `1e-6 × input_peak` for identical options in 2D and 3D. |
| Numerical safety | All outputs finite and non-negative; no denominator-floor activation at default settings on either deterministic phantom. |
| Default feature retention | Relative to ordinary RL at 25 iterations, no point, thin-line, or dim-line recovery loss greater than 2 percentage points on either phantom. |
| Initialization change | At 5 iterations, reduce MSE by at least 15% on both phantoms and do not reduce any required feature recovery by more than 1 percentage point; also show no >5% MSE regression at 25 iterations. |
| Boundary change | At 25 iterations, improve border MSE by at least 10% on both phantoms; at 5 and 10 iterations, do not worsen global or border MSE by more than 5%. Current unconditional reflect padding fails the early-iteration criterion. |
| PSF validation | Reject wrong dimensionality and metadata-known sampling mismatch; warn or require explicit preparation for even shape or peak/centroid offset >0.25 voxel. No implicit resampling. |
| TV discretization change | Meet default feature-retention criteria, improve at least one predeclared global/border metric by ≥5% without worsening another by >2%, and document the discrete gradient/divergence adjoint convention and voxel-spacing treatment. |
| Real data | Pass on at least three calibrated biological datasets plus bead data, with no systematic dim-structure loss in blinded review and no material deterioration in bead localization/FWHM or border artifacts. |
| Reproducibility | Regenerate the CSV/JSON evidence without non-finite metrics; metric changes from the reviewed baseline should remain within 0.1% unless intentionally explained. |

## Verification performed

```text
Validation script: 100 result rows, 0 non-finite metric cells
PSF checks:        8 checks, 0 failures/unexpected acceptances
Harness tests:     5 passed
RL/PSF operations: 17 passed, 611 deselected
Grid/PSF tests:    4 passed, 15 deselected
Example workflows: 2 passed
RL-TV widget test: 1 passed
Ruff:              all added files passed
```

The only emitted test warning was the existing `ome_zarr.writer.Scaler` deprecation warning.

## Limitations

This study uses synthetic ground truth and intentionally mismatched boundary assumptions so each mechanism can be isolated. It does not include measured aberrations, background subtraction artifacts, bleaching, spatially varying PSFs, or biological expert labels. The photon-regime sweep uses one deterministic seed per level, so it is a sensitivity check rather than an uncertainty estimate. The evidence is sufficient to rank implementation-level causes and rule out the default numerical guards, but not to certify a new default, preset, initialization, or padding policy.
