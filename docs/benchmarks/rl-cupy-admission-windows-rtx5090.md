# CuPy Richardson-Lucy admission evidence

- Generated: `2026-07-29T21:46:15.679475+00:00`
- Schema: `napari-vipp-cupy-rl-admission-evidence` version 1
- Device: `NVIDIA GeForce RTX 5090`
- Platform: `Windows-10-10.0.19045-SP0`

This deterministic, machine-local scientific-parity evidence supports
public exposure inside the reviewed exact region on this development
branch. It is not a portable performance, cross-platform, or released-
package promotion claim, and it does not waive exact-workload parity.

## Policy gate

- NRMSE: `<= 2e-06`
- Maximum absolute error: `<= 1e-6 + 5e-6 × CPU peak`
- Shape, float32 dtype, finite masks, and complete finiteness must match.

## final odd 164

- Fixtures: **164**
- Manifest SHA-256: `3238340a510f933ebb0e5e12839e2bd4600b533f1c0326cd638cdd2ff6291641`

| Filter epsilon | Iterations | Failures | Worst gate score | Worst fixture |
|---:|---:|---:|---:|:---|
| 1e-08 | 10 | 0/164 | 0.432110591 | `sparse-sweep-s5009-128x129-sparse_noisy_beads` |
| 1e-08 | 25 | 0/164 | 0.864347739 | `sparse-sweep-s5009-128x129-sparse_noisy_beads` |
| 1e-08 | 50 | 4/164 | 1.22312068 | `core-s100-128x129-zero_heavy_dynamic_range` |
| 1e-07 | 10 | 0/164 | 0.375308839 | `core-s100-128x129-zero_heavy_dynamic_range` |
| 1e-07 | 25 | 1/164 | 1.15109703 | `sparse-sweep-s5069-128x129-sparse_noisy_beads` |
| 1e-07 | 50 | 5/164 | 2.16212373 | `sparse-sweep-s5063-128x129-sparse_noisy_beads` |
| 1e-06 | 10 | 1/164 | 1.27254791 | `sparse-sweep-s5055-128x129-sparse_noisy_beads` |
| 1e-06 | 25 | 1/164 | 3.60687394 | `sparse-sweep-s5025-128x129-sparse_noisy_beads` |
| 1e-06 | 50 | 6/164 | 9.25679433 | `core-s102-128x129-zero_heavy_dynamic_range` |

## provisional floor rejection 36

- Fixtures: **36**
- Manifest SHA-256: `d2a4b96a5c58c8c1cd10bf8de3137ae12b4793e4b45ed2a4798dda8788fe26ce`

| Filter epsilon | Iterations | Failures | Worst gate score | Worst fixture |
|---:|---:|---:|---:|:---|
| 1e-10 | 10 | 0/36 | 0.375308839 | `core-s100-128x129-zero_heavy_dynamic_range` |
| 1e-10 | 25 | 1/36 | 1.22063921 | `core-s102-47x53-zero_heavy_dynamic_range` |
| 1e-10 | 50 | 5/36 | 4.05980866 | `core-s102-47x53-zero_heavy_dynamic_range` |
| 1e-10 | 100 | 7/36 | 1.72648673 | `core-s100-128x129-zero_heavy_dynamic_range` |

## even psf comparison 40

- Fixtures: **40**
- Manifest SHA-256: `54cec35b037f3de98829b6fc18e8646c0209d2afde9b2d6325aec898bebd8865`

| Filter epsilon | Iterations | Failures | Worst gate score | Worst fixture |
|---:|---:|---:|---:|:---|
| 1e-08 | 5 | 6/40 | 1889.41126 | `even-study-s901-63x65-sparse_beads` |
| 1e-08 | 10 | 9/40 | 410758.296 | `even-study-s901-63x65-zero_heavy_dynamic_range` |
| 1e-08 | 25 | 14/40 | 488389.894 | `even-study-s901-63x65-zero_heavy_dynamic_range` |
| 1e-07 | 5 | 2/40 | 2.76559334 | `even-study-s900-63x65-zero_heavy_dynamic_range` |
| 1e-07 | 10 | 9/40 | 377209.218 | `even-study-s901-63x65-zero_heavy_dynamic_range` |
| 1e-07 | 25 | 14/40 | 409007.301 | `even-study-s900-17x19x21-zero_heavy_dynamic_range` |
| 1e-06 | 5 | 2/40 | 3.28602223 | `even-study-s900-63x65-zero_heavy_dynamic_range` |
| 1e-06 | 10 | 8/40 | 109.175747 | `even-study-s901-63x65-zero_heavy_dynamic_range` |
| 1e-06 | 25 | 14/40 | 494052.993 | `even-study-s901-63x65-zero_heavy_dynamic_range` |

## Evidence-backed initial contract

- `filter_epsilon == 1e-08`
- `iterations <= 25`
- every PSF extent is odd;
- normalized PSF and default-safe clipping/scale controls;
- public exposure on this development branch, limited to this exact
  reviewed region; and
- exact-workload CPU/GPU parity before timing or optimizer selection.

Higher epsilon values are not automatically safer: the ratio update has
a threshold branch, and the 1e-7/1e-6 comparison contains parity failures.
The authored CPU default `filter_epsilon=1e-12` remains unchanged and
uses CPU fallback. VIPP must never silently raise it to qualify a GPU run.
Cross-platform support and released-package promotion require their
own validation and are not claimed by this artifact.

Raw per-fixture metrics, environment versions, generator contract, and
source hashes are retained in the sibling JSON artifact.
