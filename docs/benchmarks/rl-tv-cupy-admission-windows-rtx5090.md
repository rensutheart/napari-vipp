# CuPy Richardson-Lucy TV admission evidence

- Device: `NVIDIA GeForce RTX 5090` (`cuda:0`)
- Python / CuPy: `3.12.9` / `14.1.1`
- Fixtures: `164` inherited adversarial + `96` independent holdout
- Status: **all reviewed developer-hidden gates passed**

The lambda-zero profile remains ordinary Richardson-Lucy and uses its
strict numerical policy. The positive profile is only the exact shipped
tuple (lambda 0.002, TV epsilon 1e-6, filter epsilon 1e-12, floor 0.05)
at exactly 10 or 25 iterations; this is not a continuous parameter claim.

## Numerical matrices

| Profile | Iterations | Cases | Failures | Worst gate score | Threshold-active cases | Floor-active cases | Minimum raw denominator |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Lambda zero / strict RL | 10 | 260 | 0 | 0.443692 | n/a | n/a | n/a |
| Lambda zero / strict RL | 25 | 260 | 0 | 0.942629 | n/a | n/a | n/a |
| Positive shipped default | 10 | 260 | 0 | 0.457442 | 173 | 0 | 0.992914379 |
| Positive shipped default | 25 | 260 | 0 | 0.443837 | 181 | 0 | 0.992867529 |

## Maintained microscopy phantoms

| Phantom | Iterations | NRMSE | Max feature delta | Max MSE/border/flux relative delta |
| --- | ---: | ---: | ---: | ---: |
| maintained-2d-phantom | 10 | 4.43002e-07 | 2.09422e-07 | 9.21031e-07 |
| maintained-2d-phantom | 25 | 5.24903e-07 | 1.95134e-07 | 2.10313e-06 |
| maintained-3d-phantom | 10 | 5.64592e-07 | 8.34465e-08 | 3.79565e-07 |
| maintained-3d-phantom | 25 | 7.59239e-07 | 9.66562e-08 | 9.48771e-07 |

## Interpretation

- Positive TV is nonlinear, so its versioned 0.5% numerical screen is
  separate from lambda-zero ordinary RL. Feature-recovery, MSE, border
  MSE, and flux gates prevent the aggregate tolerance from hiding a
  material microscopy regression on the maintained phantoms.
- Guard activity is diagnostic, not a reason to alter authored values.
  The evidence records threshold and denominator-floor activity for
  every positive-profile fixture.
- This single-machine artifact supports developer-hidden admission.
  Calibrated bead/biological datasets, blinded review, and Linux/laptop
  evidence remain required before public promotion.
