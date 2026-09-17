# Seeded segmentation development qualification

2026-09-17; Windows 11, Python 3.12.9, CPU. This qualifies the unreleased
Propagation implementation and the scoped watershed examples. It is not a
passing whole-application, GPU, native-installer or cross-platform qualification.
The machine-readable [receipt](qualification.json) records package versions,
wheel identity, scientific checks and the limits of each test run.

## Scientific and packaging checks

- Four acquired 996-by-996 fields: exact Centrosome equality through shared
  execution and exported Python, zero differences across 3,968,064 pixels per
  path. See the [2D evidence](../cellprofiler-propagation/README.md).
- Seven acquired full mitochondrial volumes, two crops and an analytical
  phantom: exact watershed reference and repeat agreement. The phantom assigns
  all 25,770 foreground voxels correctly. See the [3D evidence](README.md).
- Random Walker was evaluated separately. It showed no established quality
  advantage and exceeded a 45-second limit on a mammalian crop; a public node
  remains deferred.
- The final 456 focused source tests cover the kernel, graph integration, cache
  identity, 3D examples, compute planning, documentation, example catalogue,
  chooser/launcher/layout and workflow-schema checks. They passed in the clean
  CPU environment after all integration corrections.
- A clean environment without system packages passed 104 installed-wheel
  feature/example tests, direct/exported reference checks, dependency consistency
  and manifest validation. The saved J05 real-data export also matched exactly.
  This includes execution of the Propagation lane from the installed showcase
  resource. Imported runtime modules came from the wheel, not the source checkout.
- Ruff, source manifest validation and distribution builds passed. The companion
  manual passed its 12 content contracts, 50 cross-repository routes and strict
  build; the changed task/reference pages were visually reviewed.

The clean CPU development environment and archived data are local review
artifacts, not a published release. The original user working checkouts were
not modified.

## Full-suite limitation and follow-up

The complete source suite was attempted in a venv that inherited the machine's
system packages: **1,048 failed, 8,907 passed, 376 skipped and 13 errors**, in
2,110.82 seconds. This historical run predates the final integration corrections.
All 1,061 failure/error sections were classified:

| Observed traceback signature | Sections |
| --- | ---: |
| Qt event-loop `NoneType.clear` exception, including all 13 errors | 973 |
| Missing CUDA NVRTC/cublas library | 82 |
| Unavailable CUDA runtime, truncated reason | 1 |
| New Centrosome dependency missing from mocked version dictionaries | 2 |
| New example/node absent from maintained example catalogues | 3 |

The five feature-related gaps were reproduced against the unchanged baseline
and corrected: version fixtures include Centrosome; the standalone 3D validation
example is documented under `examples/validation`; the bundled exhaustive
showcase includes a valid 2D Propagation lane and required sample.

Nine selected unchanged baseline cases produced five failures and four passes
in the inherited environment. Against the earlier clean installed feature wheel, the
same test names produced seven passes and two expected CuPy skips. These bounded
checks support the environment explanation for the sampled cases; the receipt
distinguishes that earlier wheel from the final 104-test artifact. Signature
classification does **not** establish that every other failure was preexisting,
or that test bodies obscured by Qt errors would pass. The complete suite was not
rerun in the clean environment. Focused post-correction results are recorded in
the receipt.

Full logs, the classified traceback index and comparison runs are retained in
the worktree's `.cache/propagation` directory. Biological segmentation accuracy,
Fiji equivalence and full paper reproduction remain outside this qualification.
