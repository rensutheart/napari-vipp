# Contributing to napari-vipp

Thank you for helping improve napari-vipp. The project is an early alpha, so
small, focused changes with clear tests and documentation are especially
valuable.

## Before You Start

- Search the [existing issues](https://github.com/rensutheart/napari-vipp/issues)
  before opening a new one.
- For a substantial feature or a change to workflow compatibility, open an
  issue before investing in an implementation.
- Report suspected vulnerabilities privately as described in
  [SECURITY.md](SECURITY.md), not in a public issue.

By participating, you agree to follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## Development Setup

VIPP supports Python 3.12 and newer. Create an isolated environment and install
the editable package with its development dependencies.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

On macOS or Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run a development instance with:

```bash
python scripts/launch_vipp_sample.py
```

The [developer notes](docs/developer-notes.md) and
[architecture reference](docs/architecture.md) explain the main extension
points and internal boundaries.

## Making A Change

1. Create a branch from the current default branch.
2. Keep the change narrowly scoped and preserve unrelated work.
3. Add or update tests for observable behavior and regressions.
4. Update user documentation when behavior, terminology, or limitations change.
5. Add a concise entry under `Unreleased` in [CHANGELOG.md](CHANGELOG.md) for a
   user-visible change.

For scientific operations, also document:

- the algorithm and authoritative reference;
- accepted axes, data types, units, and physical-scale assumptions;
- 2D, 3D, and leading-dimension behavior;
- edge cases and known limitations; and
- validation against an analytical phantom, trusted reference implementation,
  or licensed benchmark dataset where appropriate.

Do not describe a method as validated, reproducible, or equivalent to another
tool unless the repository contains evidence that supports that exact claim.

### Scientific behavior requirements

Scientific correctness takes precedence over convenience. A contribution must
not silently make an invalid request look successful. In particular:

- Treat axis names/types, physical scale, units, origin, and grid alignment as
  part of an image input. Equal array shapes do not prove equal physical grids.
- Never resample, reorder, project, normalize, clip, cast, repair a parameter,
  or substitute a dimensional interpretation implicitly. Expose the policy,
  record it in carried history/provenance where it affects results, and reject
  ambiguous requests when no scientifically safe default exists.
- State whether an algorithm is exact, sampled, approximate, iterative, or
  truncated. Sampling and approximation controls must be visible to users and
  covered by tests; a display optimization must not enter a scientific result.
- Preserve input buffers. Tests for a new operation should include read-only
  arrays and verify that cached upstream data is not changed. If returning an
  alias or view is intentional, document and test that contract explicitly.
- Test dtype and numeric-range behavior, including NaN/Inf, signed values,
  boolean data, and wide integers when supported. Any lossy conversion needs an
  explicit user-visible policy.
- Presentation layers, thumbnails, histograms, and viewer settings must remain
  detached from scientific cache arrays and cache keys unless they are an
  explicit operation parameter.
- File and live-layer inputs must enter background work as stable revisions.
  Batch output publication must happen only after every source revision has
  been reverified; failed or cancelled work must not leave apparently complete
  outputs.
- Workflow/schema, metadata history, batch manifests, atomic persistence, and
  cancellation/stale-result behavior are part of reproducibility. Update their
  tests whenever a change affects those contracts.

For a bug fix that replaces a silent behavior, prefer a clear failure with an
actionable message over preserving backward compatibility. Add a regression
that demonstrates both the former hazard and the intended scientific contract.

## Required Checks

Run the focused tests for the code you changed while developing, then run the
complete checks before requesting review:

```bash
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest
python -m build
```

Qt tests run through a virtual display in Linux CI and with Qt's offscreen
platform on Windows CI. If a UI test fails only in CI, include the failing job,
Python version, traceback, and any available screenshot or Qt diagnostics in
the issue or pull request.

## Pull Requests

A reviewable pull request should explain:

- the problem and why the selected approach addresses it;
- user-visible changes and compatibility implications;
- tests and manual checks performed;
- scientific references or validation evidence, when applicable; and
- remaining limitations or follow-up work.

Material use of generative AI should be disclosed in the pull request. The
contributor remains responsible for checking generated code, citations,
licenses, tests, and scientific claims.

Contributions are submitted under the repository's
[BSD 3-Clause License](LICENSE).
