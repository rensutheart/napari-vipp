# VIPP Alpha Release Runbook

Last reviewed: 2026-08-04

This runbook covers publishing napari-vipp to PyPI, creating a GitHub release,
publishing the companion documentation site, and confirming discovery on
napari hub.

## Scope

- Target package version: set `<version>` from the release milestone before
  starting; do not reuse the current package version by accident.
- Current prepared target: `0.13.0a1`.
- Release maturity: Alpha
- Distribution channels: PyPI, GitHub release, napari hub index

## 1. Preconditions

1. You have push/tag permission on GitHub for this repository.
2. You have upload permission for the `napari-vipp` project on PyPI.
3. You have a PyPI API token available as `TWINE_PASSWORD`.
4. You have a clean git working tree on the release commit.
5. The companion `vipp-mkdocs` repository has a reviewed release page and a
   clean, pushed release commit.

Never paste, print, commit, or place the PyPI token in shell history. Load it
into the process environment without echoing it and clear it after upload.

Recommended local tools:

- Python 3.12 or 3.13 for the CPU package; CPython 3.12 for CUDA qualification
- `python -m pip install -U build twine`

## 2. Verify Metadata

Confirm these are set:

- `pyproject.toml` version matches the target version
- `src/napari_vipp/__init__.py` exposes the same target version, if it carries
  an explicit version
- `pyproject.toml` classifier includes `Development Status :: 3 - Alpha`
- `pyproject.toml` license is `BSD-3-Clause`
- README has a clear alpha disclaimer
- README has a clear license section
- CHANGELOG has an `Unreleased` section or a dated section for the target
  version with the release highlights
- the companion documentation version and release page match the target

Required checks:

- `python -X utf8 -m npe2 validate src/napari_vipp/napari.yaml`
- `python -m ruff check .`
- `python -m pytest`
- the release-candidate branch and final `main` commit are green on every
  configured operating system and Python version

For a batch/provenance release, also use `Batch workspace...` -> `Open batch
demo...`. Move through all three representatives with the slider and confirm
that both paired sources and downstream previews change together without
changing the workflow hash. Complete the generated run and confirm three
completed items, nine outputs, retained workspace statuses, and a finalized
latest manifest, archive, and three item sidecars.

For a release containing durable GPU execution, repeat the batch smoke with a
mixed admitted CPU/GPU pipeline and inspect the manifest rather than relying on
badges alone. Confirm BatchConfig/manifest schema 2, configured and effective
requests/hashes, exact implementation version/parity identity for every
computed node, per-output execution digest links, and true cleanup. Run the
saved `vipp_batch_pipeline.py --progress` once with its recorded request and
once with an explicit mode or node override. Confirm both item and operation
progress, then cancel a third run and verify cancelled status, no unpublished
output, finalized manifests/checkpoints, and exit code 130.

Also replay the same graph with `Prefer GPU`. Confirm every scientifically
eligible reviewed public provider is used without requiring a CPU-speed win,
an unsupported dtype/parameter node receives an explained ordinary CPU
decision, no cast or parameter rewrite occurs, and the manifest records
`prefer_gpu`, visible fallback, and the exact actual implementation per node.
Confirm per-node preferences remain stored but inactive, the Custom-only
`Find fastest pipeline…` action is hidden, and an explicit strict Prefer-GPU
request is rejected before calculation or publication.

Also run a CPU-only environment smoke with no CuPy/cuCIM installed: import the
plugin and an exported script, replay CPU and Auto workflows, and run a fully
skipped batch without an accelerator import/probe. Use deterministic tests to
exercise visible one-retry OOM provenance, strict no-retry behavior, and
cleanup-false publication blocking; do not induce a real OOM in an uncontrolled
operator run.

For a release that changes runtime or responsive UI code, also confirm the
batch representative strip remains usable at a 420 px dock width on Windows
and that macOS cache status reports RAM without launching a subprocess. The
automated suite must exercise Windows, macOS, and POSIX dispatch without
assuming `os.sysconf` exists on Windows.

### Bounded Pre-Feature Windows Evidence

On 2026-08-04, the operator reported that the native-Windows napari UI smoke
passed on development commit `ff21040`. The local schema-4 Custom workflow
loaded the representative private ND2 acquisition and exercised the intended
channel, navigation/display behavior, and backend-badge presentation. This is
an operator attestation only: the retained last-run JSON was later overwritten
by an Auto run, so it does not independently preserve the exact mixed-backend
assignment or cleanup result. Treat the pass only as bounded UI regression
evidence for that editable checkout.

That smoke predates the addition of `Prefer GPU`; it did not qualify that mode,
an immutable release artifact, the CPU-only path, saved batch/CLI replay,
cancellation/OOM behavior, or another platform. The Prefer-GPU change also
changes the candidate source commit. Build the final wheel and source archive
from the eventual immutable release commit and repeat all applicable CPU, CUDA,
interactive, batch, generated-CLI, provenance, fallback, cancellation, and
cleanup checks. The operator checklist below therefore remains unchecked.

## 3. Freeze And Tag The Exact Release Commit

Only tag after the versioned candidate is reviewed, pushed, clean, and green in
GitHub Actions. Confirm that it contains the current `origin/main` and record
the immutable commit id:

```powershell
git fetch origin --tags
git status --short
git merge-base --is-ancestor origin/main HEAD
git rev-parse HEAD
git tag -a "v<version>" -m "napari-vipp <version> alpha"
git rev-parse "v<version>^{}"
```

The two commit ids must match. Never move or recreate a published release tag;
prepare a new version if the tagged candidate needs code changes.

## 4. Build And Qualify Tagged Artifacts

With `HEAD` still at the clean tagged commit, build into an empty
version-specific directory. Do not use a broad root-level `dist/*` upload glob:

```powershell
python -m pip install -U build twine
python -m build --outdir "dist/<version>"
python -m twine check "dist/<version>/*"
Get-FileHash -Algorithm SHA256 "dist/<version>/*"
```

Expected output artifacts:

- `dist/<version>/napari_vipp-<version>.tar.gz`
- `dist/<version>/napari_vipp-<version>-py3-none-any.whl`

Using a version-specific directory prevents an upload command from including
artifacts from an older release. It does not make an existing same-version
artifact safe: confirm both files were produced from the tagged commit, record
their SHA-256 hashes, and repeat the clean-wheel CPU and CUDA acceptance smokes
against these files rather than an editable checkout.

## 5. Build And Publish Documentation

In the companion `vipp-mkdocs` repository:

```powershell
python -m pip install -r requirements.txt
python -m mkdocs build --strict
```

Review the rendered `0.13.0a1` release page, workflow-schema upgrade guidance,
batch workspace instructions, architecture boundaries, and known limitations.
Commit and push the docs release before or alongside the package release, then
confirm the hosted documentation resolves from the `Documentation` project URL.

## 6. Publish Tag, Package, And GitHub Prerelease

Push the already-qualified immutable tag first:

```powershell
git push origin "v<version>"
```

Then publish exactly the two hash-recorded artifacts. PyPI uploads cannot be
replaced, so recheck the directory and version before entering credentials:

Set token in the shell (PowerShell):

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "<pypi-api-token>"
python -m twine upload "dist/<version>/*"
```

Post-upload validation:

- Open https://pypi.org/project/napari-vipp/
- Confirm the target version is visible
- Confirm README renders alpha disclaimer
- Confirm license metadata shows BSD-3-Clause terms
- Confirm `Requires-Python` and optional GPU extras match the tagged metadata

Prepare the release notes body below in a temporary file, then create the
release with GitHub CLI (or use the equivalent GitHub UI fields):

```powershell
gh release create v<version> --prerelease --verify-tag --notes-file release-notes.md "dist/<version>/napari_vipp-<version>.tar.gz" "dist/<version>/napari_vipp-<version>-py3-none-any.whl"
```

If using the GitHub UI:

1. GitHub repository -> Releases -> Draft a new release
2. Tag: `v<version>`
3. Title: `napari-vipp v<version> (Alpha)`
4. Mark as pre-release: enabled
5. Attach the wheel and source distribution built from the tagged commit
6. Add release notes (suggested template below)

Write release highlights from the target CHANGELOG section. The following is a
structure, not a reusable list of 0.11 features:

```markdown
## napari-vipp v<version> (Alpha)

This is an early alpha build and is still in active development.

### Important
- Breaking changes are expected between releases.
- Validate outputs before publication or production use.
- This release is distributed under the BSD 3-Clause License.

### Highlights
- Summarize the target release's user-visible changes.
- Call out workflow/schema compatibility changes.
- Link new guides, validation reports, or examples.
```

## 7. napari Hub Listing/Refresh

napari hub indexes packages from PyPI metadata for napari plugins.

After PyPI upload:

1. Wait for napari hub index refresh (can take some time).
2. Check: https://napari-hub.org/plugins/napari-vipp
3. Confirm:
   - plugin appears
   - version updates to the target version
   - README/disclaimer is visible

If not updated after indexing delay:

- Verify `napari.manifest` entry point in `pyproject.toml`
- Verify `src/napari_vipp/napari.yaml` is included in the wheel/sdist
- Re-check PyPI metadata and release files

## 8. Post-Release Follow-up

1. Announce alpha status clearly in repository and release channels.
2. Open a tracking milestone for issues found in the released alpha.
3. Plan next versioning strategy.

## Operator Checklist

- [ ] Versioned release-candidate branch CI passes on Python 3.12 and 3.13 for
      Windows, Linux, and macOS
- [ ] Final `main` release commit CI passes and exact commit id is recorded
- [ ] Tests pass locally
- [ ] Manual UI smoke pass completed for graph search, tunnel manager, graph
      notes, insert-on-wire mapping, workflow save/load, cache modes, and
      example workflows
- [ ] Deterministic batch demo completed with 3 items, 9 outputs, exact
      ground-truth validation, manifest archive, and 3 finalized sidecars
- [ ] Durable CPU/GPU replay verified configured/effective requests, exact node
      identities, output digest links, both progress levels, cancellation 130,
      structured OOM policy, and cleanup-gated publication
- [ ] Prefer-GPU UI and durable replay verified all eligible reviewed GPU
      placements, explained unsupported-node CPU decisions, dormant per-node
      preferences, visible-only fallback, and exact provenance
- [ ] CPU-only import/generated/batch replay passed without optional GPU imports
- [ ] Clean tagged wheel/sdist build, Twine, content, `pip check`, manifest,
      compute-policy-resource, and entry-point checks pass; SHA-256 hashes saved
- [ ] Companion documentation strict build passes and release page is published
- [ ] Git tag pushed
- [ ] Uploaded to PyPI
- [ ] GitHub pre-release published with wheel and sdist attached
- [ ] napari hub page shows latest version
