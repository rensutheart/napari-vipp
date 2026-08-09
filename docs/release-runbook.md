# VIPP Alpha Release Runbook

Last reviewed: 2026-08-09

This runbook covers publishing napari-vipp to PyPI, creating a GitHub release,
publishing the companion documentation site, and confirming discovery on
napari hub.

## Scope

- Target package version: set `<version>` from the release milestone before
  starting; do not reuse the current package version by accident.
- Current prepared target: `0.13.0a4`.
- Release maturity: Alpha
- Distribution channels: PyPI, GitHub release, napari hub index

## 1. Preconditions

1. You have push/tag permission on GitHub for this repository.
2. You have upload permission for the `napari-vipp` project on PyPI.
3. You have a PyPI API token available to paste into Twine's hidden password
   prompt.
4. You have a clean git working tree on the release commit.
5. The companion `vipp-mkdocs` repository has a reviewed release page and a
   clean, pushed release commit.

This repository does not currently contain a PyPI Trusted Publishing workflow,
so this runbook documents the manual Twine route for 0.13.0a4. Never paste,
print, commit, or place the PyPI token in a command, script, or shell
environment variable. Enter it only at Twine's hidden password prompt.

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
badges alone. Confirm BatchConfig/manifest schema 3, configured and effective
requests/hashes, exact implementation version/parity identity for every
computed node, per-output execution digest links, and true cleanup. Run the
saved `vipp_batch_pipeline.py --progress` once with its recorded request and
once with an explicit mode or node override. Confirm both item and operation
progress, then cancel a third run and verify cancelled status, no unpublished
output, finalized manifests/checkpoints, and exit code 130.

For 0.13.0a1 and later, include a conventional-TIFF axis smoke. Use a fixture
that the reader reports as `QYX` but whose acquisition record establishes that
Q is Z. First leave `Declare axes` blank and confirm a workflow requiring `ZYX`
is stopped by representative scientific preflight before the output directory,
run artifacts, or GPU setup appear. Then set `QYX -> ZYX`, preview and run, and
confirm both paths use `ZYX`. Inspect the schema-3 config and manifest for the
exact declaration plus raw `QYX` and effective `ZYX`. Confirm `Reorder Axes`
still transposes without renaming Q, a later item whose raw axes do not match the
declaration fails at that item, and the declared Z calibration is not presented
as newly measured or corrected.

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

For any release whose policy declares cuCIM candidates, make one explicit
distribution decision before freezing the commit:

1. **Distributed and admitted:** retain the exact final downstream wheel,
   inspect its full payload/metadata/licences, publish it at an immutable URL,
   record its SHA-256/source/patch/toolchain manifest, expose a packaged
   checksum-and-probe approval command, update the policy to that artifact, and
   repeat clean-host background/measurement, durable-run, memory, cancellation,
   cleanup, and licence/packaging qualification;
2. **Pinned local build:** publish no wheel, but ship an exact source
   tag/commit/recipe, complete local wheel licences/notices, a canonical payload
   digest, a machine-readable build manifest, and an existing-environment
   checksum/probe approval path; or
3. **Unavailable:** state prominently in README, CHANGELOG, release notes,
   installation, compute, and validation pages that ordinary users cannot
   install cuCIM and that the affected providers normally remain on CPU.

VIPP 0.13.0a1 through 0.13.0a4 use option 2. Every user builds and keeps their
own archive; policy allows a per-user archive SHA only when its installed
payload, pinned source/recipe, environment, and workload pass the exact
reviewed gates.

Never publish the historical `586D...134CF8` research wheel. Its exact archive
is no longer retained, and the installed copy contains Windows-materialized
licence links rather than the actual Apache-2.0 and third-party licence text.
The pinned local recipe replaces that historical build for self-use; it does
not authorize hosting or redistribution.

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

### Bounded 0.13.0a1 Windows Acceptance Evidence

On 2026-08-06, a clean, non-editable, release-style Windows environment running
VIPP 0.13.0a1 launched the synthetic-volume VIPP example in napari. The
operator inspected the running example and confirmed that it looked correct.
This records the example launch and visual acceptance only; it is not a pass
for every manual UI scenario in the checklist.

Separately on 2026-08-06, an automated headless run in that clean environment
executed Subtract Background with `Prefer GPU`, selected
`cucim-subtract_background-v2` through cuCIM without fallback, produced the
expected `(31, 37)` `uint16` output, and reported successful cleanup. This is
bounded evidence for that operation and environment. It does not qualify the
final tagged artifacts, durable batch or generated-CLI replay, cancellation or
OOM handling, the CPU-only path, other manual workflows, or another platform;
those checklist items remain separate release gates.

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

With `HEAD` still at the clean tagged commit, build into a new directory named
for both the version and tagged commit. Refuse to reuse an existing directory;
old same-version artifacts can otherwise look uploadable even though they came
from another commit. Do not use a broad root-level `dist/*` upload glob:

```powershell
python -m pip install -U build twine
$releaseVersion = "<version>"
$taggedSha = (git rev-parse "v$releaseVersion^{}").Trim()
$artifactDir = "dist/$releaseVersion-$($taggedSha.Substring(0, 12))"
if (Test-Path -LiteralPath $artifactDir) {
    throw "Artifact directory already exists; choose a clean checkout or inspect it without reusing it: $artifactDir"
}
New-Item -ItemType Directory -Path $artifactDir | Out-Null
python -m build --outdir $artifactDir
python -m twine check `
  "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl" `
  "$artifactDir/napari_vipp-$releaseVersion.tar.gz"
Get-FileHash -Algorithm SHA256 `
  "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl", `
  "$artifactDir/napari_vipp-$releaseVersion.tar.gz"
```

Expected output artifacts:

- `dist/<version>-<tagged-short-sha>/napari_vipp-<version>.tar.gz`
- `dist/<version>-<tagged-short-sha>/napari_vipp-<version>-py3-none-any.whl`

Using a version-and-commit-specific directory prevents an upload command from
including artifacts from an older release. It does not make an existing
same-version artifact safe: confirm both files were produced from the tagged
commit, record their SHA-256 hashes, and repeat the clean-wheel CPU and CUDA
acceptance smokes against these files rather than an editable checkout.

## 5. Build And Publish Documentation

In the companion `vipp-mkdocs` repository:

```powershell
python -m pip install -r requirements.txt
python -m mkdocs build --strict
```

Review the rendered `0.13.0a4` release page, workflow-schema upgrade guidance,
batch workspace instructions, architecture boundaries, Windows CUDA/cuCIM
installation boundary, and known limitations.
Commit and push the docs release before the package release. A push to the docs
repository publishes only the `nightly` manual: confirm the nightly release
page and Windows CUDA guide resolve. Do not mistake that for the numbered
release snapshot. Step 6 explicitly publishes and verifies the numbered manual
after the application tag is pushed and before PyPI upload.

## 6. Publish Tag, Numbered Manual, Package, And GitHub Prerelease

Push the already-qualified immutable tag first:

```powershell
git push origin "v<version>"
```

Next, dispatch the companion repository's numbered documentation deployment:

```powershell
gh workflow run docs-deploy.yml `
  --repo rensutheart/vipp-mkdocs `
  --ref main `
  -f version="<version>" `
  -f make_stable=true
```

Wait for that exact `workflow_dispatch` run to succeed. Before any package
upload, open both of these URLs and confirm they show the target version rather
than a 404 or `nightly` content:

- `https://rensutheart.github.io/vipp-mkdocs/<version>/`
- `https://rensutheart.github.io/vipp-mkdocs/<version>/getting-started/windows-cuda/`

Only then publish exactly the two hash-recorded artifacts. PyPI uploads cannot be
replaced, so recheck the directory, version, and explicit filenames before
entering credentials. Pass only the token username on the command line and
paste the API token into Twine's hidden password prompt:

```powershell
python -m twine upload --username "__token__" `
  "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl" `
  "$artifactDir/napari_vipp-$releaseVersion.tar.gz"
```

Do not put the token in `TWINE_PASSWORD`, the command line, or a PowerShell
assignment. At the `Password:` prompt, paste the token; Twine does not echo the
input or add it to shell history.

Post-upload validation:

- Open https://pypi.org/project/napari-vipp/
- Confirm the target version is visible
- Confirm README renders alpha disclaimer
- Confirm license metadata shows BSD-3-Clause terms
- Confirm `Requires-Python` and optional GPU extras match the tagged metadata

Prepare the release notes body below in a temporary file, then create the
release with GitHub CLI (or use the equivalent GitHub UI fields):

```powershell
gh release create "v$releaseVersion" --prerelease --verify-tag `
  --notes-file release-notes.md `
  "$artifactDir/napari_vipp-$releaseVersion.tar.gz" `
  "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl"
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
- [ ] Conventional-TIFF `QYX` batch is blocked before output/GPU setup without a
      declaration, succeeds as `ZYX` with `QYX -> ZYX`, records raw/effective
      axes in schema 3, and keeps declaration distinct from reorder/calibration
- [ ] Prefer-GPU UI and durable replay verified all eligible reviewed GPU
      placements, explained unsupported-node CPU decisions, dormant per-node
      preferences, visible-only fallback, and exact provenance
- [ ] CPU-only import/generated/batch replay passed without optional GPU imports
- [ ] cuCIM distribution decision is complete: distributed artifact, pinned
      private local-build route, or unavailable; every public surface matches
      that decision and the chosen provenance/approval path was requalified
- [ ] Clean tagged wheel/sdist build, Twine, content, `pip check`, manifest,
      compute-policy-resource, and entry-point checks pass; SHA-256 hashes saved
- [ ] Companion documentation strict build passes and release page is published
- [ ] Git tag pushed
- [ ] Numbered release manual and Windows CUDA guide resolve before package upload
- [ ] Uploaded to PyPI
- [ ] GitHub pre-release published with wheel and sdist attached
- [ ] napari hub page shows latest version
