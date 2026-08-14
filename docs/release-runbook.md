# VIPP Alpha Release Runbook

Last reviewed: 2026-08-14

This runbook covers publishing napari-vipp to PyPI, creating a GitHub release,
publishing the companion documentation site, and confirming discovery on
napari hub.

## Scope

- Target package version: set `<version>` from the release milestone before
  starting; do not reuse the current package version by accident.
- Current published release: `0.13.0a6`.
- Current prepared target: `0.13.0a7`.
- Release maturity: Alpha
- Distribution channels: PyPI, GitHub release, napari hub index

## 1. Preconditions

1. You have push/tag permission on GitHub for this repository.
2. You have upload permission for the `napari-vipp` project on PyPI.
3. The project-scoped PyPI API token is stored as the protected
   `PYPI_API_TOKEN` secret in this repository's `pypi` GitHub environment.
4. You have a clean git working tree on the release commit.
5. The companion `vipp-mkdocs` repository has a reviewed release page and a
   clean, pushed release commit.

The `pypi` environment accepts deployments only from `main`. Store a
project-scoped token there, never in the repository or command history, and
rotate it if its source copy is exposed. Dispatch the workflow only from
`main` and only after the exact tag's GitHub prerelease assets and numbered
manual have been verified. PyPI Trusted Publishing may replace the token later
without changing the artifact-verification flow.

Recommended local tools:

- Python 3.12 or 3.13 for the CPU package; CPython 3.12 for CUDA qualification
- for the release artifacts and isolated Windows setup build, a dedicated
  64-bit CPython 3.12.10 environment with
  `python -m pip install ".[installer-build]" twine`; this pins the build
  backend, wheel tooling, and PyInstaller 6.21.0 used by the reproducibility
  gate

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
- `release-notes.md` has human-readable **Features added** and **Bug fixes**
  sections, useful feature subcategories, current installation/upgrade notes,
  contributor credit, and public links; ordinary prose is not forced into
  narrow hard-wrapped columns
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

VIPP 0.13.0a1 through the published 0.13.0a6 use option 2, and the prepared
0.13.0a7 target retains that decision. Every user builds and keeps their own
archive; policy allows a per-user archive SHA only when its installed payload,
pinned source/recipe, environment, and workload pass the exact reviewed gates.

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

Merge the reviewed release-candidate pull request into `main` before tagging,
then wait for the final `main` GitHub Actions run to pass. The protected PyPI
workflow requires the release tag to be an ancestor of `origin/main`; a tag on
an unmerged release branch is not publishable. Update the local checkout to the
exact remote `main` commit:

```powershell
git fetch origin --tags
git switch main
git pull --ff-only origin main
git status --short
```

Only tag after that versioned `main` candidate is reviewed, pushed, clean, and
green in GitHub Actions. Confirm that `HEAD` equals the current `origin/main`
and record the immutable commit id:

```powershell
if ((git rev-parse HEAD).Trim() -ne (git rev-parse origin/main).Trim()) {
    throw "HEAD is not the exact origin/main release commit"
}
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
$releaseVersion = "<version>"
$taggedSha = (git rev-parse "v$releaseVersion^{}").Trim()
$artifactDir = "dist/$releaseVersion-$($taggedSha.Substring(0, 12))"
if (Test-Path -LiteralPath $artifactDir) {
    throw "Artifact directory already exists; choose a clean checkout or inspect it without reusing it: $artifactDir"
}
New-Item -ItemType Directory -Path $artifactDir | Out-Null
$releasePython = "C:\path\to\cpython-3.12.10\python.exe"
$builderVenv = Join-Path $env:TEMP "vipp-release-builder-$releaseVersion"
if (Test-Path -LiteralPath $builderVenv) {
    throw "Release-builder environment already exists: $builderVenv"
}
& $releasePython -m venv $builderVenv
$builderPython = Join-Path $builderVenv "Scripts\python.exe"
& $builderPython -m pip install ".[installer-build]" twine
$env:PYTHONHASHSEED = "0"
$env:SOURCE_DATE_EPOCH = (git show -s --format=%ct $taggedSha).Trim()
& $builderPython -m build --sdist --no-isolation --outdir $artifactDir
& $builderPython -m build --wheel --no-isolation --outdir $artifactDir
& $builderPython -m twine check `
  "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl" `
  "$artifactDir/napari_vipp-$releaseVersion.tar.gz"
Get-FileHash -Algorithm SHA256 `
  "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl", `
  "$artifactDir/napari_vipp-$releaseVersion.tar.gz"
```

For a release that publishes the separate Windows cuCIM local-build installer,
create it from this same clean tagged checkout and artifact directory:

```powershell
$cucimInstaller = "$artifactDir/napari-vipp-cucim-installer-$releaseVersion-windows.zip"
& $builderPython scripts/package_cucim_windows_installer.py --output $cucimInstaller
Get-FileHash -Algorithm SHA256 $cucimInstaller
```

The packager refuses a dirty tree. Inspect `bundle-manifest.json` inside the ZIP
and confirm its VIPP version and source commit equal `$releaseVersion` and
`$taggedSha`; the archive must contain no wheel. Publish its SHA-256 alongside
the download.

For a release that includes the standalone Windows installer, build it from the
exact wheel already in this artifact directory. Dependencies may resolve
online, but the top-level VIPP package is the hash-recorded wheel embedded in
the EXE:

```powershell
$wheel = "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl"
& $builderPython scripts/package_windows_installer.py build `
  --wheel $wheel `
  --output-directory $artifactDir
$stagingExe = "$artifactDir/VIPP-Setup-$releaseVersion-Windows-x86_64-SIGNING-STAGING.exe"
$buildManifest = "$artifactDir/VIPP-Setup-$releaseVersion-Windows-x86_64-SIGNING-STAGING-build.json"
```

This refuses a dirty or untagged tree and creates only a signing-staging name.
For a local or pull-request smoke, add `--development`; that permits unfinished
source but creates only a visibly marked `DEVELOPMENT` EXE.

Use the real code-signing certificate from the Windows certificate store. The
hook invokes Windows SDK `signtool.exe` with SHA-256 and an RFC 3161 timestamp,
then verifies the result. Never put a private key or password in the repository:

```powershell
.\scripts\sign_windows_installer.ps1 `
  -InputPath $stagingExe `
  -CertificateThumbprint '<approved-40-hex-thumbprint>'
```

Only then create the official filename and release metadata. Omit the final
cuCIM option when that separate, optional local-build ZIP is not being attached:

```powershell
& $builderPython scripts/package_windows_installer.py finalize `
  --signed-staging-executable $stagingExe `
  --build-manifest $buildManifest `
  --output-directory $artifactDir `
  --expected-signer-thumbprint '<approved-40-hex-thumbprint>' `
  --cucim-bundle $cucimInstaller
```

Finalization rechecks the clean tag, signer thumbprint, timestamp certificate,
copied signature, embedded wheel, and optional cuCIM version/commit. It creates
the reserved official EXE, release JSON, persistent third-party notices, and
`SHA256SUMS-Windows-<version>.txt`. This signed path has no unsigned override.

If the release decision is explicitly to ship unsigned, keep the signing-staging
EXE unsigned and use the separate fail-closed command:

```powershell
& $builderPython scripts/package_windows_installer.py finalize-unsigned `
  --unsigned-staging-executable $stagingExe `
  --build-manifest $buildManifest `
  --output-directory $artifactDir `
  --cucim-bundle $cucimInstaller
```

This path still requires the clean exact tag, matching reproducible wheel,
unchanged staging bytes, exact frozen payload, and optional same-tag cuCIM
bundle. It also requires Windows to report `NotSigned` and creates only
`VIPP-Setup-<version>-Windows-x86_64-UNSIGNED.exe`, its release JSON, notices,
and SHA-256 sidecar. It cannot create the filename reserved for a signed
installer. Every public surface must state **Unknown publisher**, require the
official GitHub source and matching checksum, explain **More info > Run
anyway**, provide the manual fallback, and tell users not to bypass an actual
antivirus detection or disable security.

Expected output artifacts:

- `dist/<version>-<tagged-short-sha>/napari_vipp-<version>.tar.gz`
- `dist/<version>-<tagged-short-sha>/napari_vipp-<version>-py3-none-any.whl`
- when the signed Windows bootstrapper is part of the release,
  `dist/<version>-<tagged-short-sha>/VIPP-Setup-<version>-Windows-x86_64.exe`
- when an explicitly unsigned alpha bootstrapper is selected instead,
  `dist/<version>-<tagged-short-sha>/VIPP-Setup-<version>-Windows-x86_64-UNSIGNED.exe`
- with that bootstrapper, its `-release.json`,
  `-THIRD-PARTY-NOTICES.txt`, and `SHA256SUMS-Windows-<version>.txt` sidecars
- when selected for the release,
  `dist/<version>-<tagged-short-sha>/napari-vipp-cucim-installer-<version>-windows.zip`

The repository build and fail-closed signing gate do not replace clean-machine
acceptance. Before publishing, run managed-CPU and qualified-GPU install,
launch, update, repair, and uninstall smokes. The quick-start link must use the
exact finalized asset name. If the release does not include the `.exe`, every
quick-start surface must say **not yet published** rather than showing a dead
download.

Record the exact-artifact CPU, CUDA, path, rollback, and novice results with the
[Windows installer field-acceptance form](windows-installer-field-acceptance.md).
Leave anything that was not actually exercised as **not run**; an older build or
an automated test cannot stand in for the tagged executable.

During those smokes, change each install-relevant field after a successful
review and confirm **Install VIPP** stays disabled until **Check these settings**
finishes for the new selection. Exercise a slow or interrupted GPU dependency
download: a transfer that continues receiving data may run for several minutes,
the configured 120-second timeout applies to network idleness rather than total
setup duration, and retries remain bounded. After a terminal network failure,
verify ownership-bound rollback leaves no incomplete copy active and preserves
any previous working copy. **Try again** must rerun checking and resolution for
the exact current selection, show the new review, and require a fresh
confirmation.

Using a version-and-commit-specific directory prevents an upload command from
including artifacts from an older release. It does not make an existing
same-version artifact safe: confirm both files were produced from the tagged
commit, record their SHA-256 hashes, and repeat the clean-wheel CPU and CUDA
acceptance smokes against these files rather than an editable checkout.

### Scheduled cuCIM release canary

The `Windows cuCIM release canary` workflow provides two deliberately separate
levels of evidence:

- its weekly hosted-Windows job checks out the configured immutable release
  tag, reproduces the no-wheel cuCIM installer ZIP, compares it byte-for-byte
  with the published GitHub asset, and verifies the embedded tag, commit,
  entry point, and no-wheel declaration; and
- its real CUDA job runs only when a maintainer explicitly selects
  `run_real_canary` or sets `VIPP_CUCIM_CANARY_ENABLED=true`. It requires an
  up-to-date self-hosted Windows runner labelled `vipp-cuda13` and the protected
  `cuda-canary` environment variable `VIPP_CUCIM_CANARY_PYTHON`, pointing to a
  dedicated, already released VIPP CUDA environment. It saves a redacted Doctor
  report and, for releases that contain the aggregate admission harness, runs
  the complete `quick` profile and retains its aggregate/operation evidence.
  Older pre-harness tags record that limitation explicitly rather than claiming
  a pass. An optional `VIPP_CUCIM_CANARY_WORK_ROOT` retains the large pinned
  source/build cache.

Keep `VIPP_CUCIM_CANARY_TAG` on `v0.13.0a6` until the complete next prerelease
and its cuCIM ZIP are public. Then move it to that exact published tag (the
prepared target is `v0.13.0a7`). A skipped real-CUDA job is truthful static
bundle evidence only; it must never be recorded as a GPU build, installation,
Compute Doctor, or operation acceptance pass.

Normal CI also installs the built wheel and sdist in clean jobs on Windows,
Linux, and macOS, alternating CPython 3.12 and 3.13 while covering both archive
formats on every operating system. These package jobs do not replace the
managed installer or real-GPU acceptance paths.

Before a GPU release candidate, validate the live public catalogue and run its
owned qualification plan from the clean tagged checkout:

```powershell
python scripts/run_gpu_admission.py --check
python scripts/run_gpu_admission.py `
  --profile full `
  --output .\gpu-admission-aggregate.json `
  --artifacts .\gpu-admission-artifacts `
  --device-index 0
```

Use `--profile quick` for the protected scheduled/manual canary. Both profiles
fail if a public implementation is unaccounted for, an evidence owner is not
actually invoked, a required facet is missing, a pytest owner skips, or an
artifact fails its declared schema. Promote evidence only from the reviewed
clean commit and hardware context named by the release; a dirty-worktree pass
is useful integration evidence but not a canonical qualification record.

## 5. Build And Publish Documentation

In the companion `vipp-mkdocs` repository:

```powershell
python -m pip install -r requirements.txt
python -m mkdocs build --strict
```

Review the rendered `<version>` release page, installer-first quick start,
workflow-schema upgrade guidance,
batch workspace instructions, architecture boundaries, Windows CUDA/cuCIM
installation boundary, and known limitations. When a Windows `.exe` is
included, the quick start must lead with its exact asset and truthful signing
status. An unsigned alpha must provide checksum-first SmartScreen instructions
and a manual fallback. If no installer is included, explicitly say it is not
yet published.
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

If this release includes a Windows bootstrapper, publish the GitHub
prerelease and its exact installer asset **before** promoting the versioned
manual. Otherwise the prioritized Quick Start would point to a file that does
not exist. Attach the already qualified wheel, sdist, and selected signed or
explicitly unsigned `.exe`, then verify the public asset URL, signing-status
guidance, size, and SHA-256. The example below shows the signed path; substitute
the exact `-UNSIGNED` asset and sidecars when that release decision applies:

```powershell
gh release create "v$releaseVersion" --prerelease --verify-tag `
  --notes-file release-notes.md `
  "$artifactDir/napari_vipp-$releaseVersion.tar.gz" `
  "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl" `
  "$artifactDir/VIPP-Setup-$releaseVersion-Windows-x86_64.exe" `
  "$artifactDir/VIPP-Setup-$releaseVersion-Windows-x86_64-release.json" `
  "$artifactDir/VIPP-Setup-$releaseVersion-Windows-x86_64-THIRD-PARTY-NOTICES.txt" `
  "$artifactDir/SHA256SUMS-Windows-$releaseVersion.txt"
```

When selected, append `$cucimInstaller` to that command. cuCIM remains a
separate optional release asset and is never payload inside the primary EXE.

Record that the GitHub prerelease is already public and skip the later
`gh release create` command. If verification fails, stop before deploying the
manual or uploading to PyPI; never publish a temporary or unsigned download at
the final asset name.

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
- `https://rensutheart.github.io/vipp-mkdocs/<version>/getting-started/`
- `https://rensutheart.github.io/vipp-mkdocs/<version>/getting-started/windows-cuda/`

Only then publish exactly the two hash-recorded artifacts. PyPI uploads cannot
be replaced, so dispatch the protected workflow from `main` with the already
qualified tag:

```powershell
gh workflow run publish-pypi.yml `
  --repo rensutheart/napari-vipp `
  --ref main `
  -f tag="v$releaseVersion"
```

Wait for that exact workflow run to succeed. It checks out the immutable tag,
downloads exactly the wheel and source archive already attached to its GitHub
prerelease, validates their metadata, and publishes them with the project-
scoped token held only in the protected `pypi` environment. It does not rebuild
the distributions or accept a broad artifact glob.

Post-upload validation:

- Open https://pypi.org/project/napari-vipp/
- Confirm the target version is visible
- Confirm README renders alpha disclaimer
- Confirm license metadata shows BSD-3-Clause terms
- Confirm `Requires-Python` and optional GPU extras match the tagged metadata

Prepare the release notes body below in a temporary file, then create the
release with GitHub CLI (or use the equivalent GitHub UI fields). Skip this
creation step when the signed-installer ordering above already published the
same prerelease; inspect that existing release instead:

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

Write release highlights from the target CHANGELOG section in plain language.
Use feature subcategories to make a large release scannable, keep fixes
separate, explain upgrade or optional-component boundaries, and credit external
contributors. Do not turn prose into a narrow column with manual line breaks.
The following is a structure, not a reusable feature list:

```markdown
## napari-vipp v<version> (Alpha)

This is an early alpha build and is still in active development.

### Important
- Breaking changes are expected between releases.
- Validate outputs before publication or production use.
- This release is distributed under the BSD 3-Clause License.

### Features added

#### <User-facing feature group>

Explain what changed and why it matters in ordinary prose.

### Bug fixes

- Summarize corrected behavior in user-facing terms.

### Installation and upgrading

- Call out workflow/schema compatibility and optional-component boundaries.
- Link the exact public installer, guides, validation reports, or examples.
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
- [ ] If published, the separate cuCIM installer ZIP came from the clean tagged
      commit, contains no wheel, and its source commit/file hashes plus archive
      SHA-256 were verified and recorded
- [ ] Windows-installer status is truthful everywhere: if signed, the reserved
      filename is Authenticode-signed and timestamped; if explicitly unsigned,
      the filename contains `-UNSIGNED`, Windows reports `NotSigned`, and every
      public surface requires the official source and matching SHA-256 before
      **More info > Run anyway**. Either shipped artifact is attached,
      clean-machine accepted, and linked first; if no installer is shipped,
      Quick Start says **not yet published**
- [ ] Windows installer invalidates reviewed state after every install-relevant
      edit; GPU downloads tolerate continuing multi-minute transfers with the
      bounded retry/120-second idle-timeout policy; terminal network failure
      rolls back safely; and Try again rechecks the exact current selection
      before a fresh confirmation
- [ ] Clean tagged wheel/sdist build, Twine, content, `pip check`, manifest,
      compute-policy-resource, and entry-point checks pass; SHA-256 hashes saved
- [ ] Companion documentation strict build passes; its Quick Start, release
      page, and installer/manual/advanced hierarchy match the released artifacts
- [ ] Git tag pushed
- [ ] Numbered release manual, Quick Start, and Windows CUDA guide resolve before
      package upload
- [ ] Uploaded to PyPI
- [ ] GitHub pre-release published with wheel and sdist attached
- [ ] napari hub page shows latest version
