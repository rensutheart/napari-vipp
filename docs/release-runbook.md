# VIPP Release Runbook

Last reviewed: 2026-08-15

This runbook is intentionally risk-based. A release version change does not,
by itself, invalidate evidence for an unchanged installer, GPU provider,
workflow schema, or user interface.

The governing rule is:

> required work = cheap release invariants + gates for changed domains + the
> release-maturity gate

The current reusable qualification baseline is
[`v0.13.0a7`](release-qualification-baseline.md). The large qualification
procedures remain available as conditional tools; they are not a checklist to
repeat for every alpha.

## Release Lanes

| Lane | Versions | Purpose | Expected human release work |
| --- | --- | --- | --- |
| Iterative alpha | `X.Y.ZaN` | Put coherent development slices in users' hands quickly. This is the default for current VIPP development. | About 10–20 minutes after CI, excluding automated build/deploy time. |
| Release candidate | `X.Y.ZrcN` | Freeze a feature line and test the combined user journey before production. | About 20–45 minutes, plus only the invalidated domain checks. |
| Stable production | `X.Y.Z` | Publish a supported production milestone. A first stable or major release refreshes all qualification baselines. | Comprehensive review, preferably carried forward from an unchanged passing RC. |
| Stable hotfix | `X.Y.(Z+1)` | Correct a narrow production defect. | Alpha-like fast path plus the affected production regression. |

`0.13.0a7` was an iterative alpha, even though it contained unusually broad
GPU and installer changes. Future `aN` releases use the lightest lane unless
their actual changes invalidate a heavier domain.

## Checks Required For Every Release

These facts are unique to each artifact and cannot be carried forward:

1. The version, changelog, release notes, and package metadata agree.
2. The exact final `main` commit passed the normal CI matrix.
3. The immutable tag resolves to that exact clean commit.
4. The wheel and source archive are built once, pass metadata checks, and have
   recorded SHA-256 hashes.
5. Published GitHub and PyPI bytes match those hashes.
6. The numbered documentation resolves; when selected, the `stable` alias
   points to the same version.

Normal CI already runs manifest validation, Ruff, the full test suite on every
supported OS/Python combination, package builds, and clean wheel/source-archive
install jobs. Do not repeat the full local suite merely to duplicate a green
exact-commit CI run. Run local focused tests while developing and use CI as the
release-wide test record.

## Declare The Changed Domains

The release declaration covers the union of changes from the previous public
tag to the candidate—not merely the final version/docs PR. Start with the
changed paths, combine the release-impact declarations from every merged PR in
that range, and have a maintainer confirm the result because shared utilities
can affect more than one domain.

```powershell
git diff --name-only "<previous-public-tag>...HEAD"
```

```yaml
tier: alpha
changed:
  core_ui: true
  workflow_schema: true
  gpu_scientific: false
  windows_installer: false
  dependencies_toolchain: false
  packaging_release: false
  documentation: true
carried_forward:
  gpu_scientific: v0.13.0a7
  windows_installer: v0.13.0a7
```

When unsure, activate the domain and run its affected checks. Do not respond
to uncertainty by running every gate in the repository.

## Change-Triggered Gates

| Changed domain | Typical invalidators | Additional alpha gate |
| --- | --- | --- |
| Core or UI | User-visible node, graph, inspector, viewer, worker, or interaction changes | Focused automated tests and one manual smoke of the changed behavior. Do not replay unrelated UI scenarios. |
| Workflow/schema | Workflow, batch, manifest, provenance, source identity, serializer, or migration changes | Affected old-version fixtures, save/reopen and round-trip checks, and one representative generated/batch path when relevant. |
| GPU/scientific | Provider algorithm, numerical contract, dtype/axes behavior, compute policy, dispatch, fallback, cleanup, memory, provenance, catalogue, or CUDA/CuPy pins | Qualify only the changed provider or shared layer on real hardware. Run the full catalogue only for a catalogue-wide/shared-layer change, an RC, or a production baseline refresh. |
| Windows installer | Installer/repair/update/rollback/network/path/ownership code, dependency resolution, PyInstaller, signing policy, or supported Windows/Python route | Run packaging tests plus the affected lifecycle scenario. Full new/install/repair/update/rollback/uninstall and fresh-account matrices are reserved for cross-cutting installer changes, RCs, or production refreshes. |
| Dependencies/toolchain | Runtime/build dependency, Python support, compiler, build backend, or packaging tool changes | Clean package installs first, then only the scientific/installer domains affected by that dependency. Test-only dependency changes do not invalidate runtime evidence. |
| Packaging/release | Package data, entry points, build backend, resource layout, publishing workflows, or release asset composition | Exact artifact content/entry-point checks. Reproducibility comparison is required when packaging changed, not for an unrelated app-only alpha. |
| Documentation | Installation, upgrade, schema, UI, limitation, download name, navigation, or site configuration changes | Strict documentation build and manual inspection of changed pages only. Screenshots are refreshed only when their pictured UI changed. |

Path examples and the exact carry-forward boundaries are recorded in
[`release-qualification-baseline.md`](release-qualification-baseline.md).

## Evidence Carry-Forward

Behavioral evidence remains valid while all of its declared inputs remain
unchanged:

- implementation and contract;
- runtime/build dependency fingerprint;
- supported platform and ABI;
- test/evidence procedure; and
- relevant scheduled canary.

A version string, changelog, unrelated feature, or documentation edit does not
invalidate behavioral evidence. Exact artifact facts—hashes, embedded version,
signature state, and public URLs—are always regenerated.

When a domain is invalidated:

1. run focused checks while implementing the change;
2. run the smallest real-environment scenario that covers the changed risk;
3. update the qualification baseline for that domain; and
4. carry the new baseline forward until another declared invalidator changes.

A failed installer packaging job invalidates the packaging domain. A failed
real-GPU canary invalidates the GPU domain. A stale GPU canary calls for a quick
smoke, not automatically for the full historical matrix.

## Fast Path: Iterative Alpha

This is the normal VIPP release path.

### 1. Prepare one release change

Use one application release PR and, when user documentation changed, at most
one companion-documentation PR. Include:

- the version bump;
- a concise changelog/release-note section;
- the union changed-domain declaration for everything since the previous
  public tag;
- focused verification performed for the changed feature; and
- links to carried-forward baselines for unchanged domains.

Write documentation so it remains true before and after publication. Avoid
separate “pending”, “now public”, and “stable promoted” commits. The versioned
manual is not deployed until the package is public.

### 2. Merge once and trust exact-commit CI

Merge the reviewed release PR into `main`, wait for the final `main` CI run,
and stop if any job fails. Do not run another full local pytest solely because
CI already passed it.

```powershell
git fetch origin --tags
git switch main
git pull --ff-only origin main
git status --short
git rev-parse HEAD
```

### 3. Tag once

```powershell
$releaseVersion = "<version>"
$releaseSha = (git rev-parse HEAD).Trim()
if ($releaseSha -ne (git rev-parse origin/main).Trim()) {
    throw "HEAD is not the exact origin/main release commit"
}
git tag -a "v$releaseVersion" -m "napari-vipp $releaseVersion"
if ((git rev-parse "v$releaseVersion^{}").Trim() -ne $releaseSha) {
    throw "Tag does not resolve to the release commit"
}
```

Never move a published tag. If code changes after tagging, prepare the next
version instead.

### 4. Build once

Build the wheel and source archive once from the clean tag, or reuse retained
CI artifacts only when they are cryptographically bound to that exact commit.
Run Twine metadata validation and record hashes.

The canonical manual build sequence is:

```powershell
$releaseVersion = "<version>"
$taggedSha = (git rev-parse "v$releaseVersion^{}").Trim()
$artifactDir = Join-Path $env:TEMP `
    "vipp-release-$releaseVersion-$($taggedSha.Substring(0, 12))"
if (Test-Path -LiteralPath $artifactDir) {
    throw "Refusing to reuse release directory: $artifactDir"
}
New-Item -ItemType Directory -Path $artifactDir | Out-Null

$releasePython = "C:\path\to\cpython-3.12.10\python.exe"
$builderVenv = "$artifactDir-builder"
& $releasePython -m venv $builderVenv
$builderPython = Join-Path $builderVenv "Scripts\python.exe"
& $builderPython -m pip install ".[installer-build]" twine

$env:PYTHONHASHSEED = "0"
$env:SOURCE_DATE_EPOCH = (git show -s --format=%ct $taggedSha).Trim()
& $builderPython -m build --sdist --no-isolation --outdir $artifactDir
& $builderPython -m build --wheel --no-isolation --outdir $artifactDir
& $builderPython -m twine check `
    "$artifactDir/napari_vipp-$releaseVersion.tar.gz" `
    "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl"
Get-FileHash -Algorithm SHA256 `
    "$artifactDir/napari_vipp-$releaseVersion.tar.gz", `
    "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl"
```

Routine alphas may publish only the wheel and source archive. If a Windows EXE
is useful for alpha testers, it may also be rebuilt from the exact wheel. When
installer code and its dependency/toolchain inputs are unchanged, require only
the automated frozen-payload, embedded-wheel, version, signature-state, and
hash checks; carry forward lifecycle evidence. Do not repeat manual
install/repair/update/rollback/uninstall testing.

When publishing an alpha EXE, build it from the exact wheel above. Choose one
finalization route—signed or explicitly unsigned—rather than both:

```powershell
$wheel = "$artifactDir/napari_vipp-$releaseVersion-py3-none-any.whl"
& $builderPython scripts/package_windows_installer.py build `
    --wheel $wheel `
    --output-directory $artifactDir
$stagingExe = `
    "$artifactDir/VIPP-Setup-$releaseVersion-Windows-x86_64-SIGNING-STAGING.exe"
$buildManifest = `
    "$artifactDir/VIPP-Setup-$releaseVersion-Windows-x86_64-SIGNING-STAGING-build.json"

# Explicitly unsigned alpha:
& $builderPython scripts/package_windows_installer.py finalize-unsigned `
    --unsigned-staging-executable $stagingExe `
    --build-manifest $buildManifest `
    --output-directory $artifactDir

# Signed release instead:
# .\scripts\sign_windows_installer.ps1 -InputPath $stagingExe `
#     -CertificateThumbprint '<approved-thumbprint>'
# & $builderPython scripts/package_windows_installer.py finalize `
#     --signed-staging-executable $stagingExe `
#     --build-manifest $buildManifest `
#     --output-directory $artifactDir `
#     --expected-signer-thumbprint '<approved-thumbprint>'
```

Detailed Windows build and signing commands live in
[`packaging/windows/README.md`](../packaging/windows/README.md). Full installer
field procedures live in
[`windows-installer-field-acceptance.md`](windows-installer-field-acceptance.md)
and are conditional, not alpha defaults.

### 5. Publish in one direction

1. Push the immutable tag.
2. Create the GitHub prerelease and attach the already-qualified artifacts.
3. Dispatch `publish-pypi.yml` with the exact tag; it downloads the GitHub
   wheel/source archive rather than rebuilding them.
4. Verify the PyPI version and hashes.
5. Deploy the companion numbered manual once with `make_stable=true` when that
   alpha should be the default manual.
6. Verify GitHub, PyPI, numbered/stable docs, and repository cleanliness.

napari hub indexing is asynchronous and non-blocking once PyPI metadata and the
napari manifest are correct. Record it as pending propagation rather than
holding the release session open.

### Alpha stop conditions

Stop publication only for:

- failed exact-commit CI;
- a failed gate in a changed domain;
- mismatched tag/version/hash/artifact metadata;
- unsafe or missing release assets; or
- a failed publishing workflow.

An unrun gate for an unchanged domain is not a failure; it is carried-forward
evidence. An aspirational qualification item outside the release claim is a
roadmap item, not a release blocker.

## Release Candidate

An RC tests the combined feature line rather than every historical scenario:

- all universal release checks;
- all domains changed since the last production baseline;
- one representative end-to-end UI workflow;
- the supported saved-workflow compatibility corpus;
- a quick real-GPU workflow if the feature line includes GPU behavior; and
- an exact-EXE happy-path install/launch/uninstall if an installer will ship.

Run the full GPU catalogue only if GPU shared infrastructure/catalogue inputs
changed. Run the full installer matrix only if installer cross-cutting inputs
changed. Later RCs test their delta from the prior RC.

## Stable Production Release

Prefer an unchanged passing RC followed by version/release-metadata-only
promotion. Carry the RC behavioral evidence forward and regenerate only exact
stable-artifact facts plus a representative happy path.

A first stable release, major release, support-matrix expansion, or production
release without a qualifying RC refreshes all relevant baselines. If product
code, schema, dependencies, GPU contracts, or installer behavior changes after
the RC, publish another RC rather than folding the change directly into the
stable release.

Stable hotfixes use the fast path plus the exact production regression and any
domain it invalidates.

## Minimal Operator Checklist

- [ ] Release tier and changed domains are declared.
- [ ] Focused checks for changed domains pass.
- [ ] Unchanged expensive domains cite a valid qualification baseline.
- [ ] Final `main` CI passes and its commit is recorded.
- [ ] Tag resolves to that exact commit.
- [ ] Wheel/source archive metadata and hashes pass.
- [ ] An optional EXE passes its exact-artifact integrity checks.
- [ ] GitHub prerelease and PyPI bytes match.
- [ ] Numbered documentation resolves; `stable` is moved if intended.
- [ ] napari hub is either updated or truthfully recorded as asynchronously pending.

Anything beyond this list must be justified by the release tier or a declared
changed domain.

## Automation Follow-Up

The next release-infrastructure slice should make this fast path one manual
workflow with inputs for the tag, tier, and optional EXE.
It should build once, publish the same bytes to GitHub/PyPI, deploy docs once,
and report the exact artifacts. A path-based report can pre-fill the domain
declaration, but a maintainer still confirms it.

Current-version documentation should also move to one shared data value so a
routine version bump does not require mechanical edits across dozens of pages.
Neither automation item expands release gates; it only removes repeated
operator work.
