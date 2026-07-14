# VIPP Alpha Release Runbook

Last reviewed: 2026-07-14

This runbook covers publishing napari-vipp to PyPI, creating a GitHub release,
and confirming discovery on napari hub.

## Scope

- Target package version: set `<version>` from the release milestone before
  starting; do not reuse the current package version by accident.
- Release maturity: Alpha
- Distribution channels: PyPI, GitHub release, napari hub index

## 1. Preconditions

1. You have push/tag permission on GitHub for this repository.
2. You have upload permission for the `napari-vipp` project on PyPI.
3. You have a PyPI API token available as `TWINE_PASSWORD`.
4. You have a clean git working tree on the release commit.

Recommended local tools:

- Python 3.12+
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

Required checks:

- `python -m npe2 validate src/napari_vipp/napari.yaml`
- `python -m ruff check .`
- `python -m pytest`

For a batch/provenance release, also use `Run batch...` -> `Open batch demo...` and
complete the generated three-item run. Confirm three completed items, nine
outputs, and a finalized latest manifest, archive, and three item sidecars.

## 3. Build Artifacts

From repository root:

```powershell
python -m pip install -U build twine
python -m build --outdir "dist/<version>"
python -m twine check "dist/<version>/*"
```

Expected output artifacts:

- `dist/<version>/napari_vipp-<version>.tar.gz`
- `dist/<version>/napari_vipp-<version>-py3-none-any.whl`

Using a version-specific directory prevents an upload command from including
artifacts from an older release.

## 4. Publish To PyPI

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

## 5. Create Git Tag And GitHub Release

Create and push tag:

```powershell
git tag -a v<version> -m "napari-vipp <version> alpha"
git push origin v<version>
```

Prepare the release notes body below in a temporary file, then create the
release with GitHub CLI (or use the equivalent GitHub UI fields):

```powershell
gh release create v<version> --prerelease --verify-tag --notes-file release-notes.md
```

If using the GitHub UI:

1. GitHub repository -> Releases -> Draft a new release
2. Tag: `v<version>`
3. Title: `napari-vipp v<version> (Alpha)`
4. Mark as pre-release: enabled
5. Add release notes (suggested template below)

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

## 6. napari Hub Listing/Refresh

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

## 7. Post-Release Follow-up

1. Announce alpha status clearly in repository and release channels.
2. Open a tracking milestone for issues found in the released alpha.
3. Plan next versioning strategy.

## Operator Checklist

- [ ] Tests pass locally
- [ ] Manual UI smoke pass completed for graph search, tunnel manager, graph
      notes, insert-on-wire mapping, workflow save/load, cache modes, and
      example workflows
- [ ] Deterministic batch demo completed with 3 items, 9 outputs, exact
      ground-truth validation, manifest archive, and 3 finalized sidecars
- [ ] Build and twine checks pass
- [ ] Uploaded to PyPI
- [ ] Git tag pushed
- [ ] GitHub pre-release published
- [ ] napari hub page shows latest version
