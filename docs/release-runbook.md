# VIPP 0.8.0a1 Alpha Release Runbook

This runbook covers publishing napari-vipp to PyPI, creating a GitHub release,
and confirming discovery on napari hub.

## Scope

- Package version: 0.8.0a1
- Release maturity: Alpha
- Distribution channels: PyPI, GitHub release, napari hub index

## 1. Preconditions

1. You have push/tag permission on GitHub for this repository.
2. You have upload permission for the `napari-vipp` project on PyPI.
3. You have a PyPI API token available as `TWINE_PASSWORD`.
4. You have a clean git working tree on the release commit.

Recommended local tools:

- Python 3.10+
- `python -m pip install -U build twine`

## 2. Verify Metadata

Confirm these are set:

- `pyproject.toml` version is `0.8.0a1`
- `pyproject.toml` classifier includes `Development Status :: 3 - Alpha`
- README has a clear alpha disclaimer

Optional but recommended checks:

- `python -m npe2 validate src/napari_vipp/napari.yaml`
- `python -m pytest -q`

## 3. Build Artifacts

From repository root:

```powershell
python -m pip install -U build twine
python -m build
python -m twine check dist/*
```

Expected output artifacts:

- `dist/napari_vipp-0.8.0a1.tar.gz`
- `dist/napari_vipp-0.8.0a1-py3-none-any.whl`

## 4. Publish To PyPI

Set token in the shell (PowerShell):

```powershell
$env:TWINE_USERNAME = "__token__"
$env:TWINE_PASSWORD = "<pypi-api-token>"
python -m twine upload dist/*
```

Post-upload validation:

- Open https://pypi.org/project/napari-vipp/
- Confirm version `0.8.0a1` is visible
- Confirm README renders alpha disclaimer

## 5. Create Git Tag And GitHub Release

Create and push tag:

```powershell
git tag -a v0.8.0a1 -m "napari-vipp 0.8.0a1 alpha"
git push origin v0.8.0a1
```

Create release page in GitHub UI:

1. GitHub repository -> Releases -> Draft a new release
2. Tag: `v0.8.0a1`
3. Title: `napari-vipp v0.8.0a1 (Alpha)`
4. Mark as pre-release: enabled
5. Add release notes (suggested template below)

Suggested release notes body:

```markdown
## napari-vipp v0.8.0a1 (Alpha)

This is an early alpha build and is still in active development.

### Important
- Breaking changes are expected between releases.
- Validate outputs before publication or production use.

### Highlights
- Added insert-on-wire graph editing for splicing nodes into existing pipelines.
- Added smarter graph connector routing with compact close-node curves.
- Added VIPP-local Z/T/C slider planning and calibrated preview scale handling.
- Improved axis-aware thumbnails for RGB/composite previews.
```

## 6. napari Hub Listing/Refresh

napari hub indexes packages from PyPI metadata for napari plugins.

After PyPI upload:

1. Wait for napari hub index refresh (can take some time).
2. Check: https://napari-hub.org/plugins/napari-vipp
3. Confirm:
   - plugin appears
   - version updates to 0.8.0a1
   - README/disclaimer is visible

If not updated after indexing delay:

- Verify `napari.manifest` entry point in `pyproject.toml`
- Verify `src/napari_vipp/napari.yaml` is included in the wheel/sdist
- Re-check PyPI metadata and release files

## 7. Post-Release Follow-up

1. Announce alpha status clearly in repository and release channels.
2. Open a tracking milestone for issues found in 0.8.0 alpha.
3. Plan next versioning strategy.

## Operator Checklist

- [ ] Tests pass locally
- [ ] Build and twine checks pass
- [ ] Uploaded to PyPI
- [ ] Git tag pushed
- [ ] GitHub pre-release published
- [ ] napari hub page shows latest version
