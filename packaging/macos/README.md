# macOS PKG Packaging

VIPP's macOS desktop deliverable is a native, offline `.pkg`, not a frozen
PyInstaller application and not a DMG. A PKG is already a double-clickable Mac
installer. It can install the managed Python/napari environment and create a
normal `VIPP.app`, while preserving napari's plugin model. Wrapping it in a DMG
would add another artifact without adding installation capability.

The builder has two deliberately separate unsigned lanes:

- `build --development` creates a local/CI `-DEVELOPMENT.pkg` that can never be
  finalized as a release;
- a clean build at the exact `v<version>` alpha tag creates
  `-SIGNING-STAGING.pkg`, and `finalize-unsigned` alone may copy the reviewed
  bytes to the public, unmistakably named `-UNSIGNED.pkg` plus release
  evidence.

The second lane is a narrow exception for iterative `X.Y.ZaN` alphas. It does
not permit an unsigned beta, release candidate, or stable release. The public
alpha remains unsigned and unnotarized, so macOS will warn and managed Macs may
refuse it. Developer ID signing and notarization remain the production goal.

## Installed layout

| Property | Contract |
| --- | --- |
| Compute | CPU only; macOS does not use the NVIDIA CUDA extras |
| Python | Managed CPython 3.12 environment |
| UI | napari 0.9.0 with PySide6 6.9.3 |
| Environment | `~/Library/vipp` |
| Application | `~/Applications/VIPP.app` |
| Free disk space | Approximately 3 GB |
| User scope | Current user only; no shell initialization or system Python changes |
| Architectures | Separate `arm64` and `x86_64` packages |
| Minimum installer guard | macOS 13 |

Separate architecture packages keep conda's native dependencies auditable;
this is not a Universal 2 bundle. The app shortcut is supplied by `menuinst`
and launches the real managed Python. The VIPP launcher can therefore start its
full child process normally and napari remains extensible.

The conda recipe contains two exact-version local packages:

- `napari-vipp`, installed from the wheel built from the same source checkout;
- `vipp-menu`, containing the macOS menu metadata and generated ICNS icon.

Constructor resolves the rest from conda-forge and embeds the complete solved
environment for offline installation. The release manifest records the source,
wheel, staging PKG, and local-package digests. Finalization also preserves
architecture-qualified constructor information, licence inventory, explicit
lockfile, package list, and a SHA-256 checksum file.

The alpha lockfile still records the build's temporary local channel URI for
the two VIPP wrapper packages. Their exact hashes remain in the build/release
manifests and their bytes are embedded in the offline PKG. A signed production
build should instead publish those packages to a durable channel and configure
constructor `channels_remap` so its lockfile has durable provenance.

## Build a local development PKG

Build on the same architecture as the target package:

```bash
conda env create \
  --prefix .venv-macos-installer \
  --file packaging/macos/builder-environment.yml
conda run --prefix .venv-macos-installer \
  python -m build --wheel --no-isolation --outdir dist/python
conda run --prefix .venv-macos-installer \
  python scripts/package_macos_installer.py build \
  --development \
  --wheel dist/python/napari_vipp-0.14.0a3-py3-none-any.whl \
  --output-directory dist/macos-development
```

The expected Apple Silicon filename is:

```text
VIPP-0.14.0a3-macOS-arm64-DEVELOPMENT.pkg
```

Use `x86_64` instead of `arm64` for a package built natively on Intel. A quick
plan that performs no package build is available with `--plan-only`.

## Build and finalize an explicitly unsigned alpha

Use a clean checkout at the immutable exact tag. The source version must have
the `X.Y.ZaN` form, the expected `v<version>` tag must point at `HEAD`, and the
working tree must be clean. The builder also exports that exact commit, rebuilds
the wheel with the pinned toolchain, and compares archive contents with the
supplied release wheel before constructing the PKG.

```bash
git switch --detach v0.14.0a3
test "$(git describe --tags --exact-match)" = "v0.14.0a3"
test -z "$(git status --porcelain --untracked-files=all)"

conda run --prefix .venv-macos-installer \
  python -m build --wheel --no-isolation --outdir dist/python
conda run --prefix .venv-macos-installer \
  python scripts/package_macos_installer.py build \
  --wheel dist/python/napari_vipp-0.14.0a3-py3-none-any.whl \
  --output-directory dist/macos-staging

conda run --prefix .venv-macos-installer \
  python scripts/package_macos_installer.py finalize-unsigned \
  --unsigned-staging-installer \
    dist/macos-staging/VIPP-0.14.0a3-macOS-arm64-SIGNING-STAGING.pkg \
  --build-manifest \
    dist/macos-staging/VIPP-0.14.0a3-macOS-arm64-SIGNING-STAGING-build.json \
  --output-directory dist/macos-release
```

`finalize-unsigned` rechecks the clean exact tag, architecture, complete staging
hash and size, unsigned signature state, PKG archive, and every constructor
evidence digest. It publishes atomically and refuses to overwrite any output.
For Apple Silicon the final release directory contains only:

```text
VIPP-0.14.0a3-macOS-arm64-UNSIGNED.pkg
VIPP-0.14.0a3-macOS-arm64-UNSIGNED-release.json
VIPP-0.14.0a3-macOS-arm64-UNSIGNED-constructor-info.json
VIPP-0.14.0a3-macOS-arm64-UNSIGNED-licenses.json
VIPP-0.14.0a3-macOS-arm64-UNSIGNED-lockfile.txt
VIPP-0.14.0a3-macOS-arm64-UNSIGNED-package-list.txt
SHA256SUMS-macOS-arm64-0.14.0a3.txt
```

The staging PKG and its generic constructor sidecars are review inputs, not
release assets. Never publish a `-SIGNING-STAGING` or `-DEVELOPMENT` package.

The `Build unsigned release installers` workflow accepts an existing tag,
checks out its complete history, verifies the tag/version/main ancestry and
native architecture, runs the focused tests, and builds/finalizes separate
Apple Silicon and Intel artifact sets. It intentionally retains Actions
artifacts for human release review; attaching them to the GitHub prerelease is
a separate publication decision.

## Verify and install the unsigned alpha

Download the PKG and matching checksum file only from the official GitHub
prerelease. In Terminal, from the download directory:

```bash
shasum -a 256 -c SHA256SUMS-macOS-arm64-0.14.0a3.txt
pkgutil --check-signature VIPP-0.14.0a3-macOS-arm64-UNSIGNED.pkg
```

Every checksum line must report `OK`. Because the alpha is deliberately
unsigned, `pkgutil` must report `Status: no signature`; a Developer ID identity
is not expected for this filename.

Double-click the `.pkg`. If macOS blocks it, first confirm the official release
URL, exact `-UNSIGNED.pkg` filename, and SHA-256. Then open **System Settings >
Privacy & Security** and use **Open Anyway** for this specific installer. Never
disable Gatekeeper globally and never run a command that removes quarantine
from unrelated files. Work/school security policy may not offer an override;
use the manual Python installation route on those Macs.

The current installer does not claim Windows-style update/repair/uninstall
parity. Remove this alpha by moving only these installer-owned paths to Trash:

```text
~/Applications/VIPP.app
~/Library/vipp
```

Before publishing the first macOS alpha, review the complete constructor
licence inventory (especially Qt/PySide redistribution obligations) and record
fresh install/launch evidence for both architectures. Also manually exercise
the browser-download quarantine and **Open Anyway** journey on a clean Mac; the
automated runner uses `installer -allowUntrusted` only to validate the package
lifecycle in an ephemeral account.

## Signed production gate

An ordinary public release should replace the alpha exception with all of the
following:

1. Import Apple **Developer ID Application** and **Developer ID Installer**
   identities into an ephemeral build keychain.
2. Supply constructor's executable-content and installer signing identities
   without exposing certificate material in manifests or logs.
3. Submit the signed PKG with `xcrun notarytool ... --wait`, then run
   `xcrun stapler staple` and `xcrun stapler validate`.
4. Require `pkgutil --check-signature` and
   `spctl --assess --type install --verbose=4` to accept the exact artifact.
5. Test install, first launch, representative workflow/export, replacement,
   failure cleanup, and removal on clean Apple Silicon and Intel Macs.
6. Publish the two VIPP conda packages to a durable release channel and remap
   the build channel to it.

Apple certificates and notarization credentials are intentionally absent from
the repository.
