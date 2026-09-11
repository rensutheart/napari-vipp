# Update discovery and reader packaging

Engineering record, updated 2026-09-11. Public instructions belong in the
[manual](https://rensutheart.github.io/vipp-mkdocs/nightly/getting-started/updating/).

## Implemented update boundary

- `core/updates.py` compares published GitHub releases using PEP 440 versions
  and derives official asset URLs. Drafts and development/local tags are
  excluded. Prereleases default on for alpha installs and off for final ones.
- `ui/updates.py` owns asynchronous Qt networking and a persistent release cache.
  Unreleased after 0.15.0a4: automatic checks run eight seconds after every
  launch, independent of a previous launch's check/failure. Repeat checks in
  a single session remain daily; explicit checks bypass that interval.
  Automatic results change only the badge, without opening a dialog.
- The request uses the public releases list (up to 100 entries), a generic
  user agent, HTTPS, no API redirects, an 8-MiB ceiling, a 15-second socket
  timeout, and a 20-second total deadline. No workflow, file path, image,
  installed version, or hardware details enter the request. One transient retry
  shares the same total deadline. TLS verification is never bypassed. Typed
  local diagnostics distinguish TLS, proxy, DNS, HTTP and timeout failures.
- `ui/update_dialog.py` separates status, installed/latest versions, actions and
  preferences. A failed check takes priority over cached availability; cached
  releases and the last-success time are explicitly labeled. Managed download
  handoff requires a fresh successful check and a separate user click.
- A matching native installer is offered only with its published checksum
  sidecar. Unreleased: `core/update_install.py` and `ui/update_download.py`
  support an explicit Windows desktop **Download & open update** action.
  Detection must bind this desktop's actual prefix and installed package to
  valid active ownership, version and candidate marker; nearby, retired,
  editable, plugin, pip/conda and macOS installations do not qualify.
- The download snapshot binds the exact release, asset names, ownership and
  compute track. HTTPS asset redirects are allowlisted; checksum/installer
  streams have independent size limits, bounded reads and a total deadline.
  Partial, cancelled or hash-mismatched downloads are not executable handoffs.
  Before handoff, ownership and downloaded bytes are checked again; Windows
  denies writes/deletion during the final hash check and process creation.
- Setup opens with the current owned root and CPU/CUDA profile selected. Its
  existing review/approval and side-by-side installation transaction remain
  authoritative. This path does not auto-approve, quit/kill VIPP, restart the
  application, or mutate an arbitrary Python environment. macOS and unmanaged
  sessions retain explicit browser/manual routes. Unsigned installers remain
  identified as unsigned; a checksum is not a publisher signature.
- Download progress/cancellation stays off the GUI thread. Late completions
  cannot launch after cancellation, owner shutdown or while a close-confirmation
  question is awaiting the user's answer. No workflow or scientific state is
  passed to the updater. Successful setup handoff is not reported as a completed
  installation.
- `VIPP_DISABLE_UPDATE_CHECKS=1` suppresses automatic requests, not explicit
  manual checks. Tests isolate settings and use fake network replies.

## Reader packaging — implemented, unreleased

Approved scope: the ordinary plugin and managed desktop installations have
the same native readers. This supersedes the original lean-plugin proposal.
Windows managed CPU still requests `app`; GPU adds `gpu-cuda13`. Native readers
are base requirements, so both routes receive them without special extras.
The macOS conda recipe mirrors those requirements, including `imagecodecs` for
ND2's legacy codec support.

| Default dependency range | Native route |
| --- | --- |
| `czifile>=2026.8.16,<2027` | Zeiss CZI, without the BioIO CZI wrapper |
| `liffile>=2026.7.14,<2027` | Leica LIF/LOF/XLIF |
| `nd2[legacy]>=0.11.1,<0.12` | Nikon ND2, including legacy compression |
| `oiffile>=2026.2.8,<2027`, `oirfile>=2026.7.28,<2027` | Olympus OIF/OIB/OIR |
| `imagecodecs>=2026.8.16,<2027` | CZI codecs and legacy ND2 compression |

These pure-Python reader projects use BSD licences. The codec distribution
contains native libraries and their own notices; PyPI/conda distribution
licence files must travel with the packages. Required metadata dependencies
are resolved transitively; plotting and BioIO wrappers are not default extras.
Version ranges accept maintenance releases, not unreviewed next-year reader
API changes. The minimum CZI/codec pair covers chunked compression introduced
in the August reader release. Effective NumPy minimum is 2.1 via these readers;
the qualified CUDA stack's NumPy 2.5.1 remains compatible and is not changed.

The historical `czi`, `nd2`, `microscope` and `bioformats` extras remain valid
for compatibility. The broad extras are not used by default. Optional setup
uses only `bioio>=3.4,<4` and `bioio-bioformats>=2,<3` for Bio-Formats, avoiding
unnecessary native BioIO wrappers. Bio-Formats 2 uses `bffile`/`scyjava`/`cjdk`:
Java and Maven artifacts may be downloaded at first use, not by a status check.

## Image Source diagnostics and setup boundary

- **Reader support** is a collapsed file-source section, never a Settings item.
  An import failure reveals relevant actions without expanding the full list.
- Dependency metadata checks do not import readers. Explicit load checks run
  in disposable processes with a 30-second limit. A timeout/native crash is a
  failed check, not a hung GUI. Leaving the inspector cancels its process.
- Ready means dependencies/imports worked (plus a tiny lossless JPEG 2000
  codec round trip where applicable). It does not validate Java startup, a
  vendor acquisition, or its scientific metadata.
- Retry preserves the authored source and its saved identity/axis contract.
  A stale source-load generation cannot publish a new recovery action.
- Setup is a separate process with two explicit decisions: review/download,
  then approve installation after the application closes. It never kills the
  user's session or auto-reopens a workflow.
- Only fixed catalog recipes are accepted; exception text is never executed.
  Resolution pins every installed distribution. Plans reject replacements,
  source builds, non-PyPI artifact hosts and missing SHA-256 hashes. Downloaded
  wheels are installed offline with hashes and no dependency re-resolution.
- The parent PID/creation time and sibling environment processes must be gone;
  the installed-version snapshot and reviewed requirements are checked again
  immediately before installation. Users must not reopen the environment
  during setup. Existing package files are not selected for replacement.
- Shared/system, read-only or externally managed Python is refused. Existing
  broken/incompatible installations get retry/help rather than an implicit
  scientific-stack upgrade. Failed installation may leave some additions;
  errors are shown and there is no destructive rollback.

## Qualification and release gates

`test_reader_support.py` covers catalog/recipe parity, missing versus broken
imports, isolated probe teardown, inline actions, stale failures, hash/host
validation, immutable installed packages and close-before-install guards.
Existing source inspection/error/metadata and installer tests also run.

Clean wheel/sdist CI now requires all native imports and codec checks. A new
Intel macOS wheel job complements Apple Silicon; the installed macOS PKG
smoke job checks readers on both architectures. These are mandatory checks,
not evidence that a locally edited workflow has already run remotely.

Retain dated clean-install and frozen-corpus results before release. Windows
checks do not establish native macOS runtime correctness. OIF companion-tree
tests and LOF/XLIF API support are not blanket real-acquisition qualification.
Do not change source selectors or accept a new reader topology to make a file
pass. Native readers still have format boundaries, including CZI multi-file
and topography limitations.

### Local evidence — 2026-09-07

- Fresh Windows Python 3.12 environment installed the ordinary `[app]` package
  with no microscope extra. All six native reader/codec checks and `pip check`
  passed. BioIO/Bio-Formats was absent for the native-file acceptance run.
- The installed wheel (not the source checkout) passed 16 tests over eight
  byte/hash-verified public ND2, LIF, CZI/multiscene, OIR, OIB and LSM files.
  These included the corpus's strict metadata and declared pixel evidence.
  Reader versions were CZI 2026.8.16, LIF 2026.7.14, ND2 0.11.3,
  OIF 2026.2.8, OIR 2026.9.6 and imagecodecs 2026.8.16.
- In that disposable environment, real Bio-Formats setup resolved, downloaded,
  installed and import-checked the approved additions after the parent exited.
  All 141 pre-existing distribution versions stayed unchanged. No Java runtime
  was started. The Windows venv redirector is explicitly excluded from the
  helper's own process guard; the calling application is not.
- Reader/source/installer regressions passed 146 tests (one native macOS tool
  test skipped). The broader source UI run exposed six synthetic batch-demo
  hash failures from the preceding Binary Threshold default; both packaged
  demo copies now explicitly author `Foreground: Above`. All six then passed
  in a 44-test focused rerun. No scientific hash check was weakened.
- Ruff, manifest validation and manual content/strict-build checks passed.
  Reader UI and the new manual page were inspected at narrow/desktop widths
  in light/dark themes. Native macOS execution remains a release gate above.

Local logs are `.cache/reader-clean-corpus.log`, `reader-clean-probes.json`,
`reader-setup-integration.json`, `reader-regressions.log` and
`reader-final-ui-tests.log`; these are development evidence, not published
release artifacts. Public fixture hashes remain in corpus-v4; nothing was
added to Git from the downloaded image cache.

Sources: [czifile](https://pypi.org/project/czifile/),
[liffile](https://pypi.org/project/liffile/), [nd2](https://pypi.org/project/nd2/),
[oiffile](https://pypi.org/project/oiffile/), [oirfile](https://pypi.org/project/oirfile/),
[imagecodecs](https://pypi.org/project/imagecodecs/),
[BioIO requirements](https://bioio-devs.github.io/bioio/), and
[bioio-czi](https://pypi.org/project/bioio-czi/).
