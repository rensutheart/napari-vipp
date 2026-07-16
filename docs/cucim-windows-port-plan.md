# cuCIM native-Windows port and distribution plan

Date: 2026-07-16
Status: approved planning direction; the existing `cucim.skimage` wheel remains
research evidence and no downstream cuCIM package is a supported VIPP
dependency yet.

## Purpose and relationship to the GPU plan

This document owns the work needed to make a maintainable native-Windows cuCIM
distribution, including the optional `cucim.clara`/`CuImage` native image-I/O
surface. It covers the upstream fork, Windows C++ port, package matrix, CI,
release engineering, installation support, and upstream contribution strategy.

The [production GPU implementation plan](gpu-production-implementation-plan.md)
continues to own VIPP's provider contracts, execution policy, scientific parity,
memory and cancellation behavior, operation promotion, UI, and product support
matrix. A successful Windows cuCIM build does not itself enable a VIPP GPU
operation. The
[native-Windows source evaluation](cucim-windows-source-evaluation.md) remains
the evidence record for the first `cucim.skimage` build and benchmark.

## Bottom line

The preferred path is a thin, upstream-tracking cuCIM fork with Windows CI and
small upstreamable patches. It is not a permanent algorithm fork and it is not
a series of unrecorded edits applied to disposable source checkouts.

Development proceeds on two tracks:

1. Package the already working Python/`cucim.skimage` surface as an explicitly
   experimental Windows preview so wider benchmarks can continue.
2. Port the actively used native stack (`libcucim`, the `_cucim` Python
   extension, and the `cuslide2` image-I/O plugin) to Windows, then add Clara to
   the release only after its correctness and packaging gates pass.

VIPP must not wait for Clara before evaluating the high-value operations already
shown to benefit from cuCIM. Conversely, a preview skimage wheel must not be
described as full cuCIM.

## 1. Upstream stability and maintenance contract

cuCIM is active and suitable for pinned use, but it is not an API- or
dependency-frozen library. The
[upstream changelog](https://github.com/rapidsai/cucim/blob/main/CHANGELOG.md)
shows a roughly two-month RAPIDS release cadence and recent breaking changes in
the scikit-image compatibility and nvImageCodec layers. CUDA, CuPy, NumPy,
scikit-image, Python, and nvImageCodec compatibility move between releases.

Two local adaptations from the pinned `v26.06.00` evaluation were already
superseded on upstream `main`: the NumPy `reshape` correction was merged in
[PR 1106](https://github.com/rapidsai/cucim/pull/1106), and CuPy wheel
dependencies gained the `[ctk]` extra in
[PR 1104](https://github.com/rapidsai/cucim/pull/1104). This is evidence that
the fork must continuously drop patches accepted upstream instead of carrying
copies indefinitely.

The maintenance rules are:

- every downstream release is based on an immutable upstream tag and commit;
- Windows changes remain platform abstractions, build changes, tests, and
  packaging wherever possible;
- scientific Python algorithms are not modified downstream unless a separate
  parity-reviewed fix is unavoidable;
- each upstream release is evaluated before adoption rather than pulled into a
  published wheel automatically;
- accepted upstream fixes are removed from the downstream patch series;
- VIPP pins an admitted downstream release range and never follows the fork's
  development branch;
- build provenance records the upstream tag/commit, downstream commit, compiler,
  Python ABI, CUDA major/minor, dependency lock, and artifact digest.

## 2. Scope

### 2.1 Initial supported targets

| Dimension | Initial target |
| --- | --- |
| Operating system | Native 64-bit Windows 10/11 on x86-64 |
| Python | CPython 3.11, 3.12, 3.13, and 3.14 |
| CUDA packaging | Separate CUDA 12.x and CUDA 13.x distributions |
| GPU vendor | NVIDIA only |
| Validation devices | RTX 5090/Blackwell and RTX 4060 Laptop/Ada initially |
| Linux | Continue to use and compare against official upstream packages |
| macOS | No NVIDIA execution; VIPP remains CPU-only |

CUDA 12 and CUDA 13 must remain distinct package tracks because
`cupy-cuda12x` and `cupy-cuda13x` conflict. CUDA 12 supplies the broader driver
and legacy-architecture path; CUDA 13 supplies the current stack for newer
systems. Supporting CUDA 11 would require an older CuPy/cuCIM maintenance line
and is not part of the initial port.

### 2.2 Included surfaces

- `cucim.skimage` and the Python helpers already present in the preview wheel;
- `cucim.clara.CuImage` metadata, pyramid, associated-image, and region-read
  behavior supported by the modern native plugin;
- ordinary host-file I/O on Windows;
- CPU-staged and CUDA-device image decode paths where nvImageCodec supports
  them;
- the per-process image cache after concurrency and memory tests pass;
- the `cuslide2` plugin and the formats its upstream tests and declared support
  matrix validate;
- actionable capability reporting when an optional codec or native feature is
  unavailable.

### 2.3 Explicit non-goals

- GPUDirect Storage/cuFile on Windows. NVIDIA's GDS implementation is
  Linux-specific; Windows uses ordinary file I/O and staged transfers.
- NVIDIA execution on Apple Silicon or current macOS.
- Porting every historical benchmark, example, or deprecated native target
  merely because it exists in the repository.
- Preserving a legacy plugin when `cuslide2` supplies the required supported
  behavior.
- Installing or updating NVIDIA display drivers automatically.
- Publishing an unofficial wheel under metadata that implies NVIDIA/RAPIDS
  built or supports it.
- Allowing cuCIM arrays or native objects to leak through VIPP's host-output,
  workflow, batch, or export contracts.

"Maximum benefit" therefore means all maintainable, user-relevant Windows
functionality. It does not mean emulating unavailable kernel facilities or
shipping unused legacy targets.

## 3. Distribution and repository model

### 3.1 Fork topology

Create a public fork of `rapidsai/cucim` in the VIPP-maintaining organization.
Configure two remotes locally:

- `upstream`: `rapidsai/cucim`, read-only for synchronization;
- `origin`: the VIPP-maintained fork, used for Windows branches and releases.

Keep the fork's `main` synchronized with upstream. Development branches should
contain one reviewable portability concern each. Downstream release branches
are cut from an immutable upstream tag and receive only the accepted Windows
patch series.

Recommended refs are:

| Ref | Purpose |
| --- | --- |
| `main` | Mirror of upstream `main` |
| `windows/dynlib` | Windows library loading abstraction |
| `windows/file-io` | Windows ordinary file-I/O implementation |
| `windows/cmake` | MSVC and Windows target/link/install rules |
| `windows/cuslide2` | nvImageCodec and plugin discovery port |
| `release/<upstream>-windows` | Frozen downstream release preparation |
| downstream release tag | Immutable source for published artifacts |

Do not accumulate the entire port in one unreviewable branch. Small commits
make upstream review, regression bisection, and patch removal practical.

### 3.2 Package identity

Until RAPIDS publishes or endorses Windows wheels, use a clearly distinct
distribution name and describe it as an unofficial downstream build. The
Python import namespace remains `cucim` for compatibility, so pip must prevent
the upstream and downstream distributions from being installed together.

The final names, publication index, and trademark wording require approval
before public release. Working names such as `vipp-cucim-cu12` and
`vipp-cucim-cu13` are acceptable for private or preview artifacts but are not a
branding decision.

Every release must include:

- the upstream and downstream source identifiers;
- Apache-2.0 and third-party notices;
- an explicit unofficial-build statement;
- supported and tested platform tables;
- SHA-256 digests and a machine-readable artifact manifest;
- dependency and toolchain versions;
- known feature differences, especially the absence of GDS on Windows;
- reproducible build commands and CI run references.

## 4. Target Windows artifact

### 4.1 Preview wheel

The preview artifact contains Python sources and runtime-compiled CUDA kernels,
but no Clara native library. The current CPython 3.12 tag is produced by
upstream packaging that assumes a vendored native extension. It must not be
changed by renaming the wheel or editing metadata after the build.

For the first preview release, build and test explicit Python-minor wheels for
all admitted Python/CUDA combinations. An ABI-independent skimage-only wheel may
be considered later after the build metadata is corrected and all supported
Python versions prove identical behavior.

### 4.2 Full wheel with Clara

The full artifact contains native Python and C++ binaries and is genuinely
CPython-ABI-specific. Its required payload is expected to include:

- `cucim/clara/_cucim.cp3xx-win_amd64.pyd`;
- `libcucim.dll` or the approved Windows library name;
- the Windows `cuslide2` plugin DLL;
- Python `cucim`, `cucim.core`, `cucim.skimage`, and `cucim.clara` sources;
- licenses, notices, version files, and plugin configuration;
- no bundled duplicate of a dependency DLL when the declared nvImageCodec/CUDA
  package supplies and owns that DLL.

The initial full release matrix is eight wheels:

| Python | CUDA 12 | CUDA 13 |
| --- | --- | --- |
| 3.11 | `cp311-cp311-win_amd64` | `cp311-cp311-win_amd64` |
| 3.12 | `cp312-cp312-win_amd64` | `cp312-cp312-win_amd64` |
| 3.13 | `cp313-cp313-win_amd64` | `cp313-cp313-win_amd64` |
| 3.14 | `cp314-cp314-win_amd64` | `cp314-cp314-win_amd64` |

Each CUDA track depends on exactly one matching CuPy distribution with `[ctk]`
and the matching nvImageCodec CUDA-major package. A clean user environment must
need a compatible NVIDIA display driver, not MSVC, CMake, nvcc, or a system-wide
CUDA Toolkit.

## 5. Current Windows blockers

The first source audit found a feasible porting surface, not a metadata-only
fix. The principal blockers are:

| Area | Current assumption | Required Windows result |
| --- | --- | --- |
| Version files | Repository-relative symbolic links | Ordinary packaged files or a build-generated version module |
| RAPIDS Python backend | Invokes Unix `which` | Platform-neutral executable discovery |
| CMake/compiler | GCC flags, `_GLIBCXX` ABI, RPATH and soname assumptions | MSVC-safe options and Windows install/runtime rules |
| Dynamic libraries | `dlopen`, `dlsym`, `dlclose`, `.so` names | `LoadLibraryEx`, `GetProcAddress`, `FreeLibrary`, safe DLL search paths |
| File handles | POSIX descriptors, `pread`/`pwrite`, `mmap`, `off_t`/`ssize_t` | 64-bit Windows file offsets, positioned reads, mapping, Unicode paths |
| Shared cache | POSIX shared-memory assumptions | Disabled initially or implemented with named Windows mappings after tests |
| GDS/cuFile | Linux kernel and filesystem integration | Compile-time unsupported capability; ordinary I/O remains functional |
| Plugins | ELF module naming, `$ORIGIN`, soname and `.so` discovery | DLL naming, discovery, lifetime, and dependency-directory control |
| nvImageCodec | Linux `.so` dynamic wrapper | Load the official Windows `nvimgcodec_0.dll` and extensions safely |
| Python binding | Linux `.so` build/install layout | Per-CPython `_cucim.pyd` built and imported with MSVC |
| Tests | Linux paths and POSIX fixtures | Native-Windows unit, integration, concurrency, packaging, and GPU tests |

The exact compatible nvImageCodec 0.8 release already publishes a Windows
x86-64 wheel with the core DLL, codec DLLs, headers, and Python 3.11-3.14
extensions. That removes a major external dependency risk, but installing it
does not port `libcucim` or Clara by itself.

## 6. Work plan

### Phase 0 — Establish the fork and upstream agreement

**Work**

- Create the fork and configure `upstream`/`origin` remotes.
- Post a concise design proposal on
  [upstream issue 454](https://github.com/rapidsai/cucim/issues/454) before the
  native port expands.
- Confirm with maintainers which native plugin is current and which Windows
  abstractions they would accept.
- Record the upstream tag/commit and generate a downstream patch inventory.
- Add a Windows-port status document in the fork.

**Acceptance**

- Fork governance and release ownership are named.
- Every initial patch has an upstream issue/PR destination or an explicit
  downstream-only reason.
- No public artifact uses ambiguous upstream branding.

### Phase 1 — Make the skimage-only build a real preview package

**Work**

- Replace the source-checkout symlink workaround with build-safe version
  generation.
- Make executable discovery platform-neutral.
- Correct classifiers, dependency extras, package name, feature declarations,
  and console entry points.
- Ensure Clara is reported as unavailable rather than exposing a broken CLI.
- Parameterize the builder for Python 3.11-3.14 and CUDA 12/13.
- Add deterministic source pins, artifact manifests, checksums, and clean-install
  probes.

**Acceptance**

- Eight preview wheel jobs build from clean checkouts.
- Each wheel installs in a clean environment with only the driver as a system
  prerequisite and executes representative real CUDA kernels.
- `cucim.is_available("skimage")` is true and Clara capability is false with an
  actionable explanation.
- The existing selected upstream test and benchmark suites pass on the RTX 5090
  and RTX 4060 Laptop validation hosts.

### Phase 2 — Make the native core compile with MSVC

**Work**

- Add compiler- and platform-conditioned CMake flags.
- Exclude Linux-only GDS targets when building Windows.
- Make dependency acquisition and install paths Windows-safe.
- Implement the Windows dynamic-library helper with restricted DLL search
  behavior; never search the current working directory implicitly.
- Build `libcucim.dll` with an intentional export/import contract.
- Build a minimal native-core test executable before attempting Python
  packaging.

**Acceptance**

- Clean Release and Debug configurations compile with the supported Visual
  Studio Build Tools version.
- Dependency DLLs are resolved from declared package/application directories.
- Core/plugin load and unload tests pass repeatedly without handle leaks.
- Linux builds and tests remain unchanged by the platform conditionals.

### Phase 3 — Port ordinary file I/O and cache foundations

**Work**

- Implement 64-bit positioned read/write behavior using Windows APIs without a
  shared seek-position race.
- Add UTF-16 path conversion at the OS boundary and preserve Python Unicode
  paths.
- Implement or adapt read-only file mappings used by image access.
- Compile GDS/cuFile APIs as an explicit unsupported capability on Windows.
- Enable only the in-process cache first.
- Defer cross-process shared-memory cache until named mapping, cleanup, crash,
  permission, and collision semantics are designed and tested.

**Acceptance**

- Large-file offsets beyond 4 GiB, Unicode paths, spaces, long paths, and
  concurrent positioned reads pass.
- Resource handles and mappings return to baseline after repeated open/read/
  close and failure cycles.
- Unsupported GDS requests fail with a typed capability result rather than a
  compile error or crash.
- In-process cache eviction, capacity, concurrency, and cleanup tests pass.

### Phase 4 — Build the Python extension and Clara core

**Work**

- Build `_cucim.pyd` for one initial Python 3.12/CUDA 13 configuration.
- Package `libcucim.dll` beside the extension using an explicit runtime loading
  policy.
- Adapt plugin-root and configuration discovery to Windows paths.
- Bind only native APIs whose backing implementation passes the Windows tests;
  unavailable filesystem/GDS APIs must say why they are unavailable.
- Test interpreter shutdown, repeated import, subprocess import, and virtual
  environments.

**Acceptance**

- `import cucim.clara` succeeds without PATH modification or copying DLLs into
  system directories.
- `CuImage` opens, reports metadata, reads host regions, closes, and releases all
  handles for a minimal supported fixture.
- Malformed, truncated, inaccessible, and unsupported files fail safely.
- The corresponding official Linux build produces equivalent public metadata
  and region values for the shared fixture.

### Phase 5 — Port and validate `cuslide2`/nvImageCodec

**Work**

- Replace the Linux nvImageCodec wrapper with a platform abstraction that loads
  `nvimgcodec_0.dll` from the installed dependency package.
- Port plugin target names, install layout, discovery, and lifetime management.
- Validate the upstream-supported tiled-TIFF/whole-slide formats rather than
  claiming every nvImageCodec codec automatically.
- Exercise CPU and CUDA output devices, pyramid levels, batches, associated
  images, interpolation, cache interaction, and asynchronous streams.
- Compare Windows outputs with official Linux cuCIM for the same corpus.

**Acceptance**

- The mandatory whole-slide corpus passes metadata and pixel-region parity.
- Edge, out-of-bounds, non-aligned, multi-tile, batch, and multi-threaded reads
  pass.
- CUDA reads synchronize correctly and expose no use-after-free on stream or
  image destruction.
- Repeated reads and failure injection show no material host, device, handle, or
  thread leak.
- Performance is measured, but correctness and safety remain release blockers
  even when the speedup is large.

### Phase 6 — Expand the ABI/CUDA matrix

**Work**

- Expand the proven Python 3.12/CUDA 13 native build to all eight release jobs.
- Add CUDA-major-specific dependency constraints and conflict detection.
- Run clean install, uninstall, repair, and upgrade tests.
- Verify the CUDA 12 build on the two available validation devices and add an
  older NVIDIA architecture before making a legacy-coverage claim.
- Compare Windows native results with supported Linux packages.

**Acceptance**

- All eight wheels build, install, import, and pass the required test tiers.
- One environment can never contain both CuPy CUDA-major packages or both
  upstream and downstream cuCIM distributions without a clear resolver error.
- Wheel inspection finds no absolute build paths, undeclared DLLs, debug
  runtimes, or files from another Python ABI.

### Phase 7 — Release engineering, VIPP admission, and upstreaming

**Work**

- Add signed/checksummed release artifacts and a machine-readable manifest.
- Publish PowerShell installation, diagnosis, repair, and uninstall scripts.
- Submit the port upstream as small PRs with Linux non-regression evidence.
- Run the GPU-plan provider admission gates independently for each VIPP
  operation.
- Document supported versus merely expected GPUs and formats.

**Acceptance**

- A clean user machine follows one supported command path and completes the
  post-install kernel plus Clara probes.
- Release artifacts reproduce from the tagged fork source and pinned toolchain.
- VIPP remains fully importable and CPU-capable when the downstream package is
  absent or broken.
- Only operation implementations clearing the production GPU plan's parity,
  memory, cancellation, provenance, and benefit gates are enabled.

## 7. CI and validation design

### 7.1 Test tiers

| Tier | Trigger | Required evidence |
| --- | --- | --- |
| Static and configure | Every Windows-port PR | Formatting, generated-file check, CMake configure, dependency and license audit |
| Native build | Every Windows-port PR | MSVC core, plugin, and Python-extension builds for the primary configuration |
| CPU/native unit | Every Windows-port PR | DLL loader, file I/O, mapping, cache, metadata, failure and cleanup tests |
| Package smoke | Every Windows-port PR | Clean wheel install/import/uninstall without developer tools on PATH |
| GPU smoke | Trusted/self-hosted PR or queued validation | CuPy kernel, skimage kernels, CuImage CUDA region read, synchronization |
| Full GPU/WSI | Scheduled and release candidate | Upstream selected tests, WSI corpus, concurrency, leak, memory and benchmark suites |
| Matrix release | Release candidate | Python 3.11-3.14 by CUDA 12/13, artifact inspection and install scripts |
| Linux parity | Release candidate | Official Linux cuCIM on shared scientific and WSI fixtures |

GitHub-hosted Windows runners may build artifacts and run non-GPU tests. They
do not replace real NVIDIA validation. The RTX 5090 and RTX 4060 Laptop can be
release-validation runners or manually attested hosts; a public CI runner must
not expose personal credentials or unrelated files.

### 7.2 Required Clara fixture coverage

The repository must use redistributable or generated fixtures with recorded
licenses and hashes. At minimum, cover:

- a small generic tiled TIFF;
- a pyramidal SVS fixture if redistribution terms allow it;
- a multi-level/sub-IFD TIFF fixture;
- an associated image when supported;
- tiles that cross storage boundaries and image edges;
- a file larger than 4 GiB, generated or sparsely constructed where practical;
- malformed and truncated variants;
- Unicode, spaced, and long Windows paths.

Additional vendor formats enter the public matrix only with legal fixtures,
metadata/pixel expectations, and Linux/Windows parity evidence.

### 7.3 Benchmark rules

- Record cold process, warm process, resident-device, and end-to-end timings.
- Synchronize every timed CUDA boundary.
- Report decode, host-to-device, device-to-host, cache-hit, and cache-miss costs
  separately where observable.
- Record device, compute capability, VRAM, power mode, driver, CUDA, Python,
  cuCIM, CuPy, nvImageCodec, image hash, region, and repetition counts.
- Never promote a VIPP operation or I/O path from a single GPU result.
- Treat laptop power/thermal behavior as part of the result rather than noise to
  hide.

## 8. Installation and support experience

Provide an `install-cucim-windows.ps1` entry point with parameters similar to:

```powershell
.\install-cucim-windows.ps1 -Cuda Auto -Python python
```

The script should:

1. verify native Windows x86-64 and a supported Python version;
2. detect the NVIDIA GPU and driver with `nvidia-smi`;
3. select CUDA 12 or 13, defaulting conservatively when both are possible;
4. detect and stop on conflicting CuPy or cuCIM distributions;
5. create or use an explicitly selected virtual environment;
6. install exactly one matching CuPy `[ctk]` distribution, nvImageCodec, and
   downstream cuCIM wheel;
7. verify package hashes when installing release assets directly;
8. run a real CuPy kernel, representative skimage operations, and the Clara
   fixture probe when Clara is included;
9. emit a JSON diagnostic report with no raw GPU serial/UUID;
10. print exact repair or uninstall commands on failure.

The script must not install a display driver, alter machine-wide CUDA settings,
copy DLLs into Windows system directories, or silently modify an existing
environment containing conflicting GPU packages.

## 9. Upstream contribution sequence

Submit changes in an order that gives upstream useful improvements even if the
complete port is not accepted immediately:

1. remove or generate the version-file symlinks during packaging;
2. replace Unix-only executable discovery in the Python build;
3. isolate GCC/ELF CMake flags behind platform conditions;
4. add the cross-platform dynamic-library abstraction and its tests;
5. add Windows ordinary file I/O and mapping abstractions;
6. make GDS an explicit platform capability;
7. port nvImageCodec DLL discovery and `cuslide2` target rules;
8. add `_cucim.pyd` packaging and Windows unit/package CI;
9. add optional real-GPU Windows validation if upstream has suitable runners;
10. propose official Windows wheels only after the preceding pieces are stable.

Each PR should link issue 454, explain Linux behavior preservation, include
tests, and avoid unrelated formatting or algorithm changes.

## 10. Risks and go/no-go gates

| Risk | Mitigation / gate |
| --- | --- |
| Upstream changes faster than the port | Pin tags, automate range comparisons, keep patches small, adopt releases deliberately |
| Native port becomes a permanent large fork | Upstream early, avoid algorithm divergence, publish patch inventory and ownership |
| DLL search or packaging is unsafe | Restricted search directories, artifact inspection, clean-machine package tests |
| Windows file semantics change results | Linux/Windows metadata and pixel-region parity corpus |
| Clara works but leaks under repetition | Handle/thread/host/device leak suites and failure injection are release blockers |
| CUDA/Python matrix is too costly | Prove one primary configuration first; expand only after the architecture passes |
| Old GPUs are claimed without evidence | Separate CUDA 12 artifact and validate an older architecture before advertising it |
| Users confuse preview and full builds | Distinct capability manifest, package description, versioning, and install probes |
| nvImageCodec changes its C API | Pin its supported range and require an explicit update review per upstream release |
| Fork owner is unavailable | Document release keys, runner ownership, upstream sync, and rollback procedures |

The native Clara effort stops or is re-scoped if any of these conditions holds:

- a required proprietary dependency has no redistributable Windows runtime;
- ordinary Windows file I/O cannot preserve the public `CuImage` correctness
  contract without an unmaintainable rewrite;
- the tested native stack remains crash- or leak-prone after the bounded port;
- maintaining the patch series consumes more effort than its measured VIPP
  value justifies;
- upstream rejects necessary platform abstractions and no sustainable fork
  owner is approved.

A no-go for Clara does not invalidate the skimage-only provider. The preview
package and individual VIPP operation admission remain independently removable.

## 11. Completion definition

The Windows cuCIM port is complete only when:

- the fork and upstream synchronization process are documented and exercised;
- all eight full wheels build reproducibly from a downstream release tag;
- clean Windows environments install without a compiler or system CUDA Toolkit;
- skimage and Clara capability probes pass on both initial validation devices;
- the required whole-slide corpus passes Windows/Linux parity, concurrency,
  malformed-input, large-file, cleanup, and memory tests;
- GDS is reported as unavailable without disabling ordinary Clara I/O;
- installation, diagnosis, repair, uninstall, checksums, licenses, and support
  boundaries are published;
- upstream PRs have been opened for every generally useful portability change;
- VIPP still treats cuCIM as optional and enables only separately admitted
  operations;
- rollback consists of removing the downstream extra/artifacts without changing
  the base CPU application or saved-workflow compatibility.
