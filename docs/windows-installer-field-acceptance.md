# Windows Installer Field Acceptance

Last reviewed: 2026-08-27. Current Windows release: `0.14.0a2`; the focused
capacity/activity checklist introduced for `0.14.0a1` remains applicable.

Use this page for a tagged VIPP installer on a fresh Windows account. It is a
short evidence record, not an installation guide. The ordinary installation
steps remain in the [quick start](quick-start.md), and CUDA details remain in
the [GPU guide](gpu-guide.md).

This form covers only the Windows `.exe`. `0.14.0a2` also publishes separate
native, offline, CPU-only macOS PKGs for Apple Silicon (`arm64`) and Intel
(`x86_64`). Those explicitly unsigned and unnotarized packages have their own
architecture-qualified checksums and are documented in the
[macOS packaging guide](../packaging/macos/README.md); do not record them as
Windows acceptance evidence here.

Do not publish names, usernames, local paths, support reports, or screenshots
without reviewing them for private information.

## Record the exact release

- Release tag:
- Public release URL:
- Installer filename:
- Expected SHA-256:
- SHA-256 shown by PowerShell `Get-FileHash <downloaded-installer>`:
- Windows version:
- Account type: standard / administrator
- Account path includes spaces: yes / no
- Account path includes a non-ASCII character: yes / no
- Tester or evidence-record identifier:
- Start time:
- Finish time:

If this form is returned publicly, remove the tester's name, account name,
local paths, computer name, and the contents of the private Compute Doctor
support report first.

Stop if the filename, tag, or checksum does not match the official GitHub
release. An alpha installer may show **Unknown publisher**, but it must also say
`UNSIGNED` in its filename and the warning must match the release instructions.

Introduced in `0.13.0a8` and retained through `0.14.0a2`, Windows supplies
canonical Local App Data through
`SHGetKnownFolderPath(FOLDERID_LocalAppData)`. Managed setup accepts only
`VIPP\environments\cpu` or `VIPP\environments\cuda13` beneath it; custom
managed roots are rejected. The fixed CPU location supports spaces and
non-ASCII characters. CUDA supports spaces but requires the complete canonical
path to be ASCII because of the pinned CuPy 14.1.1 NVRTC path limitation. If
canonical Local App Data is non-ASCII, setup must make one-click CUDA
unavailable before environment creation or download and offer CPU.

## CPU route

Complete this on an account without an existing VIPP installation.

- [ ] The installer opened from the downloaded `.exe`.
- [ ] The recommended CPU choice was understandable without Python knowledge.
- [ ] Installation completed and created the expected VIPP shortcuts.
- [ ] **VIPP (Automatic)** opened successfully.
- [ ] Compute Doctor clearly said whether CUDA was unavailable or not selected.
- [ ] A bundled example opened, calculated, and produced the expected visible result.
- [ ] A workflow was saved, closed, reopened, and calculated again.
- [ ] A small batch completed and its output folder was understandable.
- [ ] Repair/update preserved the working setup.
- [ ] Uninstall removed VIPP-owned shortcuts and files without removing unrelated data.

Result: pass / fail

What was confusing, if anything:

## Qualified NVIDIA CUDA route

Complete this on a machine that satisfies the published CUDA requirements.

- GPU model:
- Driver version:
- Compute Doctor support-report filename retained privately:

- [ ] The installer offered CUDA only after checking the machine.
- [ ] The managed CUDA location exactly matched canonical Local App Data plus
      `VIPP\environments\cuda13`; spaces, if present, did not prevent setup.
- [ ] Supplying any other managed root was rejected before resolution or
      download; setup did not expose a custom-location chooser.
- [ ] If canonical Local App Data contained a non-ASCII character, one-click
      CUDA was unavailable and setup offered the fixed CPU route instead.
- [ ] Reviewing an expert-selected existing CUDA environment remained a
      separate non-mutating operation and never suggested moving, editing, or
      converting that environment into a managed installation.
- [ ] Installation completed and the Automatic, CPU, and Prefer-GPU shortcuts opened.
- [ ] Compute Doctor separately reported **CUDA and GPU** and **VIPP GPU coverage**.
- [ ] A reviewed GPU operation actually ran on the GPU; its execution report records
      the GPU implementation rather than merely detecting CUDA.
- [ ] A deliberately unsupported or CPU-selected step visibly explained its CPU
      fallback and still produced the correct result.
- [ ] The same workflow completed through Automatic and CPU-only launch profiles.
- [ ] Repair/update preserved the working setup.
- [ ] For an installer-owned CUDA copy already under a non-ASCII root, setup
      separately reported any recovery from a prior interrupted transaction;
      after that recovery, the newly blocked selection performed no new
      mutation. Setup clearly opened Installed apps for ownership-bound removal
      and did not offer a second or custom managed CUDA copy or describe an
      in-place/fallback migration.
- [ ] Uninstall removed VIPP-owned shortcuts and files without removing unrelated data.

Result: pass / fail

What was confusing, if anything:

## 0.13.0a8 changed-feature smoke

Complete these checks in a qualified CUDA installation. Repeat the portable
example on CPU-only if that route is part of this acceptance run; a visible CPU
explanation is a pass where the exact GPU region is unavailable.

- [ ] Compute Doctor reports **CUDA and GPU** and **VIPP GPU coverage** without
      offering an optional cuCIM installation or requiring a separate provider
      bundle.
- [ ] Calculate the basic **Measure Objects** and **Measure Objects + Intensity**
      schemas through Prefer GPU. Their execution reports name the reviewed
      CuPy implementations, match the CPU tables, and leave no private GPU
      memory after cleanup.
- [ ] Add **Remove Outliers (Binary)** after a small Boolean segmentation mask.
      Check both foreground removal and background filling, compare CPU and GPU
      outputs, and confirm that noncanonical grayscale `uint8` input is rejected
      rather than silently thresholded.
- [ ] Load a reviewed QYX TIFF, choose `QYX -> ZYX` in **Image Source**, and
      confirm that **Rescale Axes** exposes Z controls. Save, reopen, and run the
      workflow; the declaration and resized Z metadata must remain intact.
- [ ] Drop a disconnected compatible node onto a green-highlighted wire and
      confirm that it is inserted between the exact endpoints as one Undo action.
- [ ] Inspect several **Intensity & Contrast** nodes and confirm that each shows
      the exact input and output histograms. Integer Clip bounds must use
      whole-number controls, while Sigma Filter keeps a practical slider and a
      wider direct-entry range.
- [ ] Open **Portable GPU Segmentation Bridge**, calculate the graph, and
      confirm that the 19-voxel speck is removed, 31 enclosed cavity voxels are
      restored, and the final component volumes are 685, 599, 595, and 561.
- [ ] Inspect the execution report rather than relying on badges. Each eligible
      threshold and boolean-cleanup step records its actual implementation;
      any fallback names the unsupported dtype, parameter, or environment.
- [ ] Temporarily feed a `uint16` image directly into a reviewed float32 GPU
      operation. Confirm that its **GPU tip** explains the lossless conversion,
      **Add conversion** inserts a visible **Convert Dtype** node on the correct
      input, and one Undo restores the original graph.
- [ ] In **Prefer GPU**, calculate the affected node and confirm that an
      applicable GPU tip remains available afterward instead of disappearing
      because a CPU result was cached.
- [ ] Run **Find fastest pipeline** in a Custom workflow. Confirm that results
      are grouped by node and implementation, and that completed measurements
      remain inspectable even if VIPP says neither assignment is decisively
      faster and leaves the saved backends unchanged. Apply a clean measured
      assignment and confirm that the fixed CPU Image Source row does not make
      the proposal stale.
- [ ] Open the bundled 3D RL/RL-TV example and confirm that both branches retain
      25 iterations and `filter_epsilon=1e-12`; GPU eligibility must not rewrite
      either authored value.

Result: pass / fail

What was confusing, if anything:

## 0.14.0a1-introduced installer capacity and activity smoke

Run this focused section for installer issue
[#42](https://github.com/rensutheart/napari-vipp/issues/42). It qualifies the
changed presentation; the unchanged transactional install/update/repair and
rollback lifecycle can retain its existing release evidence.

- [ ] Before confirmation, CPU review shows approximately 250 MiB download,
      1.5 GiB installed, and 2.5 GiB peak working space.
- [ ] CUDA review shows approximately 1.5 GiB download, 5 GiB installed, and
      7 GiB peak working space.
- [ ] The approximate values are visually separate from the enforced disk
      minimums: CPU 5 GiB on the installation drive and 1 GiB on temp/records
      drives; CUDA 15 GiB and 5 GiB respectively.
- [ ] CUDA-ready wording says **at least 15 GiB of free disk space on the
      installation drive** and does not compare any disk figure with VRAM.
- [ ] Setup names each active phase and keeps elapsed time visible. Work without
      an observable total uses an indeterminate activity bar; byte progress is
      determinate only when the underlying tool provides trustworthy totals.
- [ ] During a deliberately quiet operation, a heartbeat eventually confirms
      activity without replacing the latest concrete activity message or
      declaring success, failure, or a stall.
- [ ] Advanced details shows the exact setup-log path and **Open setup log**
      opens that file once it exists.
- [ ] An actual reported network stall and an actual failure remain visibly
      distinct from the quiet-operation heartbeat, with an actionable terminal
      outcome and retained log location.

Result: pass / fail

What was confusing, if anything:

## Cancellation and network rollback

Use a machine or account where a previous VIPP installation is known to work.
Do not simulate power loss on a machine containing irreplaceable work.

- [ ] Cancelling setup reached a clear terminal outcome.
- [ ] The previous VIPP installation still opened and calculated its acceptance example.
- [ ] No half-created shortcut or partially selected environment was presented as ready.
- [ ] Repeating setup with the network unavailable failed with a useful action.
- [ ] The previous VIPP installation again remained usable after that failure.
- [ ] Retrying with the network restored completed successfully.

Result: pass / fail

Retained log or run-record identifiers:

## Novice understanding check

Ask these after the tester finishes; do not coach the answer.

- Could the tester explain which shortcut they would normally use?
- Could the tester tell whether their last calculation used CPU or GPU?
- Could the tester find the one next action when Compute Doctor reported a problem?
- Could the tester identify the input, output, axes, and pixel-size information in the
  example?
- Could the tester replace the example input with one owned image, save the workflow,
  and run a batch without terminal help?

Record each answer as **yes**, **partly**, or **no**, followed by one short note.

## Acceptance decision

- CPU fresh-account route: pass / fail / not run
- CUDA fresh-account route: pass / fail / not run
- Spaces path coverage: pass / fail / not run
- CPU non-ASCII path coverage: pass / fail / not run
- CUDA non-ASCII preflight guidance: pass / fail / not run
- Cancellation rollback: pass / fail / not run
- Network-failure rollback: pass / fail / not run
- Novice path: pass / fail / not run
- Release gate decision: accepted / still pending
- Exact remaining problem and owner:

A check that was not run stays **not run**. Automated tests, an older development
installer, WSL, or a different release artifact must not be recorded as a pass for
this tagged Windows installer.
