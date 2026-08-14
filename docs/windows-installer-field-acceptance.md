# Windows Installer Field Acceptance

Last reviewed: 2026-08-14

Use this page for a tagged VIPP installer on a fresh Windows account. It is a
short evidence record, not an installation guide. The ordinary installation
steps remain in the [quick start](quick-start.md), and CUDA/cuCIM details remain
in the [GPU guide](gpu-guide.md).

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
- [ ] Installation completed and the Automatic, CPU, and Prefer-GPU shortcuts opened.
- [ ] Compute Doctor separately reported **CUDA and GPU**, **Optional cuCIM**, and
      **VIPP GPU coverage**.
- [ ] A reviewed GPU operation actually ran on the GPU; its execution report records
      the GPU implementation rather than merely detecting CUDA.
- [ ] A deliberately unsupported or CPU-selected step visibly explained its CPU
      fallback and still produced the correct result.
- [ ] The same workflow completed through Automatic and CPU-only launch profiles.
- [ ] Repair/update preserved the working setup.
- [ ] Uninstall removed VIPP-owned shortcuts and files without removing unrelated data.

If optional cuCIM is included in this acceptance run:

- [ ] The official no-wheel ZIP checksum matched before extraction.
- [ ] `Install VIPP cuCIM.cmd` built cuCIM locally for this exact VIPP environment.
- [ ] Compute Doctor changed only the optional cuCIM and admitted-coverage results
      that the add-on genuinely enabled.

Result: pass / fail

What was confusing, if anything:

## 0.13.0a7 guided-GPU feature smoke

Complete these checks in a qualified CUDA installation. Repeat the portable
example on CPU-only if that route is part of this acceptance run; a visible CPU
explanation is a pass where the exact GPU region is unavailable.

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
      faster and leaves the saved backends unchanged.
- [ ] Open the bundled 3D RL/RL-TV example and confirm that both branches retain
      25 iterations and `filter_epsilon=1e-12`; GPU eligibility must not rewrite
      either authored value.

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
- Spaces and non-ASCII path coverage: pass / fail / not run
- Cancellation rollback: pass / fail / not run
- Network-failure rollback: pass / fail / not run
- Novice path: pass / fail / not run
- Release gate decision: accepted / still pending
- Exact remaining problem and owner:

A check that was not run stays **not run**. Automated tests, an older development
installer, WSL, or a different release artifact must not be recorded as a pass for
this tagged Windows installer.
