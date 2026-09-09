# VIPP 0.15.0a3

VIPP 0.15.0a3 adds a reviewed path from a recorded batch run to a portable reproducibility package, verified interrupted-run recovery, and clearer object QC and desktop workflow handling.

**Alpha software.** Preserve original data, workflows, environments and decisive outputs. Review the included checks and limitations before using this version for scientific analysis.

## Review, share and repeat a recorded analysis

- **Export reproducibility package…** prepares an offline report, portable workflow and batch settings, advanced Python runners, software records and redacted evidence. Review the exact included files and sharing omissions before saving a local ZIP. Packages exclude raw images, result files and previews; preparation and export do not upload data or install software.
- Recorded-run packages use the archived workflow and settings, not a later edited graph. Explanatory workflow notes are retained with folder hiding and optional filename anonymisation; review free text for sensitive information.
- Reopen the portable workflow in VIPP, choose **Reproduce original run** or **Use workflow on new data**, relink inputs and choose a new output folder. Reproduction checks original input contents and image selections before Run. A different or unknown VIPP version requires explicit acknowledgement.
- Clearer check details, version guidance and completed-run status keep input verification distinct from output validation. Correcting a mismatched input folder and checking again preserves the fresh verification at Run handoff. A completed-run badge does not authorize another run without a new check.

Matching input hashes and VIPP versions do not prove identical outputs, dependencies, hardware or scientific validity. The package is neither a complete data archive nor a locked environment, and its redacted evidence cannot resume the original run.

## Resume an interrupted batch

**Resume saved run…** and newly generated batch runners' `--resume MANIFEST` continue the saved workflow/settings while leaving the open graph unchanged. Only whole completed items with verified input/output bytes, settings, runtime and original receipt evidence are reused. The report distinguishes verified reuse from new writes, and ordinary/resumed runs share a destination lock.

Manifest schema **6** adds recovery snapshots and run lineage. Earlier manifests cannot supply verified resume evidence. Keep original archives and sidecars; moved runs, partial-item recovery and cross-version recovery are not supported.

## Object QC and workflow handling

- **Find Label Boundaries** creates a Boolean QC mask inside objects, outside objects or on both sides, with explicit 2D/3D and Face/Full connectivity. It preserves the grid and upstream labels and processes time/channel blocks independently on CPU. It does not retain object IDs or create meshes.
- **Filter result** shows current input, kept and removed object counts for supported label/mask filters using a cached background diagnostic. Stale, bypassed and uncached results do not display misleading counts.
- Drag a local workflow JSON onto the graph, tab strip or inspector to open it in a new tab without replacing existing work. Recorded workflows use the same reproduction choice as **Open**.
- **Separate Overlapping Objects** adds an annotated segmentation and two-mesh example. One **Mesh Objects, Colours & Refinement** example now retains the authored interactive workflow; the duplicate tuned entry is removed.
- Installed desktop launches gain VIPP window/shortcut branding and a title-bar-free startup window. Drag the logo or background to move it, or use the small top-right minimize button; it does not stay above other windows. Plugin/manual launches retain napari branding. Initial workflow-dock space and the Batch/Display toolbar divider are clearer.

## Broader RL GPU execution, with explicit uncertainty

Ordinary Richardson–Lucy and RL-TV admit more authored parameter combinations, including even/larger PSFs. Numerical-difference advisories appear in Compute and execution provenance version **2**. Permission to execute is not proof of CPU equivalence or restoration quality; benchmark comparison criteria remain separate. Finite float32 input, axes, PSF mass, memory, environment and cancellation requirements still apply. CPU fallback clears GPU-result advisories.

## Compatibility and qualification

Workflow and batch-configuration schema numbers remain **6**; the new manifest schema is **6**, compared with **5** in 0.15.0a2. New operations, reproduction references and generated runners require a compatible runtime even where a schema number is unchanged. Keep older evidence unchanged and regenerate Python exports for the exact installed version.

User-reported end-to-end acceptance is recorded separately from independent output comparison. The [qualification declaration](https://github.com/rensutheart/napari-vipp/blob/v0.15.0a3/docs/release-qualification-baseline.md) describes changed-domain checks and the scope of carried-forward evidence. Matching inputs, versions or a successful test suite do not guarantee scientific validity for a new dataset.

Use the [0.15.0a3 release page](https://github.com/rensutheart/napari-vipp/releases/tag/v0.15.0a3) for distributions, Windows setup and separate Apple Silicon/Intel macOS installers with their checksums, and the [0.15.0a3 manual](https://rensutheart.github.io/vipp-mkdocs/0.15.0a3/) for instructions. Desktop installers are explicitly unsigned alpha builds; macOS packages are also unnotarized. Verify downloads and follow the platform-specific setup guidance.
