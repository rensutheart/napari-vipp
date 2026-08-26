# VIPP 0.14.0a1

VIPP 0.14.0a1 makes scientific image selection explicit and durable. The same
selected item, reader evidence, source revision, axes, metadata, and effective
per-sample parameters now travel through interactive calculation, batch,
generated execution, replay, export, checkpoints, manifests, and provenance.

This remains alpha software. Keep original data and workflows, test
representative sources through the intended reader, and review important
outputs before using them for scientific conclusions or publication.

## SourceItem identity and compatibility

- Added the frozen SourceItem v1 contract. A logical selector is distinct from
  the observed container revision and the resolved reader/backend evidence, so
  changed bytes or an unexpected item topology fail visibly instead of silently
  selecting something else.
- Workflow schema 5 and batch config/manifest schema 4 write the canonical
  SourceItem representation. Existing workflow v3/v4 and batch v1-v3 records
  retain deterministic compatibility and migration paths.
- Stable item keys replace order-dependent series selection where the reader can
  provide them. Public identities remain privacy-safe; exact local paths are not
  added to portable provenance.
- Multifile `.vsi`/ETS and `.oif` companion trees are treated as one source
  container. Missing, empty, or changed companions invalidate the revision.

## Truthful microscope-reader contracts

- Reader inspection and full read now share normalized metadata for the claimed
  ND2, LIF, CZI, OIR/OIB, VSI/IMS, and LSM routes in the frozen corpus.
- Olympus OIB inspection, returned pixels, and ImageState now agree on the
  authoritative CZYX shape. Its Z/Y/X calibration and channel wavelengths are
  retained.
- Optional-reader and Bio-Formats/Java readiness failures preserve a structured
  stage, backend, and remediation instead of becoming a generic import error.
- Leica LIF channel identity now preserves the reader's semantic Alexa dye
  names rather than substituting display LUT colours. Combined objective labels
  such as `63x, 1.3NA` are normalized into magnification and numerical aperture.
- Readers declare whether inspection/data access is lazy, whether region or
  level reads are supported, and an estimated decoded size where available.
  Native LIF, CZI, OIR, OIB, and LSM pixel access remains truthfully eager.

The PR2729 Leica source deliberately has different reader presentations:
`liffile` exposes one combined TMZCYX item while Bio-Formats exposes four
logical items. VIPP pins the selected backend and version and refuses an
unreviewed topology change; it does not claim that those views are equivalent.

## Presentation preview without changing analysis

- Local OME-Zarr 0.4 and 0.5 image and label groups expose declared levels and
  coordinate transforms.
- VIPP can choose a lower declared level for display and slices requested T/Z/C
  positions plus the Y/X region before computing. Observable object and byte
  reads are reported where the store permits it.
- Every lower-level result is labelled
  `Preview level N - analysis remains full resolution`. The canonical
  SourceItem and scientific graph input stay at level 0.
- Cooperative cancellation and generation checks prevent a superseded preview
  from publishing. Label previews preserve label semantics and do not use
  intensity-image presentation rules.
- A single-level source truthfully reports that no lower-level preview exists.

## Reviewed per-sample numeric parameters

- Batch rows can override explicitly selected numeric scientific parameters for
  one stable SourceItem while a blank cell visibly inherits the saved workflow
  value.
- Values pass through the normal parameter contract. Duplicate, stale,
  zero-match, multi-match, and invalid typed rows stop preflight before output
  publication.
- The base workflow is never mutated. Effective values and workflow hashes are
  shared by representative preview, batch, generated runner/CLI, checkpoints,
  manifests, and item provenance.

## Clearer retained Batch workspace

- Reopening a workflow with an attached Batch workspace performs metadata-only
  sample detection in the background and restores exact per-sample overrides
  without requiring a manual Preview click. Changed or missing sources remain
  quarantined for review.
- A compact status and activity indicator stays beside the Batch toolbar while
  the lower run section retains detailed per-item and per-operation progress.
- `Ask before overwrite (recommended)` lists the exact existing outputs and
  requires one-run consent before replacing them. Cancel is the safe default
  and leaves files untouched. Duplicate destinations, source overlaps, and
  explicitly protected outputs remain hard errors; headless execution retains
  the fail-closed `error` policy.

## Understandable Windows setup activity

- The reviewed setup summary now separates rounded capacity estimates from the
  conservative disk-space gates that are actually enforced. CPU setup shows
  approximately 250 MiB to download, 1.5 GiB installed, and 2.5 GiB peak
  temporary working space. CUDA setup shows approximately 1.5 GiB to download,
  5 GiB installed, and 7 GiB peak temporary working space.
- Those estimates do not replace the enforced minimums: managed CPU setup needs
  5 GiB free on the installation drive and 1 GiB on every drive used for
  Windows temporary files or VIPP installer records; managed CUDA setup needs
  15 GiB and 5 GiB respectively. All of these values describe disk storage.
  Setup does not compare them with GPU memory (VRAM).
- Setup identifies its current phase and elapsed time. It uses an indeterminate
  activity bar for work whose progress cannot be observed and reports byte
  progress only when a dependency tool supplies trustworthy totals.
- A quiet-operation heartbeat confirms that setup is still working without
  hiding the latest concrete activity. Advanced details retain the setup-log
  path and offer a direct way to open the log; a slow operation is kept
  distinct from an actual reported stall or failure.

## Large-source loading

- Local source reads provide decoded-memory preflight, truthful progress,
  cooperative cancellation, and stale-generation protection.
- Progress reports only work the reader can observe; a monolithic eager reader
  is never given an invented internal percentage.

## Remaining limitations

- Scientific graph execution still materializes the complete selected level-0
  image. The lower OME-Zarr level is display-only; arbitrary operation-level
  lazy or chunked graph execution is not included.
- Presentation preview is limited to local OME-Zarr 0.4/0.5. Remote stores,
  HCS plate/well/field traversal, and IMS pyramid support are not claimed.
- Native LIF, CZI, OIR, OIB, and LSM pixel access is eager. Large eager readers
  can warn/refuse before decode, but cannot provide chunk-level progress.
- Bio-Formats-backed VSI/IMS requires the optional reader stack, Java, and the
  required codecs. VIPP reports missing prerequisites but does not bundle them
  into the base Python package.
- Per-sample overrides are numeric scalar parameters only. Expressions,
  filename rules, source selectors, topology changes, CSV import/export, and
  semantic-axis iteration remain deferred.

## Qualification scope

Focused core, reader-contract, schema/migration, preview, and override tests are
part of the release evidence. The complete qualification scope also includes
the integrated release suite, the strict verified-cache corpus profile in its
qualified optional-reader environment, exact-tag artifacts, and the
release-specific Windows installer.

The release-domain declaration for the complete change from `v0.13.0a9` is:

```yaml
tier: alpha
changed:
  core_ui: true
  workflow_schema_provenance: true
  gpu_scientific_shared_axes_metadata: true
  windows_installer: true
  dependencies_toolchain: false
  packaging_release_metadata: true
  documentation: true
carried_forward:
  full_gpu_catalogue: v0.13.0a8
  windows_installer_transactional_lifecycle: v0.13.0a8
  dependencies_toolchain: v0.13.0a8
```

Focused real-GPU parity is still required for the changed shared axis/metadata
boundary. The changed installer capacity/activity presentation receives
focused qualification, while only its unchanged transactional lifecycle is
carried forward from `v0.13.0a8`. The unchanged full GPU catalogue is also
carried forward; exact `0.14.0a1` package, installer, hash, and public URL facts
are regenerated for the immutable release tag.
