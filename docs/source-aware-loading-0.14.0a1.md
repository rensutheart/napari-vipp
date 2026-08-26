# VIPP `0.14.0a1` Source-Aware Loading Implementation Record

Status: `0.14.0a1` implementation and release-scope record, based on public
corpus revision `0.14.0a1-v3`. Qualification and publication evidence are
recorded separately; this page does not assert the outcome of an external gate.

## Release Decision

`0.14.0a1` makes Image Source a trustworthy scientific boundary. A source is
not merely a path plus a numeric series index: it is a selected scientific item
inside a versioned container, interpreted by a known reader, with pixels, axes,
metadata, capabilities, and provenance that agree.

The minimum coherent release floor is:

1. a truthful inspection/read contract for the claimed source formats;
2. durable `SourceItem v1` identity and source-revision checking;
3. propagation of that item through interactive, batch, generated, replay,
   export, checkpoint, and provenance surfaces; and
4. an Image Source UI that exposes the selected item, its metadata, its reader,
   and any limitation or refusal before scientific execution.

Pyramid-aware presentation preview and typed per-sample scalar parameters are
the two feature packages built on that floor. Both are included in a1 with
their own acceptance scope. Any future unfinished package moves explicitly to
the next alpha; it must not weaken or hold an otherwise coherent SourceItem
release.
Installer capacity/activity issue
[#42](https://github.com/rensutheart/napari-vipp/issues/42) is integrated in a1
and receives its own focused installer-presentation gate.

## Evidence Baseline

The acceptance corpus contains 18 datasets, 20 downloaded artifacts, and
210.41 MiB of source data. It covers OME-Zarr 0.4/0.5, OME-TIFF, ImageJ TIFF,
NumPy containers, ordinary raster data, Nikon ND2, Leica LIF, Zeiss CZI/LSM,
Olympus OIR/OIB/VSI, and Imaris IMS. It includes single and multiple items,
images and labels, pyramids, companion files, Bio-Formats, RGB associated
images, signed and unsigned types, and explicit non-Z semantic axes.

The corpus is evidence for source behavior, not for segmentation quality and
not for every possible dtype. Tiny deterministic fixtures remain the correct
place for exhaustive empty, singleton, integer-boundary, NaN, Inf, `float64`,
and malformed-source contracts. Ordinary CI remains network-free. Original
public bytes are used only from a verified opt-in cache.

The release record combines network-free contract tests with a strict opt-in profile
for the verified public cache. It checks reader format, stable item identity,
shape, dtype, axes, bounded metadata parity, representative decoded pixels,
capabilities, source revision, persistence/execution propagation, preview
behavior, and per-sample overrides. Olympus OIB now passes the predeclared CZYX
contract. The strict profile still requires the qualified optional-reader and
Java environment and is a release gate rather than routine network CI.

## Canonical Source Contract

The implementation separates three records that are currently mixed together:

- **Source container:** local URI, normalized format, all files or objects that
  form the container, and an exact content-revision proof. A `.vsi` plus its
  `.ets` tree and an `.oif` plus its companion directory are one container.
- **Logical selector:** the authored, reader-neutral choice of item key and
  kind, plus any reviewed axis declaration. A numeric index is only a legacy
  hint and never the durable identity.
- **Resolved SourceItem:** the logical selector bound to the observed container
  revision, selected adapter and version, normalized shape/dtype/axes/kind,
  metadata, available levels, and declared reader capabilities.

Changing source bytes invalidates the resolved proof without silently changing
the logical selector. Changing reader implementation or version must not
silently change item topology or array meaning. If the same logical item cannot
be resolved unambiguously through the recorded adapter, VIPP stops and offers
an explicit rebind rather than choosing another scene or backend.

Every adapter must satisfy these invariants before an item can enter the graph:

- inspection does not materialize the complete pixel array;
- the returned pixel rank and shape exactly match the normalized axes;
- dtype and logical shape describe the returned pixels, not padded storage
  chunks;
- any permutation is applied atomically to both pixels and metadata;
- raw reader axes and normalized VIPP axes remain distinguishable, inferred
  axes are never marked explicit, multi-character axes remain single tokens,
  and axis names/cardinality are validated;
- item kind distinguishes ordinary images, labels, and associated RGB images;
- metadata-only inspection and full read agree for the bounded a1 metadata
  subset;
- the selected item key agrees across inspection and read;
- per-scene metadata belongs to that scene rather than to whichever scene the
  reader inspected last, and a failed scene selection is never ignored;
- backend, version, capabilities, and any normalization are recorded; and
- a contract mismatch fails before graph publication with an actionable error.

## Bounded Metadata Contract For a1

VIPP does not need to model every private vendor field in this alpha. It does
need to preserve the scientific facts already declared by the corpus when the
source and supported reader expose them:

- ordered axis name/type, size, unit, scale, translation, source position, and
  confidence;
- time spacing and spatial calibration, without losing anisotropy;
- channel order, name, colour, fluorophore, and excitation/emission wavelength
  where present;
- objective name/model, magnification, numerical aperture, immersion, and
  refractive index where present;
- acquisition date, instrument, and detector where reliably available;
- item kind, series/scene key and name, format, reader and reader version; and
- pyramid level shapes/transforms and truthful lazy, preview, region-read, and
  chunk capabilities.

Metadata normalization must distinguish at least these states: value present,
not present in the source, not exposed by the chosen reader, and exposed but not
yet mapped by VIPP. The UI must not display a missing mapping as a factual
default such as unitless scale 1 or "no objective." Raw vendor metadata may be
retained as diagnostic evidence, but it is not the canonical scientific
contract and need not round-trip exhaustively in a1.

Known semantic axes remain semantic. `T`, `C`, `Z`, `Y`, and `X` retain their
meaning; FLIM lifetime-bin `H` must not become Z; unknown `Q` remains unknown
until the user explicitly maps it; and filenames never determine axes. Packed
sample axis `S` is normalized explicitly to RGB/RGBA without being confused
with an acquisition channel axis. The same reviewed axis-declaration component
is used by interactive Image Source and batch configuration and is persisted in
the SourceItem evidence.

## Format Readiness Matrix

| Source family | Candidate support | Remaining limitation |
| --- | --- | --- |
| OME-Zarr 0.4/0.5 | Stable image/label group identity, declared levels/transforms, lower-level display-only preview, requested-axis/region slicing, label semantics, cancellation, observable I/O, and unchanged level-0 analysis | Local stores only; the scientific graph still materializes level 0 |
| OME-TIFF / TIFF / NPZ | Stable item keys, deterministic legacy migration, guarded axes, and NPZ header inspection without decoding every member | Single-level TIFF truthfully has no lower preview; arbitrary semantic-axis iteration is deferred |
| Nikon ND2 | Pixel-lazy inspection/data, decoded-size estimate, calibration, channels, objective/acquisition metadata, and inspection/read parity | Requires the optional `nd2` reader |
| Leica LIF | Calibration, channels, objective metadata, and inspection/read parity | Native `liffile` access is eager; PR2729 is one combined TMZCYX item in `liffile` versus four Bio-Formats items, so VIPP pins/refuses backend topology changes rather than inferring equivalence |
| Zeiss CZI | Stable same-shape scene keys, distinct scene pixels, calibration, channels, objective metadata, and inspection/read parity | Native `czifile` pixel access is eager; pyramid support is not claimed |
| Olympus OIR | Authoritative TCYX, 10-second T spacing, XY calibration, channels/objective metadata, and inspection/read parity | Native `oirfile` pixel access is eager |
| Olympus OIB | Authoritative CZYX pixels and ImageState, anisotropic calibration, channel wavelengths, and inspection/read parity | Native `oiffile` pixel access is eager |
| Olympus VSI/ETS | All companions bind into source revision; primary and RGB macro items, calibration/channels/objective, decoded-size reporting, and actionable Java/Bio-Formats errors | Optional Java/codecs are required and the roughly 116 MiB decode remains cache-gated |
| Imaris IMS | Logical TCZYX shape instead of padded storage shape, metadata parity, decoded-size reporting, and actionable Java/Bio-Formats errors | No pyramid enumeration or cheap lower-level-read claim yet |
| Zeiss LSM | Stable main/thumbnail identity, CYX versus packed RGB semantics, calibration/channels/objective metadata, and inspection/read parity | Native `tifffile` LSM pixel access is eager |
| FLIM and RGB controls | Semantic `H` and packed sample `S` remain non-Z/non-acquisition-C; unknown axes require reviewed mappings | General semantic-axis batch iteration is deferred |

Native OIF companion support is implemented only after the shared multifile
container contract exists. A deterministic OIF fixture may then be derived from
the licensed OIB source. LOF/XLIF remain capability-advertised but unclaimed as
validated until a licensed frozen source is added.

## Delivered Work Packages And Qualification Order

### 0. Corpus contract (delivered)

- Record the final corpus-v3 manifest hash and keep previous revisions
  immutable.
- Build the deterministic mixed-member NPZ and reorderable two-series OME-TIFF
  fixtures with pinned derivation hashes.
- Add small malformed, missing-companion, source-drift, axis-mismatch,
  empty/singleton, numeric-boundary, NaN, and Inf fixtures.
- Add reader-double cases for failed scene selection, per-scene metadata, packed
  sample axes, multi-character/unknown axes, and an NPZ member that must not be
  decompressed during inspection.
- Expand cache-gated tests to assert canonical metadata with unit-normalized
  tolerances, decoded hashes or stable plane samples, item kind, scene
  distinction, and all known refusal paths.
- Provide a strict acceptance profile in which required optional readers and
  Java are failures. Ordinary CI may skip the entire opt-in profile when its
  verified cache is absent; it must not silently skip individual claimed
  formats inside a configured profile.

Delivered result: the network-free contract remains deterministic, and the
strict cache profile fails on any regression of OIB or another claimed reader,
an unavailable required optional runtime, or a declared refusal path.

### 1. Reader-adapter contract and shared metadata extraction (delivered)

- Replace or extend `ImageSeriesInfo` so inspection produces the complete
  normalized per-item record rather than only shape/dtype/axis strings.
- Introduce a small backend registry with explicit `probe`, `inspect`, and
  `read` stages plus typed source errors. Preserve error code, stage, format,
  backend, item, and remediation through background workers instead of
  converting failures to undifferentiated strings.
- Give every adapter explicit capabilities: pixel-lazy inspection, lazy data,
  levels, preview read, exact region read, companions, and estimated decoded
  bytes.
- Use the same per-format extraction and normalization functions during
  inspection and read; do not parse a poorer metadata path for inspection.
- Store normalized metadata per scene. A reader that cannot select the requested
  scene must stop rather than returning metadata or pixels for its previous
  scene.
- Replace character-by-character fallback parsing with one token-aware axis
  normalizer. Preserve raw axes, mark guessed axes as inferred, distinguish
  acquisition C from packed RGB/S, and validate unique axes and channel
  cardinality.
- Validate shape/dtype/axes/kind/item key at the adapter boundary.
- Select readers deterministically and record adapter/version. Fall back only
  for a declared availability failure, never for corrupt data or a semantic
  mismatch.

Exit: synthetic adapter-contract tests prove inspection/read parity and refuse
silent fallback, transposition, and padded-shape publication.

### 2. Real-reader correctness (delivered; strict cache gate pending)

Split this work by independent reader families so review remains tractable:

1. ND2, LIF, CZI, and LSM metadata plus stable scene/item behavior;
2. Olympus OIR and OIB metadata plus the OIB C/Z correction; and
3. Bio-Formats-backed VSI and IMS metadata, multifile discovery, codec/runtime
   preflight, and decoded-size reporting.

For multifile sources, reader discovery returns a deterministic relative member
inventory. Exact source revision hashes every required member; missing,
duplicate, inaccessible, or changed companions have distinct errors. Reader
handles are closed on success, failure, cancellation, and source replacement.

Delivered result: focused reader contracts pass their declared structural,
pixel, axes, metadata, and error checks. OIB has no expected failure.
Unsupported capabilities remain explicit rather than simulated; the strict
real-cache profile remains an integrated release gate.

### 3. `SourceItem v1` identity and migration (delivered)

- Define frozen canonical selector, resolved item, container-member, revision,
  reader, capability, and metadata-evidence records.
- Resolve saved selection by stable key; use a legacy index only when it maps to
  one unchanged item.
- Keep exact content revision separate from logical selection so a changed file
  is reported as changed rather than as a different authored item.
- Bind all participating files/objects, the adapter/version, axis declaration,
  and analysis level into resolved evidence.
- Keep public identities privacy-safe and exclude private absolute paths.
- Bump workflow schema to 5, batch config to 4, and manifest/item records to 4;
  continue reading workflow v3/v4 and batch v1-v3 goldens, reject unknown future
  versions, and write only the canonical form.

Exit: selection survives order reversal, same-shaped scenes remain distinct, a
one-byte/object mutation invalidates stale evidence, and legacy documents
migrate and re-save deterministically without invented meaning.

### 4. SourceItem propagation through execution surfaces (delivered)

- Replace path-plus-series assumptions in interactive Image Source,
  `SourcePayload`, caches, batch planning/naming, output collision handling,
  manifests, checkpoints, generated runners, CLI execution, replay, export, and
  provenance.
- Make cache keys include source revision, logical item, adapter/version, axis
  declaration, and analysis level.
- Use one resolver everywhere; no surface may independently reinterpret an
  index, filename, or display name.
- Preserve scientific hashes where migrated meaning is unchanged.

Exit: all routes serialize the same canonical SourceItem and produce the same
effective scientific identity. A mismatch stops before any output write.

This is the minimum `0.14.0a1` release floor.

### 5. Image Source user journey (delivered)

- Separate container selection from item selection.
- List every item with kind, axes, shape, dtype, reader, level availability, and
  the bounded metadata subset.
- Distinguish unavailable, source-absent, reader-unexposed, and VIPP-unmapped
  metadata.
- Use the shared interactive/batch axis declaration control and show its
  provenance.
- Show source verification, inspection, decode/materialization, and
  cancellation phases. Where the reader can estimate decoded size, show it
  before loading and warn before unreasonable host-memory demand.
- Carry typed errors and progress/cancellation callbacks through the worker
  boundary; inspection failures must not be silently suppressed by the UI.
- Avoid a second complete in-memory copy where the adapter can prove safe VIPP
  ownership. Otherwise retain bounded chunked copying and report its memory
  cost honestly.
- Preserve the last valid item while a replacement loads; stale workers cannot
  publish. Missing readers, Java, codecs, companions, items, or source revisions
  produce specific remediation.

Exit: save/reopen and refresh behave predictably, no thumbnail/label/scene is
substituted silently, and large/eager loads do not block the GUI without status.

### 6. OME-Zarr presentation preview (delivered)

- Scope the first preview adapter to local OME-Zarr 0.4/0.5.
- Choose a declared lower level and read only required T/Z/C selections and Y/X
  chunks for display.
- Label it `Preview level N - analysis remains full resolution`.
- Keep it outside graph inputs, scientific cache, complete-image statistics,
  output data, and scientific provenance.
- Generation-own and cancel preview, exact verification, and level-0 work.
  A provisional display may appear while hashing continues, but scientific
  publication waits for exact source verification.
- Measure objects/bytes read, peak RAM, time to first preview, cancellation,
  cache bounds, handle release, and stale-result rejection.

Exit: the preview demonstrably reads less than level 0, transforms and label
semantics are correct, and final analysis is byte-equivalent to the unchanged
level-0 path. Unsupported readers make no pyramid or region-read claim.

### 7. Typed per-sample scalar parameters (delivered)

- Key rows by source-node ID plus the logical SourceItem selector; bind observed
  revision separately.
- Begin with an explicit whitelist of numeric authored scientific scalars.
  Blank inherits the saved workflow value. Exclude source/output paths,
  selectors, topology, expressions, code, device/cache controls, and derived
  fields.
- Validate values through each node's normal scientific contract before any
  item runs.
- Reject duplicate, stale, zero-match, and multi-match rows; a new item without
  a row visibly inherits the default.
- Use one detached resolver for representative preview, full batch, saved
  runner, and CLI without mutating the base workflow.
- Record authored/effective values, source revisions, config hash, and effective
  workflow digest in plans, manifests, checkpoints, and provenance.

Exit: the frozen BBBC016 pair uses its two predeclared thresholds and produces
distinct expected outputs with preview/batch/runner/CLI parity and an unchanged
base workflow.

### 8. Integrated release qualification

- Run one end-to-end case combining reordered item discovery, save/reopen,
  lower-resolution presentation with level-0 analysis, two item-specific
  thresholds, generated/CLI execution, and provenance.
- Publish a truthful support matrix distinguishing validated, optional,
  capability-only, and deferred formats/features.
- Document selection/revision/backend identity, metadata availability states,
  axis declarations, preview semantics, migrations, optional-reader setup, and
  every explicit non-goal.
- Run exact-main ordinary CI once, the strict cached public-source profile,
  focused UI acceptance, schema goldens, strict docs, and only the changed
  release qualification domains.

Expected changed domains are core/UI, workflow schema, Windows installer, and
documentation. Add focused GPU scientific parity for shared axis/metadata
changes, focused installer capacity/activity acceptance for #42, dependency
toolchain only when reader declarations change, and the normal
packaging/release checks for the final tag. The unchanged installer
transactional lifecycle and full GPU catalogue are carried forward rather than
repeating those complete matrices.

## Release Blockers

The alpha does not ship with any of these conditions:

- OIB pixels and axes regress from the authoritative passing CZYX contract;
- a claimed corpus format lacks inspection/read parity for declared metadata;
- returned rank, shape, dtype, kind, or item key disagrees with inspection;
- inferred axes are reported as explicit, a packed sample axis is treated as an
  acquisition channel, or per-scene metadata comes from a different scene;
- Java/Bio-Formats, codec, optional-reader, or companion failures are not
  actionable;
- SourceItem differs across interactive, batch, generated, replay, export, or
  provenance paths;
- legacy migration can retarget an item or invent axes/calibration;
- a preview can enter scientific execution or survive after its source is
  superseded;
- per-sample effective parameters differ between execution routes; or
- the strict configured public-data profile contains an unexplained skip.

## Explicit Deferrals

The following do not block `0.14.0a1`:

- exhaustive vendor-private metadata and opaque metadata round-trip;
- automatic dependency installation or silent backend switching;
- making eager LIF/CZI readers lazy;
- operation-level lazy/chunked execution and progressive scientific outputs;
- remote stores, plate/well/field traversal, and semantic-axis batch iteration;
- IMS pyramid support before controlled evidence proves level enumeration,
  transforms, and cheap reads;
- Responsive Volume Crop and exact source-window Crop pushdown;
- arbitrary per-item expressions, code, topology changes, bypass, or CSV
  override import/export;
- CZI line scans without Y, misleading-filename OIR, derived channel-last data,
  and the larger extended/stress corpus;
- broad whole-slide, clinical, facility-specific, LOF, or XLIF claims without a
  licensed frozen acceptance source; and
- scientific validation of downstream algorithms, which requires separate
  method-specific evidence.

## Implementation Map

The main seams are `core/io/model.py`, `core/io/registry.py`, new typed-error and
backend-registry seams under `core/io/`, the format adapters under `core/io/`,
`core/metadata.py`, `core/source_identity.py`,
`core/file_sources.py`, `ui/file_sources.py`, `ui/source_adapter.py`, Image
Source controls in `_widget.py`, workflow/batch schema and execution modules,
and execution provenance/export. Tests belong beside each seam; the opt-in
real-source contract remains in `test_public_image_io.py` with network-free
manifest and fixture tests in ordinary CI.

Implementation followed the dependency order above: executable corpus
contracts first, then adapter/reader correctness, SourceItem identity and
propagation, source UI, OME-Zarr presentation preview, and typed overrides.
Release qualification must preserve those expectations rather than weakening a
failing corpus or migration case.
