# Public Data Corpus For `0.14.0a1`

Status: release acceptance baseline `0.14.0a1-v4`, reviewed on 2026-08-26.

The first 0.14 alpha needs real data diversity, but a larger pile of images is
not automatically a better test. This corpus is selected against the feature
contract before implementation so that datasets are not chosen because they
happen to pass VIPP. It combines public biological data, standards fixtures,
and tiny generated boundary cases.

The canonical machine record is
[`validation/public-data/corpus-v4.json`](validation/public-data/corpus-v4.json).
It names the immutable
[`v3` base](validation/public-data/corpus-v3.json) and its SHA-256; `v3` and `v2`
in turn preserve the earlier corpus revisions.
Public image bytes are cached outside Git and are never fetched by ordinary CI.

## Selection Rules

Every accepted source must satisfy all of these rules:

1. It closes a predeclared coverage gap that is not already covered more
   cheaply by another source.
2. Its landing page, scientific accession or version, licence, exact download
   URL, byte count, and SHA-256 are recorded.
3. A directory-backed store has a frozen inventory of every object path, byte
   count, and SHA-256. S3 ETags are advisory and never substitute for content
   hashes.
4. The expected axes, shape, dtype, item structure, scale, and pyramid levels
   are recorded before the corresponding implementation is accepted.
5. A derived fixture names every source hash and deterministic transformation.
   It is never described as an unmodified public image.
6. Upstream drift fails closed. Refreshing a source requires review and a new
   corpus version; it never silently changes a test baseline.
7. Prefer CC0, CC BY, or similarly permissive material. Licence or attribution
   uncertainty moves a dataset out of the corpus even when it is technically
   attractive.

This is an I/O and workflow acceptance corpus, not a representative sample of
all microscopy or biology. Scientific algorithm validation continues to use
method-specific phantoms and reference packs.

## Tier 0: Deterministic Contract Fixtures

Always-on CI should remain fast and network-free. Tiny arrays generated in the
test process cover `bool`, signed and unsigned integer widths, `float32`,
`float64`, empty and singleton dimensions, extrema, NaN/Inf rejection, and
numeric control boundaries.

Two deterministic derived containers are required before SourceItem merges:

- a mixed-member NPZ with `bool`, `uint8`, `uint16`, and `float32` members; and
- a two-series OME-TIFF whose inspection order can be deliberately reversed.

Their recipes must use fixed source members from the public manifest, canonical
member order, fixed ZIP timestamps where relevant, exact dtype/endian rules,
and frozen output hashes. Ordinary `numpy.savez` output bytes are not a stable
container golden because ZIP metadata can vary.

Generated fixtures provide exhaustive boundary control. Public images provide
external validity. Requiring one public dataset for every possible NumPy dtype
would add size and licence risk without improving the contract.

## Tier 1: Public Alpha Acceptance

The frozen public tier is 220,627,149 bytes (210.41 MiB) across twenty
downloaded artifacts. It runs from a verified cache in trusted manual or
release acceptance, not on every pull request. The `v3` increase is a focused
50.81 MiB format tier: each source reaches a distinct reader or container
contract rather than merely adding another filename extension.

| Source | Objective coverage | Frozen size |
| --- | --- | ---: |
| IDR0062 `6001240_labels.zarr`, OME-NGFF 0.5 / Zarr 3 | Three-level `uint16` CZYX image, `int8` label group, calibrated axes, image-versus-label SourceItems, lower-level preview versus level-0 analysis | 34,323,650 B |
| The same IDR0062 image in OME-NGFF 0.4 / Zarr 2 | Format-version compatibility and equivalent item/transform meaning without conflating container identity | 46,162,906 B |
| OME PR2729 Leica LIF | Four logical images in Bio-Formats versus one combined `M`-axis view in `liffile`; VIPP pins the reader/backend and refuses an unreviewed topology change rather than inferring cross-reader equivalence | 227,429 B |
| OME `BF007.nd2` | Very small calibrated Nikon `uint16` YX smoke, including channel and objective metadata | 270,336 B |
| BIA S-BIAD2080 Nikon ND2 | Real four-channel `uint16` CYX fetal-mouse-ovary acquisition with calibrated XY and objective metadata | 20,885,504 B |
| BIA S-BIAD1390 Leica LIF | Real three-channel anisotropic Z stack with inspection/read calibration, channels, and objective parity; native pixel access remains eager | 5,041,111 B downloaded |
| BIA S-BIAD1305 Zeiss CZI | Tiny real two-channel `uint8` CYX acquisition with calibrated XY and objective metadata | 601,024 B |
| Zenodo 7015307 CZI | Purpose-built two-scene `uint16` TZYX reader contract with calibrated ZYX | 5,266,784 B |
| OME imagesc-105684 Olympus OIR | Modern Olympus/Evident two-channel `uint16` TCYX sequence with 10-second time spacing through the native `oirfile` path | 17,517,607 B |
| OME imagesc-71616 Olympus OIB | Real two-channel six-plane volume through native `oiffile`; validates authoritative CZYX pixels, ImageState, calibration, and channel wavelengths | 26,001,408 B |
| Zenodo 6094961 Olympus VSI | Multifile VSI/ETS companion discovery, JPEG2000 decoding, a calibrated TCZYX primary image, and a distinct RGB macro scene through Bio-Formats | 122,550 B downloaded |
| OME Bitplane LZ4 Imaris IMS | True two-channel anisotropic volume in HDF5 with Imaris LZ4; checks logical dimensions rather than padded storage chunks | 1,006,389 B |
| Zenodo 14510432 Zeiss LSM | Legacy four-channel CYX image plus a separate RGB thumbnail series through `tifffile` | 8,632,357 B |
| OME artificial multi-channel 4D OME-TIFF | Signed `int8`, full TCZYX, five dimensions, truthful single-level fallback | 7,889,665 B |
| OME FLIM ModuloAlongC OME-TIFF | Semantic `H` lifetime-bin axis as a negative control: a known non-spatial axis must not silently become Z | 461,842 B |
| BioImage.IO StarDist sample TIFF | True `float32` CYX data and fractional intensity-control behavior | 12,193,476 B |
| BBBC007 image and outline archives | Real `uint8` images, boolean hand outlines, paired identity, and heterogeneous XY shapes | 7,088,307 B |
| BBBC016 images and plate map | 144 channel images covering 72 biological fields, official dose/control structure, and stable per-sample parameter rows | 26,934,804 B |

### Coverage accounting

File count, image count, and field count are deliberately kept separate. The
corpus covers 97 biological fields of view: 72 in BBBC016, 16 two-channel
fields in BBBC007, the IDR0062 volume, the BioImage.IO sample, three real BIA
vendor acquisitions, and the OIR, OIB, Imaris IMS, and LSM acquisitions.
Standards fixtures and synthetic reader-contract scenes are reported
separately; treating every channel image, pyramid level, label, or archive
member as another biological field would overstate diversity.

The source-reader matrix now includes OME-Zarr 0.4 and 0.5, OME-TIFF/ImageJ
TIFF, Leica LIF, Nikon ND2, Zeiss CZI/LSM, Olympus OIR/OIB/VSI, and Imaris IMS.
It spans native `nd2`, `liffile`, `czifile`, `oirfile`, `oiffile`, and
`tifffile` paths plus Java/Bio-Formats fallback. Public data cover `bool`,
`int8`, `uint8`, `uint16`, and `float32`; YX, CYX, CZYX, TZYX, TCYX, TCZYX,
TCZYXS, SYX, TMZCYX, and a semantic non-Z axis; single and multiple items;
multiscales and labels; calibrated anisotropic volumes; archive and companion
members; and reader-specific singleton or series presentations. Generated Tier
0 fixtures retain exhaustive numeric boundary coverage, including wider
integers and `float64`.

### Why these sources

The [official OME sample archive](https://downloads.openmicroscopy.org/images/)
supplies small standards and vendor-format fixtures under explicit licences.
The [IDR OME-NGFF catalogue](https://idr.github.io/ome-ngff-samples/) publishes
format versions, dimensions, study provenance, and licences. IDR0062 is a
particularly compact real 3D image-plus-label example; its
[study record](https://idr.openmicroscopy.org/study/idr0062/) supplies the data
DOI and CC BY 4.0 terms.

The [Broad Bioimage Benchmark Collection](https://bbbc.broadinstitute.org/bbbc/)
is used only where biological structure adds a specific test contract.
[BBBC007](https://bbbc.broadinstitute.org/BBBC007) contributes a public-domain
image/outline pair. [BBBC016](https://bbbc.broadinstitute.org/BBBC016) provides
an official 11-dose-plus-control design with adjacent duplicate wells and three
fields per well, making it more objective for per-sample parameter tests than
visually choosing two convenient images.

The float case is the DOI-versioned
[BioImage.IO StarDist sample on Zenodo](https://zenodo.org/records/6326367).
Its TIFF is native `float32`; no conversion is needed merely to manufacture a
float coverage cell.

The vendor tier uses public records with explicit redistribution terms. The
[BioImage Archive](https://www.ebi.ac.uk/bioimage-archive/) sources contribute
real Nikon, Leica, and Zeiss acquisitions under CC0 or CC BY 4.0. The
[Zenodo CZI dimension set](https://zenodo.org/records/7015307) was created
specifically to test CZI reader dimensions, so its two-scene file is an
objective SourceItem contract rather than an image chosen because VIPP already
opens it successfully. OME `BF007.nd2` remains the cheap reader-smoke case;
the larger BIA ND2 independently checks real multichannel metadata.

The Olympus sources deliberately exercise three different contracts. OIR is a
modern self-contained native file, OIB reaches the native `oiffile` adapter,
and VSI requires a sidecar ETS file plus Bio-Formats/JPEG2000. The compact
Imaris source adds HDF5/LZ4 storage through that fallback path, while the LSM
source covers a still-used legacy Zeiss container and its associated thumbnail.
All five have explicit CC BY 4.0 records and frozen byte identities.

### Predeclared per-sample case

The first parameter-override acceptance uses the lexically stable BBBC016
members `O01f00d0` and `O24f00d0`. They are endpoint wells in the official
plate order, not images selected after looking at VIPP output. Thresholds are
fixed at 85 and 170, exactly one-third and two-thirds of the inclusive `uint8`
maximum.

The test must prove:

- each stable SourceItem resolves exactly one typed override;
- an item with no row inherits the base workflow value visibly;
- representative preview, batch, saved runner, CLI, effective hashes, and
  provenance agree; and
- the saved base workflow is byte-for-byte unchanged.

Scientific segmentation quality is not inferred from this wiring test. If an
algorithm needs a biological accuracy claim, it requires a separate validated
method pack.

## Required `0.14.0a1` Journeys

The corpus is accepted only when it drives these user-visible journeys:

### SourceItem identity

- discover two or more items in one container and assign distinct identities;
- save and reopen the same logical item after an adapter reverses inspection
  order;
- copy a source, mutate one byte or object, and reject the stale revision;
- preserve the same item through inspector, interactive source, batch naming,
  manifests, checkpoints, generated execution, replay, export, and provenance;
- inspect metadata without materializing pixels; and
- keep optional-reader absence a truthful skip/capability result, not a package
  import failure.

For each frozen vendor original, the cache-gated acceptance test verifies the
download hash before asking VIPP to inspect and open every declared item. It
asserts item key, name, shape, dtype, axes, and the bounded calibration,
channel, objective, and acquisition facts exposed by that reader. The files
originally revealed metadata-only parity gaps, missing Java readiness
diagnostics, and the OIB C/Z disagreement. VIPP 0.14.0a1 closes those
contracts without weakening the predeclared expectations. Remaining eager
readers, optional-runtime requirements, cache-gated large decodes, and
reader-specific topology differences stay explicit limitations.

### Pyramid preview

- discover all declared levels and transforms in both frozen OME-NGFF stores;
- read only the chosen lower-level objects for the first preview;
- show the exact preview level while keeping analysis fixed to level 0;
- preserve label rendering and avoid intensity-statistics leakage for labels;
- cancel superseded reads and prevent stale publication; and
- make a single-level OME-TIFF fallback without inventing pyramid support.

The acceptance harness records object keys and bytes read, elapsed time, peak
RAM, cancellation, and final level-0 parity. A successful-looking thumbnail is
not sufficient evidence.

### Typed per-sample parameters

- resolve rows by the stable composite source-node plus logical SourceItem key;
- validate exact typed values through the normal scientific parameter
  contract;
- reject duplicate, stale, zero-match, and multi-match rows before execution;
- bind source revisions separately from authored selectors; and
- prove effective-workflow and provenance parity across all execution routes.

## Extended And Stress Candidates

These are not part of the 210.41 MiB alpha tier. They should be downloaded only
when a named acceptance gap justifies their cost, then frozen before use.

| Candidate | Use only when | Reason for deferral |
| --- | --- | --- |
| IDR0033 `BR00109990_C2.zarr` | The compact IDR0062 image/label groups do not sufficiently test sibling image groups | Nine translated sibling images are excellent identity coverage but store about 331 MB |
| OME eight-series FRET LIF | PR2729 plus the real anisotropic BIA LIF still cannot expose a particular reader-specific series bug | About 49 MB and redundant for the initial happy path |
| scikit-image `kidney.tif` | A converted channel-last fixture becomes a changed contract | The distributed TIFF is ZCYX; scikit-image's in-memory helper presents ZYXC, so it cannot truthfully provide original-file channel-last coverage without a documented derivation |
| Zenodo 8305531 CZI line scans | A real TX/ZX/TZX acquisition with no Y axis becomes a changed source contract | Valuable protection against silent YX assumptions, but it does not drive ordinary image pipelines and follows the positive CZI path |
| Derived Olympus OIF companion fixture | A real native OIF decode becomes a changed acceptance claim | Companion-tree identity and missing/empty companion failures are covered deterministically; derive and freeze pixels from the licensed OIB only if a separate real-read gap justifies it |
| OME Olympus OIR filename/axis negative control | The positive OIR path is stable and filename-based axis inference needs an external guard | Its name says Z stack while authoritative metadata is TYX, which is useful but secondary to the positive TCYX source |
| BBBC024 volumetric labels | Later 3D skeleton/measurement or memory work changes | A selected label member materializes to about 117.6 MB |
| BBBC005 blur series | Later sensitivity or large-batch stress work changes | The archive is about 1.88 GB and unnecessary for SourceItem semantics |

Large multiscale candidates from the
[OME-NGFF challenge catalogue](https://ome.github.io/ome2024-ngff-challenge/)
are appropriate for cancellation, peak-memory, and time-to-first-preview stress
tests. They must remain opt-in and must not become routine release downloads.

## Explicit Exclusion

The Cell Tracking Challenge `Fluo-N3DH-CHO` set would be technically useful for
anisotropic time-series labels, but the provider's
[dataset conditions](https://celltrackingchallenge.net/datasets/) require
permission for public non-challenge scientific use and prohibit cloning a
dataset or part of it. VIPP must not mirror it or publish a derived subset
without written permission. A technically attractive dataset is not an
objective choice when its reuse rights are unsuitable.

## Reproducibility And Storage

- Do not add public archives or Zarr stores to Git. The repository has no Git
  LFS contract for them.
- Cache data under a short user-selected root referenced by
  `VIPP_PUBLIC_DATA_ROOT`; all manifest paths are relative and privacy-safe.
- Download to a sibling staging path, verify completely, then promote. Never
  place `.part`, receipt, or status files inside a Zarr store because those
  files would change its source identity.
- Verify both the compressed artifact and every declared member before opening
  an archived vendor file. The S-BIAD1390 ZIP and its extracted LIF therefore
  have separate byte counts and SHA-256 values; the VSI archive additionally
  pins the `.vsi`, `.ets`, README, and checksum members and preserves their
  relative directory layout.
- Verify a restored cache before every acceptance run. Record the corpus
  version and manifest SHA-256 in release evidence.
- Do not read live public URLs during ordinary CI. Network-free tests validate
  manifest structure, downloader behavior, and generated micro-fixtures.
- When an upstream object differs, keep the old evidence, investigate, and
  create a reviewed corpus revision. Never bless the new bytes automatically.

The two frozen OME-Zarr inventories are:

- [`idr0062-6001240-ngff-v05-objects-v1.json`](validation/public-data/idr0062-6001240-ngff-v05-objects-v1.json), 286 objects; and
- [`idr0062-6001240-ngff-v04-objects-v1.json`](validation/public-data/idr0062-6001240-ngff-v04-objects-v1.json), 1,475 objects.

They use the same `napari-vipp-local-source-v1` aggregate identity as VIPP's
scientific source verification. The inventories were made from byte-preserving
S3 object downloads; the `ome_zarr download` command is unsuitable for this
purpose because it may rewrite a store's chunk layout while preserving array
meaning.

## Change Control

A corpus revision is required when a URL, licence, source byte, object key,
expected shape/dtype/axis/scale, derived recipe, or acceptance role changes.
Adding an unrelated dataset is not a harmless documentation edit: it must name
the missing contract it closes and the existing candidate it cannot replace.

`v4` preserves the byte-exact `v3` manifest and names its hash. It corrects the
BIA S-BIAD1390 LIF expectation to the `liffile` semantic dye-coordinate names
(`ALEXA 488`, `ALEXA 546`, and `ALEXA 405`); the former Green/Red/Blue values
are display LUT colors in the Leica XML. No source byte or other expectation
changed. `v3` preserves the byte-exact `v2` manifest and adds OIR,
OIB, VSI, IMS, and LSM contracts while preserving both historical manifests.
The OIB expectation remains the predeclared CZYX contract and records the
release's passing CZYX inspection/read state. PR2729 continues to record the
different `liffile` and Bio-Formats topologies; a pinned backend change is a
review/refusal boundary, not a silent identity migration.

The corpus can grow after a1, but the default should remain the smallest set
that independently exercises the changed product contracts.
