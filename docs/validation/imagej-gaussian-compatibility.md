# ImageJ Gaussian compatibility qualification

Status: unreleased after 0.16.0a3; local Windows CPU evidence, 2026-10-06.
Tracking: [issue 73](https://github.com/rensutheart/napari-vipp/issues/73).

## Contract

`imagej_gaussian_blur` / **ImageJ Gaussian Blur** is a separate operation; the
existing SciPy Gaussian operations keep their scientific behavior. The target
is the direct-convolution branch of ImageJ 1.54p's GaussianBlur, using sigma
0–8.5 pixels, scalar uint8/uint16/finite float32, canonical trailing YX axes and
independent leading planes. It is CPU-only. Larger sigma, unsupported dtypes,
empty inputs and non-finite float data fail explicitly. No conversion,
resampling, normalization, ROI filtering or physical-unit sigma conversion is
implicit. Sigma zero returns a new buffer.

The source port preserves the dtype-dependent kernel accuracy, polynomial tail
correction, nearest-edge tail sums, float32 paired-neighbor accumulation,
X-then-Y order and final integer rounding. It preserves input buffers, axes,
scale, origin and carried acquisition metadata. Authored sigma and the target
algorithm enter history and shared execution provenance. Finiteness checking
uses bounded float chunks; filtering checks cancellation within each direction.
The implementation retains the full output and several float32 work buffers
for one XY plane (approximately five planes at peak, plus indexing/edge
temporaries). The generic CPU planning estimate does not yet model all of that
workspace; this is not a memory-bounded whole-image executor.

Host planning projects the exact input shape and dtype without reading pixels
or running the filter. Calibrated leading axes, channels, source and acquisition
metadata survive that projection; value ranges remain deferred. The operation
does not inherit upstream finiteness, signed-zero or extrema facts. Those
value-level claims remain outside its reviewed planning contract, including
the sigma-zero case.

Primary source:
[ImageJ 1.54p GaussianBlur.java](https://github.com/imagej/ImageJ/blob/v1.54p/ij/plugin/filter/GaussianBlur.java).
ImageJ is public domain; source attribution remains in the module.

## Independent numerical evidence

`scripts/generate_imagej_gaussian_reference.py` compiles the small Java harness
and executes unmodified, official ImageJ bytecode. It does not invoke VIPP's
Gaussian implementation. The jar is hash-pinned:

`2e1a09961dfb41cee66ddc821b2577a41a072566ce45a49bae69267099741e20`

`imagej_gaussian_reference_v1.json` records ImageJ/JDK versions, platform and
generator/harness hashes. All 37 cases match bit-for-bit, including float32
bit patterns: uint8/uint16/float32, 1×1 / 2×3 / 7×9 planes, and sigma
0 / 0.5 / 1.5 / 8.5, plus a 41×41 float32 negative-zero plane at sigma 1.5.
The latter exercises distinct interior and edge accumulation: the interior
direct neighbor sum preserves negative zero, while the edge branch starts its
neighbor sum at positive zero. Focused kernel/reference checks pass 85 tests.
Tests also exercise constant fields, centered impulse
symmetry and mass, unsigned extrema, signed float values, read-only and strided
buffers, independent leading planes, invalid types/values, arithmetic overflow,
cancellation, calibrated shared execution, saved workflow roundtrip and
generated Python agreement.

An acquired tissue reproduction independently exported all stages from the
supplied ImageJ macro. On three complete 1024×1024 uint16 nuclear planes,
ordinary VIPP Gaussian differed at 2,972,594 pixels; 25 threshold pixels and
233 final ROI pixels consequently differed. Replaying each downstream
operation with the corresponding preceding ImageJ stage gave exact conversion,
Default threshold, 10 square-3×3 dilations, three square-3×3 closes and
4-connected 2D hole filling. The separate ImageJ Gaussian operation then gave
zero differing pixels at every stage and in full/cropped masks and unchanged
marker crops, through both TIFF-source and original-OIR-source workflows.

The source OIR contained three complete planes and a fourth partial plane.
Bio-Formats excluded the partial plane while the native reader returned a zero
plane. The original-OIR workflow explicitly removed its last plane before
channel extraction; no reader was repaired or implicitly changed. Source
hashes, source-channel XML, partial-plane evidence, workflow files, stage
comparisons, implementation fingerprints and QC remain beside the private
research input. No acquired pixels are included in this repository.

## Catalog and planning preservation

The exhaustive inspector showcase adds one sigma-1.5 ImageJ Gaussian node,
connected through the new **Nuclear uint16** tunnel to port zero of the existing
`split_axis_1` node. Its input is the first lane's calibrated uint16 nuclear
ZYX volume. Every earlier scientific node, parameter, connection and tunnel is
unchanged; the complete document also matches the preceding checkout after
removing the added branch, position and explanatory note sentence.

The new showcase scientific digest is
`ac43f4fd8333102af760d20a8056237e20413a5ce589ac22bdae2224d613bc42`.
An exact branch-removal regression retains the preceding digest
`8d6ad9b4559389d64f154527482bfe35aef9d4c3643d4a112f8e6fc3c7ac25fb`,
and every older lane-preservation fingerprint remains pinned unchanged.
Both original and restored/reserialized documents are checked.

Focused array-facts and workflow-golden modules pass 200 tests in the recorded
Windows development environment. New planning regressions use unreadable
pixel descriptors and forbidden kernels for uint8, uint16 and float32 CZYX
inputs, verify retained calibration and deferred statistics, and require
conservative value-fact handling at sigma zero and 1.5.

## Limits

Numerical reproduction is not biological validation of a tissue mask.
Coverage, nucleus-to-tissue assumptions, background correction and biological
marker assignments require independent review. The reproduction did not run
Coloc 2 measurements or qualify Pearson/Manders equivalence. Fiji's ROI ZIP and
drawn QC overlay remain ancillary Fiji exports; the VIPP graph reproduces the
scientific mask and marker crop outputs.

Golden coverage is bounded evidence rather than proof of every input value or
platform. Native Linux/macOS execution, ImageJ downsampling, calibrated sigma,
encoded RGB, 3D blur and GPU parity remain unqualified/unsupported here.
The fixed Crop Stack margins saved for the acquired example reproduce that
sample; the external runner recomputes XY bounds with a 20-pixel margin for a
new input. There is no new dynamic crop-to-mask node.
