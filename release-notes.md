# VIPP 0.13.0a2

VIPP 0.13.0a2 is a focused compute-planning correctness release on the 0.13
GPU alpha. Fresh workflows whose dynamic multi-output host operations feed
accelerator-capable nodes now receive exact per-port planning descriptors
without requiring a preliminary CPU calculation. Unresolved placeholders can
no longer be promoted into apparently resolved downstream values.

In particular, the bundled red-channel object-intensity example now plans and
runs under **Prefer GPU** from a fresh session. If cuCIM is not installed, its
measurement node receives an explained CPU decision instead of the workflow
aborting at Otsu's scientifically correct dtype validation. The release retains
the unified execution, GPU admission, workflow, batch, interface, and
scientific-tool improvements introduced in 0.13.0a1.

CPU remains the portable scientific reference implementation. GPU support is
deliberately selective in this alpha: VIPP accelerates only the combinations of
operation, data type, dimensionality, parameters, dependencies, memory, and
environment that have been reviewed. When a call does not qualify, VIPP keeps
the data on CPU and explains why instead of silently changing its type or
parameters.

## Fixed in 0.13.0a2

- Planning now publishes exact shape, dtype, axes, and channel metadata for
  every exposed `Split Channels` output, including nonzero source ports.
- An unresolved host transform can no longer become falsely resolved through
  a downstream shape-preserving accelerator projection. Its descendants remain
  unresolved and safely defer to CPU until an exact contract or concrete value
  is available.
- **Prefer GPU** can run the fresh red-channel object-intensity example without
  a CPU warm-up. Missing cuCIM produces the intended explained CPU decision for
  measurement instead of a preflight failure at Otsu.

## Install or upgrade

VIPP supports CPython 3.12 and 3.13 for CPU use:

```bash
python -m pip install "napari[pyqt6]>=0.6" "napari-vipp==0.13.0a2"
vipp
```

The optional CUDA route is CPython 3.12-only in this alpha. On native Windows
with a compatible NVIDIA driver, install the pinned CUDA 13 environment with:

```powershell
py -3.12 -m venv ".venv-vipp-gpu-cu13"
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install "napari[pyqt6]>=0.6" "napari-vipp[gpu-cuda13]==0.13.0a2"
& ".\.venv-vipp-gpu-cu13\Scripts\vipp-compute-doctor.exe" --track cuda13
& ".\.venv-vipp-gpu-cu13\Scripts\vipp.exe"
```

Only the NVIDIA display driver is a machine-wide prerequisite for this standard
route. It does not require a separate CUDA Toolkit, `nvcc`, Visual Studio, or
CMake.

When moving from 0.12.0a3 or 0.13.0a1, preserve the old environment and
workflow, then open a duplicate in 0.13.0a2. Older schema-3 workflows
deliberately open in CPU mode and become schema 4 only when saved. Version-1
batch configurations likewise load with explicit CPU intent; version-2
configurations keep their saved compute request. Both load without source-axis
declarations and become version 3 when reviewed and saved. Regenerate exported
Python and saved batch runners because generated programs are locked to the
exact VIPP version that created them.

## Features retained from 0.13.0a1

### GPU acceleration that stays under your control

The main toolbar now offers four clear ways to run a workflow:

- **CPU** always uses the reference CPU implementation.
- **Auto** is the new-session default. It begins with reviewed safe GPU choices
  and learns only from compatible, successful, fallback-free whole-pipeline
  runs on the current machine.
- **Prefer GPU** uses every reviewed GPU implementation that is scientifically
  eligible, even when it has not proved faster than CPU. Unsupported calls
  receive an explained CPU decision.
- **Custom** lets you choose CPU, CuPy, or cuCIM for individual implemented
  nodes and use the whole-pipeline optimizer.

Calculated node cards show what actually ran with compact CPU, CuPy, cuCIM, or
amber CPU-fallback badges. Setup diagnostics provide repair guidance, while
RAM, discrete VRAM, and unified memory are presented in a form appropriate to
the host. One-node benchmarking and **Find fastest pipeline…** use the exact
current workload and present a proposal for review; they do not rewrite the
workflow until the user accepts it.

Reviewed accelerator regions now exist for rolling-ball background estimation
and subtraction, median filtering, 2D and 3D Gaussian blur,
Richardson-Lucy and Richardson-Lucy TV deconvolution, Canny edge detection,
Otsu thresholding, Sigma Filter, connected-components labeling, and the basic
object-measurement profiles with and without intensity. This is region-level
coverage rather than a promise that every call to those nodes runs on GPU.

Compatible arrays can remain on the device between adjacent GPU operations.
VIPP also adds memory admission before work starts, fair sharing of the
accelerator within the process, truthful nested progress, cooperative
cancellation, and classified out-of-memory handling. A retry on CPU occurs
only when the failure is a recognized retryable device OOM, cleanup succeeds,
and the selected policy allows visible fallback.

### The same execution contract everywhere

Interactive calculation, collection batch runs, saved batch runners, generated
Python and CLI programs, and exported output now use the same headless
execution service. Each route can carry the requested compute mode, fallback
policy, and per-node choices, and can record the implementation that actually
ran, its decision reason, progress, fallback or OOM details, cleanup state, and
environment identity.

Workflow persistence advances to schema 4, and collection batch configurations
and manifests advance to schema 3. Saved runners accept compute overrides and
report both item-level and operation-level progress. Cooperative cancellation
finishes the relevant checkpoint and manifest state, returns exit code 130 from
the CLI, and prevents unpublished output from escaping.

Output is prepared privately and promoted only after successful execution and
validation. Generated programs now stage the complete output and provenance
set, reject duplicate destinations, and roll back caught publication failures.
For production collection processing, the saved batch runner remains the
durable choice; the generated `batch_process()` folder loop is a simpler
convenience and does not provide the same pairing, checkpoint, manifest, or
replay guarantees.

### Clear TIFF page interpretation for batches

Some ordinary TIFF files describe their page dimension as the generic `Q`
rather than saying what the pages mean. The Batch workspace now presents the
source-specific `Image stack` setting with plain-language choices. It begins at
`Automatic (recommended)` for a new unsaved source;
`Something else (advanced)...` stays hidden unless someone deliberately
chooses it.

VIPP does not blindly turn Q into Z. If an exact `QYX` representative reaches a
workflow step that explicitly requires `ZYX`, the workspace selects
`Pages are depth slices (Z stack)`, retries the check, and displays a short
notice. The notice explains why the choice changed, that it will be saved, and
that pixel order is unchanged. Users can select `Use the file's labels unchanged`
to opt out,
and VIPP respects that decision instead of suggesting Z again for the same
source.

The automatic choice is UI-only until it resolves. Saving before any change is
needed stores no declaration and reloads as file-information-unchanged. Loaded
historic or headless configs with no declaration behave the same way, so they
are never automatically reinterpreted. After the guarded suggestion is applied,
saving records the concrete `QYX -> ZYX` declaration and loading restores the
Z-stack choice.

The declaration is guarded: the left side must match the axes reported by the
reader exactly. It changes their meaning in place and does not move pixels.
`Reorder Axes`, by contrast, moves pixels and their complete metadata records
but never turns Q into Z. Before a run creates its output folder, artifacts, or
GPU context, VIPP inspects a representative source set and checks the
workflow's scientific axis contract through deterministic image-axis and rank
changes such as slicing, projection, channel extraction, and splitting. This
includes every image output of multi-output analysis nodes; an unsupported
image transform now fails closed instead of ending the check silently. This
catches a deterministic `QYX`/`ZYX` mistake before a large collection is
attempted. Every later item is still checked as it is read, so the
representative check is not presented as proof that every file is uniform.

For every source that is successfully read, version-3 manifests preserve the
reader's raw axes, the effective axes, and the declaration used. The embedded
config still preserves the intended declaration when an item is skipped or
fails before reading. Versions 1 and 2 of the batch configuration remain
loadable, but contain no declaration unless the user reviews and saves them in
the new format.

### Several workflows can stay open at once

A movable workflow tab bar now keeps independent live workflows in one VIPP
window. Each tab retains its own graph, calculated results, caches, undo and
redo history, inspector state, file path, dirty state, and Batch workspace.
Opening or creating a workflow no longer replaces the one already in view.

A collection batch remains attached to the tab that started it while other
tabs remain editable. Progress and completion return to the originating tab,
and VIPP prevents actions that would invalidate the active run until it has
finished or cancellation and cleanup have completed.

Graph authoring is easier to follow as well. Image Source cards show their
current napari layer, file, sample, or collection representative. Named output
tunnels can be rerouted by dragging their source badge to another compatible
output; the change is validated, atomic, undoable, and protected against cycles
and incompatible types.

### Better thumbnails and inspection without changing the science

Thumbnail backing detail now has persistent Low, Standard, High, and Very High
choices. These settings improve HiDPI display, downsampling, and graph zoom
without changing card size, pipeline data, or scientific results.

Full-stack `uint8` and `uint16` percentile contrast now uses exact
native-integer histograms instead of sorting a floating-point copy. Min-max
contrast uses an exact native reduction. Presentation statistics have their own
Auto, CPU, and Prefer-GPU routing, progress, cancellation, and fallback status,
kept separate from the scientific CPU/GPU badge. Stack contrast remains based
on the full output; responsive Slice contrast uses the sampled current view and
can vary slightly with the selected detail.

Recalculating an inspected result preserves compatible napari camera, slice,
translation, rotation, zoom, colormap, and display styling. Display profiles
are remembered per node and output and saved independently from scientific
parameters, with an explicit reset action when the defaults are wanted.

### New filtering and thresholding tools

**Sigma Filter** is a new edge-preserving Lee filter with a defined CPU
contract and a reviewed CuPy implementation for its admitted region. It handles
channels and leading dimensions plane by plane while preserving supported
native `uint8`, `uint16`, and `float32` data.

**ImageJ Auto Threshold (8-bit)** provides an explicit conversion-and-threshold
node targeting ImageJ 1.54p `Default` and `Triangle` behavior for documented
scalar inputs. It is separate from VIPP's existing generic scikit-image
threshold nodes, which are unchanged.

Connected-components labeling now has a complete reviewed CPU/GPU path for its
eligible Boolean 2D and 3D region. GPU results must match the SciPy label IDs
exactly; numeric masks, unsupported shapes, oversized blocks, and unqualified
environments remain visibly on CPU.

### Colocalization becomes easier to explore and export

New **Colocalization Scatter Plot** and **Masked Colocalization Scatter Plot**
nodes turn native-range density plots and threshold guides into durable graph
outputs. The inspector adds a responsive, resizable scatter view with linked
colormap controls, immediate guide movement while exact counts are recomputed,
and PNG or TIFF export at the selected display size.

Threshold-independent density is reused while the exact full ROI is recounted,
rapid requests are coalesced, and large masked histograms are accumulated in
bounded chunks. Interactive views are capped at 1,024 bins per axis; graph
nodes can request up to 4,096 bins and preserve their configured output size,
native populated ranges, and optional symmetric percentile clipping.

## Other fixes retained from 0.13.0a1

- **A generic TIFF page axis no longer fails once for every batch item.** A
  `QYX` input that reaches a demonstrated `ZYX` requirement receives one exact,
  visible Z-stack suggestion instead of a wall of node errors. Preview and
  execution apply the same saved declaration, respect an opt-out, and stop other
  deterministic errors before output or GPU setup rather than after attempting
  the collection.
- **ND2 navigation follows the file's real dimension order.** Metadata
  normalization now respects the reader's ordered dimensions, restoring the
  correct T, Z, and C sliders and slice updates for affected files.
- **Recalculation no longer resets the inspected view.** Replacing a compatible
  inspected layer preserves the user's napari camera, zoom, translation,
  displayed dimensions, slice position, and display styling.
- **Crop Stack keeps graph-port types intact.** Cropped images, masks, and
  labels retain their respective types, so cropped ROI masks restore correctly
  when connected to masked analysis nodes.
- **Colocalization stays in native intensity units.** Pixel and object
  calculations no longer jointly rescale and clip both channels to 0–255.
  Thresholds and intensity sums now use the original finite intensity range.
  Pearson population names now distinguish any-channel and both-channel
  populations, and Fiji M1/M2 and thresholded tM1/tM2 are reported separately.
- **Automatic colocalization thresholds more closely follow the stated Fiji
  target.** The Costes search now follows the documented Fiji Coloc 2 3.1.0
  `SimpleStepper` details, including native one-unit steps and source-compatible
  population and rounding behavior. Independent golden parity is still pending,
  so this remains experimental rather than a claim of exact Fiji equivalence.
- **Canceled or failed work cannot masquerade as a fresh result.** Last coherent
  outputs and their actual implementation badges are preserved where safe;
  stale values are shown as previous or pending rather than relabeled as though
  the new request produced them.
- **Cache and optimizer decisions are harder to confuse.** Structural cache
  identity, stale-source rejection, manual barriers, exact workload evidence,
  transfer modeling, timing-surface separation, and optimizer revalidation have
  all been tightened.
- **Memory failures and cleanup are handled more defensively.** Memory
  accounting, device synchronization, cancellation cleanup, and publication
  gates prevent unsafe partial state from being reused. A cleanup failure
  quarantines further calculation until restart instead of allowing uncertain
  accelerator state to continue.
- **Batch and generated outputs are published more safely.** Source identity,
  measurement assembly, staging, provenance links, duplicate-destination
  checks, cancellation records, and rollback around final promotion have been
  strengthened. A multi-file promotion interrupted at the final operating-
  system boundary can still leave an explicitly recorded `partial` item.
- **Optimizer feedback is more truthful.** Progress distinguishes the current
  operation from overall search progress, time-limit exhaustion is explained,
  and nodes beyond intentional manual or cache boundaries no longer request an
  impossible comparison target.

## Important scientific compatibility notes

Colocalization results can change materially from 0.12.0a3. The new native-unit
intensities, Costes search, Pearson domains, and Manders definitions target Fiji
Coloc 2 3.1.0, but independent upstream golden-fixture parity is still pending.
Existing `manders_m1` and `manders_m2` columns now alias thresholded tM1 and tM2
for workflow compatibility. Preserve old results and perform an external Fiji
or other reference comparison before using the revised values in consequential
analysis. The ImageJ Auto Threshold node is likewise source-aligned but still
awaits independent golden parity.

Review upgraded workflows before saving them. In particular, compare decisive
intermediate and final results around colocalization, cropped mask ports, ND2
axis order, and any node for which GPU execution is enabled. Do not combine
0.12 and 0.13 numerical results without that review.

An axis declaration is not a calibration measurement. `QYX -> ZYX` preserves
the scale, unit, and origin already attached to the leading position; it cannot
discover the physical Z step that the TIFF failed to describe. Verify Output
Metadata and use `Set Pixel Size / Units` before any analysis that depends on
physical Z distance.

## GPU and cuCIM boundaries

The `gpu-cuda13` extra installs the pinned NumPy, SciPy, scikit-image, CuPy,
and CUDA package track. Public GPU evidence for this alpha remains limited to
the recorded native-Windows, 64-bit CPython 3.12, CUDA 13, RTX 5090
environment. Installation on another GPU or operating system is not itself a
claim that VIPP has qualified acceleration there. Linux GPU qualification,
RTX 40-series evidence, and Apple GPU acceleration remain future work; macOS
uses CPU in this alpha.

The ordinary CUDA extra neither distributes nor requires cuCIM. Windows users
who want the reviewed cuCIM-backed background and basic-measurement regions can
build cuCIM 26.6.0 from tag `v26.06.00` at commit
`3c15781c207eab93a317dd9803a6e726fe01f7c4` using VIPP's fixed recipe, then
install their private wheel through the manifest-verifying helper. VIPP does
not host that wheel, and the local build omits Clara whole-slide I/O. Without
an approved local cuCIM build, those affected regions remain on CPU while
independently eligible CuPy and CuPyX regions can still use the GPU. See the
[Windows CUDA and cuCIM guide](https://rensutheart.github.io/vipp-mkdocs/0.13.0a2/getting-started/windows-cuda/)
for the complete procedure and exact provenance pin.

VIPP remains alpha software. For consequential work, record the application
version, workflow, compute request, implementations that actually ran,
environment, inputs, batch configuration, manifests, fallbacks, and validation
evidence.
