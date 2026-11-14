# Apple GPU functionality proposal

Date: 2026-10-30

Status: Proposed feasibility and implementation plan. No Apple GPU operation is
qualified by this document, and no release or delivery date is assigned.

## User outcome and scope

Enable selected existing VIPP nodes to process microscopy images on Apple
Silicon GPUs through existing Metal-capable libraries. Keep the same node
parameters, image meaning, workflow behavior and visible CPU fallback.

The proposal is for VIPP operation adapters and qualification. Evaluate
upstream functions and primitives before choosing a dependency; do not assume
that a library's Metal support establishes coverage for a VIPP node. An
operation without a suitable existing-library path remains on CPU.

This expands the Apple investigation in the
[production GPU plan](gpu-production-implementation-plan.md) into a concrete
candidate list. It remains a [product idea](product-ideas.md), outside the
committed release scope in the [active roadmap](planning.md). The
[durable execution contract](durable-gpu-execution.md) continues to govern
admission, provenance, fallback, cancellation and publication.

## Existing-library candidates

| Candidate | Proposed role | Boundary of the current evidence |
| --- | --- | --- |
| pyclesperanto with its Metal backend | Evaluate existing microscopy image-processing functions for Gaussian filtering and later morphology, median filtering and labeling. | A Metal package has been released, but its metadata classifies it Alpha. The general operation API does not prove Metal execution, equivalence or speed for those operations. |
| MLX | Evaluate existing array operations for views, exact conversions and threshold comparisons; retain its FFT primitives as a later restoration research option. | NumPy-like syntax does not make it a CuPyX/SciPy image-operation substitute. Verify dtypes, evaluation, ownership and synchronization on the chosen Mac. |
| PyTorch MPS | Secondary candidate if the first two libraries fail a concrete operation or packaging requirement. | Avoid adding a second large runtime without demonstrated operation coverage and end-to-end value. MPS does not support float64. |

Choose one array runtime for the first slice. Do not require both MLX and
pyclesperanto, or claim that their device arrays interoperate without copies.
If a later library requires a different array domain, measure and record the
bridge explicitly before admitting a mixed-library path.

## Proposed functionality

All rows below are candidates, not an advertised Apple support matrix. The
first slice has four node candidates; promotion of each remains independent.

| Stage | Existing VIPP node and operation ID | Proposed scope and qualification condition |
| --- | --- | --- |
| First slice | Extract Channel (`extract_channel`) | Use the declared channel axis; preserve all other axes, grid and values. Evaluate bool, uint8, uint16 and float32 only where the runtime preserves their types. Establish safe view ownership and lifetime without inferring RGB from shape. |
| First slice | Convert Dtype (`convert_dtype`) | Only lossless uint8/uint16 to float32 with `scaling="preserve"`. No implicit rescaling, clipping or float-to-integer conversion. An unsupported native dtype stays CPU. |
| First slice | Binary Threshold (`binary_threshold`) | Scalar float32 to Boolean masks. Preserve VIPP's Above, Below, In range and Outside range comparison rules, finite authored thresholds, nonfinite-input behavior and leading axes. RGB/RGBA reduction stays CPU. |
| First slice, conditional | Gaussian Blur (`gaussian_blur`) | Finite float32, independent Y/X planes, declared channel axis excluded. An existing library function plus thin argument/buffer adapters must match SciPy-reflect boundaries, sigma/radius rules and zero-sigma behavior, and pass VIPP's existing Gaussian tolerance gate. Otherwise retain CPU. |
| Next candidates | Gaussian Blur 3D (`gaussian_blur_3d`) | Independent Z/Y/X sigma on resolved spatial axes. Keep channel, time and other leading dimensions independent. Qualify separately from 2D Gaussian. |
| Next candidates | Dilate, Erode, Opening and Closing (`dilate`, `erode`, `opening`, `closing`) | Existing binary morphology functions or their composition must reproduce square Y/X footprints, even/odd footprint origin, zero-outside boundaries and iteration rules. No implicit sphere/box substitution. |
| Next candidates | Connected Components (`label_connected_components`) | Existing labeling function with VIPP's connectivity, independent spatial blocks and deterministic output-label contract. Matching component membership alone is insufficient if integer label identities differ. |
| Next candidates | Fill Holes (`fill_holes`), Remove Small Objects (`remove_small_objects`) | Start with Boolean masks and explicit connectivity. Treat filling all holes separately from bounded-size hole filling; retain CPU for integer-label preservation unless independently qualified. |
| Library-match investigation | Median Filter (`median_filter`) | Slice-wise 2D uint8/uint16 or finite float32; current odd-window normalization, SciPy-reflect borders and bitwise CPU agreement. Check upstream neighborhood limits. Keep CPU if a library's boundary/window behavior differs. |
| Deferred precision investigation | Automatic Otsu thresholding, Canny, Sigma Filter and precision-sensitive measurements | Existing VIPP implementations use double precision, ordered reductions or exact tie/mask rules. No promotion through silent float32 substitution or looser existing contracts. |
| Deferred restoration investigation | Richardson-Lucy and Richardson-Lucy TV | Assess existing FFT/convolution and array primitives only after the first useful slice. Verify PSF normalization, convolution boundary/centering, stopping rules and current precision requirements. MLX has FFT primitives, not an established VIPP-equivalent Richardson-Lucy implementation. |

The first three nodes primarily establish device-resident input preparation
and masks. Gaussian is the first proposed substantive image-processing speed
test; a successful threshold-only demonstration is not enough to advertise a
generally accelerated Apple workflow.

## Implementation sequence and decisions

### 1. Record library and hardware coverage

- Use a real Apple Silicon Mac. An available M1 Max is a useful initial
  feasibility host; an M5 Max with 36 GB is a separate target for measured
  results, not a presumed equivalent configuration.
- Record exact macOS, Python, package, Metal backend and hardware versions;
  install from a clean isolated environment and check upstream licenses.
- Exercise the four first-slice candidates against small CPU reference cases.
  Include borders, singleton dimensions and noncontiguous arrays immediately.
- Produce an operation-level promote/defer/reject report. Select one runtime
  only if existing-library behavior and dependency packaging are suitable.

Exit: a library choice supported by actual Mac results, or a documented reason
to retain CPU. Availability of a wheel or a working import is not this gate.

### 2. Add one optional VIPP runtime and the smallest useful slice

- Implement the existing runtime boundary in
  `src/napari_vipp/core/compute_registry.py`; keep optional imports and device
  initialization lazy so ordinary CPU startup does not load the accelerator.
- Provide runtime-owned conversion, synchronization, view/allocation lifetime,
  exception classification and cleanup. Use unified-memory accounting with one
  shared budget rather than adding fictitious dedicated VRAM to system RAM.
- Register operation implementations only for the exact tested dtype, shape,
  axes and parameter regions. Keep existing CPU and CUDA choices intact.
- Connect existing Auto, Prefer GPU, Custom and CPU behavior, visible reasons
  for fallback, and implementation/environment provenance. Do not add a
  parallel Apple-only workflow engine or bypass the shared executor.
- Admit the Gaussian path only after both scientific and end-to-end timing
  gates pass. If it fails, report a limited feasibility result and reconsider
  the next existing-library candidate before enabling a production slice.

Exit: a useful, bounded Apple path with honest fallback and evidence for the
nodes it actually accelerates. No workflow-schema change is proposed;
implementation identity, caches, export and replay still require regression
checks when wiring the runtime.

### 3. Expand functionality in separate changes

Evaluate 3D Gaussian and binary morphology first, then labeling and mask
cleanup. Each change needs its own contract, implementation identity, measured
memory cost and real-Mac evidence. Median, precision-sensitive operations and
restoration remain separate investigations until suitable library behavior is
demonstrated. A failed candidate does not block already-qualified operations.

### 4. Package and document the admitted subset

Add an optional Apple dependency set only after versions and clean installation
are qualified. Keep the existing macOS CPU installer behavior until the new
route passes packaging checks. Define exact architecture/OS/Python coverage
without implying that an Intel Mac or an untested Apple generation is covered.

Before any advertised capability changes, update the companion
[vipp-mkdocs manual](https://github.com/rensutheart/vipp-mkdocs) with installation,
compute selection, admitted operations, diagnostics and remaining limitations.
The current proposal changes no user instructions or executable behavior.

## Scientific and operational acceptance

Reuse the CPU references and existing parity policies in
`src/napari_vipp/core/compute_specs.py` and
`src/napari_vipp/core/compute_benchmark_adapter.py`.

- Conversions and masks must meet their exact policies. Gaussian must pass
  the full existing float32 comparison, including both NRMSE and maximum
  absolute-error limits; NRMSE alone is insufficient. Do not globally relax
  tolerances to accommodate a backend.
- Cover borders, odd/even and tiny shapes, zero sigma, signed zero, NaN/Inf
  where admitted, integer extrema, semantic channel/time axes, read-only and
  strided arrays, and unchanged input buffers/calibration.
- Operations requiring double precision remain CPU unless an existing-library
  path demonstrably preserves the current scientific contract. In particular,
  do not assume an MLX NumPy conversion preserves float64: upstream documents
  default conversion to float32, while Metal/MPS has no native double type.
- Exercise cancellation and stale-result retirement, allocation/view lifetime,
  memory pressure, safe fallback and clean failure without apparently complete
  outputs. Use the shared interactive, batch and generated-Python paths.
- Record actual implementation and fallback reasons in provenance; saved CUDA
  preferences must open safely on an Apple-only machine. Existing cache and
  scientific-result identity rules still apply.

## Performance evidence

Benchmark the same VIPP node and complete workflow against the current CPU
implementation on the same Mac. Include warmup/compilation separately, explicit
GPU synchronization, array conversion, returned host output and memory peaks.
Measure both isolated nodes and a connected chain such as Extract Channel,
lossless conversion, Gaussian Blur and Binary Threshold.

Use representative 2D microscopy images and 3D stacks where admitted, plus
small inputs where dispatch overhead may favor CPU. For unified memory, report
owned allocations, shared-buffer reuse and conversion/synchronization costs
without double-counting aliases. Retain artifacts identifying dataset shape,
dtype, parameters, repetitions, hardware and software versions.

Promotion requires repeatable useful end-to-end gains in a named workload
region and passing scientific checks. A GPU model, core count, library demo or
generic compute benchmark is not a VIPP speedup claim. Keep CPU where the
candidate is slower or unqualified.

## Evidence to retain before promotion

- Exact dependency/install record and runtime diagnostic output.
- Per-operation input regions, reference comparisons and failed/deferred cases.
- End-to-end timing and unified-memory measurements on named Mac hardware.
- Shared-execution, fallback, cancellation, provenance and clean-install checks.
- Explicit implementation owner, release decision and companion manual change.

## Primary upstream references

Checked 2026-10-30. Recheck the selected versions during feasibility; these
sources describe building blocks, not VIPP qualification.

- [pyclesperanto project and backend overview](https://github.com/clEsperanto/pyclesperanto)
- [Released pyclesperanto Metal package and metadata](https://pypi.org/project/pyclesperanto-metal/)
- [pyclesperanto image-operation API](https://clesperanto-doc.readthedocs.io/en/latest/docs/pyclesperanto/api/filters.html)
- [MLX array framework](https://ml-explore.github.io/mlx/build/html/index.html)
- [MLX FFT primitives](https://ml-explore.github.io/mlx/build/html/python/fft.html)
- [MLX NumPy conversion and dtype behavior](https://ml-explore.github.io/mlx/build/html/usage/numpy.html)
- [PyTorch Metal/MPS and double-precision limitation](https://docs.pytorch.org/docs/stable/notes/mps.html)
