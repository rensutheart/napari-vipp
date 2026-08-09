# Durable GPU Execution

This page describes the GPU-development branch contract for collection batch
runs, generated Python, command-line replay, and standalone output export. These
surfaces now use the same headless execution service as interactive VIPP. They
therefore preserve the requested compute mode and per-node choices, report the
implementation that actually ran, and apply the same scientific-admission,
fallback, memory, cancellation, and cleanup rules.

This is branch-scoped functionality, not a released cross-platform GPU-support
claim. Only operation regions and runtime environments that have passed their
documented gates can run on a GPU. Everything else follows the explicit CPU or
fallback decision described below.

## One Execution Contract

The durable surfaces all submit a `ComputeRequest` to
`core.execution.execute_pipeline_request()`:

- `mode`: `cpu`, `auto`, `prefer_gpu`, or `custom`;
- `node_preferences`: stable choices keyed by workflow node ID;
- `fallback_policy`: `visible` or `strict`;
- runtime/device selection and accelerator memory cap/reserve;
- the precision and workload policy identifiers; and
- the explicit experimental-admission flag.

No surface calls a GPU implementation directly. The shared planner decides the
eligible implementation for the exact node data and parameters, and the shared
transactional executor keeps device values private until synchronization and
cleanup succeed. Manual nodes are included in full batch and generated runs,
matching interactive full-pipeline behavior.

The four policies have distinct purposes:

| Mode | GPU selection contract |
| --- | --- |
| `cpu` | Use only the authoritative host implementation. |
| `auto` | With no exact compatible history, use reviewed `public_auto_candidate` GPU defaults wherever they pass the current safety gates. Accelerated-only history causes the next global Auto run to measure CPU once on the same execution surface; a complete pair then selects under the 1.20x/20-ms gate. This remains the new-session default. |
| `prefer_gpu` | Consider every reviewed public GPU implementation, including `public_custom`, and use an eligible GPU without requiring it to beat CPU. |
| `custom` | Apply the active per-node CPU/library/exact preferences; this is the only mode that exposes node benchmarking and `Find fastest pipeline…`. |

Prefer GPU bypasses only the performance gate. Scientific parity, dtype,
parameter, shape, optional-dependency, environment, provider, and memory gates
remain mandatory. VIPP never inserts a cast, changes a parameter, or admits a
developer-hidden provider merely to place more work on GPU. Developer-hidden
implementations are considered only when the request explicitly enables
experimental admission, and such a run is not a public support claim. If every
eligible GPU candidate has complete comparable timing evidence, Prefer GPU
selects the fastest GPU; otherwise it selects deterministically by stable
implementation ID. Missing timing evidence is not itself a reason to use CPU.

On native Windows, Auto, Prefer GPU, and Custom use the same CUDA-device
admission rule: the exact pinned Python, scientific-stack, CUDA runtime,
CuPy/provider, and cuCIM provenance gates must pass; the matching numeric driver
API must be at least `13030`; and the selected NVIDIA CUDA device must report a
numeric compute capability of at least 7.5. The GPU model itself is recorded as
provenance rather than used as an exact-model allowlist. Auto and Prefer GPU do
not run a local parity benchmark merely because a new qualifying model is
encountered; the explicit Custom benchmark and Find-Fastest actions provide
that local comparison when wanted.

A Prefer-GPU request requires `fallback_policy: visible`; `strict` is invalid.
An unsupported node receives an explained ordinary CPU planning decision,
because “GPU wherever possible” deliberately includes CPU elsewhere. A
retryable device OOM still follows the visible one-retry rule described below.

CPU-only installations remain first-class. Importing VIPP, loading workflows,
importing a generated program, planning a CPU batch, and completely skipping a
batch item do not import or initialize CuPy or cuCIM. `Auto` and `Prefer GPU`
use CPU normally when no admitted accelerator is available. A Custom request
follows its saved `visible` or `strict` policy rather than producing an
optional-package import traceback.

Successful, fallback-free completed full-pipeline runs may append only their
wall time and exact assignment to machine-local history. With no compatible
history, Auto uses reviewed GPU defaults. If compatible history contains an
accelerated assignment but no CPU observation, the next global Auto run
executes the authoritative CPU assignment once on that same execution surface.
Once both observations exist, a later matching Auto run selects acceleration
only when it clears the reviewed 1.20x/20-ms benefit margin; otherwise it
selects CPU. Interactive, batch, and registry-lifecycle timing surfaces are
keyed separately and never mixed. Auto never silently benchmarks multiple
implementations; node and pipeline comparisons remain explicit Custom actions.
Raw image data and workflow documents are not stored in this timing history.
Before the optional authoritative CPU comparison, the executor conservatively
estimates its additional host peak. It proceeds only when physical RAM—and, on
Windows, system commit—retain a safety reserve. Otherwise Auto keeps the
reviewed safe assignment and records a warning that the missing CPU observation
was skipped rather than risking an avoidable host OOM.

## Saved And Effective Compute Requests

Workflow schema 4 stores portable authored intent, including the serialized
`prefer_gpu` mode. Batch configuration schema 3 retains the full `compute`
object and adds guarded source-axis declarations, so a saved collection run
also retains its source interpretation, runtime, device, memory, and
experimental settings. Batch config version 1 had no compute contract and is
migrated to an explicit CPU request; version 2 keeps its saved compute request.
Both older versions load without axis declarations and become version 3 only
when reviewed and saved. VIPP never guesses that an old batch intended to use
an accelerator or that a generic TIFF page axis meant Z. Per-node preferences
are preserved when another global mode is active but remain dormant unless the
mode is `custom`.

The precedence rules are deliberate:

1. A new Batch workspace captures the current toolbar request when its config is
   built or saved.
2. A loaded batch config retains its saved request while the toolbar compute
   request remains exactly as it was at load time.
3. Changing any toolbar compute setting after load selects the current toolbar
   request for the next preview, save, or run. This is a complete replacement,
   not an ambiguous merge with the loaded request.
4. A saved `vipp_batch_pipeline.py` uses its config request by default. Its CLI
   overlays only the explicitly supplied mode, fallback, and per-node options;
   the other saved fields remain unchanged.
5. An exported workflow function uses its embedded schema-4 request unless the
   caller supplies a complete `ComputeRequest` or mapping. The exported CLI
   similarly overlays only options written on the command line.

Unknown node IDs and malformed preferences fail before batch output artifacts
are produced. A run-time override does not mutate the workflow or saved config.
The batch manifest records both `configured_request` and `effective_request`,
whether an override was used, the effective-request fingerprint, the saved
config hash, and an `effective_sha256` that changes when the run override
changes.

## Run A Saved Batch

Create the artifacts from `Batch workspace...` with `Save generated Python
runner` enabled. The saved runner defaults to its sibling
`vipp_batch_config.json` and to the workflow path recorded by that config.

From a CPU development environment on Windows:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python .\results\vipp_batch_pipeline.py --progress
```

From the established CUDA 13 development environment:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13
.\.venv-gpu-cu13\Scripts\python.exe .\results\vipp_batch_pipeline.py --progress
```

The equivalent Linux preparation is:

```bash
bash scripts/setup_gpu_dev.sh --track cuda13
./.venv-gpu-cu13/bin/python ./results/vipp_batch_pipeline.py --progress
```

Use explicit overrides only when the run should differ from the saved config:

```powershell
.\.venv-gpu-cu13\Scripts\python.exe .\results\vipp_batch_pipeline.py `
  --config .\results\vipp_batch_config.json `
  --workflow .\results\vipp_batch_workflow.json `
  --compute-mode custom `
  --fallback-policy visible `
  --node-preference gaussian_blur_1=library:cupyx `
  --node-preference otsu_threshold_1=cpu `
  --progress
```

To request every eligible reviewed GPU implementation without running the
optimizer first:

```powershell
.\.venv-gpu-cu13\Scripts\python.exe .\results\vipp_batch_pipeline.py `
  --compute-mode prefer_gpu `
  --progress
```

When the CLI changes the mode to `prefer_gpu` and no fallback override is
provided, it uses `visible`. Supplying `--fallback-policy strict` with
`prefer_gpu` is rejected as an invalid request.

`--node-preference` is repeatable and uses
`NODE_ID=PREFERENCE`. Stable preference forms are `auto`, `cpu`, `best_gpu`,
`library:<library-id>`, and `implementation:<implementation-id>`. Prefer values
written by VIPP or copied from a reviewed config; an exact implementation pin
may be unavailable on another computer. These preferences are carried through
all modes for lossless round trips but affect planning only in `custom`.

The runner exits with:

- `0` when the batch finishes without recorded failures;
- `1` when a finalized batch contains failures;
- `2` when setup or execution fails before a normal batch result is returned;
  or
- `130` after cooperative cancellation.

Use one `Ctrl+C` to request cancellation and wait for the active operation to
synchronize and clean up. A second `Ctrl+C` is an emergency interrupt and can
bypass normal finalization.

## Generated Python And Standalone Export

`Export Python...` emits an immutable, version-locked workflow program. Its
callable API accepts the same durable execution controls:

```python
from threading import Event

from napari_vipp.core.compute import ComputeRequest

cancel = Event()
results = run_pipeline(
    src_input=load_image("input.ome.tif"),
    compute_request=ComputeRequest(mode="auto", fallback_policy="visible"),
    progress_callback=progress_callback,
    cancel_event=cancel,
)
save_image(
    results["gaussian_blur_1"],
    "gaussian.ome.tif",
    image_state=results.image_states["gaussian_blur_1"],
    provenance=results,
    output_node_id="gaussian_blur_1",
)
```

The generated CLI supports `--compute-mode` (including `prefer_gpu`),
`--fallback-policy`, repeatable `--node-preference`, `--progress`, and
`--provenance`/`--no-provenance`. It
returns `130` for cancellation and `2` for execution or publication failure.
With provenance enabled, a successful output has an atomic sibling such as
`gaussian.ome.tif.vipp-provenance.json`; a failed or cancelled single-output run
also attempts to write a failure sidecar at the requested destination name.

The generated `batch_process()` folder loop is deliberately only a convenience
for one varying primary source. Local files opened by its built-in
`load_image()` helper are content-hashed before reading, verified again after
materialization, and bound to each output sidecar. For each item, all requested
outputs and sidecars are written to a private staging directory, checked for
duplicate destinations and cleanup success, and committed as one
rollback-protected set within one destination directory. The loop still does
**not** provide the saved batch runner's multi-source pairing, collision plan,
final source-identity recheck immediately before publication, checkpoints,
manifest, or durable replay guarantees. Use
`vipp_batch_pipeline.py` for production collection processing.

## Exact Provenance

Every successful CPU or GPU execution produces a formal execution report. Its
durable serialization records:

- the requested compute policy and environment fingerprint;
- the actual decision for every completed computed node, including pruned
  intermediates;
- the runtime, array domain, implementation library, stable implementation ID
  and version, parity policy, and cache-equivalence group;
- whether that implementation identity is complete;
- requested preference, decision reason, benchmark digest, and memory estimate;
- classified fallback records and warnings; and
- completed, failed, or cancelled outcome plus cleanup evidence.

A generated `PipelineResults` exposes `execution_report`,
`effective_compute_request`, `node_compute_provenance`, and
`execution_provenance`. `results.provenance_for(node_id)` binds that run to one
specific output node and port and includes its own digest. `save_image()` uses
this output-specific document for the atomic `.vipp-provenance.json` sidecar.

Batch manifest schema 3 stores the full execution document and its SHA-256 on
each item. A source successfully read during item execution also records its
raw axes, effective axes, and optional declaration; the embedded config retains
the intended declaration for sources skipped or failed before reading. Every
published output record has `provenance_status: produced`
and an `execution_provenance_sha256` link to that exact item execution. The
run's checkpoint sidecars contain the same item link; separate output
provenance files are unnecessary because the authoritative batch manifest
already binds each output path and status to the digest.

An implementation identity can be marked `identity_complete: false` for a
custom or external planner that did not supply a matching versioned
declaration. Such a record remains honest decision evidence, but it is not an
exact implementation-reproduction claim.

## OOM, Fallback, And Cleanup

An ordinary Prefer-GPU CPU decision means that no reviewed GPU candidate passed
all admission gates for that node. It is recorded with the exact reason and is
not an attempted-device fallback.

`visible` fallback permits one CPU retry of a complete transactional device
segment only after a classified, retryable runtime out-of-memory failure. The
GPU attempt is synchronized and cleaned first. `strict` does not retry that
segment; it returns the typed memory failure. Other device defects are not
misreported as OOM and are not silently retried.

An insufficient-memory preflight records its typed failure, runtime/segment,
and required/available bytes before any kernel or writer runs. An eligibility
or availability fallback selected before execution remains visible in each
node decision and the compact `fallbacks` summary; `fallback_records` is the
more detailed audit trail for an attempted device segment and CPU retry.

Host `MemoryError` is classified separately from GPU OOM. Windows diagnostics
distinguish available physical RAM from remaining commit headroom because
either can limit a large CPU allocation; the `GlobalMemoryStatusEx` page-file
fields represent system commit limit/headroom, not just page-file size.

Each OOM retry produces a structured record containing the segment/runtime and
node IDs, typed reason and reason code, exception type/message, attempt counts,
whether the CPU retry succeeded, cleanup result, the planned memory estimate,
and the available runtime memory snapshot when one could be captured. A later
item is replanned independently; one item's fallback does not rewrite the saved
request for the rest of the batch.

Publication fails closed. Batch outputs remain private until source
reverification and execution cleanup are both proven. A false or unknown GPU
cleanup result blocks promotion, records a failure in the item/manifest, and
leaves no newly published output for that item. Standalone generated output
publication follows the same rule and reports publication failures separately
from successful scientific calculation.

In an interactive VIPP process, a false cleanup result from pipeline execution,
a node benchmark, **Find fastest**, or collection batch means the accelerator
runtime is no longer safe to reuse. VIPP requests cooperative cancellation of
every other active compute owner, preserves its last coherent interactive
result, and disables new calculation, policy changes (including policy-changing
undo/redo), benchmark/optimizer work, and batch starts until the application is
restarted. **Find fastest** also rolls back benchmark records written by the
unsafe analysis. If an individual rollback cannot be written safely, VIPP
writes a durable poison marker first and moves the complete local timing-store
file to an `.unsafe-*` quarantine name under the store's cross-process lock. A
restart resolves that marker or refuses to open the active store, so suspect
evidence is never silently reused. If even the marker cannot be written, the
restart alert names the active file that must be moved manually. This
process-level quarantine is separate from ordinary visible CPU fallback, which
requires cleanup to have succeeded.

## Progress And Cancellation

Batch has two progress levels:

- overall item progress reports the item number, batch ID, and final item
  status; and
- current-operation progress reports the containing item plus node/operation,
  completed checkpoint, total checkpoints, and message.

The Batch workspace displays both bars. `--progress` prints both streams in the
saved runner. The generated workflow CLI prints operation-level updates.

Cancellation is cooperative. VIPP checks the shared token between nodes,
device segments, iterative/tiled operation checkpoints, private output staging,
source verification, and batch items. A monolithic CPU/GPU library call or file
writer cannot report invented internal percentages and may finish its current
call before cancellation is observed. Once multi-output promotion begins after
all data is staged and verified, VIPP finishes that short promotion boundary to
avoid creating an avoidable partial publication set.

The active item and its unpublished outputs become `cancelled`; later unstarted
items become `skipped`. The latest manifest, run archive, and item checkpoints
are finalized on the normal cooperative-cancellation path, and the CLI exits
`130`.

## Current Limitations

- GPU execution remains limited to admitted operation regions and qualified
  environments on the GPU development branch. Unsupported dtype, parameter,
  shape, dependency, or platform regions use CPU or fail according to policy.
- `Auto` is hardware- and workload-dependent. Reproduction requires the
  recorded environment and actual implementation provenance, not only the
  authored request.
- Floating-point implementations can differ slightly across GPU models and
  driver/JIT combinations while remaining inside their declared parity
  tolerances; integer operations with a bitwise parity contract remain exact.
  Publications should report the VIPP version, GPU model, compute capability,
  driver API, CUDA runtime, CuPy/cuCIM and CPU scientific-stack versions,
  workflow parameters, and actual per-node implementations from the execution
  provenance.
- `Prefer GPU` is an accelerator-placement preference, not a performance
  promise. It can deliberately choose a GPU that is equal to or slower than
  CPU, while still refusing scientifically or operationally inadmissible
  regions.
- Current collection batching is local-file, sorted-position pairing. It does
  not iterate semantic T/C/Z combinations or discover plate/well/field HCS
  layouts.
- Atomicity is per artifact. A failure during final promotion of several
  outputs can produce an explicitly recorded `partial` item.
- Item checkpoints are a recovery trail, not automatic resume. The generated
  folder convenience is not a durable batch substitute.
- Generated workflow programs require the exact VIPP version that created
  them. Their built-in `load_image()` helper captures and verifies exact local
  source identities, and their folder helper privately stages and transactionally
  commits each requested output set. Callers supplying raw arrays must also
  supply any source identity they want recorded. The saved batch runner remains
  the durable surface for multi-source capture, a final source recheck before
  publication, collision planning, checkpoints, manifests, and replay.
- Native Linux, broader multi-device performance characterization, Apple M1 Max
  provider feasibility, and feature-complete cuCIM/Clara packaging remain
  release work. Qualifying native-Windows NVIDIA CUDA devices are available to
  Auto, Prefer GPU, and Custom; CPU execution remains the portable
  Windows/macOS/Linux path.

For the owning module boundaries, see [Architecture](architecture.md). For
operation-specific admitted regions and setup constraints, see the
[production GPU implementation plan](gpu-production-implementation-plan.md).
