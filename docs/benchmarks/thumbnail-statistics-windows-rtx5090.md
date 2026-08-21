# Thumbnail-statistics CPU/CuPy calibration on Windows RTX 5090

This is machine-local screening evidence collected on 2026-08-05 from the
current GPU-development worktree. It is not a portable speed claim and is not
a durable optimizer record. The worktree was still under development when the
measurements were made, so release evidence should be regenerated from the
eventual committed revision.

The benchmark exercised the production
`ThumbnailStatisticsEngine`, not standalone NumPy or CuPy approximations.
Every reported GPU result produced exactly the same thumbnail contrast limits
as the CPU implementation. Private image paths, filenames, pixels, input
hashes, and calculated contrast-limit values are intentionally omitted.

## What cold and warm mean

- **CPU** is one complete production CPU-engine call.
- **Cold GPU** is the first `Prefer GPU` call in a fresh Python process. It
  includes lazy CuPy import, CUDA probing and context setup, host-to-device
  transfer, the first dtype-specific kernel, result download, and normal
  production cleanup.
- **Warm GPU** is a subsequent complete production-engine call in the same
  process. It still includes transfer, runtime ownership, and cleanup; it does
  not pretend that the source remains resident between unrelated thumbnail
  requests.
- **Exact parity** means the final CPU and GPU limit tuples compared equal. It
  is not a tolerance-based image-quality assertion.

Synthetic cases were placed in separate fresh processes so one dtype or size
could not inherit another case's initialized CUDA context. The short screening
run below used one CPU and one warm-GPU sample per cell. The calibration tool
defaults to three samples for each median and should be used for release-grade
local threshold review.

## Synthetic threshold screening

The inputs are deterministic, contiguous native-integer stacks. Times are
seconds for the complete production call; speedup is CPU time divided by GPU
time.

| dtype | Input | CPU | Cold GPU | Warm GPU | Cold speedup | Warm speedup | Parity |
|---|---:|---:|---:|---:|---:|---:|---|
| `uint8` | 32 MiB | 0.1407 | 1.0574 | 0.0119 | 0.13x | 11.78x | Exact |
| `uint8` | 384 MiB | 1.4499 | 0.9851 | 0.0973 | 1.47x | 14.90x | Exact |
| `uint8` | 512 MiB | 2.4213 | 1.1111 | 0.1483 | 2.18x | 16.33x | Exact |
| `uint16` | 32 MiB | 0.0683 | 0.9379 | 0.0126 | 0.07x | 5.43x | Exact |
| `uint16` | 384 MiB | 0.8856 | 1.0302 | 0.1022 | 0.86x | 8.66x | Exact |
| `uint16` | 512 MiB | 1.0770 | 0.9728 | 0.0955 | 1.11x | 11.28x | Exact |

Within this measured matrix, the sustained cold crossover was 384 MiB for
`uint8` and 512 MiB for `uint16`. Warm GPU execution was already faster at the
smallest common size tested, 32 MiB. These observations agree with the current
production Auto thresholds:

| dtype | Cold Auto threshold | Warm Auto threshold |
|---|---:|---:|
| `uint8` | 384 MiB | 32 MiB |
| `uint16` | 512 MiB | 32 MiB |

These are conservative defaults for this implementation, not universal
hardware constants. The exact crossover can change with the GPU, driver, CuPy
build, host memory bandwidth, array contiguity, channel topology, and competing
GPU work.

## Anonymized representative ND2 screening

The real-acquisition source was evaluated without recording any direct source
identifier or image-derived limit value.

| Workload | Shape | Input bytes | CPU | Cold GPU | Warm GPU | Warm speedup | Parity |
|---|---|---:|---:|---:|---:|---:|---|
| Full two-channel stack | `T19 Z9 C2 Y1150 X822`, `uint16` | 646,585,200 | 1.2901 | 1.3477 | 0.4619 | 2.79x | Exact per channel |
| Extracted single-channel stack | `T19 Z9 Y1150 X822`, `uint16` | 323,292,600 | 0.5847 | Not selected by cold Auto | 0.1013 | 5.77x | Exact |

The full stack sits above the current 512 MiB cold `uint16` threshold, but its
single cold GPU observation was about 4.5% slower than CPU. Once warm, GPU was
about 2.79x faster. The extracted channel sits below the cold threshold, so
Auto correctly kept its first calculation on CPU; after a successful `uint16`
GPU call warmed the engine, the 32 MiB threshold selected GPU and the measured
call was 5.77x faster.

This near-crossover full-stack result is why Auto is described as an informed
heuristic rather than a promise. `Prefer GPU` remains the explicit override for
a user who values accelerator use even on a first call, while CPU remains
available for deterministic host-only presentation work.

## Reproduce or recalibrate

The privacy-safe calibration utility is
[`scripts/benchmark_thumbnail_statistics.py`](../../scripts/benchmark_thumbnail_statistics.py).
From the repository root in PowerShell:

```powershell
& .\.venv-gpu-cu13\Scripts\python.exe `
  scripts\benchmark_thumbnail_statistics.py `
  --sizes-mib 2,4,8,32,128,256,384,512 `
  --dtypes uint8,uint16 `
  --cpu-rounds 3 `
  --warm-gpu-rounds 3 `
  --require-gpu `
  --output thumbnail-statistics-calibration.json
```

Add a private channel-selected acquisition without writing its path or data to
the artifact:

```powershell
& .\.venv-gpu-cu13\Scripts\python.exe `
  scripts\benchmark_thumbnail_statistics.py `
  --nd2 "D:\private-data\representative-sample.nd2" `
  --nd2-channel-index 1 `
  --require-gpu `
  --output thumbnail-statistics-calibration.json
```

By default the ND2 case retains the complete time and Z stack and selects only
the requested channel. `--nd2-time-index` can deliberately select one time
point. The JSON records only anonymized workload metadata and timing/provenance
fields; it omits the supplied path, filename, pixels, content hash, and contrast
limits.

## 2026-08-20 float32 and resident-output addendum

The qualified `float32` Percentile and Min-max implementation uses bounded
radix reductions rather than an unbounded device sort. Protected RTX tests
matched the authoritative CPU limits bit-for-bit across non-finite values,
signed zero, subnormals, interpolation boundaries, channel layouts, and
strided inputs. Cancellation and every healthy terminal path returned the
private runtime pool to zero live and reserved bytes.

The normal host-input path still starts from the scientific host result and
therefore performs one explicit full-image H2D upload. A separate measurement
borrowed the same device-resident output before its existing scientific scope
was released. It did not remove the required scientific D2H result transfer;
it removed only the later redundant thumbnail H2D upload.

| float32 input | Host-input body | Resident body | Time saved | Relative reduction |
|---:|---:|---:|---:|---:|
| 2 MiB | 3.641 ms | 2.998 ms | 0.643 ms | 17.7% |
| 32 MiB | 23.801 ms | 18.653 ms | 5.148 ms | 21.6% |
| 128 MiB | 93.078 ms | 68.267 ms | 24.810 ms | 26.7% |
| 512 MiB | 369.204 ms | 277.246 ms | 91.958 ms | 24.9% |

The absolute gain is negligible for the bundled 576 KiB example and becomes
material at 128 MiB. Production therefore requests the resident shortcut only
for one warm, selected, retained `float32` image card under Prefer GPU, with a
128 MiB minimum. Other cards keep the asynchronous, cancellable host-input
worker so rapid scientific edits are not delayed by presentation scans.

The resident result records `resident_borrow`, zero logical input H2D bytes,
and only bounded auxiliary/D2H metadata. It returns immutable host limits and
does not retain or release the borrowed scientific array. Recoverable
presentation failure is a soft miss; cancellation propagates; any scratch
release failure is fatal and marks the CUDA runtime unhealthy. Resident scan
wall time is excluded from completed-run scientific timing history.

This remains machine-local screening evidence, not a portable speed promise.
The anonymized measurement artifact was written outside the repository at
`D:\Temp\vipp-float32-resident-benchmark-20260820.json` with SHA-256
`189dadbcdf1daae9c0f03c89e94e95701c66d6e85b5838fc17ad76287aaee0c1`.
